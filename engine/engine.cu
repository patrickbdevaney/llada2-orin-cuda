// LLaDA2.2-mini Orin engine core: weight loading and the forward pass of one 32-token block.
//
// forward_block(ids, pos0, ctx_blocks, commit) runs all layers for one block of 32 tokens at positions
// pos0 .. pos0+31 against ctx_blocks committed cache blocks:
//   * a DENOISING step (commit = false): the block (with mask tokens) attends cache + itself; logits come back for
//     the sampler; the cache is untouched.
//   * a COMMIT pass (commit = true): same forward over the final tokens, then every layer's fresh K/V is quantized
//     into cache block ctx_blocks (attn::kv_commit). Prefill is a sequence of commit passes, one per 32 tokens.
// Everything after the embedding upload runs on the GPU without a host round trip (routing and the expert work
// lists are built by ops::route_schedule; the GEMMs read their item counts from device memory), so a step can be
// captured into a CUDA graph.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <array>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ops.cu"
#include "attn.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace eng {

constexpr int T = 32, HID = 2048, NH = 16, NKV = 4, HD = 128, E = 256, TOPK = 8;
constexpr int GW = 4, GMS = 2;                       // GEMM warps per block, strips per warp (rows % 128 == 0)
constexpr int PB = 16, TTM = PB * T;                  // batched prefill: up to 16 blocks (512 tokens) per pass

struct Ent { std::string file; size_t off, bytes; std::string dtype; std::vector<long> shape; };

struct Index {
    std::map<std::string, std::string> cfg;
    std::map<std::string, Ent> t;                    // key "file:name"
    std::map<int, std::vector<std::array<long, 3>>> experts;   // layer -> [E] (bits, gu_off, d_off)
    std::map<std::string, size_t> fsize;
    int ci(const std::string& k) const { return std::stoi(cfg.at(k)); }
    double cf(const std::string& k) const { return std::stod(cfg.at(k)); }
    const Ent& get(const std::string& f, const std::string& n) const {
        auto it = t.find(f + ":" + n);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s:%s\n", f.c_str(), n.c_str()); exit(1); }
        return it->second;
    }
};

inline Index read_index(const std::string& dir) {
    Index ix;
    std::ifstream in(dir + "/engine.idx");
    if (!in) { fprintf(stderr, "no %s/engine.idx\n", dir.c_str()); exit(1); }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string kind; s >> kind;
        if (kind == "C") { std::string k, v; s >> k >> v; ix.cfg[k] = v; }
        else if (kind == "T") {
            Ent e; std::string name; s >> e.file >> name >> e.off >> e.bytes >> e.dtype;
            long d; while (s >> d) e.shape.push_back(d);
            ix.t[e.file + ":" + name] = e;
        } else if (kind == "E") {
            int l, x; long b, g, dd; s >> l >> x >> b >> g >> dd;
            auto& v = ix.experts[l];
            if (int(v.size()) <= x) v.resize(x + 1);
            v[x] = {b, g, dd};
        } else if (kind == "F") { std::string f; size_t b; s >> f >> b; ix.fsize[f] = b; }
    }
    return ix;
}

// Read a file region straight into device memory through a pinned bounce buffer (no full host copy: on the Orin
// host and device memory are the same 7.4 GiB).
inline void file_to_device(const std::string& path, size_t off, size_t bytes, void* dst) {
    static void* bounce = nullptr;
    constexpr size_t CH = 64 << 20;
    if (!bounce) CK(cudaMallocHost(&bounce, CH));
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fseeko(f, off_t(off), SEEK_SET);
    for (size_t done = 0; done < bytes; done += CH) {
        const size_t n = std::min(CH, bytes - done);
        if (fread(bounce, 1, n, f) != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
        CK(cudaMemcpy(static_cast<uint8_t*>(dst) + done, bounce, n, cudaMemcpyHostToDevice));
    }
    fclose(f);
}

inline size_t file_bytes(const std::string& p) {
    struct stat st;
    if (stat(p.c_str(), &st)) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    return size_t(st.st_size);
}

template <class X> X* dev_copy(const std::vector<X>& v) {
    X* d; CK(cudaMalloc(&d, v.size() * sizeof(X)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice));
    return d;
}

inline moe8::ExpertTable single_table(const uint8_t* base, size_t gu_off, size_t d_off, int bits) {
    return moe8::ExpertTable{base, dev_copy(std::vector<uint64_t>{gu_off}), dev_copy(std::vector<uint64_t>{d_off}),
                             dev_copy(std::vector<int8_t>{int8_t(bits)})};
}

struct Layer {
    bool moe = false;
    int I = 512;                                     // MLP intermediate (5120 for the dense layer 0)
    moe8::ExpertTable qkv, dense, mlp, experts;      // mlp: layer-0 dense MLP; experts: E routed + shared at E
    float *g_in, *g_post, *gq, *gk, *bias = nullptr;
    __nv_bfloat16* router = nullptr;
    uint8_t* kv = nullptr;                           // [4 heads][max_blocks][KV_BLOCK_BYTES]
    half *kf, *vf;                                   // this pass's fresh K/V [tokens][4][128] (kept for commit)
};

struct Act { moe8::Act a; int8_t* q; float* s; int* sum; };

inline Act alloc_act(int rows, int K) {
    Act x;
    CK(cudaMalloc(&x.q, size_t(rows) * K));
    CK(cudaMalloc(&x.s, size_t(rows) * (K / 128) * 4));
    CK(cudaMalloc(&x.sum, size_t(rows) * (K / 128) * 4));
    x.a = moe8::Act{x.q, x.s, x.sum};
    return x;
}

__global__ void set_positions(int* pos, int pos0) { pos[threadIdx.x] = pos0 + threadIdx.x; }

__global__ void half2float(const half* a, float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __half2float(a[i]);
}

class Engine {
public:
    int L = 20, V = 157184, mask_id = 156895, max_blocks = 0, nlayers = 20;
    float eps = 1e-6f, theta = 3e6f, rscale = 2.5f;
    std::vector<Layer> layers;
    moe8::ExpertTable lm;
    bool lm_split = false;                            // lm is the high plane; logits are screening values
    const uint8_t *lm_hi = nullptr, *lm_lo = nullptr;
    moe8::ExpertTable lm8{};                          // LLADA_LM_CHECK only
    float *final_norm, *signs, *inv_freq;
    const uint16_t* embed = nullptr;                  // host mmap of embed.bin (fp16 rows)
    size_t weight_bytes = 0, kv_bytes = 0;

    // buffers
    float *h, *hn, *qkv, *rlogits, *Y, *logits, *Opart, *Mpart, *Lpart, *ones;
    half *q, *o, *Z, *hbuf;
    int *pos, *iota;
    Act xA, oA, zA, z0A;
    ops::Sched sched;
    moe8::Work *items_dense, *items_dense16, *items_pf, *items_pf16;
    cudaStream_t st, st2, st3;                       // st2 / st3: the NT2 / NT4 expert GEMMs, concurrent with NT1 on st
    cudaStream_t st4;                                // the shared expert, concurrent with routing
    cudaEvent_t ev_a, ev_b, ev_c, ev_s, ev_sd;
    moe8::Work* items_shared;                        // decode: the shared expert's one item (slots 256 .. 287)
    int* stok_shared;                                // slot -> token for it (slot 256 + t -> t), independent of routing
    int max_chunks = 0;

    void load(const std::string& dir, int max_ctx_tokens, int only_layers = -1) {
        Index ix = read_index(dir);
        L = ix.ci("num_hidden_layers");
        nlayers = only_layers > 0 ? only_layers : L;
        V = ix.ci("vocab_size"); mask_id = ix.ci("mask_id");
        eps = float(ix.cf("rms_norm_eps")); theta = float(ix.cf("rope_theta")); rscale = float(ix.cf("routed_scaling_factor"));
        const int first_moe = ix.ci("first_k_dense_replace");
        max_blocks = max_ctx_tokens / T + 1;
        const int nb = ix.ci("nonexpert_bits");
        auto small_path = dir + "/small.bin";
        // small tensors: one device copy of small.bin
        const size_t small_bytes = file_bytes(small_path);
        uint8_t* small; CK(cudaMalloc(&small, small_bytes));
        file_to_device(small_path, 0, small_bytes, small);
        weight_bytes += small_bytes;
        auto sp = [&](const std::string& n) { return small + ix.get("small.bin", n).off; };
        final_norm = reinterpret_cast<float*>(sp("final_norm"));
        signs = reinterpret_cast<float*>(sp("signs"));
        layers.resize(nlayers);
        for (int li = 0; li < nlayers; ++li) {
            Layer& Ly = layers[li];
            Ly.moe = li >= first_moe;
            Ly.I = Ly.moe ? ix.ci("moe_intermediate_size") : ix.ci("intermediate_size");
            const std::string nef = "nonexpert_L" + std::to_string(li) + ".bin", exf = "experts_L" + std::to_string(li) + ".bin";
            const size_t ne_bytes = file_bytes(dir + "/" + nef);
            const size_t ex_bytes = Ly.moe ? file_bytes(dir + "/" + exf) : 0;
            const size_t ex_pad = (ex_bytes + 4095) & ~size_t(4095);
            uint8_t* arena; CK(cudaMalloc(&arena, ex_pad + ne_bytes));
            if (Ly.moe) file_to_device(dir + "/" + exf, 0, ex_bytes, arena);
            file_to_device(dir + "/" + nef, 0, ne_bytes, arena + ex_pad);
            weight_bytes += ex_pad + ne_bytes;
            const uint8_t* ne = arena + ex_pad;
            Ly.qkv = single_table(ne, 0, ix.get(nef, "qkv").off, nb);
            Ly.dense = single_table(ne, 0, ix.get(nef, "dense").off, nb);
            const size_t gu = ix.get(nef, "gate_up").off, dn = ix.get(nef, "down").off;
            uint8_t* nem = arena + ex_pad;
            moe8::relayout_matrix(nem + ix.get(nef, "qkv").off, 3072, HID, nb);
            moe8::relayout_matrix(nem + ix.get(nef, "dense").off, HID, HID, nb);
            moe8::relayout_matrix(nem + gu, 2 * Ly.I, HID, nb);       // shared expert (MoE) / dense MLP (layer 0)
            moe8::relayout_matrix(nem + dn, HID, Ly.I, nb);
            if (Ly.moe) {
                const auto& ex = ix.experts.at(li);
                std::vector<uint64_t> go(E + 1), doff(E + 1);
                std::vector<int8_t> bits(E + 1);
                for (int e = 0; e < E; ++e) {
                    bits[e] = int8_t(ex[e][0]);
                    go[e] = ex[e][0] ? uint64_t(ex[e][1]) : 0;
                    doff[e] = ex[e][0] ? uint64_t(ex[e][2]) : 0;
                    bool dup = false;                   // an aliased entry must not be relaid twice
                    for (int x = 0; x < e && !dup; ++x) dup = ex[x][0] && go[x] == go[e];
                    if (ex[e][0] && !dup) {
                        moe8::relayout_matrix(arena + go[e], 2 * Ly.I, HID, bits[e]);
                        moe8::relayout_matrix(arena + doff[e], HID, Ly.I, bits[e]);
                    }
                }
                go[E] = ex_pad + gu; doff[E] = ex_pad + dn; bits[E] = int8_t(nb);
                Ly.experts = moe8::ExpertTable{arena, dev_copy(go), dev_copy(doff), dev_copy(bits)};
                Ly.router = reinterpret_cast<__nv_bfloat16*>(sp("L" + std::to_string(li) + ".router"));
                Ly.bias = reinterpret_cast<float*>(sp("L" + std::to_string(li) + ".expert_bias"));
            } else {
                Ly.mlp = single_table(ne, gu, dn, nb);
            }
            const std::string p = "L" + std::to_string(li) + ".";
            Ly.g_in = reinterpret_cast<float*>(sp(p + "input_layernorm.weight"));
            Ly.g_post = reinterpret_cast<float*>(sp(p + "post_attention_layernorm.weight"));
            Ly.gq = reinterpret_cast<float*>(sp(p + "attention.query_layernorm.weight"));
            Ly.gk = reinterpret_cast<float*>(sp(p + "attention.key_layernorm.weight"));
            const size_t kvb = size_t(NKV) * max_blocks * attn::KV_BLOCK_BYTES;
            CK(cudaMalloc(&Ly.kv, kvb));
            kv_bytes += kvb;
            CK(cudaMalloc(&Ly.kf, TTM * NKV * HD * 2));
            CK(cudaMalloc(&Ly.vf, TTM * NKV * HD * 2));
        }
        if (ix.t.count("lm_head_split.bin:lm_head_hi") && !getenv("LLADA_NO_LM_SPLIT")) {
            // screened readout (s07d): stream the high nibble plane, rescore candidates from both planes
            const Ent& hi = ix.get("lm_head_split.bin", "lm_head_hi");
            const Ent& lo = ix.get("lm_head_split.bin", "lm_head_lo");
            uint8_t* lmw; CK(cudaMalloc(&lmw, hi.bytes + lo.bytes));
            file_to_device(dir + "/lm_head_split.bin", hi.off, hi.bytes, lmw);
            file_to_device(dir + "/lm_head_split.bin", lo.off, lo.bytes, lmw + hi.bytes);
            moe8::relayout_matrix(lmw, V, HID, 4);
            moe8::relayout_matrix(lmw + hi.bytes, V, HID, 4);
            weight_bytes += hi.bytes + lo.bytes;
            lm = single_table(lmw, 0, 0, 5);
            lm_hi = lmw; lm_lo = lmw + hi.bytes; lm_split = true;
            if (getenv("LLADA_LM_CHECK")) {                 // also the exact 8-bit head, for readout comparisons
                const Ent& e = ix.get("lm_head.bin", "lm_head");
                uint8_t* w8; CK(cudaMalloc(&w8, e.bytes));
                file_to_device(dir + "/lm_head.bin", e.off, e.bytes, w8);
                moe8::relayout_matrix(w8, V, HID, 8);
                lm8 = single_table(w8, 0, 0, 8);
            }
        } else {
            const Ent& e = ix.get("lm_head.bin", "lm_head");
            uint8_t* lmw; CK(cudaMalloc(&lmw, e.bytes));
            file_to_device(dir + "/lm_head.bin", e.off, e.bytes, lmw);
            moe8::relayout_matrix(lmw, V, HID, ix.ci("lm_bits"));
            weight_bytes += e.bytes;
            lm = single_table(lmw, 0, 0, ix.ci("lm_bits"));
        }
        {
            const std::string ep = dir + "/embed.bin";
            const int fd = open(ep.c_str(), O_RDONLY);
            if (fd < 0) { fprintf(stderr, "no %s\n", ep.c_str()); exit(1); }
            const size_t eb = file_bytes(ep);
            embed = static_cast<const uint16_t*>(mmap(nullptr, eb, PROT_READ, MAP_SHARED, fd, 0));
            close(fd);
        }
        {
            std::vector<float> f(32);
            for (int i = 0; i < 32; ++i) f[i] = float(1.0 / std::pow(double(theta), 2.0 * i / 64.0));
            inv_freq = dev_copy(f);
        }
        alloc_buffers();
        set_smem();
    }

    void alloc_buffers() {
        int plo, phi; CK(cudaDeviceGetStreamPriorityRange(&plo, &phi));
        const int pmain = phi;                       // the critical path outranks the shared expert
        CK(cudaStreamCreateWithPriority(&st, cudaStreamDefault, pmain));
        CK(cudaStreamCreateWithPriority(&st2, cudaStreamNonBlocking, pmain));
        CK(cudaStreamCreateWithPriority(&st3, cudaStreamNonBlocking, pmain));
        CK(cudaEventCreateWithFlags(&ev_a, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ev_b, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&ev_c, cudaEventDisableTiming));
        CK(cudaStreamCreateWithPriority(&st4, cudaStreamNonBlocking, plo));   // fills SMs routing leaves idle
        CK(cudaEventCreateWithFlags(&ev_s, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&ev_sd, cudaEventDisableTiming));
        CK(cudaMalloc(&h, TTM * HID * 4)); CK(cudaMalloc(&hn, TTM * HID * 4));
        CK(cudaMalloc(&qkv, TTM * 3072 * 4)); CK(cudaMalloc(&rlogits, size_t(TTM) * ops::ROUTER_KC * E * 4));
        CK(cudaMalloc(&q, TTM * NH * HD * 2)); CK(cudaMalloc(&o, TTM * NH * HD * 2));
        const int slots = TTM * TOPK + TTM;
        CK(cudaMalloc(&Z, size_t(std::max(slots * 512, TTM * 5120)) * 2));
        CK(cudaMalloc(&Y, size_t(slots) * HID * 4));
        CK(cudaMalloc(&logits, size_t(T) * V * 4));
        CK(cudaMalloc(&hbuf, TTM * HID * 2));
        max_chunks = 4;                                    // attn::pick_bpc: <= 4 chunks (decode), <= 2 (prefill)
        CK(cudaMalloc(&Opart, size_t(PB) * NKV * max_chunks * 128 * HD * 4));
        CK(cudaMalloc(&Mpart, size_t(PB) * NKV * max_chunks * 128 * 4));
        CK(cudaMalloc(&Lpart, size_t(PB) * NKV * max_chunks * 128 * 4));
        CK(cudaMalloc(&pos, TTM * 4));
        xA = alloc_act(TTM, HID); oA = alloc_act(TTM, HID); zA = alloc_act(slots, 512); z0A = alloc_act(TTM, 5120);
        std::vector<float> one(slots, 1.f);
        ones = dev_copy(one);
        std::vector<int> io(TTM);
        for (int i = 0; i < TTM; ++i) io[i] = i;
        iota = dev_copy(io);
        items_dense = dev_copy(std::vector<moe8::Work>{{0, 0, T}});
        items_shared = dev_copy(std::vector<moe8::Work>{{E, T * TOPK, T}});
        {
            std::vector<int> st_(T * TOPK + T, 0);
            for (int t = 0; t < T; ++t) st_[T * TOPK + t] = t;
            stok_shared = dev_copy(st_);
        }
        items_dense16 = dev_copy(std::vector<moe8::Work>{{0, 0, 16}, {0, 16, 16}});
        std::vector<moe8::Work> pf, pf16;
        for (int j = 0; j < PB; ++j) { pf.push_back({0, 32 * j, 32}); pf16.push_back({0, 32 * j, 16}); pf16.push_back({0, 32 * j + 16, 16}); }
        items_pf = dev_copy(pf); items_pf16 = dev_copy(pf16);
        CK(cudaMalloc(&sched.slot_token, slots * 4)); CK(cudaMalloc(&sched.slot_weight, slots * 4));
        CK(cudaMalloc(&sched.tok_slots, TTM * (TOPK + 1) * 4));
        CK(cudaMalloc(&sched.work1, ops::W1_CAP * sizeof(moe8::Work))); CK(cudaMalloc(&sched.work2, ops::W2_CAP * sizeof(moe8::Work)));
        CK(cudaMalloc(&sched.work4, ops::W4_CAP * sizeof(moe8::Work)));
        CK(cudaMalloc(&sched.nwork, 3 * 4));
        CK(cudaMalloc(&sched.topk, T * TOPK * 4)); CK(cudaMalloc(&sched.topw, T * TOPK * 4));
        CK(cudaMalloc(&sched.bsel, TTM * TOPK * 4)); CK(cudaMalloc(&sched.bselw, TTM * TOPK * 4));
        CK(cudaMalloc(&sched.bmask, PB * E * 4)); CK(cudaMalloc(&sched.bpre, PB * E * 4));
    }

    void set_smem() {
        auto set = [](auto fn, int bytes) { CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)); };
        // every variant at its largest K: NT1/NT2 up to the layer-0 down projection (K = 5120), NT4 up to 2048
        auto ls = [](int NT, int K) { return moe8::smem_bytes(NT, K); };
        set(moe8::gemm<GW, 1, GMS, 0>, ls(1, 5120)); set(moe8::gemm<GW, 1, GMS, 1>, ls(1, 5120));
        set(moe8::gemm<GW, 1, GMS, 2>, ls(1, 5120)); set(moe8::gemm<GW, 2, GMS, 0>, ls(2, 5120));
        set(moe8::gemm<GW, 2, GMS, 1>, ls(2, 5120)); set(moe8::gemm<GW, 2, GMS, 2>, ls(2, 5120));
        set(moe8::gemm<GW, 4, GMS, 0>, ls(4, HID)); set(moe8::gemm<GW, 4, GMS, 1>, ls(4, HID));
        set(moe8::gemm<GW, 4, GMS, 2>, ls(4, HID)); set(moe8::gemm<GW, 4, GMS, 3>, ls(4, HID));
        set(attn::attn_partial, attn::SMEM_BYTES);
        set(ops::router_mma, ops::ROUTER_SMEM);
    }

    template <int NT, int MODE>
    void gemm(const moe8::ExpertTable& tab, const moe8::Work* items, const int* nwork, int grid_y, const int* slot_token,
              const moe8::Act& X, const float* sw, int rows, int K, half* Zo, float* Yo, int ksplit = 1, int groups = 1,
              bool swap = false, cudaStream_t s = nullptr) {
        if (!s) s = st;
        if ((rows / 16) % (GW * GMS * groups) || (K / 128 / ksplit) % 4) { fprintf(stderr, "gemm shape %d x %d groups %d\n", rows, K, groups); exit(1); }
        moe8::GemmArgs g{tab, items, nwork, slot_token, X, sw, MODE == 0 ? K : rows, MODE == 0 ? rows / 2 : K, Zo, Yo, groups,
                         swap};
        if (MODE == 1 && ksplit > 1) CK(cudaMemsetAsync(Yo, 0, size_t(T) * rows * 4, st));
        const int rb = rows / 16 / (GW * GMS * groups);
        const dim3 grid = swap ? dim3(grid_y, rb, ksplit) : dim3(rb, grid_y, ksplit);
        moe8::gemm<GW, NT, GMS, MODE><<<grid, GW * 32, moe8::smem_bytes(NT, K), s>>>(g);
    }

    // embeddings for 32 token ids -> h (fp32)
    // Embedding rows are gathered on the host (the table is an mmap) into a pinned staging buffer, then uploaded.
    // Split so a captured graph can replay the upload while the host refills the buffer before each launch.
    uint16_t* emb_rows = nullptr;                     // pinned [TTM][HID]
    void embed_fill(const int* ids, int n) {
        if (!emb_rows) CK(cudaMallocHost(&emb_rows, size_t(TTM) * HID * 2));
        for (int t = 0; t < n; ++t) std::memcpy(emb_rows + size_t(t) * HID, embed + size_t(ids[t]) * HID, HID * 2);
    }
    void embed_upload(int n) {
        CK(cudaMemcpyAsync(hbuf, emb_rows, size_t(n) * HID * 2, cudaMemcpyHostToDevice, st));
        half2float<<<n * HID / 256, 256, 0, st>>>(hbuf, h, n * HID);
    }
    void embed_block(const int* ids, int n = T) {
        CK(cudaStreamSynchronize(st));                // the staging buffer may still feed an earlier upload
        embed_fill(ids, n);
        embed_upload(n);
    }

    // lm_head for the 32 rows of xA: the split head writes fp16 screening values over `logits` (decode.cu reads
    // them as half), the 8-bit head fp32 logits.
    void lm_gemm() {
        if (lm_split) gemm<4, 3>(lm, items_dense, nullptr, 1, nullptr, xA.a, ones, V, HID, reinterpret_cast<half*>(logits), nullptr, 1, 4);
        else gemm<4, 1>(lm, items_dense, nullptr, 1, nullptr, xA.a, ones, V, HID, nullptr, logits, 1, 4);
    }

    // Optional section profiler: mark(k) records an event; time since the previous mark is charged to section k.
    std::vector<std::pair<int, cudaEvent_t>> prof_ev;
    std::vector<cudaEvent_t> prof_pool;
    bool profiling = false;
    std::vector<double> prof_acc;
    void mark(int k) {
        if (!profiling) return;
        if (prof_ev.size() == prof_pool.size()) { cudaEvent_t ev; CK(cudaEventCreate(&ev)); prof_pool.push_back(ev); }
        cudaEvent_t ev = prof_pool[prof_ev.size()];
        CK(cudaEventRecord(ev, st));
        prof_ev.push_back({k, ev});
    }
    // sync and add the marked sections' ms into acc[k]
    void prof_collect() {
        std::vector<double>& acc = prof_acc;
        CK(cudaStreamSynchronize(st));
        for (size_t i = 1; i < prof_ev.size(); ++i) {
            float ms; CK(cudaEventElapsedTime(&ms, prof_ev[i - 1].second, prof_ev[i].second));
            if (int(acc.size()) <= prof_ev[i].first) acc.resize(prof_ev[i].first + 1);
            acc[prof_ev[i].first] += ms;
        }
        prof_ev.clear();
    }

    // st2 / st3 join st's timeline (fork) and st waits for their work (join)
    void fork() { CK(cudaEventRecord(ev_a, st)); CK(cudaStreamWaitEvent(st2, ev_a, 0)); CK(cudaStreamWaitEvent(st3, ev_a, 0)); }
    void join() {
        CK(cudaEventRecord(ev_b, st2)); CK(cudaStreamWaitEvent(st, ev_b, 0));
        CK(cudaEventRecord(ev_c, st3)); CK(cudaStreamWaitEvent(st, ev_c, 0));
    }

    // xA already holds this layer's input (the previous MoE layer's fused combine + norm): reset per forward
    bool in_ready = false;
    void layer_step(int li, int ctx_blocks) {
        Layer& Ly = layers[li];
        mark(0);
        // ---- attention
        if (li == 0) in_ready = false;
        if (!in_ready) ops::norm_rot_q8<<<T, 256, 0, st>>>(h, signs, Ly.g_in, eps, xA.a, nullptr);
        in_ready = false;
        gemm<4, 1>(Ly.qkv, items_dense, nullptr, 1, nullptr, xA.a, ones, 3072, HID, nullptr, qkv, 1, 2);
        mark(1);
        ops::qk_norm_rope<<<dim3(T, 3), 256, 0, st>>>(qkv, pos, Ly.gq, Ly.gk, eps, inv_freq, q, Ly.kf, Ly.vf);
        mark(2);
        const int bpc = attn::pick_bpc(ctx_blocks, 1), nchunks = attn::chunks_for(ctx_blocks, bpc);
        attn::Args aa{q, Ly.kf, Ly.vf, Ly.kv, max_blocks, ctx_blocks, bpc, nchunks, 0.08838834764831845f, Opart, Mpart, Lpart};
        attn::attn_partial<<<dim3(NKV, nchunks, 1), attn::ATT_NTH, attn::SMEM_BYTES, st>>>(aa);
        attn::attn_combine<<<dim3(NKV, 128 / attn::COMB_ROWS, 1), 32 * attn::COMB_ROWS, 0, st>>>(Opart, Mpart, Lpart, nchunks, ctx_blocks, bpc, o, oA.q, oA.s, oA.sum);
        mark(3);
        gemm<4, 2>(Ly.dense, items_dense, nullptr, 1, nullptr, oA.a, ones, HID, HID, nullptr, h, 1, 2);
        mark(4);
        // ---- MLP
        if (!Ly.moe) {
            ops::norm_rot_q8<<<T, 256, 0, st>>>(h, signs, Ly.g_post, eps, xA.a, nullptr);
            gemm<4, 0>(Ly.mlp, items_dense, nullptr, 1, iota, xA.a, ones, 2 * Ly.I, HID, Z, nullptr, 1, 4);
            ops::quant_rows_q8<half><<<T, 256, 0, st>>>(Z, Ly.I, z0A.a);
            gemm<2, 2>(Ly.mlp, items_dense16, nullptr, 2, nullptr, z0A.a, ones, HID, Ly.I, nullptr, h, 1, 2);
            mark(5);
            return;
        }
        ops::norm_rot_q8<<<T, 256, 0, st>>>(h, signs, Ly.g_post, eps, xA.a, hn);
        mark(16);
        {   // shared expert (slots 256 .. 287, weight 1) on st4: needs only x', so it streams while routing runs
            const int S0 = T * TOPK;
            CK(cudaEventRecord(ev_s, st)); CK(cudaStreamWaitEvent(st4, ev_s, 0));
            gemm<4, 0>(Ly.experts, items_shared, nullptr, 1, stok_shared, xA.a, ones, 2 * Ly.I, HID, Z, nullptr, 1, 2, false, st4);
            const moe8::Act zs{zA.q + size_t(S0) * 512, zA.s + size_t(S0) * 4, zA.sum + size_t(S0) * 4};
            moe8::fwht512_q8<<<T / 8, 256, 0, st4>>>(Z + size_t(S0) * 512, zs, T);
            gemm<4, 1>(Ly.experts, items_shared, nullptr, 1, nullptr, zA.a, ones, HID, Ly.I, nullptr, Y, 1, 4, false, st4);
            CK(cudaEventRecord(ev_sd, st4));
        }
        ops::router_mma<<<dim3(ops::ROUTER_KC, 1), 256, ops::ROUTER_SMEM, st>>>(hn, Ly.router, rlogits);
        mark(17);
        ops::route_schedule<<<1, 256, 0, st>>>(rlogits, Ly.bias, rscale, sched);
        mark(6);
        // grid_y caps: at most 48 items in all; NT2 <= 28 (256 / 9), NT4 <= 15 (256 / 17)
        fork();
        gemm<4, 0>(Ly.experts, sched.work4, sched.nwork + 2, 16, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 2, false, st3);
        gemm<2, 0>(Ly.experts, sched.work2, sched.nwork + 1, 29, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 2, false, st2);
        gemm<1, 0>(Ly.experts, sched.work1, sched.nwork, 48, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 4);
        join();
        mark(7);
        moe8::fwht512_q8<<<T * TOPK / 8, 256, 0, st>>>(Z, zA.a, T * TOPK);      // routed slots
        mark(8);
        fork();
        gemm<4, 1>(Ly.experts, sched.work4, sched.nwork + 2, 16, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 4, false, st3);
        gemm<2, 1>(Ly.experts, sched.work2, sched.nwork + 1, 29, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 2, false, st2);
        gemm<1, 1>(Ly.experts, sched.work1, sched.nwork, 48, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 8);
        join();
        mark(9);
        CK(cudaStreamWaitEvent(st, ev_sd, 0));               // the shared expert's slots
        // + the next input norm (next layer, or the final norm: γ is folded into the weights either way)
        ops::moe_combine_norm<<<T, 256, 0, st>>>(Y, sched.tok_slots, TOPK + 1, signs, h, eps, xA.a);
        in_ready = true;
        mark(10);
    }

    // Full block forward. pos0: position of the block's first token; logits: compute lm_head for all 32 rows.
    // Positions are set by a kernel (its argument is captured at launch): a host staging buffer reused by
    // back-to-back async copies would race -- consecutive prefill commit passes do not synchronize.
    void begin_block(const int* ids, int pos0) {
        set_positions<<<1, T, 0, st>>>(pos, pos0);
        embed_block(ids);
    }

    // ---- CUDA graph of one denoising step (forward + caller's tail). Built once; between blocks only the
    // position kernel and the attention launches (context length) change, patched in the executable graph.
    struct StepGraph {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        std::vector<std::pair<cudaGraphNode_t, attn::Args>> partial;   // one per layer
        std::vector<cudaGraphNode_t> combine, setpos;
        int ctx = -1;
    };
    // Captures: positions, embedding upload (from emb_rows, filled by the caller), all layers, final norm and
    // lm_head, then tail() (the decoder's logits reduction and copy-out).
    template <class Tail>
    void capture_step(StepGraph& G, int ctx_blocks, Tail&& tail) {
        CK(cudaStreamSynchronize(st));
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        set_positions<<<1, T, 0, st>>>(pos, ctx_blocks * T);
        embed_upload(T);
        for (int li = 0; li < nlayers; ++li) layer_step(li, ctx_blocks);
        if (!in_ready) ops::norm_rot_q8<<<T, 256, 0, st>>>(h, signs, final_norm, eps, xA.a, nullptr);
        in_ready = false;
        lm_gemm();
        tail();
        CK(cudaStreamEndCapture(st, &g));
        size_t n = 0;
        CK(cudaGraphGetNodes(g, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        CK(cudaGraphGetNodes(g, nodes.data(), &n));
        for (auto nd : nodes) {
            cudaGraphNodeType ty;
            CK(cudaGraphNodeGetType(nd, &ty));
            if (ty != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams p;
            CK(cudaGraphKernelNodeGetParams(nd, &p));
            if (p.func == reinterpret_cast<void*>(attn::attn_partial))
                G.partial.push_back({nd, *static_cast<attn::Args*>(p.kernelParams[0])});
            else if (p.func == reinterpret_cast<void*>(attn::attn_combine)) G.combine.push_back(nd);
            else if (p.func == reinterpret_cast<void*>(set_positions)) G.setpos.push_back(nd);
        }
        CK(cudaGraphInstantiate(&G.exec, g, 0));
        G.graph = g;                                     // node handles stay valid only while the graph lives
        G.ctx = ctx_blocks;
    }
    void update_step(StepGraph& G, int ctx_blocks) {
        if (G.ctx == ctx_blocks) return;
        const int bpc = attn::pick_bpc(ctx_blocks, 1), nchunks = attn::chunks_for(ctx_blocks, bpc);
        for (auto& [nd, a0] : G.partial) {
            attn::Args a = a0;
            a.nblocks = ctx_blocks; a.blocks_per_chunk = bpc; a.nchunks = nchunks;
            void* args[] = {&a};
            cudaKernelNodeParams p{reinterpret_cast<void*>(attn::attn_partial), dim3(NKV, nchunks, 1), dim3(attn::ATT_NTH),
                                   unsigned(attn::SMEM_BYTES), args, nullptr};
            CK(cudaGraphExecKernelNodeSetParams(G.exec, nd, &p));
        }
        for (auto nd : G.combine) {
            const float *op = Opart, *mp = Mpart, *lp = Lpart;
            half* oo = o;
            int nc = nchunks, nb = ctx_blocks, bp = bpc;
            int8_t* q8 = oA.q; float* s8 = oA.s; int* sum8 = oA.sum;
            void* args[] = {&op, &mp, &lp, &nc, &nb, &bp, &oo, &q8, &s8, &sum8};
            cudaKernelNodeParams p{reinterpret_cast<void*>(attn::attn_combine), dim3(NKV, 128 / attn::COMB_ROWS, 1),
                                   dim3(32 * attn::COMB_ROWS), 0, args, nullptr};
            CK(cudaGraphExecKernelNodeSetParams(G.exec, nd, &p));
        }
        for (auto nd : G.setpos) {
            int* pp = pos;
            int p0 = ctx_blocks * T;
            void* args[] = {&pp, &p0};
            cudaKernelNodeParams p{reinterpret_cast<void*>(set_positions), dim3(1), dim3(T), 0, args, nullptr};
            CK(cudaGraphExecKernelNodeSetParams(G.exec, nd, &p));
        }
        G.ctx = ctx_blocks;
    }

    // Batched prefill: nb <= PB full blocks of prompt ids, committed as cache blocks ctx0 .. ctx0 + nb - 1. Layer by
    // layer: all nb blocks' K/V are committed first, then block j attends cache blocks < ctx0 + j plus its own fresh
    // K/V -- exactly what nb sequential commit passes compute. Weights stream once per pass instead of once per block.
    void dbg(int line) {
        static const bool on = getenv("LLADA_SYNC") != nullptr;
        if (!on) return;
        cudaError_t e = cudaStreamSynchronize(st);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "prefill: error after line %d: %s\n", line, cudaGetErrorString(e)); exit(1); }
    }
    void prefill(const int* ids, int nb, int ctx0) {
        const int TT = nb * T;
        mark(14);
        set_positions<<<1, TT, 0, st>>>(pos, ctx0 * T);
        dbg(__LINE__);
        embed_block(ids, TT);
        mark(15);
        for (int li = 0; li < nlayers; ++li) {
            Layer& Ly = layers[li];
            mark(0);
            ops::norm_rot_q8<<<TT, 256, 0, st>>>(h, signs, Ly.g_in, eps, xA.a, nullptr);
            dbg(__LINE__);
            gemm<4, 1>(Ly.qkv, items_pf, nullptr, nb, nullptr, xA.a, ones, 3072, HID, nullptr, qkv, 1, 2, true);
            dbg(__LINE__);
            mark(1);
            ops::qk_norm_rope<<<dim3(TT, 3), 256, 0, st>>>(qkv, pos, Ly.gq, Ly.gk, eps, inv_freq, q, Ly.kf, Ly.vf);
            dbg(__LINE__);
            attn::kv_commit<<<dim3(NKV, nb), 128, 0, st>>>(Ly.kf, Ly.vf, Ly.kv, max_blocks, ctx0);
            dbg(__LINE__);
            mark(2);
            const int bpc = attn::pick_bpc(ctx0 + nb - 1, nb), nchunks = attn::chunks_for(ctx0 + nb - 1, bpc);
            attn::Args aa{q, Ly.kf, Ly.vf, Ly.kv, max_blocks, ctx0, bpc, nchunks, 0.08838834764831845f, Opart, Mpart, Lpart};
            attn::attn_partial<<<dim3(NKV, nchunks, nb), attn::ATT_NTH, attn::SMEM_BYTES, st>>>(aa);
            dbg(__LINE__);
            attn::attn_combine<<<dim3(NKV, 128 / attn::COMB_ROWS, nb), 32 * attn::COMB_ROWS, 0, st>>>(Opart, Mpart, Lpart, nchunks, ctx0, bpc, o, oA.q, oA.s, oA.sum);
            dbg(__LINE__);
            mark(3);
            gemm<4, 2>(Ly.dense, items_pf, nullptr, nb, nullptr, oA.a, ones, HID, HID, nullptr, h, 1, 2, true);
            dbg(__LINE__);
            mark(4);
            if (!Ly.moe) {
                ops::norm_rot_q8<<<TT, 256, 0, st>>>(h, signs, Ly.g_post, eps, xA.a, nullptr);
                dbg(__LINE__);
                gemm<4, 0>(Ly.mlp, items_pf, nullptr, nb, iota, xA.a, ones, 2 * Ly.I, HID, Z, nullptr, 1, 4, true);
                dbg(__LINE__);
                ops::quant_rows_q8<half><<<TT, 256, 0, st>>>(Z, Ly.I, z0A.a);
                dbg(__LINE__);
                gemm<2, 2>(Ly.mlp, items_pf16, nullptr, 2 * nb, nullptr, z0A.a, ones, HID, Ly.I, nullptr, h, 1, 2, true);
                dbg(__LINE__);
                mark(5);
                continue;
            }
            ops::norm_rot_q8<<<TT, 256, 0, st>>>(h, signs, Ly.g_post, eps, xA.a, hn);
            dbg(__LINE__);
            mark(16);
            ops::router_mma<<<dim3(ops::ROUTER_KC, nb), 256, ops::ROUTER_SMEM, st>>>(hn, Ly.router, rlogits);
            dbg(__LINE__);
            mark(17);
            ops::route_batch_select<<<nb, 256, 0, st>>>(rlogits, Ly.bias, rscale, sched);
            dbg(__LINE__);
            ops::route_batch_plan<<<1, 256, 0, st>>>(nb, E, sched);
            dbg(__LINE__);
            ops::route_batch_place<<<nb, 256, 0, st>>>(nb, sched);
            dbg(__LINE__);
            mark(6);
            const int slots = TT * TOPK + TT;
            gemm<1, 0>(Ly.experts, sched.work1, sched.nwork, ops::W1_CAP, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 4, true);
            dbg(__LINE__);
            gemm<2, 0>(Ly.experts, sched.work2, sched.nwork + 1, ops::W2_CAP, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 2, true);
            dbg(__LINE__);
            gemm<4, 0>(Ly.experts, sched.work4, sched.nwork + 2, ops::W4_CAP, sched.slot_token, xA.a, sched.slot_weight, 2 * Ly.I, HID, Z, nullptr, 1, 2, true);
            dbg(__LINE__);
            mark(7);
            moe8::fwht512_q8<<<(slots + 7) / 8, 256, 0, st>>>(Z, zA.a, slots);
            dbg(__LINE__);
            mark(8);
            gemm<1, 1>(Ly.experts, sched.work1, sched.nwork, ops::W1_CAP, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 8, true);
            dbg(__LINE__);
            gemm<2, 1>(Ly.experts, sched.work2, sched.nwork + 1, ops::W2_CAP, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 4, true);
            dbg(__LINE__);
            gemm<4, 1>(Ly.experts, sched.work4, sched.nwork + 2, ops::W4_CAP, nullptr, zA.a, sched.slot_weight, HID, Ly.I, nullptr, Y, 1, 4, true);
            dbg(__LINE__);
            mark(9);
            ops::moe_combine_unrot<<<TT, 256, 0, st>>>(Y, sched.tok_slots, TOPK + 1, signs, h, TT);
            dbg(__LINE__);
            mark(10);
        }
    }

    // Quantize the fresh block's K/V (left by the last forward) into the cache as block ctx_blocks.
    void commit_kv(int ctx_blocks) {
        for (int li = 0; li < nlayers; ++li)
            attn::kv_commit<<<NKV, 128, 0, st>>>(layers[li].kf, layers[li].vf, layers[li].kv, max_blocks, ctx_blocks);
    }

    void forward_block(const int* ids, int pos0, int ctx_blocks, bool commit, bool want_logits) {
        mark(14);
        begin_block(ids, pos0);
        mark(15);
        for (int li = 0; li < nlayers; ++li) layer_step(li, ctx_blocks);
        if (commit) commit_kv(ctx_blocks);
        const bool final_ready = in_ready;
        in_ready = false;
        mark(11);
        if (want_logits) {
            if (!final_ready) ops::norm_rot_q8<<<T, 256, 0, st>>>(h, signs, final_norm, eps, xA.a, nullptr);
            lm_gemm();
            mark(12);
        }
    }
};

}  // namespace eng
