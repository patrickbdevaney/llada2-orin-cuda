// Block decoding for LLaDA2.2-mini (greedy JointThresholdInDel, the model card's serving default): a C++ port of
// modeling_llada2_moe.py's _joint_decode_block with temperature 0, on top of eng::Engine.
//
// Per step the GPU reduces each block row's logits to what the scheduler can ever ask for -- the top-2 tokens with
// their softmax probabilities and the best token that is neither DELETE nor SPLIT -- so only 32 small records cross
// to the host, not 32 x 157,184 logits.
#include <algorithm>
#include <random>
#include <set>
#include "engine.cu"

namespace dec {

struct RowTop {
    int i1, i2, isd;      // argmax, runner-up, best excluding DELETE/SPLIT
    float p1, p2, psd;    // their probabilities under the full softmax
};

// Per-row partial record over a vocabulary slice: top-2 (ties: lower index), best non-DELETE/SPLIT, and the
// sum of exp(x - m1) over the slice.
struct TopRec { float m1, m2, msd, s; int i1, i2, isd; };

__device__ __forceinline__ void rec_merge(TopRec& a, const TopRec& b) {
    const float M = fmaxf(a.m1, b.m1);
    const float s = (a.m1 == -INFINITY ? 0.f : a.s * __expf(a.m1 - M)) + (b.m1 == -INFINITY ? 0.f : b.s * __expf(b.m1 - M));
    TopRec x = a, y = b;
    if (y.m1 > x.m1 || (y.m1 == x.m1 && y.i1 < x.i1)) { TopRec t = x; x = y; y = t; }   // x holds the max
    const bool y1second = y.m1 > x.m2 || (y.m1 == x.m2 && y.i1 < x.i2);
    a.m1 = x.m1; a.i1 = x.i1;
    a.m2 = y1second ? y.m1 : x.m2; a.i2 = y1second ? y.i1 : x.i2;
    const bool sd = b.msd > a.msd || (b.msd == a.msd && b.isd < a.isd);
    a.msd = sd ? b.msd : a.msd; a.isd = sd ? b.isd : a.isd;
    a.s = s;
}

constexpr int TOP_SPLIT = 16;
// grid (T rows, TOP_SPLIT slices), block 256: partial records -> part[row][slice]
__global__ void logits_top_part(const float* logits, int V, int del_id, int split_id, TopRec* part) {
    const int row = blockIdx.x, sl = blockIdx.y, n = (V + TOP_SPLIT - 1) / TOP_SPLIT;
    const int v0 = sl * n, v1 = min(V, v0 + n);
    const float* l = logits + size_t(row) * V;
    TopRec r{-INFINITY, -INFINITY, -INFINITY, 0.f, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    for (int v = v0 + threadIdx.x; v < v1; v += blockDim.x) {
        const float x = l[v];
        if (x > r.m1) { r.s = r.s * __expf(r.m1 - x) + 1.f; r.m2 = r.m1; r.i2 = r.i1; r.m1 = x; r.i1 = v; }
        else { r.s += __expf(x - r.m1); if (x > r.m2) { r.m2 = x; r.i2 = v; } }
        if (v != del_id && v != split_id && x > r.msd) { r.msd = x; r.isd = v; }
    }
    __shared__ TopRec sm[256];
    sm[threadIdx.x] = r;
    __syncthreads();
    for (int w = 128; w; w >>= 1) {
        if (threadIdx.x < w) rec_merge(sm[threadIdx.x], sm[threadIdx.x + w]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[row * TOP_SPLIT + sl] = sm[0];
}

// grid T rows, block 32: merge the slices -> RowTop (probabilities under the full softmax)
__global__ void logits_top_merge(const TopRec* part, RowTop* out) {
    const int row = blockIdx.x;
    __shared__ TopRec sm[TOP_SPLIT];
    if (threadIdx.x < TOP_SPLIT) sm[threadIdx.x] = part[row * TOP_SPLIT + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        TopRec r = sm[0];
        for (int i = 1; i < TOP_SPLIT; ++i) rec_merge(r, sm[i]);     // fixed order: deterministic
        const float lse = r.m1 + __logf(r.s);
        out[row] = RowTop{r.i1, r.i2, r.isd, __expf(r.m1 - lse), __expf(r.m2 - lse), __expf(r.msd - lse)};
    }
}

// ---- screened readout (engine lm_split): logits are approximations from the high nibble plane; every row keeps
// its top-8 and the full softmax sum, then the 8 candidates are rescored exactly from both planes.
constexpr int NCAND = 8;
struct CandRec { float v[NCAND]; int i[NCAND]; float m, s; };   // v sorted desc; s = sum exp(x - m)

// grid (T rows, TOP_SPLIT slices), 256 threads: per thread its top-2 + sum-exp, then 8 rounds of block argmax.
// logits: fp16 screening values [T][V]; V % (8 * TOP_SPLIT) == 0.
__global__ void cand_part(const half* logits, int V, CandRec* part) {
    const int row = blockIdx.x, sl = blockIdx.y, n = V / TOP_SPLIT;
    const int v0 = sl * n, v1 = v0 + n, tid = threadIdx.x;
    const half* l = logits + size_t(row) * V;
    // per thread: its top-2 (a row's best 8 sit in 4096 interleaved thread slots, so losing one to a 3-way
    // collision is negligible -- and every survivor is rescored exactly afterwards)
    float tv[2] = {-INFINITY, -INFINITY};
    int ti[2] = {0x7fffffff, 0x7fffffff};
    float m = -INFINITY, sum = 0.f;
    for (int v = v0 + 8 * tid; v < v1; v += 8 * blockDim.x) {   // 8 halves per step
        const uint4 raw = *reinterpret_cast<const uint4*>(l + v);
        const half2* h2 = reinterpret_cast<const half2*>(&raw);
        float xs[8];
#pragma unroll
        for (int k = 0; k < 4; ++k) { const float2 f = __half22float2(h2[k]); xs[2 * k] = f.x; xs[2 * k + 1] = f.y; }
        float m8 = xs[0];
#pragma unroll
        for (int k = 1; k < 8; ++k) m8 = fmaxf(m8, xs[k]);
        if (m8 > m) { sum *= __expf(m - m8); m = m8; }
#pragma unroll
        for (int k = 0; k < 8; ++k) sum += __expf(xs[k] - m);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const float x = xs[k];
            if (x > tv[1]) {
                if (x > tv[0]) { tv[1] = tv[0]; ti[1] = ti[0]; tv[0] = x; ti[0] = v + k; }
                else { tv[1] = x; ti[1] = v + k; }
            }
        }
    }
    __shared__ float sm_m[256], sm_s[256];
    __shared__ float wv[8 * NCAND]; __shared__ int wi[8 * NCAND];
    __shared__ CandRec out;
    sm_m[tid] = m; sm_s[tid] = sum;
    // per warp: pop its top-NCAND from the lanes' sorted lists (shuffles only)
    const int lane = tid & 31, warp = tid >> 5;
    int head = 0;
    for (int r = 0; r < NCAND; ++r) {
        float x = head == 0 ? tv[0] : head == 1 ? tv[1] : -INFINITY;
        int xi = head == 0 ? ti[0] : head == 1 ? ti[1] : 0x7fffffff, who = lane;
        for (int o = 16; o; o >>= 1) {
            const float y = __shfl_xor_sync(0xffffffffu, x, o);
            const int yi = __shfl_xor_sync(0xffffffffu, xi, o), yw = __shfl_xor_sync(0xffffffffu, who, o);
            if (y > x || (y == x && yi < xi)) { x = y; xi = yi; who = yw; }
        }
        if (lane == 0) { wv[warp * NCAND + r] = x; wi[warp * NCAND + r] = xi; }
        if (lane == who) ++head;
    }
    __syncthreads();
    if (warp == 0) {                                     // sum-exp over the block, then top-NCAND of the 8 warp lists
        float M = -INFINITY;
        for (int k = lane; k < 256; k += 32) M = fmaxf(M, sm_m[k]);
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
        float S = 0.f;
        for (int k = lane; k < 256; k += 32) S += sm_m[k] == -INFINITY ? 0.f : sm_s[k] * __expf(sm_m[k] - M);
        for (int o = 16; o; o >>= 1) S += __shfl_xor_sync(0xffffffffu, S, o);
        // lane l < 8 walks warp l's sorted list
        int hd = 0;
        for (int r = 0; r < NCAND; ++r) {
            float x = lane < 8 && hd < NCAND ? wv[lane * NCAND + hd] : -INFINITY;
            int xi = lane < 8 && hd < NCAND ? wi[lane * NCAND + hd] : 0x7fffffff, who = lane;
            for (int o = 16; o; o >>= 1) {
                const float y = __shfl_xor_sync(0xffffffffu, x, o);
                const int yi = __shfl_xor_sync(0xffffffffu, xi, o), yw = __shfl_xor_sync(0xffffffffu, who, o);
                if (y > x || (y == x && yi < xi)) { x = y; xi = yi; who = yw; }
            }
            if (lane == 0) { out.v[r] = x; out.i[r] = xi; }
            if (lane == who) ++hd;
        }
        if (lane == 0) { out.m = M; out.s = S; }
    }
    __syncthreads();
    if (tid == 0) part[row * TOP_SPLIT + sl] = out;
}

// exact logit of row r of the split lm_head for activation row t (int8, fragment order, per-group scale and sum)
__device__ float exact_logit(const uint8_t* hi, const uint8_t* lo, const moe8::Act& x, int t, int r, int lane) {
    constexpr int TB = moe8::tile_bytes(4), KT = 2048 / 128;
    const int S = r >> 4, i = r & 15, g8 = i & 7, h8 = i >> 3;
    float acc = 0.f;
    for (int gk = 0; gk < KT; ++gk) {                 // device strip layout: codes of all tiles, then their metas
        const uint8_t* th = hi + size_t(S) * KT * TB + gk * 1024;
        const uint8_t* tl = lo + size_t(S) * KT * TB + gk * 1024;
        const uint8_t* ml = lo + size_t(S) * KT * TB + moe8::strip_meta_off(KT, 4) + gk * moe8::META;
        int P = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int c = lane * 4 + b, j = c >> 5, rem = c & 31, half16 = rem >> 4, tt = (rem & 15) >> 2, bb = rem & 3;
            const int word = 2 * j + half16, off = ((word >> 2) * 32 + g8 * 4 + tt) * 16 + (word & 3) * 4 + bb;
            const int code = (((th[off] >> (4 * h8)) & 15) << 4) | ((tl[off] >> (4 * h8)) & 15);
            P += code * int(x.q[size_t(t) * 2048 + moe8::frag_pos_i8(gk * 128 + c)]);
        }
        const float sw = __half2float(*reinterpret_cast<const half*>(ml + 4 * g8 + 2 * h8));
        const float sx = x.s[t * KT + gk];
        const int z = ml[32 + 2 * g8 + h8];
        acc += sw * sx * float(P);
        if (lane == 0) acc -= sw * sx * float(z * x.sum[t * KT + gk]);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// grid T rows, 256 threads (warp k rescores candidate k): merge the slices' candidates, rescore exactly, correct
// the softmax sum (screening terms of the candidates replaced by exact ones), and emit the RowTop.
__global__ void cand_finish(const CandRec* part, const uint8_t* hi, const uint8_t* lo, moe8::Act x, int del_id,
                            int split_id, RowTop* out) {
    const int row = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float cv[NCAND], ce[NCAND];
    __shared__ int ci[NCAND];
    __shared__ float Ms, Ss;
    __shared__ CandRec pr[TOP_SPLIT];
    for (int k = tid; k < TOP_SPLIT * int(sizeof(CandRec) / 4); k += blockDim.x)
        reinterpret_cast<int*>(pr)[k] = reinterpret_cast<const int*>(part + row * TOP_SPLIT)[k];
    __syncthreads();
    if (tid == 0) {                                      // merge TOP_SPLIT sorted lists (fixed order: deterministic)
        float M = -INFINITY;
        for (int k = 0; k < TOP_SPLIT; ++k) M = fmaxf(M, pr[k].m);
        float S = 0.f;
        int hd[TOP_SPLIT];
        for (int k = 0; k < TOP_SPLIT; ++k) { S += pr[k].s * __expf(pr[k].m - M); hd[k] = 0; }
        for (int r = 0; r < NCAND; ++r) {
            int bk = -1; float bv = -INFINITY; int bi = 0x7fffffff;
            for (int k = 0; k < TOP_SPLIT; ++k) {
                if (hd[k] >= NCAND) continue;
                const CandRec& c = pr[k];
                if (c.v[hd[k]] > bv || (c.v[hd[k]] == bv && c.i[hd[k]] < bi)) { bv = c.v[hd[k]]; bi = c.i[hd[k]]; bk = k; }
            }
            cv[r] = bv; ci[r] = bi; ++hd[bk];
        }
        Ms = M; Ss = S;
    }
    __syncthreads();
    const float e = exact_logit(hi, lo, x, row, ci[warp], lane);
    if (lane == 0) ce[warp] = e;
    __syncthreads();
    if (tid == 0) {
        const float M = Ms;
        float S = Ss;
        for (int k = 0; k < NCAND; ++k) S += __expf(ce[k] - M) - __expf(cv[k] - M);
        int a1 = -1, a2 = -1, asd = -1;
        for (int k = 0; k < NCAND; ++k) {
            auto better = [&](int u, int w) { return w < 0 || ce[u] > ce[w] || (ce[u] == ce[w] && ci[u] < ci[w]); };
            if (better(k, a1)) { a2 = a1; a1 = k; } else if (better(k, a2)) a2 = k;
            if (ci[k] != del_id && ci[k] != split_id && better(k, asd)) asd = k;
        }
        const float lse = M + __logf(S);
        out[row] = RowTop{ci[a1], ci[a2], ci[asd], __expf(ce[a1] - lse), __expf(ce[a2] - lse), __expf(ce[asd] - lse)};
    }
}

struct Params {
    float threshold = 0.5f, editing_threshold = 0.0f;
    int steps = 32, max_post_steps = 16, max_steps_per_block = 1000;
    int mask_id = 156895, delete_id = 156930, split_id = 156931, eos_id = 156892;
};

struct Stats { long steps = 0, blocks = 0, commits = 0, kv_only_commits = 0, prefill_passes = 0; double gpu_ms = 0; };

class Decoder {
public:
    eng::Engine& e;
    Params P;
    Stats S;
    RowTop* d_top;
    TopRec* d_part;
    CandRec* d_cand;
    RowTop* h_top;
    std::mt19937 rng{0};
    bool trace = getenv("LLADA_TRACE") != nullptr;
    int ctx_blocks = 0;                      // committed blocks
    int reused_blocks = 0;                   // prefix-cache hits of the last generate()
    std::vector<int> committed;              // all committed tokens (for the caller)
    std::vector<int> last_fwd;               // ids of the last denoising forward of the current block

    explicit Decoder(eng::Engine& en) : e(en) {
        CK(cudaMalloc(&d_top, eng::T * sizeof(RowTop)));
        CK(cudaMalloc(&d_part, eng::T * TOP_SPLIT * sizeof(TopRec)));
        CK(cudaMalloc(&d_cand, eng::T * TOP_SPLIT * sizeof(CandRec)));
        CK(cudaMallocHost(&h_top, eng::T * sizeof(RowTop)));
    }

    void reset() { ctx_blocks = 0; committed.clear(); last_fwd.clear(); }

    std::vector<int> generate(const std::vector<int>& prompt, int max_new, bool eos_stop = true) {
        return generate(prompt, max_new, eos_stop, [](const int*, int) {});
    }

    // If the last denoising forward ran on exactly these ids (the block ended stable), its fresh K/V are already
    // the block's final K/V: only the cache quantization runs. Otherwise a full commit pass.
    void commit(const std::vector<int>& blk) {
        if (last_fwd == blk) {
            e.commit_kv(ctx_blocks);
            ++S.kv_only_commits;
        } else {
            e.forward_block(blk.data(), ctx_blocks * eng::T, ctx_blocks, true, false);
        }
        last_fwd.clear();
        if (e.profiling) e.prof_collect();
        ++ctx_blocks; ++S.commits;
        committed.insert(committed.end(), blk.begin(), blk.end());
    }

    void logits_tail() {
        if (e.lm_split) {
            cand_part<<<dim3(eng::T, TOP_SPLIT), 256, 0, e.st>>>(reinterpret_cast<const half*>(e.logits), e.V, d_cand);
            cand_finish<<<eng::T, 32 * NCAND, 0, e.st>>>(d_cand, e.lm_hi, e.lm_lo, e.xA.a, P.delete_id, P.split_id, d_top);
        } else {
            logits_top_part<<<dim3(eng::T, TOP_SPLIT), 256, 0, e.st>>>(e.logits, e.V, P.delete_id, P.split_id, d_part);
            logits_top_merge<<<eng::T, 32, 0, e.st>>>(d_part, d_top);
        }
        e.mark(13);
        CK(cudaMemcpyAsync(h_top, d_top, eng::T * sizeof(RowTop), cudaMemcpyDeviceToHost, e.st));
    }

    // LLADA_LM_CHECK: exact 8-bit logits for the same activations, compared with the screened RowTop
    long ck_rows = 0, ck_i1 = 0, ck_i2 = 0; double ck_dp = 0, ck_dpmax = 0;
    void check_readout() {
        std::vector<RowTop> scr(h_top, h_top + eng::T);
        e.gemm<4, 1>(e.lm8, e.items_dense, nullptr, 1, nullptr, e.xA.a, e.ones, e.V, eng::HID, nullptr, e.logits, 1, 4);
        logits_top_part<<<dim3(eng::T, TOP_SPLIT), 256, 0, e.st>>>(e.logits, e.V, P.delete_id, P.split_id, d_part);
        logits_top_merge<<<eng::T, 32, 0, e.st>>>(d_part, d_top);
        CK(cudaMemcpyAsync(h_top, d_top, eng::T * sizeof(RowTop), cudaMemcpyDeviceToHost, e.st));
        CK(cudaStreamSynchronize(e.st));
        for (int i = 0; i < eng::T; ++i) {
            ++ck_rows; ck_i1 += scr[i].i1 == h_top[i].i1; ck_i2 += scr[i].i2 == h_top[i].i2;
            const double d = std::fabs(double(scr[i].p1) - h_top[i].p1);
            ck_dp += d; ck_dpmax = std::max(ck_dpmax, d);
        }
        std::copy(scr.begin(), scr.end(), h_top);              // decode continues with the screened readout
        if (S.steps % 50 == 49)
            fprintf(stderr, "readout check: %ld rows, top-1 agree %.5f, top-2 agree %.5f, |dp1| mean %.2e max %.2e\n",
                    ck_rows, double(ck_i1) / ck_rows, double(ck_i2) / ck_rows, ck_dp / ck_rows, ck_dpmax);
    }

    // One denoising step. Normally a replay of the captured step graph (the profiler and LLADA_NOGRAPH use plain
    // launches); the host only gathers the embedding rows and patches the context length when it changes.
    eng::Engine::StepGraph G;
    bool use_graph = getenv("LLADA_NOGRAPH") == nullptr;
    void step_logits(const std::vector<int>& blk) {
        if (use_graph && !e.profiling) {
            CK(cudaStreamSynchronize(e.st));
            e.embed_fill(blk.data(), eng::T);
            if (!G.exec) e.capture_step(G, ctx_blocks, [&] { logits_tail(); });
            e.update_step(G, ctx_blocks);
            CK(cudaGraphLaunch(G.exec, e.st));
        } else {
            e.forward_block(blk.data(), ctx_blocks * eng::T, ctx_blocks, false, true);
            logits_tail();
        }
        last_fwd = blk;
        CK(cudaStreamSynchronize(e.st));
        if (const char* dp = getenv("LLADA_DUMPX")) {    // TEMP: readout inputs (int8 q, group scales, ids) per step
            static FILE* df = fopen(dp, "wb");
            std::vector<int8_t> q(eng::T * 2048); std::vector<float> sc(eng::T * 16);
            CK(cudaMemcpy(q.data(), e.xA.a.q, q.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sc.data(), e.xA.a.s, sc.size() * 4, cudaMemcpyDeviceToHost));
            fwrite(q.data(), 1, q.size(), df); fwrite(sc.data(), 4, sc.size(), df); fwrite(blk.data(), 4, eng::T, df); fflush(df);
        }
        if (e.lm8.base) check_readout();
        if (e.profiling) e.prof_collect();
        ++S.steps;
    }

    static std::vector<int> transfer_schedule(int n, int steps) {
        std::vector<int> s(steps, n / steps);
        for (int i = 0; i < n % steps; ++i) s[i]++;
        return s;
    }

    // Decode the current (not yet committed) block in place. blk: 32 ids, masks = mask_id; prompt tokens fixed.
    void decode_block(std::vector<int>& blk) {
        const int B = eng::T;
        std::vector<bool> prompt(B), orig(B);
        int init_masks = 0;
        for (int i = 0; i < B; ++i) { prompt[i] = blk[i] != P.mask_id; orig[i] = !prompt[i]; init_masks += orig[i]; }
        if (!init_masks) return;
        auto sched = transfer_schedule(init_masks, P.steps);
        std::set<std::vector<int>> seen;
        int post_steps = 0, step_id = 0;
        while (true) {
            const std::vector<int> old = blk;
            std::vector<bool> mask(B);
            int nmask = 0, orig_left = 0;
            for (int i = 0; i < B; ++i) {
                mask[i] = old[i] == P.mask_id && !prompt[i];
                nmask += mask[i];
                orig_left += orig[i] && old[i] == P.mask_id;
            }
            const int new_masks = nmask - orig_left;
            post_steps = orig_left == 0 ? post_steps + 1 : 0;
            if (nmask == 0 && post_steps > P.max_post_steps) break;
            if (step_id >= P.max_steps_per_block) break;
            step_logits(blk);
            if (trace && step_id == 0 && S.blocks == 0)
                for (int i = 0; i < B; ++i)
                    fprintf(stderr, "  TOP %d %d %d %.4f %d %.4f\n", i, int(mask[i]), h_top[i].i1, h_top[i].p1, h_top[i].i2, h_top[i].p2);
            std::vector<int> x0(B);
            std::vector<float> p0(B);
            for (int i = 0; i < B; ++i) { x0[i] = h_top[i].i1; p0[i] = h_top[i].p1; }
            // M2T
            std::vector<bool> m2t(B, false);
            if (nmask) {
                if (step_id < int(sched.size())) {
                    const int need = sched[step_id] + new_masks;
                    int high = 0;
                    for (int i = 0; i < B; ++i) high += mask[i] && p0[i] > P.threshold;
                    if (high >= need) {
                        for (int i = 0; i < B; ++i) m2t[i] = mask[i] && p0[i] > P.threshold;
                    } else {
                        const int k = std::min(need, nmask);
                        std::vector<int> order;
                        for (int i = 0; i < B; ++i) if (mask[i]) order.push_back(i);
                        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p0[a] > p0[b]; });
                        for (int j = 0; j < k; ++j) m2t[order[j]] = true;
                    }
                } else {
                    m2t = mask;
                }
            }
            // T2T
            std::vector<bool> t2t(B, false);
            for (int i = 0; i < B; ++i) t2t[i] = !mask[i] && !prompt[i] && p0[i] > P.editing_threshold && old[i] != x0[i];
            // final round: resolve every mask, no DELETE/SPLIT on written positions
            bool final_round;
            if (P.max_post_steps > 0) {
                final_round = post_steps >= P.max_post_steps;
            } else {
                bool any = false, covered = true;
                for (int i = 0; i < B; ++i)
                    if (mask[i] && orig[i]) { any = true; covered = covered && m2t[i]; }
                final_round = any && covered;
            }
            if (final_round) {
                m2t = mask;
                for (int i = 0; i < B; ++i)
                    if ((m2t[i] || t2t[i]) && (x0[i] == P.split_id || x0[i] == P.delete_id)) x0[i] = h_top[i].isd;
            }
            if (trace) {
                int nm = 0, nt = 0;
                for (int i = 0; i < B; ++i) { nm += m2t[i]; nt += t2t[i]; }
                fprintf(stderr, "  step %2d: masks %2d (orig %2d) post %2d | m2t %2d t2t %2d%s", step_id, nmask, orig_left,
                        post_steps, nm, nt, final_round ? " FINAL" : "");
                for (int i = 0; i < B; ++i) if (t2t[i]) fprintf(stderr, " [%d:%d->%d p=%.2f]", i, old[i], x0[i], p0[i]);
                fprintf(stderr, "\n");
            }
            const std::vector<bool> snapshot = orig;
            bool any_fill = false;
            for (int i = 0; i < B; ++i)
                if (m2t[i] || t2t[i]) { blk[i] = x0[i]; any_fill = true; }
            if (any_fill && !final_round && seen.count(blk)) {      // anti-loop resample (greedy: next-best token)
                for (int it = 0; it < 5; ++it) {
                    std::vector<int> changed;
                    for (int i = 0; i < B; ++i) if (blk[i] != old[i]) changed.push_back(i);
                    if (changed.empty()) break;
                    const int p = changed[std::uniform_int_distribution<int>(0, int(changed.size()) - 1)(rng)];
                    if (!m2t[p] && !t2t[p]) continue;
                    blk[p] = h_top[p].i1 != blk[p] ? h_top[p].i1 : h_top[p].i2;
                    if (!seen.count(blk)) break;
                }
            }
            seen.insert(blk);
            // DELETE / SPLIT, fixed length
            std::vector<int> nb;
            std::vector<bool> ntr;
            for (int i = 0; i < B; ++i) {
                if (blk[i] == P.delete_id) continue;
                if (blk[i] == P.split_id) {
                    nb.push_back(P.mask_id); ntr.push_back(false);
                    nb.push_back(old[i]); ntr.push_back(snapshot[i]);
                } else {
                    nb.push_back(blk[i]); ntr.push_back(snapshot[i]);
                }
            }
            nb.resize(std::min<size_t>(nb.size(), B)); ntr.resize(nb.size());
            while (int(nb.size()) < B) { nb.push_back(P.mask_id); ntr.push_back(false); }
            blk = nb;
            orig = ntr;
            ++step_id;
            bool stable = blk == old;
            for (int i = 0; i < B && stable; ++i) stable = blk[i] != P.mask_id;
            if (stable) break;
        }
        if (trace) fprintf(stderr, "block done after %d steps\n", step_id);
        ++S.blocks;
    }

    // Prefill the prompt (all full blocks as commit passes) and generate up to max_new tokens.
    // Returns the generated ids (up to and including EOS).
    // on_block(tokens, n): called with each decoded block's new tokens (streaming).
    // Prefix cache: committed blocks that match the new prompt's leading full blocks are kept (an agent's next turn
    // re-sends the whole conversation), so only the rest is prefilled.
    template <class CB>
    std::vector<int> generate(const std::vector<int>& prompt, int max_new, bool eos_stop, CB&& on_block) {
        const int B = eng::T;
        const int full = int(prompt.size()) / B;
        int keep = 0;
        while (keep < full && keep < ctx_blocks &&
               std::equal(prompt.begin() + keep * B, prompt.begin() + (keep + 1) * B, committed.begin() + keep * B))
            ++keep;
        ctx_blocks = keep;
        committed.resize(size_t(keep) * B);
        reused_blocks = keep;
        for (int b = keep; b < full; b += eng::PB) {          // batched prefill of the prompt's full blocks
            const int nb = std::min(eng::PB, full - b);
            e.prefill(prompt.data() + size_t(b) * B, nb, ctx_blocks);
            if (e.profiling) e.prof_collect();
            ctx_blocks += nb; S.commits += nb; S.prefill_passes++;
            committed.insert(committed.end(), prompt.begin() + b * B, prompt.begin() + (b + nb) * B);
        }
        last_fwd.clear();
        std::vector<int> pending(prompt.begin() + full * B, prompt.end());
        std::vector<int> out;
        const int total = int(prompt.size()) + max_new;
        while (ctx_blocks * B < total) {
            std::vector<int> blk = pending;
            pending.clear();
            const int fixed = int(blk.size());
            blk.resize(B, P.mask_id);
            decode_block(blk);
            commit(blk);
            bool eos = false;
            const size_t before = out.size();
            for (int i = fixed; i < B; ++i) {
                out.push_back(blk[i]);
                if (blk[i] == P.eos_id) { eos = true; break; }
            }
            on_block(out.data() + before, int(out.size() - before));
            if (eos && eos_stop) break;
        }
        return out;
    }
};

}  // namespace dec
