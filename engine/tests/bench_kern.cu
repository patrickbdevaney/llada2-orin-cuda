// Per-kernel timing on a real layer of the loaded engine: each GEMM of one denoising step repeated in isolation,
// with the weight bytes it streams -> effective GB/s.   ./bench_kern <engine dir> [layer] [ctx_blocks]
#include <chrono>
#include "decode.cu"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    const int li = argc > 2 ? atoi(argv[2]) : 10, cb = argc > 3 ? atoi(argv[3]) : 4;
    eng::Engine e;
    e.load(dir, 4096);
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    for (int b = 0; b < cb; ++b) e.forward_block(ids.data(), b * 32, b, true, false);
    e.begin_block(ids.data(), cb * 32);
    for (int l = 0; l <= li; ++l) e.layer_step(l, cb);           // leaves layer li's schedule and activations
    CK(cudaStreamSynchronize(e.st));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto timeit = [&](const char* name, double bytes, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        const int R = 50;
        cudaEventRecord(a, e.st);
        for (int i = 0; i < R; ++i) fn();
        cudaEventRecord(b, e.st); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b);
        const double us = 1e3 * ms / R;
        printf("  %-28s %8.1f us  %7.2f MB  %6.1f GB/s\n", name, us, bytes / 1e6, bytes / us / 1e3);
    };
    const int nb = e.layers[li].experts.base ? 8 : 8;
    auto mat_bytes = [](int rows, int K, int bits) { return double(rows / 16) * (K / 128) * moe8::tile_bytes(bits); };
    eng::Layer& Ly = e.layers[li];
    const int nbits = 8, lmb = 8;
    (void)nb;
    printf("layer %d (ctx %d blocks):\n", li, cb);
    moe8::Work* items8 = eng::dev_copy(std::vector<moe8::Work>{{0, 0, 8}, {0, 8, 8}, {0, 16, 8}, {0, 24, 8}});
    timeit("norm_rot_q8", 32 * 2048 * 4, [&] { ops::norm_rot_q8<<<32, 256, 0, e.st>>>(e.h, e.signs, Ly.g_in, e.eps, e.xA.a, nullptr); });
    for (int G : {1, 2, 3})
        timeit((std::string("qkv gemm groups ") + std::to_string(G)).c_str(), mat_bytes(3072, 2048, nbits),
               [&] { e.gemm<4, 1>(Ly.qkv, e.items_dense, nullptr, 1, nullptr, e.xA.a, e.ones, 3072, 2048, nullptr, e.qkv, 1, G); });
    for (int ks : {2, 4})
        timeit((std::string("qkv gemm ksplit ") + std::to_string(ks)).c_str(), mat_bytes(3072, 2048, nbits),
               [&] { e.gemm<4, 1>(Ly.qkv, e.items_dense, nullptr, 1, nullptr, e.xA.a, e.ones, 3072, 2048, nullptr, e.qkv, ks, 1); });
    for (int G : {1, 2})
        timeit((std::string("qkv NT2 x2 swap groups ") + std::to_string(G)).c_str(), mat_bytes(3072, 2048, nbits),
               [&] { e.gemm<2, 1>(Ly.qkv, e.items_dense16, nullptr, 2, nullptr, e.xA.a, e.ones, 3072, 2048, nullptr, e.qkv, 1, G, true); });
    for (int G : {1, 2})
        timeit((std::string("qkv NT1 x4 swap groups ") + std::to_string(G)).c_str(), mat_bytes(3072, 2048, nbits),
               [&] { e.gemm<1, 1>(Ly.qkv, items8, nullptr, 4, nullptr, e.xA.a, e.ones, 3072, 2048, nullptr, e.qkv, 1, G, true); });
    for (int ks : {1, 4})
        timeit((std::string("dense gemm ksplit ") + std::to_string(ks)).c_str(), mat_bytes(2048, 2048, nbits),
               [&] { e.gemm<4, 2>(Ly.dense, e.items_dense, nullptr, 1, nullptr, e.oA.a, e.ones, 2048, 2048, nullptr, e.hn, ks); });
    // routed experts: bytes from the actual work lists
    std::vector<moe8::Work> w1(256), w2(128);
    int nw[2];
    CK(cudaMemcpy(nw, e.sched.nwork, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w1.data(), e.sched.work1, nw[0] * sizeof(moe8::Work), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w2.data(), e.sched.work2, nw[1] * sizeof(moe8::Work), cudaMemcpyDeviceToHost));
    std::vector<int8_t> bits(257);
    CK(cudaMemcpy(bits.data(), Ly.experts.bits, 257, cudaMemcpyDeviceToHost));
    double gu1 = 0, gu2 = 0, d1 = 0, d2 = 0;
    std::set<int> uniq;
    for (int i = 0; i < nw[0]; ++i) { gu1 += mat_bytes(1024, 2048, bits[w1[i].expert]); d1 += mat_bytes(2048, 512, bits[w1[i].expert]); uniq.insert(w1[i].expert); }
    for (int i = 0; i < nw[1]; ++i) { gu2 += mat_bytes(1024, 2048, bits[w2[i].expert]); d2 += mat_bytes(2048, 512, bits[w2[i].expert]); uniq.insert(w2[i].expert); }
    printf("  work items: %d NT1, %d NT2 (incl. 2 shared), %zu distinct experts\n", nw[0], nw[1], uniq.size());
    auto& S = e.sched;
    for (int G : {1, 2, 4, 8}) {
        printf("  -- groups %d\n", G);
        timeit("moe gate_up NT1", gu1, [&] { e.gemm<1, 0>(Ly.experts, S.work1, S.nwork, 64, S.slot_token, e.xA.a, S.slot_weight, 1024, 2048, e.Z, nullptr, 1, G); });
        timeit("moe gate_up NT2", gu2, [&] { e.gemm<2, 0>(Ly.experts, S.work2, S.nwork + 1, 40, S.slot_token, e.xA.a, S.slot_weight, 1024, 2048, e.Z, nullptr, 1, G); });
        timeit("moe down NT1", d1, [&] { e.gemm<1, 1>(Ly.experts, S.work1, S.nwork, 64, nullptr, e.zA.a, S.slot_weight, 2048, 512, nullptr, e.Y, 1, G); });
        timeit("moe down NT2", d2, [&] { e.gemm<2, 1>(Ly.experts, S.work2, S.nwork + 1, 40, nullptr, e.zA.a, S.slot_weight, 2048, 512, nullptr, e.Y, 1, G); });
    }
    {   // locality test: same item shapes, experts consecutive in memory (and by bit width)
        std::vector<std::pair<int, int>> byb;
        for (int x = 0; x < 256; ++x) byb.push_back({bits[x], x});
        for (int B : {2, 3, 4}) {
            std::vector<moe8::Work> wl;
            int n = 0;
            for (int x = 0; x < 256 && n < nw[0]; ++x) if (bits[x] == B) { wl.push_back({x, w1[n].slot0, w1[n].ntok}); ++n; }
            if (n < 8) continue;
            double by = 0; for (auto& q : wl) by += mat_bytes(1024, 2048, B);
            moe8::Work* dw = eng::dev_copy(wl);
            timeit(("gate_up NT1 " + std::to_string(n) + " consecutive " + std::to_string(B) + "-bit").c_str(), by,
                   [&] { e.gemm<1, 0>(Ly.experts, dw, nullptr, n, S.slot_token, e.xA.a, S.slot_weight, 1024, 2048, e.Z, nullptr, 1, 1); });
            // same experts, but in a shuffled order
            std::vector<moe8::Work> sh = wl;
            std::mt19937 r(1); std::shuffle(sh.begin(), sh.end(), r);
            for (int i = 0; i < n; ++i) { sh[i].slot0 = wl[i].slot0; sh[i].ntok = wl[i].ntok; }
            moe8::Work* ds = eng::dev_copy(sh);
            timeit(("gate_up NT1 " + std::to_string(n) + " shuffled " + std::to_string(B) + "-bit").c_str(), by,
                   [&] { e.gemm<1, 0>(Ly.experts, ds, nullptr, n, S.slot_token, e.xA.a, S.slot_weight, 1024, 2048, e.Z, nullptr, 1, 1); });
        }
        int c[9] = {0}; for (int x = 0; x < 256; ++x) c[bits[x]]++;
        printf("  layer bit histogram: 2:%d 3:%d 4:%d 8:%d\n", c[2], c[3], c[4], c[8]);
    }
    for (int G : {16}) timeit(("moe down NT1 groups " + std::to_string(G)).c_str(), d1, [&] { e.gemm<1, 1>(Ly.experts, S.work1, S.nwork, 64, nullptr, e.zA.a, S.slot_weight, 2048, 512, nullptr, e.Y, 1, G); });
    timeit("fwht512_q8", 288 * 512 * 3, [&] { moe8::fwht512_q8<<<36, 256, 0, e.st>>>(e.Z, e.zA.a, 288); });
    timeit("moe_combine_unrot (Y read)", 288 * 2048 * 4, [&] { ops::moe_combine_unrot<<<32, 256, 0, e.st>>>(e.Y, S.tok_slots, 9, e.signs, e.hn, 32); });
    timeit("router_mma", 256 * 2048 * 2, [&] { ops::router_mma<<<dim3(ops::ROUTER_KC, 1), 256, ops::ROUTER_SMEM, e.st>>>(e.hn, Ly.router, e.rlogits); });
    timeit("route_schedule", 0, [&] { ops::route_schedule<<<1, 256, 0, e.st>>>(e.rlogits, Ly.bias, e.rscale, S); });
    timeit("lm_head (screen plane, fp16 out)", e.lm_split ? double(e.V / 16) * 16 * moe8::tile_bytes(4) : mat_bytes(e.V, 2048, lmb), [&] { e.lm_gemm(); });
    dec::RowTop* top; CK(cudaMalloc(&top, 32 * sizeof(dec::RowTop)));
    dec::TopRec* part; CK(cudaMalloc(&part, 32 * dec::TOP_SPLIT * sizeof(dec::TopRec)));
    if (e.lm_split) {
        dec::CandRec* cr; CK(cudaMalloc(&cr, 32 * dec::TOP_SPLIT * sizeof(dec::CandRec)));
        timeit("cand_part", double(e.V) * 32 * 4, [&] { dec::cand_part<<<dim3(32, dec::TOP_SPLIT), 256, 0, e.st>>>(reinterpret_cast<const half*>(e.logits), e.V, cr); });
        timeit("cand_finish", 0, [&] { dec::cand_finish<<<32, 256, 0, e.st>>>(cr, e.lm_hi, e.lm_lo, e.xA.a, 156930, 156931, top); });
    }
    timeit("logits_top (part+merge)", double(e.V) * 32 * 4, [&] {
        dec::logits_top_part<<<dim3(32, dec::TOP_SPLIT), 256, 0, e.st>>>(e.logits, e.V, 156930, 156931, part);
        dec::logits_top_merge<<<32, 32, 0, e.st>>>(part, top); });
    {
        const int bpc = attn::pick_bpc(cb, 1), nch = attn::chunks_for(cb, bpc);
        attn::Args aa{e.q, Ly.kf, Ly.vf, Ly.kv, e.max_blocks, cb, bpc, nch, 0.0883883f, e.Opart, e.Mpart, e.Lpart};
        timeit("attn_partial", double(cb) * 4 * attn::KV_BLOCK_BYTES, [&] { attn::attn_partial<<<dim3(4, nch, 1), attn::ATT_NTH, attn::SMEM_BYTES, e.st>>>(aa); });
    }
    // raw streaming reference: copy of the lm_head matrix
    {
        void* dst; const size_t n = 256u << 20; CK(cudaMalloc(&dst, n));
        timeit("cudaMemcpy D2D 256MB (r+w)", 2.0 * n, [&] { CK(cudaMemcpyAsync(dst, e.lm.base, n, cudaMemcpyDeviceToDevice, e.st)); });
        cudaFree(dst);
    }
    return 0;
}
