// MoE stage timing on real routed work (layer li after a forward): each launch alone and the concurrent stage
// exactly as Engine::layer_step issues it.   ./bench_stage <engine dir> [layer] [ctx_blocks]
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
    for (int l = 0; l <= li; ++l) e.layer_step(l, cb);
    CK(cudaStreamSynchronize(e.st));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto timeit = [&](const std::string& name, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        const int R = 50;
        cudaEventRecord(a, e.st);
        for (int i = 0; i < R; ++i) fn();
        cudaEventRecord(b, e.st); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("  %-36s %8.1f us\n", name.c_str(), 1e3 * ms / R);
    };
    eng::Layer& Ly = e.layers[li];
    auto& S = e.sched;
    int nw[3]; CK(cudaMemcpy(nw, S.nwork, 12, cudaMemcpyDeviceToHost));
    printf("layer %d items: NT1 %d, NT2 %d, NT4 %d\n", li, nw[0], nw[1], nw[2]);
    const int H = 2048, I = Ly.I;
    auto gu4 = [&](cudaStream_t s) { e.gemm<4, 0>(Ly.experts, S.work4, S.nwork + 2, 16, S.slot_token, e.xA.a, S.slot_weight, 2 * I, H, e.Z, nullptr, 1, 1, false, s); };
    auto gu2 = [&](cudaStream_t s) { e.gemm<2, 0>(Ly.experts, S.work2, S.nwork + 1, 40, S.slot_token, e.xA.a, S.slot_weight, 2 * I, H, e.Z, nullptr, 1, 2, false, s); };
    auto gu1 = [&](cudaStream_t s) { e.gemm<1, 0>(Ly.experts, S.work1, S.nwork, 48, S.slot_token, e.xA.a, S.slot_weight, 2 * I, H, e.Z, nullptr, 1, 4, false, s); };
    auto dn4 = [&](cudaStream_t s) { e.gemm<4, 1>(Ly.experts, S.work4, S.nwork + 2, 16, nullptr, e.zA.a, S.slot_weight, H, I, nullptr, e.Y, 1, 2, false, s); };
    auto dn2 = [&](cudaStream_t s) { e.gemm<2, 1>(Ly.experts, S.work2, S.nwork + 1, 40, nullptr, e.zA.a, S.slot_weight, H, I, nullptr, e.Y, 1, 4, false, s); };
    auto dn1 = [&](cudaStream_t s) { e.gemm<1, 1>(Ly.experts, S.work1, S.nwork, 48, nullptr, e.zA.a, S.slot_weight, H, I, nullptr, e.Y, 1, 8, false, s); };
    timeit("gate_up NT4 alone", [&] { gu4(e.st); });
    {   // the NT4 items split into adjacent smaller items (the repeat reads of a strip hit L2)
        std::vector<moe8::Work> w4(nw[2]), h2, q1;
        CK(cudaMemcpy(w4.data(), S.work4, nw[2] * sizeof(moe8::Work), cudaMemcpyDeviceToHost));
        for (auto& w : w4) {
            for (int o = 0; o < w.ntok; o += 16) h2.push_back({w.expert, w.slot0 + o, std::min(16, w.ntok - o)});
            for (int o = 0; o < w.ntok; o += 8) q1.push_back({w.expert, w.slot0 + o, std::min(8, w.ntok - o)});
        }
        moe8::Work* d2 = eng::dev_copy(h2); moe8::Work* d1 = eng::dev_copy(q1);
        const int n2 = int(h2.size()), n1 = int(q1.size());
        for (int G : {1, 2}) timeit("gate_up hot as NT2 halves g" + std::to_string(G), [&] { e.gemm<2, 0>(Ly.experts, d2, nullptr, n2, S.slot_token, e.xA.a, S.slot_weight, 2 * I, H, e.Z, nullptr, 1, G); });
        for (int G : {2, 4}) timeit("gate_up hot as NT1 quarters g" + std::to_string(G), [&] { e.gemm<1, 0>(Ly.experts, d1, nullptr, n1, S.slot_token, e.xA.a, S.slot_weight, 2 * I, H, e.Z, nullptr, 1, G); });
        for (int G : {2, 4}) timeit("down hot as NT2 halves g" + std::to_string(G), [&] { e.gemm<2, 1>(Ly.experts, d2, nullptr, n2, nullptr, e.zA.a, S.slot_weight, H, I, nullptr, e.Y, 1, G); });
        for (int G : {4, 8}) timeit("down hot as NT1 quarters g" + std::to_string(G), [&] { e.gemm<1, 1>(Ly.experts, d1, nullptr, n1, nullptr, e.zA.a, S.slot_weight, H, I, nullptr, e.Y, 1, G); });
    }
    timeit("gate_up NT2 alone", [&] { gu2(e.st); });
    timeit("gate_up NT1 alone", [&] { gu1(e.st); });
    timeit("gate_up NT1+NT2+NT4 serial", [&] { gu4(e.st); gu2(e.st); gu1(e.st); });
    timeit("gate_up stage (3 streams)", [&] { e.fork(); gu4(e.st3); gu2(e.st2); gu1(e.st); e.join(); });
    timeit("down NT4 alone", [&] { dn4(e.st); });
    timeit("down NT2 alone", [&] { dn2(e.st); });
    timeit("down NT1 alone", [&] { dn1(e.st); });
    timeit("down stage (3 streams)", [&] { e.fork(); dn4(e.st3); dn2(e.st2); dn1(e.st); e.join(); });
    timeit("fwht512_q8", [&] { moe8::fwht512_q8<<<36, 256, 0, e.st>>>(e.Z, e.zA.a, 288); });
    return 0;
}
