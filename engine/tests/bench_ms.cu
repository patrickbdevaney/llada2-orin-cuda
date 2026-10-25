// NT1 expert GEMM: strips per warp (MS) and occupancy variants on N same-width items.   ./bench_ms <dir> [layer]
#include "decode.cu"
template <int W, int MS, int MODE>
double run(eng::Engine& e, const moe8::ExpertTable& tab, const moe8::Work* w, int n, const int* stok, const moe8::Act& X,
           int rows, int K, int G) {
    moe8::GemmArgs g{tab, w, nullptr, stok, X, e.sched.slot_weight, MODE == 0 ? K : rows, MODE == 0 ? rows / 2 : K, e.Z, e.Y, G, 0};
    if ((rows / 16) % (W * MS * G)) return -1;
    const int rb = rows / 16 / (W * MS * G), sm = moe8::smem_bytes(1, K);
    CK(cudaFuncSetAttribute(moe8::gemm<W, 1, MS, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; ++i) moe8::gemm<W, 1, MS, MODE><<<dim3(rb, n), W * 32, sm, e.st>>>(g);
    double best = 1e9;
    for (int r = 0; r < 3; ++r) {
        cudaEventRecord(a, e.st); for (int i = 0; i < 20; ++i) moe8::gemm<W, 1, MS, MODE><<<dim3(rb, n), W * 32, sm, e.st>>>(g);
        cudaEventRecord(b, e.st); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b); best = std::min(best, 1e3 * ms / 20);
    }
    return best;
}
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    const int li = argc > 2 ? atoi(argv[2]) : 10;
    eng::Engine e; e.load(dir, 4096);
    std::vector<int> ids(32); for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    e.begin_block(ids.data(), 0); for (int l = 0; l <= li; ++l) e.layer_step(l, 0);
    CK(cudaStreamSynchronize(e.st));
    eng::Layer& Ly = e.layers[li];
    std::vector<int8_t> bits(257); CK(cudaMemcpy(bits.data(), Ly.experts.bits, 257, cudaMemcpyDeviceToHost));
    std::vector<int> stok(288); for (int i = 0; i < 288; ++i) stok[i] = i % 32;
    CK(cudaMemcpy(e.sched.slot_token, stok.data(), 288 * 4, cudaMemcpyHostToDevice));
    for (int B : {2, 3}) {
        std::vector<int> ex; for (int x = 0; x < 256; ++x) if (bits[x] == B) ex.push_back(x);
        const int N = std::min<int>(40, ex.size()); if (N < 8) continue;
        std::vector<moe8::Work> w; for (int i = 0; i < N; ++i) w.push_back({ex[i], (i * 8) % 256, 8});
        moe8::Work* dw = eng::dev_copy(w);
        const double gub = N * 64.0 * 16 * moe8::tile_bytes(B), dnb = N * 128.0 * 4 * moe8::tile_bytes(B);
        printf("%d-bit, %d items:\n", B, N);
        auto pr = [&](const char* nm, double t, double by) { if (t > 0) printf("  %-22s %7.1f us %5.1f GB/s\n", nm, t, by / t / 1e3); };
        pr("gate_up W4 MS2 g4", run<4, 2, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 4), gub);
        pr("gate_up W4 MS4 g2", run<4, 4, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 2), gub);
        pr("gate_up W4 MS4 g1", run<4, 4, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 1), gub);
        pr("gate_up W4 MS1 g8", run<4, 1, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 8), gub);
        pr("gate_up W8 MS2 g2", run<8, 2, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 2), gub);
        pr("gate_up W2 MS2 g8", run<2, 2, 0>(e, Ly.experts, dw, N, e.sched.slot_token, e.xA.a, 1024, 2048, 8), gub);
        pr("down W4 MS2 g8", run<4, 2, 1>(e, Ly.experts, dw, N, nullptr, e.zA.a, 2048, 512, 8), dnb);
        pr("down W4 MS4 g4", run<4, 4, 1>(e, Ly.experts, dw, N, nullptr, e.zA.a, 2048, 512, 4), dnb);
        pr("down W4 MS1 g16", run<4, 1, 1>(e, Ly.experts, dw, N, nullptr, e.zA.a, 2048, 512, 16), dnb);
        pr("down W8 MS2 g4", run<8, 2, 1>(e, Ly.experts, dw, N, nullptr, e.zA.a, 2048, 512, 4), dnb);
    }
    return 0;
}
