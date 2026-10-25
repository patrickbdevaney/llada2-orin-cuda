// GEMM cost vs number of work items (distinct experts of one bit width, 8 tokens each): per-item slope and fixed
// overhead.   ./bench_items <engine dir> [layer]
#include "decode.cu"
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(getenv("HOME")) + "/models/llada2mini-orin";
    const int li = argc > 2 ? atoi(argv[2]) : 10;
    eng::Engine e;
    e.load(dir, 4096);
    std::vector<int> ids(32);
    for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    e.begin_block(ids.data(), 0);
    for (int l = 0; l <= li; ++l) e.layer_step(l, 0);
    CK(cudaStreamSynchronize(e.st));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto timeit = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        const int R = 30;
        cudaEventRecord(a, e.st);
        for (int i = 0; i < R; ++i) fn();
        cudaEventRecord(b, e.st); CK(cudaEventSynchronize(b)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, a, b);
        return 1e3 * ms / R;
    };
    eng::Layer& Ly = e.layers[li];
    std::vector<int8_t> bits(257);
    CK(cudaMemcpy(bits.data(), Ly.experts.bits, 257, cudaMemcpyDeviceToHost));
    auto& S = e.sched;
    // slot_token for slots 0..255 = slot % 32 (any valid token rows)
    std::vector<int> stok(288); for (int i = 0; i < 288; ++i) stok[i] = i % 32;
    CK(cudaMemcpy(S.slot_token, stok.data(), 288 * 4, cudaMemcpyHostToDevice));
    for (int B : {2, 3}) {
        std::vector<int> ex; for (int x = 0; x < 256; ++x) if (bits[x] == B) ex.push_back(x);
        const double tb = moe8::tile_bytes(B);
        printf("%d-bit (%zu experts in layer):\n", B, ex.size());
        for (int N : {8, 16, 24, 32, 48}) {
            if (N > int(ex.size())) break;
            std::vector<moe8::Work> w; for (int i = 0; i < N; ++i) w.push_back({ex[i], (i * 8) % 256, 8});
            moe8::Work* dw = eng::dev_copy(w);
            const double gub = N * 64.0 * 16 * tb, dnb = N * 128.0 * 4 * tb;
            for (int G : {2, 4}) {
                const double t1 = timeit([&] { e.gemm<1, 0>(Ly.experts, dw, nullptr, N, S.slot_token, e.xA.a, S.slot_weight, 1024, 2048, e.Z, nullptr, 1, G); });
                const double t2 = timeit([&] { e.gemm<1, 1>(Ly.experts, dw, nullptr, N, nullptr, e.zA.a, S.slot_weight, 2048, 512, nullptr, e.Y, 1, 2 * G); });
                printf("  N %2d g%d: gate_up %7.1f us (%5.1f GB/s)   down g%d %7.1f us (%5.1f GB/s)\n", N, G, t1, gub / t1 / 1e3, 2 * G, t2, dnb / t2 / 1e3);
            }
        }
    }
    return 0;
}
