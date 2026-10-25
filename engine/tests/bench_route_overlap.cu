// Does the shared expert overlap routing? Times router+schedule, the shared path, and both as layer_step issues them.
#include "decode.cu"
int main() {
    eng::Engine e; e.load(std::string(getenv("HOME")) + "/models/llada2mini-orin", 4096);
    std::vector<int> ids(32); for (int i = 0; i < 32; ++i) ids[i] = 1000 + 37 * i;
    const int li = 10;
    e.begin_block(ids.data(), 0); for (int l = 0; l <= li; ++l) e.layer_step(l, 0);
    CK(cudaStreamSynchronize(e.st));
    eng::Layer& Ly = e.layers[li];
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    auto timeit = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 3; ++i) fn();
        cudaEventRecord(a, e.st); for (int i = 0; i < 50; ++i) fn(); cudaEventRecord(b, e.st); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); printf("  %-34s %7.1f us\n", name, 1e3 * ms / 50);
    };
    const int S0 = 256;
    const moe8::Act zs{e.zA.q + size_t(S0) * 512, e.zA.s + size_t(S0) * 4, e.zA.sum + size_t(S0) * 4};
    auto shared = [&](cudaStream_t s) {
        e.gemm<4, 0>(Ly.experts, e.items_shared, nullptr, 1, e.stok_shared, e.xA.a, e.ones, 1024, 2048, e.Z, nullptr, 1, 2, false, s);
        moe8::fwht512_q8<<<4, 256, 0, s>>>(e.Z + size_t(S0) * 512, zs, 32);
        e.gemm<4, 1>(Ly.experts, e.items_shared, nullptr, 1, nullptr, e.zA.a, e.ones, 2048, 512, nullptr, e.Y, 1, 4, false, s);
    };
    auto route = [&] {
        ops::router_mma<<<dim3(ops::ROUTER_KC, 1), 256, ops::ROUTER_SMEM, e.st>>>(e.hn, Ly.router, e.rlogits);
        ops::route_schedule<<<1, 256, 0, e.st>>>(e.rlogits, Ly.bias, e.rscale, e.sched);
    };
    timeit("router + route_schedule", route);
    timeit("shared gate_up+fwht+down", [&] { shared(e.st); });
    timeit("  shared gate_up only", [&] { e.gemm<4, 0>(Ly.experts, e.items_shared, nullptr, 1, e.stok_shared, e.xA.a, e.ones, 1024, 2048, e.Z, nullptr, 1, 2, false, e.st); });
    timeit("  shared gate_up g1", [&] { e.gemm<4, 0>(Ly.experts, e.items_shared, nullptr, 1, e.stok_shared, e.xA.a, e.ones, 1024, 2048, e.Z, nullptr, 1, 1, false, e.st); });
    timeit("  shared down only", [&] { e.gemm<4, 1>(Ly.experts, e.items_shared, nullptr, 1, nullptr, e.zA.a, e.ones, 2048, 512, nullptr, e.Y, 1, 4, false, e.st); });
    timeit("  shared down g1", [&] { e.gemm<4, 1>(Ly.experts, e.items_shared, nullptr, 1, nullptr, e.zA.a, e.ones, 2048, 512, nullptr, e.Y, 1, 1, false, e.st); });
    timeit("both (side stream)", [&] {
        CK(cudaEventRecord(e.ev_s, e.st)); CK(cudaStreamWaitEvent(e.st4, e.ev_s, 0));
        shared(e.st4); CK(cudaEventRecord(e.ev_sd, e.st4));
        route(); CK(cudaStreamWaitEvent(e.st, e.ev_sd, 0)); });
    return 0;
}
