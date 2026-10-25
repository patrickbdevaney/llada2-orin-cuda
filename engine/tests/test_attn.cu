// attn.cu check: random Q/K/V; cache built by kv_commit; fp64 CPU reference over the dequantized cache + fresh block.
//   nvcc -O3 -arch=sm_87 -std=c++17 -I.. test_attn.cu -o test_attn && ./test_attn [nblocks] [blocks_per_chunk]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "attn.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
using attn::D;

static float h2f(uint16_t h) { half x = *reinterpret_cast<half*>(&h); return __half2float(x); }

int main(int argc, char** argv) {
    const int nblocks = argc > 1 ? atoi(argv[1]) : 64, bpc = argc > 2 ? atoi(argv[2]) : 8;
    const int maxb = nblocks + 1, nchunks = attn::chunks_for(nblocks, bpc);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto randh = [&](size_t n, float sc, std::vector<half>& v) { v.resize(n); for (auto& x : v) x = __float2half(nd(rng) * sc); };
    std::vector<half> Q, Kall, Vall;                      // Kall/Vall: (nblocks + 1) blocks of [32][4][128]
    randh(32 * 16 * D, 1.f, Q);
    randh(size_t(nblocks + 1) * 32 * 4 * D, 1.f, Kall);
    randh(size_t(nblocks + 1) * 32 * 4 * D, 1.f, Vall);
    for (size_t i = 0; i < Kall.size(); ++i) if ((i % D) == 5) Kall[i] = __float2half(__half2float(Kall[i]) * 8.f + 3.f);  // outlier channel
    half *dQ, *dK, *dV, *dO; uint8_t* dC; float *dOp, *dMp, *dLp;
    const size_t blkn = 32 * 4 * D;
    CK(cudaMalloc(&dQ, Q.size() * 2)); CK(cudaMalloc(&dK, Kall.size() * 2)); CK(cudaMalloc(&dV, Vall.size() * 2));
    CK(cudaMalloc(&dO, 32 * 16 * D * 2));
    CK(cudaMalloc(&dC, size_t(4) * maxb * attn::KV_BLOCK_BYTES));
    CK(cudaMalloc(&dOp, size_t(4) * nchunks * 128 * D * 4)); CK(cudaMalloc(&dMp, 4 * nchunks * 128 * 4)); CK(cudaMalloc(&dLp, 4 * nchunks * 128 * 4));
    CK(cudaMemcpy(dQ, Q.data(), Q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, Kall.data(), Kall.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, Vall.data(), Vall.size() * 2, cudaMemcpyHostToDevice));
    for (int b = 0; b < nblocks; ++b) attn::kv_commit<<<4, 128>>>(dK + b * blkn, dV + b * blkn, dC, maxb, b);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> C(size_t(4) * maxb * attn::KV_BLOCK_BYTES);
    CK(cudaMemcpy(C.data(), dC, C.size(), cudaMemcpyDeviceToHost));

    // dequantized keys/values per head: [ctx][128]; context = cached blocks then the fresh block (exact fp16)
    const int ctx = (nblocks + 1) * 32;
    std::vector<double> Kd(size_t(4) * ctx * D), Vd(size_t(4) * ctx * D);
    double kerr = 0, kden = 0;
    for (int h = 0; h < 4; ++h)
        for (int b = 0; b <= nblocks; ++b)
            for (int tk = 0; tk < 32; ++tk)
                for (int c = 0; c < D; ++c) {
                    const size_t src = size_t(b) * blkn + (size_t(tk) * 4 + h) * D + c, dst = (size_t(h) * ctx + b * 32 + tk) * D + c;
                    if (b == nblocks) { Kd[dst] = __half2float(Kall[src]); Vd[dst] = __half2float(Vall[src]); continue; }
                    const uint8_t* blk = C.data() + (size_t(h) * maxb + b) * attn::KV_BLOCK_BYTES;
                    const uint16_t* km = reinterpret_cast<const uint16_t*>(blk + attn::KOFF_META);
                    const uint16_t* vm = reinterpret_cast<const uint16_t*>(blk + attn::VOFF_META);
                    const int kc = (blk[tk * 64 + c / 2] >> (4 * (c & 1))) & 15;
                    const int vc = (blk[attn::VOFF + tk * 64 + c / 2] >> (4 * (c & 1))) & 15;
                    Kd[dst] = kc * h2f(km[c]) + h2f(km[128 + c]);
                    Vd[dst] = vc * h2f(vm[tk]) + h2f(vm[32 + tk]);
                    const double o = __half2float(Kall[src]);
                    kerr += (Kd[dst] - o) * (Kd[dst] - o); kden += o * o;
                }
    printf("kv_commit K rel quant err %.3e (4-bit per-channel)\n", std::sqrt(kerr / kden));

    attn::Args a{dQ, dK + size_t(nblocks) * blkn, dV + size_t(nblocks) * blkn, dC, maxb, nblocks, bpc, nchunks,
                 1.f / std::sqrt(float(D)), dOp, dMp, dLp};
    CK(cudaFuncSetAttribute(attn::attn_partial, cudaFuncAttributeMaxDynamicSharedMemorySize, attn::SMEM_BYTES));
    auto run = [&]() {
        attn::attn_partial<<<dim3(4, nchunks, 1), attn::ATT_NTH, attn::SMEM_BYTES>>>(a);
        attn::attn_combine<<<dim3(4, 128 / attn::COMB_ROWS, 1), 32 * attn::COMB_ROWS>>>(dOp, dMp, dLp, nchunks, nblocks, bpc, dO, nullptr, nullptr, nullptr);
    };
    run();
    CK(cudaDeviceSynchronize());
    std::vector<half> O(32 * 16 * D);
    CK(cudaMemcpy(O.data(), dO, O.size() * 2, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    std::vector<double> sc(ctx);
    for (int tk = 0; tk < 32; ++tk)
        for (int qh = 0; qh < 16; ++qh) {
            const int h = qh / 4;
            double mx = -1e300;
            for (int j = 0; j < ctx; ++j) {
                double s = 0;
                for (int c = 0; c < D; ++c) s += double(__half2float(Q[(tk * 16 + qh) * D + c])) * Kd[(size_t(h) * ctx + j) * D + c];
                sc[j] = s / std::sqrt(double(D)); mx = std::max(mx, sc[j]);
            }
            double l = 0;
            for (int j = 0; j < ctx; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
            for (int c = 0; c < D; ++c) {
                double o = 0;
                for (int j = 0; j < ctx; ++j) o += sc[j] * Vd[(size_t(h) * ctx + j) * D + c];
                o /= l;
                const double gpu = __half2float(O[(tk * 16 + qh) * D + c]);
                num += (gpu - o) * (gpu - o); den += o * o;
            }
        }
    printf("ctx %d tokens (%d cached blocks, %d per chunk, %d chunks): attention rel L2 err %.3e\n",
           ctx, nblocks, bpc, nchunks, std::sqrt(num / den));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) run();
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) run();
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 20;
    printf("attention (one layer): %.3f ms, cache stream %.1f GB/s\n", ms, 4.0 * nblocks * attn::KV_BLOCK_BYTES / (ms * 1e6));
    cudaEventRecord(e0);
    for (int i = 0; i < 20; ++i) attn::attn_partial<<<dim3(4, nchunks, 1), attn::ATT_NTH, attn::SMEM_BYTES>>>(a);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms2; cudaEventElapsedTime(&ms2, e0, e1); ms2 /= 20;
    printf("  partial alone %.3f ms, combine %.3f ms\n", ms2, ms - ms2);
    return std::sqrt(num / den) < 1e-2 ? 0 : 2;
}
