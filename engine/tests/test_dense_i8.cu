// Dense projection through moe_i8.cu's gemm<.., NT, MS, 1>: one work item holding all T tokens as slots.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "moe_i8.cu"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
template <class T> std::vector<T> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / sizeof(T); fseek(f, 0, SEEK_SET);
    std::vector<T> v(n); if (fread(v.data(), sizeof(T), n, f) != n) exit(1); fclose(f); return v;
}
template <class T> T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "dense";
    auto dims = rd<int>(dir + "/dims.bin");
    const int R = dims[0], K = dims[1], bits = dims[2], T = dims[3];
    auto blob = rd<uint8_t>(dir + "/blob.bin");
    auto x = rd<uint16_t>(dir + "/x.bin");
    auto ref = rd<float>(dir + "/ref.bin");
    std::vector<int8_t> xq(size_t(T) * K); std::vector<float> xs(size_t(T) * (K / 128)); std::vector<int> xsum(xs.size());
    for (int r = 0; r < T; ++r)
        for (int g = 0; g < K / 128; ++g) {
            float m = 0.f;
            for (int c = g * 128; c < g * 128 + 128; ++c) m = std::max(m, std::fabs(__half2float(*reinterpret_cast<const half*>(&x[size_t(r) * K + c]))));
            const float s = std::max(m / 127.f, 1e-30f); int sum = 0;
            for (int c = g * 128; c < g * 128 + 128; ++c) {
                const int q = std::max(-127, std::min(127, int(std::nearbyint(__half2float(*reinterpret_cast<const half*>(&x[size_t(r) * K + c])) / s))));
                xq[size_t(r) * K + moe8::frag_pos_i8(c)] = int8_t(q); sum += q;
            }
            xs[size_t(r) * (K / 128) + g] = s; xsum[size_t(r) * (K / 128) + g] = sum;
        }
    std::vector<uint64_t> off{0}; std::vector<int8_t> bv{int8_t(bits)};
    std::vector<moe8::Work> work{{0, 0, T}};
    std::vector<float> sw(T, 1.f);
    moe8::ExpertTable tab{up(blob), up(off), up(off), up(bv)};
    moe8::relayout_matrix(const_cast<uint8_t*>(tab.base), R, K, bits);   // packlib -> device strip layout
    float* dY; CK(cudaMalloc(&dY, size_t(T) * R * 4));
    moe8::GemmArgs g{tab, up(work), nullptr, nullptr, moe8::Act{up(xq), up(xs), up(xsum)}, up(sw), R, K, nullptr, dY};
    constexpr int W = 4, MS = 2, NT = 4;
    const int sm = moe8::smem_bytes(NT, K);
    CK(cudaFuncSetAttribute(moe8::gemm<W, NT, MS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm));
    auto run = [&]() { moe8::gemm<W, NT, MS, 1><<<dim3(R / 16 / (W * MS), 1), W * 32, sm>>>(g); };
    run(); CK(cudaDeviceSynchronize());
    std::vector<float> y(size_t(T) * R);
    CK(cudaMemcpy(y.data(), dY, y.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); ++i) { num += (y[i] - ref[i]) * double(y[i] - ref[i]); den += double(ref[i]) * ref[i]; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; ++i) run();
    cudaEventRecord(e0); for (int i = 0; i < 50; ++i) run(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 50;
    printf("%dx%d %d-bit T=%d: rel L2 err %.3e, %.3f ms, %.1f GB/s\n", R, K, bits, T, std::sqrt(num / den), ms, blob.size() / (ms * 1e6));
    return std::sqrt(num / den) < 5e-3 ? 0 : 2;
}
