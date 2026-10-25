// Correctness + bandwidth check of moe_gemm.cu against gen_moe_case.py's fp32 reference.
//   nvcc -O3 -arch=sm_87 -std=c++17 -I.. test_moe.cu -o test_moe && ./test_moe case
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "moe_gemm.cu"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

template <class T> std::vector<T> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t n = ftell(f) / sizeof(T); fseek(f, 0, SEEK_SET);
    std::vector<T> v(n);
    if (fread(v.data(), sizeof(T), n, f) != n) exit(1);
    fclose(f);
    return v;
}
template <class T> T* up(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "case";
    const int H = 2048, I = 512, K = 8;
    auto blob = rd<uint8_t>(dir + "/blob.bin");
    auto gu_off = rd<uint64_t>(dir + "/gu_off.bin");
    auto d_off = rd<uint64_t>(dir + "/d_off.bin");
    auto bits = rd<int8_t>(dir + "/bits.bin");
    auto x = rd<uint16_t>(dir + "/x.bin");
    auto idx = rd<int32_t>(dir + "/idx.bin");
    auto wt = rd<float>(dir + "/w.bin");
    auto ref = rd<float>(dir + "/ref.bin");
    const int T = int(x.size() / H), E = int(bits.size());
    {   // activations to fragment order (the engine's rmsnorm+FWHT kernel writes it directly)
        std::vector<uint16_t> xp(x.size());
        for (int r = 0; r < T; ++r)
            for (int c = 0; c < H; ++c) xp[size_t(r) * H + moe::frag_pos(c)] = x[size_t(r) * H + c];
        x.swap(xp);
    }
    std::vector<float> sx(size_t(T) * (H / 128), 0.f);   // group sums of the fp16 activations
    for (int r = 0; r < T; ++r)
        for (int c = 0; c < H; ++c)   // permutation stays inside a group, so c / 128 is still the group
            sx[size_t(r) * (H / 128) + c / 128] += __half2float(*reinterpret_cast<const half*>(&x[size_t(r) * H + c]));

    // host scheduling: counting sort of (token, k) by expert -> slots; items of <= 32 slots
    std::vector<int> cnt(E, 0);
    for (int i = 0; i < T * K; ++i) cnt[idx[i]]++;
    std::vector<int> start(E + 1, 0);
    for (int e = 0; e < E; ++e) start[e + 1] = start[e] + cnt[e];
    std::vector<int> fill(start.begin(), start.end() - 1), slot_tok(T * K), tok_slots(T * K);
    std::vector<float> slot_w(T * K);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const int e = idx[t * K + k], s = fill[e]++;
            slot_tok[s] = t; slot_w[s] = wt[t * K + k]; tok_slots[t * K + k] = s;
        }
    std::vector<moe::Work> w1, w2;   // NT = 1 items (<= 8 tokens), NT = 2 items (<= 16)
    size_t wbytes = 0;
    for (int e = 0; e < E; ++e)
        if (cnt[e]) {
            for (int s = 0; s < cnt[e]; s += 16) {
                const int n = std::min(16, cnt[e] - s);
                (n <= 8 ? w1 : w2).push_back({e, start[e] + s, n});
            }
            wbytes += size_t(2 * I / 16) * (H / 128) * moe::tile_bytes(bits[e]) +
                      size_t(H / 16) * (I / 128) * moe::tile_bytes(bits[e]);
        }
    std::vector<moe::Work> work(w1);
    work.insert(work.end(), w2.begin(), w2.end());
    const int n1 = int(w1.size()), n2 = int(w2.size());
    printf("T=%d, %d+%d (NT1+NT2) work items, %d active experts, %.1f MB weights read\n", T, n1, n2,
           int(std::count_if(cnt.begin(), cnt.end(), [](int c) { return c > 0; })), wbytes / 1e6);

    moe::ExpertTable tab{up(blob), up(gu_off), up(d_off), up(bits)};
    moe::Work* d_work = up(work);
    int* d_tok = up(slot_tok);
    int* d_ts = up(tok_slots);
    float* d_w = up(slot_w);
    half* d_x = reinterpret_cast<half*>(up(x));
    float* d_sx = up(sx);
    float* d_sz; CK(cudaMalloc(&d_sz, size_t(T) * K * (I / 128) * sizeof(float)));
    half* d_z; CK(cudaMalloc(&d_z, size_t(T) * K * I * sizeof(half)));
    float* d_acc; CK(cudaMalloc(&d_acc, size_t(T) * H * sizeof(float)));
    float* d_y; CK(cudaMalloc(&d_y, size_t(T) * K * H * sizeof(float)));

#ifndef WGV
#define WGV 4
#endif
#ifndef WDV
#define WDV 4
#endif
    constexpr int WG = WGV, WD = WDV;
#ifndef MSV
#define MSV 2
#endif
    constexpr int MS = MSV;
    const int sg1 = moe::smem_bytes(1, H), sg2 = moe::smem_bytes(2, H), sd1 = moe::smem_bytes(1, I), sd2 = moe::smem_bytes(2, I);
    CK(cudaFuncSetAttribute(moe::moe_gemm<WG, 1, MS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sg1));
    CK(cudaFuncSetAttribute(moe::moe_gemm<WG, 2, MS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sg2));
    CK(cudaFuncSetAttribute(moe::moe_gemm<WD, 1, MS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sd1));
    CK(cudaFuncSetAttribute(moe::moe_gemm<WD, 2, MS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sd2));
    moe::GemmArgs ga{tab, d_work, n1, d_tok, d_x, d_sx, d_w, H, I, d_z, d_y};
    moe::GemmArgs ga2 = ga; ga2.work = d_work + n1; ga2.nwork = n2;
    moe::GemmArgs gd = ga; gd.X = d_z; gd.SX = d_sz;
    moe::GemmArgs gd2 = gd; gd2.work = d_work + n1; gd2.nwork = n2;
    auto gate_up = [&]() {
        if (n1) moe::moe_gemm<WG, 1, MS, 0><<<dim3(2 * I / 16 / (WG * MS), n1), WG * 32, sg1>>>(ga);
        if (n2) moe::moe_gemm<WG, 2, MS, 0><<<dim3(2 * I / 16 / (WG * MS), n2), WG * 32, sg2>>>(ga2);
    };
    auto down = [&]() {
        if (n1) moe::moe_gemm<WD, 1, MS, 1><<<dim3(H / 16 / (WD * MS), n1), WD * 32, sd1>>>(gd);
        if (n2) moe::moe_gemm<WD, 2, MS, 1><<<dim3(H / 16 / (WD * MS), n2), WD * 32, sd2>>>(gd2);
    };
    auto run = [&]() {
        gate_up();
        moe::fwht512<<<T * K, 256>>>(d_z, d_sz, T * K);
        down();
        moe::moe_combine<<<T, 128>>>(d_y, d_ts, K, H, d_acc);
    };
    run();
    CK(cudaDeviceSynchronize());
    std::vector<float> acc(size_t(T) * H);
    CK(cudaMemcpy(acc.data(), d_acc, acc.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < acc.size(); ++i) {
        const double d = acc[i] - ref[i];
        num += d * d; den += double(ref[i]) * ref[i]; mx = std::max(mx, std::fabs(d));
    }
    const double rel = std::sqrt(num / den);
    printf("rel L2 err %.3e, max abs err %.3e, ref rms %.3e\n", rel, mx, std::sqrt(den / acc.size()));

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) run();
    const int R = 50;
    cudaEventRecord(a);
    for (int i = 0; i < R; ++i) run();
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= R;
    printf("MoE layer step: %.3f ms, %.1f GB/s effective weight stream\n", ms, wbytes / (ms * 1e6));
    // per kernel
    float t3[3] = {0, 0, 0};
    cudaEvent_t ev[4]; for (auto& e : ev) cudaEventCreate(&e);
    for (int i = 0; i < R; ++i) {
        cudaEventRecord(ev[0]);
        gate_up();
        cudaEventRecord(ev[1]);
        moe::fwht512<<<T * K, 256>>>(d_z, d_sz, T * K);
        cudaEventRecord(ev[2]);
        down();
        moe::moe_combine<<<T, 128>>>(d_y, d_ts, K, H, d_acc);
        cudaEventRecord(ev[3]); CK(cudaEventSynchronize(ev[3]));
        for (int k = 0; k < 3; ++k) { float m; cudaEventElapsedTime(&m, ev[k], ev[k + 1]); t3[k] += m / R; }
    }
    printf("gate_up %.3f ms | fwht %.3f ms | down+combine %.3f ms\n", t3[0], t3[1], t3[2]);
    return rel < 5e-3 ? 0 : 2;
}
