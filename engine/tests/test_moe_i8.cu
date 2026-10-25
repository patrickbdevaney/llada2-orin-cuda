// Correctness + speed of the W·A8 expert path (moe_i8.cu) against gen_moe_case.py --fmt i8's reference.
//   nvcc -O3 -arch=sm_87 -std=c++17 -I.. test_moe_i8.cu -o test_moe_i8 && ./test_moe_i8 case12_i8
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

#ifndef WGV
#define WGV 4
#endif
#ifndef WDV
#define WDV 4
#endif
#ifndef MSV
#define MSV 2
#endif

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "case12_i8";
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

    // A8 activations as the engine's producer kernel writes them
    std::vector<int8_t> xq(size_t(T) * H);
    std::vector<float> xs(size_t(T) * (H / 128));
    std::vector<int> xsum(size_t(T) * (H / 128));
    for (int r = 0; r < T; ++r)
        for (int g = 0; g < H / 128; ++g) {
            float m = 0.f;
            for (int c = g * 128; c < g * 128 + 128; ++c)
                m = std::max(m, std::fabs(__half2float(*reinterpret_cast<const half*>(&x[size_t(r) * H + c]))));
            const float s = std::max(m / 127.f, 1e-30f);
            int sum = 0;
            for (int c = g * 128; c < g * 128 + 128; ++c) {
                const float v = __half2float(*reinterpret_cast<const half*>(&x[size_t(r) * H + c]));
                const int q = std::max(-127, std::min(127, int(std::nearbyint(v / s))));
                xq[size_t(r) * H + moe8::frag_pos_i8(c)] = int8_t(q);
                sum += q;
            }
            xs[size_t(r) * (H / 128) + g] = s;
            xsum[size_t(r) * (H / 128) + g] = sum;
        }

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
    std::vector<moe8::Work> w1, w2;
    size_t wbytes = 0;
    for (int e = 0; e < E; ++e)
        if (cnt[e]) {
            for (int s = 0; s < cnt[e]; s += 16) {
                const int n = std::min(16, cnt[e] - s);
                (n <= 8 ? w1 : w2).push_back({e, start[e] + s, n});
            }
            wbytes += size_t(2 * I / 16) * (H / 128) * moe8::tile_bytes(bits[e]) +
                      size_t(H / 16) * (I / 128) * moe8::tile_bytes(bits[e]);
        }
    std::vector<moe8::Work> work(w1);
    work.insert(work.end(), w2.begin(), w2.end());
    const int n1 = int(w1.size()), n2 = int(w2.size());
    printf("T=%d, %d+%d (NT1+NT2) items, %.1f MB weights\n", T, n1, n2, wbytes / 1e6);

    moe8::ExpertTable tab{up(blob), up(gu_off), up(d_off), up(bits)};
    {   // packlib -> device strip layout, as Engine::load (unused experts alias offset 0: only the routed ones)
        std::vector<int> used(bits.size(), 0);
        for (int v : idx) used[v] = 1;
        for (int e = 0; e < int(bits.size()); ++e) {
            if (!bits[e] || !used[e]) continue;
            moe8::relayout_matrix(const_cast<uint8_t*>(tab.base) + gu_off[e], 2 * I, H, bits[e]);
            moe8::relayout_matrix(const_cast<uint8_t*>(tab.base) + d_off[e], H, I, bits[e]);
        }
    }
    moe8::Work* d_work = up(work);
    int* d_tok = up(slot_tok);
    int* d_ts = up(tok_slots);
    float* d_w = up(slot_w);
    moe8::Act ax{up(xq), up(xs), up(xsum)};
    half* d_z; CK(cudaMalloc(&d_z, size_t(T) * K * I * sizeof(half)));
    int8_t* zq; float* zs; int* zsum;
    CK(cudaMalloc(&zq, size_t(T) * K * I)); CK(cudaMalloc(&zs, size_t(T) * K * 4 * 4)); CK(cudaMalloc(&zsum, size_t(T) * K * 4 * 4));
    moe8::Act az{zq, zs, zsum};
    float* d_acc; CK(cudaMalloc(&d_acc, size_t(T) * H * sizeof(float)));
    float* d_y; CK(cudaMalloc(&d_y, size_t(T) * K * H * sizeof(float)));

    constexpr int WG = WGV, WD = WDV, MS = MSV;
    const int sg1 = moe8::smem_bytes(1, H), sg2 = moe8::smem_bytes(2, H), sd1 = moe8::smem_bytes(1, I), sd2 = moe8::smem_bytes(2, I);
    CK(cudaFuncSetAttribute(moe8::gemm<WG, 1, MS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sg1));
    CK(cudaFuncSetAttribute(moe8::gemm<WG, 2, MS, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, sg2));
    CK(cudaFuncSetAttribute(moe8::gemm<WD, 1, MS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sd1));
    CK(cudaFuncSetAttribute(moe8::gemm<WD, 2, MS, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, sd2));
    moe8::GemmArgs ga{tab, d_work, nullptr, d_tok, ax, d_w, H, I, d_z, d_y};
    moe8::GemmArgs ga2 = ga; ga2.work = d_work + n1;
    moe8::GemmArgs gd = ga; gd.X = az;
    moe8::GemmArgs gd2 = gd; gd2.work = d_work + n1;
    auto gate_up = [&]() {
        if (n1) moe8::gemm<WG, 1, MS, 0><<<dim3(2 * I / 16 / (WG * MS), n1), WG * 32, sg1>>>(ga);
        if (n2) moe8::gemm<WG, 2, MS, 0><<<dim3(2 * I / 16 / (WG * MS), n2), WG * 32, sg2>>>(ga2);
    };
    auto down = [&]() {
        if (n1) moe8::gemm<WD, 1, MS, 1><<<dim3(H / 16 / (WD * MS), n1), WD * 32, sd1>>>(gd);
        if (n2) moe8::gemm<WD, 2, MS, 1><<<dim3(H / 16 / (WD * MS), n2), WD * 32, sd2>>>(gd2);
    };
    auto run = [&]() {
        gate_up();
        moe8::fwht512_q8<<<(T * K + 7) / 8, 256>>>(d_z, az, T * K);
        down();
        moe8::combine<<<T, 128>>>(d_y, d_ts, K, H, d_acc);
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

    cudaEvent_t ev[5]; for (auto& e : ev) cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) run();
    const int R = 50;
    float t4[4] = {0, 0, 0, 0};
    for (int i = 0; i < R; ++i) {
        cudaEventRecord(ev[0]); gate_up();
        cudaEventRecord(ev[1]); moe8::fwht512_q8<<<(T * K + 7) / 8, 256>>>(d_z, az, T * K);
        cudaEventRecord(ev[2]); down();
        cudaEventRecord(ev[3]); moe8::combine<<<T, 128>>>(d_y, d_ts, K, H, d_acc);
        cudaEventRecord(ev[4]); CK(cudaEventSynchronize(ev[4]));
        for (int k = 0; k < 4; ++k) { float m; cudaEventElapsedTime(&m, ev[k], ev[k + 1]); t4[k] += m / R; }
    }
    const float tot = t4[0] + t4[1] + t4[2] + t4[3];
    printf("MoE layer step %.3f ms (%.1f GB/s) | gate_up %.3f | fwht_q8 %.3f | down %.3f | combine %.3f\n",
           tot, wbytes / (tot * 1e6), t4[0], t4[1], t4[2], t4[3]);
    return rel < 5e-3 ? 0 : 2;
}
