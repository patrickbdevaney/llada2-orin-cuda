// Per-layer glue kernels for the LLaDA2.2-mini Orin engine (decode: T = 32 tokens of one diffusion block).
// The residual stream h is fp32 [T, 2048]. Matrices run through moe_i8.cu's W·A8 GEMM, whose activations are
// moe8::Act (int8 in fragment order + per-128-group scale and int sum), produced here.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "moe_i8.cu"

namespace ops {

constexpr int HID = 2048;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int warp_isum(int v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// In-place FWHT over n = 2 * blockDim.x values in shared memory (thread i owns butterflies of index i).
__device__ __forceinline__ void fwht_smem(float* v, int n) {
    const int half_n = n >> 1;
    for (int hlen = 1; hlen < n; hlen <<= 1) {
        for (int i = threadIdx.x; i < half_n; i += blockDim.x) {
            const int blk = i / hlen, off = i % hlen, a = blk * 2 * hlen + off, b = a + hlen;
            const float x = v[a], y = v[b];
            v[a] = x + y;
            v[b] = x - y;
        }
        __syncthreads();
    }
}

// Quantize v[0..n) (shared, fp32) to the Act row `row`: symmetric int8 per 128-group, fragment order.
// Block of 256 threads; n % 128 == 0, n <= 8192.
__device__ void quant_row_smem(const float* v, int n, moe8::Act out, int row) {
    const int ng = n / 128, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int gi = warp; gi < ng; gi += blockDim.x / 32) {            // one warp per group: max, then sum
        // lane: values 4 lane .. 4 lane + 3, consecutive in the fragment order too -> one 32-bit store
        const float4 x = *reinterpret_cast<const float4*>(v + gi * 128 + 4 * lane);
        const float xv[4] = {x.x, x.y, x.z, x.w};
        float m = fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w)));
        m = warp_max(m);
        const float s = fmaxf(m / 127.f, 1e-30f);
        int sum = 0;
        uint32_t wd = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int qq = max(-127, min(127, __float2int_rn(xv[e] / s)));
            wd |= uint32_t(qq & 0xFF) << (8 * e);
            sum += qq;
        }
        *reinterpret_cast<uint32_t*>(const_cast<int8_t*>(out.q) + size_t(row) * n + moe8::frag_pos_i8(gi * 128 + 4 * lane)) = wd;
        sum = warp_isum(sum);
        if (lane == 0) {
            const_cast<float*>(out.s)[size_t(row) * ng + gi] = s;
            const_cast<int*>(out.sum)[size_t(row) * ng + gi] = sum;
        }
    }
}

template <int N>
__device__ __forceinline__ void warp_fwht(float (&x)[N], int lane);

// x' = A8( RMSnorm_noγ(h) ⊙ d · H / sqrt(2048) ) per token; optionally hn = RMSnorm(h) ⊙ γ (fp32, router input).
// grid T, block 256. FWHT as in moe_combine_unrot: bits 0-7 in registers/shuffles, one transpose, bits 8-10.
// The work after the load: x = thread i's columns 8 i .. 8 i + 7 of token t's row; v, red: shared scratch.
__device__ __forceinline__ void norm_rot_body(float (&x)[8], const float* signs, const float* gamma, float eps,
                                              moe8::Act out, float* hn, int t, float* v, float* red) {
    const int i = threadIdx.x, lane = i & 31;
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
    ss = warp_sum(ss);
    if (lane == 0) red[i >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot += red[w];
    const float r = rsqrtf(tot / HID + eps);
    const float4* s4 = reinterpret_cast<const float4*>(signs + i * 8);
    const float4 sa = s4[0], sb = s4[1];
    const float sg[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
    if (hn) {
        const float4* g4 = reinterpret_cast<const float4*>(gamma + i * 8);
        const float4 ga = g4[0], gb = g4[1];
        float4* o4 = reinterpret_cast<float4*>(hn + size_t(t) * HID + i * 8);
        o4[0] = make_float4(x[0] * r * ga.x, x[1] * r * ga.y, x[2] * r * ga.z, x[3] * r * ga.w);
        o4[1] = make_float4(x[4] * r * gb.x, x[5] * r * gb.y, x[6] * r * gb.z, x[7] * r * gb.w);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] *= r * sg[j];
    warp_fwht<8>(x, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[i * 8 + j] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = v[i + 256 * j];
#pragma unroll
    for (int hh = 1; hh < 8; hh <<= 1)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (!(j & hh)) { const float p = x[j], q = x[j + hh]; x[j] = p + q; x[j + hh] = p - q; }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) v[i + 256 * j] = x[j] * 0.02209708691207961f;   // 1/sqrt(2048)
    __syncthreads();
    quant_row_smem(v, HID, out, t);
}
__global__ void norm_rot_q8(const float* h, const float* signs, const float* gamma, float eps, moe8::Act out,
                            float* hn) {
    __shared__ __align__(16) float v[HID];
    __shared__ float red[8];
    const int t = blockIdx.x, i = threadIdx.x;
    const float4* x4 = reinterpret_cast<const float4*>(h + size_t(t) * HID + i * 8);
    const float4 a = x4[0], b = x4[1];
    float x[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    norm_rot_body(x, signs, gamma, eps, out, hn, t, v, red);
}

// Rows of fp16 or fp32 [rows, n] -> Act (no rotation). grid rows, block 256. n <= 8192.
template <typename T>
__global__ void quant_rows_q8(const T* x, int n, moe8::Act out) {
    __shared__ __align__(16) float v[8192];
    const int r = blockIdx.x;
    for (int c = threadIdx.x; c < n; c += blockDim.x) v[c] = float(x[size_t(r) * n + c]);
    __syncthreads();
    quant_row_smem(v, n, out, r);
}

// Router logits on the tensor cores. hn (fp32) is split as bf16 hi + lo (x = hi + lo to ~2^-17 |x|) and each
// tile takes two m16n8k16 bf16 MMAs with fp32 accumulation, so the logits keep fp32-grade accuracy (Wg is bf16).
// K is split over ROUTER_KC CTAs: part[block][kc][token][expert]; route_core adds the kc partials in a fixed order.
// grid (ROUTER_KC, 32-token blocks), 256 threads: warp w = experts 32 w .. 32 w + 31, both 16-token halves.
// Dynamic shared memory ROUTER_SMEM.
constexpr int ROUTER_KC = 8, ROUTER_KW = HID / ROUTER_KC, ROUTER_LD = ROUTER_KW + 8;   // +8: conflict-free A reads
constexpr int ROUTER_SMEM = 2 * 32 * ROUTER_LD * 2;
__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(256) router_mma(const float* hn, const __nv_bfloat16* Wg, float* part) {
    constexpr int KW = ROUTER_KW, LD = ROUTER_LD;
    extern __shared__ __align__(16) uint8_t rsm[];
    __nv_bfloat16 (*xh)[LD] = reinterpret_cast<__nv_bfloat16 (*)[LD]>(rsm);
    __nv_bfloat16 (*xl)[LD] = reinterpret_cast<__nv_bfloat16 (*)[LD]>(rsm + 32 * LD * 2);
    const int kc = blockIdx.x, blk = blockIdx.y;
    hn += size_t(blk) * 32 * HID + kc * KW;
    part += (size_t(blk) * ROUTER_KC + kc) * 32 * 256;
    for (int i = threadIdx.x; i < 32 * KW / 4; i += blockDim.x) {
        const int t = i / (KW / 4), c = (i % (KW / 4)) * 4;
        const float4 v = *reinterpret_cast<const float4*>(hn + size_t(t) * HID + c);
        const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const __nv_bfloat16 hb = __float2bfloat16_rn(vv[q]);
            xh[t][c + q] = hb;
            xl[t][c + q] = __float2bfloat16_rn(vv[q] - __bfloat162float(hb));
        }
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    constexpr int NN = 4;                                 // n-tiles (8 experts) per warp
    const int e0 = warp * 8 * NN;
    const __nv_bfloat16* W = Wg + size_t(e0) * HID + kc * KW;
    float acc[2][NN][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.f;
    auto w32 = [](const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); };
#pragma unroll 4
    for (int k0 = 0; k0 < KW; k0 += 16) {
        uint32_t b[NN][2];
#pragma unroll
        for (int n = 0; n < NN; ++n) {
            const __nv_bfloat16* wr = W + size_t(n * 8 + g) * HID + k0 + 2 * t;
            b[n][0] = __ldg(reinterpret_cast<const unsigned*>(wr));
            b[n][1] = __ldg(reinterpret_cast<const unsigned*>(wr + 8));
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int r = m * 16 + g;
            const uint32_t ah[4] = {w32(&xh[r][k0 + 2 * t]), w32(&xh[r + 8][k0 + 2 * t]), w32(&xh[r][k0 + 2 * t + 8]), w32(&xh[r + 8][k0 + 2 * t + 8])};
            const uint32_t al[4] = {w32(&xl[r][k0 + 2 * t]), w32(&xl[r + 8][k0 + 2 * t]), w32(&xl[r][k0 + 2 * t + 8]), w32(&xl[r + 8][k0 + 2 * t + 8])};
#pragma unroll
            for (int n = 0; n < NN; ++n) { mma_bf16(acc[m][n], ah, b[n][0], b[n][1]); mma_bf16(acc[m][n], al, b[n][0], b[n][1]); }
        }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < NN; ++n) {
            const int tok = m * 16 + g, ex = e0 + n * 8 + 2 * t;
            *reinterpret_cast<float2*>(part + tok * 256 + ex) = make_float2(acc[m][n][0], acc[m][n][1]);
            *reinterpret_cast<float2*>(part + (tok + 8) * 256 + ex) = make_float2(acc[m][n][2], acc[m][n][3]);
        }
}

// exclusive prefix sum over a 256-thread block; *total gets the sum. ws: 8 ints of shared scratch.
__device__ __forceinline__ int block_excl_scan256(int v, int* ws, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) ws[warp] = x;
    __syncthreads();
    int off = 0, tot = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) { off += w < warp ? ws[w] : 0; tot += ws[w]; }
    __syncthreads();
    *total = tot;
    return off + x - v;
}

struct Sched {
    int* slot_token;
    float* slot_weight;
    int* tok_slots;        // [tokens][9]: 8 routed slots then the shared-expert slot
    moe8::Work* work1;     // <= 8 slots per item (NT = 1)
    moe8::Work* work2;     // 9 .. 16 (NT = 2)
    moe8::Work* work4;     // 17 .. 32 (NT = 4, batched prefill only)
    int* nwork;            // [3]
    int* topk;             // [32][8] (decode)
    float* topw;
    // batched prefill scratch
    int* bsel;             // [tokens][8]
    float* bselw;
    unsigned* bmask;       // [blocks][256] token masks per expert
    int* bpre;             // [blocks][256] first slot of (block, expert)
};
constexpr int W1_CAP = 256, W2_CAP = 256, W4_CAP = 512;   // W4: 4096 slots / 17 + 16 shared items

// Routing of one 32-token block in shared memory (256 threads, thread = expert):
//   score = sigmoid(logit); s = score + bias; pool = top-48 experts by max_t s; per token top-8 of s in the pool;
//   w = score / sum * routed_scale. Leaves sel/selw and, per expert, the bit mask of tokens that picked it.
struct RouteSmem {
    float sr[32][257];     // sigmoid score + bias (score = sr - bias); +1 pad: token rows on distinct banks
    __align__(16) float key[256];
    int pool[48];          // pool experts in index order
    int sel[32][8];
    float selw[32][8];
    int ws[8];
    unsigned tokmask[256];
    int startx[256];
};

// logits: the block's router_mma partials [ROUTER_KC][32][256]
__device__ void route_core(const float* logits, const float* bias, float routed_scale, RouteSmem& S) {
    constexpr int T = 32, E = 256, C = 48, K = 8;
    const int e = threadIdx.x;
    S.tokmask[e] = 0u;
    float lg[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = 0.f;
#pragma unroll
        for (int kc = 0; kc < ROUTER_KC; ++kc) v += logits[(kc * T + t) * E + e];   // fixed order
        lg[t] = v;
    }
    const float be = bias[e];
    float bmax = -INFINITY;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float s = 1.f / (1.f + __expf(-lg[t])) + be;
        S.sr[t][e] = s;
        bmax = fmaxf(bmax, s);
    }
    S.key[e] = bmax;
    __syncthreads();
    int rank;                                           // rank by (key desc, index asc): top-48 = rank < 48
    {
        int r0 = 0, r1 = 0, r2 = 0, r3 = 0;               // four chains: the adds do not serialise
        const float4* k4 = reinterpret_cast<const float4*>(S.key);
#pragma unroll 8
        for (int j4 = 0; j4 < E / 4; ++j4) {
            const float4 kv = k4[j4];
            const int j = 4 * j4;
            r0 += (kv.x > bmax) || (kv.x == bmax && j < e);
            r1 += (kv.y > bmax) || (kv.y == bmax && j + 1 < e);
            r2 += (kv.z > bmax) || (kv.z == bmax && j + 2 < e);
            r3 += (kv.w > bmax) || (kv.w == bmax && j + 3 < e);
        }
        rank = r0 + r1 + r2 + r3;
    }
    const bool inp = rank < C;
    int npool;
    const int pidx = block_excl_scan256(inp, S.ws, &npool);
    if (inp) S.pool[pidx] = e;
    __syncthreads();
    {   // per token top-8 within the pool by rank: thread = (token e / 8, pool slots e % 8 + 8 r); order (value desc,
        // index asc) -- the insertion order of the reference
        const int t = e >> 3, j0 = e & 7;
        float v[C];
#pragma unroll
        for (int jj = 0; jj < C; ++jj) v[jj] = S.sr[t][S.pool[jj]];
#pragma unroll
        for (int r = 0; r < C / 8; ++r) {
            const int jj = j0 + 8 * r;
            const float x = S.sr[t][S.pool[jj]];
            int r4[4] = {0, 0, 0, 0};
#pragma unroll
            for (int q = 0; q < C; ++q) r4[q & 3] += (v[q] > x) || (v[q] == x && q < jj);
            const int rk = r4[0] + r4[1] + r4[2] + r4[3];
            if (rk < K) S.sel[t][rk] = S.pool[jj];
        }
    }
    __syncthreads();
    if (e < T) {
        float sum = 0.f;
        float scv[K];
        for (int k = 0; k < K; ++k) { const int x = S.sel[e][k]; scv[k] = S.sr[e][x] - bias[x]; sum += scv[k]; }
        for (int k = 0; k < K; ++k) S.selw[e][k] = scv[k] / (sum + 1e-20f) * routed_scale;
    }
    {   // a token picks 8 distinct experts, so per-expert token masks give counts and in-expert ranks
        const int t = e >> 3, k = e & 7;
        atomicOr(&S.tokmask[S.sel[t][k]], 1u << t);
    }
    __syncthreads();
}

// Decode: routing + work lists for one block, one CTA. Slots [0, 256): routed (token, k) pairs grouped by expert in
// token order; [256, 288): the shared expert. One item per routed expert (its weights stream once): <= 8 slots ->
// work1 (NT1), 9 .. 16 -> work2 (NT2), 17 .. 32 -> work4 (NT4). The shared expert has no item here: it does not
// depend on routing, so Engine::layer_step runs it on a side stream while this kernel and the router run.
__global__ void route_schedule(const float* logits, const float* bias, float routed_scale, Sched o) {
    constexpr int T = 32, K = 8;
    __shared__ RouteSmem S;
    const int e = threadIdx.x;
    route_core(logits, bias, routed_scale, S);
    if (e < T)
        for (int k = 0; k < K; ++k) { o.topk[e * K + k] = S.sel[e][k]; o.topw[e * K + k] = S.selw[e][k]; }
    const int c = __popc(S.tokmask[e]);
    int ctot;
    const int start = block_excl_scan256(c, S.ws, &ctot);
    S.startx[e] = start;
    const int k1 = c >= 1 && c <= 8, k2 = c > 8 && c <= 16, k4 = c > 16;
    int t1, t2, t4;
    const int w1 = block_excl_scan256(k1, S.ws, &t1);   // (its barriers also publish startx)
    const int w2 = block_excl_scan256(k2, S.ws, &t2);
    const int w4 = block_excl_scan256(k4, S.ws, &t4);
    const moe8::Work wk{e, start, c};
    if (k1) o.work1[w1] = wk;
    if (k2) o.work2[w2] = wk;
    if (k4) o.work4[w4] = wk;
    if (e == 0) {
        o.nwork[0] = t1;
        o.nwork[1] = t2;
        o.nwork[2] = t4;
    }
    {
        const int t = e >> 3, k = e & 7, x = S.sel[t][k];
        const int sl = S.startx[x] + __popc(S.tokmask[x] & ((1u << t) - 1u));
        o.slot_token[sl] = t;
        o.slot_weight[sl] = S.selw[t][k];
        o.tok_slots[t * (K + 1) + k] = sl;
    }
    if (e < T) {
        o.slot_token[T * K + e] = e;
        o.slot_weight[T * K + e] = 1.f;
        o.tok_slots[e * (K + 1) + K] = T * K + e;
    }
}

// Batched prefill (nb blocks, TT = 32 nb tokens), three kernels. Routing stays per 32-token block (the model's
// block pool); the slots of all blocks are merged per expert so each expert's weights stream once per 32 slots.
// 1) grid nb: route each block; selections and token masks to global.
__global__ void route_batch_select(const float* logits, const float* bias, float routed_scale, Sched o) {
    __shared__ RouteSmem S;
    const int j = blockIdx.x, e = threadIdx.x;
    route_core(logits + size_t(j) * ROUTER_KC * 32 * 256, bias, routed_scale, S);
    const int t = e >> 3, k = e & 7;
    o.bsel[(j * 32 + t) * 8 + k] = S.sel[t][k];
    o.bselw[(j * 32 + t) * 8 + k] = S.selw[t][k];
    o.bmask[j * 256 + e] = S.tokmask[e];
}
// 2) one CTA, thread = expert: slot ranges (expert-major, then block, then token) and the work lists; items of
// <= 32 slots: <= 8 -> work1, 9..16 -> work2, else work4. The shared expert's slots [8 TT, 9 TT) go in work4.
__global__ void route_batch_plan(int nb, int shared_expert, Sched o) {
    __shared__ int ws[8];
    const int e = threadIdx.x, TT = nb * 32;
    int c = 0;
    for (int j = 0; j < nb; ++j) c += __popc(o.bmask[j * 256 + e]);
    int ctot;
    const int start = block_excl_scan256(c, ws, &ctot);
    int run = start;
    for (int j = 0; j < nb; ++j) { o.bpre[j * 256 + e] = run; run += __popc(o.bmask[j * 256 + e]); }
    int n1 = 0, n2 = 0, n4 = 0;
    for (int q = 0; q < c; q += 32) { const int n = min(32, c - q); (n <= 8 ? n1 : n <= 16 ? n2 : n4)++; }
    int t1, t2, t4;
    int w1 = block_excl_scan256(n1, ws, &t1);
    int w2 = block_excl_scan256(n2, ws, &t2);
    int w4 = block_excl_scan256(n4, ws, &t4);
    for (int q = 0; q < c; q += 32) {
        const int n = min(32, c - q);
        const moe8::Work wk{e, start + q, n};
        if (n <= 8) o.work1[w1++] = wk; else if (n <= 16) o.work2[w2++] = wk; else o.work4[w4++] = wk;
    }
    if (e < nb) o.work4[t4 + e] = moe8::Work{shared_expert, TT * 8 + 32 * e, 32};
    if (e == 0) { o.nwork[0] = t1; o.nwork[1] = t2; o.nwork[2] = t4 + nb; }
}
// 3) grid nb, thread = (token, k) pair of the block: place the slots.
__global__ void route_batch_place(int nb, Sched o) {
    const int j = blockIdx.x, e = threadIdx.x, t = e >> 3, k = e & 7, TT = nb * 32, tok = j * 32 + t;
    const int x = o.bsel[tok * 8 + k];
    const unsigned m = o.bmask[j * 256 + x];
    const int sl = o.bpre[j * 256 + x] + __popc(m & ((1u << t) - 1u));
    o.slot_token[sl] = tok;
    o.slot_weight[sl] = o.bselw[tok * 8 + k];
    o.tok_slots[tok * 9 + k] = sl;
    if (k == 0) {
        o.slot_token[TT * 8 + tok] = tok;
        o.slot_weight[TT * 8 + tok] = 1.f;
        o.tok_slots[tok * 9 + 8] = TT * 8 + tok;
    }
}

// q/k RMSNorm per head + partial RoPE (first 64 dims, rotate_half, cos/sin rounded to bf16 as the reference
// model computes them); v passes through. qkv fp32 [T, 3072] (q 16 heads, k 4, v 4). grid (T, 3), 256 threads:
// warp = head; lane l holds dims l, l+32, l+64, l+96, so the rotate_half pairs (d, d+32) share a lane.
// inv_freq[i] = theta^(-2i/64), from the host.
__global__ void qk_norm_rope(const float* qkv, const int* pos, const float* gq, const float* gk, float eps,
                             const float* inv_freq, half* q, half* k, half* v) {
    const int t = blockIdx.x, hd = blockIdx.y * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
    const float* src = qkv + size_t(t) * 3072 + hd * 128;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = src[l + 32 * i];
    if (hd >= 20) {                                     // v head
#pragma unroll
        for (int i = 0; i < 4; ++i) v[(size_t(t) * 4 + (hd - 20)) * 128 + l + 32 * i] = __float2half(x[i]);
        return;
    }
    const float r = rsqrtf(warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) / 128.f + eps);
    const float* gm = hd < 16 ? gq : gk;
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) n[i] = x[i] * r * gm[l + 32 * i];
    float sn, cs;
    sincosf(float(pos[t]) * inv_freq[l], &sn, &cs);
    cs = __bfloat162float(__float2bfloat16(cs)); sn = __bfloat162float(__float2bfloat16(sn));
    const float o0 = n[0] * cs - n[1] * sn, o1 = n[1] * cs + n[0] * sn;
    half* dst = hd < 16 ? q + (size_t(t) * 16 + hd) * 128 : k + (size_t(t) * 4 + (hd - 16)) * 128;
    dst[l] = __float2half(o0); dst[l + 32] = __float2half(o1);
    dst[l + 64] = __float2half(n[2]); dst[l + 96] = __float2half(n[3]);
}

// Warp FWHT of 32 * N values, lane-major (value index = lane * N + j): log2(N) stages in registers, 5 by shuffle.
template <int N>
__device__ __forceinline__ void warp_fwht(float (&x)[N], int lane) {
#pragma unroll
    for (int h = 1; h < N; h <<= 1)
#pragma unroll
        for (int j = 0; j < N; ++j)
            if (!(j & h)) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const bool hi = lane & o;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const float y = __shfl_xor_sync(0xffffffffu, x[j], o);
            x[j] = hi ? y - x[j] : x[j] + y;
        }
    }
}

// h[t] += (sum of the token's slots of Y, routed + shared, in the Q basis) · Qᵀ, Qᵀ = H diag(d) / sqrt(2048).
// grid T, 256 threads. Thread i first owns values 8i .. 8i+7: index bits 0-2 in registers, 3-7 by shuffle; one
// shared-memory transpose then gives it values i + 256 j, and bits 8-10 finish in registers.
__global__ void moe_combine_unrot(const float* Y, const int* tok_slots, int nk, const float* signs, float* h, int T) {
    __shared__ __align__(16) float v[HID];
    const int t = blockIdx.x, i = threadIdx.x, lane = i & 31;
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = 0.f;
    for (int k = 0; k < nk; ++k) {
        const float4* y = reinterpret_cast<const float4*>(Y + size_t(tok_slots[t * nk + k]) * HID + i * 8);
        const float4 a = y[0], b = y[1];
        x[0] += a.x; x[1] += a.y; x[2] += a.z; x[3] += a.w; x[4] += b.x; x[5] += b.y; x[6] += b.z; x[7] += b.w;
    }
    warp_fwht<8>(x, lane);                               // bits 0-2 local, lane bits = index bits 3-7
#pragma unroll
    for (int j = 0; j < 8; ++j) v[i * 8 + j] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = v[i + 256 * j];
#pragma unroll
    for (int hh = 1; hh < 8; hh <<= 1)                    // bits 8-10
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (!(j & hh)) { const float a = x[j], b = x[j + hh]; x[j] = a + b; x[j + hh] = a - b; }
    float* hp = h + size_t(t) * HID;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = i + 256 * j;
        hp[c] += x[j] * signs[c] * 0.02209708691207961f;
    }
}

// moe_combine_unrot, then the next layer's input norm (norm_rot_q8 without hn) on the updated rows -> out: one
// launch instead of two, and h is not re-read. grid T, block 256.
__global__ void moe_combine_norm(const float* Y, const int* tok_slots, int nk, const float* signs, float* h, float eps,
                                 moe8::Act out) {
    __shared__ __align__(16) float v[HID];
    __shared__ float red[8];
    const int t = blockIdx.x, i = threadIdx.x, lane = i & 31;
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = 0.f;
    for (int k = 0; k < nk; ++k) {
        const float4* y = reinterpret_cast<const float4*>(Y + size_t(tok_slots[t * nk + k]) * HID + i * 8);
        const float4 a = y[0], b = y[1];
        x[0] += a.x; x[1] += a.y; x[2] += a.z; x[3] += a.w; x[4] += b.x; x[5] += b.y; x[6] += b.z; x[7] += b.w;
    }
    warp_fwht<8>(x, lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[i * 8 + j] = x[j];
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = v[i + 256 * j];
#pragma unroll
    for (int hh = 1; hh < 8; hh <<= 1)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (!(j & hh)) { const float a = x[j], b = x[j + hh]; x[j] = a + b; x[j + hh] = a - b; }
    float* hp = h + size_t(t) * HID;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = i + 256 * j;
        x[j] = hp[c] + x[j] * signs[c] * 0.02209708691207961f;
        hp[c] = x[j];
    }
    __syncthreads();                                     // all reads of the transposed v are done
#pragma unroll
    for (int j = 0; j < 8; ++j) v[i + 256 * j] = x[j];
    __syncthreads();
    float y[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = v[i * 8 + j];
    __syncthreads();
    norm_rot_body(y, signs, nullptr, eps, out, nullptr, t, v, red);
}

__global__ void add_inplace(float* h, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] += y[i];
}

}  // namespace ops
