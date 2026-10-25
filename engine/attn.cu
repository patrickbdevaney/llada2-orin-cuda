// Decode attention for LLaDA2.2-mini on Orin: one 32-token diffusion block against the quantized KV cache.
//
// Cache format "4c/4" (chosen by s06kv, QUANT_PLAN §10): per layer, per KV head, per committed 32-token block
//   K codes [32 tok][128 ch] 4-bit (byte = ch 2i low nibble | ch 2i+1 high), K scale[128] fp16, K min[128] fp16
//     (per CHANNEL over the block's 32 tokens, KIVI-style: keys carry per-channel outliers)
//   V codes [32 tok][128 d] 4-bit, V scale[32] fp16, V min[32] fp16 (per TOKEN)
//   value = code * scale + min.  4736 B per block per head = 11.8 KB per token over 20 layers x 4 heads.
// The block being denoised is never in the cache: its fresh K/V (fp16) are attended exactly, and kv_commit
// quantizes them once the block is final.
//
// attn_partial: grid (kv_head, chunk). A CTA takes the 128 queries of one KV head (32 tokens x 4 GQA heads, row r =
// token r/4, head 4*kvh + r%4), 8 warps x 16 rows, and a chunk of cache blocks (the last chunk index = the fresh
// block). Per block: cp.async the codes (double buffer) -> dequantize to fp16 in padded shared memory -> S = Q Kᵀ
// (mma.m16n8k16, ldmatrix) -> online softmax in registers -> O += P V (P reused from the S accumulators, V via
// ldmatrix.trans). Writes un-normalised partials; attn_combine merges chunks.
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace attn {

constexpr int D = 128, BLK = 32, QR = 128;               // head dim, block tokens, query rows per KV head
constexpr int KV_BLOCK_BYTES = 32 * 64 + 128 * 4 + 32 * 64 + 32 * 4;   // 4736
constexpr int KOFF_META = 2048, VOFF = 2560, VOFF_META = 4608;
constexpr int LDS = D + 8;                                // padded fp16 row stride (272 B: ldmatrix conflict-free)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(unsigned(__cvta_generic_to_shared(smem))), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(unsigned(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t r[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(unsigned(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

struct Args {
    const half* Q;          // [nbatch][32 tok][16 heads][128]  (post q-norm, RoPE)
    const half* Kf;         // [nbatch][32 tok][4 heads][128]   fresh blocks
    const half* Vf;
    const uint8_t* cache;   // this layer: [4 heads][max_blocks][KV_BLOCK_BYTES]
    int max_blocks;
    int nblocks;            // committed cache blocks visible to batch block 0 (block j sees nblocks + j)
    int blocks_per_chunk;
    int nchunks;            // chunk slots per (batch block, head) in the partial buffers (>= any block's count)
    float scale;            // 1/sqrt(128)
    float* Opart;           // [nbatch][4][nchunks][128 rows][128]  un-normalised
    float* Mpart;           // [nbatch][4][nchunks][128]  running max (log2 domain, scaled)
    float* Lpart;           // [nbatch][4][nchunks][128]
};

// chunks of batch block j: cache blocks split into runs of bpc, the fresh block appended to the last run
__host__ __device__ inline int chunks_for(int nblocks, int bpc) { return nblocks <= 0 ? 1 : (nblocks + bpc - 1) / bpc; }

#ifndef ATTN_STAGES
#define ATTN_STAGES 2
#endif
constexpr int STAGES = ATTN_STAGES;                       // cache blocks in flight (cp.async ring)
// Q tile, two K/V tile pairs (dequantize block b+1 while computing block b), staging ring
constexpr int SMEM_BYTES = QR * LDS * 2 + 2 * 2 * BLK * LDS * 2 + STAGES * KV_BLOCK_BYTES;

// 8 nibbles of w -> 4 half2 in channel order (c0,c1) (c2,c3) (c4,c5) (c6,c7), each value = nibble exactly:
// (w >> 4k) & 0x000f000f | 0x6400 0x6400 is half2(1024 + n_k, 1024 + n_{k+4}); subtract 1024, then pair up.
__device__ __forceinline__ void nib8_to_h2(uint32_t w, half2 out[4]) {
    uint32_t r[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t x;
        asm("lop3.b32 %0, %1, %2, %3, 0xea;\n" : "=r"(x) : "r"(w >> (4 * k)), "r"(0x000f000fu), "r"(0x64006400u));
        r[k] = x;                                            // (n_k, n_{k+4})
    }
    const half2 bias = __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400));
    const uint32_t p01 = __byte_perm(r[0], r[1], 0x5410), p23 = __byte_perm(r[2], r[3], 0x5410);
    const uint32_t p45 = __byte_perm(r[0], r[1], 0x7632), p67 = __byte_perm(r[2], r[3], 0x7632);
    out[0] = __hsub2(*reinterpret_cast<const half2*>(&p01), bias);
    out[1] = __hsub2(*reinterpret_cast<const half2*>(&p23), bias);
    out[2] = __hsub2(*reinterpret_cast<const half2*>(&p45), bias);
    out[3] = __hsub2(*reinterpret_cast<const half2*>(&p67), bias);
}

// dequantize one staged cache block into a K/V tile pair (256 threads, 16 values each per matrix), in half2
// arithmetic: code (exact in fp16) * scale + min is one fused rounding, as in fp32-then-round.
__device__ __forceinline__ void dequant_block(const uint8_t* st, half* Ks, half* Vs, int i) {
    const int tok = i >> 3, c0 = (i & 7) * 16;
    const uint2 kc = *reinterpret_cast<const uint2*>(st + tok * 64 + c0 / 2);
    const uint2 vc = *reinterpret_cast<const uint2*>(st + VOFF + tok * 64 + c0 / 2);
    const uint4* ks4 = reinterpret_cast<const uint4*>(st + KOFF_META + c0 * 2);
    const uint4* km4 = reinterpret_cast<const uint4*>(st + KOFF_META + 256 + c0 * 2);
    uint4 ksv[2] = {ks4[0], ks4[1]}, kmv[2] = {km4[0], km4[1]};
    const half2* ks2 = reinterpret_cast<const half2*>(ksv);  // channel pairs c0 + 2j
    const half2* km2 = reinterpret_cast<const half2*>(kmv);
    const half* vsm = reinterpret_cast<const half*>(st + VOFF_META);
    const half2 vs = __half2half2(vsm[tok]), vm = __half2half2(vsm[32 + tok]);
    half2 kh[8], vh[8];
    nib8_to_h2(kc.x, kh); nib8_to_h2(kc.y, kh + 4);
    nib8_to_h2(vc.x, vh); nib8_to_h2(vc.y, vh + 4);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        kh[j] = __hfma2(kh[j], ks2[j], km2[j]);
        vh[j] = __hfma2(vh[j], vs, vm);
    }
    uint4* kd = reinterpret_cast<uint4*>(Ks + tok * LDS + c0);
    uint4* vd = reinterpret_cast<uint4*>(Vs + tok * LDS + c0);
    const uint4* kq = reinterpret_cast<const uint4*>(kh);
    const uint4* vq = reinterpret_cast<const uint4*>(vh);
    kd[0] = kq[0]; kd[1] = kq[1];
    vd[0] = vq[0]; vd[1] = vq[1];
}

// The fresh block (fp16, exact) into a K/V tile pair. NTH threads, each 32 / (NTH / 8) tokens x 16 channels.
template <int NTH>
__device__ __forceinline__ void load_fresh(const half* Kf, const half* Vf, int kvh, half* Ks, half* Vs) {
    for (int i = threadIdx.x; i < 256; i += NTH) {
        const int tok = i >> 3, c0 = (i & 7) * 16;
        const uint4* ks = reinterpret_cast<const uint4*>(Kf + (size_t(tok) * 4 + kvh) * D + c0);
        const uint4* vs = reinterpret_cast<const uint4*>(Vf + (size_t(tok) * 4 + kvh) * D + c0);
        uint4* kd = reinterpret_cast<uint4*>(Ks + tok * LDS + c0);
        uint4* vd = reinterpret_cast<uint4*>(Vs + tok * LDS + c0);
        kd[0] = ks[0]; kd[1] = ks[1];
        vd[0] = vs[0]; vd[1] = vs[1];
    }
}

constexpr int ATT_WARPS = 4, ATT_NTH = ATT_WARPS * 32, ATT_MT = QR / 16 / ATT_WARPS;   // 2 m-tiles (32 rows) per warp

// grid (kv_head, chunk, batch block), 128 threads. Per chunk: its run of cache blocks, plus the fresh block on the
// last chunk. A warp owns 32 query rows (two m16 tiles), so every K and V fragment read from shared memory feeds
// two mma: 96 KB of shared-memory reads per block instead of 160 with 16 rows per warp.
// Software pipeline, one barrier per block: iteration b computes tile pair b&1 while dequantizing block b+1 into
// the other pair; cp.async stages block b+2.
__global__ void __launch_bounds__(ATT_NTH, 2) attn_partial(Args a) {
    extern __shared__ __align__(16) uint8_t smem[];
    half* Qs = reinterpret_cast<half*>(smem);
    half* KV = Qs + QR * LDS;                               // pair p: K at KV + 2p*BLK*LDS, V right after
    uint8_t* stage = reinterpret_cast<uint8_t*>(KV + 4 * BLK * LDS);
    const int kvh = blockIdx.x, chunk = blockIdx.y, jb = blockIdx.z;
    const int nbl = a.nblocks + jb;
    const int nch = chunks_for(nbl, a.blocks_per_chunk);
    if (chunk >= nch) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int b0 = chunk * a.blocks_per_chunk;
    const int ncache = nbl <= 0 ? 0 : min(a.blocks_per_chunk, nbl - b0);
    const bool last = chunk == nch - 1;
    const int nb = ncache + (last ? 1 : 0);                  // block ncache (if last) is the fresh block
    const uint8_t* cbase = a.cache + (size_t(kvh) * a.max_blocks + b0) * KV_BLOCK_BYTES;
    const half* Q = a.Q + size_t(jb) * 32 * 16 * D;
    const half* Kf = a.Kf + size_t(jb) * 32 * 4 * D;
    const half* Vf = a.Vf + size_t(jb) * 32 * 4 * D;

    for (int v = threadIdx.x; v < QR * 16; v += ATT_NTH) {  // Q rows (row r: token r/4, head 4 kvh + r%4)
        const int r = v >> 4, c = (v & 15) * 8;
        cp_async16(Qs + r * LDS + c, Q + (size_t(r >> 2) * 16 + 4 * kvh + (r & 3)) * D + c);
    }
    auto stage_block = [&](int b) {
        uint8_t* dst = stage + (b % STAGES) * KV_BLOCK_BYTES;
        const uint8_t* src = cbase + size_t(b) * KV_BLOCK_BYTES;
        for (int v = threadIdx.x; v < KV_BLOCK_BYTES / 16; v += ATT_NTH) cp_async16(dst + 16 * v, src + 16 * v);
    };
    auto fill = [&](int b, int pair) {                       // block b -> tile pair (dequant or fresh copy)
        half* Ks = KV + 2 * pair * BLK * LDS;
        if (b < ncache) {
            const uint8_t* stp = stage + (b % STAGES) * KV_BLOCK_BYTES;
            dequant_block(stp, Ks, Ks + BLK * LDS, threadIdx.x);
            dequant_block(stp, Ks, Ks + BLK * LDS, threadIdx.x + ATT_NTH);
        } else {
            load_fresh<ATT_NTH>(Kf, Vf, kvh, Ks, Ks + BLK * LDS);
        }
    };
#pragma unroll
    for (int i = 0; i < STAGES; ++i) {
        if (i < ncache) stage_block(i);
        cp_commit();
    }
    cp_wait<STAGES - 1>();
    __syncthreads();
    fill(0, 0);

    float o[ATT_MT][16][4];
#pragma unroll
    for (int m = 0; m < ATT_MT; ++m)
#pragma unroll
        for (int n = 0; n < 16; ++n) o[m][n][0] = o[m][n][1] = o[m][n][2] = o[m][n][3] = 0.f;
    float mrow[ATT_MT][2], lrow[ATT_MT][2];
#pragma unroll
    for (int m = 0; m < ATT_MT; ++m) { mrow[m][0] = mrow[m][1] = -INFINITY; lrow[m][0] = lrow[m][1] = 0.f; }
    const float sl2 = a.scale * 1.4426950408889634f;       // scores in log2 units
    const int qrow0 = warp * 16 * ATT_MT;

    for (int b = 0; b < nb; ++b) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (b + STAGES < ncache) stage_block(b + STAGES);
        cp_commit();
        if (b + 1 < nb) fill(b + 1, (b + 1) & 1);
        const half* Ks = KV + 2 * (b & 1) * BLK * LDS;
        const half* Vs = Ks + BLK * LDS;
        float s[ATT_MT][4][4];
#pragma unroll
        for (int m = 0; m < ATT_MT; ++m)
#pragma unroll
            for (int n = 0; n < 4; ++n) s[m][n][0] = s[m][n][1] = s[m][n][2] = s[m][n][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < D / 16; ++kk) {
            uint32_t qa[ATT_MT][4];
#pragma unroll
            for (int m = 0; m < ATT_MT; ++m)
                ldsm_x4(qa[m], Qs + (qrow0 + m * 16 + (lane & 15)) * LDS + kk * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t kb[4];
                const int key = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                ldsm_x4(kb, Ks + key * LDS + kk * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
                for (int m = 0; m < ATT_MT; ++m) {
                    mma16816(s[m][2 * np], qa[m], kb[0], kb[1]);
                    mma16816(s[m][2 * np + 1], qa[m], kb[2], kb[3]);
                }
            }
        }
        uint32_t ph[ATT_MT][4][2];
#pragma unroll
        for (int m = 0; m < ATT_MT; ++m) {
            // Lazy rescaling: a row's reference max moves only when this block's max exceeds it by more than
            // RESCALE_SLACK (log2 units), so p = 2^(s - m) <= 2^SLACK, exponentiated in fp32 and then rounded to fp16
            // (full fp16 relative precision at any magnitude); O is rescaled only when some row of the warp moved.
            constexpr float RESCALE_SLACK = 8.f;
            float mnew[2], corr[2];
            bool moved = false;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float mx = -INFINITY;
#pragma unroll
                for (int n = 0; n < 4; ++n) mx = fmaxf(mx, fmaxf(s[m][n][2 * h], s[m][n][2 * h + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const bool mv = mx * sl2 > mrow[m][h] + RESCALE_SLACK;
                mnew[h] = mv ? mx * sl2 : mrow[m][h];
                corr[h] = mv ? exp2f(mrow[m][h] - mnew[h]) : 1.f;
                mrow[m][h] = mnew[h];
                moved |= mv;
            }
            float rs[2] = {0.f, 0.f};
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float p0 = exp2f(s[m][n][2 * h] * sl2 - mnew[h]), p1 = exp2f(s[m][n][2 * h + 1] * sl2 - mnew[h]);
                    ph[m][n][h] = pack_h2(p0, p1);
                    rs[h] += p0 + p1;
                }
#pragma unroll
            for (int h = 0; h < 2; ++h) lrow[m][h] = lrow[m][h] * corr[h] + rs[h];
            if (__any_sync(0xffffffffu, moved)) {
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    o[m][n][0] *= corr[0]; o[m][n][1] *= corr[0];
                    o[m][n][2] *= corr[1]; o[m][n][3] *= corr[1];
                }
            }
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {                     // keys 16kk .. 16kk + 15
#pragma unroll
            for (int np = 0; np < 8; ++np) {                 // dim tiles 2np, 2np+1
                uint32_t vb[4];
                const int key = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                ldsm_x4_t(vb, Vs + key * LDS + np * 16 + (lane >> 4) * 8);
#pragma unroll
                for (int m = 0; m < ATT_MT; ++m) {
                    const uint32_t pa[4] = {ph[m][2 * kk][0], ph[m][2 * kk][1], ph[m][2 * kk + 1][0], ph[m][2 * kk + 1][1]};
                    mma16816(o[m][2 * np], pa, vb[0], vb[1]);
                    mma16816(o[m][2 * np + 1], pa, vb[2], vb[3]);
                }
            }
        }
    }
    const size_t base = ((size_t(jb) * 4 + kvh) * a.nchunks + chunk) * QR;
#pragma unroll
    for (int m = 0; m < ATT_MT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = lrow[m][h];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            const int r = qrow0 + m * 16 + g + 8 * h;
            if (t == 0) { a.Mpart[base + r] = mrow[m][h]; a.Lpart[base + r] = l; }
            float* op = a.Opart + (base + r) * D;
#pragma unroll
            for (int n = 0; n < 16; ++n)
                *reinterpret_cast<float2*>(op + n * 8 + 2 * t) = make_float2(o[m][n][2 * h], o[m][n][2 * h + 1]);
        }
}

// int8 activation fragment order within a 128-group (= moe8::frag_pos_i8)
__device__ __forceinline__ int frag_pos(int c) {
    const int base = c & ~127, r = c & 127;
    return base + 32 * ((r >> 2) & 3) + 8 * (r >> 5) + 4 * ((r >> 4) & 1) + (r & 3);
}

// O[j][tok][head][d] = sum_c 2^(m_c - M) O_c / sum_c 2^(m_c - M) l_c. grid (kvh, 128 / 8 rows, batch block),
// 256 threads: warp = one query row (token, head), lane = dims 4 lane .. 4 lane + 3.
// With q8 set, the row (one head = one 128-group of the dense layer's input) goes straight to the int8 activation
// layout the dense GEMM reads (symmetric per group, fragment order, + scale and int sum) and O is not written.
constexpr int COMB_ROWS = 8;
__global__ void attn_combine(const float* Opart, const float* Mpart, const float* Lpart, int nchunks, int nblocks,
                             int bpc, half* O, int8_t* q8, float* s8, int* sum8) {
    const int kvh = blockIdx.x, jb = blockIdx.z, lane = threadIdx.x & 31;
    const int r = blockIdx.y * COMB_ROWS + (threadIdx.x >> 5);
    const int nch = chunks_for(nblocks + jb, bpc);
    const size_t b0 = (size_t(jb) * 4 + kvh) * nchunks;
    float M = -INFINITY;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, Mpart[(b0 + c) * QR + r]);
    float4 num = make_float4(0.f, 0.f, 0.f, 0.f);
    float den = 0.f;
    for (int c = 0; c < nch; ++c) {
        const size_t i = (b0 + c) * QR + r;
        const float w = exp2f(Mpart[i] - M);
        const float4 v = *reinterpret_cast<const float4*>(Opart + i * D + 4 * lane);
        num.x += w * v.x; num.y += w * v.y; num.z += w * v.z; num.w += w * v.w;
        den += w * Lpart[i];
    }
    const float inv = 1.f / den;
    const float val[4] = {num.x * inv, num.y * inv, num.z * inv, num.w * inv};
    const int head = 4 * kvh + (r & 3);
    const size_t row = size_t(jb) * 32 + (r >> 2);
    if (!q8) {
        half2* o = reinterpret_cast<half2*>(O + (row * 16 + head) * D + 4 * lane);
        o[0] = __floats2half2_rn(val[0], val[1]);
        o[1] = __floats2half2_rn(val[2], val[3]);
        return;
    }
    float m = fmaxf(fmaxf(fabsf(val[0]), fabsf(val[1])), fmaxf(fabsf(val[2]), fabsf(val[3])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sc = fmaxf(m / 127.f, 1e-30f);
    int sm = 0;
    char4 qv;
    int8_t* qq = reinterpret_cast<int8_t*>(&qv);
#pragma unroll
    for (int k = 0; k < 4; ++k) { qq[k] = int8_t(max(-127, min(127, __float2int_rn(val[k] / sc)))); sm += qq[k]; }
    // dims 4 lane .. 4 lane + 3 are consecutive in the fragment order too (frag_pos keeps runs of 4)
    *reinterpret_cast<char4*>(q8 + row * 2048 + frag_pos(head * D + 4 * lane)) = qv;
    for (int o = 16; o; o >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, o);
    if (lane == 0) { s8[row * 16 + head] = sc; sum8[row * 16 + head] = sm; }
}

// chunking policy: ~4 chunks per head fill the GPU (4 heads x 4 chunks x 2 CTAs/SM on 8 SMs is one wave)
inline int pick_bpc(int nblocks, int nbatch) {
    const int want = nbatch >= 4 ? 1 : 4 / nbatch;           // batched prefill already has CTAs to spare
    return std::max(1, (nblocks + want - 1) / want);
}

// Quantize a final block's fresh K/V ([32 tok][4 heads][128] fp16) into cache block `blk` of every head.
// One block of 128 threads per head: thread = channel for K (min/max over 32 tokens), then = token-dim for V.
// grid (4 heads, nb): batch block j of the fresh K/V goes to cache block blk + j.
__global__ void kv_commit(const half* Kf, const half* Vf, uint8_t* cache, int max_blocks, int blk) {
    const int h = blockIdx.x, c = threadIdx.x;
    Kf += size_t(blockIdx.y) * 32 * 4 * D;
    Vf += size_t(blockIdx.y) * 32 * 4 * D;
    blk += blockIdx.y;
    uint8_t* dst = cache + (size_t(h) * max_blocks + blk) * KV_BLOCK_BYTES;
    __shared__ uint8_t kq[32][128], vq[32][128];
    __shared__ float vred[32][2][4];
    // K: per channel c over tokens
    float mn = INFINITY, mx = -INFINITY;
    for (int tk = 0; tk < 32; ++tk) {
        const float v = __half2float(Kf[(size_t(tk) * 4 + h) * D + c]);
        mn = fminf(mn, v); mx = fmaxf(mx, v);
    }
    half sh = __float2half(fmaxf((mx - mn) / 15.f, 6.2e-5f)), mh = __float2half(mn);
    const float s = __half2float(sh), m0 = __half2float(mh);
    for (int tk = 0; tk < 32; ++tk) {
        const float v = __half2float(Kf[(size_t(tk) * 4 + h) * D + c]);
        kq[tk][c] = uint8_t(max(0, min(15, __float2int_rn((v - m0) / s))));
    }
    reinterpret_cast<half*>(dst + KOFF_META)[c] = sh;
    reinterpret_cast<half*>(dst + KOFF_META)[128 + c] = mh;
    // V: per token over the 128 dims (thread c = dim); reduce min/max per token across the block
    const int w = c >> 5, l = c & 31;
    for (int tk = 0; tk < 32; ++tk) {
        float a0 = __half2float(Vf[(size_t(tk) * 4 + h) * D + c]), b0 = a0;
        for (int o = 16; o; o >>= 1) {
            a0 = fminf(a0, __shfl_xor_sync(0xffffffffu, a0, o));
            b0 = fmaxf(b0, __shfl_xor_sync(0xffffffffu, b0, o));
        }
        if (l == 0) { vred[tk][0][w] = a0; vred[tk][1][w] = b0; }
    }
    __syncthreads();
    half* vmeta = reinterpret_cast<half*>(dst + VOFF_META);
    for (int tk = 0; tk < 32; ++tk) {
        const float vmn = fminf(fminf(vred[tk][0][0], vred[tk][0][1]), fminf(vred[tk][0][2], vred[tk][0][3]));
        const float vmx = fmaxf(fmaxf(vred[tk][1][0], vred[tk][1][1]), fmaxf(vred[tk][1][2], vred[tk][1][3]));
        const half vsh = __float2half(fmaxf((vmx - vmn) / 15.f, 6.2e-5f)), vmh = __float2half(vmn);
        const float vs = __half2float(vsh), vm = __half2float(vmh);
        const float v = __half2float(Vf[(size_t(tk) * 4 + h) * D + c]);
        vq[tk][c] = uint8_t(max(0, min(15, __float2int_rn((v - vm) / vs))));
        if (c == 0) { vmeta[tk] = vsh; vmeta[32 + tk] = vmh; }
    }
    __syncthreads();
    for (int v = c; v < 32 * 64; v += 128) {               // pack nibbles: byte = ch 2i | ch 2i+1 << 4
        const int tk = v >> 6, i = v & 63;
        dst[v] = kq[tk][2 * i] | (kq[tk][2 * i + 1] << 4);
        dst[VOFF + v] = vq[tk][2 * i] | (vq[tk][2 * i + 1] << 4);
    }
}

}  // namespace attn
