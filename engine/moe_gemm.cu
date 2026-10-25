// Routed-expert GEMMs for LLaDA2.2-mini on Orin (sm_87): mixed 2/3/4-bit tiles (quant/packlib.py format).
//
// Decode shape: one denoising step runs a 32-token block; block routing picks a pool of <= 48 experts per layer and
// each token its top-8 inside it, so an expert sees ~5 tokens. That is a weight-streaming problem: swap-AB puts the
// 16 weight rows of a strip in mma.m16n8k16's M and the tokens in N=8, so every byte of weights is read once and
// tokens ride along. A block gathers its item's token rows into shared memory once; each warp then streams MS strips
// (16 rows each) through a register ring DEPTH k-tiles deep, reusing every activation fragment across its strips.
//
//   moe_gemm<.., 0> gate_up : Z[slot, :] = SiLU(x' Wg'ᵀ) * (x' Wu'ᵀ)   x' = rmsnorm_noγ(h) Q, rows gathered per slot
//   fwht512     : Z' = Z H512 (in place)
//   moe_gemm<.., 1> down    : Y[slot, :] = w_slot * (Z' Wd'ᵀ)
//   moe_combine : acc[token, :] = sum of the token's slots        acc is in the Q-rotated basis (undo once per token)
//
// A slot is one (token, expert) pair; work items list each active expert's contiguous slot range.
#include <cuda_fp16.h>
#include <cstdint>

namespace moe {

constexpr int GROUP = 128;
constexpr int META = 48;
constexpr int MAX_NT = 2;            // token tiles per work item: launches are split by NT (1: <= 8 tokens, 2: <= 16),
                                     // larger counts are chunked; registers are sized per NT instantiation
#ifndef MOE_DEPTH
#define MOE_DEPTH 4
#endif
#ifndef MOE_MINB
#define MOE_MINB 1
#endif
constexpr int DEPTH = MOE_DEPTH;     // tiles in flight per warp

__host__ __device__ constexpr int tile_wbytes(int bits) { return bits == 4 ? 1024 : bits == 3 ? 768 : 512; }
__host__ __device__ constexpr int tile_bytes(int bits) { return tile_wbytes(bits) + META; }

struct Work {
    int expert;
    int slot0;     // first slot of this item
    int ntok;      // <= 8 * NT of the launch it is in
};

struct ExpertTable {
    const uint8_t* base;
    const uint64_t* gu_off;   // [E] byte offsets into base
    const uint64_t* d_off;
    const int8_t* bits;       // [E] 2, 3, 4 (0 = pruned, never scheduled)
};

template <int BITS>
struct Raw {
    uint4 lo;
    uint4 hi4;     // 4-bit second half
    uint2 hi3;     // 3-bit top-bit plane
    uint32_t sz;   // scale r0 | scale r0+8 << 16 (fp16 bits)
    uint32_t zz;   // zero r0 | zero r0+8 << 16
};

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// k16 step j of the tile as RAW codes in the 4 A registers (half2 of 1024 + code, exact): a0 (r0), a1 (r0+8),
// a2 (r0, +8), a3 (r0+8, +8). Scale and zero are applied once per group in fp32 (see strip_mm_nt), so the per-weight
// cost is one shift + one LOP3; dequantizing every weight (sub + mul) left the kernel ALU-bound at ~45 GB/s.
template <int BITS>
__device__ __forceinline__ void code_step(const Raw<BITS>& r, int j, uint32_t a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t q;
        if (BITS == 4) {
            const uint32_t w = j < 4 ? word(r.lo, j) : word(r.hi4, j - 4);
            q = (w >> (4 * i)) & 0x000F000Fu;
        } else {
            const uint32_t w = word(r.lo, j >> 1);
            q = (w >> (2 * ((j & 1) * 4 + i))) & 0x00030003u;
            if (BITS == 3) {
                const uint32_t h = (j >> 2) ? r.hi3.y : r.hi3.x;
                const int bp = (j & 3) * 4 + i;                 // bit of this pair in the high plane -> bit 2
                q |= (bp >= 2 ? (h >> (bp - 2)) : (h << (2 - bp))) & 0x00040004u;
            }
        }
        a[i] = q | 0x64006400u;
    }
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm(                                                    // not volatile: lets loads be scheduled across it
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Activations are stored in B-FRAGMENT ORDER inside every 128-column group: logical column c = 16j + r (r < 16)
// sits at 32t + 4j + q with t = (r % 8) / 2, q = 2 (r / 8) + r % 2. Lane t's whole tile of B operands for one token is
// then 64 contiguous bytes (4 x uint4) instead of 16 scattered words. Producers (fwht kernels) write this order.
__host__ __device__ __forceinline__ int frag_pos(int c) {
    const int base = c & ~127, j = (c & 127) >> 4, r = c & 15;
    return base + 32 * ((r & 7) >> 1) + 4 * j + 2 * (r >> 3) + (r & 1);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(unsigned(__cvta_generic_to_shared(smem))), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(unsigned(__cvta_generic_to_shared(smem))), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int BITS>
__device__ __forceinline__ void load_raw(const uint8_t* tile, int lane, Raw<BITS>& r) {
    r.lo = __ldg(reinterpret_cast<const uint4*>(tile) + lane);
    if (BITS == 4) r.hi4 = __ldg(reinterpret_cast<const uint4*>(tile + 512) + lane);
    if (BITS == 3) r.hi3 = __ldg(reinterpret_cast<const uint2*>(tile + 512) + lane);
    const uint8_t* m = tile + tile_wbytes(BITS);
    const int g = lane >> 2;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(m);
    r.sz = uint32_t(__ldg(sc + g)) | (uint32_t(__ldg(sc + g + 8)) << 16);
    r.zz = uint32_t(__ldg(m + 32 + g)) | (uint32_t(__ldg(m + 32 + g + 8)) << 16);
}

// Shared memory for one work item's B: the NT*8 gathered token rows for ALL k-tiles as [kt][q][token][t] 16-byte
// units (lane t's 64-byte run of a tile is 4 units q; the 8 lanes of a 128-bit load phase hit 8 different units),
// then their group sums [token][kt].
__host__ __device__ constexpr int smem_bytes(int NT, int K) { return (K / GROUP) * 8 * NT * (GROUP * 2 + 4); }

struct GemmArgs {
    ExpertTable tab;
    const Work* work;
    int nwork;
    const int* slot_token;   // MODE 0: B row of a slot = its token's row of X
    const half* X;           // MODE 0: x' [tokens, H] ; MODE 1: Z' [slots, I]   (fragment order)
    const float* SX;         // group sums of X rows
    const float* slot_weight;
    int H, I;
    half* Z;                 // MODE 0 out: SiLU(g) * u [slots, I] (logical order)
    float* Y;                // MODE 1 out: w_slot * (Z' Wd'ᵀ) [slots, H]
};

// Why this shape (measured on the Orin, tests/ and bench/): weights stream at the full ~100 GB/s through per-lane
// __ldg register rings, but the activations do not -- for 8 tokens a k-tile of B is 2 KB against 560 B of 2-bit
// weights, and gathered rows read through L1 touch 16 lines per load, which held a warp-per-strip kernel to
// ~32 GB/s. So B is gathered ONCE per block into conflict-free shared memory and each warp reuses every B
// fragment across MS strips; A stays in registers, DEPTH k-tiles ahead per strip.
template <int BITS, int NT, int MS, int MODE>
__device__ __forceinline__ void gemm_body(const GemmArgs& g_, const Work& w, const uint4* Bs, const float* Ss,
                                          const uint8_t* mat, int ktiles) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    constexpr int TB = tile_bytes(BITS), NTOK = 8 * NT;
    const int strip0 = (blockIdx.x * (blockDim.x >> 5) + warp) * MS;
    const uint8_t* sp[MS];
#pragma unroll
    for (int m = 0; m < MS; ++m) sp[m] = mat + size_t(strip0 + m) * ktiles * TB;
    Raw<BITS> buf[DEPTH][MS];
#pragma unroll
    for (int i = 0; i < DEPTH; ++i)
#pragma unroll
        for (int m = 0; m < MS; ++m) load_raw<BITS>(sp[m] + i * TB, lane, buf[i][m]);
    cp_wait<0>();
    __syncthreads();                                       // B staged (its copy overlapped the first A loads)
    float c[MS][NT][4];
#pragma unroll
    for (int m = 0; m < MS; ++m)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) c[m][nt][0] = c[m][nt][1] = c[m][nt][2] = c[m][nt][3] = 0.f;
    for (int kt0 = 0; kt0 < ktiles; kt0 += DEPTH) {
#pragma unroll
        for (int i = 0; i < DEPTH; ++i) {
            const int kt = kt0 + i;
            Raw<BITS> cur[MS];
#pragma unroll
            for (int m = 0; m < MS; ++m) {
                cur[m] = buf[i][m];
                if (kt + DEPTH < ktiles) load_raw<BITS>(sp[m] + (kt + DEPTH) * TB, lane, buf[i][m]);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int n = nt * 8 + g;
                uint4 bq[4];
#pragma unroll
                for (int q = 0; q < 4; ++q) bq[q] = Bs[((kt * 4 + q) * NTOK + n) * 4 + t];
                const float x0 = Ss[(nt * 8 + 2 * t) * ktiles + kt], x1 = Ss[(nt * 8 + 2 * t + 1) * ktiles + kt];
#pragma unroll
                for (int m = 0; m < MS; ++m) {
                    float p[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};   // even / odd k16 chains
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        uint32_t aa[4];
                        code_step<BITS>(cur[m], j, aa);
                        mma16816(p[j & 1], aa, word(bq[j >> 1], (j & 1) * 2), word(bq[j >> 1], (j & 1) * 2 + 1));
                    }
                    const float s0 = __half2float(__ushort_as_half(uint16_t(cur[m].sz & 0xFFFF)));
                    const float s1 = __half2float(__ushort_as_half(uint16_t(cur[m].sz >> 16)));
                    const float z0 = 1024.f + float(cur[m].zz & 0xFF), z1 = 1024.f + float((cur[m].zz >> 16) & 0xFF);
                    c[m][nt][0] += s0 * ((p[0][0] + p[1][0]) - z0 * x0);
                    c[m][nt][1] += s0 * ((p[0][1] + p[1][1]) - z0 * x1);
                    c[m][nt][2] += s1 * ((p[0][2] + p[1][2]) - z1 * x0);
                    c[m][nt][3] += s1 * ((p[0][3] + p[1][3]) - z1 * x1);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MS; ++m) {
        const int strip = strip0 + m;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int n = nt * 8 + 2 * t + e;
                if (n < w.ntok) {
                    const int slot = w.slot0 + n;
                    if (MODE == 0) {                       // rows 0..7 gate, 8..15 up of intermediate col strip*8+g
                        const float gt = c[m][nt][e], up = c[m][nt][2 + e];
                        g_.Z[size_t(slot) * g_.I + strip * 8 + g] = __float2half(gt / (1.f + __expf(-gt)) * up);
                    } else {
                        float* o = g_.Y + size_t(slot) * g_.H + strip * 16 + g;
                        const float sw = g_.slot_weight[slot];
                        o[0] = sw * c[m][nt][e];
                        o[8] = sw * c[m][nt][2 + e];
                    }
                }
            }
        }
    }
}

// grid: x = row blocks of WARPS*MS strips, y = work item. MODE 0: gate_up (rows 2I interleaved gate/up, K = H);
// MODE 1: down (rows H, K = I). Dynamic shared memory = smem_bytes(NT, K). ktiles must be a multiple of DEPTH.
template <int WARPS, int NT, int MS, int MODE>
__global__ void __launch_bounds__(WARPS * 32, MOE_MINB) moe_gemm(GemmArgs g_) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int NTOK = 8 * NT;
    const Work w = g_.work[blockIdx.y];
    const int K = MODE == 0 ? g_.H : g_.I, ktiles = K / GROUP;
    uint4* Bs = reinterpret_cast<uint4*>(smem);
    float* Ss = reinterpret_cast<float*>(smem + ktiles * NTOK * GROUP * 2);
    auto row_of = [&](int n) -> size_t {                    // B row of token n (n >= ntok aliases token 0)
        const int sl = w.slot0 + (n < w.ntok ? n : 0);
        return size_t(MODE == 0 ? g_.slot_token[sl] : sl);
    };
    for (int v = threadIdx.x; v < NTOK * ktiles * 16; v += WARPS * 32) {
        const int n = v / (ktiles * 16), r = v % (ktiles * 16), kt = r >> 4, cc = r & 15;
        cp_async16(Bs + ((kt * 4 + (cc & 3)) * NTOK + n) * 4 + (cc >> 2), g_.X + row_of(n) * K + kt * GROUP + 8 * cc);
    }
    for (int v = threadIdx.x; v < NTOK * ktiles; v += WARPS * 32)
        cp_async4(Ss + v, g_.SX + row_of(v / ktiles) * ktiles + v % ktiles);
    cp_commit();
    const int bits = g_.tab.bits[w.expert];
    const uint8_t* mat = g_.tab.base + (MODE == 0 ? g_.tab.gu_off[w.expert] : g_.tab.d_off[w.expert]);
    if (bits == 2) gemm_body<2, NT, MS, MODE>(g_, w, Bs, Ss, mat, ktiles);
    else if (bits == 3) gemm_body<3, NT, MS, MODE>(g_, w, Bs, Ss, mat, ktiles);
    else gemm_body<4, NT, MS, MODE>(g_, w, Bs, Ss, mat, ktiles);
}

// In-place Z H512 per slot (logical order in, fragment order out) + SZ [slots, 4] sums of the rounded outputs per
// 128-group (for the zero-point term of moe_down). One block of 256 threads per slot, 2 values each.
__global__ void fwht512(half* Z, float* SZ, int nslots) {
    __shared__ float v[512];
    __shared__ float part[16];
    const int s = blockIdx.x;
    if (s >= nslots) return;
    half* z = Z + size_t(s) * 512;
    const int i = threadIdx.x;
    v[i] = __half2float(z[i]);
    v[i + 256] = __half2float(z[i + 256]);
    __syncthreads();
    for (int hlen = 1; hlen < 512; hlen <<= 1) {
        const int blk = i / hlen, off = i % hlen;
        const int a = blk * 2 * hlen + off, b = a + hlen;
        const float x = v[a], y = v[b];
        __syncthreads();
        v[a] = x + y;
        v[b] = x - y;
        __syncthreads();
    }
    const float sc = 0.04419417382415922f;                 // 1/sqrt(512)
    const half h0 = __float2half(v[i] * sc), h1 = __float2half(v[i + 256] * sc);
    z[frag_pos(i)] = h0;
    z[frag_pos(i + 256)] = h1;
    float a0 = __half2float(h0), a1 = __half2float(h1);    // warp w covers group w/4 (a0) and 2 + w/4 (a1)
    for (int o = 16; o; o >>= 1) {
        a0 += __shfl_xor_sync(0xffffffffu, a0, o);
        a1 += __shfl_xor_sync(0xffffffffu, a1, o);
    }
    if ((i & 31) == 0) { part[i >> 5] = a0; part[8 + (i >> 5)] = a1; }
    __syncthreads();
    if (i < 4) {                                            // fixed summation order: reproducible
        const int base = (i < 2 ? 0 : 8) + (i & 1) * 4;
        SZ[size_t(s) * 4 + i] = part[base] + part[base + 1] + part[base + 2] + part[base + 3];
    }
}

// acc[t, :] = sum_k Y[tok_slots[t, k], :]   (K = 8 slots per token, rotated basis). One block per token.
__global__ void moe_combine(const float* Y, const int* tok_slots, int K, int H, float* acc) {
    const int t = blockIdx.x;
    for (int h = threadIdx.x * 4; h < H; h += blockDim.x * 4) {
        float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < K; ++k) {
            const float4 v = *reinterpret_cast<const float4*>(Y + size_t(tok_slots[t * K + k]) * H + h);
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
        *reinterpret_cast<float4*>(acc + size_t(t) * H + h) = s;
    }
}

}  // namespace moe
