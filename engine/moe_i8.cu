// Routed-expert GEMMs, W(2/3/4)·A8 path: weights enter mma.m16n8k32 as raw u8 codes (packlib format "i8"),
// activations as int8 with one fp32 scale per token and 128-group. Per group the tensor cores give the exact int32
//   P = sum_k q_k xq_k   and   W·x = sum_g s_w s_x (P_g - z_g Sxq_g),   Sxq_g = sum of the group's int8 activations.
// Against the fp16 path (moe_gemm.cu) this needs ~half the code-extraction ALU work (one shift + one mask yields
// 4 codes, not 2), half the mma instructions, and half the activation bytes; that fp16 path measured ALU-bound at
// ~45 GB/s once loads ran at ~110 GB/s.
//
// Shapes and structure as moe_gemm.cu: one denoising step of a 32-token block, swap-AB (16 weight rows in M, tokens
// in N = 8), the block gathers its item's token rows into conflict-free shared memory once, each warp streams MS
// strips through a register ring DEPTH k-tiles deep.
//   gemm<.., 0> gate_up : Z[slot, :] = SiLU(x' Wg'ᵀ) * (x' Wu'ᵀ)   (fp16, logical order)
//   fwht512_q8          : Z' = A8(Z H512) in fragment order, + scales and int sums
//   gemm<.., 1> down    : Y[slot, :] = w_slot * (Z' Wd'ᵀ)
//   combine             : acc[token, :] = sum of the token's slots (Q basis)
#include <cuda_fp16.h>
#include <cstdint>

namespace moe8 {

constexpr int GROUP = 128;
constexpr int META = 48;
#ifndef MOE_DEPTH
#define MOE_DEPTH 2
#endif
#ifndef MOE_MINB
#define MOE_MINB 3
#endif
constexpr int DEPTH = MOE_DEPTH;

// bits 5 = the HIGH nibble plane of a split 8-bit matrix (s07d): a 4-bit tile whose codes enter the mma as 16 h
__host__ __device__ constexpr int tile_wbytes(int bits) { return bits == 8 ? 2048 : bits == 4 || bits == 5 ? 1024 : bits == 3 ? 768 : 512; }
__host__ __device__ constexpr int tile_bytes(int bits) { return tile_wbytes(bits) + META; }

// Device layout of a strip (16 rows x ktot tiles), rewritten at load by relayout_strips from packlib's
// [tile = codes | meta] x ktot into
//   [codes of tiles 0 .. ktot-1][meta of tiles 0 .. ktot-1]          (same bytes)
// so the code stream is contiguous and sector aligned (an interleaved 48 B meta cost ~20% of DRAM throughput), and
// each 48 B meta is lane-ordered: u32 scale pair (rows g | g + 8 << 16) at 4 g, u16 zero pair (g | g + 8 << 8)
// at 32 + 2 g, g = 0..7 -- two loads per lane and tile instead of four.
__host__ __device__ constexpr size_t strip_meta_off(int ktot, int bits) { return size_t(ktot) * tile_wbytes(bits); }

struct Work {
    int expert;
    int slot0;
    int ntok;      // <= 8 * NT of the launch it is in
};

struct ExpertTable {
    const uint8_t* base;
    const uint64_t* gu_off;
    const uint64_t* d_off;
    const int8_t* bits;
};

// int8 activations in FRAGMENT ORDER inside every 128-group: logical k = 32j + 16h + 4t + e sits at 32t + 8j + 4h + e,
// so lane t's B operands for a whole tile are 32 contiguous bytes (2 x uint4).
__host__ __device__ __forceinline__ int frag_pos_i8(int c) {
    const int base = c & ~127, r = c & 127;
    return base + 32 * ((r >> 2) & 3) + 8 * (r >> 5) + 4 * ((r >> 4) & 1) + (r & 3);
}

struct Act {                 // one activation matrix: rows of int8 (fragment order) + per-group scale and int sum
    const int8_t* q;         // [rows, K]
    const float* s;          // [rows, K/128]
    const int* sum;          // [rows, K/128]
};

template <int BITS>
struct Raw {
    uint4 lo;
    uint4 hi4;     // 4-bit: words 4..7; 8-bit: uint4 1
    uint2 hi3;
    uint4 w8b, w8c;   // 8-bit: uint4 2, 3
    uint32_t sz;   // fp16 scale bits rows g | g+8 << 16
    uint32_t zz;   // zero rows g | g+8 << 8
};

// one tile from global memory: codes at `code`, its (relaid) meta at `meta`
template <int BITS>
__device__ __forceinline__ void load_raw(const uint8_t* code, const uint8_t* meta, int lane, Raw<BITS>& r) {
    r.lo = __ldg(reinterpret_cast<const uint4*>(code) + lane);
    if (BITS == 4 || BITS == 5 || BITS == 8) r.hi4 = __ldg(reinterpret_cast<const uint4*>(code + 512) + lane);
    if (BITS == 8) {
        r.w8b = __ldg(reinterpret_cast<const uint4*>(code + 1024) + lane);
        r.w8c = __ldg(reinterpret_cast<const uint4*>(code + 1536) + lane);
    }
    if (BITS == 3) r.hi3 = __ldg(reinterpret_cast<const uint2*>(code + 512) + lane);
    const int g = lane >> 2;
    r.sz = __ldg(reinterpret_cast<const uint32_t*>(meta) + g);
    r.zz = __ldg(reinterpret_cast<const uint16_t*>(meta + 32) + g);
}

// In-place packlib -> device strip layout (see strip_meta_off). grid: strips (grid-stride), 256 threads, dynamic
// shared memory ktot * tile_bytes(bits).
__global__ void relayout_strips(uint8_t* mat, int nstrips, int ktot, int bits) {
    extern __shared__ __align__(16) uint8_t rs[];
    const int TB = tile_bytes(bits), WB = tile_wbytes(bits), SB = ktot * TB;
    for (int s = blockIdx.x; s < nstrips; s += gridDim.x) {
        uint8_t* p = mat + size_t(s) * SB;
        for (int i = threadIdx.x; i < SB / 16; i += blockDim.x) reinterpret_cast<uint4*>(rs)[i] = reinterpret_cast<const uint4*>(p)[i];
        __syncthreads();
        for (int i = threadIdx.x; i < ktot * WB / 16; i += blockDim.x) {
            const int kt = i / (WB / 16), c = i % (WB / 16);
            reinterpret_cast<uint4*>(p)[i] = reinterpret_cast<const uint4*>(rs + kt * TB)[c];
        }
        for (int i = threadIdx.x; i < ktot * 8; i += blockDim.x) {
            const int kt = i >> 3, g = i & 7;
            const uint8_t* m = rs + kt * TB + WB;
            const uint16_t* sc = reinterpret_cast<const uint16_t*>(m);
            uint8_t* o = p + size_t(ktot) * WB + kt * META;
            reinterpret_cast<uint32_t*>(o)[g] = uint32_t(sc[g]) | (uint32_t(sc[g + 8]) << 16);
            reinterpret_cast<uint16_t*>(o + 32)[g] = uint16_t(m[32 + g] | (m[32 + g + 8] << 8));
        }
        __syncthreads();
    }
}
inline void relayout_matrix(uint8_t* mat, int rows, int K, int bits, cudaStream_t st = nullptr) {
    const int ktot = K / GROUP, sm = ktot * tile_bytes(bits), ns = rows / 16;
    cudaFuncSetAttribute(relayout_strips, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    relayout_strips<<<ns < 1024 ? ns : 1024, 256, sm, st>>>(mat, ns, ktot, bits);
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// k32 step j of the tile: the 4 A registers of u8 codes (packlib "i8" layout).
template <int BITS>
__device__ __forceinline__ void code_step(const Raw<BITS>& r, int j, uint32_t a[4]) {
    if (BITS == 8) {
        const uint4& v = j == 0 ? r.lo : j == 1 ? r.hi4 : j == 2 ? r.w8b : r.w8c;
        a[0] = v.x; a[1] = v.y; a[2] = v.z; a[3] = v.w;
    } else if (BITS == 5) {
        const uint4& v = j < 2 ? r.lo : r.hi4;
        const uint32_t w0 = word(v, (2 * j) & 3), w1 = word(v, (2 * j + 1) & 3);
        a[0] = (w0 << 4) & 0xF0F0F0F0u;
        a[1] = w0 & 0xF0F0F0F0u;
        a[2] = (w1 << 4) & 0xF0F0F0F0u;
        a[3] = w1 & 0xF0F0F0F0u;
    } else if (BITS == 4) {
        const uint4& v = j < 2 ? r.lo : r.hi4;
        const uint32_t w0 = word(v, (2 * j) & 3), w1 = word(v, (2 * j + 1) & 3);
        a[0] = w0 & 0x0F0F0F0Fu;
        a[1] = (w0 >> 4) & 0x0F0F0F0Fu;
        a[2] = w1 & 0x0F0F0F0Fu;
        a[3] = (w1 >> 4) & 0x0F0F0F0Fu;
    } else {
        const uint32_t w = word(r.lo, j);
#pragma unroll
        for (int k = 0; k < 4; ++k) a[k] = (w >> (2 * k)) & 0x03030303u;
        if (BITS == 3) {
            const uint32_t h = j < 2 ? r.hi3.x : r.hi3.y;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = (j & 1) * 4 + k;               // bit i of each byte -> bit 2
                a[k] |= (i >= 2 ? (h >> (i - 2)) : (h << (2 - i))) & 0x04040404u;
            }
        }
    }
}

// int -> float for |x| < 2^22 with full-rate ALU ops (I2F is quarter rate): x + 1.5 * 2^23 as float bits.
// Bound here: |sum_k q_k xq_k - z * Sxq| <= 255 * 127 * 128 < 2^22 per 128-group.
// HW: true selects I2F (quarter-rate pipe, but one issue slot): a win where conversions are sparse per tile (NT <= 2).
template <bool HW = false>
__device__ __forceinline__ float i2f_small(int x) {
    if (HW) return __int2float_rn(x);
    return __int_as_float(x + 0x4B400000) - 12582912.0f;
}

__device__ __forceinline__ void mma_i8(int c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(unsigned(__cvta_generic_to_shared(smem))), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(unsigned(__cvta_generic_to_shared(smem))), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// shared memory per block: B rows [kt][2 q][NTOK][4 t][16 B], then scales [kt][token], then int sums [kt][token]
__host__ __device__ constexpr int smem_bytes(int NT, int K) { return (K / GROUP) * 8 * NT * (GROUP + 8); }


struct GemmArgs {
    ExpertTable tab;
    const Work* work;
    const int* nwork;        // optional device count: blocks with blockIdx.y >= *nwork exit (GPU-side scheduling)
    const int* slot_token;   // MODE 0: B row of a slot is its token's row
    Act X;                   // MODE 0: x' [tokens, H] ; MODE 1: Z' [slots, I]
    const float* slot_weight;
    int H, I;
    half* Z;                 // MODE 0 out [slots, I]
    float* Y;                // MODE 1 out [slots, H] = w_slot * (.) ; MODE 2: Y [slots, H] += (.)  (residual add)
    int groups;              // strip groups per warp (grid.x = rows / 16 / (WARPS * MS * groups))
    int swap;                // grid (items, row blocks) instead of (row blocks, items): blocks that share a weight
                             // strip are adjacent in launch order, so a strip read by several items hits L2
};

// Write one finished strip group (MS strips starting at `strip`). Inlined at each ring position.
template <int NT, int MS, int MODE>
__device__ __forceinline__ void epilogue(const GemmArgs& g_, const Work& w, float (&c)[MS][NT][4], int sg, int strip0,
                                         int t, int g) {
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
                    if (MODE == 0) {
                        const float gt = c[m][nt][e], up = c[m][nt][2 + e];
                        g_.Z[size_t(slot) * g_.I + strip * 8 + g] = __float2half(__fdividef(gt, 1.f + __expf(-gt)) * up);
                    } else if (MODE == 1) {
                        float* o = g_.Y + size_t(slot) * g_.H + strip * 16 + g;
                        const float sw = g_.slot_weight[slot];
                        if (gridDim.z > 1) {               // split-K: Y was zeroed, partial sums meet in L2
                            atomicAdd(o, sw * c[m][nt][e]);
                            atomicAdd(o + 8, sw * c[m][nt][2 + e]);
                        } else {
                            o[0] = sw * c[m][nt][e];
                            o[8] = sw * c[m][nt][2 + e];
                        }
                    } else if (MODE == 3) {                // fp16 rows [slot][H] into Z (the screened readout)
                        half* o = g_.Z + size_t(slot) * g_.H + strip * 16 + g;
                        o[0] = __float2half(c[m][nt][e]);
                        o[8] = __float2half(c[m][nt][2 + e]);
                    } else {                               // one writer per element unless split-K
                        float* o = g_.Y + size_t(slot) * g_.H + strip * 16 + g;
                        if (gridDim.z > 1) {
                            atomicAdd(o, c[m][nt][e]);
                            atomicAdd(o + 8, c[m][nt][2 + e]);
                        } else {
                            o[0] += c[m][nt][e];
                            o[8] += c[m][nt][2 + e];
                        }
                    }
                }
            }
            c[m][nt][0] = c[m][nt][1] = c[m][nt][2] = c[m][nt][3] = 0.f;
        }
    }
    (void)sg;
}

#ifndef MOE_D2
#define MOE_D2 4
#endif
#ifndef MOE_D3
#define MOE_D3 4
#endif
#ifndef MOE_D4
#define MOE_D4 4
#endif
// Ring depth (tile steps in flight per strip) by bit width. Must divide every launch's ktiles (4, 16, 40 and the
// split-K quarters), so strip-group boundaries fall on the last ring slot and the epilogue is inlined once.
__host__ __device__ constexpr int depth_for(int bits) { return bits == 2 ? MOE_D2 : bits == 3 ? MOE_D3 : bits == 4 || bits == 5 ? MOE_D4 : DEPTH; }

// Per-lane load pointers of one strip, at the next tile step to load: this lane's code bytes (16 lane), the 3-bit
// high plane (8 lane), the scale pair (4 g) and the zero pair (32 + 2 g) of the tile's meta. Loads within a ring
// pass then use compile-time offsets (i steps), and the pointers advance once per pass.
template <int BITS>
struct LanePtr {
    const uint8_t *c, *h, *s, *z;
    __device__ __forceinline__ void at(const uint8_t* strip, size_t moff, int step, int lane) {
        const int g = lane >> 2;
        c = strip + size_t(step) * tile_wbytes(BITS) + 16 * lane;
        h = strip + size_t(step) * tile_wbytes(BITS) + 512 + 8 * lane;
        s = strip + moff + size_t(step) * META + 4 * g;
        z = strip + moff + size_t(step) * META + 32 + 2 * g;
    }
    __device__ __forceinline__ void advance(size_t dc, size_t dm) { c += dc; h += dc; s += dm; z += dm; }
    __device__ __forceinline__ void load(int i, Raw<BITS>& r) const {   // tile step +i
        constexpr int WB = tile_wbytes(BITS);
        r.lo = __ldg(reinterpret_cast<const uint4*>(c + i * WB));
        if (BITS == 4 || BITS == 5 || BITS == 8) r.hi4 = __ldg(reinterpret_cast<const uint4*>(c + i * WB + 512));
        if (BITS == 8) {
            r.w8b = __ldg(reinterpret_cast<const uint4*>(c + i * WB + 1024));
            r.w8c = __ldg(reinterpret_cast<const uint4*>(c + i * WB + 1536));
        }
        if (BITS == 3) r.hi3 = __ldg(reinterpret_cast<const uint2*>(h + i * WB));
        r.sz = __ldg(reinterpret_cast<const uint32_t*>(s + i * META));
        r.zz = __ldg(reinterpret_cast<const uint16_t*>(z + i * META));
    }
};

template <int BITS, int NT, int MS, int MODE>
__device__ __forceinline__ void gemm_body(const GemmArgs& g_, const Work& w, const uint8_t* smem, const uint8_t* mat,
                                          int ktiles, int ktot, int kb) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    constexpr int TB = tile_bytes(BITS), WB = tile_wbytes(BITS), NTOK = 8 * NT, D = depth_for(BITS);
    const uint4* Bs = reinterpret_cast<const uint4*>(smem);
    const float* Ss = reinterpret_cast<const float*>(smem + ktiles * NTOK * GROUP);
    const int* Qs = reinterpret_cast<const int*>(Ss + ktiles * NTOK);
    // The warp's strips: `groups` groups of MS strips, streamed as one ring of groups * ktiles tile steps, so the
    // pipeline does not drain between groups and the block's B gather is paid once. Strip m of group sg is
    // strip0 + sg * warps * MS + m. The ring is filled D steps ahead; ktiles is a multiple of D, so group ends fall
    // on pass ends, where the pointers jump to the next group.
    const int warps = blockDim.x >> 5, ng = g_.groups > 0 ? g_.groups : 1, nf = ng * ktiles;
    const int strip0 = ((g_.swap ? int(blockIdx.y) : int(blockIdx.x)) * ng * warps + warp) * MS;
    const size_t strip_bytes = size_t(ktot) * TB, moff = strip_meta_off(ktot, BITS);
    const size_t cjump = size_t(warps * MS) * strip_bytes - size_t(ktiles) * WB;
    const size_t mjump = size_t(warps * MS) * strip_bytes - size_t(ktiles) * META;
    LanePtr<BITS> p[MS], safe[MS];                // safe: the matrix's first tiles (the last pass reloads them)
#pragma unroll
    for (int m = 0; m < MS; ++m) {
        p[m].at(mat + size_t(strip0 + m) * strip_bytes, moff, kb, lane);
        safe[m].at(mat, moff, 0, lane);
    }
    Raw<BITS> buf[D][MS];
#pragma unroll
    for (int i = 0; i < D; ++i)                   // nf >= D always
#pragma unroll
        for (int m = 0; m < MS; ++m) p[m].load(i, buf[i][m]);
    int lkt = D;
#pragma unroll
    for (int m = 0; m < MS; ++m) {
        p[m].advance(size_t(D) * WB, size_t(D) * META);
        if (lkt == ktiles) p[m].advance(cjump, mjump);
    }
    if (lkt == ktiles) lkt = 0;
    cp_wait_all();
    __syncthreads();
    float c[MS][NT][4];
#pragma unroll
    for (int m = 0; m < MS; ++m)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) c[m][nt][0] = c[m][nt][1] = c[m][nt][2] = c[m][nt][3] = 0.f;
    int sg = 0, kt0 = 0;
    for (int f0 = 0; f0 < nf; f0 += D) {
        // unconditional reload (the last pass re-reads the matrix's first tiles): a predicated load would make the
        // compiler copy the ring registers instead of renaming them
        LanePtr<BITS> q[MS];
#pragma unroll
        for (int m = 0; m < MS; ++m) q[m] = f0 + D < nf ? p[m] : safe[m];
#pragma unroll
        for (int i = 0; i < D; ++i) {
            const int kt = kt0 + i;
            Raw<BITS> cur[MS];
#pragma unroll
            for (int m = 0; m < MS; ++m) {
                cur[m] = buf[i][m];
                q[m].load(i, buf[i][m]);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int n = nt * 8 + g;
                const uint4 b0 = Bs[((kt * 2 + 0) * NTOK + n) * 4 + t], b1 = Bs[((kt * 2 + 1) * NTOK + n) * 4 + t];
                const int m0 = nt * 8 + 2 * t;            // tokens m0, m0 + 1: adjacent ([kt][token] layout)
                const float2 xs = *reinterpret_cast<const float2*>(Ss + kt * NTOK + m0);
                const int2 xq = *reinterpret_cast<const int2*>(Qs + kt * NTOK + m0);
                const float xs0 = xs.x, xs1 = xs.y;
                const int xq0 = xq.x, xq1 = xq.y;
#pragma unroll
                for (int m = 0; m < MS; ++m) {
                    // one accumulator chain, started at (conversion bias - z * Sxq): the zero-point term and the
                    // int->float bias come out of the tensor core, nothing integer is left after the mma
                    constexpr bool HW = NT <= 2;
                    constexpr int BIAS = HW ? 0 : 0x4B400000;
                    const int z0 = int(cur[m].zz & 0xFF), z1 = int(cur[m].zz >> 8);
                    int pp[4] = {BIAS - z0 * xq0, BIAS - z0 * xq1, BIAS - z1 * xq0, BIAS - z1 * xq1};
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        uint32_t aa[4];
                        code_step<BITS>(cur[m], j, aa);
                        const uint4& bb = j < 2 ? b0 : b1;
                        mma_i8(pp, aa, word(bb, (j & 1) * 2), word(bb, (j & 1) * 2 + 1));
                    }
                    const float s0 = __half2float(__ushort_as_half(uint16_t(cur[m].sz & 0xFFFF)));
                    const float s1 = __half2float(__ushort_as_half(uint16_t(cur[m].sz >> 16)));
                    auto cvt = [](int x) { return HW ? __int2float_rn(x) : __int_as_float(x) - 12582912.0f; };
                    c[m][nt][0] += s0 * xs0 * cvt(pp[0]);
                    c[m][nt][1] += s0 * xs1 * cvt(pp[1]);
                    c[m][nt][2] += s1 * xs0 * cvt(pp[2]);
                    c[m][nt][3] += s1 * xs1 * cvt(pp[3]);
                }
            }
        }
        lkt += D;
#pragma unroll
        for (int m = 0; m < MS; ++m) {
            p[m].advance(size_t(D) * WB, size_t(D) * META);
            if (lkt == ktiles) p[m].advance(cjump, mjump);
        }
        if (lkt == ktiles) lkt = 0;
        kt0 += D;
        if (kt0 == ktiles) {                      // group done: epilogue, then the next group's accumulators
            epilogue<NT, MS, MODE>(g_, w, c, sg, strip0 + sg * warps * MS, t, g);
            kt0 = 0; ++sg;
        }
    }
}

// grid: x = row blocks of WARPS*MS*groups strips, y = work item, z = K splits (MODE 1/2 only; ktiles/z a multiple of DEPTH).
// Dynamic shared memory = smem_bytes(NT, K).
// MODE 0: gate_up (rows 2I, K = H) -> Z ; MODE 1: rows H, K = I -> Y weighted ; MODE 2: as 1, accumulated into Y ;
// MODE 3: as 1, fp16 into Z, unweighted (lm_head screen).
// Occupancy: NT <= 2 (the routed experts) is register-capped for 3 blocks/SM, which hides each block's B-gather
// prologue behind other blocks' streaming; NT = 4 would spill under that cap.
template <int WARPS, int NT, int MS, int MODE>
__global__ void __launch_bounds__(WARPS * 32, NT <= 2 ? MOE_MINB : 1) gemm(GemmArgs g_) {
    extern __shared__ __align__(16) uint8_t smem[];
    constexpr int NTOK = 8 * NT;
    const int item = g_.swap ? int(blockIdx.x) : int(blockIdx.y);
    if (g_.nwork && item >= *g_.nwork) return;
    const Work w = g_.work[item];
    const int K = MODE == 0 ? g_.H : g_.I, ktot = K / GROUP;
    const int kb = blockIdx.z * ktot / gridDim.z, ktiles = (blockIdx.z + 1) * ktot / gridDim.z - kb;
    uint8_t* Bs = smem;    // (MODE 1/2 read B rows by slot, MODE 0 by the slot's token)
    float* Ss = reinterpret_cast<float*>(smem + ktiles * NTOK * GROUP);
    int* Qs = reinterpret_cast<int*>(Ss + ktiles * NTOK);
    auto row_of = [&](int n) -> size_t {
        const int sl = w.slot0 + (n < w.ntok ? n : 0);
        return size_t(MODE == 0 ? g_.slot_token[sl] : sl);
    };
    for (int v = threadIdx.x; v < NTOK * ktiles * 8; v += WARPS * 32) {     // 8 x 16 B per token row and tile
        const int n = v / (ktiles * 8), r = v % (ktiles * 8), kt = r >> 3, cc = r & 7;   // cc = 2 t + q
        cp_async16(Bs + 16 * (((kt * 2 + (cc & 1)) * NTOK + n) * 4 + (cc >> 1)), g_.X.q + row_of(n) * K + (kb + kt) * GROUP + 16 * cc);
    }
    for (int v = threadIdx.x; v < NTOK * ktiles; v += WARPS * 32) {
        const int n = v / ktiles, kt = v % ktiles;
        const size_t src = row_of(n) * ktot + kb + kt;
        cp_async4(Ss + kt * NTOK + n, g_.X.s + src);
        cp_async4(Qs + kt * NTOK + n, g_.X.sum + src);
    }
    cp_commit();
    const int bits = g_.tab.bits[w.expert];
    const uint8_t* mat = g_.tab.base + (MODE == 0 ? g_.tab.gu_off[w.expert] : g_.tab.d_off[w.expert]);
    if (bits == 2) gemm_body<2, NT, MS, MODE>(g_, w, smem, mat, ktiles, ktot, kb);
    else if (bits == 3) gemm_body<3, NT, MS, MODE>(g_, w, smem, mat, ktiles, ktot, kb);
    else if (bits == 4) gemm_body<4, NT, MS, MODE>(g_, w, smem, mat, ktiles, ktot, kb);
    else if (NT == 4 && MODE == 3 && bits == 5) gemm_body<5, NT, MS, MODE>(g_, w, smem, mat, ktiles, ktot, kb);
    else gemm_body<8, NT, MS, MODE>(g_, w, smem, mat, ktiles, ktot, kb);
}

// Z' = A8(Z H512) per slot: fp16 logical-order Z in; int8 fragment-order Z', per-group scale and int sum out.
// One warp per slot, lane l owns values 16 l .. 16 l + 15 (4 butterfly stages in registers, 5 by shuffle); a
// 128-group is 8 lanes. grid ceil(nslots / 8), 256 threads.
__global__ void fwht512_q8(const half* Z, Act out8, int nslots) {
    const int s = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (s >= nslots) return;
    float x[16];
    {
        const uint4* z = reinterpret_cast<const uint4*>(Z + size_t(s) * 512 + lane * 16);
        const uint4 a = z[0], b = z[1];
        const half2* ha = reinterpret_cast<const half2*>(&a);
        const half2* hb = reinterpret_cast<const half2*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __half22float2(ha[j]), fb = __half22float2(hb[j]);
            x[2 * j] = fa.x; x[2 * j + 1] = fa.y; x[8 + 2 * j] = fb.x; x[8 + 2 * j + 1] = fb.y;
        }
    }
#pragma unroll
    for (int h = 1; h < 16; h <<= 1)
#pragma unroll
        for (int j = 0; j < 16; ++j)
            if (!(j & h)) { const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const bool hi = lane & o;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float y = __shfl_xor_sync(0xffffffffu, x[j], o);
            x[j] = hi ? y - x[j] : x[j] + y;
        }
    }
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) { x[j] *= 0.04419417382415922f; m = fmaxf(m, fabsf(x[j])); }   // 1/sqrt(512)
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float sc = fmaxf(m / 127.f, 1e-30f);
    int8_t* q = const_cast<int8_t*>(out8.q) + size_t(s) * 512;
    int sum = 0;
    const float inv = 1.f / sc;
#pragma unroll
    for (int a = 0; a < 4; ++a) {                 // values 4a .. 4a + 3 are consecutive in the fragment order
        uint32_t wd = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int qq = max(-127, min(127, __float2int_rn(x[4 * a + e] * inv)));
            wd |= uint32_t(qq & 0xFF) << (8 * e);
            sum += qq;
        }
        *reinterpret_cast<uint32_t*>(q + frag_pos_i8(lane * 16 + 4 * a)) = wd;
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((lane & 7) == 0) {
        const_cast<float*>(out8.s)[size_t(s) * 4 + (lane >> 3)] = sc;
        const_cast<int*>(out8.sum)[size_t(s) * 4 + (lane >> 3)] = sum;
    }
}

// acc[t, :] = sum_k Y[tok_slots[t, k], :] in a fixed order. One block per token.
__global__ void combine(const float* Y, const int* tok_slots, int K, int H, float* acc) {
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

}  // namespace moe8
