#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define GL_KTILE 64
#define GL_ROW_BYTES (GL_KTILE * 2)
#define GL_SEGS (GL_ROW_BYTES / 16)

#ifndef KT_TRANS
#define KT_TRANS 1
#endif

__device__ __forceinline__ void split_bf16(float x, __nv_bfloat16* hi, __nv_bfloat16* lo) {
    const __nv_bfloat16 h = __float2bfloat16_rn(x);
    *hi = h;
    *lo = __float2bfloat16_rn(x - __bfloat162float(h));
}

__device__ __forceinline__ void mma_bf16_hilo(float c[4], const uint32_t ah[4], const uint32_t al[4],
                                              const uint32_t bh[2], const uint32_t bl[2]) {
#define CLEF_MMA_HILO(A, B)                                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"                     \
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])                              \
                 : "r"((A)[0]), "r"((A)[1]), "r"((A)[2]), "r"((A)[3]), "r"((B)[0]), "r"((B)[1]))
    CLEF_MMA_HILO(ah, bh);
    CLEF_MMA_HILO(ah, bl);
    CLEF_MMA_HILO(al, bh);
#undef CLEF_MMA_HILO
}

__device__ __forceinline__ uint32_t ldm_swizzled(const uint8_t* base, int row, int seg) {
    return (uint32_t)__cvta_generic_to_shared(base + row * GL_ROW_BYTES + (seg ^ (row % 8)) * 16);
}

__device__ __forceinline__ void st_hi_lo(uint8_t* hi, uint8_t* lo, int row, int col, float v) {
    const int seg = col / 8, e = col - (col / 8) * 8;
    const int off = row * GL_ROW_BYTES + (seg ^ (row % 8)) * 16;
    split_bf16(v, (__nv_bfloat16*)(hi + off) + e, (__nv_bfloat16*)(lo + off) + e);
}

#define CHUNK 64
#define KD 128
#define VD 128
#define KS (KD + 1)
#define LS (CHUNK + 1)

#define LROW CHUNK
#define CHUNK_THREADS 256
#define WU_THREADS 128

#define RH_ROWS 8
#ifndef V_SPLIT

#define V_SPLIT 4
#endif
#define VS (VD / V_SPLIT)

#define WN_STRIDE (VS / GL_WARPS_N)
#define WS_NS (WN_STRIDE / GL_NT)

#define U_WM ((VS + 15) / 16)
#define U_WN (8 / U_WM)
#define U_NSTRIDE (KD / U_WN)
#define U_NS (U_NSTRIDE / GL_NT)
#define PT 32

#define RHS_COLS (KD + VD)

extern "C" __global__ void gdn_norm_qk(const float* __restrict__ q_in,
                                       const float* __restrict__ k_in,
                                       uint8_t* __restrict__ q_hi,
                                       uint8_t* __restrict__ q_lo,
                                       uint8_t* __restrict__ k_hi,
                                       uint8_t* __restrict__ k_lo,
                                       float* __restrict__ k_out,
                                       int rows, int cols, float q_scale,
                                       int heads, int nchunks) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= rows) return;

    const float* __restrict__ qr = q_in + (size_t)row * cols;
    const float* __restrict__ kr = k_in + (size_t)row * cols;
    float sq = 0.0f, sk = 0.0f;
    for (int p = lane; p < cols; p += 32) {
        sq += qr[p] * qr[p];
        sk += kr[p] * kr[p];
    }
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, s);
        sk += __shfl_xor_sync(0xffffffffu, sk, s);
    }
    const float rq = rsqrtf(sq + 1e-6f) * q_scale;
    const float rk = rsqrtf(sk + 1e-6f);

    const int t = row / heads, h = row - (row / heads) * heads;
    const int idx = h * nchunks + t / CHUNK;
    const int r = t - (t / CHUNK) * CHUNK;
    float* __restrict__ ko = k_out + (size_t)row * cols;
    for (int p = lane; p < cols; p += 32) {
        const int tile = p / GL_KTILE;
        const int c = p - tile * GL_KTILE;
        const int seg = c / 8, e = c - (c / 8) * 8;
        const size_t off = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                         + (size_t)r * GL_ROW_BYTES + (seg ^ (r % 8)) * 16 + e * 2;
        split_bf16(qr[p] * rq, (__nv_bfloat16*)(q_hi + off), (__nv_bfloat16*)(q_lo + off));
        split_bf16(kr[p] * rk, (__nv_bfloat16*)(k_hi + off), (__nv_bfloat16*)(k_lo + off));
        ko[p] = kr[p] * rk;
    }
}

#define GL_WARPS_M 4
#define GL_WARPS_N 2
#define GL_M_TILE 64
#define GL_NT 8
#define GL_NSUB 4
extern "C" __global__ void gdn_chunk_l(const uint8_t* __restrict__ k_hi,
                                       const uint8_t* __restrict__ k_lo,
                                       const float* __restrict__ g_ptr,
                                       const float* __restrict__ beta_ptr,
                                       float* __restrict__ l_out,
                                       float* __restrict__ gcum_out,
                                       int seq_len, int heads, int nchunks) {
    const int idx = blockIdx.x;
    const int h = idx / nchunks;
    const int c = idx - h * nchunks;
    const int t0 = c * CHUNK;
    const int n = min(CHUNK, seq_len - t0);

    __shared__ __align__(128) uint8_t kh[CHUNK * GL_ROW_BYTES];
    __shared__ __align__(128) uint8_t kl[CHUNK * GL_ROW_BYTES];
    __shared__ float lsh[CHUNK * LS];
    __shared__ float gcum[CHUNK];
    __shared__ float bsh[CHUNK];
    for (int i = threadIdx.x; i < CHUNK * CHUNK; i += CHUNK_THREADS)
        lsh[(i / CHUNK) * LS + (i % CHUNK)] = 0.0f;
    if (threadIdx.x < CHUNK) {
        const int r = threadIdx.x;
        gcum[r] = (r < n) ? g_ptr[(t0 + r) * heads + h] : 0.0f;
        bsh[r] = (r < n) ? beta_ptr[(t0 + r) * heads + h] : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float acc = 0.0f;
        for (int r = 0; r < CHUNK; ++r) {
            acc += gcum[r];
            gcum[r] = acc;
        }
    }

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % GL_WARPS_M;
    const int warp_n = warp / GL_WARPS_M;

    float d[GL_NSUB][4];
    #pragma unroll
    for (int ns = 0; ns < GL_NSUB; ++ns) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) d[ns][e] = 0.0f;
    }
    __syncthreads();
    for (int tile = 0; tile < KD / GL_KTILE; ++tile) {
        const int k0 = tile * GL_KTILE;
        for (int i = threadIdx.x; i < CHUNK * GL_SEGS; i += CHUNK_THREADS) {
            const int r = i / GL_SEGS, p = i - (i / GL_SEGS) * GL_SEGS;
            const size_t src = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                             + (size_t)r * GL_ROW_BYTES + p * 16;
            const int dst = r * GL_ROW_BYTES + p * 16;

            const unsigned mask = (r < n) ? 0xffffffffu : 0u;
            uint4 vh = *reinterpret_cast<const uint4*>(k_hi + src);
            uint4 vl = *reinterpret_cast<const uint4*>(k_lo + src);
            vh.x &= mask; vh.y &= mask; vh.z &= mask; vh.w &= mask;
            vl.x &= mask; vl.y &= mask; vl.z &= mask; vl.w &= mask;
            *reinterpret_cast<uint4*>(kh + dst) = vh;
            *reinterpret_cast<uint4*>(kl + dst) = vl;
        }
        __syncthreads();

        #pragma unroll
        for (int ksub = 0; ksub < GL_KTILE / 16; ++ksub) {
            uint32_t ah[4], al[4];
            {
                const int a_row = warp_m * 16 + (lane % 16);
                const uint32_t addr = ldm_swizzled(kh, a_row, ksub * 2 + (lane / 16));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3])
                             : "r"(addr));
            }
            {
                const int a_row = warp_m * 16 + (lane % 16);
                const uint32_t addr = ldm_swizzled(kl, a_row, ksub * 2 + (lane / 16));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3])
                             : "r"(addr));
            }
            #pragma unroll
            for (int ns = 0; ns < GL_NSUB; ++ns) {
                uint32_t bh[2], bl[2];
                const int b_row = warp_n * (GL_NSUB * GL_NT) + ns * GL_NT + (lane % 8);
                const int b_seg = ksub * 2 + ((lane / 8) & 1);
                {
                    const uint32_t addr = ldm_swizzled(kh, b_row, b_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                }
                {
                    const uint32_t addr = ldm_swizzled(kl, b_row, b_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                }
                mma_bf16_hilo(d[ns], ah, al, bh, bl);
            }
        }
        __syncthreads();
    }

    const int r0 = warp_m * 16 + lane / 4;
    const int r1 = r0 + 8;
    const float b0 = bsh[r0];
    const float b1 = bsh[r1];
    const float g0 = gcum[r0];
    const float g1 = gcum[r1];
    #pragma unroll
    for (int ns = 0; ns < GL_NSUB; ++ns) {
        const int c0 = warp_n * (GL_NSUB * GL_NT) + ns * GL_NT + (lane % 4) * 2;
        const int c1 = c0 + 1;
        if (c0 < r0) lsh[r0 * LS + c0] = b0 * __expf(g0 - gcum[c0]) * d[ns][0];
        if (c1 < r0) lsh[r0 * LS + c1] = b0 * __expf(g0 - gcum[c1]) * d[ns][1];
        if (c0 < r1) lsh[r1 * LS + c0] = b1 * __expf(g1 - gcum[c0]) * d[ns][2];
        if (c1 < r1) lsh[r1 * LS + c1] = b1 * __expf(g1 - gcum[c1]) * d[ns][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < CHUNK * CHUNK; i += CHUNK_THREADS)
        l_out[(size_t)idx * CHUNK * CHUNK + i] = lsh[(i / CHUNK) * LS + (i % CHUNK)];
    if (threadIdx.x < CHUNK) gcum_out[(size_t)idx * CHUNK + threadIdx.x] = gcum[threadIdx.x];
}

extern "C" __global__ void gdn_chunk_wu(const float* __restrict__ k_ptr,
                                        const float* __restrict__ v_ptr,
                                        const float* __restrict__ beta_ptr,
                                        const float* __restrict__ l_in,
                                        const float* __restrict__ gcum_in,
                                        uint8_t* __restrict__ w_hi,
                                        uint8_t* __restrict__ w_lo,
                                        float* __restrict__ u_out,
                                        int seq_len, int heads, int nchunks) {
    const int idx = blockIdx.x;
    const bool is_w = (blockIdx.y == 0);
    const int h = idx / nchunks;
    const int c = idx - h * nchunks;
    const int t0 = c * CHUNK;
    const int n = min(CHUNK, seq_len - t0);

    __shared__ float lsh[CHUNK * LROW];
    __shared__ float rhs_s[RH_ROWS * KD];
    __shared__ float bsh[CHUNK];
    for (int i = threadIdx.x; i < CHUNK * CHUNK; i += WU_THREADS)
        lsh[(i / CHUNK) * LROW + (i % CHUNK)] = l_in[(size_t)idx * CHUNK * CHUNK + i];
    __syncthreads();

    __shared__ float asc[CHUNK];
    if (threadIdx.x < CHUNK) {
        const int r = threadIdx.x;
        asc[r] = (r < n) ? __expf(gcum_in[(size_t)idx * CHUNK + r]) : 0.0f;
        bsh[r] = (r < n) ? beta_ptr[(size_t)(t0 + r) * heads + h] : 0.0f;
    }
    __syncthreads();

    const int col = threadIdx.x;
    const float* __restrict__ src = is_w ? k_ptr : v_ptr;
    const int src_dim = is_w ? KD : VD;

    float y[CHUNK];
    #pragma unroll
    for (int slice = 0; slice < CHUNK / RH_ROWS; ++slice) {
        for (int i = threadIdx.x; i < RH_ROWS * KD; i += WU_THREADS) {
            const int r = slice * RH_ROWS + i / KD, cc = i % KD;

            const int rr = (r < n) ? r : (n - 1);

            const float b = bsh[rr];
            const float a = is_w ? asc[rr] : 1.0f;
            const float v = b * a * src[((size_t)(t0 + rr) * heads + h) * src_dim + cc];
            rhs_s[i] = (r < n) ? v : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int ii = 0; ii < RH_ROWS; ++ii) {
            const int i = slice * RH_ROWS + ii;
            const float* __restrict__ lr = &lsh[i * LROW];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            #pragma unroll
            for (int j = 0; j < CHUNK; j += 4) {
                const float4 l4 = *reinterpret_cast<const float4*>(&lr[j]);
                if (j + 0 < i) a0 -= l4.x * y[j + 0];
                if (j + 1 < i) a1 -= l4.y * y[j + 1];
                if (j + 2 < i) a2 -= l4.z * y[j + 2];
                if (j + 3 < i) a3 -= l4.w * y[j + 3];
            }
            y[i] = rhs_s[ii * KD + col] + ((a0 + a1) + (a2 + a3));
        }
        __syncthreads();
    }
    if (is_w) {
        const size_t wbase = (size_t)idx * (KD / GL_KTILE) * CHUNK * GL_ROW_BYTES;
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i) {
            const int tile = col / GL_KTILE;
            st_hi_lo(w_hi + wbase + tile * CHUNK * GL_ROW_BYTES,
                     w_lo + wbase + tile * CHUNK * GL_ROW_BYTES,
                     i, col - tile * GL_KTILE, y[i]);
        }
    } else {
        float* __restrict__ dst = u_out;
        #pragma unroll
        for (int i = 0; i < CHUNK; ++i)
            dst[((size_t)idx * CHUNK + i) * VD + col] = y[i];
    }
}

#define S_WARPS_M 4

#if VS >= 32
#define S_WARPS_N 4
#else
#define S_WARPS_N 2
#endif
#define S_WARPS (S_WARPS_M * S_WARPS_N)
#define STATE_THREADS (S_WARPS * 32)

#define WQ_SEGS_PER_THREAD ((CHUNK * GL_SEGS) / STATE_THREADS)
static_assert((CHUNK * GL_SEGS) % STATE_THREADS == 0, "the staging must divide evenly over the block");
#define S_WN_STRIDE (VS / S_WARPS_N)
#define S_WS_NS (S_WN_STRIDE / GL_NT)
#define S_U_WM ((VS + 15) / 16)

#define S_U_WN (S_WARPS / S_U_WM)
#define S_U_NSTRIDE (KD / S_U_WN)
#define S_U_NS (S_U_NSTRIDE / GL_NT)
static_assert(S_U_NSTRIDE * S_U_WN == KD, "the state update must cover all of KD");
static_assert(S_U_NSTRIDE % GL_NT == 0, "and in whole m16n8 tiles");
static_assert(S_WS_NS > 0, "the W@S accumulator would be zero-sized");

__device__ __forceinline__ void gdn_prefetch_wq(const uint8_t* __restrict__ q_hi,
                                                const uint8_t* __restrict__ q_lo,
                                                const uint8_t* __restrict__ w_hi,
                                                const uint8_t* __restrict__ w_lo,
                                                int idx, int n, int tile, int slot,
                                                uint4* rw, uint4* rq) {
    const int i = threadIdx.x + slot * STATE_THREADS;
    const int r = i / GL_SEGS, p = i - (i / GL_SEGS) * GL_SEGS;
    const size_t src = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                     + (size_t)r * GL_ROW_BYTES + p * 16;
    rw[0] = *reinterpret_cast<const uint4*>(w_hi + src);
    rw[1] = *reinterpret_cast<const uint4*>(w_lo + src);
    const unsigned mask = (r < n) ? 0xffffffffu : 0u;
    uint4 vq = *reinterpret_cast<const uint4*>(q_hi + src);
    uint4 wq = *reinterpret_cast<const uint4*>(q_lo + src);
    vq.x &= mask; vq.y &= mask; vq.z &= mask; vq.w &= mask;
    wq.x &= mask; wq.y &= mask; wq.z &= mask; wq.w &= mask;
    rq[0] = vq;
    rq[1] = wq;
}

__device__ __forceinline__ void gdn_commit_wq(uint8_t* ab_hi, uint8_t* ab_lo, int slot,
                                              const uint4* rw, const uint4* rq) {
    constexpr int QOFF = CHUNK * GL_ROW_BYTES;
    const int i = threadIdx.x + slot * STATE_THREADS;
    const int r = i / GL_SEGS, p = i - (i / GL_SEGS) * GL_SEGS;
    const int dst = r * GL_ROW_BYTES + p * 16;
    *reinterpret_cast<uint4*>(ab_hi + dst) = rw[0];
    *reinterpret_cast<uint4*>(ab_lo + dst) = rw[1];
    *reinterpret_cast<uint4*>(ab_hi + QOFF + dst) = rq[0];
    *reinterpret_cast<uint4*>(ab_lo + QOFF + dst) = rq[1];
}

extern "C" __global__ void gdn_state(const uint8_t* __restrict__ q_hi,
                                     const uint8_t* __restrict__ q_lo,
                                     const uint8_t* __restrict__ k_hi,
                                     const uint8_t* __restrict__ k_lo,
                                     const uint8_t* __restrict__ w_hi,
                                     const uint8_t* __restrict__ w_lo,
                                     const float* __restrict__ u_ptr,
                                     const float* __restrict__ gcum_ptr,
                                     float* __restrict__ delta_out,
                                     float* __restrict__ o_ptr,
                                     int seq_len, int heads, int nchunks,
                                     int segs,

                                     const int* __restrict__ chunk_start,

                                    const float* state_in,
                                    float* state_out,

                                    const int* __restrict__ state_slot,

                                    const int* __restrict__ walk_ptr,

                                    int* dbg) {
    const int h = blockIdx.x;
    const int vs = blockIdx.y;
    const int seg = blockIdx.z;
    const int j0 = vs * VS;

    #ifndef KD_STRIDE

    #define KD_STRIDE (KD + 3)
    #endif
    __shared__ float s_s[VS * KD_STRIDE];

    __shared__ __align__(128) uint8_t ab_hi[2 * CHUNK * GL_ROW_BYTES];
    __shared__ __align__(128) uint8_t ab_lo[2 * CHUNK * GL_ROW_BYTES];
    __shared__ __align__(128) uint8_t st_hi[VS * GL_ROW_BYTES];
    __shared__ __align__(128) uint8_t st_lo[VS * GL_ROW_BYTES];
    __shared__ float d_s[CHUNK * VS];
    __shared__ float gcum[CHUNK];
    __shared__ float egd[CHUNK];

    constexpr int QOFF = CHUNK * GL_ROW_BYTES;

    const int sblk = blockIdx.x * V_SPLIT + blockIdx.y;

    const size_t slot_stride = (size_t)heads * V_SPLIT * VS * KD;
    const int slot0 = state_slot != 0 ? state_slot[0] : 0;
    if (state_in != 0) {
        for (int i = threadIdx.x; i < VS * KD_STRIDE; i += STATE_THREADS) {
            const int j = i / KD_STRIDE, p = i - j * KD_STRIDE;

            s_s[i] = (p < KD)
                ? state_in[(size_t)slot0 * slot_stride + ((size_t)sblk * VS + j) * KD + p]
                : 0.0f;
        }
    } else {
        for (int i = threadIdx.x; i < VS * KD_STRIDE; i += STATE_THREADS) s_s[i] = 0.0f;
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % S_WARPS_M;
    const int warp_n = warp / S_WARPS_M;

    const int um = warp % S_U_WM;
    const int un = warp / S_U_WM;

    const int walk = walk_ptr ? min(*walk_ptr, seq_len) : seq_len;
    const int walk_chunks = min(nchunks, (walk + CHUNK - 1) / CHUNK);
    const int c_begin = (walk_chunks * seg) / segs;
    const int c_end = (walk_chunks * (seg + 1)) / segs;

    uint4 rw[KD / GL_KTILE][WQ_SEGS_PER_THREAD][2], rq[KD / GL_KTILE][WQ_SEGS_PER_THREAD][2];
    {
        const int n0 = min(CHUNK, seq_len - c_begin * CHUNK);
        const int idx0 = h * nchunks + c_begin;
        #pragma unroll
        for (int t = 0; t < KD / GL_KTILE; ++t)
            for (int sl = 0; sl < WQ_SEGS_PER_THREAD; ++sl)
                gdn_prefetch_wq(q_hi, q_lo, w_hi, w_lo, idx0, n0, t, sl, rw[t][sl], rq[t][sl]);
    }
    for (int c = c_begin; c < c_end; ++c) {
        const int idx = h * nchunks + c;
        const int t0 = c * CHUNK;
        const int n = min(CHUNK, seq_len - t0);
        if (chunk_start != 0 && chunk_start[c] != 0) {
            const int prev = state_slot != 0 ? state_slot[c - 1] : 0;
            const int next = state_slot != 0 ? state_slot[c] : 0;
            if (state_out != 0) {
                for (int i = threadIdx.x; i < VS * KD; i += STATE_THREADS) {
                    const int j = i / KD, p = i - j * KD;
                    state_out[(size_t)prev * slot_stride + ((size_t)sblk * VS + j) * KD + p] =
                        s_s[j * KD_STRIDE + p];
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < VS * KD_STRIDE; i += STATE_THREADS) {
                const int j = i / KD_STRIDE, p = i - j * KD_STRIDE;
                s_s[i] = (state_in != 0 && p < KD)
                    ? state_in[(size_t)next * slot_stride + ((size_t)sblk * VS + j) * KD + p]
                    : 0.0f;
            }
            __syncthreads();
        }
        if (threadIdx.x < CHUNK) gcum[threadIdx.x] = gcum_ptr[(size_t)idx * CHUNK + threadIdx.x];
        __syncthreads();

        uint8_t* abh = ab_hi;
        uint8_t* abl = ab_lo;
        const float gtot = gcum[CHUNK - 1];
        if (threadIdx.x < CHUNK) egd[threadIdx.x] = __expf(gtot - gcum[threadIdx.x]);

        float dWS[S_WS_NS][4], dQS[S_WS_NS][4];
        #pragma unroll
        for (int ns = 0; ns < S_WS_NS; ++ns)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                dWS[ns][e] = 0.0f;
                dQS[ns][e] = 0.0f;
            }
        __syncthreads();
        for (int tile = 0; tile < KD / GL_KTILE; ++tile) {
            const int k0 = tile * GL_KTILE;

            #pragma unroll
            for (int sl = 0; sl < WQ_SEGS_PER_THREAD; ++sl)
                gdn_commit_wq(abh, abl, sl, rw[tile][sl], rq[tile][sl]);
            if (c + 1 < c_end) {
                const int n1 = min(CHUNK, seq_len - (c + 1) * CHUNK);
                #pragma unroll
                for (int sl = 0; sl < WQ_SEGS_PER_THREAD; ++sl)
                    gdn_prefetch_wq(q_hi, q_lo, w_hi, w_lo, h * nchunks + (c + 1), n1, tile, sl,
                                    rw[tile][sl], rq[tile][sl]);
            }
            for (int i = threadIdx.x; i < VS * GL_SEGS; i += STATE_THREADS) {
                const int r = i / GL_SEGS, seg = i - (i / GL_SEGS) * GL_SEGS;
                const int off = r * GL_ROW_BYTES + (seg ^ (r % 8)) * 16;
                __nv_bfloat16* sh = (__nv_bfloat16*)(st_hi + off);
                __nv_bfloat16* sl = (__nv_bfloat16*)(st_lo + off);
                const float* __restrict__ srow = s_s + r * KD_STRIDE + k0 + seg * 8;
                #pragma unroll
                for (int e = 0; e < 8; ++e) split_bf16(srow[e], &sh[e], &sl[e]);
            }
            __syncthreads();

            #pragma unroll
            for (int ksub = 0; ksub < GL_KTILE / 16; ++ksub) {
                uint32_t ah[4], al[4];
                const int a_row = warp_m * 16 + (lane % 16);
                const int a_seg = ksub * 2 + (lane / 16);
                {
                    const uint32_t addr = ldm_swizzled(abh, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3]) : "r"(addr));
                }
                {
                    const uint32_t addr = ldm_swizzled(abl, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3]) : "r"(addr));
                }
                #pragma unroll
                for (int ns = 0; ns < S_WS_NS; ++ns) {
                    uint32_t bh[2], bl[2];
                    const int b_row = warp_n * S_WN_STRIDE + ns * GL_NT + (lane % 8);
                    const int b_seg = ksub * 2 + ((lane / 8) & 1);
                    {
                        const uint32_t addr = ldm_swizzled(st_hi, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                    }
                    {
                        const uint32_t addr = ldm_swizzled(st_lo, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                    }
                    mma_bf16_hilo(dWS[ns], ah, al, bh, bl);
                }

                {
                    const uint32_t addr = ldm_swizzled(abh + QOFF, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3]) : "r"(addr));
                }
                {
                    const uint32_t addr = ldm_swizzled(abl + QOFF, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3]) : "r"(addr));
                }
                #pragma unroll
                for (int ns = 0; ns < S_WS_NS; ++ns) {
                    uint32_t bh[2], bl[2];
                    const int b_row = warp_n * S_WN_STRIDE + ns * GL_NT + (lane % 8);
                    const int b_seg = ksub * 2 + ((lane / 8) & 1);
                    {
                        const uint32_t addr = ldm_swizzled(st_hi, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                    }
                    {
                        const uint32_t addr = ldm_swizzled(st_lo, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                    }
                    mma_bf16_hilo(dQS[ns], ah, al, bh, bl);
                }
            }
            __syncthreads();
        }

        const int r0 = warp_m * 16 + lane / 4;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ns = 0; ns < S_WS_NS; ++ns) {
            const int c0 = warp_n * S_WN_STRIDE + ns * GL_NT + (lane % 4) * 2;
            const int c1 = c0 + 1;
            const int i0 = r0, i1 = r1;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = (e < 2) ? i0 : i1;
                const int jj = (e % 2 == 0) ? c0 : c1;
                const float ws = (e == 0) ? dWS[ns][0] : (e == 1) ? dWS[ns][1]
                               : (e == 2) ? dWS[ns][2] : dWS[ns][3];
                const float qs = (e == 0) ? dQS[ns][0] : (e == 1) ? dQS[ns][1]
                               : (e == 2) ? dQS[ns][2] : dQS[ns][3];

                const float u0 = u_ptr[(size_t)idx * CHUNK * VD + (size_t)i * VD + (j0 + jj)];
                const float d = u0 - ws;

                d_s[i * VS + jj] = d * egd[i];
                delta_out[(size_t)idx * CHUNK * VD + (size_t)i * VD + (j0 + jj)] = d;
                if (i < n) {
                    o_ptr[((size_t)(t0 + i) * heads + h) * VD + (j0 + jj)] = __expf(gcum[i]) * qs;
                }
            }
        }
        __syncthreads();

        {
            for (int i = threadIdx.x; i < VS * CHUNK; i += STATE_THREADS) {
                const int ii = i / VS, jj = i - (i / VS) * VS;
                const int seg = ii / 8, e = ii - (ii / 8) * 8;
                const int off = jj * GL_ROW_BYTES + (seg ^ (jj % 8)) * 16;
                split_bf16(d_s[ii * VS + jj], (__nv_bfloat16*)(st_hi + off) + e,
                                              (__nv_bfloat16*)(st_lo + off) + e);
            }
            for (int i = threadIdx.x; i < (KD / GL_KTILE) * (CHUNK * GL_SEGS); i += STATE_THREADS) {
                const int per_tile = CHUNK * GL_SEGS;
                const int tile = i / per_tile;
                const int rem = i - tile * per_tile;
                const int r = rem / GL_SEGS, s = rem - r * GL_SEGS;
                const size_t src = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                                 + (size_t)r * GL_ROW_BYTES + s * 16;
#if KT_TRANS

                const int dst = tile * (CHUNK * GL_ROW_BYTES) + r * GL_ROW_BYTES + s * 16;
                const unsigned mask = (r < n) ? 0xffffffffu : 0u;
                const uint4 vh = *reinterpret_cast<const uint4*>(k_hi + src);
                const uint4 vl = *reinterpret_cast<const uint4*>(k_lo + src);
                *reinterpret_cast<uint4*>(abh + dst) =
                    make_uint4(vh.x & mask, vh.y & mask, vh.z & mask, vh.w & mask);
                *reinterpret_cast<uint4*>(abl + dst) =
                    make_uint4(vl.x & mask, vl.y & mask, vl.z & mask, vl.w & mask);
#else
                const unsigned mask = (r < n) ? 0xffffffffu : 0u;
                const uint4 vh = *reinterpret_cast<const uint4*>(k_hi + src);
                const uint4 vl = *reinterpret_cast<const uint4*>(k_lo + src);
                const uint32_t hw[4] = {vh.x & mask, vh.y & mask, vh.z & mask, vh.w & mask};
                const uint32_t lw[4] = {vl.x & mask, vl.y & mask, vl.z & mask, vl.w & mask};

                const int c0 = (s ^ (r % 8)) * 8;
                const int seg = r / 8, e = r - (r / 8) * 8;
                #pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const int p = tile * GL_KTILE + c0 + q;
                    const int off = p * GL_ROW_BYTES + (seg ^ (p % 8)) * 16 + e * 2;
                    *reinterpret_cast<uint16_t*>(abh + off) =
                        (uint16_t)(hw[q >> 1] >> ((q & 1) * 16));
                    *reinterpret_cast<uint16_t*>(abl + off) =
                        (uint16_t)(lw[q >> 1] >> ((q & 1) * 16));
                }
#endif
            }
            __syncthreads();

            float dS[S_U_NS][4];
            #pragma unroll
            for (int ns = 0; ns < S_U_NS; ++ns)
                #pragma unroll
                for (int e = 0; e < 4; ++e) dS[ns][e] = 0.0f;

            #pragma unroll
            for (int ksub = 0; ksub < CHUNK / 16; ++ksub) {
                uint32_t ah[4], al[4];
                const int a_row = um * 16 + (lane % 16);
                const int a_seg = ksub * 2 + (lane / 16);
                {
                    const uint32_t addr = ldm_swizzled(st_hi, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3]) : "r"(addr));
                }
                {
                    const uint32_t addr = ldm_swizzled(st_lo, a_row, a_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3]) : "r"(addr));
                }
                #pragma unroll
                for (int ns = 0; ns < S_U_NS; ++ns) {
                    uint32_t bh[2], bl[2];
#if KT_TRANS

                    const int p_blk = un * S_U_NSTRIDE + ns * GL_NT;
                    const int ktile = p_blk / GL_KTILE;
                    const int seg = (p_blk % GL_KTILE) / GL_SEGS;
                    const int irow = ksub * 16 + (lane & 15);
                    const int off = ktile * (CHUNK * GL_ROW_BYTES) + irow * GL_ROW_BYTES
                                  + ((seg ^ (irow % GL_SEGS)) * 16);
                    {
                        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(abh + off);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                    }
                    {
                        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(abl + off);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                    }
#else
                    const int b_row = un * S_U_NSTRIDE + ns * GL_NT + (lane % 8);
                    const int b_seg = ksub * 2 + ((lane / 8) & 1);
                    {
                        const uint32_t addr = ldm_swizzled(abh, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                    }
                    {
                        const uint32_t addr = ldm_swizzled(abl, b_row, b_seg);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                     : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                    }
#endif
                    mma_bf16_hilo(dS[ns], ah, al, bh, bl);
                }
            }

            const float sdecay = __expf(gtot);
            const int m0 = um * 16 + lane / 4;
            const int m1 = m0 + 8;
            #pragma unroll
            for (int ns = 0; ns < S_U_NS; ++ns) {
                const int n0 = un * S_U_NSTRIDE + ns * GL_NT + (lane % 4) * 2;
                const int n1 = n0 + 1;
                s_s[m0 * KD_STRIDE + n0] = sdecay * s_s[m0 * KD_STRIDE + n0] + dS[ns][0];
                s_s[m0 * KD_STRIDE + n1] = sdecay * s_s[m0 * KD_STRIDE + n1] + dS[ns][1];
                s_s[m1 * KD_STRIDE + n0] = sdecay * s_s[m1 * KD_STRIDE + n0] + dS[ns][2];
                s_s[m1 * KD_STRIDE + n1] = sdecay * s_s[m1 * KD_STRIDE + n1] + dS[ns][3];
            }
            __syncthreads();
        }
    }
    if (dbg != 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        dbg[0] = state_slot != 0 ? state_slot[0] : -1;
        dbg[1] = state_slot != 0 ? state_slot[nchunks - 1] : -1;
        dbg[2] = (int)slot_stride;
        dbg[3] = state_out != 0 ? 1 : 0;
        dbg[4] = nchunks;
    }
    if (state_out != 0) {
        for (int i = threadIdx.x; i < VS * KD; i += STATE_THREADS) {
            const int j = i / KD, p = i - j * KD;
            const int last = state_slot != 0 ? state_slot[nchunks - 1] : 0;
            state_out[(size_t)last * slot_stride + ((size_t)sblk * VS + j) * KD + p] =
                s_s[j * KD_STRIDE + p];
        }
    }
}

extern "C" __global__ void gdn_chunk_out(const uint8_t* __restrict__ q_hi,
                                         const uint8_t* __restrict__ q_lo,
                                         const uint8_t* __restrict__ k_hi,
                                         const uint8_t* __restrict__ k_lo,
                                         const float* __restrict__ delta_ptr,
                                         const float* __restrict__ gcum_ptr,
                                         float* __restrict__ o_ptr,
                                         int seq_len, int heads, int nchunks,

                                         const int* __restrict__ walk_ptr) {
    const int idx = blockIdx.x;
    const int h = idx / nchunks;
    const int c = idx - h * nchunks;
    const int t0 = c * CHUNK;
    const int n = min(CHUNK, seq_len - t0);

    __shared__ __align__(128) uint8_t ab[2 * 2 * CHUNK * GL_ROW_BYTES];
    uint8_t* const ab_hi = ab;
    uint8_t* const ab_lo = ab + 2 * CHUNK * GL_ROW_BYTES;
    __shared__ __align__(128) uint8_t qk_hi[CHUNK * GL_ROW_BYTES];
    __shared__ __align__(128) uint8_t qk_lo[CHUNK * GL_ROW_BYTES];
    __shared__ float gcum[CHUNK];

    constexpr int QOFF = CHUNK * GL_ROW_BYTES;
    if (threadIdx.x < CHUNK) gcum[threadIdx.x] = gcum_ptr[(size_t)idx * CHUNK + threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % GL_WARPS_M;
    const int warp_n = warp / GL_WARPS_M;

    float d[GL_NSUB][4];
    #pragma unroll
    for (int ns = 0; ns < GL_NSUB; ++ns)
        #pragma unroll
        for (int e = 0; e < 4; ++e) d[ns][e] = 0.0f;
    for (int tile = 0; tile < KD / GL_KTILE; ++tile) {
        const int k0 = tile * GL_KTILE;
        for (int i = threadIdx.x; i < CHUNK * GL_SEGS; i += CHUNK_THREADS) {
            const int r = i / GL_SEGS, p = i - (i / GL_SEGS) * GL_SEGS;
            const size_t src = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                             + (size_t)r * GL_ROW_BYTES + p * 16;
            const int qdst = r * GL_ROW_BYTES + p * 16;
            const int kdst = QOFF + r * GL_ROW_BYTES + p * 16;

            const unsigned mask = (r < n) ? 0xffffffffu : 0u;
            uint4 vq = *reinterpret_cast<const uint4*>(q_hi + src);
            uint4 wq = *reinterpret_cast<const uint4*>(q_lo + src);
            uint4 vk = *reinterpret_cast<const uint4*>(k_hi + src);
            uint4 wk = *reinterpret_cast<const uint4*>(k_lo + src);
            vq.x &= mask; vq.y &= mask; vq.z &= mask; vq.w &= mask;
            wq.x &= mask; wq.y &= mask; wq.z &= mask; wq.w &= mask;
            vk.x &= mask; vk.y &= mask; vk.z &= mask; vk.w &= mask;
            wk.x &= mask; wk.y &= mask; wk.z &= mask; wk.w &= mask;
            *reinterpret_cast<uint4*>(ab_hi + qdst) = vq;
            *reinterpret_cast<uint4*>(ab_lo + qdst) = wq;
            *reinterpret_cast<uint4*>(ab_hi + kdst) = vk;
            *reinterpret_cast<uint4*>(ab_lo + kdst) = wk;
        }
        __syncthreads();

        #pragma unroll
        for (int ksub = 0; ksub < GL_KTILE / 16; ++ksub) {
            uint32_t ah[4], al[4];
            const int a_row = warp_m * 16 + (lane % 16);
            const int a_seg = ksub * 2 + (lane / 16);
            {
                const uint32_t addr = ldm_swizzled(ab_hi, a_row, a_seg);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3]) : "r"(addr));
            }
            {
                const uint32_t addr = ldm_swizzled(ab_lo, a_row, a_seg);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3]) : "r"(addr));
            }
            #pragma unroll
            for (int ns = 0; ns < GL_NSUB; ++ns) {
                uint32_t bh[2], bl[2];
                const int b_row = warp_n * (GL_NSUB * GL_NT) + ns * GL_NT + (lane % 8);
                const int b_seg = ksub * 2 + ((lane / 8) & 1);
                {
                    const uint32_t addr = ldm_swizzled(ab_hi + QOFF, b_row, b_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
                }
                {
                    const uint32_t addr = ldm_swizzled(ab_lo + QOFF, b_row, b_seg);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
                }
                mma_bf16_hilo(d[ns], ah, al, bh, bl);
            }
        }
        __syncthreads();
    }

    const int r0 = warp_m * 16 + lane / 4;
    const int r1 = r0 + 8;
    const float g0 = gcum[r0];
    const float g1 = gcum[r1];
    #pragma unroll
    for (int ns = 0; ns < GL_NSUB; ++ns) {
        const int c0 = warp_n * (GL_NSUB * GL_NT) + ns * GL_NT + (lane % 4) * 2;
        const int c1 = c0 + 1;

        const float e00 = __expf(g0 - gcum[c0]);
        const float e01 = __expf(g0 - gcum[c1]);
        const float e10 = __expf(g1 - gcum[c0]);
        const float e11 = __expf(g1 - gcum[c1]);
        st_hi_lo(qk_hi, qk_lo, r0, c0, (c0 <= r0) ? d[ns][0] * e00 : 0.0f);
        st_hi_lo(qk_hi, qk_lo, r0, c1, (c1 <= r0) ? d[ns][1] * e01 : 0.0f);
        st_hi_lo(qk_hi, qk_lo, r1, c0, (c0 <= r1) ? d[ns][2] * e10 : 0.0f);
        st_hi_lo(qk_hi, qk_lo, r1, c1, (c1 <= r1) ? d[ns][3] * e11 : 0.0f);
    }
    __syncthreads();

    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;

#if !KT_TRANS
    for (int i = threadIdx.x; i < CHUNK * VD; i += CHUNK_THREADS) {
        const int jp = i / VD, col = i - (i / VD) * VD;
        st_hi_lo(ab_hi, ab_lo, col, jp,
                 delta_ptr[(size_t)idx * CHUNK * VD + (size_t)jp * VD + col]);
    }
#else
    for (int i = threadIdx.x; i < CHUNK * (VD / 8); i += CHUNK_THREADS) {
        const int jp = i / (VD / 8), g = i - jp * (VD / 8);
        const int col = g * 8;
        const float4* src = reinterpret_cast<const float4*>(
            delta_ptr + (size_t)idx * CHUNK * VD + (size_t)jp * VD + col);
        const float4 a = src[0], b = src[1];
        const int tile = col / GL_KTILE;
        const int seg = (col - tile * GL_KTILE) / GL_SEGS;
        const int off = tile * (CHUNK * GL_ROW_BYTES) + jp * GL_ROW_BYTES
                      + ((seg ^ (jp % GL_SEGS)) * 16);
        __nv_bfloat16* dh = reinterpret_cast<__nv_bfloat16*>(ab_hi + off);
        __nv_bfloat16* dl = reinterpret_cast<__nv_bfloat16*>(ab_lo + off);
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        #pragma unroll
        for (int q = 0; q < 8; ++q) split_bf16(v[q], &dh[q], &dl[q]);
    }
#endif
    __syncthreads();

    const int sm = warp % 4;
    const int sn = warp / 4;
    float od[8][4];
    #pragma unroll
    for (int ns = 0; ns < 8; ++ns)
        #pragma unroll
        for (int e = 0; e < 4; ++e) od[ns][e] = 0.0f;

    #pragma unroll
    for (int ksub = 0; ksub < CHUNK / 16; ++ksub) {
        uint32_t ah[4], al[4];
        const int a_row = sm * 16 + (lane % 16);
        const int a_seg = ksub * 2 + (lane / 16);
        {
            const uint32_t addr = ldm_swizzled(qk_hi, a_row, a_seg);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(ah[0]), "=r"(ah[1]), "=r"(ah[2]), "=r"(ah[3]) : "r"(addr));
        }
        {
            const uint32_t addr = ldm_swizzled(qk_lo, a_row, a_seg);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                         : "=r"(al[0]), "=r"(al[1]), "=r"(al[2]), "=r"(al[3]) : "r"(addr));
        }
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) {
            uint32_t bh[2], bl[2];
#if KT_TRANS

            const int irow = ksub * 16 + (lane & 15);
            const int off = sn * (CHUNK * GL_ROW_BYTES) + irow * GL_ROW_BYTES
                          + ((ns ^ (irow % GL_SEGS)) * 16);
            {
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(ab_hi + off);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
            }
            {
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(ab_lo + off);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
            }
#else
            const int b_row = sn * 64 + ns * GL_NT + (lane % 8);
            const int b_seg = ksub * 2 + ((lane / 8) & 1);
            {
                const uint32_t addr = ldm_swizzled(ab_hi, b_row, b_seg);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(bh[0]), "=r"(bh[1]) : "r"(addr));
            }
            {
                const uint32_t addr = ldm_swizzled(ab_lo, b_row, b_seg);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                             : "=r"(bl[0]), "=r"(bl[1]) : "r"(addr));
            }
#endif
            mma_bf16_hilo(od[ns], ah, al, bh, bl);
        }
    }

    {
        __syncthreads();
        float* tile = reinterpret_cast<float*>(ab);
        #pragma unroll
        for (int ns = 0; ns < 8; ++ns) {
            const int c0 = sn * 64 + ns * GL_NT + (lane % 4) * 2;
            const int r0 = sm * 16 + lane / 4;
            const int r1 = r0 + 8;
            const int w0 = 4 * (r0 % 8);
            const int w1 = 4 * (r1 % 8);
            tile[r0 * VD + (c0 ^ w0)] = od[ns][0];
            tile[r0 * VD + ((c0 + 1) ^ w0)] = od[ns][1];
            tile[r1 * VD + (c0 ^ w1)] = od[ns][2];
            tile[r1 * VD + ((c0 + 1) ^ w1)] = od[ns][3];
        }
        __syncthreads();
        for (int e = threadIdx.x * 4; e < CHUNK * VD; e += CHUNK_THREADS * 4) {
            const int i = e / VD, col = e - i * VD;
            if (i < n) {
                const float4 v = *reinterpret_cast<const float4*>(
                    &tile[i * VD + (col ^ (4 * (i % 8)))]);
                float4* __restrict__ dst = reinterpret_cast<float4*>(
                    &o_ptr[((size_t)(t0 + i) * heads + h) * VD + col]);
                if (walk_ptr == 0 || t0 + i < *walk_ptr) {
                    const float4 cur = *dst;
                    *dst = make_float4(cur.x + v.x, cur.y + v.y, cur.z + v.z, cur.w + v.w);
                } else {
                    *dst = v;
                }
            }
        }
    }
}

#define PREP_THREADS 256
#define PREP_ROW_WARPS 4

#define CP_ROWS 16
#define CP_CH   1024
#define CP_THREADS 256
#define CP_PER_THREAD (CP_CH / CP_THREADS)

extern "C" __global__ void gdn_conv_prep_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cw,
    float* __restrict__ v_out,
    uint8_t* __restrict__ q_hi, uint8_t* __restrict__ q_lo,
    uint8_t* __restrict__ k_hi, uint8_t* __restrict__ k_lo,
    float* __restrict__ kn,
    int rows, int channels, int k,
    const int* __restrict__ seq_start,
    const float* __restrict__ history,
    const int* __restrict__ slot_map,
    int heads, int kh, int kd, int nchunks, float q_scale)
{
    const int t0 = blockIdx.y * CP_ROWS;
    const int c0 = blockIdx.x * CP_CH;
    if (t0 >= rows || c0 >= channels) {
        return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ float stage[CP_ROWS * CP_CH];

    {
        const int my_c = c0 + tid * CP_PER_THREAD;
        const int nr = min(CP_ROWS, rows - t0);
        const int lo0 = seq_start ? seq_start[t0] : 0;
        const bool uniform = seq_start == 0 || seq_start[t0 + nr - 1] == lo0;
        if (uniform) {
            float wt[CP_PER_THREAD][4];
            #pragma unroll
            for (int j = 0; j < CP_PER_THREAD; ++j) {
                #pragma unroll
                for (int tap = 0; tap < 4; ++tap) wt[j][tap] = cw[(size_t)(my_c + j) * k + tap];
            }

            auto src_at = [&](int src, int t) -> float4 {
                if (src >= lo0) {
                    return *reinterpret_cast<const float4*>(x + (size_t)src * channels + my_c);
                }
                if (history != 0 && src >= lo0 - (k - 1)) {
                    const int slot = slot_map ? slot_map[t] : 0;
                    return *reinterpret_cast<const float4*>(
                        history + (size_t)(slot * (k - 1) + (src - lo0 + (k - 1))) * channels + my_c);
                }
                return make_float4(0.f, 0.f, 0.f, 0.f);
            };

            float4 w0 = src_at(t0 - 3, t0);
            float4 w1 = src_at(t0 - 2, t0);
            float4 w2 = src_at(t0 - 1, t0);
            float4 w3;
            float* dst = stage + tid * CP_PER_THREAD;
            #pragma unroll 4
            for (int r = 0; r < nr; ++r) {
                w3 = src_at(t0 + r, t0 + r);
                float ax = fmaf(wt[0][0], w0.x, 0.0f);
                float ay = fmaf(wt[1][0], w0.y, 0.0f);
                float az = fmaf(wt[2][0], w0.z, 0.0f);
                float aw = fmaf(wt[3][0], w0.w, 0.0f);
                ax = fmaf(wt[0][1], w1.x, ax); ay = fmaf(wt[1][1], w1.y, ay);
                az = fmaf(wt[2][1], w1.z, az); aw = fmaf(wt[3][1], w1.w, aw);
                ax = fmaf(wt[0][2], w2.x, ax); ay = fmaf(wt[1][2], w2.y, ay);
                az = fmaf(wt[2][2], w2.z, az); aw = fmaf(wt[3][2], w2.w, aw);
                ax = fmaf(wt[0][3], w3.x, ax); ay = fmaf(wt[1][3], w3.y, ay);
                az = fmaf(wt[2][3], w3.z, az); aw = fmaf(wt[3][3], w3.w, aw);
                w0 = w1; w1 = w2; w2 = w3;
                dst[0] = ax / (1.0f + __expf(-ax));
                dst[1] = ay / (1.0f + __expf(-ay));
                dst[2] = az / (1.0f + __expf(-az));
                dst[3] = aw / (1.0f + __expf(-aw));
                dst += CP_CH;
            }
        } else {
            for (int r = 0; r < nr; ++r) {
                const int t = t0 + r;
                const int lo = seq_start ? seq_start[t] : 0;
                float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
                for (int tap = 0; tap < k; ++tap) {
                    const int src = t - (k - 1) + tap;
                    float4 xv;
                    if (src >= lo) {
                        xv = *reinterpret_cast<const float4*>(x + (size_t)src * channels + my_c);
                    } else if (history != 0 && src < lo && src >= lo - (k - 1)) {
                        const int slot = slot_map ? slot_map[t] : 0;
                        xv = *reinterpret_cast<const float4*>(
                            history + (size_t)(slot * (k - 1) + (src - lo + (k - 1))) * channels + my_c);
                    } else {
                        xv = make_float4(0.f, 0.f, 0.f, 0.f);
                    }

                    acc.x += cw[(size_t)(my_c + 0) * k + tap] * xv.x;
                    acc.y += cw[(size_t)(my_c + 1) * k + tap] * xv.y;
                    acc.z += cw[(size_t)(my_c + 2) * k + tap] * xv.z;
                    acc.w += cw[(size_t)(my_c + 3) * k + tap] * xv.w;
                }
                float* dst = stage + r * CP_CH + tid * CP_PER_THREAD;
                dst[0] = acc.x / (1.0f + __expf(-acc.x));
                dst[1] = acc.y / (1.0f + __expf(-acc.y));
                dst[2] = acc.z / (1.0f + __expf(-acc.z));
                dst[3] = acc.w / (1.0f + __expf(-acc.w));
            }
        }
    }
    __syncthreads();

    const int ratio = heads / kh;
    const int qk_ch = 2 * kh * kd;
    const int nr = min(CP_ROWS, rows - t0);
    if (c0 < qk_ch && warp < (CP_CH / kd)) {
        const bool is_q = c0 < kh * kd;
        const int h = (c0 - (is_q ? 0 : kh * kd)) / kd + warp;
        if (h < kh) {
            const int base = warp * kd;
            for (int r = 0; r < nr; ++r) {
                const int t = t0 + r;
                const float* row = stage + r * CP_CH + base;
                float s = 0.0f;
                #pragma unroll 4
                for (int p = lane; p < kd; p += 32) {
                    s += row[p] * row[p];
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    s += __shfl_xor_sync(0xffffffffu, s, off);
                }

                const float rq = rsqrtf(s + 1e-6f) * (is_q ? q_scale : 1.0f);
                const int rr = t - (t / CHUNK) * CHUNK;
                for (int rep = 0; rep < ratio; ++rep) {
                    const int he = h * ratio + rep;

                    const int idx = he * nchunks + t / CHUNK;
                    float* ko = kn + (size_t)(t * heads + he) * kd;
                    #pragma unroll 4
                    for (int p = lane; p < kd; p += 32) {
                        const int tile = p / GL_KTILE;
                        const int c = p - tile * GL_KTILE;
                        const int seg = c / 8, e = c - (c / 8) * 8;
                        const size_t off = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                                         + (size_t)rr * GL_ROW_BYTES + (seg ^ (rr % 8)) * 16 + e * 2;
                        const float val = row[p] * rq;
                        if (is_q) {
                            split_bf16(val, (__nv_bfloat16*)(q_hi + off), (__nv_bfloat16*)(q_lo + off));
                        } else {
                            split_bf16(val, (__nv_bfloat16*)(k_hi + off), (__nv_bfloat16*)(k_lo + off));
                            ko[p] = val;
                        }
                    }
                }
            }
        }
    } else if (c0 >= qk_ch) {
        const int vdim = heads * kd;
        const int vc = c0 - qk_ch;
        const int n4 = CP_CH / 4;
        for (int r = 0; r < nr; ++r) {
            const int t = t0 + r;
            const float4* vs = reinterpret_cast<const float4*>(stage + r * CP_CH);
            float4* vo = reinterpret_cast<float4*>(v_out + (size_t)t * vdim + vc);
            for (int j = tid; j < n4; j += CP_THREADS) {
                vo[j] = vs[j];
            }
        }
    }
}

extern "C" __global__ void gdn_prep_kernel(const float* __restrict__ mixed,
                                           float* __restrict__ v_out,
                                           uint8_t* __restrict__ q_hi,
                                           uint8_t* __restrict__ q_lo,
                                           uint8_t* __restrict__ k_hi,
                                           uint8_t* __restrict__ k_lo,
                                           float* __restrict__ kn,
                                           int seq_len, int heads, int kh, int kd,
                                           int nchunks, int conv_dim, float q_scale) {
    const int t = blockIdx.x;
    if (t >= seq_len) {
        return;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* __restrict__ row = mixed + (size_t)t * conv_dim;
    const int ratio = heads / kh;
    if (warp < PREP_ROW_WARPS) {
        const int per = kh / PREP_ROW_WARPS;
        for (int i = 0; i < per; ++i) {
            const int h = warp * per + i;
            const float* __restrict__ qr = row + h * kd;
            const float* __restrict__ kr = row + kh * kd + h * kd;
            float sq = 0.0f, sk = 0.0f;
            for (int p = lane; p < kd; p += 32) {
                sq += qr[p] * qr[p];
                sk += kr[p] * kr[p];
            }
            #pragma unroll
            for (int s = 16; s > 0; s >>= 1) {
                sq += __shfl_xor_sync(0xffffffffu, sq, s);
                sk += __shfl_xor_sync(0xffffffffu, sk, s);
            }

            const float rq = rsqrtf(sq + 1e-6f) * q_scale;
            const float rk = rsqrtf(sk + 1e-6f);
            for (int rep = 0; rep < ratio; ++rep) {
                const int he = h * ratio + rep;
                const int idx = he * nchunks + t / CHUNK;
                const int r = t - (t / CHUNK) * CHUNK;
                float* __restrict__ ko = kn + (size_t)(t * heads + he) * kd;
                for (int p = lane; p < kd; p += 32) {
                    const int tile = p / GL_KTILE;
                    const int c = p - tile * GL_KTILE;
                    const int seg = c / 8, e = c - (c / 8) * 8;
                    const size_t off = ((size_t)idx * (KD / GL_KTILE) + tile) * (CHUNK * GL_ROW_BYTES)
                                     + (size_t)r * GL_ROW_BYTES + (seg ^ (r % 8)) * 16 + e * 2;
                    split_bf16(qr[p] * rq, (__nv_bfloat16*)(q_hi + off), (__nv_bfloat16*)(q_lo + off));
                    split_bf16(kr[p] * rk, (__nv_bfloat16*)(k_hi + off), (__nv_bfloat16*)(k_lo + off));
                    ko[p] = kr[p] * rk;
                }
            }
        }
    } else {
        const int vdim = heads * kd;
        const int n4 = vdim / 4;
        const float4* __restrict__ vs = reinterpret_cast<const float4*>(row + 2 * kh * kd);
        float4* __restrict__ vo = reinterpret_cast<float4*>(v_out + (size_t)t * vdim);
        for (int j = (warp - PREP_ROW_WARPS) * 32 + lane; j < n4;
             j += (PREP_THREADS / 32 - PREP_ROW_WARPS) * 32) {
            vo[j] = vs[j];
        }
    }
}
