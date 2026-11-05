#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_bf16.h>

#ifndef HD_MAX

#define HD_MAX 256
#endif
#define M_ROWS 16

#ifndef M_TILES
#define M_TILES 1
#endif
#define N_KEYS 8
#define K_ATOM 16
#ifndef BQ
#define BQ 128
#endif

#ifndef BK
#define BK 16
#endif

__device__ __forceinline__ void att_cp_async_16(void* dst, const void* src) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(src));
}
__device__ __forceinline__ void att_cp_commit() { asm volatile("cp.async.commit_group;\n"); }

#define ATT_CONSUMER_THREADS (8 * 32)

#ifndef ATT_NBUF
#if defined(ATT_WS)
#define ATT_NBUF 2
#elif defined(Q_IN_REGS) && BK <= 64
#define ATT_NBUF 1
#else
#define ATT_NBUF 1
#endif
#endif
#ifndef ATT_NBUF_V
#if defined(ATT_WS)
#define ATT_NBUF_V 2
#elif defined(Q_IN_REGS) && BK <= 64
#define ATT_NBUF_V 2
#else
#define ATT_NBUF_V 1
#endif
#endif

#ifdef ATT_WS
#define ATT_PRODUCER_WARPS 4
#define ATT_PRODUCER_THREADS (ATT_PRODUCER_WARPS * 32)
#define ATT_CTA_THREADS (ATT_CONSUMER_THREADS + ATT_PRODUCER_THREADS)

#define ATT_BAR_FULL(b)  (2 + (b))
#define ATT_BAR_EMPTY(b) (2 + ATT_NBUF + (b))

#define SETMAX_INC(n) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n))
#define SETMAX_DEC(n) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n))
#ifndef MMA_REG_MAX
#define MMA_REG_MAX 232
#endif
#ifndef PRODUCER_REG_MAX
#define PRODUCER_REG_MAX 40
#endif
__device__ __forceinline__ void att_arrive_full(int b)  { asm volatile("bar.arrive %0, %1;" :: "r"(ATT_BAR_FULL(b)),  "n"(ATT_CTA_THREADS)); }
__device__ __forceinline__ void att_sync_full(int b)    { asm volatile("bar.sync %0, %1;"   :: "r"(ATT_BAR_FULL(b)),  "n"(ATT_CTA_THREADS)); }
__device__ __forceinline__ void att_arrive_empty(int b) { asm volatile("bar.arrive %0, %1;" :: "r"(ATT_BAR_EMPTY(b)), "n"(ATT_CTA_THREADS)); }
__device__ __forceinline__ void att_sync_empty(int b)   { asm volatile("bar.sync %0, %1;"   :: "r"(ATT_BAR_EMPTY(b)), "n"(ATT_CTA_THREADS)); }
#endif
__device__ __forceinline__ void att_consumer_sync() {
    asm volatile("bar.sync 1, %0;" :: "r"(ATT_CONSUMER_THREADS));
}
__device__ __forceinline__ void att_cp_wait()   { asm volatile("cp.async.wait_group 0;\n"); }

#ifndef NCG

#define NCG 2
#endif

#define NWARPS (NCG * (BQ / (M_TILES * M_ROWS)))

#define RG (min(warp / NCG, (BQ / (M_TILES * M_ROWS)) - 1))
#define CG (warp % NCG)
#define NCOLS (HD_MAX / NCG)
#define WARPS_N (NCOLS / N_KEYS)

__device__ __forceinline__ void mma_bf16(float* d, uint4 a, uint2 b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ uint4 ldmatrix_x4(const uint32_t addr) {
    uint4 r;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(addr));
    return r;
}

__device__ __forceinline__ uint2 ldmatrix_x2_trans(const uint32_t addr) {
    uint2 r;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r.x), "=r"(r.y) : "r"(addr));
    return r;
}

__device__ __forceinline__ uint2 ldmatrix_x2(const uint32_t addr) {
    uint2 r;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r.x), "=r"(r.y) : "r"(addr));
    return r;
}

extern "C" __global__ void to_bf16_kernel(const float* __restrict__ x, unsigned short* __restrict__ out,
                                          long long n,

                                          const int* __restrict__ off_tokens_dev,

                                          int off_stride) {
    const long long local = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const long long off = off_tokens_dev != 0 ? (long long)(*off_tokens_dev) * off_stride : 0;

    const long long dst = local + off;
    if (local + 8 <= n) {
        unsigned short h[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            h[k] = __bfloat16_as_ushort(__float2bfloat16(x[local + k]));
        }

        uint4 packed;
        packed.x = (uint32_t)h[0] | ((uint32_t)h[1] << 16);
        packed.y = (uint32_t)h[2] | ((uint32_t)h[3] << 16);
        packed.z = (uint32_t)h[4] | ((uint32_t)h[5] << 16);
        packed.w = (uint32_t)h[6] | ((uint32_t)h[7] << 16);
        *(uint4*)&out[dst] = packed;
    } else {
        for (long long k = local; k < n; ++k) {
            out[off + k] = __bfloat16_as_ushort(__float2bfloat16(x[k]));
        }
    }
}

extern "C" __global__ void to_bf16_scatter_kernel(const float* __restrict__ x,
                                                  unsigned short* __restrict__ out,
                                                  long long n,

                                                  const int* __restrict__ pos_map,
                                                  int stride) {
    const long long local = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (local + 8 > n) {
        return;
    }
    const int token = (int)(local / stride);
    const long long dst = (long long)pos_map[token] * stride + (local - (long long)token * stride);
    unsigned short h[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        h[k] = __bfloat16_as_ushort(__float2bfloat16(x[local + k]));
    }
    uint4 packed;
    packed.x = (uint32_t)h[0] | ((uint32_t)h[1] << 16);
    packed.y = (uint32_t)h[2] | ((uint32_t)h[3] << 16);
    packed.z = (uint32_t)h[4] | ((uint32_t)h[5] << 16);
    packed.w = (uint32_t)h[6] | ((uint32_t)h[7] << 16);
    *reinterpret_cast<uint4*>(out + dst) = packed;
}

extern "C" __global__ void
#ifdef ATT_WS
__launch_bounds__(ATT_CTA_THREADS, 1)
#endif
attention_kernel(const unsigned short* __restrict__ q,
                                            const unsigned short* __restrict__ k,
                                            const unsigned short* __restrict__ v,
                                            float* __restrict__ out,
                                            int seq, int heads, int kv_heads, int head_dim,
                                            float scale,
                                            const int* __restrict__ seq_start,
                                            const int* __restrict__ seq_end,
                                            int seq_end_scalar_arg,
                                            const int* __restrict__ real_end_dev,

                                            const int* __restrict__ cache_len_dev,

                                            const int* __restrict__ cache_base_dev,

                                            const int* __restrict__ cpos_map,
                                            const int* __restrict__ cbase_map,
                                            const int* __restrict__ cslot_map
#ifdef ATT_PARTIAL

                                            , float* __restrict__ partial, int sp, int q_len,

                                            int tiles_per_split
#endif
                                            ) {
    const int seq_end_scalar = real_end_dev != nullptr ? real_end_dev[0] : seq_end_scalar_arg;

    const int cl = cache_len_dev != nullptr ? cache_len_dev[0] : 0;

    const int cbase = cache_base_dev != nullptr ? cache_base_dev[0] : 0;
    const bool has_cache = cache_len_dev != nullptr;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (head_dim > HD_MAX) {
        return;
    }
    const int gqa = heads / kv_heads;
#ifdef ATT_GQA_PACK

    const int kvh = blockIdx.x;
    #define RPH (BQ / gqa)

    #define QHOF(r) (kvh * gqa + (r) / RPH)
    #define QTOK(r) (t0 + (r) % RPH)
    #define EHOF(r) (kvh * gqa + (((r) - t0) / RPH))
    #define ETOK(r) (t0 + (((r) - t0) % RPH))
#else
    const int h = blockIdx.x;
    const int kvh = h / gqa;
    #define QHOF(r) (h)
    #define QTOK(r) (t0 + (r))
    #define EHOF(r) (h)
    #define ETOK(r) (r)
#endif
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

#ifdef ATT_GQA_PACK
    const int t0 = blockIdx.y * (BQ / gqa);
#else
    const int t0 = blockIdx.y * BQ;
#endif

    const int t0w = t0 + RG * (M_TILES * M_ROWS);

    #define QPAD (HD_MAX + 8)
#ifndef KSTRIDE
    #define KSTRIDE (HD_MAX + 8)
    #endif
    #ifndef VSTRIDE
    #define VSTRIDE (HD_MAX + 8)
    #endif

#define ATT_KB(t) (ATT_NBUF == 1 ? 0 : (ATT_NBUF == 2 ? ((t) & 1) : ((t) % ATT_NBUF)))
#define ATT_VBUF(t) (ATT_NBUF_V == 1 ? 0 : (ATT_NBUF_V == 2 ? ((t) & 1) : ((t) % ATT_NBUF_V)))
#ifdef Q_IN_REGS

    __shared__ __align__(16) unsigned short att_kbuf[ATT_NBUF][BK * KSTRIDE];
    #define ks (att_kbuf[ATT_KB(tile)])
#else
    __shared__ __align__(16) unsigned short qs[BQ * QPAD];
#endif

    #ifndef KSTRIDE
    #define KSTRIDE (HD_MAX + 8)
    #endif
#ifndef Q_IN_REGS

    __shared__ __align__(16) unsigned short att_kbuf[1][BK * KSTRIDE];
    #define ks (att_kbuf[0])
#endif

    #ifndef VSTRIDE
    #define VSTRIDE (HD_MAX + 8)
    #endif

    __shared__ __align__(16) unsigned short att_vbuf[ATT_NBUF_V][BK * VSTRIDE];

    #define vs (att_vbuf[ATT_VBUF(tile)])

#ifndef Q_IN_REGS
    for (int e = tid; e < BQ * head_dim; e += nthreads) {
        const int r = e / head_dim;
        const int d = e % head_dim;
        const int t = QTOK(r);
        qs[r * QPAD + d] = (t < seq) ? q[((size_t)t * heads + QHOF(r)) * head_dim + d] : 0;
    }
#endif

#ifdef Q_IN_REGS

    uint4 qf[HD_MAX / K_ATOM];
    #pragma unroll
    for (int kk = 0; kk < HD_MAX / K_ATOM; ++kk) {
        uint32_t w[4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int r = RG * (M_TILES * M_ROWS) + (m & 1) * 8 + (lane >> 2);
            const int col = kk * K_ATOM + (m >> 1) * 8 + 2 * (lane & 3);
            const int t = QTOK(r);
            const unsigned short* src = q + ((size_t)t * heads + QHOF(r)) * head_dim + col;
            w[m] = (t < seq) ? ((uint32_t)src[0] | ((uint32_t)src[1] << 16)) : 0u;
        }
        qf[kk] = make_uint4(w[0], w[1], w[2], w[3]);
    }
#endif

    float acc[M_TILES][WARPS_N][4];
    #pragma unroll
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int n = 0; n < WARPS_N; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[mt][n][i] = 0.0f;
            }
        }
    }
    float m_lo[M_TILES], m_hi[M_TILES], l_lo[M_TILES], l_hi[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        m_lo[mt] = -1.0e30f; m_hi[mt] = -1.0e30f; l_lo[mt] = 0.0f; l_hi[mt] = 0.0f;
    }
    const unsigned quad = 0xFu << (lane & ~3);

    const int seq_end_eff = seq_end_scalar > 0 ? seq_end_scalar : seq;

#ifdef ATT_NONCAUSAL

    const int stage_last = (seq_end ? seq_end[0] : seq_end_eff) - 1;
#else

    const int b_row = min(t0 + BQ - 1, seq - 1);
    const int b = seq_end ? min(b_row, seq_end[t0] - 1) : b_row;
    const int stage_last = seq_end ? min(b, seq_end[b] - 1) : min(b, seq_end_eff - 1);
#endif
    const int stage_max = cpos_map ? cpos_map[max(stage_last, 0)] : cl + stage_last;
#ifdef ATT_NONCAUSAL

    const int s_max_block = stage_max;
#else
    const int s_max_block = cpos_map ? cpos_map[b] : cl + t0 + BQ - 1;
#endif

#ifdef ATT_PARTIAL

    const int sp_eff = (int)gridDim.z > 1 ? (int)blockIdx.z : sp;

    const int blk_tok = ETOK(t0) < seq ? ETOK(t0) : seq - 1;
    const int blk_base = cbase_map ? cbase_map[blk_tok]
                                   : (has_cache ? cbase : (seq_start ? seq_start[blk_tok] : 0));
    const int total_tiles = (s_max_block / BK) + 1;

    const int base_tile = (blk_base / BK < total_tiles) ? (blk_base / BK) : 0;
    const int tile0 = base_tile + sp_eff * tiles_per_split;
    const int n_tiles = max(0, min(tiles_per_split, total_tiles - tile0));
#else
    const int sp_eff = 0;
    const int tile0 = 0;
    const int n_tiles = (s_max_block / BK) + 1;
#endif

    int att_buf = 0;

    auto stage_k = [&](int t, int b, int tid, int nthreads) {
        const int s0 = (tile0 + t) * BK;

        const int rows = max(0, min(BK, stage_max + 1 - s0));
        for (int c = tid; c < (BK - rows) * head_dim; c += nthreads) {
            const int d = c / (BK - rows);
            const int r = rows + (c % (BK - rows));
            att_kbuf[b][r * KSTRIDE + d] = 0;
        }
        for (int c = tid; c < rows * (head_dim / 8); c += nthreads) {
            const int r = c / (head_dim / 8);
            const int seg = c % (head_dim / 8);
            const int s = s0 + r;
            const unsigned short* kg = k + ((size_t)s * kv_heads + kvh) * head_dim + seg * 8;
            att_cp_async_16(&att_kbuf[b][r * KSTRIDE + seg * 8], kg);
        }
        att_cp_commit();
    };
    auto stage_v = [&](int t, int b, int tid, int nthreads) {
        const int s0 = (tile0 + t) * BK;

        const int rows = max(0, min(BK, stage_max + 1 - s0));
        for (int c = tid; c < (BK - rows) * head_dim; c += nthreads) {
            const int d = c / (BK - rows);
            const int r = rows + (c % (BK - rows));
            att_vbuf[b][r * VSTRIDE + d] = 0;
        }
        for (int c = tid; c < rows * (head_dim / 8); c += nthreads) {
            const int r = c / (head_dim / 8);
            const int seg = c % (head_dim / 8);
            const int s = s0 + r;
            const unsigned short* vg = v + ((size_t)s * kv_heads + kvh) * head_dim + seg * 8;
            att_cp_async_16(&att_vbuf[b][r * VSTRIDE + seg * 8], vg);
        }
        att_cp_commit();
    };

#ifdef ATT_WS

#ifndef ATT_REG_RENDEZVOUS
#define ATT_REG_RENDEZVOUS 1
#endif
    if (tid >= ATT_CONSUMER_THREADS) {
        SETMAX_DEC(PRODUCER_REG_MAX);
#if ATT_REG_RENDEZVOUS
        asm volatile("bar.arrive 8, %0;" :: "n"(ATT_CTA_THREADS));
#endif
        const int ptid = tid - ATT_CONSUMER_THREADS;
        for (int t = 0; t < n_tiles; ++t) {
            if (t >= ATT_NBUF) att_sync_empty(ATT_KB(t));
            stage_k(t, ATT_KB(t), ptid, ATT_PRODUCER_THREADS);
            stage_v(t, ATT_VBUF(t), ptid, ATT_PRODUCER_THREADS);
            att_cp_wait();
            att_arrive_full(ATT_KB(t));
        }
        return;
    }
#if ATT_REG_RENDEZVOUS

    asm volatile("bar.sync 8, %0;" :: "n"(ATT_CTA_THREADS));
#endif
    SETMAX_INC(MMA_REG_MAX);
#else
    stage_k(0, 0, tid, nthreads);
    stage_v(0, 0, tid, nthreads);
#endif
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int s0 = (tile0 + tile) * BK;

#ifdef ATT_WS
        att_sync_full(ATT_KB(tile));
#else
        att_cp_wait(); att_consumer_sync();
#endif
#ifdef ATT_PARTIAL
#endif

        float sc[M_TILES][BK / N_KEYS][4];
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < BK / N_KEYS; ++nt) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sc[mt][nt][i] = 0.0f;
                }
            }
        }
        #pragma unroll
        for (int ksub = 0; ksub < HD_MAX / K_ATOM; ++ksub) {
            uint4 a[M_TILES];
#ifdef Q_IN_REGS
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) { a[mt] = qf[ksub]; }
#else
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int a_row = RG * (M_TILES * M_ROWS) + mt * M_ROWS
                                + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int a_col = (lane >> 4) * 8 + ksub * K_ATOM;
                a[mt] = ldmatrix_x4((uint32_t)__cvta_generic_to_shared(qs + a_row * QPAD + a_col));
            }
#endif
            #pragma unroll
            for (int nt = 0; nt < BK / N_KEYS; ++nt) {
                const int b_key = nt * N_KEYS + (lane % 8);
                const int b_dim = ksub * K_ATOM + ((lane / 8) & 1) * 8;

                const uint2 b = ldmatrix_x2((uint32_t)__cvta_generic_to_shared(ks + b_key * KSTRIDE + b_dim));
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    mma_bf16(sc[mt][nt], a[mt], b);
                }
            }
        }

#ifndef ATT_WS
        att_consumer_sync();
        if (tile + 1 < n_tiles) { stage_k(tile + 1, ATT_KB(tile + 1), tid, nthreads);
                                  stage_v(tile + 1, ATT_VBUF(tile + 1), tid, nthreads); }
#endif

        const float slog2 = scale;
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < BK / N_KEYS; ++nt) {
                const int col0 = nt * N_KEYS + (lane & 3) * 2;
                const int row_lo = t0w + mt * M_ROWS + (lane >> 2);
                const int row_hi = row_lo + 8;

                const int key0 = s0 + col0;
                const int key1 = key0 + 1;

                const int tl = ETOK(row_lo);
                const int th = ETOK(row_hi);
                const int r_lo = tl < seq ? tl : seq - 1;
                const int r_hi = th < seq ? th : seq - 1;
                const int pos_lo = cpos_map ? cpos_map[r_lo] : cl + r_lo;
                const int pos_hi = cpos_map ? cpos_map[r_hi] : cl + r_hi;
                const int start_lo = cbase_map ? cbase_map[r_lo]
                    : (has_cache ? cbase : (seq_start ? seq_start[r_lo] : 0));
                const int start_hi = cbase_map ? cbase_map[r_hi]
                    : (has_cache ? cbase : (seq_start ? seq_start[r_hi] : 0));

                const int last_lo = (seq_end ? seq_end[r_lo] : seq_end_eff) - 1;
                const int last_hi = (seq_end ? seq_end[r_hi] : seq_end_eff) - 1;
                const int end_lo = (cpos_map ? cpos_map[min(last_lo, seq - 1)] : cl + last_lo) + 1;
                const int end_hi = (cpos_map ? cpos_map[min(last_hi, seq - 1)] : cl + last_hi) + 1;

#ifdef ATT_NONCAUSAL

                const bool ok_lo = pos_lo < seq;
                const bool ok_hi = pos_hi < seq;
#else
                const bool ok_lo = pos_lo < end_lo;
                const bool ok_hi = pos_hi < end_hi;
#endif

                const int row_lo_abs = pos_lo;
                const int row_hi_abs = pos_hi;

#ifdef ATT_NONCAUSAL

                sc[mt][nt][0] = (ok_lo && key0 >= start_lo && key0 < end_lo) ? sc[mt][nt][0] * slog2 : -INFINITY;
                sc[mt][nt][1] = (ok_lo && key1 >= start_lo && key1 < end_lo) ? sc[mt][nt][1] * slog2 : -INFINITY;
                sc[mt][nt][2] = (ok_hi && key0 >= start_hi && key0 < end_hi) ? sc[mt][nt][2] * slog2 : -INFINITY;
                sc[mt][nt][3] = (ok_hi && key1 >= start_hi && key1 < end_hi) ? sc[mt][nt][3] * slog2 : -INFINITY;
#else
                sc[mt][nt][0] = (ok_lo && key0 <= row_lo_abs && key0 >= start_lo && key0 < end_lo) ? sc[mt][nt][0] * slog2 : -INFINITY;
                sc[mt][nt][1] = (ok_lo && key1 <= row_lo_abs && key1 >= start_lo && key1 < end_lo) ? sc[mt][nt][1] * slog2 : -INFINITY;
                sc[mt][nt][2] = (ok_hi && key0 <= row_hi_abs && key0 >= start_hi && key0 < end_hi) ? sc[mt][nt][2] * slog2 : -INFINITY;
                sc[mt][nt][3] = (ok_hi && key1 <= row_hi_abs && key1 >= start_hi && key1 < end_hi) ? sc[mt][nt][3] * slog2 : -INFINITY;
#endif
            }

            const float m_lo_prev = m_lo[mt];
            const float m_hi_prev = m_hi[mt];
            #pragma unroll
            for (int nt = 0; nt < BK / N_KEYS; ++nt) {
                m_lo[mt] = fmaxf(m_lo[mt], fmaxf(sc[mt][nt][0], sc[mt][nt][1]));
                m_hi[mt] = fmaxf(m_hi[mt], fmaxf(sc[mt][nt][2], sc[mt][nt][3]));
            }

            m_lo[mt] = fmaxf(m_lo[mt], -1.0e30f);
            m_hi[mt] = fmaxf(m_hi[mt], -1.0e30f);

            float m_lo_new = m_lo[mt], m_hi_new = m_hi[mt];
            m_lo_new = fmaxf(m_lo_new, __shfl_xor_sync(quad, m_lo_new, 1));
            m_lo_new = fmaxf(m_lo_new, __shfl_xor_sync(quad, m_lo_new, 2));
            m_hi_new = fmaxf(m_hi_new, __shfl_xor_sync(quad, m_hi_new, 1));
            m_hi_new = fmaxf(m_hi_new, __shfl_xor_sync(quad, m_hi_new, 2));

            const float alpha_lo = __expf(m_lo_prev - m_lo_new);
            const float alpha_hi = __expf(m_hi_prev - m_hi_new);
            m_lo[mt] = m_lo_new;
            m_hi[mt] = m_hi_new;

            float psum_lo = 0.0f, psum_hi = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < BK / N_KEYS; ++nt) {
                sc[mt][nt][0] = __expf(sc[mt][nt][0] - m_lo[mt]);
                sc[mt][nt][1] = __expf(sc[mt][nt][1] - m_lo[mt]);
                sc[mt][nt][2] = __expf(sc[mt][nt][2] - m_hi[mt]);
                sc[mt][nt][3] = __expf(sc[mt][nt][3] - m_hi[mt]);
                psum_lo += sc[mt][nt][0] + sc[mt][nt][1];
                psum_hi += sc[mt][nt][2] + sc[mt][nt][3];
            }
            psum_lo += __shfl_xor_sync(quad, psum_lo, 1);
            psum_lo += __shfl_xor_sync(quad, psum_lo, 2);
            psum_hi += __shfl_xor_sync(quad, psum_hi, 1);
            psum_hi += __shfl_xor_sync(quad, psum_hi, 2);
            l_lo[mt] = l_lo[mt] * alpha_lo + psum_lo;
            l_hi[mt] = l_hi[mt] * alpha_hi + psum_hi;

#ifdef ATT_NO_SKIP

            #pragma unroll
            for (int n = 0; n < WARPS_N; ++n) {
                acc[mt][n][0] *= alpha_lo;
                acc[mt][n][1] *= alpha_lo;
                acc[mt][n][2] *= alpha_hi;
                acc[mt][n][3] *= alpha_hi;
            }
#else
            if (alpha_lo != 1.0f) {
                #pragma unroll
                for (int n = 0; n < WARPS_N; ++n) {
                    acc[mt][n][0] *= alpha_lo;
                    acc[mt][n][1] *= alpha_lo;
                }
            }
            if (alpha_hi != 1.0f) {
                #pragma unroll
                for (int n = 0; n < WARPS_N; ++n) {
                    acc[mt][n][2] *= alpha_hi;
                    acc[mt][n][3] *= alpha_hi;
                }
            }
#endif
        }

        #pragma unroll
        for (int kt = 0; kt < BK / K_ATOM; ++kt) {
            #pragma unroll
            for (int n = 0; n < WARPS_N; ++n) {
                const int b_out = CG * NCOLS + n * N_KEYS;

                const int v_row = kt * K_ATOM + lane;
                const uint2 b = ldmatrix_x2_trans((uint32_t)__cvta_generic_to_shared(vs + v_row * VSTRIDE + b_out));
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    const int t0i = kt * 2;
                    unsigned short p01[2] = { __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i][0])),
                                              __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i][1])) };
                    unsigned short p23[2] = { __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i][2])),
                                              __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i][3])) };
                    unsigned short p45[2] = { __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i + 1][0])),
                                              __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i + 1][1])) };
                    unsigned short p67[2] = { __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i + 1][2])),
                                              __bfloat16_as_ushort(__float2bfloat16(sc[mt][t0i + 1][3])) };
                    uint4 a;
                    a.x = (uint32_t)p01[0] | ((uint32_t)p01[1] << 16);
                    a.y = (uint32_t)p23[0] | ((uint32_t)p23[1] << 16);
                    a.z = (uint32_t)p45[0] | ((uint32_t)p45[1] << 16);
                    a.w = (uint32_t)p67[0] | ((uint32_t)p67[1] << 16);
                    mma_bf16(acc[mt][n], a, b);
                }
            }
        }
#ifdef ATT_WS
        if (tile + ATT_NBUF < n_tiles) {
            att_arrive_empty(ATT_KB(tile));
        }
#else
        att_consumer_sync();
#endif
    }

    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        const float inv_lo = (l_lo[mt] > 0.0f) ? 1.0f / l_lo[mt] : 0.0f;
        const float inv_hi = (l_hi[mt] > 0.0f) ? 1.0f / l_hi[mt] : 0.0f;
        const int rlo = t0w + mt * M_ROWS + (lane >> 2);
        const int rhi = rlo + 8;
        #pragma unroll
        for (int n = 0; n < WARPS_N; ++n) {
            const int col = CG * NCOLS + n * N_KEYS + (lane & 3) * 2;
#ifndef ATT_PARTIAL
            if (ETOK(rlo) < seq && col < head_dim) {
                out[((size_t)ETOK(rlo) * heads + EHOF(rlo)) * head_dim + col] = acc[mt][n][0] * inv_lo;
            }
            if (ETOK(rlo) < seq && col + 1 < head_dim) {
                out[((size_t)ETOK(rlo) * heads + EHOF(rlo)) * head_dim + col + 1] = acc[mt][n][1] * inv_lo;
            }
            if (ETOK(rhi) < seq && col < head_dim) {
                out[((size_t)ETOK(rhi) * heads + EHOF(rhi)) * head_dim + col] = acc[mt][n][2] * inv_hi;
            }
            if (ETOK(rhi) < seq && col + 1 < head_dim) {
                out[((size_t)ETOK(rhi) * heads + EHOF(rhi)) * head_dim + col + 1] = acc[mt][n][3] * inv_hi;
            }
#endif

#ifdef ATT_PARTIAL
            if ((lane & 3) == 0) {
                if (ETOK(rlo) < seq) {
                    float* r = partial + (((size_t)sp_eff * q_len + ETOK(rlo)) * heads + EHOF(rlo)) * (head_dim + 2);

                    r[0] = m_lo[mt];
                    r[1] = l_lo[mt];
                }
                if (ETOK(rhi) < seq) {
                    float* r = partial + (((size_t)sp_eff * q_len + ETOK(rhi)) * heads + EHOF(rhi)) * (head_dim + 2);
                    r[0] = m_hi[mt];
                    r[1] = l_hi[mt];
                }
            }
            if (ETOK(rlo) < seq && col < head_dim) {
                partial[(((size_t)sp_eff * q_len + ETOK(rlo)) * heads + EHOF(rlo)) * (head_dim + 2) + 2 + col] = acc[mt][n][0];
            }
            if (ETOK(rlo) < seq && col + 1 < head_dim) {
                partial[(((size_t)sp_eff * q_len + ETOK(rlo)) * heads + EHOF(rlo)) * (head_dim + 2) + 2 + col + 1] = acc[mt][n][1];
            }
            if (ETOK(rhi) < seq && col < head_dim) {
                const size_t ix = (((size_t)sp_eff * q_len + ETOK(rhi)) * heads + EHOF(rhi)) * (head_dim + 2) + 2 + col;
                partial[ix] = acc[mt][n][2];
            }
            if (ETOK(rhi) < seq && col + 1 < head_dim) {
                partial[(((size_t)sp_eff * q_len + ETOK(rhi)) * heads + EHOF(rhi)) * (head_dim + 2) + 2 + col + 1] = acc[mt][n][3];
            }
#endif
        }
    }
}
