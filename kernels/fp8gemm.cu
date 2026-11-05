#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cstdint>

#ifndef FP8_PLAIN_ATOM
#define FP8_PLAIN_ATOM 0
#endif

#ifndef GATED_GEMM_FP8
#define GATED_GEMM_FP8 0
#endif
#if GATED_GEMM_FP8
#define NMAT 2
#else
#define NMAT 1
#endif

#define KCHUNK_FP8 32

#ifndef N_TILE
#define N_TILE 64
#endif
#define N_SUB (N_TILE / 8)

#ifndef SUB_CHUNKS
#define SUB_CHUNKS 4
#endif
#define KTILE_FP8 (KCHUNK_FP8 * SUB_CHUNKS)
#define A_INT4_PER_ROW (KTILE_FP8 / 16)
#define B_INT4_PER_ROW (KTILE_FP8 / 16)

#ifndef SMEM_ROW_INT4
#define SMEM_ROW_INT4 (KTILE_FP8 / 16)
#endif
#define SMEM_ROW_BYTES (SMEM_ROW_INT4 * 16)

#define SMEM_SWIZZLE_MOD (SMEM_ROW_BYTES >= 128 ? 8 : (SMEM_ROW_BYTES >= 64 ? 4 : 2))

#define SMEM_SWIZZLE_ROW(row) (((row) * SMEM_ROW_BYTES) >> 7)
#ifndef STAGES

#ifndef STAGES
#define STAGES 3
#endif
#endif

#ifndef EPI_N
#define EPI_N 32
#endif

#ifndef EPI_BUFS
#define EPI_BUFS 2
#endif

#ifndef EPI_SEPARATE
#define EPI_SEPARATE 1
#endif

#ifndef EPI_SWZ
#define EPI_SWZ 1
#endif

#ifndef FP8_PREFETCH
#define FP8_PREFETCH 0
#endif

#ifndef FP8_DIRECT_STORE
#define FP8_DIRECT_STORE 0
#endif

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    const uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(s), "l"(gmem_src) : "memory");
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::
                 "n"(N));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void tma_store_2d(const void* smem, const void* map, int inner, int outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::
        "l"((unsigned long long)(uintptr_t)map), "r"(inner), "r"(outer),
        "r"(smem_u32(smem)) : "memory");
}

__device__ __forceinline__ void mma_barrier() {
    asm volatile("bar.sync 15, 256;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d(void* smem, const void* map, int inner, int outer,
                                            uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n" ::
        "r"(smem_u32(smem)), "l"((unsigned long long)(uintptr_t)map),
        "r"(inner), "r"(outer), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::
                 "r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar)));
}

__device__ __forceinline__ void cp_async_mbarrier_arrive_noinc(uint64_t* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n"
        "CLEF_FP8_MB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra CLEF_FP8_MB_WAIT;\n}\n" :: "r"(smem_u32(bar)), "r"(phase));
}

#ifndef WARPS_M
#define WARPS_M 4
#endif
#ifndef WARPS_N
#define WARPS_N 2
#endif
#ifndef WARP_M_ROWS
#define WARP_M_ROWS 16
#endif
#define M_ATOMS (WARP_M_ROWS / 16)
#define NUM_MMA_WARPS (WARPS_M * WARPS_N)

#define PRODUCER_WARPS 4
#define CTA_THREADS ((NUM_MMA_WARPS + PRODUCER_WARPS) * 32)

#define STORE_WARP_ID (NUM_MMA_WARPS + 1)
#define EPILOG_READY_ID 3
#define MMA_SYNC_ID 4
#define EPILOG_THREADS ((NUM_MMA_WARPS + 1) * 32)

__device__ __forceinline__ void mma_sync() {
    asm volatile("bar.sync %0, %1;\n" :: "n"(MMA_SYNC_ID), "n"(NUM_MMA_WARPS * 32) : "memory");
}

__device__ __forceinline__ void epilog_ready_arrive() {
    asm volatile("bar.arrive %0, %1;\n" :: "n"(EPILOG_READY_ID), "n"(EPILOG_THREADS) : "memory");
}
__device__ __forceinline__ void epilog_ready_sync() {
    asm volatile("bar.sync %0, %1;\n" :: "n"(EPILOG_READY_ID), "n"(EPILOG_THREADS) : "memory");
}

__device__ __forceinline__ void epilog_free_sync(int buf) {
    if (buf) {
        asm volatile("bar.sync 2, %0;\n" :: "n"(EPILOG_THREADS) : "memory");
    } else {
        asm volatile("bar.sync 1, %0;\n" :: "n"(EPILOG_THREADS) : "memory");
    }
}
__device__ __forceinline__ void epilog_free_arrive(int buf) {
    if (buf) {
        asm volatile("bar.arrive 2, %0;\n" :: "n"(EPILOG_THREADS) : "memory");
    } else {
        asm volatile("bar.arrive 1, %0;\n" :: "n"(EPILOG_THREADS) : "memory");
    }
}

#define BLOCK_THREADS CTA_THREADS
#define M_TILE (WARPS_M * WARP_M_ROWS)

#ifndef MMA_REG_MAX
#define MMA_REG_MAX 128
#endif

#ifndef PRODUCER_REG_MAX
#define PRODUCER_REG_MAX 40
#endif
#define SETMAXNREG_INC(n) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n))
#define SETMAXNREG_DEC(n) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n))
#define N_BLOCK (WARPS_N * N_TILE)

#define A_SMEM_ROWS M_TILE
#define B_SMEM_ROWS N_BLOCK
#define A_STAGE_BYTES (A_SMEM_ROWS * SMEM_ROW_BYTES)
#define B_STAGE_BYTES (B_SMEM_ROWS * SMEM_ROW_BYTES)

#if EPI_SEPARATE
extern "C" __global__ void __launch_bounds__(384, 1) fp8_gemm(
#else
extern "C" __global__ void __launch_bounds__(384, 2) fp8_gemm(
#endif
const __grid_constant__ CUtensorMap map_a,
                                         const __grid_constant__ CUtensorMap map_b,
#if GATED_GEMM_FP8

                                    const __grid_constant__ CUtensorMap map_bu,
#endif
                                         const uint8_t* __restrict__ a_ptr,
                                    const uint8_t* __restrict__ b_ptr,
                                    const uint16_t* __restrict__ sa,
                                    const uint16_t* __restrict__ sb,
#if GATED_GEMM_FP8
                                    const uint16_t* __restrict__ sbu,
#endif
                                    float* __restrict__ c,

                                    unsigned long long add_addr,
                                    const __grid_constant__ CUtensorMap map_c,
                                    int m, int n, int k,

                                    int sa_rows,

                                    int a_rows,

                                    int log_swizzle) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    const int m_tiles = (m + M_TILE - 1) / M_TILE;
    const int n_tiles = (n + N_BLOCK - 1) / N_BLOCK;
    const int n_work_total = m_tiles * n_tiles;
    const int my_work = (n_work_total - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;

    const bool sw_ok = log_swizzle > 0 && log_swizzle < 31
                       && (m_tiles & ((1 << log_swizzle) - 1)) == 0;
    const int sw = sw_ok ? (1 << log_swizzle) : 1;
    const int log_sw = sw_ok ? log_swizzle : 0;
    const int blk_major = m_tiles / sw;

    auto work_base = [&](int w, int* mb, int* nb) {
        const int offset = w & (sw - 1);
        const int extra = w >> log_sw;
        const int n_tile = extra / blk_major;
        const int m_tile = (extra % blk_major) * sw + offset;
        *mb = m_tile * M_TILE;
        *nb = n_tile * N_BLOCK;
    };

    const bool STAGED_EPI = (n % 4 == 0) && !FP8_DIRECT_STORE;
    constexpr int TILE_GATE_ID = 4;
    constexpr int TILE_GATE_THREADS = 64;

    __shared__ __align__(1024) uint8_t a_smem[STAGES][A_STAGE_BYTES];

    __shared__ uint16_t sa_s[M_TILE];
    __shared__ uint16_t sb_s[NMAT][N_BLOCK];

    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
    __shared__ __align__(1024) uint8_t b_smem[NMAT][STAGES][B_STAGE_BYTES];

#if EPI_SEPARATE

#if EPI_SEPARATE
    __shared__ __align__(1024) float epi_own[EPI_BUFS][M_TILE * EPI_N];
#endif
#endif

    const int k_tiles = (k + KTILE_FP8 - 1) / KTILE_FP8;

    float d[NMAT][M_ATOMS][N_SUB][4];
    #pragma unroll
    for (int mat = 0; mat < NMAT; ++mat)
    #pragma unroll
    for (int a = 0; a < M_ATOMS; ++a)
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            d[mat][a][ns][0] = d[mat][a][ns][1] = d[mat][a][ns][2] = d[mat][a][ns][3] = 0.0f;
        }

    auto issue_tma = [&](int flat) {
        if (warp != NUM_MMA_WARPS || lane != 0) return;
        const int wi = flat / k_tiles;
        const int kt = flat - wi * k_tiles;
        const int st = flat % STAGES;
        int mb, nb;
        work_base((int)blockIdx.x + wi * (int)gridDim.x, &mb, &nb);
        mbar_arrive_expect_tx(&full[st], A_STAGE_BYTES + NMAT * B_STAGE_BYTES);
        tma_load_2d(a_smem[st], &map_a, kt * KTILE_FP8, mb, &full[st]);
        tma_load_2d(b_smem[0][st], &map_b, kt * KTILE_FP8, nb, &full[st]);
#if GATED_GEMM_FP8
        tma_load_2d(b_smem[1][st], &map_bu, kt * KTILE_FP8, nb, &full[st]);
#endif
    };
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);
            mbar_init(&empty[s], NUM_MMA_WARPS);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }
    __syncthreads();
    if (warp >= NUM_MMA_WARPS) {
        SETMAXNREG_DEC(PRODUCER_REG_MAX);
        if (warp == STORE_WARP_ID && STAGED_EPI) {
            constexpr int HALVES = N_BLOCK / EPI_N;
#if EPI_SEPARATE
            float* epi = &epi_own[0][0];
#else
            float* epi = reinterpret_cast<float*>(a_smem);
#endif

            int half_flat = 0;
            for (int wi = 0; wi < my_work; ++wi) {
            int m_base, n_base;
            work_base((int)blockIdx.x + wi * (int)gridDim.x, &m_base, &n_base);
            #pragma unroll
            for (int half = 0; half < HALVES; ++half, ++half_flat) {
#if EPI_SEPARATE
                const int eidx = half_flat;
#else
                const int eidx = half;
#endif
                const int buf = eidx & (EPI_BUFS - 1);
                if (eidx >= EPI_BUFS && lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read %0;\n" ::
                                 "n"(EPI_BUFS - 1));
                }
                epilog_free_arrive(buf);
                epilog_ready_sync();
                if (lane == 0) {
                    tma_store_2d(epi + buf * (M_TILE * EPI_N), &map_c,
                                 (n_base + half * EPI_N) * 4, m_base);
                    asm volatile("cp.async.bulk.commit_group;\n");
                }
            }
            if (!EPI_SEPARATE) {
                if (lane == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;\n");
                }

                asm volatile("bar.arrive %0, %1;\n" :: "n"(TILE_GATE_ID), "n"(TILE_GATE_THREADS) : "memory");
            }
            }
            return;
        }

        const int total = my_work * k_tiles;
        #pragma unroll
        for (int t = 0; t < STAGES - 1; ++t) {
            if (t < total) {
                issue_tma(t);
            }
        }
        for (int flat = STAGES - 1; flat < total; ++flat) {
            const int st = flat % STAGES;
            if (STAGED_EPI && !EPI_SEPARATE && flat % k_tiles == 0 && flat > 0) {
                asm volatile("bar.sync %0, %1;\n" :: "n"(TILE_GATE_ID), "n"(TILE_GATE_THREADS) : "memory");
            }
            if (flat >= STAGES) {
                mbar_wait(&empty[st], ((flat - STAGES) / STAGES) & 1);
            }
            issue_tma(flat);
        }
        return;
    }
    SETMAXNREG_INC(MMA_REG_MAX);

    int g = 0;

    int epi_half_flat = 0;
    for (int wi = 0; wi < my_work; ++wi) {
    int m_base, n_base;
    work_base((int)blockIdx.x + wi * (int)gridDim.x, &m_base, &n_base);
    const int m_warp = m_base + warp_m * WARP_M_ROWS;
    const int n_warp = n_base + warp_n * N_TILE;
    mma_sync();

    {
        for (int i = threadIdx.x; i < M_TILE; i += NUM_MMA_WARPS * 32) {
            const int r = m_base + i;
            const int rc = min(r, sa_rows - 1);
            const unsigned ok = (r < a_rows && r < sa_rows) ? 0xffffu : 0u;
            sa_s[i] = (uint16_t)(sa[rc] & ok);
        }
        for (int i = threadIdx.x; i < N_BLOCK; i += NUM_MMA_WARPS * 32) {
            const int r = n_base + i;
            const int rc = min(r, n - 1);
            const unsigned ok = (r < n) ? 0xffffu : 0u;
            sb_s[0][i] = (uint16_t)(sb[rc] & ok);
#if GATED_GEMM_FP8
            sb_s[1][i] = (uint16_t)(sbu[rc] & ok);
#endif
        }
    }
    mma_sync();

    #pragma unroll
    for (int mat = 0; mat < NMAT; ++mat)
    #pragma unroll
    for (int a = 0; a < M_ATOMS; ++a)
        #pragma unroll
        for (int ns = 0; ns < N_SUB; ++ns) {
            d[mat][a][ns][0] = d[mat][a][ns][1] = d[mat][a][ns][2] = d[mat][a][ns][3] = 0.0f;
        }
    for (int tile = 0; tile < k_tiles; ++tile, ++g) {
        const int stage = g % STAGES;
        const int k0 = tile * KTILE_FP8;
        mbar_wait(&full[stage], (g / STAGES) & 1);

        auto load_b_one = [&](int mat, int ksub, int ns) -> uint2 {
            const int b_row = warp_n * N_TILE + ns * 8 + (lane % 8);
            const int b_seg = ksub * (KCHUNK_FP8 / 16) + ((lane / 8) & 1);

            const int b_scol = b_seg ^ (SMEM_SWIZZLE_ROW(b_row) & (SMEM_SWIZZLE_MOD - 1));
            const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                b_smem[mat][stage] + (b_row * SMEM_ROW_INT4 + b_scol) * 16);
            uint2 r;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(r.x), "=r"(r.y)
                : "r"(addr));
            return r;
        };
        auto load_a_one = [&](int ksub, int atom) -> uint4 {
            const int a_row = warp_m * WARP_M_ROWS + atom * 16 + (lane % 16);
            const int a_seg = ksub * (KCHUNK_FP8 / 16) + (lane / 16);

            const int a_scol = a_seg ^ (SMEM_SWIZZLE_ROW(a_row) & (SMEM_SWIZZLE_MOD - 1));
            const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                a_smem[stage] + (a_row * SMEM_ROW_INT4 + a_scol) * 16);
            uint4 r;
            asm volatile(
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                : "r"(addr));
            return r;
        };

        auto mma_one = [&](int mat, int atom, int ns, uint4 a, uint2 b) {
#if FP8_PLAIN_ATOM

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[mat][atom][ns][0]), "+f"(d[mat][atom][ns][1]),
                  "+f"(d[mat][atom][ns][2]), "+f"(d[mat][atom][ns][3])
                : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
                  "r"(b.x), "r"(b.y));
#else

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X."
                "f32.e4m3.e4m3.f32.ue8m0 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                "{%10}, {%12,%13}, {%11}, {%14,%15};\n"
                : "+f"(d[mat][atom][ns][0]), "+f"(d[mat][atom][ns][1]),
                  "+f"(d[mat][atom][ns][2]), "+f"(d[mat][atom][ns][3])
                : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
                  "r"(b.x), "r"(b.y),
                  "r"(0x7f7f7f7fu), "r"(0x7f7f7f7fu),
                  "h"((unsigned short)0), "h"((unsigned short)0),
                  "h"((unsigned short)0), "h"((unsigned short)0));
#endif
        };

#if FP8_PREFETCH

        uint4 a_pipe[2][M_ATOMS];
        uint2 b_pipe[NMAT][2][N_SUB];
        if (k0 < k) {
            #pragma unroll
            #pragma unroll
            for (int mat = 0; mat < NMAT; ++mat)
            for (int ns = 0; ns < N_SUB; ++ns) b_pipe[mat][0][ns] = load_b_one(mat, 0, ns);
            #pragma unroll
            for (int atom = 0; atom < M_ATOMS; ++atom) a_pipe[0][atom] = load_a_one(0, atom);
        }

#define KSUB(K)                                                                       \
        if (k0 + (K) * KCHUNK_FP8 < k) {                                              \
            constexpr int cur = (K) & 1;                                              \
            if ((K) + 1 < SUB_CHUNKS && k0 + ((K) + 1) * KCHUNK_FP8 < k) {            \
                _Pragma("unroll")                                                     \
                for (int ns = 0; ns < N_SUB; ++ns)                                    \
                    for (int mat = 0; mat < NMAT; ++mat)                            \
                    b_pipe[mat][1 - cur][ns] = load_b_one(mat, (K) + 1, ns);           \
                _Pragma("unroll")                                                     \
                for (int atom = 0; atom < M_ATOMS; ++atom)                            \
                    a_pipe[1 - cur][atom] = load_a_one((K) + 1, atom);                \
            }                                                                         \
            _Pragma("unroll")                                                         \
            for (int atom = 0; atom < M_ATOMS; ++atom) {                              \
                _Pragma("unroll")                                                     \
                for (int ns = 0; ns < N_SUB; ++ns) {                                  \
                    for (int mat = 0; mat < NMAT; ++mat)                            \
                    mma_one(mat, atom, ns, a_pipe[cur][atom], b_pipe[mat][cur][ns]);   \
                }                                                                     \
            }                                                                         \
        }
        KSUB(0)
        KSUB(1)
        KSUB(2)
        KSUB(3)
#undef KSUB
#else
        #pragma unroll
        for (int ksub = 0; ksub < SUB_CHUNKS; ++ksub) {
            if (k0 + ksub * KCHUNK_FP8 >= k) {
                continue;
            }

            uint2 b_ser[NMAT][N_SUB];
            #pragma unroll
            #pragma unroll
            for (int mat = 0; mat < NMAT; ++mat)
            for (int ns = 0; ns < N_SUB; ++ns) b_ser[mat][ns] = load_b_one(mat, ksub, ns);
            #pragma unroll
            for (int atom = 0; atom < M_ATOMS; ++atom) {
                const uint4 a_ser = load_a_one(ksub, atom);
                #pragma unroll
                for (int ns = 0; ns < N_SUB; ++ns) {
                    for (int mat = 0; mat < NMAT; ++mat)
                    mma_one(mat, atom, ns, a_ser, b_ser[mat][ns]);
                }
            }
        }
#endif
        if (lane == 0) {
            mbar_arrive(&empty[stage]);
        }
    }
    if (STAGED_EPI) {
#if EPI_SEPARATE
        float* epi = &epi_own[0][0];
#else
        float* epi = reinterpret_cast<float*>(a_smem);
#endif
        const int ec_row = lane >> 2;
        const int ec_col = (lane & 3) * 2;

        #pragma unroll
        for (int half = 0; half < N_BLOCK / EPI_N; ++half, ++epi_half_flat) {
            const int n_off = half * EPI_N;

#if EPI_SEPARATE
            const int eidx = epi_half_flat;
#else
            const int eidx = half;
#endif

            float* ebuf = epi + (eidx & (EPI_BUFS - 1)) * (M_TILE * EPI_N);
            epilog_free_sync(eidx & (EPI_BUFS - 1));
            #pragma unroll
            for (int atom = 0; atom < M_ATOMS; ++atom) {
                const float s_lo = __uint_as_float(
                    (uint32_t)sa_s[warp_m * WARP_M_ROWS + atom * 16 + ec_row] << 16);
                const float s_hi = __uint_as_float(
                    (uint32_t)sa_s[warp_m * WARP_M_ROWS + atom * 16 + ec_row + 8] << 16);
                #pragma unroll
                for (int ns = 0; ns < N_SUB; ++ns) {
                    const int col_local = warp_n * N_TILE + ns * 8 + ec_col - n_off;
                    if (col_local < 0 || col_local + 1 >= EPI_N) {
                        continue;
                    }
                    const int n_loc = warp_n * N_TILE + ns * 8;
                    const float t0 = __uint_as_float((uint32_t)sb_s[0][n_loc + ec_col + 0] << 16);
                    const float t1 = __uint_as_float((uint32_t)sb_s[0][n_loc + ec_col + 1] << 16);
                    const int er0 = warp_m * WARP_M_ROWS + atom * 16 + ec_row;

                    const float* __restrict__ add_ptr = (const float*)(uintptr_t)add_addr;
                    const int abs_row = m_base + er0;
                    const int abs_col = n_base + warp_n * N_TILE + ns * 8 + ec_col;
                    const bool live_r0 = (abs_row < m);
                    const bool live_r1 = (abs_row + 8 < m);
                    const bool live_c0 = (abs_col < n);
                    const bool live_c1 = (abs_col + 1 < n);
                    const size_t abs0 = (size_t)abs_row * n + abs_col;
                    float v00 = d[0][atom][ns][0] * s_lo * t0;
                    float v01 = d[0][atom][ns][1] * s_lo * t1;
                    float v10 = d[0][atom][ns][2] * s_hi * t0;
                    float v11 = d[0][atom][ns][3] * s_hi * t1;
#if GATED_GEMM_FP8

                    {
                        const float t0u = __uint_as_float((uint32_t)sb_s[1][n_loc + ec_col + 0] << 16);
                        const float t1u = __uint_as_float((uint32_t)sb_s[1][n_loc + ec_col + 1] << 16);
                        const float u00 = d[1][atom][ns][0] * s_lo * t0u;
                        const float u01 = d[1][atom][ns][1] * s_lo * t1u;
                        const float u10 = d[1][atom][ns][2] * s_hi * t0u;
                        const float u11 = d[1][atom][ns][3] * s_hi * t1u;
                        v00 = (v00 / (1.0f + __expf(-v00))) * u00;
                        v01 = (v01 / (1.0f + __expf(-v01))) * u01;
                        v10 = (v10 / (1.0f + __expf(-v10))) * u10;
                        v11 = (v11 / (1.0f + __expf(-v11))) * u11;
                    }
#endif
                    if (add_ptr) {
                        if (live_r0 && live_c0) v00 += add_ptr[abs0];
                        if (live_r0 && live_c1) v01 += add_ptr[abs0 + 1];
                        if (live_r1 && live_c0) v10 += add_ptr[abs0 + 8 * (size_t)n];
                        if (live_r1 && live_c1) v11 += add_ptr[abs0 + 8 * (size_t)n + 1];
                    }
                    const float2 lo = make_float2(v00, v01);
                    const float2 hi = make_float2(v10, v11);
#if EPI_N == 32 && EPI_SWZ

                    const int sw = (n >= (int)EPI_N && (n & 31) == 0) ? 4 * ec_row : 0;
                    *reinterpret_cast<float2*>(&ebuf[er0 * EPI_N + (col_local ^ sw)]) = lo;
                    *reinterpret_cast<float2*>(&ebuf[(er0 + 8) * EPI_N + (col_local ^ sw)]) = hi;
#else
                    *reinterpret_cast<float2*>(&ebuf[er0 * EPI_N + col_local]) = lo;
                    *reinterpret_cast<float2*>(&ebuf[(er0 + 8) * EPI_N + col_local]) = hi;
#endif
                }
            }

            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            epilog_ready_arrive();
        }
        continue;
    }

    const int rows_here = min(WARP_M_ROWS, m - m_warp);
    if (rows_here <= 0 || n - n_warp <= 0) {
        continue;
    }
    const int c_row = lane >> 2;
    const int c_col = (lane & 3) * 2;

    const int r0 = c_row;
    const int r1 = c_row + 8;
    const int c0 = c_col;
    const int c1 = c_col + 1;
    #pragma unroll
    for (int atom = 0; atom < M_ATOMS; ++atom) {
        const int m_atom = m_warp + atom * 16;
        const int rows_atom = min(16, m - m_atom);
        if (rows_atom <= 0) {
            continue;
        }
        float* c_tile = c + (size_t)m_atom * n + n_warp;
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        const int cols_here = min(8, n - (n_warp + ns * 8));
        if (cols_here <= 0) {
            continue;
        }

        const int n_loc = n_warp - n_base + ns * 8;
        const float t0 = __uint_as_float((uint32_t)sb_s[0][n_loc + c_col + 0] << 16);
        const float t1 = __uint_as_float((uint32_t)sb_s[0][n_loc + c_col + 1] << 16);
        float* cs = c_tile + ns * 8;
        const float* __restrict__ add_ptr = (const float*)(uintptr_t)add_addr;

        #define CLEF_FP8_STORE(off, val)                                                     \
            do {                                                                             \
                const size_t abs = (size_t)(cs - c) + (off);                                  \
                cs[(size_t)(off)] = add_ptr ? ((val) + add_ptr[abs]) : (val);                 \
            } while (0)
        if (r0 < rows_atom) {
            const float s_lo = __uint_as_float((uint32_t)sa_s[warp_m * WARP_M_ROWS + atom * 16 + r0] << 16);
            float v00 = d[0][atom][ns][0] * s_lo * t0;
            float v01 = d[0][atom][ns][1] * s_lo * t1;
#if GATED_GEMM_FP8
            {
                const float t0u = __uint_as_float((uint32_t)sb_s[1][n_loc + c_col + 0] << 16);
                const float t1u = __uint_as_float((uint32_t)sb_s[1][n_loc + c_col + 1] << 16);
                const float u00 = d[1][atom][ns][0] * s_lo * t0u;
                const float u01 = d[1][atom][ns][1] * s_lo * t1u;
                v00 = (v00 / (1.0f + __expf(-v00))) * u00;
                v01 = (v01 / (1.0f + __expf(-v01))) * u01;
            }
#endif
            if (c0 < cols_here) CLEF_FP8_STORE(r0 * n + c0, v00);
            if (c1 < cols_here) CLEF_FP8_STORE(r0 * n + c1, v01);
        }
        if (r1 < rows_atom) {
            const float s_hi = __uint_as_float((uint32_t)sa_s[warp_m * WARP_M_ROWS + atom * 16 + r1] << 16);
            float v10 = d[0][atom][ns][2] * s_hi * t0;
            float v11 = d[0][atom][ns][3] * s_hi * t1;
#if GATED_GEMM_FP8
            {
                const float t0u = __uint_as_float((uint32_t)sb_s[1][n_loc + c_col + 0] << 16);
                const float t1u = __uint_as_float((uint32_t)sb_s[1][n_loc + c_col + 1] << 16);
                const float u10 = d[1][atom][ns][2] * s_hi * t0u;
                const float u11 = d[1][atom][ns][3] * s_hi * t1u;
                v10 = (v10 / (1.0f + __expf(-v10))) * u10;
                v11 = (v11 / (1.0f + __expf(-v11))) * u11;
            }
#endif
            if (c0 < cols_here) CLEF_FP8_STORE(r1 * n + c0, v10);
            if (c1 < cols_here) CLEF_FP8_STORE(r1 * n + c1, v11);
        }
        #undef CLEF_FP8_STORE
    }
    }
    }
}
