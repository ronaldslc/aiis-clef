#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

#include "nvfp4_quant.h"

#ifndef WARPS_M
#define WARPS_M 4
#endif
#ifndef WARPS_N
#define WARPS_N 2
#endif
#ifndef WARP_M_ROWS
#define WARP_M_ROWS 32
#endif
#ifndef N_TILE
#define N_TILE 64
#endif
#define M_TILE (WARPS_M * WARP_M_ROWS)

#define N_BLOCK (WARPS_N * N_TILE)

#define SUB_CHUNKS 2

#ifndef SF_BLK_MN
#define SF_BLK_MN ((M_TILE) >= 128 ? ((M_TILE) % 128 == 0 ? 128 : \
                                    ((M_TILE) % 96 == 0 ? 96 : \
                                     ((M_TILE) % 64 == 0 ? 64 : 32))) : 128)
#endif
#define SF_ROWS SF_BLK_MN

#define SF_ATOMS_M ((M_TILE / SF_BLK_MN) > 0 ? (M_TILE / SF_BLK_MN) : 1)

#define SF_CHUNK_STRIDE_A (SF_BLK_MN * 4)

#define SF_CHUNK_STRIDE_B 512

__device__ __forceinline__ int swizzled_scale(const uint8_t* s, int row, int block, int k) {
    const int k1_count = (k / 16) / 4;
    const int blk_sub = SF_BLK_MN / 32;
    const int m_rest = row / SF_BLK_MN;
    const int m_within = row % 32;
    const int m_band = (row / 32) % blk_sub;
    const int m_atom = m_within * blk_sub + m_band;
    const int k1 = block / 4;
    const int k2 = block % 4;
    return (m_rest * k1_count + k1) * SF_CHUNK_STRIDE_A + m_atom * 4 + k2;
}

__device__ __forceinline__ int swizzled_scale_base(int row, int k, int conflict_free) {
    const int k1_count = (k / 16) / 4;
    if (conflict_free) {
        return (row / SF_BLK_MN) * k1_count * SF_CHUNK_STRIDE_A + (row % SF_BLK_MN) * 4;
    }
    return (row / SF_BLK_MN) * k1_count * SF_CHUNK_STRIDE_A +
           (row % 32) * (SF_BLK_MN / 32) * 4 + ((row / 32) % (SF_BLK_MN / 32)) * 4;
}

#define SF_STAGE_BYTES_A (SUB_CHUNKS * SF_CHUNK_STRIDE_A)
#define SF_STAGE_BYTES_B (SUB_CHUNKS * SF_CHUNK_STRIDE_B)

#define SF_STAGE_BYTES_A_TOTAL (SF_ATOMS_M * SF_STAGE_BYTES_A)
#define SF_STAGE_BYTES (SF_STAGE_BYTES_A_TOTAL + SF_STAGE_BYTES_B)

__device__ __forceinline__ int sf_word_a(int row, int ks, int k, int conflict_free) {
    const int blk_sub = SF_BLK_MN / 32;
    const int m_atom = (row % 32) * blk_sub + ((row / 32) % blk_sub);
    const int atom = row / SF_BLK_MN;
    const int words_per_atom = (SF_CHUNK_STRIDE_A / 4) * SUB_CHUNKS;
    return atom * words_per_atom + ks * (SF_CHUNK_STRIDE_A / 4) + m_atom;
}

__device__ __forceinline__ int sf_word_b(int row, int ks, int k) {
    const int m_atom = (row % 32) * 4 + ((row / 32) % 4);
    return ks * (SF_CHUNK_STRIDE_B / 4) + m_atom;
}

#define KCHUNK 64

#define KTILE (KCHUNK * SUB_CHUNKS)
#define A_INT4_PER_ROW (KTILE / 32)
#define B_BYTES_PER_ROW (KTILE / 2)
#define B_INT4_PER_ROW (B_BYTES_PER_ROW / 16)

#ifndef WARPS_N
#define WARPS_N 2
#endif

#define NUM_MMA_WARPS (WARPS_M * WARPS_N)

#ifndef PRODUCER_WARPS
#define PRODUCER_WARPS 4
#endif
#define CTA_THREADS ((NUM_MMA_WARPS + PRODUCER_WARPS) * 32)

#ifndef SF_CONFLICT_FREE
#define SF_CONFLICT_FREE 1
#endif

#ifndef MMA_REG_MAX
#define MMA_REG_MAX 232
#endif
#ifndef PRODUCER_REG_MAX
#define PRODUCER_REG_MAX 40
#endif
#define USE_SETMAXNREG 1

#if USE_SETMAXNREG
#define SETMAXNREG_INC(n) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(n))
#define SETMAXNREG_DEC(n) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(n))
#else
#define SETMAXNREG_INC(n) do {} while (0)
#define SETMAXNREG_DEC(n) do {} while (0)
#endif
#define BLOCK_THREADS (WARPS_M * WARPS_N * 32)

#ifndef WARP_M_ROWS
#define WARP_M_ROWS 32
#endif
#define M_ATOMS (WARP_M_ROWS / 16)
static_assert(SF_STAGE_BYTES_A_TOTAL == SF_ROWS * SUB_CHUNKS * 4 * SF_ATOMS_M,
              "the staged scale region must be exactly SF_ROWS rows per k-chunk per atom for A");

#if GATED_GEMM
static_assert(SF_STAGE_BYTES_B == 128 * SUB_CHUNKS * 4,
              "the gated build stages the whole 128-row atom of B scales");
static_assert(N_BLOCK <= 128, "and no tile can be wider than the atom it indexes");
#else
static_assert(SF_STAGE_BYTES_B == N_BLOCK * SUB_CHUNKS * 4,
              "and N_BLOCK rows for B, whose atom is 128 in every variant");
#endif
static_assert(M_TILE <= SF_BLK_MN || M_TILE % SF_BLK_MN == 0,
              "the scale-factor atom must divide the CTA tile's M, or be taller than it");
static_assert(SF_BLK_MN == 128 || SF_BLK_MN == 96 || SF_BLK_MN == 64 || SF_BLK_MN == 32,
              "the atom is one of the four multiples of 32 that tile_atom_to_shape_SF_blk accepts");
static_assert(M_TILE % SF_BLK_MN == 0 || M_TILE <= SF_BLK_MN,
              "a forced SF_BLK_MN must divide the tile's M");

#define A_STAGE_BYTES ((M_TILE / 2) * 128)
#define B_STAGE_BYTES ((N_BLOCK / 2) * 128)

#define N_SUB (N_TILE / 8)

#ifndef STAGES
#define STAGES 3
#endif

#ifndef EPI_N
#define EPI_N 32
#endif
#ifndef EPI_BUFS
#define EPI_BUFS 2
#endif

__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    const uint32_t sm = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(sm), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async_4(void* smem_dst, const void* gmem_src) {
    const uint32_t sm = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                 "r"(sm), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void tma_load_2d(void* smem, const void* map, int inner, int outer,
                                            uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];\n" ::
        "r"(smem_u32(smem)), "l"((unsigned long long)(uintptr_t)map),
        "r"(inner), "r"(outer), "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void bulk_copy_1d(void* smem, const void* gmem, int bytes, uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::
        "r"(smem_u32(smem)), "l"((unsigned long long)(uintptr_t)gmem), "r"(bytes),
        "r"(smem_u32(bar)) : "memory");
}

#define LOG_SWIZZLE_DEFAULT 0

#ifndef GATED_GEMM
#define GATED_GEMM 0
#endif
#ifndef GATED_NO_EPI
#define GATED_NO_EPI 0
#endif

#ifndef GATED_NO_QUAD
#define GATED_NO_QUAD 0
#endif
#ifndef GATED_NO_SILU
#define GATED_NO_SILU 0
#endif

#ifndef GATED_NO_SCALE
#define GATED_NO_SCALE 0
#endif

#define MMA_BARRIER_ID 2
__device__ __forceinline__ void mma_barrier() {
    asm volatile("bar.sync %0, %1;\n" :: "n"(MMA_BARRIER_ID), "n"(NUM_MMA_WARPS * 32) : "memory");
}

#define STORE_WARP_ID (NUM_MMA_WARPS + 1)
#define EPILOG_READY_ID 3
#define EPILOG_THREADS ((NUM_MMA_WARPS + 1) * 32)

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

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::
                 "r"(smem_u32(bar)), "r"(bytes));
}

__device__ __forceinline__ void tma_store_2d(const void* smem, const void* map, int inner, int outer) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n" ::
        "l"((unsigned long long)(uintptr_t)map), "r"(inner), "r"(outer),
        "r"(smem_u32(smem)) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar)));
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n"
        "CLEF_MB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra CLEF_MB_WAIT;\n}\n" :: "r"(smem_u32(bar)), "r"(phase));
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::
                 "n"(N));
}

extern "C" __global__ void __launch_bounds__(CTA_THREADS, 1) nvfp4_gemm(const __grid_constant__ CUtensorMap map_a,
                                      const __grid_constant__ CUtensorMap map_b,
                                      const __grid_constant__ CUtensorMap map_c,
                                      const uint8_t* __restrict__ a_ptr,
                                      const uint8_t* __restrict__ b_ptr,
                                      const uint8_t* __restrict__ sa,
                                      const uint8_t* __restrict__ sb,
                                      float* __restrict__ c,
                                      int m, int n, int k,
                                      float global_a, float global_b,

                                      int a_rows,

                                      int n_splits,

                                      int log_swizzle) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    int sf_row_a[M_ATOMS];
    #pragma unroll
    for (int ma = 0; ma < M_ATOMS; ++ma) {
        const int a_srow = warp_m * WARP_M_ROWS + ma * 16 + 8 * ((lane % 4) % 2) + lane / 4;
        const int blk_sub = SF_BLK_MN / 32;

        const int m_atom = SF_CONFLICT_FREE ? (a_srow % SF_BLK_MN)
                                            : (a_srow % 32) * blk_sub + ((a_srow / 32) % blk_sub);
        sf_row_a[ma] = (a_srow / SF_BLK_MN) * ((SF_CHUNK_STRIDE_A / 4) * SUB_CHUNKS) + m_atom;
    }
    int sf_row_b[N_SUB];
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        const int b_srow = warp_n * N_TILE + ns * 8 + (lane / 4);
        sf_row_b[ns] = (b_srow % 32) * 4 + ((b_srow / 32) % 4);
    }

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

    __shared__ __align__(128) uint8_t a_smem[STAGES][A_STAGE_BYTES];
    __shared__ __align__(128) uint8_t b_smem[STAGES][B_STAGE_BYTES];

    __shared__ __align__(8) uint64_t full[STAGES];
    __shared__ __align__(8) uint64_t empty[STAGES];

    __shared__ uint32_t sfa_smem[STAGES][SF_ROWS * SUB_CHUNKS * SF_ATOMS_M];

    __shared__ __align__(1024) float epi_smem[EPI_BUFS][M_TILE * EPI_N];

    __shared__ uint32_t sfb_smem[STAGES][N_BLOCK * SUB_CHUNKS];

    const int k_bytes = k >> 1;

    const int k_tiles = (k + KTILE - 1) / KTILE;

    const int tiles_per_split = (k_tiles + n_splits - 1) / n_splits;
    const int tile_begin = blockIdx.z * tiles_per_split;
    const int tile_end = min(tile_begin + tiles_per_split, k_tiles);
    const int b_segs = B_BYTES_PER_ROW / 16;

    float acc[M_ATOMS][N_SUB][4];

    const int sf_rows = ((a_rows + SF_BLK_MN - 1) / SF_BLK_MN) * SF_BLK_MN;

    const bool is_load_warp = (warp == NUM_MMA_WARPS);
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
        if (warp == STORE_WARP_ID && (n % 4) == 0) {
            for (int wi = 0; wi < my_work; ++wi) {
                const int w = (int)blockIdx.x + wi * (int)gridDim.x;
                int m_base, n_base;
                work_base(w, &m_base, &n_base);
                #pragma unroll
                for (int half = 0; half < N_BLOCK / EPI_N; ++half) {
                    const int buf = half & (EPI_BUFS - 1);
                    const int n_off = half * EPI_N;

                    asm volatile("cp.async.bulk.wait_group.read 1;\n");
                    epilog_free_arrive(buf);
                    epilog_ready_sync();
                    if (lane == 0) {
                        tma_store_2d(epi_smem[buf], &map_c, (n_base + n_off) * 4, m_base);
                        asm volatile("cp.async.bulk.commit_group;\n");
                    }
                }
            }

            asm volatile("cp.async.bulk.wait_group.read 0;\n");
            return;
        }
        if (is_load_warp) {
            auto issue_tile = [&](int flat, int kt, int m_base, int n_base) {
                const int st = flat % STAGES;
                if (lane == 0) {
                    mbar_arrive_expect_tx(&full[st], (uint32_t)(A_STAGE_BYTES + B_STAGE_BYTES
                                                               + SF_STAGE_BYTES_A_TOTAL + SF_STAGE_BYTES_B));
                    tma_load_2d(a_smem[st], &map_a, (kt * KTILE) >> 1, m_base, &full[st]);
                    tma_load_2d(b_smem[st], &map_b, (kt * KTILE) >> 1, n_base, &full[st]);

                    const int k1_count = (k / 16) / 4;

                    #pragma unroll
                    for (int atom = 0; atom < SF_ATOMS_M; ++atom) {
                        bulk_copy_1d(sfa_smem[st] + atom * (SF_STAGE_BYTES_A / 4),
                                     sa + (size_t)(m_base / SF_BLK_MN + atom) * k1_count * SF_CHUNK_STRIDE_A
                                        + (size_t)kt * SUB_CHUNKS * SF_CHUNK_STRIDE_A,
                                     SF_STAGE_BYTES_A, &full[st]);
                    }
                    bulk_copy_1d(sfb_smem[st],
                                 sb + (size_t)(n_base / 128) * k1_count * SF_CHUNK_STRIDE_B
                                    + (size_t)kt * SUB_CHUNKS * SF_CHUNK_STRIDE_B,
                                 SF_STAGE_BYTES_B, &full[st]);
                }
            };

            int flat = 0;
            for (int wi = 0; wi < my_work; ++wi) {
                int m_base, n_base;
                work_base((int)blockIdx.x + wi * (int)gridDim.x, &m_base, &n_base);
                for (int kt = 0; kt < k_tiles; ++kt, ++flat) {
                    if (flat >= STAGES) {
                        mbar_wait(&empty[flat % STAGES], ((flat - STAGES) / STAGES) & 1);
                    }
                    issue_tile(flat, kt, m_base, n_base);
                }
            }
        }
        return;
    }
    SETMAXNREG_INC(MMA_REG_MAX);

    int g = 0;
    for (int wi = 0; wi < my_work; ++wi) {
        const int w = (int)blockIdx.x + wi * (int)gridDim.x;
        int m_base, n_base;
        work_base(w, &m_base, &n_base);
        const int m_warp = m_base + warp_m * WARP_M_ROWS;
        const int n_warp = n_base + warp_n * N_TILE;

        #pragma unroll
        for (int ma = 0; ma < M_ATOMS; ++ma) {
            #pragma unroll
            for (int s = 0; s < N_SUB; ++s) {
                acc[ma][s][0] = acc[ma][s][1] = acc[ma][s][2] = acc[ma][s][3] = 0.0f;
            }
        }
    for (int kt = 0; kt < k_tiles; ++kt, ++g) {
        const int k0 = kt * KTILE;
        const int k0_bytes = k0 >> 1;
        const int stage = g % STAGES;
        mbar_wait(&full[stage], (g / STAGES) & 1);

        uint32_t af[2][M_ATOMS][4];
        uint32_t bf[2][N_SUB][2];

        uint32_t saf[2][M_ATOMS];
        uint32_t sbf[2][N_SUB];

        auto load_ksub = [&](int ks, int buf) {
            const int a_seg = ks * (KCHUNK / 32) + (lane / 16);
            #pragma unroll
            for (int ma = 0; ma < M_ATOMS; ++ma) {
                const int a_row = warp_m * WARP_M_ROWS + ma * 16 + (lane % 16);
                const int a_scol = a_seg ^ ((a_row / 2) % 4);
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    a_smem[stage] + (a_row * 4 + a_scol) * 16);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(af[buf][ma][0]), "=r"(af[buf][ma][1]),
                      "=r"(af[buf][ma][2]), "=r"(af[buf][ma][3])
                    : "r"(addr));
            }
            #pragma unroll
            for (int np = 0; np < N_SUB; np += 2) {
                uint32_t b4[4];
                {
                    const int b_row = warp_n * N_TILE + (np + (lane / 16)) * 8 + (lane % 8);
                    const int b_seg = ks * (KCHUNK / 32) + ((lane / 8) & 1);
                    const int b_scol = b_seg ^ ((b_row / 2) % 4);
                    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        b_smem[stage] + (b_row * 4 + b_scol) * 16);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b4[0]), "=r"(b4[1]), "=r"(b4[2]), "=r"(b4[3])
                        : "r"(addr));
                }
                bf[buf][np][0] = b4[0];
                bf[buf][np][1] = b4[1];
                bf[buf][np + 1][0] = b4[2];
                bf[buf][np + 1][1] = b4[3];
            }
            #pragma unroll
            for (int ma = 0; ma < M_ATOMS; ++ma) {
                saf[buf][ma] = sfa_smem[stage][sf_row_a[ma] + ks * (SF_CHUNK_STRIDE_A / 4)];
            }
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                sbf[buf][ns] = sfb_smem[stage][sf_row_b[ns] + ks * (SF_CHUNK_STRIDE_B / 4)];
            }
        };
        if (k0 < k) {
            load_ksub(0, 0);
        }
        #pragma unroll
        for (int ksub = 0; ksub < SUB_CHUNKS; ++ksub) {
            if (k0 + ksub * KCHUNK >= k) {
                continue;
            }
            const int cur = ksub & 1;
            if (ksub + 1 < SUB_CHUNKS) {
                load_ksub(ksub + 1, (ksub + 1) & 1);
            }
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                const uint32_t b0 = bf[cur][ns][0], b1 = bf[cur][ns][1];
                const uint32_t scale_b = sbf[cur][ns];
                const unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                #pragma unroll
                for (int ma = 0; ma < M_ATOMS; ++ma) {
                    const uint32_t scale_a = saf[cur][ma];

                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
                        "f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                        "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
                        : "+f"(acc[ma][ns][0]), "+f"(acc[ma][ns][1]),
                          "+f"(acc[ma][ns][2]), "+f"(acc[ma][ns][3])
                        : "r"(af[cur][ma][0]), "r"(af[cur][ma][1]),
                          "r"(af[cur][ma][2]), "r"(af[cur][ma][3]),
                          "r"(b0), "r"(b1),
                          "r"(scale_a), "h"(bidA), "h"(tidA),
                          "r"(scale_b), "h"(bidB), "h"(tidB));
                }
            }
        }
        if (lane == 0) {
            mbar_arrive(&empty[stage]);
        }
    }

    const float gscale = 1.0f / (global_a * global_b);
    const int c_row = lane >> 2;
    const int c_col = (lane & 3) * 2;
    if ((n % 4) != 0) {
        #pragma unroll
        for (int ma = 0; ma < M_ATOMS; ++ma) {
            const int r_lo = m_warp + ma * 16;
            const int rows_here = min(16, m - r_lo);
            if (rows_here <= 0 || n - n_warp <= 0) {
                continue;
            }
            float* c_tile = c + (size_t)r_lo * n + n_warp;
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                const int cols_here = min(8, n - (n_warp + ns * 8));
                if (cols_here <= 0) {
                    continue;
                }
                float* cs = c_tile + ns * 8;
                const int r0 = c_row;
                const int r1 = c_row + 8;
                const int c0 = c_col;
                const int c1 = c_col + 1;
                #define CLEF_NVFP4_STORE(off, val)                                             \
                    do {                                                                       \
                        if (n_splits > 1) atomicAdd(&cs[(size_t)(off)], (val));                 \
                        else cs[(size_t)(off)] = (val);                                        \
                    } while (0)
                if (r0 < rows_here) {
                    if (c0 < cols_here) CLEF_NVFP4_STORE(r0 * n + c0, acc[ma][ns][0] * gscale);
                    if (c1 < cols_here) CLEF_NVFP4_STORE(r0 * n + c1, acc[ma][ns][1] * gscale);
                }
                if (r1 < rows_here) {
                    if (c0 < cols_here) CLEF_NVFP4_STORE(r1 * n + c0, acc[ma][ns][2] * gscale);
                    if (c1 < cols_here) CLEF_NVFP4_STORE(r1 * n + c1, acc[ma][ns][3] * gscale);
                }
                #undef CLEF_NVFP4_STORE
            }
        }
        return;
    }
    #pragma unroll
    for (int half = 0; half < N_BLOCK / EPI_N; ++half) {
        const int n_off = half * EPI_N;
        const int buf = half & (EPI_BUFS - 1);
        epilog_free_sync(buf);
        #pragma unroll
        for (int ma = 0; ma < M_ATOMS; ++ma) {
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                const int col_local = warp_n * N_TILE + ns * 8 + c_col - n_off;
                if (col_local < 0 || col_local + 1 >= EPI_N) {
                    continue;
                }
                const int r0 = warp_m * WARP_M_ROWS + ma * 16 + c_row;

                const int sw = 4 * c_row;
                epi_smem[buf][r0 * EPI_N + (col_local ^ sw)]             = acc[ma][ns][0] * gscale;
                epi_smem[buf][r0 * EPI_N + ((col_local + 1) ^ sw)]       = acc[ma][ns][1] * gscale;
                epi_smem[buf][(r0 + 8) * EPI_N + (col_local ^ sw)]       = acc[ma][ns][2] * gscale;
                epi_smem[buf][(r0 + 8) * EPI_N + ((col_local + 1) ^ sw)] = acc[ma][ns][3] * gscale;
            }
        }

        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        epilog_ready_arrive();
    }
    }
}

#if GATED_GEMM

extern "C" __global__ void __launch_bounds__(CTA_THREADS, 1) nvfp4_gemm_gated(
                                      const __grid_constant__ CUtensorMap map_a,
                                      const __grid_constant__ CUtensorMap map_bg,
                                      const __grid_constant__ CUtensorMap map_bu,
                                      const uint8_t* __restrict__ a_ptr,
                                      const uint8_t* __restrict__ bg_ptr,
                                      const uint8_t* __restrict__ bu_ptr,
                                      const uint8_t* __restrict__ sa,
                                      const uint8_t* __restrict__ sbg,
                                      const uint8_t* __restrict__ sbu,
                                      uint8_t* __restrict__ out_packed,
                                      uint8_t* __restrict__ out_scales,
                                      int m, int n, int k,
                                      float global_a, float global_bg, float global_bu,
                                      int a_rows,
                                      int log_swizzle,

                                      float out_global) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;

    __shared__ __align__(128) uint8_t a_smem[STAGES][A_STAGE_BYTES];
    __shared__ __align__(1024) uint8_t b_smem[2][STAGES][B_STAGE_BYTES];
    __shared__ __align__(8) uint64_t full[STAGES];
    __shared__ __align__(8) uint64_t empty[STAGES];
    __shared__ uint32_t sfa_smem[STAGES][SF_ROWS * SUB_CHUNKS * SF_ATOMS_M];

    __shared__ uint32_t sfb_smem[2][STAGES][SF_STAGE_BYTES_B / 4];

    __shared__ __align__(16) uint8_t epi_packed[M_TILE * (N_BLOCK / 2)];

    const int k_bytes = k >> 1;
    const int k_tiles = (k + KTILE - 1) / KTILE;
    const int b_segs = B_BYTES_PER_ROW / 16;
    const int k1_count = (k / 16) / 4;

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

    float accg[M_ATOMS][N_SUB][4];
    float accu[M_ATOMS][N_SUB][4];
    int sf_row_a[M_ATOMS];

    const bool is_load_warp = (warp == NUM_MMA_WARPS);
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
        if (is_load_warp) {
            auto issue_tile = [&](int flat, int kt, int m_base, int n_base) {
                const int st = flat % STAGES;
                if (lane == 0) {
                    mbar_arrive_expect_tx(&full[st], (uint32_t)(A_STAGE_BYTES + 2 * B_STAGE_BYTES
                                                               + SF_STAGE_BYTES_A_TOTAL
                                                               + 2 * SF_STAGE_BYTES_B));
                    tma_load_2d(a_smem[st], &map_a, (kt * KTILE) >> 1, m_base, &full[st]);
                    tma_load_2d(b_smem[0][st], &map_bg, (kt * KTILE) >> 1, n_base, &full[st]);
                    tma_load_2d(b_smem[1][st], &map_bu, (kt * KTILE) >> 1, n_base, &full[st]);
                    #pragma unroll
                    for (int atom = 0; atom < SF_ATOMS_M; ++atom) {
                        bulk_copy_1d(sfa_smem[st] + atom * (SF_STAGE_BYTES_A / 4),
                                     sa + (size_t)(m_base / SF_BLK_MN + atom) * k1_count * SF_CHUNK_STRIDE_A
                                        + (size_t)kt * SUB_CHUNKS * SF_CHUNK_STRIDE_A,
                                     SF_STAGE_BYTES_A, &full[st]);
                    }
                    bulk_copy_1d(sfb_smem[0][st],
                                 sbg + (size_t)(n_base / 128) * k1_count * SF_CHUNK_STRIDE_B
                                     + (size_t)kt * SUB_CHUNKS * SF_CHUNK_STRIDE_B,
                                 SF_STAGE_BYTES_B, &full[st]);
                    bulk_copy_1d(sfb_smem[1][st],
                                 sbu + (size_t)(n_base / 128) * k1_count * SF_CHUNK_STRIDE_B
                                     + (size_t)kt * SUB_CHUNKS * SF_CHUNK_STRIDE_B,
                                 SF_STAGE_BYTES_B, &full[st]);
                }
            };
            int flat = 0;
            for (int wi = 0; wi < my_work; ++wi) {
                int m_base, n_base;
                work_base((int)blockIdx.x + wi * (int)gridDim.x, &m_base, &n_base);
                for (int kt = 0; kt < k_tiles; ++kt, ++flat) {
                    if (flat >= STAGES) {
                        mbar_wait(&empty[flat % STAGES], ((flat - STAGES) / STAGES) & 1);
                    }
                    issue_tile(flat, kt, m_base, n_base);
                }
            }
        }
        return;
    }

    const int c_row = lane >> 2;
    const int c_col = (lane & 3) * 2;
    static_assert(M_TILE <= SF_BLK_MN, "this variant stages exactly one atom of A scales");
    int sf_row_a_local[M_ATOMS];
    #pragma unroll
    for (int ma = 0; ma < M_ATOMS; ++ma) {
        sf_row_a_local[ma] = warp_m * WARP_M_ROWS + ma * 16 + 8 * ((lane % 4) % 2) + lane / 4;
    }

    int sf_row_b_local[N_SUB];
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        sf_row_b_local[ns] = warp_n * N_TILE + ns * 8 + (lane / 4);
    }
    int sf_row_b[N_SUB];

    const float gscale_g = 1.0f / (global_a * global_bg);
    const float gscale_u = 1.0f / (global_a * global_bu);
    const int blocks_per_row = n / 16;
    const int out_k1_count = blocks_per_row / 4;

    int g = 0;
    for (int wi = 0; wi < my_work; ++wi) {
        const int w = (int)blockIdx.x + wi * (int)gridDim.x;
        int m_base, n_base;
        work_base(w, &m_base, &n_base);
        #pragma unroll
        for (int ma = 0; ma < M_ATOMS; ++ma) {
            sf_row_a[ma] = (m_base % SF_BLK_MN) + sf_row_a_local[ma];
            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
                const int b = (n_base % 128) + sf_row_b_local[ns];
                sf_row_b[ns] = (b % 32) * 4 + ((b / 32) % 4);
            }
            #pragma unroll
            for (int s = 0; s < N_SUB; ++s) {
                accg[ma][s][0] = accg[ma][s][1] = accg[ma][s][2] = accg[ma][s][3] = 0.0f;
                accu[ma][s][0] = accu[ma][s][1] = accu[ma][s][2] = accu[ma][s][3] = 0.0f;
            }
        }
        for (int kt = 0; kt < k_tiles; ++kt, ++g) {
            const int k0 = kt * KTILE;
            const int stage = g % STAGES;
            mbar_wait(&full[stage], (g / STAGES) & 1);

            uint32_t af[2][M_ATOMS][4];
            uint32_t bfg[2][N_SUB][2], bfu[2][N_SUB][2];
            uint32_t saf[2][M_ATOMS], sbfg[2][N_SUB], sbfu[2][N_SUB];

            auto load_ksub = [&](int ks, int buf) {
                const int a_seg = ks * (KCHUNK / 32) + (lane / 16);
                #pragma unroll
                for (int ma = 0; ma < M_ATOMS; ++ma) {
                    const int a_row = warp_m * WARP_M_ROWS + ma * 16 + (lane % 16);
                    const int a_scol = a_seg ^ ((a_row / 2) % 4);
                    const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                        a_smem[stage] + (a_row * 4 + a_scol) * 16);
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(af[buf][ma][0]), "=r"(af[buf][ma][1]),
                          "=r"(af[buf][ma][2]), "=r"(af[buf][ma][3])
                        : "r"(addr));
                }
                #pragma unroll
                for (int np = 0; np < N_SUB; np += 2) {
                    #pragma unroll
                    for (int mat = 0; mat < 2; ++mat) {
                        const int b_row = warp_n * N_TILE + (np + (lane / 16)) * 8 + (lane % 8);
                        const int b_seg = ks * (KCHUNK / 32) + ((lane / 8) & 1);
                        const int b_scol = b_seg ^ ((b_row / 2) % 4);
                        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                            b_smem[mat][stage] + (b_row * 4 + b_scol) * 16);
                        uint32_t b4[4];
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b4[0]), "=r"(b4[1]), "=r"(b4[2]), "=r"(b4[3])
                            : "r"(addr));
                        if (mat == 0) {
                            bfg[buf][np][0] = b4[0]; bfg[buf][np][1] = b4[1];
                            bfg[buf][np + 1][0] = b4[2]; bfg[buf][np + 1][1] = b4[3];
                        } else {
                            bfu[buf][np][0] = b4[0]; bfu[buf][np][1] = b4[1];
                            bfu[buf][np + 1][0] = b4[2]; bfu[buf][np + 1][1] = b4[3];
                        }
                    }
                }
                #pragma unroll
                for (int ma = 0; ma < M_ATOMS; ++ma) {
                    saf[buf][ma] = sfa_smem[stage][sf_row_a[ma] + ks * (SF_CHUNK_STRIDE_A / 4)];
                }
                #pragma unroll
                for (int ns = 0; ns < N_SUB; ++ns) {
                    sbfg[buf][ns] = sfb_smem[0][stage][sf_row_b[ns] + ks * (SF_CHUNK_STRIDE_B / 4)];
                    sbfu[buf][ns] = sfb_smem[1][stage][sf_row_b[ns] + ks * (SF_CHUNK_STRIDE_B / 4)];
                }
            };
            if (k0 < k) {
                load_ksub(0, 0);
            }
            #pragma unroll
            for (int ksub = 0; ksub < SUB_CHUNKS; ++ksub) {
                if (k0 + ksub * KCHUNK >= k) {
                    continue;
                }
                const int cur = ksub & 1;
                if (ksub + 1 < SUB_CHUNKS) {
                    load_ksub(ksub + 1, (ksub + 1) & 1);
                }
                #pragma unroll
                for (int ns = 0; ns < N_SUB; ++ns) {
                    #pragma unroll
                    for (int ma = 0; ma < M_ATOMS; ++ma) {
                        const unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                        const uint32_t scale_a = saf[cur][ma];
                        asm volatile(
                            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
                            "f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                            "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accg[ma][ns][0]), "+f"(accg[ma][ns][1]),
                              "+f"(accg[ma][ns][2]), "+f"(accg[ma][ns][3])
                            : "r"(af[cur][ma][0]), "r"(af[cur][ma][1]),
                              "r"(af[cur][ma][2]), "r"(af[cur][ma][3]),
                              "r"(bfg[cur][ns][0]), "r"(bfg[cur][ns][1]),
                              "r"(scale_a), "h"(bidA), "h"(tidA),
                              "r"(sbfg[cur][ns]), "h"(bidB), "h"(tidB));
                        asm volatile(
                            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col."
                            "f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                            "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
                            : "+f"(accu[ma][ns][0]), "+f"(accu[ma][ns][1]),
                              "+f"(accu[ma][ns][2]), "+f"(accu[ma][ns][3])
                            : "r"(af[cur][ma][0]), "r"(af[cur][ma][1]),
                              "r"(af[cur][ma][2]), "r"(af[cur][ma][3]),
                              "r"(bfu[cur][ns][0]), "r"(bfu[cur][ns][1]),
                              "r"(scale_a), "h"(bidA), "h"(tidA),
                              "r"(sbfu[cur][ns]), "h"(bidB), "h"(tidB));
                    }
                }
            }
            if (lane == 0) {
                mbar_arrive(&empty[stage]);
            }
        }

#if GATED_NO_EPI
        if (m_base < 0) {
            out_packed[0] = (uint8_t)(accg[0][0][0] + accu[0][0][0]);
        }
#else

        #pragma unroll
        for (int half = 0; half < N_BLOCK / EPI_N; ++half) {
            const int n_off = half * EPI_N;

            bool live_blk[EPI_N / 16];
            #pragma unroll
            for (int b = 0; b < EPI_N / 16; ++b) {
                const int ns0 = (half % (N_TILE / EPI_N)) * (EPI_N / 8) + b * 2;
                const int c0 = warp_n * N_TILE + ns0 * 8 + c_col - half * EPI_N;
                live_blk[b] = (c0 >= 0 && c0 + 1 < EPI_N);
            }

            #pragma unroll
            for (int blk = 0; blk < EPI_N / 16; ++blk) {
                float vv[M_ATOMS][2][4];
                #pragma unroll
                for (int ma = 0; ma < M_ATOMS; ++ma) {
                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        float m0 = 0.0f;
                        #pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            const int ns = (half % (N_TILE / EPI_N)) * (EPI_N / 8) + blk * 2 + s;
                            const int src = r * 2;
                            float v0 = 0.0f, v1 = 0.0f;
                            if (live_blk[blk]) {
                                const float gg = accg[ma][ns][src + 0] * gscale_g;
                                const float gu = accu[ma][ns][src + 0] * gscale_u;
#if GATED_NO_SILU
                                v0 = gg * gu;
#else
                                v0 = (gg / (1.0f + __expf(-gg))) * gu;
#endif
                                const float g1 = accg[ma][ns][src + 1] * gscale_g;
                                const float u1 = accu[ma][ns][src + 1] * gscale_u;
#if GATED_NO_SILU
                                v1 = g1 * u1;
#else
                                v1 = (g1 / (1.0f + __expf(-g1))) * u1;
#endif
                                m0 = fmaxf(m0, fmaxf(fabsf(v0), fabsf(v1)));
                            }
                            vv[ma][r][s * 2 + 0] = v0;
                            vv[ma][r][s * 2 + 1] = v1;
                        }

#if !GATED_NO_QUAD
                        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1));
                        m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2));
#endif
                        const int row = m_base + warp_m * WARP_M_ROWS + ma * 16 + c_row + r * 8;
                        const int block_global = (n_base + n_off + blk * 16) / 16;

#if GATED_NO_SCALE
                        const uint8_t stored = 0x38;
#else
                        const uint8_t stored = (row < a_rows) ? to_e4m3((m0 / 6.0f) * out_global)
                                                             : (uint8_t)0;
#endif
                        if ((lane & 3) == 0 && live_blk[blk]) {
#if GATED_NO_SCALE

#endif

                            const int m_rest = row / SF_BLK_MN;
                            const int k1 = block_global / 4;
                            const int k2 = block_global % 4;
                            out_scales[(size_t)(m_rest * out_k1_count + k1) * SF_CHUNK_STRIDE_A
                                       + (size_t)(row % SF_BLK_MN) * 4 + k2] = stored;
                        }
                        const float stored_f = from_e4m3(stored);
                        const float inv_scale = (stored_f > 0.0f) ? out_global / stored_f : 0.0f;
                        if (row < a_rows && live_blk[blk]) {
                            #pragma unroll
                            for (int s = 0; s < 2; ++s) {
                                const uint8_t lo = to_e2m1(vv[ma][r][s * 2 + 0] * inv_scale);
                                const uint8_t hi = to_e2m1(vv[ma][r][s * 2 + 1] * inv_scale);
                                epi_packed[(warp_m * WARP_M_ROWS + ma * 16 + c_row + r * 8)
                                           * (N_BLOCK / 2) + (n_off / 2) + blk * 8 + s * 4
                                           + (lane & 3)] = (uint8_t)(lo | (hi << 4));
                            }
                        }
                    }
                }
            }
        }
        mma_barrier();
        if (threadIdx.x < (unsigned)(2 * M_TILE)) {
            const int r = threadIdx.x >> 1;
            const int seg = threadIdx.x & 1;
            const int row = m_base + r;
            if (row < a_rows) {
                const uint4 v = *reinterpret_cast<const uint4*>(
                    &epi_packed[r * (N_BLOCK / 2) + seg * 16]);
                *reinterpret_cast<uint4*>(out_packed + (size_t)row * (n / 2)
                                          + (size_t)n_base / 2 + seg * 16) = v;
            }
        }
        mma_barrier();
#endif
    }
}
#endif
