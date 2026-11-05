#include <cuda_runtime.h>
#include <cstdint>

#define KCHUNK_BF16 16

#ifndef N_TILE
#define N_TILE 8
#endif
#define N_SUB (N_TILE / 8)
#define SUB_CHUNKS 4
#define KTILE_BF16 (KCHUNK_BF16 * SUB_CHUNKS)
#define A_BYTES_PER_ROW (KTILE_BF16 * 2)
#define B_BYTES_PER_ROW (KTILE_BF16 * 2)
#define A_INT4_PER_ROW (A_BYTES_PER_ROW / 16)
#define B_INT4_PER_ROW (B_BYTES_PER_ROW / 16)

#define WARPS_M 4
#ifndef WARPS_N
#define WARPS_N 2
#endif
#define BLOCK_THREADS (WARPS_M * WARPS_N * 32)
#define M_TILE (WARPS_M * 16)
#define N_BLOCK (WARPS_N * N_TILE)

extern "C" __global__ void bf16_gemm(const uint16_t* __restrict__ a_ptr,
                                     const uint16_t* __restrict__ b_ptr,
                                     float* __restrict__ c,
                                     int m, int n, int k,

                                     int a_rows,

                                     int n_splits,

                                     float* __restrict__ partials,
                                     int c_elems) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_m = warp % WARPS_M;
    const int warp_n = warp / WARPS_M;
    const int m_base = blockIdx.y * M_TILE;
    const int n_base = blockIdx.x * N_BLOCK;
    const int m_warp = m_base + warp_m * 16;
    const int n_warp = n_base + warp_n * N_TILE;

    __shared__ __align__(128) uint8_t a_smem[M_TILE * A_BYTES_PER_ROW];
    __shared__ __align__(128) uint8_t b_smem[N_BLOCK * B_BYTES_PER_ROW];

    const int k_row_bytes = k * 2;
    const uint8_t* a_bytes = reinterpret_cast<const uint8_t*>(a_ptr);
    const uint8_t* b_bytes = reinterpret_cast<const uint8_t*>(b_ptr);

    const int k_tiles = (k + KTILE_BF16 - 1) / KTILE_BF16;

    const int tiles_per_split = (k_tiles + n_splits - 1) / n_splits;
    const int tile_begin = blockIdx.z * tiles_per_split;
    const int tile_end = min(tile_begin + tiles_per_split, k_tiles);

    float d[N_SUB][4];
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        d[ns][0] = d[ns][1] = d[ns][2] = d[ns][3] = 0.0f;
    }
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int k0 = tile * KTILE_BF16;

        const int a_segs_here = (k_row_bytes - k0 * 2 + 15) >> 4;
        for (int i = threadIdx.x; i < M_TILE * A_INT4_PER_ROW; i += BLOCK_THREADS) {
            const int row = i / A_INT4_PER_ROW;
            const int seg = i % A_INT4_PER_ROW;
            uint8_t* dst = a_smem + (row * A_INT4_PER_ROW + (seg ^ (row % 8))) * 16;
            if (m_base + row < a_rows && seg < a_segs_here) {
                const uint8_t* src =
                    a_bytes + (size_t)(m_base + row) * k_row_bytes + k0 * 2 + seg * 16;
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }

        const int b_segs_here = (k_row_bytes - k0 * 2 + 15) >> 4;
        for (int i = threadIdx.x; i < N_BLOCK * B_INT4_PER_ROW; i += BLOCK_THREADS) {
            const int row = i / B_INT4_PER_ROW;
            const int seg = i % B_INT4_PER_ROW;
            uint8_t* dst = b_smem + (row * B_INT4_PER_ROW + (seg ^ (row % 8))) * 16;
            if (n_base + row < n && seg < b_segs_here) {
                const uint8_t* src =
                    b_bytes + (size_t)(n_base + row) * k_row_bytes + k0 * 2 + seg * 16;
                *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int ksub = 0; ksub < SUB_CHUNKS; ++ksub) {
            if (k0 + ksub * KCHUNK_BF16 >= k) {
                continue;
            }

            uint32_t a[4];
            {
                const int a_row = warp_m * 16 + (lane % 16);
                const int a_seg = (ksub * (KCHUNK_BF16 / 8) + (lane / 16)) ^ (a_row % 8);
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    a_smem + a_row * A_BYTES_PER_ROW + a_seg * 16);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                    : "r"(addr));
            }

            #pragma unroll
            for (int ns = 0; ns < N_SUB; ++ns) {
            uint32_t b[2];
            {
                const int b_row = warp_n * N_TILE + ns * 8 + (lane % 8);
                const int b_seg = (ksub * (KCHUNK_BF16 / 8) + ((lane / 8) & 1)) ^ (b_row % 8);
                const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
                    b_smem + b_row * B_BYTES_PER_ROW + b_seg * 16);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                    : "=r"(b[0]), "=r"(b[1])
                    : "r"(addr));
            }

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[ns][0]), "+f"(d[ns][1]), "+f"(d[ns][2]), "+f"(d[ns][3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
        }
        __syncthreads();
    }

    const int rows_here = min(16, m - m_warp);
    if (rows_here <= 0 || n - n_warp <= 0) {
        return;
    }
    float* c_tile = c + (size_t)m_warp * n + n_warp;
    const int c_row = lane >> 2;
    const int c_col = (lane & 3) * 2;
    const int r0 = c_row;
    const int r1 = c_row + 8;
    const int c0 = c_col;
    const int c1 = c_col + 1;
    #pragma unroll
    for (int ns = 0; ns < N_SUB; ++ns) {
        const int cols_here = min(8, n - (n_warp + ns * 8));
        if (cols_here <= 0) {
            continue;
        }
        float* cs = c_tile + ns * 8;

        #define CLEF_BF16_STORE(off, val)                                              \
            do {                                                                       \
                if (n_splits > 1)                                                      \
                    partials[(size_t)blockIdx.z * c_elems + (size_t)(cs - c) + (off)] = (val); \
                else cs[(size_t)(off)] = (val);                                        \
            } while (0)
        if (r0 < rows_here) {
            if (c0 < cols_here) CLEF_BF16_STORE(r0 * n + c0, d[ns][0]);
            if (c1 < cols_here) CLEF_BF16_STORE(r0 * n + c1, d[ns][1]);
        }
        if (r1 < rows_here) {
            if (c0 < cols_here) CLEF_BF16_STORE(r1 * n + c0, d[ns][2]);
            if (c1 < cols_here) CLEF_BF16_STORE(r1 * n + c1, d[ns][3]);
        }
        #undef CLEF_BF16_STORE
    }
}

extern "C" __global__ void bf16_split_reduce(const float* __restrict__ partials,
                                             float* __restrict__ out,
                                             int elems, int n_splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elems) {
        return;
    }
    float acc = 0.0f;
    for (int sp = 0; sp < n_splits; ++sp) {
        acc += partials[(size_t)sp * elems + i];
    }
    out[i] = acc;
}
