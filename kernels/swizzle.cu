#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void swizzle_blockscale(const uint8_t* __restrict__ src,
                                              uint8_t* __restrict__ dst,
                                              int rows,
                                              int cols,
                                              int rows_padded,
                                              int cols_padded,
                                              int conflict_free,
                                              int blk_mn,
                                              int chunk) {
    const int k1_count = cols_padded / 4;
    const long long total = (long long)rows_padded * cols_padded;
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int blk_sub = blk_mn / 32;

    const int k2 = (int)(index % 4);
    const int m_atom = (int)((index / 4) % blk_mn);
    const int k1 = (int)((index / chunk) % k1_count);
    const int m1 = (int)(index / ((long long)chunk * k1_count));

    int m;
    if (conflict_free) {
        m = m1 * blk_mn + m_atom;
    } else {
        const int m_within = m_atom / blk_sub;
        const int m_band = m_atom % blk_sub;
        m = m1 * blk_mn + m_band * 32 + m_within;
    }
    const int k = k1 * 4 + k2;
    if (m >= rows || k >= cols) {
        return;
    }
    dst[index] = src[(long long)m * cols + k];
}
