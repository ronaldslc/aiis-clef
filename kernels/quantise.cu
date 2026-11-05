#include <cuda_runtime.h>
#include <cstdint>

#include "nvfp4_quant.h"

extern "C" __global__ void quantise_nvfp4_kernel(const float* __restrict__ x,
                                                 uint8_t* __restrict__ packed,
                                                 uint8_t* __restrict__ scales,
                                                 int rows, int cols, float global_scale,
                                                 int blk_mn, int chunk) {
    const int row = blockIdx.y + blockIdx.z * gridDim.y;
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = cols / 16;
    if (block >= blocks_per_row) {
        return;
    }

    const float* src = x + (size_t)row * cols + block * 16;

    const bool live = row < rows;

    float amax = 0.0f;
    if (live) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(src[i]));
        }
    }

    const float block_scale = amax / 6.0f;

    const uint8_t stored = live ? to_e4m3(block_scale * global_scale) : (uint8_t)0;

    {
        const int k1_count = blocks_per_row / 4;
        const int m_rest = row / blk_mn;
        const int k1 = block / 4;
        const int k2 = block % 4;
        const int m_atom = row % blk_mn;
        scales[(m_rest * k1_count + k1) * chunk + m_atom * 4 + k2] = stored;
    }
    if (live) {
        const float stored_f = from_e4m3(stored);
        const float inv_scale = (stored_f > 0.0f) ? global_scale / stored_f : 0.0f;

        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const uint8_t lo = to_e2m1(src[pair * 2] * inv_scale);
            const uint8_t hi = to_e2m1(src[pair * 2 + 1] * inv_scale);
            packed[(size_t)row * (cols / 2) + block * 8 + pair] = (uint8_t)(lo | (hi << 4));
        }
    }
}

#define RNG_ROWS_PER_BLOCK 8

extern "C" __global__ void rms_norm_gated_quantise_nvfp4_kernel(
        const float* __restrict__ x, const float* __restrict__ gate,
        const float* __restrict__ weight,
        uint8_t* __restrict__ packed, uint8_t* __restrict__ scales,
        int rows, int cols, float eps, float global_scale,
        int blk_mn, int chunk, int vh, int rows_live) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * RNG_ROWS_PER_BLOCK + warp;
    if (row >= rows) {
        return;
    }
    const bool live = row < rows_live;

    const float4* x4 = reinterpret_cast<const float4*>(x + (size_t)row * cols);
    const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)row * cols);
    const float4* w4 = reinterpret_cast<const float4*>(weight);

    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
    float acc = 0.0f;
    if (live) {
        xv = x4[lane];
        acc = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const float inv = live ? 1.0f / sqrtf(__shfl_sync(0xffffffffu, acc, 0) / (float)cols + eps) : 0.0f;

    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) {
        const float4 gv = g4[lane], wv = w4[lane];

        v.x = xv.x * inv * wv.x * __fdividef(gv.x, 1.0f + __expf(-gv.x));
        v.y = xv.y * inv * wv.y * __fdividef(gv.y, 1.0f + __expf(-gv.y));
        v.z = xv.z * inv * wv.z * __fdividef(gv.z, 1.0f + __expf(-gv.z));
        v.w = xv.w * inv * wv.w * __fdividef(gv.w, 1.0f + __expf(-gv.w));
    }

    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));

    const uint8_t stored = live ? to_e4m3((amax / 6.0f) * global_scale) : (uint8_t)0;

    const int blocks_per_row = (vh * cols) / 16;
    const int k1_count = blocks_per_row / 4;
    const int r_flat = row / vh;
    const int block = (row - r_flat * vh) * (cols / 16) + (lane / 4);
    if ((lane & 3) == 0) {
        const int m_rest = r_flat / blk_mn;
        const int k1 = block / 4;
        const int k2 = block - k1 * 4;
        scales[(size_t)(m_rest * k1_count + k1) * chunk + (size_t)(r_flat % blk_mn) * 4 + k2] = stored;
    }
    if (!live) {
        return;
    }
    const float stored_f = from_e4m3(stored);
    const float inv_scale = (stored_f > 0.0f) ? global_scale / stored_f : 0.0f;
    uint8_t* dst = packed + (size_t)row * (cols / 2) + (lane * 4) / 2;
    dst[0] = (uint8_t)(to_e2m1(v.x * inv_scale) | (to_e2m1(v.y * inv_scale) << 4));
    dst[1] = (uint8_t)(to_e2m1(v.z * inv_scale) | (to_e2m1(v.w * inv_scale) << 4));
}

extern "C" __global__ void __launch_bounds__(1024, 1) rms_norm_gated_quantise_fp8_kernel(
        const float* __restrict__ x, const float* __restrict__ gate,
        const float* __restrict__ weight,
        uint8_t* __restrict__ out, uint16_t* __restrict__ scales,
        int rows, int cols, float eps, int vh, int rows_live) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r_flat = blockIdx.x;
    const int row = r_flat * vh + warp;
    const bool live = row < rows_live;
    __shared__ float s_warp[32];

    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f);
    float acc = 0.0f;
    if (live) {
        xv = *reinterpret_cast<const float4*>(x + (size_t)row * cols + lane * 4);
        acc = (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const float inv = live ? 1.0f / sqrtf(__shfl_sync(0xffffffffu, acc, 0) / (float)cols + eps) : 0.0f;

    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) {
        const float4 gv = *reinterpret_cast<const float4*>(gate + (size_t)row * cols + lane * 4);
        const float4 wv = *reinterpret_cast<const float4*>(weight + lane * 4);
        v.x = xv.x * inv * wv.x * __fdividef(gv.x, 1.0f + __expf(-gv.x));
        v.y = xv.y * inv * wv.y * __fdividef(gv.y, 1.0f + __expf(-gv.y));
        v.z = xv.z * inv * wv.z * __fdividef(gv.z, 1.0f + __expf(-gv.z));
        v.w = xv.w * inv * wv.w * __fdividef(gv.w, 1.0f + __expf(-gv.w));
    }
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    }
    if (lane == 0) {
        s_warp[warp] = amax;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < vh) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
        }
        if (lane == 0) {
            s_warp[0] = t;
        }
    }
    __syncthreads();

    const float scale = s_warp[0] / 448.0f;
    if (threadIdx.x == 0) {
        scales[r_flat] = (uint16_t)(__float_as_uint(scale) >> 16);
    }
    if (!live) {
        return;
    }
    const float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    uchar4 q;
    q.x = to_e4m3(fabsf(v.x) * inv_scale) | ((v.x < 0.0f) ? 0x80u : 0x00u);
    q.y = to_e4m3(fabsf(v.y) * inv_scale) | ((v.y < 0.0f) ? 0x80u : 0x00u);
    q.z = to_e4m3(fabsf(v.z) * inv_scale) | ((v.z < 0.0f) ? 0x80u : 0x00u);
    q.w = to_e4m3(fabsf(v.w) * inv_scale) | ((v.w < 0.0f) ? 0x80u : 0x00u);
    *reinterpret_cast<uchar4*>(out + (size_t)row * cols + lane * 4) = q;
}

extern "C" __global__ void rms_norm_quantise_nvfp4_kernel(const float* __restrict__ x,
                                                          const float* __restrict__ weight,
                                                          uint8_t* __restrict__ packed,
                                                          uint8_t* __restrict__ scales,
                                                          int rows, int cols, float eps,
                                                          float global_scale,
                                                          int blk_mn, int chunk) {
    const int row = blockIdx.x;
    const int sub = threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const bool live = row < rows;
    __shared__ float s_part[8];

    float acc = 0.0f;
    if (live) {
        const int c4 = cols >> 2;
        const float4* xr4 = reinterpret_cast<const float4*>(x + (size_t)row * cols);
        #pragma unroll 4
        for (int i = sub; i < c4; i += 256) {
            const float4 t = xr4[i];
            acc += (t.x * t.x + t.y * t.y) + (t.z * t.z + t.w * t.w);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        s_part[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_part[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, off);
        }
        if (lane == 0) {
            s_part[0] = t;
        }
    }
    __syncthreads();

    const float inv = live ? 1.0f / sqrtf(s_part[0] / (float)cols + eps) : 0.0f;

    float v[16];
    const float* wrow = weight + sub * 16;
    float amax = 0.0f;
    if (live) {
        const float* src = x + (size_t)row * cols + sub * 16;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = src[i] * inv * (1.0f + wrow[i]);
            amax = fmaxf(amax, fabsf(v[i]));
        }
    }
    const float block_scale = amax / 6.0f;
    const uint8_t stored = live ? to_e4m3(block_scale * global_scale) : (uint8_t)0;
    {
        const int blocks_per_row = cols / 16;
        const int k1_count = blocks_per_row / 4;
        const int m_rest = row / blk_mn;
        const int k1 = sub / 4;
        const int k2 = sub % 4;
        const int m_atom = row % blk_mn;
        scales[(m_rest * k1_count + k1) * chunk + m_atom * 4 + k2] = stored;
    }
    if (live) {
        const float stored_f = from_e4m3(stored);
        const float inv_scale = (stored_f > 0.0f) ? global_scale / stored_f : 0.0f;
        #pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const uint8_t lo = to_e2m1(v[pair * 2] * inv_scale);
            const uint8_t hi = to_e2m1(v[pair * 2 + 1] * inv_scale);
            packed[(size_t)row * (cols / 2) + sub * 8 + pair] = (uint8_t)(lo | (hi << 4));
        }
    }
}

extern "C" __global__ void rms_norm_fp8_bf16_kernel(const float* __restrict__ x,
                                                    const float* __restrict__ weight,
                                                    uint8_t* __restrict__ q_out,
                                                    uint16_t* __restrict__ q_scales,
                                                    uint16_t* __restrict__ bf_out,
                                                    int rows, int cols, float eps) {
    const int row = blockIdx.x;
    const int sub = threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const bool live = row < rows;
    __shared__ float s_part[8];

    const int c4 = cols >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(x + (size_t)row * cols);

    float acc = 0.0f;
    if (live) {
        #pragma unroll 4
        for (int i = sub; i < c4; i += 256) {
            const float4 t = xr4[i];
            acc += (t.x * t.x + t.y * t.y) + (t.z * t.z + t.w * t.w);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        s_part[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_part[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, off);
        }
        if (lane == 0) {
            s_part[0] = t;
        }
    }
    __syncthreads();

    const float inv = live ? 1.0f / sqrtf(s_part[0] / (float)cols + eps) : 0.0f;

    float v[16];
    float amax = 0.0f;
    if (live) {
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float4 xv = xr4[sub + g * 256];
            const float4 wv = w4[sub + g * 256];

            v[g * 4 + 0] = xv.x * inv * (1.0f + wv.x);
            v[g * 4 + 1] = xv.y * inv * (1.0f + wv.y);
            v[g * 4 + 2] = xv.z * inv * (1.0f + wv.z);
            v[g * 4 + 3] = xv.w * inv * (1.0f + wv.w);
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[g * 4 + 0]), fabsf(v[g * 4 + 1])),
                                     fmaxf(fabsf(v[g * 4 + 2]), fabsf(v[g * 4 + 3]))));
        }
    }
    __syncthreads();
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    }
    if (lane == 0) {
        s_part[warp] = amax;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_part[lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
        }
        if (lane == 0) {
            s_part[0] = t;
        }
    }
    __syncthreads();

    const float scale = s_part[0] / 448.0f;
    if (live && sub == 0) {
        q_scales[row] = (uint16_t)(__float_as_uint(scale) >> 16);
    }
    const float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    if (live) {
        uint8_t* qo = q_out + (size_t)row * cols;
        uint16_t* bo = bf_out + (size_t)row * cols;
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            uchar4 q;
            uint2 b;
            const float t0 = v[g * 4 + 0], t1 = v[g * 4 + 1], t2 = v[g * 4 + 2], t3 = v[g * 4 + 3];
            q.x = to_e4m3(fabsf(t0) * inv_scale) | ((t0 < 0.0f) ? 0x80u : 0x00u);
            q.y = to_e4m3(fabsf(t1) * inv_scale) | ((t1 < 0.0f) ? 0x80u : 0x00u);
            q.z = to_e4m3(fabsf(t2) * inv_scale) | ((t2 < 0.0f) ? 0x80u : 0x00u);
            q.w = to_e4m3(fabsf(t3) * inv_scale) | ((t3 < 0.0f) ? 0x80u : 0x00u);

            const uint32_t b0 = __float_as_uint(t0), b1 = __float_as_uint(t1);
            const uint32_t b2 = __float_as_uint(t2), b3 = __float_as_uint(t3);
            b.x = ((b0 + 0x7FFFu + ((b0 >> 16) & 1u)) >> 16) | (((b1 + 0x7FFFu + ((b1 >> 16) & 1u)) >> 16) << 16);
            b.y = ((b2 + 0x7FFFu + ((b2 >> 16) & 1u)) >> 16) | (((b3 + 0x7FFFu + ((b3 >> 16) & 1u)) >> 16) << 16);
            const int e = 4 * (sub + g * 256);
            *reinterpret_cast<uchar4*>(qo + e) = q;
            *reinterpret_cast<uint2*>(bo + e) = b;
        }
    }
}

#ifndef SQ_ELEMS
#define SQ_ELEMS 16
#endif
#ifndef SQ_THREADS
#define SQ_THREADS 768
#endif

extern "C" __global__ void __launch_bounds__(SQ_THREADS, 2)
swiglu_quantise_fp8_kernel(const float* __restrict__ gate,
                                                      const float* __restrict__ up,
                                                      uint8_t* __restrict__ out,
                                                      uint16_t* __restrict__ scales,
                                                      int rows, int cols) {
    const int row = blockIdx.x;
    const int sub = threadIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float v[SQ_ELEMS];
    #pragma unroll
    for (int e = 0; e < SQ_ELEMS; ++e) {
        const int c = sub + e * SQ_THREADS;
        v[e] = 0.0f;
        if (c < cols) {
            const float g = gate[(size_t)row * cols + c];
            v[e] = (g / (1.0f + __expf(-g))) * up[(size_t)row * cols + c];
        }
    }
    float amax = 0.0f;
    #pragma unroll
    for (int e = 0; e < SQ_ELEMS; ++e) {
        amax = fmaxf(amax, fabsf(v[e]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    }

    __shared__ float s_warp[32];
    if (lane == 0) {
        s_warp[warp] = amax;
    }
    __syncthreads();
    if (warp == 0) {
        float t = (lane < SQ_THREADS / 32) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
        }
        if (lane == 0) {
            s_warp[0] = t;
        }
    }
    __syncthreads();
    const float scale = s_warp[0] / 448.0f;
    if (threadIdx.x == 0) {
        scales[row] = (uint16_t)(__float_as_uint(scale) >> 16);
    }
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    #pragma unroll
    for (int e = 0; e < SQ_ELEMS; ++e) {
        const int c = sub + e * SQ_THREADS;
        if (c < cols) {
            out[(size_t)row * cols + c] = to_e4m3(fabsf(v[e]) * inv) |
                                          ((v[e] < 0.0f) ? 0x80u : 0x00u);
        }
    }
}

extern "C" __global__ void quantise_fp8_kernel(const float* __restrict__ x,
                                               uint8_t* __restrict__ out,
                                               uint16_t* __restrict__ scales,
                                               int rows, int cols) {
    const int row = blockIdx.x;
    const float* src = x + (size_t)row * cols;

    const int tpr = min(256, max(32, ((cols + 255) / 256) * 32));
    const int wpr = tpr >> 5;
    const int rpb = 256 / tpr;
    const int rib = threadIdx.x / tpr;
    const int sub = threadIdx.x % tpr;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r = blockIdx.x * rpb + rib;
    const bool active = r < rows;

    __shared__ float s_part[8];
    float amax = 0.0f;
    if (active) {
        #pragma unroll 4
        for (int c = sub; c < cols; c += tpr) {
            amax = fmaxf(amax, fabsf(src[c]));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, off));
    }
    if (lane == 0) {
        s_part[warp] = amax;
    }
    if (wpr > 1) {
        __syncthreads();
        if ((warp % wpr) == 0) {
            float t = (lane < wpr) ? s_part[warp + lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, off));
            }
            if (lane == 0) {
                s_part[warp] = t;
            }
        }
        __syncthreads();
    }
    if (!active) {
        return;
    }
    const float scale =
        ((wpr > 1) ? s_part[(warp / wpr) * wpr] : __shfl_sync(0xffffffffu, amax, 0)) / 448.0f;
    if (sub == 0) {
        scales[r] = (uint16_t)(__float_as_uint(scale) >> 16);
    }
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    for (int c = sub; c < cols; c += tpr) {
        out[(size_t)r * cols + c] = to_e4m3(fabsf(src[c]) * inv) |
                                    ((src[c] < 0.0f) ? 0x80u : 0x00u);
    }
}

#ifndef SQN_THREADS
#define SQN_THREADS 768
#endif
#ifndef SQN_ELEMS
#define SQN_ELEMS 16
#endif

extern "C" __global__ void __launch_bounds__(SQN_THREADS, 2)
swiglu_quantise_nvfp4_kernel(const float* __restrict__ gate,
                             const float* __restrict__ up,
                             uint8_t* __restrict__ packed,
                             uint8_t* __restrict__ scales,
                             int rows, int cols, float global_scale,
                             int blk_mn, int chunk) {
    const int row = blockIdx.y + blockIdx.z * gridDim.y;
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    const int blocks_per_row = cols / 16;
    if (block >= blocks_per_row) {
        return;
    }

    const bool live = row < rows;
    const float* gsrc = gate + (size_t)row * cols + block * 16;
    const float* usrc = up + (size_t)row * cols + block * 16;

    float v[SQN_ELEMS];
    #pragma unroll
    for (int i = 0; i < SQN_ELEMS; ++i) {
        v[i] = 0.0f;
    }
    if (live) {
        const float4* g4 = reinterpret_cast<const float4*>(gsrc);
        const float4* u4 = reinterpret_cast<const float4*>(usrc);
        #pragma unroll
        for (int j = 0; j < SQN_ELEMS / 4; ++j) {
            const float4 gv = g4[j];
            const float4 uv = u4[j];
            const float g0 = gv.x, g1 = gv.y, g2 = gv.z, g3 = gv.w;
            v[4 * j + 0] = (g0 / (1.0f + __expf(-g0))) * uv.x;
            v[4 * j + 1] = (g1 / (1.0f + __expf(-g1))) * uv.y;
            v[4 * j + 2] = (g2 / (1.0f + __expf(-g2))) * uv.z;
            v[4 * j + 3] = (g3 / (1.0f + __expf(-g3))) * uv.w;
        }
    }
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < SQN_ELEMS; ++i) {
        amax = fmaxf(amax, fabsf(v[i]));
    }

    const uint8_t stored = live ? to_e4m3((amax / 6.0f) * global_scale) : (uint8_t)0;

    {
        const int k1_count = blocks_per_row / 4;
        const int m_rest = row / blk_mn;
        const int k1 = block / 4;
        const int k2 = block % 4;
        const int m_atom = row % blk_mn;
        scales[(m_rest * k1_count + k1) * chunk + m_atom * 4 + k2] = stored;
    }
    if (!live) {
        return;
    }

    const float stored_f = from_e4m3(stored);
    const float inv_scale = (stored_f > 0.0f) ? global_scale / stored_f : 0.0f;
    if (!live) {
        return;
    }
    #pragma unroll
    for (int pair = 0; pair < SQN_ELEMS / 2; ++pair) {
        const uint8_t lo = to_e2m1(v[pair * 2] * inv_scale);
        const uint8_t hi = to_e2m1(v[pair * 2 + 1] * inv_scale);
        packed[(size_t)row * (cols / 2) + block * 8 + pair] = (uint8_t)(lo | (hi << 4));
    }
}
