#include <cuda_runtime.h>
#include <cstdint>

#define HD_PER_LANE_MAX 8
#define WARPS_PER_BLOCK 4

#ifndef Q_PER_WARP
#define Q_PER_WARP 4
#endif

#define ROWS_PER_BLOCK (WARPS_PER_BLOCK * Q_PER_WARP)
#define MAX_OPTIONS 128

extern "C" __global__ void layer_norm_kernel(const float* __restrict__ x,
                                             const float* __restrict__ weight,
                                             const float* __restrict__ bias,
                                             float* __restrict__ out,
                                             int rows, int cols, float eps,

                                             const int* __restrict__ base_p) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* xr = x + (size_t)(row + *base_p) * cols;
    float* orow = out + (size_t)row * cols;

    __shared__ float s_sum[256];
    __shared__ float s_sq[256];

    const int c4 = cols >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    float sum = 0.0f;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < c4; i += blockDim.x) {
        const float4 v = xr4[i];
        sum += (v.x + v.y) + (v.z + v.w);
        sq += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }
    s_sum[threadIdx.x] = sum;
    s_sq[threadIdx.x] = sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)cols;
    const float var = s_sq[0] / (float)cols - mean * mean;
    const float inv = 1.0f / sqrtf(var + eps);

    float4* or4 = reinterpret_cast<float4*>(orow);
    if (weight && bias) {
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        for (int i = threadIdx.x; i < c4; i += blockDim.x) {
            const float4 v = xr4[i], wv = w4[i], bv = b4[i];
            or4[i] = make_float4((v.x - mean) * inv * wv.x + bv.x, (v.y - mean) * inv * wv.y + bv.y,
                                 (v.z - mean) * inv * wv.z + bv.z, (v.w - mean) * inv * wv.w + bv.w);
        }
    } else if (weight) {
        const float4* w4 = reinterpret_cast<const float4*>(weight);
        for (int i = threadIdx.x; i < c4; i += blockDim.x) {
            const float4 v = xr4[i], wv = w4[i];
            or4[i] = make_float4((v.x - mean) * inv * wv.x, (v.y - mean) * inv * wv.y,
                                 (v.z - mean) * inv * wv.z, (v.w - mean) * inv * wv.w);
        }
    } else {
        for (int i = threadIdx.x; i < c4; i += blockDim.x) {
            const float4 v = xr4[i];
            or4[i] = make_float4((v.x - mean) * inv, (v.y - mean) * inv,
                                 (v.z - mean) * inv, (v.w - mean) * inv);
        }
    }
}

extern "C" __global__ void gelu_erf_kernel(const float* __restrict__ x,
                                           float* __restrict__ out, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const float v = x[i];
        out[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    }
}

extern "C" __global__ void f32_to_bf16_kernel(const float* __restrict__ x,
                                              uint16_t* __restrict__ out, int total,
                                              int cols,

                                              const int* __restrict__ base_p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const uint32_t bits = __float_as_uint(x[i + *base_p * cols]);
        const uint32_t lsb = (bits >> 16) & 1u;
        out[i] = (uint16_t)((bits + 0x7FFFu + lsb) >> 16);
    }
}

extern "C" __global__ void bias_add_kernel(const float* __restrict__ x,
                                           const float* __restrict__ bias,
                                           float* __restrict__ out,
                                           int rows, int cols) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < rows * cols && cols % 4 == 0) {
        const int col = idx % cols;
        const float4 v = *reinterpret_cast<const float4*>(x + idx);
        const float4 bv = *reinterpret_cast<const float4*>(bias + col);
        *reinterpret_cast<float4*>(out + idx) =
            make_float4(v.x + bv.x, v.y + bv.y, v.z + bv.z, v.w + bv.w);
    } else {
        for (int k = idx; k < rows * cols; ++k) out[k] = x[k] + bias[k % cols];
    }
}

extern "C" __global__ void add_kernel(float* __restrict__ out,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b, int total) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < total) {
        const float4 x = *reinterpret_cast<const float4*>(a + i);
        const float4 y = *reinterpret_cast<const float4*>(b + i);
        *reinterpret_cast<float4*>(out + i) =
            make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    } else {
        for (int k = i; k < total; ++k) out[k] = a[k] + b[k];
    }
}

extern "C" __global__ void mha_kernel(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      float* __restrict__ out,
                                      int q_len, int kv_len, int width, int heads, float scale) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;
    const int t = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (t >= q_len) {
        return;
    }
    const int head_dim = width / heads;
    const int per_lane = head_dim / 32;
    if (per_lane < 1 || per_lane > HD_PER_LANE_MAX || head_dim % 32 != 0) {
        return;
    }

    const float* q_t = q + ((size_t)t * heads + h) * head_dim;
    float qreg[HD_PER_LANE_MAX];
    #pragma unroll
    for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
        qreg[i] = (i < per_lane) ? q_t[lane * per_lane + i] : 0.0f;
    }

    float acc[HD_PER_LANE_MAX];
    #pragma unroll
    for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        const float* k_s = k + ((size_t)s * heads + h) * head_dim;

        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            if (i < per_lane) {
                part += qreg[i] * k_s[lane * per_lane + i];
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            part += __shfl_xor_sync(0xffffffffu, part, off);
        }
        const float score = part * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        const float* v_s = v + ((size_t)s * heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            if (i < per_lane) {
                acc[i] = acc[i] * alpha + p * v_s[lane * per_lane + i];
            }
        }
    }

    float* o_t = out + ((size_t)t * heads + h) * head_dim;
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
        if (i < per_lane) {
            o_t[lane * per_lane + i] = acc[i] * inv;
        }
    }
}

#ifndef KV_SPLITS
#define KV_SPLITS 16
#endif

extern "C" __global__ void mha_partial_kernel(const float* __restrict__ q,
                                              const float* __restrict__ k,
                                              const float* __restrict__ v,
                                              float* __restrict__ partial,
                                              int q_len, int kv_len, int width, int heads, float scale,

                                              int kv_base,

                                              const int* __restrict__ kv_real_p,

                                              int per_split) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;
    const int t0 = blockIdx.y * ROWS_PER_BLOCK + warp * Q_PER_WARP;
    const int split = blockIdx.z;
    if (t0 >= q_len) {
        return;
    }
    const int head_dim = width / heads;
    const int per_lane = head_dim / 32;
    if (per_lane < 1 || per_lane > HD_PER_LANE_MAX || head_dim % 32 != 0) {
        return;
    }

    const int s_begin = kv_base + split * per_split;
    const int kv_real = *kv_real_p;
    const int s_end = min(min(kv_len, s_begin + per_split), kv_real);

    float qreg[Q_PER_WARP][HD_PER_LANE_MAX];
    float acc[Q_PER_WARP][HD_PER_LANE_MAX];
    float m[Q_PER_WARP];
    float l[Q_PER_WARP];
    bool live[Q_PER_WARP];
    #pragma unroll
    for (int j = 0; j < Q_PER_WARP; ++j) {
        live[j] = (t0 + j) < q_len;
        const float* q_t = q + ((size_t)(t0 + j) * heads + h) * head_dim;
        m[j] = -INFINITY;
        l[j] = 0.0f;
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            qreg[j][i] = (live[j] && i < per_lane) ? q_t[lane * per_lane + i] : 0.0f;
            acc[j][i] = 0.0f;
        }
    }
    for (int s = s_begin; s < s_end; ++s) {
        const float* k_s = k + ((size_t)(s - kv_base) * heads + h) * head_dim;
        float kv[HD_PER_LANE_MAX];
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            kv[i] = (i < per_lane) ? k_s[lane * per_lane + i] : 0.0f;
        }
        float part[Q_PER_WARP];
        #pragma unroll
        for (int j = 0; j < Q_PER_WARP; ++j) {
            float p = 0.0f;
            #pragma unroll
            for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
                if (i < per_lane) p += qreg[j][i] * kv[i];
            }
            part[j] = p;
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < Q_PER_WARP; ++j) {
                part[j] += __shfl_xor_sync(0xffffffffu, part[j], off);
            }
        }

        const float* v_s = v + ((size_t)(s - kv_base) * heads + h) * head_dim;
        float vv[HD_PER_LANE_MAX];
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            vv[i] = (i < per_lane) ? v_s[lane * per_lane + i] : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < Q_PER_WARP; ++j) {
            if (!live[j]) continue;
            const float score = part[j] * scale;
            const float m_new = fmaxf(m[j], score);
            const float alpha = __expf(m[j] - m_new);
            const float p = __expf(score - m_new);
            l[j] = l[j] * alpha + p;
            m[j] = m_new;
            #pragma unroll
            for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
                if (i < per_lane) acc[j][i] = acc[j][i] * alpha + p * vv[i];
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < Q_PER_WARP; ++j) {
        if (!live[j]) continue;
        float* row = partial + (((size_t)split * q_len + (t0 + j)) * heads + h) * (head_dim + 2);
        if (lane == 0) {
            row[0] = m[j];
            row[1] = l[j];
        }
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            if (i < per_lane) {
                row[2 + lane * per_lane + i] = acc[j][i];
            }
        }
    }
}

extern "C" __global__ void mha_partial_kernel_pl2(const float* __restrict__ q,
                                                  const float* __restrict__ k,
                                                  const float* __restrict__ v,
                                                  float* __restrict__ partial,
                                                  int q_len, int kv_len, int width, int heads,
                                                  float scale, int kv_base, const int* __restrict__ kv_real_p, int per_split) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;
    const int t0 = blockIdx.y * ROWS_PER_BLOCK + warp * Q_PER_WARP;
    const int split = blockIdx.z;
    if (t0 >= q_len) {
        return;
    }
    if (width / heads != 64) {
        return;
    }

    const int s_begin = kv_base + split * per_split;
    const int kv_real = *kv_real_p;
    const int s_end = min(min(kv_len, s_begin + per_split), kv_real);

    float qreg[Q_PER_WARP][2];
    float acc[Q_PER_WARP][2];
    float m[Q_PER_WARP];
    float l[Q_PER_WARP];
    bool live[Q_PER_WARP];
    #pragma unroll
    for (int j = 0; j < Q_PER_WARP; ++j) {
        live[j] = (t0 + j) < q_len;
        const float* q_t = q + ((size_t)(t0 + j) * heads + h) * 64;
        m[j] = -INFINITY;
        l[j] = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            qreg[j][i] = live[j] ? q_t[lane * 2 + i] : 0.0f;
            acc[j][i] = 0.0f;
        }
    }

    int s = s_begin;
    for (; s + 4 <= s_end; s += 4) {
        float2 kv[4];
        float2 vv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float* k_s = k + ((size_t)(s + u - kv_base) * heads + h) * 64;
            kv[u] = *reinterpret_cast<const float2*>(k_s + lane * 2);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float* v_s = v + ((size_t)(s + u - kv_base) * heads + h) * 64;
            vv[u] = *reinterpret_cast<const float2*>(v_s + lane * 2);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float part[Q_PER_WARP];
            #pragma unroll
            for (int j = 0; j < Q_PER_WARP; ++j) {
                part[j] = qreg[j][0] * kv[u].x + qreg[j][1] * kv[u].y;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                #pragma unroll
                for (int j = 0; j < Q_PER_WARP; ++j) {
                    part[j] += __shfl_xor_sync(0xffffffffu, part[j], off);
                }
            }
            #pragma unroll
            for (int j = 0; j < Q_PER_WARP; ++j) {
                if (!live[j]) continue;
                const float score = part[j] * scale;
                const float m_new = fmaxf(m[j], score);
                const float alpha = __expf(m[j] - m_new);
                const float p = __expf(score - m_new);
                l[j] = l[j] * alpha + p;
                m[j] = m_new;
                acc[j][0] = acc[j][0] * alpha + p * vv[u].x;
                acc[j][1] = acc[j][1] * alpha + p * vv[u].y;
            }
        }
    }
    for (; s < s_end; ++s) {
        const float* k_s = k + ((size_t)(s - kv_base) * heads + h) * 64;
        const float2 kv = *reinterpret_cast<const float2*>(k_s + lane * 2);
        float part[Q_PER_WARP];
        #pragma unroll
        for (int j = 0; j < Q_PER_WARP; ++j) {
            part[j] = qreg[j][0] * kv.x + qreg[j][1] * kv.y;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int j = 0; j < Q_PER_WARP; ++j) {
                part[j] += __shfl_xor_sync(0xffffffffu, part[j], off);
            }
        }
        const float* v_s = v + ((size_t)(s - kv_base) * heads + h) * 64;
        const float2 vv = *reinterpret_cast<const float2*>(v_s + lane * 2);
        #pragma unroll
        for (int j = 0; j < Q_PER_WARP; ++j) {
            if (!live[j]) continue;
            const float score = part[j] * scale;
            const float m_new = fmaxf(m[j], score);
            const float alpha = __expf(m[j] - m_new);
            const float p = __expf(score - m_new);
            l[j] = l[j] * alpha + p;
            m[j] = m_new;
            acc[j][0] = acc[j][0] * alpha + p * vv.x;
            acc[j][1] = acc[j][1] * alpha + p * vv.y;
        }
    }

    #pragma unroll
    for (int j = 0; j < Q_PER_WARP; ++j) {
        if (!live[j]) continue;
        float* row = partial + (((size_t)split * q_len + (t0 + j)) * heads + h) * (64 + 2);
        if (lane == 0) {
            row[0] = m[j];
            row[1] = l[j];
        }
        row[2 + lane * 2] = acc[j][0];
        row[2 + lane * 2 + 1] = acc[j][1];
    }
}

extern "C" __global__ void mha_combine_kernel(const float* __restrict__ partial,
                                              float* __restrict__ out,
                                              int q_len, int width, int heads,

                                              int n_splits) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;
    const int t = blockIdx.y * WARPS_PER_BLOCK + warp;
    if (t >= q_len) {
        return;
    }
    const int head_dim = width / heads;
    const int per_lane = head_dim / 32;
    if (per_lane < 1 || per_lane > HD_PER_LANE_MAX || head_dim % 32 != 0) {
        return;
    }

    double M = -INFINITY;
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* row = partial + (((size_t)sp * q_len + t) * heads + h) * (head_dim + 2);
        M = fmax(M, (double)row[0]);
    }
    double l = 0.0;
    float acc[HD_PER_LANE_MAX];
    #pragma unroll
    for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
        acc[i] = 0.0f;
    }
    for (int sp = 0; sp < n_splits; ++sp) {
        const float* row = partial + (((size_t)sp * q_len + t) * heads + h) * (head_dim + 2);
        const double w = exp((double)row[0] - M);
        l += (double)row[1] * w;
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
            if (i < per_lane) {
                acc[i] += (float)((double)row[2 + lane * per_lane + i] * w);
            }
        }
    }

    float* o_t = out + ((size_t)t * heads + h) * head_dim;
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int i = 0; i < HD_PER_LANE_MAX; ++i) {
        if (i < per_lane) {
            o_t[lane * per_lane + i] = acc[i] * inv;
        }
    }
}

extern "C" __global__ void mean_span_kernel(const float* __restrict__ x,
                                            const int* __restrict__ starts,
                                            const int* __restrict__ ends,
                                            float* __restrict__ out,
                                            int n_spans, int cols) {
    const int span = blockIdx.x;
    if (span >= n_spans) {
        return;
    }
    const int s = starts[span];
    const int e = ends[span];
    float* orow = out + (size_t)span * cols;
    const float inv = (e > s) ? 1.0f / (float)(e - s) : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        double acc = 0.0;
        for (int t = s; t < e; ++t) {
            acc += (double)x[(size_t)t * cols + c];
        }
        orow[c] = (float)acc * inv;
    }
}

extern "C" __global__ void l2_normalise_kernel(float* __restrict__ out,
                                               const float* __restrict__ x,
                                               int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const float* xr = x + (size_t)row * cols;
    float* orow = out + (size_t)row * cols;

    __shared__ double s_part[256];
    double acc = 0.0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const double v = (double)xr[c];
        acc += v * v;
    }
    s_part[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const double norm = sqrt(s_part[0]);
    const float inv = (norm > 1e-12) ? (float)(1.0 / norm) : 1.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        orow[c] = xr[c] * inv;
    }
}

extern "C" __global__ void summary_softmax_kernel(const float* __restrict__ routed,
                                                  const float* __restrict__ fields,
                                                  const int* __restrict__ counts,
                                                  float* __restrict__ out,
                                                  int n_questions, int width, float scale) {
    const int q = blockIdx.x;
    if (q >= n_questions) {
        return;
    }
    const int count = counts[q];
    if (count <= 0 || count > MAX_OPTIONS) {
        return;
    }

    int offset = 0;
    for (int i = 0; i < q; ++i) {
        offset += counts[i];
    }

    __shared__ float s_red[256];
    __shared__ float s_w[MAX_OPTIONS];

    const float* field = fields + (size_t)q * width;
    for (int j = 0; j < count; ++j) {
        const float* row = routed + (size_t)(offset + j) * width;
        float part = 0.0f;
        for (int d = threadIdx.x; d < width; d += blockDim.x) {
            part += row[d] * field[d];
        }
        s_red[threadIdx.x] = part;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride) {
                s_red[threadIdx.x] += s_red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            s_w[j] = s_red[0] * scale;
        }
        __syncthreads();
    }

    float max = -INFINITY;
    for (int j = 0; j < count; ++j) {
        max = fmaxf(max, s_w[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < count; ++j) {
        s_w[j] = __expf(s_w[j] - max);
        sum += s_w[j];
    }
    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;

    float* orow = out + (size_t)q * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < count; ++j) {
            acc += (s_w[j] * inv) * routed[(size_t)(offset + j) * width + d];
        }
        orow[d] = acc;
    }
}

extern "C" __global__ void accumulate_indexed_kernel(float* __restrict__ dst,
                                                     const float* __restrict__ src,
                                                     const int* __restrict__ idx,
                                                     int rows, int cols) {
    const int r = blockIdx.y + blockIdx.z * gridDim.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        dst[(size_t)r * cols + c] += src[(size_t)idx[r] * cols + c];
    }
}

extern "C" __global__ void gather_rows_kernel(const float* __restrict__ src,
                                              const int* __restrict__ idx,
                                              float* __restrict__ out,
                                              int rows, int cols) {
    const int r = blockIdx.y + blockIdx.z * gridDim.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        out[(size_t)r * cols + c] = src[(size_t)idx[r] * cols + c];
    }
}

extern "C" __global__ void head_pair_dot_kernel(const float* __restrict__ a,
                                                const float* __restrict__ b,
                                                const int* __restrict__ idx,
                                                float* __restrict__ out,
                                                int rows, int cols, float scale) {
    const int r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    const float* ar = a + (size_t)r * cols;
    const float* br = b + (size_t)idx[r] * cols;

    __shared__ double s_part[256];
    double acc = 0.0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        acc += (double)ar[c] * (double)br[c];
    }
    s_part[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            s_part[threadIdx.x] += s_part[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[r] = (float)s_part[0] * scale;
    }
}

extern "C" __global__ void head_logit_kernel(const float* __restrict__ prior,
                                             const float* __restrict__ cosine,
                                             const float* __restrict__ residual,
                                             float* __restrict__ out,
                                             int n, float joint_scale, float gate) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = prior[i] + gate * (joint_scale * cosine[i] + residual[i]);
    }
}

extern "C" __global__ void head_features_kernel(const float* __restrict__ field,
                                                const float* __restrict__ option,
                                                float* __restrict__ out,
                                                int rows, int width) {
    const int part = blockIdx.x;

    const int r = blockIdx.y + blockIdx.z * gridDim.y;
    if (r >= rows) {
        return;
    }
    const float* f = field + (size_t)r * width;
    const float* o = option + (size_t)r * width;
    float* dst = out + (size_t)r * 4 * width + (size_t)part * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        float v;
        if (part == 0) {
            v = f[d];
        } else if (part == 1) {
            v = o[d];
        } else if (part == 2) {
            v = f[d] * o[d];
        } else {
            v = fabsf(f[d] - o[d]);
        }
        dst[d] = v;
    }
}
