#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__ void rms_norm_kernel(const float* __restrict__ x,
                                           const float* __restrict__ weight,
                                           float* __restrict__ out,
                                           int rows, int cols, float eps) {
    const int tpr = min(256, max(32, ((cols + 255) / 256) * 32));
    const int wpr = tpr >> 5;
    const int rpb = 256 / tpr;
    const int rib = threadIdx.x / tpr;
    const int sub = threadIdx.x % tpr;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int row = blockIdx.x * rpb + rib;
    const float* xr = x + (size_t)row * cols;
    float* orow = out + (size_t)row * cols;
    const bool active = row < rows;

    __shared__ float s_part[8];
    float acc = 0.0f;
    const int c4 = cols >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    if (active) {
        #pragma unroll 4
        for (int i = sub; i < c4; i += tpr) {
            const float4 v = xr4[i];
            acc += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
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
    if ((warp % wpr) == 0) {
        float t = (lane < wpr) ? s_part[warp + lane] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, off);
        }
        if (lane == 0) {
            s_part[warp] = t;
        }
    }
    __syncthreads();
    if (active) {
        const float inv = 1.0f / sqrtf(s_part[(warp / wpr) * wpr] / (float)cols + eps);
        float4* or4 = reinterpret_cast<float4*>(orow);
        if (weight) {
            const float4* w4 = reinterpret_cast<const float4*>(weight);
            for (int i = sub; i < c4; i += tpr) {
                const float4 v = xr4[i], wv = w4[i];
                or4[i] = make_float4(v.x * inv * (1.0f + wv.x), v.y * inv * (1.0f + wv.y),
                                     v.z * inv * (1.0f + wv.z), v.w * inv * (1.0f + wv.w));
            }
        } else {
            for (int i = sub; i < c4; i += tpr) {
                const float4 v = xr4[i];
                or4[i] = make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
            }
        }
    }
}

extern "C" __global__ void qk_norm_rope_kernel(const float* __restrict__ x,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ cos_t,
                                               const float* __restrict__ sin_t,
                                               float* __restrict__ out,
                                               float* __restrict__ gate_out,
                                               int rows, int heads, int head_dim, int rotary_dim,
                                               float eps,
                                               const int* __restrict__ seq_start,
                                               const int* __restrict__ pos_base_dev,
                                               const int* __restrict__ pos_map) {
    __shared__ float srow[8][256];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= rows) {
        return;
    }
    const int t = row / heads;
    const int h = row - t * heads;
    const int c4 = head_dim >> 2;
    const int stride4 = gate_out ? 2 * c4 : c4;
    const float4* __restrict__ x4 =
        reinterpret_cast<const float4*>(x) + (size_t)row * stride4;
    float4* __restrict__ o4 = reinterpret_cast<float4*>(out) + (size_t)row * c4;
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(weight);

    float4 a = x4[lane];
    float4 b = x4[lane + 32];
    float acc = 0.0f;
    acc += (a.x * a.x + a.y * a.y) + (a.z * a.z + a.w * a.w);
    acc += (b.x * b.x + b.y * b.y) + (b.z * b.z + b.w * b.w);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    const float inv = 1.0f / sqrtf(__shfl_sync(0xffffffffu, acc, 0) / (float)head_dim + eps);

    const float4 w0 = w4[lane], w1 = w4[lane + 32];
    a = make_float4(a.x * inv * (1.0f + w0.x), a.y * inv * (1.0f + w0.y),
                    a.z * inv * (1.0f + w0.z), a.w * inv * (1.0f + w0.w));
    b = make_float4(b.x * inv * (1.0f + w1.x), b.y * inv * (1.0f + w1.y),
                    b.z * inv * (1.0f + w1.z), b.w * inv * (1.0f + w1.w));

    float* r = srow[warp];
    *reinterpret_cast<float4*>(&r[4 * lane]) = a;
    *reinterpret_cast<float4*>(&r[4 * lane + 128]) = b;
    __syncwarp();

    const int pos = pos_map != 0
        ? pos_map[t]
        : (seq_start ? t - seq_start[t] : t)
          + (pos_base_dev != 0 ? pos_base_dev[0] : 0);

    const int half = rotary_dim >> 1;
    for (int p = lane; p < half; p += 32) {
        const float c = cos_t[pos * rotary_dim + p];
        const float s = sin_t[pos * rotary_dim + p];
        const float lo = r[p];
        const float hi = r[p + half];
        r[p] = lo * c - hi * s;
        r[p + half] = hi * c + lo * s;
    }
    __syncwarp();

    o4[lane] = *reinterpret_cast<const float4*>(&r[4 * lane]);
    o4[lane + 32] = *reinterpret_cast<const float4*>(&r[4 * lane + 128]);
    if (gate_out) {
        float4* __restrict__ g4 = reinterpret_cast<float4*>(gate_out) + (size_t)row * c4;
        g4[lane] = x4[c4 + lane];
        g4[lane + 32] = x4[c4 + lane + 32];
    }
}

extern "C" __global__ void rms_norm_gated_kernel(const float* __restrict__ x,
                                                 const float* __restrict__ gate,
                                                 const float* __restrict__ weight,
                                                 float* __restrict__ out,
                                                 int rows, int cols, float eps) {
    const int tpr = min(256, max(32, ((cols + 255) / 256) * 32));
    const int wpr = tpr >> 5;
    const int rpb = 256 / tpr;
    const int row = blockIdx.x * rpb + (threadIdx.x / tpr);
    const int sub = threadIdx.x % tpr;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float* xr = x + (size_t)row * cols;
    float* orow = out + (size_t)row * cols;
    const bool active = row < rows;

    __shared__ float s_part[8];
    float acc = 0.0f;
    const int c4 = cols >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    if (active) {
        #pragma unroll 4
        for (int i = sub; i < c4; i += tpr) {
            const float4 v = xr4[i];
            acc += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        s_part[warp] = acc;
    }
    if (wpr > 1) {
        __syncthreads();
        if ((warp % wpr) == 0) {
            float t = (lane < wpr) ? s_part[warp + lane] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                t += __shfl_down_sync(0xffffffffu, t, off);
            }
            if (lane == 0) {
                s_part[warp] = t;
            }
        }
        __syncthreads();
    }
    if (active) {
        const float total =
            (wpr > 1) ? s_part[(warp / wpr) * wpr] : __shfl_sync(0xffffffffu, acc, 0);
        const float inv = 1.0f / sqrtf(total / (float)cols + eps);

        const float4* g4 = gate ? reinterpret_cast<const float4*>(gate + (size_t)row * cols) : nullptr;
        float4* or4 = reinterpret_cast<float4*>(orow);
        if (weight) {
            const float4* w4 = reinterpret_cast<const float4*>(weight);
            for (int i = sub; i < c4; i += tpr) {
                const float4 v = xr4[i], wv = w4[i];
                const float4 gv = g4 ? g4[i] : make_float4(0.f, 0.f, 0.f, 0.f);

                or4[i] = make_float4(
                    v.x * inv * wv.x * __fdividef(gv.x, 1.0f + __expf(-gv.x)),
                    v.y * inv * wv.y * __fdividef(gv.y, 1.0f + __expf(-gv.y)),
                    v.z * inv * wv.z * __fdividef(gv.z, 1.0f + __expf(-gv.z)),
                    v.w * inv * wv.w * __fdividef(gv.w, 1.0f + __expf(-gv.w)));
            }
        } else {
            for (int i = sub; i < c4; i += tpr) {
                const float4 v = xr4[i];
                const float4 gv = g4 ? g4[i] : make_float4(0.f, 0.f, 0.f, 0.f);
                or4[i] = make_float4(
                    v.x * inv * __fdividef(gv.x, 1.0f + __expf(-gv.x)),
                    v.y * inv * __fdividef(gv.y, 1.0f + __expf(-gv.y)),
                    v.z * inv * __fdividef(gv.z, 1.0f + __expf(-gv.z)),
                    v.w * inv * __fdividef(gv.w, 1.0f + __expf(-gv.w)));
            }
        }
    }
}

extern "C" __global__ void swiglu_kernel(const float* __restrict__ gate,
                                         const float* __restrict__ up,
                                         float* __restrict__ out, int total) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < total) {
        const float4 g = *reinterpret_cast<const float4*>(gate + i);
        const float4 u = *reinterpret_cast<const float4*>(up + i);
        *reinterpret_cast<float4*>(out + i) = make_float4(
            (g.x / (1.0f + __expf(-g.x))) * u.x, (g.y / (1.0f + __expf(-g.y))) * u.y,
            (g.z / (1.0f + __expf(-g.z))) * u.z, (g.w / (1.0f + __expf(-g.w))) * u.w);
    } else {
        for (int k = i; k < total; ++k) {
            const float g = gate[k];
            out[k] = (g / (1.0f + __expf(-g))) * up[k];
        }
    }
}

extern "C" __global__ void causal_conv_silu_kernel(const float* __restrict__ x,
                                                   const float* __restrict__ w,
                                                   float* __restrict__ out,
                                                   int rows, int channels, int k,

                                                   const int* __restrict__ seq_start,

                                                   const float* __restrict__ history,

                                                   const int* __restrict__ slot_map) {
    const int t = blockIdx.y + blockIdx.z * gridDim.y;
    if (t >= rows) {
        return;
    }
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) {
        return;
    }
    float acc = 0.0f;
    const int lo = seq_start ? seq_start[t] : 0;
    for (int tap = 0; tap < k; ++tap) {
        const int src = t - (k - 1) + tap;
        if (src >= lo) {
            acc += w[ch * k + tap] * x[(size_t)src * channels + ch];
        } else if (history != 0 && src < lo && src >= lo - (k - 1)) {
            const int slot = slot_map ? slot_map[t] : 0;
            acc += w[ch * k + tap]
                 * history[(size_t)(slot * (k - 1) + (src - lo + (k - 1))) * channels + ch];
        }
    }
    out[(size_t)t * channels + ch] = acc / (1.0f + __expf(-acc));
}

extern "C" __global__ void rope_kernel(float* __restrict__ out,
                                       const float* __restrict__ x,
                                       const float* __restrict__ cos_t,
                                       const float* __restrict__ sin_t,
                                       int rows, int heads, int head_dim, int rotary_dim,

                                       const int* __restrict__ seq_start,

                                       const int* __restrict__ pos_base_dev,

                                       const int* __restrict__ pos_map) {
    const int half = rotary_dim / 2;
    const int pairs_per_head = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * heads * pairs_per_head) {
        return;
    }
    const int row = idx / (heads * pairs_per_head);
    const int rem = idx % (heads * pairs_per_head);
    const int head = rem / pairs_per_head;
    const int j = rem % pairs_per_head;

    const size_t base = ((size_t)row * heads + head) * head_dim;

    const int pos = pos_map != 0
        ? pos_map[row]
        : (seq_start ? row - seq_start[row] : row)
          + (pos_base_dev != 0 ? pos_base_dev[0] : 0);
    if (j < half) {
        const float c = cos_t[pos * rotary_dim + j];
        const float s = sin_t[pos * rotary_dim + j];
        const float lo = x[base + j];
        const float hi = x[base + j + half];
        out[base + j] = lo * c - hi * s;
        out[base + j + half] = hi * c + lo * s;
    } else {
        out[base + 2 * j] = x[base + 2 * j];
        out[base + 2 * j + 1] = x[base + 2 * j + 1];
    }
}

extern "C" __global__ void copy_range_u8_kernel(const unsigned char* __restrict__ src, long long src_off,
                                                unsigned char* __restrict__ dst, long long dst_off,
                                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[dst_off + i] = src[src_off + i];
    }
}

extern "C" __global__ void copy_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst, int total) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < total) {
        *reinterpret_cast<float4*>(dst + i) = *reinterpret_cast<const float4*>(src + i);
    } else {
        for (int k = i; k < total; ++k) dst[k] = src[k];
    }
}

extern "C" __global__ void copy_u8_kernel(const unsigned char* __restrict__ src,
                                          unsigned char* __restrict__ dst, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void sigmoid_mul_kernel(float* __restrict__ out,
                                              const float* __restrict__ x,
                                              const float* __restrict__ gate, int total) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < total) {
        const float4 v = *reinterpret_cast<const float4*>(x + i);
        const float4 g = *reinterpret_cast<const float4*>(gate + i);
        *reinterpret_cast<float4*>(out + i) = make_float4(
            v.x * 1.0f / (1.0f + __expf(-g.x)), v.y * 1.0f / (1.0f + __expf(-g.y)),
            v.z * 1.0f / (1.0f + __expf(-g.z)), v.w * 1.0f / (1.0f + __expf(-g.w)));
    } else {
        for (int k = i; k < total; ++k) {
            out[k] = x[k] * (1.0f / (1.0f + __expf(-gate[k])));
        }
    }
}

extern "C" __global__ void gdn_gates_kernel(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            const float* __restrict__ a_log,
                                            const float* __restrict__ dt_bias,
                                            float* __restrict__ decay,
                                            float* __restrict__ beta,
                                            int tokens, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads) {
        return;
    }
    const int h = i % heads;
    beta[i] = 1.0f / (1.0f + __expf(-b[i]));
    const float x = a[i] + dt_bias[h];
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    decay[i] = -__expf(a_log[h]) * sp;
}

extern "C" __global__ void split_q_gate_kernel(const float* __restrict__ qg,
                                              float* __restrict__ q,
                                              float* __restrict__ gate,
                                              int tokens, int heads, int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * head_dim) {
        return;
    }
    const int t = i / (heads * head_dim);
    const int rem = i % (heads * head_dim);
    const int h = rem / head_dim;
    const int d = rem % head_dim;
    const size_t src = ((size_t)t * heads + h) * head_dim * 2;
    q[i] = qg[src + d];
    gate[i] = qg[src + head_dim + d];
}

extern "C" __global__ void embed_gather_kernel(const float* __restrict__ table,
                                               const int* __restrict__ ids,
                                               float* __restrict__ out,
                                               int tokens, int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * hidden) {
        return;
    }
    const int t = i / hidden;
    const int d = i % hidden;
    out[i] = table[(size_t)ids[t] * hidden + d];
}

extern "C" __global__ void split_qkv_kernel(const float* __restrict__ mixed,
                                           float* __restrict__ q,
                                           float* __restrict__ k,
                                           float* __restrict__ v,
                                           int tokens, int kh, int kd, int vh, int vd) {
    const int conv_dim = 2 * kh * kd + vh * vd;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * conv_dim) {
        return;
    }
    const int t = i / conv_dim;
    const int c = i % conv_dim;
    if (c < kh * kd) {
        q[t * kh * kd + c] = mixed[i];
    } else if (c < 2 * kh * kd) {
        k[t * kh * kd + (c - kh * kd)] = mixed[i];
    } else {
        v[t * vh * vd + (c - 2 * kh * kd)] = mixed[i];
    }
}

extern "C" __global__ void repeat_interleave_kernel(const float* __restrict__ src,
                                                   float* __restrict__ out,
                                                   int tokens, int kh, int ratio, int dim) {
    const int out_heads = kh * ratio;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * out_heads * dim) {
        return;
    }
    const int t = i / (out_heads * dim);
    const int rem = i % (out_heads * dim);
    const int oh = rem / dim;
    const int d = rem % dim;
    const int h = oh / ratio;
    out[i] = src[(t * kh + h) * dim + d];
}

extern "C" __global__ void rope_tables_kernel(const int* __restrict__ positions,
                                              float* __restrict__ cos_out,
                                              float* __restrict__ sin_out,
                                              int rows, int rotary_dim, float theta) {
    const int half = rotary_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * rotary_dim) {
        return;
    }
    const int row = idx / rotary_dim;
    const int j = idx % rotary_dim;
    const int i = j % half;
    const float inv_freq = 1.0f / powf(theta, 2.0f * (float)i / (float)rotary_dim);
    const float f = (float)positions[row] * inv_freq;
    cos_out[idx] = cosf(f);
    sin_out[idx] = sinf(f);
}

extern "C" __global__ void conv_hist_save_kernel(const float* __restrict__ src,
                                                 float* __restrict__ hist,
                                                 const int* __restrict__ saves,
                                                 int n_saves, int rows, int k, int channels) {
    const int i = blockIdx.x;
    if (i >= n_saves) {
        return;
    }
    const int row = saves[2 * i];
    const int slot = saves[2 * i + 1];
    const int hist_rows = k - 1;
    const int per_slot = hist_rows * channels;
    const int first = row + 1 - hist_rows;
    if (row < 0 || first < 0 || row >= rows) {
        return;
    }
    const float* s = src + (size_t)first * channels;
    float* d = hist + (size_t)slot * per_slot;
    for (int c = threadIdx.x; c < per_slot; c += blockDim.x) {
        d[c] = s[c];
    }
}
