#include <cuda_runtime.h>
#include <cstdint>

#define HD_PER_LANE 8
#define HD_MAX 256

#ifndef ATT_ROWS
#define ATT_ROWS 2
#endif

extern "C" __global__ void attention_kernel(const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            float* __restrict__ out,
                                            int seq, int heads, int kv_heads, int head_dim,
                                            float scale,

    const int* __restrict__ seq_start) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;

    const int t0 = blockIdx.y * (2 * ATT_ROWS) + warp * ATT_ROWS;
    if (head_dim > HD_MAX) {
        return;
    }

    const int group = heads / kv_heads;
    const int kvh = h / group;

    float qreg[ATT_ROWS][HD_PER_LANE];
    float acc[ATT_ROWS][HD_PER_LANE];
    float m[ATT_ROWS];
    float l[ATT_ROWS];
    bool live[ATT_ROWS];

    #pragma unroll
    for (int r = 0; r < ATT_ROWS; ++r) {
        const int t = t0 + r;

        live[r] = (t < seq);
        const float* q_t = q + ((size_t)t * heads + h) * head_dim;
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE; ++i) {
            qreg[r][i] = live[r] ? q_t[lane * HD_PER_LANE + i] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE; ++i) {
            acc[r][i] = 0.0f;
        }
        m[r] = -INFINITY;
        l[r] = 0.0f;
    }

    const int s_max = min(t0 + ATT_ROWS - 1, seq - 1);

    const int s_min = seq_start ? seq_start[t0] : 0;
    for (int s = 0; s <= s_max; ++s) {
        const float* k_s = k + ((size_t)s * kv_heads + kvh) * head_dim;
        const float* v_s = v + ((size_t)s * kv_heads + kvh) * head_dim;

        float kreg[HD_PER_LANE];
        float vreg[HD_PER_LANE];
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE; ++i) {
            kreg[i] = k_s[lane * HD_PER_LANE + i];
            vreg[i] = v_s[lane * HD_PER_LANE + i];
        }

        #pragma unroll
        for (int r = 0; r < ATT_ROWS; ++r) {
            if (!live[r] || s > t0 + r || s < s_min) {
                continue;
            }

            float part = 0.0f;
            #pragma unroll
            for (int i = 0; i < HD_PER_LANE; ++i) {
                part += qreg[r][i] * kreg[i];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                part += __shfl_xor_sync(0xffffffffu, part, off);
            }
            const float score = part * scale;

            const float m_new = fmaxf(m[r], score);
            const float alpha = __expf(m[r] - m_new);
            const float p = __expf(score - m_new);
            l[r] = l[r] * alpha + p;
            m[r] = m_new;

            #pragma unroll
            for (int i = 0; i < HD_PER_LANE; ++i) {
                acc[r][i] = acc[r][i] * alpha + p * vreg[i];
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ATT_ROWS; ++r) {
        const int t = t0 + r;
        if (t >= seq) {
            continue;
        }
        float* o_t = out + ((size_t)t * heads + h) * head_dim;
        const float inv = 1.0f / l[r];
        #pragma unroll
        for (int i = 0; i < HD_PER_LANE; ++i) {
            o_t[lane * HD_PER_LANE + i] = acc[r][i] * inv;
        }
    }
}
