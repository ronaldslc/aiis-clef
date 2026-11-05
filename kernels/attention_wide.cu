#include <cuda_runtime.h>
#include <cstdint>

#define HD_MAX 256
#define DPL_MAX 64

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
    if (head_dim > HD_MAX) {
        return;
    }

    const int gqa = heads / kv_heads;
    const int kvh = h / gqa;

    const int row_in_warp = lane >> 2;
    const int sub = lane & 3;
    const int dpl = head_dim >> 2;

    const int t0 = blockIdx.y * 16 + warp * 8;
    const int t = t0 + row_in_warp;
    const bool live = (t < seq);

    float qreg[DPL_MAX];
    float acc[DPL_MAX];
    float m = -INFINITY;
    float l = 0.0f;

    const float* q_t = q + ((size_t)t * heads + h) * head_dim;
    #pragma unroll
    for (int i = 0; i < DPL_MAX; ++i) {
        qreg[i] = (live && i < dpl) ? q_t[sub * dpl + i] : 0.0f;
    }
    #pragma unroll
    for (int i = 0; i < DPL_MAX; ++i) {
        acc[i] = 0.0f;
    }

    const int s_max_warp = min(t0 + 7, seq - 1);

    const int s_min = seq_start ? seq_start[t0] : 0;

    const unsigned mask = 0xFu << (row_in_warp * 4);
    for (int s = 0; s <= s_max_warp; ++s) {
        if (s > t || s < s_min) {
            continue;
        }
        const float* k_s = k + ((size_t)s * kv_heads + kvh) * head_dim;
        const float* v_s = v + ((size_t)s * kv_heads + kvh) * head_dim;

        float kreg[DPL_MAX];
        float vreg[DPL_MAX];
        #pragma unroll
        for (int i = 0; i < DPL_MAX; ++i) {
            kreg[i] = (i < dpl) ? k_s[sub * dpl + i] : 0.0f;
            vreg[i] = (i < dpl) ? v_s[sub * dpl + i] : 0.0f;
        }

        float part = 0.0f;
        #pragma unroll
        for (int i = 0; i < DPL_MAX; ++i) {
            part += qreg[i] * kreg[i];
        }
        part += __shfl_xor_sync(mask, part, 1);
        part += __shfl_xor_sync(mask, part, 2);
        const float score = part * scale;

        const float m_new = fmaxf(m, score);
        const float alpha = __expf(m - m_new);
        const float p = __expf(score - m_new);
        l = l * alpha + p;
        m = m_new;

        #pragma unroll
        for (int i = 0; i < DPL_MAX; ++i) {
            acc[i] = acc[i] * alpha + p * vreg[i];
        }
    }
    if (!live) {
        return;
    }

    float* o_t = out + ((size_t)t * heads + h) * head_dim;
    const float inv = 1.0f / l;
    #pragma unroll
    for (int i = 0; i < DPL_MAX; ++i) {
        if (i < dpl) {
            o_t[sub * dpl + i] = acc[i] * inv;
        }
    }
}
