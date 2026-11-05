#include <cuda_runtime.h>
#include <cstdint>

#define V_SPLIT 4

#define MAX_ROWS 16

#define I_SPLIT 8
#define BLOCK_THREADS (32 * I_SPLIT)

extern "C" __global__ void gdn_scan(const float* __restrict__ q_ptr,
                                    const float* __restrict__ k_ptr,
                                    const float* __restrict__ v_ptr,
                                    const float* __restrict__ g_ptr,
                                    const float* __restrict__ beta_ptr,
                                    float* __restrict__ out,
                                    int seq_len, int num_heads, int k_dim, int v_dim) {
    const int head = blockIdx.x;
    const int vsplit = blockIdx.y;
    const int j_local = threadIdx.x;
    const int iq = threadIdx.y;

    const int j_global = vsplit * (v_dim / V_SPLIT) + j_local;
    const int j_count = v_dim / V_SPLIT;
    const int rows = k_dim / I_SPLIT;
    const int i0 = iq * rows;

    __shared__ float s_k[MAX_ROWS * I_SPLIT];
    __shared__ float s_q[MAX_ROWS * I_SPLIT];
    __shared__ float s_v[128];
    __shared__ float s_kv[128];
    __shared__ float s_part[I_SPLIT][128];
    __shared__ float s_g, s_beta;

    float s[MAX_ROWS];
    #pragma unroll
    for (int r = 0; r < MAX_ROWS; ++r) {
        if (r < rows) {
            s[r] = 0.0f;
        }
    }

    const size_t head_stride_k = (size_t)num_heads * k_dim;
    const size_t head_stride_v = (size_t)num_heads * v_dim;
    for (int t = 0; t < seq_len; ++t) {
        if (iq == 0) {
            if (j_local == 0) {
                s_g = g_ptr[t * num_heads + head];
                s_beta = beta_ptr[t * num_heads + head];
            }
            s_v[j_local] = v_ptr[(size_t)t * head_stride_v + (size_t)head * v_dim + j_global];
        }
        for (int r = threadIdx.y * 32 + threadIdx.x; r < k_dim; r += BLOCK_THREADS) {
            s_k[r] = k_ptr[(size_t)t * head_stride_k + (size_t)head * k_dim + r];
            s_q[r] = q_ptr[(size_t)t * head_stride_k + (size_t)head * k_dim + r];
        }
        __syncthreads();

        {
            float nk = 0.0f, nq = 0.0f;
            for (int r = iq * (k_dim / I_SPLIT); r < (iq + 1) * (k_dim / I_SPLIT); ++r) {
                nk += s_k[r] * s_k[r];
                nq += s_q[r] * s_q[r];
            }
            s_part[iq][j_local] = nk;
            s_part[iq][32 + j_local] = nq;
            __syncthreads();
            if (iq == 0) {
                float tk = 0.0f, tq = 0.0f;
                for (int p = 0; p < I_SPLIT; ++p) {
                    tk += s_part[p][j_local];
                    tq += s_part[p][32 + j_local];
                }
                s_part[0][j_local] = rsqrtf(tk + 1e-6f);
                s_part[0][32 + j_local] = rsqrtf(tq + 1e-6f) / sqrtf((float)k_dim);
            }
            __syncthreads();
            const float rk = s_part[0][j_local];
            const float rq = s_part[0][32 + j_local];
            for (int r = iq * (k_dim / I_SPLIT); r < (iq + 1) * (k_dim / I_SPLIT); ++r) {
                s_k[r] *= rk;
                s_q[r] *= rq;
            }
            __syncthreads();
        }

        const float decay = __expf(s_g);
        #pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) {
            if (r < rows) {
                s[r] *= decay;
            }
        }

        {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < MAX_ROWS; ++r) {
                if (r < rows) {
                    acc += s[r] * s_k[i0 + r];
                }
            }
            s_part[iq][j_local] = acc;
        }
        __syncthreads();
        if (iq == 0) {
            float acc = 0.0f;
            for (int p = 0; p < I_SPLIT; ++p) {
                acc += s_part[p][j_local];
            }
            s_kv[j_local] = acc;
        }
        __syncthreads();

        const float delta = (s_v[j_local] - s_kv[j_local]) * s_beta;
        #pragma unroll
        for (int r = 0; r < MAX_ROWS; ++r) {
            if (r < rows) {
                s[r] += s_k[i0 + r] * delta;
            }
        }

        {
            float acc = 0.0f;
            #pragma unroll
            for (int r = 0; r < MAX_ROWS; ++r) {
                if (r < rows) {
                    acc += s[r] * s_q[i0 + r];
                }
            }
            s_part[iq][j_local] = acc;
        }
        __syncthreads();
        if (iq == 0) {
            float acc = 0.0f;
            for (int p = 0; p < I_SPLIT; ++p) {
                acc += s_part[p][j_local];
            }
            out[(size_t)t * head_stride_v + (size_t)head * v_dim + j_global] = acc;
        }
        __syncthreads();
    }
}
