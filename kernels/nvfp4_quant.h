#ifndef AIDS_NVFP4_QUANT_H
#define AIDS_NVFP4_QUANT_H

#include <cuda_runtime.h>
#include <cstdint>

#ifndef CLEF_E2M1_FAST
#define CLEF_E2M1_FAST 0
#endif

__device__ __constant__ float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint8_t to_e2m1(float v) {
    const uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    const float a = fabsf(v);
#if CLEF_E2M1_FAST
    const uint8_t best = (uint8_t)((a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f)
                                 + (a > 2.5f) + (a > 3.5f) + (a > 5.0f));
    return (uint8_t)(sign | best);
#else
    uint8_t best = 0;
    float best_err = 1e30f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float err = fabsf(a - kE2M1[i]);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return (uint8_t)(sign | best);
#endif
}

__device__ __forceinline__ uint8_t to_e4m3(float v) {
    if (!(v > 0.0f)) {
        return 0;
    }

    int exp = 0;
    float m = frexpf(v, &exp);
    int e4 = exp + 6;
    float mant = (m * 2.0f - 1.0f) * 8.0f;
    int mi = (int)(mant + 0.5f);
    if (mi == 8) {
        mi = 0;
        ++e4;
    }
    if (e4 <= 0) {
        const int sub = (int)(v * 512.0f + 0.5f);
        return (uint8_t)(sub > 7 ? 7 : sub);
    }
    if (e4 > 15 || (e4 == 15 && mi > 6)) {
        return 0x7E;
    }
    return (uint8_t)((e4 << 3) | (mi & 0x7));
}

__device__ __forceinline__ float from_e4m3(uint8_t b) {
    const int e = (b >> 3) & 0x0F;
    const float mant = (float)(b & 0x07);
    if (e == 0) {
        return mant * (1.0f / 512.0f);
    }
    return (1.0f + mant / 8.0f) * exp2f((float)(e - 7));
}

#endif
