#ifndef HALTON_H
#define HALTON_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HALTON_UNITY float(1 - 0.0000001)

template <int B>
class Halton {
public:
    __device__ explicit Halton(float seed) : value_(seed) {
        // ignored
    }

    __device__ float nextFloat() {
        float const r = HALTON_UNITY - value_;

        float const n = floorf(logf(r) * (1.0f / logf(B)));
        float const h = powf(B, n);

        return value_ += fminf(h * (1+B) - 1, 1.0f/B);
    }

private:
    float value_;
};

template <>
__device__ float Halton<2>::nextFloat() {
    float const r = HALTON_UNITY - value_;

    float const h = __int_as_float(__float_as_int(r) & 0x7F800000);

    return value_ += fminf(3*h-1, .5f);
}

#endif