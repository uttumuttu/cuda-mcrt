#ifndef RANDOMSTATE_H
#define RANDOMSTATE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#ifndef __int_as_float
static inline float __int_as_float(int x) {
    return *((float*)&x);
}
#endif

class RandomState {
public:
    void setSeed(int seed) {
        state_ = seed;
    }

    __device__ float nextFloat() {
        state_ = state_ * 1664525 + 1013904223;

        return (unsigned int)(state_) / (float(1 << 16) * float(1 << 16));
    }

    __device__ int nextInt(int modulus) {
        state_ = state_ * 1664525 + 1013904223;

        return state_ % modulus;
    }

private:
    int state_;
};

#endif