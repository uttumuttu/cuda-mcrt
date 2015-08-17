#ifndef RAY_H
#define RAY_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cutil_math.h>

#define RAY_DISTANCE_EPSILON 1e-4f

struct Ray {
    __device__ Ray(float3 pos_, float3 dir_) : pos(pos_), dir(dir_), inv_dir(make_float3(1) / dir) {
        // ignored
    }
    float3 pos;
    float3 dir;
    float3 inv_dir;
};

#endif