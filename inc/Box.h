#ifndef BOX_H
#define BOX_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Ray.h"
#include "HitInfo.h"

#include <cutil_math.h>
#include <math_constants.h>

class Box {
public:
    __device__ bool traceShadow(Ray const& /*ray*/, float /*distance*/) const {
        // We assume that the box cannot cast shadows (i.e., all sources
        // of light are inside the box), resulting in a massive optimization.

        return false;
    }

    __device__ bool trace(Ray const& ray, HitInfo& info) const {
        float3 const corner = sgn(ray.dir);

        // TODO: refactor for multiply-and-add
        float3 t = (corner - ray.pos) * ray.inv_dir;

        t.x = t.x < RAY_DISTANCE_EPSILON ? CUDART_INF_F : t.x;
        t.y = t.y < RAY_DISTANCE_EPSILON ? CUDART_INF_F : t.y;
        t.z = t.z < RAY_DISTANCE_EPSILON ? CUDART_INF_F : t.z;

        info.time = fminf(t.x, fminf(t.y, t.z));

        info.material.type = MT_DIFFUSE;
        info.normal = make_float3(0);

        float3 const p = ray.pos + ray.dir * info.time;

        // XXX: direct comparison against CUDART_INF_F is faster
        //      than !isinf
        bool const inside = p.x*p.x-(1 + RAY_DISTANCE_EPSILON)<0
                          & p.y*p.y-(1 + RAY_DISTANCE_EPSILON)<0
                          & p.z*p.z-(1 + RAY_DISTANCE_EPSILON)<0 & info.time < CUDART_INF_F;

        // XXX: Do not remove the common return statement;
        //      it will result in slightly lower performance.
        if(t.x == info.time) {
            info.normal.x = -corner.x;
            float c = corner.x * 0.5f + 0.5f;
            info.material.color = make_float3(1-c, c*0.4f, 0.3f + c*0.6f);
            return inside;
        } else if(t.y == info.time) {
            info.normal.y = -corner.y;
            info.material.color = make_float3(1);
            return inside;
        } else {
            info.normal.z = -corner.z;
            info.material.color = make_float3(1);
            return inside;
        }
    }

private:
    __device__ static inline float3 sgn(float3 u) {
        return make_float3(
            copysignf(1,u.x),
            copysignf(1,u.y),
            copysignf(1,u.z));
    }
};

#endif