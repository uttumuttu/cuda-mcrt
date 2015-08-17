#ifndef SPHERE_H
#define SPHERE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Ray.h"
#include "HitInfo.h"

#include <cutil_math.h>

#define USE_SPHERE_SHADOW_OPTIMIZATION 1

#define USE_SPHERE_SHADOW_EPSILON_OPTIMIZATION 1

class Sphere {
public:
#if USE_SPHERE_SHADOW_OPTIMIZATION
    __device__ bool traceShadow(Ray const& ray, float /*distance*/) const {
        // The optimization relies on the light source being
        // on the scene boundary.

        float3 const rel_pos = ray.pos - pos_;

        float const b = dot(rel_pos, ray.dir);
        float const c = dot(rel_pos, rel_pos) + neg_sq_radius_;

#if USE_SPHERE_SHADOW_EPSILON_OPTIMIZATION
        return b*b-c >= 0 & b < 0 & c >= 0;
#else
        float const D = b*b - c;

        return D >= 0 & RAY_DISTANCE_EPSILON+b <= 0 & D <= sq(RAY_DISTANCE_EPSILON+b);
#endif
    }
#else
    __device__ bool traceShadow(Ray const& ray, float distance) const {
        HitInfo info;

        bool hit = trace(ray, info);

        return hit & info.time <= distance;
    }
#endif

    __device__ Sphere() {
        // TODO: probably not a good idea to have a default ctor
    }

    __device__ Sphere(float3 const& pos, float radius, Material const& material) 
        : pos_(pos), neg_sq_radius_(-radius*radius), inv_radius_(1.0f / radius), material_(material) {
        // ignored
    }

    __device__ bool trace(Ray const& ray, HitInfo& info) const {
        // |rel_pos + ray_dir*t| = radius

        float3 const rel_pos = ray.pos - pos_;

        float const b = dot(rel_pos, ray.dir);
        float const c = dot(rel_pos, rel_pos) + neg_sq_radius_;

        float const d = sqrtf(b*b - c);

        float const t1 = -b-d;
        float const t2 = -b+d;

        info.time = t1 >= RAY_DISTANCE_EPSILON ? t1 : t2;

        info.material = material_;

        float3 const rel_hit = rel_pos + ray.dir * info.time;

        int const isInside = signbit(t1 - RAY_DISTANCE_EPSILON);

        info.normal = rel_hit * (inv_radius_ * (1-2*isInside));

        info.material.refractiveIndex = 
            info.material.refractiveIndex * (1-isInside) + isInside;

        return d >= 0 & info.time >= RAY_DISTANCE_EPSILON;
    }

private:
    template <typename T>
    __device__ static inline T sq(T x) {
        return x * x;
    }

private:
    float3 pos_;
    float neg_sq_radius_;
    float inv_radius_;
    Material material_;
};

#endif