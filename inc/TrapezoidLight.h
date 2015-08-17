#ifndef TRAPEZOIDLIGHT_H
#define TRAPEZOIDLIGHT_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cutil_math.h>

#include "HitInfo.h"
#include "Ray.h"
#include "RandomState.h"

#include "Hemisphere.h"
#include "Sampler2D.h"

class TrapezoidLight {
public:
    __device__ inline bool traceShadow(Ray const& ray, float distance) const {
        // Assume no shadows are casted by the light.
        return false;
    }

    __device__ inline TrapezoidLight() {
        // TODO: a default ctor may not be a good idea
    }

    /** @param power In Watts. */
    __device__ inline TrapezoidLight(float3 const& pos, float3 const& b1, float3 const& b2, float3 const& power)
        : pos_(pos - b1 - b2), b1_(b1*2), b2_(b2*2) {
        normal_ = normalize(cross(b1, b2));

        u_bound_ = dot(b1_,b1_) + RAY_DISTANCE_EPSILON;
        v_bound_ = dot(b2_,b2_) + RAY_DISTANCE_EPSILON;

        area_ = length(cross(b1, b2));

        // For diffuse emitters we have power = radiance * area * pi
        // (AGI, Equation 2.12).

        radiance_ = power / (area_ * CUDART_PI_F);
    }

    __device__ inline float3 randomPosition(float u, float v) const {
        return pos_ + b1_ * u + b2_ * v;
    }

    __device__ inline float3 randomDirection(float u, float v) const {
        return Hemisphere::cosineWeightedDirection(normal_, u, v);
    }

    __device__ inline float3 power() const {
        return radiance_ * (area_ * CUDART_PI_F);
    }

    __device__ inline bool trace(Ray const& ray, HitInfo& info) const {
        // (rel_pos + ray.dir * t) * normal_ = 0

        float3 const rel_pos = ray.pos - pos_;

        // The single negation here is nothing compared to
        // the gains achieved by using multiply-and-add on rel_hit below.
        info.time = -dot(rel_pos, normal_) / dot(ray.dir, normal_);

        float3 const rel_hit = rel_pos + ray.dir * info.time;

        float const u = dot(rel_hit, b1_);
        float const v = dot(rel_hit, b2_);

        info.material.type = MT_EMITTING;

        // Profiling shows that this condition should be kept
        // (probably because there rarely is divergence:
        // the if-case is almost always the case).
        if(dot(rel_pos, normal_) >= 0) {
            info.normal = normal_;

            info.material.color = radiance_;
        } else {
            info.normal.x = -normal_.x;
            info.normal.y = -normal_.y;
            info.normal.z = -normal_.z;

            info.material.color = make_float3(0);
        }

        return info.time >= RAY_DISTANCE_EPSILON
            & u >= -RAY_DISTANCE_EPSILON & u <= u_bound_
            & v >= -RAY_DISTANCE_EPSILON & v <= v_bound_;
    }

    template <typename S, typename T>
    __device__ inline float3 sampleIrradiance(
                    float3 const& position, float3 const& normal,
                    S const& scene, RandomState& rng,
                    T& sampler, int const nSamples) const {
        float3 const relativePosition = pos_ - position;

        float const dot0 = -dot(normal_, relativePosition);

        if(dot0 < 0) {
            return make_float3(0);
        }

        // In area formulation, we have (AGI, Equation 2.12)
        //
        //   E = \int_{area} L(x <- w) V(x,y) G(x,y) dA_y,
        //
        // where
        //
        //   G(x,y) = cos(N_x, w) * cos(N_y, w) / r^2(x,y).
        //
        // Since L(x <- w) is constant, we only integrate
        // V(x,y) * G(x,y).

        float score = 0;

#pragma unroll
        for(int i=0; i<nSamples; ++i) {
            float u, v;
            sampler.next(u, v, i, rng);

            float3 const relativeLightPosition = b1_ * u + b2_ * v;

            float3 const direction = relativePosition + relativeLightPosition;

            // Let the compiler sort out the optimization for sqrtf and rsqrtf.
            float const sq_distance = dot(direction, direction);
            float const distance = sqrtf(sq_distance);
            float const inv_distance = rsqrtf(sq_distance);

            // dot1 is not scaled by inv_distance yet
            float const dot1 = dot(normal, direction);

            // multiplication by the constant dot2 = dot0 
            // (yet unscaled by inv_distance) is moved outside the loop 

            // We could continue if dot1 < 0, but
            // this would lead to thread divergence.

            Ray const ray(position, direction * inv_distance);

            float const contribution = dot1 * sq(sq(inv_distance));

            score += dot1 < 0 | scene.traceShadow(ray, distance - RAY_DISTANCE_EPSILON)
                ? 0 : contribution;
        }

        // The score was Monte Carlo integrated,
        // so we must multiply by \int_{area} dA_y = area
        // and divide by sample count.
        
        return radiance_ * (score * dot0 * area_ / nSamples);
    }

private:
    template <typename T>
    __device__ static inline T sq(T x) {
        return x * x;
    }

private:
    float3 pos_;

    float3 b1_;
    float3 b2_;

    // auxiliary variables

    float3 normal_;

    float3 radiance_;

    float area_;

    float u_bound_;
    float v_bound_;
};

#endif