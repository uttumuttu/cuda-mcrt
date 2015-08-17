#ifndef HEMISPHERE_H
#define HEMISPHERE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cutil_math.h>
#include <math_constants.h>

#include "Ray.h"

class Hemisphere {
private:
    __device__ static float3 rotate(
            float3 const& normal, 
            float x, float y, float const z) {
        float3 const right = fabsf(normal.x) <= .25f
            ? make_float3(1,0,0)
            : make_float3(0,1,0);

        float const w = dot(right, normal);

        float const scale = rsqrtf(1-w*w);

        x *= scale;
        y *= scale;

        return right * x
             + cross(right, normal) * y
             + normal * (z - w * x);
    }
public:

    /**
     * Creates a cosine-weighted direction above a hemisphere.
     *
     * @param normal Unit surface normal.
     * @param u A uniformly distributed surface parameter in [0, 1).
     * @param v A uniformly distributed surface parameter in [0, 1).
     */
    __device__ static float3 cosineWeightedDirection(
            float3 const& normal, float const u, float const v) {
        // See AGI page 66 for derivation.

        float const phi = u * (2 * CUDART_PI_F);

        float const sin_theta = sqrtf(1-v); // = sin(acos(sqrt(v)))
        float const cos_theta = sqrtf(v);   // = cos(acos(sqrt(v)))

        float const x = cosf(phi) * sin_theta;
        float const y = sinf(phi) * sin_theta;
        float const z = cos_theta;

        return rotate(normal, x,y,z);
    }

    __device__ static inline float fresnelCoefficient(
            float3 const& normal, float3 const& incidentDirection,
            float incidentRefractiveIndex,
            float excitantRefractiveIndex) {
        float n1 = incidentRefractiveIndex; // shorthands
        float n2 = excitantRefractiveIndex;

        // Fresnel coefficient at normal incidence 
        // (Wann Jensen, Equations 2.28 and 2.29)
        float f0 = square( (n1-n2) / (n1+n2) );

        float cosine = -dot(normal, incidentDirection);

        // Wann Jensen, Equation 2.30 (Schlick approximation).
        return f0 + (1 - f0) * quintic(1 - cosine);
    }

    /**
     * @param normal Unit surface normal (outwards).
     *
     * @param incidentDirection Incident direction (inside the surface).
     *
     * @param incidentRefractiveIndex Refractive index in the incident medium.
     *
     * @param excitantRefractiveIndex Refractive index in the excitant medium.
     *
     * @param excitantDirection
     *         Contains the excitant direction (inside the surface)
     *         if there was no total internal reflection; undefined otherwise.
     *
     * @return True if there was no total internal reflection;
     *         false if there was total internal reflection.
     */
    __device__ static inline bool computeRefraction(
            float3 const& normal, 
            float3 const& incidentDirection,
            float incidentRefractiveIndex,
            float excitantRefractiveIndex,
            float3& excitantDirection) {
        // See Wann Jensen Equation 2.32 for derivation.
        float n1 = incidentRefractiveIndex; // shorthands
        float n2 = excitantRefractiveIndex;

        // Relative index of refraction.
        float eta = n1 / n2;

        float cosine = -dot(normal, incidentDirection);

        float outOfPlaneCoefficient = -sqrtf(1 - square(eta) * (1 - square(cosine)));

        float inPlaneCoefficient = -eta;

        excitantDirection = 
            normal * (outOfPlaneCoefficient - cosine * inPlaneCoefficient)
            - incidentDirection * inPlaneCoefficient;

        return outOfPlaneCoefficient < 0;
    }

private:
    __device__ static inline float square(float x) {
        return x*x;
    }

    __device__ static inline float quintic(float x) {
        float xx = x*x;
        float xxx = xx*x;
        return xx*xxx;
    }

private: // prevent instantiation
    Hemisphere();
    Hemisphere(Hemisphere const&);
    Hemisphere& operator = (Hemisphere const&);
};

#endif