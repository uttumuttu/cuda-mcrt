#ifndef SCENE_H
#define SCENE_H

#include "Box.h"
#include "Sphere.h"
#include "TrapezoidLight.h"
#include "Composite.h"

class Scene {
public:
    __device__ void setup(float time) {
        time *= 0.5f;

        float rad1 = time;
        float rad2 = time + (2*CUDART_PI_F/3);
        float rad3 = time + (2*CUDART_PI_F/3*2);

        sphere1_ = Sphere(
            make_float3(.5f*cosf(rad1), -.25f+.4f*cosf(time*1.5f), .5f*sinf(rad1)), .35f,
            Material::diffuse(make_float3(1,1,.3f)));

        sphere2_ = Sphere(
            make_float3(.5f*cosf(rad2), -.25f+.4f*sinf(time*1.3f), .5f*sinf(rad2)), .35f,
            Material::specular(make_float3(1)));

        sphere3_ = Sphere(
            make_float3(.5f*cosf(rad3), -.25f+.4f*cosf(time*1.1f), .5f*sinf(rad3)), .35f,
            Material::dielectric(make_float3(1), 1.6));

        light_ = TrapezoidLight(
            make_float3(0,1 - RAY_DISTANCE_EPSILON,0),
            make_float3(0.3f,0,0),
            make_float3(0,0,0.3f),
            make_float3(60));

        eyePosition_ = make_float3(cosf(time*0.9f), sinf(time*0.8f), 2.2f);
    }

    /**
     * @return The set of objects that matters on the first shooting
     *         of a caustic photon ray.
     */
    __device__ Sphere getCausticObjects() const {
        return sphere3_;
    }

    __device__ Composite<Composite<TrapezoidLight, Box>, Composite<Sphere, Composite<Sphere, Sphere> > > getScene() const {
        return cons(cons(light_, box_), cons(sphere1_, cons(sphere2_, sphere3_)));
    }

    __device__ float3 getEyePosition() const {
        return eyePosition_;
    }

    __device__ TrapezoidLight const& getLight() const {
        return light_;
    }

private:
    __align__(16) TrapezoidLight light_;
    __align__(16) Sphere sphere1_;
    __align__(16) Sphere sphere2_;
    __align__(16) Sphere sphere3_;
    __align__(16) Box box_;
    __align__(16) float3 eyePosition_;
};

#endif