#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cutil_math.h>

#include <cstdio>
#include <cstdlib>

#include <cmath>

#define MAX_BOUNCES 16

#define PHOTON_RADIUS .15f

#define SHUTTER_TIME_SECONDS 0.01f

#define EXPOSURE_CONSTANT 10

#define USE_MORTON_ORDER 0

#include "RandomState.h"
#include "Halton.h"

#include "Ray.h"
#include "Material.h"
#include "HitInfo.h"

#include "Hemisphere.h"
#include "Sampler2D.h"
#include "Photonmap.h"

#include "Scene.h"

namespace {

template <typename S, typename L, typename T>
static __device__ float3 sampleIncomingDirectRadiance(
        float3 const& pos, float3 const& normal,
        S const& scene, L const& light, RandomState& rng,
        int const nSamples1D) {
    T sampler(nSamples1D, rng);

    float3 const directIrradiance = light.sampleIrradiance(
            pos, normal, scene, rng, sampler,
            nSamples1D * nSamples1D);

    return directIrradiance / CUDART_PI_F; // AGI Equation 2.12
}

template <typename T>
__device__ static inline T sq(T x) {
    return x * x;
}

template <typename S, typename L>
static __device__ float3 sampleIncidentRadiance(
        Ray& ray, S const& scene, L const& light, DefaultPhotonMap const* photonMap, RandomState& rng) {
    HitInfo info;

    float3 cumulatedReflectance = make_float3(1);

    float currentRefractiveIndex = 1;

    for(int nBounces=0; nBounces<MAX_BOUNCES; ++nBounces) {
        if( scene.trace(ray, info) ) {
            float3 const hitPosition = ray.pos + ray.dir * info.time;

            if(info.material.type == MT_EMITTING) {
                return cumulatedReflectance * info.material.color; // emitted radiance
            } else if(info.material.type == MT_DIFFUSE) {
                float3 const hitReflectance = info.material.color;

                float3 const hitNormal = info.normal;

                float3 const incomingDirectRadiance = sampleIncomingDirectRadiance<S, L, LatticeSampler2D>(
                        hitPosition, hitNormal, scene, light, rng, 5);

                float3 score = make_float3(0);

                int const nSamples1D = 6;
                int const nSamples2D = sq(nSamples1D);

                LatticeSampler2D sampler(nSamples1D, rng);

                for(int i=0; i<nSamples2D; ++i) {
                    float u, v;
                    sampler.next(u, v, i, rng);

                    ray = Ray(
                        hitPosition,
                        Hemisphere::cosineWeightedDirection(hitNormal, u, v));

                    bool const hit = scene.trace(ray, info);

                    if(hit & info.material.type == MT_DIFFUSE) {
                        score += sampleIncomingDirectRadiance<S, L, SukharevSampler2D>(
                            ray.pos + ray.dir * info.time, info.normal,
                            scene, light, rng, 2) * info.material.color;
                    }
                }

                float3 const incomingIndirectRadiance = (score * cumulatedReflectance) / nSamples2D;

                //float3 const causticsRadiance = photonMap->powerDensity(hitPosition, hitNormal, PHOTON_RADIUS) / CUDART_PI_F;

                float3 const causticsRadiance = make_float3(0);

                return (incomingDirectRadiance + incomingIndirectRadiance + causticsRadiance) * hitReflectance;
            } else if(info.material.type == MT_SPECULAR) {
                cumulatedReflectance *= info.material.color;

                ray = Ray(hitPosition, reflect(ray.dir, info.normal));
            } else {
                cumulatedReflectance *= info.material.color;

                float3 reflectedDirection = reflect(ray.dir, info.normal);

                float3 refractedDirection;

                int noTotalInternalReflection = 
                    Hemisphere::computeRefraction(
                        info.normal, ray.dir,
                        currentRefractiveIndex, info.material.refractiveIndex,
                        refractedDirection);

                // TODO: We currently cannot "bifurcate" paths,
                //       so we'll trace the path (refraction or
                //       reflection) with the highest weight.

                float fresnelCoefficient = Hemisphere::fresnelCoefficient(
                        info.normal, ray.dir,
                        currentRefractiveIndex,
                        info.material.refractiveIndex) * noTotalInternalReflection + (1-noTotalInternalReflection);

                if(fresnelCoefficient > .5f) {
                    cumulatedReflectance *= fresnelCoefficient;

                    ray = Ray(hitPosition, reflectedDirection);
                } else {
                    cumulatedReflectance *= (1 - fresnelCoefficient);

                    ray = Ray(hitPosition, refractedDirection);

                    currentRefractiveIndex = info.material.refractiveIndex;
                }
            }
        } else {
            return make_float3(0);
        }
    }
    return make_float3(0);
}

static inline __device__ float3 toneMap(float3 const& radiance) {
    float3 const irradiation = radiance * (SHUTTER_TIME_SECONDS * 2 * CUDART_PI_F);

    float3 tone = irradiation * (-EXPOSURE_CONSTANT);

    tone.x = expf(tone.x);
    tone.y = expf(tone.y);
    tone.z = expf(tone.z);

    return 255 + tone * (-255);
}

template <typename S, typename L>
static __device__ float3 pixelTone(
        int x, int y, int width, int height,
        float3 eyePos,
        S const& scene, L const& light, DefaultPhotonMap const* photonMap, RandomState& rng) {
    float3 front = normalize(-eyePos);
    float3 up = make_float3(0,1,0);

    up = normalize(up - front*dot(front,up));

    float3 right = cross(front, up);

    float3 tone = make_float3(0);

    int const sampleCount = 1;

    for(int i=0; i<sampleCount; ++i) {
        float u = ((i&1) + .5f) / 1;
        float v = ((i>>1) + .5f) / 1;

        u = 2*(x + u) / width - 1;
        v = 2*(y + v) / height - 1;

        v *= (-3.0f/4.0f);

        Ray eye(
            eyePos, normalize(right * u + up * v + front));

        tone += toneMap(sampleIncidentRadiance(eye, scene, light, photonMap, rng));
    }

    return tone / sampleCount;
}

static inline int idiv_ceil(int x, int y) {
    return x/y + ((x%y) ? 1:0);
}

static void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#if USE_MORTON_ORDER
struct MortonOrder {
public:
    __device__ static void indexToCoordinate(int index, int& x, int& y) {
        x = pack((index >> 1) & 0x55555555);
        y = pack( index       & 0x55555555);
    }

private:
    __device__ static int pack(int n) {
        int x = 0;
#pragma unroll
        for(int i=0; i<16; ++i) {
            x |= ((n >> (2*i)) & 1) << i;
        }
        return x;
    }
};

static unsigned int nextPow2(unsigned int v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}
#else
#define BLOCK_WIDTH 5
#define BLOCK_HEIGHT 32
#endif

} // anonymous namespace

static __global__ void raytracingKernel(
        uchar4* color, DefaultPhotonMap const* photonMap, RandomState* random,
        int width, int height, float timeSeconds) {
#if USE_MORTON_ORDER
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int x, y;

    MortonOrder::indexToCoordinate(index, x, y);
#else
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
#endif

    __shared__ Scene scene;

    scene.setup(timeSeconds);

    if(x < width & y < height) {
        int const index = y * width + x;

        int const x = index % width;
        int const y = index / width;

        RandomState rng(random[index]); // create a local copy

        rng = random[index];

        float3 tone = pixelTone(x, y, width, height, 
            scene.getEyePosition(),
            scene.getScene(), scene.getLight(), photonMap, rng);

        random[index] = rng;

        color[index] = make_uchar4(tone.x, tone.y, tone.z, 0);
    }
}

void raytracing(
        uchar4* color, DefaultPhotonMap const* photonMap, RandomState* random,
        unsigned int image_width, unsigned int image_height, float timeSeconds) {
#if USE_MORTON_ORDER
    unsigned int temp = max(image_width, image_height);

    if(temp && (temp & (temp-1))) {
        temp = nextPow2(temp);
    }

    printf("morton size: %dx%d\n", temp, temp);

    dim3 nBlocks(idiv_ceil(temp*temp, 192), 1);
    dim3 nThreads(192, 1);
#else
    dim3 nBlocks (idiv_ceil(image_width, BLOCK_WIDTH), idiv_ceil(image_height, BLOCK_HEIGHT));
    dim3 nThreads(BLOCK_WIDTH, BLOCK_HEIGHT);
#endif
    cudaFuncSetCacheConfig(raytracingKernel, cudaFuncCachePreferL1);

    raytracingKernel<<< nBlocks, nThreads >>>(color, photonMap, random, image_width, image_height, timeSeconds);

    cudaThreadSynchronize();
    checkCUDAError("kernel failed!");
}
