#include "Photonmap.h"
#include "Scene.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>

namespace {

__device__ static inline void computeRefraction(Ray& ray, HitInfo const& info, float& currentRefractiveIndex) {
    float3 reflectedDirection = reflect(ray.dir, info.normal);

    float3 refractedDirection;

    int noTotalInternalReflection = 
        Hemisphere::computeRefraction(
            info.normal, ray.dir,
            currentRefractiveIndex, info.material.refractiveIndex,
            refractedDirection);

    float fresnelCoefficient = Hemisphere::fresnelCoefficient(
            info.normal, ray.dir,
            currentRefractiveIndex,
            info.material.refractiveIndex) * noTotalInternalReflection + (1-noTotalInternalReflection);

    float3 const hitPosition = ray.pos + ray.dir * info.time;

    if(fresnelCoefficient > .5f) {
        ray = Ray(hitPosition, reflectedDirection);
    } else {
        ray = Ray(hitPosition, refractedDirection);

        currentRefractiveIndex = info.material.refractiveIndex;
    }
}

template <typename C, typename S, int N>
static __device__ void tracePhotonPath(
        float3 position, float3 direction, float3 power,
        C const& causticObjects,
        S const& scene, PhotonList<N>& list, RandomState& rng) {
    Ray ray(position, direction);
    HitInfo info;

    float currentRefractiveIndex = 1;

    bool hit = causticObjects.trace(ray, info);

    if(hit & info.material.type == MT_DIELECTRIC) {
        computeRefraction(ray, info, currentRefractiveIndex);

        for(int i=1; i<N; ++i) {
            if(scene.trace(ray, info)) {
                float3 const hitPosition = ray.pos + ray.dir * info.time;

                if(info.material.type == MT_DIFFUSE) {
                    list.addUnchecked(Photon(hitPosition, info.normal, power));
                    return;
                } else if(info.material.type == MT_DIELECTRIC) {
                    computeRefraction(ray, info, currentRefractiveIndex);
                } else {
                    return;
                }
            } else {
                return;
            }
        }
    }
}

} // anonymous namespace

namespace {

template <typename T>
__host__ __device__ static inline T sq(T x) {
    return x*x;
}

} // anonymous namespace

static __global__ void photonmappingKernel(int nPhotonPaths, int nSamples1D, DefaultPhotonList* lists, RandomState* rngs, float time) {
    __shared__ Scene scene;

    scene.setup(time);

    int index = blockIdx.x*blockDim.x + threadIdx.x;

    int const nSamples0D = 1;
    int const nSamples2D = sq(nSamples1D);
    int const nSamples3D = nSamples2D * nSamples1D;
    int const nSamples4D = sq(nSamples2D);

    if(index < nPhotonPaths) {
        DefaultPhotonList list; // local list
        RandomState rng = rngs[index]; // local prng

        if(index < nSamples4D) {
            float u = (((index / nSamples0D) % nSamples1D) + 0.5f) / nSamples1D;
            float v = (((index / nSamples1D) % nSamples1D) + 0.5f) / nSamples1D;

            float s = (((index / nSamples2D) % nSamples1D) + 0.5f) / nSamples1D;
            float t = (((index / nSamples3D) % nSamples1D) + 0.5f) / nSamples1D;

            tracePhotonPath(
                scene.getLight().randomPosition(u, v),
                scene.getLight().randomDirection(s, t),
                scene.getLight().power() / nPhotonPaths, 
                scene.getCausticObjects(),
                scene.getScene(), list, rng);
        }

        lists[index] = list;
        rngs[index] = rng;
    }
}

namespace {
inline int idiv_ceil(int x, int y) {
    return x/y + ((x%y) ? 1:0);
}

static void checkCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
}

void photonmapping(int nPhotonPaths, DefaultPhotonList* lists, RandomState* rngs, float timeSeconds) {
    int nSamples1D = floor(sqrt(sqrt(double(nPhotonPaths))));

    dim3 nBlocks(max(1, idiv_ceil(nPhotonPaths, 256)), 1);
    dim3 nThreads(256,1);

    cudaFuncSetCacheConfig(photonmappingKernel, cudaFuncCachePreferL1);

    photonmappingKernel<<< nBlocks, nThreads >>>(nPhotonPaths, nSamples1D, lists, rngs, timeSeconds);

    cudaThreadSynchronize();
    checkCUDAError("kernel failed!");
}
