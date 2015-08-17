#ifndef SAMPLER2D_H
#define SAMPLER2D_H

#include "RandomState.h"
#include "Halton.h"

class SukharevSampler2D {
public:
    __device__ SukharevSampler2D(int nSamples1D, RandomState& /*rng*/)
        : nSamples1D_(nSamples1D) {
        // ignored
    }

    __device__ void next(float& u, float& v, int index, RandomState& /*rng*/) {
        // The compiler is clever enough to optimize this part.
        u = ((index%nSamples1D_) + .5f) / nSamples1D_;
        v = ((index/nSamples1D_) + .5f) / nSamples1D_;
    }

private:
    int const nSamples1D_;
};

class StratifiedSampler2D {
public:
    __device__ StratifiedSampler2D(int nSamples1D, RandomState& /*rng*/) 
        : nSamples1D_(nSamples1D) {
        // ignored
    }

    __device__ void next(float& u, float& v, int index, RandomState& rng) {
        u = ((index%nSamples1D_) + rng.nextFloat()) / nSamples1D_;
        v = ((index/nSamples1D_) + rng.nextFloat()) / nSamples1D_;
    }

private:
    int const nSamples1D_;
};

#define LATTICE_PHI 1.6180339887498948482f

class LatticeSampler2D {
public:
    __device__ LatticeSampler2D(int nSamples1D, RandomState& rng) :
        invSamples2D_(1.0f/(nSamples1D * nSamples1D)), uOffset_(rng.nextFloat() * invSamples2D_), v_(rng.nextFloat()) {
        // ignored
    }

    __device__ void next(float& u, float& v, int index, RandomState& /*rng*/) {
        u = index * invSamples2D_ + uOffset_;
        v = v_ = fracf(v_ + LATTICE_PHI);
    }
private:
    float const invSamples2D_;
    float const uOffset_; // pre-scaled by invSamples2D_ to encourage multiply-and-add
    float v_;
};

class HaltonSampler2D {
public:
    __device__ HaltonSampler2D(int nSamples1D, RandomState& rng)
        : invSamples2D_(1.0f / (nSamples1D * nSamples1D)),
          offset_(rng.nextFloat() * invSamples2D_), halton_(rng.nextFloat()) {
        // ignored
    }

    __device__ void next(float& u, float& v, int index, RandomState& /*rng*/) {
        u = index * invSamples2D_ + offset_;
        v = halton_.nextFloat();
    }

private:
    float const invSamples2D_;
    float const offset_;
    Halton<2> halton_;
};

#endif