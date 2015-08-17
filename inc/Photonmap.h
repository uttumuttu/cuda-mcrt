#ifndef PHOTONMAP_H
#define PHOTONMAP_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cutil_math.h>
#include <math_constants.h>

#define MAX_PHOTON_PATH_LENGTH 4
#define PHOTONMAP_GRID_SIZE 16
#define PHOTONMAP_LIST_LENGTH 1024

/** A photon hit. */
struct Photon {
    /** Creates an uninitialized photon. */
    __device__ Photon() {
        // ignored
    }

    __device__ Photon(float3 const& pos_, float3 const& normal_, float3 const& power_)
        : pos(pos_), normal(normal_), power(power_) {
        // ignored
    }

    float3 pos;
    float3 normal;
    float3 power;
};

/**
 * @param M Maximum number of photons in the list.
 */
template <int M>
class PhotonList {
public:
    /** Creates an empty photon list. */
    __host__ __device__ inline PhotonList() : size_(0) {
        // ignored
    }

    inline void clear() {
        size_ = 0;
    }

    /**
     * Adds a photon to the list if there is space.
     */
    inline void addChecked(Photon const& photon) {
        if(size_ < M) {
            photons_[size_++] = photon;
        }
    }

    /** 
     * Adds a photon to the list without checking
     * for space (for efficiency).
     */
    __device__ inline void addUnchecked(Photon const& photon) {
        photons_[size_++] = photon;
    }

    __host__ __device__ inline int size() const {
        return size_;
    }

    /** Does not check bounds. */
    __host__ __device__ inline Photon const& operator [] (int index) const {
        return photons_[index];
    }
private:
    int size_;
    Photon photons_[M];
};

/**
 * A three-dimensional photon map fixed to space
 * [-1,1]^3.
 *
 * @param N Number of lists along a single dimension.
 * @param M Maximum number of photons in a single list.
 */
template <int N, int M>
class PhotonMap {
public:
    template <int K>
    void construct(int nSrcLists, PhotonList<K>* srcLists) {
        for(int i=0; i<N*N*N; ++i) {
            lists_[i].clear();
        }

        int srcTotal = 0;
        int dstTotal = 0;

        for(int i=0; i<nSrcLists; ++i) {
            PhotonList<K> const& srcList = srcLists[i];

            int const nPhotons = srcList.size();

            srcTotal += nPhotons;

            for(int j=0; j<nPhotons; ++j) {
                Photon const& photon = srcList[j];

                lists_[worldToIndex(photon.pos)].addChecked(photon);
            }
        }

        for(int i=0; i<N*N*N; ++i) {
            dstTotal += lists_[i].size();
        }

        printf("photon map of %d photons created from %d photons (loss: %d percents)\n", 
            dstTotal, srcTotal, (int)(100 * (srcTotal-dstTotal)/float(srcTotal)));
    }

    __device__ float3 powerDensity(float3 const& pos, float3 const& normal, float const h) const {
        float3 score = make_float3(0);

        int3 const min = worldToGrid(pos - h - 1e-3); // TODO: no magic numbers
        int3 const max = worldToGrid(pos + h + 1e-3);

        float const h_sq = h*h;
        float const h_sq_inv = 1 / h_sq;

        for(int z=min.z; z<=max.z; ++z) {
            for(int y=min.y; y<=max.y; ++y) {
                for(int x=min.x; x<=max.x; ++x) {
                    PhotonList<M> const& list = lists_[gridToIndex(make_int3(x,y,z))];

                    int nPhotons = list.size();

                    for(int i=0; i<nPhotons; ++i) {
                        Photon const& photon = list[i];

                        float3 const delta = pos - photon.pos;

                        float const weight = 1 - dot(delta,delta) * h_sq_inv;

                        float const cosine = dot(normal, photon.normal);

                        float3 const contribution1 = photon.power * weight;
                        float3 const contribution2 = make_float3(0);

                        score += weight >= 0 & cosine >= 0
                            ? contribution1 : contribution2;
                    }
                }
            }
        }
        // The factor 0.5f comes frmo the Epanechnikov filter.
        return score / (0.5f * CUDART_PI_F * h_sq);
    }

private:
    __host__ __device__ int3 worldToGrid(float3 const& pos) const {
        return clamp(
            make_int3(N * (1 + pos) / 2),
            make_int3(0), make_int3(N-1));
    }

    __host__ __device__ int gridToIndex(int3 const& pos) const {
        return pos.z*N*N + pos.y*N + pos.x;
    }

    __host__ __device__ int worldToIndex(float3 const& pos) const {
        return gridToIndex(worldToGrid(pos));
    }

private:
    PhotonList<M> lists_[N*N*N];
};

typedef PhotonList<MAX_PHOTON_PATH_LENGTH> DefaultPhotonList;
typedef PhotonMap<PHOTONMAP_GRID_SIZE, PHOTONMAP_LIST_LENGTH> DefaultPhotonMap;

#endif