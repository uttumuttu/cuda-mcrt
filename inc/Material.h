#ifndef MATERIAL_H
#define MATERIAL_H

enum MaterialType {
    MT_EMITTING,
    MT_DIFFUSE,
    MT_SPECULAR,
    MT_DIELECTRIC,
};

class Material {
public:
    __device__ Material() {
        // ignored
    }

private:
    __device__ Material(MaterialType const& type_, float3 const& color_) 
        : type(type_), color(color_) {
        // ignored
    }

    __device__ Material(MaterialType const& type_, float3 const& color_, float refractiveIndex_)
        : type(type_), color(color_), refractiveIndex(refractiveIndex_) {
        // ignored
    }

public:
    __device__ static Material emitting(float3 const& radiance) {
        return Material(MT_EMITTING, radiance);
    }

    __device__ static Material diffuse(float3 const& reflectance) {
        return Material(MT_DIFFUSE, reflectance);
    }

    __device__ static Material specular(float3 const& reflectance) {
        return Material(MT_SPECULAR, reflectance);
    }

    __device__ static Material dielectric(float3 const& reflectance, float refractiveIndex) {
        return Material(MT_DIELECTRIC, reflectance, refractiveIndex);
    }

public:
    MaterialType type;

    /**
     * For non-emitting materials, the color
     * represents reflectance (in zero-one range).
     *
     * For emitting materials, the color
     * represents radiance (assumed constant in
     * all directions).
     */
    float3 color;

    /** Only meaningful for dielectric materials. */
    float refractiveIndex;
};

#endif