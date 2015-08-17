#ifndef HITINFO_H
#define HITINFO_H

#include "Material.h"

struct HitInfo {
    float time;
    float3 normal;
    Material material;
};

#endif