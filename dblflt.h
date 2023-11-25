#pragma once

#include "HighPrecision.h"

#pragma pack(push, 4)
struct MattDblflt {
    using TemplateSubType = MattDblflt;
    MattDblflt() = default;

    CUDA_CRAP
    MattDblflt(const MattDblflt &other) :
        x{ other.x },
        y{ other.y } {
        static_assert(sizeof(*this) == 8, "!");
    }

    CUDA_CRAP
    MattDblflt(float x, float y) :
        x{ x },
        y{ y } {
    }

    CUDA_CRAP
    explicit MattDblflt(double other) :
        x{ (float)(other - (double)(float)other) },
        y{ (float)other } {
    }

    CUDA_CRAP
    explicit MattDblflt(float other) :
        x{ 0 },
        y{ other } {
    }

    float x; // head
    float y; // tail
};

using dblflt = MattDblflt;
#pragma pack(pop)


//#ifndef __CUDACC__
//using dblflt = MattDblflt;
//#endif

struct MattQFltflt {
    float x; // MSB
    float y;
    float z;
    float w; // LSB
};

struct MattDbldbl {
    double head;
    double tail;
};

struct MattQDbldbl {
    double x; // MSB
    double y;
    double z;
    double w; // LSB
};