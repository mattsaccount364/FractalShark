#pragma once

#include "HighPrecision.h"

#pragma pack(push, 4)
struct MattDblflt {
    using TemplateSubType = MattDblflt;
    MattDblflt() = default;

    CUDA_CRAP
    MattDblflt(const MattDblflt &other) :
        head{ other.head },
        tail{ other.tail } {
        static_assert(sizeof(*this) == 8, "!");
    }

    CUDA_CRAP
    MattDblflt(float head, float tail) :
        head{ head },
        tail{ tail } {
    }

#if !defined(__CUDA_ARCH__)
    CUDA_CRAP
    explicit MattDblflt(double other) :
        head{ (float)other },
        tail{ (float)(other - (double)(float)other) } {
    }
#endif

    //CUDA_CRAP
    //explicit MattDblflt(double other) {

    //    auto a = (float)(other - (double)(float)other);
    //    auto b = (float)other;

    //    float t1, t2;
    //    tail = a + b;
    //    t1 = tail - a;
    //    t2 = tail - t1;
    //    t1 = b - t1;
    //    t2 = a - t2;
    //    head = t1 + t2;
    //}

    CUDA_CRAP
    explicit MattDblflt(float other) :
        head{ other },
        tail{ 0 } {
    }

    float head; // head / most significant bits
    float tail; // tail / least significant bits
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