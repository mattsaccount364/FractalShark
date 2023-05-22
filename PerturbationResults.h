#pragma once

#include <vector>
#include <stdint.h>
#include "HighPrecision.h"

template<class T>
class PerturbationResults {
public:
    HighPrecision hiX, hiY, radiusX, radiusY, maxRadius;
    size_t scrnX, scrnY;
    size_t MaxIterations;

    std::vector<T> x;
    std::vector<T> x2;
    std::vector<T> y;
    std::vector<T> y2;
    std::vector<uint8_t> bad;
    std::vector<uint32_t> bad_counts;

    void clear() {
        x.clear();
        y.clear();
        x2.clear();
        y2.clear();
        bad.clear();
        bad_counts.clear();
        hiX = {};
        hiY = {};
        radiusX = {};
        radiusY = {};
        maxRadius = {};
    }
};