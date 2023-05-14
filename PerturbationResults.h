#pragma once

#include <vector>
#include <stdint.h>
#include "HighPrecision.h"

class PerturbationResults {
public:
    HighPrecision hiX, hiY, radiusX, radiusY, maxRadius;
    size_t scrnX, scrnY;
    size_t MaxIterations;

    std::vector<double> x;
    std::vector<double> x2;
    std::vector<double> y;
    std::vector<double> y2;
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