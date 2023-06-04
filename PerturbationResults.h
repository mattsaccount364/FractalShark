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

    template<class Other>
    void Copy(PerturbationResults<Other>& other) {
        clear();

        x.reserve(other.x.size());
        y.reserve(other.y.size());
        x2.reserve(other.x2.size());
        y2.reserve(other.y2.size());
        bad.reserve(other.bad.size());
        bad_counts.reserve(other.bad_counts.size());

        for (size_t i = 0; i < other.x.size(); i++) {
            x.push_back((T)other.x[i]);
            y.push_back((T)other.y[i]);
            x2.push_back((T)other.x2[i]);
            y2.push_back((T)other.y2[i]);
        }

        bad = other.bad;
        bad_counts = other.bad_counts;
    }
};