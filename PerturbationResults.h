#pragma once

#include <vector>
#include <stdint.h>
#include "HighPrecision.h"

#include "HDRFloatComplex.h"

template<class T>
class PerturbationResults {
public:
    HighPrecision hiX, hiY, radiusX, radiusY;
    T maxRadius;
    size_t MaxIterations;
    size_t PeriodMaybeZero;  // Zero if not worked out

    std::vector<T> x;
    std::vector<T> y;
    std::vector<uint8_t> bad;

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;

    void clear() {
        hiX = {};
        hiY = {};
        radiusX = {};
        radiusY = {};
        maxRadius = {};
        MaxIterations = 0;
        PeriodMaybeZero = 0;

        x.clear();
        y.clear();
        bad.clear();

        AuthoritativePrecision = false;
        ReuseX.clear();
        ReuseY.clear();
    }

    template<class Other>
    void Copy(const PerturbationResults<Other>& other) {
        clear();

        x.reserve(other.x.size());
        y.reserve(other.y.size());
        bad.reserve(other.bad.size());

        ReuseX.reserve(other.ReuseX.size());
        ReuseY.reserve(other.ReuseY.size());

        for (size_t i = 0; i < other.x.size(); i++) {
            x.push_back((T)other.x[i]);
            y.push_back((T)other.y[i]);
        }

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX = other.ReuseX;
        ReuseY = other.ReuseY;

        bad = other.bad;
    }

    template<class U>
    HDRFloatComplex<U> GetComplex(size_t index) const {
        return { x[index], y[index] };
    }
};