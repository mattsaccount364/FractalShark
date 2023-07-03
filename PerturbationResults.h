#pragma once

#include <vector>
#include <stdint.h>
#include "HighPrecision.h"

template<class T>
class PerturbationResults {
public:
    HighPrecision hiX, hiY, radiusX, radiusY;
    T maxRadius;
    size_t MaxIterations;

    std::vector<T> x;
    std::vector<T> x2;
    std::vector<T> y;
    std::vector<T> y2;
    std::vector<uint8_t> bad;

    bool m_Periodic;

    void clear() {
        x.clear();
        y.clear();
        x2.clear();
        y2.clear();
        bad.clear();
        hiX = {};
        hiY = {};
        radiusX = {};
        radiusY = {};
        maxRadius = {};
        m_Periodic = {};
    }

    template<class Other>
    void Copy(PerturbationResults<Other>& other) {
        clear();

        x.reserve(other.x.size());
        y.reserve(other.y.size());
        x2.reserve(other.x2.size());
        y2.reserve(other.y2.size());
        bad.reserve(other.bad.size());

        for (size_t i = 0; i < other.x.size(); i++) {
            x.push_back((T)other.x[i]);
            y.push_back((T)other.y[i]);
            x2.push_back((T)other.x2[i]);
            y2.push_back((T)other.y2[i]);
        }

        bad = other.bad;

        m_Periodic = other.m_Periodic;
    }
};