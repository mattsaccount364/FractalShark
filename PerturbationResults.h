#pragma once

#include <vector>
#include <stdint.h>

#include "HighPrecision.h"
#include "HDRFloatComplex.h"
#include "RefOrbitCalc.h"

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

    void InitReused() {
        HighPrecision Zero = 0;

        //assert(RequiresReuse());
        Zero.precision(AuthoritativeReuseExtraPrecision);

        ReuseX.push_back(Zero);
        ReuseY.push_back(Zero);
    }

    template<class T, RefOrbitCalc::CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void InitResults(
        const HighPrecision& cx,
        const HighPrecision& cy,
        const HighPrecision& minX,
        const HighPrecision& minY,
        const HighPrecision& maxX,
        const HighPrecision& maxY,
        size_t NumIterations,
        size_t GuessReserveSize) {
        radiusX = fabs(maxX - cx) + fabs(minX - cx);
        radiusY = fabs(maxY - cy) + fabs(minY - cy);

        hiX = cx;
        hiY = cy;
        maxRadius = (T)((radiusX > radiusY) ? radiusX : radiusY);
        MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : NumIterations;

        x.reserve(ReserveSize);
        y.reserve(ReserveSize);

        if constexpr (Bad == RefOrbitCalc::CalcBad::Enable) {
            bad.reserve(ReserveSize);
        }

        x.push_back(T(0));
        y.push_back(T(0));

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AuthoritativePrecision = cx.precision();
            ReuseX.reserve(ReserveSize);
            ReuseY.reserve(ReserveSize);

            InitReused();
        }
        else if constexpr (Reuse == RefOrbitCalc::ReuseMode::DontSaveForReuse) {
            AuthoritativePrecision = 0;
        }
    }

    template<RefOrbitCalc::CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void TrimResults() {
        x.shrink_to_fit();
        y.shrink_to_fit();

        if constexpr (Bad == RefOrbitCalc::CalcBad::Enable) {
            bad.shrink_to_fit();
        }

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            ReuseX.shrink_to_fit();
            ReuseY.shrink_to_fit();
        }
    }
};