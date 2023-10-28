#pragma once

#include <vector>
#include <stdint.h>
#include <type_traits>

#include "HighPrecision.h"
#include "HDRFloatComplex.h"
#include "RefOrbitCalc.h"
#include "LAReference.h"

#include "GPU_Render.h"

// If true, choose type == float/double for primitives.
// If false, choose type == T::TemplateSubType for HdrFloat subtypes.
// This is kind of a headache.  std::conditional by itself is not adequate here.
template<bool, typename T>
class SubTypeChooser {
public:
    using type = typename T::TemplateSubType;
};

template<typename T>
class SubTypeChooser<true, T> {
public:
    using type = T;
};

template<class T, CalcBad Bad>
class PerturbationResults {
public:
    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<
        std::is_fundamental<T>::value,
        T>::type;

    HighPrecision hiX, hiY, radiusX, radiusY;
    T radiusXHdr, radiusYHdr;
    T maxRadius;
    IterType MaxIterations;
    IterType PeriodMaybeZero;  // Zero if not worked out

    std::vector<MattReferenceSingleIter<T, Bad>> orb;

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;

    std::unique_ptr<LAReference<IterType, SubType>> LaReference;

    void clear() {
        hiX = {};
        hiY = {};
        radiusX = {};
        radiusY = {};
        radiusXHdr = {};
        radiusYHdr = {};
        maxRadius = {};
        MaxIterations = 0;
        PeriodMaybeZero = 0;

        orb.clear();

        AuthoritativePrecision = false;
        ReuseX.clear();
        ReuseY.clear();
    }

    // TODO WTF is this for again?
    template<class Other, CalcBad Bad = CalcBad::Disable>
    void Copy(const PerturbationResults<Other, Bad>& other) {
        clear();

        //hiX = other.hiX;
        //hiY = other.hiY;

        //radiusX = other.radiusX;
        //radiusY = other.radiusY;
        //radiusXHdr = other.radiusXHdr;
        //radiusYHdr = other.radiusYHdr;
        //maxRadius = other.maxRadius;

        orb.reserve(other.orb.size());

        ReuseX.reserve(other.ReuseX.size());
        ReuseY.reserve(other.ReuseY.size());

        for (size_t i = 0; i < other.orb.size(); i++) {
            if constexpr (Bad == CalcBad::Enable) {
                orb.push_back({ (T)other.orb[i].x, (T)other.orb[i].y, other.orb[i].bad != 0 });
            }
            else {
                orb.push_back({ (T)other.orb[i].x, (T)other.orb[i].y });
            }
        }

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX = other.ReuseX;
        ReuseY = other.ReuseY;
    }

    template<class U>
    HDRFloatComplex<U> GetComplex(size_t index) const {
        return { orb[index].x, orb[index].y };
    }

    void InitReused() {
        HighPrecision Zero = 0;

        //assert(RequiresReuse());
        Zero.precision(AuthoritativeReuseExtraPrecision);

        ReuseX.push_back(Zero);
        ReuseY.push_back(Zero);
    }

    template<class T, CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void InitResults(
        const HighPrecision& cx,
        const HighPrecision& cy,
        const HighPrecision& minX,
        const HighPrecision& minY,
        const HighPrecision& maxX,
        const HighPrecision& maxY,
        size_t NumIterations,
        size_t GuessReserveSize) {
        //radiusX = fabs(maxX - cx) + fabs(minX - cx);
        //radiusY = fabs(maxY - cy) + fabs(minY - cy);
        radiusX = maxX - minX;
        radiusY = maxY - minY;
        radiusXHdr = (T)radiusX;
        HdrReduce(radiusXHdr);
        radiusYHdr = (T)radiusY;
        HdrReduce(radiusYHdr);

        hiX = cx;
        hiY = cy;
        maxRadius = (T)(radiusX > radiusY ? radiusX : radiusY);
        HdrReduce(maxRadius);

        MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : NumIterations;

        orb.reserve(ReserveSize);

        orb.push_back({});

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

    template<CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void TrimResults() {
        orb.shrink_to_fit();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            ReuseX.shrink_to_fit();
            ReuseY.shrink_to_fit();
        }
    }
};