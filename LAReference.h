#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.*;
//import fractalzoomer.functions.Fractal;
//
//import java.util.Arrays;

#include <cmath>
#include "HDRFloat.h"
#include "LAstep.h"
#include "LAInfoDeep.h"
#include "LAInfoI.h"

template<class T>
class ATInfo;

class LAStageInfo;
class RefOrbitCalc;

template<class T>
class PerturbationResults;

class LAReference {
private:
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    friend class GPU_LAReference;

    static const int lowBound = 64;
    static const double log16;

public:
    LAReference(const PerturbationResults<HDRFloat> &PerturbationResults) :
        m_PerturbationResults(PerturbationResults),
        UseAT{},
        AT{},
        LAStageCount{},
        isValid{} {
    }

    bool UseAT;

    ATInfo<HDRFloat> AT;

    int32_t LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<float>> LAs;
    std::vector<LAStageInfo> LAStages;

    const PerturbationResults<HDRFloat>& m_PerturbationResults;

    void addToLA(LAInfoDeep<float> la);
    int32_t LAsize();
    bool CreateLAFromOrbit(int32_t maxRefIteration);
    bool CreateNewLAStage(int32_t maxRefIteration);

public:
    void GenerateApproximationData(HDRFloat radius, int32_t maxRefIteration);
    void CreateATFromLA(HDRFloat radius);

public:
    bool isLAStageInvalid(int32_t LAIndex, HDRFloatComplex dc);
    int32_t getLAIndex(int32_t CurrentLAStage);
    int32_t getMacroItCount(int32_t CurrentLAStage);

    LAstep<HDRFloatComplex>
    getLA(int32_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */int32_t j, int32_t iterations, int32_t max_iterations);
};
