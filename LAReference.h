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

template<class SubType>
class GPU_LAReference;

template<class SubType>
class LAReference {
private:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;

    friend class GPU_LAReference<SubType>;

    static const int lowBound = 64;
    static const SubType log16;

public:
    LAReference(const PerturbationResults<HDRFloat> &PerturbationResults) :
        m_PerturbationResults(PerturbationResults),
        UseAT{},
        AT{},
        LAStageCount{},
        isValid{} {
    }

    bool UseAT;

    ATInfo<SubType> AT;

    int32_t LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<SubType>> LAs;
    std::vector<LAStageInfo> LAStages;

    const PerturbationResults<HDRFloat>& m_PerturbationResults;

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

    LAstep<SubType>
    getLA(int32_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */int32_t j, int32_t iterations, int32_t max_iterations);
};
