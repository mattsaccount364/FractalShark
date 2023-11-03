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

template<typename IterType, class T>
class ATInfo;

template<typename IterType>
class LAStageInfo;

class RefOrbitCalc;

template<typename IterType, class T, CalcBad Bad>
class PerturbationResults;

template<typename IterType, class SubType>
class GPU_LAReference;

template<typename IterType, class SubType>
class LAReference {
private:
    using HDRFloat = HDRFloat<SubType>;
    using HDRFloatComplex = HDRFloatComplex<SubType>;

    friend class GPU_LAReference<IterType, float>;
    friend class GPU_LAReference<IterType, double>;

    static const int lowBound = 64;
    static const SubType log16;

public:
    LAReference() :
        UseAT{},
        AT{},
        LAStageCount{},
        isValid{} {
    }

    bool UseAT;

    ATInfo<IterType, SubType> AT;

    IterType LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<IterType, SubType>> LAs;
    std::vector<LAStageInfo<IterType>> LAStages;

    IterType LAsize();
    template<typename PerturbType>
    bool CreateLAFromOrbit(
        const PerturbationResults<IterType, PerturbType, CalcBad::Disable>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateLAFromOrbitMT(
        const PerturbationResults<IterType, PerturbType, CalcBad::Disable>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateNewLAStage(
        const PerturbationResults<IterType, PerturbType, CalcBad::Disable>& PerturbationResults,
        IterType maxRefIteration);

public:
    template<typename PerturbType>
    void GenerateApproximationData(
        const PerturbationResults<IterType, PerturbType, CalcBad::Disable>& PerturbationResults,
        HDRFloat radius,
        IterType maxRefIteration);
    void CreateATFromLA(HDRFloat radius);

public:
    bool isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc);
    IterType getLAIndex(IterType CurrentLAStage);
    IterType getMacroItCount(IterType CurrentLAStage);

    LAstep<IterType, SubType>
    getLA(IterType LAIndex,
        HDRFloatComplex dz,
        /*HDRFloatComplex dc, */IterType j,
        IterType iterations,
        IterType max_iterations);
};
