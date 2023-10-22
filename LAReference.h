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
    LAReference(const PerturbationResults<HDRFloat> *PerturbationResults) :
        m_PerturbationResults(PerturbationResults),
        UseAT{},
        AT{},
        LAStageCount{},
        isValid{} {
    }

    bool UseAT;

    ATInfo<SubType> AT;

    IterType LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<SubType>> LAs;
    std::vector<LAStageInfo> LAStages;

    const PerturbationResults<HDRFloat> *m_PerturbationResults;

    IterType LAsize();
    bool CreateLAFromOrbit(IterType maxRefIteration);
    bool CreateLAFromOrbitMT(IterType maxRefIteration);
    bool CreateNewLAStage(IterType maxRefIteration);

public:
    void GenerateApproximationData(HDRFloat radius, IterType maxRefIteration);
    void CreateATFromLA(HDRFloat radius);

public:
    bool isLAStageInvalid(IterType LAIndex, HDRFloatComplex dc);
    IterType getLAIndex(IterType CurrentLAStage);
    IterType getMacroItCount(IterType CurrentLAStage);

    LAstep<SubType>
    getLA(IterType LAIndex,
        HDRFloatComplex dz,
        /*HDRFloatComplex dc, */IterType j,
        IterType iterations,
        IterType max_iterations);
};
