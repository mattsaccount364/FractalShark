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

    friend class LAReferenceCuda;

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

    void init();

    bool UseAT;

    ATInfo<HDRFloat> AT;

    size_t LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<float>> LAs;
    std::vector<LAInfoI> LAIs;

    std::vector<LAStageInfo> LAStages;

    const PerturbationResults<HDRFloat>& m_PerturbationResults;

    void addToLA(LAInfoDeep<float> la);
    size_t LAsize();
    void addToLAI(LAInfoI lai);
    void popLA();
    void popLAI();
    bool CreateLAFromOrbit(size_t maxRefIteration);
    bool CreateNewLAStage(size_t maxRefIteration);

public:
    void GenerateApproximationData(HDRFloat radius, size_t maxRefIteration);
    void CreateATFromLA(HDRFloat radius);

public:
    bool isLAStageInvalid(size_t LAIndex, HDRFloatComplex dc);
    size_t getLAIndex(size_t CurrentLAStage);
    size_t getMacroItCount(size_t CurrentLAStage);

    LAstep<HDRFloatComplex>
    getLA(size_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */size_t j, size_t iterations, size_t max_iterations);
};
