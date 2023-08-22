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
#include "PerturbationResults.h"

class ATInfo;
class LAStageInfo;

class RefOrbitCalc;

class LAStageInfo {
public:
    size_t LAIndex;
    size_t MacroItCount;
};

class  LAInfoI {
public:
    size_t StepLength, NextStageLAIndex;

    LAInfoI() {
        StepLength = 0;
        NextStageLAIndex = 0;
    }

    LAInfoI(const LAInfoI &other) {
        StepLength = other.StepLength;
        NextStageLAIndex = other.NextStageLAIndex;
    }
};

class LAReference {
private:
    using HDRFloat = HDRFloat<float>;
    using HDRFloatComplex = HDRFloatComplex<float>;

    static const int lowBound = 64;
    static const double log16;

public:
    LAReference(const PerturbationResults<HDRFloat> &PerturbationResults) :
        m_PerturbationResults(PerturbationResults) {
    }

    void init();

    bool UseAT;

    ATInfo AT;

    int LAStageCount;

    bool isValid;

    bool DoublePrecisionPT;

private:
    static constexpr int MaxLAStages = 512;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep> LAs;
    std::vector<LAInfoI> LAIs;

    std::vector<LAStageInfo> LAStages;

    const PerturbationResults<HDRFloat>& m_PerturbationResults;

    void addToLA(LAInfoDeep la);
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
    bool isLAStageInvalid(int LAIndex, HDRFloatComplex dc);
    size_t getLAIndex(int CurrentLAStage);
    size_t getMacroItCount(int CurrentLAStage);
    LAstep getLA(int LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */int j, int iterations, int max_iterations);
};
