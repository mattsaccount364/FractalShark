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

template<typename IterType, class T, class SubType>
class GPU_LAReference;

template<typename IterType, class T, class SubType>
class LAReference {
private:
    using HDRFloat = ::HDRFloat<SubType>;
    using HDRFloatComplex = ::HDRFloatComplex<SubType>;

    friend class GPU_LAReference<IterType, HDRFloatComplex, float>;
    friend class GPU_LAReference<IterType, HDRFloatComplex, double>;
    friend class GPU_LAReference<IterType, HDRFloatComplex, CudaDblflt<dblflt>>;
    friend class GPU_LAReference<IterType, HDRFloatComplex, CudaDblflt<MattDblflt>>;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAReference<IterType, float, float>;
    friend class LAReference<IterType, double, double>;
    friend class LAReference<IterType, ::HDRFloat<float>, float>;
    friend class LAReference<IterType, ::HDRFloat<double>, double>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>;

    static const int lowBound = 64;
    static const SubType log16;

public:
    LAReference() :
        UseAT{},
        AT{},
        LAStageCount{}, 
        isValid{} {
    }

    template<class OtherT, class Other>
    LAReference(const LAReference<IterType, OtherT, Other>& other) {
        UseAT = other.UseAT;
        AT = other.AT;
        LAStageCount = other.LAStageCount;
        isValid = other.isValid;
        LAs.resize(other.LAs.size());
        for (int i = 0; i < other.LAs.size(); i++) {
            LAs[i] = other.LAs[i];
        }
        LAStages.resize(other.LAStages.size());
        for (int i = 0; i < other.LAStages.size(); i++) {
            LAStages[i] = other.LAStages[i];
        }
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
