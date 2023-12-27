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

template<typename IterType, class HDRFloat, class SubType>
class ATInfo;

template<typename IterType>
class LAStageInfo;

class RefOrbitCalc;

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

template<typename IterType, class T, class SubType>
class GPU_LAReference;

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAReference {
private:
    static constexpr bool IsHDR =
        std::is_same<Float, ::HDRFloat<float>>::value ||
        std::is_same<Float, ::HDRFloat<double>>::value ||
        std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<Float, ::HDRFloat<CudaDblflt<dblflt>>>::value;
    using FloatComplexT =
        std::conditional<
        IsHDR,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

    friend class GPU_LAReference<IterType, Float, float>;
    friend class GPU_LAReference<IterType, Float, double>;
    friend class GPU_LAReference<IterType, Float, CudaDblflt<dblflt>>;
    friend class GPU_LAReference<IterType, Float, CudaDblflt<MattDblflt>>;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAReference<IterType, float, float, PExtras>;
    friend class LAReference<IterType, double, double, PExtras>;
    friend class LAReference<IterType, CudaDblflt<dblflt>, CudaDblflt<dblflt>, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<float>, float, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<double>, double, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>, PExtras>;

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
    LAReference(const LAReference<IterType, OtherT, Other, PExtras>& other) {
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

    ATInfo<IterType, Float, SubType> AT;

    IterType LAStageCount;

    bool isValid;

private:
    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;
    std::vector<LAInfoDeep<IterType, Float, SubType>> LAs;
    std::vector<LAStageInfo<IterType>> LAStages;

    IterType LAsize();
    template<typename PerturbType>
    bool CreateLAFromOrbit(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateLAFromOrbitMT(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateNewLAStage(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);

public:
    template<typename PerturbType>
    void GenerateApproximationData(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        Float radius,
        IterType maxRefIteration,
        bool UseSmallExponents);
    void CreateATFromLA(Float radius, bool UseSmallExponents);

public:
    bool isLAStageInvalid(IterType LAIndex, FloatComplexT dc);
    IterType getLAIndex(IterType CurrentLAStage);
    IterType getMacroItCount(IterType CurrentLAStage);

    LAstep<IterType, Float, SubType>
    getLA(IterType LAIndex,
        FloatComplexT dz,
        /*FloatComplexT dc, */IterType j,
        IterType iterations,
        IterType max_iterations);
};
