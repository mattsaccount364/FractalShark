#pragma once

#include "HDRFloatComplex.h"

template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAInfoDeep;
template <typename IterType, class Float, class SubType> class GPU_LAInfoDeep;

namespace FractalShark::LA {

template <class Complex, class Float>
CUDA_CRAP bool
IsThresholdExceeded(const Complex &delta, const Float &threshold)
{
    return HdrCompareToBothPositiveReducedGE(delta.chebychevNorm(), threshold);
}

template <class Step, class Float, class Complex>
CUDA_CRAP Step
Prepare(const Complex &ref, const Complex &dz, const Float &threshold)
{
    // Keep the entry's quadratic term; Evaluate applies the linear coefficients.
    Complex preparedDz = dz * (ref * Float{2} + dz);
    preparedDz.Reduce();

    Step result{};
    result.unusable = IsThresholdExceeded(preparedDz, threshold);
    result.newDzDeep = preparedDz;
    return result;
}

template <class Complex>
CUDA_CRAP Complex
Evaluate(const Complex &preparedDz, const Complex &dc, const Complex &zCoeff, const Complex &cCoeff)
{
    return preparedDz * zCoeff + dc * cCoeff;
}

// Advance a fresh rendering pixel through initialization and the LA hierarchy.
// The orbit reader keeps CPU decompression and GPU storage outside this operation.
template <class Reference, typename IterType, class Complex, class OrbitReader>
#if defined(_MSC_VER) && !defined(__CUDACC__)
// Inline into the CPU caller; the renderer chooses the compilation boundary.
__forceinline
#endif
    CUDA_CRAP void
    AdvancePixel(Reference &reference,
                 IterType maxIterations,
                 const Complex &deltaC,
                 IterType maxRefIteration,
                 IterType referencePeriod,
                 const OrbitReader &readOrbit,
                 IterType &iterations,
                 IterType &referenceIteration,
                 Complex &deltaZ,
                 Complex &z)
{
    reference.InitializePixel(maxIterations, deltaC, iterations, deltaZ);
    referenceIteration = 0;
    z = deltaC;

    if (iterations != 0 && referenceIteration < maxRefIteration) {
        z = readOrbit(referenceIteration) + deltaZ;
    } else if (iterations != 0 && referencePeriod != 0) {
        referenceIteration %= referencePeriod;
        z = readOrbit(referenceIteration) + deltaZ;
    }

    IterType currentStage = reference.IsValid() ? reference.GetLAStageCount() : 0;
    while (currentStage > 0) {
        --currentStage;
        const IterType laIndex = reference.getLAIndex(currentStage);
        if (reference.isLAStageInvalid(laIndex, deltaC)) {
            continue;
        }

        const IterType macroItCount = reference.getMacroItCount(currentStage);
        IterType j = referenceIteration;
        while (iterations < maxIterations) {
            const auto step = reference.getLA(laIndex, deltaZ, j, iterations, maxIterations);
            if (step.unusable) {
                referenceIteration = step.nextStageLAindex;
                break;
            }

            iterations += step.step;
            deltaZ = step.Evaluate(deltaC);
            z = step.getZ(deltaZ);
            ++j;

            auto zNorm = z.chebychevNorm();
            HdrReduce(zNorm);
            auto deltaNorm = deltaZ.chebychevNorm();
            HdrReduce(deltaNorm);
            if (HdrCompareToBothPositiveReducedLT(zNorm, deltaNorm) || j >= macroItCount) {
                deltaZ = z;
                j = 0;
            }
        }

        if (iterations >= maxIterations) {
            break;
        }
    }
}

} // namespace FractalShark::LA

template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAstep {
public:
    static constexpr bool IsHDR = std::is_same_v<Float, ::HDRFloat<float>> ||
                                  std::is_same_v<Float, ::HDRFloat<double>> ||
                                  std::is_same_v<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>;
    using HDRFloatComplex =
        std::conditional_t<IsHDR, ::HDRFloatComplex<SubType>, ::FloatComplex<SubType>>;

    CUDA_CRAP
    LAstep() : step{}, nextStageLAindex{}, LAjdeep{}, Refp1Deep{}, newDzDeep{}, unusable{true} {}

    IterType step;
    IterType nextStageLAindex;
    const LAInfoDeep<IterType, Float, SubType, PExtras> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;
    bool unusable;

    CUDA_CRAP HDRFloatComplex
    Evaluate(HDRFloatComplex deltaC) const
    {
        return LAjdeep->Evaluate(newDzDeep, deltaC);
    }

    CUDA_CRAP void
    EvaluateDzdcDeep(const HDRFloatComplex &dz, HDRFloatComplex &dzdc, const Float &scalingFactor) const
    {
        LAjdeep->EvaluateDzdc(dz, dzdc, scalingFactor);
    }

    CUDA_CRAP HDRFloatComplex
    getZ(HDRFloatComplex deltaZ) const
    {
        return Refp1Deep + deltaZ;
    }
};

template <typename IterType, class Float, class SubType> class GPU_LAstep {
public:
    static constexpr bool IsHDR = std::is_same_v<Float, ::HDRFloat<float>> ||
                                  std::is_same_v<Float, ::HDRFloat<double>> ||
                                  std::is_same_v<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>;
    using HDRFloatComplex =
        std::conditional_t<IsHDR, ::HDRFloatComplex<SubType>, ::FloatComplex<SubType>>;

    CUDA_CRAP
    GPU_LAstep() : step{}, nextStageLAindex{}, LAjdeep{}, Refp1Deep{}, newDzDeep{}, unusable{true} {}

    IterType step;
    IterType nextStageLAindex;
    const GPU_LAInfoDeep<IterType, Float, SubType> *LAjdeep;
    HDRFloatComplex Refp1Deep;
    HDRFloatComplex newDzDeep;
    bool unusable;

    CUDA_CRAP HDRFloatComplex
    Evaluate(HDRFloatComplex deltaC) const
    {
        return LAjdeep->Evaluate(newDzDeep, deltaC);
    }

    CUDA_CRAP HDRFloatComplex
    getZ(HDRFloatComplex deltaZ) const
    {
        return Refp1Deep + deltaZ;
    }
};
