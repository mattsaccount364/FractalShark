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
