#pragma once

#include "ATInfo.h"
#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAInfoI.h"
#include "LAParameters.h"
#include "LAstep.h"

template <typename IterType, class Float, class SubType> class GPU_LAInfoDeep;

template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAInfoDeep {
public:
    using HDRFloat = Float;
    static constexpr bool IsHDR = std::is_same<HDRFloat, ::HDRFloat<float>>::value ||
                                  std::is_same<HDRFloat, ::HDRFloat<double>>::value ||
                                  std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
                                  std::is_same<HDRFloat, ::HDRFloat<CudaDblflt<dblflt>>>::value;

    using HDRFloatComplex =
        std::conditional_t<IsHDR, ::HDRFloatComplex<SubType>, ::FloatComplex<SubType>>;
    using T = SubType;

public:
    HDRFloatComplex Ref;
    HDRFloatComplex ZCoeff;
    HDRFloatComplex CCoeff;
    HDRFloat LAThreshold;
    HDRFloat LAThresholdC;
    HDRFloat MinMag;
    LAInfoI<IterType> LAi;

public:
    CUDA_CRAP LAInfoDeep();

    template <class Float2, class SubType2, PerturbExtras PExtras2>
    CUDA_CRAP LAInfoDeep(const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other);
    CUDA_CRAP LAInfoDeep(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z);
    CUDA_CRAP bool DetectPeriod(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z);
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP HDRFloatComplex getZCoeff() const;
    CUDA_CRAP HDRFloatComplex getCCoeff() const;
    CUDA_CRAP bool Step(const LAParametersRuntime<Float> &parameters,
                        LAInfoDeep &out,
                        HDRFloatComplex z) const;

    CUDA_CRAP bool isLAThresholdZero() const;
    CUDA_CRAP bool isZCoeffZero() const;
    CUDA_CRAP LAInfoDeep Step(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z);

    CUDA_CRAP bool Composite(const LAParametersRuntime<Float> &parameters,
                             LAInfoDeep &out,
                             const LAInfoDeep &la);
    CUDA_CRAP LAInfoDeep Composite(const LAParametersRuntime<Float> &parameters, const LAInfoDeep &la);
    CUDA_CRAP LAstep<IterType, Float, SubType, PExtras> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex preparedDz, HDRFloatComplex dc) const;
    CUDA_CRAP void EvaluateDzdz(HDRFloatComplex &dz,
                                HDRFloatComplex &dzdz,
                                const HDRFloatComplex &dc) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc);
    CUDA_CRAP void EvaluateDzdc(const HDRFloatComplex &dz,
                                HDRFloatComplex &dzdc,
                                const HDRFloat &ScalingFactor) const;
    CUDA_CRAP HDRFloatComplex EvaluateDzdc2(HDRFloatComplex z,
                                            HDRFloatComplex dzdc2,
                                            HDRFloatComplex dzdc);
    CUDA_CRAP HDRFloatComplex DzdzStep(const HDRFloatComplex &dz) const;
    CUDA_CRAP void EvaluateDerivatives(HDRFloatComplex &dz,
                                       HDRFloatComplex &dzdz,
                                       HDRFloatComplex &dzdc,
                                       const HDRFloat &ScalingFactor) const;
    CUDA_CRAP void CreateAT(ATInfo<IterType, Float, SubType> &Result,
                            const LAInfoDeep &next,
                            bool UseSmallExponents);
    CUDA_CRAP HDRFloat getLAThreshold() const;
    CUDA_CRAP HDRFloat getLAThresholdC() const;
    CUDA_CRAP void SetLAi(const LAInfoI<IterType> &other);
    CUDA_CRAP const LAInfoI<IterType> &GetLAi() const;

private:
    CUDA_CRAP static HDRFloat
    RestrictThreshold(HDRFloat threshold,
                      HDRFloat magnitude,
                      HDRFloat coefficientMagnitude,
                      HDRFloat scale)
    {
        HDRFloat candidate = magnitude / coefficientMagnitude * scale;
        HdrReduce(candidate);
        if constexpr (IsHDR) {
            return HDRFloat::minBothPositiveReduced(threshold, candidate);
        } else {
            return std::min(threshold, candidate);
        }
    }
};

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep()
    : Ref{}, ZCoeff{}, CCoeff{}, LAThreshold{}, LAThresholdC{}, MinMag{}, LAi{}
{
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
template <class Float2, class SubType2, PerturbExtras PExtras2>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep(
    const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other)
{

    this->Ref = static_cast<HDRFloatComplex>(other.Ref);
    this->LAThreshold = static_cast<HDRFloat>(other.LAThreshold);
    this->ZCoeff = static_cast<HDRFloatComplex>(other.ZCoeff);
    this->CCoeff = static_cast<HDRFloatComplex>(other.CCoeff);
    this->LAThresholdC = static_cast<HDRFloat>(other.LAThresholdC);
    this->MinMag = static_cast<HDRFloat>(other.MinMag);
    this->LAi = other.LAi;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep(const LAParametersRuntime<Float> &parameters,
                                                          HDRFloatComplex z)
    : Ref{z}, ZCoeff{SubType{1}, SubType{0}}, CCoeff{SubType{1}, SubType{0}}, LAThreshold{1},
      LAThresholdC{1}, MinMag{parameters.m_DetectionMethod == 1 ? HDRFloat{4} : HDRFloat{}}, LAi{}
{
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAInfoDeep<IterType, Float, SubType, PExtras>::DetectPeriod(const LAParametersRuntime<Float> &parameters,
                                                            HDRFloatComplex z)
{
    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {

            return z.chebychevNorm().compareToBothPositive(MinMag *
                                                           parameters.m_PeriodDetectionThreshold2) < 0;
        } else {
            return z.chebychevNorm() < (MinMag * parameters.m_PeriodDetectionThreshold2);
        }
    } else {

        if constexpr (IsHDR) {
            return (z.chebychevNorm() / ZCoeff.chebychevNorm() * parameters.m_LAThresholdScale)
                       .compareToBothPositive(LAThreshold * parameters.m_PeriodDetectionThreshold) < 0;
        } else {
            return (z.chebychevNorm() / ZCoeff.chebychevNorm() * parameters.m_LAThresholdScale) <
                   (LAThreshold * parameters.m_PeriodDetectionThreshold);
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::getRef() const
{
    return Ref;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::getZCoeff() const
{
    return ZCoeff;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::getCCoeff() const
{
    return CCoeff;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAInfoDeep<IterType, Float, SubType, PExtras>::Step(const LAParametersRuntime<Float> &parameters,
                                                    LAInfoDeep &out,
                                                    HDRFloatComplex z) const
{

    const HDRFloat ChebyMagz = z.chebychevNorm();

    const HDRFloat ChebyMagZCoeff{ZCoeff.chebychevNorm()};
    const HDRFloat ChebyMagCCoeff{CCoeff.chebychevNorm()};

    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {
            HDRFloat outMin = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
            out.MinMag = outMin;
        } else {
            HDRFloat outMin = std::min(ChebyMagz, MinMag);
            out.MinMag = outMin;
        }
    }

    HDRFloat outLAThreshold =
        RestrictThreshold(LAThreshold, ChebyMagz, ChebyMagZCoeff, parameters.m_LAThresholdScale);
    HDRFloat outLAThresholdC =
        RestrictThreshold(LAThresholdC, ChebyMagz, ChebyMagCCoeff, parameters.m_LAThresholdCScale);

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;

    const HDRFloatComplex z2{z * HDRFloat(2)};
    HDRFloatComplex outZCoeff{z2 * ZCoeff};
    HdrReduce(outZCoeff);
    HDRFloatComplex outCCoeff{z2 * CCoeff + HDRFloat{1}};
    HdrReduce(outCCoeff);

    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;

    out.Ref = Ref;

    if (parameters.m_DetectionMethod == 1) {

        if constexpr (IsHDR) {
            return out.MinMag.compareToBothPositive(MinMag *
                                                    (parameters.m_Stage0PeriodDetectionThreshold2)) < 0;
        } else {
            return out.MinMag < (MinMag * parameters.m_Stage0PeriodDetectionThreshold2);
        }

    } else {
        if constexpr (IsHDR) {
            return out.LAThreshold.compareToBothPositive(
                       LAThreshold * (parameters.m_Stage0PeriodDetectionThreshold)) < 0;
        } else {
            return out.LAThreshold < (LAThreshold * (parameters.m_Stage0PeriodDetectionThreshold));
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAInfoDeep<IterType, Float, SubType, PExtras>::isLAThresholdZero() const
{
    if constexpr (IsHDR) {
        return LAThreshold.compareTo(HDRFloat{0}) == 0;
    } else {
        return LAThreshold == 0;
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAInfoDeep<IterType, Float, SubType, PExtras>::isZCoeffZero() const
{
    if constexpr (IsHDR) {
        return ZCoeff.getRe().compareTo(HDRFloat{0}) == 0 && ZCoeff.getIm().compareTo(HDRFloat{0}) == 0;
    } else {
        return ZCoeff.getRe() == 0 && ZCoeff.getIm() == 0;
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>
LAInfoDeep<IterType, Float, SubType, PExtras>::Step(const LAParametersRuntime<Float> &parameters,
                                                    HDRFloatComplex z)
{

    LAInfoDeep Result = LAInfoDeep();

    Step(parameters, Result, z);
    return Result;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAInfoDeep<IterType, Float, SubType, PExtras>::Composite(const LAParametersRuntime<Float> &parameters,
                                                         LAInfoDeep &out,
                                                         const LAInfoDeep &la)
{

    HDRFloatComplex z = la.Ref;
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloat ChebyMagZCoeff = ZCoeff.chebychevNorm();
    HDRFloat ChebyMagCCoeff = CCoeff.chebychevNorm();

    HDRFloat outLAThreshold =
        RestrictThreshold(LAThreshold, ChebyMagz, ChebyMagZCoeff, parameters.m_LAThresholdScale);
    HDRFloat outLAThresholdC =
        RestrictThreshold(LAThresholdC, ChebyMagz, ChebyMagCCoeff, parameters.m_LAThresholdCScale);

    HDRFloatComplex z2 = z * HDRFloat(2);
    HDRFloatComplex outZCoeff = z2 * ZCoeff;
    HdrReduce(outZCoeff);

    HDRFloatComplex outCCoeff = z2 * CCoeff;
    HdrReduce(outCCoeff);

    ChebyMagZCoeff = outZCoeff.chebychevNorm();
    ChebyMagCCoeff = outCCoeff.chebychevNorm();
    HDRFloat temp = outLAThreshold;

    HDRFloat nextThreshold = la.LAThreshold;
    HDRFloatComplex LAZCoeff = la.ZCoeff;
    HDRFloatComplex LACCoeff = la.CCoeff;

    HDRFloat temp1 = nextThreshold / ChebyMagZCoeff;
    HdrReduce(temp1);

    HDRFloat temp2 = nextThreshold / ChebyMagCCoeff;
    HdrReduce(temp2);

    if constexpr (IsHDR) {
        outLAThreshold = HDRFloat::minBothPositiveReduced(outLAThreshold, temp1);
        outLAThresholdC = HDRFloat::minBothPositiveReduced(outLAThresholdC, temp2);
    } else {
        outLAThreshold = std::min(outLAThreshold, temp1);
        outLAThresholdC = std::min(outLAThresholdC, temp2);
    }
    outZCoeff = outZCoeff * LAZCoeff;
    HdrReduce(outZCoeff);

    outCCoeff = outCCoeff * LAZCoeff + LACCoeff;
    HdrReduce(outCCoeff);

    out.LAThreshold = outLAThreshold;
    out.LAThresholdC = outLAThresholdC;
    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;
    out.Ref = Ref;

    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {
            temp = HDRFloat::minBothPositiveReduced(ChebyMagz, MinMag);
            out.MinMag = HDRFloat::minBothPositiveReduced(temp, la.MinMag);
            return temp.compareToBothPositive(MinMag * parameters.m_PeriodDetectionThreshold2) < 0;
        } else {
            temp = std::min(ChebyMagz, MinMag);
            out.MinMag = std::min(temp, la.MinMag);
            return temp < (MinMag * parameters.m_PeriodDetectionThreshold2);
        }
    } else {
        if constexpr (IsHDR) {
            return temp.compareToBothPositive(LAThreshold * parameters.m_PeriodDetectionThreshold) < 0;
        } else {
            return temp < (LAThreshold * parameters.m_PeriodDetectionThreshold);
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>
LAInfoDeep<IterType, Float, SubType, PExtras>::Composite(const LAParametersRuntime<Float> &parameters,
                                                         const LAInfoDeep &la)
{

    LAInfoDeep Result = LAInfoDeep();

    Composite(parameters, Result, la);
    return Result;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAstep<IterType, Float, SubType, PExtras>
LAInfoDeep<IterType, Float, SubType, PExtras>::Prepare(HDRFloatComplex dz) const
{
    return FractalShark::LA::Prepare<LAstep<IterType, Float, SubType, PExtras>>(Ref, dz, LAThreshold);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::Evaluate(HDRFloatComplex preparedDz,
                                                        HDRFloatComplex dc) const
{
    return FractalShark::LA::Evaluate(preparedDz, dc, ZCoeff, CCoeff);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdz(
    HDRFloatComplex &dz, HDRFloatComplex &dzdz, [[maybe_unused]] const HDRFloatComplex &dc) const
{
    dzdz = dzdz * HDRFloat{2.0} * (dz + HDRFloatComplex(Ref)) * HDRFloatComplex(ZCoeff);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdc(HDRFloatComplex z, HDRFloatComplex dzdc)
{
    return dzdc * HDRFloat(2) * z * ZCoeff + CCoeff;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdc(const HDRFloatComplex &dz,
                                                            HDRFloatComplex &dzdc,
                                                            const HDRFloat &ScalingFactor) const
{
    dzdc = dzdc * HDRFloat{2.0} * (dz + HDRFloatComplex(Ref)) * HDRFloatComplex(ZCoeff) +
           HDRFloatComplex(CCoeff) * ScalingFactor;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDzdc2(HDRFloatComplex z,
                                                             HDRFloatComplex dzdc2,
                                                             HDRFloatComplex dzdc)
{
    return (dzdc2 * z + dzdc.square()) * HDRFloat(2) * ZCoeff;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAInfoDeep<IterType, Float, SubType, PExtras>::CreateAT(ATInfo<IterType, Float, SubType> &Result,
                                                        const LAInfoDeep &next,
                                                        bool UseSmallExponents)
{
    Result.ZCoeff = ZCoeff;
    Result.CCoeff = ZCoeff * CCoeff;
    HdrReduce(Result.CCoeff);

    Result.InvZCoeff = ZCoeff.reciprocal();
    HdrReduce(Result.InvZCoeff);

    Result.CCoeffSqrInvZCoeff = Result.CCoeff * Result.CCoeff * Result.InvZCoeff;
    HdrReduce(Result.CCoeffSqrInvZCoeff);

    Result.CCoeffInvZCoeff = Result.CCoeff * Result.InvZCoeff;
    HdrReduce(Result.CCoeffInvZCoeff);

    Result.RefC = next.getRef() * ZCoeff;
    HdrReduce(Result.RefC);

    Result.CCoeffNormSqr = Result.CCoeff.norm_squared();
    HdrReduce(Result.CCoeffNormSqr);

    Result.RefCNormSqr = Result.RefC.norm_squared();
    HdrReduce(Result.RefCNormSqr);

    HDRFloat lim;
    if constexpr (IsHDR) {
        lim = HDRFloat(32, 1);
        if constexpr (std::is_same<HDRFloat, ::HDRFloat<double>>::value) {
            if (!UseSmallExponents) {
                lim.setExp(256);
            }
        }
        HdrReduce(lim);
        Result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim);
        HdrReduce(Result.SqrEscapeRadius);

        Result.ThresholdC = HDRFloat::minBothPositive(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    } else {
        lim = 4294967296.0f;
        Result.SqrEscapeRadius = std::min(ZCoeff.norm_squared() * LAThreshold, lim);
        Result.ThresholdC = std::min(LAThresholdC, lim / Result.CCoeff.chebychevNorm());
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP typename LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex
LAInfoDeep<IterType, Float, SubType, PExtras>::DzdzStep(const HDRFloatComplex &dz) const
{
    return HDRFloat{2.0f} * (dz + HDRFloatComplex(Ref)) * HDRFloatComplex(ZCoeff);
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAInfoDeep<IterType, Float, SubType, PExtras>::EvaluateDerivatives(HDRFloatComplex &dz,
                                                                   HDRFloatComplex &dzdz,
                                                                   HDRFloatComplex &dzdc,
                                                                   const HDRFloat &ScalingFactor) const
{
    const HDRFloatComplex step = DzdzStep(dz);

    dzdz = dzdz * step;
    dzdc = dzdc * step + HDRFloatComplex(CCoeff) * ScalingFactor;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloat
LAInfoDeep<IterType, Float, SubType, PExtras>::getLAThreshold() const
{
    return LAThreshold;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloat
LAInfoDeep<IterType, Float, SubType, PExtras>::getLAThresholdC() const
{
    return LAThresholdC;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP void
LAInfoDeep<IterType, Float, SubType, PExtras>::SetLAi(const LAInfoI<IterType> &other)
{
    this->LAi = other;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP const LAInfoI<IterType> &
LAInfoDeep<IterType, Float, SubType, PExtras>::GetLAi() const
{
    return this->LAi;
}
