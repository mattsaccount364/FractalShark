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
    LAInfoI<IterType> LAi;

public:
    CUDA_CRAP LAInfoDeep();
    LAInfoDeep(const LAInfoDeep &) = default;
    LAInfoDeep &operator=(const LAInfoDeep &) = default;

    template <class Float2, class SubType2, PerturbExtras PExtras2>
    CUDA_CRAP LAInfoDeep(const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other);
    CUDA_CRAP explicit LAInfoDeep(HDRFloatComplex z);
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP HDRFloatComplex getZCoeff() const;
    CUDA_CRAP HDRFloatComplex getCCoeff() const;

    CUDA_CRAP bool isLAThresholdZero() const;
    CUDA_CRAP bool isZCoeffZero() const;

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
                            bool useSmallExponents,
                            HDRFloat thresholdC);
    CUDA_CRAP HDRFloat getLAThreshold() const;
    CUDA_CRAP void SetLAi(const LAInfoI<IterType> &other);
    CUDA_CRAP const LAInfoI<IterType> &GetLAi() const;
};

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep()
    : Ref{}, ZCoeff{}, CCoeff{}, LAThreshold{}, LAi{}
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
    this->LAi = other.LAi;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP
LAInfoDeep<IterType, Float, SubType, PExtras>::LAInfoDeep(HDRFloatComplex z)
    : Ref{z}, ZCoeff{SubType{1}, SubType{0}}, CCoeff{SubType{1}, SubType{0}}, LAThreshold{1}, LAi{}
{
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
LAInfoDeep<IterType, Float, SubType, PExtras>::CreateAT(ATInfo<IterType, Float, SubType> &result,
                                                        const LAInfoDeep &next,
                                                        bool useSmallExponents,
                                                        HDRFloat thresholdC)
{
    result.ZCoeff = ZCoeff;
    result.CCoeff = ZCoeff * CCoeff;
    HdrReduce(result.CCoeff);

    result.InvZCoeff = ZCoeff.reciprocal();
    HdrReduce(result.InvZCoeff);

    result.CCoeffSqrInvZCoeff = result.CCoeff * result.CCoeff * result.InvZCoeff;
    HdrReduce(result.CCoeffSqrInvZCoeff);

    result.CCoeffInvZCoeff = result.CCoeff * result.InvZCoeff;
    HdrReduce(result.CCoeffInvZCoeff);

    result.RefC = next.getRef() * ZCoeff;
    HdrReduce(result.RefC);

    result.CCoeffNormSqr = result.CCoeff.norm_squared();
    HdrReduce(result.CCoeffNormSqr);

    result.RefCNormSqr = result.RefC.norm_squared();
    HdrReduce(result.RefCNormSqr);

    HDRFloat lim;
    if constexpr (IsHDR) {
        lim = HDRFloat(32, 1);
        if constexpr (std::is_same<HDRFloat, ::HDRFloat<double>>::value) {
            if (!useSmallExponents) {
                lim.setExp(256);
            }
        }
        HdrReduce(lim);
        result.SqrEscapeRadius = HDRFloat::minBothPositive(ZCoeff.norm_squared() * LAThreshold, lim);
        HdrReduce(result.SqrEscapeRadius);

        result.ThresholdC = HDRFloat::minBothPositive(thresholdC, lim / result.CCoeff.chebychevNorm());
    } else {
        lim = 4294967296.0f;
        result.SqrEscapeRadius = std::min(ZCoeff.norm_squared() * LAThreshold, lim);
        result.ThresholdC = std::min(thresholdC, lim / result.CCoeff.chebychevNorm());
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

// Construction metadata is stored in a parallel table, never in runtime rows.
template <class Float> struct LAConstructionInfo {
    Float LAThresholdC{};
    Float MinMag{};
};

// Only local builder state combines a row and its construction metadata.
template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAConstructionEntry : public LAInfoDeep<IterType, Float, SubType, PExtras> {
public:
    using HDRFloat = Float;
    using HDRFloatComplex = typename LAInfoDeep<IterType, Float, SubType, PExtras>::HDRFloatComplex;
    static constexpr bool IsHDR = LAInfoDeep<IterType, Float, SubType, PExtras>::IsHDR;
    LAConstructionInfo<Float> m_Construction;

    LAConstructionEntry() = default;
    LAConstructionEntry(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z)
        : LAInfoDeep<IterType, Float, SubType, PExtras>{z},
          m_Construction{Float{1}, parameters.m_DetectionMethod == 1 ? Float{4} : Float{}}
    {
    }
    LAConstructionEntry(const LAInfoDeep<IterType, Float, SubType, PExtras> &row,
                        const LAConstructionInfo<Float> &construction)
        : LAInfoDeep<IterType, Float, SubType, PExtras>{row}, m_Construction{construction}
    {
    }
    CUDA_CRAP bool DetectPeriod(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z);
    CUDA_CRAP bool Step(const LAParametersRuntime<Float> &parameters,
                        LAConstructionEntry &out,
                        HDRFloatComplex z) const;
    CUDA_CRAP LAConstructionEntry Step(const LAParametersRuntime<Float> &parameters, HDRFloatComplex z);
    CUDA_CRAP bool Composite(const LAParametersRuntime<Float> &parameters,
                             LAConstructionEntry &out,
                             const LAConstructionEntry &la);
    CUDA_CRAP LAConstructionEntry Composite(const LAParametersRuntime<Float> &parameters,
                                            const LAConstructionEntry &la);

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
CUDA_CRAP bool
LAConstructionEntry<IterType, Float, SubType, PExtras>::DetectPeriod(
    const LAParametersRuntime<Float> &parameters, HDRFloatComplex z)
{
    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {

            return z.chebychevNorm().compareToBothPositive(m_Construction.MinMag *
                                                           parameters.m_PeriodDetectionThreshold2) < 0;
        } else {
            return z.chebychevNorm() < (m_Construction.MinMag * parameters.m_PeriodDetectionThreshold2);
        }
    } else {

        if constexpr (IsHDR) {
            return (z.chebychevNorm() / this->ZCoeff.chebychevNorm() * parameters.m_LAThresholdScale)
                       .compareToBothPositive(this->LAThreshold *
                                              parameters.m_PeriodDetectionThreshold) < 0;
        } else {
            return (z.chebychevNorm() / this->ZCoeff.chebychevNorm() * parameters.m_LAThresholdScale) <
                   (this->LAThreshold * parameters.m_PeriodDetectionThreshold);
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAConstructionEntry<IterType, Float, SubType, PExtras>::Step(
    const LAParametersRuntime<Float> &parameters, LAConstructionEntry &out, HDRFloatComplex z) const
{

    const HDRFloat ChebyMagz = z.chebychevNorm();

    const HDRFloat ChebyMagZCoeff{this->ZCoeff.chebychevNorm()};
    const HDRFloat ChebyMagCCoeff{this->CCoeff.chebychevNorm()};

    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {
            HDRFloat outMin = HDRFloat::minBothPositiveReduced(ChebyMagz, m_Construction.MinMag);
            out.m_Construction.MinMag = outMin;
        } else {
            HDRFloat outMin = std::min(ChebyMagz, m_Construction.MinMag);
            out.m_Construction.MinMag = outMin;
        }
    }

    HDRFloat outLAThreshold =
        RestrictThreshold(this->LAThreshold, ChebyMagz, ChebyMagZCoeff, parameters.m_LAThresholdScale);
    HDRFloat outLAThresholdC = RestrictThreshold(
        m_Construction.LAThresholdC, ChebyMagz, ChebyMagCCoeff, parameters.m_LAThresholdCScale);

    out.LAThreshold = outLAThreshold;
    out.m_Construction.LAThresholdC = outLAThresholdC;

    const HDRFloatComplex z2{z * HDRFloat(2)};
    HDRFloatComplex outZCoeff{z2 * this->ZCoeff};
    HdrReduce(outZCoeff);
    HDRFloatComplex outCCoeff{z2 * this->CCoeff + HDRFloat{1}};
    HdrReduce(outCCoeff);

    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;

    out.Ref = this->Ref;

    if (parameters.m_DetectionMethod == 1) {

        if constexpr (IsHDR) {
            return out.m_Construction.MinMag.compareToBothPositive(
                       m_Construction.MinMag * (parameters.m_Stage0PeriodDetectionThreshold2)) < 0;
        } else {
            return out.m_Construction.MinMag <
                   (m_Construction.MinMag * parameters.m_Stage0PeriodDetectionThreshold2);
        }

    } else {
        if constexpr (IsHDR) {
            return out.LAThreshold.compareToBothPositive(
                       this->LAThreshold * (parameters.m_Stage0PeriodDetectionThreshold)) < 0;
        } else {
            return out.LAThreshold < (this->LAThreshold * (parameters.m_Stage0PeriodDetectionThreshold));
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAConstructionEntry<IterType, Float, SubType, PExtras>
LAConstructionEntry<IterType, Float, SubType, PExtras>::Step(
    const LAParametersRuntime<Float> &parameters, HDRFloatComplex z)
{

    LAConstructionEntry Result = LAConstructionEntry();

    Step(parameters, Result, z);
    return Result;
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP bool
LAConstructionEntry<IterType, Float, SubType, PExtras>::Composite(
    const LAParametersRuntime<Float> &parameters,
    LAConstructionEntry &out,
    const LAConstructionEntry &la)
{

    HDRFloatComplex z = la.Ref;
    HDRFloat ChebyMagz = z.chebychevNorm();

    HDRFloat ChebyMagZCoeff = this->ZCoeff.chebychevNorm();
    HDRFloat ChebyMagCCoeff = this->CCoeff.chebychevNorm();

    HDRFloat outLAThreshold =
        RestrictThreshold(this->LAThreshold, ChebyMagz, ChebyMagZCoeff, parameters.m_LAThresholdScale);
    HDRFloat outLAThresholdC = RestrictThreshold(
        m_Construction.LAThresholdC, ChebyMagz, ChebyMagCCoeff, parameters.m_LAThresholdCScale);

    HDRFloatComplex z2 = z * HDRFloat(2);
    HDRFloatComplex outZCoeff = z2 * this->ZCoeff;
    HdrReduce(outZCoeff);

    HDRFloatComplex outCCoeff = z2 * this->CCoeff;
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
    out.m_Construction.LAThresholdC = outLAThresholdC;
    out.ZCoeff = outZCoeff;
    out.CCoeff = outCCoeff;
    out.Ref = this->Ref;

    if (parameters.m_DetectionMethod == 1) {
        if constexpr (IsHDR) {
            temp = HDRFloat::minBothPositiveReduced(ChebyMagz, m_Construction.MinMag);
            out.m_Construction.MinMag = HDRFloat::minBothPositiveReduced(temp, la.m_Construction.MinMag);
            return temp.compareToBothPositive(m_Construction.MinMag *
                                              parameters.m_PeriodDetectionThreshold2) < 0;
        } else {
            temp = std::min(ChebyMagz, m_Construction.MinMag);
            out.m_Construction.MinMag = std::min(temp, la.m_Construction.MinMag);
            return temp < (m_Construction.MinMag * parameters.m_PeriodDetectionThreshold2);
        }
    } else {
        if constexpr (IsHDR) {
            return temp.compareToBothPositive(this->LAThreshold *
                                              parameters.m_PeriodDetectionThreshold) < 0;
        } else {
            return temp < (this->LAThreshold * parameters.m_PeriodDetectionThreshold);
        }
    }
}

template <typename IterType, class Float, class SubType, PerturbExtras PExtras>
CUDA_CRAP LAConstructionEntry<IterType, Float, SubType, PExtras>
LAConstructionEntry<IterType, Float, SubType, PExtras>::Composite(
    const LAParametersRuntime<Float> &parameters, const LAConstructionEntry &la)
{

    LAConstructionEntry Result = LAConstructionEntry();

    Composite(parameters, Result, la);
    return Result;
}
