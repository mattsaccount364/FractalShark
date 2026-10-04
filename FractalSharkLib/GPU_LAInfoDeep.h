#pragma once

#include "ATInfo.h"
#include "HDRFloat.h"
#include "HDRFloatComplex.h"
#include "LAInfoDeep.h"
#include "LAInfoI.h"
#include "LAstep.h"
#include <cstddef>
#include <type_traits>

namespace FractalShark::LA {

template <class CpuRow, class GpuRow>
constexpr void
CheckRowLayout()
{
    static_assert(std::is_standard_layout_v<CpuRow> && std::is_standard_layout_v<GpuRow>);
    // Legacy double-float copies normalize their components. Matching uploads intentionally
    // preserve stored bits, as they did before compaction; assignment is a memberwise copy.
    static_assert(std::is_trivially_copy_assignable_v<CpuRow> &&
                  std::is_trivially_copy_assignable_v<GpuRow>);
    static_assert(sizeof(CpuRow) == sizeof(GpuRow) && alignof(CpuRow) == alignof(GpuRow));
    static_assert(offsetof(CpuRow, Ref) == offsetof(GpuRow, Ref));
    static_assert(offsetof(CpuRow, ZCoeff) == offsetof(GpuRow, ZCoeff));
    static_assert(offsetof(CpuRow, CCoeff) == offsetof(GpuRow, CCoeff));
    static_assert(offsetof(CpuRow, LAThreshold) == offsetof(GpuRow, LAThreshold));
    static_assert(offsetof(CpuRow, LAi) == offsetof(GpuRow, LAi));
    if constexpr (std::is_same_v<typename CpuRow::HDRFloat, ::HDRFloat<float>> &&
                  std::is_same_v<decltype(CpuRow{}.LAi.StepLength), uint32_t>) {
        static_assert(sizeof(CpuRow) == 52 && sizeof(GpuRow) == 52);
        static_assert(std::is_trivially_copyable_v<CpuRow> && std::is_trivially_copyable_v<GpuRow>);
    }
}

} // namespace FractalShark::LA

template <typename IterType, class Float, class SubType> class GPU_LAInfoDeep {
public:
    static constexpr bool IsHDR = std::is_same<Float, ::HDRFloat<float>>::value ||
                                  std::is_same<Float, ::HDRFloat<double>>::value ||
                                  std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
                                  std::is_same<Float, ::HDRFloat<CudaDblflt<dblflt>>>::value;
    using HDRFloat = Float;
    using HDRFloatComplex =
        std::conditional_t<IsHDR, ::HDRFloatComplex<SubType>, ::FloatComplex<SubType>>;

    HDRFloatComplex Ref;
    HDRFloatComplex ZCoeff;
    HDRFloatComplex CCoeff;
    HDRFloat LAThreshold;
    LAInfoI<IterType> LAi;

    GPU_LAInfoDeep() = default;
    GPU_LAInfoDeep(const GPU_LAInfoDeep &) = default;
    GPU_LAInfoDeep &operator=(const GPU_LAInfoDeep &) = default;

    template <class Float2, class SubType2>
    CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType> &operator=(
        const GPU_LAInfoDeep<IterType, Float2, SubType2> &other);

    template <class Float2, class SubType2, PerturbExtras PExtras2>
    CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType> &operator=(
        const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other);

    CUDA_CRAP GPU_LAstep<IterType, Float, SubType> Prepare(HDRFloatComplex dz) const;
    CUDA_CRAP HDRFloatComplex getRef() const;
    CUDA_CRAP HDRFloatComplex Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const;
    CUDA_CRAP const LAInfoI<IterType> &GetLAi() const;
};

template <typename IterType, class Float, class SubType>
template <class Float2, class SubType2>
CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType> &
GPU_LAInfoDeep<IterType, Float, SubType>::operator=(
    const GPU_LAInfoDeep<IterType, Float2, SubType2> &other)
{
    if (this == &other) {
        return *this;
    }

    this->Ref = HDRFloatComplex(other.Ref);
    this->LAThreshold = HDRFloat(other.LAThreshold);
    this->ZCoeff = HDRFloatComplex(other.ZCoeff);
    this->CCoeff = HDRFloatComplex(other.CCoeff);
    this->LAi = other.LAi;

    return *this;
}

template <typename IterType, class Float, class SubType>
template <class Float2, class SubType2, PerturbExtras PExtras2>
CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType> &
GPU_LAInfoDeep<IterType, Float, SubType>::operator=(
    const LAInfoDeep<IterType, Float2, SubType2, PExtras2> &other)
{

    this->Ref = HDRFloatComplex(other.Ref);
    this->LAThreshold = HDRFloat(other.LAThreshold);
    this->ZCoeff = HDRFloatComplex(other.ZCoeff);
    this->CCoeff = HDRFloatComplex(other.CCoeff);
    this->LAi = other.LAi;

    return *this;
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAstep<IterType, Float, SubType>
GPU_LAInfoDeep<IterType, Float, SubType>::Prepare(HDRFloatComplex dz) const
{
    return FractalShark::LA::Prepare<GPU_LAstep<IterType, Float, SubType>>(Ref, dz, LAThreshold);
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAInfoDeep<IterType, Float, SubType>::getRef() const
{
    return Ref;
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP GPU_LAInfoDeep<IterType, Float, SubType>::HDRFloatComplex
GPU_LAInfoDeep<IterType, Float, SubType>::Evaluate(HDRFloatComplex newdz, HDRFloatComplex dc) const
{
    return FractalShark::LA::Evaluate(newdz, dc, ZCoeff, CCoeff);
}

template <typename IterType, class Float, class SubType>
CUDA_CRAP const LAInfoI<IterType> &
GPU_LAInfoDeep<IterType, Float, SubType>::GetLAi() const
{
    return this->LAi;
}
