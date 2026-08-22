#pragma once

// GPU kernel communication structures for reference-orbit and Newton-Raphson kernels.
// Include HpSharkFloat.h rather than including this header directly.

enum class PeriodicityResult { Unknown, Continue, PeriodFound, Escaped };

#if !defined(__CUDA_ARCH__)
[[maybe_unused]] static std::string
PeriodicityStrResult(PeriodicityResult periodicityStatus)
{
    switch (periodicityStatus) {
        case PeriodicityResult::Continue:
            return "Continue";
        case PeriodicityResult::PeriodFound:
            return "PeriodFound";
        case PeriodicityResult::Escaped:
            return "Escaped";
        default:
            return "Unknown";
    }
}
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

struct alignas(16) HpSharkReferencePackedCarryPrefixDescriptor {
    uint32_t AggregateTransform;
    uint32_t PrefixTransform;
    uint32_t State;
    uint32_t Padding;
};

enum class HpSharkReferenceIterationKind : uint32_t {
    Zero = 0u,
    LinearOnly = 1u,
    Ntt = 2u,
};

constexpr uint32_t HpSharkReferencePlanRealProduct = 1u << 0;
constexpr uint32_t HpSharkReferencePlanImagProduct = 1u << 1;
constexpr uint32_t HpSharkReferencePlanDzdcP1 = 1u << 2;
constexpr uint32_t HpSharkReferencePlanDzdcP2 = 1u << 3;
constexpr uint32_t HpSharkReferencePlanDzdcP3 = 1u << 4;
constexpr uint32_t HpSharkReferencePlanRealLinear = 1u << 5;
constexpr uint32_t HpSharkReferencePlanImagLinear = 1u << 6;
constexpr uint32_t HpSharkReferencePlanDzdcOne = 1u << 7;

struct alignas(16) HpSharkReferenceIterationPlan {
    uint32_t Kind;
    uint32_t PlanSlot;
    uint32_t ActiveN;
    uint32_t LimbCount;
    uint32_t Flags;

    uint32_t ZRealCoefficientShift;
    uint32_t ZImagCoefficientShift;
    uint32_t DzdcRealCoefficientShift;
    uint32_t DzdcImagCoefficientShift;
    uint32_t ZRealResidualBitShift;
    uint32_t ZImagResidualBitShift;
    uint32_t DzdcRealResidualBitShift;
    uint32_t DzdcImagResidualBitShift;

    uint32_t RealCoefficientCount;
    uint32_t ImagCoefficientCount;
    uint32_t DzdcRealCoefficientCount;
    uint32_t DzdcImagCoefficientCount;

    int32_t RealExponent;
    int32_t ImagExponent;
    int32_t DzdcRealExponent;
    int32_t DzdcImagExponent;

    uint64_t RealProductBitOffset;
    uint64_t ImagProductBitOffset;
    uint64_t DzdcRealProductBitOffset;
    uint64_t DzdcImagProductBitOffset;
    uint64_t RealLinearBitOffset;
    uint64_t ImagLinearBitOffset;
    uint64_t DzdcRealLinearBitOffset;
};

template <class SharkFloatParams> struct HpSharkReferenceWorkspace {
    static constexpr uint32_t MaxFusedN = 32u * 1024u * 1024u;
    static constexpr uint32_t MaxFusedStages = 25;
    static constexpr uint32_t MinFusedN =
        SharkNTT::NextPow2U32(static_cast<uint32_t>(SharkFloatParams::ReferenceNTTPlan.L));
    static constexpr uint32_t MinFusedStages = SharkNTT::CeilLog2U32(MinFusedN);
    static constexpr uint32_t PlanCacheEntryCount = MaxFusedStages - MinFusedStages + 1u;
    static constexpr uint32_t MaxFusedLimbs = (MaxFusedN * 16u) / 32u + 4u;
    static constexpr uint32_t MaxCarryPrefixParts = (MaxFusedLimbs + 31u) / 32u;
    static constexpr uint32_t CarryPrefixControlCount = 4u;

    static_assert(MinFusedN >= 2u && (MinFusedN & (MinFusedN - 1u)) == 0u);
    static_assert(PlanCacheEntryCount <= 32u);

    uint64_t *ZReal;
    uint64_t *ZImag;
    uint64_t *DzdcReal;
    uint64_t *DzdcImag;
    uint64_t *RealOutput;
    uint64_t *ImagOutput;
    uint64_t *DzdcRealOutput;
    uint64_t *DzdcImagOutput;
    int64_t *RealLimbs;
    int64_t *ImagLimbs;
    int64_t *DzdcRealLimbs;
    int64_t *DzdcImagLimbs;
    uint32_t *MagnitudeDigits;
    uint32_t *Magnitude;
    uint64_t *StageOmegas;
    uint64_t *StageOmegasInverse;
    uint64_t *ForwardTwiddles;
    uint64_t *InverseTwiddles;
    SharkNTT::Plan Plans[PlanCacheEntryCount];
    SharkNTT::RootTables PlanRoots[PlanCacheEntryCount];
    uint32_t ValidPlanMask;
    uint32_t GeneratedStages;
    uint32_t ActualPrecisionLimbs;
    uint32_t IgnoredPrecisionBits;
    uint32_t ActiveMinFusedN;
    uint32_t ActiveMaxFusedN;
    uint32_t ActiveMinFusedStages;
    uint32_t ActiveMaxFusedStages;
    uint32_t ActiveMaxFusedLimbs;
    uint32_t ActiveMaxCarryPrefixParts;
    uint32_t ActivePlanCacheEntryCount;
    HpSharkReferenceIterationPlan IterationPlan;
};

template <class SharkFloatParams>
constexpr uint32_t
HpSharkReferenceOneShotRequiredStage()
{
    constexpr uint32_t ProductCoefficientCount =
        2u * static_cast<uint32_t>(SharkFloatParams::ReferenceNTTPlan.L) - 1u;
    constexpr uint32_t ProductRequiredStage =
        SharkNTT::CeilLog2U32(SharkNTT::NextPow2U32(ProductCoefficientCount));
    constexpr uint32_t AlignmentRequiredStage =
        HpSharkReferenceWorkspace<SharkFloatParams>::MinFusedStages + 2u;
    return ProductRequiredStage > AlignmentRequiredStage ? ProductRequiredStage : AlignmentRequiredStage;
}

template <class SharkFloatParams> struct HpSharkReferenceResults {
    alignas(16) typename SharkFloatParams::Float RadiusY;
    alignas(16) HpSharkFloat<SharkFloatParams> ZReal;
    alignas(16) HpSharkFloat<SharkFloatParams> ZImag;
    alignas(16) HpSharkFloat<SharkFloatParams> CReal;
    alignas(16) HpSharkFloat<SharkFloatParams> CImag;
    alignas(16) HpSharkFloat<SharkFloatParams> DzdcReal;
    alignas(16) HpSharkFloat<SharkFloatParams> DzdcImag;
    alignas(16) HpSharkFloat<SharkFloatParams> One;
    alignas(16) PeriodicityResult PeriodicityStatus;
    alignas(16) typename SharkFloatParams::Float DzdcX;
    alignas(16) typename SharkFloatParams::Float DzdcY;
    alignas(16) typename SharkFloatParams::Float D2Real;
    alignas(16) typename SharkFloatParams::Float D2Imag;
    alignas(16) uint64_t OutputIterCount;
    alignas(16) uint64_t MaxRuntimeIters;

    static constexpr size_t MaxOutputIters = 1024;
    alignas(16) typename SharkFloatParams::ReferenceIterT OutputIters[MaxOutputIters];

    alignas(16) HpSharkReferenceWorkspace<SharkFloatParams> *Workspace;

    // Host-side lifecycle state. Device code never dereferences these members.
    alignas(16) HpSharkReferenceResults<SharkFloatParams> *DeviceResults;
    alignas(16) uint64_t *DeviceDebugStorage;
    alignas(16) uintptr_t Stream;
    alignas(16) void *KernelArgs[2];
    alignas(16) void *OwnedWorkspaceStorage;
    alignas(16) size_t OwnedWorkspaceStorageBytes;
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
