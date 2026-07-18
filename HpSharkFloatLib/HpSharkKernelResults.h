#pragma once

// GPU kernel communication structs for reference orbit and Newton-Raphson kernels.
// Included from HpSharkFloat.h after the HpSharkFloat class definition.
// Do not include this header directly — include HpSharkFloat.h instead.

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
#endif // !__CUDA_ARCH__

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif
template <class SharkFloatParams> struct alignas(16) HpSharkComboResults {
    SharkNTT::RootTables Roots;
    alignas(16) HpSharkFloat<SharkFloatParams> A;
    alignas(16) HpSharkFloat<SharkFloatParams> B;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultX2;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2XY;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultY2;

    // NR derivative: dz/dc inputs and multiply outputs (gated by EnableNewtonRaphson)
    alignas(16) HpSharkFloat<SharkFloatParams> DzdcReal;
    alignas(16) HpSharkFloat<SharkFloatParams> DzdcImag;
    alignas(16) HpSharkFloat<SharkFloatParams> ResultW0; // dzdcR * 2zR
    alignas(16) HpSharkFloat<SharkFloatParams> ResultW1; // dzdcI * 2zI
    alignas(16) HpSharkFloat<SharkFloatParams> ResultW2; // dzdcR * 2zI
    alignas(16) HpSharkFloat<SharkFloatParams> ResultW3; // dzdcI * 2zR
};

template <class SharkFloatParams> struct HpSharkAddComboResults {
    alignas(16) HpSharkFloat<SharkFloatParams> A_X2;
    alignas(16) HpSharkFloat<SharkFloatParams> B_Y2;
    alignas(16) HpSharkFloat<SharkFloatParams> C_A;
    alignas(16) HpSharkFloat<SharkFloatParams> D_2X;
    alignas(16) HpSharkFloat<SharkFloatParams> E_B;
    alignas(16) HpSharkFloat<SharkFloatParams> Result1_A_B_C;
    alignas(16) HpSharkFloat<SharkFloatParams> Result2_D_E;

    // NR derivative add inputs and outputs (gated by EnableNewtonRaphson)
    alignas(16) HpSharkFloat<SharkFloatParams> W0;             // dzdcR * 2zR (input from multiply)
    alignas(16) HpSharkFloat<SharkFloatParams> W1;             // dzdcI * 2zI
    alignas(16) HpSharkFloat<SharkFloatParams> W2;             // dzdcR * 2zI
    alignas(16) HpSharkFloat<SharkFloatParams> W3;             // dzdcI * 2zR
    alignas(16) HpSharkFloat<SharkFloatParams> One;            // constant 1.0 for dzdc +1
    alignas(16) HpSharkFloat<SharkFloatParams> ResultDzdcReal; // W0 - W1 + 1
    alignas(16) HpSharkFloat<SharkFloatParams> ResultDzdcImag; // W2 + W3
};

// Ref2 evaluates its fused NTT pipeline cooperatively. The descriptor lives on
// the device; every pointed-to buffer is allocated once by the Ref2 invocation
// glue and reused for the lifetime of the orbit session.
struct alignas(16) HpSharkReference2CarryPrefixDescriptor {
    uint64_t AggregateTransform;
    uint64_t PrefixTransform;
    uint32_t State;
    uint32_t Padding;
};

template <class SharkFloatParams> struct HpSharkReference2Workspace {
    static constexpr uint32_t MaxFusedN = 32u * 1024u * 1024u;
    static constexpr uint32_t MaxFusedStages = 25;
    static constexpr uint32_t MaxFusedLimbs = (MaxFusedN * 16u) / 32u + 4u;
    static constexpr uint32_t MaxCarryPrefixParts = (MaxFusedLimbs + 31u) / 32u;

    uint64_t *ZReal;
    uint64_t *ZImag;
    uint64_t *CReal;
    uint64_t *CImag;
    uint64_t *DzdcReal;
    uint64_t *DzdcImag;
    uint64_t *One;
    uint64_t *RealOutput;
    uint64_t *ImagOutput;
    uint64_t *DzdcRealOutput;
    uint64_t *DzdcImagOutput;
    uint64_t *Product;
    int64_t *RealLimbs;
    int64_t *ImagLimbs;
    int64_t *DzdcRealLimbs;
    int64_t *DzdcImagLimbs;
    uint32_t *MagnitudeDigits;
    uint32_t *Magnitude;
    uint64_t *CarryPrefixTransforms;
    HpSharkReference2CarryPrefixDescriptor *CarryPrefixDescriptors;
    uint32_t *CarryPrefixControl;
    SharkNTT::RootTables Roots;
    uint32_t CachedN;
};

template <class SharkFloatParams> struct HpSharkReferenceResults {

    alignas(16) typename SharkFloatParams::Float RadiusY;
    alignas(16) HpSharkComboResults<SharkFloatParams> Multiply;
    alignas(16) HpSharkAddComboResults<SharkFloatParams> Add;
    alignas(16) PeriodicityResult PeriodicityStatus;
    alignas(16) typename SharkFloatParams::Float dzdcX;
    alignas(16) typename SharkFloatParams::Float dzdcY;
    alignas(16) typename SharkFloatParams::Float d2Real;
    alignas(16) typename SharkFloatParams::Float d2Imag;
    alignas(16) uint64_t OutputIterCount;
    alignas(16) uint64_t MaxRuntimeIters;

    static constexpr auto MaxOutputIters = 1024;
    alignas(16) typename SharkFloatParams::ReferenceIterT OutputIters[MaxOutputIters];

    // Device-visible Ref2 storage descriptor.  It is null until the Ref2
    // invocation path initializes its fixed-capacity workspace.
    alignas(16) HpSharkReference2Workspace<SharkFloatParams> *Reference2Workspace;

    // Host only
    alignas(16) HpSharkReferenceResults<SharkFloatParams> *comboGpu;
    alignas(16) uint64_t *d_tempProducts;
    alignas(16) uintptr_t stream; // cudaStream_t
    alignas(16) void *kernelArgs[3];
    alignas(16) void *d_reference2WorkspaceStorage;
    alignas(16) size_t reference2WorkspaceStorageBytes;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif
