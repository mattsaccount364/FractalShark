#include "stdafx.h"

#include "Fractal.h"
#include "PerturbationResults.h"
#include "RefOrbitCalc.h"

#include "Exceptions.h"
#include "ImaginaOrbit.h"
#include "LAReference.h"
#include "MpirSerialization.h"
#include "OrbitParameterPack.h"
#include "PrecisionCalculator.h"
#include "RecommendedSettings.h"
#include "ScopedMpir.h"

#include "HpSharkFloat.h"
#include "KernelInvoke.cuh"

#include <fstream>
#include <math.h>
#include <memory>
#include <vector>

#include <psapi.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>

template <class Type> struct ThreadPtrs {
    std::atomic<Type *> In;
    std::atomic<Type *> Out;
};

#define ENABLE_PREFETCH(ARG0, ARG1) _mm_prefetch(ARG0, ARG1)
// #define ENABLE_PREFETCH(ARG0, ARG1)

#define CheckStartCriteria                                                                              \
    while (true) {                                                                                      \
        _mm_pause();                                                                                    \
        expected = ThreadMemory->In.load(std::memory_order_relaxed);                                    \
        if (expected == nullptr) {                                                                      \
            continue;                                                                                   \
        }                                                                                               \
        if (ThreadMemory->In.compare_exchange_weak(                                                     \
                expected, nullptr, std::memory_order_acquire, std::memory_order_relaxed)) {             \
            break;                                                                                      \
        }                                                                                               \
    }                                                                                                   \
    if (expected == (void *)0x1) {                                                                      \
        break;                                                                                          \
    }                                                                                                   \
    ENABLE_PREFETCH((const char *)expected, _MM_HINT_T0);                                               \
    ok = expected;

#define CheckFinishCriteria                                                                             \
    expected = nullptr;                                                                                 \
    for (;;) {                                                                                          \
        _mm_pause();                                                                                    \
        bool result = ThreadMemory->Out.compare_exchange_weak(expected, ok, std::memory_order_relaxed); \
        if (result) {                                                                                   \
            break;                                                                                      \
        }                                                                                               \
    }

static inline void
prefetch_range(void *addr, std::size_t len)
{
    constexpr uintptr_t prefetch_stride = 64;
    void *vp = addr;
    void *end =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(len));
    while (vp < end) {
        ENABLE_PREFETCH((const char *)vp, _MM_HINT_T0);
        vp = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vp) +
                                      static_cast<uintptr_t>(prefetch_stride));
    }
}

static inline void
PrefetchHighPrec(const HighPrecision &target)
{
    ENABLE_PREFETCH((const char *)target.backend(), _MM_HINT_T0);
    size_t lastindex = abs(target.backend()->_mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target.backend()->_mp_d, total);
}

static inline void
PrefetchHighPrec(const mpf_t &target)
{
    ENABLE_PREFETCH((const char *)&target[0], _MM_HINT_T0);
    size_t lastindex = abs(target[0]._mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target[0]._mp_d, total);
}

RefOrbitCalc::RefOrbitCalc(const Fractal &fractal, uint64_t commitLimitInBytes)
    : m_PerturbationAlg{PerturbationAlg::Auto}, m_Fractal(fractal), m_PerturbationGuessCalcX(0),
      m_PerturbationGuessCalcY(0), m_RefOrbitOptions{AddPointOptions::DontSave}, m_GuessReserveSize(),
      m_GenerationNumber(), m_CommitLimitInBytes{commitLimitInBytes}
{

    // Get number of CPU cores and whether hyperthreading is enabled
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    m_NumCpuCores = sysinfo.dwNumberOfProcessors;

    // Find whether hyperthreading is enabled via GetLogicalProcessorInformation
    m_HyperthreadingEnabled = false;
    DWORD returnLength = 0;
    GetLogicalProcessorInformation(nullptr, &returnLength);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{
            returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)};
        GetLogicalProcessorInformation(buffer.data(), &returnLength);
        for (const auto &info : buffer) {
            if (info.Relationship == RelationProcessorCore) {
                if (info.ProcessorCore.Flags == LTP_PC_SMT) {
                    m_HyperthreadingEnabled = true;
                    break;
                }
            }
        }
    }
}

bool
RefOrbitCalc::RequiresReuse() const
{
    switch (GetPerturbationAlg()) {
        case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4:
            return true;
        default:
            return false;
    }
}

template <typename IterType, class T, PerturbExtras PExtras>
void
RefOrbitCalc::OptimizeMemory()
{
    PROCESS_MEMORY_COUNTERS_EX checkHappy;
    const size_t OMGAlotOfMemory = m_CommitLimitInBytes / 2;

    if (OMGAlotOfMemory == 0) {
        return;
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));

    // Clear out the memory if the type of the orbit is different from the currently selected algorithm.
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        // Iterate over all elements in the vector and remove the ones that are not of the desired type.
        auto removalFn = [](auto &elt) {
            using eltType = std::decay_t<decltype(elt)>;
            if constexpr (std::is_same_v<eltType,
                                         std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>) {
                return false; // Keep the element if it is of the desired type
            } else {
                return true; // Remove the element if it is not of the desired type
            }
        };

        m_C.erase(std::remove_if(m_C.begin(), m_C.end(), removalFn), m_C.end());
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ClearPerturbationResults(PerturbationResultType::MediumRes);
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ClearPerturbationResults(PerturbationResultType::HighRes);
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ::MessageBox(
            nullptr, L"Watch the memory use... this is just a warning", L"", MB_OK | MB_APPLMODAL);
        assert(false);
    }
}

template <typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetLast()
{
    return GetElt<IterType, T, PExtras>(m_C.size() - 1);
}

template <typename IterType, class T, PerturbExtras PExtras>
const PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetLastConst() const
{
    return GetEltConst<IterType, T, PExtras>(m_C.size() - 1);
}

template <typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetElt(size_t i)
{
    // Given std::vector<AwesomeVariantUniquePtr> m_C;
    // We want to get the last element of the vector.
    // Use std::visit

    // This is a lambda that will be called for each element in the vector.
    // It will return the last element.
    auto lambda = [&](auto &elt) -> PerturbationResults<IterType, T, PExtras> * {
        using eltType = std::decay_t<decltype(elt)>;
        if constexpr (std::is_same<eltType,
                                   std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>::value) {
            return elt.get();
        } else {
            return nullptr;
        }
    };

    return std::visit(lambda, m_C[i]);
}

template <typename IterType, class T, PerturbExtras PExtras>
const PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetEltConst(size_t i) const
{
    auto lambda = [&](const auto &elt) -> PerturbationResults<IterType, T, PExtras> * {
        using eltType = std::decay_t<decltype(elt)>;
        if constexpr (std::is_same<eltType,
                                   std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>::value) {
            return elt.get();
        } else {
            return nullptr;
        }
    };

    return std::visit(lambda, m_C[i]);
}

void
RefOrbitCalc::SetOptions(AddPointOptions options)
{
    m_RefOrbitOptions = options;
}

template <typename IterType,
          class T,
          class SubType,
          PerturbExtras PExtras,
          RefOrbitCalc::BenchmarkMode BenchmarkState>
void
RefOrbitCalc::AddPerturbationReferencePoint()
{
    if (m_PerturbationGuessCalcX == HighPrecision{} && m_PerturbationGuessCalcY == HighPrecision{}) {
        m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision(2);
        m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision(2);
    }

    if (GetPerturbationAlg() == PerturbationAlg::ST) {
        AddPerturbationReferencePointST<IterType,
                                        T,
                                        SubType,
                                        false,
                                        BenchmarkState,
                                        PExtras,
                                        ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                     m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MT) {
        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         false,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                      m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::STPeriodicity) {
        AddPerturbationReferencePointST<IterType,
                                        T,
                                        SubType,
                                        true,
                                        BenchmarkState,
                                        PExtras,
                                        ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                     m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3) {
        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                      m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed) {
        // TODO: we're hardcoding SaveForReuse3 here.  This is the single-threaded case.
        // Single threaded only supports one option, so we're hardcoding it here.
        AddPerturbationReferencePointST<IterType,
                                        T,
                                        SubType,
                                        true,
                                        BenchmarkState,
                                        PExtras,
                                        ReuseMode::SaveForReuse3>(m_PerturbationGuessCalcX,
                                                                  m_PerturbationGuessCalcY);

        // If you want to test, use single threaded for both high-precision and
        // medium precision orbit calculations.
        // AddPerturbationReferencePointST<IterType, T, SubType, true, BenchmarkState, PExtras,
        // ReuseMode::SaveForReuse3>(
        //    m_PerturbationGuessCalcX,
        //    m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1) {

        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::SaveForReuse1>(m_PerturbationGuessCalcX,
                                                                   m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2) {
        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::SaveForReuse2>(m_PerturbationGuessCalcX,
                                                                   m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3) {
        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::SaveForReuse3>(m_PerturbationGuessCalcX,
                                                                   m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4) {
        AddPerturbationReferencePointMT3<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::SaveForReuse4>(m_PerturbationGuessCalcX,
                                                                   m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity5) {
        AddPerturbationReferencePointMT5<IterType,
                                         T,
                                         SubType,
                                         true,
                                         BenchmarkState,
                                         PExtras,
                                         ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                      m_PerturbationGuessCalcY);
    } else if (GetPerturbationAlg() == PerturbationAlg::GPU) {
        AddPerturbationReferencePointGPU<uint64_t,
                                         HDRFloat<float>,
                                         float,
                                         true,
                                         BenchmarkState,
                                         PerturbExtras::Disable,
                                         ReuseMode::DontSaveForReuse>(m_PerturbationGuessCalcX,
                                                                      m_PerturbationGuessCalcY);
    }
}

size_t
RefOrbitCalc::GetNextGenerationNumber() const
{
    ++m_GenerationNumber;
    return m_GenerationNumber;
}

template <typename IterType,
          class T,
          class PerturbationResultsType,
          PerturbExtras PExtras,
          RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::InitResults(PerturbationResultsType &results,
                          const HighPrecision &initX,
                          const HighPrecision &initY)
{

    results.InitResults(Reuse,
                        initX,
                        initY,
                        m_Fractal.GetMinX(),
                        m_Fractal.GetMinY(),
                        m_Fractal.GetMaxX(),
                        m_Fractal.GetMaxY(),
                        m_Fractal.GetNumIterations<IterType>(),
                        m_GuessReserveSize);
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          PerturbExtras PExtras,
          RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::AddPerturbationReferencePointST(HighPrecision cx, HighPrecision cy)
{
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PushbackResults(std::move(newArray));
    auto *results = GetLast<IterType, T, PExtras>();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    std::unique_ptr<MPIRBoundedAllocator> boundedAllocator;
    std::unique_ptr<MPIRBumpAllocator> bumpAllocator;

    InitAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);

    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

        results->InitReused();
    }

    {
        mpf_t cx_mpf;
        mpf_init(cx_mpf);
        mpf_set(cx_mpf, cx.backend());

        mpf_t cy_mpf;
        mpf_init(cy_mpf);
        mpf_set(cy_mpf, cy.backend());

        mpf_t zx;
        mpf_init(zx);

        mpf_t zy;
        mpf_init(zy);

        mpf_t zx2;
        mpf_init(zx2);

        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        IterTypeFull i;

        const T small_float = T((SubType)1.1754944e-38);
        // Note: results->bad is not here.  See end of this function.
        SubType glitch = (SubType)0.0000001;

        T dzdcX = T{1};
        T dzdcY = T{0};

        constexpr bool floatOrDouble = std::is_same<T, double>::value || std::is_same<T, float>::value;
        T cx_cast;
        T cy_cast;
        if constexpr (floatOrDouble) {
            cx_cast = (T)mpf_get_d(cx_mpf);
            cy_cast = (T)mpf_get_d(cy_mpf);
        } else {
            int32_t cx_exponent, cy_exponent;
            double cx_mantissa, cy_mantissa;

            cx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cx_mantissa, cx_mpf));
            cy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cy_mantissa, cy_mpf));

            cx_cast = T{cx_exponent, static_cast<SubType>(cx_mantissa)};
            cy_cast = T{cy_exponent, static_cast<SubType>(cy_mantissa)};
        }

        static const T HighOne = T{1.0};
        static const T HighTwo = T{2.0};
        static const T TwoFiftySix = T(256);

        RefOrbitCompressor<IterType, T, PExtras> compressor{
            *results, m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low)};

        SimpleIntermediateOrbitCompressor<IterType, T, PExtras> intermediateCompressor{
            *results, m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Intermediate)};

        MaxIntermediateOrbitCompressor<IterType, T, PExtras> maxIntermediateCompressor{
            *results, m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Intermediate)};

        mpf_set(zx, cx_mpf);
        mpf_set(zy, cy_mpf);

        for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
            mpf_mul_2exp(zx2, zx, 1); // Multiply by 2

            T double_zx;
            T double_zy;

            if constexpr (floatOrDouble) {
                double_zx = (T)mpf_get_d(zx);
                double_zy = (T)mpf_get_d(zy);
            } else {
                double_zx = T{zx};
                double_zy = T{zy};
            }

            if constexpr (PExtras == PerturbExtras::Disable) {
                results->AddUncompressedIteration({double_zx, double_zy});
            } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
                compressor.MaybeAddCompressedIteration({double_zx, double_zy, i + 1});
            } else if constexpr (PExtras == PerturbExtras::Bad) {
                results->AddUncompressedIteration({double_zx, double_zy, false});
            }

            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                          Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                results->AddUncompressedReusedEntry(zx, zy);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                intermediateCompressor.MaybeAddCompressedIteration(zx, zy, i + 1);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                maxIntermediateCompressor.MaybeAddCompressedIteration(zx, zy, i + 1);
            }

            if constexpr (PExtras == PerturbExtras::Bad) {
                const T sq_x = double_zx * double_zx;
                const T sq_y = double_zy * double_zy;
                const T norm = HdrReduce((sq_x + sq_y) * glitch);

                // TODO This is stupid - we can fix this by using double_zx/double_zy:
                const auto zx_reduced = HdrReduce(HdrAbs((T)mpf_get_d(zx)));
                const auto zy_reduced = HdrReduce(HdrAbs((T)mpf_get_d(zy)));
                const bool underflow = (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                                        HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                                        HdrCompareToBothPositiveReducedLE(norm, small_float));
                results->SetBad(underflow);
            }

            if constexpr (Periodicity) {
                // x^2+2*I*x*y-y^2
                // dzdc = 2.0 * z * dzdc + real(1.0);
                // dzdc = 2.0 * (zx + zy * i) * (dzdcX + dzdcY * i) + HighPrecision(1.0);
                // dzdc = 2.0 * (zx * dzdcX + zx * dzdcY * i + zy * i * dzdcX + zy * i * dzdcY * i) +
                // HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i *
                // dzdcX + 2.0 * zy * i * dzdcY * i + HighPrecision(1.0); dzdc = 2.0 * zx * dzdcX + 2.0 *
                // zx * dzdcY * i + 2.0 * zy * i * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0);
                //
                // dzdcX = 2.0 * zx * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0)
                // dzdcY = 2.0 * zx * dzdcY + 2.0 * zy * dzdcX

                HdrReduce(dzdcX);
                auto dzdcX1 = HdrAbs(dzdcX);

                HdrReduce(dzdcY);
                auto dzdcY1 = HdrAbs(dzdcY);

                HdrReduce(double_zx);
                auto zxCopy1 = HdrAbs(double_zx);

                HdrReduce(double_zy);
                auto zyCopy1 = HdrAbs(double_zy);

                T n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

                T r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
                auto n3 = results->GetMaxRadius() * r0 * HighTwo;
                HdrReduce(n3);

                if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                    if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                        results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                        break;
                    }
                } else {
                    auto dzdcXOrig = dzdcX;
                    dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                    dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
                }
            }

            // zx = zx * zx - zy * zy + cx;
            mpf_mul(temp_mpf, zx, zx);
            mpf_mul(temp2_mpf, zy, zy);
            mpf_sub(zx, temp_mpf, temp2_mpf);
            mpf_add(zx, zx, cx_mpf);

            // zy = zx2 * zy + cy;
            mpf_mul(zy, zx2, zy);
            mpf_add(zy, zy, cy_mpf);

            // !!!!!!!!!!!!!!!!!!!!!!!!!
            T tempZX = double_zx + cx_cast;
            T tempZY = double_zy + cy_cast;
            T zn_size = tempZX * tempZX + tempZY * tempZY;
            if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
                break;
            }
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            results->SetBad(false);
        }

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
            maxIntermediateCompressor.CompleteResults();
        }

        results->CompleteResults<Reuse>(bumpAllocator->GetAllocated(1));
        m_GuessReserveSize = results->GetCompressedOrUncompressedOrbitSize();
    } // End of scope for allocators.

    ShutdownAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);
}

void
RefOrbitCalc::GetEstimatedPrecision(uint64_t authoritativePrecisionInBits,
                                    int64_t &deltaPrecision,
                                    int64_t &extraPrecision) const
{

    const bool reuse = RequiresReuse();
    // const bool reuse = false;
    const auto NewPrec = PrecisionCalculator::GetPrecision(
        m_Fractal.GetMinX(), m_Fractal.GetMinY(), m_Fractal.GetMaxX(), m_Fractal.GetMaxY(), reuse);
    // const int64_t NewPrec = m_Fractal.GetMinX().precisionInBits();
    deltaPrecision = static_cast<int64_t>(NewPrec - authoritativePrecisionInBits);
    extraPrecision = static_cast<int64_t>(AuthoritativeReuseExtraPrecisionInBits -
                                          AuthoritativeMinExtraPrecisionInBits);
}

template <typename IterType, class T>
bool
RefOrbitCalc::GetReuseResults(
    const HighPrecision &cx,
    const HighPrecision &cy,
    const PerturbationResults<IterType, T, PerturbExtras::Disable> &existingAuthoritativeResults,
    PerturbationResults<IterType, T, PerturbExtras::Disable> *&outResults)
{

    int64_t deltaPrecision;
    int64_t extraPrecision;
    GetEstimatedPrecision(
        existingAuthoritativeResults.GetAuthoritativePrecisionInBits(), deltaPrecision, extraPrecision);

    // This all generally works and only starts to suffer precision problems after
    // about 2^AuthoritativeReuseExtraPrecisionInBits. The problem naturally is the original
    // reference orbit is calculated only to so many digits.
    if (deltaPrecision >= extraPrecision) {
        //::MessageBox(nullptr, L"Regenerating authoritative orbit is required 1", L"", MB_OK |
        //:MB_APPLMODAL);
        return false;
    }

    // Verify that existing results are for an orbit that's "nearby" the current point
    // we're trying to perturb.
    const auto &existingResultsHiX = existingAuthoritativeResults.GetHiX();
    const auto &existingResultsHiY = existingAuthoritativeResults.GetHiY();
    const auto deltaX = T{abs(cx - existingResultsHiX)};
    const auto deltaY = T{abs(cy - existingResultsHiY)};

    if (HdrCompareToBothPositiveReducedGT(deltaX, existingAuthoritativeResults.GetMaxRadius()) ||
        HdrCompareToBothPositiveReducedGT(deltaY, existingAuthoritativeResults.GetMaxRadius())) {

        // const std::string deltaStr =
        //     cx.str() + ", " + cy.str() + ", " +
        //     existingResultsHiX.str() + ", " + existingResultsHiY.str() + ", " +
        //     deltaX.str() + ", " + deltaY.str();
        // const std::string outputStr = "Regenerating authoritative orbit is required 2: " + deltaStr;
        //::MessageBoxA(nullptr, outputStr.c_str(), "", MB_OK | MB_APPLMODAL);
        return false;
    }

    outResults = GetLast<IterType, T, PerturbExtras::Disable>();
    outResults->SetIntermediateCachedPrecision(deltaPrecision, extraPrecision);
    return true;
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          RefOrbitCalc::ReuseMode Reuse>
bool
RefOrbitCalc::AddPerturbationReferencePointSTReuse(HighPrecision cx, HighPrecision cy)
{
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PushbackResults(std::move(newArray));
    auto *existingResults =
        GetUsefulPerturbationResultsMutable<IterType, T, true, PerturbExtras::Disable>();

    // TODO Lame hack with < 5.
    if (existingResults == nullptr || existingResults->GetPeriodMaybeZero() < 5) {

        m_C.pop_back();
        return false;
    }

    PerturbationResults<IterType, T, PerturbExtras::Disable> *results = nullptr;
    bool reuse = GetReuseResults<IterType, T>(cx, cy, *existingResults, results);
    if (!reuse) {
        m_C.pop_back();
        return false;
    }

    MPIRPrecision prec(AuthoritativeReuseExtraPrecisionInBits);
    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(
        *results, cx, cy);

    mpf_t zx, zy;
    mpf_init(zx);
    mpf_init(zy);

    IterTypeFull i;

    mpf_t HighOne;
    mpf_init(HighOne);
    mpf_set_d(HighOne, 1.0);

    mpf_t HighTwo;
    mpf_init(HighTwo);
    mpf_set_d(HighTwo, 2.0);

    static const T TwoFiftySix = T(256.0);

    mpf_t DeltaReal;
    mpf_init2(DeltaReal, existingResults->GetAuthoritativePrecisionInBits());
    mpf_sub(DeltaReal, cx.backend(), existingResults->GetHiX().backend());
    mpf_set_prec(DeltaReal, AuthoritativeReuseExtraPrecisionInBits);

    mpf_t DeltaImaginary;
    mpf_init2(DeltaImaginary, existingResults->GetAuthoritativePrecisionInBits());
    mpf_sub(DeltaImaginary, cy.backend(), existingResults->GetHiY().backend());
    mpf_set_prec(DeltaImaginary, AuthoritativeReuseExtraPrecisionInBits);

    mpf_t DeltaSub0X;
    mpf_init(DeltaSub0X);
    mpf_set(DeltaSub0X, DeltaReal);

    mpf_t DeltaSub0Y;
    mpf_init(DeltaSub0Y);
    mpf_set(DeltaSub0Y, DeltaImaginary);

    mpf_t DeltaSubNX;
    mpf_init(DeltaSubNX);

    mpf_t DeltaSubNY;
    mpf_init(DeltaSubNY);

    IterTypeFull RefIteration = 0;
    IterTypeFull MaxRefIteration = existingResults->GetCountOrbitEntries() - 1;

    T dzdcX = T{1};
    T dzdcY = T{0};

    T zxCopy;
    T zyCopy;

    T zn_size;
    T normDeltaSubN;

    T tempZXLow;
    T tempZYLow;

    T tempDeltaSubNXLow;
    T tempDeltaSubNYLow;

    mpf_set(zx, cx.backend());
    mpf_set(zy, cy.backend());

    mpf_t DeltaSubNXOrig;
    mpf_init(DeltaSubNXOrig);

    mpf_t DeltaSubNYOrig;
    mpf_init(DeltaSubNYOrig);

    mpf_t temp_mpf;
    mpf_init(temp_mpf);

    mpf_t temp2_mpf;
    mpf_init(temp2_mpf);

    IntermediateRuntimeDecompressor<IterType, T, PerturbExtras::Disable> intermediateCompressor{
        *existingResults};

    constexpr bool floatOrDouble = std::is_same<T, double>::value || std::is_same<T, float>::value;

    const mpf_t *ReuseX = nullptr;
    const mpf_t *ReuseY = nullptr;

    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
        if constexpr (Periodicity) {
            if constexpr (floatOrDouble) {
                zxCopy = (T)mpf_get_d(zx);
                zyCopy = (T)mpf_get_d(zy);
            } else {
                zxCopy = T{zx};
                zyCopy = T{zy};
            }
        }

        mpf_set(DeltaSubNXOrig, DeltaSubNX);
        mpf_set(DeltaSubNYOrig, DeltaSubNY);

        existingResults->GetCompressedReuseEntries(intermediateCompressor, RefIteration, ReuseX, ReuseY);

        // DeltaSubNX =
        //     DeltaSubNXOrig * ((*ReuseX) * HighTwo + DeltaSubNXOrig) -
        //     DeltaSubNYOrig * ((*ReuseY) * HighTwo + DeltaSubNYOrig) +
        //     DeltaSub0X;
        mpf_mul(temp_mpf, *ReuseX, HighTwo);
        mpf_add(temp_mpf, temp_mpf, DeltaSubNXOrig);
        mpf_mul(temp2_mpf, *ReuseY, HighTwo);
        mpf_add(temp2_mpf, temp2_mpf, DeltaSubNYOrig);
        mpf_mul(DeltaSubNX, DeltaSubNXOrig, temp_mpf);
        mpf_mul(temp_mpf, DeltaSubNYOrig, temp2_mpf);
        mpf_sub(DeltaSubNX, DeltaSubNX, temp_mpf);
        mpf_add(DeltaSubNX, DeltaSubNX, DeltaSub0X);

        // DeltaSubNY =
        //     DeltaSubNXOrig * ((*ReuseY) * HighTwo + DeltaSubNYOrig) +
        //     DeltaSubNYOrig * ((*ReuseX) * HighTwo + DeltaSubNXOrig) +
        //     DeltaSub0Y;
        mpf_mul(temp_mpf, *ReuseY, HighTwo);
        mpf_add(temp_mpf, temp_mpf, DeltaSubNYOrig);
        mpf_mul(temp2_mpf, *ReuseX, HighTwo);
        mpf_add(temp2_mpf, temp2_mpf, DeltaSubNXOrig);
        mpf_mul(DeltaSubNY, DeltaSubNXOrig, temp_mpf);
        mpf_mul(temp_mpf, DeltaSubNYOrig, temp2_mpf);
        mpf_add(DeltaSubNY, DeltaSubNY, temp_mpf);
        mpf_add(DeltaSubNY, DeltaSubNY, DeltaSub0Y);

        // tempDeltaSubNXLow = (T)DeltaSubNX;
        // tempDeltaSubNYLow = (T)DeltaSubNY;
        if constexpr (floatOrDouble) {
            tempDeltaSubNXLow = (T)mpf_get_d(DeltaSubNX);
            tempDeltaSubNYLow = (T)mpf_get_d(DeltaSubNY);
            tempZXLow = (T)mpf_get_d(zx);
            tempZYLow = (T)mpf_get_d(zy);
        } else {
            tempDeltaSubNXLow = T{DeltaSubNX};
            tempDeltaSubNYLow = T{DeltaSubNY};
            tempZXLow = T{zx};
            tempZYLow = T{zy};
        }

        ++RefIteration;

        // results->AddUncompressedIteration({ tempZXLow, tempZYLow });

        existingResults->GetCompressedReuseEntries(intermediateCompressor, RefIteration, ReuseX, ReuseY);

        // zx = (*ReuseX) + DeltaSubNX;
        mpf_add(zx, *ReuseX, DeltaSubNX);

        // zy = (*ReuseY) + DeltaSubNY;
        mpf_add(zy, *ReuseY, DeltaSubNY);

        zn_size = tempZXLow * tempZXLow + tempZYLow * tempZYLow;
        HdrReduce(zn_size);
        normDeltaSubN = tempDeltaSubNXLow * tempDeltaSubNXLow + tempDeltaSubNYLow * tempDeltaSubNYLow;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            // DeltaSubNX = zx;
            mpf_set(DeltaSubNX, zx);

            // DeltaSubNY = zy;
            mpf_set(DeltaSubNY, zy);

            RefIteration = 0;
        }

        if constexpr (Periodicity) {
            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

            T r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
            auto n3 = results->GetMaxRadius() * r0 * T(2.0);
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    // TODO: note that this implementation simply does not seem
                    // to work with period-2 orbits.  It's not clear why.
                    // T reducedZx = (T)mpf_get_d(zx);
                    // T reducedZy = (T)mpf_get_d(zy);
                    // results->AddUncompressedIteration({ reducedZx, reducedZy });

                    // Break before adding the result.
                    results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                    break;
                }
            } else {
                auto dzdcXOrig = dzdcX;
                dzdcX = T(2.0) * (zxCopy * dzdcX - zyCopy * dzdcY) + T(1.0);
                dzdcY = T(2.0) * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
            break;
        }

        if constexpr (floatOrDouble) {
            T reducedZx = (T)mpf_get_d(zx);
            T reducedZy = (T)mpf_get_d(zy);

            results->AddUncompressedIteration({reducedZx, reducedZy});
        } else {
            T reducedZx = T{zx};
            T reducedZy = T{zy};

            results->AddUncompressedIteration({reducedZx, reducedZy});
        }
    }

    mpf_clear(zx);
    mpf_clear(zy);
    mpf_clear(HighOne);
    mpf_clear(HighTwo);
    mpf_clear(DeltaReal);
    mpf_clear(DeltaImaginary);
    mpf_clear(DeltaSub0X);
    mpf_clear(DeltaSub0Y);
    mpf_clear(DeltaSubNX);
    mpf_clear(DeltaSubNY);
    mpf_clear(DeltaSubNXOrig);
    mpf_clear(DeltaSubNYOrig);
    mpf_clear(temp_mpf);
    mpf_clear(temp2_mpf);

    results->CompleteResults<ReuseMode::DontSaveForReuse>(nullptr);
    m_GuessReserveSize = results->GetCompressedOrUncompressedOrbitSize();

    return true;
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          RefOrbitCalc::ReuseMode Reuse>
bool
RefOrbitCalc::AddPerturbationReferencePointMT3Reuse(HighPrecision cx, HighPrecision cy)
{
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PushbackResults(std::move(newArray));
    auto *existingResults =
        GetUsefulPerturbationResultsMutable<IterType, T, true, PerturbExtras::Disable>();

    // TODO Lame hack with < 5.
    if (existingResults == nullptr || existingResults->GetPeriodMaybeZero() < 5) {

        m_C.pop_back();
        return false;
    }

    PerturbationResults<IterType, T, PerturbExtras::Disable> *results = nullptr;
    bool reuse = GetReuseResults<IterType, T>(cx, cy, *existingResults, results);
    if (!reuse) {
        m_C.pop_back();
        return false;
    }

    MPIRPrecision prec(AuthoritativeReuseExtraPrecisionInBits);
    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(
        *results, cx, cy);

    mpf_t zx, zy;
    mpf_init(zx);
    mpf_init(zy);

    IterTypeFull i;

    mpf_t HighOne;
    mpf_init(HighOne);
    mpf_set_d(HighOne, 1.0);

    mpf_t HighTwo;
    mpf_init(HighTwo);
    mpf_set_d(HighTwo, 2.0);

    static const T TwoFiftySix = T(256.0);

    mpf_t DeltaReal;
    mpf_init2(DeltaReal, existingResults->GetAuthoritativePrecisionInBits());
    mpf_sub(DeltaReal, cx.backend(), existingResults->GetHiX().backend());
    mpf_set_prec(DeltaReal, AuthoritativeReuseExtraPrecisionInBits);

    mpf_t DeltaImaginary;
    mpf_init2(DeltaImaginary, existingResults->GetAuthoritativePrecisionInBits());
    mpf_sub(DeltaImaginary, cy.backend(), existingResults->GetHiY().backend());
    mpf_set_prec(DeltaImaginary, AuthoritativeReuseExtraPrecisionInBits);

    mpf_t DeltaSub0X;
    mpf_init(DeltaSub0X);
    mpf_set(DeltaSub0X, DeltaReal);

    mpf_t DeltaSub0Y;
    mpf_init(DeltaSub0Y);
    mpf_set(DeltaSub0Y, DeltaImaginary);

    mpf_t DeltaSubNX;
    mpf_init(DeltaSubNX);

    mpf_t DeltaSubNY;
    mpf_init(DeltaSubNY);

    IterTypeFull RefIteration = 0;
    IterTypeFull MaxRefIteration = existingResults->GetCountOrbitEntries() - 1;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    T zxCopy;
    T zyCopy;

    T zn_size;
    T normDeltaSubN;

    mpf_set(zx, cx.backend());
    mpf_set(zy, cy.backend());

    struct ThreadZxData {
        ThreadZxData()
            : ReferenceIteration{}, DeltaSubNXOrig{}, DeltaSubNYOrig{}, DeltaSub0X{}, DeltaSubNX{},
              OutZx{}, OutZxLow{}, OutDeltaSubNXLow{}
        {

            mpf_init(DeltaSubNX);
            mpf_init(OutZx);
        }

        ~ThreadZxData()
        {
            mpf_clear(DeltaSubNX);
            mpf_clear(OutZx);
        }

        IterTypeFull ReferenceIteration;
        const mpf_t *DeltaSubNXOrig;
        const mpf_t *DeltaSubNYOrig;
        const mpf_t *DeltaSub0X;
        mpf_t DeltaSubNX;
        mpf_t OutZx;
        T OutZxLow;
        T OutDeltaSubNXLow;
    };

    struct ThreadZyData {
        ThreadZyData()
            : ReferenceIteration{}, DeltaSubNXOrig{}, DeltaSubNYOrig{}, DeltaSub0Y{}, DeltaSubNY{},
              OutZy{}, OutZyLow{}, OutDeltaSubNYLow{}
        {

            mpf_init(DeltaSubNY);
            mpf_init(OutZy);
        }

        ~ThreadZyData()
        {
            mpf_clear(DeltaSubNY);
            mpf_clear(OutZy);
        }

        IterTypeFull ReferenceIteration;
        const mpf_t *DeltaSubNXOrig;
        const mpf_t *DeltaSubNYOrig;
        const mpf_t *DeltaSub0Y;
        mpf_t DeltaSubNY;
        mpf_t OutZy;
        T OutZyLow;
        T OutDeltaSubNYLow;
    };

    auto *ThreadZxMemory =
        (ThreadPtrs<ThreadZxData> *)_aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    if (ThreadZxMemory == nullptr) {
        throw FractalSharkSeriousException("Memory allocation failure site " + std::to_string(__LINE__));
    }

    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto *ThreadZyMemory =
        (ThreadPtrs<ThreadZyData> *)_aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    if (ThreadZyMemory == nullptr) {
        throw FractalSharkSeriousException("Memory allocation failure site " + std::to_string(__LINE__));
    }

    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    constexpr bool floatOrDouble = std::is_same<T, double>::value || std::is_same<T, float>::value;

    auto ThreadSqZx = [&](ThreadPtrs<ThreadZxData> *ThreadMemory) {
        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        IntermediateRuntimeDecompressor<IterType, T, PerturbExtras::Disable> intermediateDecompressor{
            *existingResults};

        IntermediateMaxRuntimeDecompressor<IterType, T, PerturbExtras::Disable>
            maxIntermediateDecompressor{*existingResults};

        const mpf_t *ReuseX{};
        const mpf_t *ReuseY{};

        for (;;) {
            ThreadZxData *expected = ThreadMemory->In.load();
            ThreadZxData *ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0X);
            // PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            // PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            // TODO factor out
            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                          Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                existingResults->GetUncompressedReuseEntries(ok->ReferenceIteration, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                existingResults->GetCompressedReuseEntries(
                    intermediateDecompressor, ok->ReferenceIteration, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                existingResults->GetMaxCompressedReuseEntries(
                    maxIntermediateDecompressor, ok->ReferenceIteration, ReuseX, ReuseY);
            } else {
                assert(false);
            }

            // ok->DeltaSubNX =
            //     (*ok->DeltaSubNXOrig) * ((*ReuseX) * HighTwo + (*ok->DeltaSubNXOrig)) -
            //     (*ok->DeltaSubNYOrig) * ((*ReuseY) * HighTwo + (*ok->DeltaSubNYOrig)) +
            //     (*ok->DeltaSub0X);

            mpf_mul(temp_mpf, *ReuseX, HighTwo);
            mpf_add(temp_mpf, temp_mpf, *ok->DeltaSubNXOrig);
            mpf_mul(temp2_mpf, *ReuseY, HighTwo);
            mpf_add(temp2_mpf, temp2_mpf, *ok->DeltaSubNYOrig);
            mpf_mul(ok->DeltaSubNX, *ok->DeltaSubNXOrig, temp_mpf);
            mpf_mul(temp_mpf, *ok->DeltaSubNYOrig, temp2_mpf);
            mpf_sub(ok->DeltaSubNX, ok->DeltaSubNX, temp_mpf);
            mpf_add(ok->DeltaSubNX, ok->DeltaSubNX, *ok->DeltaSub0X);

            // TODO
            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                          Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                existingResults->GetUncompressedReuseEntries(ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                existingResults->GetCompressedReuseEntries(
                    intermediateDecompressor, ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                existingResults->GetMaxCompressedReuseEntries(
                    maxIntermediateDecompressor, ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else {
                assert(false);
            }

            mpf_add(ok->OutZx, *ReuseX, ok->DeltaSubNX);
            if constexpr (floatOrDouble) {
                ok->OutDeltaSubNXLow = (T)mpf_get_d(ok->DeltaSubNX);
                ok->OutZxLow = (T)mpf_get_d(ok->OutZx);
            } else {
                ok->OutDeltaSubNXLow = T(ok->DeltaSubNX);
                ok->OutZxLow = T{ok->OutZx};
            }

            // Give result back.
            CheckFinishCriteria;
        }

        mpf_clear(temp_mpf);
        mpf_clear(temp2_mpf);
    };

    auto ThreadSqZy = [&](ThreadPtrs<ThreadZyData> *ThreadMemory) {
        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        IntermediateRuntimeDecompressor<IterType, T, PerturbExtras::Disable> intermediateDecompressor{
            *existingResults};

        IntermediateMaxRuntimeDecompressor<IterType, T, PerturbExtras::Disable>
            maxIntermediateDecompressor{*existingResults};

        const mpf_t *ReuseX{};
        const mpf_t *ReuseY{};

        for (;;) {
            ThreadZyData *expected = ThreadMemory->In.load();
            ThreadZyData *ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0Y);
            // PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            // PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            // TODO factor out
            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                          Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                existingResults->GetUncompressedReuseEntries(ok->ReferenceIteration, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                existingResults->GetCompressedReuseEntries(
                    intermediateDecompressor, ok->ReferenceIteration, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                existingResults->GetMaxCompressedReuseEntries(
                    maxIntermediateDecompressor, ok->ReferenceIteration, ReuseX, ReuseY);
            } else {
                assert(false);
            }

            // ok->DeltaSubNY =
            //     (*ok->DeltaSubNXOrig) * ((*ReuseY) * HighTwo + (*ok->DeltaSubNYOrig)) +
            //     (*ok->DeltaSubNYOrig) * ((*ReuseX) * HighTwo + (*ok->DeltaSubNXOrig)) +
            //     (*ok->DeltaSub0Y);

            mpf_mul(temp_mpf, *ReuseY, HighTwo);
            mpf_add(temp_mpf, temp_mpf, *ok->DeltaSubNYOrig);
            mpf_mul(temp2_mpf, *ReuseX, HighTwo);
            mpf_add(temp2_mpf, temp2_mpf, *ok->DeltaSubNXOrig);
            mpf_mul(ok->DeltaSubNY, *ok->DeltaSubNXOrig, temp_mpf);
            mpf_mul(temp_mpf, *ok->DeltaSubNYOrig, temp2_mpf);
            mpf_add(ok->DeltaSubNY, ok->DeltaSubNY, temp_mpf);
            mpf_add(ok->DeltaSubNY, ok->DeltaSubNY, *ok->DeltaSub0Y);

            // TODO
            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                          Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                existingResults->GetUncompressedReuseEntries(ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                existingResults->GetCompressedReuseEntries(
                    intermediateDecompressor, ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                existingResults->GetMaxCompressedReuseEntries(
                    maxIntermediateDecompressor, ok->ReferenceIteration + 1, ReuseX, ReuseY);
            } else {
                assert(false);
            }

            mpf_add(ok->OutZy, *ReuseY, ok->DeltaSubNY);
            if constexpr (floatOrDouble) {
                ok->OutDeltaSubNYLow = (T)mpf_get_d(ok->DeltaSubNY);
                ok->OutZyLow = (T)mpf_get_d(ok->OutZy);
            } else {
                ok->OutDeltaSubNYLow = T(ok->DeltaSubNY);
                ok->OutZyLow = T{ok->OutZy};
            }

            // Give result back.
            CheckFinishCriteria;
        }

        mpf_clear(temp_mpf);
        mpf_clear(temp2_mpf);
    };

    auto *threadZxdata = (ThreadZxData *)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto *threadZydata = (ThreadZyData *)_aligned_malloc(sizeof(ThreadZyData), 64);

    new (threadZxdata)(ThreadZxData){};
    new (threadZydata)(ThreadZyData){};

    std::unique_ptr<std::thread> tZx(DEBUG_NEW std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(DEBUG_NEW std::thread(ThreadSqZy, ThreadZyMemory));

    ScopedAffinity scopedAffinity{
        *this, GetCurrentThread(), tZx->native_handle(), tZy->native_handle(), nullptr};

    ThreadZxData *expectedZx = nullptr;
    ThreadZyData *expectedZy = nullptr;

    T tempDeltaSubNXLow = T(0);
    T tempDeltaSubNYLow = T(0);

    T Low256 = T(256);
    T tempZYLow = T(0);
    T tempZXLow = T(0);

    bool RanOnce = false;
    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
        threadZxdata->ReferenceIteration = RefIteration;
        threadZxdata->DeltaSubNXOrig = &DeltaSubNX;
        threadZxdata->DeltaSubNYOrig = &DeltaSubNY;
        threadZxdata->DeltaSub0X = &DeltaSub0X;

        ThreadZxMemory->In.store(threadZxdata, std::memory_order_release);

        threadZydata->ReferenceIteration = RefIteration;
        threadZydata->DeltaSubNXOrig = &DeltaSubNX;
        threadZydata->DeltaSubNYOrig = &DeltaSubNY;
        threadZydata->DeltaSub0Y = &DeltaSub0Y;

        ThreadZyMemory->In.store(threadZydata, std::memory_order_relaxed);

        ++RefIteration;

        // TODO Lame conditional.
        // Could erase first elt but that's O(n) in size of vector
        if (RanOnce) {
            results->AddUncompressedIteration({tempZXLow, tempZYLow});
        }

        if constexpr (Periodicity) {
            if constexpr (floatOrDouble) {
                zxCopy = (T)mpf_get_d(zx);
                zyCopy = (T)mpf_get_d(zy);
            } else {
                zxCopy = T{zx};
                zyCopy = T{zy};
            }

            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

            T r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
            auto n3 = results->GetMaxRadius() * r0 * T(2.0);
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    // TODO: note that this implementation simply does not seem
                    // to work with period-2 orbits.  It's not clear why.
                    // T reducedZx = (T)mpf_get_d(zx);
                    // T reducedZy = (T)mpf_get_d(zy);
                    // results->AddUncompressedIteration({ reducedZx, reducedZy });

                    // Break before adding the result.
                    results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                    break;
                }
            } else {
                auto dzdcXOrig = dzdcX;
                dzdcX = T(2.0) * (zxCopy * dzdcX - zyCopy * dzdcY) + T(1.0);
                dzdcY = T(2.0) * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        bool done1 = false;
        bool done2 = false;

        for (;;) {
            expectedZy = threadZydata;

            _mm_pause();
            if (!done2 && ThreadZyMemory->Out.compare_exchange_weak(
                              expectedZy, nullptr, std::memory_order_release)) {
                done2 = true;
                mpf_set(zy, threadZydata->OutZy);

                tempDeltaSubNYLow = threadZydata->OutDeltaSubNYLow;
                tempZYLow = threadZydata->OutZyLow;
            }

            expectedZx = threadZxdata;

            _mm_pause();
            if (!done1 && ThreadZxMemory->Out.compare_exchange_weak(
                              expectedZx, nullptr, std::memory_order_release)) {
                done1 = true;
                mpf_set(zx, threadZxdata->OutZx);

                tempDeltaSubNXLow = threadZxdata->OutDeltaSubNXLow;
                tempZXLow = threadZxdata->OutZxLow;
            }

            if (done1 && done2) {
                break;
            }
        }

        mpf_set(DeltaSubNX, threadZxdata->DeltaSubNX);
        mpf_set(DeltaSubNY, threadZydata->DeltaSubNY);

        zn_size = tempZXLow * tempZXLow + tempZYLow * tempZYLow;
        HdrReduce(zn_size);
        normDeltaSubN = tempDeltaSubNXLow * tempDeltaSubNXLow + tempDeltaSubNYLow * tempDeltaSubNYLow;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedGT(zn_size, Low256)) {
            break;
        }

        if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            mpf_set(DeltaSubNX, zx);
            mpf_set(DeltaSubNY, zy);
            RefIteration = 0;
        }

        RanOnce = true;
    }

    bool res1 = false, res2 = false;
    while (!res1) {
        expectedZx = nullptr;
        res1 = ThreadZxMemory->In.compare_exchange_strong(
            expectedZx, (ThreadZxData *)0x1, std::memory_order_release);
    }

    while (!res2) {
        expectedZy = nullptr;
        res2 = ThreadZyMemory->In.compare_exchange_strong(
            expectedZy, (ThreadZyData *)0x1, std::memory_order_release);
    }

    // results->x.erase(results->x.begin());
    // results->y.erase(results->y.begin());

    tZx->join();
    tZy->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);

    mpf_clear(zx);
    mpf_clear(zy);
    mpf_clear(HighOne);
    mpf_clear(HighTwo);
    mpf_clear(DeltaReal);
    mpf_clear(DeltaImaginary);
    mpf_clear(DeltaSub0X);
    mpf_clear(DeltaSub0Y);
    mpf_clear(DeltaSubNX);
    mpf_clear(DeltaSubNY);

    results->CompleteResults<ReuseMode::DontSaveForReuse>(nullptr);
    m_GuessReserveSize = results->GetCompressedOrUncompressedOrbitSize();

    return true;
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          PerturbExtras PExtras,
          RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::AddPerturbationReferencePointMT3(HighPrecision cx, HighPrecision cy)
{
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PushbackResults(std::move(newArray));
    auto *results = GetLast<IterType, T, PExtras>();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    std::unique_ptr<MPIRBoundedAllocator> boundedAllocator;
    std::unique_ptr<MPIRBumpAllocator> bumpAllocator;

    InitAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);

    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                  Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

        results->InitReused();
    }

    std::unique_ptr<ThreadMemory> reusedAllocator;

    {
        mpf_t cx_mpf;
        mpf_init(cx_mpf);
        mpf_set(cx_mpf, cx.backend());

        mpf_t cy_mpf;
        mpf_init(cy_mpf);
        mpf_set(cy_mpf, cy.backend());

        mpf_t zx;
        mpf_init(zx);

        mpf_t zy;
        mpf_init(zy);

        mpf_t zx2;
        mpf_init(zx2);

        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        constexpr bool floatOrDouble = std::is_same<T, double>::value || std::is_same<T, float>::value;
        T cx_cast;
        T cy_cast;
        if constexpr (floatOrDouble) {
            cx_cast = (T)mpf_get_d(cx_mpf);
            cy_cast = (T)mpf_get_d(cy_mpf);
        } else {
            int32_t cx_exponent, cy_exponent;
            double cx_mantissa, cy_mantissa;

            cx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cx_mantissa, cx_mpf));
            cy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cy_mantissa, cy_mpf));

            cx_cast = T{cx_exponent, static_cast<SubType>(cx_mantissa)};
            cy_cast = T{cy_exponent, static_cast<SubType>(cy_mantissa)};
        }

        T dzdcX = T{1.0};
        T dzdcY = T{0.0};

        const T small_float = T((SubType)1.1754944e-38);
        // Note: results->bad is not here.  See end of this function.
        SubType glitch = (SubType)0.0000001;

        struct ThreadZxData {
            ThreadZxData()
            {
                mpf_init(zx);
                mpf_init(zx_sq);
                zx_low = T{0.0};
            }

            ~ThreadZxData()
            {
                mpf_clear(zx);
                mpf_clear(zx_sq);
            }

            mpf_t zx;
            mpf_t zx_sq;
            T zx_low;
        };

        struct ThreadZyData {
            ThreadZyData()
            {
                mpf_init(zy);
                mpf_init(zy_sq);
                zy_low = T{0.0};
            }

            ~ThreadZyData()
            {
                mpf_clear(zy);
                mpf_clear(zy_sq);
            }

            mpf_t zy;
            mpf_t zy_sq;
            T zy_low;
        };

        struct ThreadReusedData {
            ThreadReusedData()
            {
                mpf_init(zx);
                mpf_init(zy);
            }

            ~ThreadReusedData()
            {
                mpf_clear(zx);
                mpf_clear(zy);
            }

            mpf_t zx;
            mpf_t zy;
        };

        auto *ThreadZxMemory =
            (ThreadPtrs<ThreadZxData> *)_aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
        if (ThreadZxMemory == nullptr) {
            throw FractalSharkSeriousException("Memory allocation failure site " +
                                               std::to_string(__LINE__));
        }

        memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

        auto *ThreadZyMemory =
            (ThreadPtrs<ThreadZyData> *)_aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
        if (ThreadZyMemory == nullptr) {
            throw FractalSharkSeriousException("Memory allocation failure site " +
                                               std::to_string(__LINE__));
        }

        memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

        auto *ThreadReusedMemory =
            (ThreadPtrs<ThreadReusedData> *)_aligned_malloc(sizeof(ThreadPtrs<ThreadReusedData>), 64);
        if (ThreadReusedMemory == nullptr) {
            throw FractalSharkSeriousException("Memory allocation failure site " +
                                               std::to_string(__LINE__));
        }

        memset(ThreadReusedMemory, 0, sizeof(*ThreadReusedMemory));

        auto InitTls = [&]() {
            if constexpr (Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse4) {
                boundedAllocator->InitTls();
            } else {
                bumpAllocator->InitTls();
            }
        };

        auto ShutdownTls = [&]() {
            if constexpr (Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3 &&
                          Reuse != RefOrbitCalc::ReuseMode::SaveForReuse4) {
                boundedAllocator->ShutdownTls();
            } else {
                bumpAllocator->ShutdownTls();
            }
        };

        auto ThreadSqZx = [&InitTls, &ShutdownTls](ThreadPtrs<ThreadZxData> *ThreadMemory) {
            InitTls();

            for (;;) {
                ThreadZxData *expected = ThreadMemory->In.load();
                ThreadZxData *ok = nullptr;

                CheckStartCriteria;
                // PrefetchHighPrec(ok->zx);

                // ok->zx_low = (T)mpf_get_d(ok->zx);
                if constexpr (floatOrDouble) {
                    ok->zx_low = (T)mpf_get_d(ok->zx);
                } else {
                    int32_t zx_exponent;
                    double zx_mantissa;
                    zx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zx_mantissa, ok->zx));
                    ok->zx_low = T{zx_exponent, static_cast<SubType>(zx_mantissa)};
                }

                mpf_mul(ok->zx_sq, ok->zx, ok->zx);

                // Give result back.
                CheckFinishCriteria;
            }

            ShutdownTls();
        };

        auto ThreadSqZy = [&InitTls, &ShutdownTls](ThreadPtrs<ThreadZyData> *ThreadMemory) {
            InitTls();

            for (;;) {
                ThreadZyData *expected = ThreadMemory->In.load();
                ThreadZyData *ok = nullptr;

                CheckStartCriteria;
                // PrefetchHighPrec(ok->zy);

                // ok->zy_low = (T)mpf_get_d(ok->zy);
                if constexpr (floatOrDouble) {
                    ok->zy_low = (T)mpf_get_d(ok->zy);
                } else {
                    int32_t zy_exponent;
                    double zy_mantissa;
                    zy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zy_mantissa, ok->zy));
                    ok->zy_low = T{zy_exponent, static_cast<SubType>(zy_mantissa)};
                }

                mpf_mul(ok->zy_sq, ok->zy, ok->zy);

                // Give result back.
                CheckFinishCriteria;
            }

            ShutdownTls();
        };

        const int32_t IntermediateCompressionErrorExp =
            m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Intermediate);

        auto ThreadReused =
            [results,
             &bumpAllocator,
             &reusedAllocator,
             &InitTls,
             &ShutdownTls,
             IntermediateCompressionErrorExp](ThreadPtrs<ThreadReusedData> *ThreadMemory) {
                InitTls();

                SimpleIntermediateOrbitCompressor<IterType, T, PExtras> intermediateCompressor{
                    *results, IntermediateCompressionErrorExp};

                MaxIntermediateOrbitCompressor<IterType, T, PExtras> maxIntermediateCompressor{
                    *results, IntermediateCompressionErrorExp};

                // Initialize to 1 because the array starts with a zero at the front
                size_t index = 1;

                for (;;) {
                    ThreadReusedData *expected = ThreadMemory->In.load();
                    ThreadReusedData *ok = nullptr;

                    CheckStartCriteria;

                    // SaveForReuse1 shouldn't be done on this thread.
                    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {
                        results->AddUncompressedReusedEntry(ok->zx, ok->zy, index);
                    } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
                        intermediateCompressor.MaybeAddCompressedIteration(ok->zx, ok->zy, index);
                    } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                        maxIntermediateCompressor.MaybeAddCompressedIteration(ok->zx, ok->zy, index);
                    } else {
                        assert(false);
                    }

                    index++;

                    // Give result back.
                    CheckFinishCriteria;
                }

                if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
                    maxIntermediateCompressor.CompleteResults();
                }

                auto allocatorIndex = bumpAllocator->GetAllocatorIndex();
                reusedAllocator = bumpAllocator->GetAllocated(allocatorIndex);

                ShutdownTls();
            };

        auto *threadZxdata = (ThreadZxData *)_aligned_malloc(sizeof(ThreadZxData), 64);
        auto *threadZydata = (ThreadZyData *)_aligned_malloc(sizeof(ThreadZyData), 64);
        auto *threadReuseddata = (ThreadReusedData *)_aligned_malloc(sizeof(ThreadReusedData), 64);

        new (threadZxdata)(ThreadZxData){};
        new (threadZydata)(ThreadZyData){};
        new (threadReuseddata)(ThreadReusedData){};

        std::unique_ptr<std::thread> tZx(DEBUG_NEW std::thread(ThreadSqZx, ThreadZxMemory));
        std::unique_ptr<std::thread> tZy(DEBUG_NEW std::thread(ThreadSqZy, ThreadZyMemory));

        std::unique_ptr<std::thread> tReuse;

        // Mode 2/3 we use another thread
        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                      Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                      Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

            tReuse =
                std::unique_ptr<std::thread>(DEBUG_NEW std::thread(ThreadReused, ThreadReusedMemory));
        }

        ScopedAffinity scopedAffinity{*this,
                                      GetCurrentThread(),
                                      tZx->native_handle(),
                                      tZy->native_handle(),
                                      tReuse ? tReuse->native_handle() : nullptr};

        ThreadZxData *expectedZx = nullptr;
        ThreadZyData *expectedZy = nullptr;
        ThreadReusedData *expectedReused = nullptr;

        bool done1 = false;
        bool done2 = false;

        mpf_t zy_sq_orig;
        mpf_init(zy_sq_orig);

        mpf_set(zx, cx_mpf);
        mpf_set(zy, cy_mpf);

        bool periodicity_should_break = false;

        static const T HighOne = T{1.0};
        static const T HighTwo = T{2.0};
        static const T TwoFiftySix = T(256);

        bool zyStarted = false;

        T double_zx_last = T{0.0};
        T double_zy_last = T{0.0};

        RefOrbitCompressor<IterType, T, PExtras> compressor{
            *results, m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low)};

        for (IterTypeFull i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
            // Start Zx squaring thread
            mpf_set(threadZxdata->zx, zx);

            if (!zyStarted) {
                mpf_set(threadZydata->zy, zy);
            }

            ThreadZxMemory->In.store(threadZxdata, std::memory_order_release);

            if (!zyStarted) {
                // Start Zy squaring thread
                ThreadZyMemory->In.store(threadZydata, std::memory_order_relaxed);

                zyStarted = true;
            }

            T double_zx = double_zx_last;
            T double_zy = double_zy_last;

            SubType zn_size;

            if (i > 0) {
                if constexpr (PExtras == PerturbExtras::Disable) {
                    results->AddUncompressedIteration({double_zx, double_zy});
                } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
                    compressor.MaybeAddCompressedIteration({double_zx, double_zy, i});
                } else if constexpr (PExtras == PerturbExtras::Bad) {
                    results->AddUncompressedIteration({double_zx, double_zy, false});
                }

                if constexpr (PExtras == PerturbExtras::Bad) {
                    const T norm = HdrReduce((double_zx * double_zx + double_zy * double_zy) * glitch);
                    const auto zx_reduced = HdrReduce(HdrAbs((T)double_zx));
                    const auto zy_reduced = HdrReduce(HdrAbs((T)double_zy));

                    const bool underflow = (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                                            HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                                            HdrCompareToBothPositiveReducedLE(norm, small_float));
                    results->SetBad(underflow);
                }

                // Note: not T.
                const SubType tempZX = (SubType)double_zx + (SubType)cx_cast;
                const SubType tempZY = (SubType)double_zy + (SubType)cy_cast;
                zn_size = tempZX * tempZX + tempZY * tempZY;

                if constexpr (Periodicity) {
                    HdrReduce(dzdcX);
                    auto dzdcX1 = HdrAbs(dzdcX);

                    HdrReduce(dzdcY);
                    auto dzdcY1 = HdrAbs(dzdcY);

                    HdrReduce(double_zx);
                    auto zxCopy1 = HdrAbs(double_zx);

                    HdrReduce(double_zy);
                    auto zyCopy1 = HdrAbs(double_zy);

                    T n2 = HdrMaxPositiveReduced(zxCopy1, zyCopy1);

                    T r0 = HdrMaxPositiveReduced(dzdcX1, dzdcY1);
                    auto n3 = results->GetMaxRadius() * r0 * HighTwo; // TODO optimize HDRFloat *2.
                    HdrReduce(n3);

                    if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                        if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                            periodicity_should_break = true;
                        }
                    } else {
                        auto dzdcXOrig = dzdcX;
                        dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                        dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
                    }
                }

                if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                              Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                              Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

                    for (;;) {
                        expectedReused = threadReuseddata;

                        _mm_pause();
                        if (ThreadReusedMemory->Out.compare_exchange_weak(
                                expectedReused, nullptr, std::memory_order_release)) {
                            break;
                        }
                    }
                }
            } else {
                zn_size = 0;
            }

            if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1) {
                results->AddUncompressedReusedEntry(zx, zy, i + 1);
            } else if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                                 Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                                 Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

                mpf_set(threadReuseddata->zx, zx);
                mpf_set(threadReuseddata->zy, zy);

                ThreadReusedMemory->In.store(threadReuseddata, std::memory_order_release);
            }

            // zy = zx * 2 * zy + cy;

            // Store in temp
            mpf_mul(temp_mpf, zx, zy);
            mpf_mul_ui(temp_mpf, temp_mpf, 2);
            mpf_add(zy, temp_mpf, cy_mpf);

            done1 = false;
            done2 = false;
            bool quitting = false;

            for (;;) {
                expectedZy = threadZydata;

                _mm_pause();
                if (!done2 && ThreadZyMemory->Out.compare_exchange_weak(
                                  expectedZy, nullptr, std::memory_order_release)) {
                    done2 = true;

                    PrefetchHighPrec(threadZydata->zy_sq);

                    if constexpr (Periodicity) {
                        if (periodicity_should_break) {
                            results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                            quitting = true;
                        }
                    }

                    if (zn_size > 256) {
                        quitting = true;
                    }

                    if (!quitting) {
                        mpf_set(zy_sq_orig, threadZydata->zy_sq);
                        double_zy_last = threadZydata->zy_low;

                        // Restart right away!
                        mpf_set(threadZydata->zy, zy);

                        ThreadZyMemory->In.store(threadZydata, std::memory_order_release);
                    }
                }

                expectedZx = threadZxdata;

                _mm_pause();
                if (!done1 && ThreadZxMemory->Out.compare_exchange_weak(
                                  expectedZx, nullptr, std::memory_order_release)) {
                    done1 = true;

                    double_zx_last = threadZxdata->zx_low;
                    PrefetchHighPrec(threadZxdata->zx_sq);
                }

                if (done1 && done2) {
                    break;
                }
            }

            // zx = threadZxdata->zx_sq - zy_sq_orig + cx;
            mpf_sub(temp_mpf, threadZxdata->zx_sq, zy_sq_orig);
            mpf_add(zx, temp_mpf, cx_mpf);

            if (!quitting) {
                continue;
            }

            break;
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            results->SetBad(false);
        }

        bool res1 = false, res2 = false, res3 = false;
        while (!res1) {
            expectedZx = nullptr;
            res1 = ThreadZxMemory->In.compare_exchange_strong(
                expectedZx, (ThreadZxData *)0x1, std::memory_order_release);
        }

        while (!res2) {
            expectedZy = nullptr;
            res2 = ThreadZyMemory->In.compare_exchange_strong(
                expectedZy, (ThreadZyData *)0x1, std::memory_order_release);
        }

        while (!res3) {
            expectedReused = nullptr;
            res3 = ThreadReusedMemory->In.compare_exchange_strong(
                expectedReused, (ThreadReusedData *)0x1, std::memory_order_release);
        }

        tZx->join();
        tZy->join();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                      Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
                      Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

            tReuse->join();
        }

        _aligned_free(ThreadZxMemory);
        _aligned_free(ThreadZyMemory);
        _aligned_free(ThreadReusedMemory);

        threadZxdata->~ThreadZxData();
        threadZydata->~ThreadZyData();
        threadReuseddata->~ThreadReusedData();

        _aligned_free(threadZxdata);
        _aligned_free(threadZydata);
        _aligned_free(threadReuseddata);

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1) {
            std::ignore = bumpAllocator->GetAllocated(1); // Destruct the return value
        }

        results->CompleteResults<Reuse>(std::move(reusedAllocator));
        m_GuessReserveSize = results->GetCompressedOrUncompressedOrbitSize();
    } // End of scope for boundedAllocator and bumpAllocator

    ShutdownAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          PerturbExtras PExtras,
          RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::AddPerturbationReferencePointMT5(HighPrecision cx, HighPrecision cy)
{
    ::MessageBox(
        nullptr, L"AddPerturbationReferencePointMT5 disabled, using MT3", L"", MB_OK | MB_APPLMODAL);
    AddPerturbationReferencePointMT3<IterType, T, SubType, Periodicity, BenchmarkState, PExtras, Reuse>(
        cx, cy);
}

template <class P, class F>
static void
DispatchOne(F &&f)
{
    // Use unique_ptr so we always clean up even on exceptions
    auto comboResults = std::make_unique<HpSharkReferenceResults<P>>();
    f.template operator()<P>(*comboResults);
}

template <class F>
void
DispatchByPrecision(uint64_t prec, F &&f)
{
    // Round up prec to nearest power of 2:
    auto precRounded = prec;
    if ((precRounded & (precRounded - 1)) != 0) {
        uint32_t p = 1;
        while (p < precRounded) {
            p <<= 1;
        }
        precRounded = p;
    }

    if (precRounded < 256) {
        precRounded = 256;
    }

    if (precRounded > 524288) {
        throw std::invalid_argument("Unsupported precision");
    }

    switch (precRounded) {
        case 256:
            return DispatchOne<ProdSharkParams1>(std::forward<F>(f));
        case 512:
            return DispatchOne<ProdSharkParams2>(std::forward<F>(f));
        case 1024:
            return DispatchOne<ProdSharkParams3>(std::forward<F>(f));
        case 2048:
            return DispatchOne<ProdSharkParams4>(std::forward<F>(f));
        case 4096:
            return DispatchOne<ProdSharkParams5>(std::forward<F>(f));
        case 8192:
            return DispatchOne<ProdSharkParams6>(std::forward<F>(f));
        case 16384:
            return DispatchOne<ProdSharkParams7>(std::forward<F>(f));
        case 32768:
            return DispatchOne<ProdSharkParams8>(std::forward<F>(f));
        case 65536:
            return DispatchOne<ProdSharkParams9>(std::forward<F>(f));
        case 131072:
            return DispatchOne<ProdSharkParams10>(std::forward<F>(f));
        case 262144:
            return DispatchOne<ProdSharkParams11>(std::forward<F>(f));
        case 524288:
            return DispatchOne<ProdSharkParams12>(std::forward<F>(f));
        default:
            throw std::invalid_argument("Unsupported NumIters");
    }
}

template <typename IterType,
          class T,
          class SubType,
          bool Periodicity,
          RefOrbitCalc::BenchmarkMode BenchmarkState,
          PerturbExtras PExtras,
          RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::AddPerturbationReferencePointGPU(HighPrecision cx, HighPrecision cy)
{
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PushbackResults(std::move(newArray));
    auto *results = GetLast<IterType, T, PExtras>();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    mpf_t cx_mpf;
    mpf_init(cx_mpf);
    mpf_set(cx_mpf, cx.backend());

    mpf_t cy_mpf;
    mpf_init(cy_mpf);
    mpf_set(cy_mpf, cy.backend());

    const auto NumIters = m_Fractal.GetNumIterations<IterType>();
    const auto PrecInBits = HighPrecision::defaultPrecisionInBits();
    const auto PrecInLimbs = (PrecInBits + 31) / 32;

    auto lamb = [&]<class P>(HpSharkReferenceResults<P> &combo) {
        combo.RadiusY = results->GetMaxRadius();

        InvokeHpSharkReferenceKernelProd(launchParams, combo, cx_mpf, cy_mpf, NumIters);

        for (size_t i = 0; i < combo.EscapedIteration; ++i) {
            results->AddUncompressedIteration(combo.OutputIters[i]);
        }

        results->SetPeriodMaybeZero(combo.Period);

        // If OutputIters is a raw new[] owned by combo:
        delete[] combo.OutputIters;

        m_GuessReserveSize = results->GetCompressedOrUncompressedOrbitSize();
    };

    DispatchByPrecision(PrecInLimbs, lamb);

    results->CompleteResults<ReuseMode::DontSaveForReuse>(nullptr);

    mpf_clear(cx_mpf);
    mpf_clear(cy_mpf);
}

template <typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
bool
RefOrbitCalc::IsPerturbationResultUsefulHere(size_t i) const
{

    const auto *PerturbationResults = GetEltConst<IterType, T, PExtras>(i);

    if (PerturbationResults) {
        if constexpr (Authoritative == true) {
            return PerturbationResults->GetAuthoritativePrecisionInBits() != 0 &&
                   (PerturbationResults->GetMaxIterations() >
                        PerturbationResults->GetCountOrbitEntries() ||
                    PerturbationResults->GetMaxIterations() >= m_Fractal.GetNumIterations<IterType>());
        } else {
            const auto term1 = PerturbationResults->GetHiX() >= m_Fractal.GetMinX();
            const auto term2 = PerturbationResults->GetHiX() <= m_Fractal.GetMaxX();
            const auto term3 = PerturbationResults->GetHiY() >= m_Fractal.GetMinY();
            const auto term4 = PerturbationResults->GetHiY() <= m_Fractal.GetMaxY();
            return term1 && term2 && term3 && term4 &&
                   (PerturbationResults->GetMaxIterations() >
                        PerturbationResults->GetCountOrbitEntries() ||
                    PerturbationResults->GetMaxIterations() >= m_Fractal.GetNumIterations<IterType>());
        }
    }

    return false;
}

template <typename IterType,
          class T,
          class SubType,
          PerturbExtras PExtras,
          RefOrbitCalc::Extras Ex,
          class ConvertTType>
const PerturbationResults<IterType, ConvertTType, PExtras> *
RefOrbitCalc::GetUsefulPerturbationResults() const
{
    const auto *resultsExisting =
        GetUsefulPerturbationResultsConst<IterType, ConvertTType, false, PExtras>();

    if (resultsExisting != nullptr) {
        if constexpr (Ex == RefOrbitCalc::Extras::IncludeLAv2) {
            if (resultsExisting->GetLaReference() == nullptr) {
                resultsExisting = nullptr;
            }
        }

        m_LastUsedRefOrbit = resultsExisting;
    }

    return resultsExisting;
}

template <typename IterType,
          class T,
          class SubType,
          PerturbExtras PExtras,
          RefOrbitCalc::Extras Ex,
          class ConvertTType>
PerturbationResults<IterType, ConvertTType, PExtras> *
RefOrbitCalc::GetAndCreateUsefulPerturbationResults()
{
    constexpr bool ForceCompressDecompressForTesting = false;
    constexpr bool IsHdrDblflt = std::is_same<ConvertTType, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    constexpr bool IsDblflt = std::is_same<ConvertTType, CudaDblflt<MattDblflt>>::value;
    constexpr bool UsingDblflt = IsHdrDblflt || IsDblflt;
    constexpr auto PExtrasHackYay =
        (PExtras == PerturbExtras::SimpleCompression) ? PerturbExtras::Disable : PExtras;
    bool added = false;

    auto GenLAResults = [&](auto &results, AddPointOptions options_to_use) {
        if (results->GetLaReference() == nullptr) {
            if constexpr (Introspection::TestPExtras<PExtras>::value) {
                auto temp = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(
                    m_Fractal.GetLAParameters(),
                    options_to_use,
                    results->GenFilename(GrowableVectorTypes::LAInfoDeep, L"", true),
                    results->GenFilename(GrowableVectorTypes::LAStageInfo, L"", true));

                // TODO the presumption here is results size fits in the target IterType size
                temp->GenerateApproximationData(*results, results->GetMaxRadius(), UsingDblflt);

                added = true;

                results->SetLaReference(std::move(temp));
            }
        }
    };

    if (RequiresReuse()) {
        if (m_PerturbationGuessCalcX == HighPrecision{} && m_PerturbationGuessCalcY == HighPrecision{}) {
            m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision{2};
            m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision{2};
        }

        PerturbationResults<IterType, T, PExtrasHackYay> *results =
            GetUsefulPerturbationResultsMutable<IterType, T, false, PExtrasHackYay>();
        if (results == nullptr) {
            switch (GetPerturbationAlg()) {
                case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
                    added = AddPerturbationReferencePointSTReuse<IterType,
                                                                 T,
                                                                 SubType,
                                                                 true,
                                                                 BenchmarkMode::Disable,
                                                                 ReuseMode::SaveForReuse1>(
                        m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                    break;
                case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
                    added = AddPerturbationReferencePointMT3Reuse<IterType,
                                                                  T,
                                                                  SubType,
                                                                  true,
                                                                  BenchmarkMode::Disable,
                                                                  ReuseMode::SaveForReuse1>(
                        m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                    break;
                case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
                    added = AddPerturbationReferencePointMT3Reuse<IterType,
                                                                  T,
                                                                  SubType,
                                                                  true,
                                                                  BenchmarkMode::Disable,
                                                                  ReuseMode::SaveForReuse2>(
                        m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                    break;
                case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
                    added = AddPerturbationReferencePointMT3Reuse<IterType,
                                                                  T,
                                                                  SubType,
                                                                  true,
                                                                  BenchmarkMode::Disable,
                                                                  ReuseMode::SaveForReuse3>(
                        m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                    break;
                case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4:
                    added = AddPerturbationReferencePointMT3Reuse<IterType,
                                                                  T,
                                                                  SubType,
                                                                  true,
                                                                  BenchmarkMode::Disable,
                                                                  ReuseMode::SaveForReuse4>(
                        m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                    break;
                default:
                    ::MessageBox(nullptr, L"Some stupid bug #2343 :(", L"", MB_OK | MB_APPLMODAL);
                    assert(false);
                    break;
            }
        }
    }

    PerturbationResults<IterType, T, PExtras> *results =
        GetUsefulPerturbationResultsMutable<IterType, T, false, PExtras>();
    if (results == nullptr) {
        if (added) {
            ::MessageBox(nullptr, L"Why didn't this work! :(", L"", MB_OK | MB_APPLMODAL);
        }

        if constexpr (UsingDblflt) {
            PerturbationResults<IterType, ConvertTType, PExtras> *results_converted =
                GetUsefulPerturbationResultsMutable<IterType, ConvertTType, false, PExtras>();

            if (results_converted != nullptr) {
                return results_converted;
            }
        }

        AddPerturbationReferencePoint<IterType, T, SubType, PExtras, BenchmarkMode::Disable>();
        added = true;

        results = GetLast<IterType, T, PExtras>();

        // This is a hack for testing, but it's only a hack
        // if ForceCompressDecompressForTesting is true.
        if constexpr (ForceCompressDecompressForTesting && PExtras == PerturbExtras::Disable) {
            auto compressedResults =
                results->CompressMax(m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low),
                                     GetNextGenerationNumber(),
                                     true);

            auto decompressedResults = compressedResults->DecompressMax<PerturbExtras::Disable>(
                m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low),
                GetNextGenerationNumber());
            results = PushbackResults(std::move(decompressedResults));
        }
    }

    // This is a weird case.  Suppose you generate a Perturbation-only set of results
    // with things set to DontSave.  Then you generate a an LA-enabled orbit with
    // things set to EnableWithSave.  The LA-enabled orbit will not be saved.
    AddPointOptions options_to_use = m_RefOrbitOptions;

    if (results->GetRefOrbitOptions() == AddPointOptions::DontSave &&
        (m_RefOrbitOptions == AddPointOptions::EnableWithSave ||
         m_RefOrbitOptions == AddPointOptions::EnableWithoutSave)) {

        options_to_use = AddPointOptions::DontSave;
    }

    if constexpr (std::is_same<T, float>::value || // TODO: these are new.  Maybe OK to keep here.
                  std::is_same<T, double>::value || std::is_same<T, HDRFloat<float>>::value ||
                  std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Ex == Extras::IncludeLAv2) {
            static_assert(
                PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::SimpleCompression, "!");
            GenLAResults(results, options_to_use);
        } else {
            results->ClearLaReference();
        }
    }

    auto OptionalSave = [&](const auto *results) {
        if ((options_to_use == AddPointOptions::EnableWithSave ||
             options_to_use == AddPointOptions::EnableWithoutSave) &&
            added) {
            results->WriteMetadata();
        }
    };

    if constexpr (UsingDblflt) {
        auto *resultsExisting =
            GetUsefulPerturbationResultsMutable<IterType, ConvertTType, false, PExtras>();

        // The second part of this accounts for the case where the
        // malicious user deleted the LA reference.  We want to
        // recompute it if it's not present as a file.
        if ((resultsExisting == nullptr) ||
            (resultsExisting != nullptr && resultsExisting->GetLaReference() == nullptr &&
             Ex == Extras::IncludeLAv2)) {

            // This save is for the 64-bit calculations.
            OptionalSave(results);

            // Now generate the 2x32 results:
            auto results2(std::make_unique<PerturbationResults<IterType, ConvertTType, PExtras>>(
                options_to_use, GetNextGenerationNumber()));
            results2->CopyPerturbationResults<true>(*results);

            auto *results2Raw = results2.get();

            m_LastUsedRefOrbit = results2Raw;

            // Save those.
            OptionalSave(results2Raw);
            PushbackResults(std::move(results2));
            return results2Raw;
        } else {
            m_LastUsedRefOrbit = resultsExisting;
            OptionalSave(resultsExisting);
            return resultsExisting;
        }
    } else {
        m_LastUsedRefOrbit = results;
        OptionalSave(results);
        return results;
    }
}

template <typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetUsefulPerturbationResultsMutable()
{
    std::vector<PerturbationResults<IterType, T, PExtras> *> useful_results;

    if (m_C.size() > MaxStoredOrbits) {
        m_C.erase(m_C.begin());
    }

    for (size_t i = 0; i < m_C.size(); i++) {
        auto *cur_elt = GetElt<IterType, T, PExtras>(i);

        if (cur_elt) {
            if (IsPerturbationResultUsefulHere<IterType, T, Authoritative, PExtras>(i)) {
                useful_results.push_back(GetElt<IterType, T, PExtras>(i));
            }
        }
    }

    PerturbationResults<IterType, T, PExtras> *results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }

    return results;
}

template <typename IterType, class T, bool Authoritative, PerturbExtras PExtras>
const PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::GetUsefulPerturbationResultsConst() const
{
    std::vector<const PerturbationResults<IterType, T, PExtras> *> useful_results;

    for (size_t i = 0; i < m_C.size(); i++) {

        const auto *cur_elt = GetEltConst<IterType, T, PExtras>(i);
        if (cur_elt) {
            if (IsPerturbationResultUsefulHere<IterType, T, Authoritative, PExtras>(i)) {
                useful_results.push_back(GetEltConst<IterType, T, PExtras>(i));
            }
        }
    }

    const PerturbationResults<IterType, T, PExtras> *results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }

    return results;
}

template <typename IterType,
          class SrcT,
          PerturbExtras SrcEnableBad,
          class DestT,
          PerturbExtras DestEnableBad>
PerturbationResults<IterType, DestT, DestEnableBad> *
RefOrbitCalc::CopyUsefulPerturbationResults(PerturbationResults<IterType, SrcT, SrcEnableBad> &src_array)
requires((SrcEnableBad == PerturbExtras::Bad && DestEnableBad == PerturbExtras::Bad) ||
         (SrcEnableBad == PerturbExtras::Disable && DestEnableBad == PerturbExtras::Disable))
{

    if constexpr (std::is_same<SrcT, double>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        m_C.push_back(std::move(newarray));
        auto *dest = GetLast<IterType, DestT, DestEnableBad>();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    } else if constexpr (std::is_same<SrcT, float>::value) {
        return nullptr;
    } else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, HDRFloat<float>, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        m_C.push_back(std::move(newarray));
        auto *dest = GetLast<IterType, DestT, DestEnableBad>();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    } else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        m_C.push_back(std::move(newarray));
        auto *dest = GetLast<IterType, DestT, DestEnableBad>();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    } else {
        return nullptr;
    }
}

void
RefOrbitCalc::ClearPerturbationResults(PerturbationResultType type)
{
    auto IsMarkedToDelete = [&](const auto &val) -> bool {
        // Erase results as needed.
        // Note: erase full results in dbl-float case -- we'll reconvert
        // the whole thing if needed from double/Hdrfloat<double> etc.
        if (type == PerturbationResultType::All ||
            (type == PerturbationResultType::MediumRes && val->GetAuthoritativePrecisionInBits() == 0) ||
            (type == PerturbationResultType::HighRes && val->GetAuthoritativePrecisionInBits() != 0) ||
            (type == PerturbationResultType::LAOnly && val->Is2X32)) {
            return true;
        }

        return false;
    };

    auto ClearLAIfNeeded = [&](const auto &o) {
        if (type == PerturbationResultType::LAOnly) {
            o->ClearLaReference();
        }
    };

    auto ClearOne = [&](auto &arr) {
        // First, erase some results completely as needed
        arr.erase(std::remove_if(arr.begin(), arr.end(), IsMarkedToDelete), arr.end());

        // Now just erase LA results of what's left, if needed
        std::for_each(arr.begin(), arr.end(), ClearLAIfNeeded);
    };

    m_C.clear();

    m_PerturbationGuessCalcX = {};
    m_PerturbationGuessCalcY = {};

    ResetLastUsedOrbit();
}

void
RefOrbitCalc::ResetLastUsedOrbit()
{
    m_LastUsedRefOrbit = {};
}

void
RefOrbitCalc::ResetGuess(HighPrecision x, HighPrecision y)
{
    m_PerturbationGuessCalcX = x;
    m_PerturbationGuessCalcY = y;
}

void
RefOrbitCalc::SaveAllOrbits()
{
    auto lambda = [&](auto &elt) {
        const auto *results = elt.get();

        // Remove const / references / pointers
        using StrippedType = typename std::decay<decltype(results)>::type;
        using StrippedType2 = typename std::remove_pointer<StrippedType>::type;
        constexpr auto CurPExtras = StrippedType2::LocalPExtras;

        if constexpr (CurPExtras != PerturbExtras::MaxCompression) {
            auto resultsCopy =
                elt->CopyPerturbationResults(AddPointOptions::EnableWithSave, GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }
    };

    for (size_t i = 0; i < m_C.size(); i++) {
        std::visit(lambda, m_C[i]);
    }
}

void
RefOrbitCalc::LoadAllOrbits()
{
    // Matches the extension of a filename
    auto extmatch = [](std::string fn) -> bool {
        if (fn.substr(fn.find_last_of(".") + 1) == "met") {
            return true;
        }
        return false;
    };

    // Removes the extension from a filename
    auto stripext = [](std::string fn) -> std::string {
        size_t lastindex = fn.find_last_of(".");
        std::string n = fn.substr(0, lastindex);
        return n;
    };

    // Convert narrow string to wide string
    auto narrowtowide = [](std::string narrow) -> std::wstring {
        std::wstring wide;
        std::transform(
            narrow.begin(), narrow.end(), std::back_inserter(wide), [](char c) { return (wchar_t)c; });
        return wide;
    };

    // This is such a stupid implementation of this but whatever
    // TODO: make this not stupid
    // The idea is to load all the relevant files in the current directory.
    auto lambda = [&]<typename IterType, PerturbExtras PExtras>() {
        std::string path = ".";
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            auto file = entry.path().string();
            if (extmatch(file)) {
                auto next_gen = GetNextGenerationNumber();
                auto stripfn = stripext(file);
                auto widefn = narrowtowide(stripfn);
                auto results1 = std::make_unique<PerturbationResults<IterType, double, PExtras>>(
                    widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results1->ReadMetadata()) {
                    m_C.push_back(std::move(results1));
                    continue;
                }
                results1 = nullptr;

                auto results2 = std::make_unique<PerturbationResults<IterType, float, PExtras>>(
                    widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results2->ReadMetadata()) {
                    m_C.push_back(std::move(results2));
                    continue;
                }
                results2 = nullptr;

                auto results3 =
                    std::make_unique<PerturbationResults<IterType, CudaDblflt<MattDblflt>, PExtras>>(
                        widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results3->ReadMetadata()) {
                    m_C.push_back(std::move(results3));
                    continue;
                }
                results3 = nullptr;

                auto results4 =
                    std::make_unique<PerturbationResults<IterType, HDRFloat<double>, PExtras>>(
                        widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results4->ReadMetadata()) {
                    m_C.push_back(std::move(results4));
                    continue;
                }
                results4 = nullptr;

                auto results5 =
                    std::make_unique<PerturbationResults<IterType, HDRFloat<float>, PExtras>>(
                        widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results5->ReadMetadata()) {
                    m_C.push_back(std::move(results5));
                    continue;
                }
                results5 = nullptr;

                auto results6 = std::make_unique<
                    PerturbationResults<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PExtras>>(
                    widefn, AddPointOptions::OpenExistingWithSave, next_gen);
                if (results6->ReadMetadata()) {
                    m_C.push_back(std::move(results6));
                    continue;
                }
                results6 = nullptr;
            }
        }
    };

    lambda.template operator()<uint32_t, PerturbExtras::Disable>();
    lambda.template operator()<uint32_t, PerturbExtras::Bad>();
    lambda.template operator()<uint32_t, PerturbExtras::SimpleCompression>();
    lambda.template operator()<uint64_t, PerturbExtras::Disable>();
    lambda.template operator()<uint64_t, PerturbExtras::Bad>();
    lambda.template operator()<uint64_t, PerturbExtras::SimpleCompression>();
}

void
RefOrbitCalc::SetPerturbationAlg(PerturbationAlg alg)
{
    m_PerturbationAlg = alg;
}

RefOrbitCalc::PerturbationAlg
RefOrbitCalc::GetPerturbationAlg() const
{
    if (m_PerturbationAlg == PerturbationAlg::Auto) {
        HighPrecision zoomFactor = m_Fractal.GetZoomFactor();

        if (zoomFactor < HighPrecision{10}.power(150)) {
            return PerturbationAlg::STPeriodicity;
        } else if (zoomFactor < HighPrecision{10}.power(2000)) {
            return PerturbationAlg::MTPeriodicity3;
        } else {
            // Roll the dice and hope the intermediate precision orbit
            // works! :-)
            // Note MTPeriodicity3PerturbMTHighMTMed4 is broken.
            return PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3;
        }
    } else {
        return m_PerturbationAlg;
    }
}

std::string
RefOrbitCalc::GetPerturbationAlgStr() const
{
    // Return a string representation of the current perturbation algorithm:
    /*
        enum class PerturbationAlg {
        ST,
        MT,
        STPeriodicity,
        MTPeriodicity3,
        MTPeriodicity3PerturbMTHighSTMed,
        MTPeriodicity3PerturbMTHighMTMed1,
        MTPeriodicity3PerturbMTHighMTMed2,
        MTPeriodicity3PerturbMTHighMTMed3,
        MTPeriodicity3PerturbMTHighMTMed4,
        MTPeriodicity5
    };
    */

    switch (GetPerturbationAlg()) {
        case PerturbationAlg::ST:
            return "ST";
        case PerturbationAlg::MT:
            return "MT";
        case PerturbationAlg::STPeriodicity:
            return "STPeriodicity";
        case PerturbationAlg::MTPeriodicity3:
            return "MTPeriodicity2";
        case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
            return "MTPeriodicity2PerturbMTHighSTMed";
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
            return "MTPeriodicity2PerturbMTHighMTMed1";
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
            return "MTPeriodicity2PerturbMTHighMTMed2";
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
            return "MTPeriodicity2PerturbMTHighMTMed3";
        case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4:
            return "MTPeriodicity2PerturbMTHighMTMed4";
        case PerturbationAlg::MTPeriodicity5:
            return "MTPeriodicity5";
        case PerturbationAlg::GPU:
            return "GPU";
        default:
            return "Unknown";
    }
}

void
RefOrbitCalc::GetSomeDetails(RefOrbitDetails &details) const
{

    details = {};

    auto lambda = [&](auto &&arg) {
        if (arg == nullptr) {
            return;
        }

        uint64_t LAMilliseconds = 0;
        uint64_t LASize = 0;

        if (arg->GetLaReference() != nullptr) {
            LAMilliseconds = arg->GetLaReference()->GetBenchmarkLA().GetDeltaInMs();
            LASize = arg->GetLaReference()->GetLAs().GetSize();
        }

        int64_t deltaPrecisionCached = 0;
        int64_t extraPrecisionCached = 0;
        arg->GetIntermediatePrecision(deltaPrecisionCached, extraPrecisionCached);

        details = {arg->GetPeriodMaybeZero(),
                   arg->GetCompressedOrUncompressedOrbitSize(),
                   arg->GetCountOrbitEntries(),
                   arg->GetReuseSize(),
                   arg->GetCompressionErrorExp(),
                   arg->GetIntermediateCompressionErrorExp(),
                   deltaPrecisionCached,
                   extraPrecisionCached,
                   arg->GetBenchmarkOrbit(),
                   LAMilliseconds,
                   LASize,
                   GetPerturbationAlgStr(),
                   arg->GetHiZoomFactor()};
    };

    std::visit(lambda, m_LastUsedRefOrbit);

    static_assert(static_cast<int>(RenderAlgorithmEnum::MAX) == 61, "Fix me");
    static_assert(std::tuple_size<RenderAlgorithmsTupleT>() == 62, "Fix me");
}

void
RefOrbitCalc::SaveOrbit(CompressToDisk desiredCompression, std::wstring filename) const
{
    auto resultsSaver = [this, &filename](const auto &resultsToSave) {
        const auto compressedResults = resultsToSave->Compress(
            m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low), GetNextGenerationNumber());
        compressedResults->SaveOrbit(filename);
    };

    auto resultsSaverMax = [this, &filename](const auto &resultsToSave) {
        const auto compressedResults =
            resultsToSave->CompressMax(m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low),
                                       GetNextGenerationNumber(),
                                       false);
        compressedResults->SaveOrbit(filename);
    };

    bool lastOrbitNull = true;

    auto lambda = [this, &lastOrbitNull, resultsSaver, resultsSaverMax, desiredCompression, filename](
                      auto &&results) {
        if (results == nullptr) {
            return;
        }

        lastOrbitNull = false; // TODO we should use monostate properly

        using StrippedType = typename std::decay<decltype(results)>::type;
        using StrippedType2 = typename std::remove_pointer<StrippedType>::type;
        constexpr auto PExtras = StrippedType2::LocalPExtras;
        using IterType = decltype(Introspection::Extract(*results))::IterType_;

        if (desiredCompression == CompressToDisk::Disable) {
            results->SaveOrbit(filename);
        } else if (desiredCompression == CompressToDisk::SimpleCompression) {
            if constexpr (PExtras != PerturbExtras::Bad &&
                          !Introspection::IsDblFlt<decltype(*results)>()) {

                if constexpr (PExtras == PerturbExtras::Disable) {
                    resultsSaver(results);
                } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
                    // Already compressed
                    results->SaveOrbit(filename);
                } else {
                    throw FractalSharkSeriousException("Currently unsupported type 1");
                }
            } else {
                throw FractalSharkSeriousException("Currently unsupported type 3");
            }
        } else if (desiredCompression == CompressToDisk::MaxCompression) {
            if constexpr (PExtras != PerturbExtras::Bad &&
                          !Introspection::IsDblFlt<decltype(*results)>()) {

                resultsSaverMax(results);
            } else if constexpr (PExtras != PerturbExtras::Bad) {
                if constexpr (results->IsHDR) {
                    const auto *relatedResults =
                        GetUsefulPerturbationResultsConst<IterType, HDRFloat<double>, false, PExtras>();

                    resultsSaverMax(relatedResults);
                } else {
                    const auto *relatedResults =
                        GetUsefulPerturbationResultsConst<IterType, double, false, PExtras>();

                    resultsSaverMax(relatedResults);
                }
            } else {
                throw FractalSharkSeriousException("Currently unsupported type 1");
            }
        } else if (desiredCompression == CompressToDisk::MaxCompressionImagina) {
            if constexpr (!Introspection::IsDblFlt<decltype(*results)>() &&
                          PExtras != PerturbExtras::Bad) {

                SaveOrbitResults(*results, filename);
            } else if constexpr (PExtras != PerturbExtras::Bad) {
                if constexpr (results->IsHDR) {
                    const auto *relatedResults =
                        GetUsefulPerturbationResultsConst<IterType, HDRFloat<double>, false, PExtras>();
                    SaveOrbitResults(*relatedResults, filename);
                } else {
                    const auto *relatedResults =
                        GetUsefulPerturbationResultsConst<IterType, double, false, PExtras>();
                    SaveOrbitResults(*relatedResults, filename);
                }
            } else {
                throw FractalSharkSeriousException("Currently unsupported type 2");
            }
        } else {
            throw FractalSharkSeriousException("Unknown CompressToDisk");
        }
    };

    // Check if m_LastUsedRefOrbit holds anything
    std::visit(lambda, m_LastUsedRefOrbit);

    if (lastOrbitNull) {
        // Save location only, no orbit.
        SaveOrbitResults(filename);
    }
}

void
RefOrbitCalc::DiffOrbit(CompressToDisk compression,
                        std::wstring outFile,
                        std::wstring filename1,
                        std::wstring filename2) const
{

    const auto results1 = LoadOrbitConst(compression, filename1, nullptr);
    const auto results2 = LoadOrbitConst(compression, filename2, nullptr);

    auto lambda = [&](const auto &results1, const auto &results2) {
        using decl1 = decltype(*results1);
        using decl2 = decltype(*results2);
        if constexpr (std::is_same_v<decl1, decl2> && !Introspection::IsDblFlt<decl1>() &&
                      !Introspection::IsDblFlt<decl2>()) {
            results1->DiffOrbit(*results2, outFile);
        } else {
            throw FractalSharkSeriousException("Different types");
        }
    };

    try {
        std::visit(lambda, results1, results2);
    } catch (const std::exception &e) {
        const auto outstr = std::string("Error diffing orbits: ") + e.what();
        ::MessageBoxA(nullptr, "Error diffing orbits", "", MB_OK | MB_APPLMODAL);
    }
}

template <typename IterType, class T, PerturbExtras PExtras>
void
RefOrbitCalc::SaveOrbitResults(const PerturbationResults<IterType, T, PExtras> &results,
                               std::wstring imagFilename) const
{

    std::ofstream file(imagFilename, std::ios::binary);
    if (!file.is_open()) {
        throw FractalSharkSeriousException("Failed to open file for writing");
    }

    // Note: the flush() calls are just for debugging and the IO performance is
    // irrelevant on this path so it's fine to leave them in.

    Imagina::IMFileHeader fileHeader;
    using ResultsSubType = typename std::remove_reference<decltype(results)>::type::SubType;
    if (std::is_same_v<ResultsSubType, float>) {
        fileHeader.Magic = Imagina::SharksMagicNumber;
    } else if (std::is_same_v<ResultsSubType, double>) {
        fileHeader.Magic = Imagina::IMMagicNumber;
    } else {
        throw FractalSharkSeriousException("Invalid SubType");
    }

    fileHeader.Reserved = 0;
    fileHeader.LocationOffset = sizeof(fileHeader);
    fileHeader.ReferenceOffset = 0;
    file.write(reinterpret_cast<const char *>(&fileHeader), sizeof(fileHeader));
    file.flush();

    const uint64_t locationOffset = file.tellp();

    {
        Imagina::HRReal halfH{};
        const auto radius = results.GetMaxRadius();
        halfH = Imagina::HRReal{radius};
        file.write(reinterpret_cast<const char *>(&halfH), sizeof(Imagina::HRReal));
        file.flush();
    }

    {
        const uint64_t iterationLimit = results.GetMaxIterations() - 1;
        file.write((const char *)&iterationLimit, sizeof(iterationLimit));
        file.flush();
    }

    results.SaveOrbitLocation(file);

    const uint64_t referenceOffset = file.tellp();
    auto compressedResults =
        results.CompressMax(m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low),
                            GetNextGenerationNumber(),
                            false);

    static_assert(std::is_trivially_copyable_v<Imagina::ReferenceHeader>, "");
    // static_assert(std::is_trivially_copyable_v<ReferenceTrivialContent>, "");
    //  static_assert(std::is_trivially_copyable_v<LAReferenceTrivialContent>, "");

    {
        Imagina::ReferenceHeader referenceHeader;
        referenceHeader.ExtendedRange = results.IsHDR;
        file.write(reinterpret_cast<const char *>(&referenceHeader), sizeof(referenceHeader));
    }

    compressedResults->SaveOrbitBin(file);

    file.seekp(0);
    fileHeader.ReferenceOffset = referenceOffset;
    fileHeader.LocationOffset = locationOffset;

    file.write(reinterpret_cast<const char *>(&fileHeader), sizeof(fileHeader));
    file.flush();
}

void
RefOrbitCalc::SaveOrbitResults(std::wstring imagFilename) const
{
    // In this case, there's no orbit.  Instead, save only the location.
    // This path is only relevant with double/float precision and shallow depths.

    std::ofstream file(imagFilename, std::ios::binary);
    if (!file.is_open()) {
        throw FractalSharkSeriousException("Failed to open file for writing");
    }

    Imagina::IMFileHeader fileHeader{};
    fileHeader.Magic = Imagina::IMMagicNumber;
    fileHeader.Reserved = 0;
    fileHeader.LocationOffset = sizeof(fileHeader);
    fileHeader.ReferenceOffset = 0;
    file.write(reinterpret_cast<const char *>(&fileHeader), sizeof(fileHeader));

    HighPrecision orbitX, orbitY;

    {
        Imagina::HRReal halfH{};

        auto minX = m_Fractal.GetMinX();
        auto maxX = m_Fractal.GetMaxX();
        auto minY = m_Fractal.GetMinY();
        auto maxY = m_Fractal.GetMaxY();

        PointZoomBBConverter zoomConverter{minX, minY, maxX, maxY};

        const double radiusY{double{maxY - minY} / double{2.0}};
        orbitX = zoomConverter.GetPtX();
        orbitY = zoomConverter.GetPtY();

        halfH = Imagina::HRReal{radiusY};
        file.write(reinterpret_cast<const char *>(&halfH), sizeof(Imagina::HRReal));
        file.flush();
    }

    {
        const uint64_t iterationLimit = m_Fractal.GetNumIterations<IterTypeFull>();
        file.write((const char *)&iterationLimit, sizeof(iterationLimit));
        file.flush();
    }

    orbitX.SaveToStream(file);
    orbitY.SaveToStream(file);

    file.close();
}

const PerturbationResultsBase *
RefOrbitCalc::LoadOrbit(ImaginaSettings imaginaSettings,
                        CompressToDisk compression,
                        RenderAlgorithm renderAlg,
                        std::wstring imagFilename,
                        RecommendedSettings *recommendedSettings)
{

    auto lambda = [&](auto &ptr) -> const PerturbationResultsBase * {
        const auto *retval = ptr.get();
        m_LastUsedRefOrbit = retval;
        m_C.push_back(std::move(ptr));
        return retval;
    };

    if (imaginaSettings == ImaginaSettings::UseSaved) {
        auto decompressedResults = LoadOrbitConst(compression, imagFilename, recommendedSettings);

        // get the unique_ptr out of the AwesomeVariantUniquePtr

        auto decompressedResultsPtr = std::visit(lambda, decompressedResults);
        return decompressedResultsPtr;
    } else {
        auto helper = [&]<typename DestIterType, typename T>() -> const PerturbationResultsBase * {
            if (renderAlg.RequiresCompression) {
                constexpr auto PExtrasDest = PerturbExtras::SimpleCompression;
                auto decompressedResults = LoadOrbitConvert<DestIterType, T, PExtrasDest>(
                    compression, imagFilename, recommendedSettings);

                auto decompressedResultsPtr = std::visit(lambda, decompressedResults);
                return decompressedResultsPtr;
            } else {
                constexpr auto PExtrasDest = PerturbExtras::Disable;
                auto decompressedResults = LoadOrbitConvert<DestIterType, T, PExtrasDest>(
                    compression, imagFilename, recommendedSettings);
                auto decompressedResultsPtr = std::visit(lambda, decompressedResults);
                return decompressedResultsPtr;
            }
        };

        auto helperT = [&]<typename T>() -> const PerturbationResultsBase * {
            const auto RuntimeIterType = m_Fractal.GetIterType();
            if (RuntimeIterType == IterTypeEnum::Bits64) {
                return helper.template operator()<uint64_t, T>();
            } else {
                return helper.template operator()<uint32_t, T>();
            }
        };

        switch (renderAlg.Algorithm) {
            case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
            case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
            case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
            case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
                return helperT.template operator()<HDRFloat<double>>();

            case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
            case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
            case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
            case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
            case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
            case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
            case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
            case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
            case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
                return helperT.template operator()<double>();

            case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
            case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
            case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
            case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
                return helperT.template operator()<HDRFloat<float>>();

            case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
            case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
            case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
            case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
            case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
            case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
                return helperT.template operator()<float>();

            default:
                ::MessageBox(nullptr, L"Unknown render algorithm", L"Error", MB_OK | MB_ICONERROR);
                return helperT.template operator()<HDRFloat<double>>();
        }
    }
}

template <typename DestIterType, class DestT, PerturbExtras DestPExtras>
RefOrbitCalc::AwesomeVariantUniquePtr
RefOrbitCalc::LoadOrbitConvert(CompressToDisk compression,
                               std::wstring imagFilename,
                               RecommendedSettings *recommendedSettings)
{

    OrbitParameterPack params{};
    LoadOrbitConstInternal(params, compression, imagFilename, recommendedSettings);

    // Depending on the header, read the rest of the file and determine the type
    // Extended range implies the use of high precision types
    // For this example, let's assume HDRFloat<HRReal>

    auto TypeHelper = [&]<typename DestIterType, typename T, PerturbExtras PExtrasDest>()
        -> std::unique_ptr<PerturbationResults<DestIterType, T, PExtrasDest>> {
        auto results =
            std::make_unique<PerturbationResults<DestIterType, T, PerturbExtras::MaxCompression>>(
                AddPointOptions::EnableWithoutSave, GetNextGenerationNumber());

        const auto saturatedIterationLimit = params.GetSaturatedIterationCount<DestIterType>();

        results->LoadOrbitBin(std::move(params.orbitX),
                              std::move(params.orbitY),
                              saturatedIterationLimit,
                              params.halfH,
                              *params.file);

        auto decompressedResults = results->DecompressMax<PExtrasDest>(
            m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low), GetNextGenerationNumber());
        return decompressedResults;
    };

    if (params.m_OrbitType == OrbitParameterPack::IncludedOrbit::OrbitIncluded) {
        AwesomeVariantUniquePtr retval;
        // params.fileHeader.Magic == Imagina::IMMagicNumber
        retval = TypeHelper.template operator()<DestIterType, DestT, DestPExtras>();
        return retval;
    } else {
        // TODO use monostate?
        return {};
    }
}

RefOrbitCalc::AwesomeVariantUniquePtr
RefOrbitCalc::LoadOrbitConst(CompressToDisk compression,
                             std::wstring imagFilename,
                             RecommendedSettings *recommendedSettings) const
{

    OrbitParameterPack params{};

    LoadOrbitConstInternal(params, compression, imagFilename, recommendedSettings);

    // Depending on the header, read the rest of the file and determine the type
    // Extended range implies the use of HDR types

    auto TypeHelper = [&]<typename IterType, typename T, PerturbExtras PExtrasDest>()
        -> std::unique_ptr<PerturbationResults<IterType, T, PExtrasDest>> {
        auto results = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::MaxCompression>>(
            AddPointOptions::EnableWithoutSave, GetNextGenerationNumber());

        // The range of params.iterationLimit should already be checked
        // so we can safely cast it to IterType
        results->LoadOrbitBin(std::move(params.orbitX),
                              std::move(params.orbitY),
                              static_cast<IterType>(params.iterationLimit),
                              params.halfH,
                              *params.file);

        auto decompressedResults = results->DecompressMax<PExtrasDest>(
            m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low), GetNextGenerationNumber());
        return decompressedResults;
    };

    AwesomeVariantUniquePtr retval;
    auto CombinedHelper =
        [&]<RenderAlgorithmEnum Algorithm, typename IterType, typename T, PerturbExtras PExtrasDest>() {
            if (recommendedSettings != nullptr) {
                recommendedSettings->SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(Algorithm));
            }

            if (params.m_OrbitType == OrbitParameterPack::IncludedOrbit::OrbitIncluded) {
                retval = TypeHelper.template operator()<IterType, T, PerturbExtras::Disable>();
            } else {
                // TODO use monostate?
                retval = {};
            }
        };

    if (params.extendedRange) {
        if (params.fileHeader.Magic == Imagina::IMMagicNumber) {
            if (recommendedSettings->GetIterType() == IterTypeEnum::Bits64) {
                CombinedHelper.operator()<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2,
                                          uint64_t,
                                          HDRFloat<double>,
                                          PerturbExtras::Disable>();
            } else {
                CombinedHelper.operator()<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2,
                                          uint32_t,
                                          HDRFloat<double>,
                                          PerturbExtras::Disable>();
            }
        } else if (params.fileHeader.Magic == Imagina::SharksMagicNumber) {
            if (recommendedSettings->GetIterType() == IterTypeEnum::Bits64) {
                CombinedHelper.operator()<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2,
                                          uint64_t,
                                          HDRFloat<float>,
                                          PerturbExtras::Disable>();
            } else {
                CombinedHelper.operator()<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2,
                                          uint32_t,
                                          HDRFloat<float>,
                                          PerturbExtras::Disable>();
            }
        } else {
            throw FractalSharkSeriousException("Invalid file format");
        }
    } else {
        if (params.fileHeader.Magic == Imagina::IMMagicNumber) {
            if (recommendedSettings->GetIterType() == IterTypeEnum::Bits64) {
                CombinedHelper.operator()<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2,
                                          uint64_t,
                                          double,
                                          PerturbExtras::Disable>();
            } else {
                CombinedHelper.operator()<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2,
                                          uint32_t,
                                          double,
                                          PerturbExtras::Disable>();
            }
        } else if (params.fileHeader.Magic == Imagina::SharksMagicNumber) {
            if (recommendedSettings->GetIterType() == IterTypeEnum::Bits64) {
                CombinedHelper.operator()<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2,
                                          uint64_t,
                                          float,
                                          PerturbExtras::Disable>();
            } else {
                CombinedHelper.operator()<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2,
                                          uint32_t,
                                          float,
                                          PerturbExtras::Disable>();
            }
        } else {
            throw FractalSharkSeriousException("Invalid file format");
        }
    }

    return retval;
}

void
RefOrbitCalc::LoadOrbitConstInternal(OrbitParameterPack &params,
                                     CompressToDisk compression,
                                     std::wstring imagFilename,
                                     RecommendedSettings *recommendedSettings) const
{

    if (compression != CompressToDisk::MaxCompressionImagina) {
        throw FractalSharkSeriousException("Invalid compression type");
    }

    constexpr bool singleStepHelper = false;

    // Read the ReferenceHeader to determine the type
    auto file = std::make_unique<std::ifstream>(imagFilename, std::ios::binary);
    if (!file->is_open()) {
        throw FractalSharkSeriousException("Failed to open file for reading");
    }

    // Read the header to determine the type
    Imagina::IMFileHeader fileHeader;
    file->read(reinterpret_cast<char *>(&fileHeader), sizeof(fileHeader));

    if (fileHeader.Magic != Imagina::IMMagicNumber && fileHeader.Magic != Imagina::SharksMagicNumber) {

        throw FractalSharkSeriousException("Invalid file format");
    }

    // Seek to the location offset to read precision information
    file->seekg(fileHeader.LocationOffset);
    Imagina::HRReal halfH;
    file->read(reinterpret_cast<char *>(&halfH), sizeof(Imagina::HRReal));

    // Based on the precision of halfH, determine the type
    uint64_t precision = -std::min(0ll, halfH.getExp()) + AuthoritativeMinExtraPrecisionInBits;

    // Convert to zoom factor
    HighPrecision zoomFactor{};
    halfH.GetHighPrecision(zoomFactor);
    zoomFactor = HighPrecision{2} / zoomFactor;
    std::string zoomFactorStr = zoomFactor.str();

    uint64_t iterationLimit;
    file->read((char *)&iterationLimit, sizeof(iterationLimit));

    if constexpr (singleStepHelper) {
        uint64_t curPos1 = file->tellg();
    }

    HighPrecision orbitX{precision, *file};
    std::string orbitXStr = orbitX.str();

    HighPrecision orbitY{precision, *file};
    std::string orbitYStr = orbitY.str();

    if constexpr (singleStepHelper) {
        std::string orbitXStr2 = orbitX.str();
        std::string orbitYStr2 = orbitY.str();
        uint64_t curPos2 = file->tellg();
    }

    RecommendedSettings settingsOut{orbitX, orbitY, zoomFactor, {}, iterationLimit};

    bool extendedRange = false;
    OrbitParameterPack::IncludedOrbit includedOrbit = OrbitParameterPack::IncludedOrbit::NoOrbit;

    if (fileHeader.ReferenceOffset) {
        includedOrbit = OrbitParameterPack::IncludedOrbit::OrbitIncluded;
        file->seekg(fileHeader.ReferenceOffset);

        {
            Imagina::ReferenceHeader referenceHeader;
            file->read(reinterpret_cast<char *>(&referenceHeader), sizeof(referenceHeader));

            extendedRange = referenceHeader.ExtendedRange;
        }
    } else {
        // In this case, no orbit was saved, only the location.
        // Let's look at zoomFactor to figure out if we should use HDRFloat or not.
        // This is the same cut-off as used in Fractal::GetRenderAlgorithm
        if (zoomFactor < HighPrecision{1e34}) {
            extendedRange = false;
        } else {
            extendedRange = true;
        }
    }

    params = OrbitParameterPack(std::move(fileHeader),
                                std::move(orbitX),
                                std::move(orbitY),
                                iterationLimit,
                                halfH,
                                extendedRange,
                                includedOrbit,
                                std::move(file));

    if (recommendedSettings != nullptr) {
        *recommendedSettings = settingsOut;
    }
}

template <typename IterType, class T, PerturbExtras PExtras>
void
RefOrbitCalc::DrawPerturbationResultsHelper()
{
    // TODO can we just integrate all this with DrawFractal

    for (size_t i = 0; i < m_C.size(); i++) {
        const auto *curResult = GetElt<IterType, T, PExtras>(i);

        if (IsPerturbationResultUsefulHere<IterType, T, false, PExtras>(i)) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            const GLint scrnX =
                Convert<HighPrecision, GLint>(m_Fractal.XFromCalcToScreen(curResult->GetHiX()));
            const GLint scrnY =
                static_cast<GLint>(m_Fractal.GetRenderHeight()) -
                Convert<HighPrecision, GLint>(m_Fractal.YFromCalcToScreen(curResult->GetHiY()));

            // Coordinates are weird in OGL mode.
            glVertex2i(scrnX, scrnY);
        }
    }
}

void
RefOrbitCalc::DrawPerturbationResults()
{

    auto drawbatch = [&]<typename IterType>() {
        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, CudaDblflt<MattDblflt>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType,
                                      HDRFloat<CudaDblflt<MattDblflt>>,
                                      PerturbExtras::Disable>();

        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, CudaDblflt<MattDblflt>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>();

        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::SimpleCompression>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::SimpleCompression>();
        DrawPerturbationResultsHelper<IterType,
                                      CudaDblflt<MattDblflt>,
                                      PerturbExtras::SimpleCompression>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::SimpleCompression>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::SimpleCompression>();
        DrawPerturbationResultsHelper<IterType,
                                      HDRFloat<CudaDblflt<MattDblflt>>,
                                      PerturbExtras::SimpleCompression>();
    };

    drawbatch.template operator()<uint32_t>();
    drawbatch.template operator()<uint64_t>();
}

template <RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::InitAllocatorsIfNeeded(std::unique_ptr<MPIRBoundedAllocator> &boundedAllocator,
                                     std::unique_ptr<MPIRBumpAllocator> &bumpAllocator)
{

    if constexpr (Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse4) {

        boundedAllocator = std::make_unique<MPIRBoundedAllocator>();
        boundedAllocator->InitScopedAllocators();
        boundedAllocator->InitTls();
    } else {
        bumpAllocator = std::make_unique<MPIRBumpAllocator>();
        bumpAllocator->InitScopedAllocators();
        bumpAllocator->InitTls();
    }
}

void
RefOrbitCalc::PushbackResults(auto results)
{
    using StrippedType = typename std::decay<decltype(results)>::type;
    // decayedT is a unique_ptr<PerturbationResults<IterType, T, PExtras>>
    // Figure out what IterType, T, and PExtras are
    using StrippedType2 = typename StrippedType::element_type;
    constexpr auto PExtras = StrippedType2::LocalPExtras;
    using IterType = decltype(Introspection::Extract(*results))::IterType_;
    using T = decltype(Introspection::Extract(*results))::Float_;

    OptimizeMemory<IterType, T, PExtras>();
    m_C.push_back(std::move(results));
}

template <RefOrbitCalc::ReuseMode Reuse>
void
RefOrbitCalc::ShutdownAllocatorsIfNeeded(std::unique_ptr<MPIRBoundedAllocator> &boundedAllocator,
                                         std::unique_ptr<MPIRBumpAllocator> &bumpAllocator)
{

    if constexpr (Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3 &&
                  Reuse != RefOrbitCalc::ReuseMode::SaveForReuse4) {

        boundedAllocator->ShutdownTls();
    } else {
        bumpAllocator->ShutdownTls();
    }
}

RefOrbitCalc::ScopedAffinity::ScopedAffinity(
    RefOrbitCalc &refOrbitCalc, HANDLE thread1, HANDLE thread2, HANDLE thread3, HANDLE thread4)
    : m_RefOrbitCalc(refOrbitCalc), m_Thread1(thread1), m_Thread2(thread2), m_Thread3(thread3),
      m_Thread4(thread4)
{

    // SetCpuAffinityAsNeeded();
}

RefOrbitCalc::ScopedAffinity::~ScopedAffinity()
{
    // Note, ignore threads 2/3/4 -- they're temporaries in RefOrbitCalc.
    SetThreadAffinityMask(m_Thread1, 0xFFFFFFFF);

    // Reset to default priority:
    // SetThreadPriority(m_Thread1, THREAD_PRIORITY_NORMAL);
}

void
RefOrbitCalc::ScopedAffinity::SetCpuAffinityAsNeeded()
{

    if ((m_RefOrbitCalc.m_HyperthreadingEnabled == false && m_RefOrbitCalc.m_NumCpuCores < 4) ||
        (m_RefOrbitCalc.m_HyperthreadingEnabled == true && m_RefOrbitCalc.m_NumCpuCores < 8)) {
        return;
    }

    if (m_RefOrbitCalc.m_HyperthreadingEnabled) {
        if (m_Thread1) {
            SetThreadAffinityMask(m_Thread1, 0x1 << 3);
        }

        if (m_Thread2) {
            SetThreadAffinityMask(m_Thread2, 0x1 << 5);
        }

        if (m_Thread3) {
            SetThreadAffinityMask(m_Thread3, 0x1 << 7);
        }

        if (m_Thread4) {
            SetThreadAffinityMask(m_Thread4, 0x1 << 9);
        }
    } else {
        if (m_Thread1) {
            SetThreadAffinityMask(m_Thread1, 0x1 << 1);
        }

        if (m_Thread2) {
            SetThreadAffinityMask(m_Thread2, 0x1 << 2);
        }

        if (m_Thread3) {
            SetThreadAffinityMask(m_Thread3, 0x1 << 3);
        }

        if (m_Thread4) {
            SetThreadAffinityMask(m_Thread4, 0x1 << 4);
        }
    }

    // SetThreadPriority(m_Thread1, THREAD_PRIORITY_ABOVE_NORMAL);
    // SetThreadPriority(m_Thread2, THREAD_PRIORITY_ABOVE_NORMAL);
    // SetThreadPriority(m_Thread3, THREAD_PRIORITY_ABOVE_NORMAL);
    // SetThreadPriority(m_Thread4, THREAD_PRIORITY_ABOVE_NORMAL);
}

#include "RefOrbitCalcTemplates.h"
