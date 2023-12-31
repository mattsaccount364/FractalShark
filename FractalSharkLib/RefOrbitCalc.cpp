#include "stdafx.h"

#include "RefOrbitCalc.h"
#include "Fractal.h"
#include "PerturbationResults.h"

#include <vector>
#include <memory>
#include <math.h>
#include <fstream>

#include <psapi.h>

#include <string>
#include <iostream>
#include <filesystem>

// TODO we really should be using std::variant all over.  Oh well!

template<class Type>
struct ThreadPtrs {
    std::atomic<Type*> In;
    std::atomic<Type*> Out;
};

#define ENABLE_PREFETCH(ARG0, ARG1) _mm_prefetch(ARG0, ARG1)
//#define ENABLE_PREFETCH(ARG0, ARG1)

#define CheckStartCriteria \
    _mm_pause(); \
    if (expected == nullptr || \
        ThreadMemory->In.compare_exchange_weak( \
            expected, \
            nullptr, \
            std::memory_order_relaxed) == false) { \
        continue; \
    } \
 \
    if (expected == (void*)0x1) { \
        break; \
    } \
    ENABLE_PREFETCH((const char*)expected, _MM_HINT_T0); \
    ok = expected; \

#define CheckFinishCriteria \
    expected = nullptr; \
    for (;;) { \
        _mm_pause(); \
        bool result = ThreadMemory->Out.compare_exchange_weak( \
            expected, \
            ok, \
            std::memory_order_relaxed); \
        if (result) { \
            break; \
        } \
    } \

static inline void prefetch_range(void* addr, std::size_t len) {
    constexpr uintptr_t prefetch_stride = 64;
    void* vp = addr;
    void* end = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(len));
    while (vp < end) {
        ENABLE_PREFETCH((const char*)vp, _MM_HINT_T0);
        vp = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(vp) + static_cast<uintptr_t>(prefetch_stride));
    }
}

static inline void PrefetchHighPrec(const HighPrecision& target) {
    ENABLE_PREFETCH((const char*)&target.backend().data(), _MM_HINT_T0);
    size_t lastindex = abs(target.backend().data()->_mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target.backend().data()->_mp_d, total);
}

RefOrbitCalc::RefOrbitCalc(Fractal& Fractal)
    : m_PerturbationAlg{ PerturbationAlg::MTPeriodicity3 },
      m_Fractal(Fractal),
      m_PerturbationGuessCalcX(0),
      m_PerturbationGuessCalcY(0),
      m_RefOrbitOptions{ AddPointOptions::DontSave },
      m_GuessReserveSize(),
      m_GenerationNumber(),
      m_LaGenerationNumber() {
}

std::wstring RefOrbitCalc::UseFileBackedOrbits() const {
    if (m_RefOrbitOptions == AddPointOptions::SaveToFile) {
        return L".FullOrbit";
    }
    else {
        return L"";
    }
}

bool RefOrbitCalc::RequiresCompression() const {
    switch (m_Fractal.GetRenderAlgorithm()) {
        case RenderAlgorithm::Gpu1x32PerturbedRCLAv2:
        case RenderAlgorithm::Gpu1x32PerturbedRCLAv2PO:
        case RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO:

        case RenderAlgorithm::Gpu2x32PerturbedRCLAv2:
        case RenderAlgorithm::Gpu2x32PerturbedRCLAv2PO:
        case RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO:

        case RenderAlgorithm::Gpu1x64PerturbedRCLAv2:
        case RenderAlgorithm::Gpu1x64PerturbedRCLAv2PO:
        case RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO:

        case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2:
        case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO:
        case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO:

        case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2:
        case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO:
        case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO:

        case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2:
        case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO:
        case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO:
            return true;

        default:
            return false;
    }
}

bool RefOrbitCalc::RequiresBadCalc() const {
    switch (m_Fractal.GetRenderAlgorithm()) {
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        return true;
    default:
        return false;
    }
}

bool RefOrbitCalc::RequiresReuse() const {
    switch (m_PerturbationAlg) {
    case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
    case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed:
        return true;
    default:
        return false;
    }
}

bool RefOrbitCalc::IsThisPerturbationArrayUsed(void* check) const {
    switch (m_Fractal.GetRenderAlgorithm()) {
    case RenderAlgorithm::CpuHDR32:
    case RenderAlgorithm::CpuHigh:
    case RenderAlgorithm::Cpu64:
    case RenderAlgorithm::CpuHDR64:
    case RenderAlgorithm::Gpu1x64:
    case RenderAlgorithm::Gpu2x64:
    case RenderAlgorithm::Gpu4x64:
    case RenderAlgorithm::Gpu1x32:
    case RenderAlgorithm::Gpu2x32:
    case RenderAlgorithm::Gpu4x32:
        return false;
    case RenderAlgorithm::Cpu32PerturbedBLAHDR:
    case RenderAlgorithm::Cpu32PerturbedBLAV2HDR:
    case RenderAlgorithm::Cpu32PerturbedRCBLAV2HDR:
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2:
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2PO:
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO:
    case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2:
    case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO:
    case RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResultsHDRFloat ||
            check == &c32e.m_PerturbationResultsHDRFloat ||
            check == &c32c.m_PerturbationResultsHDRFloat ||
            check == &c64d.m_PerturbationResultsHDRFloat ||
            check == &c64e.m_PerturbationResultsHDRFloat ||
            check == &c64c.m_PerturbationResultsHDRFloat;
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::Cpu64PerturbedBLAV2HDR:
    case RenderAlgorithm::Cpu64PerturbedRCBLAV2HDR:
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2:
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2PO:
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO:
    case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2:
    case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO:
    case RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResultsHDRDouble ||
            check == &c32e.m_PerturbationResultsHDRDouble ||
            check == &c32c.m_PerturbationResultsHDRDouble ||
            check == &c64d.m_PerturbationResultsHDRDouble ||
            check == &c64e.m_PerturbationResultsHDRDouble ||
            check == &c64c.m_PerturbationResultsHDRDouble;
    case RenderAlgorithm::Gpu1x32PerturbedLAv2:
    case RenderAlgorithm::Gpu1x32PerturbedLAv2PO:
    case RenderAlgorithm::Gpu1x32PerturbedLAv2LAO:
    case RenderAlgorithm::Gpu1x32PerturbedRCLAv2:
    case RenderAlgorithm::Gpu1x32PerturbedRCLAv2PO:
    case RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResultsFloat ||
            check == &c32e.m_PerturbationResultsFloat ||
            check == &c32c.m_PerturbationResultsFloat ||
            check == &c64d.m_PerturbationResultsFloat ||
            check == &c64e.m_PerturbationResultsFloat ||
            check == &c64c.m_PerturbationResultsFloat;
    case RenderAlgorithm::Gpu2x32PerturbedLAv2:
    case RenderAlgorithm::Gpu2x32PerturbedLAv2PO:
    case RenderAlgorithm::Gpu2x32PerturbedLAv2LAO:
    case RenderAlgorithm::Gpu2x32PerturbedRCLAv2:
    case RenderAlgorithm::Gpu2x32PerturbedRCLAv2PO:
    case RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResults2xFloat ||
            check == &c32e.m_PerturbationResults2xFloat ||
            check == &c32c.m_PerturbationResults2xFloat ||
            check == &c64d.m_PerturbationResults2xFloat ||
            check == &c64e.m_PerturbationResults2xFloat ||
            check == &c64c.m_PerturbationResults2xFloat;
    case RenderAlgorithm::Cpu64PerturbedBLA:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
    case RenderAlgorithm::Gpu1x64PerturbedLAv2:
    case RenderAlgorithm::Gpu1x64PerturbedLAv2PO:
    case RenderAlgorithm::Gpu1x64PerturbedLAv2LAO:
    case RenderAlgorithm::Gpu1x64PerturbedRCLAv2:
    case RenderAlgorithm::Gpu1x64PerturbedRCLAv2PO:
    case RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResultsDouble ||
            check == &c32e.m_PerturbationResultsDouble ||
            check == &c32c.m_PerturbationResultsDouble ||
            check == &c64d.m_PerturbationResultsDouble ||
            check == &c64e.m_PerturbationResultsDouble ||
            check == &c64c.m_PerturbationResultsDouble;
    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2:
    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO:
    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO:
    case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2:
    case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO:
    case RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO:
        return
            check == &c32d.m_PerturbationResultsHDR2xFloat ||
            check == &c32e.m_PerturbationResultsHDR2xFloat ||
            check == &c32c.m_PerturbationResultsHDR2xFloat ||
            check == &c64d.m_PerturbationResultsHDR2xFloat ||
            check == &c64e.m_PerturbationResultsHDR2xFloat ||
            check == &c64c.m_PerturbationResultsHDR2xFloat;
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        // TODO
        //CalcGpuPerturbationFractalBLA<double, double>(MemoryOnly);
        assert(false);
        return false;
    default:
        assert(false);
        return false;
    }

    static_assert(static_cast<int>(RenderAlgorithm::MAX) == 62, "Fix me");
}

void RefOrbitCalc::OptimizeMemory() {
    PROCESS_MEMORY_COUNTERS_EX checkHappy;
    const size_t OMGAlotOfMemory = 128llu * 1024llu * 1024llu * 1024llu;
    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));

    auto lambda = [&]<typename T>(T & container) {
        if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResultsDouble)) {
                container.m_PerturbationResultsDouble.clear();
            }

            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResultsFloat)) {
                container.m_PerturbationResultsFloat.clear();
            }

            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResults2xFloat)) {
                container.m_PerturbationResults2xFloat.clear();
            }

            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResultsHDRDouble)) {
                container.m_PerturbationResultsHDRDouble.clear();
            }

            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResultsHDRFloat)) {
                container.m_PerturbationResultsHDRFloat.clear();
            }

            if (!IsThisPerturbationArrayUsed(&container.m_PerturbationResultsHDR2xFloat)) {
                container.m_PerturbationResultsHDR2xFloat.clear();
            }
        }
    };

    lambda(c32d);
    lambda(c32e);
    lambda(c32c);
    lambda(c64d);
    lambda(c64e);
    lambda(c64c);

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
        ::MessageBox(NULL, L"Watch the memory use... this is just a warning", L"", MB_OK | MB_APPLMODAL);
        assert(false);
    }
}

template<
    typename IterType,
    class T,
    PerturbExtras PExtras>
std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>> &
RefOrbitCalc::GetPerturbationResults() {
    auto lambda = [&]<typename U>(U& container) -> std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>& {
        if constexpr (std::is_same<T, double>::value) {
            return container.m_PerturbationResultsDouble;
        }
        else if constexpr (std::is_same<T, float>::value) {
            return container.m_PerturbationResultsFloat;
        }
        else if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
            return container.m_PerturbationResults2xFloat;
        }
        else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            return container.m_PerturbationResultsHDRDouble;
        }
        else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            return container.m_PerturbationResultsHDRFloat;
        }
        else if constexpr (std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
            return container.m_PerturbationResultsHDR2xFloat;
        }
    };

    if constexpr (std::is_same<IterType, uint32_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c32d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c32e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c32c);
        }
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c64d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c64e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c64c);
        }
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras> *
RefOrbitCalc::AddPerturbationResults(std::unique_ptr<PerturbationResults<IterType, T, PExtras>> results) {
    auto lambda = [&]<typename U>(U & container) -> PerturbationResults<IterType, T, PExtras>* {
        if constexpr (std::is_same<T, double>::value) {
            container.m_PerturbationResultsDouble.push_back(std::move(results));
            return container.m_PerturbationResultsDouble[container.m_PerturbationResultsDouble.size() - 1].get();
        }
        else if constexpr (std::is_same<T, float>::value) {
            container.m_PerturbationResultsFloat.push_back(std::move(results));
            return container.m_PerturbationResultsFloat[container.m_PerturbationResultsFloat.size() - 1].get();
        }
        else if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
            container.m_PerturbationResults2xFloat.push_back(std::move(results));
            return container.m_PerturbationResults2xFloat[container.m_PerturbationResults2xFloat.size() - 1].get();
        }
        else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            container.m_PerturbationResultsHDRDouble.push_back(std::move(results));
            return container.m_PerturbationResultsHDRDouble[container.m_PerturbationResultsHDRDouble.size() - 1].get();
        }
        else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            container.m_PerturbationResultsHDRFloat.push_back(std::move(results));
            return container.m_PerturbationResultsHDRFloat[container.m_PerturbationResultsHDRFloat.size() - 1].get();
        }
        else if constexpr (std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
            container.m_PerturbationResultsHDR2xFloat.push_back(std::move(results));
            return container.m_PerturbationResultsHDR2xFloat[container.m_PerturbationResultsHDR2xFloat.size() - 1].get();
        }
    };

    if constexpr (std::is_same<IterType, uint32_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c32d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c32e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c32c);
        }
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c64d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c64e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c64c);
        }
    }
}

template<
    typename IterType,
    class T,
    PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras>&
RefOrbitCalc::GetPerturbationResults(size_t index) {
    auto lambda = [&]<typename U>(U & container) -> PerturbationResults<IterType, T, PExtras>& {
        if constexpr (std::is_same<T, double>::value) {
            return *container.m_PerturbationResultsDouble[index];
        }
        else if constexpr (std::is_same<T, float>::value) {
            return *container.m_PerturbationResultsFloat[index];
        }
        else if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
            return *container.m_PerturbationResults2xFloat[index];
        }
        else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            return *container.m_PerturbationResultsHDRDouble[index];
        }
        else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            return *container.m_PerturbationResultsHDRFloat[index];
        }
        else if constexpr (std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
            return *container.m_PerturbationResultsHDR2xFloat[index];
        }
    };

    if constexpr (std::is_same<IterType, uint32_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c32d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c32e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c32c);
        }
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value) {
        if constexpr (PExtras == PerturbExtras::Disable) {
            return lambda(c64d);
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            return lambda(c64e);
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            return lambda(c64c);
        }
    }
}

template<
    typename IterType,
    class T,
    class SubType,
    PerturbExtras PExtras,
    RefOrbitCalc::BenchmarkMode BenchmarkState>
void RefOrbitCalc::AddPerturbationReferencePoint() {
    // TODO Hookup PExtras, move RequiresBadCalc out
    if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
        m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision(2);
        m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision(2);
    }

    //if (RequiresBadCalc()) {
    //    if (m_PerturbationAlg == PerturbationAlg::ST) {
    //        AddPerturbationReferencePointST<IterType, T, SubType, false, BenchmarkState, PerturbExtras::Bad, ReuseMode::DontSaveForReuse>(
    //            m_PerturbationGuessCalcX,
    //            m_PerturbationGuessCalcY);
    //    }
    //    else if (m_PerturbationAlg == PerturbationAlg::MT) {
    //        AddPerturbationReferencePointMT3<IterType, T, SubType, false, BenchmarkState, PerturbExtras::Bad, ReuseMode::DontSaveForReuse>(
    //            m_PerturbationGuessCalcX,
    //            m_PerturbationGuessCalcY);
    //    }
    //    else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
    //        AddPerturbationReferencePointST<IterType, T, SubType, true, BenchmarkState, PerturbExtras::Bad, ReuseMode::DontSaveForReuse>(
    //            m_PerturbationGuessCalcX,
    //            m_PerturbationGuessCalcY);
    //    }
    //    else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3 ||
    //             m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed ||
    //             m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed) {
    //        // Note: this path we don't bother saving/reusing.  Useless for low zoom depths.
    //        AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PerturbExtras::Bad, ReuseMode::DontSaveForReuse>(
    //            m_PerturbationGuessCalcX,
    //            m_PerturbationGuessCalcY);
    //    }
    //    else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity5) {
    //        AddPerturbationReferencePointMT5<IterType, T, SubType, true, BenchmarkState, PerturbExtras::Bad, ReuseMode::DontSaveForReuse>(
    //            m_PerturbationGuessCalcX,
    //            m_PerturbationGuessCalcY);
    //    }
    //}
    //else {
        if (m_PerturbationAlg == PerturbationAlg::ST) {
            AddPerturbationReferencePointST<IterType, T, SubType, false, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MT) {
            AddPerturbationReferencePointMT3<IterType, T, SubType, false, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
            AddPerturbationReferencePointST<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3) {
            AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed ||
                 m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed) {
            AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::SaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity5) {
            AddPerturbationReferencePointMT5<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
    //}
}

template<class T>
static void AddReused(T &results, const HighPrecision& zx, const HighPrecision& zy) {
    HighPrecision ReducedZx;
    HighPrecision ReducedZy;

    ReducedZx = zx;
    ReducedZy = zy;

    //assert(RequiresReuse());
    ReducedZx.precision(AuthoritativeReuseExtraPrecision);
    ReducedZy.precision(AuthoritativeReuseExtraPrecision);

    results.AddReusedEntry(std::move(ReducedZx), std::move(ReducedZy));
}

size_t RefOrbitCalc::GetNextGenerationNumber() {
    ++m_GenerationNumber;
    return m_GenerationNumber;
}

size_t RefOrbitCalc::GetNextLaGenerationNumber() {
    ++m_LaGenerationNumber;
    return m_LaGenerationNumber;
}

template<typename IterType, class T, class PerturbationResultsType, PerturbExtras PExtras, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::InitResults(
    PerturbationResultsType& results,
    const HighPrecision& initX,
    const HighPrecision& initY) {
    // We're going to add new results, so clear out the old ones.
    OptimizeMemory();

    results.InitResults<T, PExtras, Reuse>(
        initX,
        initY,
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        m_Fractal.GetNumIterations<IterType>(),
        m_GuessReserveSize,
        GetNextGenerationNumber());
}

template<
    typename IterType,
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState,
    PerturbExtras PExtras,
    RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointST(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<IterType, T, PExtras>();
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(UseFileBackedOrbits());
    PerturbationResultsArray.push_back(std::move(newArray));
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    IterTypeFull i;

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    T dzdcX = T{ 1 };
    T dzdcY = T{ 0 };

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    RefOrbitCompressor<IterType, T, PExtras> compressor{
        *results,
        m_Fractal.GetCompressionErrorExp() };

    zx = cx;
    zy = cy;
    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++)
    {
        zx2 = zx * 2;
        zy2 = zy * 2;

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        if constexpr (PExtras == PerturbExtras::Disable) {
            results->AddUncompressedIteration({ double_zx, double_zy });
        } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            compressor.MaybeAddCompressedIteration({ double_zx, double_zy, i + 1 });
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            results->AddUncompressedIteration({ double_zx, double_zy, false });
        }

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(*results, zx, zy);
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            const T sq_x = double_zx * double_zx;
            const T sq_y = double_zy * double_zy;
            const T norm = HdrReduce((sq_x + sq_y) * glitch);

            const auto zx_reduced = HdrReduce(HdrAbs((T)zx));
            const auto zy_reduced = HdrReduce(HdrAbs((T)zy));
            const bool underflow =
                (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(norm, small_float));
            results->SetBad(underflow);
        }

        if constexpr (Periodicity) {
            // x^2+2*I*x*y-y^2
            //dzdc = 2.0 * z * dzdc + real(1.0);
            //dzdc = 2.0 * (zx + zy * i) * (dzdcX + dzdcY * i) + HighPrecision(1.0);
            //dzdc = 2.0 * (zx * dzdcX + zx * dzdcY * i + zy * i * dzdcX + zy * i * dzdcY * i) + HighPrecision(1.0);
            //dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX + 2.0 * zy * i * dzdcY * i + HighPrecision(1.0);
            //dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0);
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
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
            }
        }

        zx = zx * zx - zy * zy + cx;
        zy = zx2 * zy + cy;

        T tempZX = double_zx + (T)cx;
        T tempZY = double_zy + (T)cy;
        T zn_size = tempZX * tempZX + tempZY * tempZY;
        if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
            break;
        }
    }

    if constexpr (PExtras == PerturbExtras::Bad) {
        results->SetBad(false);
    }

    results->TrimResults<PExtras, Reuse>();
    m_GuessReserveSize = results->GetCountOrbitEntries();

    if (m_RefOrbitOptions == AddPointOptions::SaveToFile) {
        results->Write(false);
    }
}

template<
    typename IterType,
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState>
bool RefOrbitCalc::AddPerturbationReferencePointSTReuse(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<IterType, T, PerturbExtras::Disable>();
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(UseFileBackedOrbits());
    PerturbationResultsArray.push_back(std::move(newArray));

    auto* existingResults = GetUsefulPerturbationResults<IterType, T, true, PerturbExtras::Disable>();
    if (existingResults == nullptr || existingResults->GetReuseSize() < 5) {
        // TODO Lame hack with < 5.
        PerturbationResultsArray.pop_back();
        return false;
    }

    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    auto NewPrec = m_Fractal.GetPrecision(
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        RequiresReuse());
    uint32_t precNum = AuthoritativeReuseExtraPrecision;

    // This all generally works and only starts to suffer precision problems after
    // about 10^AuthoritativeReuseExtraPrecision. The problem naturally is the original
    // reference orbit is calculated only to so many digits.
    if (NewPrec - existingResults->GetAuthoritativePrecision() >= AuthoritativeReuseExtraPrecision - AuthoritativeMinExtraPrecision) {
        //::MessageBox(NULL, L"Regenerating authoritative orbit is required", L"", MB_OK | MB_APPLMODAL);
        PerturbationResultsArray.pop_back();
        return false;
    }

    // TODO seems like we should be able to avoid all these annoying .precision calls below
    // via this mechanism.  Read the awful boost docs more...
    scoped_mpfr_precision prec(AuthoritativeReuseExtraPrecision);

    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);

    HighPrecision zx, zy;
    IterTypeFull i;

    HighPrecision HighOne = 1.0;
    HighPrecision HighTwo = 2.0;
    static const T TwoFiftySix = T(256.0);
    HighPrecision DeltaReal = cx - existingResults->GetHiX();
    HighPrecision DeltaImaginary = cy - existingResults->GetHiY();
    HighPrecision DeltaSub0X = DeltaReal;
    HighPrecision DeltaSub0Y = DeltaImaginary;
    HighPrecision DeltaSubNX = 0;
    HighPrecision DeltaSubNY = 0;

    // Must come after subtraction above
    cx.precision(precNum);
    cy.precision(precNum);
    HighOne.precision(precNum);
    HighTwo.precision(precNum);
    DeltaReal.precision(precNum);
    DeltaImaginary.precision(precNum);
    DeltaSub0X.precision(precNum);
    DeltaSub0Y.precision(precNum);
    DeltaSubNX.precision(precNum);
    DeltaSubNY.precision(precNum);

    IterTypeFull RefIteration = 0;
    IterTypeFull MaxRefIteration = existingResults->GetCountOrbitEntries() - 1;

    T dzdcX = T{ 1 };
    T dzdcY = T{ 0 };

    T zxCopy;
    T zyCopy;

    T zn_size;
    T normDeltaSubN;

    T tempZXLow;
    T tempZYLow;

    T tempDeltaSubNXLow;
    T tempDeltaSubNYLow;

    zx = cx;
    zy = cy;

    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
        if constexpr (Periodicity) {
            zxCopy = T(zx);
            zyCopy = T(zy);
        }

        const HighPrecision DeltaSubNXOrig = DeltaSubNX;
        const HighPrecision DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX =
            DeltaSubNXOrig * (existingResults->GetReuseXEntry(RefIteration) * HighTwo + DeltaSubNXOrig) -
            DeltaSubNYOrig * (existingResults->GetReuseYEntry(RefIteration) * HighTwo + DeltaSubNYOrig) +
            DeltaSub0X;
        tempDeltaSubNXLow = (T)DeltaSubNX;

        DeltaSubNY =
            DeltaSubNXOrig * (existingResults->GetReuseYEntry(RefIteration) * HighTwo + DeltaSubNYOrig) +
            DeltaSubNYOrig * (existingResults->GetReuseXEntry(RefIteration) * HighTwo + DeltaSubNXOrig) +
            DeltaSub0Y;
        tempDeltaSubNYLow = (T)DeltaSubNY;

        ++RefIteration;

        tempZXLow = (T)zx;
        tempZYLow = (T)zy;

        zx = existingResults->GetReuseXEntry(RefIteration) + DeltaSubNX;
        zy = existingResults->GetReuseYEntry(RefIteration) + DeltaSubNY;
        zn_size = tempZXLow * tempZXLow + tempZYLow * tempZYLow;
        HdrReduce(zn_size);
        normDeltaSubN = tempDeltaSubNXLow * tempDeltaSubNXLow + tempDeltaSubNYLow * tempDeltaSubNYLow;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
            break;
        }

        if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = zx;
            DeltaSubNY = zy;
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
                    // Break before adding the result.
                    results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                    break;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = T(2.0) * (zxCopy * dzdcX - zyCopy * dzdcY) + T(1.0);
                dzdcY = T(2.0) * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        T reducedZx = (T)zx;
        T reducedZy = (T)zy;

        results->AddUncompressedIteration({ reducedZx, reducedZy });
    }

    results->TrimResults<PerturbExtras::Disable, ReuseMode::DontSaveForReuse>();
    m_GuessReserveSize = results->GetCountOrbitEntries();

    return true;
}

template<
    typename IterType,
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState>
bool RefOrbitCalc::AddPerturbationReferencePointMT3Reuse(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<IterType, T, PerturbExtras::Disable>();
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(UseFileBackedOrbits());
    PerturbationResultsArray.push_back(std::move(newArray));

    auto* existingResults = GetUsefulPerturbationResults<IterType, T, true, PerturbExtras::Disable>();
    if (existingResults == nullptr || existingResults->GetReuseSize() < 5) {
        // TODO Lame hack with < 5.
        //::MessageBox(NULL, L"Authoritative not found", L"", MB_OK | MB_APPLMODAL);
        PerturbationResultsArray.pop_back();
        return false;
    }

    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    auto NewPrec = m_Fractal.GetPrecision(
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        RequiresReuse());
    uint32_t precNum = AuthoritativeReuseExtraPrecision;

    // This all generally works and only starts to suffer precision problems after
    // about 10^AuthoritativeReuseExtraPrecision. The problem naturally is the original
    // reference orbit is calculated only to so many digits.
    if (NewPrec - existingResults->GetAuthoritativePrecision() >= AuthoritativeReuseExtraPrecision - AuthoritativeMinExtraPrecision) {
        //::MessageBox(NULL, L"Regenerating authoritative orbit is required", L"", MB_OK | MB_APPLMODAL);
        PerturbationResultsArray.pop_back();
        return false;
    }

    // TODO seems like we should be able to avoid all these annoying .precision calls below
    // via this mechanism.  Read the awful boost docs more...
    scoped_mpfr_precision prec(AuthoritativeReuseExtraPrecision);
    scoped_mpfr_precision_options precOptions(boost::multiprecision::variable_precision_options::assume_uniform_precision);

    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);
    
    HighPrecision zx, zy;
    IterTypeFull i;

    HighPrecision HighOne = 1.0;
    HighPrecision HighTwo = 2.0;
    static const T TwoFiftySix = T(256.0);
    HighPrecision DeltaReal = cx - existingResults->GetHiX();
    HighPrecision DeltaImaginary = cy - existingResults->GetHiY();
    HighPrecision DeltaSub0X = DeltaReal;
    HighPrecision DeltaSub0Y = DeltaImaginary;
    HighPrecision DeltaSubNX = 0;
    HighPrecision DeltaSubNY = 0;

    // Must come after subtraction above
    cx.precision(precNum);
    cy.precision(precNum);
    HighOne.precision(precNum);
    HighTwo.precision(precNum);
    DeltaReal.precision(precNum);
    DeltaImaginary.precision(precNum);
    DeltaSub0X.precision(precNum);
    DeltaSub0Y.precision(precNum);
    DeltaSubNX.precision(precNum);
    DeltaSubNY.precision(precNum);

    IterTypeFull RefIteration = 0;
    IterTypeFull MaxRefIteration = existingResults->GetCountOrbitEntries() - 1;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    T zxCopy;
    T zyCopy;

    T zn_size;
    T normDeltaSubN;

    zx = cx;
    zy = cy;

    struct ThreadZxData {
        ThreadZxData(uint32_t precNum) {
            ReferenceIteration = 0;
            DeltaSubNX.precision(precNum);
        }

        IterTypeFull ReferenceIteration;
        const HighPrecision *DeltaSubNXOrig;
        const HighPrecision *DeltaSubNYOrig;
        const HighPrecision *DeltaSub0X;
        HighPrecision DeltaSubNX;
    };

    struct ThreadZyData {
        ThreadZyData(uint32_t precNum) {
            ReferenceIteration = 0;
            DeltaSubNY.precision(precNum);
        }

        IterTypeFull ReferenceIteration;
        const HighPrecision *DeltaSubNXOrig;
        const HighPrecision *DeltaSubNYOrig;
        const HighPrecision *DeltaSub0Y;
        HighPrecision DeltaSubNY;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto ThreadSqZx = [&](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        scoped_mpfr_precision_options precOptions(boost::multiprecision::variable_precision_options::assume_uniform_precision);
        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0X);
            //PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            //PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            ok->DeltaSubNX =
                (*ok->DeltaSubNXOrig) * (existingResults->GetReuseXEntry(ok->ReferenceIteration) * HighTwo + (*ok->DeltaSubNXOrig)) -
                (*ok->DeltaSubNYOrig) * (existingResults->GetReuseYEntry(ok->ReferenceIteration) * HighTwo + (*ok->DeltaSubNYOrig)) +
                (*ok->DeltaSub0X);

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto ThreadSqZy = [&](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        scoped_mpfr_precision_options precOptions(boost::multiprecision::variable_precision_options::assume_uniform_precision);
        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0Y);
            //PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            //PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            ok->DeltaSubNY =
                (*ok->DeltaSubNXOrig) * (existingResults->GetReuseYEntry(ok->ReferenceIteration) * HighTwo + (*ok->DeltaSubNYOrig)) +
                (*ok->DeltaSubNYOrig) * (existingResults->GetReuseXEntry(ok->ReferenceIteration) * HighTwo + (*ok->DeltaSubNXOrig)) +
                (*ok->DeltaSub0Y);

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);

    new (threadZxdata) (ThreadZxData){precNum};
    new (threadZydata) (ThreadZyData){precNum};

    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);
    //SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    //SetThreadPriority(tZx->native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
    //SetThreadPriority(tZy->native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;

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

        ThreadZxMemory->In.store(
            threadZxdata,
            std::memory_order_release);

        threadZydata->ReferenceIteration = RefIteration;
        threadZydata->DeltaSubNXOrig = &DeltaSubNX;
        threadZydata->DeltaSubNYOrig = &DeltaSubNY;
        threadZydata->DeltaSub0Y = &DeltaSub0Y;

        ThreadZyMemory->In.store(
            threadZydata,
            std::memory_order_relaxed);

        ++RefIteration;

        // Lame conditional.
        // Could erase first elt but that's O(n) in size of vector
        if (RanOnce) {
            results->AddUncompressedIteration({ tempZXLow, tempZYLow });
        }

        if constexpr (Periodicity) {
            zxCopy = T(zx);
            zyCopy = T(zy);

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
                    // Break before adding the result.
                    results->SetPeriodMaybeZero((IterType)results->GetCountOrbitEntries());
                    break;
                }
            }
            else {
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
            if (!done2 &&
                ThreadZyMemory->Out.compare_exchange_weak(expectedZy,
                    nullptr,
                    std::memory_order_release)) {
                done2 = true;
                tempDeltaSubNYLow = T(threadZydata->DeltaSubNY);
                zy = existingResults->GetReuseYEntry(RefIteration) + threadZydata->DeltaSubNY;
                tempZYLow = T(zy);
            }

            expectedZx = threadZxdata;

            _mm_pause();
            if (!done1 &&
                ThreadZxMemory->Out.compare_exchange_weak(expectedZx,
                    nullptr,
                    std::memory_order_release)) {
                done1 = true;
                tempDeltaSubNXLow = T(threadZxdata->DeltaSubNX);
                zx = existingResults->GetReuseXEntry(RefIteration) + threadZxdata->DeltaSubNX;
                tempZXLow = T(zx);
            }

            if (done1 && done2) {
                break;
            }
        }

        DeltaSubNY = threadZydata->DeltaSubNY;
        DeltaSubNX = threadZxdata->DeltaSubNX;

        zn_size = tempZXLow * tempZXLow + tempZYLow * tempZYLow;
        HdrReduce(zn_size);
        normDeltaSubN = tempDeltaSubNXLow * tempDeltaSubNXLow + tempDeltaSubNYLow * tempDeltaSubNYLow;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedGT(zn_size, Low256)) {
            break;
        }

        if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = zx;
            DeltaSubNY = zy;
            RefIteration = 0;
        }

        RanOnce = true;
    }

    bool res1 = false, res2 = false;
    while (!res1) {
        expectedZx = nullptr;
        res1 = ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);
    }

    while (!res2) {
        expectedZy = nullptr;
        res2 = ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);
    }

    //results->x.erase(results->x.begin());
    //results->y.erase(results->y.begin());

    tZx->join();
    tZy->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);

    results->TrimResults<PerturbExtras::Disable, ReuseMode::DontSaveForReuse>();
    m_GuessReserveSize = results->GetCountOrbitEntries();

    return true;
}

template<
    typename IterType, 
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState,
    PerturbExtras PExtras,
    RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT3(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<IterType, T, PExtras>();
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(UseFileBackedOrbits());
    PerturbationResultsArray.push_back(std::move(newArray));
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision zx, zy;

    T dzdcX = T{ 1.0 };
    T dzdcY = T{ 0.0 };

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        HighPrecision zx;
        HighPrecision zx_sq;
    };

    struct ThreadZyData {
        HighPrecision zy;
        HighPrecision zy_sq;
    };

    struct ThreadReusedData {
        HighPrecision zx;
        HighPrecision zy;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto* ThreadReusedMemory = (ThreadPtrs<ThreadReusedData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadReusedData>), 64);
    memset(ThreadReusedMemory, 0, sizeof(*ThreadReusedMemory));

    auto ThreadSqZx = [](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zx);

            ok->zx_sq = ok->zx * ok->zx;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto ThreadSqZy = [](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zy);

            ok->zy_sq = ok->zy * ok->zy;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto ThreadReused = [&](ThreadPtrs<ThreadReusedData>* ThreadMemory) {
        for (;;) {
            ThreadReusedData* expected = ThreadMemory->In.load();
            ThreadReusedData* ok = nullptr;

            CheckStartCriteria;

            AddReused(*results, ok->zx, ok->zy);

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);
    auto* threadReuseddata = (ThreadReusedData*)_aligned_malloc(sizeof(ThreadReusedData), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};
    new (threadReuseddata) (ThreadReusedData){};

    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));

    std::unique_ptr<std::thread> tReuse;
    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
        //tReuse = std::make_unique<std::thread>(ThreadSqZy, ThreadReusedMemory); // TODO
    }

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);
    //SetThreadAffinityMask(tReuse->native_handle(), 0x1 << 9); // TODO

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;

    bool done1 = false;
    bool done2 = false;

    HighPrecision zy_sq_orig;

    zx = cx;
    zy = cy;

    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    bool zyStarted = false;

    RefOrbitCompressor<IterType, T, PExtras> compressor{
        *results,
        m_Fractal.GetCompressionErrorExp() };

    for (IterTypeFull i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++)
    {
        // Start Zx squaring thread
        threadZxdata->zx = zx;

        if (!zyStarted) {
            threadZydata->zy = zy;
        }

        ThreadZxMemory->In.store(
            threadZxdata,
            std::memory_order_release);

        if (!zyStarted) {
            // Start Zy squaring thread
            ThreadZyMemory->In.store(
                threadZydata,
                std::memory_order_relaxed);

            zyStarted = true;
        }

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        if constexpr (PExtras == PerturbExtras::Disable) {
            results->AddUncompressedIteration({ double_zx, double_zy });
        }
        else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            compressor.MaybeAddCompressedIteration({ double_zx, double_zy, i + 1 });
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            results->AddUncompressedIteration({ double_zx, double_zy, false });
        }

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(*results, zx, zy);  // TODO
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            const T norm = HdrReduce((double_zx * double_zx + double_zy * double_zy) * glitch);
            const auto zx_reduced = HdrReduce(HdrAbs((T)double_zx));
            const auto zy_reduced = HdrReduce(HdrAbs((T)double_zy));

            const bool underflow =
                (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(norm, small_float));
            results->SetBad(underflow);
        }

        // Note: not T.
        const SubType tempZX = (SubType)double_zx + (SubType)cx;
        const SubType tempZY = (SubType)double_zy + (SubType)cy;
        const SubType zn_size = tempZX * tempZX + tempZY * tempZY;

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
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
            }
        }

        zy = zx * 2 * zy + cy;

        done1 = false;
        done2 = false;
        bool quitting = false;

        for (;;) {
            expectedZy = threadZydata;

            _mm_pause();
            if (!done2 &&
                ThreadZyMemory->Out.compare_exchange_weak(expectedZy,
                    nullptr,
                    std::memory_order_release)) {
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
                    zy_sq_orig = threadZydata->zy_sq;

                    // Restart right away!
                    threadZydata->zy = zy;

                    ThreadZyMemory->In.store(
                        threadZydata,
                        std::memory_order_release);
                }
            }

            expectedZx = threadZxdata;

            _mm_pause();
            if (!done1 &&
                ThreadZxMemory->Out.compare_exchange_weak(expectedZx,
                    nullptr,
                    std::memory_order_release)) {
                done1 = true;

                PrefetchHighPrec(threadZxdata->zx_sq);
            }

            if (done1 && done2) {
                break;
            }
        }

        zx = threadZxdata->zx_sq - zy_sq_orig + cx;

        if (!quitting) {
            continue;
        }

        break;
    }

    if constexpr (PExtras == PerturbExtras::Bad) {
        results->SetBad(false);
    }

    bool res1 = false, res2 = false;
    while (!res1) {
        expectedZx = nullptr;
        res1 = ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);
    }

    while (!res2) {
        expectedZy = nullptr;
        res2 = ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);
    }

    tZx->join();
    tZy->join();
    //tReuse->join();  // TODO

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);
    _aligned_free(ThreadReusedMemory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();
    threadReuseddata->~ThreadReusedData();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);
    _aligned_free(threadReuseddata);

    results->TrimResults<PExtras, Reuse>();
    m_GuessReserveSize = results->GetCountOrbitEntries();

    if (m_RefOrbitOptions == AddPointOptions::SaveToFile) {
        results->Write(false);
    }
}

template<
    typename IterType, 
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState,
    PerturbExtras PExtras,
    RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT5(HighPrecision cx, HighPrecision cy) {
    ::MessageBox(NULL, L"AddPerturbationReferencePointMT5 disabled", L"", MB_OK | MB_APPLMODAL);
    AddPerturbationReferencePointMT3<IterType, T, SubType, Periodicity, BenchmarkState, PExtras, Reuse>(cx, cy);
}


bool RefOrbitCalc::RequiresReferencePoints() const {
    switch (m_Fractal.GetRenderAlgorithm()) {
        case RenderAlgorithm::CpuHigh:
        case RenderAlgorithm::CpuHDR32:
        case RenderAlgorithm::CpuHDR64:
        case RenderAlgorithm::Cpu64:
        case RenderAlgorithm::Gpu1x64:
        case RenderAlgorithm::Gpu2x64:
        case RenderAlgorithm::Gpu4x64:
        case RenderAlgorithm::Gpu1x32:
        case RenderAlgorithm::GpuHDRx32:
        case RenderAlgorithm::Gpu2x32:
        case RenderAlgorithm::Gpu4x32:
            return false;
        default:
            return true;
    }
}

template<
    typename IterType,
    class T,
    bool Authoritative,
    PerturbExtras PExtras>
bool RefOrbitCalc::IsPerturbationResultUsefulHere(size_t i) {

    auto lambda = [&]<typename T, bool Authoritative, PerturbExtras PExtras>(
        const PerturbationResults<IterType, T, PExtras> &PerturbationResults
        ) -> bool {

        //if constexpr (PExtras == PerturbExtras::EnableCompression) {
        //    if constexpr (!PerturbationResults.IsCompressed()) {
        //        return false;
        //    }
        //}
        //else {
        //    if constexpr (PerturbationResults.IsCompressed()) {
        //        return false;
        //    }
        //}

        if constexpr (Authoritative == true) {
            return
                PerturbationResults.GetAuthoritativePrecision() != 0 &&
                (PerturbationResults.GetMaxIterations() > PerturbationResults.GetCountOrbitEntries() ||
                    PerturbationResults.GetMaxIterations() >= m_Fractal.GetNumIterations<IterType>());
        }

        return
            PerturbationResults.GetHiX() >= m_Fractal.GetMinX() &&
            PerturbationResults.GetHiX() <= m_Fractal.GetMaxX() &&
            PerturbationResults.GetHiY() >= m_Fractal.GetMinY() &&
            PerturbationResults.GetHiY() <= m_Fractal.GetMaxY() &&
            (PerturbationResults.GetMaxIterations() > PerturbationResults.GetCountOrbitEntries() ||
                PerturbationResults.GetMaxIterations() >= m_Fractal.GetNumIterations<IterType>());
    };

    const auto& results = GetPerturbationResults<IterType, T, PExtras>(i);
    return lambda.template operator()<T, Authoritative, PExtras>(results);
}

template<
    typename IterType,
    class T,
    class SubType,
    PerturbExtras PExtras,
    RefOrbitCalc::Extras Ex,
    class ConvertTType>
PerturbationResults<IterType, ConvertTType, PExtras>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults() {

    bool added = false;
    constexpr auto PExtrasHackYay =
        (PExtras == PerturbExtras::EnableCompression) ? PerturbExtras::Disable : PExtras;

    if (RequiresReuse()) {
        if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
            m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / 2;
            m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / 2;
        }

        PerturbationResults<IterType, T, PExtrasHackYay>* results =
            GetUsefulPerturbationResults<IterType, T, false, PExtrasHackYay>();
        if (results == nullptr) {
            switch (GetPerturbationAlg()) {
            case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
                added = AddPerturbationReferencePointSTReuse<IterType, T, SubType, true, BenchmarkMode::Disable>
                    (m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed:
                added = AddPerturbationReferencePointMT3Reuse<IterType, T, SubType, true, BenchmarkMode::Disable>
                    (m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            default:
                ::MessageBox(NULL, L"Some stupid bug #2343 :(", L"", MB_OK | MB_APPLMODAL);
                assert(false);
                break;
            }
        }
    }

    PerturbationResults<IterType, T, PExtras>* results =
        GetUsefulPerturbationResults<IterType, T, false, PExtras>();
    if (results == nullptr) {
        if (added) {
            ::MessageBox(NULL, L"Why didn't this work! :(", L"", MB_OK | MB_APPLMODAL);
        }

        //std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtrasHackYay>>>& cur_array =
        //    GetPerturbationResults<IterType, T, PExtrasHackYay>();
        //AddPerturbationReferencePoint<IterType, T, SubType, PerturbExtras::Disable, BenchmarkMode::Disable>();
        //added = true;

        //// TODO: this is a hack.  We need to fix this.
        //if constexpr (PExtras == PerturbExtras::EnableCompression) {
        //    auto resultsIntermediate = cur_array[cur_array.size() - 1].get();
        //    auto compressedResults = resultsIntermediate->Compress(
        //        m_Fractal.GetCompressionErrorExp(),
        //        GetNextGenerationNumber(),
        //        GetNextLaGenerationNumber());
        //    results = AddPerturbationResults(std::move(compressedResults));

        //    // Here we just delete the full orbit.
        //    cur_array.pop_back();
        //}
        //else {
        //    results = cur_array[cur_array.size() - 1].get();
        //}

        std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>& cur_array =
            GetPerturbationResults<IterType, T, PExtras>();
        AddPerturbationReferencePoint<IterType, T, SubType, PExtras, BenchmarkMode::Disable>();
        added = true;

        results = cur_array[cur_array.size() - 1].get();
    }

    constexpr bool IsHdrDblflt = std::is_same<ConvertTType, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    constexpr bool IsDblflt = std::is_same<ConvertTType, CudaDblflt<MattDblflt>>::value;
    constexpr bool UseSmallExponents = IsHdrDblflt || IsDblflt;

    if constexpr (
        std::is_same<T, float>::value || // TODO: these are new.  Maybe OK to keep here.
        std::is_same<T, double>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Ex == Extras::IncludeLAv2) {
            static_assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::EnableCompression, "!");
            if (results->GetLaReference() == nullptr) {
                auto temp = std::make_unique<LAReference<IterType, T, SubType, PExtras>>();

                // TODO the presumption here is results size fits in the target IterType size
                temp->GenerateApproximationData(
                    *results,
                    results->GetMaxRadius(),
                    (IterType)results->GetCountOrbitEntries() - 1,
                    UseSmallExponents);

                results->SetLaReference(std::move(temp), GetNextLaGenerationNumber());
            }
        }
        else {
            results->ClearLaReference();
        }
    }

    if constexpr (UseSmallExponents) {
        auto* resultsExisting = GetUsefulPerturbationResults<IterType, ConvertTType, false, PExtras>();
        if (resultsExisting == nullptr) {
            auto results2(std::make_unique<PerturbationResults<IterType, ConvertTType, PExtras>>(UseFileBackedOrbits()));
            results2->Copy<true>(*results,
                GetNextGenerationNumber(),
                GetNextLaGenerationNumber());

            auto ret = AddPerturbationResults(std::move(results2));
            m_LastUsedRefOrbit = ret;
            return ret;
        }
        else {
            m_LastUsedRefOrbit = resultsExisting;
            return resultsExisting;
        }
    }
    else {
        m_LastUsedRefOrbit = results;
        return results;
    }
}

template<
    typename IterType,
    class T,
    bool Authoritative,
    PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras>* RefOrbitCalc::GetUsefulPerturbationResults() {
    std::vector<PerturbationResults<IterType, T, PExtras>*> useful_results;
    std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>> &cur_array = GetPerturbationResults<IterType, T, PExtras>();

    if (!cur_array.empty()) {
        if (cur_array.size() > 64) {
            cur_array.erase(cur_array.begin());
        }

        for (size_t i = 0; i < cur_array.size(); i++) {
            if (IsPerturbationResultUsefulHere<IterType, T, Authoritative, PExtras>(i)) {
                useful_results.push_back(cur_array[i].get());
            }
        }
    }

    PerturbationResults<IterType, T, PExtras>* results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }

    return results;
}

template<
    typename IterType,
    class SrcT,
    PerturbExtras SrcEnableBad,
    class DestT,
    PerturbExtras DestEnableBad>
PerturbationResults<IterType, DestT, DestEnableBad>* RefOrbitCalc::CopyUsefulPerturbationResults(
    PerturbationResults<IterType, SrcT, SrcEnableBad>& src_array)
{
    if constexpr (DestEnableBad != SrcEnableBad) {
        return nullptr;
    }

    static_assert(
        (DestEnableBad == PerturbExtras::Disable && SrcEnableBad == PerturbExtras::Disable) ||
        (DestEnableBad == PerturbExtras::Bad && SrcEnableBad == PerturbExtras::Bad), "!");

    auto& container = GetContainer<IterType, DestEnableBad>();

    if constexpr (std::is_same<SrcT, double>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(UseFileBackedOrbits());
        container.m_PerturbationResultsFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsFloat[container.m_PerturbationResultsFloat.size() - 1].get();
        dest->Copy<false>(
            src_array,
            GetNextGenerationNumber(),
            GetNextLaGenerationNumber());
        return dest;
    }
    else if constexpr (std::is_same<SrcT, float>::value) {
        return nullptr;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, HDRFloat<float>, DestEnableBad>>(UseFileBackedOrbits());
        container.m_PerturbationResultsHDRFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsHDRFloat[container.m_PerturbationResultsHDRFloat.size() - 1].get();
        dest->Copy<false>(
            src_array,
            GetNextGenerationNumber(),
            GetNextLaGenerationNumber());
        return dest;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(UseFileBackedOrbits());
        container.m_PerturbationResultsFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsFloat[container.m_PerturbationResultsFloat.size() - 1].get();
        dest->Copy<false>(
            src_array,
            GetNextGenerationNumber(),
            GetNextLaGenerationNumber());
        return dest;
    }
    else {
        return nullptr;
    }
}

void RefOrbitCalc::ClearPerturbationResults(PerturbationResultType type) {
    auto IsMarkedToDelete = [&](const auto& o) -> bool {
        if (type == PerturbationResultType::All ||
            (type == PerturbationResultType::MediumRes &&
                o->GetAuthoritativePrecision() == 0) ||
            (type == PerturbationResultType::HighRes &&
                o->GetAuthoritativePrecision() != 0)) {
            return true;
        }

        return false;
    };

    auto ClearOne = [&](auto &arr) {
        arr.erase(
            std::remove_if(arr.begin(), arr.end(), IsMarkedToDelete),
            arr.end());
    };

    auto ClearContainer = [&](auto& container) {
        ClearOne(container.m_PerturbationResultsDouble);
        ClearOne(container.m_PerturbationResultsFloat);
        ClearOne(container.m_PerturbationResults2xFloat);
        ClearOne(container.m_PerturbationResultsHDRDouble);
        ClearOne(container.m_PerturbationResultsHDRFloat);
        ClearOne(container.m_PerturbationResultsHDR2xFloat);
    };

    ClearContainer(c32d);
    ClearContainer(c32e);
    ClearContainer(c32c);
    ClearContainer(c64d);
    ClearContainer(c64e);
    ClearContainer(c64c);

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;
}

void RefOrbitCalc::ResetGuess(HighPrecision x, HighPrecision y) {
    m_PerturbationGuessCalcX = x;
    m_PerturbationGuessCalcY = y;
}

void RefOrbitCalc::SaveAllOrbits() {
    // Saves all the results to disk
    auto lambda = [&](auto& Container) {
        for (const auto &it : Container.m_PerturbationResultsDouble) {
            auto filebase = GetTimeAsString();
            it->Write(true, filebase);
        }

        for (const auto &it : Container.m_PerturbationResultsFloat) {
            auto filebase = GetTimeAsString();
            it->Write(true, filebase);
        }

        // TODO - does not currently make sense to save these.
        //for (const auto& it : Container.m_PerturbationResults2xFloat) {
        //    auto filebase = GetTimeAsString();
        //    it->Write(true, filebase);
        //}

        for (const auto &it : Container.m_PerturbationResultsHDRDouble) {
            auto filebase = GetTimeAsString();
            it->Write(true, filebase);
        }
    
        for (const auto &it : Container.m_PerturbationResultsHDRFloat) {
            auto filebase = GetTimeAsString();
            it->Write(true, filebase);
        }

        // TODO - does not currently make sense to save these.
        //for (const auto& it : Container.m_PerturbationResults2xFloat) {
        //    auto filebase = GetTimeAsString();
        //    it->Write(true, filebase);
        //}

        for (const auto &it : Container.m_PerturbationResultsHDR2xFloat) {
            auto filebase = GetTimeAsString();
            it->Write(true, filebase);
        }
    };

    lambda(c32d);
    lambda(c32e);
    lambda(c32c);
    lambda(c64d);
    lambda(c64e);
    lambda(c64c);
}

void RefOrbitCalc::LoadAllOrbits() {
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
        std::transform(narrow.begin(), narrow.end(), std::back_inserter(wide), [](char c) {
            return (wchar_t)c;
            });
        return wide;
    };

    // This is such a stupid implementation of this but whatever
    // TODO: make this not stupid
    // The idea is to load all the relevant files in the current directory.
    auto lambda = [&]<typename IterType, PerturbExtras PExtras>(auto& Container) {
        std::string path = ".";
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            auto file = entry.path().string();
            if (extmatch(file)) {
                auto NextGen = GetNextGenerationNumber();
                auto NextLaGen = GetNextLaGenerationNumber();
                auto stripfn = stripext(file);
                auto widefn = narrowtowide(stripfn);
                auto results1 = std::make_unique<PerturbationResults<IterType, double, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results1->Load(widefn)) {
                    Container.m_PerturbationResultsDouble.push_back(std::move(results1));
                    continue;
                }

                auto results2 = std::make_unique<PerturbationResults<IterType, float, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results2->Load(widefn)) {
                    Container.m_PerturbationResultsFloat.push_back(std::move(results2));
                    continue;
                }

                auto results3 = std::make_unique<PerturbationResults<IterType, CudaDblflt<MattDblflt>, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results3->Load(widefn)) {
                    Container.m_PerturbationResults2xFloat.push_back(std::move(results3));
                    continue;
                }

                auto results4 = std::make_unique<PerturbationResults<IterType, HDRFloat<double>, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results4->Load(widefn)) {
                    Container.m_PerturbationResultsHDRDouble.push_back(std::move(results4));
                    continue;
                }

                auto results5 = std::make_unique<PerturbationResults<IterType, HDRFloat<float>, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results5->Load(widefn)) {
                    Container.m_PerturbationResultsHDRFloat.push_back(std::move(results5));
                    continue;
                }

                auto results6 = std::make_unique<PerturbationResults<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PExtras>>(
                    UseFileBackedOrbits(),
                    NextGen,
                    NextLaGen);
                if (results6->Load(widefn)) {
                    Container.m_PerturbationResultsHDR2xFloat.push_back(std::move(results6));
                    continue;
                }
            }
        }
    };

    lambda.template operator()<uint32_t, PerturbExtras::Disable>(c32d);
    lambda.template operator()<uint32_t, PerturbExtras::Bad>(c32e);
    lambda.template operator()<uint32_t, PerturbExtras::EnableCompression>(c32c);
    lambda.template operator()<uint64_t, PerturbExtras::Disable>(c64d);
    lambda.template operator()<uint64_t, PerturbExtras::Bad >(c64e);
    lambda.template operator()<uint64_t, PerturbExtras::EnableCompression>(c64c);
}

template<typename IterType, PerturbExtras PExtras>
RefOrbitCalc::Container<IterType, PExtras>& RefOrbitCalc::GetContainer() {
    return const_cast<RefOrbitCalc::Container<IterType, PExtras>&>(
        std::as_const(*this).GetContainer<IterType, PExtras>());
}

template<typename IterType, PerturbExtras PExtras>
const RefOrbitCalc::Container<IterType, PExtras> &RefOrbitCalc::GetContainer() const {
    if constexpr (std::is_same<IterType, uint32_t>::value && PExtras == PerturbExtras::Disable) {
        return c32d;
    }
    else if constexpr (std::is_same<IterType, uint32_t>::value && PExtras == PerturbExtras::Bad) {
        return c32e;
    } 
    else if constexpr (std::is_same<IterType, uint32_t>::value && PExtras == PerturbExtras::EnableCompression) {
        return c32c;
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value && PExtras == PerturbExtras::Disable) {
        return c64d;
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value && PExtras == PerturbExtras::Bad) {
        return c64e;
    }
    else if constexpr (std::is_same<IterType, uint64_t>::value && PExtras == PerturbExtras::EnableCompression) {
        return c64c;
    }
    else {
        assert(false);
        return c32d;
    }
}

void RefOrbitCalc::GetSomeDetails(
    uint64_t& PeriodMaybeZero,
    uint64_t& CompressedIters,
    uint64_t& UncompressedIters,
    int32_t &CompressionErrorExp) {


    PeriodMaybeZero = 0;
    CompressedIters = 0;
    UncompressedIters = 0;
    CompressionErrorExp = 0;

    auto lambda = [&](auto &&arg) {
        if (arg != nullptr) {
            PeriodMaybeZero = arg->GetPeriodMaybeZero();
            CompressedIters = arg->GetCompressedOrbitSize();
            UncompressedIters = arg->GetCountOrbitEntries();
            CompressionErrorExp = arg->GetCompressionErrorExp();
        }
    };

    std::visit(lambda, m_LastUsedRefOrbit);

    static_assert(static_cast<int>(RenderAlgorithm::MAX) == 62, "Fix me");
}

template<typename IterType, class T, PerturbExtras PExtras>
void RefOrbitCalc::DrawPerturbationResultsHelper() {
    // TODO can we just integrate all this with DrawFractal

    const auto& results = GetPerturbationResults<IterType, T, PExtras>();
    for (size_t i = 0; i < results.size(); i++)
    {
        if (IsPerturbationResultUsefulHere<IterType, T, false, PExtras>(i)) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            const GLint scrnX =
                Convert<HighPrecision, GLint>(m_Fractal.XFromCalcToScreen(results[i]->GetHiX()));
            const GLint scrnY =
                static_cast<GLint>(m_Fractal.GetRenderHeight()) -
                Convert<HighPrecision, GLint>(m_Fractal.YFromCalcToScreen(results[i]->GetHiY()));

            // Coordinates are weird in OGL mode.
            glVertex2i(scrnX, scrnY);
        }
    }
}

void RefOrbitCalc::DrawPerturbationResults() {

    auto drawbatch = [&]<typename IterType>() {
        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, CudaDblflt<MattDblflt>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::Disable>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable>();

        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, CudaDblflt<MattDblflt>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::Bad>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad>();

        DrawPerturbationResultsHelper<IterType, double, PerturbExtras::EnableCompression>();
        DrawPerturbationResultsHelper<IterType, float, PerturbExtras::EnableCompression>();
        DrawPerturbationResultsHelper<IterType, CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<double>, PerturbExtras::EnableCompression>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<float>, PerturbExtras::EnableCompression>();
        DrawPerturbationResultsHelper<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression>();
    };

    drawbatch.template operator() < uint32_t > ();
    drawbatch.template operator() < uint64_t > ();
}

#include "RefOrbitCalcTemplates.h"