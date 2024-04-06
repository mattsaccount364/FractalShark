#include "stdafx.h"

#include "RefOrbitCalc.h"
#include "Fractal.h"
#include "PerturbationResults.h"

#include "ScopedMpir.h"

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
    ENABLE_PREFETCH((const char*)target.backend(), _MM_HINT_T0);
    size_t lastindex = abs(target.backend()->_mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target.backend()->_mp_d, total);
}

static inline void PrefetchHighPrec(const mpf_t& target) {
    ENABLE_PREFETCH((const char*)&target[0], _MM_HINT_T0);
    size_t lastindex = abs(target[0]._mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target[0]._mp_d, total);
}


RefOrbitCalc::RefOrbitCalc(Fractal& Fractal)
    : m_PerturbationAlg{ PerturbationAlg::Auto },
      m_Fractal(Fractal),
      m_PerturbationGuessCalcX(0),
      m_PerturbationGuessCalcY(0),
      m_RefOrbitOptions{ AddPointOptions::DontSave },
      m_GuessReserveSize(),
      m_GenerationNumber() {

    // Get number of CPU cores and whether hyperthreading is enabled
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    m_NumCpuCores = sysinfo.dwNumberOfProcessors;

    // Find whether hyperthreading is enabled via GetLogicalProcessorInformation
    m_HyperthreadingEnabled = false;
    DWORD returnLength = 0;
    GetLogicalProcessorInformation(nullptr, &returnLength);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{ returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) };
        GetLogicalProcessorInformation(buffer.data(), &returnLength);
        for (const auto& info : buffer) {
            if (info.Relationship == RelationProcessorCore) {
                if (info.ProcessorCore.Flags == LTP_PC_SMT) {
                    m_HyperthreadingEnabled = true;
                    break;
                }
            }
        }
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

bool RefOrbitCalc::RequiresReuse() const {
    switch (GetPerturbationAlg()) {
    case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
    case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
    case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
    case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
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

    static_assert(static_cast<int>(RenderAlgorithm::MAX) == 61, "Fix me");
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
        ::MessageBox(nullptr, L"Watch the memory use... this is just a warning", L"", MB_OK | MB_APPLMODAL);
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

void RefOrbitCalc::SetOptions(AddPointOptions options) {
    m_RefOrbitOptions = options;
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
    if (m_PerturbationGuessCalcX == HighPrecision{} && m_PerturbationGuessCalcY == HighPrecision{}) {
        m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision(2);
        m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision(2);
    }
    
    if (GetPerturbationAlg() == PerturbationAlg::ST) {
        AddPerturbationReferencePointST<IterType, T, SubType, false, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MT) {
        AddPerturbationReferencePointMT3<IterType, T, SubType, false, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::STPeriodicity) {
        AddPerturbationReferencePointST<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3) {
        AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed) {
        // TODO use MT version
        AddPerturbationReferencePointST<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::SaveForReuse3>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1) {

        AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::SaveForReuse1>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2) {
        AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::SaveForReuse2>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3) {
        AddPerturbationReferencePointMT3<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::SaveForReuse3>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
    else if (GetPerturbationAlg() == PerturbationAlg::MTPeriodicity5) {
        AddPerturbationReferencePointMT5<IterType, T, SubType, true, BenchmarkState, PExtras, ReuseMode::DontSaveForReuse>(
            m_PerturbationGuessCalcX,
            m_PerturbationGuessCalcY);
    }
}

// TODO remove me
//template<class T>
//static void AddReused(T &results, const HighPrecision& zx, const HighPrecision& zy) {
//    HighPrecision ReducedZx;
//    HighPrecision ReducedZy;
//
//    ReducedZx = zx;
//    ReducedZy = zy;
//
//    //assert(RequiresReuse());
//    ReducedZx.precisionInBits(AuthoritativeReuseExtraPrecisionInBits);
//    ReducedZy.precisionInBits(AuthoritativeReuseExtraPrecisionInBits);
//
//    results.AddUncompressedReusedEntry(std::move(ReducedZx), std::move(ReducedZy));
//}

size_t RefOrbitCalc::GetNextGenerationNumber() {
    ++m_GenerationNumber;
    return m_GenerationNumber;
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
        m_GuessReserveSize);
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
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PerturbationResultsArray.push_back(std::move(newArray));
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    std::unique_ptr<MPIRBoundedAllocator> boundedAllocator;
    std::unique_ptr<MPIRBumpAllocator> bumpAllocator;

    InitAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);

    if constexpr (
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {

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

    mpf_t zx2, zy2;
    mpf_init(zx2);
    mpf_init(zy2);

    mpf_t temp_mpf;
    mpf_init(temp_mpf);

    mpf_t temp2_mpf;
    mpf_init(temp2_mpf);

    IterTypeFull i;

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    T dzdcX = T{ 1 };
    T dzdcY = T{ 0 };

    constexpr bool floatOrDouble =
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value;
    T cx_cast;
    T cy_cast;
    if constexpr (floatOrDouble) {
        cx_cast = (T)mpf_get_d(cx_mpf);
        cy_cast = (T)mpf_get_d(cy_mpf);
    }
    else {
        int32_t cx_exponent, cy_exponent;
        double cx_mantissa, cy_mantissa;

        cx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cx_mantissa, cx_mpf));
        cy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cy_mantissa, cy_mpf));

        cx_cast = T{ cx_exponent, static_cast<SubType>(cx_mantissa) };
        cy_cast = T{ cy_exponent, static_cast<SubType>(cy_mantissa) };
    }

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    RefOrbitCompressor<IterType, T, PExtras> compressor{
        *results,
        m_Fractal.GetCompressionErrorExp() };

    IntermediateOrbitCompressor<IterType, T, PExtras> intermediateCompressor{
        *results,
        m_Fractal.GetCompressionErrorExp() };

    mpf_set(zx, cx_mpf);
    mpf_set(zy, cy_mpf);
    
    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++)
    {
        mpf_mul_2exp(zx2, zx, 1); // Multiply by 2
        mpf_mul_2exp(zy2, zy, 1);

        // TODO mpz_get_d_2exp
        T double_zx;
        T double_zy;

        //  = (T)mpf_get_d(zx);
        // = (T)mpf_get_d(zy);
        if constexpr (floatOrDouble) {
            double_zx = (T)mpf_get_d(zx);
            double_zy = (T)mpf_get_d(zy);
        }
        else {
            //int32_t zx_exponent, zy_exponent;
            //double zx_mantissa, zy_mantissa;

            //zx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zx_mantissa, zx));
            //zy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zy_mantissa, zy));

            //double_zx = T{ zx_exponent, static_cast<SubType>(zx_mantissa) };
            //double_zy = T{ zy_exponent, static_cast<SubType>(zy_mantissa) };

            double_zx = T{ zx };
            double_zy = T{ zy };
        }

        if constexpr (PExtras == PerturbExtras::Disable) {
            results->AddUncompressedIteration({ double_zx, double_zy });
        } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            compressor.MaybeAddCompressedIteration({ double_zx, double_zy, i + 1 });
        }
        else if constexpr (PExtras == PerturbExtras::Bad) {
            results->AddUncompressedIteration({ double_zx, double_zy, false });
        }

        if constexpr (
            Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
            Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
            Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
            // AddReused(*results, zx, zy);
            intermediateCompressor.MaybeAddCompressedIteration(zx, zy, i + 1);
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            const T sq_x = double_zx * double_zx;
            const T sq_y = double_zy * double_zy;
            const T norm = HdrReduce((sq_x + sq_y) * glitch);

            // TODO This is stupid - we can fix this by using double_zx/double_zy:
            const auto zx_reduced = HdrReduce(HdrAbs((T)mpf_get_d(zx)));
            const auto zy_reduced = HdrReduce(HdrAbs((T)mpf_get_d(zy)));
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

        //zx = zx * zx - zy * zy + cx;
        mpf_mul(temp_mpf, zx, zx);
        mpf_mul(temp2_mpf, zy, zy);
        mpf_sub(zx, temp_mpf, temp2_mpf);
        mpf_add(zx, zx, cx_mpf);

        //zy = zx2 * zy + cy;
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

    results->CompleteResults<PExtras, Reuse>(bumpAllocator->GetAllocated(1));
    m_GuessReserveSize = results->GetCountOrbitEntries();
    } // End of scope for allocators.

    ShutdownAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);
}

template<
    typename IterType,
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState>
bool RefOrbitCalc::AddPerturbationReferencePointSTReuse(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<IterType, T, PerturbExtras::Disable>();
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PerturbationResultsArray.push_back(std::move(newArray));

    auto* existingResults = GetUsefulPerturbationResults<IterType, T, true, PerturbExtras::Disable>();
    // TODO Lame hack with < 5.
    // existingResults->GetReuseSize() < 5
    if (existingResults == nullptr) {
        PerturbationResultsArray.pop_back();
        return false;
    }

    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    const int64_t NewPrec = m_Fractal.GetPrecision(
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        RequiresReuse());
    const int64_t existingPrecision = existingResults->GetAuthoritativePrecisionInBits();
    const int64_t deltaPrecision = NewPrec - existingPrecision;
    const int64_t extraPrecision = static_cast<int64_t>(AuthoritativeReuseExtraPrecisionInBits - AuthoritativeMinExtraPrecisionInBits);

    // This all generally works and only starts to suffer precision problems after
    // about 10^AuthoritativeReuseExtraPrecisionInBits. The problem naturally is the original
    // reference orbit is calculated only to so many digits.
    if(deltaPrecision >= extraPrecision) {
        //::MessageBox(nullptr, L"Regenerating authoritative orbit is required", L"", MB_OK | MB_APPLMODAL);
        PerturbationResultsArray.pop_back();
        return false;
    }

    MPIRPrecision prec(AuthoritativeReuseExtraPrecisionInBits);
    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);

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

    IntermediateCompressionHelper<IterType, T, PerturbExtras::Disable> intermediateCompressor{ *existingResults };

    constexpr bool floatOrDouble =
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value;

    const HighPrecisionT<HPDestructor::False>* ReuseX = nullptr;
    const HighPrecisionT<HPDestructor::False>* ReuseY = nullptr;

    for (i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++) {
        if constexpr (Periodicity) {
            if constexpr (floatOrDouble) {
                zxCopy = (T)mpf_get_d(zx);
                zyCopy = (T)mpf_get_d(zy);
            }
            else {
                zxCopy = T{ zx };
                zyCopy = T{ zy };
            }
        }

        mpf_set(DeltaSubNXOrig, DeltaSubNX);
        mpf_set(DeltaSubNYOrig, DeltaSubNY);

        existingResults->GetCompressedReuseEntries(
            intermediateCompressor,
            RefIteration,
            ReuseX,
            ReuseY);

        //DeltaSubNX =
        //    DeltaSubNXOrig * ((*ReuseX) * HighTwo + DeltaSubNXOrig) -
        //    DeltaSubNYOrig * ((*ReuseY) * HighTwo + DeltaSubNYOrig) +
        //    DeltaSub0X;
        mpf_mul(temp_mpf, ReuseX->backend(), HighTwo);
        mpf_add(temp_mpf, temp_mpf, DeltaSubNXOrig);
        mpf_mul(temp2_mpf, ReuseY->backend(), HighTwo);
        mpf_add(temp2_mpf, temp2_mpf, DeltaSubNYOrig);
        mpf_mul(DeltaSubNX, DeltaSubNXOrig, temp_mpf);
        mpf_mul(temp_mpf, DeltaSubNYOrig, temp2_mpf);
        mpf_sub(DeltaSubNX, DeltaSubNX, temp_mpf);
        mpf_add(DeltaSubNX, DeltaSubNX, DeltaSub0X);

        //DeltaSubNY =
        //    DeltaSubNXOrig * ((*ReuseY) * HighTwo + DeltaSubNYOrig) +
        //    DeltaSubNYOrig * ((*ReuseX) * HighTwo + DeltaSubNXOrig) +
        //    DeltaSub0Y;
        mpf_mul(temp_mpf, ReuseY->backend(), HighTwo);
        mpf_add(temp_mpf, temp_mpf, DeltaSubNYOrig);
        mpf_mul(temp2_mpf, ReuseX->backend(), HighTwo);
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
        }
        else {
            tempDeltaSubNXLow = T{ DeltaSubNX };
            tempDeltaSubNYLow = T{ DeltaSubNY };
            tempZXLow = T{ zx };
            tempZYLow = T{ zy };
        }

        ++RefIteration;

        // results->AddUncompressedIteration({ tempZXLow, tempZYLow });

        existingResults->GetCompressedReuseEntries(
            intermediateCompressor,
            RefIteration,
            ReuseX,
            ReuseY);

        //zx = (*ReuseX) + DeltaSubNX;
        mpf_add(zx, ReuseX->backend(), DeltaSubNX);

        //zy = (*ReuseY) + DeltaSubNY;
        mpf_add(zy, ReuseY->backend(), DeltaSubNY);

        zn_size = tempZXLow * tempZXLow + tempZYLow * tempZYLow;
        HdrReduce(zn_size);
        normDeltaSubN = tempDeltaSubNXLow * tempDeltaSubNXLow + tempDeltaSubNYLow * tempDeltaSubNYLow;
        HdrReduce(normDeltaSubN);

        if (HdrCompareToBothPositiveReducedLT(zn_size, normDeltaSubN) ||
            RefIteration == MaxRefIteration) {
            //DeltaSubNX = zx;
            mpf_set(DeltaSubNX, zx);

            //DeltaSubNY = zy;
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

        if (HdrCompareToBothPositiveReducedGT(zn_size, TwoFiftySix)) {
            break;
        }

        if constexpr (floatOrDouble) {
            T reducedZx = (T)mpf_get_d(zx);
            T reducedZy = (T)mpf_get_d(zy);

            results->AddUncompressedIteration({ reducedZx, reducedZy });
        }
        else {
            T reducedZx = T{ zx };
            T reducedZy = T{ zy };

            results->AddUncompressedIteration({ reducedZx, reducedZy });
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

    results->CompleteResults<PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(nullptr);
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
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PerturbationResultsArray.push_back(std::move(newArray));

    auto* existingResults = GetUsefulPerturbationResults<IterType, T, true, PerturbExtras::Disable>();
    // TODO Lame hack with < 5.
    // existingResults->GetReuseSize() < 5
    if (existingResults == nullptr) {
        PerturbationResultsArray.pop_back();
        return false;
    }

    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    const int64_t NewPrec = m_Fractal.GetPrecision(
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        RequiresReuse());
    const int64_t existingPrecision = existingResults->GetAuthoritativePrecisionInBits();
    const int64_t deltaPrecision = NewPrec - existingPrecision;
    const int64_t extraPrecision = static_cast<int64_t>(AuthoritativeReuseExtraPrecisionInBits - AuthoritativeMinExtraPrecisionInBits);

    // This all generally works and only starts to suffer precision problems after
    // about 10^AuthoritativeReuseExtraPrecisionInBits. The problem naturally is the original
    // reference orbit is calculated only to so many digits.
    if (deltaPrecision >= extraPrecision) {
        //::MessageBox(nullptr, L"Regenerating authoritative orbit is required", L"", MB_OK | MB_APPLMODAL);
        PerturbationResultsArray.pop_back();
        return false;
    }

    MPIRPrecision prec(AuthoritativeReuseExtraPrecisionInBits);
    InitResults<IterType, T, decltype(*results), PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);
    
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
        ThreadZxData() :
            DeltaSubNX{},
            ReferenceIteration{},
            DeltaSubNXOrig{},
            DeltaSubNYOrig{},
            DeltaSub0X{} {

            mpf_init(DeltaSubNX);
            mpf_init(OutZx);
        }

        ~ThreadZxData() {
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
        ThreadZyData() :
            DeltaSubNY{},
            ReferenceIteration{},
            DeltaSubNXOrig{},
            DeltaSubNYOrig{},
            DeltaSub0Y{} {
        
            mpf_init(DeltaSubNY);
            mpf_init(OutZy);
        }

        ~ThreadZyData() {
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

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    if (ThreadZxMemory == nullptr) {
        throw std::bad_alloc();
    }

    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    if (ThreadZyMemory == nullptr) {
        throw std::bad_alloc();
    }
        
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    constexpr bool floatOrDouble =
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value;

    auto ThreadSqZx = [&](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        const HighPrecisionT<HPDestructor::False>* ReuseX = nullptr;
        const HighPrecisionT<HPDestructor::False>* ReuseY = nullptr;

        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0X);
            //PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            //PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            // TODO:
            assert(false);
            // existingResults->GetReuseEntries(ok->ReferenceIteration, ReuseX, ReuseY);

            //ok->DeltaSubNX =
            //    (*ok->DeltaSubNXOrig) * ((*ReuseX) * HighTwo + (*ok->DeltaSubNXOrig)) -
            //    (*ok->DeltaSubNYOrig) * ((*ReuseY) * HighTwo + (*ok->DeltaSubNYOrig)) +
            //    (*ok->DeltaSub0X);

            mpf_mul(temp_mpf, ReuseX->backend(), HighTwo);
            mpf_add(temp_mpf, temp_mpf, *ok->DeltaSubNXOrig);
            mpf_mul(temp2_mpf, ReuseY->backend(), HighTwo);
            mpf_add(temp2_mpf, temp2_mpf, *ok->DeltaSubNYOrig);
            mpf_mul(ok->DeltaSubNX, *ok->DeltaSubNXOrig, temp_mpf);
            mpf_mul(temp_mpf, *ok->DeltaSubNYOrig, temp2_mpf);
            mpf_sub(ok->DeltaSubNX, ok->DeltaSubNX, temp_mpf);
            mpf_add(ok->DeltaSubNX, ok->DeltaSubNX, *ok->DeltaSub0X);

            // TODO
            assert(false);
            // existingResults->GetReuseEntries(ok->ReferenceIteration + 1, ReuseX, ReuseY);
            mpf_add(ok->OutZx, ReuseX->backend(), ok->DeltaSubNX);
            if constexpr (floatOrDouble) {
                ok->OutDeltaSubNXLow = (T)mpf_get_d(ok->DeltaSubNX);
                ok->OutZxLow = (T)mpf_get_d(ok->OutZx);
            }
            else {
                ok->OutDeltaSubNXLow = T(ok->DeltaSubNX);
                ok->OutZxLow = T{ ok->OutZx };
            }

            // Give result back.
            CheckFinishCriteria;
        }

        mpf_clear(temp_mpf);
        mpf_clear(temp2_mpf);
    };

    auto ThreadSqZy = [&](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        mpf_t temp_mpf;
        mpf_init(temp_mpf);

        mpf_t temp2_mpf;
        mpf_init(temp2_mpf);

        const HighPrecisionT<HPDestructor::False>* ReuseX = nullptr;
        const HighPrecisionT<HPDestructor::False>* ReuseY = nullptr;

        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(*ok->DeltaSubNXOrig);
            PrefetchHighPrec(*ok->DeltaSubNYOrig);
            PrefetchHighPrec(*ok->DeltaSub0Y);
            //PrefetchHighPrec(existingReuseX[ok->ReferenceIteration]);
            //PrefetchHighPrec(existingReuseY[ok->ReferenceIteration]);

            // TODO:
            assert(false);
            // existingResults->GetReuseEntries(ok->ReferenceIteration, ReuseX, ReuseY);

            //ok->DeltaSubNY =
            //    (*ok->DeltaSubNXOrig) * ((*ReuseY) * HighTwo + (*ok->DeltaSubNYOrig)) +
            //    (*ok->DeltaSubNYOrig) * ((*ReuseX) * HighTwo + (*ok->DeltaSubNXOrig)) +
            //    (*ok->DeltaSub0Y);
            mpf_mul(temp_mpf, ReuseY->backend(), HighTwo);
            mpf_add(temp_mpf, temp_mpf, *ok->DeltaSubNYOrig);
            mpf_mul(temp2_mpf, ReuseX->backend(), HighTwo);
            mpf_add(temp2_mpf, temp2_mpf, *ok->DeltaSubNXOrig);
            mpf_mul(ok->DeltaSubNY, *ok->DeltaSubNXOrig, temp_mpf);
            mpf_mul(temp_mpf, *ok->DeltaSubNYOrig, temp2_mpf);
            mpf_add(ok->DeltaSubNY, ok->DeltaSubNY, temp_mpf);
            mpf_add(ok->DeltaSubNY, ok->DeltaSubNY, *ok->DeltaSub0Y);

            // TODO
            assert(false);
            // existingResults->GetReuseEntries(ok->ReferenceIteration + 1, ReuseX, ReuseY);
            mpf_add(ok->OutZy, ReuseY->backend(), ok->DeltaSubNY);
            if constexpr (floatOrDouble) {
                ok->OutDeltaSubNYLow = (T)mpf_get_d(ok->DeltaSubNY);
                ok->OutZyLow = (T)mpf_get_d(ok->OutZy);
            }
            else {
                ok->OutDeltaSubNYLow = T(ok->DeltaSubNY);
                ok->OutZyLow = T{ ok->OutZy };
            }

            // Give result back.
            CheckFinishCriteria;
        }

        mpf_clear(temp_mpf);
        mpf_clear(temp2_mpf);
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};

    std::unique_ptr<std::thread> tZx(DEBUG_NEW std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(DEBUG_NEW std::thread(ThreadSqZy, ThreadZyMemory));

    ScopedAffinity scopedAffinity{
            *this,
            GetCurrentThread(),
            tZx->native_handle(),
            tZy->native_handle(),
            nullptr };

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
            if constexpr (floatOrDouble) {
                zxCopy = (T)mpf_get_d(zx);
                zyCopy = (T)mpf_get_d(zy);
            }
            else {
                zxCopy = T{ zx };
                zyCopy = T{ zy };
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
                mpf_set(zy, threadZydata->OutZy);

                tempDeltaSubNYLow = threadZydata->OutDeltaSubNYLow;
                tempZYLow = threadZydata->OutZyLow;
            }

            expectedZx = threadZxdata;

            _mm_pause();
            if (!done1 &&
                ThreadZxMemory->Out.compare_exchange_weak(expectedZx,
                    nullptr,
                    std::memory_order_release)) {
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

    results->CompleteResults<PerturbExtras::Disable, ReuseMode::DontSaveForReuse>(nullptr);
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
    auto newArray = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        m_RefOrbitOptions, GetNextGenerationNumber());
    PerturbationResultsArray.push_back(std::move(newArray));
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    InitResults<IterType, T, decltype(*results), PExtras, Reuse>(*results, cx, cy);

    std::unique_ptr<MPIRBoundedAllocator> boundedAllocator;
    std::unique_ptr<MPIRBumpAllocator> bumpAllocator;

    InitAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);

    if constexpr (
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {

        results->InitReused();
    }

    std::unique_ptr<GrowableVector<uint8_t>> reusedAllocator;

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

    mpf_t zx2, zy2;
    mpf_init(zx2);
    mpf_init(zy2);

    mpf_t temp_mpf;
    mpf_init(temp_mpf);

    mpf_t temp2_mpf;
    mpf_init(temp2_mpf);

    constexpr bool floatOrDouble =
        std::is_same<T, double>::value ||
        std::is_same<T, float>::value;
    T cx_cast;
    T cy_cast;
    if constexpr (floatOrDouble) {
        cx_cast = (T)mpf_get_d(cx_mpf);
        cy_cast = (T)mpf_get_d(cy_mpf);
    }
    else {
        int32_t cx_exponent, cy_exponent;
        double cx_mantissa, cy_mantissa;

        cx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cx_mantissa, cx_mpf));
        cy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&cy_mantissa, cy_mpf));

        cx_cast = T{ cx_exponent, static_cast<SubType>(cx_mantissa) };
        cy_cast = T{ cy_exponent, static_cast<SubType>(cy_mantissa) };
    }

    T dzdcX = T{ 1.0 };
    T dzdcY = T{ 0.0 };

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        ThreadZxData() {
            mpf_init(zx);
            mpf_init(zx_sq);
            zx_low = T{ 0.0 };
        }

        ~ThreadZxData() {
            mpf_clear(zx);
            mpf_clear(zx_sq);
        }

        mpf_t zx;
        mpf_t zx_sq;
        T zx_low;
    };

    struct ThreadZyData {
        ThreadZyData() {
            mpf_init(zy);
            mpf_init(zy_sq);
            zy_low = T{ 0.0 };
        }

        ~ThreadZyData() {
            mpf_clear(zy);
            mpf_clear(zy_sq);
        }

        mpf_t zy;
        mpf_t zy_sq;
        T zy_low;
    };

    struct ThreadReusedData {
        ThreadReusedData() {
            mpf_init(zx);
            mpf_init(zy);
        }

        ~ThreadReusedData() {
            mpf_clear(zx);
            mpf_clear(zy);
        }

        mpf_t zx;
        mpf_t zy;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    if (ThreadZxMemory == nullptr) {
        throw std::bad_alloc();
    }

    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    if (ThreadZyMemory == nullptr) {
        throw std::bad_alloc();
    }

    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto* ThreadReusedMemory = (ThreadPtrs<ThreadReusedData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadReusedData>), 64);
    if (ThreadReusedMemory == nullptr) {
        throw std::bad_alloc();
    }

    memset(ThreadReusedMemory, 0, sizeof(*ThreadReusedMemory));

    auto InitTls = [&]() {
        if constexpr (
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3) {
            boundedAllocator->InitTls();
        }
        else {
            bumpAllocator->InitTls();
        }
    };

    auto ShutdownTls = [&]() {
        if constexpr (
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
            Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3) {
            boundedAllocator->ShutdownTls();
        }
        else {
            bumpAllocator->ShutdownTls();
        }
    };

    auto ThreadSqZx = [&InitTls,&ShutdownTls](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        InitTls();

        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            //PrefetchHighPrec(ok->zx);

            // ok->zx_low = (T)mpf_get_d(ok->zx);
            if constexpr (floatOrDouble) {
                ok->zx_low = (T)mpf_get_d(ok->zx);
            }
            else {
                int32_t zx_exponent;
                double zx_mantissa;
                zx_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zx_mantissa, ok->zx));
                ok->zx_low = T{ zx_exponent, static_cast<SubType>(zx_mantissa) };
            }

            mpf_mul(ok->zx_sq, ok->zx, ok->zx);

            // Give result back.
            CheckFinishCriteria;
        }

        ShutdownTls();
    };

    auto ThreadSqZy = [&InitTls, &ShutdownTls](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        InitTls();

        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            //PrefetchHighPrec(ok->zy);

            //ok->zy_low = (T)mpf_get_d(ok->zy);
            if constexpr (floatOrDouble) {
                ok->zy_low = (T)mpf_get_d(ok->zy);
            }
            else {
                int32_t zy_exponent;
                double zy_mantissa;
                zy_exponent = static_cast<int32_t>(mpf_get_2exp_d(&zy_mantissa, ok->zy));
                ok->zy_low = T{ zy_exponent, static_cast<SubType>(zy_mantissa) };
            }

            mpf_mul(ok->zy_sq, ok->zy, ok->zy);

            // Give result back.
            CheckFinishCriteria;
        }

        ShutdownTls();
    };

    auto ThreadReused = [
        results,
        &bumpAllocator,
        &reusedAllocator,
        &InitTls,
        &ShutdownTls](ThreadPtrs<ThreadReusedData>* ThreadMemory) {

        InitTls();

        for (;;) {
            ThreadReusedData* expected = ThreadMemory->In.load();
            ThreadReusedData* ok = nullptr;

            CheckStartCriteria;

            // AddReused(*results, ok->zx, ok->zy);

            assert(false);
            // TODO

            // Give result back.
            CheckFinishCriteria;
        }

        auto index = bumpAllocator->GetAllocatorIndex();
        reusedAllocator = bumpAllocator->GetAllocated(index);

        ShutdownTls();
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);
    auto* threadReuseddata = (ThreadReusedData*)_aligned_malloc(sizeof(ThreadReusedData), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};
    new (threadReuseddata) (ThreadReusedData){};

    std::unique_ptr<std::thread> tZx(DEBUG_NEW std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(DEBUG_NEW std::thread(ThreadSqZy, ThreadZyMemory));

    std::unique_ptr<std::thread> tReuse;

    // Mode 2 we use another thread
    if constexpr (
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {

        tReuse = std::unique_ptr<std::thread>(DEBUG_NEW std::thread(ThreadReused, ThreadReusedMemory));
    }

    ScopedAffinity scopedAffinity{
        *this,
        GetCurrentThread(),
        tZx->native_handle(),
        tZy->native_handle(),
        tReuse ? tReuse->native_handle() : nullptr };

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;
    ThreadReusedData* expectedReused = nullptr;

    bool done1 = false;
    bool done2 = false;

    mpf_t zy_sq_orig;
    mpf_init(zy_sq_orig);

    mpf_set(zx, cx_mpf);
    mpf_set(zy, cy_mpf);

    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    bool zyStarted = false;

    T double_zx_last = T{ 0.0 };
    T double_zy_last = T{ 0.0 };

    RefOrbitCompressor<IterType, T, PExtras> compressor{
        *results,
        m_Fractal.GetCompressionErrorExp() };

    for (IterTypeFull i = 0; i < m_Fractal.GetNumIterations<IterType>(); i++)
    {
        // Start Zx squaring thread
        mpf_set(threadZxdata->zx, zx);

        if (!zyStarted) {
            mpf_set(threadZydata->zy, zy);
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

        T double_zx = double_zx_last;
        T double_zy = double_zy_last;

        SubType zn_size;

        if (i > 0) {
            if constexpr (PExtras == PerturbExtras::Disable) {
                results->AddUncompressedIteration({ double_zx, double_zy });
            }
            else if constexpr (PExtras == PerturbExtras::EnableCompression) {
                compressor.MaybeAddCompressedIteration({ double_zx, double_zy, i });
            }
            else if constexpr (PExtras == PerturbExtras::Bad) {
                results->AddUncompressedIteration({ double_zx, double_zy, false });
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
                }
                else {
                    auto dzdcXOrig = dzdcX;
                    dzdcX = HighTwo * (double_zx * dzdcX - double_zy * dzdcY) + HighOne;
                    dzdcY = HighTwo * (double_zx * dzdcY + double_zy * dzdcXOrig);
                }
            }

            if constexpr (
                Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
                Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2) {

                for (;;) {
                    expectedReused = threadReuseddata;

                    _mm_pause();
                    if (ThreadReusedMemory->Out.compare_exchange_weak(expectedReused,
                        nullptr,
                        std::memory_order_release)) {
                        break;
                    }
                }
            }
        }
        else {
            zn_size = 0;
        }

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1) {
            assert(false);
            // AddReused(*results, zx, zy);
            // TODO
        } else if constexpr (
            Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
            Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
            mpf_set(threadReuseddata->zx, zx);
            mpf_set(threadReuseddata->zy, zy);

            ThreadReusedMemory->In.store(
                threadReuseddata,
                std::memory_order_release);
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
                    mpf_set(zy_sq_orig, threadZydata->zy_sq);
                    double_zy_last = threadZydata->zy_low;

                    // Restart right away!
                    mpf_set(threadZydata->zy, zy);

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
        res1 = ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);
    }

    while (!res2) {
        expectedZy = nullptr;
        res2 = ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);
    }

    while (!res3) {
        expectedReused = nullptr;
        res3 = ThreadReusedMemory->In.compare_exchange_strong(expectedReused, (ThreadReusedData*)0x1, std::memory_order_release);
    }

    tZx->join();
    tZy->join();

    if constexpr (
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {

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

    results->CompleteResults<PExtras, Reuse>(std::move(reusedAllocator));
    m_GuessReserveSize = results->GetCountOrbitEntries();
    } // End of scope for boundedAllocator and bumpAllocator

    ShutdownAllocatorsIfNeeded<Reuse>(boundedAllocator, bumpAllocator);
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
    ::MessageBox(nullptr, L"AddPerturbationReferencePointMT5 disabled, using MT3", L"", MB_OK | MB_APPLMODAL);
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

        if constexpr (Authoritative == true) {
            return
                PerturbationResults.GetAuthoritativePrecisionInBits() != 0 &&
                (PerturbationResults.GetMaxIterations() > PerturbationResults.GetCountOrbitEntries() ||
                    PerturbationResults.GetMaxIterations() >= m_Fractal.GetNumIterations<IterType>());
        }

        const auto term1 = PerturbationResults.GetHiX() >= m_Fractal.GetMinX();
        const auto term2 = PerturbationResults.GetHiX() <= m_Fractal.GetMaxX();
        const auto term3 = PerturbationResults.GetHiY() >= m_Fractal.GetMinY();
        const auto term4 = PerturbationResults.GetHiY() <= m_Fractal.GetMaxY();
        return
            term1 && term2 && term3 && term4 &&
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
RefOrbitCalc::GetUsefulPerturbationResults() {
    auto* resultsExisting = GetUsefulPerturbationResults<IterType, ConvertTType, false, PExtras>();

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

template<
    typename IterType,
    class T,
    class SubType,
    PerturbExtras PExtras,
    RefOrbitCalc::Extras Ex,
    class ConvertTType>
PerturbationResults<IterType, ConvertTType, PExtras>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults() {
    constexpr bool IsHdrDblflt = std::is_same<ConvertTType, HDRFloat<CudaDblflt<MattDblflt>>>::value;
    constexpr bool IsDblflt = std::is_same<ConvertTType, CudaDblflt<MattDblflt>>::value;
    constexpr bool UsingDblflt = IsHdrDblflt || IsDblflt;
    constexpr auto PExtrasHackYay =
        (PExtras == PerturbExtras::EnableCompression) ? PerturbExtras::Disable : PExtras;
    bool added = false;

    auto GenLAResults = [&](auto& results, AddPointOptions options_to_use) {
        if (results->GetLaReference() == nullptr) {
            auto temp = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(
                m_Fractal.GetLAParameters(),
                options_to_use,
                results->GenFilename(GrowableVectorTypes::LAInfoDeep, L"", true),
                results->GenFilename(GrowableVectorTypes::LAStageInfo, L"", true));

            // TODO the presumption here is results size fits in the target IterType size
            temp->GenerateApproximationData(
                *results,
                results->GetMaxRadius(),
                UsingDblflt);

            added = true;

            results->SetLaReference(std::move(temp));
        }
    };

    if (RequiresReuse()) {
        if (m_PerturbationGuessCalcX == HighPrecision{} && m_PerturbationGuessCalcY == HighPrecision{}) {
            m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision{ 2 };
            m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision{ 2 };
        }

        PerturbationResults<IterType, T, PExtrasHackYay>* results =
            GetUsefulPerturbationResults<IterType, T, false, PExtrasHackYay>();
        if (results == nullptr) {
            switch (GetPerturbationAlg()) {
            case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
                added = AddPerturbationReferencePointSTReuse<IterType, T, SubType, true, BenchmarkMode::Disable>
                    (m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1:
            case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2:
            case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3:
                added = AddPerturbationReferencePointMT3Reuse<IterType, T, SubType, true, BenchmarkMode::Disable>
                    (m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            default:
                ::MessageBox(nullptr, L"Some stupid bug #2343 :(", L"", MB_OK | MB_APPLMODAL);
                assert(false);
                break;
            }
        }
    }

    PerturbationResults<IterType, T, PExtras>* results =
        GetUsefulPerturbationResults<IterType, T, false, PExtras>();
    if (results == nullptr) {
        if (added) {
            ::MessageBox(nullptr, L"Why didn't this work! :(", L"", MB_OK | MB_APPLMODAL);
        }

        if constexpr (UsingDblflt) {
            PerturbationResults<IterType, ConvertTType, PExtras>* results_converted =
                GetUsefulPerturbationResults<IterType, ConvertTType, false, PExtras>();

            if (results_converted != nullptr) {
                return results_converted;
            }
        }

        std::vector<std::unique_ptr<PerturbationResults<IterType, T, PExtras>>>& cur_array =
            GetPerturbationResults<IterType, T, PExtras>();
        AddPerturbationReferencePoint<IterType, T, SubType, PExtras, BenchmarkMode::Disable>();
        added = true;

        results = cur_array[cur_array.size() - 1].get();
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

    if constexpr (
        std::is_same<T, float>::value || // TODO: these are new.  Maybe OK to keep here.
        std::is_same<T, double>::value ||
        std::is_same<T, HDRFloat<float>>::value ||
        std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Ex == Extras::IncludeLAv2) {
            static_assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::EnableCompression, "!");
            GenLAResults(results, options_to_use);
        }
        else {
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
        auto* resultsExisting = GetUsefulPerturbationResults<IterType, ConvertTType, false, PExtras>();


        // The second part of this accounts for the case where the
        // malicious user deleted the LA reference.  We want to
        // recompute it if it's not present as a file.
        if ((resultsExisting == nullptr) ||
            (resultsExisting != nullptr &&
                resultsExisting->GetLaReference() == nullptr &&
                Ex == Extras::IncludeLAv2)) {

            // This save is for the 64-bit calculations.
            OptionalSave(results);

            // Now generate the 2x32 results:
            auto results2(std::make_unique<PerturbationResults<IterType, ConvertTType, PExtras>>(
                options_to_use, GetNextGenerationNumber()));
            results2->CopyPerturbationResults<true>(*results);

            auto ret = AddPerturbationResults(std::move(results2));
            m_LastUsedRefOrbit = ret;

            // Save those.
            OptionalSave(ret);
            return ret;
        }
        else {
            m_LastUsedRefOrbit = resultsExisting;
            OptionalSave(resultsExisting);
            return resultsExisting;
        }
    }
    else {
        m_LastUsedRefOrbit = results;
        OptionalSave(results);
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
        if (cur_array.size() > MaxStoredOrbits) {
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
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        container.m_PerturbationResultsFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsFloat[container.m_PerturbationResultsFloat.size() - 1].get();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, float>::value) {
        return nullptr;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, HDRFloat<float>, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        container.m_PerturbationResultsHDRFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsHDRFloat[container.m_PerturbationResultsHDRFloat.size() - 1].get();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
        auto newarray = std::make_unique<PerturbationResults<IterType, float, DestEnableBad>>(
            m_RefOrbitOptions, GetNextGenerationNumber());
        container.m_PerturbationResultsFloat.push_back(std::move(newarray));
        auto* dest = container.m_PerturbationResultsFloat[container.m_PerturbationResultsFloat.size() - 1].get();
        dest->CopyPerturbationResults<false>(src_array);
        return dest;
    }
    else {
        return nullptr;
    }
}

void RefOrbitCalc::ClearPerturbationResults(PerturbationResultType type) {
    auto IsMarkedToDelete = [&](const auto& val) -> bool {
        // Erase results as needed.
        // Note: erase full results in dbl-float case -- we'll reconvert
        // the whole thing if needed from double/Hdrfloat<double> etc.
        if (type == PerturbationResultType::All ||
            (type == PerturbationResultType::MediumRes &&
                val->GetAuthoritativePrecisionInBits() == 0) ||
            (type == PerturbationResultType::HighRes &&
                val->GetAuthoritativePrecisionInBits() != 0) ||
            (type == PerturbationResultType::LAOnly &&
                val->Is2X32)) {
            return true;
        }

        return false;
    };

    auto ClearLAIfNeeded = [&](const auto& o) {
        if (type == PerturbationResultType::LAOnly) {
            o->ClearLaReference();
        }
    };

    auto ClearOne = [&](auto &arr) {
        // First, erase some results completely as needed
        arr.erase(
            std::remove_if(arr.begin(), arr.end(), IsMarkedToDelete),
            arr.end());

        // Now just erase LA results of what's left, if needed
        std::for_each(arr.begin(), arr.end(), ClearLAIfNeeded);
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
        for (const auto& it : Container.m_PerturbationResultsDouble) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResultsFloat) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResults2xFloat) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResultsHDRDouble) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResultsHDRFloat) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResults2xFloat) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
        }

        for (const auto& it : Container.m_PerturbationResultsHDR2xFloat) {
            auto resultsCopy = it->CopyPerturbationResults(
                AddPointOptions::EnableWithSave,
                GetNextGenerationNumber());
            resultsCopy->WriteMetadata();
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
                auto next_gen = GetNextGenerationNumber();
                auto stripfn = stripext(file);
                auto widefn = narrowtowide(stripfn);
                auto results1 = std::make_unique<PerturbationResults<IterType, double, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results1->ReadMetadata()) {
                    Container.m_PerturbationResultsDouble.push_back(std::move(results1));
                    continue;
                }
                results1 = nullptr;

                auto results2 = std::make_unique<PerturbationResults<IterType, float, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results2->ReadMetadata()) {
                    Container.m_PerturbationResultsFloat.push_back(std::move(results2));
                    continue;
                }
                results2 = nullptr;

                auto results3 = std::make_unique<PerturbationResults<IterType, CudaDblflt<MattDblflt>, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results3->ReadMetadata()) {
                    Container.m_PerturbationResults2xFloat.push_back(std::move(results3));
                    continue;
                }
                results3 = nullptr;

                auto results4 = std::make_unique<PerturbationResults<IterType, HDRFloat<double>, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results4->ReadMetadata()) {
                    Container.m_PerturbationResultsHDRDouble.push_back(std::move(results4));
                    continue;
                }
                results4 = nullptr;

                auto results5 = std::make_unique<PerturbationResults<IterType, HDRFloat<float>, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results5->ReadMetadata()) {
                    Container.m_PerturbationResultsHDRFloat.push_back(std::move(results5));
                    continue;
                }
                results5 = nullptr;

                auto results6 = std::make_unique<PerturbationResults<IterType, HDRFloat<CudaDblflt<MattDblflt>>, PExtras>>(
                    widefn,
                    AddPointOptions::OpenExistingWithSave,
                    next_gen);
                if (results6->ReadMetadata()) {
                    Container.m_PerturbationResultsHDR2xFloat.push_back(std::move(results6));
                    continue;
                }
                results6 = nullptr;
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

void RefOrbitCalc::SetPerturbationAlg(PerturbationAlg alg) {
    m_PerturbationAlg = alg;
}

RefOrbitCalc::PerturbationAlg RefOrbitCalc::GetPerturbationAlg() const {
    if (m_PerturbationAlg == PerturbationAlg::Auto) {
        HighPrecision zoomFactor = m_Fractal.GetZoomFactor();
        if (zoomFactor < HighPrecision{ 1e100 }) {
            return PerturbationAlg::STPeriodicity;
        }
        else {
            return PerturbationAlg::MTPeriodicity3;
        }
    }
    else {
        return m_PerturbationAlg;
    }
}

std::string RefOrbitCalc::GetPerturbationAlgStr() const {
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
    case PerturbationAlg::MTPeriodicity5:
        return "MTPeriodicity5";
    default:
        return "Unknown";
    }
}

void RefOrbitCalc::GetSomeDetails(
    uint64_t& InternalPeriodMaybeZero,
    uint64_t& CompressedIters,
    uint64_t& UncompressedIters,
    int32_t &CompressionErrorExp,
    uint64_t &OrbitMilliseconds,
    uint64_t &LAMilliseconds,
    uint64_t &LASize,
    std::string& PerturbationAlg) {

    InternalPeriodMaybeZero = 0;
    CompressedIters = 0;
    UncompressedIters = 0;
    CompressionErrorExp = 0;
    OrbitMilliseconds = 0;
    LAMilliseconds = 0;
    LASize = 0;

    auto lambda = [&](auto &&arg) {
        if (arg != nullptr) {
            InternalPeriodMaybeZero = arg->GetPeriodMaybeZero();
            CompressedIters = arg->GetCompressedOrbitSize();
            UncompressedIters = arg->GetCountOrbitEntries();
            CompressionErrorExp = arg->GetCompressionErrorExp();
            OrbitMilliseconds = arg->GetBenchmarkOrbit();

            if (arg->GetLaReference() != nullptr) {
                LAMilliseconds = arg->GetLaReference()->GetBenchmarkLA().GetDeltaInMs();
                LASize = arg->GetLaReference()->GetLAs().GetSize();
            }
        }
    };

    std::visit(lambda, m_LastUsedRefOrbit);

    PerturbationAlg = GetPerturbationAlgStr();

    static_assert(static_cast<int>(RenderAlgorithm::MAX) == 61, "Fix me");
}

void RefOrbitCalc::SaveOrbitAsText() const {
    auto lambda = [&](auto&& arg) {
        if (arg != nullptr) {
            arg->SaveOrbitAsText();
        }
    };

    std::visit(lambda, m_LastUsedRefOrbit);
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

template<RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::InitAllocatorsIfNeeded(
    std::unique_ptr<MPIRBoundedAllocator>& boundedAllocator,
    std::unique_ptr<MPIRBumpAllocator>& bumpAllocator) {
    if constexpr (
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3) {

        boundedAllocator = std::make_unique<MPIRBoundedAllocator>();
        boundedAllocator->InitScopedAllocators();
        boundedAllocator->InitTls();
    }
    else {
        bumpAllocator = std::make_unique<MPIRBumpAllocator>();
        bumpAllocator->InitScopedAllocators();
        bumpAllocator->InitTls();
    }
}

template<RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::ShutdownAllocatorsIfNeeded(
    std::unique_ptr<MPIRBoundedAllocator>& boundedAllocator,
    std::unique_ptr<MPIRBumpAllocator>& bumpAllocator) {
    if constexpr (
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse1 &&
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse2 &&
        Reuse != RefOrbitCalc::ReuseMode::SaveForReuse3) {
        boundedAllocator->ShutdownTls();
    }
    else {
        bumpAllocator->ShutdownTls();
    }
}

RefOrbitCalc::ScopedAffinity::ScopedAffinity(
    RefOrbitCalc &refOrbitCalc,
    HANDLE thread1,
    HANDLE thread2,
    HANDLE thread3,
    HANDLE thread4) :
    m_RefOrbitCalc(refOrbitCalc),
    m_Thread1(thread1),
    m_Thread2(thread2),
    m_Thread3(thread3),
    m_Thread4(thread4) {

    //SetCpuAffinityAsNeeded();
}

RefOrbitCalc::ScopedAffinity::~ScopedAffinity() {
    // Note, ignore threads 2/3/4 -- they're temporaries in RefOrbitCalc.
    SetThreadAffinityMask(m_Thread1, 0xFFFFFFFF);

    // Reset to default priority:
    //SetThreadPriority(m_Thread1, THREAD_PRIORITY_NORMAL);
}

void RefOrbitCalc::ScopedAffinity::SetCpuAffinityAsNeeded() {

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
    }
    else {
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

    //SetThreadPriority(m_Thread1, THREAD_PRIORITY_ABOVE_NORMAL);
    //SetThreadPriority(m_Thread2, THREAD_PRIORITY_ABOVE_NORMAL);
    //SetThreadPriority(m_Thread3, THREAD_PRIORITY_ABOVE_NORMAL);
    //SetThreadPriority(m_Thread4, THREAD_PRIORITY_ABOVE_NORMAL);
}


#include "RefOrbitCalcTemplates.h"

