#include "stdafx.h"

#include "RefOrbitCalc.h"
#include "Fractal.h"
#include "PerturbationResults.h"

#include <vector>
#include <memory>
#include <math.h>

#include <psapi.h>

struct scoped_mpfr_precision
{
    unsigned saved_digits10;
    scoped_mpfr_precision(unsigned digits10) : saved_digits10(HighPrecision::thread_default_precision())
    {
        HighPrecision::default_precision(digits10);
    }
    ~scoped_mpfr_precision()
    {
        HighPrecision::default_precision(saved_digits10);
    }
    void reset(unsigned digits10)
    {
        HighPrecision::default_precision(digits10);
    }
    void reset()
    {
        HighPrecision::default_precision(saved_digits10);
    }
};

struct scoped_mpfr_precision_options
{
    boost::multiprecision::variable_precision_options saved_options;
    scoped_mpfr_precision_options(boost::multiprecision::variable_precision_options opts) : saved_options(HighPrecision::thread_default_variable_precision_options())
    {
        HighPrecision::thread_default_variable_precision_options(opts);
    }
    ~scoped_mpfr_precision_options()
    {
        HighPrecision::thread_default_variable_precision_options(saved_options);
    }
    void reset(boost::multiprecision::variable_precision_options opts)
    {
        HighPrecision::thread_default_variable_precision_options(opts);
    }
};



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
      m_GuessReserveSize() {
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
    case RenderAlgorithm::GpuHDRx32Perturbed:
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2:
        return check == &m_PerturbationResultsHDRFloat;
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::Cpu64PerturbedBLAV2HDR:
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2:
        return check == &m_PerturbationResultsHDRDouble;
    case RenderAlgorithm::Gpu1x32Perturbed:
    case RenderAlgorithm::Gpu1x32PerturbedPeriodic:
        return check == &m_PerturbationResultsFloat;
    case RenderAlgorithm::Cpu64PerturbedBLA:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
    case RenderAlgorithm::Gpu1x64Perturbed:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
        return check == &m_PerturbationResultsDouble;
    case RenderAlgorithm::Gpu2x32Perturbed:
        // TODO
        //CalcGpuPerturbationFractalBLA<dblflt, dblflt>(MemoryOnly);
        assert(false);
        return false;
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        // TODO
        //CalcGpuPerturbationFractalBLA<double, double>(MemoryOnly);
        assert(false);
        return false;
    default:
        return false;
    }
}

void RefOrbitCalc::OptimizeMemory() {
    PROCESS_MEMORY_COUNTERS_EX checkHappy;
    const size_t OMGAlotOfMemory = 128llu * 1024llu * 1024llu * 1024llu;
    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));

    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsDouble)) {
            m_PerturbationResultsDouble.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsFloat)) {
            m_PerturbationResultsFloat.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsHDRDouble)) {
            m_PerturbationResultsHDRDouble.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsHDRFloat)) {
            m_PerturbationResultsHDRFloat.clear();
        }
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
        ::MessageBox(NULL, L"Watch the memory use... this is just a warning", L"", MB_OK);
        assert(false);
    }
}

template<class T, CalcBad Bad>
std::vector<std::unique_ptr<PerturbationResults<T, Bad>>> &
RefOrbitCalc::GetPerturbationResults() {
    if constexpr (std::is_same<T, double>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return m_PerturbationResultsDoubleB;
        }
        else {
            return m_PerturbationResultsDouble;
        }
    }
    else if constexpr (std::is_same<T, float>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return m_PerturbationResultsFloatB;
        }
        else {
            return m_PerturbationResultsFloat;
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return m_PerturbationResultsHDRDoubleB;
        }
        else {
            return m_PerturbationResultsHDRDouble;
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return m_PerturbationResultsHDRFloatB;
            }
        else {
            return m_PerturbationResultsHDRFloat;
        }
    }
}

template<class T, class SubType, RefOrbitCalc::BenchmarkMode BenchmarkState>
void RefOrbitCalc::AddPerturbationReferencePoint() {
    if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
        m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / HighPrecision(2);
        m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / HighPrecision(2);
    }

    if (RequiresBadCalc()) {
        if (m_PerturbationAlg == PerturbationAlg::ST) {
            AddPerturbationReferencePointST<T, SubType, false, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MT) {
            AddPerturbationReferencePointMT3<T, SubType, false, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
            AddPerturbationReferencePointST<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3 ||
                 m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed ||
                 m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed) {
            // Note: this path we don't bother saving/reusing.  Useless for low zoom depths.
            AddPerturbationReferencePointMT3<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity5) {
            AddPerturbationReferencePointMT5<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
    }
    else {
        if (m_PerturbationAlg == PerturbationAlg::ST) {
            AddPerturbationReferencePointST<T, SubType, false, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MT) {
            AddPerturbationReferencePointMT3<T, SubType, false, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
            AddPerturbationReferencePointST<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3) {
            AddPerturbationReferencePointMT3<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed ||
                 m_PerturbationAlg == PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed) {
            AddPerturbationReferencePointMT3<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::SaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity5) {
            AddPerturbationReferencePointMT5<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
    }
}

template void RefOrbitCalc::AddPerturbationReferencePoint<float, float, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<double, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Disable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Disable>();

template void RefOrbitCalc::AddPerturbationReferencePoint<float, float, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<double, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<HDRFloat<double>, double, RefOrbitCalc::BenchmarkMode::Enable>();
template void RefOrbitCalc::AddPerturbationReferencePoint<HDRFloat<float>, float, RefOrbitCalc::BenchmarkMode::Enable>();

template<class T>
static void AddReused(T &results, const HighPrecision& zx, const HighPrecision& zy) {
    HighPrecision ReducedZx;
    HighPrecision ReducedZy;

    ReducedZx = zx;
    ReducedZy = zy;

    //assert(RequiresReuse());
    ReducedZx.precision(AuthoritativeReuseExtraPrecision);
    ReducedZy.precision(AuthoritativeReuseExtraPrecision);

    results.ReuseX.push_back(ReducedZx);
    results.ReuseY.push_back(ReducedZy);
}

template<class T, class PerturbationResultsType, CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::InitResults(PerturbationResultsType& results, const HighPrecision& initX, const HighPrecision& initY) {
    results.InitResults<T, Bad, Reuse>(
        initX,
        initY,
        m_Fractal.GetMinX(),
        m_Fractal.GetMinY(),
        m_Fractal.GetMaxX(),
        m_Fractal.GetMaxY(),
        m_Fractal.GetNumIterations(),
        m_GuessReserveSize);
}

template<
    class T,
    class SubType,
    bool Periodicity,
    RefOrbitCalc::BenchmarkMode BenchmarkState,
    CalcBad Bad,
    RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointST(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<T, Bad>();

    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T, Bad>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    InitResults<T, decltype(*results), Bad, Reuse>(*results, cx, cy);

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    IterType i;

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    zx = cx;
    zy = cy;
    for (i = 0; i < m_Fractal.GetNumIterations(); i++)
    {
        zx2 = zx * 2;
        zy2 = zy * 2;

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        results->orb.push_back({ double_zx, double_zy });

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(*results, zx, zy);
        }

        if constexpr (Bad == CalcBad::Enable) {
            const T sq_x = double_zx * double_zx;
            const T sq_y = double_zy * double_zy;
            const T norm = HdrReduce((sq_x + sq_y) * glitch);

            const auto zx_reduced = HdrReduce(HdrAbs((T)zx));
            const auto zy_reduced = HdrReduce(HdrAbs((T)zy));
            const bool underflow =
                (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(norm, small_float));
            results->orb[results->orb.size() - 1].bad = underflow;
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
            auto n3 = results->maxRadius * r0 * HighTwo;
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    results->PeriodMaybeZero = (IterType)results->orb.size();
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

    if constexpr (Bad == CalcBad::Enable) {
        results->orb[results->orb.size() - 1].bad = false;
    }

    results->TrimResults<Bad, Reuse>();
    m_GuessReserveSize = results->orb.size();
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState>
bool RefOrbitCalc::AddPerturbationReferencePointSTReuse(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<T, CalcBad::Disable>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T, CalcBad::Disable>>());

    auto* existingResults = GetUsefulPerturbationResults<T, SubType, true, CalcBad::Disable>();
    if (existingResults == nullptr || existingResults->ReuseX.size() < 5) {
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
    if (NewPrec - existingResults->AuthoritativePrecision >= AuthoritativeReuseExtraPrecision - AuthoritativeMinExtraPrecision) {
        //::MessageBox(NULL, L"Regenerating authoritative orbit is required", L"", MB_OK);
        PerturbationResultsArray.pop_back();
        return false;
    }

    // TODO seems like we should be able to avoid all these annoying .precision calls below
    // via this mechanism.  Read the awful boost docs more...
    scoped_mpfr_precision prec(AuthoritativeReuseExtraPrecision);

    InitResults<T, decltype(*results), CalcBad::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);

    HighPrecision zx, zy;
    IterType i;

    HighPrecision HighOne = 1.0;
    HighPrecision HighTwo = 2.0;
    static const T TwoFiftySix = T(256.0);
    HighPrecision DeltaReal = cx - existingResults->hiX;
    HighPrecision DeltaImaginary = cy - existingResults->hiY;
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

    IterType RefIteration = 0;
    IterType MaxRefIteration = existingResults->orb.size() - 1;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

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

    const std::vector<HighPrecision> &existingReuseX = existingResults->ReuseX;
    const std::vector<HighPrecision> &existingReuseY = existingResults->ReuseY;
    for (i = 0; i < m_Fractal.GetNumIterations(); i++) {
        if constexpr (Periodicity) {
            zxCopy = T(zx);
            zyCopy = T(zy);
        }

        const HighPrecision DeltaSubNXOrig = DeltaSubNX;
        const HighPrecision DeltaSubNYOrig = DeltaSubNY;

        DeltaSubNX =
            DeltaSubNXOrig * (existingReuseX[RefIteration] * HighTwo + DeltaSubNXOrig) -
            DeltaSubNYOrig * (existingReuseY[RefIteration] * HighTwo + DeltaSubNYOrig) +
            DeltaSub0X;
        tempDeltaSubNXLow = (T)DeltaSubNX;

        DeltaSubNY =
            DeltaSubNXOrig * (existingReuseY[RefIteration] * HighTwo + DeltaSubNYOrig) +
            DeltaSubNYOrig * (existingReuseX[RefIteration] * HighTwo + DeltaSubNXOrig) +
            DeltaSub0Y;
        tempDeltaSubNYLow = (T)DeltaSubNY;

        ++RefIteration;

        tempZXLow = (T)zx;
        tempZYLow = (T)zy;

        zx = existingReuseX[RefIteration] + DeltaSubNX;
        zy = existingReuseY[RefIteration] + DeltaSubNY;
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
            auto n3 = results->maxRadius * r0 * T(2.0);
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    // Break before adding the result.
                    results->PeriodMaybeZero = (IterType)results->orb.size();
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

        results->orb.push_back({ reducedZx, reducedZy });
    }

    results->TrimResults<CalcBad::Disable, ReuseMode::DontSaveForReuse>();
    m_GuessReserveSize = results->orb.size();

    return true;
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState>
bool RefOrbitCalc::AddPerturbationReferencePointMT3Reuse(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<T, CalcBad::Disable>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T, CalcBad::Disable>>());

    auto* existingResults = GetUsefulPerturbationResults<T, SubType, true, CalcBad::Disable>();
    if (existingResults == nullptr || existingResults->ReuseX.size() < 5) {
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
    if (NewPrec - existingResults->AuthoritativePrecision >= AuthoritativeReuseExtraPrecision - AuthoritativeMinExtraPrecision) {
        //::MessageBox(NULL, L"Regenerating authoritative orbit is required", L"", MB_OK);
        PerturbationResultsArray.pop_back();
        return false;
    }

    // TODO seems like we should be able to avoid all these annoying .precision calls below
    // via this mechanism.  Read the awful boost docs more...
    scoped_mpfr_precision prec(AuthoritativeReuseExtraPrecision);
    scoped_mpfr_precision_options precOptions(boost::multiprecision::variable_precision_options::assume_uniform_precision);

    InitResults<T, decltype(*results), CalcBad::Disable, ReuseMode::DontSaveForReuse>(*results, cx, cy);
    
    HighPrecision zx, zy;
    IterType i;

    HighPrecision HighOne = 1.0;
    HighPrecision HighTwo = 2.0;
    static const T TwoFiftySix = T(256.0);
    HighPrecision DeltaReal = cx - existingResults->hiX;
    HighPrecision DeltaImaginary = cy - existingResults->hiY;
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

    IterType RefIteration = 0;
    IterType MaxRefIteration = existingResults->orb.size() - 1;

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

        IterType ReferenceIteration;
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

        IterType ReferenceIteration;
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

    const std::vector<HighPrecision>& existingReuseX = existingResults->ReuseX;
    const std::vector<HighPrecision>& existingReuseY = existingResults->ReuseY;

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
                (*ok->DeltaSubNXOrig) * (existingReuseX[ok->ReferenceIteration] * HighTwo + (*ok->DeltaSubNXOrig)) -
                (*ok->DeltaSubNYOrig) * (existingReuseY[ok->ReferenceIteration] * HighTwo + (*ok->DeltaSubNYOrig)) +
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
                (*ok->DeltaSubNXOrig) * (existingReuseY[ok->ReferenceIteration] * HighTwo + (*ok->DeltaSubNYOrig)) +
                (*ok->DeltaSubNYOrig) * (existingReuseX[ok->ReferenceIteration] * HighTwo + (*ok->DeltaSubNXOrig)) +
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

    for (i = 0; i < m_Fractal.GetNumIterations(); i++) {
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
            results->orb.push_back({ tempZXLow, tempZYLow });
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
            auto n3 = results->maxRadius * r0 * T(2.0);
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    // Break before adding the result.
                    results->PeriodMaybeZero = (IterType)results->orb.size();
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
                zy = existingReuseY[RefIteration] + threadZydata->DeltaSubNY;
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
                zx = existingReuseX[RefIteration] + threadZxdata->DeltaSubNX;
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

    results->TrimResults<CalcBad::Disable, ReuseMode::DontSaveForReuse>();
    m_GuessReserveSize = results->orb.size();

    return true;
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT3(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<T, Bad>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T, Bad>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision zx, zy;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    InitResults<T, decltype(*results), Bad, Reuse>(*results, cx, cy);

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

    for (IterType i = 0; i < m_Fractal.GetNumIterations(); i++)
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

        results->orb.push_back({double_zx, double_zy });

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(*results, zx, zy);  // TODO
        }

        if constexpr (Bad == CalcBad::Enable) {
            const T norm = HdrReduce((double_zx * double_zx + double_zy * double_zy) * glitch);
            const auto zx_reduced = HdrReduce(HdrAbs((T)double_zx));
            const auto zy_reduced = HdrReduce(HdrAbs((T)double_zy));

            const bool underflow =
                (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(norm, small_float));
            results->orb[results->orb.size() - 1].bad = underflow;
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
            auto n3 = results->maxRadius * r0 * HighTwo;
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
                        results->PeriodMaybeZero = (IterType)results->orb.size();
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

    if constexpr (Bad == CalcBad::Enable) {
        results->orb[results->orb.size() - 1].bad = false;
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

    results->TrimResults<Bad, Reuse>();
    m_GuessReserveSize = results->orb.size();
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT5(HighPrecision cx, HighPrecision cy) {
    auto& PerturbationResultsArray = GetPerturbationResults<T, Bad>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T, Bad>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    HighPrecision zx_sq, zy_sq;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    InitResults<T, decltype(*results), Bad, Reuse>(*results, cx, cy);

    const T small_float = T((SubType)1.1754944e-38);
    // Note: results->bad is not here.  See end of this function.
    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        HighPrecision zx;
        HighPrecision* zx_sq;
    };

    struct ThreadZyData {
        HighPrecision zy;
        HighPrecision* zy_sq;
    };

    struct Thread1Data {
        HighPrecision* zx_sq;
        HighPrecision* zy_sq;
        HighPrecision* zx2;
        HighPrecision* zy2;
        HighPrecision* zx;
        HighPrecision zy;
        HighPrecision* cx;
    };

    struct Thread2Data {
        HighPrecision zx2;
        HighPrecision* zy2;
        HighPrecision zy;
        HighPrecision* cy;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto* Thread1Memory = (ThreadPtrs<Thread1Data> *)
        _aligned_malloc(sizeof(ThreadPtrs<Thread1Data>), 64);
    memset(Thread1Memory, 0, sizeof(*Thread1Memory));

    auto* Thread2Memory = (ThreadPtrs<Thread2Data>*)
        _aligned_malloc(sizeof(ThreadPtrs<Thread2Data>), 64);
    memset(Thread2Memory, 0, sizeof(*Thread2Memory));

    auto ThreadSqZx = [](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zx);

            *ok->zx_sq = ok->zx * ok->zx;

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

            *ok->zy_sq = ok->zy * ok->zy;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto Thread1 = [](ThreadPtrs<Thread1Data>* ThreadMemory,
        ThreadZxData* threadZxdata,
        ThreadZyData* threadZydata,
        ThreadPtrs<ThreadZxData>* ThreadZxMemory,
        ThreadPtrs<ThreadZyData>* ThreadZyMemory) {
            HighPrecision temp3;

            ThreadZxData* expectedZx = nullptr;
            ThreadZyData* expectedZy = nullptr;

            for (;;) {
                Thread1Data* expected = ThreadMemory->In.load();
                Thread1Data* ok = nullptr;

                CheckStartCriteria;

                PrefetchHighPrec(*ok->cx);
                PrefetchHighPrec(*ok->zx);

                // Wait for squaring
                bool zxOk = false;
                bool zyOk = false;
                for (;;) {
                    expectedZx = threadZxdata;

                    if (!zxOk && ThreadZxMemory->Out.compare_exchange_weak(expectedZx, nullptr, std::memory_order_relaxed)) {
                        zxOk = true;

                        std::atomic_thread_fence(std::memory_order_release);
                        PrefetchHighPrec(*ok->zx_sq);
                    }

                    expectedZy = threadZydata;

                    if (!zyOk && ThreadZyMemory->Out.compare_exchange_weak(expectedZy, nullptr, std::memory_order_relaxed)) {
                        zyOk = true;
                        std::atomic_thread_fence(std::memory_order_release);
                        PrefetchHighPrec(*ok->zy_sq);
                    }

                    if (zxOk && zyOk) {
                        break;
                    }
                }

                temp3 = *ok->zx_sq - *ok->zy_sq;
                *ok->zx = temp3 + *ok->cx;
                *ok->zx2 = *ok->zx * 2;

                // Give result back.
                CheckFinishCriteria;
            }
    };

    auto Thread2 = [](ThreadPtrs<Thread2Data>* ThreadMemory) {
        HighPrecision temp1;
        for (;;) {
            Thread2Data* expected = ThreadMemory->In.load();
            Thread2Data* ok = nullptr;

            CheckStartCriteria;

            // _mm_prefetch((const char*)ok->zx_sq, _MM_HINT_T0);
            // _mm_prefetch((const char*)&ok->zx_sq->backend().data(), _MM_HINT_T0);
            PrefetchHighPrec(*ok->cy);
            PrefetchHighPrec(ok->zx2);
            PrefetchHighPrec(ok->zy);

            temp1 = ok->zx2 * ok->zy;
            ok->zy = temp1 + *ok->cy;
            *ok->zy2 = ok->zy * 2;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);
    auto* thread1data = (Thread1Data*)_aligned_malloc(sizeof(Thread1Data), 64);
    auto* thread2data = (Thread2Data*)_aligned_malloc(sizeof(Thread2Data), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};
    new (thread1data) (Thread1Data){};
    new (thread2data) (Thread2Data){};

    threadZxdata->zx_sq = &zx_sq;

    threadZydata->zy_sq = &zy_sq;

    thread1data->zx2 = &zx2;
    thread1data->zy2 = &zy2;
    thread1data->zx_sq = &zx_sq;
    thread1data->zy_sq = &zy_sq;
    thread1data->zx = &zx;
    thread1data->cx = &cx;

    thread2data->zy2 = &zy2;
    thread2data->cy = &cy;

    // Five threads + use rest for HDRFloat
    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));
    std::unique_ptr<std::thread> t1(new std::thread(Thread1, Thread1Memory, threadZxdata, threadZydata, ThreadZxMemory, ThreadZyMemory));
    std::unique_ptr<std::thread> t2(new std::thread(Thread2, Thread2Memory));

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);
    SetThreadAffinityMask(t1->native_handle(), 0x1 << 9);
    SetThreadAffinityMask(t2->native_handle(), 0x1 << 11);

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;
    Thread1Data* expected1 = nullptr;
    Thread2Data* expected2 = nullptr;

    bool done1 = false;
    bool done2 = false;

    zx = cx;
    zy = cy;
    zx2 = zx * 2;
    zy2 = zy * 2;

    thread2data->zy = zy;

    T zxCopy;
    T zyCopy;
    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    for (IterType i = 0; i < m_Fractal.GetNumIterations(); i++)
    {
        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };
        }

        // Start Thread 2: zy = 2 * zx * zy + cy;
        thread2data->zx2 = zx2;

        Thread2Memory->In.store(
            thread2data,
            std::memory_order_release);

        // Start Zx squaring thread
        threadZxdata->zx = zx;

        ThreadZxMemory->In.store(
            threadZxdata,
            std::memory_order_release);

        // Start Zy squaring thread
        threadZydata->zy = zy;

        ThreadZyMemory->In.store(
            threadZydata,
            std::memory_order_release);

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(*results, zx, zy);
        }

        // Start Thread 1: zx = zx * zx - zy * zy + cx;
        thread1data->zy = zy;

        Thread1Memory->In.store(
            thread1data,
            std::memory_order_release);

        results->orb.push_back({ double_zx, double_zy });

        if constexpr (Bad == CalcBad::Enable) {
            //T norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
            //bool underflow = (HdrAbs(double_zx) <= small_float ||
            //    HdrAbs(double_zy) <= small_float ||
            //    norm <= small_float);

            const T norm = HdrReduce((double_zx * double_zx + double_zy * double_zy) * glitch);
            const auto zx_reduced = HdrReduce(HdrAbs((T)double_zx));
            const auto zy_reduced = HdrReduce(HdrAbs((T)double_zy));

            const bool underflow =
                (HdrCompareToBothPositiveReducedLE(zx_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(zy_reduced, small_float) ||
                 HdrCompareToBothPositiveReducedLE(norm, small_float));

            results->orb[results->orb.size() - 1].bad = underflow;
        }

        // Note: not T.
        const SubType tempZX = (SubType)double_zx + (SubType)cx;
        const SubType tempZY = (SubType)double_zy + (SubType)cy;
        const SubType zn_size = tempZX * tempZX + tempZY * tempZY;

        if constexpr (Periodicity) {
            //function p  = findPeriodM3(c0,dx,dy,n,doCont,mpow)
            //% in ball centered on c0 find period (up to n) of nucleus
            //% use 1nd order Taylor ball
            //% M-power mpow set
            //% doCont = 0 normally
            //
            //r0 = min(abs(dx),abs(dy));
            //z = c0*0;
            //r = (r0);
            //p = [];
            //maxR = 1e5;
            //az = abs(z);
            //
            //for k=1:n
            //    r = (az+r).^mpow - az.^mpow + r0;
            //    z = z.^mpow + c0;
            //    az = abs(z);
            //    if(r>az)
            //        p = [p k];
            //        fprintf('findPeriodBallM3: N-period found: %d\n',k);
            //        if(~doCont)
            //            break;
            //        end
            //    end
            //    if(az>maxR | r>maxR)
            //        fprintf('Ball: escaping\n',k);
            //        break;
            //    end
            //end

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
            auto n3 = results->maxRadius * r0 * HighTwo;
            HdrReduce(n3);

            if (HdrCompareToBothPositiveReducedLT(n2, n3)) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    periodicity_should_break = true;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        done1 = false;
        done2 = false;

        for (;;) {
            expected1 = thread1data;

            if (!done1 &&
                Thread1Memory->Out.compare_exchange_weak(expected1,
                    nullptr,
                    std::memory_order_release)) {
                done1 = true;
            }

            expected2 = thread2data;

            if (!done2 &&
                Thread2Memory->Out.compare_exchange_weak(expected2,
                    nullptr,
                    std::memory_order_release)) {
                done2 = true;
            }

            if (done1 && done2) {
                break;
            }
        }

        zy = thread2data->zy;

        if (zn_size > 256) {
            break;
        }

        if constexpr (Periodicity) {
            if (periodicity_should_break) {
                results->PeriodMaybeZero = (IterType)results->orb.size();
                break;
            }
        }
    }

    if constexpr (Bad == CalcBad::Enable) {
        results->orb[results->orb.size() - 1].bad = false;
    }

    bool resZx = false, resZy = false;
    bool res1 = false, res2 = false;
    while (!resZx) {
        expectedZx = nullptr;
        resZx = ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);
    }

    while (!resZy) {
        expectedZy = nullptr;
        resZy = ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);
    }

    while (!res1) {
        expected1 = nullptr;
        res1 = Thread1Memory->In.compare_exchange_strong(expected1, (Thread1Data*)0x1, std::memory_order_release);
    }

    while (!res2) {
        expected2 = nullptr;
        res2 = Thread2Memory->In.compare_exchange_strong(expected2, (Thread2Data*)0x1, std::memory_order_release);
    }

    tZx->join();
    tZy->join();
    t1->join();
    t2->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);
    _aligned_free(Thread1Memory);
    _aligned_free(Thread2Memory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();
    thread1data->~Thread1Data();
    thread2data->~Thread2Data();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);
    _aligned_free(thread1data);
    _aligned_free(thread2data);

    results->TrimResults<Bad, Reuse>();
    m_GuessReserveSize = results->orb.size();
}


bool RefOrbitCalc::RequiresReferencePoints() const {
    switch (m_Fractal.GetRenderAlgorithm()) {
    case RenderAlgorithm::Cpu64PerturbedBLA:
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::Gpu1x32Perturbed:
    case RenderAlgorithm::Gpu1x32PerturbedPeriodic:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32Perturbed:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
    case RenderAlgorithm::Gpu1x64Perturbed:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
    case RenderAlgorithm::Gpu2x32Perturbed:
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2:
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2:
        return true;
    }

    return false;
}

template<class T, bool Authoritative, CalcBad Bad>
bool RefOrbitCalc::IsPerturbationResultUsefulHere(size_t i) const {

    auto lambda = [&]<typename T, bool Authoritative, CalcBad Bad>(
        const PerturbationResults<T, Bad> &PerturbationResults
        ) -> bool {

        if constexpr (Authoritative == true) {
            return PerturbationResults.AuthoritativePrecision != 0 &&
                (PerturbationResults.MaxIterations > PerturbationResults.orb.size() ||
                    PerturbationResults.MaxIterations >= m_Fractal.GetNumIterations());
        }

        return
            PerturbationResults.hiX >= m_Fractal.GetMinX() &&
            PerturbationResults.hiX <= m_Fractal.GetMaxX() &&
            PerturbationResults.hiY >= m_Fractal.GetMinY() &&
            PerturbationResults.hiY <= m_Fractal.GetMaxY() &&
            (PerturbationResults.MaxIterations > PerturbationResults.orb.size() ||
                PerturbationResults.MaxIterations >= m_Fractal.GetNumIterations());
    };

    if constexpr (std::is_same<T, double>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsDoubleB[i]);
        }
        else {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsDouble[i]);
        }
    }
    else if constexpr (std::is_same<T, float>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsFloatB[i]);
        }
        else {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsFloat[i]);
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsHDRDoubleB[i]);
        }
        else {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsHDRDouble[i]);
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsHDRFloatB[i]);
        }
        else {
            return lambda.template operator() <T, Authoritative, Bad>(*m_PerturbationResultsHDRFloat[i]);
        }
    }
}

template<class T, class SubType, CalcBad Bad, RefOrbitCalc::Extras Ex>
PerturbationResults<T, Bad>* RefOrbitCalc::GetAndCreateUsefulPerturbationResults() {
    bool added = false;
    if (RequiresReuse()) {
        if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
            m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / 2;
            m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / 2;
        }

        PerturbationResults<T, Bad>* results = GetUsefulPerturbationResults<T, SubType, false, Bad>();
        if (results == nullptr) {
            switch (GetPerturbationAlg()) {
            case PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed:
                added = AddPerturbationReferencePointSTReuse<T, SubType, true, BenchmarkMode::Disable>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            case PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed:
                added = AddPerturbationReferencePointMT3Reuse<T, SubType, true, BenchmarkMode::Disable>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
                break;
            default:
                ::MessageBox(NULL, L"Some stupid bug #2343 :(", L"", MB_OK);
                assert(false);
                break;
            }
        }
    }

    PerturbationResults<T, Bad>* results = GetUsefulPerturbationResults<T, SubType, false, Bad>();
    if (results == nullptr) {
        if (added) {
            ::MessageBox(NULL, L"Why didn't this work! :(", L"", MB_OK);
        }
        std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>* cur_array = GetPerturbationArray<T, Bad>();
        AddPerturbationReferencePoint<T, SubType, BenchmarkMode::Disable>();
        added = true;

        results = (*cur_array)[cur_array->size() - 1].get();
    }

    if constexpr (std::is_same<T, HDRFloat<float>>::value ||
                  std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Ex == Extras::IncludeLAv2) {
            static_assert(Bad == CalcBad::Disable, "!");
            if (results->LaReference == nullptr) {
                results->LaReference = std::make_unique<LAReference<SubType>>(results);
                results->LaReference->GenerateApproximationData(results->maxRadius, (IterType)results->orb.size() - 1);
            }
        }
        else {
            results->LaReference = nullptr;
        }
    }

    return results;
}

template PerturbationResults<double, CalcBad::Enable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    double,
    double,
    CalcBad::Enable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<float, CalcBad::Enable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    float,
    float,
    CalcBad::Enable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<HDRFloat<double>, CalcBad::Enable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<double>,
    double,
    CalcBad::Enable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<HDRFloat<float>, CalcBad::Enable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<float>,
    float,
    CalcBad::Enable,
    RefOrbitCalc::Extras::None>();

template PerturbationResults<double, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    double,
    double,
    CalcBad::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<float, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    float,
    float,
    CalcBad::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<HDRFloat<double>, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<double>,
    double,
    CalcBad::Disable,
    RefOrbitCalc::Extras::None>();
template PerturbationResults<HDRFloat<float>, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<float>,
    float,
    CalcBad::Disable,
    RefOrbitCalc::Extras::None>();

///////////

template PerturbationResults<HDRFloat<double>, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<double>,
    double,
    CalcBad::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();
template PerturbationResults<HDRFloat<float>, CalcBad::Disable>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<
    HDRFloat<float>,
    float,
    CalcBad::Disable,
    RefOrbitCalc::Extras::IncludeLAv2>();


template<class T, CalcBad Bad>
std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>* RefOrbitCalc::GetPerturbationArray() {
    std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>* cur_array = nullptr;

    if constexpr (std::is_same<T, double>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            cur_array = &m_PerturbationResultsDoubleB;
        }
        else {
            cur_array = &m_PerturbationResultsDouble;
        }
    }
    else if constexpr (std::is_same<T, float>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            cur_array = &m_PerturbationResultsFloatB;
        }
        else {
            cur_array = &m_PerturbationResultsFloat;
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            cur_array = &m_PerturbationResultsHDRDoubleB;
        }
        else {
            cur_array = &m_PerturbationResultsHDRDouble;
        }
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        if constexpr (Bad == CalcBad::Enable) {
            cur_array = &m_PerturbationResultsHDRFloatB;
        }
        else {
            cur_array = &m_PerturbationResultsHDRFloat;
        }
    }

    return cur_array;
}

template<class T, class SubType, bool Authoritative, CalcBad Bad>
PerturbationResults<T, Bad>* RefOrbitCalc::GetUsefulPerturbationResults() {
    std::vector<PerturbationResults<T, Bad>*> useful_results;
    std::vector<std::unique_ptr<PerturbationResults<T, Bad>>>* cur_array = GetPerturbationArray<T, Bad>();

    if (!cur_array->empty()) {
        if (cur_array->size() > 64) {
            cur_array->erase(cur_array->begin());
        }

        for (size_t i = 0; i < cur_array->size(); i++) {
            if (IsPerturbationResultUsefulHere<T, Authoritative, Bad>(i)) {
                useful_results.push_back((*cur_array)[i].get());
            }
        }
    }

    PerturbationResults<T, Bad>* results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }

    return results;
}

template<class SrcT, CalcBad SrcEnableBad, class DestT, CalcBad DestEnableBad>
PerturbationResults<DestT, DestEnableBad>* RefOrbitCalc::CopyUsefulPerturbationResults(
    PerturbationResults<SrcT, SrcEnableBad>& src_array)
{
    if constexpr (DestEnableBad != SrcEnableBad) {
        return nullptr;
    }

    if constexpr (DestEnableBad == CalcBad::Disable && SrcEnableBad == CalcBad::Disable) {
        if constexpr (std::is_same<SrcT, double>::value) {
            m_PerturbationResultsFloat.push_back(
                std::make_unique<PerturbationResults<float, DestEnableBad>>());
            auto* dest = m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else if constexpr (std::is_same<SrcT, float>::value) {
            return nullptr;
        }
        else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
            m_PerturbationResultsHDRFloat.push_back(
                std::make_unique<PerturbationResults<HDRFloat<float>, DestEnableBad>>());
            auto* dest = m_PerturbationResultsHDRFloat[m_PerturbationResultsHDRFloat.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
            m_PerturbationResultsFloat.push_back(
                std::make_unique<PerturbationResults<float, DestEnableBad>>());
            auto* dest = m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else {
            return nullptr;
        }
    }
    else if constexpr (DestEnableBad == CalcBad::Enable && SrcEnableBad == CalcBad::Enable) {
        if constexpr (std::is_same<SrcT, double>::value) {
            m_PerturbationResultsFloatB.push_back(
                std::make_unique<PerturbationResults<float, DestEnableBad>>());
            auto* dest = m_PerturbationResultsFloatB[m_PerturbationResultsFloatB.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else if constexpr (std::is_same<SrcT, float>::value) {
            return nullptr;
        }
        else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
            m_PerturbationResultsHDRFloatB.push_back(
                std::make_unique<PerturbationResults<HDRFloat<float>, DestEnableBad>>());
            auto* dest = m_PerturbationResultsHDRFloatB[m_PerturbationResultsHDRFloatB.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
            m_PerturbationResultsFloatB.push_back(
                std::make_unique<PerturbationResults<float, DestEnableBad>>());
            auto* dest = m_PerturbationResultsFloatB[m_PerturbationResultsFloatB.size() - 1].get();
            dest->Copy(src_array);
            return dest;
        }
        else {
            return nullptr;
        }
    }
    else {
        return nullptr;
    }
}

template PerturbationResults<float, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    double,
    CalcBad::Enable,
    float,
    CalcBad::Enable>(PerturbationResults<double, CalcBad::Enable>&);
template PerturbationResults<HDRFloat<float>, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<double>,
    CalcBad::Enable,
    HDRFloat<float>,
    CalcBad::Enable>(PerturbationResults<HDRFloat<double>, CalcBad::Enable>&);
template PerturbationResults<float, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<float>,
    CalcBad::Enable,
    float,
    CalcBad::Enable>(PerturbationResults<HDRFloat<float>, CalcBad::Enable>&);

template PerturbationResults<float, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    double,
    CalcBad::Enable,
    float,
    CalcBad::Disable> (PerturbationResults<double, CalcBad::Enable>&);
template PerturbationResults<HDRFloat<float>, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<double>,
    CalcBad::Enable,
    HDRFloat<float>,
    CalcBad::Disable>(PerturbationResults<HDRFloat<double>, CalcBad::Enable>&);
template PerturbationResults<float, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<float>,
    CalcBad::Enable,
    float,
    CalcBad::Disable>(PerturbationResults<HDRFloat<float>, CalcBad::Enable>&);

template PerturbationResults<float, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    double,
    CalcBad::Disable,
    float,
    CalcBad::Enable>(PerturbationResults<double, CalcBad::Disable>&);
template PerturbationResults<HDRFloat<float>, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<double>,
    CalcBad::Disable,
    HDRFloat<float>,
    CalcBad::Enable>(PerturbationResults<HDRFloat<double>, CalcBad::Disable>&);
template PerturbationResults<float, CalcBad::Enable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<float>,
    CalcBad::Disable,
    float,
    CalcBad::Enable>(PerturbationResults<HDRFloat<float>, CalcBad::Disable>&);

template PerturbationResults<float, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    double,
    CalcBad::Disable,
    float,
    CalcBad::Disable>(PerturbationResults<double, CalcBad::Disable>&);
template PerturbationResults<HDRFloat<float>, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<double>,
    CalcBad::Disable,
    HDRFloat<float>,
    CalcBad::Disable>(PerturbationResults<HDRFloat<double>, CalcBad::Disable>&);
template PerturbationResults<float, CalcBad::Disable>*
RefOrbitCalc::CopyUsefulPerturbationResults<
    HDRFloat<float>,
    CalcBad::Disable,
    float,
    CalcBad::Disable>(PerturbationResults<HDRFloat<float>, CalcBad::Disable>&);




void RefOrbitCalc::ClearPerturbationResults(PerturbationResultType type) {
    auto IsMarkedToDelete = [&](const auto& o) -> bool {
        if (type == PerturbationResultType::All ||
            (type == PerturbationResultType::MediumRes &&
                o->AuthoritativePrecision == 0) ||
            (type == PerturbationResultType::HighRes &&
                o->AuthoritativePrecision != 0)) {
            return true;
        }

        return false;
    };

    auto ClearOne = [&](auto &arr) {
        arr.erase(
            std::remove_if(arr.begin(), arr.end(), IsMarkedToDelete),
            arr.end());
    };

    ClearOne(m_PerturbationResultsDouble);
    ClearOne(m_PerturbationResultsFloat);
    ClearOne(m_PerturbationResultsHDRDouble);
    ClearOne(m_PerturbationResultsHDRFloat);

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;
}

void RefOrbitCalc::ResetGuess(HighPrecision x, HighPrecision y) {
    m_PerturbationGuessCalcX = x;
    m_PerturbationGuessCalcY = y;
}
