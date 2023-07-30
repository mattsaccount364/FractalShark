#include "stdafx.h"

#include "RefOrbitCalc.h"
#include "Fractal.h"

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

RefOrbitCalc::RefOrbitCalc(Fractal& Fractal)
: m_Fractal(Fractal) {
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
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
        return check == &m_PerturbationResultsHDRFloat;
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
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
        ClearPerturbationResults();
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ::MessageBox(NULL, L"Watch the memory use... this is just a warning", L"", MB_OK);
        assert(false);
    }
}

template<class T>
std::vector<std::unique_ptr<PerturbationResults<T>>>& RefOrbitCalc::GetPerturbationResults() {
    if constexpr (std::is_same<T, double>::value) {
        return m_PerturbationResultsDouble;
    }
    else if constexpr (std::is_same<T, float>::value) {
        return m_PerturbationResultsFloat;
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        return m_PerturbationResultsHDRDouble;
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        return m_PerturbationResultsHDRFloat;
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
            AddPerturbationReferencePointMT2<T, SubType, false, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
            AddPerturbationReferencePointST<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity2) {
            AddPerturbationReferencePointMT2<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity2Perturb) {
            AddPerturbationReferencePointMT2<T, SubType, true, BenchmarkState, CalcBad::Enable, ReuseMode::SaveForReuse>(
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
            AddPerturbationReferencePointMT2<T, SubType, false, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
            AddPerturbationReferencePointST<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity2) {
            AddPerturbationReferencePointMT2<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::DontSaveForReuse>(
                m_PerturbationGuessCalcX,
                m_PerturbationGuessCalcY);
        }
        else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity2Perturb) {
            AddPerturbationReferencePointMT2<T, SubType, true, BenchmarkState, CalcBad::Disable, ReuseMode::SaveForReuse>(
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
static void AddReused(T* results, const HighPrecision& zx, const HighPrecision& zy) {
    HighPrecision ReducedZx;
    HighPrecision ReducedZy;

    ReducedZx = zx;
    ReducedZy = zy;

#ifndef FULL_PREC_TEST_ONLY
    ReducedZx.precision(50);
    ReducedZy.precision(50);
#endif

    results->ReuseX.push_back(ReducedZx);
    results->ReuseY.push_back(ReducedZy);
}

template<class T>
static void InitReused(T* results) {
    HighPrecision Zero = 0;

#ifndef FULL_PREC_TEST_ONLY
    Zero.precision(50);
#endif

    results->ReuseX.push_back(Zero);
    results->ReuseY.push_back(Zero);
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, RefOrbitCalc::CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResultsArray = GetPerturbationResults<T>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision radiusX = fabs(m_Fractal.GetMaxX() - initX) + fabs(m_Fractal.GetMinX() - initX);
    HighPrecision radiusY = fabs(m_Fractal.GetMaxY() - initY) + fabs(m_Fractal.GetMinY() - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    results->AuthoritativePrecision = initX.precision();
    results->ReuseX.reserve(m_Fractal.GetNumIterations());
    results->ReuseY.reserve(m_Fractal.GetNumIterations());

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_Fractal.GetNumIterations() + 1; // +1 for push_back(0) below

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    unsigned int i;

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_Fractal.GetNumIterations());
    results->y.reserve(m_Fractal.GetNumIterations());

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.reserve(m_Fractal.GetNumIterations());
    }

    results->x.push_back(T(0));
    results->y.push_back(T(0));

    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
        InitReused(results);
    }

    // Note: results->bad is not here.  See end of this function.

    //SubType glitch;
    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

    SubType glitch = (SubType)0.0000001;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    T zxCopy;
    T zyCopy;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    zx = cx;
    zy = cy;
    for (i = 0; i < m_Fractal.GetNumIterations(); i++)
    {
        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };
        }

        zx2 = zx * 2;
        zy2 = zy * 2;

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(results, zx, zy);
        }

        if constexpr (Bad == CalcBad::Enable) {
            T sq_x = double_zx * double_zx;
            T sq_y = double_zy * double_zy;
            T norm = (sq_x + sq_y) * glitch;

            bool underflow = (HdrAbs((T)zx) <= small_float ||
                HdrAbs((T)zy) <= small_float ||
                norm <= small_float);
            results->bad.push_back(underflow);
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

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n2 < n3) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    break;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        zx = zx * zx - zy * zy + cx;
        zy = zx2 * zy + cy;

        T tempZX = double_zx + (T)cx;
        T tempZY = double_zy + (T)cy;
        T zn_size = tempZX * tempZX + tempZY * tempZY;
        if (zn_size > TwoFiftySix) {
            break;
        }
    }

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.push_back(false);
        assert(results->bad.size() == results->x.size());
    }
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, RefOrbitCalc::CalcBad Bad>
bool RefOrbitCalc::AddPerturbationReferencePointSTReuse(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResultsArray = GetPerturbationResults<T>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T>>());

    auto* existingResults = GetUsefulPerturbationResults<T, SubType, true>();
    if (existingResults == nullptr) {
        PerturbationResultsArray.pop_back();
        return false;
    }

    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

#ifndef FULL_PREC_TEST_ONLY
    auto NewPrec = m_Fractal.GetPrecision(m_Fractal.GetMinX(), m_Fractal.GetMinY(), m_Fractal.GetMaxX(), m_Fractal.GetMaxY());
    auto ExistingPrec = existingResults->AuthoritativePrecision;
    auto FullPrec = NewPrec - ExistingPrec + ExtraPrecision;
    uint32_t precNum = static_cast<uint32_t>(FullPrec);

    // This all generally works but suffers precision problems after about 10^35.
    // The problem naturally is the original reference orbit is calculated only to so many digits.
    // The idea here is mostly to see how much we can squeeze out of MPIR etc before it looks bad.
    // 10^35 seems about it.  35 = 20 + 15, where 20 is what we're adding in "GetPrecision" above.
    // So, if you increase precision on the reference, then adjust ExtraPrecision accordingly.
    // 
    // Comment this out to find out where things start failing.

    if (precNum > ExtraPrecision + MpirEstPrecision) {
        PerturbationResultsArray.pop_back();
        return false;
    }

    scoped_mpfr_precision prec(precNum);
#endif

    HighPrecision radiusX = fabs(m_Fractal.GetMaxX() - initX) + fabs(m_Fractal.GetMinX() - initX);
    HighPrecision radiusY = fabs(m_Fractal.GetMaxY() - initY) + fabs(m_Fractal.GetMinY() - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    results->AuthoritativePrecision = 0;
    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_Fractal.GetNumIterations() + 1; // +1 for push_back(0) below

#ifndef FULL_PREC_TEST_ONLY
    radiusX.precision(precNum);
    radiusY.precision(precNum);
    cx.precision(precNum);
    cy.precision(precNum);
#endif

    HighPrecision zx, zy;
    unsigned int i;

    results->x.reserve(m_Fractal.GetNumIterations());
    results->y.reserve(m_Fractal.GetNumIterations());

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.reserve(m_Fractal.GetNumIterations());
        results->bad.resize(m_Fractal.GetNumIterations()); // TODO
        assert(false);
        // Not implemented yet
    }

    results->x.push_back(T(0));
    results->y.push_back(T(0));

    HighPrecision HighOne = 1.0;
    HighPrecision HighTwo = 2.0;
    HighPrecision TwoFiftySix = 256.0;
    HighPrecision DeltaReal = initX - existingResults->hiX;
    HighPrecision DeltaImaginary = initY - existingResults->hiY;
    HighPrecision DeltaSub0X = DeltaReal;
    HighPrecision DeltaSub0Y = DeltaImaginary;
    HighPrecision DeltaSubNX = 0;
    HighPrecision DeltaSubNY = 0;

#ifndef FULL_PREC_TEST_ONLY
    HighOne.precision(precNum);
    HighTwo.precision(precNum);
    TwoFiftySix.precision(precNum);
    DeltaReal.precision(precNum);
    DeltaImaginary.precision(precNum);
    DeltaSub0X.precision(precNum);
    DeltaSub0Y.precision(precNum);
    DeltaSubNX.precision(precNum);
    DeltaSubNY.precision(precNum);
#endif

    size_t RefIteration = 0;
    size_t MaxRefIteration = existingResults->x.size() - 1;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    T zxCopy;
    T zyCopy;

    HighPrecision tempZX;
    HighPrecision tempZY;
    HighPrecision zn_size;
    HighPrecision normDeltaSubN;
    HighPrecision reuseX, reuseY;

#ifndef FULL_PREC_TEST_ONLY
    tempZX.precision(precNum);
    tempZY.precision(precNum);
    zn_size.precision(precNum);
    normDeltaSubN.precision(precNum);

    reuseX.precision(precNum);
    reuseY.precision(precNum);
#endif

    zx = cx;
    zy = cy;

    std::vector<HighPrecision> existingReuseX;
    std::vector<HighPrecision> existingReuseY;

    existingReuseX.reserve(existingResults->ReuseX.size());
    existingReuseY.reserve(existingResults->ReuseY.size());
    for (i = 0; i < existingResults->ReuseX.size(); i++) {
        auto tempX = existingResults->ReuseX[i];
        tempX.precision(precNum);
        existingReuseX.push_back(tempX);

        auto tempY = existingResults->ReuseY[i];
        tempY.precision(precNum);
        existingReuseY.push_back(tempY);
    }

    for (i = 0; i < m_Fractal.GetNumIterations(); i++) {
        if constexpr (Periodicity) {
            zxCopy = T(zx);
            zyCopy = T(zy);
        }

        const HighPrecision DeltaSubNXOrig = DeltaSubNX;
        const HighPrecision DeltaSubNYOrig = DeltaSubNY;

        reuseX = existingReuseX[RefIteration];
        reuseY = existingReuseY[RefIteration];

        DeltaSubNX =
            DeltaSubNXOrig * (reuseX * HighTwo + DeltaSubNXOrig) -
            DeltaSubNYOrig * (reuseY * HighTwo + DeltaSubNYOrig) +
            DeltaSub0X;

        DeltaSubNY =
            DeltaSubNXOrig * (reuseY * HighTwo + DeltaSubNYOrig) +
            DeltaSubNYOrig * (reuseX * HighTwo + DeltaSubNXOrig) +
            DeltaSub0Y;

        ++RefIteration;

        reuseX = existingReuseX[RefIteration];
        reuseY = existingReuseY[RefIteration];

        tempZX = reuseX + DeltaSubNX;
        tempZY = reuseY + DeltaSubNY;
        zn_size = tempZX * tempZX + tempZY * tempZY;
        normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

        if (zn_size > TwoFiftySix) {
            break;
        }

        if (zn_size < normDeltaSubN ||
            RefIteration == MaxRefIteration) {
            DeltaSubNX = tempZX;
            DeltaSubNY = tempZY;
            RefIteration = 0;
        }

        zx = tempZX;
        zy = tempZY;

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

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * T(2.0);
            HdrReduce(n3);

            if (n2 < n3) {
                if constexpr (BenchmarkState == BenchmarkMode::Disable) {
                    // Break before adding the result.
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

        results->x.push_back(reducedZx);
        results->y.push_back(reducedZy);
    }

    if constexpr (Bad == CalcBad::Enable) {
        //results->bad.push_back(false); // TODO
        //assert(results->bad.size() == results->x.size());
        results->bad.resize(results->x.size());
    }
    return true;
}

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

static inline void PrefetchHighPrec(HighPrecision& target) {
    ENABLE_PREFETCH((const char*)&target.backend().data(), _MM_HINT_T0);
    size_t lastindex = abs(target.backend().data()->_mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target.backend().data()->_mp_d, total);
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, RefOrbitCalc::CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT2(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResultsArray = GetPerturbationResults<T>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision radiusX = fabs(m_Fractal.GetMaxX() - initX) + fabs(m_Fractal.GetMinX() - initX);
    HighPrecision radiusY = fabs(m_Fractal.GetMaxY() - initY) + fabs(m_Fractal.GetMinY() - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    HighPrecision zx, zy;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    results->AuthoritativePrecision = initX.precision();
    results->ReuseX.reserve(m_Fractal.GetNumIterations());
    results->ReuseY.reserve(m_Fractal.GetNumIterations());

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_Fractal.GetNumIterations() + 1; // +1 for push_back(0) below

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_Fractal.GetNumIterations());
    results->y.reserve(m_Fractal.GetNumIterations());

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.reserve(m_Fractal.GetNumIterations());
    }

    results->x.push_back(T(0));
    results->y.push_back(T(0));

    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
        InitReused(results);
    }

    // Note: results->bad is not here.  See end of this function.

    //SubType glitch;
    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        HighPrecision zx;
        HighPrecision zx_sq;
    };

    struct ThreadZyData {
        HighPrecision zy;
        HighPrecision zy_sq;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

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

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};

    //threadZxdata->zx_sq = &zx_sq;

    //threadZydata->zy_sq = &zy_sq;

    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;

    bool done1 = false;
    bool done2 = false;

    HighPrecision zy_sq_orig;

    zx = cx;
    zy = cy;

    T zxCopy;
    T zyCopy;
    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    bool zyStarted = false;

    for (size_t i = 0; i < m_Fractal.GetNumIterations(); i++)
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

        T double_zx;
        T double_zy;

        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };

            double_zx = zxCopy;
            double_zy = zyCopy;
        }
        else {
            double_zx = (T)zx;
            double_zy = (T)zy;
        }

        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AddReused(results, zx, zy);
        }

        if constexpr (Bad == CalcBad::Enable) {
            T norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
            //HdrReduce(norm);
            bool underflow = (HdrAbs(double_zx) <= small_float ||
                HdrAbs(double_zy) <= small_float ||
                norm <= small_float);
            results->bad.push_back(underflow);
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

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n2 < n3) {
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
                        results->m_Periodic = true;
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
        results->bad.push_back(false);
        assert(results->bad.size() == results->x.size());
    }

    expectedZx = nullptr;
    ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);

    expectedZy = nullptr;
    ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);

    tZx->join();
    tZy->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);
}

template<class T, class SubType, bool Periodicity, RefOrbitCalc::BenchmarkMode BenchmarkState, RefOrbitCalc::CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
void RefOrbitCalc::AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResultsArray = GetPerturbationResults<T>();
    PerturbationResultsArray.push_back(std::make_unique<PerturbationResults<T>>());
    auto* results = PerturbationResultsArray[PerturbationResultsArray.size() - 1].get();

    HighPrecision radiusX = fabs(m_Fractal.GetMaxX() - initX) + fabs(m_Fractal.GetMinX() - initX);
    HighPrecision radiusY = fabs(m_Fractal.GetMaxY() - initY) + fabs(m_Fractal.GetMinY() - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    HighPrecision zx_sq, zy_sq;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    results->AuthoritativePrecision = initX.precision();
    results->ReuseX.reserve(m_Fractal.GetNumIterations());
    results->ReuseY.reserve(m_Fractal.GetNumIterations());

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_Fractal.GetNumIterations() + 1; // +1 for push_back(0) below

    //volatile double tempX = Convert<HighPrecision, double>(initX);
    //volatile double tempY = Convert<HighPrecision, double>(initY);
    //volatile double intX = Convert<HighPrecision, double>(XFromCalcToScreen(initX));
    //volatile double intY = Convert<HighPrecision, double>(YFromCalcToScreen(initY));

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_Fractal.GetNumIterations());
    results->y.reserve(m_Fractal.GetNumIterations());

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.reserve(m_Fractal.GetNumIterations());
    }

    results->x.push_back(T(0));
    results->y.push_back(T(0));

    if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
        InitReused(results);
    }

    // Note: results->bad is not here.  See end of this function.

    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

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

    for (size_t i = 0; i < m_Fractal.GetNumIterations(); i++)
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
            AddReused(results, zx, zy);
        }

        // Start Thread 1: zx = zx * zx - zy * zy + cx;
        thread1data->zy = zy;

        Thread1Memory->In.store(
            thread1data,
            std::memory_order_release);

        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        if constexpr (Bad == CalcBad::Enable) {
            T norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
            bool underflow = (HdrAbs(double_zx) <= small_float ||
                HdrAbs(double_zy) <= small_float ||
                norm <= small_float);
            results->bad.push_back(underflow);
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

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n2 < n3) {
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
                results->m_Periodic = true;
                break;
            }
        }
    }

    if constexpr (Bad == CalcBad::Enable) {
        results->bad.push_back(false);
        assert(results->bad.size() == results->x.size());
    }

    expectedZx = nullptr;
    ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);

    expectedZy = nullptr;
    ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);

    expected1 = nullptr;
    Thread1Memory->In.compare_exchange_strong(expected1, (Thread1Data*)0x1, std::memory_order_release);

    expected2 = nullptr;
    Thread2Memory->In.compare_exchange_strong(expected2, (Thread2Data*)0x1, std::memory_order_release);

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
}


bool RefOrbitCalc::RequiresReferencePoints() const {
    switch (m_Fractal.GetRenderAlgorithm()) {
    case RenderAlgorithm::Cpu64PerturbedBLA:
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::Gpu1x32Perturbed:
    case RenderAlgorithm::Gpu1x32PerturbedPeriodic:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
    case RenderAlgorithm::Gpu1x64Perturbed:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
    case RenderAlgorithm::Gpu2x32Perturbed:
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
        return true;
    }

    return false;
}

bool RefOrbitCalc::IsReferencePerturbationEnabled() const {
    switch (m_PerturbationAlg) {
    case PerturbationAlg::MTPeriodicity2Perturb:
        return true;
    default:
        return false;
    }
}

template<class T, bool Authoritative>
bool RefOrbitCalc::IsPerturbationResultUsefulHere(size_t i) const {
    if constexpr (std::is_same<T, double>::value) {
        if constexpr (Authoritative == true) {
            return m_PerturbationResultsDouble[i]->AuthoritativePrecision != 0 &&
                (m_PerturbationResultsDouble[i]->MaxIterations > m_PerturbationResultsDouble[i]->x.size() ||
                    m_PerturbationResultsDouble[i]->MaxIterations >= m_Fractal.GetNumIterations());
        }

        return
            m_PerturbationResultsDouble[i]->hiX >= m_Fractal.GetMinX() &&
            m_PerturbationResultsDouble[i]->hiX <= m_Fractal.GetMaxX() &&
            m_PerturbationResultsDouble[i]->hiY >= m_Fractal.GetMinY() &&
            m_PerturbationResultsDouble[i]->hiY <= m_Fractal.GetMaxY() &&
            (m_PerturbationResultsDouble[i]->MaxIterations > m_PerturbationResultsDouble[i]->x.size() ||
                m_PerturbationResultsDouble[i]->MaxIterations >= m_Fractal.GetNumIterations());
    }
    else if constexpr (std::is_same<T, float>::value) {
        if constexpr (Authoritative == true) {
            return m_PerturbationResultsFloat[i]->AuthoritativePrecision != 0 &&
                (m_PerturbationResultsFloat[i]->MaxIterations > m_PerturbationResultsFloat[i]->x.size() ||
                    m_PerturbationResultsFloat[i]->MaxIterations >= m_Fractal.GetNumIterations());
        }

        return
            m_PerturbationResultsFloat[i]->hiX >= m_Fractal.GetMinX() &&
            m_PerturbationResultsFloat[i]->hiX <= m_Fractal.GetMaxX() &&
            m_PerturbationResultsFloat[i]->hiY >= m_Fractal.GetMinY() &&
            m_PerturbationResultsFloat[i]->hiY <= m_Fractal.GetMaxY() &&
            (m_PerturbationResultsFloat[i]->MaxIterations > m_PerturbationResultsFloat[i]->x.size() ||
                m_PerturbationResultsFloat[i]->MaxIterations >= m_Fractal.GetNumIterations());
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        if constexpr (Authoritative == true) {
            return m_PerturbationResultsHDRDouble[i]->AuthoritativePrecision != 0 &&
                (m_PerturbationResultsHDRDouble[i]->MaxIterations > m_PerturbationResultsHDRDouble[i]->x.size() ||
                    m_PerturbationResultsHDRDouble[i]->MaxIterations >= m_Fractal.GetNumIterations());
        }

        return
            m_PerturbationResultsHDRDouble[i]->hiX >= m_Fractal.GetMinX() &&
            m_PerturbationResultsHDRDouble[i]->hiX <= m_Fractal.GetMaxX() &&
            m_PerturbationResultsHDRDouble[i]->hiY >= m_Fractal.GetMinY() &&
            m_PerturbationResultsHDRDouble[i]->hiY <= m_Fractal.GetMaxY() &&
            (m_PerturbationResultsHDRDouble[i]->MaxIterations > m_PerturbationResultsHDRDouble[i]->x.size() ||
                m_PerturbationResultsHDRDouble[i]->MaxIterations >= m_Fractal.GetNumIterations());
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        if constexpr (Authoritative == true) {
            return m_PerturbationResultsHDRFloat[i]->AuthoritativePrecision != 0 &&
                (m_PerturbationResultsHDRFloat[i]->MaxIterations > m_PerturbationResultsHDRFloat[i]->x.size() ||
                    m_PerturbationResultsHDRFloat[i]->MaxIterations >= m_Fractal.GetNumIterations());
        }

        return
            m_PerturbationResultsHDRFloat[i]->hiX >= m_Fractal.GetMinX() &&
            m_PerturbationResultsHDRFloat[i]->hiX <= m_Fractal.GetMaxX() &&
            m_PerturbationResultsHDRFloat[i]->hiY >= m_Fractal.GetMinY() &&
            m_PerturbationResultsHDRFloat[i]->hiY <= m_Fractal.GetMaxY() &&
            (m_PerturbationResultsHDRFloat[i]->MaxIterations > m_PerturbationResultsHDRFloat[i]->x.size() ||
                m_PerturbationResultsHDRFloat[i]->MaxIterations >= m_Fractal.GetNumIterations());
    }
}

template<class T, class SubType>
PerturbationResults<T>* RefOrbitCalc::GetAndCreateUsefulPerturbationResults() {
    bool added = false;
    if (IsReferencePerturbationEnabled()) {
        if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
            m_PerturbationGuessCalcX = (m_Fractal.GetMaxX() + m_Fractal.GetMinX()) / 2;
            m_PerturbationGuessCalcY = (m_Fractal.GetMaxY() + m_Fractal.GetMinY()) / 2;
        }

        PerturbationResults<T>* results = GetUsefulPerturbationResults<T, SubType, false>();
        if (results == nullptr) {
            added = AddPerturbationReferencePointSTReuse<T, SubType, true, BenchmarkMode::Disable, CalcBad::Disable>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
        }
    }

    PerturbationResults<T>* results = GetUsefulPerturbationResults<T, SubType, false>();
    if (results == nullptr) {
        if (added) {
            ::MessageBox(NULL, L"Why didn't this work! :(", L"", MB_OK);
        }
        std::vector<std::unique_ptr<PerturbationResults<T>>>* cur_array = GetPerturbationArray<T>();
        AddPerturbationReferencePoint<T, SubType, BenchmarkMode::Disable>();

        results = (*cur_array)[cur_array->size() - 1].get();
    }

    return results;
}

template PerturbationResults<double>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<double, double>();
template PerturbationResults<float>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<float, float>();
template PerturbationResults<HDRFloat<double>>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<HDRFloat<double>, double>();
template PerturbationResults<HDRFloat<float>>*
RefOrbitCalc::GetAndCreateUsefulPerturbationResults<HDRFloat<float>, float>();

template<class T>
std::vector<std::unique_ptr<PerturbationResults<T>>>* RefOrbitCalc::GetPerturbationArray() {
    std::vector<std::unique_ptr<PerturbationResults<T>>>* cur_array = nullptr;

    if constexpr (std::is_same<T, double>::value) {
        cur_array = &m_PerturbationResultsDouble;
    }
    else if constexpr (std::is_same<T, float>::value) {
        cur_array = &m_PerturbationResultsFloat;
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        cur_array = &m_PerturbationResultsHDRDouble;
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        cur_array = &m_PerturbationResultsHDRFloat;
    }

    return cur_array;
}

template<class T, class SubType, bool Authoritative>
PerturbationResults<T>* RefOrbitCalc::GetUsefulPerturbationResults() {
    std::vector<PerturbationResults<T>*> useful_results;
    std::vector<std::unique_ptr<PerturbationResults<T>>>* cur_array = GetPerturbationArray<T>();

    if (!cur_array->empty()) {
        if (cur_array->size() > 64) {
            cur_array->erase(cur_array->begin());
        }

        for (size_t i = 0; i < cur_array->size(); i++) {
            if (IsPerturbationResultUsefulHere<T, Authoritative>(i)) {
                useful_results.push_back((*cur_array)[i].get());
            }
        }
    }

    PerturbationResults<T>* results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }

    return results;
}

template<class SrcT, class DestT>
PerturbationResults<DestT>* RefOrbitCalc::CopyUsefulPerturbationResults(
    PerturbationResults<SrcT>& src_array)
{
    if constexpr (std::is_same<SrcT, double>::value) {
        m_PerturbationResultsFloat.push_back(std::make_unique<PerturbationResults<float>>());
        auto* dest = m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1].get();
        dest->Copy(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, float>::value) {
        return nullptr;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
        m_PerturbationResultsHDRFloat.push_back(std::make_unique<PerturbationResults<HDRFloat<float>>>());
        auto* dest = m_PerturbationResultsHDRFloat[m_PerturbationResultsHDRFloat.size() - 1].get();
        dest->Copy(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
        m_PerturbationResultsFloat.push_back(std::make_unique<PerturbationResults<float>>());
        auto* dest = m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1].get();
        dest->Copy(src_array);
        return dest;
    }
    else {
        return nullptr;
    }
}

template PerturbationResults<float>*
RefOrbitCalc::CopyUsefulPerturbationResults<double, float>(PerturbationResults<double>&);
template PerturbationResults<HDRFloat<float>>*
RefOrbitCalc::CopyUsefulPerturbationResults<HDRFloat<double>, HDRFloat<float>>(PerturbationResults<HDRFloat<double>>&);
template PerturbationResults<float>*
RefOrbitCalc::CopyUsefulPerturbationResults<HDRFloat<float>, float>(PerturbationResults<HDRFloat<float>>&);

void RefOrbitCalc::ClearPerturbationResults() {
    m_PerturbationResultsDouble.clear();
    m_PerturbationResultsFloat.clear();
    m_PerturbationResultsHDRDouble.clear();
    m_PerturbationResultsHDRFloat.clear();

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;
}

void RefOrbitCalc::ResetGuess(HighPrecision x, HighPrecision y) {
    m_PerturbationGuessCalcX = x;
    m_PerturbationGuessCalcY = y;
}
