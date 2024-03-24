#include "stdafx.h"
#include "LAReference.h"
#include "LAParameters.h"
#include "RefOrbitCalc.h"
#include "PerturbationResults.h"

#include <deque>
#include <future>


//#include <Windows.h>

// Imagina uses 4 for this periodDivisor number instead.
// Smaller = generally better perf, but more memory usage.
// Larger = generally worse perf, but less memory usage.  The idea is that if compression
// is enabled, then the priority is to reduce memory usage, so we want to use a larger
// number here.
template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
const int LAReference<IterType, Float, SubType, PExtras>::periodDivisor =
    PExtras == PerturbExtras::EnableCompression ? 8 : 2;

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
IterType LAReference<IterType, Float, SubType, PExtras>::LAsize() {
    return (IterType)m_LAs.GetSize();
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType, PExtras>::CreateLAFromOrbit(
    const LAParameters& la_parameters,
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    IterType maxRefIteration) {

    auto compressionHelper{ std::make_unique<CompressionHelper<IterType, Float, PExtras>>(PerturbationResults) };

    {
        m_IsValid = false;
        m_LAStages.MutableResize(MaxLAStages);

        //m_LAs.reserve(maxRefIteration);

        m_UseAT = false;
        m_LAStageCount = 0;

        m_LAStages[0].LAIndex = 0;
    }

    IterType Period = 0;

    LAInfoDeep<IterType, Float, SubType, PExtras> LA{ la_parameters, FloatComplexT()};
    LA = LA.Step(
        la_parameters,
        PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;
        bool PeriodDetected = LA.Step(
            la_parameters,
            NewLA,
            PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
        if (!PeriodDetected) {
            LA = NewLA;
            continue;
        }

        Period = i;
        LAI.StepLength = Period;

        LA.SetLAi(LAI);
        m_LAs.PushBack(LA);

        LAI.NextStageLAIndex = i;

        if (i + 1 < maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i)).Step(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
            i += 1;
        }
        break;
    }

    m_LAStageCount = 1;

    IterType PeriodBegin = Period;
    IterType PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, 0)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
            Period = (IterType)std::round(std::pow(static_cast<double>(maxRefIteration), 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            m_LAs.PushBack(LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(
                    *compressionHelper,
                    maxRefIteration)));

            m_LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        m_LAs.PopBack();

        LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(*compressionHelper, 0)).Step(
                la_parameters, 
                PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
        Period = (IterType)std::round(std::pow(static_cast<double>(maxRefIteration), 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    // TODO can we multithread this
    // const auto numPerThread = maxRefIteration / std::thread::hardware_concurrency();

    for (; i < maxRefIteration; i++) {
        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
        const bool PeriodDetected{
            LA.Step(
                la_parameters,
                NewLA,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i)) };

        if (!PeriodDetected && i < PeriodEnd) {
            LA = NewLA;
            continue;
        }

        LAI.StepLength = i - PeriodBegin;

        LA.SetLAi(LAI);
        m_LAs.PushBack(LA);

        LAI.NextStageLAIndex = i;
        PeriodBegin = i;
        PeriodEnd = PeriodBegin + Period;

        const IterType ip1{ i + 1 };
        const bool detected{
            NewLA.DetectPeriod(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, ip1)) };

        if (detected || ip1 >= maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i)).Step(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, ip1));
            i++;
        }
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    m_LAs.PushBack(LA);

    m_LAStages[0].MacroItCount = LAsize();

    auto LA2 = LAInfoDeep<IterType, Float, SubType, PExtras>(
        la_parameters,
        PerturbationResults.GetComplex<SubType>(*compressionHelper, maxRefIteration));
    LA2.SetLAi({});
    m_LAs.PushBack(LA2);

    return true;
}

//
// Based on e704d5b40895f453156e09d9542a589545cc6144 from fractal-zoomer, LAReference.java
// The function below has some comments in the style // l:<number>.
// These are references to the line numbers in the original Java code.
//

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType, PExtras>::CreateLAFromOrbitMT(
    const LAParameters& la_parameters,
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    IterType maxRefIteration) {

    auto compressionHelper{ std::make_unique<CompressionHelper<IterType, Float, PExtras>> (PerturbationResults) };

    // This is not supported.
    if (m_AddPointOptions == AddPointOptions::OpenExistingWithSave) {
        assert(false);
        ::MessageBox(nullptr, L"AddPointOptions::OpenExistingWithSave is not supported", L"", MB_OK | MB_APPLMODAL);
        return false;
    }

    // The idea here is to scale the number of threads according to the
    // expected work.  If the work is small, we don't want to use a lot
    // of threads.  If the work is large, we want to use a lot of threads.
    // The work is expected to be large if the number of iterations is large.
    constexpr size_t WorkThreshholdForThreads = 50000;

    size_t ThreadCount = maxRefIteration / WorkThreshholdForThreads;
    if (ThreadCount > std::thread::hardware_concurrency()) {
        ThreadCount = std::thread::hardware_concurrency();
    }
    else if (ThreadCount == 0) {
        ThreadCount = 1;
    }

    if (ThreadCount == 1) {
        // If we only have one thread, then we don't need to do any
        // special handling.  We can just call the single-threaded
        // version of this function.
        return CreateLAFromOrbit(la_parameters, PerturbationResults, maxRefIteration);
    }

    {
        m_IsValid = false;
        m_LAStages.MutableResize(MaxLAStages);

        m_UseAT = false;
        m_LAStageCount = 0;

        m_LAStages[0].LAIndex = 0;
    }

    IterType Period = 0;

    LAInfoDeep<IterType, Float, SubType, PExtras> LA{ la_parameters, FloatComplexT() };
    LA = LA.Step(la_parameters, PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;
        bool PeriodDetected = LA.Step(
            la_parameters,
            NewLA,
            PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
        if (!PeriodDetected) {
            LA = NewLA;
            continue;
        }

        Period = i;
        LAI.StepLength = Period;

        LA.SetLAi(LAI);
        m_LAs.PushBack(LA);

        LAI.NextStageLAIndex = i;

        if (i + 1 < maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
            i += 1;
        }
        break;
    }

    m_LAStageCount = 1;

    IterType PeriodBegin = Period;
    IterType PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, 0)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
            Period = (IterType)std::round(std::pow(static_cast<double>(maxRefIteration), 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            m_LAs.PushBack(LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters, 
                PerturbationResults.GetComplex<SubType>(*compressionHelper, maxRefIteration)));

            m_LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        m_LAs.PopBack();

        LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
            la_parameters, 
            PerturbationResults.GetComplex<SubType>(*compressionHelper, 0)).Step(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, 1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
        Period = (IterType)std::round(std::pow(static_cast<double>(maxRefIteration), 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    std::deque<std::shared_future<int64_t>> StartIndexFuture(ThreadCount);
    std::deque<std::promise<int64_t>> StartIndexPromise(ThreadCount);
    std::vector<GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>>> LAsPerThread;
    std::vector<LAInfoDeep<IterType, Float, SubType, PExtras>> LastLAPerThread(ThreadCount);
    std::deque<std::atomic_int64_t> FinishIndex(ThreadCount);

    // We don't ever save these files.  But we give the option of memory-only.
    AddPointOptions OptionsToUse = m_AddPointOptions;
    if (OptionsToUse == AddPointOptions::EnableWithSave) {
        OptionsToUse = AddPointOptions::EnableWithoutSave;
    }

    for (size_t k = 0; k < ThreadCount; k++) {
        std::wstring filename = L"LAsPerThread" + std::to_wstring(k) + L".dat";
        LAsPerThread.emplace_back(OptionsToUse, filename);
    }

    for (auto threadIndex = 0; threadIndex < StartIndexPromise.size(); threadIndex++) {
        StartIndexFuture[threadIndex] = StartIndexPromise[threadIndex].get_future();
    }

    // l:477
    auto Starter = [ // MOAR VARIABLES
            &la_parameters,
            &i,
            &LA,
            &LAI,
            &PeriodBegin,
            &PeriodEnd,
            &PerturbationResults,
            Period,
            &StartIndexFuture,
            &StartIndexPromise,
            &FinishIndex,
            &LastLAPerThread,
            ThreadCount,
            maxRefIteration,
            this
        ]() {

        auto compressionHelper{ std::make_unique<CompressionHelper<IterType, Float, PExtras>>(PerturbationResults) };

        const auto threadBoundary = maxRefIteration / ThreadCount;

        const auto ThreadID = 0;
        auto NextThread = 1;

        // l:487
        for (; i < maxRefIteration; i++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            const bool PeriodDetected{ LA.Step(
                la_parameters,
                NewLA,
                PerturbationResults.GetComplex<SubType>(*compressionHelper,i)) };

            if (!PeriodDetected && i < PeriodEnd) {
                LA = NewLA;
                continue;
            }

            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            LAI.NextStageLAIndex = i;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            // l:504
            const IterType ip1{ i + 1 };
            const bool detected{ NewLA.DetectPeriod(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, ip1)) };

            if (detected || ip1 >= maxRefIteration) {
                LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, i));
            }
            else {
                LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, i)).Step(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(*compressionHelper, ip1));
                i++;
            }

            // l:520
            if (i > threadBoundary) {

                if (i >= maxRefIteration) {
                    break;
                }

                if (NextThread < StartIndexFuture.size()) {
                    auto nextStart = StartIndexFuture[NextThread].get();

                    if (nextStart < 0) {
                        return;
                    }

                    if (static_cast<int64_t>(i) == nextStart - 1) {
                        i++;
                        break;
                    }
                    else if (static_cast<int64_t>(i) >= nextStart) {
                        NextThread++;
                    }
                }
            }
        }

        FinishIndex[ThreadID] = static_cast<int64_t>(i);
        LAI.StepLength = i - PeriodBegin;

        // TODO I don't get this bit
        LA.SetLAi(LAI);
        LastLAPerThread[ThreadID] = LA;
    };

    auto Worker = [
            &la_parameters,
            &PerturbationResults,
            Period,
            &StartIndexFuture,
            &StartIndexPromise,
            &FinishIndex,
            &LAsPerThread,
            &LastLAPerThread,
            ThreadCount,
            maxRefIteration,
            this
        ](uint32_t ThreadID) {

        auto compressionHelper{ std::make_unique<CompressionHelper<IterType, Float, PExtras>>(PerturbationResults) };

        auto NextThread = ThreadID + 1;
        const auto LastThread = ThreadCount - 1;

        const IterTypeFull intermediate_j = static_cast<IterTypeFull>(maxRefIteration) * ThreadID / ThreadCount;
        const IterType Begin = static_cast<IterType>(intermediate_j);
        auto j = Begin;

        const IterTypeFull intermediate_end = static_cast<IterTypeFull>(maxRefIteration) * NextThread / ThreadCount;
        const IterType End = static_cast<IterType>(intermediate_end);

        // l:586
        LAInfoI<IterType> LAI_;
        LAI_.NextStageLAIndex = j;

        LAInfoDeep<IterType, Float, SubType, PExtras> LA_2(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(
                *compressionHelper,
                j));
        LA_2 = LA_2.Step(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(
                *compressionHelper,
                j + 1));
        auto j2 = j + 2;

        LAInfoDeep<IterType, Float, SubType, PExtras> LA_(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(
                *compressionHelper,
                j - 1));
        LA_ = LA_.Step(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(
                *compressionHelper,
                j));
        auto j1 = j + 1;

        // l: 598
        IterType PeriodBegin = 0;
        IterType PeriodEnd = 0;
        bool PeriodDetected = false;
        bool PeriodDetected2 = false;

        // l:603
        for (; j2 < maxRefIteration || j1 < maxRefIteration; j1++, j2++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            PeriodDetected = LA_.Step(
                la_parameters,
                NewLA,
                PerturbationResults.GetComplex<SubType>(*compressionHelper, j1));

            if (PeriodDetected) {
                LAI_.NextStageLAIndex = j1;
                PeriodBegin = j1;
                PeriodEnd = PeriodBegin + Period;

                if (j1 + 1 >= maxRefIteration) {
                    LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(*compressionHelper, j1));
                    j1 += 1;
                }
                else {
                    LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(*compressionHelper, j1)).Step(
                            la_parameters,
                            PerturbationResults.GetComplex<SubType>(*compressionHelper, j1 + 1));
                    j1 += 2;
                }
                break;
            }

            LA_ = NewLA;

            // l:626
            if (j2 < maxRefIteration) {
                LAInfoDeep<IterType, Float, SubType, PExtras> NewLA2{};
                PeriodDetected2 = LA_2.Step(
                    la_parameters,
                    NewLA2,
                    PerturbationResults.GetComplex<SubType>(
                        *compressionHelper,
                        j2));

                if (PeriodDetected2) {
                    LAI_.NextStageLAIndex = j2;
                    PeriodBegin = j2;
                    PeriodEnd = PeriodBegin + Period;

                    auto jp1 = j2 + 1;

                    if (jp1 >= maxRefIteration) {
                        LA_2 = LAInfoDeep<IterType, Float, SubType, PExtras>(
                            la_parameters,
                            PerturbationResults.GetComplex<SubType>(*compressionHelper, j2));
                        j2++;
                    }
                    else {
                        LA_2 = LAInfoDeep<IterType, Float, SubType, PExtras>(
                            la_parameters,
                            PerturbationResults.GetComplex<SubType>(*compressionHelper, j2)).Step(
                                la_parameters,
                                PerturbationResults.GetComplex<SubType>(*compressionHelper, jp1));
                        j2 += 2;
                    }
                    break;
                }

                LA_2 = NewLA2;
            }
        }

        // l:652
        if (PeriodDetected2) {
            LA_ = LA_2;
            j = j2;
        }
        else if (PeriodDetected) {
            j = j1;
        }
        else {
            j = maxRefIteration;
        }

        // l:663
        //Just for protection
        if (ThreadID == LastThread || (j >= Begin && j < End)) {
            StartIndexPromise[ThreadID].set_value(j);
        }
        else {
            auto nextStart = StartIndexFuture[NextThread].get();

            if (nextStart < 0) {
                return;
            }

            //Abort the current thread and leave its task to the previous
            StartIndexPromise[ThreadID].set_value(nextStart);
            FinishIndex[ThreadID].store(-1, std::memory_order_release);
            return;
        }

        // l:679
        for (; j < maxRefIteration; j++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            PeriodDetected = LA_.Step(
                la_parameters,
                NewLA,
                PerturbationResults.GetComplex<SubType>(
                    *compressionHelper,
                    j));

            if (!PeriodDetected && j < PeriodEnd) {
                LA_ = NewLA;
                continue;
            }

            LAI_.StepLength = j - PeriodBegin;

            LA_.SetLAi(LAI_);
            LAsPerThread[ThreadID].PushBack(LA_);

            LAI_.NextStageLAIndex = j;
            PeriodBegin = j;
            PeriodEnd = PeriodBegin + Period;

            const IterType jp1{ j + 1 };
            const bool detected{ NewLA.DetectPeriod(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(*compressionHelper,jp1)) };

            if (detected || jp1 >= maxRefIteration) {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, j));
            }
            else {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(*compressionHelper, j)).Step(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(*compressionHelper, jp1));
                j++;
            }

            if (j > End) {
                if (ThreadID == LastThread) {
                    ::MessageBox(nullptr, L"Thread finished unexpected", L"", MB_OK | MB_APPLMODAL);
                    DebugBreak();
                }

                if (j >= maxRefIteration) {
                    break;
                }

                if (NextThread < StartIndexFuture.size()) {
                    auto nextStart = StartIndexFuture[NextThread].get();

                    if (nextStart < 0) { //The next tread had an exception
                        return;
                    }

                    if (static_cast<int64_t>(j) == nextStart - 1) {
                        j++;
                        break;
                    }
                    else if (static_cast<int64_t>(j) >= nextStart) {
                        NextThread++;
                    }
                }
            }
        }

        FinishIndex[ThreadID].store(static_cast<int64_t>(j), std::memory_order_release);

        LAI_.StepLength = j - PeriodBegin;
        LA_.SetLAi(LAI_);
        LastLAPerThread[ThreadID] = LA_;
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(Starter));
    for (uint32_t t = 1; t < ThreadCount; t++) {
        threads.push_back(std::make_unique<std::thread>(Worker, t));
    }

    for (uint32_t t = 0; t < ThreadCount; t++) {
        threads[t]->join();
    }

    {
        auto lastThreadToAdd = 0;

        auto index = 0;
        auto j = index;
        while ((index < threads.size() - 1) &&
               (FinishIndex[j] > StartIndexFuture[index + 1].get())) {
            index++; //Skip, if there is a missalignment
        }
        index++;

        for (; index < threads.size(); index++) {
            const auto& threadData = LAsPerThread[index];
            for (int k = 0; k < threadData.GetSize(); k++) {
                m_LAs.PushBack(threadData[k]);
            }

            //If this thread managed to search
            if (FinishIndex[index] > StartIndexFuture[index].get()) {
                lastThreadToAdd = index;
            }

            j = index;
            while ((index < threads.size() - 1) &&
                (FinishIndex[j] > StartIndexFuture[index + 1].get())) {
                index++; //Skip, if there is a missalignment
            }
        }

        m_LAs.PushBack(LastLAPerThread[lastThreadToAdd]);
    }

    m_LAStages[0].MacroItCount = LAsize();

    m_LAs.PushBack(LAInfoDeep<IterType, Float, SubType, PExtras>(
        la_parameters,
        PerturbationResults.GetComplex<SubType>(*compressionHelper, maxRefIteration)));

    return true;
}

//#pragma optimize( "", off )
template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType, PExtras>::CreateNewLAStage(
    const LAParameters& la_parameters,
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    IterType maxRefIteration) {

    LAInfoDeep<IterType, Float, SubType, PExtras> LA;
    LAInfoI<IterType> LAI{};
    IterType i;
    IterType PeriodBegin;
    IterType PeriodEnd;

    IterType PrevStage = m_LAStageCount - 1;
    IterType CurrentStage = m_LAStageCount;
    IterType PrevStageLAIndex = m_LAStages[PrevStage].LAIndex;
    IterType PrevStageMacroItCount = m_LAStages[PrevStage].MacroItCount;
    LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLA = m_LAs[PrevStageLAIndex];
    const LAInfoI<IterType> &PrevStageLAI = m_LAs[PrevStageLAIndex].GetLAi();

    LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLAp1 = m_LAs[PrevStageLAIndex + 1];
    const LAInfoI<IterType> PrevStageLAIp1 = m_LAs[PrevStageLAIndex + 1].GetLAi();

    IterType Period = 0;

    auto compressionHelper{ std::make_unique<CompressionHelper<IterType, Float, PExtras>>(PerturbationResults) };

    if (PrevStage > MaxLAStages) {
        ::MessageBox(nullptr, L"Too many stages :(", L"", MB_OK | MB_APPLMODAL);
    }

    if (CurrentStage >= MaxLAStages) {
        ::MessageBox(nullptr, L"Too many stages :(", L"", MB_OK | MB_APPLMODAL);
    }

    m_LAStages[CurrentStage].LAIndex = LAsize();

    LA = PrevStageLA.Composite(la_parameters, PrevStageLAp1);
    LAI.NextStageLAIndex = 0;
    i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
    IterType j;

    for (j = 2; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;

        NewLA = LAInfoDeep<IterType, Float, SubType, PExtras>();

        IterType PrevStageLAIndexj = PrevStageLAIndex + j;
        LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLAj = m_LAs[PrevStageLAIndexj];
        const LAInfoI<IterType> *PrevStageLAIj = &PrevStageLAj.GetLAi();
        bool PeriodDetected = LA.Composite(la_parameters, NewLA, PrevStageLAj);

        if (PeriodDetected) {
            if (PrevStageLAj.isLAThresholdZero()) break;
            Period = i;

            LAI.StepLength = Period;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            LAI.NextStageLAIndex = j;

            IterType PrevStageLAIndexjp1 = PrevStageLAIndexj + 1;
            LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLAjp1 = m_LAs[PrevStageLAIndexjp1];
            const LAInfoI<IterType> &PrevStageLAIjp1 = m_LAs[PrevStageLAIndexjp1].GetLAi();

            if (NewLA.DetectPeriod(la_parameters, PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
                i += PrevStageLAIj->StepLength;
                j++;
            }
            else {
                LA = PrevStageLAj.Composite(la_parameters, PrevStageLAjp1);
                i += PrevStageLAIj->StepLength + PrevStageLAIjp1.StepLength;
                j += 2;
            }
            break;
        }
        LA = NewLA;
        PrevStageLAIj = &m_LAs[PrevStageLAIndex + j].GetLAi();
        i += PrevStageLAIj->StepLength;
    }
    m_LAStageCount++;
    if (m_LAStageCount > MaxLAStages) {
        ::MessageBox(nullptr, L"Too many stages (2) :(", L"", MB_OK | MB_APPLMODAL);
    }

    PeriodBegin = Period;
    PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > PrevStageLAI.StepLength * lowBound) {
            LA = PrevStageLA.Composite(la_parameters, PrevStageLAp1);
            i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
            LAI.NextStageLAIndex = 0;

            j = 2;

            double Ratio = ((double)(maxRefIteration)) / PrevStageLAI.StepLength;
            double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
            Period = PrevStageLAI.StepLength * (IterType)std::round(std::pow(Ratio, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            LAInfoDeep<IterType, Float, SubType, PExtras> LA2(
                la_parameters, 
                PerturbationResults.GetComplex<SubType>(*compressionHelper, maxRefIteration));
            LA2.SetLAi({}); // mrenz This one is new
            m_LAs.PushBack(LA2);

            m_LAStages[CurrentStage].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > PrevStageLAI.StepLength * lowBound) {
        m_LAs.PopBack();

        LA = PrevStageLA.Composite(la_parameters, PrevStageLAp1);
        i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
        LAI.NextStageLAIndex = 0;

        j = 2;

        double Ratio = ((double)(Period)) / PrevStageLAI.StepLength;

        double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
        Period = PrevStageLAI.StepLength * ((IterType)std::round(std::pow(Ratio, 1.0 / NthRoot)));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    for (; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;

        NewLA = LAInfoDeep<IterType, Float, SubType, PExtras>();
        IterType PrevStageLAIndexj = PrevStageLAIndex + j;

        LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLAj = m_LAs[PrevStageLAIndexj];
        bool PeriodDetected = LA.Composite(la_parameters, NewLA, PrevStageLAj);

        if (PeriodDetected || i >= PeriodEnd) {
            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            m_LAs.PushBack(LA);

            LAI.NextStageLAIndex = j;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            LAInfoDeep<IterType, Float, SubType, PExtras> PrevStageLAjp1 = m_LAs[PrevStageLAIndexj + 1];

            if (NewLA.DetectPeriod(la_parameters, PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
            }
            else {
                LA = PrevStageLAj.Composite(la_parameters, PrevStageLAjp1);

                const LAInfoI<IterType> &PrevStageLAIj = m_LAs[PrevStageLAIndexj].GetLAi();
                i += PrevStageLAIj.StepLength;
                j++;
            }
        }
        else {
            LA = NewLA;
        }
        const LAInfoI<IterType> &PrevStageLAIj = m_LAs[PrevStageLAIndex + j].GetLAi();
        i += PrevStageLAIj.StepLength;
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    m_LAs.PushBack(LA);

    m_LAStages[CurrentStage].MacroItCount = LAsize() - m_LAStages[CurrentStage].LAIndex;

    LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
        la_parameters,
        PerturbationResults.GetComplex<SubType>(*compressionHelper, maxRefIteration));
    LA.SetLAi({});
    m_LAs.PushBack(LA);
    return true;
}
//#pragma optimize( "", on )

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
void LAReference<IterType, Float, SubType, PExtras>::GenerateApproximationData(
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    Float radius,
    bool UseSmallExponents) {

    const IterType maxRefIteration = (IterType)PerturbationResults.GetCountOrbitEntries() - 1;

    if (maxRefIteration == 0) {
        m_IsValid = false;
        return;
    }

    m_BenchmarkDataLA.StartTimer();

    bool PeriodDetected;
    if (m_LAParameters.GetThreading() == LAParameters::LAThreadingAlgorithm::MultiThreaded) {
        PeriodDetected = CreateLAFromOrbitMT(m_LAParameters, PerturbationResults, maxRefIteration);
    }
    else {
        PeriodDetected = CreateLAFromOrbit(m_LAParameters, PerturbationResults, maxRefIteration);
    }

    auto finish = [&]() {
        m_LAs.Trim();
        m_LAStages.Trim();
        m_BenchmarkDataLA.StopTimer();
    };

    if (!PeriodDetected) {
        finish();
        return;
    }

    while (true) {
        PeriodDetected = CreateNewLAStage(m_LAParameters, PerturbationResults, maxRefIteration);
        if (!PeriodDetected) break;
    }

    CreateATFromLA(radius, UseSmallExponents);
    m_IsValid = true;

    finish();
}

// TODO - this is a mess
#define InitializeApproximationData(IterType, T, SubType, PExtras) \
template void LAReference<IterType, T, SubType, PExtras>::GenerateApproximationData<T>( \
    const PerturbationResults<IterType, T, PExtras>& PerturbationResults, \
    T radius, \
    bool UseSmallExponents);

InitializeApproximationData(uint32_t, float, float, PerturbExtras::Disable);
InitializeApproximationData(uint64_t, float, float, PerturbExtras::Disable);
InitializeApproximationData(uint32_t, double, double, PerturbExtras::Disable);
InitializeApproximationData(uint64_t, double, double, PerturbExtras::Disable);
InitializeApproximationData(uint32_t, ::HDRFloat<float>, float, PerturbExtras::Disable);
InitializeApproximationData(uint64_t, ::HDRFloat<float>, float, PerturbExtras::Disable);
InitializeApproximationData(uint32_t, ::HDRFloat<double>, double, PerturbExtras::Disable);
InitializeApproximationData(uint64_t, ::HDRFloat<double>, double, PerturbExtras::Disable);

InitializeApproximationData(uint32_t, float, float, PerturbExtras::EnableCompression);
InitializeApproximationData(uint64_t, float, float, PerturbExtras::EnableCompression);
InitializeApproximationData(uint32_t, double, double, PerturbExtras::EnableCompression);
InitializeApproximationData(uint64_t, double, double, PerturbExtras::EnableCompression);
InitializeApproximationData(uint32_t, ::HDRFloat<float>, float, PerturbExtras::EnableCompression);
InitializeApproximationData(uint64_t, ::HDRFloat<float>, float, PerturbExtras::EnableCompression);
InitializeApproximationData(uint32_t, ::HDRFloat<double>, double, PerturbExtras::EnableCompression);
InitializeApproximationData(uint64_t, ::HDRFloat<double>, double, PerturbExtras::EnableCompression);

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
void LAReference<IterType, Float, SubType, PExtras>::CreateATFromLA(Float radius, bool UseSmallExponents) {
    Float SqrRadius;

    if constexpr (IsHDR) {
        SqrRadius = radius.square();
        SqrRadius.Reduce();
    }
    else {
        SqrRadius = radius * radius;
    }

    for (auto Stage = m_LAStageCount; Stage > 0; ) {
        Stage--;
        IterType LAIndex = m_LAStages[Stage].LAIndex;
        m_LAs[LAIndex].CreateAT(m_AT, m_LAs[LAIndex + 1], UseSmallExponents);
        m_AT.StepLength = m_LAs[LAIndex].GetLAi().StepLength;
        if (m_AT.StepLength > 0 && m_AT.Usable(SqrRadius)) {
            m_UseAT = true;
            return;
        }
    }
    m_UseAT = false;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
bool LAReference<IterType, Float, SubType, PExtras>::isLAStageInvalid(IterType LAIndex, FloatComplexT dc) {
    return (dc.chebychevNorm().compareToBothPositiveReduced((m_LAs[LAIndex]).getLAThresholdC()) >= 0);
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
IterType LAReference<IterType, Float, SubType, PExtras>::getLAIndex(IterType CurrentLAStage) {
    return m_LAStages[CurrentLAStage].LAIndex;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
IterType LAReference<IterType, Float, SubType, PExtras>::getMacroItCount(IterType CurrentLAStage) {
    return m_LAStages[CurrentLAStage].MacroItCount;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
LAstep<IterType, Float, SubType, PExtras>
LAReference<IterType, Float, SubType, PExtras>::getLA(
    IterType LAIndex,
    FloatComplexT dz,
    /*FloatComplexT dc, */ IterType j,
    IterType iterations,
    IterType max_iterations) {

    IterType LAIndexj = LAIndex + j;
    const LAInfoI<IterType> &LAIj = m_LAs[LAIndexj].GetLAi();

    LAstep<IterType, Float, SubType, PExtras> las;

    IterType l = LAIj.StepLength;
    bool usable = iterations + l <= max_iterations;
    //if (l < IterTypeMax) { // TODO - lame
    //    usable = iterations + l <= max_iterations;
    //}

    if (usable) {
        LAInfoDeep<IterType, Float, SubType, PExtras> &LAj = m_LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (FloatComplexT)m_LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }
    else {
        las = LAstep<IterType, Float, SubType, PExtras>();
        las.unusable = true;
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;

}

#define InitializeLAReference(IterType, T, SubType, PExtras) \
template class LAReference<IterType, T, SubType, PExtras>;

InitializeLAReference(uint32_t, HDRFloat<float>, float, PerturbExtras::Disable);
InitializeLAReference(uint64_t, HDRFloat<float>, float, PerturbExtras::Disable);
InitializeLAReference(uint32_t, HDRFloat<double>, double, PerturbExtras::Disable);
InitializeLAReference(uint64_t, HDRFloat<double>, double, PerturbExtras::Disable);

InitializeLAReference(uint32_t, HDRFloat<float>, float, PerturbExtras::EnableCompression);
InitializeLAReference(uint64_t, HDRFloat<float>, float, PerturbExtras::EnableCompression);
InitializeLAReference(uint32_t, HDRFloat<double>, double, PerturbExtras::EnableCompression);
InitializeLAReference(uint64_t, HDRFloat<double>, double, PerturbExtras::EnableCompression);
