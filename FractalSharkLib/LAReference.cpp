#include "stdafx.h"
#include "LAReference.h"
#include "LAParameters.h"
#include "RefOrbitCalc.h"
#include "PerturbationResults.h"

#include <deque>


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
    LA = LA.Step(la_parameters, PerturbationResults.GetComplex<SubType>(1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;
        bool PeriodDetected = LA.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(i));
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
                PerturbationResults.GetComplex<SubType>(i)).Step(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(i));
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
                PerturbationResults.GetComplex<SubType>(0)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(1));
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
                PerturbationResults.GetComplex<SubType>(maxRefIteration)));

            m_LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        m_LAs.PopBack();

        LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
            la_parameters,
            PerturbationResults.GetComplex<SubType>(0)).Step(
                la_parameters, 
                PerturbationResults.GetComplex<SubType>(1));
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
        const bool PeriodDetected{ LA.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(i)) };

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
        const bool detected{ NewLA.DetectPeriod(la_parameters, PerturbationResults.GetComplex<SubType>(ip1)) };

        if (detected || ip1 >= maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(i));
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(i)).Step(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(ip1));
            i++;
        }
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    m_LAs.PushBack(LA);

    m_LAStages[0].MacroItCount = LAsize();

    auto LA2 = LAInfoDeep<IterType, Float, SubType, PExtras>(
        la_parameters,
        PerturbationResults.GetComplex<SubType>(maxRefIteration));
    LA2.SetLAi({});
    m_LAs.PushBack(LA2);

    return true;
}

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType, PExtras>::CreateLAFromOrbitMT(
    const LAParameters& la_parameters,
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    IterType maxRefIteration) {

    {
        m_IsValid = false;
        m_LAStages.MutableResize(MaxLAStages);

        m_UseAT = false;
        m_LAStageCount = 0;

        m_LAStages[0].LAIndex = 0;
    }

    IterType Period = 0;

    LAInfoDeep<IterType, Float, SubType, PExtras> LA{ la_parameters, FloatComplexT() };
    LA = LA.Step(la_parameters, PerturbationResults.GetComplex<SubType>(1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType, PExtras> NewLA;
        bool PeriodDetected = LA.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(i));
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
                PerturbationResults.GetComplex<SubType>(i)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(i));
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
                PerturbationResults.GetComplex<SubType>(0)).Step(
                    la_parameters, 
                    PerturbationResults.GetComplex<SubType>(1));
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
                PerturbationResults.GetComplex<SubType>(maxRefIteration)));

            m_LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        m_LAs.PopBack();

        LA = LAInfoDeep<IterType, Float, SubType, PExtras>(
            la_parameters, 
            PerturbationResults.GetComplex<SubType>(0)).Step(
                la_parameters,
                PerturbationResults.GetComplex<SubType>(1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log2(static_cast<double>(maxRefIteration)) / periodDivisor);
        Period = (IterType)std::round(std::pow(static_cast<double>(maxRefIteration), 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    // TODO can we multithread this
    // const auto numPerThread = maxRefIteration / ;

    const int32_t ThreadCount = (int32_t)std::thread::hardware_concurrency();
    //const int32_t ThreadCount = (int32_t)3;
    std::deque<std::atomic_uint64_t> Start, Release;
    Start.resize(ThreadCount);
    Release.resize(ThreadCount);

    std::vector<std::vector<LAInfoDeep<IterType, Float, SubType, PExtras>>> ThreadLAs;
    ThreadLAs.resize(ThreadCount);
    //for (size_t i = 1; i < ThreadCount; i++) {
    //    ThreadLAs[i].reserve...
    //}

    volatile IterType numbers1[128] = { 0 };
    volatile IterType numbers2[128] = { 0 };

    auto Starter = [ // MOAR VARIABLES
            &i,
            &LA,
            &LAI,
            &PeriodBegin,
            &PeriodEnd,
            &PerturbationResults,
            &numbers1,
            &numbers2,
            Period,
            &Release,
            &Start,
            &ThreadLAs,
            ThreadCount,
            maxRefIteration,
            this
        ]() {

        const auto ThreadID = 0;
        const auto NextThread = 1;
        const auto LastThread = ThreadCount - 1;

        for (; i < maxRefIteration; i++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            const bool PeriodDetected{ LA.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(i)) };

            if (!PeriodDetected && i < PeriodEnd) {
                LA = NewLA;
                continue;
            }

            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            ThreadLAs[ThreadID].push_back(LA);

            LAI.NextStageLAIndex = i;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            const IterType ip1{ i + 1 };
            const bool detected{ NewLA.DetectPeriod(la_parameters, PerturbationResults.GetComplex<SubType>(ip1)) };

            if (detected || ip1 >= maxRefIteration) {
                LA = LAInfoDeep<IterType, Float, SubType, PExtras>(PerturbationResults.GetComplex<SubType>(i));
            }
            else {
                LA = LAInfoDeep<IterType, Float, SubType, PExtras>(PerturbationResults.GetComplex<SubType>(i)).Step(
                    la_parameters,
                    PerturbationResults.GetComplex<SubType>(ip1));
                i++;
            }

            for (size_t next_thread_eat = NextThread; next_thread_eat < ThreadCount; next_thread_eat++) {
                auto cur_end = static_cast<IterType>(
                    static_cast<IterTypeFull>(maxRefIteration) * next_thread_eat / ThreadCount);

                auto next_thread_upperbound = static_cast<IterType>(
                    static_cast<IterTypeFull>(maxRefIteration) * (next_thread_eat + 1) / ThreadCount);

                if (i <= cur_end) {
                    break;
                } else if (i > cur_end && i < next_thread_upperbound) {
                    // Wait for the next thread to finish the first bit.
                    while (Start[next_thread_eat].load(std::memory_order_acquire) == 0);
                    auto result = Start[next_thread_eat].load(std::memory_order_acquire);

                    if (i == result - 1) {

                        // Let the next thread go.
                        Release[next_thread_eat].store(1, std::memory_order_release);

                        i++;

                        if (ThreadID == LastThread) {
                            LAI.StepLength = i - PeriodBegin;

                            LA.SetLAi(LAI);
                            ThreadLAs[ThreadID].push_back(LA);
                        }

                        return;
                    } else if (i >= result) {
                        DebugBreak();
                        ::MessageBox(nullptr, L"I have another bug here yay :(", L"", MB_OK | MB_APPLMODAL);
                        //break;
                    }

                    break;
                }
                else {
                    // Lock out the next thread.
                    Release[next_thread_eat].store(2, std::memory_order_release);
                }
            }
        }

        ::MessageBox(nullptr, L"Thread 0 finished unexpected", L"", MB_OK | MB_APPLMODAL);
        DebugBreak();
    };

    auto Worker = [
            &PerturbationResults,
            &numbers1,
            &numbers2,
            Period,
            &Release,
            &Start,
            &ThreadLAs,
            ThreadCount,
            maxRefIteration,
            this
        ](int32_t ThreadID) {

        //const auto PrevThread = ThreadID - 1;
        const auto NextThread = ThreadID + 1;
        const auto LastThread = ThreadCount - 1;

        IterTypeFull intermediate_j = static_cast<IterTypeFull>(maxRefIteration) * ThreadID / ThreadCount;
        IterType j = static_cast<IterType>(intermediate_j);

        //if (PrevThread > 0) {
        //    while (Start[PrevThread].load(std::memory_order_acquire) == 0);
        //    auto result = Start[PrevThread].load(std::memory_order_acquire);

        //    if (result > j && result != UINT64_MAX) {
        //        uint64_t expected = 0;
        //        bool res = Start[ThreadID].compare_exchange_strong(expected, UINT64_MAX);
        //        if (!res) {
        //            DebugBreak();
        //            ::MessageBox(nullptr, L"Confused thread situation", L"", MB_OK | MB_APPLMODAL);
        //        }

        //        return;
        //    }
        //}

        IterTypeFull intermediate_end = static_cast<IterTypeFull>(maxRefIteration) * NextThread / ThreadCount;
        IterType End = static_cast<IterType>(intermediate_end);

        LAInfoDeep<IterType, Float, SubType, PExtras> LA_(PerturbationResults.GetComplex<SubType>(j));
        LA_ = LA_.Step(la_parameters, PerturbationResults.GetComplex<SubType>(j + 1));
        LAInfoI<IterType> LAI_;
        LAI_.NextStageLAIndex = j;
        j += 2;

        IterType PeriodBegin = 0;
        IterType PeriodEnd = 0;

        numbers1[ThreadID] = j;
        numbers2[ThreadID] = End;

        //auto result = Start[NextThread].load(std::memory_order_acquire);

        for (; j < maxRefIteration; j++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            bool PeriodDetected = LA_.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(j));

            if (!PeriodDetected) {
                LA_ = NewLA;
                continue;
            }

            LAI_.NextStageLAIndex = j;
            PeriodBegin = j;
            PeriodEnd = PeriodBegin + Period;

            if (j + 1 < maxRefIteration) {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    PerturbationResults.GetComplex<SubType>(j)).Step(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(j + 1));
                j += 2;
            }
            else {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    PerturbationResults.GetComplex<SubType>(j));
                j += 1;
            }
            break;
        }

        uint64_t expected = 0;
        bool res = Start[ThreadID].compare_exchange_strong(expected, j);
        if (!res) {
            DebugBreak();
            ::MessageBox(nullptr, L"Confused thread situation", L"", MB_OK | MB_APPLMODAL);
        }

        {
            for (;;) {
                auto result = Release[ThreadID].load(std::memory_order_acquire);
                if (result == 2) { // 2 = don't use this thread, the prior one went overboard
                    return;
                }
                else if (result == 1) { // 1 == go
                    break;
                }
                else { // 0 = keep waiting.
                    continue;
                }
            }
        }

        //auto next_thread_eat = NextThread;
        //while (j > End) {
        //    End = static_cast<IterType>(
        //        static_cast<IterTypeFull>(maxRefIteration) * next_thread_eat / ThreadCount);
        //    if (j > test) {
        //        uint64_t expected = 0;
        //        bool res = Start[next_thread_eat].compare_exchange_strong(expected, UINT64_MAX);
        //        if (!res) {
        //            DebugBreak();
        //            ::MessageBox(nullptr, L"Confused thread situation", L"", MB_OK | MB_APPLMODAL);
        //        }

        //        next_thread_eat++;
        //    }
        //}

        for (size_t next_thread_eat = NextThread; next_thread_eat < ThreadCount; next_thread_eat++) {
            auto cur_end = static_cast<IterType>(
                static_cast<IterTypeFull>(maxRefIteration) * next_thread_eat / ThreadCount);

            if (j < cur_end) {
                // Let the next thread go.
                Release[next_thread_eat].store(1, std::memory_order_release);
                break;
            }
            else {
                // Lock out the next thread.
                Release[next_thread_eat].store(2, std::memory_order_release);
            }
        }

        //Sleep(10000000);
        for (; j < maxRefIteration; j++) {
            LAInfoDeep<IterType, Float, SubType, PExtras> NewLA{};
            const bool PeriodDetected{ LA_.Step(la_parameters, NewLA, PerturbationResults.GetComplex<SubType>(j)) };

            if (!PeriodDetected && j < PeriodEnd) {
                LA_ = NewLA;
                continue;
            }

            LAI_.StepLength = j - PeriodBegin;

            LA_.SetLAi(LAI_);
            ThreadLAs[ThreadID].push_back(LA_);

            LAI_.NextStageLAIndex = j;
            PeriodBegin = j;
            PeriodEnd = PeriodBegin + Period;

            const IterType jp1{ j + 1 };
            const bool detected{ NewLA.DetectPeriod(la_parameters, PerturbationResults.GetComplex<SubType>(jp1)) };

            if (detected || jp1 >= maxRefIteration) {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    PerturbationResults.GetComplex<SubType>(j));
            }
            else {
                LA_ = LAInfoDeep<IterType, Float, SubType, PExtras>(
                    PerturbationResults.GetComplex<SubType>(j)).Step(
                        la_parameters,
                        PerturbationResults.GetComplex<SubType>(jp1));
                j++;
            }

            //if (j > End) {
            //    if (ThreadID == LastThread) {
            //        DebugBreak();
            //        ::MessageBox(nullptr, L"I have another bug here", L"", MB_OK | MB_APPLMODAL);
            //    }
            //    while (Start[NextThread].load(std::memory_order_acquire) == 0);
            //    auto result = Start[NextThread].load(std::memory_order_acquire);
            //    if (j == result - 1) {
            //        j++;
            //        break;
            //    }
            //    else if (j >= result) {
            //        DebugBreak();
            //        ::MessageBox(nullptr, L"I have another bug here yay :(", L"", MB_OK | MB_APPLMODAL);
            //        //break;
            //    }
            //}

            if (ThreadID >= LastThread) {
                DebugBreak();
                ::MessageBox(nullptr, L"I have another bug here", L"", MB_OK | MB_APPLMODAL);
            }

            for (size_t next_thread_eat = NextThread; next_thread_eat < ThreadCount; next_thread_eat++) {

                auto cur_end = static_cast<IterType>(
                    static_cast<IterTypeFull>(maxRefIteration) * next_thread_eat / ThreadCount);

                auto next_thread_eat_end = static_cast<IterType>(
                    static_cast<IterTypeFull>(maxRefIteration) * (next_thread_eat + 1) / ThreadCount);

                if (j <= cur_end) {
                    break;
                } else if (j > cur_end && j < next_thread_eat_end) {

                    // Let the next thread go.
                    Release[next_thread_eat].store(1, std::memory_order_release);

                    // Wait for the next thread to finish the first bit.
                    while (Start[next_thread_eat].load(std::memory_order_acquire) == 0);

                    auto result = Start[next_thread_eat].load(std::memory_order_acquire);
                    if (j == result - 1) {
                        j++;

                        if (ThreadID == LastThread) {
                            LAI_.StepLength = j - PeriodBegin;

                            LA_.SetLAi(LAI_);
                            ThreadLAs[ThreadID].push_back(LA_);
                        }

                        return;
                    }
                    else if (j >= result) {
                        // So for next time, this fails.  Fuck if I know what I'm doing
                        DebugBreak();
                        ::MessageBox(nullptr, L"I have another bug here yay :(", L"", MB_OK | MB_APPLMODAL);
                        //break;
                    }
                }
                else {
                    // Lock out the next thread.
                    Release[next_thread_eat].store(2, std::memory_order_release);
                }
            }
        }

        //if (ThreadID == LastThread) {
        //    LAI_.StepLength = j - PeriodBegin;

        //    LA_.SetLAi(LAI_);
        //    ThreadLAs[ThreadID].push_back(LA_);
        //}



        //}
        //else {
        //    uint64_t expected = 0;
        //    bool res = Start[ThreadID].compare_exchange_strong(expected, j);
        //    if (!res) {
        //        DebugBreak();
        //        ::MessageBox(nullptr, L"Confused thread situation", L"", MB_OK | MB_APPLMODAL);
        //    }

        //    ThreadLAs[ThreadID].clear();

        //    if (ThreadID == LastThread) {
        //        LAI_.StepLength = j - PeriodBegin;

        //        LA_.SetLAi(LAI_);
        //        ThreadLAs[ThreadID].push_back(LA_);
        //    }
        //}
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(Starter));
    for (int32_t t = 1; t < ThreadCount; t++) {
        threads.push_back(std::make_unique<std::thread>(Worker, t));
    }

    for (int32_t t = 0; t < ThreadCount; t++) {
        threads[t]->join();
    }

    for (auto t = 0; t < ThreadLAs.size(); t++) {
        for (auto k = 0; k < ThreadLAs[t].size(); k++) {
            m_LAs.PushBack(ThreadLAs[t][k]);
        }
    }

    m_LAStages[0].MacroItCount = LAsize();

    m_LAs.PushBack(LAInfoDeep<IterType, Float, SubType, PExtras>(PerturbationResults.GetComplex<SubType>(maxRefIteration)));

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
                PerturbationResults.GetComplex<SubType>(maxRefIteration));
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
        PerturbationResults.GetComplex<SubType>(maxRefIteration));
    LA.SetLAi({});
    m_LAs.PushBack(LA);
    return true;
}
//#pragma optimize( "", on )

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
template<typename PerturbType>
void LAReference<IterType, Float, SubType, PExtras>::GenerateApproximationData(
    const LAParameters& la_parameters,
    const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
    Float radius,
    IterType maxRefIteration,
    bool UseSmallExponents) {

    if (maxRefIteration == 0) {
        m_IsValid = false;
        return;
    }

    m_BenchmarkDataLA.StartTimer();

    //bool PeriodDetected = CreateLAFromOrbitMT(PerturbationResults, maxRefIteration);
    bool PeriodDetected = CreateLAFromOrbit(la_parameters, PerturbationResults, maxRefIteration);
    if (!PeriodDetected) return;

    while (true) {
        PeriodDetected = CreateNewLAStage(la_parameters, PerturbationResults, maxRefIteration);
        if (!PeriodDetected) break;
    }

    CreateATFromLA(radius, UseSmallExponents);
    m_IsValid = true;

    m_BenchmarkDataLA.StopTimer();
}

// TODO - this is a mess
#define InitializeApproximationData(IterType, T, SubType, PExtras) \
template void LAReference<IterType, T, SubType, PExtras>::GenerateApproximationData<T>( \
    const LAParameters &la_parameters, \
    const PerturbationResults<IterType, T, PExtras>& PerturbationResults, \
    T radius, \
    IterType maxRefIteration, \
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
