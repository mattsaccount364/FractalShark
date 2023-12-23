#include "stdafx.h"
#include "LAReference.h"
#include "RefOrbitCalc.h"
#include "PerturbationResults.h"

//#include <Windows.h>

template<typename IterType, class Float, class SubType>
const SubType LAReference<IterType, Float, SubType>::log16 = (SubType)std::log(16);

template<typename IterType, class Float, class SubType>
IterType LAReference<IterType, Float, SubType>::LAsize() {
    return (IterType)LAs.size();
}

template<typename IterType, class Float, class SubType>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType>::CreateLAFromOrbit(
    const PerturbationResults<IterType, PerturbType, PerturbExtras::Disable>& PerturbationResults,
    IterType maxRefIteration) {

    {
        isValid = false;
        LAStages.resize(MaxLAStages);

        //LAs.reserve(maxRefIteration);

        UseAT = false;
        LAStageCount = 0;

        LAStages[0].LAIndex = 0;
    }

    IterType Period = 0;

    LAInfoDeep<IterType, Float, SubType> LA{ FloatComplexT()};
    LA = LA.Step(PerturbationResults.GetComplex<SubType>(1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType> NewLA;
        bool PeriodDetected = LA.Step(NewLA, PerturbationResults.GetComplex<SubType>(i));
        if (!PeriodDetected) {
            LA = NewLA;
            continue;
        }

        Period = i;
        LAI.StepLength = Period;

        LA.SetLAi(LAI);
        LAs.push_back(LA);

        LAI.NextStageLAIndex = i;

        if (i + 1 < maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i)).Step(PerturbationResults.GetComplex<SubType>(i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i));
            i += 1;
        }
        break;
    }

    LAStageCount = 1;

    IterType PeriodBegin = Period;
    IterType PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(0)).Step(PerturbationResults.GetComplex<SubType>(1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log(maxRefIteration) / log16);
            Period = (IterType)std::round(std::pow(maxRefIteration, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAs.push_back(LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(maxRefIteration)));

            LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        LAs.pop_back();

        LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(0)).Step(PerturbationResults.GetComplex<SubType>(1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log(Period) / log16);
        Period = (IterType)std::round(std::pow(Period, 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    // TODO can we multithread this
    // const auto numPerThread = maxRefIteration / std::thread::hardware_concurrency();

    for (; i < maxRefIteration; i++) {
        LAInfoDeep<IterType, Float, SubType> NewLA{};
        const bool PeriodDetected{ LA.Step(NewLA, PerturbationResults.GetComplex<SubType>(i)) };

        if (!PeriodDetected && i < PeriodEnd) {
            LA = NewLA;
            continue;
        }

        LAI.StepLength = i - PeriodBegin;

        LA.SetLAi(LAI);
        LAs.push_back(LA);

        LAI.NextStageLAIndex = i;
        PeriodBegin = i;
        PeriodEnd = PeriodBegin + Period;

        const IterType ip1{ i + 1 };
        const bool detected{ NewLA.DetectPeriod(PerturbationResults.GetComplex<SubType>(ip1)) };

        if (detected || ip1 >= maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i));
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i)).Step(
                PerturbationResults.GetComplex<SubType>(ip1));
            i++;
        }
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    LAs.push_back(LA);

    LAStages[0].MacroItCount = LAsize();

    auto LA2 = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(maxRefIteration));
    LA2.SetLAi({});
    LAs.push_back(LA2);

    return true;
}

template<typename IterType, class Float, class SubType>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType>::CreateLAFromOrbitMT(
    const PerturbationResults<IterType, PerturbType, PerturbExtras::Disable>& PerturbationResults,
    IterType maxRefIteration) {

    {
        isValid = false;
        LAStages.resize(MaxLAStages);

        LAs.reserve(maxRefIteration);

        UseAT = false;
        LAStageCount = 0;

        LAStages[0].LAIndex = 0;
    }

    IterType Period = 0;

    LAInfoDeep<IterType, Float, SubType> LA{ FloatComplexT() };
    LA = LA.Step(PerturbationResults.GetComplex<SubType>(1));

    LAInfoI<IterType> LAI{};
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    IterType i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<IterType, Float, SubType> NewLA;
        bool PeriodDetected = LA.Step(NewLA, PerturbationResults.GetComplex<SubType>(i));
        if (!PeriodDetected) {
            LA = NewLA;
            continue;
        }

        Period = i;
        LAI.StepLength = Period;

        LA.SetLAi(LAI);
        LAs.push_back(LA);

        LAI.NextStageLAIndex = i;

        if (i + 1 < maxRefIteration) {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i)).Step(PerturbationResults.GetComplex<SubType>(i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i));
            i += 1;
        }
        break;
    }

    LAStageCount = 1;

    IterType PeriodBegin = Period;
    IterType PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(0)).Step(PerturbationResults.GetComplex<SubType>(1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log(maxRefIteration) / log16);
            Period = (IterType)std::round(std::pow(maxRefIteration, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAs.push_back(LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(maxRefIteration)));

            LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        LAs.pop_back();

        LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(0)).Step(PerturbationResults.GetComplex<SubType>(1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log(Period) / log16);
        Period = (IterType)std::round(std::pow(Period, 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    // TODO can we multithread this
    // const auto numPerThread = maxRefIteration / ;

    const int32_t ThreadCount = (int32_t)std::thread::hardware_concurrency();
    std::vector<IterType> Start;
    Start.resize(ThreadCount);

    auto Starter = [&]() {
        for (; i < maxRefIteration; i++) {
            LAInfoDeep<IterType, Float, SubType> NewLA{};
            const bool PeriodDetected{ LA.Step(NewLA, PerturbationResults.GetComplex<SubType>(i)) };

            if (!PeriodDetected && i < PeriodEnd) {
                LA = NewLA;
                continue;
            }

            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAI.NextStageLAIndex = i;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            const IterType ip1{ i + 1 };
            const bool detected{ NewLA.DetectPeriod(PerturbationResults.GetComplex<SubType>(ip1)) };

            if (detected || ip1 >= maxRefIteration) {
                LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i));
            }
            else {
                LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(i)).Step(
                    PerturbationResults.GetComplex<SubType>(ip1));
                i++;
            }

            if (i > maxRefIteration / ThreadCount) {
                while (!Start[1]);

                if (i == Start[1] - 1) {
                    i++;
                    break;
                }
                else if (i >= Start[1]) {
                    ::MessageBox(NULL, L"Confused thread situation", L"", MB_OK);
                }
            }
        }
    };

    std::vector<std::vector<LAInfoDeep<IterType, Float, SubType>>> ThreadLAs;
    ThreadLAs.resize(ThreadCount);
    //for (size_t i = 1; i < ThreadCount; i++) {
    //    ThreadLAs[i].reserve...
    //}

    volatile IterType numbers1[128] = { 0 };
    volatile IterType numbers2[128] = { 0 };

    auto Worker = [&numbers1, &numbers2, Period, &Start, &ThreadLAs, ThreadCount, maxRefIteration, this](int32_t ThreadID) {
        IterType j = maxRefIteration * ThreadID / ThreadCount;
        IterType End = maxRefIteration * (ThreadID + 1) / ThreadCount;
        LAInfoDeep<IterType, Float, SubType> LA_ = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(j));
        LA_ = LA_.Step(PerturbationResults.GetComplex<SubType>(j + 1));
        LAInfoI<IterType> LAI_;
        LAI_.NextStageLAIndex = j;
        j += 2;

        IterType PeriodBegin = 0;
        IterType PeriodEnd = 0;

        numbers1[ThreadID] = j;
        numbers2[ThreadID] = End;

        for (; j < maxRefIteration; j++) {
            LAInfoDeep<IterType, Float, SubType> NewLA;
            bool PeriodDetected = LA_.Step(NewLA, PerturbationResults.GetComplex<SubType>(j));

            if (PeriodDetected) {
                LAI_.NextStageLAIndex = j;
                PeriodBegin = j;
                PeriodEnd = PeriodBegin + Period;

                if (j + 1 < maxRefIteration) {
                    LA_ = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(j)).Step(PerturbationResults.GetComplex<SubType>(j + 1));
                    j += 2;
                }
                else {
                    LA_ = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(j));
                    j += 1;
                }
                break;
            }
            LA_ = NewLA;
        }

        Start[ThreadID] = j;
        for (; j < maxRefIteration; j++) {
            LAInfoDeep<IterType, Float, SubType> NewLA{};
            const bool PeriodDetected{ LA_.Step(NewLA, PerturbationResults.GetComplex<SubType>(j)) };

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

            const IterType ip1{ j + 1 };
            const bool detected{ NewLA.DetectPeriod(PerturbationResults.GetComplex<SubType>(ip1)) };

            if (detected || ip1 >= maxRefIteration) {
                LA_ = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(j));
            }
            else {
                LA_ = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(j)).Step(
                    PerturbationResults.GetComplex<SubType>(ip1));
                j++;
            }

            if (j > End) {
                if (ThreadID == ThreadCount - 1) {
                    DebugBreak();
                    ::MessageBox(NULL, L"I have another bug here", L"", MB_OK);
                }
                while (!Start[ThreadID + 1]);
                if (j == Start[ThreadID + 1] - 1) {
                    j++;
                    break;
                }
                else if (j >= Start[ThreadID + 1]) {
                    DebugBreak();
                    ::MessageBox(NULL, L"I have another bug here yay :(", L"", MB_OK);
                    //break;
                }
            }
        }

        if (ThreadID == ThreadCount - 1) {
            LAI_.StepLength = j - PeriodBegin;

            LA_.SetLAi(LAI_);
            LAs.push_back(LA_);

            auto LA2 = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(maxRefIteration));
            LA2.SetLAi({});
            LAs.push_back(LA2);
        }
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(Starter));
    for (int32_t t = 1; t < ThreadCount; t++) {
        threads.push_back(std::make_unique<std::thread>(Worker, t));
    }

    for (int32_t t = 0; t < ThreadCount; t++) {
        threads[t]->join();
    }

    // TODO concat

    LAStages[0].MacroItCount = LAsize();

    return true;
}

//#pragma optimize( "", off )
template<typename IterType, class Float, class SubType>
template<typename PerturbType>
bool LAReference<IterType, Float, SubType>::CreateNewLAStage(
    const PerturbationResults<IterType, PerturbType, PerturbExtras::Disable>& PerturbationResults,
    IterType maxRefIteration) {

    LAInfoDeep<IterType, Float, SubType> LA;
    LAInfoI<IterType> LAI{};
    IterType i;
    IterType PeriodBegin;
    IterType PeriodEnd;

    IterType PrevStage = LAStageCount - 1;
    IterType CurrentStage = LAStageCount;
    IterType PrevStageLAIndex = LAStages[PrevStage].LAIndex;
    IterType PrevStageMacroItCount = LAStages[PrevStage].MacroItCount;
    LAInfoDeep<IterType, Float, SubType> PrevStageLA = LAs[PrevStageLAIndex];
    const LAInfoI<IterType> &PrevStageLAI = LAs[PrevStageLAIndex].GetLAi();

    LAInfoDeep<IterType, Float, SubType> PrevStageLAp1 = LAs[PrevStageLAIndex + 1];
    const LAInfoI<IterType> PrevStageLAIp1 = LAs[PrevStageLAIndex + 1].GetLAi();

    IterType Period = 0;

    if (PrevStage > MaxLAStages) {
        ::MessageBox(NULL, L"Too many stages :(", L"", MB_OK);
    }

    if (CurrentStage >= MaxLAStages) {
        ::MessageBox(NULL, L"Too many stages :(", L"", MB_OK);
    }

    LAStages[CurrentStage].LAIndex = LAsize();

    LA = PrevStageLA.Composite(PrevStageLAp1);
    LAI.NextStageLAIndex = 0;
    i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
    IterType j;

    for (j = 2; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<IterType, Float, SubType> NewLA;

        NewLA = LAInfoDeep<IterType, Float, SubType>();

        IterType PrevStageLAIndexj = PrevStageLAIndex + j;
        LAInfoDeep<IterType, Float, SubType> PrevStageLAj = LAs[PrevStageLAIndexj];
        const LAInfoI<IterType> *PrevStageLAIj = &PrevStageLAj.GetLAi();
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected) {
            if (PrevStageLAj.isLAThresholdZero()) break;
            Period = i;

            LAI.StepLength = Period;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAI.NextStageLAIndex = j;

            IterType PrevStageLAIndexjp1 = PrevStageLAIndexj + 1;
            LAInfoDeep<IterType, Float, SubType> PrevStageLAjp1 = LAs[PrevStageLAIndexjp1];
            const LAInfoI<IterType> &PrevStageLAIjp1 = LAs[PrevStageLAIndexjp1].GetLAi();

            if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
                i += PrevStageLAIj->StepLength;
                j++;
            }
            else {
                LA = PrevStageLAj.Composite(PrevStageLAjp1);
                i += PrevStageLAIj->StepLength + PrevStageLAIjp1.StepLength;
                j += 2;
            }
            break;
        }
        LA = NewLA;
        PrevStageLAIj = &LAs[PrevStageLAIndex + j].GetLAi();
        i += PrevStageLAIj->StepLength;
    }
    LAStageCount++;
    if (LAStageCount > MaxLAStages) {
        ::MessageBox(NULL, L"Too many stages (2) :(", L"", MB_OK);
    }

    PeriodBegin = Period;
    PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > PrevStageLAI.StepLength * lowBound) {
            LA = PrevStageLA.Composite(PrevStageLAp1);
            i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
            LAI.NextStageLAIndex = 0;

            j = 2;

            double Ratio = ((double)(maxRefIteration)) / PrevStageLAI.StepLength;
            double NthRoot = std::round(std::log(Ratio) / log16); // log16
            Period = PrevStageLAI.StepLength * (IterType)std::round(std::pow(Ratio, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAInfoDeep<IterType, Float, SubType> LA2(PerturbationResults.GetComplex<SubType>(maxRefIteration));
            LA2.SetLAi({}); // mrenz This one is new
            LAs.push_back(LA2);

            LAStages[CurrentStage].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > PrevStageLAI.StepLength * lowBound) {
        LAs.pop_back();

        LA = PrevStageLA.Composite(PrevStageLAp1);
        i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
        LAI.NextStageLAIndex = 0;

        j = 2;

        double Ratio = ((double)(Period)) / PrevStageLAI.StepLength;

        double NthRoot = std::round(std::log(Ratio) / log16);
        Period = PrevStageLAI.StepLength * ((IterType)std::round(std::pow(Ratio, 1.0 / NthRoot)));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    for (; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<IterType, Float, SubType> NewLA;

        NewLA = LAInfoDeep<IterType, Float, SubType>();
        IterType PrevStageLAIndexj = PrevStageLAIndex + j;

        LAInfoDeep<IterType, Float, SubType> PrevStageLAj = LAs[PrevStageLAIndexj];
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected || i >= PeriodEnd) {
            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAI.NextStageLAIndex = j;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            LAInfoDeep<IterType, Float, SubType> PrevStageLAjp1 = LAs[PrevStageLAIndexj + 1];

            if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
            }
            else {
                LA = PrevStageLAj.Composite(PrevStageLAjp1);

                const LAInfoI<IterType> &PrevStageLAIj = LAs[PrevStageLAIndexj].GetLAi();
                i += PrevStageLAIj.StepLength;
                j++;
            }
        }
        else {
            LA = NewLA;
        }
        const LAInfoI<IterType> &PrevStageLAIj = LAs[PrevStageLAIndex + j].GetLAi();
        i += PrevStageLAIj.StepLength;
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    LAs.push_back(LA);

    LAStages[CurrentStage].MacroItCount = LAsize() - LAStages[CurrentStage].LAIndex;

    LA = LAInfoDeep<IterType, Float, SubType>(PerturbationResults.GetComplex<SubType>(maxRefIteration));
    LA.SetLAi({});
    LAs.push_back(LA);
    return true;
}
//#pragma optimize( "", on )

template<typename IterType, class Float, class SubType>
template<typename PerturbType>
void LAReference<IterType, Float, SubType>::GenerateApproximationData(
    const PerturbationResults<IterType, PerturbType, PerturbExtras::Disable>& PerturbationResults,
    Float radius,
    IterType maxRefIteration,
    bool UseSmallExponents) {

    if (maxRefIteration == 0) {
        isValid = false;
        return;
    }

    //bool PeriodDetected = CreateLAFromOrbitMT(maxRefIteration);
    bool PeriodDetected = CreateLAFromOrbit(PerturbationResults, maxRefIteration);
    if (!PeriodDetected) return;

    while (true) {
        PeriodDetected = CreateNewLAStage(PerturbationResults, maxRefIteration);
        if (!PeriodDetected) break;
    }

    CreateATFromLA(radius, UseSmallExponents);
    isValid = true;
}

// TODO - this is a mess
template void LAReference<uint32_t, float, float>::GenerateApproximationData<float>(
    const PerturbationResults<uint32_t, float, PerturbExtras::Disable>& PerturbationResults,
    float radius,
    uint32_t maxRefIteration,
    bool UseSmallExponents);
//template void LAReference<uint32_t, float, float>::GenerateApproximationData<double>(
//    const PerturbationResults<uint32_t, double, PerturbExtras::Disable>& PerturbationResults,
//    float radius,
//    uint32_t maxRefIteration,
//    bool UseSmallExponents);
//
template void LAReference<uint64_t, float, float>::GenerateApproximationData<float>(
    const PerturbationResults<uint64_t, float, PerturbExtras::Disable>& PerturbationResults,
    float radius,
    uint64_t maxRefIteration,
    bool UseSmallExponents);
//template void LAReference<uint64_t, float, float>::GenerateApproximationData<double>(
//    const PerturbationResults<uint64_t, double, PerturbExtras::Disable>& PerturbationResults,
//    float radius,
//    uint64_t maxRefIteration,
//    bool UseSmallExponents);
//
//template void LAReference<uint32_t, double, double>::GenerateApproximationData<float>(
//    const PerturbationResults<uint32_t, float, PerturbExtras::Disable>& PerturbationResults,
//    double radius,
//    uint32_t maxRefIteration,
//    bool UseSmallExponents);
template void LAReference<uint32_t, double, double>::GenerateApproximationData<double>(
    const PerturbationResults<uint32_t, double, PerturbExtras::Disable>& PerturbationResults,
    double radius,
    uint32_t maxRefIteration,
    bool UseSmallExponents);
//
//template void LAReference<uint64_t, double, double>::GenerateApproximationData<float>(
//    const PerturbationResults<uint64_t, float, PerturbExtras::Disable>& PerturbationResults,
//    double radius,
//    uint64_t maxRefIteration,
//    bool UseSmallExponents);
template void LAReference<uint64_t, double, double>::GenerateApproximationData<double>(
    const PerturbationResults<uint64_t, double, PerturbExtras::Disable>& PerturbationResults,
    double radius,
    uint64_t maxRefIteration,
    bool UseSmallExponents);

///

template void LAReference<uint32_t, ::HDRFloat<float>, float>::GenerateApproximationData<HDRFloat<float>>(
    const PerturbationResults<uint32_t, ::HDRFloat<float>, PerturbExtras::Disable>& PerturbationResults,
    ::HDRFloat<float> radius,
    uint32_t maxRefIteration,
    bool UseSmallExponents);
//template void LAReference<uint32_t, ::HDRFloat<float>, float>::GenerateApproximationData<HDRFloat<double>>(
//    const PerturbationResults<uint32_t, ::HDRFloat<double>, PerturbExtras::Disable>& PerturbationResults,
//    ::HDRFloat<float> radius,
//    uint32_t maxRefIteration,
//    bool UseSmallExponents);

template void LAReference<uint64_t, ::HDRFloat<float>, float>::GenerateApproximationData<HDRFloat<float>>(
    const PerturbationResults<uint64_t, ::HDRFloat<float>, PerturbExtras::Disable>& PerturbationResults,
    ::HDRFloat<float> radius,
    uint64_t maxRefIteration,
    bool UseSmallExponents);
//template void LAReference<uint64_t, ::HDRFloat<float>, float>::GenerateApproximationData<HDRFloat<double>>(
//    const PerturbationResults<uint64_t, ::HDRFloat<double>, PerturbExtras::Disable>& PerturbationResults,
//    ::HDRFloat<float> radius,
//    uint64_t maxRefIteration,
//    bool UseSmallExponents);

//template void LAReference<uint32_t, ::HDRFloat<double>, double>::GenerateApproximationData<HDRFloat<float>>(
//    const PerturbationResults<uint32_t, ::HDRFloat<float>, PerturbExtras::Disable>& PerturbationResults,
//    ::HDRFloat<double> radius,
//    uint32_t maxRefIteration,
//    bool UseSmallExponents);
template void LAReference<uint32_t, ::HDRFloat<double>, double>::GenerateApproximationData<HDRFloat<double>>(
    const PerturbationResults<uint32_t, ::HDRFloat<double>, PerturbExtras::Disable>& PerturbationResults,
    ::HDRFloat<double> radius,
    uint32_t maxRefIteration,
    bool UseSmallExponents);

//template void LAReference<uint64_t, ::HDRFloat<double>, double>::GenerateApproximationData<HDRFloat<float>>(
//    const PerturbationResults<uint64_t, ::HDRFloat<float>, PerturbExtras::Disable>& PerturbationResults,
//    ::HDRFloat<double> radius,
//    uint64_t maxRefIteration,
//    bool UseSmallExponents);
template void LAReference<uint64_t, ::HDRFloat<double>, double>::GenerateApproximationData<HDRFloat<double>>(
    const PerturbationResults<uint64_t, ::HDRFloat<double>, PerturbExtras::Disable>& PerturbationResults,
    ::HDRFloat<double> radius,
    uint64_t maxRefIteration,
    bool UseSmallExponents);

template<typename IterType, class Float, class SubType>
void LAReference<IterType, Float, SubType>::CreateATFromLA(Float radius, bool UseSmallExponents) {
    Float SqrRadius;

    if constexpr (IsHDR) {
        SqrRadius = radius.square();
        SqrRadius.Reduce();
    }
    else {
        SqrRadius = radius * radius;
    }

    for (auto Stage = LAStageCount; Stage > 0; ) {
        Stage--;
        IterType LAIndex = LAStages[Stage].LAIndex;
        LAs[LAIndex].CreateAT(AT, LAs[LAIndex + 1], UseSmallExponents);
        AT.StepLength = LAs[LAIndex].GetLAi().StepLength;
        if (AT.StepLength > 0 && AT.Usable(SqrRadius)) {
            UseAT = true;
            return;
        }
    }
    UseAT = false;
}

template<typename IterType, class Float, class SubType>
bool LAReference<IterType, Float, SubType>::isLAStageInvalid(IterType LAIndex, FloatComplexT dc) {
    return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
}

template<typename IterType, class Float, class SubType>
IterType LAReference<IterType, Float, SubType>::getLAIndex(IterType CurrentLAStage) {
    return LAStages[CurrentLAStage].LAIndex;
}

template<typename IterType, class Float, class SubType>
IterType LAReference<IterType, Float, SubType>::getMacroItCount(IterType CurrentLAStage) {
    return LAStages[CurrentLAStage].MacroItCount;
}

template<typename IterType, class Float, class SubType>
LAstep<IterType, Float, SubType>
LAReference<IterType, Float, SubType>::getLA(
    IterType LAIndex,
    FloatComplexT dz,
    /*FloatComplexT dc, */ IterType j,
    IterType iterations,
    IterType max_iterations) {

    IterType LAIndexj = LAIndex + j;
    const LAInfoI<IterType> &LAIj = LAs[LAIndexj].GetLAi();

    LAstep<IterType, Float, SubType> las;

    IterType l = LAIj.StepLength;
    bool usable = iterations + l <= max_iterations;
    //if (l < IterTypeMax) { // TODO - lame
    //    usable = iterations + l <= max_iterations;
    //}

    if (usable) {
        LAInfoDeep<IterType, Float, SubType> &LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (FloatComplexT)LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }
    else {
        las = LAstep<IterType, Float, SubType>();
        las.unusable = true;
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;

}

template class LAReference<uint32_t, HDRFloat<float>, float>;
template class LAReference<uint64_t, HDRFloat<float>, float>;
template class LAReference<uint32_t, HDRFloat<double>, double>;
template class LAReference<uint64_t, HDRFloat<double>, double>;