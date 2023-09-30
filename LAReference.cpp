#include "stdafx.h"
#include "LAReference.h"
#include "RefOrbitCalc.h"
#include "PerturbationResults.h"

//#include <Windows.h>

const double LAReference::log16 = std::log(16);

int32_t LAReference::LAsize() {
    return (int32_t)LAs.size();
}

bool LAReference::CreateLAFromOrbit(int32_t maxRefIteration) {

    {
        isValid = false;
        LAStages.resize(MaxLAStages);

        LAs.reserve(maxRefIteration);

        UseAT = false;
        LAStageCount = 0;

        LAStages[0].LAIndex = 0;
    }

    int32_t Period = 0;

    LAInfoDeep<float> LA{HDRFloatComplex()};
    LA = LA.Step(m_PerturbationResults.GetComplex<float>(1));

    LAInfoI LAI = LAInfoI();
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    int32_t i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep<float> NewLA;
        bool PeriodDetected;

        NewLA = LAInfoDeep<float>();
        PeriodDetected = LA.Step(NewLA, m_PerturbationResults.GetComplex<float>(i));

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
            LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(i)).Step(m_PerturbationResults.GetComplex<float>(i + 1));
            i += 2;
        }
        else {
            LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(i));
            i += 1;
        }
        break;
    }

    LAStageCount = 1;

    int32_t PeriodBegin = Period;
    int32_t PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(0)).Step(m_PerturbationResults.GetComplex<float>(1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log(maxRefIteration) / log16);
            Period = (int32_t)std::round(std::pow(maxRefIteration, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAs.push_back(LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(maxRefIteration)));

            LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        LAs.pop_back();

        LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(0)).Step(m_PerturbationResults.GetComplex<float>(1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log(Period) / log16);
        Period = (int32_t)std::round(std::pow(Period, 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    const auto numPerThread = maxRefIteration / std::thread::hardware_concurrency();

    for (; i < maxRefIteration; i++) {
        LAInfoDeep<float> NewLA{};
        bool PeriodDetected{ LA.Step(NewLA, m_PerturbationResults.GetComplex<float>(i)) };

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

        int32_t ip1{ i + 1 };
        bool detected{ NewLA.DetectPeriod(m_PerturbationResults.GetComplex<float>(ip1)) };

        if (detected || ip1 >= maxRefIteration) {
            LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(i));
        }
        else {
            LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(i)).Step(
                m_PerturbationResults.GetComplex<float>(ip1));
            i++;
        }
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    LAs.push_back(LA);

    LAStages[0].MacroItCount = LAsize();

    auto LA2 = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(maxRefIteration));
    LA2.SetLAi({});
    LAs.push_back(LA2);

    return true;
}

//#pragma optimize( "", off )
bool LAReference::CreateNewLAStage(int32_t maxRefIteration) {
    LAInfoDeep<float> LA;
    LAInfoI LAI = LAInfoI();
    int32_t i;
    int32_t PeriodBegin;
    int32_t PeriodEnd;

    int32_t PrevStage = LAStageCount - 1;
    int32_t CurrentStage = LAStageCount;
    int32_t PrevStageLAIndex = LAStages[PrevStage].LAIndex;
    int32_t PrevStageMacroItCount = LAStages[PrevStage].MacroItCount;
    LAInfoDeep<float> PrevStageLA = LAs[PrevStageLAIndex];
    const LAInfoI &PrevStageLAI = LAs[PrevStageLAIndex].GetLAi();

    LAInfoDeep<float> PrevStageLAp1 = LAs[PrevStageLAIndex + 1];
    const LAInfoI PrevStageLAIp1 = LAs[PrevStageLAIndex + 1].GetLAi();

    int32_t Period = 0;

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
    int32_t j;

    for (j = 2; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<float> NewLA;

        NewLA = LAInfoDeep<float>();

        int32_t PrevStageLAIndexj = PrevStageLAIndex + j;
        LAInfoDeep<float> PrevStageLAj = LAs[PrevStageLAIndexj];
        const LAInfoI *PrevStageLAIj = &PrevStageLAj.GetLAi();
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected) {
            if (PrevStageLAj.isLAThresholdZero()) break;
            Period = i;

            LAI.StepLength = Period;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAI.NextStageLAIndex = j;

            int32_t PrevStageLAIndexjp1 = PrevStageLAIndexj + 1;
            LAInfoDeep<float> PrevStageLAjp1 = LAs[PrevStageLAIndexjp1];
            const LAInfoI &PrevStageLAIjp1 = LAs[PrevStageLAIndexjp1].GetLAi();

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
            Period = PrevStageLAI.StepLength * (int32_t)std::round(std::pow(Ratio, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAInfoDeep<float> LA2(m_PerturbationResults.GetComplex<float>(maxRefIteration));
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
        Period = PrevStageLAI.StepLength * ((int32_t)std::round(std::pow(Ratio, 1.0 / NthRoot)));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    for (; j < PrevStageMacroItCount; j++) {
        LAInfoDeep<float> NewLA;

        NewLA = LAInfoDeep<float>();
        int32_t PrevStageLAIndexj = PrevStageLAIndex + j;

        LAInfoDeep<float> PrevStageLAj = LAs[PrevStageLAIndexj];
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected || i >= PeriodEnd) {
            LAI.StepLength = i - PeriodBegin;

            LA.SetLAi(LAI);
            LAs.push_back(LA);

            LAI.NextStageLAIndex = j;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            LAInfoDeep<float> PrevStageLAjp1 = LAs[PrevStageLAIndexj + 1];

            if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
            }
            else {
                LA = PrevStageLAj.Composite(PrevStageLAjp1);

                const LAInfoI &PrevStageLAIj = LAs[PrevStageLAIndexj].GetLAi();
                i += PrevStageLAIj.StepLength;
                j++;
            }
        }
        else {
            LA = NewLA;
        }
        const LAInfoI &PrevStageLAIj = LAs[PrevStageLAIndex + j].GetLAi();
        i += PrevStageLAIj.StepLength;
    }

    LAI.StepLength = i - PeriodBegin;

    LA.SetLAi(LAI);
    LAs.push_back(LA);

    LAStages[CurrentStage].MacroItCount = LAsize() - LAStages[CurrentStage].LAIndex;

    LA = LAInfoDeep<float>(m_PerturbationResults.GetComplex<float>(maxRefIteration));
    LA.SetLAi({});
    LAs.push_back(LA);
    return true;
}
//#pragma optimize( "", on )

void LAReference::GenerateApproximationData(HDRFloat radius, int32_t maxRefIteration) {

    if (maxRefIteration == 0) {
        isValid = false;
        return;
    }

    bool PeriodDetected = CreateLAFromOrbit(maxRefIteration);
    if (!PeriodDetected) return;

    while (true) {
        PeriodDetected = CreateNewLAStage(maxRefIteration);
        if (!PeriodDetected) break;
    }

    CreateATFromLA(radius);
    isValid = true;
}

void LAReference::CreateATFromLA(HDRFloat radius) {

    HDRFloat SqrRadius = radius.square();
    SqrRadius.Reduce();

    for (auto Stage = LAStageCount; Stage > 0; ) {
        Stage--;
        int32_t LAIndex = LAStages[Stage].LAIndex;
        LAs[LAIndex].CreateAT(AT, LAs[LAIndex + 1]);
        AT.StepLength = LAs[LAIndex].GetLAi().StepLength;
        if (AT.StepLength > 0 && AT.Usable(SqrRadius)) {
            UseAT = true;
            return;
        }
    }
    UseAT = false;
}

bool LAReference::isLAStageInvalid(int32_t LAIndex, HDRFloatComplex dc) {
    return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
}

int32_t LAReference::getLAIndex(int32_t CurrentLAStage) {
    return LAStages[CurrentLAStage].LAIndex;
}

int32_t LAReference::getMacroItCount(int32_t CurrentLAStage) {
    return LAStages[CurrentLAStage].MacroItCount;
}

LAstep<HDRFloatComplex<float>>
LAReference::getLA(int32_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ int32_t j, int32_t iterations, int32_t max_iterations) {

    int32_t LAIndexj = LAIndex + j;
    const LAInfoI &LAIj = LAs[LAIndexj].GetLAi();

    LAstep<HDRFloatComplex> las;

    int32_t l = LAIj.StepLength;
    bool usuable = iterations + l <= max_iterations;

    if (usuable) {
        LAInfoDeep<float> &LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (HDRFloatComplex)LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }
    else {
        las = LAstep<HDRFloatComplex>();
        las.unusable = true;
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;

}
