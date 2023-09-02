#include "stdafx.h"
#include "LAReference.h"
#include "RefOrbitCalc.h"

#include <Windows.h>

const double LAReference::log16 = std::log(16);

void LAReference::init() {

    isValid = false;
    LAStages.resize(MaxLAStages);

    LAs.reserve(DEFAULT_SIZE);
    LAIs.reserve(DEFAULT_SIZE);

    UseAT = false;
    LAStageCount = 0;

    LAStages[0].LAIndex = 0;
}

void LAReference::addToLA(LAInfoDeep la) {
    LAs.push_back(la);
}

size_t LAReference::LAsize() {
    return LAs.size();
}

void LAReference::addToLAI(LAInfoI lai) {
    LAIs.push_back(lai);
}

void LAReference::popLA() {
    LAs.pop_back();
}

void LAReference::popLAI() {
    LAIs.pop_back();
}


bool LAReference::CreateLAFromOrbit(size_t maxRefIteration) {

    init();

    size_t Period = 0;

    LAInfoDeep LA;


    LA = LAInfoDeep(HDRFloatComplex());
    LA.Step(LA, m_PerturbationResults.GetComplex<float>(1));

    LAInfoI LAI = LAInfoI();
    LAI.NextStageLAIndex = 0;

    if (LA.isZCoeffZero()) {
        return false;
    }

    size_t i;
    for (i = 2; i < maxRefIteration; i++) {

        LAInfoDeep NewLA;
        bool PeriodDetected;

        NewLA = LAInfoDeep();
        PeriodDetected = LA.Step(NewLA, m_PerturbationResults.GetComplex<float>(i));

        if (PeriodDetected) {
            Period = i;
            LAI.StepLength = Period;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));

            LAI.NextStageLAIndex = i;

            if (i + 1 < maxRefIteration) {
                LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(i)).Step(m_PerturbationResults.GetComplex<float>(i + 1));
                i += 2;
            }
            else {
                LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(i));
                i += 1;
            }
            break;
        }
        LA = NewLA;
    }

    LAStageCount = 1;

    size_t PeriodBegin = Period;
    size_t PeriodEnd = PeriodBegin + Period;

    if (Period == 0) {
        if (maxRefIteration > lowBound) {
            LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(0)).Step(m_PerturbationResults.GetComplex<float>(1));
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = std::round(std::log(maxRefIteration) / log16);
            Period = (size_t)std::round(std::pow(maxRefIteration, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));
            addToLA(LAInfoDeep(m_PerturbationResults.GetComplex<float>(maxRefIteration)));

            LAStages[0].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > lowBound) {
        popLA();
        popLAI();

        LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(0)).Step(m_PerturbationResults.GetComplex<float>(1));
        LAI.NextStageLAIndex = 0;
        i = 2;

        double NthRoot = std::round(std::log(Period) / log16);
        Period = (size_t)std::round(std::pow(Period, 1.0 / NthRoot));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    for (; i < maxRefIteration; i++) {
        LAInfoDeep NewLA;
        bool PeriodDetected;

        NewLA = LAInfoDeep();
        PeriodDetected = LA.Step(NewLA, m_PerturbationResults.GetComplex<float>(i));

        if (PeriodDetected || i >= PeriodEnd) {
            LAI.StepLength = i - PeriodBegin;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));

            LAI.NextStageLAIndex = i;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            size_t ip1 = i + 1;

            bool detected;

            detected = NewLA.DetectPeriod(m_PerturbationResults.GetComplex<float>(ip1));

            if (detected || ip1 >= maxRefIteration) {
                LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(i));
            }
            else {
                LA = LAInfoDeep(m_PerturbationResults.GetComplex<float>(i)).Step(
                    m_PerturbationResults.GetComplex<float>(ip1));
                i++;
            }
        }
        else {
            LA = NewLA;
        }
    }

    LAI.StepLength = i - PeriodBegin;

    addToLA(LA);
    addToLAI(LAInfoI(LAI));

    LAStages[0].MacroItCount = LAsize();

    addToLA(LAInfoDeep(m_PerturbationResults.GetComplex<float>(maxRefIteration)));
    addToLAI(LAInfoI());

    return true;
}


bool LAReference::CreateNewLAStage(size_t maxRefIteration) {
    LAInfoDeep LA;
    LAInfoI LAI = LAInfoI();
    size_t i;
    size_t PeriodBegin;
    size_t PeriodEnd;

    size_t PrevStage = LAStageCount - 1;
    size_t CurrentStage = LAStageCount;
    size_t PrevStageLAIndex = LAStages[PrevStage].LAIndex;
    size_t PrevStageMacroItCount = LAStages[PrevStage].MacroItCount;
    LAInfoDeep PrevStageLA = LAs[PrevStageLAIndex];
    LAInfoI PrevStageLAI = LAIs[PrevStageLAIndex];

    LAInfoDeep PrevStageLAp1 = LAs[PrevStageLAIndex + 1];
    LAInfoI PrevStageLAIp1 = LAIs[PrevStageLAIndex + 1];

    size_t Period = 0;

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
    size_t j;

    for (j = 2; j < PrevStageMacroItCount; j++) {
        LAInfoDeep NewLA;

        NewLA = LAInfoDeep();

        size_t PrevStageLAIndexj = PrevStageLAIndex + j;
        LAInfoDeep PrevStageLAj = LAs[PrevStageLAIndexj];
        LAInfoI PrevStageLAIj = LAIs[PrevStageLAIndexj];
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected) {
            if (PrevStageLAj.isLAThresholdZero()) break;
            Period = i;

            LAI.StepLength = Period;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));

            LAI.NextStageLAIndex = j;

            size_t PrevStageLAIndexjp1 = PrevStageLAIndexj + 1;
            LAInfoDeep PrevStageLAjp1 = LAs[PrevStageLAIndexjp1];
            LAInfoI PrevStageLAIjp1 = LAIs[PrevStageLAIndexjp1];

            if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
                i += PrevStageLAIj.StepLength;
                j++;
            }
            else {
                LA = PrevStageLAj.Composite(PrevStageLAjp1);
                i += PrevStageLAIj.StepLength + PrevStageLAIjp1.StepLength;
                j += 2;
            }
            break;
        }
        LA = NewLA;
        PrevStageLAIj = LAIs[PrevStageLAIndex + j];
        i += PrevStageLAIj.StepLength;
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
            Period = PrevStageLAI.StepLength * (size_t)std::round(std::pow(Ratio, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }
        else {
            LAI.StepLength = maxRefIteration;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));


            addToLA(LAInfoDeep(m_PerturbationResults.GetComplex<float>(maxRefIteration)));

            LAStages[CurrentStage].MacroItCount = 1;

            return false;
        }
    }
    else if (Period > PrevStageLAI.StepLength * lowBound) {
        popLA();
        popLAI();

        LA = PrevStageLA.Composite(PrevStageLAp1);
        i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
        LAI.NextStageLAIndex = 0;

        j = 2;

        double Ratio = ((double)(Period)) / PrevStageLAI.StepLength;

        double NthRoot = std::round(std::log(Ratio) / log16);
        Period = PrevStageLAI.StepLength * ((size_t)std::round(std::pow(Ratio, 1.0 / NthRoot)));

        PeriodBegin = 0;
        PeriodEnd = Period;
    }

    for (; j < PrevStageMacroItCount; j++) {
        LAInfoDeep NewLA;

        NewLA = LAInfoDeep();
        size_t PrevStageLAIndexj = PrevStageLAIndex + j;

        LAInfoDeep PrevStageLAj = LAs[PrevStageLAIndexj];
        LAInfoI PrevStageLAIj = LAIs[PrevStageLAIndexj];
        bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

        if (PeriodDetected || i >= PeriodEnd) {
            LAI.StepLength = i - PeriodBegin;

            addToLA(LA);
            addToLAI(LAInfoI(LAI));

            LAI.NextStageLAIndex = j;
            PeriodBegin = i;
            PeriodEnd = PeriodBegin + Period;

            LAInfoDeep PrevStageLAjp1 = LAs[PrevStageLAIndexj + 1];

            if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                LA = PrevStageLAj;
            }
            else {
                LA = PrevStageLAj.Composite(PrevStageLAjp1);
                i += PrevStageLAIj.StepLength;
                j++;
            }
        }
        else {
            LA = NewLA;
        }
        PrevStageLAIj = LAIs[PrevStageLAIndex + j];
        i += PrevStageLAIj.StepLength;
    }

    LAI.StepLength = i - PeriodBegin;

    addToLA(LA);
    addToLAI(LAInfoI(LAI));

    LAStages[CurrentStage].MacroItCount = LAsize() - LAStages[CurrentStage].LAIndex;

    addToLA(LAInfoDeep(m_PerturbationResults.GetComplex<float>(maxRefIteration)));
    addToLAI(LAInfoI());
    return true;
}

void LAReference::GenerateApproximationData(HDRFloat radius, size_t maxRefIteration) {

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
        size_t LAIndex = LAStages[Stage].LAIndex;
        LAs[LAIndex].CreateAT(AT, LAs[LAIndex + 1]);
        AT.StepLength = LAIs[LAIndex].StepLength;
        if (AT.StepLength > 0 && AT.Usable(SqrRadius)) {
            UseAT = true;
            return;
        }
    }
    UseAT = false;
}

bool LAReference::isLAStageInvalid(size_t LAIndex, HDRFloatComplex dc) {
    return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
}

size_t LAReference::getLAIndex(size_t CurrentLAStage) {
    return LAStages[CurrentLAStage].LAIndex;
}

size_t LAReference::getMacroItCount(size_t CurrentLAStage) {
    return LAStages[CurrentLAStage].MacroItCount;
}

LAstep LAReference::getLA(size_t LAIndex, HDRFloatComplex dz, /*HDRFloatComplex dc, */ size_t j, size_t iterations, size_t max_iterations) {

    size_t LAIndexj = LAIndex + j;
    LAInfoI LAIj = LAIs[LAIndexj];

    LAstep las;

    size_t l = LAIj.StepLength;
    bool usuable = iterations + l <= max_iterations;

    if (usuable) {
        LAInfoDeep &LAj = LAs[LAIndexj];

        las = LAj.Prepare(dz);

        if (!las.unusable) {
            las.LAjdeep = &LAj;
            las.Refp1Deep = (HDRFloatComplex)LAs[LAIndexj + 1].getRef();
            las.step = LAIj.StepLength;
        }
    }
    else {
        las = LAstep();
        las.unusable = true;
    }

    las.nextStageLAindex = LAIj.NextStageLAIndex;

    return las;

}
