#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.*;
//import fractalzoomer.functions.Fractal;
//
//import java.util.Arrays;

class LAStageInfo {
    int LAIndex;
    int MacroItCount;
    bool UseDoublePrecision;
};

class  LAInfoI {
private:
    int StepLength, NextStageLAIndex;

public:
    LAInfoI() {
        StepLength = 0;
        NextStageLAIndex = 0;
    }
    LAInfoI(LAInfoI other) {
        StepLength = other.StepLength;
        NextStageLAIndex = other.NextStageLAIndex;
    }
};

class LAReference {
private:
    static int lowBound = 64;
    static double log16 = Math.log(16);

    static MantExp doubleRadiusLimit = MantExp(0x1.0p-896);
    static MantExp doubleThresholdLimit = MantExp(0x1.0p-768);
public:
    bool UseAT;

    ATInfo AT;

    int LAStageCount;

    bool isValid;

    bool DoublePrecisionPT;

private:
    static int MaxLAStages = 512;
    static int DEFAULT_SIZE = 10000;
    GenericLAInfo[] LAs;
    int LAcurrentIndex;
    int LAIcurrentIndex;
    LAInfoI[] LAIs;

    LAStageInfo[] LAStages;

    void init(bool deepZoom) {

        isValid = false;
        LAStages = LAStageInfo[MaxLAStages];
        LAcurrentIndex = 0;
        LAIcurrentIndex = 0;

        LAs = GenericLAInfo[DEFAULT_SIZE];
        LAIs = LAInfoI[DEFAULT_SIZE];

        UseAT = false;
        LAStageCount = 0;

        for(int i = 0; i < LAStages.length; i++) {
            LAStages[i] = LAStageInfo();
        }

        LAStages[0].UseDoublePrecision = !deepZoom;

        LAStages[0].LAIndex = 0;
    }

    void addToLA(GenericLAInfo la) {
        LAs[LAcurrentIndex] = la;
        LAcurrentIndex++;
        if (LAcurrentIndex >= LAs.length) {
            LAs = Arrays.copyOf(LAs, LAs.length << 1);
        }
    }

    int LAsize() {
        return LAcurrentIndex;
    }

    void addToLAI(LAInfoI lai) {
        LAIs[LAIcurrentIndex] = lai;
        LAIcurrentIndex++;
        if (LAIcurrentIndex >= LAIs.length) {
            LAIs = Arrays.copyOf(LAIs, LAIs.length << 1);
        }
    }

    void popLA() {
        if(LAcurrentIndex > 0) {
            LAcurrentIndex--;
            LAs[LAcurrentIndex] = null;
        }
    }

    void popLAI() {
        if(LAIcurrentIndex > 0) {
            LAIcurrentIndex--;
            LAIs[LAIcurrentIndex] = null;
        }
    }


    bool CreateLAFromOrbit(DoubleReference ref, DeepReference refDeep, int maxRefIteration, bool deepZoom) {

        init(deepZoom);

        int Period = 0;

        GenericLAInfo LA;

        if(deepZoom) {
            LA = LAInfoDeep(MantExpComplex());
        }
        else {
            LA = LAInfo(Complex());
        }

        if(deepZoom) {
            LA = LA.Step(Fractal.getArrayDeepValue(refDeep, 1));
        }
        else {
            LA = LA.Step(Fractal.getArrayValue(ref, 1));
        }

        LAInfoI LAI = LAInfoI();
        LAI.NextStageLAIndex = 0;

        if(LA.isZCoeffZero()) {
            return false;
        }

        int i;
        for (i = 2; i < maxRefIteration; i++) {

            GenericLAInfo NewLA;
            bool PeriodDetected;

            if(deepZoom) {
                NewLA = LAInfoDeep();
                PeriodDetected = LA.Step(NewLA, Fractal.getArrayDeepValue(refDeep, i));
            }
            else {
                NewLA = LAInfo();
                PeriodDetected = LA.Step(NewLA, Fractal.getArrayValue(ref, i));
            }

            if (PeriodDetected) {
                Period = i;
                LAI.StepLength = Period;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));

                LAI.NextStageLAIndex = i;

                if (i + 1 < maxRefIteration) {
                    if(deepZoom) {
                        LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, i)).Step(Fractal.getArrayDeepValue(refDeep, i + 1));
                    }
                    else {
                        LA = LAInfo(Fractal.getArrayValue(ref, i)).Step(Fractal.getArrayValue(ref, i + 1));
                    }
                    i += 2;
                } else {
                    if(deepZoom) {
                        LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, i));
                    }
                    else {
                        LA = LAInfo(Fractal.getArrayValue(ref, i));
                    }
                    i += 1;
                }
                break;
            }
            LA = NewLA;
        }

        LAStageCount = 1;

        int PeriodBegin = Period;
        int PeriodEnd = PeriodBegin + Period;

        if (Period == 0) {
            if (maxRefIteration > lowBound) {
                if(deepZoom) {
                    LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, 0)).Step(Fractal.getArrayDeepValue(refDeep, 1));
                }
                else {
                    LA = LAInfo(Fractal.getArrayValue(ref, 0)).Step(Fractal.getArrayValue(ref, 1));
                }
                LAI.NextStageLAIndex = 0;
                i = 2;

                double NthRoot = Math.round(Math.log(maxRefIteration) / log16);
                Period = (int)Math.round(Math.pow(maxRefIteration, 1.0 / NthRoot));

                PeriodBegin = 0;
                PeriodEnd = Period;
            } else {
                LAI.StepLength = maxRefIteration;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));

                if(deepZoom) {
                    addToLA(LAInfoDeep(Fractal.getArrayDeepValue(refDeep, maxRefIteration)));
                }
                else {
                    addToLA(LAInfo(Fractal.getArrayValue(ref, maxRefIteration)));
                }

                LAStages[0].MacroItCount = 1;

                return false;
            }
        } else if (Period > lowBound) {
            popLA();
            popLAI();

            if(deepZoom) {
                LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, 0)).Step(Fractal.getArrayDeepValue(refDeep, 1));
            }
            else {
                LA = LAInfo(Fractal.getArrayValue(ref, 0)).Step(Fractal.getArrayValue(ref, 1));
            }
            LAI.NextStageLAIndex = 0;
            i = 2;

            double NthRoot = Math.round(Math.log(Period) / log16);
            Period = (int)Math.round(Math.pow(Period, 1.0 / NthRoot));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }

        for (; i < maxRefIteration; i++) {
            GenericLAInfo NewLA;
            bool PeriodDetected;

            if(deepZoom) {
                NewLA = LAInfoDeep();
                PeriodDetected = LA.Step(NewLA, Fractal.getArrayDeepValue(refDeep, i));
            }
            else {
                NewLA = LAInfo();
                PeriodDetected = LA.Step(NewLA, Fractal.getArrayValue(ref, i));
            }

            if (PeriodDetected || i >= PeriodEnd) {
                LAI.StepLength = i - PeriodBegin;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));

                LAI.NextStageLAIndex = i;
                PeriodBegin = i;
                PeriodEnd = PeriodBegin + Period;

                int ip1 = i + 1;

                bool detected;

                if(deepZoom) {
                    detected = NewLA.DetectPeriod(Fractal.getArrayDeepValue(refDeep, ip1));
                }
                else {
                    detected = NewLA.DetectPeriod(Fractal.getArrayValue(ref, ip1));
                }

                if (detected || ip1 >= maxRefIteration) {
                    if(deepZoom) {
                        LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, i));
                    }
                    else {
                        LA = LAInfo(Fractal.getArrayValue(ref, i));
                    }
                } else {
                    if(deepZoom) {
                        LA = LAInfoDeep(Fractal.getArrayDeepValue(refDeep, i)).Step(Fractal.getArrayDeepValue(refDeep, ip1));
                    }
                    else {
                        LA = LAInfo(Fractal.getArrayValue(ref, i)).Step(Fractal.getArrayValue(ref, ip1));
                    }
                    i++;
                }
            } else {
                LA = NewLA;
            }
        }

        LAI.StepLength = i - PeriodBegin;

        addToLA(LA);
        addToLAI(LAInfoI(LAI));

        LAStages[0].MacroItCount = LAsize();

        if(deepZoom) {
            addToLA(LAInfoDeep(Fractal.getArrayDeepValue(refDeep, maxRefIteration)));
        }
        else {
            addToLA(LAInfo(Fractal.getArrayValue(ref, maxRefIteration)));
        }

        addToLAI(LAInfoI());

        return true;
    }


    bool CreateNewLAStage(DoubleReference ref, DeepReference refDeep, int maxRefIteration, bool deepZoom) throws Exception {
        GenericLAInfo LA;
        LAInfoI LAI = LAInfoI();
        int i;
        int PeriodBegin;
        int PeriodEnd;

        int PrevStage = LAStageCount - 1;
        int CurrentStage = LAStageCount;
        int PrevStageLAIndex = LAStages[PrevStage].LAIndex;
        int PrevStageMacroItCount = LAStages[PrevStage].MacroItCount;
        GenericLAInfo PrevStageLA = LAs[PrevStageLAIndex];
        LAInfoI PrevStageLAI = LAIs[PrevStageLAIndex];

        GenericLAInfo PrevStageLAp1 = LAs[PrevStageLAIndex + 1];
        LAInfoI PrevStageLAIp1 = LAIs[PrevStageLAIndex + 1];

        int Period = 0;

        if (CurrentStage >= MaxLAStages) throw Exception("Too many stages");

        LAStages[CurrentStage].UseDoublePrecision = !deepZoom;
        LAStages[CurrentStage].LAIndex = LAsize();

        LA = PrevStageLA.Composite(PrevStageLAp1);
        LAI.NextStageLAIndex = 0;
        i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
        int j;

        for (j = 2; j < PrevStageMacroItCount; j++) {
            GenericLAInfo NewLA;

            if(deepZoom) {
                NewLA = LAInfoDeep();
            }
            else {
                NewLA = LAInfo();
            }

            int PrevStageLAIndexj = PrevStageLAIndex + j;
            GenericLAInfo PrevStageLAj = LAs[PrevStageLAIndexj];
            LAInfoI PrevStageLAIj = LAIs[PrevStageLAIndexj];
            bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

            if (PeriodDetected) {
                if (PrevStageLAj.isLAThresholdZero()) break;
                Period = i;

                LAI.StepLength = Period;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));

                LAI.NextStageLAIndex = j;

                int PrevStageLAIndexjp1 = PrevStageLAIndexj + 1;
                GenericLAInfo PrevStageLAjp1 = LAs[PrevStageLAIndexjp1];
                LAInfoI PrevStageLAIjp1 = LAIs[PrevStageLAIndexjp1];

                if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                    LA = PrevStageLAj;
                    i += PrevStageLAIj.StepLength;
                    j++;
                } else {
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
        if (LAStageCount > MaxLAStages) throw Exception("Too many stages");

        PeriodBegin = Period;
        PeriodEnd = PeriodBegin + Period;

        if (Period == 0) {
            if (maxRefIteration > PrevStageLAI.StepLength * lowBound) {
                LA = PrevStageLA.Composite(PrevStageLAp1);
                i = PrevStageLAI.StepLength + PrevStageLAIp1.StepLength;
                LAI.NextStageLAIndex = 0;

                j = 2;

                double Ratio = ((double)(maxRefIteration)) / PrevStageLAI.StepLength;
                double NthRoot = Math.round(Math.log(Ratio) / log16); // log16
                Period = PrevStageLAI.StepLength * (int)Math.round(Math.pow(Ratio, 1.0 / NthRoot));

                PeriodBegin = 0;
                PeriodEnd = Period;
            } else {
                LAI.StepLength = maxRefIteration;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));


                if(deepZoom) {
                    addToLA(LAInfoDeep(Fractal.getArrayDeepValue(refDeep, maxRefIteration)));
                }
                else {
                    addToLA(LAInfo(Fractal.getArrayValue(ref, maxRefIteration)));
                }

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

            double NthRoot = Math.round(Math.log(Ratio) / log16);
            Period = PrevStageLAI.StepLength * ((int)Math.round(Math.pow(Ratio, 1.0 / NthRoot)));

            PeriodBegin = 0;
            PeriodEnd = Period;
        }

        for (; j < PrevStageMacroItCount; j++) {
            GenericLAInfo NewLA;

            if(deepZoom) {
                NewLA = LAInfoDeep();
            }
            else {
                NewLA = LAInfo();
            }

            int PrevStageLAIndexj = PrevStageLAIndex + j;

            GenericLAInfo PrevStageLAj = LAs[PrevStageLAIndexj];
            LAInfoI PrevStageLAIj = LAIs[PrevStageLAIndexj];
            bool PeriodDetected = LA.Composite(NewLA, PrevStageLAj);

            if (PeriodDetected || i >= PeriodEnd) {
                LAI.StepLength = i - PeriodBegin;

                addToLA(LA);
                addToLAI(LAInfoI(LAI));

                LAI.NextStageLAIndex = j;
                PeriodBegin = i;
                PeriodEnd = PeriodBegin + Period;

                GenericLAInfo PrevStageLAjp1 = LAs[PrevStageLAIndexj + 1];

                if (NewLA.DetectPeriod(PrevStageLAjp1.getRef()) || j + 1 >= PrevStageMacroItCount) {
                    LA = PrevStageLAj;
                } else {
                    LA = PrevStageLAj.Composite(PrevStageLAjp1);
                    i += PrevStageLAIj.StepLength;
                    j++;
                }
            } else {
                LA = NewLA;
            }
            PrevStageLAIj = LAIs[PrevStageLAIndex + j];
            i += PrevStageLAIj.StepLength;
        }

        LAI.StepLength = i - PeriodBegin;

        addToLA(LA);
        addToLAI(LAInfoI(LAI));

        LAStages[CurrentStage].MacroItCount = LAsize() - LAStages[CurrentStage].LAIndex;

        if(deepZoom) {
            addToLA(LAInfoDeep(Fractal.getArrayDeepValue(refDeep, maxRefIteration)));
        }
        else {
            addToLA(LAInfo(Fractal.getArrayValue(ref, maxRefIteration)));
        }

        addToLAI(LAInfoI());
        return true;
    }

    bool DoubleUsableAtPrevStage(int stage, MantExp radius) {
            int LAIndex = LAStages[stage].LAIndex;
            GenericLAInfo LA = LAs[LAIndex];
            return radius.compareToBothPositiveReduced(doubleRadiusLimit) > 0
                    || (LA.getLAThreshold().compareToBothPositiveReduced(doubleThresholdLimit) > 0
                         && LA.getLAThresholdC().compareToBothPositiveReduced(doubleThresholdLimit) > 0);

    }

    public:
    void GenerateApproximationData(MantExp radius, ReferenceData refData, ReferenceDeepData refDeepData, int maxRefIteration, bool deepZoom, Fractal f, int period) {

        try {
            if(maxRefIteration == 0) {
                isValid = false;
                return;
            }

            bool PeriodDetected = CreateLAFromOrbit(refData.Reference, refDeepData.Reference, maxRefIteration, deepZoom);

            if (deepZoom && !f.useFullFloatExp()) {
                if (DoubleUsableAtPrevStage(0, radius)) {
                    DoublePrecisionPT = true;
                }
            }

            if (!PeriodDetected) return;

            bool convertedToDouble = false;

            while (true) {
                int PrevStage = LAStageCount - 1;
                int CurrentStage = LAStageCount;

                PeriodDetected = CreateNewLAStage(refData.Reference, refDeepData.Reference, maxRefIteration, deepZoom);

                if (deepZoom && !f.useFullFloatExp()) {
                    if (DoubleUsableAtPrevStage(CurrentStage, radius)) {
                        ConvertStageToDouble(PrevStage);
                        convertedToDouble = true;
                    }
                }

                if (!PeriodDetected) break;
            }

            CreateATFromLA(radius);

            if (deepZoom && !f.useFullFloatExp()) {
                if (radius.compareToBothPositiveReduced(doubleRadiusLimit) > 0) {
                    ConvertStageToDouble(LAStageCount - 1);
                    convertedToDouble = true;
                }
            }

            //Recreate data to save memory
            if (deepZoom && refData.Reference == null && (convertedToDouble || DoublePrecisionPT)) {
                int length = maxRefIteration + 1;
                refData.createAndSetShortcut(length, false, 0);
                DoubleReference reference = refData.Reference;
                DeepReference deepReference = refDeepData.Reference;

                for (int i = 0; i < length; i++) {
                    Fractal.setArrayValue(reference, i, Fractal.getArrayDeepValue(deepReference, i).toComplex());
                }

                reference.setLengthOverride(deepReference.length());
            }
        }
        catch (Exception ex) {
            isValid = false;
            ex.printStackTrace();
            return;
        }

        isValid = true;
    }

    void CreateATFromLA(MantExp radius) {

        MantExp SqrRadius = radius.square();
        SqrRadius.Reduce();

        for (int Stage = LAStageCount; Stage > 0; ) {
            Stage--;
            int LAIndex = LAStages[Stage].LAIndex;
            AT = LAs[LAIndex].CreateAT(LAs[LAIndex + 1]);
            AT.StepLength = LAIs[LAIndex].StepLength;
            if (AT.StepLength > 0 && AT.Usable(SqrRadius)) {
                UseAT = true;
                return;
            }
        }
        UseAT = false;
    }

    private:
    void ConvertStageToDouble(int Stage) {
        int LAIndex = LAStages[Stage].LAIndex;
        int MacroItCount = LAStages[Stage].MacroItCount;

        LAStages[Stage].UseDoublePrecision = true;

        for (int i = 0; i <= MacroItCount; i++) {
            LAs[LAIndex + i] = LAInfo((LAInfoDeep)LAs[LAIndex + i]);
        }
    }

    public:
    bool isLAStageInvalid(int LAIndex, Complex dc) {
        return (dc.chebychevNorm() >= ((LAInfo)LAs[LAIndex]).LAThresholdC);
    }

    bool isLAStageInvalid(int LAIndex, MantExpComplex dc) {
        return (dc.chebychevNorm().compareToBothPositiveReduced((LAs[LAIndex]).getLAThresholdC()) >= 0);
    }

    bool useDoublePrecisionAtStage(int stage) {
        return LAStages[stage].UseDoublePrecision;
    }

    int getLAIndex(int CurrentLAStage) {
        return LAStages[CurrentLAStage].LAIndex;
    }

    int getMacroItCount(int CurrentLAStage) {
        return LAStages[CurrentLAStage].MacroItCount;
    }

    LAstep getLA(int LAIndex, Complex dz, Complex dc, int j, int iterations, int max_iterations) {

        int LAIndexj = LAIndex + j;
        LAInfoI LAIj = LAIs[LAIndexj];

        LAstep las;

        int l = LAIj.StepLength;
        bool usuable  = iterations + l <= max_iterations;

        if(usuable) {
            LAInfo LAj = (LAInfo) LAs[LAIndexj];
            las = LAj.Prepare(dz, dc);

            if(!las.unusable) {
                las.LAj = LAj;
                las.Refp1 = (Complex) LAs[LAIndexj + 1].getRef();
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

    LAstep getLA(int LAIndex, MantExpComplex dz, MantExpComplex dc, int j, int iterations, int max_iterations) {

        int LAIndexj = LAIndex + j;
        LAInfoI LAIj = LAIs[LAIndexj];

        LAstep las;

        int l = LAIj.StepLength;
        bool usuable = iterations + l <= max_iterations;

        if(usuable) {
            LAInfoDeep LAj = (LAInfoDeep) LAs[LAIndexj];

            las = LAj.Prepare(dz, dc);

            if(!las.unusable) {
                las.LAjdeep = LAj;
                las.Refp1Deep = (MantExpComplex) LAs[LAIndexj + 1].getRef();
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

}
