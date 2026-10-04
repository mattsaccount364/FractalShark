#include "ATInfo.h"
#include "LAstep.h"
#include "TestFramework.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

template <typename IterType, class Float, class SubType>
ATInfo<IterType, Float, SubType>
MakeInitialization(double referenceReal, IterType stepLength)
{
    ATInfo<IterType, Float, SubType> info;
    info.StepLength = stepLength;
    info.ThresholdC = Float{1000};
    info.SqrEscapeRadius = Float{256};
    info.RefC = {Float{referenceReal}, Float{0}};
    info.ZCoeff = {Float{1}, Float{0}};
    info.CCoeff = {Float{1}, Float{0}};
    info.InvZCoeff = {Float{1}, Float{0}};
    info.CCoeffNormSqr = Float{1};
    info.RefCNormSqr = info.RefC.norm_squared();
    HdrReduce(info.RefCNormSqr);
    return info;
}

template <typename IterType, class Float, class SubType>
void
CheckInitialization()
{
    auto info = MakeInitialization<IterType, Float, SubType>(1, IterType{5});
    auto delta = info.RefC;
    const auto fallbackDelta = decltype(delta){SubType{0}, SubType{0}};
    IterType iterations = 99;
    info.InitializePixel(true, IterType{100}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{20});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 26.0);
    ASSERT_EQ(static_cast<SubType>(delta.getIm()), 0.0);

    info.RefC = {};
    info.InitializePixel(true, IterType{17}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{15});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 0.0);
    info.InitializePixel(true, IterType{4}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{0});
    info.InitializePixel(true, IterType{0}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{0});

    info.RefC = {Float{-1}, Float{0}};
    info.StepLength = 1;
    info.InitializePixel(true, IterType{101}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{101});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), -1.0);

    info.RefC = {Float{1}, Float{0}};
    info.SqrEscapeRadius = Float{1};
    info.InitializePixel(true, IterType{10}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{2});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 2.0);

    info.ThresholdC = Float{1};
    info.SqrEscapeRadius = Float{256};
    info.InitializePixel(true, IterType{1}, {Float{1}, Float{0}}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{1});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 2.0);
    info.InitializePixel(true, IterType{10}, {Float{1.25}, Float{0}}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{0});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 0.0);
    delta = info.RefC;
    iterations = 99;
    info.InitializePixel(false, IterType{10}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{0});
    ASSERT_EQ(static_cast<SubType>(delta.getRe()), 0.0);
    if constexpr (std::is_same_v<Float, HDRFloat<SubType>>) {
        ASSERT_EQ(delta.getRe().exp, 0);
        ASSERT_EQ(delta.getIm().exp, 0);
        // A sentinel-exponent zero would retain this tiny value instead.
        const Float tiny{-1000, SubType{1}};
        const auto sum = delta + decltype(delta){tiny, tiny};
        ASSERT_EQ(sum.getRe().mantissa, SubType{0});
        ASSERT_EQ(sum.getIm().mantissa, SubType{0});
    }
    info.StepLength = 0;
    info.InitializePixel(true, IterType{10}, {}, fallbackDelta, iterations, delta);
    ASSERT_EQ(iterations, IterType{0});
    ASSERT_EQ(static_cast<SubType>(delta.getIm()), 0.0);
    if constexpr (std::is_same_v<Float, HDRFloat<SubType>>) {
        ASSERT_EQ(delta.getRe().exp, 0);
        ASSERT_EQ(delta.getIm().exp, 0);
    }

    if constexpr (sizeof(IterType) == 8) {
        info.StepLength = IterType{1} << 33;
        info.InitializePixel(true, info.StepLength * 4, {}, fallbackDelta, iterations, delta);
        ASSERT_EQ(iterations, info.StepLength * 4);
        ASSERT_EQ(static_cast<SubType>(delta.getRe()), 26.0);
    }
}

struct EvaluationStep {
    uint64_t step;
    uint64_t nextStageLAindex;
    bool unusable;
    FloatComplex<double> m_Delta;
    FloatComplex<double> m_Reference;

    FloatComplex<double>
    Evaluate(const FloatComplex<double> &deltaC) const
    {
        return m_Delta + deltaC;
    }
    FloatComplex<double>
    getZ(const FloatComplex<double> &deltaZ) const
    {
        return m_Reference + deltaZ;
    }
};

struct EvaluationStage {
    uint64_t m_Index;
    uint64_t m_MacroCount;
    bool m_Invalid;
};

struct EvaluationRequest {
    uint64_t m_LAIndex;
    uint64_t m_Index;
    uint64_t m_Iterations;
    FloatComplex<double> m_Delta;
};

struct EvaluationReference {
    bool m_Valid = true;
    bool m_Initialize = false;
    ATInfo<uint64_t, double, double> m_Initialization =
        MakeInitialization<uint64_t, double, double>(100, 5);
    std::vector<EvaluationStage> m_Stages;
    std::vector<EvaluationStep> m_Steps;
    std::vector<EvaluationRequest> m_Requests;

    bool
    IsValid() const
    {
        return m_Valid;
    }
    uint64_t
    GetLAStageCount() const
    {
        return m_Stages.size();
    }
    void
    InitializePixel(uint64_t maxIterations,
                    const FloatComplex<double> &deltaC,
                    uint64_t &iterations,
                    FloatComplex<double> &deltaZ) const
    {
        m_Initialization.InitializePixel(m_Valid && m_Initialize,
                                         maxIterations,
                                         deltaC,
                                         FloatComplex<double>{0, 0},
                                         iterations,
                                         deltaZ);
    }
    uint64_t
    getLAIndex(uint64_t stage) const
    {
        return m_Stages.at(stage).m_Index;
    }
    uint64_t
    getMacroItCount(uint64_t stage) const
    {
        return m_Stages.at(stage).m_MacroCount;
    }
    bool
    IsLAStageInvalid(uint64_t index, const FloatComplex<double> &) const
    {
        return m_Stages.at(index).m_Invalid;
    }
    EvaluationStep
    getLA(uint64_t index,
          const FloatComplex<double> &deltaZ,
          uint64_t j,
          uint64_t iterations,
          uint64_t maxIterations)
    {
        const auto result = m_Steps.at(m_Requests.size());
        m_Requests.push_back({index, j, iterations, deltaZ});
        if (!result.unusable)
            ASSERT_TRUE(result.step <= maxIterations - iterations);
        return result;
    }
};

struct EvaluationOutputs {
    uint64_t m_Iterations = 99;
    uint64_t m_ReferenceIteration = 99;
    FloatComplex<double> m_Delta{99, 99};
    FloatComplex<double> m_Z{99, 99};
};

EvaluationOutputs
EvaluateReference(EvaluationReference &reference,
                  uint64_t maxIterations,
                  FloatComplex<double> deltaC,
                  uint64_t maxRefIteration,
                  uint64_t period,
                  uint64_t &orbitReads)
{
    EvaluationOutputs result;
    const auto readOrbit = [&](uint64_t) {
        ++orbitReads;
        return FloatComplex<double>{7, 0};
    };
    FractalShark::LA::AdvancePixel(reference,
                                   maxIterations,
                                   deltaC,
                                   maxRefIteration,
                                   period,
                                   readOrbit,
                                   result.m_Iterations,
                                   result.m_ReferenceIteration,
                                   result.m_Delta,
                                   result.m_Z);
    return result;
}

template <class SubType>
void
CheckBackendFallbackZeros()
{
    ATInfo<uint32_t, HDRFloat<SubType>, SubType> info;
    HDRFloatComplex<SubType> delta;
    uint32_t iterations = 99;
    const HDRFloatComplex<SubType> cpuZero{SubType{0}, SubType{0}};
    const HDRFloatComplex<SubType> gpuZero{HDRFloat<SubType>{0}, HDRFloat<SubType>{0}};
    const HDRFloatComplex<SubType> tiny{HDRFloat<SubType>{-1000, SubType{1}}, HDRFloat<SubType>{}};
    info.InitializePixel(false, 10, {}, cpuZero, iterations, delta);
    ASSERT_EQ(iterations, uint32_t{0});
    ASSERT_EQ(delta.getRe().exp, 0);
    ASSERT_EQ((delta + tiny).getRe().mantissa, SubType{0});
    info.InitializePixel(false, 10, {}, gpuZero, iterations, delta);
    ASSERT_EQ(iterations, uint32_t{0});
    ASSERT_EQ(delta.getRe().exp, HDRFloat<SubType>::MIN_BIG_EXPONENT());
    ASSERT_EQ((delta + tiny).getRe().exp, -1000);
    ASSERT_EQ((delta + tiny).getRe().mantissa, SubType{1});
}
} // namespace

TEST(LAInitialization_BoundariesAndArithmeticAcrossTypes)
{
    CheckInitialization<uint32_t, double, double>();
    CheckInitialization<uint64_t, double, double>();
    CheckInitialization<uint32_t, HDRFloat<float>, float>();
    CheckInitialization<uint64_t, HDRFloat<double>, double>();
}

TEST(LAInitialization_ImmediateEscapeAndCoordinateTransforms)
{
    auto info = MakeInitialization<uint64_t, double, double>(100, 1);
    uint64_t iterations = 0;
    FloatComplex<double> delta;
    info.InitializePixel(true, 100, {}, {}, iterations, delta);
    ASSERT_EQ(iterations, uint64_t{1});
    ASSERT_EQ(delta.getRe(), 100.0);
    info.CCoeff = {2, 0};
    info.RefC = {1, 1};
    const auto parameter = info.getC({0.5, 0});
    ASSERT_EQ(parameter.getRe(), 2.0);
    ASSERT_EQ(parameter.getIm(), 1.0);
    info.InvZCoeff = {0.5, 0};
    const auto restored = info.getDZ({4, 6});
    ASSERT_EQ(restored.getRe(), 2.0);
    ASSERT_EQ(restored.getIm(), 3.0);
    ATInfo<uint64_t, double, double> empty;
    empty.InitializePixel(true, 100, {}, {}, iterations, delta);
    ASSERT_EQ(iterations, uint64_t{0});
    ASSERT_EQ(delta.getRe(), 0.0);
}

TEST(LAInitialization_PrecisionConversionPreservesOutputs)
{
    const auto original = MakeInitialization<uint64_t, HDRFloat<double>, double>(1, 5);
    const ATInfo<uint64_t, HDRFloat<float>, float> converted(original);
    uint64_t iterations = 0;
    HDRFloatComplex<float> delta;
    converted.InitializePixel(true, 100, {}, {}, iterations, delta);
    ASSERT_EQ(iterations, uint64_t{20});
    ASSERT_EQ(static_cast<float>(delta.getRe()), 26.0);
}

TEST(LAInitialization_MetadataRoundTripKeepsLegacyFields)
{
    const auto path = std::filesystem::temp_directory_path() / "fractalshark-la-initialization.meta";
    auto original = MakeInitialization<uint64_t, HDRFloat<double>, double>(1, 5);
    original.CCoeffSqrInvZCoeff = {3, 4};
    original.CCoeffInvZCoeff = {5, 6};
    {
        std::ofstream stream(path);
        ASSERT_TRUE(original.WriteMetadata(stream));
    }
    ATInfo<uint64_t, HDRFloat<double>, double> restored;
    {
        std::ifstream stream(path);
        ASSERT_TRUE(restored.ReadMetadata(stream));
        ASSERT_FALSE(stream.fail());
    }
    ASSERT_EQ(restored.StepLength, original.StepLength);
    ASSERT_EQ(static_cast<double>(restored.CCoeffSqrInvZCoeff.getRe()), 3.0);
    ASSERT_EQ(static_cast<double>(restored.CCoeffInvZCoeff.getIm()), 6.0);
    ASSERT_EQ(static_cast<double>(restored.factor), 4294967296.0);
    uint64_t iterations = 0;
    HDRFloatComplex<double> delta;
    restored.InitializePixel(true, 100, {}, {}, iterations, delta);
    ASSERT_EQ(iterations, uint64_t{20});
    ASSERT_EQ(static_cast<double>(delta.getRe()), 26.0);
    std::filesystem::remove(path);
}

TEST(LAEvaluation_InvalidReferenceResetsAllOutputs)
{
    EvaluationReference reference;
    reference.m_Valid = false;
    reference.m_Initialize = true;
    reference.m_Stages = {{10, 4, false}};
    uint64_t reads = 0;
    const auto result = EvaluateReference(reference, 100, {1, 2}, 8, 0, reads);
    ASSERT_EQ(result.m_Iterations, uint64_t{0});
    ASSERT_EQ(result.m_ReferenceIteration, uint64_t{0});
    ASSERT_EQ(result.m_Delta.getRe(), 0.0);
    ASSERT_EQ(result.m_Z.getRe(), 1.0);
    ASSERT_EQ(result.m_Z.getIm(), 2.0);
    ASSERT_TRUE(reference.m_Requests.empty());
    ASSERT_EQ(reads, uint64_t{0});
}

TEST(LAEvaluation_InitializationRestoresAbsoluteValueAndPeriodFallback)
{
    for (uint64_t maxReference : {uint64_t{0}, uint64_t{8}}) {
        EvaluationReference reference;
        reference.m_Initialize = true;
        uint64_t reads = 0;
        const auto result = EvaluateReference(reference, 12, {}, maxReference, 7, reads);
        ASSERT_EQ(result.m_Iterations, uint64_t{5});
        ASSERT_EQ(result.m_ReferenceIteration, uint64_t{0});
        ASSERT_EQ(result.m_Delta.getRe(), 100.0);
        ASSERT_EQ(result.m_Z.getRe(), 107.0);
        ASSERT_EQ(reads, uint64_t{1});
    }
}

TEST(LAEvaluation_UnusableStepMapsIntoFinerStage)
{
    EvaluationReference reference;
    reference.m_Stages = {{10, 10, false}, {20, 10, false}};
    reference.m_Steps = {{0, 3, true, {}, {}}, {2, 0, false, {2, 1}, {3, -1}}, {0, 7, true, {}, {}}};
    uint64_t reads = 0;
    const auto result = EvaluateReference(reference, 100, {}, 8, 0, reads);
    ASSERT_EQ(reference.m_Requests.size(), size_t{3});
    ASSERT_EQ(reference.m_Requests[0].m_LAIndex, uint64_t{20});
    ASSERT_EQ(reference.m_Requests[1].m_LAIndex, uint64_t{10});
    ASSERT_EQ(reference.m_Requests[1].m_Index, uint64_t{3});
    ASSERT_EQ(reference.m_Requests[2].m_Index, uint64_t{4});
    ASSERT_EQ(reference.m_Requests[2].m_Iterations, uint64_t{2});
    ASSERT_EQ(result.m_Iterations, uint64_t{2});
    ASSERT_EQ(result.m_ReferenceIteration, uint64_t{7});
    ASSERT_EQ(result.m_Delta.getRe(), 2.0);
    ASSERT_EQ(result.m_Delta.getIm(), 1.0);
    ASSERT_EQ(result.m_Z.getRe(), 5.0);
    ASSERT_EQ(result.m_Z.getIm(), 0.0);
    ASSERT_EQ(reads, uint64_t{0});
}

TEST(LAEvaluation_InvalidStageFallsBackAndBudgetStopsDescent)
{
    EvaluationReference reference;
    reference.m_Stages = {{10, 10, false}, {20, 10, true}};
    reference.m_Steps = {{2, 0, false, {1, 0}, {2, 0}}};
    uint64_t reads = 0;
    const auto result = EvaluateReference(reference, 2, {}, 8, 0, reads);
    ASSERT_EQ(reference.m_Requests.size(), size_t{1});
    ASSERT_EQ(reference.m_Requests[0].m_LAIndex, uint64_t{10});
    ASSERT_EQ(result.m_Iterations, uint64_t{2});
    ASSERT_EQ(result.m_ReferenceIteration, uint64_t{0});
    ASSERT_EQ(result.m_Delta.getRe(), 1.0);
    ASSERT_EQ(result.m_Z.getRe(), 3.0);

    reference.m_Requests.clear();
    reference.m_Stages[1].m_Invalid = false;
    EvaluateReference(reference, 2, {}, 8, 0, reads);
    ASSERT_EQ(reference.m_Requests.size(), size_t{1});
    ASSERT_EQ(reference.m_Requests[0].m_LAIndex, uint64_t{20});
}

TEST(LAEvaluation_RebasesOnMagnitudeAndStageEnd)
{
    for (const bool stageEnd : {false, true}) {
        EvaluationReference reference;
        reference.m_Stages = {{10, stageEnd ? uint64_t{1} : uint64_t{10}, false}};
        reference.m_Steps = {{2, 0, false, {4, 0}, {stageEnd ? 3.0 : -3.0, 0}}, {0, 2, true, {}, {}}};
        uint64_t reads = 0;
        const auto result = EvaluateReference(reference, 100, {}, 8, 0, reads);
        const double expected = stageEnd ? 7.0 : 1.0;
        ASSERT_EQ(reference.m_Requests[1].m_Index, uint64_t{0});
        ASSERT_EQ(reference.m_Requests[1].m_Delta.getRe(), expected);
        ASSERT_EQ(result.m_Iterations, uint64_t{2});
        ASSERT_EQ(result.m_ReferenceIteration, uint64_t{2});
        ASSERT_EQ(result.m_Delta.getRe(), expected);
        ASSERT_EQ(result.m_Z.getRe(), expected);
    }
}

TEST(LAEvaluation_ZeroBudgetDoesNotReadSteps)
{
    EvaluationReference reference;
    reference.m_Stages = {{10, 4, false}};
    uint64_t reads = 0;
    const auto result = EvaluateReference(reference, 0, {0.25, -0.5}, 8, 0, reads);
    ASSERT_TRUE(reference.m_Requests.empty());
    ASSERT_EQ(reads, uint64_t{0});
    ASSERT_EQ(result.m_Iterations, uint64_t{0});
    ASSERT_EQ(result.m_Delta.getRe(), 0.0);
    ASSERT_EQ(result.m_Z.getRe(), 0.25);
    ASSERT_EQ(result.m_Z.getIm(), -0.5);
}

TEST(LAInitialization_PreservesCpuAndCudaFallbackZeroForms)
{
    CheckBackendFallbackZeros<float>();
    CheckBackendFallbackZeros<double>();
}
