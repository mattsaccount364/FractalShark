#include "TestFramework.h"
#include "LAParameters.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Default construction
// ---------------------------------------------------------------------------

TEST(LAParam_DefaultConstruction)
{
    LAParameters p;
    ASSERT_EQ(p.GetDetectionMethod(), 1);
    ASSERT_TRUE(p.GetThreading() == LAParameters::LAThreadingAlgorithm::MultiThreaded);
}

TEST(LAParam_DefaultExponents)
{
    LAParameters p;
    ASSERT_EQ(p.GetLAThresholdScaleExp(), -24);
    ASSERT_EQ(p.GetLAThresholdCScaleExp(), -24);
    ASSERT_EQ(p.GetStage0PeriodDetectionThreshold2Exp(), -6);
    ASSERT_EQ(p.GetPeriodDetectionThreshold2Exp(), -3);
    ASSERT_EQ(p.GetStage0PeriodDetectionThresholdExp(), -10);
    ASSERT_EQ(p.GetPeriodDetectionThresholdExp(), -10);
}

TEST(LAParam_FloatsMatchExponents)
{
    LAParameters p;
    ASSERT_NEAR(p.GetLAThresholdScale(), std::exp2(-24.0f), 1e-15f);
    ASSERT_NEAR(p.GetLAThresholdCScale(), std::exp2(-24.0f), 1e-15f);
    ASSERT_NEAR(p.GetStage0PeriodDetectionThreshold2(), std::exp2(-6.0f), 1e-10f);
    ASSERT_NEAR(p.GetPeriodDetectionThreshold2(), std::exp2(-3.0f), 1e-10f);
    ASSERT_NEAR(p.GetPeriodDetectionThreshold(), std::exp2(-10.0f), 1e-10f);
}

// ---------------------------------------------------------------------------
// Exponent adjustment
// ---------------------------------------------------------------------------

TEST(LAParam_AdjustThresholdScale)
{
    LAParameters p;
    float orig = p.GetLAThresholdScale();
    p.AdjustLAThresholdScaleExponent(1);
    ASSERT_NEAR(p.GetLAThresholdScale(), orig * 2.0f, 1e-15f);
    ASSERT_EQ(p.GetLAThresholdScaleExp(), -23);
}

TEST(LAParam_AdjustThresholdCScale)
{
    LAParameters p;
    p.AdjustLAThresholdCScaleExponent(-5);
    ASSERT_EQ(p.GetLAThresholdCScaleExp(), -29);
    ASSERT_NEAR(p.GetLAThresholdCScale(), std::exp2(-29.0f), 1e-15f);
}

TEST(LAParam_AdjustPeriodDetection)
{
    LAParameters p;
    p.AdjustPeriodDetectionThreshold2Exponent(3);
    ASSERT_EQ(p.GetPeriodDetectionThreshold2Exp(), 0);
    ASSERT_NEAR(p.GetPeriodDetectionThreshold2(), 1.0f, 1e-10f);
}

// ---------------------------------------------------------------------------
// SetDefaults variants
// ---------------------------------------------------------------------------

TEST(LAParam_SetDefaults_MaxAccuracy)
{
    LAParameters p;
    p.AdjustLAThresholdScaleExponent(100); // perturb first
    p.SetDefaults(LAParameters::LADefaults::MaxAccuracy);
    ASSERT_EQ(p.GetLAThresholdScaleExp(), -24);
    ASSERT_EQ(p.GetLAThresholdCScaleExp(), -24);
}

TEST(LAParam_SetDefaults_MaxPerf)
{
    LAParameters p;
    p.SetDefaults(LAParameters::LADefaults::MaxPerf);
    // MaxPerf adds +12 to threshold scale exponents
    ASSERT_EQ(p.GetLAThresholdScaleExp(), -24 + 12);
    ASSERT_EQ(p.GetLAThresholdCScaleExp(), -24 + 12);
    // Period detection unchanged from defaults
    ASSERT_EQ(p.GetStage0PeriodDetectionThreshold2Exp(), -6);
}

TEST(LAParam_SetDefaults_MinMemory)
{
    LAParameters p;
    p.SetDefaults(LAParameters::LADefaults::MinMemory);
    // MinMemory adds +3 to period detection threshold2 exponents
    ASSERT_EQ(p.GetStage0PeriodDetectionThreshold2Exp(), -6 + 3);
    ASSERT_EQ(p.GetPeriodDetectionThreshold2Exp(), -3 + 3);
    // Threshold scales unchanged
    ASSERT_EQ(p.GetLAThresholdScaleExp(), -24);
}

// ---------------------------------------------------------------------------
// Threading
// ---------------------------------------------------------------------------

TEST(LAParam_Threading)
{
    LAParameters p;
    ASSERT_TRUE(p.GetThreading() == LAParameters::LAThreadingAlgorithm::MultiThreaded);
    p.SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    ASSERT_TRUE(p.GetThreading() == LAParameters::LAThreadingAlgorithm::SingleThreaded);
}
