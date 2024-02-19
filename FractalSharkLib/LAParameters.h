#pragma once

#include "HighPrecision.h"

class LAParameters {
public:
    LAParameters();

    enum class LAThreadingAlgorithm {
        SingleThreaded,
        MultiThreaded
    };

    enum class LADefaults {
        MaxAccuracy,
        MaxPerf,
        MinMemory
    };

    CUDA_CRAP int GetDetectionMethod() const;
    CUDA_CRAP float GetLAThresholdScale() const;
    CUDA_CRAP float GetLAThresholdCScale() const;
    CUDA_CRAP float GetStage0PeriodDetectionThreshold2() const;
    CUDA_CRAP float GetPeriodDetectionThreshold2() const;
    CUDA_CRAP float GetStage0PeriodDetectionThreshold() const;
    CUDA_CRAP float GetPeriodDetectionThreshold() const;

    CUDA_CRAP int32_t GetLAThresholdScaleExp() const;
    CUDA_CRAP int32_t GetLAThresholdCScaleExp() const;
    CUDA_CRAP int32_t GetStage0PeriodDetectionThreshold2Exp() const;
    CUDA_CRAP int32_t GetPeriodDetectionThreshold2Exp() const;
    CUDA_CRAP int32_t GetStage0PeriodDetectionThresholdExp() const;
    CUDA_CRAP int32_t GetPeriodDetectionThresholdExp() const;

    CUDA_CRAP void AdjustLAThresholdScaleExponent(int32_t delta_exponent);
    CUDA_CRAP void AdjustLAThresholdCScaleExponent(int32_t delta_exponent);
    CUDA_CRAP void AdjustStage0PeriodDetectionThreshold2Exponent(int32_t delta_exponent);
    CUDA_CRAP void AdjustPeriodDetectionThreshold2Exponent(int32_t delta_exponent);
    CUDA_CRAP void AdjustStage0PeriodDetectionThresholdExponent(int32_t delta_exponent);
    CUDA_CRAP void AdjustPeriodDetectionThresholdExponent(int32_t delta_exponent);

    CUDA_CRAP void SetThreading(LAThreadingAlgorithm algorithm);
    CUDA_CRAP LAThreadingAlgorithm GetThreading() const;

    CUDA_CRAP void SetDefaults(LADefaults defaults);

    bool ReadMetadata(std::ifstream& metafile);
    bool WriteMetadata(std::ofstream& metafile) const;

private:
    bool ReadLine(std::ifstream &metafile, int32_t &value, const char *name);

    void PopulateFloatsFromExponents();

    int32_t m_DetectionMethod;
    float m_LAThresholdScale;
    float m_LAThresholdCScale;
    float m_Stage0PeriodDetectionThreshold2;
    float m_PeriodDetectionThreshold2;
    float m_Stage0PeriodDetectionThreshold;
    float m_PeriodDetectionThreshold;

    int32_t m_LAThresholdScaleExponent;
    int32_t m_LAThresholdCScaleExponent;
    int32_t m_Stage0PeriodDetectionThreshold2Exponent;
    int32_t m_PeriodDetectionThreshold2Exponent;
    int32_t m_Stage0PeriodDetectionThresholdExponent;
    int32_t m_PeriodDetectionThresholdExponent;

    LAThreadingAlgorithm m_ThreadingAlgorithm;

    static constexpr int32_t DefaultDetectionMethod = 1;
    static constexpr int32_t DefaultLAThresholdScaleExponent = -24;
    static constexpr int32_t DefaultLAThresholdCScaleExponent = -24;
    static constexpr int32_t DefaultStage0PeriodDetectionThreshold2Exponent = -6;
    static constexpr int32_t DefaultPeriodDetectionThreshold2Exponent = -3;
    static constexpr int32_t DefaultStage0PeriodDetectionThresholdExponent = -10;
    static constexpr int32_t DefaultPeriodDetectionThresholdExponent = -10;

    static constexpr LAThreadingAlgorithm DefaultThreadingAlgorithm = LAThreadingAlgorithm::MultiThreaded;
 };

