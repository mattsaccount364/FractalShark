#include "LAParameters.h"

#include <fstream>

LAParameters::LAParameters() :
    m_DetectionMethod{ DefaultDetectionMethod },
    m_LAThresholdScaleExponent{ DefaultLAThresholdScaleExponent },
    m_LAThresholdCScaleExponent{ DefaultLAThresholdCScaleExponent },
    m_Stage0PeriodDetectionThreshold2Exponent{ DefaultStage0PeriodDetectionThreshold2Exponent },
    m_PeriodDetectionThreshold2Exponent{ DefaultPeriodDetectionThreshold2Exponent },
    m_Stage0PeriodDetectionThresholdExponent{ DefaultStage0PeriodDetectionThresholdExponent },
    m_PeriodDetectionThresholdExponent{ DefaultPeriodDetectionThresholdExponent }
{
    PopulateFloatsFromExponents();
}

bool LAParameters::ReadLine(std::ifstream& metafile, int32_t& value, const char* name) {
    std::string line;
    metafile >> line;
    if (line != name) {
        return false;
    }

    metafile >> value;
    return true;
}

void LAParameters::PopulateFloatsFromExponents() {
    m_LAThresholdScale = static_cast<float>(std::exp2(m_LAThresholdScaleExponent));
    m_Stage0PeriodDetectionThreshold2 = static_cast<float>(std::exp2(m_Stage0PeriodDetectionThreshold2Exponent));
    m_PeriodDetectionThreshold = static_cast<float>(std::exp2(m_PeriodDetectionThresholdExponent));
    m_PeriodDetectionThreshold2 = static_cast<float>(std::exp2(m_PeriodDetectionThreshold2Exponent));
    m_LAThresholdCScale = static_cast<float>(std::exp2(m_LAThresholdCScaleExponent));
    m_Stage0PeriodDetectionThreshold = static_cast<float>(std::exp2(m_Stage0PeriodDetectionThresholdExponent));
}

bool LAParameters::ReadMetadata(std::ifstream &metafile) {
    std::string line;
    metafile >> line;
    if (line != "LAParameters:") {
        return false;
    }

    bool ret = true;
    ret &= ReadLine(metafile, m_DetectionMethod, "DetectionMethod:");
    ret &= ReadLine(metafile, m_LAThresholdScaleExponent, "LAThresholdScale:");
    ret &= ReadLine(metafile, m_LAThresholdCScaleExponent, "LAThresholdCScale:");
    ret &= ReadLine(metafile, m_Stage0PeriodDetectionThreshold2Exponent, "Stage0PeriodDetectionThreshold2:");
    ret &= ReadLine(metafile, m_PeriodDetectionThreshold2Exponent, "PeriodDetectionThreshold2:");
    ret &= ReadLine(metafile, m_Stage0PeriodDetectionThresholdExponent, "Stage0PeriodDetectionThreshold:");
    ret &= ReadLine(metafile, m_PeriodDetectionThresholdExponent, "PeriodDetectionThreshold:");
    if (!ret) {
        return false;
    }

    PopulateFloatsFromExponents();
    return true;
}

bool LAParameters::WriteMetadata(std::ofstream &metafile) const {
    metafile << "LAParameters:" << std::endl;
    metafile << "DetectionMethod: " << m_DetectionMethod << std::endl;
    metafile << "LAThresholdScale: " << m_LAThresholdScaleExponent << std::endl;
    metafile << "LAThresholdCScale: " << m_LAThresholdCScaleExponent << std::endl;
    metafile << "Stage0PeriodDetectionThreshold2: " << m_Stage0PeriodDetectionThreshold2Exponent << std::endl;
    metafile << "PeriodDetectionThreshold2: " << m_PeriodDetectionThreshold2Exponent << std::endl;
    metafile << "Stage0PeriodDetectionThreshold: " << m_Stage0PeriodDetectionThresholdExponent << std::endl;
    metafile << "PeriodDetectionThreshold: " << m_PeriodDetectionThresholdExponent << std::endl;
    return true;
}

int LAParameters::GetDetectionMethod() const {
    return m_DetectionMethod;
}

float LAParameters::GetLAThresholdScale() const {
    return m_LAThresholdScale;
}

float LAParameters::GetLAThresholdCScale() const {
    return m_LAThresholdCScale;
}

float LAParameters::GetStage0PeriodDetectionThreshold2() const {
    return m_Stage0PeriodDetectionThreshold2;
}

float LAParameters::GetPeriodDetectionThreshold2() const {
    return m_PeriodDetectionThreshold2;
}

float LAParameters::GetStage0PeriodDetectionThreshold() const {
    return m_Stage0PeriodDetectionThreshold;
}

float LAParameters::GetPeriodDetectionThreshold() const {
    return m_PeriodDetectionThreshold;
}

int32_t LAParameters::GetLAThresholdScaleExp() const {
    return m_LAThresholdScaleExponent;
}

int32_t LAParameters::GetLAThresholdCScaleExp() const {
    return m_LAThresholdCScaleExponent;
}

int32_t LAParameters::GetStage0PeriodDetectionThreshold2Exp() const {
    return m_Stage0PeriodDetectionThreshold2Exponent;
}

int32_t LAParameters::GetPeriodDetectionThreshold2Exp() const {
    return m_PeriodDetectionThreshold2Exponent;
}

int32_t LAParameters::GetStage0PeriodDetectionThresholdExp() const {
    return m_Stage0PeriodDetectionThresholdExponent;
}

int32_t LAParameters::GetPeriodDetectionThresholdExp() const {
    return m_PeriodDetectionThresholdExponent;
}

void LAParameters::AdjustLAThresholdScaleExponent(int32_t delta_exponent) {
    m_LAThresholdScaleExponent += delta_exponent;
    PopulateFloatsFromExponents();
}

void LAParameters::AdjustLAThresholdCScaleExponent(int32_t delta_exponent) {
    m_LAThresholdCScaleExponent += delta_exponent;
    PopulateFloatsFromExponents();
}

void LAParameters::AdjustStage0PeriodDetectionThreshold2Exponent(int32_t delta_exponent) {
    m_Stage0PeriodDetectionThreshold2Exponent += delta_exponent;
    PopulateFloatsFromExponents();
}

void LAParameters::AdjustPeriodDetectionThreshold2Exponent(int32_t delta_exponent) {
    m_PeriodDetectionThreshold2Exponent += delta_exponent;
    PopulateFloatsFromExponents();
}

void LAParameters::AdjustStage0PeriodDetectionThresholdExponent(int32_t delta_exponent) {
    m_Stage0PeriodDetectionThresholdExponent += delta_exponent;
    PopulateFloatsFromExponents();
}

void LAParameters::AdjustPeriodDetectionThresholdExponent(int32_t delta_exponent) {
    m_PeriodDetectionThresholdExponent += delta_exponent;
    PopulateFloatsFromExponents();
}