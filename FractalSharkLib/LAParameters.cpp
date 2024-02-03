#include "LAParameters.h"


int LAParameters::GetDefaultDetectionMethod() const {
    return 1;
}

// These apply to both detection methods
float LAParameters::GetDefaultLAThresholdScale() const {
    return 0x1.0p-24; // Imagina: 0x1.0p-24
}

float LAParameters::GetDefaultLAThresholdCScale() const {
    return 0x1.0p-24; // Imagina: 0x1.0p-24
}

// These apply to detection method 1
float LAParameters::GetDefaultStage0PeriodDetectionThreshold2() const {
    return 0x1.0p-6; // Imagina: 0x1.0p-6
}

float LAParameters::GetDefaultPeriodDetectionThreshold2() const {
    return 0x1.0p-3; // Imagina: 0x1.0p-3
}

// These apply to detection method 2
float LAParameters::GetDefaultStage0PeriodDetectionThreshold() const {
    return 0x1.0p-10; // Imagina: 0x1.0p-10
}

float LAParameters::GetDefaultPeriodDetectionThreshold() const {
    return 0x1.0p-10; // Imagina: 0x1.0p-10
}