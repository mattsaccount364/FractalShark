#pragma once

#include "BenchmarkData.h"

class BenchmarkDataCollection {
public:
    BenchmarkData m_Overall;
    BenchmarkData m_PerPixel;
    BenchmarkData m_RefOrbitSave;
    BenchmarkData m_RefOrbitLoad;
};