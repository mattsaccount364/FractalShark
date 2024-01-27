#pragma once

#include <stdint.h>
#include "HighPrecision.h"

class BenchmarkData {
public:
    BenchmarkData();
    BenchmarkData(const BenchmarkData&) = default;
    BenchmarkData(BenchmarkData&&) = default;
    BenchmarkData& operator=(const BenchmarkData&) = default;
    BenchmarkData& operator=(BenchmarkData&&) = default;
    ~BenchmarkData() = default;

    void StartTimer();
    void StopTimer();

    uint64_t GetDeltaInMs() const;

private:
    uint64_t m_freq;
    uint64_t m_startTime;
    uint64_t m_endTime;

    uint64_t m_DeltaTime;
};
