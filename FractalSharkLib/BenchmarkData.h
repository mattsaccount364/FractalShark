#pragma once

#include <stdint.h>
#include "HighPrecision.h"

class WaitCursor;

class BenchmarkData {
public:
    BenchmarkData();
    BenchmarkData(const BenchmarkData &);
    BenchmarkData(BenchmarkData &&);
    BenchmarkData &operator=(const BenchmarkData &);
    BenchmarkData &operator=(BenchmarkData &&);
    ~BenchmarkData();

    void StartTimer();
    void StopTimer();

    uint64_t GetDeltaInMs() const;

private:
    uint64_t m_freq;
    uint64_t m_startTime;
    uint64_t m_endTime;

    uint64_t m_DeltaTime;

    std::unique_ptr<WaitCursor> m_WaitCursor;
};

struct ScopedBenchmarkStopper {
    ScopedBenchmarkStopper(BenchmarkData &data) :
        m_Data(data) {
        m_Data.StartTimer();
    }

    ~ScopedBenchmarkStopper() {
        m_Data.StopTimer();
    }

private:
    BenchmarkData &m_Data;
};