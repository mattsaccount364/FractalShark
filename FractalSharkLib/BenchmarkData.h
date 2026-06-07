#pragma once

#include "HighPrecision.h"
#include <stdint.h>

class BenchmarkData {
public:
    BenchmarkData();
    BenchmarkData(const BenchmarkData &);
    BenchmarkData(BenchmarkData &&) noexcept;
    BenchmarkData &operator=(const BenchmarkData &);
    BenchmarkData &operator=(BenchmarkData &&) noexcept;
    ~BenchmarkData();

    void StartTimer();
    void StopTimer();

    uint64_t GetDeltaInMs() const;

private:
    uint64_t m_freq;
    uint64_t m_startTime;
    uint64_t m_endTime;

    uint64_t m_DeltaTime;
};

struct ScopedBenchmarkStopper {
    ScopedBenchmarkStopper(BenchmarkData &data) : m_Data{&data} { m_Data->StartTimer(); }
    ScopedBenchmarkStopper(BenchmarkData *data) : m_Data{data}
    {
        if (m_Data != nullptr) {
            m_Data->StartTimer();
        }
    }

    ~ScopedBenchmarkStopper()
    {
        if (m_Data != nullptr) {
            m_Data->StopTimer();
        }
    }

private:
    BenchmarkData *m_Data;
};
