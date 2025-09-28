#pragma once

#include <stdint.h>

class WaitCursor;

class BenchmarkTimer {
public:
    BenchmarkTimer();
    BenchmarkTimer(const BenchmarkTimer &);
    BenchmarkTimer(BenchmarkTimer &&) noexcept;
    BenchmarkTimer &operator=(const BenchmarkTimer &);
    BenchmarkTimer &operator=(BenchmarkTimer &&) noexcept;
    ~BenchmarkTimer();

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
    ScopedBenchmarkStopper(BenchmarkTimer &data) : m_Data(data) { m_Data.StartTimer(); }

    ~ScopedBenchmarkStopper() { m_Data.StopTimer(); }

private:
    BenchmarkTimer &m_Data;
};
