#pragma once

#include <stdint.h>

class BenchmarkTimer {
public:
    BenchmarkTimer();
    BenchmarkTimer(const BenchmarkTimer &) = default;
    BenchmarkTimer(BenchmarkTimer &&) noexcept = default;
    BenchmarkTimer &operator=(const BenchmarkTimer &) = default;
    BenchmarkTimer &operator=(BenchmarkTimer &&) noexcept = default;
    ~BenchmarkTimer() = default;

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
    ScopedBenchmarkStopper(BenchmarkTimer &data) : m_Data{&data} { m_Data->StartTimer(); }
    ScopedBenchmarkStopper(BenchmarkTimer *data) : m_Data{data}
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
    BenchmarkTimer *m_Data;
};
