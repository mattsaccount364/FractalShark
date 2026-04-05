#include "BenchmarkTimer.h"
#include "Environment.h"

BenchmarkTimer::BenchmarkTimer() : m_freq{}, m_startTime{}, m_endTime{}, m_DeltaTime{}
{
    m_freq = Environment::HighResFrequency();
}

BenchmarkTimer::~BenchmarkTimer() {}

BenchmarkTimer::BenchmarkTimer(const BenchmarkTimer &other)
    : m_freq{other.m_freq}, m_startTime{other.m_startTime}, m_endTime{other.m_endTime},
      m_DeltaTime{other.m_DeltaTime}
{
}

BenchmarkTimer::BenchmarkTimer(BenchmarkTimer &&other) noexcept
    : m_freq{other.m_freq}, m_startTime{other.m_startTime}, m_endTime{other.m_endTime},
      m_DeltaTime{other.m_DeltaTime}
{
}

BenchmarkTimer &
BenchmarkTimer::operator=(const BenchmarkTimer &other)
{
    if (this != &other) {
        m_freq = other.m_freq;
        m_startTime = other.m_startTime;
        m_endTime = other.m_endTime;
        m_DeltaTime = other.m_DeltaTime;
    }

    return *this;
}

BenchmarkTimer &
BenchmarkTimer::operator=(BenchmarkTimer &&other) noexcept
{
    if (this != &other) {
        m_freq = other.m_freq;
        m_startTime = other.m_startTime;
        m_endTime = other.m_endTime;
        m_DeltaTime = other.m_DeltaTime;
    }

    return *this;
}

void
BenchmarkTimer::StartTimer()
{
    m_startTime = 0;
    m_endTime = 1;
    m_startTime = Environment::HighResCounter();
}

void
BenchmarkTimer::StopTimer()
{
    m_endTime = Environment::HighResCounter();
    m_DeltaTime = m_endTime - m_startTime;
}

uint64_t
BenchmarkTimer::GetDeltaInMs() const
{
    double timeTakenMs = (double)m_DeltaTime * 1000.0 / (double)m_freq;
    return (uint64_t)timeTakenMs;
}
