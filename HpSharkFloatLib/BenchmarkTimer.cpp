#include "BenchmarkTimer.h"
#include "Environment.h"

BenchmarkTimer::BenchmarkTimer() : m_freq{}, m_startTime{}, m_endTime{}, m_DeltaTime{}
{
    m_freq = Environment::HighResFrequency();
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
