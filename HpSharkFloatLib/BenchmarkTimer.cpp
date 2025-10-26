#include "BenchmarkTimer.h"

#define NOMINMAX
#include <windows.h>

BenchmarkTimer::BenchmarkTimer() : m_freq{}, m_startTime{}, m_endTime{}, m_DeltaTime{}
{

    LARGE_INTEGER freq;

    QueryPerformanceFrequency(&freq);
    m_freq = freq.QuadPart;
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

    LARGE_INTEGER startTime;
    QueryPerformanceCounter(&startTime);
    m_startTime = startTime.QuadPart;
}

void
BenchmarkTimer::StopTimer()
{
    LARGE_INTEGER endTime;
    QueryPerformanceCounter(&endTime);
    m_endTime = endTime.QuadPart;

    m_DeltaTime = m_endTime - m_startTime;
}

uint64_t
BenchmarkTimer::GetDeltaInMs() const
{
    double timeTakenMs = (double)m_DeltaTime * 1000.0 / (double)m_freq;
    return (uint64_t)timeTakenMs;
}
