#include "stdafx.h"
#include "BenchmarkData.h"
#include "Environment.h"

BenchmarkData::BenchmarkData() : m_freq{}, m_startTime{}, m_endTime{}, m_DeltaTime{}
{
    m_freq = Environment::HighResFrequency();
}

BenchmarkData::~BenchmarkData() {}

BenchmarkData::BenchmarkData(const BenchmarkData &other)
    : m_freq{other.m_freq}, m_startTime{other.m_startTime}, m_endTime{other.m_endTime},
      m_DeltaTime{other.m_DeltaTime}
{
}

BenchmarkData::BenchmarkData(BenchmarkData &&other) noexcept
    : m_freq{other.m_freq}, m_startTime{other.m_startTime}, m_endTime{other.m_endTime},
      m_DeltaTime{other.m_DeltaTime}
{
}

BenchmarkData &
BenchmarkData::operator=(const BenchmarkData &other)
{
    if (this != &other) {
        m_freq = other.m_freq;
        m_startTime = other.m_startTime;
        m_endTime = other.m_endTime;
        m_DeltaTime = other.m_DeltaTime;
    }

    return *this;
}

BenchmarkData &
BenchmarkData::operator=(BenchmarkData &&other) noexcept
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
BenchmarkData::StartTimer()
{
    m_startTime = 0;
    m_endTime = 1;
    m_startTime = Environment::HighResCounter();
}

void
BenchmarkData::StopTimer()
{
    m_endTime = Environment::HighResCounter();
    m_DeltaTime = m_endTime - m_startTime;
}

uint64_t
BenchmarkData::GetDeltaInMs() const
{
    double timeTakenMs = (double)m_DeltaTime * 1000.0 / (double)m_freq;
    return (uint64_t)timeTakenMs;
}
