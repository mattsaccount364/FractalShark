#include "stdafx.h"
#include "BenchmarkData.h"
#include "Environment.h"

BenchmarkData::BenchmarkData() : m_freq{}, m_startTime{}, m_endTime{}, m_DeltaTime{}
{
    m_freq = Environment::HighResFrequency();
}

BenchmarkData::~BenchmarkData() = default;

BenchmarkData::BenchmarkData(const BenchmarkData &) = default;

BenchmarkData::BenchmarkData(BenchmarkData &&) noexcept = default;

BenchmarkData &BenchmarkData::operator=(const BenchmarkData &) = default;

BenchmarkData &BenchmarkData::operator=(BenchmarkData &&) noexcept = default;

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
