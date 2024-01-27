#include "stdafx.h"
#include "BenchmarkData.h"

#include <windows.h>

BenchmarkData::BenchmarkData() :
    m_freq{},
    m_startTime{},
    m_endTime{},
    m_DeltaTime{} {

    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&m_freq));
}

void BenchmarkData::StartTimer() {
    m_startTime = 0;
    m_endTime = 1;

    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_startTime));
    return;
}

void BenchmarkData::StopTimer() {
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_endTime));

    m_DeltaTime = m_endTime - m_startTime;
}

uint64_t BenchmarkData::GetDeltaInMs() const {
    double timeTaken = (double)m_DeltaTime / (double)m_freq;
    return (uint64_t)(timeTaken * 1000.0);
}
