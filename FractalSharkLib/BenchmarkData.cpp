#include "stdafx.h"
#include "BenchmarkData.h"

#include <windows.h>

BenchmarkData::BenchmarkData() :
    m_freq{},
    m_startTime{},
    m_endTime{},
    m_DeltaTime{} {

    LARGE_INTEGER freq;

    QueryPerformanceFrequency(&freq);
    m_freq = freq.QuadPart;
}

void BenchmarkData::StartTimer() {
    m_startTime = 0;
    m_endTime = 1;

    LARGE_INTEGER startTime;
    QueryPerformanceCounter(&startTime);
    m_startTime = startTime.QuadPart;
}

void BenchmarkData::StopTimer() {
    LARGE_INTEGER endTime;
    QueryPerformanceCounter(&endTime);
    m_endTime = endTime.QuadPart;

    m_DeltaTime = m_endTime - m_startTime;
}

uint64_t BenchmarkData::GetDeltaInMs() const {
    double timeTakenMs = (double)m_DeltaTime * 1000.0 / (double)m_freq;
    return (uint64_t)timeTakenMs;
}
