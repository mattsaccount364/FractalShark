#pragma once

#include "HpSharkFloat.h"
#include "HpSharkTestConfig.h"

#include <iostream>
#include <string>
#include <vector>

// Collected timing from a single perf test iteration (shared by NR and non-NR paths).
struct PerfTimingResult {
    double hostMs = 0.0;
    double gpuMs = 0.0;
    double cpuRefMs = 0.0;
    double cpuRef2Ms = 0.0;
};

// Print a tab-separated perf summary table.
// hostLabel column shows "Host-MT(ms)" or "Host-ST(ms)" (or "MPIR-MT/ST" etc.) based on useMT.
inline void
PrintPerfSummaryTable(const char *viewName,
                      bool useMT,
                      const std::vector<PerfTimingResult> &timings,
                      const char *hostPrefix = "Host")
{
    const std::string hostLabel = std::string(hostPrefix) + (useMT ? "-MT(ms)" : "-ST(ms)");

    std::cout << "\n=== " << viewName << " PERF SUMMARY (tab-separated) ===" << std::endl;

    // Header
    std::cout << "Iter\t" << hostLabel << "\tCPU-ref(ms)\tCPU-ref2(ms)";
    if constexpr (HpShark::TestGpu) {
        std::cout << "\tGPU(ms)";
    }
    std::cout << std::endl;

    // Per-iteration rows
    double totalHost = 0, totalCpu = 0, totalCpu2 = 0, totalGpu = 0;
    for (size_t i = 0; i < timings.size(); ++i) {
        std::cout << i << "\t" << timings[i].hostMs;
        totalHost += timings[i].hostMs;
        std::cout << "\t" << timings[i].cpuRefMs << "\t" << timings[i].cpuRef2Ms;
        totalCpu += timings[i].cpuRefMs;
        totalCpu2 += timings[i].cpuRef2Ms;
        if constexpr (HpShark::TestGpu) {
            std::cout << "\t" << timings[i].gpuMs;
            totalGpu += timings[i].gpuMs;
        }
        std::cout << std::endl;
    }

    // Totals
    std::cout << "Total\t" << totalHost << "\t" << totalCpu << "\t" << totalCpu2;
    if constexpr (HpShark::TestGpu) {
        std::cout << "\t" << totalGpu;
    }
    std::cout << std::endl;

    // Speedup
    if constexpr (HpShark::TestGpu) {
        if (totalGpu > 0) {
            std::cout << "Speedup\t" << (totalHost / totalGpu) << "x\t" << (totalCpu / totalGpu) << "x\t"
                      << (totalCpu2 / totalGpu) << "x";
            std::cout << "\t(vs GPU)" << std::endl;
        }
    }
}
