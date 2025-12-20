#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace HpShark {
struct LaunchParams;
}

class TestTracker {
public:
    static constexpr size_t NumTests = 20000; // keep your existing value if different

    enum class VariantStatus : uint8_t { NotRun, Passed, Failed };

    struct VariantResult {
        VariantStatus status = VariantStatus::NotRun;
        std::string relativeError;
        std::string acceptableError;
    };

    struct PerTest {
        PerTest();

        // Per-variant results for this test index
        std::unordered_map<std::string, VariantResult> Variants;

        uint64_t TestMs = 0;

        // Launch config (optional, but useful for reporting)
        int NumBlocks = 0;
        int ThreadsPerBlock = 0;

        // Explicit “ran” bit; avoids inferring from map emptiness / launch params
        bool RanAnyVariant = false;
    };

    TestTracker();

    bool CheckAllTestsPassed() const;

    void AddTime(size_t testIndex, uint64_t ms);

    void MarkSuccess(const HpShark::LaunchParams *launchParams,
                     size_t testIndex,
                     const std::string &description);

    void MarkFailed(const HpShark::LaunchParams *launchParams,
                    size_t testIndex,
                    const std::string &description,
                    const std::string &relativeError,
                    const std::string &acceptableError);

private:
    std::vector<PerTest> m_Tests;
};
