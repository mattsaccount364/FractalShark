#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <gmp.h>

class TestTracker {
public:
    struct PerTest {
        PerTest();

        bool Failed;
        uint64_t TestMs;
        std::string RelativeError;
        std::string AcceptableError;
    };

    constexpr static auto NumTests = 1024;

    TestTracker();

    bool CheckAllTestsPassed() const;
    void AddTime(size_t testIndex, uint64_t ms);
    void MarkFailed(size_t testIndex);
    void MarkFailed(
        size_t testIndex,
        const mpf_t relativeError,
        const mpf_t acceptableError);

private:
    std::vector<PerTest> m_Tests;
};