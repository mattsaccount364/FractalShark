#pragma once

#include <cstdint>
#include <gmp.h>
#include <map>
#include <string>
#include <vector>

class TestTracker {
public:
    struct PerTest {
        PerTest();

        std::map<std::string, bool> DescToFailure;
        uint64_t TestMs;
        std::string RelativeError;
        std::string AcceptableError;
    };

    constexpr static auto NumTests = 20000;

    TestTracker();

    bool CheckAllTestsPassed() const;
    void AddTime(size_t testIndex, uint64_t ms);

    void MarkSuccess(size_t testIndex, const std::string &description);

    void MarkFailed(size_t testIndex,
                    const std::string &description,
                    const std::string &relativeError,
                    const std::string &acceptableError);

private:
    std::vector<PerTest> m_Tests;
};
