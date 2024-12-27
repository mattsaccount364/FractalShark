#include "TestTracker.h"
#include "HpSharkFloat.cuh"

#include <iostream>
#include <assert.h>

TestTracker::PerTest::PerTest()
    :
    DescToFailure{},
    TestMs{},
    RelativeError{},
    AcceptableError{} {
}
 
TestTracker::TestTracker()
    : m_Tests(NumTests) {
}

bool TestTracker::CheckAllTestsPassed() const {
    std::map<std::string, size_t> DescToFailureCount;

    size_t totalTests = 0;

    for (int i = 0; i < m_Tests.size(); ++i) {
        bool anyFailed = false;
        std::string combinedDesc;
        if (m_Tests[i].DescToFailure.empty()) {
            continue;
        }

        // Only count tests with some entry, success or failed.
        // This is to avoid counting tests that were not run.
        totalTests++;

        for (const auto &kv : m_Tests[i].DescToFailure) {
            if (kv.second == false) {
                // This variant passed.
                continue;
            }

            if (combinedDesc == "") {
                combinedDesc = kv.first;
            } else {
                // Append
                combinedDesc += ", " + kv.first;
            }

            DescToFailureCount[kv.first]++;
            anyFailed = true;
        }

        if (anyFailed) {
            std::cout << "Test " << i << " failed!  " <<
                combinedDesc << ", " <<
                "Error: " << m_Tests[i].RelativeError << " " <<
                "Acceptable error: " << m_Tests[i].AcceptableError << std::endl;
        }
    }

    // Print out times for those that are non-zero
    for (int i = 0; i < m_Tests.size(); ++i) {
        if (m_Tests[i].TestMs > 10) {
            std::cout << "Test " << i << " took " << m_Tests[i].TestMs << " ms" << std::endl;
        }
    }

    // Print failure rate for each desc in DescToFailureCount:
    if (!DescToFailureCount.empty()) {
        std::cout << "Some tests failed!" << std::endl;

        for (const auto &kv : DescToFailureCount) {
            auto numFailed = kv.second;
            double failurePercent = (numFailed * 100.0) / totalTests;
            std::cout << "numFailed: " << numFailed << std::endl;
            std::cout << "totalTests: " << totalTests << std::endl;
            std::cout << "Failure rate " << kv.first << ": " << failurePercent << "%" << std::endl;
        }
        return false;
    }

    std::cout << "All tests passed!" << std::endl;
    return true;
}

void TestTracker::AddTime(size_t testIndex, uint64_t ms) {
    m_Tests[testIndex].TestMs = ms;
}

void TestTracker::MarkSuccess(
    size_t testIndex,
    const std::string &description) {

    if (testIndex >= NumTests) {
        std::cout << "Test index out of range!" << std::endl;
        assert(false);
    }

    m_Tests[testIndex].DescToFailure[description] = false;
}

void TestTracker::MarkFailed(
    size_t testIndex,
    const std::string &description,
    const std::string &relativeError,
    const std::string &acceptableError) {

    if (testIndex >= NumTests) {
        std::cout << "Test index out of range!" << std::endl;
        assert(false);
    }

    m_Tests[testIndex].DescToFailure[description] = true;
    m_Tests[testIndex].RelativeError = relativeError;
    m_Tests[testIndex].AcceptableError = acceptableError;
}