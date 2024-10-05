#include "TestTracker.h"
#include "HpGpu.cuh"

#include <iostream>
#include <assert.h>

TestTracker::PerTest::PerTest()
    : Failed{},
    TestMs{} {
}
 
TestTracker::TestTracker()
    : m_Tests(NumTests) {
}

bool TestTracker::CheckAllTestsPassed() const {
    bool anyFailed = false;
    for (int i = 0; i < NumTests; ++i) {
        if (m_Tests[i].Failed) {
            std::cout << "Test " << i << " failed!  Delta: " << m_Tests[i].Delta << std::endl;
            anyFailed = true;
        }
    }

    // Print out times for those that are non-zero
    for (int i = 0; i < NumTests; ++i) {
        if (m_Tests[i].TestMs > 0) {
            std::cout << "Test " << i << " took " << m_Tests[i].TestMs << " ms" << std::endl;
        }
    }

    if (anyFailed) {
        std::cout << "Some tests failed!" << std::endl;
        return false;
    }

    std::cout << "All tests passed!" << std::endl;
    return true;
}

void TestTracker::AddTime(size_t testIndex, uint64_t ms) {
    m_Tests[testIndex].TestMs = ms;
}

void TestTracker::MarkFailed(size_t testIndex) {
    m_Tests[testIndex].Failed = true;

}

void TestTracker::MarkFailed(size_t testIndex, const mpf_t delta) {
    if (testIndex >= NumTests) {
        std::cout << "Test index out of range!" << std::endl;
        assert(false);
    }

    m_Tests[testIndex].Failed = true;
    m_Tests[testIndex].Delta = MpfToString(delta, LowPrec);
}