#include "TestTracker.h"
#include "HpSharkFloat.h"
#include "TestVerbose.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static const char *
StatusToString(TestTracker::VariantStatus s)
{
    switch (s) {
        case TestTracker::VariantStatus::NotRun:
            return "NotRun";
        case TestTracker::VariantStatus::Passed:
            return "Passed";
        case TestTracker::VariantStatus::Failed:
            return "Failed";
        default:
            return "Unknown";
    }
}

TestTracker::PerTest::PerTest()
    : Variants(), TestMs(0), NumBlocks(0), ThreadsPerBlock(0), RanAnyVariant(false)
{
}

TestTracker::TestTracker() : m_Tests(NumTests) {}

bool
TestTracker::CheckAllTestsPassed() const
{
    // Per-variant stats (correct denominators)
    std::unordered_map<std::string, size_t> ranByVariant;
    std::unordered_map<std::string, size_t> failedByVariant;

    size_t ranTests = 0;
    size_t failedTests = 0;

    // Track slow tests for summary
    std::vector<std::pair<uint64_t, size_t>> slow; // (ms, testIndex)
    slow.reserve(m_Tests.size());

    for (size_t i = 0; i < m_Tests.size(); ++i) {
        const auto &t = m_Tests[i];

        if (!t.RanAnyVariant) {
            continue;
        }
        ranTests++;

        if (t.TestMs != 0) {
            slow.emplace_back(t.TestMs, i);
        }

        bool anyFailed = false;
        std::string failedVariants;

        for (const auto &kv : t.Variants) {
            const std::string &desc = kv.first;
            const VariantResult &vr = kv.second;

            if (vr.status == VariantStatus::NotRun) {
                continue;
            }

            ranByVariant[desc]++;

            if (vr.status == VariantStatus::Failed) {
                failedByVariant[desc]++;
                anyFailed = true;
                if (!failedVariants.empty())
                    failedVariants += ", ";
                failedVariants += desc;
            }

            if (SharkVerbose == VerboseMode::Debug) {
                std::cout << "Test " << i << " [" << t.NumBlocks << "x" << t.ThreadsPerBlock << "] "
                          << "Variant: " << desc << " -> " << StatusToString(vr.status) << "\n";
                if (vr.status == VariantStatus::Failed) {
                    std::cout << "  Error: " << vr.relativeError
                              << "  Acceptable: " << vr.acceptableError << "\n";
                }
            }
        }

        if (anyFailed) {
            failedTests++;
            std::cout << "Test " << i << " FAILED"
                      << " [" << t.NumBlocks << "x" << t.ThreadsPerBlock << "] " << failedVariants
                      << "\n";
        } else if (SharkVerbose == VerboseMode::Debug) {
            std::cout << "Test " << i << " PASSED"
                      << " [" << t.NumBlocks << "x" << t.ThreadsPerBlock << "] "
                      << "Time: " << t.TestMs << " ms\n";
        }
    }

    // Slow test summary: top 10
    if (!slow.empty()) {
        std::sort(
            slow.begin(), slow.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

        const size_t topN = (slow.size() < 10) ? slow.size() : 10;
        std::cout << "Top " << topN << " slowest tests:\n";
        for (size_t k = 0; k < topN; ++k) {
            auto ms = slow[k].first;
            auto idx = slow[k].second;
            std::cout << "  Test " << idx << ": " << ms << " ms"
                      << " [" << m_Tests[idx].NumBlocks << "x" << m_Tests[idx].ThreadsPerBlock << "]\n";
        }
    }

    if (ranTests == 0) {
        std::cout << "No tests were run.\n";
        return true;
    }

    if (!failedByVariant.empty()) {
        std::cout << "Some tests failed! (" << failedTests << "/" << ranTests << " tests)\n";
        for (const auto &kv : failedByVariant) {
            const std::string &desc = kv.first;
            const size_t failed = kv.second;
            const size_t ran = ranByVariant[desc];
            const double pct = ran ? (failed * 100.0) / double(ran) : 0.0;

            std::cout << "  Variant \"" << desc << "\": " << failed << "/" << ran << " failed (" << pct
                      << "%)\n";
        }
        return false;
    }

    std::cout << "All tests passed! (" << ranTests << " tests)\n";
    return true;
}

void
TestTracker::AddTime(size_t testIndex, uint64_t ms)
{
    if (testIndex >= m_Tests.size()) {
        std::cerr << "AddTime: test index out of range: " << testIndex << "\n";
        std::terminate();
    }
    m_Tests[testIndex].TestMs = ms;
}

void
TestTracker::MarkSuccess(const HpShark::LaunchParams *launchParams,
                         size_t testIndex,
                         const std::string &description)
{
    if (testIndex >= m_Tests.size()) {
        std::cerr << "MarkSuccess: test index out of range: " << testIndex << "\n";
        std::terminate();
    }

    auto &t = m_Tests[testIndex];
    t.RanAnyVariant = true;

    auto &vr = t.Variants[description];
    vr.status = VariantStatus::Passed;
    vr.relativeError.clear();
    vr.acceptableError.clear();

    if (launchParams) {
        t.NumBlocks = launchParams->NumBlocks;
        t.ThreadsPerBlock = launchParams->ThreadsPerBlock;
    }
}

void
TestTracker::MarkFailed(const HpShark::LaunchParams *launchParams,
                        size_t testIndex,
                        const std::string &description,
                        const std::string &relativeError,
                        const std::string &acceptableError)
{
    if (testIndex >= m_Tests.size()) {
        std::cerr << "MarkFailed: test index out of range: " << testIndex << "\n";
        std::terminate();
    }

    auto &t = m_Tests[testIndex];
    t.RanAnyVariant = true;

    auto &vr = t.Variants[description];
    vr.status = VariantStatus::Failed;
    vr.relativeError = relativeError;
    vr.acceptableError = acceptableError;

    if (launchParams) {
        t.NumBlocks = launchParams->NumBlocks;
        t.ThreadsPerBlock = launchParams->ThreadsPerBlock;
    }
}
