#pragma once

#include "HpSharkTestConfig.h"

#include <cstddef>
#include <cstdint>

class TestTracker;

enum class Operator { ReferenceOrbit2 };

template <Operator sharkOperator>
inline constexpr bool IsReferenceOrbitOperator = sharkOperator == Operator::ReferenceOrbit2;

template <Operator sharkOperator>
const char *
OperatorToString()
{
    if constexpr (sharkOperator == Operator::ReferenceOrbit2) {
        return "Operator::ReferenceOrbit2";
    } else {
        return "Unknown";
    }
}

template <class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(int testNum, const char *num1, const char *num2);

template <class SharkFloatParams, Operator sharkOperator> bool TestAllBinaryOp(int testBase);

template <Operator sharkOperator>
bool TestBinaryOperatorPerf(const HpShark::LaunchParams &launchParams,
                            int testBase,
                            int numIters,
                            int internalTestLoopCount,
                            BasicCorrectnessMode mode);

struct FullReferencePerfLimbSelection {
    uint32_t m_StorageLimbs = 0;
    uint32_t m_EffectiveLimbs = 0;
};

struct FullReferencePerfPrecision {
    uint64_t m_RequiredPrecisionBits = 0;
    uint64_t m_RequestedPrecisionLimbs = 0;
    FullReferencePerfLimbSelection m_ProductionSelection;
    FullReferencePerfLimbSelection m_DefaultSelection;
    bool m_DefaultIsBenchmarkPreset = false;
};

FullReferencePerfPrecision GetFullReferencePerfPrecision(Operator referenceOperator, size_t view);
uint32_t GetFullReferencePerfEffectiveLimbs(Operator referenceOperator,
                                            uint64_t requestedLimbs,
                                            uint32_t storageLimbs);
uint32_t GetMinimumFullReferencePerfEffectiveLimbs(Operator referenceOperator, uint32_t storageLimbs);
bool IsValidFullReferencePerfLimbSelection(Operator referenceOperator,
                                           const FullReferencePerfLimbSelection &selection);

// Generic "run view perf test" driver. Views 5/30/32 provide exact input and expected-period
// overrides; all views derive their precision and supported storage size from the shared preset.
template <Operator sharkOperator>
bool TestFullReferencePerfView(TestTracker &Tests,
                               int numBlocks,
                               int numThreads,
                               int testBase,
                               int numIters,
                               int internalTestLoopCount,
                               bool useMT, // no default: MainTestCuda always passes it explicitly
                               size_t view,
                               const FullReferencePerfLimbSelection &limbSelection);

#include "TestNewtonRaphson.h"
