#pragma once

#include "HpSharkTestConfig.h"

class TestTracker;

enum class Operator {
    Add,
    MultiplyNTT,
    ReferenceOrbit,
    ReferenceOrbit2,
};

template <Operator sharkOperator>
inline constexpr bool IsReferenceOrbitOperator =
    sharkOperator == Operator::ReferenceOrbit || sharkOperator == Operator::ReferenceOrbit2;

template <Operator sharkOperator>
const char *
OperatorToString()
{
    if constexpr (sharkOperator == Operator::Add) {
        return "Operator::Add";
    } else if constexpr (sharkOperator == Operator::MultiplyNTT) {
        return "Operator::MultiplyNTT";
    } else if constexpr (sharkOperator == Operator::ReferenceOrbit) {
        return "Operator::ReferenceOrbit";
    } else if constexpr (sharkOperator == Operator::ReferenceOrbit2) {
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

template <Operator sharkOperator>
bool TestFullReferencePerfView5(TestTracker &Tests,
                                int numBlocks,
                                int numThreads,
                                int testBase,
                                int numIters,
                                int internalTestLoopCount,
                                bool useMT = true);

template <Operator sharkOperator>
bool TestFullReferencePerfView30(TestTracker &Tests,
                                 int numBlocks,
                                 int numThreads,
                                 int testBase,
                                 int numIters,
                                 int internalTestLoopCount,
                                 bool useMT = true);

template <Operator sharkOperator>
bool TestFullReferencePerfView32(TestTracker &Tests,
                                 int numBlocks,
                                 int numThreads,
                                 int testBase,
                                 int numIters,
                                 int internalTestLoopCount,
                                 bool useMT = true);

#include "TestNewtonRaphson.h"
