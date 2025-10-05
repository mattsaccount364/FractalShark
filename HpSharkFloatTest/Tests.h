#pragma once

enum class Operator { Add, MultiplyFFT, MultiplyNTT, ReferenceOrbit };

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
    } else {
        return "Unknown";
    }
}

template <class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(int testNum, const char *num1, const char *num2);

template <class SharkFloatParams, Operator sharkOperator> bool TestAllBinaryOp(int testBase);

template <Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase, int numIters, int internalTestLoopCount);

template <Operator sharkOperator>
bool TestFullReferencePerf(int testBase, int internalTestLoopCount);
