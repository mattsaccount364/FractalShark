#pragma once

enum class Operator {
    Add,
    MultiplyKaratsubaV2,
    MultiplyFFT,
    ReferenceOrbit
};

template<Operator sharkOperator>
const char *OperatorToString() {
    if constexpr (sharkOperator == Operator::Add) {
        return "Operator::Add";
    } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
        return "Operator::MultiplyKaratsubaV2";
    } else if constexpr (sharkOperator == Operator::ReferenceOrbit) {
        return "Operator::ReferenceOrbit";
    } else {
        return "Unknown";
    }
}

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(int testNum, const char *num1, const char *num2);

template<class SharkFloatParams, Operator sharkOperator>
bool TestAllBinaryOp(int testBase);

template<Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase);
