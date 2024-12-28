#pragma once
enum class Operator {
    Add,
    MultiplyN2,
    MultiplyKaratsubaV1,
    MultiplyKaratsubaV2,
};

template<Operator sharkOperator>
const char *OperatorToString() {
    if constexpr (sharkOperator == Operator::Add) {
        return "Operator::Add";
    } else if constexpr (sharkOperator == Operator::MultiplyN2) {
        return "Operator::MultiplyN2";
    } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV1) {
        return "Operator::MultiplyKaratsubaV1";
    } else if constexpr (sharkOperator == Operator::MultiplyKaratsubaV2) {
        return "Operator::MultiplyKaratsubaV2";
    } else {
        return "Unknown";
    }
}

// Structure to hold carry information for each block
struct CarryInfo {
    uint32_t carryOut;    // Carry-out from the block's computation
};

struct GlobalAddBlockData {
    int32_t AIsBiggerMagnitude;
};

template<class SharkFloatParams, Operator sharkOperator>
void TestBinOperatorTwoNumbers(int testNum, const char *num1, const char *num2);

template<class SharkFloatParams, Operator sharkOperator>
bool TestAllBinaryOp(int testBase);

template<class SharkFloatParams, Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase);
