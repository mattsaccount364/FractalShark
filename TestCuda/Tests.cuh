enum class Operator {
    Add,
    Multiply
};

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
