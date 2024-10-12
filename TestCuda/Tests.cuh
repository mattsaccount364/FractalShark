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

template<Operator sharkOperator>
void TestBinOperatorTwoNumbers(int testNum, const char *num1, const char *num2);

template<Operator sharkOperator>
bool TestAllBinaryOp(int testBase);

template<Operator sharkOperator>
bool TestBinaryOperatorPerf(int testBase);
