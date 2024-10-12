#pragma once
void TestMultiplyTwoNumbers(int testNum, const char *num1, const char *num2);
bool CheckAllTestsPassed();

void ComputeMultiplyGpu(void *kernelArgs[]);
void ComputeMultiplyGpuTestLoop(void *kernelArgs[]);