#pragma once
bool CheckAllTestsPassed();

template<class SharkFloatParams>
void ComputeAddGpuTestLoop(void *kernelArgs[]);

template<class SharkFloatParams>
void ComputeAddGpu(void *kernelArgs[]);