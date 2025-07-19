#include "Multiply.cu"
#include "Add.cu"

template<class SharkFloatParams>
__device__ void ReferenceHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    HpSharkReferenceResults<SharkFloatParams> *SharkRestrict reference,
    uint64_t *tempData) {
    
    MultiplyHelperKaratsubaV2Separates<SharkFloatParams>(
        &reference->Add.C_A,
        &reference->Add.E_B,
        &reference->Multiply.ResultX2,
        &reference->Multiply.ResultXY,
        &reference->Multiply.ResultY2,
        grid,
        block,
        tempData);

    //reference->Add.A_X2.DeepCopyGPU(grid, block, reference->Multiply.ResultX2);
    //reference->Add.B_Y2.DeepCopyGPU(grid, block, reference->Multiply.ResultY2);
    ////reference->Add.C_A.DeepCopyGPU(grid, block, reference->Multiply.A);
    //reference->Add.D_2X.DeepCopyGPU(grid, block, reference->Multiply.ResultXY);
    ////reference->Add.E_B.DeepCopyGPU(grid, block, reference->Multiply.B);

    AddHelperSeparates<SharkFloatParams>(
        grid,
        block,
        &reference->Multiply.ResultX2,
        &reference->Multiply.ResultY2,
        &reference->Add.C_A,
        &reference->Multiply.ResultXY,
        &reference->Add.E_B,
        &reference->Multiply.A,
        &reference->Multiply.B,
        tempData);
}

template<class SharkFloatParams>
__global__ void HpSharkReferenceGpuKernel(
    HpSharkReferenceResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    ReferenceHelper<SharkFloatParams>(grid, block, combo, tempData);
}

template<class SharkFloatParams>
__global__ void HpSharkReferenceGpuLoop(
    HpSharkReferenceResults<SharkFloatParams> *SharkRestrict combo,
    uint64_t numIters,
    uint64_t *tempData) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int32_t i = 0; i < numIters; ++i) {
        ReferenceHelper(grid, block, combo, tempData);
    }
}

template<class SharkFloatParams>
void ComputeHpSharkReferenceGpu(void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits; // Adjust as necessary
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)HpSharkReferenceGpuKernel<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpu: " << cudaGetErrorString(err) << std::endl;
        DebugBreak();
    }
}

template<class SharkFloatParams>
void ComputeHpSharkReferenceGpuLoop(cudaStream_t & /*stream*/, void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits; // Adjust as necessary

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)HpSharkReferenceGpuLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpuTestLoop: " << cudaGetErrorString(err) << std::endl;
        DebugBreak();
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeHpSharkReferenceGpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(cudaStream_t &stream, void *kernelArgs[]);

#ifdef SHARK_INCLUDE_KERNELS
ExplicitInstantiateAll();
#endif
