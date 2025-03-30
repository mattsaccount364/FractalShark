#include "HpSharkFloat.cuh"
#include "BenchmarkTimer.h"
#include "TestTracker.h"
#include "KernelInvoke.cuh"
#include "Tests.h"
#include "DebugChecksum.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cassert>
#include <cstring>

#include <iostream>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose>
__device__ SharkForceInlineReleaseOnly void
EraseCurrentDebugState(
    RecordIt record,
    DebugState<SharkFloatParams> *debugChecksumArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block) {

    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto maxPurposes = static_cast<int>(DebugStatePurpose::NumPurposes);
    constexpr auto curPurpose = static_cast<int>(Purpose);
    debugChecksumArray[CallIndex * maxPurposes + curPurpose].Erase(
        record, grid, block, Purpose, RecursionDepth, CallIndex);
}

template<
    class SharkFloatParams,
    DebugStatePurpose Purpose,
    typename ArrayType>
__device__ SharkForceInlineReleaseOnly void
StoreCurrentDebugState(
    RecordIt record,
    DebugState<SharkFloatParams> *debugChecksumArray,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ArrayType *arrayToChecksum,
    size_t arraySize) {
    
    constexpr auto CurPurpose = static_cast<int>(Purpose);
    constexpr auto RecursionDepth = 0;
    constexpr auto CallIndex = 0;
    constexpr auto UseConvolutionHere = UseConvolution::No; 

    debugChecksumArray[CurPurpose].Reset(
        record, UseConvolutionHere, grid, block, arrayToChecksum, arraySize, Purpose, RecursionDepth, CallIndex);
}

template <class SharkFloatParams>
__device__ void AddHelper (
    cg::grid_group &grid,
    cg::thread_block &block,
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *OutXY,
    uint32_t *g_extResult) {

    // --- Constants and Parameters ---
    constexpr int32_t guard = 2;
    constexpr int32_t actualDigits = SharkFloatParams::GlobalNumUint32;
    constexpr int32_t extDigits = SharkFloatParams::GlobalNumUint32 + guard;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NewN = SharkFloatParams::GlobalNumUint32;

    constexpr auto Checksum_offset = AdditionalGlobalSyncSpace;
    auto *SharkRestrict debugChecksumArray =
        reinterpret_cast<DebugState<SharkFloatParams>*>(&g_extResult[Checksum_offset]);

    const RecordIt record =
        (block.thread_index().x == 0 && block.group_index().x == 0) ?
        RecordIt::Yes :
        RecordIt::No;

    if constexpr (SharkDebugChecksums) {
        grid.sync();

        EraseCurrentDebugState<SharkFloatParams, DebugStatePurpose::Invalid>(
            record, debugChecksumArray, grid, block);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::ADigits, uint32_t>(
            record, debugChecksumArray, grid, block, A->Digits, NewN);
        StoreCurrentDebugState<SharkFloatParams, DebugStatePurpose::BDigits, uint32_t>(
            record, debugChecksumArray, grid, block, B->Digits, NewN);

        grid.sync();
    }

    // We assume that the kernel is launched with at least extDigits threads.
    if (idx >= extDigits)
        return;

    // Shared variables used for reductions and final decisions.
    __shared__ int32_t shiftA, shiftB;
    __shared__ int32_t newAExponent, newBExponent;
    __shared__ bool normA_isZero, normB_isZero;
    __shared__ bool sameSign;
    __shared__ bool AIsBiggerMagnitude;
    __shared__ int32_t diff;
    __shared__ int32_t outExponent;

    // Thread 0 does the normalization reductions for both inputs.
    if (idx == 0) {
        const uint32_t *extA = A->Digits;
        const uint32_t *extB = B->Digits;
        newAExponent = A->Exponent;
        newBExponent = B->Exponent;
        sameSign = (A->IsNegative == B->IsNegative);

        // ----- Extended Normalization for A -----
        int32_t msdA = extDigits - 1;
        while (msdA >= 0 && extA[msdA] == 0) {
            msdA--;
        }
        if (msdA < 0) {
            normA_isZero = true;
            shiftA = 0;
        } else {
            normA_isZero = false;
            // Use CUDA intrinsic for count-leading-zeros
            int32_t clzA = __clz(extA[msdA]);
            int32_t current_msbA = msdA * 32 + (31 - clzA);
            int32_t totalExtBits = extDigits * 32;
            shiftA = (totalExtBits - 1) - current_msbA;
            newAExponent -= shiftA;
        }
        // ----- Extended Normalization for B -----
        int32_t msdB = extDigits - 1;
        while (msdB >= 0 && extB[msdB] == 0) {
            msdB--;
        }
        if (msdB < 0) {
            normB_isZero = true;
            shiftB = 0;
        } else {
            normB_isZero = false;
            int32_t clzB = __clz(extB[msdB]);
            int32_t current_msbB = msdB * 32 + (31 - clzB);
            int32_t totalExtBits = extDigits * 32;
            shiftB = (totalExtBits - 1) - current_msbB;
            newBExponent -= shiftB;
        }

        // ----- Compute Effective Exponents -----
        int32_t effExpA = normA_isZero ? -100000000 : newAExponent + (actualDigits * 32 - 32);
        int32_t effExpB = normB_isZero ? -100000000 : newBExponent + (actualDigits * 32 - 32);

        // ----- Compare Magnitudes -----
        // (For simplicity, perform a serial comparison over extDigits.)
        AIsBiggerMagnitude = false;
        if (effExpA > effExpB) {
            AIsBiggerMagnitude = true;
        } else if (effExpA < effExpB) {
            AIsBiggerMagnitude = false;
        } else {
            for (int i = extDigits - 1; i >= 0; i--) {
                // A simplified “normalized digit” computation: note that here we approximate
                // normalization by a simple left shift (this inline version assumes no word–boundary complication).
                uint32_t digitA = (i < extDigits ? extA[i] << shiftA : 0);
                uint32_t digitB = (i < extDigits ? extB[i] << shiftB : 0);
                if (digitA > digitB) { AIsBiggerMagnitude = true; break; } else if (digitA < digitB) { AIsBiggerMagnitude = false; break; }
            }
        }
        diff = AIsBiggerMagnitude ? (effExpA - effExpB) : (effExpB - effExpA);
        outExponent = AIsBiggerMagnitude ? newAExponent : newBExponent;
    }
    grid.sync();

    // --- Helper lambdas for on–the–fly normalization ---
    auto GetNormalizedDigit = [=] (const uint32_t * ext, int shift, int i) -> uint32_t {
        // Compute a left shift by 'shift' bits; here we use a simple implementation.
        int wordShift = shift / 32;
        int bitShift = shift % 32;
        int srcIdx = i - wordShift;
        uint32_t lower = (srcIdx >= 0 && srcIdx < extDigits) ? ext[srcIdx] : 0;
        uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < extDigits) ? ext[srcIdx - 1] : 0;
        return (bitShift == 0) ? lower : (lower << bitShift) | (upper >> (32 - bitShift));
    };

    auto GetShiftedNormalizedDigit = [=] (const uint32_t * ext, int shift, int rshift, int i) -> uint32_t {
        int wordShift = rshift / 32;
        int bitShift = rshift % 32;
        uint32_t lower = GetNormalizedDigit(ext, shift, i + wordShift);
        uint32_t upper = GetNormalizedDigit(ext, shift, i + wordShift + 1);
        return (bitShift == 0) ? lower : (lower >> bitShift) | (upper << (32 - bitShift));
    };

    // --- Each thread computes its aligned limb.
    uint64_t alignedA = 0, alignedB = 0;
    const uint32_t *extA = A->Digits;
    const uint32_t *extB = B->Digits;
    if (AIsBiggerMagnitude) {
        alignedA = GetNormalizedDigit(extA, shiftA, idx);
        alignedB = GetShiftedNormalizedDigit(extB, shiftB, diff, idx);
    } else {
        alignedB = GetNormalizedDigit(extB, shiftB, idx);
        alignedA = GetShiftedNormalizedDigit(extA, shiftA, diff, idx);
    }

    // --- Extended Arithmetic (digitwise) ---
    // Each thread computes a preliminary sum (or difference) for its digit.
    uint64_t prelim = 0;
    if (sameSign) {
        prelim = alignedA + alignedB;
    } else {
        if (AIsBiggerMagnitude) {
            int64_t diffVal = (int64_t)alignedA - (int64_t)alignedB;
            // We assume that the borrow will be handled in a subsequent pass.
            prelim = (uint64_t)(diffVal < 0 ? diffVal + ((uint64_t)1 << 32) : diffVal);
        } else {
            int64_t diffVal = (int64_t)alignedB - (int64_t)alignedA;
            prelim = (uint64_t)(diffVal < 0 ? diffVal + ((uint64_t)1 << 32) : diffVal);
        }
    }
    // Write preliminary result (without carry/borrow propagation) to global temporary.
    g_extResult[idx] = (uint32_t)(prelim & 0xFFFFFFFF);
    grid.sync();

    // --- Carry/Borrow Propagation ---
    // (For brevity, we perform a sequential prefix scan in thread 0.
    // In a production code you would replace this with a parallel prefix-sum algorithm.)
    __shared__ uint32_t finalCarry;
    if (idx == 0) {
        uint64_t carry = 0;
        for (int i = 0; i < extDigits; i++) {
            uint64_t sum = (uint64_t)g_extResult[i] + carry;
            g_extResult[i] = (uint32_t)(sum & 0xFFFFFFFF);
            carry = sum >> 32;
        }
        finalCarry = (uint32_t)carry;
        if (sameSign && finalCarry > 0) {
            outExponent += 1;
            // Right-shift the extended result by one bit.
            uint32_t nextBit = finalCarry & 1;
            for (int i = extDigits - 1; i >= 0; i--) {
                uint32_t current = g_extResult[i];
                g_extResult[i] = (current >> 1) | (nextBit << 31);
                nextBit = current & 1;
            }
        }
    }
    grid.sync();

    // --- Final Normalization ---
    // Thread 0 finds the most-significant digit (msd) and computes the shift needed.
    __shared__ int msdResult;
    __shared__ int shiftNeeded;
    if (idx == 0) {
        msdResult = 0;
        for (int i = extDigits - 1; i >= 0; i--) {
            if (g_extResult[i] != 0) {
                msdResult = i;
                break;
            }
        }
        int clzResult = __clz(g_extResult[msdResult]);
        int currentOverall = msdResult * 32 + (31 - clzResult);
        int desiredOverall = (actualDigits - 1) * 32 + 31;
        shiftNeeded = currentOverall - desiredOverall;
    }
    grid.sync();

    // Now, thread 0 performs the final shift and writes the fixed–precision result.
    if (idx == 0) {
        if (shiftNeeded > 0) {
            // Right-shift extResult.
            for (int i = 0; i < actualDigits; i++) {
                int wordShift = shiftNeeded / 32;
                int bitShift = shiftNeeded % 32;
                uint32_t lower = (i + wordShift < extDigits) ? g_extResult[i + wordShift] : 0;
                uint32_t upper = (i + wordShift + 1 < extDigits) ? g_extResult[i + wordShift + 1] : 0;
                OutXY->Digits[i] = (bitShift == 0) ? lower : (lower >> bitShift) | (upper << (32 - bitShift));
            }
            outExponent += shiftNeeded;
        } else if (shiftNeeded < 0) {
            int L = -shiftNeeded;
            for (int i = 0; i < actualDigits; i++) {
                int wordShift = L / 32;
                int bitShift = L % 32;
                int srcIdx = i - wordShift;
                uint32_t lower = (srcIdx >= 0 && srcIdx < extDigits) ? g_extResult[srcIdx] : 0;
                uint32_t upper = (srcIdx - 1 >= 0 && srcIdx - 1 < extDigits) ? g_extResult[srcIdx - 1] : 0;
                OutXY->Digits[i] = (bitShift == 0) ? lower : (lower << bitShift) | (upper >> (32 - bitShift));
            }
            outExponent -= L;
        } else {
            // No shifting needed; simply copy.
            memcpy(OutXY->Digits, g_extResult, actualDigits * sizeof(uint32_t));
        }
        OutXY->Exponent = outExponent;
        // Set result sign.
        OutXY->IsNegative = sameSign ? A->IsNegative : (AIsBiggerMagnitude ? A->IsNegative : B->IsNegative);
    }
    grid.sync();
}


template<class SharkFloatParams>
__global__ void AddKernel(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint32_t *g_extResult) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    // Call the AddHelper function
    AddHelper(grid, block, A, B, Out, g_extResult);
}

template<class SharkFloatParams>
__global__ void AddKernelTestLoop(
    const HpSharkFloat<SharkFloatParams> *A,
    const HpSharkFloat<SharkFloatParams> *B,
    HpSharkFloat<SharkFloatParams> *Out,
    uint64_t numIters,
    uint32_t *g_extResult) {

    // Initialize cooperative grid group
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    for (int i = 0; i < numIters; ++i) {
        AddHelper(grid, block, A, B, Out, g_extResult);
    }
}

template<class SharkFloatParams>
void ComputeAddGpu(void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32 * 2;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits * 6; // Adjust as necessary
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernel<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpu: " << cudaGetErrorString(err) << std::endl;
    }
}

template<class SharkFloatParams>
void ComputeAddGpuTestLoop(void *kernelArgs[]) {

    constexpr auto ExpandedNumDigits = SharkFloatParams::GlobalNumUint32 * 2;
    constexpr size_t SharedMemSize = sizeof(uint32_t) * ExpandedNumDigits * 6; // Adjust as necessary

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void *)AddKernelTestLoop<SharkFloatParams>,
        dim3(SharkFloatParams::GlobalNumBlocks),
        dim3(SharkFloatParams::GlobalThreadsPerBlock),
        kernelArgs,
        SharedMemSize, // Shared memory size
        0 // Stream
    );

    cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::cerr << "CUDA error in ComputeAddGpuTestLoop: " << cudaGetErrorString(err) << std::endl;
    }
}

#define ExplicitlyInstantiate(SharkFloatParams) \
    template void ComputeAddGpu<SharkFloatParams>(void *kernelArgs[]); \
    template void ComputeAddGpuTestLoop<SharkFloatParams>(void *kernelArgs[]);

#ifdef SHARK_INCLUDE_KERNELS
ExplicitInstantiateAll();
#endif