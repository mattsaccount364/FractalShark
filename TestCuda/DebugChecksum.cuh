#pragma once

#include "DebugStateRaw.h"

#ifdef __CUDACC__

#include <cstdint>
#include <cooperative_groups.h>

template <class SharkFloatParams>
struct DebugState {

    // CRC64 polynomial (ISO 3309 standard)
    static constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;
    static constexpr uint32_t CRC32_POLY = 0xEDB88320UL;

    __device__ void Reset(
        bool record,
        cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        const uint32_t *arrayToChecksum,
        size_t arraySize,
        DebugStatePurpose purpose,
        int callIndex
        );

    __device__ void Reset(
        bool record,
        cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        const uint64_t *arrayToChecksum,
        size_t arraySize,
        DebugStatePurpose purpose,
        int callIndex
    );

    __device__ void Erase(
        bool record,
        cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        DebugStatePurpose purpose,
        int callIndex
    );

    static __device__ uint64_t ComputeCRC64(
        const uint32_t *data,
        size_t size,
        uint64_t initialCrc);

    static __device__ uint64_t ComputeCRC64(
        const uint64_t *data,
        size_t size,
        uint64_t initialCrc);

    DebugStateRaw Data;
};


// Function to compute CRC64 for a single data chunk
template <class SharkFloatParams>
__device__ uint64_t DebugState<SharkFloatParams>::ComputeCRC64(
    const uint32_t *data,
    size_t size,
    uint64_t initialCrc) {

    uint64_t crc = initialCrc;
    for(size_t i = 0; i < size; i++) {
        auto word = data[i];
        crc ^= word;
        for (int bit = 0; bit < 64; ++bit) {
            if (crc & 1ULL) {
                crc = (crc >> 1) ^ CRC64_POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

// Function to compute CRC64 for a single data chunk
template <class SharkFloatParams>
__device__ uint64_t DebugState<SharkFloatParams>::ComputeCRC64(
    const uint64_t *data,
    size_t size,
    uint64_t initialCrc) {

    uint64_t crc = initialCrc;
    for (size_t i = 0; i < size; i++) {
        auto word = data[i];
        crc ^= word;
        for (int bit = 0; bit < 64; ++bit) {
            if (crc & 1ULL) {
                crc = (crc >> 1) ^ CRC64_POLY;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

template <class SharkFloatParams>
__device__ void DebugState<SharkFloatParams>::Reset(
    bool record,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const uint32_t *arrayToChecksum,
    size_t arraySize,
    DebugStatePurpose purpose,
    int callIndex)
{
    if constexpr (DebugChecksums) {
        if (record) {
            // Initialize the checksum to zero
            Data.Checksum = 0;
            Data.Block = block.group_index().x;
            Data.Thread = block.thread_index().x;
            Data.ArraySize = arraySize;

            // Set the purpose of the checksum
            Data.ChecksumPurpose = purpose;

            // Compute the checksum for the given array
            Data.Checksum = ComputeCRC64(arrayToChecksum, arraySize, 0);

            Data.CallIndex = callIndex;
        }
    }
}

template <class SharkFloatParams>
__device__ void DebugState<SharkFloatParams>::Reset(
    bool record,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const uint64_t *arrayToChecksum,
    size_t arraySize,
    DebugStatePurpose purpose,
    int callIndex)
{
    if constexpr (DebugChecksums) {
        if (record) {
            // Initialize the checksum to zero
            Data.Checksum = 0;
            Data.Block = block.group_index().x;
            Data.Thread = block.thread_index().x;
            Data.ArraySize = arraySize;

            // Set the purpose of the checksum
            Data.ChecksumPurpose = purpose;

            // Compute the checksum for the given array
            Data.Checksum = ComputeCRC64(arrayToChecksum, arraySize, 0);

            Data.CallIndex = callIndex;
        }
    }
}

template <class SharkFloatParams>
__device__ void DebugState<SharkFloatParams>::Erase(
    bool record,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    DebugStatePurpose purpose,
    int callIndex)
{
    if constexpr (DebugChecksums) {
        if (record) {
            // Initialize the checksum to zero
            Data.Checksum = 0;
            Data.Block = 0;
            Data.Thread = 0;
            Data.ArraySize = 0;
            Data.ChecksumPurpose = DebugStatePurpose::Invalid;
            Data.CallIndex = 0;
        }
    }
}

#endif // __CUDACC__