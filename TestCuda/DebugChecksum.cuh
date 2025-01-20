#pragma once

#include <cstdint>
#include <cooperative_groups.h>

template <class SharkFloatParams, typename ChecksumT>
struct DebugState {

    // CRC64 polynomial (ISO 3309 standard)
    static constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;
    static constexpr uint32_t CRC32_POLY = 0xEDB88320UL;

    enum class Purpose {
        ADigits,
        BDigits,
        XDiff,
        YDiff,
        Z0,
        Z1,
        Z2,
        Z1_temp_offset,
        Z1_offset,
        Final128,
        Convolution_offset,
        Result_offset,
        XDiff_offset,
        YDiff_offset,
        GlobalCarryOffset,
        SubtractionOffset1,
        SubractionOffset2,
        SubtractionOffset3,
        SubtractionOffset4,
        BorrowAnyOffset
    };

    __device__ void Reset(
        bool record,
        cooperative_groups::grid_group &grid,
        cooperative_groups::thread_block &block,
        const ChecksumT *arrayToChecksum,
        size_t arraySize,
        Purpose purpose
        );

    static __device__ ChecksumT ComputeCRC3264(
        const ChecksumT *data,
        size_t size,
        ChecksumT initialCrc);

    int Block;
    int Thread;
    uint64_t ArraySize;
    ChecksumT Checksum;
    Purpose ChecksumPurpose;
};


// Function to compute CRC64 for a single data chunk
template <class SharkFloatParams, typename ChecksumT>
__device__ ChecksumT DebugState<SharkFloatParams, ChecksumT>::ComputeCRC3264(
    const ChecksumT *data,
    size_t size,
    ChecksumT initialCrc) {

    if constexpr (std::is_same_v< ChecksumT, uint64_t>) {
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
    } else if constexpr (std::is_same_v< ChecksumT, uint32_t>) {
        uint32_t crc = initialCrc;
        for (size_t i = 0; i < size; i++) {
            auto word = data[i];
            crc ^= word;
            for (int bit = 0; bit < 32; ++bit) {
                if (crc & 1U) {
                    crc = (crc >> 1) ^ CRC32_POLY;
                } else {
                    crc >>= 1;
                }
            }
        }
        return static_cast<ChecksumT>(crc);
    } else {
        return 0;
    }
}

template <class SharkFloatParams, typename ChecksumT>
__device__ void DebugState<SharkFloatParams, ChecksumT>::Reset(
    bool record,
    cooperative_groups::grid_group &grid,
    cooperative_groups::thread_block &block,
    const ChecksumT *arrayToChecksum,
    size_t arraySize,
    Purpose purpose)
{
    if constexpr (SharkFloatParams::DebugChecksums) {
        if (record) {
            // Initialize the checksum to zero
            Checksum = 0;
            Block = block.group_index().x;
            Thread = block.thread_index().x;
            ArraySize = arraySize;

            // Set the purpose of the checksum
            ChecksumPurpose = purpose;

            // Compute the checksum for the given array
            Checksum = ComputeCRC3264(arrayToChecksum, arraySize, 0);
        }
    }
}
