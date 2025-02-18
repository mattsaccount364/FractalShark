#pragma once

#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include "DebugChecksum.cuh"

template <class SharkFloatParams>
struct DebugStateHost {

    // CRC64 polynomial (ISO 3309 standard)
    static constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;
    static constexpr uint32_t CRC32_POLY = 0xEDB88320UL;

    DebugStateHost();

    /**
     * @brief Constructs a DebugStateHost by copying the data from
     *        a std::vector. Internally calls the pointer+length constructor.
     *
     * @param arrayToChecksum The source data as a std::vector<uint64_t>.
     * @param purpose         The intended purpose (for debugging context).
     */
    DebugStateHost(
        const std::vector<uint32_t> &arrayToChecksum,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex
    );

    DebugStateHost(
        const std::vector<uint64_t> &arrayToChecksum,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex
    );

    /**
     * @brief Constructs a DebugStateHost by copying the data from
     *        a raw pointer and length. If SharkDebugChecksums is enabled
     *        in the template param, it also computes the CRC64.
     *
     * @param data     Pointer to the source data.
     * @param size     Number of 64-bit elements.
     * @param purpose  The intended purpose (for debugging context).
     */
    DebugStateHost(
        const uint32_t *data,
        size_t size,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex,
        UseConvolution useConvolution
    );

    DebugStateHost(
        const uint64_t *data,
        size_t size,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex,
        UseConvolution useConvolution
    );

    void Reset(
        const uint32_t *arrayToChecksum,
        size_t arraySize,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex,
        UseConvolution useConvolution
    );

    void Reset(
        const uint64_t *arrayToChecksum,
        size_t arraySize,
        DebugStatePurpose purpose,
        int recursionDepth,
        int callIndex,
        UseConvolution useConvolution
    );

    std::string GetStr() const;

    /**
     * @brief Computes CRC64 (ISO 3309) on the entire vector.
     *
     * @param arrayToChecksum Input data as a vector of 64-bit words.
     * @param initialCrc      Starting CRC value (commonly 0).
     * @return                64-bit CRC of the vector content.
     */
    static uint64_t ComputeCRC64(
        const std::vector<uint64_t> &arrayToChecksum,
        uint64_t initialCrc
    );

    static uint64_t ComputeCRC64(
        const std::vector<uint32_t> &arrayToChecksum,
        uint64_t initialCrc
    );

    std::vector<uint32_t> ArrayToChecksum32;  ///< Stores a copy of the data
    std::vector<uint64_t> ArrayToChecksum64;  ///< Stores a copy of the data
    uint64_t Checksum;         ///< Stores the computed CRC64 (if computed)
    DebugStatePurpose               ChecksumPurpose;          ///< Enum describing the purpose of this debug state
    int                   RecursionDepth;          ///< Recursion depth of the call that generated this debug state
    int                   CallIndex;                ///< Index of the call that generated this debug state
    UseConvolution  Convolution;
};

//
// Implementation
//

template <class SharkFloatParams>
DebugStateHost<SharkFloatParams>::DebugStateHost()
    : ArrayToChecksum32{}
    , ArrayToChecksum64{}
    , Checksum{}
    , ChecksumPurpose{}
    , RecursionDepth{}
    , CallIndex{}
    , Convolution{} {
    // No additional logic needed here since we rely on the other constructor
}

template <class SharkFloatParams>
uint64_t DebugStateHost<SharkFloatParams>::ComputeCRC64(
    const std::vector<uint32_t> &arrayToChecksum,
    uint64_t initialCrc
) {
    uint64_t crc = initialCrc;
    for (uint64_t word : arrayToChecksum) {
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
uint64_t DebugStateHost<SharkFloatParams>::ComputeCRC64(
    const std::vector<uint64_t> &arrayToChecksum,
    uint64_t initialCrc
)
{
    uint64_t crc = initialCrc;
    for (uint64_t word : arrayToChecksum) {
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
DebugStateHost<SharkFloatParams>::DebugStateHost(
    const std::vector<uint32_t> &arrayToChecksum,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex
)
// Forward to pointer+length constructor:
    : DebugStateHost(arrayToChecksum.data(), arrayToChecksum.size(), purpose, recursionDepth, callIndex)
{
    // No additional logic needed here since we rely on the other constructor
}

template <class SharkFloatParams>
DebugStateHost<SharkFloatParams>::DebugStateHost(
    const std::vector<uint64_t> &arrayToChecksum,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex
)
// Forward to pointer+length constructor:
    : DebugStateHost(arrayToChecksum.data(), arrayToChecksum.size(), purpose, recursionDepth, callIndex)
{
    // No additional logic needed here since we rely on the other constructor
}

template <class SharkFloatParams>
DebugStateHost<SharkFloatParams>::DebugStateHost(
    const uint32_t *data,
    size_t size,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex,
    UseConvolution useConvolution
)
    : ArrayToChecksum32{}
    , ArrayToChecksum64{}
    , Checksum(0)
    , ChecksumPurpose(purpose)
    , RecursionDepth(recursionDepth)
    , CallIndex(callIndex)
    , Convolution(useConvolution)
{
    if constexpr (SharkDebugChecksums) {
        // Copy data if valid
        if (data != nullptr && size > 0) {
            ArrayToChecksum32.assign(data, data + size);
            Checksum = ComputeCRC64(ArrayToChecksum32, 0);
        }
    }
}

template <class SharkFloatParams>
DebugStateHost<SharkFloatParams>::DebugStateHost(
    const uint64_t *data,
    size_t size,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex,
    UseConvolution useConvolution
)
    : ArrayToChecksum32{}
    , ArrayToChecksum64{}
    , Checksum(0)
    , ChecksumPurpose(purpose)
    , RecursionDepth(recursionDepth)
    , CallIndex(callIndex)
    , Convolution(useConvolution)
{
    if constexpr (SharkDebugChecksums) {
        // Copy data if valid
        if (data != nullptr && size > 0) {
            ArrayToChecksum64.assign(data, data + size);
            Checksum = ComputeCRC64(ArrayToChecksum64, 0);
        }
    }
}

template <class SharkFloatParams>
void DebugStateHost<SharkFloatParams>::Reset(
    const uint32_t *arrayToChecksum,
    size_t arraySize,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex,
    UseConvolution useConvolution
) {
    if constexpr (SharkDebugChecksums) {
        ArrayToChecksum32.assign(arrayToChecksum, arrayToChecksum + arraySize);
        Checksum = ComputeCRC64(ArrayToChecksum32, 0);

        ChecksumPurpose = purpose;
        RecursionDepth = recursionDepth;
        CallIndex = callIndex;

        Convolution = useConvolution;
    }
}

template <class SharkFloatParams>
void DebugStateHost<SharkFloatParams>::Reset(
    const uint64_t *arrayToChecksum,
    size_t arraySize,
    DebugStatePurpose purpose,
    int recursionDepth,
    int callIndex,
    UseConvolution useConvolution
) {
    if constexpr (SharkDebugChecksums) {
        ArrayToChecksum64.assign(arrayToChecksum, arrayToChecksum + arraySize);
        Checksum = ComputeCRC64(ArrayToChecksum64, 0);

        ChecksumPurpose = purpose;
        RecursionDepth = recursionDepth;
        CallIndex = callIndex;

        Convolution = useConvolution;
    }
}

template <class SharkFloatParams>
std::string DebugStateHost<SharkFloatParams>::GetStr() const
{
    std::stringstream ss;
    
    ss << "Checksum: 0x" << std::hex << Checksum;
    ss << ", DebugStatePurpose: " << std::dec << static_cast<int>(ChecksumPurpose);

    if (ArrayToChecksum32.size() > 0) {
        ss << ", Size: " << ArrayToChecksum32.size();
    } else {
        ss << ", Size: " << ArrayToChecksum64.size();
    }

    ss << ", RecursionDepth: " << RecursionDepth;
    ss << ", CallIndex: " << CallIndex;
    ss << ", Convolution: " << static_cast<int>(Convolution);

    return ss.str();
}