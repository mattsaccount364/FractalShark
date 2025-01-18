#pragma once

#include <vector>
#include <cstdint>

template <class SharkFloatParams, typename ChecksumT>
struct DebugStateHost {

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

    /**
     * @brief Constructs a DebugStateHost by copying the data from
     *        a std::vector. Internally calls the pointer+length constructor.
     *
     * @param arrayToChecksum The source data as a std::vector<uint64_t>.
     * @param purpose         The intended purpose (for debugging context).
     */
    DebugStateHost(
        const std::vector<ChecksumT> &arrayToChecksum,
        Purpose purpose
    );

    /**
     * @brief Constructs a DebugStateHost by copying the data from
     *        a raw pointer and length. If DebugChecksums is enabled
     *        in the template param, it also computes the CRC64.
     *
     * @param data     Pointer to the source data.
     * @param size     Number of 64-bit elements.
     * @param purpose  The intended purpose (for debugging context).
     */
    DebugStateHost(
        const ChecksumT *data,
        size_t size,
        Purpose purpose
    );

    std::string GetStr() const;

    /**
     * @brief Computes CRC64 (ISO 3309) on the entire vector.
     *
     * @param arrayToChecksum Input data as a vector of 64-bit words.
     * @param initialCrc      Starting CRC value (commonly 0).
     * @return                64-bit CRC of the vector content.
     */
    static ChecksumT ComputeCRC3264(
        const std::vector<ChecksumT> &arrayToChecksum,
        ChecksumT initialCrc
    );

    std::vector<ChecksumT> ArrayToChecksum;  ///< Stores a copy of the data
    ChecksumT              Checksum;         ///< Stores the computed CRC64 (if computed)
    Purpose               ChecksumPurpose;          ///< Enum describing the purpose of this debug state
};

//
// Implementation
//

template <class SharkFloatParams, typename ChecksumT>
ChecksumT DebugStateHost<SharkFloatParams, ChecksumT>::ComputeCRC3264(
    const std::vector<ChecksumT> &arrayToChecksum,
    ChecksumT initialCrc
) {
    if constexpr (std::is_same_v< ChecksumT, uint64_t>) {
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
    } else if constexpr (std::is_same_v< ChecksumT, uint32_t>) {
        uint32_t crc = initialCrc;
        for (uint32_t word : arrayToChecksum) {
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
DebugStateHost<SharkFloatParams, ChecksumT>::DebugStateHost(
    const std::vector<ChecksumT> &arrayToChecksum,
    Purpose purpose
)
    // Forward to pointer+length constructor:
    : DebugStateHost(arrayToChecksum.data(), arrayToChecksum.size(), purpose) {
    // No additional logic needed here since we rely on the other constructor
}

template <class SharkFloatParams, typename ChecksumT>
DebugStateHost<SharkFloatParams, ChecksumT>::DebugStateHost(
    const ChecksumT *data,
    size_t size,
    Purpose purpose
)
    : ArrayToChecksum{} // empty vector, will fill below if non-null data
    , Checksum(0)
    , ChecksumPurpose(purpose) {
    if constexpr (SharkFloatParams::DebugChecksums) {
        // Copy data if valid
        if (data != nullptr && size > 0) {
            ArrayToChecksum.assign(data, data + size);
            Checksum = ComputeCRC3264(ArrayToChecksum, 0);
        }
    }
}

template <class SharkFloatParams, typename ChecksumT>
std::string DebugStateHost<SharkFloatParams, ChecksumT>::GetStr() const {
    std::string retVal;
    retVal += std::string("DebugStateHost: ") + std::to_string(Checksum);
    retVal += std::string(", Purpose: ") + std::to_string(static_cast<int>(ChecksumPurpose));
    retVal += std::string(", Size: ") + std::to_string(ArrayToChecksum.size());
    return retVal;
}