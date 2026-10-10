#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <memory>

namespace Png::Detail {

// Shared launch policy and lifetime helpers for the PNG stages, not a public CUDA framework.
inline constexpr unsigned WarpSize = 32;
inline constexpr unsigned FullWarpMask = 0xffffffffu;
// Launch policy, independent of the wire format: four warp jobs per block, wider
// blocks for per-byte work, and smaller blocks for compaction and chunk headers.
inline constexpr unsigned WarpsPerBlock = 4;
inline constexpr unsigned LinearThreadsPerBlock = 256;
inline constexpr unsigned CopyThreadsPerBlock = 128;
class PendingStreamWork {
public:
    explicit PendingStreamWork(cudaStream_t stream) : m_Stream{stream} {}
    ~PendingStreamWork()
    {
        if (!m_Complete) {
            // Preserve the original error while fencing any earlier successful launches.
            // Input and workspace may be released as soon as the public call returns.
            cudaStreamSynchronize(m_Stream);
        }
    }
    PendingStreamWork(const PendingStreamWork &) = delete;
    PendingStreamWork &operator=(const PendingStreamWork &) = delete;
    void
    Complete()
    {
        m_Complete = true;
    }

private:
    cudaStream_t m_Stream;
    bool m_Complete{};
};

struct CudaAllocationDeleter {
    void
    operator()(void *allocation) const
    {
        cudaFree(allocation);
    }
};

// Grow-only allocation: a failed reserve retains the old buffer, and smaller images
// reuse its capacity. Callers fence the stream before storage can be released.
class DeviceBuffer {
public:
    ~DeviceBuffer() = default;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    cudaError_t
    Reserve(size_t bytes)
    {
        if (bytes <= m_Capacity) {
            return cudaSuccess;
        }
        void *allocation = nullptr;
        const auto error = cudaMalloc(&allocation, bytes);
        if (error != cudaSuccess) {
            return error;
        }
        Reset();
        m_Data.reset(allocation);
        m_Capacity = bytes;
        return cudaSuccess;
    }

    void
    Reset()
    {
        m_Data.reset();
        m_Capacity = 0;
    }

    template <class T>
    T *
    Data()
    {
        return static_cast<T *>(m_Data.get());
    }
    size_t
    Capacity() const
    {
        return m_Capacity;
    }

private:
    std::unique_ptr<void, CudaAllocationDeleter> m_Data;
    size_t m_Capacity{};
};

inline bool
MultiplySize(size_t first, size_t second, size_t &result)
{
    if (second != 0 && first > std::numeric_limits<size_t>::max() / second) {
        return false;
    }
    result = first * second;
    return true;
}

inline bool
AddSize(size_t first, size_t second, size_t &result)
{
    if (first > std::numeric_limits<size_t>::max() - second) {
        return false;
    }
    result = first + second;
    return true;
}

inline bool
GridFits(size_t items, size_t threads)
{
    return items != 0 && (items - 1) / threads < static_cast<size_t>(std::numeric_limits<int>::max());
}

inline unsigned
GridSize(size_t items, unsigned threads)
{
    return static_cast<unsigned>((items - 1) / threads + 1);
}

__device__ inline uint64_t
WarpSum(uint64_t value)
{
    // All lanes must participate; warp kernels only return on a whole-row/region condition.
    // Only lane zero receives the complete sum.
    for (unsigned offset = WarpSize / 2; offset != 0; offset /= 2) {
        value += __shfl_down_sync(FullWarpMask, value, offset);
    }
    return value;
}

__device__ inline void
WriteBigEndian(unsigned char *output, uint32_t value)
{
    output[0] = static_cast<unsigned char>(value >> 24);
    output[1] = static_cast<unsigned char>(value >> 16);
    output[2] = static_cast<unsigned char>(value >> 8);
    output[3] = static_cast<unsigned char>(value);
}

} // namespace Png::Detail
