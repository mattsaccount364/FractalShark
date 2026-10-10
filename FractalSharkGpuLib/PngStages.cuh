#pragma once

#include "PngCudaSupport.cuh"
#include "PngEncoder.cuh"
#include "PngFormat.cuh"
#include <vector>

struct Color16;

namespace Png::Detail {

cudaError_t FilterScanlinesAsync(const unsigned char *deviceRaw,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t channels,
                                 const unsigned char *deviceFilters,
                                 unsigned char *deviceFiltered,
                                 cudaStream_t stream);

// Each stage owns its retained capacity. Calls are ordered on the caller's stream.
class ScanlineStorage {
public:
    cudaError_t Prepare(const Color16 *pixels,
                        uint32_t width,
                        uint32_t height,
                        size_t strideBytes,
                        AlphaMode alphaMode,
                        cudaStream_t stream,
                        unsigned &channels,
                        size_t &filteredBytes);
    unsigned char *
    Data()
    {
        return m_Filtered.Data<unsigned char>();
    }
    size_t
    Bytes() const
    {
        return m_Alpha.Capacity() + m_Raw.Capacity() + m_Filters.Capacity() + m_Filtered.Capacity();
    }
    void
    Reset()
    {
        m_Alpha.Reset();
        m_Raw.Reset();
        m_Filters.Reset();
        m_Filtered.Reset();
    }

private:
    DeviceBuffer m_Alpha;
    DeviceBuffer m_Raw;
    DeviceBuffer m_Filters;
    DeviceBuffer m_Filtered;
};

class CompressionStorage {
public:
    // Shared production compressor. PNG encoding enables dynamic coding; the test-only
    // fixed/stored baseline disables it while exercising the same compression kernels.
    cudaError_t Compress(const unsigned char *input,
                         size_t bytes,
                         cudaStream_t stream,
                         size_t &resultBytes,
                         bool allowDynamic);
    unsigned char *
    Data()
    {
        return m_Zlib.Data<unsigned char>();
    }

private:
    DeviceBuffer m_Huffman;
    DeviceBuffer m_Heads;
    DeviceBuffer m_Slots;
    DeviceBuffer m_Sizes;
    DeviceBuffer m_Offsets;
    DeviceBuffer m_AdlerSums;
    DeviceBuffer m_AdlerWeighted;
    DeviceBuffer m_Scan;
    DeviceBuffer m_Zlib;
    DeviceBuffer m_ResultBytes;

public:
    size_t
    Bytes() const
    {
        return m_Huffman.Capacity() + m_Heads.Capacity() + m_Slots.Capacity() + m_Sizes.Capacity() +
               m_Offsets.Capacity() + m_AdlerSums.Capacity() + m_AdlerWeighted.Capacity() +
               m_Scan.Capacity() + m_Zlib.Capacity() + m_ResultBytes.Capacity();
    }

    void
    Reset()
    {
        m_Huffman.Reset();
        m_Heads.Reset();
        m_Slots.Reset();
        m_Sizes.Reset();
        m_Offsets.Reset();
        m_AdlerSums.Reset();
        m_AdlerWeighted.Reset();
        m_Scan.Reset();
        m_Zlib.Reset();
        m_ResultBytes.Reset();
    }
};

// Returns only after the host PNG bytes are complete; clears them on CUDA failure.
cudaError_t AssemblePng(DeviceBuffer &storage,
                        uint32_t width,
                        uint32_t height,
                        unsigned channels,
                        const unsigned char *zlib,
                        size_t zlibBytes,
                        cudaStream_t stream,
                        std::vector<unsigned char> &pngBytes);

} // namespace Png::Detail
