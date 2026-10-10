#include "stdafx.h"
#include "GPU_Types.h"
#include "PngEncoder.cuh"
#include "PngStages.cuh"

namespace Png {

class GpuPngEncoder::Storage {
public:
    int m_Device{-1};
    Detail::ScanlineStorage m_Scanlines;
    Detail::DeviceBuffer m_Png;
    Detail::CompressionStorage m_Compression;

    ~Storage()
    {
        // cudaFree must run under the device that owns this context, even if its
        // caller selected another device before destroying the encoder.
        int previous = -1;
        if (m_Device >= 0 && cudaGetDevice(&previous) == cudaSuccess &&
            cudaSetDevice(m_Device) == cudaSuccess) {
            m_Scanlines.Reset();
            m_Png.Reset();
            m_Compression.Reset();
            cudaSetDevice(previous);
        }
    }
};

GpuPngEncoder::GpuPngEncoder() : m_Storage{std::make_unique<Storage>()} {}
GpuPngEncoder::~GpuPngEncoder() = default;

size_t
GpuPngEncoder::GetWorkspaceBytes() const
{
    return m_Storage->m_Scanlines.Bytes() + m_Storage->m_Png.Capacity() +
           m_Storage->m_Compression.Bytes();
}

cudaError_t
GpuPngEncoder::Encode(const Color16 *devicePixels,
                      uint32_t width,
                      uint32_t height,
                      size_t rowStrideBytes,
                      AlphaMode alphaMode,
                      cudaStream_t stream,
                      std::vector<unsigned char> &pngBytes)
{
    pngBytes.clear();
    size_t pixelCount = 0;
    size_t minimumStride = 0;
    size_t inputSpan = 0;
    size_t inputEnd = 0;
    // Validate byte-span arithmetic as well as dimensions before dereferencing pitched input.
    if ((alphaMode != AlphaMode::Detect && alphaMode != AlphaMode::Opaque) || devicePixels == nullptr ||
        width == 0 || height == 0 || width > Format::MaximumDimension ||
        height > Format::MaximumDimension || !Detail::MultiplySize(width, height, pixelCount) ||
        !Detail::GridFits(pixelCount, Detail::LinearThreadsPerBlock) ||
        !Detail::MultiplySize(width, sizeof(Color16), minimumStride) || rowStrideBytes < minimumStride ||
        rowStrideBytes % alignof(Color16) != 0 ||
        reinterpret_cast<uintptr_t>(devicePixels) % alignof(Color16) != 0 ||
        !Detail::MultiplySize(height - 1, rowStrideBytes, inputSpan) ||
        !Detail::AddSize(inputSpan, minimumStride, inputSpan) ||
        !Detail::AddSize(reinterpret_cast<uintptr_t>(devicePixels), inputSpan, inputEnd)) {
        return cudaErrorInvalidValue;
    }
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        return error;
    }
    if (m_Storage->m_Device >= 0 && m_Storage->m_Device != device) {
        return cudaErrorInvalidDevice;
    }
    m_Storage->m_Device = device;
    Detail::PendingStreamWork pending{stream};
    unsigned channels = 0;
    size_t filteredBytes = 0;
    error = m_Storage->m_Scanlines.Prepare(
        devicePixels, width, height, rowStrideBytes, alphaMode, stream, channels, filteredBytes);
    if (error != cudaSuccess) {
        return error;
    }
    size_t zlibBytes = 0;
    error = m_Storage->m_Compression.Compress(
        m_Storage->m_Scanlines.Data(), filteredBytes, stream, zlibBytes, true);
    if (error != cudaSuccess) {
        return error;
    }
    error = Detail::AssemblePng(m_Storage->m_Png,
                                width,
                                height,
                                channels,
                                m_Storage->m_Compression.Data(),
                                zlibBytes,
                                stream,
                                pngBytes);
    // Assembly fences this stream, including all preceding stages. Avoid a second success fence.
    if (error == cudaSuccess) {
        pending.Complete();
    }
    return error;
}

} // namespace Png
