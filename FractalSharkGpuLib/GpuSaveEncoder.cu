#include "stdafx.h"

#include "AntialiasingKernel.cuh"
#include "GpuSaveEncoder.h"
#include "PngCudaSupport.cuh"
#include "PngEncoder.cuh"
#include "PngFormat.cuh"

namespace Png {

class GpuSaveEncoder::Storage {
public:
    int m_Device = -1;
    cudaStream_t m_Stream = nullptr;
    Detail::DeviceBuffer m_Iterations;
    Detail::DeviceBuffer m_Palette;
    Detail::DeviceBuffer m_Colors;
    GpuPngEncoder m_Encoder;

    ~Storage()
    {
        int previous = -1;
        if (m_Device >= 0 && cudaGetDevice(&previous) == cudaSuccess &&
            cudaSetDevice(m_Device) == cudaSuccess) {
            // Every encode has fenced its uploads. Destroy stream and retained allocations
            // on the device that created them, even if the caller changed its current device.
            if (m_Stream != nullptr) {
                cudaStreamDestroy(m_Stream);
            }
            m_Iterations.Reset();
            m_Palette.Reset();
            m_Colors.Reset();
            cudaSetDevice(previous);
        }
    }

    cudaError_t
    Initialize()
    {
        int device = -1;
        auto error = cudaGetDevice(&device);
        if (error != cudaSuccess) {
            return error;
        }
        if (m_Device >= 0 && m_Device != device) {
            return cudaErrorInvalidDevice;
        }
        m_Device = device;
        if (m_Stream == nullptr) {
            int low = 0;
            int high = 0;
            error = cudaDeviceGetStreamPriorityRange(&low, &high);
            if (error != cudaSuccess) {
                return error;
            }
            error = cudaStreamCreateWithPriority(&m_Stream, cudaStreamNonBlocking, low);
        }
        return error;
    }
};

GpuSaveEncoder::GpuSaveEncoder() : m_Storage(std::make_unique<Storage>()) {}
GpuSaveEncoder::~GpuSaveEncoder() = default;

const char *
GpuSaveEncoder::ErrorString(uint32_t result)
{
    return cudaGetErrorString(static_cast<cudaError_t>(result));
}

size_t
GpuSaveEncoder::GetWorkspaceBytes() const
{
    return m_Storage->m_Iterations.Capacity() + m_Storage->m_Palette.Capacity() +
           m_Storage->m_Colors.Capacity() + m_Storage->m_Encoder.GetWorkspaceBytes();
}

template <typename IterType>
uint32_t
GpuSaveEncoder::Encode(const IterType *iterations,
                       size_t strideElements,
                       uint32_t width,
                       uint32_t height,
                       uint32_t antialiasing,
                       IterType numIterations,
                       const Color16 *palette,
                       uint32_t paletteSize,
                       uint32_t auxDepth,
                       uint64_t rotation,
                       uint64_t maxIterations,
                       std::vector<unsigned char> &pngBytes)
{
    pngBytes.clear();
    size_t inputWidth = 0;
    size_t inputHeight = 0;
    size_t strideBytes = 0;
    size_t iterationBytes = 0;
    size_t colorBytes = 0;
    size_t paletteBytes = 0;
    size_t inputSpan = 0;
    size_t inputEnd = 0;
    if (iterations == nullptr || palette == nullptr || paletteSize == 0 || width == 0 || height == 0 ||
        width > Format::MaximumDimension || height > Format::MaximumDimension || antialiasing < 1 ||
        antialiasing > 4 || numIterations == 0 || maxIterations < 2 || auxDepth >= 64 ||
        !Detail::MultiplySize(width, antialiasing, inputWidth) ||
        !Detail::MultiplySize(height, antialiasing, inputHeight) || strideElements < inputWidth ||
        !Detail::MultiplySize(strideElements, sizeof(IterType), strideBytes) ||
        !Detail::MultiplySize(inputWidth, inputHeight, iterationBytes) ||
        !Detail::MultiplySize(iterationBytes, sizeof(IterType), iterationBytes) ||
        !Detail::MultiplySize(width, height, colorBytes) ||
        !Detail::MultiplySize(colorBytes, sizeof(Color16), colorBytes) ||
        !Detail::MultiplySize(paletteSize, sizeof(Color16), paletteBytes) ||
        !Detail::MultiplySize(inputHeight - 1, strideBytes, inputSpan) ||
        !Detail::AddSize(inputSpan, inputWidth * sizeof(IterType), inputSpan) ||
        !Detail::AddSize(reinterpret_cast<uintptr_t>(iterations), inputSpan, inputEnd)) {
        return cudaErrorInvalidValue;
    }
    auto error = m_Storage->Initialize();
    if (error != cudaSuccess) {
        return error;
    }
    Detail::PendingStreamWork pending(m_Storage->m_Stream);
    error = m_Storage->m_Iterations.Reserve(iterationBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Storage->m_Palette.Reserve(paletteBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Storage->m_Colors.Reserve(colorBytes);
    if (error != cudaSuccess) {
        return error;
    }

    // Upload active samples only: host rows are padded, device rows are packed.
    error = cudaMemcpy2DAsync(m_Storage->m_Iterations.Data<IterType>(),
                              inputWidth * sizeof(IterType),
                              iterations,
                              strideBytes,
                              inputWidth * sizeof(IterType),
                              inputHeight,
                              cudaMemcpyHostToDevice,
                              m_Storage->m_Stream);
    if (error != cudaSuccess) {
        return error;
    }
    // A snapshot's address is transient. Always upload its palette while retaining capacity.
    error = cudaMemcpyAsync(m_Storage->m_Palette.Data<Color16>(),
                            palette,
                            paletteBytes,
                            cudaMemcpyHostToDevice,
                            m_Storage->m_Stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = ::LaunchColoring(m_Storage->m_Iterations.Data<IterType>(),
                             inputWidth,
                             m_Storage->m_Colors.Data<Color16>(),
                             m_Storage->m_Palette.Data<Color16>(),
                             paletteSize,
                             auxDepth,
                             rotation,
                             maxIterations,
                             width,
                             height,
                             antialiasing,
                             numIterations,
                             FractalShark::ColoringMode::PaletteLookup,
                             m_Storage->m_Stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Storage->m_Encoder.Encode(m_Storage->m_Colors.Data<Color16>(),
                                        width,
                                        height,
                                        static_cast<size_t>(width) * sizeof(Color16),
                                        AlphaMode::Opaque,
                                        m_Storage->m_Stream,
                                        pngBytes);
    // GpuPngEncoder already fenced this stream on success. Earlier failures and exceptions
    // still need this outer guard before snapshot-owned host inputs can be released.
    if (error == cudaSuccess) {
        pending.Complete();
    }
    return error;
}

template uint32_t GpuSaveEncoder::Encode<uint32_t>(const uint32_t *,
                                                   size_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   const Color16 *,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint64_t,
                                                   uint64_t,
                                                   std::vector<unsigned char> &);
template uint32_t GpuSaveEncoder::Encode<uint64_t>(const uint64_t *,
                                                   size_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint64_t,
                                                   const Color16 *,
                                                   uint32_t,
                                                   uint32_t,
                                                   uint64_t,
                                                   uint64_t,
                                                   std::vector<unsigned char> &);

} // namespace Png
