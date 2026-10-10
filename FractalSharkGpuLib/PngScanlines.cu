#include "stdafx.h"
#include "GPU_Types.h"
#include "PngStages.cuh"

namespace Png::Detail {
namespace {
__global__ void
DetectAlpha(
    const Color16 *pixels, size_t width, size_t height, size_t rowStrideBytes, unsigned *hasAlpha)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < width * height) {
        const auto *row = reinterpret_cast<const Color16 *>(
            reinterpret_cast<const unsigned char *>(pixels) + (index / width) * rowStrideBytes);
        if (row[index % width].a != Format::MaximumSample) {
            atomicExch(hasAlpha, 1u);
        }
    }
}

__global__ void
SerializePixels(const Color16 *pixels,
                size_t width,
                size_t height,
                size_t rowStrideBytes,
                unsigned channels,
                unsigned char *raw)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < width * height) {
        const auto *row = reinterpret_cast<const Color16 *>(
            reinterpret_cast<const unsigned char *>(pixels) + (index / width) * rowStrideBytes);
        const Color16 pixel = row[index % width];
        // PNG samples are big-endian, independent of native Color16 layout and row padding.
        const size_t offset = index * channels * Format::SampleBytes;
        raw[offset] = static_cast<unsigned char>(pixel.r >> 8);
        raw[offset + 1] = static_cast<unsigned char>(pixel.r);
        raw[offset + 2] = static_cast<unsigned char>(pixel.g >> 8);
        raw[offset + 3] = static_cast<unsigned char>(pixel.g);
        raw[offset + 4] = static_cast<unsigned char>(pixel.b >> 8);
        raw[offset + 5] = static_cast<unsigned char>(pixel.b);
        if (channels == Format::RgbaChannels) {
            raw[offset + 6] = static_cast<unsigned char>(pixel.a >> 8);
            raw[offset + 7] = static_cast<unsigned char>(pixel.a);
        }
    }
}

__device__ unsigned
Paeth(unsigned left, unsigned above, unsigned upperLeft)
{
    // PNG mandates ties prefer left, then above, then upper-left.
    const int prediction = static_cast<int>(left + above) - static_cast<int>(upperLeft);
    const int leftDistance = abs(prediction - static_cast<int>(left));
    const int aboveDistance = abs(prediction - static_cast<int>(above));
    const int cornerDistance = abs(prediction - static_cast<int>(upperLeft));
    if (leftDistance <= aboveDistance && leftDistance <= cornerDistance) {
        return left;
    }
    return aboveDistance <= cornerDistance ? above : upperLeft;
}

__device__ unsigned char
FilterByte(unsigned value, unsigned left, unsigned above, unsigned upperLeft, Format::Filter filter)
{
    unsigned predictor = 0;
    if (filter == Format::Filter::Sub) {
        predictor = left;
    } else if (filter == Format::Filter::Up) {
        predictor = above;
    } else if (filter == Format::Filter::Average) {
        predictor = (left + above) / 2;
    } else if (filter == Format::Filter::Paeth) {
        predictor = Paeth(left, above, upperLeft);
    }
    return static_cast<unsigned char>(value - predictor); // PNG differences wrap modulo 256.
}

__device__ unsigned
DifferenceScore(unsigned char value)
{
    // Match the bundled LodePNG LFS_MINSUM heuristic, including its 255 - value convention.
    constexpr unsigned signedByteBoundary = 1u << (Format::ByteBits - 1);
    constexpr unsigned maximumByte = (1u << Format::ByteBits) - 1;
    return value < signedByteBoundary ? value : maximumByte - value;
}

__global__ void
SelectFilters(const unsigned char *raw,
              size_t rowBytes,
              size_t height,
              unsigned pixelBytes,
              unsigned char *filters)
{
    // One warp scores a row. Keep five scalar accumulators rather than a CUDA local array.
    const size_t row = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (row >= height) {
        return;
    }
    uint64_t noneScore = 0;
    uint64_t subScore = 0;
    uint64_t upScore = 0;
    uint64_t averageScore = 0;
    uint64_t paethScore = 0;
    for (size_t byte = lane; byte < rowBytes; byte += WarpSize) {
        const size_t index = row * rowBytes + byte;
        const unsigned value = raw[index];
        const unsigned left = byte >= pixelBytes ? raw[index - pixelBytes] : 0;
        const unsigned above = row != 0 ? raw[index - rowBytes] : 0;
        const unsigned corner = row != 0 && byte >= pixelBytes ? raw[index - rowBytes - pixelBytes] : 0;
        noneScore += value; // None uses unsigned samples; differences use signed-magnitude scoring.
        subScore += DifferenceScore(FilterByte(value, left, above, corner, Format::Filter::Sub));
        upScore += DifferenceScore(FilterByte(value, left, above, corner, Format::Filter::Up));
        averageScore += DifferenceScore(FilterByte(value, left, above, corner, Format::Filter::Average));
        paethScore += DifferenceScore(FilterByte(value, left, above, corner, Format::Filter::Paeth));
    }
    noneScore = WarpSum(noneScore);
    subScore = WarpSum(subScore);
    upScore = WarpSum(upScore);
    averageScore = WarpSum(averageScore);
    paethScore = WarpSum(paethScore);
    if (lane == 0) {
        // Strict comparisons preserve the first filter on ties, matching the CPU heuristic.
        Format::Filter best = Format::Filter::None;
        uint64_t score = noneScore;
        if (subScore < score) {
            best = Format::Filter::Sub;
            score = subScore;
        }
        if (upScore < score) {
            best = Format::Filter::Up;
            score = upScore;
        }
        if (averageScore < score) {
            best = Format::Filter::Average;
            score = averageScore;
        }
        if (paethScore < score) {
            best = Format::Filter::Paeth;
        }
        filters[row] = static_cast<unsigned char>(best);
    }
}

__global__ void
FilterScanlines(const unsigned char *raw,
                size_t rowBytes,
                size_t height,
                unsigned pixelBytes,
                const unsigned char *filters,
                unsigned char *filtered)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t filteredRowBytes = rowBytes + Format::FilterPrefixBytes;
    if (index >= filteredRowBytes * height) {
        return;
    }
    const size_t row = index / filteredRowBytes;
    const size_t column = index % filteredRowBytes;
    const auto filter = static_cast<Format::Filter>(filters[row]);
    if (column == 0) {
        filtered[index] = static_cast<unsigned char>(filter);
        return;
    }
    const size_t byte = column - Format::FilterPrefixBytes;
    const size_t source = row * rowBytes + byte;
    const unsigned value = raw[source];
    const unsigned left = byte >= pixelBytes ? raw[source - pixelBytes] : 0;
    const unsigned above = row != 0 ? raw[source - rowBytes] : 0;
    const unsigned corner = row != 0 && byte >= pixelBytes ? raw[source - rowBytes - pixelBytes] : 0;
    filtered[index] = FilterByte(value, left, above, corner, filter);
}

} // namespace

cudaError_t
ScanlineStorage::Prepare(const Color16 *pixels,
                         uint32_t width,
                         uint32_t height,
                         size_t strideBytes,
                         AlphaMode alphaMode,
                         cudaStream_t stream,
                         unsigned &channels,
                         size_t &filteredBytes)
{
    const size_t pixelCount = static_cast<size_t>(width) * height;
    // The guard is destroyed before this stack-owned alpha metadata on every error path.
    unsigned hasAlpha = 0;
    PendingStreamWork pending(stream);
    auto error = cudaSuccess;
    channels = Format::RgbChannels;
    if (alphaMode == AlphaMode::Detect) {
        error = m_Alpha.Reserve(sizeof(unsigned));
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaMemsetAsync(m_Alpha.Data<unsigned>(), 0, sizeof(unsigned), stream);
        if (error != cudaSuccess) {
            return error;
        }
        DetectAlpha<<<GridSize(pixelCount, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
            pixels, width, height, strideBytes, m_Alpha.Data<unsigned>());
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaMemcpyAsync(
            &hasAlpha, m_Alpha.Data<unsigned>(), sizeof(hasAlpha), cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            return error;
        }
        // General image callers detect transparency; fractal coloring guarantees opaque alpha.
        channels = hasAlpha != 0 ? Format::RgbaChannels : Format::RgbChannels;
    }
    size_t rowBytes = 0;
    size_t rawBytes = 0;
    if (!MultiplySize(width, channels * Format::SampleBytes, rowBytes) ||
        !MultiplySize(rowBytes, height, rawBytes) ||
        !AddSize(rowBytes, Format::FilterPrefixBytes, filteredBytes) ||
        !MultiplySize(filteredBytes, height, filteredBytes) ||
        !GridFits(filteredBytes, LinearThreadsPerBlock)) {
        return cudaErrorInvalidValue;
    }
    error = m_Raw.Reserve(rawBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Filters.Reserve(height);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Filtered.Reserve(filteredBytes);
    if (error != cudaSuccess) {
        return error;
    }
    SerializePixels<<<GridSize(pixelCount, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
        pixels, width, height, strideBytes, channels, m_Raw.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    SelectFilters<<<GridSize(height, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        m_Raw.Data<unsigned char>(),
        rowBytes,
        height,
        channels * Format::SampleBytes,
        m_Filters.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    error = FilterScanlinesAsync(m_Raw.Data<unsigned char>(),
                                 width,
                                 height,
                                 channels,
                                 m_Filters.Data<unsigned char>(),
                                 m_Filtered.Data<unsigned char>(),
                                 stream);
    if (error != cudaSuccess) {
        return error;
    }
    // Remaining launches belong to the enclosing encoder's lifetime guard.
    pending.Complete();
    return cudaSuccess;
}

} // namespace Png::Detail

namespace Png::Detail {
cudaError_t
FilterScanlinesAsync(const unsigned char *deviceRaw,
                     uint32_t width,
                     uint32_t height,
                     uint32_t channels,
                     const unsigned char *deviceFilters,
                     unsigned char *deviceFiltered,
                     cudaStream_t stream)
{
    size_t rowBytes = 0;
    size_t bytes = 0;
    if (deviceRaw == nullptr || deviceFilters == nullptr || deviceFiltered == nullptr || width == 0 ||
        height == 0 || (channels != Format::RgbChannels && channels != Format::RgbaChannels) ||
        !Detail::MultiplySize(width, channels * Format::SampleBytes, rowBytes) ||
        !Detail::AddSize(rowBytes, Format::FilterPrefixBytes, bytes) ||
        !Detail::MultiplySize(bytes, height, bytes) ||
        !Detail::GridFits(bytes, Detail::LinearThreadsPerBlock)) {
        return cudaErrorInvalidValue;
    }
    Detail::FilterScanlines<<<Detail::GridSize(bytes, Detail::LinearThreadsPerBlock),
                              Detail::LinearThreadsPerBlock,
                              0,
                              stream>>>(
        deviceRaw, rowBytes, height, channels * Format::SampleBytes, deviceFilters, deviceFiltered);
    return cudaGetLastError();
}

} // namespace Png::Detail
