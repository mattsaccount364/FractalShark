#include "stdafx.h"

#include "PngCompressionKernels.cuh"
#include "PngEncoder.cuh"
#include "PngHuffman.cuh"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "GPU_Types.h"

#include <algorithm>
#include <limits>

namespace FractalShark::Png {
namespace {

constexpr unsigned WarpSize = 32;
constexpr unsigned HashEntries = 4096;
constexpr unsigned WarpsPerBlock = 4;
constexpr size_t RegionSlotBytes = (Detail::DeflateRegionBytes * 9 + 20) / 8 + 4;
constexpr uint64_t AdlerModulus = 65521;

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

bool
MultiplySize(size_t first, size_t second, size_t &result)
{
    if (second != 0 && first > std::numeric_limits<size_t>::max() / second) {
        return false;
    }
    result = first * second;
    return true;
}

bool
AddSize(size_t first, size_t second, size_t &result)
{
    if (first > std::numeric_limits<size_t>::max() - second) {
        return false;
    }
    result = first + second;
    return true;
}

__device__ uint64_t
WarpSum(uint64_t value)
{
    for (unsigned offset = WarpSize / 2; offset != 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void
DetectAlpha(const Color16 *pixels, size_t width, size_t height, size_t stride, unsigned *hasAlpha)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < width * height) {
        const auto *row = reinterpret_cast<const Color16 *>(
            reinterpret_cast<const unsigned char *>(pixels) + (index / width) * stride);
        if (row[index % width].a != 65535) {
            atomicExch(hasAlpha, 1u);
        }
    }
}

__global__ void
SerializePixels(const Color16 *pixels,
                size_t width,
                size_t height,
                size_t stride,
                unsigned channels,
                unsigned char *raw)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < width * height) {
        const auto *row = reinterpret_cast<const Color16 *>(
            reinterpret_cast<const unsigned char *>(pixels) + (index / width) * stride);
        const Color16 pixel = row[index % width];
        const size_t offset = index * channels * 2;
        raw[offset] = static_cast<unsigned char>(pixel.r >> 8);
        raw[offset + 1] = static_cast<unsigned char>(pixel.r);
        raw[offset + 2] = static_cast<unsigned char>(pixel.g >> 8);
        raw[offset + 3] = static_cast<unsigned char>(pixel.g);
        raw[offset + 4] = static_cast<unsigned char>(pixel.b >> 8);
        raw[offset + 5] = static_cast<unsigned char>(pixel.b);
        if (channels == 4) {
            raw[offset + 6] = static_cast<unsigned char>(pixel.a >> 8);
            raw[offset + 7] = static_cast<unsigned char>(pixel.a);
        }
    }
}

__device__ unsigned
Paeth(unsigned left, unsigned above, unsigned upperLeft)
{
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
FilterByte(unsigned value, unsigned left, unsigned above, unsigned upperLeft, unsigned filter)
{
    unsigned predictor = 0;
    if (filter == 1) {
        predictor = left;
    } else if (filter == 2) {
        predictor = above;
    } else if (filter == 3) {
        predictor = (left + above) / 2;
    } else if (filter == 4) {
        predictor = Paeth(left, above, upperLeft);
    }
    return static_cast<unsigned char>(value - predictor);
}

__device__ unsigned
DifferenceScore(unsigned char value)
{
    // Match the bundled LodePNG LFS_MINSUM heuristic, including its 255 - value convention.
    return value < 128 ? value : 255u - value;
}

__global__ void
SelectFilters(const unsigned char *raw,
              size_t rowBytes,
              size_t height,
              unsigned pixelBytes,
              unsigned char *filters)
{
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
        noneScore += value;
        subScore += DifferenceScore(FilterByte(value, left, above, corner, 1));
        upScore += DifferenceScore(FilterByte(value, left, above, corner, 2));
        averageScore += DifferenceScore(FilterByte(value, left, above, corner, 3));
        paethScore += DifferenceScore(FilterByte(value, left, above, corner, 4));
    }
    noneScore = WarpSum(noneScore);
    subScore = WarpSum(subScore);
    upScore = WarpSum(upScore);
    averageScore = WarpSum(averageScore);
    paethScore = WarpSum(paethScore);
    if (lane == 0) {
        unsigned best = 0;
        uint64_t score = noneScore;
        if (subScore < score) {
            best = 1;
            score = subScore;
        }
        if (upScore < score) {
            best = 2;
            score = upScore;
        }
        if (averageScore < score) {
            best = 3;
            score = averageScore;
        }
        if (paethScore < score) {
            best = 4;
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
    const size_t filteredRowBytes = rowBytes + 1;
    if (index >= filteredRowBytes * height) {
        return;
    }
    const size_t row = index / filteredRowBytes;
    const size_t column = index % filteredRowBytes;
    const unsigned filter = filters[row];
    if (column == 0) {
        filtered[index] = static_cast<unsigned char>(filter);
        return;
    }
    const size_t byte = column - 1;
    const size_t source = row * rowBytes + byte;
    const unsigned value = raw[source];
    const unsigned left = byte >= pixelBytes ? raw[source - pixelBytes] : 0;
    const unsigned above = row != 0 ? raw[source - rowBytes] : 0;
    const unsigned corner = row != 0 && byte >= pixelBytes ? raw[source - rowBytes - pixelBytes] : 0;
    filtered[index] = FilterByte(value, left, above, corner, filter);
}

class BitWriter {
public:
    __device__ explicit BitWriter(unsigned char *output) : m_Output{output} {}

    __device__ void
    Write(unsigned value, unsigned count)
    {
        m_Bits |= static_cast<uint64_t>(value) << m_Count;
        m_Count += count;
        while (m_Count >= 8) {
            m_Output[m_Bytes++] = static_cast<unsigned char>(m_Bits);
            m_Bits >>= 8;
            m_Count -= 8;
        }
    }

    __device__ void
    Align()
    {
        if (m_Count != 0) {
            Write(0, 8 - m_Count);
        }
    }

    __device__ void
    Symbol(unsigned symbol, const Detail::Huffman::Region *table)
    {
        if (table != nullptr) {
            Write(table->m_Codes[symbol], table->m_Lengths[symbol]);
            return;
        }
        unsigned code = 0;
        unsigned count = 0;
        if (symbol <= 143) {
            code = 0x30 + symbol;
            count = 8;
        } else if (symbol <= 255) {
            code = 0x190 + symbol - 144;
            count = 9;
        } else if (symbol <= 279) {
            code = symbol - 256;
            count = 7;
        } else {
            code = 0xc0 + symbol - 280;
            count = 8;
        }
        Write(__brev(code) >> (32 - count), count);
    }

    __device__ void
    Match(unsigned length,
          unsigned distance,
          const Detail::Huffman::Region *table,
          Detail::Huffman::Region *histogram)
    {
        unsigned lengthSymbol = 285;
        unsigned lengthValue = 0;
        unsigned lengthBits = 0;
        if (length == 258) {
            lengthSymbol = 285;
        } else {
            unsigned base = 3;
            for (unsigned symbol = 257; symbol <= 284; ++symbol) {
                const unsigned extra = symbol <= 264 ? 0 : (symbol - 261) / 4;
                if (length < base + (1u << extra)) {
                    lengthSymbol = symbol;
                    lengthValue = length - base;
                    lengthBits = extra;
                    break;
                }
                base += 1u << extra;
            }
        }
        Symbol(lengthSymbol, table);
        Write(lengthValue, lengthBits);
        if (histogram != nullptr) {
            ++histogram->m_Frequencies[lengthSymbol];
            histogram->m_ExtraBits += lengthBits;
        }
        unsigned base = 1;
        for (unsigned code = 0; code < 30; ++code) {
            const unsigned extra = code < 4 ? 0 : code / 2 - 1;
            if (distance < base + (1u << extra)) {
                if (table == nullptr) {
                    Write(__brev(code) >> 27, 5);
                } else {
                    Write(table->m_Codes[Detail::Huffman::LiteralSymbols + code],
                          table->m_Lengths[Detail::Huffman::LiteralSymbols + code]);
                }
                Write(distance - base, extra);
                if (histogram != nullptr) {
                    ++histogram->m_Frequencies[Detail::Huffman::LiteralSymbols + code];
                    histogram->m_ExtraBits += extra;
                }
                break;
            }
            base += 1u << extra;
        }
    }

    __device__ size_t
    Bytes() const
    {
        return m_Bytes;
    }

private:
    unsigned char *m_Output;
    uint64_t m_Bits{};
    unsigned m_Count{};
    size_t m_Bytes{};
};

__device__ unsigned
HashAt(const unsigned char *input, unsigned position)
{
    const unsigned key = (static_cast<unsigned>(input[position]) << 16) |
                         (static_cast<unsigned>(input[position + 1]) << 8) | input[position + 2];
    return (key * 2654435761u) >> 20;
}

__device__ unsigned
CompareMatch(
    const unsigned char *input, unsigned position, unsigned previous, unsigned maximum, unsigned lane)
{
    unsigned matched = 0;
    while (matched < maximum) {
        const unsigned offset = matched + lane;
        const bool different = offset < maximum && input[position + offset] != input[previous + offset];
        const unsigned mismatches = __ballot_sync(0xffffffffu, different);
        if (mismatches != 0) {
            return matched + static_cast<unsigned>(__ffs(static_cast<int>(mismatches)) - 1);
        }
        matched += maximum - matched < WarpSize ? maximum - matched : WarpSize;
    }
    return matched;
}

__device__ void
ParseRegion(const unsigned char *source,
            unsigned bytes,
            unsigned *heads,
            unsigned lane,
            BitWriter &writer,
            const Detail::Huffman::Region *table,
            Detail::Huffman::Region *histogram)
{
    unsigned position = 0;
    while (position < bytes) {
        unsigned previous = 0;
        if (lane == 0 && position + 2 < bytes) {
            previous = heads[HashAt(source, position)];
        }
        previous = __shfl_sync(0xffffffffu, previous, 0);
        unsigned length = 0;
        if (previous != 0) {
            const unsigned maximum = bytes - position < 258 ? bytes - position : 258;
            length = CompareMatch(source, position, previous - 1, maximum, lane);
        }
        const unsigned consumed = length >= 3 ? length : 1;
        if (lane == 0) {
            if (length >= 3) {
                writer.Match(length, position - (previous - 1), table, histogram);
            } else {
                writer.Symbol(source[position], table);
                if (histogram != nullptr) {
                    ++histogram->m_Frequencies[source[position]];
                }
            }
        }
        // Insert every consumed position. atomicMax makes hash collisions deterministic
        // when the warp inserts several skipped positions into the same bucket.
        for (unsigned offset = lane; offset < consumed; offset += WarpSize) {
            const unsigned inserted = position + offset;
            if (inserted + 2 < bytes) {
                atomicMax(heads + HashAt(source, inserted), inserted + 1);
            }
        }
        __syncwarp();
        position += consumed;
    }
}

__device__ void
FinishRegion(BitWriter &writer, const Detail::Huffman::Region *table)
{
    writer.Symbol(256, table);
    writer.Write(0, 3); // Empty stored block restores byte alignment.
    writer.Align();
    writer.Write(0, 16);
    writer.Write(65535, 16);
}

__global__ void
CompressRegions(const unsigned char *input,
                size_t inputBytes,
                size_t regions,
                unsigned *hashHeads,
                unsigned char *slots,
                uint64_t *sizes,
                uint64_t *adlerSums,
                uint64_t *adlerWeighted,
                Detail::Huffman::Region *huffman)
{
    const size_t region = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (region >= regions) {
        return;
    }
    const size_t start = region * Detail::DeflateRegionBytes;
    const unsigned bytes = static_cast<unsigned>(inputBytes - start < Detail::DeflateRegionBytes
                                                     ? inputBytes - start
                                                     : Detail::DeflateRegionBytes);
    const unsigned char *source = input + start;
    unsigned *heads = hashHeads + region * HashEntries;
    unsigned char *slot = slots + region * RegionSlotBytes;
    Detail::Huffman::Region *histogram = huffman == nullptr ? nullptr : huffman + region;
    for (unsigned entry = lane; entry < HashEntries; entry += WarpSize) {
        heads[entry] = 0;
    }
    if (histogram != nullptr) {
        for (unsigned symbol = lane; symbol < Detail::Huffman::TableSymbols; symbol += WarpSize) {
            histogram->m_Frequencies[symbol] = 0;
        }
        __syncwarp();
        if (lane == 0) {
            histogram->m_Frequencies[256] = 1;
            histogram->m_ExtraBits = 0;
        }
    }
    uint64_t sum = 0;
    uint64_t weighted = 0;
    for (unsigned byte = lane; byte < bytes; byte += WarpSize) {
        sum += source[byte];
        weighted += static_cast<uint64_t>(bytes - byte) * source[byte];
    }
    sum = WarpSum(sum);
    weighted = WarpSum(weighted);
    if (lane == 0) {
        adlerSums[region] = sum;
        adlerWeighted[region] = weighted;
    }
    __syncwarp();
    BitWriter writer{slot};
    if (lane == 0) {
        writer.Write(2, 3); // BFINAL=0, fixed Huffman codes.
    }
    ParseRegion(source, bytes, heads, lane, writer, nullptr, histogram);
    if (lane == 0) {
        FinishRegion(writer, nullptr);
        sizes[region] = writer.Bytes();
    }
    __syncwarp();
    if (sizes[region] >= bytes + 5u) {
        if (lane == 0) {
            slot[0] = 0;
            slot[1] = static_cast<unsigned char>(bytes);
            slot[2] = static_cast<unsigned char>(bytes >> 8);
            slot[3] = static_cast<unsigned char>(~bytes);
            slot[4] = static_cast<unsigned char>((~bytes) >> 8);
            sizes[region] = bytes + 5u;
        }
        for (unsigned byte = lane; byte < bytes; byte += WarpSize) {
            slot[byte + 5] = source[byte];
        }
    }
}

__global__ void
OptimizeRegions(const unsigned char *input,
                size_t inputBytes,
                size_t regions,
                unsigned *hashHeads,
                unsigned char *slots,
                uint64_t *sizes,
                Detail::Huffman::Region *huffman)
{
    const size_t region = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (region >= regions) {
        return;
    }
    Detail::Huffman::Region *table = huffman + region;
    if (lane == 0) {
        Detail::Huffman::BuildRegion(table, sizes[region]);
    }
    __syncwarp();
    if (table->m_UseDynamic == 0) {
        return;
    }
    unsigned *heads = hashHeads + region * HashEntries;
    for (unsigned entry = lane; entry < HashEntries; entry += WarpSize) {
        heads[entry] = 0;
    }
    __syncwarp();
    const size_t start = region * Detail::DeflateRegionBytes;
    const unsigned bytes = static_cast<unsigned>(inputBytes - start < Detail::DeflateRegionBytes
                                                     ? inputBytes - start
                                                     : Detail::DeflateRegionBytes);
    BitWriter writer{slots + region * RegionSlotBytes};
    if (lane == 0) {
        Detail::Huffman::WriteHeader(writer, table);
    }
    ParseRegion(input + start, bytes, heads, lane, writer, table, nullptr);
    if (lane == 0) {
        FinishRegion(writer, table);
        sizes[region] = writer.Bytes();
    }
}

__global__ void
BuildLengthsFixture(const unsigned *frequencies,
                    unsigned symbols,
                    unsigned maximumBits,
                    Detail::Huffman::Workspace *workspace,
                    unsigned *lengths)
{
    Detail::Huffman::BuildLengths(frequencies, symbols, maximumBits, workspace, lengths);
}

__global__ void
EncodeRunsFixture(const unsigned *lengths,
                  unsigned count,
                  Detail::CodeLengthRun *runs,
                  unsigned *runCount)
{
    for (unsigned index = 0; index < count; ++index) {
        if (lengths[index] > 15) {
            *runCount = 0xffffffffu;
            return;
        }
    }
    *runCount = Detail::Huffman::EncodeRuns(lengths, count, runs);
}

__device__ void
WriteBigEndian(unsigned char *output, uint32_t value)
{
    output[0] = static_cast<unsigned char>(value >> 24);
    output[1] = static_cast<unsigned char>(value >> 16);
    output[2] = static_cast<unsigned char>(value >> 8);
    output[3] = static_cast<unsigned char>(value);
}

__global__ void
WriteZlibMetadata(size_t inputBytes,
                  size_t regions,
                  const uint64_t *sizes,
                  const uint64_t *offsets,
                  const uint64_t *adlerSums,
                  const uint64_t *adlerWeighted,
                  unsigned char *zlib,
                  uint64_t *resultBytes)
{
    uint64_t sum = 0;
    uint64_t weighted = 0;
    for (size_t region = threadIdx.x; region < regions; region += WarpSize) {
        const size_t end = (region + 1) * Detail::DeflateRegionBytes < inputBytes
                               ? (region + 1) * Detail::DeflateRegionBytes
                               : inputBytes;
        sum = (sum + adlerSums[region]) % AdlerModulus;
        weighted = (weighted + ((inputBytes - end) % AdlerModulus) * (adlerSums[region] % AdlerModulus) +
                    adlerWeighted[region]) %
                   AdlerModulus;
    }
    sum = WarpSum(sum);
    weighted = WarpSum(weighted);
    if (threadIdx.x == 0) {
        const size_t end = 2 + offsets[regions - 1] + sizes[regions - 1];
        zlib[0] = 0x78;
        zlib[1] = 0x01;
        zlib[end] = 1; // The only final block in the stream.
        zlib[end + 1] = 0;
        zlib[end + 2] = 0;
        zlib[end + 3] = 255;
        zlib[end + 4] = 255;
        const uint32_t adler =
            static_cast<uint32_t>((((inputBytes % AdlerModulus + weighted) % AdlerModulus) << 16) |
                                  ((1 + sum) % AdlerModulus));
        WriteBigEndian(zlib + end + 5, adler);
        *resultBytes = end + 9;
    }
}

__global__ void
CompactRegions(const unsigned char *slots,
               size_t regions,
               const uint64_t *sizes,
               const uint64_t *offsets,
               unsigned char *zlib)
{
    const size_t region = blockIdx.x;
    if (region < regions) {
        for (size_t byte = threadIdx.x; byte < sizes[region]; byte += blockDim.x) {
            zlib[2 + offsets[region] + byte] = slots[region * RegionSlotBytes + byte];
        }
    }
}

__device__ uint32_t
CrcBytes(const unsigned char *input, size_t bytes)
{
    uint32_t crc = 0xffffffffu;
    for (size_t byte = 0; byte < bytes; ++byte) {
        crc ^= input[byte];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return crc ^ 0xffffffffu;
}

// Reflected CRC polynomial multiplication, with bit 31 representing the identity.
// Scalar polynomial arithmetic avoids local matrix arrays in CUDA code.
__device__ uint32_t
CrcMultiply(uint32_t first, uint32_t second)
{
    uint32_t product = 0;
    for (uint32_t mask = 0x80000000u; mask != 0; mask >>= 1) {
        if ((first & mask) != 0) {
            product ^= second;
        }
        second = (second >> 1) ^ (0xedb88320u & (0u - (second & 1u)));
    }
    return product;
}

__device__ uint32_t
CrcCombine(uint32_t first, uint32_t second, size_t secondBytes)
{
    uint32_t factor = 0x80000000u;
    uint32_t power = 0x00800000u; // x^8.
    while (secondBytes != 0) {
        if ((secondBytes & 1) != 0) {
            factor = CrcMultiply(power, factor);
        }
        secondBytes >>= 1;
        power = CrcMultiply(power, power);
    }
    return CrcMultiply(factor, first) ^ second;
}

__global__ void
WritePngHeaders(uint32_t width,
                uint32_t height,
                unsigned channels,
                size_t zlibBytes,
                size_t chunks,
                unsigned char *png)
{
    if (threadIdx.x != 0) {
        return;
    }
    png[0] = 137;
    png[1] = 80;
    png[2] = 78;
    png[3] = 71;
    png[4] = 13;
    png[5] = 10;
    png[6] = 26;
    png[7] = 10;
    WriteBigEndian(png + 8, 13);
    png[12] = 'I';
    png[13] = 'H';
    png[14] = 'D';
    png[15] = 'R';
    WriteBigEndian(png + 16, width);
    WriteBigEndian(png + 20, height);
    png[24] = 16;
    png[25] = channels == 3 ? 2 : 6;
    png[26] = 0;
    png[27] = 0;
    png[28] = 0;
    WriteBigEndian(png + 29, CrcBytes(png + 12, 17));
    const size_t end = 33 + zlibBytes + chunks * 12;
    WriteBigEndian(png + end, 0);
    png[end + 4] = 'I';
    png[end + 5] = 'E';
    png[end + 6] = 'N';
    png[end + 7] = 'D';
    WriteBigEndian(png + end + 8, CrcBytes(png + end + 4, 4));
}

__global__ void
CopyIdat(const unsigned char *zlib, size_t bytes, unsigned char *png)
{
    const size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte < bytes) {
        const size_t chunk = byte / Detail::IdatPayloadBytes;
        png[33 + chunk * (Detail::IdatPayloadBytes + 12) + 8 + byte % Detail::IdatPayloadBytes] =
            zlib[byte];
    }
}

__global__ void
WriteIdatHeaders(size_t bytes, size_t chunks, unsigned char *png)
{
    const size_t chunk = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (chunk < chunks) {
        const size_t remaining = bytes - chunk * Detail::IdatPayloadBytes;
        const size_t payload =
            remaining < Detail::IdatPayloadBytes ? remaining : Detail::IdatPayloadBytes;
        unsigned char *header = png + 33 + chunk * (Detail::IdatPayloadBytes + 12);
        WriteBigEndian(header, static_cast<uint32_t>(payload));
        header[4] = 'I';
        header[5] = 'D';
        header[6] = 'A';
        header[7] = 'T';
    }
}

__global__ void
WriteIdatCrcs(size_t bytes, size_t chunks, unsigned char *png)
{
    const size_t chunk = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (chunk >= chunks) {
        return;
    }
    const size_t remaining = bytes - chunk * Detail::IdatPayloadBytes;
    const size_t payload = remaining < Detail::IdatPayloadBytes ? remaining : Detail::IdatPayloadBytes;
    unsigned char *header = png + 33 + chunk * (Detail::IdatPayloadBytes + 12);
    const size_t partBytes = (payload + 4 + WarpSize - 1) / WarpSize;
    const size_t begin = lane * partBytes;
    size_t length = begin < payload + 4 ? payload + 4 - begin : 0;
    length = length < partBytes ? length : partBytes;
    uint32_t crc = length != 0 ? CrcBytes(header + 4 + begin, length) : 0;
    for (unsigned stride = 1; stride < WarpSize; stride *= 2) {
        const uint32_t next = __shfl_down_sync(0xffffffffu, crc, stride);
        const size_t nextLength = __shfl_down_sync(0xffffffffu, length, stride);
        if (lane % (2 * stride) == 0) {
            crc = CrcCombine(crc, next, nextLength);
            length += nextLength;
        }
    }
    if (lane == 0) {
        WriteBigEndian(header + 8 + payload, crc);
    }
}

bool
GridFits(size_t items, size_t threads)
{
    return items != 0 && (items - 1) / threads < static_cast<size_t>(std::numeric_limits<int>::max());
}

unsigned
GridSize(size_t items, unsigned threads)
{
    return static_cast<unsigned>((items - 1) / threads + 1);
}

class CompressionStorage {
public:
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

cudaError_t
Compress(CompressionStorage &storage,
         const unsigned char *input,
         size_t bytes,
         cudaStream_t stream,
         size_t &resultBytes,
         bool allowDynamic)
{
    if (input == nullptr || bytes == 0) {
        return cudaErrorInvalidValue;
    }
    const size_t regions = (bytes - 1) / Detail::DeflateRegionBytes + 1;
    size_t headsBytes = 0;
    size_t slotsBytes = 0;
    size_t countsBytes = 0;
    size_t zlibBound = 0;
    size_t huffmanBytes = 0;
    if (regions > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !MultiplySize(regions, HashEntries * sizeof(unsigned), headsBytes) ||
        !MultiplySize(regions, RegionSlotBytes, slotsBytes) ||
        !MultiplySize(regions, sizeof(uint64_t), countsBytes) || !MultiplySize(regions, 5, zlibBound) ||
        !AddSize(zlibBound, bytes, zlibBound) || !AddSize(zlibBound, 11, zlibBound) ||
        !MultiplySize(regions, allowDynamic ? sizeof(Detail::Huffman::Region) : 0, huffmanBytes)) {
        return cudaErrorInvalidValue;
    }
    auto error = storage.m_Heads.Reserve(headsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Huffman.Reserve(huffmanBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Slots.Reserve(slotsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Sizes.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Offsets.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_AdlerSums.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_AdlerWeighted.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Zlib.Reserve(zlibBound);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_ResultBytes.Reserve(sizeof(uint64_t));
    if (error != cudaSuccess) {
        return error;
    }
    size_t scanBytes = 0;
    error = cub::DeviceScan::ExclusiveSum(nullptr,
                                          scanBytes,
                                          storage.m_Sizes.Data<uint64_t>(),
                                          storage.m_Offsets.Data<uint64_t>(),
                                          static_cast<int>(regions),
                                          stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = storage.m_Scan.Reserve(scanBytes);
    if (error != cudaSuccess) {
        return error;
    }
    CompressRegions<<<GridSize(regions, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        input,
        bytes,
        regions,
        storage.m_Heads.Data<unsigned>(),
        storage.m_Slots.Data<unsigned char>(),
        storage.m_Sizes.Data<uint64_t>(),
        storage.m_AdlerSums.Data<uint64_t>(),
        storage.m_AdlerWeighted.Data<uint64_t>(),
        allowDynamic ? storage.m_Huffman.Data<Detail::Huffman::Region>() : nullptr);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    if (allowDynamic) {
        OptimizeRegions<<<GridSize(regions, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
            input,
            bytes,
            regions,
            storage.m_Heads.Data<unsigned>(),
            storage.m_Slots.Data<unsigned char>(),
            storage.m_Sizes.Data<uint64_t>(),
            storage.m_Huffman.Data<Detail::Huffman::Region>());
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
    }
    error = cub::DeviceScan::ExclusiveSum(storage.m_Scan.Data<void>(),
                                          scanBytes,
                                          storage.m_Sizes.Data<uint64_t>(),
                                          storage.m_Offsets.Data<uint64_t>(),
                                          static_cast<int>(regions),
                                          stream);
    if (error != cudaSuccess) {
        return error;
    }
    CompactRegions<<<static_cast<unsigned>(regions), 128, 0, stream>>>(
        storage.m_Slots.Data<unsigned char>(),
        regions,
        storage.m_Sizes.Data<uint64_t>(),
        storage.m_Offsets.Data<uint64_t>(),
        storage.m_Zlib.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    WriteZlibMetadata<<<1, WarpSize, 0, stream>>>(bytes,
                                                  regions,
                                                  storage.m_Sizes.Data<uint64_t>(),
                                                  storage.m_Offsets.Data<uint64_t>(),
                                                  storage.m_AdlerSums.Data<uint64_t>(),
                                                  storage.m_AdlerWeighted.Data<uint64_t>(),
                                                  storage.m_Zlib.Data<unsigned char>(),
                                                  storage.m_ResultBytes.Data<uint64_t>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    uint64_t actualBytes = 0;
    error = cudaMemcpyAsync(&actualBytes,
                            storage.m_ResultBytes.Data<uint64_t>(),
                            sizeof(actualBytes),
                            cudaMemcpyDeviceToHost,
                            stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) {
        resultBytes = static_cast<size_t>(actualBytes);
    }
    return error;
}

} // namespace

class GpuPngEncoder::Storage {
public:
    int m_Device{-1};
    DeviceBuffer m_Alpha;
    DeviceBuffer m_Raw;
    DeviceBuffer m_Filters;
    DeviceBuffer m_Filtered;
    DeviceBuffer m_Png;
    CompressionStorage m_Compression;

    ~Storage()
    {
        // cudaFree must run under the device that owns this context, even if its
        // caller selected another device before destroying the encoder.
        int previous = -1;
        if (m_Device >= 0 && cudaGetDevice(&previous) == cudaSuccess &&
            cudaSetDevice(m_Device) == cudaSuccess) {
            m_Alpha.Reset();
            m_Raw.Reset();
            m_Filters.Reset();
            m_Filtered.Reset();
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
    return m_Storage->m_Alpha.Capacity() + m_Storage->m_Raw.Capacity() +
           m_Storage->m_Filters.Capacity() + m_Storage->m_Filtered.Capacity() +
           m_Storage->m_Png.Capacity() + m_Storage->m_Compression.Bytes();
}

cudaError_t
GpuPngEncoder::Encode(const Color16 *devicePixels,
                      uint32_t width,
                      uint32_t height,
                      size_t rowStrideBytes,
                      cudaStream_t stream,
                      std::vector<unsigned char> &pngBytes)
{
    pngBytes.clear();
    size_t pixelCount = 0;
    size_t minimumStride = 0;
    size_t inputSpan = 0;
    size_t inputEnd = 0;
    if (devicePixels == nullptr || width == 0 || height == 0 || width > 0x7fffffffu ||
        height > 0x7fffffffu || !MultiplySize(width, height, pixelCount) || !GridFits(pixelCount, 256) ||
        !MultiplySize(width, sizeof(Color16), minimumStride) || rowStrideBytes < minimumStride ||
        rowStrideBytes % alignof(Color16) != 0 ||
        reinterpret_cast<uintptr_t>(devicePixels) % alignof(Color16) != 0 ||
        !MultiplySize(height - 1, rowStrideBytes, inputSpan) ||
        !AddSize(inputSpan, minimumStride, inputSpan) ||
        !AddSize(reinterpret_cast<uintptr_t>(devicePixels), inputSpan, inputEnd)) {
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
    PendingStreamWork pending{stream};
    error = m_Storage->m_Alpha.Reserve(sizeof(unsigned));
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaMemsetAsync(m_Storage->m_Alpha.Data<unsigned>(), 0, sizeof(unsigned), stream);
    if (error != cudaSuccess) {
        return error;
    }
    DetectAlpha<<<GridSize(pixelCount, 256), 256, 0, stream>>>(
        devicePixels, width, height, rowStrideBytes, m_Storage->m_Alpha.Data<unsigned>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    unsigned hasAlpha = 0;
    error = cudaMemcpyAsync(&hasAlpha,
                            m_Storage->m_Alpha.Data<unsigned>(),
                            sizeof(hasAlpha),
                            cudaMemcpyDeviceToHost,
                            stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        return error;
    }
    const unsigned channels = hasAlpha != 0 ? 4 : 3;
    size_t rowBytes = 0;
    size_t rawBytes = 0;
    size_t filteredBytes = 0;
    if (!MultiplySize(width, channels * 2, rowBytes) || !MultiplySize(rowBytes, height, rawBytes) ||
        !AddSize(rowBytes, 1, filteredBytes) || !MultiplySize(filteredBytes, height, filteredBytes) ||
        !GridFits(filteredBytes, 256)) {
        return cudaErrorInvalidValue;
    }
    error = m_Storage->m_Raw.Reserve(rawBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Storage->m_Filters.Reserve(height);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Storage->m_Filtered.Reserve(filteredBytes);
    if (error != cudaSuccess) {
        return error;
    }
    SerializePixels<<<GridSize(pixelCount, 256), 256, 0, stream>>>(
        devicePixels, width, height, rowStrideBytes, channels, m_Storage->m_Raw.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    SelectFilters<<<GridSize(height, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        m_Storage->m_Raw.Data<unsigned char>(),
        rowBytes,
        height,
        channels * 2,
        m_Storage->m_Filters.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    error = Detail::LaunchFilterScanlines(m_Storage->m_Raw.Data<unsigned char>(),
                                          width,
                                          height,
                                          channels,
                                          m_Storage->m_Filters.Data<unsigned char>(),
                                          m_Storage->m_Filtered.Data<unsigned char>(),
                                          stream);
    if (error != cudaSuccess) {
        return error;
    }
    size_t zlibBytes = 0;
    error = Compress(m_Storage->m_Compression,
                     m_Storage->m_Filtered.Data<unsigned char>(),
                     filteredBytes,
                     stream,
                     zlibBytes,
                     true);
    if (error != cudaSuccess) {
        return error;
    }
    const size_t chunks = (zlibBytes - 1) / Detail::IdatPayloadBytes + 1;
    size_t pngSize = 0;
    if (!GridFits(zlibBytes, 256) || !MultiplySize(chunks, 12, pngSize) ||
        !AddSize(pngSize, zlibBytes, pngSize) || !AddSize(pngSize, 45, pngSize)) {
        return cudaErrorInvalidValue;
    }
    error = m_Storage->m_Png.Reserve(pngSize);
    if (error != cudaSuccess) {
        return error;
    }
    auto *png = m_Storage->m_Png.Data<unsigned char>();
    WritePngHeaders<<<1, 1, 0, stream>>>(width, height, channels, zlibBytes, chunks, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    CopyIdat<<<GridSize(zlibBytes, 256), 256, 0, stream>>>(
        m_Storage->m_Compression.m_Zlib.Data<unsigned char>(), zlibBytes, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    WriteIdatHeaders<<<GridSize(chunks, 128), 128, 0, stream>>>(zlibBytes, chunks, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    WriteIdatCrcs<<<GridSize(chunks, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        zlibBytes, chunks, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    pngBytes.resize(pngSize);
    error = cudaMemcpyAsync(pngBytes.data(), png, pngSize, cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        pngBytes.clear();
    } else {
        pending.Complete();
    }
    return error;
}

namespace Detail {

cudaError_t
LaunchFilterScanlines(const unsigned char *deviceRaw,
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
        height == 0 || (channels != 3 && channels != 4) ||
        !MultiplySize(width, channels * 2, rowBytes) || !AddSize(rowBytes, 1, bytes) ||
        !MultiplySize(bytes, height, bytes) || !GridFits(bytes, 256)) {
        return cudaErrorInvalidValue;
    }
    FilterScanlines<<<GridSize(bytes, 256), 256, 0, stream>>>(
        deviceRaw, rowBytes, height, channels * 2, deviceFilters, deviceFiltered);
    return cudaGetLastError();
}

namespace {
cudaError_t
EncodeZlibWithPolicy(const unsigned char *deviceInput,
                     size_t inputBytes,
                     cudaStream_t stream,
                     std::vector<unsigned char> &zlibBytes,
                     bool allowDynamic)
{
    zlibBytes.clear();
    CompressionStorage storage;
    PendingStreamWork pending{stream};
    size_t bytes = 0;
    auto error = Compress(storage, deviceInput, inputBytes, stream, bytes, allowDynamic);
    if (error != cudaSuccess) {
        return error;
    }
    zlibBytes.resize(bytes);
    error = cudaMemcpyAsync(
        zlibBytes.data(), storage.m_Zlib.Data<unsigned char>(), bytes, cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        zlibBytes.clear();
    } else {
        pending.Complete();
    }
    return error;
}
} // namespace

cudaError_t
EncodeZlib(const unsigned char *deviceInput,
           size_t inputBytes,
           cudaStream_t stream,
           std::vector<unsigned char> &zlibBytes)
{
    return EncodeZlibWithPolicy(deviceInput, inputBytes, stream, zlibBytes, true);
}

cudaError_t
EncodeZlibFixed(const unsigned char *deviceInput,
                size_t inputBytes,
                cudaStream_t stream,
                std::vector<unsigned char> &zlibBytes)
{
    return EncodeZlibWithPolicy(deviceInput, inputBytes, stream, zlibBytes, false);
}

cudaError_t
BuildHuffmanCodeLengths(const unsigned *deviceFrequencies,
                        unsigned symbolCount,
                        unsigned maximumBits,
                        cudaStream_t stream,
                        std::vector<unsigned> &lengths)
{
    lengths.clear();
    if (deviceFrequencies == nullptr || symbolCount < 2 || symbolCount > Huffman::LiteralSymbols ||
        maximumBits == 0 || maximumBits > Huffman::MaximumBits || symbolCount > (1u << maximumBits)) {
        return cudaErrorInvalidValue;
    }
    DeviceBuffer workspace;
    DeviceBuffer output;
    PendingStreamWork pending{stream};
    auto error = workspace.Reserve(sizeof(Huffman::Workspace));
    if (error != cudaSuccess) {
        return error;
    }
    error = output.Reserve(symbolCount * sizeof(unsigned));
    if (error != cudaSuccess) {
        return error;
    }
    BuildLengthsFixture<<<1, 1, 0, stream>>>(deviceFrequencies,
                                             symbolCount,
                                             maximumBits,
                                             workspace.Data<Huffman::Workspace>(),
                                             output.Data<unsigned>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    lengths.resize(symbolCount);
    error = cudaMemcpyAsync(lengths.data(),
                            output.Data<unsigned>(),
                            symbolCount * sizeof(unsigned),
                            cudaMemcpyDeviceToHost,
                            stream);
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error == cudaSuccess) {
        pending.Complete();
    } else {
        lengths.clear();
    }
    return error;
}

cudaError_t
EncodeCodeLengthRuns(const unsigned *deviceLengths,
                     unsigned lengthCount,
                     cudaStream_t stream,
                     std::vector<CodeLengthRun> &runs)
{
    runs.clear();
    if (deviceLengths == nullptr || lengthCount == 0 || lengthCount > Huffman::TableSymbols) {
        return cudaErrorInvalidValue;
    }
    DeviceBuffer output;
    DeviceBuffer count;
    PendingStreamWork pending{stream};
    auto error = output.Reserve(lengthCount * sizeof(CodeLengthRun));
    if (error != cudaSuccess) {
        return error;
    }
    error = count.Reserve(sizeof(unsigned));
    if (error != cudaSuccess) {
        return error;
    }
    EncodeRunsFixture<<<1, 1, 0, stream>>>(
        deviceLengths, lengthCount, output.Data<CodeLengthRun>(), count.Data<unsigned>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    unsigned actual = 0;
    error =
        cudaMemcpyAsync(&actual, count.Data<unsigned>(), sizeof(actual), cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error != cudaSuccess) {
        return error;
    }
    if (actual == 0xffffffffu) {
        return cudaErrorInvalidValue;
    }
    runs.resize(actual);
    error = cudaMemcpyAsync(runs.data(),
                            output.Data<CodeLengthRun>(),
                            actual * sizeof(CodeLengthRun),
                            cudaMemcpyDeviceToHost,
                            stream);
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(stream);
    }
    if (error == cudaSuccess) {
        pending.Complete();
    } else {
        runs.clear();
    }
    return error;
}

} // namespace Detail
} // namespace FractalShark::Png
