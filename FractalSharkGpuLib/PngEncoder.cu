#include "stdafx.h"

#include "PngCompressionKernels.cuh"
#include "PngEncoder.cuh"
#include "PngHuffman.cuh"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "GPU_Types.h"

#include <algorithm>
#include <limits>

namespace Png {
namespace {

// Pipeline: detect alpha -> serialize 16-bit samples -> choose/apply row filters ->
// compress independent regions -> scan/compact -> zlib metadata -> PNG chunks/CRCs.
// Kernels share one stream. The host reads only alpha/size metadata and final PNG bytes.
constexpr unsigned WarpSize = 32;
constexpr unsigned FullWarpMask = 0xffffffffu;
// Launch policy, independent of the wire format: four warp jobs per block, wider
// blocks for per-byte work, and smaller blocks for compaction and chunk headers.
constexpr unsigned WarpsPerBlock = 4;
constexpr unsigned LinearThreadsPerBlock = 256;
constexpr unsigned CopyThreadsPerBlock = 128;
constexpr unsigned HashBits = 12;
constexpr unsigned HashEntries = 1u << HashBits;
constexpr unsigned HashShift = Format::WordBits - HashBits;
constexpr unsigned HashMultiplier = 2654435761u;
constexpr unsigned HashKeyBytes = Format::MinimumMatchBytes;
constexpr unsigned InvalidRunCount = std::numeric_limits<unsigned>::max();
// Fixed literals require at most nine bits per byte; even a shortest match costs no
// more than its literals. Add the fixed header, end-of-block code, and alignment block.
// Stored fallback and dynamic replacements never exceed this initial fixed candidate.
constexpr size_t RegionSlotBytes =
    Format::CompressedRegionBytes(DeflateRegionBytes * Format::FixedHighLiteralBits +
                                  Format::BlockHeaderBits + Format::FixedShortLengthBits);

static_assert(DeflateRegionBytes <= Format::StoredLengthMaximum);
static_assert(IdatPayloadBytes <= Format::MaximumDimension);

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
    // All lanes must participate; warp kernels only return on a whole-row/region condition.
    // Only lane zero receives the complete sum.
    for (unsigned offset = WarpSize / 2; offset != 0; offset /= 2) {
        value += __shfl_down_sync(FullWarpMask, value, offset);
    }
    return value;
}

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

// DEFLATE fields enter least-significant bit first. Canonical Huffman codes are
// reversed before Write; only lane zero owns and advances a region's writer.
class BitWriter {
public:
    __device__ explicit BitWriter(unsigned char *output) : m_Output{output} {}

    __device__ void
    Write(unsigned value, unsigned bitCount)
    {
        m_Bits |= static_cast<uint64_t>(value) << m_BitCount;
        m_BitCount += bitCount;
        while (m_BitCount >= Format::ByteBits) {
            m_Output[m_Bytes++] = static_cast<unsigned char>(m_Bits);
            m_Bits >>= Format::ByteBits;
            m_BitCount -= Format::ByteBits;
        }
    }

    __device__ void
    Align()
    {
        if (m_BitCount != 0) {
            Write(0, Format::ByteBits - m_BitCount);
        }
    }

    __device__ void
    BlockHeader(Format::BlockType type, bool final)
    {
        Write((static_cast<unsigned>(type) << Format::FinalFlagBits) | static_cast<unsigned>(final),
              Format::BlockHeaderBits);
    }

    __device__ void
    EmptyStoredBlock(bool final)
    {
        BlockHeader(Format::BlockType::Stored, final);
        Align();
        Write(0, Format::StoredLengthBits);
        Write(Format::StoredLengthMaximum, Format::StoredLengthBits);
    }

    __device__ void
    Symbol(unsigned symbol, const Huffman::Region *table)
    {
        if (table != nullptr) {
            Write(table->m_Codes[symbol], table->m_Lengths[symbol]);
            return;
        }
        unsigned code = 0;
        unsigned count = 0;
        if (symbol <= Format::FixedLowLiteralEnd) {
            code = Format::FixedLowLiteralCode + symbol;
            count = Format::FixedLowLiteralBits;
        } else if (symbol <= Format::FixedHighLiteralEnd) {
            code = Format::FixedHighLiteralCode + symbol - Format::FixedHighLiteralBegin;
            count = Format::FixedHighLiteralBits;
        } else if (symbol <= Format::FixedShortLengthEnd) {
            code = symbol - Format::EndOfBlockSymbol;
            count = Format::FixedShortLengthBits;
        } else {
            code = Format::FixedLongLengthCode + symbol - Format::FixedLongLengthBegin;
            count = Format::FixedLongLengthBits;
        }
        Write(__brev(code) >> (Format::WordBits - count), count);
    }

    __device__ void
    Match(unsigned length, unsigned distance, const Huffman::Region *table, Huffman::Region *histogram)
    {
        unsigned lengthSymbol = Format::MaximumLengthSymbol;
        unsigned lengthValue = 0;
        unsigned lengthBits = 0;
        // The maximum match has a dedicated code; other lengths advance through the
        // RFC's ranges arithmetically, without a device-local length/distance table.
        if (length == Format::MaximumMatchBytes) {
            lengthSymbol = Format::MaximumLengthSymbol;
        } else {
            unsigned base = Format::MinimumMatchBytes;
            for (unsigned symbol = Format::FirstLengthSymbol; symbol < Format::MaximumLengthSymbol;
                 ++symbol) {
                const unsigned lengthIndex = symbol - Format::FirstLengthSymbol;
                const unsigned extra = lengthIndex < Format::LengthCodesWithoutExtraBits
                                           ? 0
                                           : lengthIndex / Format::LengthCodesPerExtraBit - 1;
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
        for (unsigned code = 0; code < Format::DistanceSymbols; ++code) {
            const unsigned extra = code < Format::DistanceCodesWithoutExtraBits
                                       ? 0
                                       : code / Format::DistanceCodesPerExtraBit - 1;
            if (distance < base + (1u << extra)) {
                if (table == nullptr) {
                    Write(__brev(code) >> (Format::WordBits - Format::FixedDistanceBits),
                          Format::FixedDistanceBits);
                } else {
                    Write(table->m_Codes[Huffman::LiteralSymbols + code],
                          table->m_Lengths[Huffman::LiteralSymbols + code]);
                }
                Write(distance - base, extra);
                if (histogram != nullptr) {
                    ++histogram->m_Frequencies[Huffman::LiteralSymbols + code];
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
    unsigned m_BitCount{};
    size_t m_Bytes{};
};

__device__ unsigned
HashAt(const unsigned char *input, unsigned position)
{
    // Multiplicative hashing mixes each three-byte key; the high HashBits select one
    // bucket. Preserve the multiplier and wraparound arithmetic for identical parsing.
    const unsigned key = (static_cast<unsigned>(input[position]) << 16) |
                         (static_cast<unsigned>(input[position + 1]) << 8) | input[position + 2];
    return (key * HashMultiplier) >> HashShift;
}

__device__ unsigned
CompareMatch(
    const unsigned char *input, unsigned position, unsigned previous, unsigned maximum, unsigned lane)
{
    // Compare 32 bytes at a time. The first ballot bit identifies the earliest
    // differing byte; even lanes beyond maximum must participate in the ballot.
    unsigned matched = 0;
    while (matched < maximum) {
        const unsigned offset = matched + lane;
        const bool different = offset < maximum && input[position + offset] != input[previous + offset];
        const unsigned mismatches = __ballot_sync(FullWarpMask, different);
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
            const Huffman::Region *table,
            Huffman::Region *histogram)
{
    // Greedy parsing is warp-uniform. Heads store position+1 so zero denotes no match.
    // Each region owns its dictionary; matches may overlap but never reference another region.
    // Replay uses the same insert order, reproducing the baseline token stream and histogram.
    unsigned position = 0;
    while (position < bytes) {
        unsigned previous = 0;
        if (lane == 0 && position + HashKeyBytes <= bytes) {
            previous = heads[HashAt(source, position)];
        }
        previous = __shfl_sync(FullWarpMask, previous, 0);
        unsigned length = 0;
        if (previous != 0) {
            const unsigned maximum = bytes - position < Format::MaximumMatchBytes
                                         ? bytes - position
                                         : Format::MaximumMatchBytes;
            length = CompareMatch(source, position, previous - 1, maximum, lane);
        }
        const unsigned consumed = length >= Format::MinimumMatchBytes ? length : 1;
        if (lane == 0) {
            if (length >= Format::MinimumMatchBytes) {
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
            if (inserted + HashKeyBytes <= bytes) {
                atomicMax(heads + HashAt(source, inserted), inserted + 1);
            }
        }
        __syncwarp();
        position += consumed;
    }
}

__device__ void
FinishRegion(BitWriter &writer, const Huffman::Region *table)
{
    // Align each independent region so a prefix scan can concatenate byte-sized slots.
    // Regions remain non-final; WriteZlibMetadata supplies the stream's only final block.
    writer.Symbol(Format::EndOfBlockSymbol, table);
    writer.EmptyStoredBlock(false);
}

__device__ unsigned
RegionByteCount(size_t inputBytes, size_t regionStart)
{
    const size_t remaining = inputBytes - regionStart;
    return static_cast<unsigned>(remaining < DeflateRegionBytes ? remaining : DeflateRegionBytes);
}

// A byte-aligned stored block uses a padded header byte and little-endian LEN/NLEN.
// This layout serves both region fallback and the final empty block in the zlib stream.
__device__ void
WriteStoredBlockHeader(unsigned char *output, unsigned payloadBytes, bool final)
{
    output[0] = static_cast<unsigned char>(final);
    output[1] = static_cast<unsigned char>(payloadBytes);
    output[2] = static_cast<unsigned char>(payloadBytes >> 8);
    output[3] = static_cast<unsigned char>(~payloadBytes);
    output[4] = static_cast<unsigned char>((~payloadBytes) >> 8);
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
                Huffman::Region *huffman)
{
    const size_t region = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (region >= regions) {
        return;
    }
    const size_t start = region * DeflateRegionBytes;
    const unsigned bytes = RegionByteCount(inputBytes, start);
    const unsigned char *source = input + start;
    unsigned *heads = hashHeads + region * HashEntries;
    unsigned char *slot = slots + region * RegionSlotBytes;
    Huffman::Region *histogram = huffman == nullptr ? nullptr : huffman + region;
    for (unsigned entry = lane; entry < HashEntries; entry += WarpSize) {
        heads[entry] = 0;
    }
    if (histogram != nullptr) {
        for (unsigned symbol = lane; symbol < Huffman::TableSymbols; symbol += WarpSize) {
            histogram->m_Frequencies[symbol] = 0;
        }
        __syncwarp();
        if (lane == 0) {
            histogram->m_Frequencies[Format::EndOfBlockSymbol] = 1;
            histogram->m_ExtraBits = 0;
        }
    }
    // Regional Adler components omit the initial one. Metadata later adds the global
    // initial state and shifts each regional weighted sum by the bytes following it.
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
        writer.BlockHeader(Format::BlockType::Fixed, false);
    }
    ParseRegion(source, bytes, heads, lane, writer, nullptr, histogram);
    if (lane == 0) {
        FinishRegion(writer, nullptr);
        sizes[region] = writer.Bytes();
    }
    __syncwarp();
    // Stored wins ties. The fixed pass still records frequencies for possible dynamic coding.
    if (sizes[region] >= bytes + Format::StoredBlockHeaderBytes) {
        if (lane == 0) {
            WriteStoredBlockHeader(slot, bytes, false);
            sizes[region] = bytes + Format::StoredBlockHeaderBytes;
        }
        for (unsigned byte = lane; byte < bytes; byte += WarpSize) {
            slot[byte + Format::StoredBlockHeaderBytes] = source[byte];
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
                Huffman::Region *huffman)
{
    const size_t region = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + threadIdx.x / WarpSize;
    const unsigned lane = threadIdx.x % WarpSize;
    if (region >= regions) {
        return;
    }
    Huffman::Region *table = huffman + region;
    // Table construction is serial in lane zero. Publish its decision to the whole
    // warp before skipping or replaying, so every collective has all lanes present.
    if (lane == 0) {
        Huffman::BuildRegion(table, sizes[region]);
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
    const size_t start = region * DeflateRegionBytes;
    const unsigned bytes = RegionByteCount(inputBytes, start);
    BitWriter writer{slots + region * RegionSlotBytes};
    if (lane == 0) {
        Huffman::WriteHeader(writer, table);
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
                    Huffman::Workspace *workspace,
                    unsigned *lengths)
{
    Huffman::BuildLengths(frequencies, symbols, maximumBits, workspace, lengths);
}

__global__ void
EncodeRunsFixture(const unsigned *lengths, unsigned count, CodeLengthRun *runs, unsigned *runCount)
{
    for (unsigned index = 0; index < count; ++index) {
        if (lengths[index] > Format::MaximumCodeBits) {
            *runCount = InvalidRunCount;
            return;
        }
    }
    *runCount = Huffman::EncodeRuns(lengths, count, runs);
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
    // s2 weights each byte by its remaining distance to the end of the entire input.
    // Shift the regional weighted sum by the suffix length, then add the initial s1=1.
    uint64_t sum = 0;
    uint64_t weighted = 0;
    for (size_t region = threadIdx.x; region < regions; region += WarpSize) {
        const size_t end = (region + 1) * DeflateRegionBytes < inputBytes
                               ? (region + 1) * DeflateRegionBytes
                               : inputBytes;
        sum = (sum + adlerSums[region]) % Format::AdlerModulus;
        weighted =
            (weighted +
             ((inputBytes - end) % Format::AdlerModulus) * (adlerSums[region] % Format::AdlerModulus) +
             adlerWeighted[region]) %
            Format::AdlerModulus;
    }
    sum = WarpSum(sum);
    weighted = WarpSum(weighted);
    if (threadIdx.x == 0) {
        const size_t end = Format::ZlibHeaderBytes + offsets[regions - 1] + sizes[regions - 1];
        zlib[0] = Format::ZlibCmf;
        zlib[1] = Format::ZlibFlags;
        WriteStoredBlockHeader(zlib + end, 0, true); // The only final block in the stream.
        const uint32_t adler = static_cast<uint32_t>(
            (((inputBytes % Format::AdlerModulus + weighted) % Format::AdlerModulus)
             << (Format::WordBits / 2)) |
            ((1 + sum) % Format::AdlerModulus));
        WriteBigEndian(zlib + end + Format::StoredBlockHeaderBytes, adler);
        *resultBytes = end + Format::StoredBlockHeaderBytes + Format::AdlerBytes;
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
            zlib[Format::ZlibHeaderBytes + offsets[region] + byte] =
                slots[region * RegionSlotBytes + byte];
        }
    }
}

__device__ uint32_t
CrcBytes(const unsigned char *input, size_t bytes)
{
    uint32_t crc = Format::CrcInitial;
    for (size_t byte = 0; byte < bytes; ++byte) {
        crc ^= input[byte];
        for (unsigned bit = 0; bit < Format::ByteBits; ++bit) {
            crc = (crc >> 1) ^ (Format::CrcPolynomial & (0u - (crc & 1u)));
        }
    }
    return crc ^ Format::CrcInitial;
}

// Reflected CRC polynomial multiplication, with bit 31 representing the identity.
// Scalar polynomial arithmetic avoids local matrix arrays in CUDA code.
__device__ uint32_t
CrcMultiply(uint32_t first, uint32_t second)
{
    uint32_t product = 0;
    for (uint32_t mask = Format::CrcIdentity; mask != 0; mask >>= 1) {
        if ((first & mask) != 0) {
            product ^= second;
        }
        second = (second >> 1) ^ (Format::CrcPolynomial & (0u - (second & 1u)));
    }
    return product;
}

__device__ uint32_t
CrcCombine(uint32_t first, uint32_t second, size_t secondBytes)
{
    // Advance the prefix by secondBytes using exponentiation by squaring, then XOR
    // the suffix CRC. The byte count is essential: CRC concatenation is order-sensitive.
    uint32_t factor = Format::CrcIdentity;
    uint32_t power = Format::CrcBytePower;
    while (secondBytes != 0) {
        if ((secondBytes & 1) != 0) {
            factor = CrcMultiply(power, factor);
        }
        secondBytes >>= 1;
        power = CrcMultiply(power, power);
    }
    return CrcMultiply(factor, first) ^ second;
}

__device__ void
WriteChunkType(unsigned char *header, char first, char second, char third, char fourth)
{
    unsigned char *type = header + Format::ChunkLengthBytes;
    type[0] = first;
    type[1] = second;
    type[2] = third;
    type[3] = fourth;
}

__device__ void
WriteChunkCrc(unsigned char *header, size_t payloadBytes)
{
    // PNG CRC covers the chunk type followed by its payload, excluding length and CRC.
    WriteBigEndian(header + Format::ChunkHeaderBytes + payloadBytes,
                   CrcBytes(header + Format::ChunkLengthBytes, Format::ChunkTypeBytes + payloadBytes));
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
    // Mandated eight-byte signature: high-bit marker, "PNG", CR/LF, DOS EOF, LF.
    // Scalar writes avoid introducing a local array into the CUDA kernel.
    png[0] = 0x89;
    png[1] = 'P';
    png[2] = 'N';
    png[3] = 'G';
    png[4] = '\r';
    png[5] = '\n';
    png[6] = 0x1a;
    png[7] = '\n';
    unsigned char *ihdr = png + Format::SignatureBytes;
    WriteBigEndian(ihdr, Format::IhdrPayloadBytes);
    WriteChunkType(ihdr, 'I', 'H', 'D', 'R');
    unsigned char *data = ihdr + Format::ChunkHeaderBytes;
    WriteBigEndian(data, width);
    WriteBigEndian(data + Format::IhdrHeightOffset, height);
    data[Format::IhdrBitDepthOffset] = Format::SampleBits;
    data[Format::IhdrColorTypeOffset] =
        channels == Format::RgbChannels ? Format::RgbColorType : Format::RgbaColorType;
    data[Format::IhdrCompressionOffset] = 0; // DEFLATE.
    data[Format::IhdrFilterOffset] = 0;      // Standard adaptive filters.
    data[Format::IhdrInterlaceOffset] = 0;   // Non-interlaced rows.
    WriteChunkCrc(ihdr, Format::IhdrPayloadBytes);
    const size_t end = Format::FirstIdatOffset + zlibBytes + chunks * Format::ChunkOverheadBytes;
    unsigned char *iend = png + end;
    WriteBigEndian(iend, 0);
    WriteChunkType(iend, 'I', 'E', 'N', 'D');
    WriteChunkCrc(iend, 0);
}

__device__ size_t
IdatChunkOffset(size_t chunk)
{
    return Format::FirstIdatOffset + chunk * (IdatPayloadBytes + Format::ChunkOverheadBytes);
}

__device__ size_t
IdatPayloadByteCount(size_t zlibBytes, size_t chunk)
{
    const size_t remaining = zlibBytes - chunk * IdatPayloadBytes;
    return remaining < IdatPayloadBytes ? remaining : IdatPayloadBytes;
}

__global__ void
CopyIdat(const unsigned char *zlib, size_t bytes, unsigned char *png)
{
    const size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (byte < bytes) {
        const size_t chunk = byte / IdatPayloadBytes;
        png[IdatChunkOffset(chunk) + Format::ChunkHeaderBytes + byte % IdatPayloadBytes] = zlib[byte];
    }
}

__global__ void
WriteIdatHeaders(size_t bytes, size_t chunks, unsigned char *png)
{
    const size_t chunk = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (chunk < chunks) {
        const size_t payload = IdatPayloadByteCount(bytes, chunk);
        unsigned char *header = png + IdatChunkOffset(chunk);
        WriteBigEndian(header, static_cast<uint32_t>(payload));
        WriteChunkType(header, 'I', 'D', 'A', 'T');
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
    const size_t payload = IdatPayloadByteCount(bytes, chunk);
    unsigned char *header = png + IdatChunkOffset(chunk);
    const size_t crcInputBytes = payload + Format::ChunkTypeBytes;
    // Each lane checks a contiguous segment of type+payload. Combine adjacent segments
    // in wire order, propagating their lengths; empty segments have CRC zero.
    const size_t partBytes = (crcInputBytes + WarpSize - 1) / WarpSize;
    const size_t begin = lane * partBytes;
    size_t length = begin < crcInputBytes ? crcInputBytes - begin : 0;
    length = length < partBytes ? length : partBytes;
    uint32_t crc = length != 0 ? CrcBytes(header + Format::ChunkLengthBytes + begin, length) : 0;
    for (unsigned stride = 1; stride < WarpSize; stride *= 2) {
        const uint32_t next = __shfl_down_sync(FullWarpMask, crc, stride);
        const size_t nextLength = __shfl_down_sync(FullWarpMask, length, stride);
        if (lane % (2 * stride) == 0) {
            crc = CrcCombine(crc, next, nextLength);
            length += nextLength;
        }
    }
    if (lane == 0) {
        WriteBigEndian(header + Format::ChunkHeaderBytes + payload, crc);
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
    const size_t regions = (bytes - 1) / DeflateRegionBytes + 1;
    size_t headsBytes = 0;
    size_t slotsBytes = 0;
    size_t countsBytes = 0;
    size_t zlibBound = 0;
    size_t huffmanBytes = 0;
    // Stored fallback caps every final slot at its raw bytes plus a stored header.
    // Add one stream header, one final empty stored block, and the Adler-32 trailer.
    if (regions > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !MultiplySize(regions, HashEntries * sizeof(unsigned), headsBytes) ||
        !MultiplySize(regions, RegionSlotBytes, slotsBytes) ||
        !MultiplySize(regions, sizeof(uint64_t), countsBytes) ||
        !MultiplySize(regions, Format::StoredBlockHeaderBytes, zlibBound) ||
        !AddSize(zlibBound, bytes, zlibBound) ||
        !AddSize(zlibBound, Format::ZlibOverheadBytes, zlibBound) ||
        !MultiplySize(regions, allowDynamic ? sizeof(Huffman::Region) : 0, huffmanBytes)) {
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
    // CUB's query reports scratch capacity without launching work. Once candidate sizes
    // are final, the exclusive scan maps independently compressed slots to wire offsets.
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
        allowDynamic ? storage.m_Huffman.Data<Huffman::Region>() : nullptr);
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
            storage.m_Huffman.Data<Huffman::Region>());
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
    CompactRegions<<<static_cast<unsigned>(regions), CopyThreadsPerBlock, 0, stream>>>(
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
    // Read only the compacted byte count here; the complete stream remains on the device
    // for PNG framing. The wait also makes this stack-owned metadata safe on return.
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
    // Validate byte-span arithmetic as well as dimensions before dereferencing pitched input.
    if (devicePixels == nullptr || width == 0 || height == 0 || width > Format::MaximumDimension ||
        height > Format::MaximumDimension || !MultiplySize(width, height, pixelCount) ||
        !GridFits(pixelCount, LinearThreadsPerBlock) ||
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
    DetectAlpha<<<GridSize(pixelCount, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
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
    // This small metadata wait chooses the packed format before reserving row buffers.
    const unsigned channels = hasAlpha != 0 ? Format::RgbaChannels : Format::RgbChannels;
    size_t rowBytes = 0;
    size_t rawBytes = 0;
    size_t filteredBytes = 0;
    if (!MultiplySize(width, channels * Format::SampleBytes, rowBytes) ||
        !MultiplySize(rowBytes, height, rawBytes) ||
        !AddSize(rowBytes, Format::FilterPrefixBytes, filteredBytes) ||
        !MultiplySize(filteredBytes, height, filteredBytes) ||
        !GridFits(filteredBytes, LinearThreadsPerBlock)) {
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
    SerializePixels<<<GridSize(pixelCount, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
        devicePixels, width, height, rowStrideBytes, channels, m_Storage->m_Raw.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    SelectFilters<<<GridSize(height, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        m_Storage->m_Raw.Data<unsigned char>(),
        rowBytes,
        height,
        channels * Format::SampleBytes,
        m_Storage->m_Filters.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    error = LaunchFilterScanlines(m_Storage->m_Raw.Data<unsigned char>(),
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
    const size_t chunks = (zlibBytes - 1) / IdatPayloadBytes + 1;
    size_t pngSize = 0;
    // Fixed framing is signature+IHDR+IEND; every IDAT adds its own header and CRC.
    if (!GridFits(zlibBytes, LinearThreadsPerBlock) ||
        !MultiplySize(chunks, Format::ChunkOverheadBytes, pngSize) ||
        !AddSize(pngSize, zlibBytes, pngSize) || !AddSize(pngSize, Format::PngFixedBytes, pngSize)) {
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
    CopyIdat<<<GridSize(zlibBytes, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
        m_Storage->m_Compression.m_Zlib.Data<unsigned char>(), zlibBytes, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    WriteIdatHeaders<<<GridSize(chunks, CopyThreadsPerBlock), CopyThreadsPerBlock, 0, stream>>>(
        zlibBytes, chunks, png);
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
        height == 0 || (channels != Format::RgbChannels && channels != Format::RgbaChannels) ||
        !MultiplySize(width, channels * Format::SampleBytes, rowBytes) ||
        !AddSize(rowBytes, Format::FilterPrefixBytes, bytes) || !MultiplySize(bytes, height, bytes) ||
        !GridFits(bytes, LinearThreadsPerBlock)) {
        return cudaErrorInvalidValue;
    }
    FilterScanlines<<<GridSize(bytes, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
        deviceRaw, rowBytes, height, channels * Format::SampleBytes, deviceFilters, deviceFiltered);
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
    if (actual == InvalidRunCount) {
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

} // namespace Png
