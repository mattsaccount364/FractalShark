#include "stdafx.h"
#include "PngHuffman.cuh"
#include "PngStages.cuh"
#include <cub/device/device_scan.cuh>

namespace Png::Detail {
namespace {
constexpr unsigned HashBits = 12;
constexpr unsigned HashEntries = 1u << HashBits;
constexpr unsigned HashShift = Format::WordBits - HashBits;
constexpr unsigned HashMultiplier = 2654435761u;
constexpr unsigned HashKeyBytes = Format::MinimumMatchBytes;
// Fixed literals require at most nine bits per byte; even a shortest match costs no
// more than its literals. Add the fixed header, end-of-block code, and alignment block.
// Stored fallback and dynamic replacements never exceed this initial fixed candidate.
constexpr size_t RegionSlotBytes =
    Format::CompressedRegionBytes(DeflateRegionBytes * Format::FixedHighLiteralBits +
                                  Format::BlockHeaderBits + Format::FixedShortLengthBits);

static_assert(DeflateRegionBytes <= Format::StoredLengthMaximum);
static_assert(IdatPayloadBytes <= Format::MaximumDimension);

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

} // namespace

cudaError_t
CompressionStorage::Compress(const unsigned char *input,
                             size_t bytes,
                             cudaStream_t stream,
                             size_t &resultBytes,
                             bool allowDynamic)
{
    // Fence before stack-owned size metadata or retained storage can leave this stage.
    uint64_t actualBytes = 0;
    PendingStreamWork pending(stream);
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
    auto error = m_Heads.Reserve(headsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Huffman.Reserve(huffmanBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Slots.Reserve(slotsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Sizes.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Offsets.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_AdlerSums.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_AdlerWeighted.Reserve(countsBytes);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Zlib.Reserve(zlibBound);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_ResultBytes.Reserve(sizeof(uint64_t));
    if (error != cudaSuccess) {
        return error;
    }
    size_t scanBytes = 0;
    // CUB's query reports scratch capacity without launching work. Once candidate sizes
    // are final, the exclusive scan maps independently compressed slots to wire offsets.
    error = cub::DeviceScan::ExclusiveSum(nullptr,
                                          scanBytes,
                                          m_Sizes.Data<uint64_t>(),
                                          m_Offsets.Data<uint64_t>(),
                                          static_cast<int>(regions),
                                          stream);
    if (error != cudaSuccess) {
        return error;
    }
    error = m_Scan.Reserve(scanBytes);
    if (error != cudaSuccess) {
        return error;
    }
    CompressRegions<<<GridSize(regions, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
        input,
        bytes,
        regions,
        m_Heads.Data<unsigned>(),
        m_Slots.Data<unsigned char>(),
        m_Sizes.Data<uint64_t>(),
        m_AdlerSums.Data<uint64_t>(),
        m_AdlerWeighted.Data<uint64_t>(),
        allowDynamic ? m_Huffman.Data<Huffman::Region>() : nullptr);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    if (allowDynamic) {
        OptimizeRegions<<<GridSize(regions, WarpsPerBlock), WarpsPerBlock * WarpSize, 0, stream>>>(
            input,
            bytes,
            regions,
            m_Heads.Data<unsigned>(),
            m_Slots.Data<unsigned char>(),
            m_Sizes.Data<uint64_t>(),
            m_Huffman.Data<Huffman::Region>());
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
    }
    error = cub::DeviceScan::ExclusiveSum(m_Scan.Data<void>(),
                                          scanBytes,
                                          m_Sizes.Data<uint64_t>(),
                                          m_Offsets.Data<uint64_t>(),
                                          static_cast<int>(regions),
                                          stream);
    if (error != cudaSuccess) {
        return error;
    }
    CompactRegions<<<static_cast<unsigned>(regions), CopyThreadsPerBlock, 0, stream>>>(
        m_Slots.Data<unsigned char>(),
        regions,
        m_Sizes.Data<uint64_t>(),
        m_Offsets.Data<uint64_t>(),
        m_Zlib.Data<unsigned char>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    WriteZlibMetadata<<<1, WarpSize, 0, stream>>>(bytes,
                                                  regions,
                                                  m_Sizes.Data<uint64_t>(),
                                                  m_Offsets.Data<uint64_t>(),
                                                  m_AdlerSums.Data<uint64_t>(),
                                                  m_AdlerWeighted.Data<uint64_t>(),
                                                  m_Zlib.Data<unsigned char>(),
                                                  m_ResultBytes.Data<uint64_t>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    // Read only the compacted byte count here; the complete stream remains on the device
    // for PNG framing. The wait also makes this stack-owned metadata safe on return.
    error = cudaMemcpyAsync(&actualBytes,
                            m_ResultBytes.Data<uint64_t>(),
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
    if (error == cudaSuccess) {
        pending.Complete();
    }
    return error;
}

} // namespace Png::Detail
