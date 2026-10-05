#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace FractalShark::Png::Detail {

// Encoder policy: regions reset the LZ77 dictionary to permit independent warp compression.
// IDAT payloads split one continuous zlib stream; they do not reset that dictionary.
inline constexpr size_t DeflateRegionBytes = 32768;
inline constexpr size_t IdatPayloadBytes = 65536;

namespace Format {

// PNG Third Edition, sections 5.2-5.5 and 11.2.1: signature, chunk framing, and IHDR.
// https://www.w3.org/TR/png-3/
inline constexpr unsigned ByteBits = 8;
inline constexpr unsigned WordBits = 32;
inline constexpr size_t WordBytes = WordBits / ByteBits;
inline constexpr unsigned SampleBits = 16;
inline constexpr unsigned SampleBytes = SampleBits / ByteBits;
inline constexpr unsigned MaximumSample = (1u << SampleBits) - 1;
inline constexpr unsigned RgbChannels = 3;
inline constexpr unsigned RgbaChannels = 4;
inline constexpr unsigned RgbColorType = 2;
inline constexpr unsigned RgbaColorType = 6;
inline constexpr uint32_t MaximumDimension = 0x7fffffffu;
inline constexpr size_t SignatureBytes = 8;
inline constexpr size_t ChunkLengthBytes = WordBytes;
inline constexpr size_t ChunkTypeBytes = 4;
inline constexpr size_t ChunkCrcBytes = 4;
inline constexpr size_t ChunkHeaderBytes = ChunkLengthBytes + ChunkTypeBytes;
inline constexpr size_t ChunkOverheadBytes = ChunkHeaderBytes + ChunkCrcBytes;
inline constexpr size_t IhdrPayloadBytes = 13;
inline constexpr size_t IhdrHeightOffset = WordBytes;
inline constexpr size_t IhdrBitDepthOffset = 2 * WordBytes;
inline constexpr size_t IhdrColorTypeOffset = IhdrBitDepthOffset + 1;
inline constexpr size_t IhdrCompressionOffset = IhdrColorTypeOffset + 1;
inline constexpr size_t IhdrFilterOffset = IhdrCompressionOffset + 1;
inline constexpr size_t IhdrInterlaceOffset = IhdrFilterOffset + 1;
inline constexpr size_t FirstIdatOffset = SignatureBytes + ChunkOverheadBytes + IhdrPayloadBytes;
inline constexpr size_t PngFixedBytes = FirstIdatOffset + ChunkOverheadBytes; // Includes empty IEND.
inline constexpr size_t FilterPrefixBytes = 1;

enum class Filter : unsigned char { None = 0, Sub = 1, Up = 2, Average = 3, Paeth = 4 };

// RFC 1951, sections 3.2.3-3.2.7: block framing and the three DEFLATE alphabets.
// https://www.rfc-editor.org/rfc/rfc1951.html
enum class BlockType : unsigned { Stored = 0, Fixed = 1, Dynamic = 2 };
inline constexpr unsigned FinalFlagBits = 1;
inline constexpr unsigned BlockTypeBits = 2;
inline constexpr unsigned BlockHeaderBits = FinalFlagBits + BlockTypeBits;
inline constexpr unsigned StoredLengthBits = 16;
inline constexpr unsigned StoredLengthMaximum = (1u << StoredLengthBits) - 1;
inline constexpr size_t StoredLengthBytes = StoredLengthBits / ByteBits;
inline constexpr size_t StoredLengthFieldBytes = 2 * StoredLengthBytes; // LEN and one's-complement NLEN.
inline constexpr size_t StoredBlockHeaderBytes = 1 + StoredLengthFieldBytes; // Byte-aligned header.
inline constexpr unsigned EndOfBlockSymbol = 256;
inline constexpr unsigned FirstLengthSymbol = EndOfBlockSymbol + 1;
inline constexpr unsigned MaximumLengthSymbol = 285;
inline constexpr unsigned MinimumMatchBytes = 3;
inline constexpr unsigned MaximumMatchBytes = 258;
inline constexpr unsigned LiteralLengthSymbols = MaximumLengthSymbol + 1;
inline constexpr unsigned DistanceSymbols = 30;
inline constexpr unsigned CodeLengthSymbols = 19;
inline constexpr unsigned MaximumCodeBits = 15;
inline constexpr unsigned MaximumHeaderCodeBits = 7;
inline constexpr unsigned MinimumLiteralCount = FirstLengthSymbol;
inline constexpr unsigned MinimumDistanceCount = 1;
inline constexpr unsigned MinimumHeaderCount = 4;
inline constexpr unsigned LiteralCountBits = 5;
inline constexpr unsigned DistanceCountBits = 5;
inline constexpr unsigned HeaderCountBits = 4;
inline constexpr unsigned HeaderLengthBits = 3;
inline constexpr unsigned DynamicHeaderPrefixBits =
    BlockHeaderBits + LiteralCountBits + DistanceCountBits + HeaderCountBits;
inline constexpr unsigned MinimumDynamicHeaderBits =
    DynamicHeaderPrefixBits + MinimumHeaderCount * HeaderLengthBits;

// A compressed region ends with an empty stored block: its header starts in the current
// bitstream, padding reaches the next byte, and LEN/NLEN occupy four more bytes.
__host__ __device__ inline constexpr uint64_t
CompressedRegionBytes(uint64_t blockBits)
{
    return (blockBits + BlockHeaderBits + ByteBits - 1) / ByteBits + StoredLengthFieldBytes;
}

// Fixed Huffman ranges, RFC 1951 section 3.2.6. These scalar constants avoid device-local tables.
inline constexpr unsigned FixedLowLiteralEnd = 143;
inline constexpr unsigned FixedLowLiteralCode = 0x30;
inline constexpr unsigned FixedLowLiteralBits = 8;
inline constexpr unsigned FixedHighLiteralBegin = FixedLowLiteralEnd + 1;
inline constexpr unsigned FixedHighLiteralEnd = EndOfBlockSymbol - 1;
inline constexpr unsigned FixedHighLiteralCode = 0x190;
inline constexpr unsigned FixedHighLiteralBits = 9;
inline constexpr unsigned FixedShortLengthEnd = 279;
inline constexpr unsigned FixedShortLengthBits = 7;
inline constexpr unsigned FixedLongLengthBegin = FixedShortLengthEnd + 1;
inline constexpr unsigned FixedLongLengthCode = 0xc0;
inline constexpr unsigned FixedLongLengthBits = 8;
inline constexpr unsigned FixedDistanceBits = 5;

// Extra-bit counts grow once per group of four length codes or two distance codes (section 3.2.5).
inline constexpr unsigned LengthCodesPerExtraBit = 4;
inline constexpr unsigned LengthCodesWithoutExtraBits = 8;
inline constexpr unsigned DistanceCodesPerExtraBit = 2;
inline constexpr unsigned DistanceCodesWithoutExtraBits = 4;

// Code-length repetition symbols, section 3.2.7; extra fields encode count minus the minimum.
inline constexpr unsigned RepeatPreviousSymbol = 16;
inline constexpr unsigned RepeatPreviousMinimum = 3;
inline constexpr unsigned RepeatPreviousMaximum = 6;
inline constexpr unsigned RepeatPreviousExtraBits = 2;
inline constexpr unsigned RepeatZeroShortSymbol = 17;
inline constexpr unsigned RepeatZeroShortMinimum = 3;
inline constexpr unsigned RepeatZeroShortMaximum = 10;
inline constexpr unsigned RepeatZeroShortExtraBits = 3;
inline constexpr unsigned RepeatZeroLongSymbol = 18;
inline constexpr unsigned RepeatZeroLongMinimum = 11;
inline constexpr unsigned RepeatZeroLongMaximum = 138;
inline constexpr unsigned RepeatZeroLongExtraBits = 7;

// RFC 1950 section 2.2: CMF selects DEFLATE with a 32 KiB window; FLG has no dictionary,
// fastest compression-level hint, and the check bits making CMF*256+FLG divisible by 31.
// https://www.rfc-editor.org/rfc/rfc1950.html
inline constexpr unsigned char ZlibCmf = 0x78;
inline constexpr unsigned char ZlibFlags = 0x01;
inline constexpr size_t ZlibHeaderBytes = 2;
inline constexpr size_t AdlerBytes = 4;
inline constexpr size_t ZlibOverheadBytes = ZlibHeaderBytes + StoredBlockHeaderBytes + AdlerBytes;
inline constexpr uint64_t AdlerModulus = 65521;

// PNG's reflected CRC-32 representation. Bit 31 is the multiplicative identity, and
// CrcBytePower is x^8, used to advance a prefix CRC by the suffix's byte count.
inline constexpr uint32_t CrcPolynomial = 0xedb88320u;
inline constexpr uint32_t CrcInitial = 0xffffffffu;
inline constexpr uint32_t CrcIdentity = 0x80000000u;
inline constexpr uint32_t CrcBytePower = 0x00800000u;

} // namespace Format

struct CodeLengthRun {
    unsigned m_Symbol;    // Literal code length or a repetition symbol from the header alphabet.
    unsigned m_Extra;     // Value written to the repetition's extra-bit field.
    unsigned m_ExtraBits; // Width of that field in bits, not bytes.
};

// Raw samples are already serialized in PNG byte order. One filter byte per row
// selects a standard PNG filter (0 through 4). Output includes scanline filter bytes.
// RGB16/RGBA16 rows are packed; output needs (width * channels * SampleBytes + 1) * height bytes.
// Launch-only: callers retain all device buffers until the supplied stream completes.
cudaError_t LaunchFilterScanlines(const unsigned char *deviceRaw,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t channels,
                                  const unsigned char *deviceFilters,
                                  unsigned char *deviceFiltered,
                                  cudaStream_t stream);

// The lower-level compression seam used by format-boundary tests. Produces one
// complete RFC 1950 zlib stream; production encoding uses the same implementation.
// Host-output seams synchronize the supplied stream before returning and clear output on CUDA errors.
cudaError_t EncodeZlib(const unsigned char *deviceInput,
                       size_t inputBytes,
                       cudaStream_t stream,
                       std::vector<unsigned char> &zlibBytes);

// Internal baseline and algorithm seams. These use the production device routines.
// Fixed coding still permits stored fallback. Huffman symbolCount is 2..LiteralLengthSymbols,
// maximumBits is 1..MaximumCodeBits, and symbolCount must fit within 2^maximumBits.
// Run encoding accepts 1..(LiteralLengthSymbols + DistanceSymbols) lengths, each 0..MaximumCodeBits.
cudaError_t EncodeZlibFixed(const unsigned char *deviceInput,
                            size_t inputBytes,
                            cudaStream_t stream,
                            std::vector<unsigned char> &zlibBytes);
cudaError_t BuildHuffmanCodeLengths(const unsigned *deviceFrequencies,
                                    unsigned symbolCount,
                                    unsigned maximumBits,
                                    cudaStream_t stream,
                                    std::vector<unsigned> &lengths);
cudaError_t EncodeCodeLengthRuns(const unsigned *deviceLengths,
                                 unsigned lengthCount,
                                 cudaStream_t stream,
                                 std::vector<CodeLengthRun> &runs);

} // namespace FractalShark::Png::Detail
