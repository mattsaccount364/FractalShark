#include "stdafx.h"
#include "PngStages.cuh"

namespace Png::Detail {
namespace {
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

} // namespace

cudaError_t
AssemblePng(DeviceBuffer &storage,
            uint32_t width,
            uint32_t height,
            unsigned channels,
            const unsigned char *zlib,
            size_t zlibBytes,
            cudaStream_t stream,
            std::vector<unsigned char> &pngBytes)
{
    PendingStreamWork pending(stream);
    const size_t chunks = (zlibBytes - 1) / IdatPayloadBytes + 1;
    size_t pngSize = 0;
    // Fixed framing is signature+IHDR+IEND; every IDAT adds its own header and CRC.
    if (!GridFits(zlibBytes, LinearThreadsPerBlock) ||
        !MultiplySize(chunks, Format::ChunkOverheadBytes, pngSize) ||
        !AddSize(pngSize, zlibBytes, pngSize) || !AddSize(pngSize, Format::PngFixedBytes, pngSize)) {
        return cudaErrorInvalidValue;
    }
    auto error = storage.Reserve(pngSize);
    if (error != cudaSuccess) {
        return error;
    }
    auto *png = storage.Data<unsigned char>();
    WritePngHeaders<<<1, 1, 0, stream>>>(width, height, channels, zlibBytes, chunks, png);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
    CopyIdat<<<GridSize(zlibBytes, LinearThreadsPerBlock), LinearThreadsPerBlock, 0, stream>>>(
        zlib, zlibBytes, png);
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

} // namespace Png::Detail
