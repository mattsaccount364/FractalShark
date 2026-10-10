#pragma once

#include "FractalSharkGpuLib/PngFormat.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

// Test-only adapters declared for FractalSharkCudaTest. Their implementation is compiled
// directly into that executable; production PNG saves call the shared GPU routines directly.
namespace PngTestSupport {

// Test-only entry point for exercising Png::Detail::FilterScanlinesAsync with
// explicitly chosen filters. Raw samples are serialized in PNG byte order. One byte per row
// selects a standard PNG filter (0 through 4). Output includes scanline filter bytes.
// RGB16/RGBA16 rows are packed; output needs
// (width * channels * Png::Format::SampleBytes + 1) * height bytes.
// Launch-only: callers retain all device buffers until the supplied stream completes.
cudaError_t LaunchFilterScanlines(const unsigned char *deviceRaw,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t channels,
                                  const unsigned char *deviceFilters,
                                  unsigned char *deviceFiltered,
                                  cudaStream_t stream);

// Test-only adapter around Png::Detail::CompressionStorage::Compress, with dynamic coding
// enabled. Downloads a complete RFC 1950 zlib stream for checksum and DEFLATE boundary checks.
// Host-output adapters synchronize before returning and clear their output on CUDA errors.
cudaError_t EncodeZlib(const unsigned char *deviceInput,
                       size_t inputBytes,
                       cudaStream_t stream,
                       std::vector<unsigned char> &zlibBytes);

// Test-only compression baseline using the same production compressor with dynamic coding
// disabled. Fixed coding still permits stored fallback; tests compare its size and bitstream
// against the normal dynamic-enabled result.
cudaError_t EncodeZlibFixed(const unsigned char *deviceInput,
                            size_t inputBytes,
                            cudaStream_t stream,
                            std::vector<unsigned char> &zlibBytes);

// Test-only fixture for Png::Huffman::BuildLengths, used for CPU-reference comparisons.
// symbolCount is 2..Png::Format::LiteralLengthSymbols; maximumBits is
// 1..Png::Format::MaximumCodeBits, and symbolCount must fit within 2^maximumBits.
cudaError_t BuildHuffmanCodeLengths(const unsigned *deviceFrequencies,
                                    unsigned symbolCount,
                                    unsigned maximumBits,
                                    cudaStream_t stream,
                                    std::vector<unsigned> &lengths);

// Test-only fixture for Png::Huffman::EncodeRuns, exposing runs for boundary and
// reconstruction checks. Accepts 1..(Png::Format::LiteralLengthSymbols +
// Png::Format::DistanceSymbols) lengths, each 0..Png::Format::MaximumCodeBits.
cudaError_t EncodeCodeLengthRuns(const unsigned *deviceLengths,
                                 unsigned lengthCount,
                                 cudaStream_t stream,
                                 std::vector<Png::CodeLengthRun> &runs);

} // namespace PngTestSupport
