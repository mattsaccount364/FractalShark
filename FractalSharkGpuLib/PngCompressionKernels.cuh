#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace FractalShark::Png::Detail {

inline constexpr size_t DeflateRegionBytes = 32768;
inline constexpr size_t IdatPayloadBytes = 65536;

struct CodeLengthRun {
    unsigned m_Symbol;
    unsigned m_Extra;
    unsigned m_ExtraBits;
};

// Raw samples are already serialized in PNG byte order. One filter byte per row
// selects a standard PNG filter (0 through 4). Output includes scanline filter bytes.
cudaError_t LaunchFilterScanlines(const unsigned char *deviceRaw,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t channels,
                                  const unsigned char *deviceFilters,
                                  unsigned char *deviceFiltered,
                                  cudaStream_t stream);

// The lower-level compression seam used by format-boundary tests. Produces one
// complete RFC 1950 zlib stream; production encoding uses the same implementation.
cudaError_t EncodeZlib(const unsigned char *deviceInput,
                       size_t inputBytes,
                       cudaStream_t stream,
                       std::vector<unsigned char> &zlibBytes);

// Internal baseline and algorithm seams. These use the production device routines.
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
