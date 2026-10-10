#include "stdafx.h"

#include "PngEncoderTestSupport.cuh"

#include "FractalSharkGpuLib/PngHuffman.cuh"
#include "FractalSharkGpuLib/PngStages.cuh"

#include <limits>

// Test-only CUDA adapters compiled directly into FractalSharkCudaTest. They expose shared
// production routines to controlled inputs and download intermediate results for assertions.
namespace PngTestSupport {
cudaError_t
LaunchFilterScanlines(const unsigned char *deviceRaw,
                      uint32_t width,
                      uint32_t height,
                      uint32_t channels,
                      const unsigned char *deviceFilters,
                      unsigned char *deviceFiltered,
                      cudaStream_t stream)
{
    return Png::Detail::FilterScanlinesAsync(
        deviceRaw, width, height, channels, deviceFilters, deviceFiltered, stream);
}

namespace {
constexpr unsigned InvalidRunCount = std::numeric_limits<unsigned>::max();

// Test-only fixture kernels make the shared production device routines callable with
// controlled inputs from the CUDA test cases.
__global__ void
BuildLengthsFixture(const unsigned *frequencies,
                    unsigned symbols,
                    unsigned maximumBits,
                    Png::Huffman::Workspace *workspace,
                    unsigned *lengths)
{
    Png::Huffman::BuildLengths(frequencies, symbols, maximumBits, workspace, lengths);
}

__global__ void
EncodeRunsFixture(const unsigned *lengths, unsigned count, Png::CodeLengthRun *runs, unsigned *runCount)
{
    for (unsigned index = 0; index < count; ++index) {
        if (lengths[index] > Png::Format::MaximumCodeBits) {
            *runCount = InvalidRunCount;
            return;
        }
    }
    *runCount = Png::Huffman::EncodeRuns(lengths, count, runs);
}

} // namespace

namespace {
cudaError_t
EncodeZlibWithPolicy(const unsigned char *deviceInput,
                     size_t inputBytes,
                     cudaStream_t stream,
                     std::vector<unsigned char> &zlibBytes,
                     bool allowDynamic)
{
    zlibBytes.clear();
    // This test-only download uses a temporary workspace. Production retains its compressor
    // workspace and passes the device zlib stream directly to PNG assembly.
    Png::Detail::CompressionStorage storage;
    Png::Detail::PendingStreamWork pending{stream};
    size_t bytes = 0;
    auto error = storage.Compress(deviceInput, inputBytes, stream, bytes, allowDynamic);
    if (error != cudaSuccess) {
        return error;
    }
    zlibBytes.resize(bytes);
    error = cudaMemcpyAsync(zlibBytes.data(), storage.Data(), bytes, cudaMemcpyDeviceToHost, stream);
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
    if (deviceFrequencies == nullptr || symbolCount < 2 || symbolCount > Png::Huffman::LiteralSymbols ||
        maximumBits == 0 || maximumBits > Png::Huffman::MaximumBits ||
        symbolCount > (1u << maximumBits)) {
        return cudaErrorInvalidValue;
    }
    Png::Detail::DeviceBuffer workspace;
    Png::Detail::DeviceBuffer output;
    Png::Detail::PendingStreamWork pending{stream};
    auto error = workspace.Reserve(sizeof(Png::Huffman::Workspace));
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
                                             workspace.Data<Png::Huffman::Workspace>(),
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
                     std::vector<Png::CodeLengthRun> &runs)
{
    runs.clear();
    if (deviceLengths == nullptr || lengthCount == 0 || lengthCount > Png::Huffman::TableSymbols) {
        return cudaErrorInvalidValue;
    }
    Png::Detail::DeviceBuffer output;
    Png::Detail::DeviceBuffer count;
    // Fence pending downloads before this stack-owned count leaves scope.
    unsigned actual = 0;
    Png::Detail::PendingStreamWork pending{stream};
    auto error = output.Reserve(lengthCount * sizeof(Png::CodeLengthRun));
    if (error != cudaSuccess) {
        return error;
    }
    error = count.Reserve(sizeof(unsigned));
    if (error != cudaSuccess) {
        return error;
    }
    EncodeRunsFixture<<<1, 1, 0, stream>>>(
        deviceLengths, lengthCount, output.Data<Png::CodeLengthRun>(), count.Data<unsigned>());
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }
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
                            output.Data<Png::CodeLengthRun>(),
                            actual * sizeof(Png::CodeLengthRun),
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

} // namespace PngTestSupport
