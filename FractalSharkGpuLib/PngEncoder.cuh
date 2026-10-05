#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct Color16;

namespace FractalShark::Png {

// A context belongs to the CUDA device on which its first operation runs. Calls on one
// context must be serialized. Input remains caller-owned until Encode returns.
// Separate contexts may encode concurrently on separate streams. Workspace grows on
// demand, is retained for reuse, and is released on its owning device at destruction.
class GpuPngEncoder {
public:
    GpuPngEncoder();
    ~GpuPngEncoder();
    GpuPngEncoder(const GpuPngEncoder &) = delete;
    GpuPngEncoder &operator=(const GpuPngEncoder &) = delete;

    // Input is native-endian, straight-alpha RGBA16 Color16 data on the current CUDA device.
    // rowStrideBytes may include padding, but must hold width pixels and preserve Color16 alignment.
    // Produces non-interlaced RGB16 for opaque input, otherwise RGBA16, without changing samples.
    // Work is ordered on stream; this call waits for completion before returning host PNG bytes.
    // CUDA failures return a cudaError_t and leave pngBytes empty, fencing previously queued work
    // before the caller can release input. C++ allocation exceptions propagate to the caller.
    cudaError_t Encode(const Color16 *devicePixels,
                       uint32_t width,
                       uint32_t height,
                       size_t rowStrideBytes,
                       cudaStream_t stream,
                       std::vector<unsigned char> &pngBytes);

    // Total reserved device capacity, including output and compression scratch, in bytes.
    size_t GetWorkspaceBytes() const;

private:
    class Storage;
    std::unique_ptr<Storage> m_Storage;
};

} // namespace FractalShark::Png
