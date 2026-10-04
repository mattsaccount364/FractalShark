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
class GpuPngEncoder {
public:
    GpuPngEncoder();
    ~GpuPngEncoder();
    GpuPngEncoder(const GpuPngEncoder &) = delete;
    GpuPngEncoder &operator=(const GpuPngEncoder &) = delete;

    cudaError_t Encode(const Color16 *devicePixels,
                       uint32_t width,
                       uint32_t height,
                       size_t rowStrideBytes,
                       cudaStream_t stream,
                       std::vector<unsigned char> &pngBytes);

    size_t GetWorkspaceBytes() const;

private:
    class Storage;
    std::unique_ptr<Storage> m_Storage;
};

} // namespace FractalShark::Png
