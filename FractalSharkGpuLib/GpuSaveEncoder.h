#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct Color16;

namespace Png {

// One save-pool lease owns this context at a time. Buffers grow for reuse; there is
// no render state, display stream, or reduction workspace. Calls return complete host bytes.
class GpuSaveEncoder {
public:
    GpuSaveEncoder();
    ~GpuSaveEncoder();
    GpuSaveEncoder(const GpuSaveEncoder &) = delete;
    GpuSaveEncoder &operator=(const GpuSaveEncoder &) = delete;

    template <typename IterType>
    uint32_t Encode(const IterType *iterations,
                    size_t strideElements,
                    uint32_t width,
                    uint32_t height,
                    uint32_t antialiasing,
                    IterType numIterations,
                    const Color16 *palette,
                    uint32_t paletteSize,
                    uint32_t auxDepth,
                    uint64_t rotation,
                    uint64_t maxIterations,
                    std::vector<unsigned char> &pngBytes);
    static const char *ErrorString(uint32_t result);
    size_t GetWorkspaceBytes() const;

private:
    class Storage;
    std::unique_ptr<Storage> m_Storage;
};

} // namespace Png
