#pragma once

#include "HighPrecision.h"
#include "GPU_Render.h"
#include "Vectors.h"

enum class IterTypeEnum {
    Bits32,
    Bits64
};

// Holds the number of iterations it took to decide if
// we were in or not in the fractal.  Has a number
// for every point on the screen.
struct ItersMemoryContainer {
    ItersMemoryContainer();
    ItersMemoryContainer(
        IterTypeEnum type,
        size_t width,
        size_t height,
        size_t total_antialiasing);
    ItersMemoryContainer(ItersMemoryContainer &&) noexcept;
    ItersMemoryContainer &operator=(ItersMemoryContainer &&) noexcept;
    ~ItersMemoryContainer();

    ItersMemoryContainer(ItersMemoryContainer &) = delete;
    ItersMemoryContainer &operator=(const ItersMemoryContainer &);

    template<typename IterType>
    IterType *GetIters() {
        if constexpr (std::is_same<IterType, uint32_t>::value) {
            return m_ItersMemory32.GetData();
        } else {
            return m_ItersMemory64.GetData();
        }
    }

    template<typename IterType>
    IterType **GetItersArray() {
        if constexpr (sizeof(IterType) == sizeof(uint32_t)) {
            return m_ItersArray32.data();
        } else {
            return m_ItersArray64.data();
        }
    }

    IterTypeFull GetItersArrayValSlow(size_t x, size_t y) const;
    void SetItersArrayValSlow(size_t x, size_t y, uint64_t val);

    void GetReductionResults(ReductionResults &results) const;

    // These include antialiasing, so 4x antialiasing implies each is ~2x screen dimension
    size_t m_Width;
    size_t m_Height;
    size_t m_Total;

    // These are the originally-input desired dimensions
    size_t m_OutputWidth;
    size_t m_OutputHeight;
    size_t m_OutputTotal;

    // These are a bit bigger than m_ScrnWidth / m_ScrnHeight, and increased
    // to account for AA.
    size_t m_RoundedWidth;
    size_t m_RoundedHeight;
    size_t m_RoundedTotal;

    // Also a bit bigger, but much closer to actual screen size.  These sizes
    // are independent of antialiasing.
    size_t m_RoundedOutputColorWidth;
    size_t m_RoundedOutputColorHeight;
    size_t m_RoundedOutputColorTotal;
    std::unique_ptr<Color16[]> m_RoundedOutputColorMemory;

    // Antialiasing for reference: 1 (1x), 2 (4x), 3 (9x), 4 (16x)
    size_t m_Antialiasing;

private:
    static std::wstring GetTempFilename();

    IterTypeEnum m_IterType;

    std::wstring m_Iters32Filename;
    GrowableVector<uint32_t> m_ItersMemory32;
    std::vector<uint32_t *> m_ItersArray32;

    std::wstring m_Iters64Filename;
    GrowableVector<uint64_t> m_ItersMemory64;
    std::vector<uint64_t *> m_ItersArray64;

};
