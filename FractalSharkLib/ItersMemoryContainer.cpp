#include "stdafx.h"
#include "ItersMemoryContainer.h"
#include "Utilities.h"

#include <algorithm>
#include <string>

std::wstring ItersMemoryContainer::GetTempFilename(uint64_t numBits) {
    static std::atomic<uint64_t> counter{};
    std::wstring result;
    for(;;) {
        std::wstring optional_suffix = L" - " + std::to_wstring(counter) + L" - " + std::to_wstring(numBits) + L" bits";

        result = L"ItersMemoryContainer" + optional_suffix + GetFileExtension(GrowableVectorTypes::ItersMemoryContainer);
        if (!Utilities::FileExists(result.c_str())) {
            break;
        }

        counter++;
    }

    return result;
}

ItersMemoryContainer::ItersMemoryContainer() :
    m_IterType(IterTypeEnum::Bits32),
    m_Iters32Filename{},
    m_ItersMemory32{},
    m_ItersArray32{},
    m_Iters64Filename{},
    m_ItersMemory64{},
    m_ItersArray64{},
    m_Width(),
    m_Height(),
    m_Total(),
    m_OutputWidth(),
    m_OutputHeight(),
    m_OutputTotal(),
    m_RoundedWidth(),
    m_RoundedHeight(),
    m_RoundedTotal(),
    m_RoundedOutputColorWidth(),
    m_RoundedOutputColorHeight(),
    m_RoundedOutputColorTotal(),
    m_Antialiasing(1) {
}

ItersMemoryContainer::ItersMemoryContainer(
    IterTypeEnum type,
    size_t width,
    size_t height,
    size_t total_antialiasing)
    : m_IterType(type),
    m_Iters32Filename{},
    m_ItersMemory32{},
    m_ItersArray32{},
    m_Iters64Filename{},
    m_ItersMemory64{},
    m_ItersArray64{},
    m_Width(),
    m_Height(),
    m_Total(),
    m_OutputWidth(),
    m_OutputHeight(),
    m_OutputTotal(),
    m_RoundedWidth(),
    m_RoundedHeight(),
    m_RoundedTotal(),
    m_RoundedOutputColorWidth(),
    m_RoundedOutputColorHeight(),
    m_RoundedOutputColorTotal(),
    m_Antialiasing(total_antialiasing) {

    // This array must be identical in size to OutputIterMatrix in CUDA
    m_Width = width * total_antialiasing;
    m_Height = height * total_antialiasing;
    m_Total = m_Width * m_Height;

    m_OutputWidth = width;
    m_OutputHeight = height;
    m_OutputTotal = m_OutputWidth * m_OutputHeight;

    const size_t w_block = m_Width / GPURenderer::NB_THREADS_W +
        (m_Width % GPURenderer::NB_THREADS_W != 0);
    const size_t h_block = m_Height / GPURenderer::NB_THREADS_H +
        (m_Height % GPURenderer::NB_THREADS_H != 0);

    m_RoundedWidth = w_block * GPURenderer::NB_THREADS_W;
    m_RoundedHeight = h_block * GPURenderer::NB_THREADS_H;
    m_RoundedTotal = m_RoundedWidth * m_RoundedHeight;

    // Find the system page size
    {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        const size_t page_size = sys_info.dwPageSize;

        // Complement page_size - 1 to get the mask
        const size_t mask = ~(page_size - 1);

        // Round to the next page boundary
        m_RoundedTotal = (m_RoundedTotal + page_size - 1) & mask;
    }

    if (m_IterType == IterTypeEnum::Bits32) {
        m_Iters32Filename = GetTempFilename(32);
        m_ItersMemory32 = GrowableVector<uint32_t>{
            AddPointOptions::EnableWithoutSave,
            m_Iters32Filename
        };

        m_ItersMemory32.MutableResize(m_RoundedTotal);

        m_ItersArray32.resize(m_RoundedHeight);
        for (size_t i = 0; i < m_RoundedHeight; i++) {
            m_ItersArray32[i] = &m_ItersMemory32[i * m_RoundedWidth];
        }
    } else {
        m_Iters64Filename = GetTempFilename(64);
        m_ItersMemory64 = GrowableVector<uint64_t>{
            AddPointOptions::EnableWithoutSave,
            m_Iters64Filename
        };

        m_ItersMemory64.MutableResize(m_RoundedTotal);

        m_ItersArray64.resize(m_RoundedHeight);
        for (size_t i = 0; i < m_RoundedHeight; i++) {
            m_ItersArray64[i] = &m_ItersMemory64[i * m_RoundedWidth];
        }
    }

    const size_t w_color_block = m_OutputWidth / GPURenderer::NB_THREADS_W_AA +
        (m_OutputWidth % GPURenderer::NB_THREADS_W_AA != 0);
    const size_t h_color_block = m_OutputHeight / GPURenderer::NB_THREADS_H_AA +
        (m_OutputHeight % GPURenderer::NB_THREADS_H_AA != 0);
    m_RoundedOutputColorWidth = w_color_block * GPURenderer::NB_THREADS_W_AA;
    m_RoundedOutputColorHeight = h_color_block * GPURenderer::NB_THREADS_H_AA;
    m_RoundedOutputColorTotal = m_RoundedOutputColorWidth * m_RoundedOutputColorHeight;
    m_RoundedOutputColorMemory = std::make_unique<Color16[]>(m_RoundedOutputColorTotal);
};

ItersMemoryContainer::ItersMemoryContainer(ItersMemoryContainer &&other) noexcept 
    : m_IterType{ std::move(other.m_IterType) },
    m_Width{ std::move(other.m_Width) },
    m_Height{ std::move(other.m_Height) },
    m_Total{ std::move(other.m_Total) },
    m_OutputWidth{ std::move(other.m_OutputWidth) },
    m_OutputHeight{ std::move(other.m_OutputHeight) },
    m_OutputTotal{ std::move(other.m_OutputTotal) },
    m_RoundedWidth{ std::move(other.m_RoundedWidth) },
    m_RoundedHeight{ std::move(other.m_RoundedHeight) },
    m_RoundedTotal{ std::move(other.m_RoundedTotal) },
    m_RoundedOutputColorWidth{ std::move(other.m_RoundedOutputColorWidth) },
    m_RoundedOutputColorHeight{ std::move(other.m_RoundedOutputColorHeight) },
    m_RoundedOutputColorTotal{ std::move(other.m_RoundedOutputColorTotal) },
    m_RoundedOutputColorMemory{ std::move(other.m_RoundedOutputColorMemory) },
    m_Antialiasing{ std::move(other.m_Antialiasing) },
    m_ItersMemory32{ std::move(other.m_ItersMemory32) },
    m_ItersMemory64{ std::move(other.m_ItersMemory64) },
    m_ItersArray32{ std::move(other.m_ItersArray32) },
    m_ItersArray64{ std::move(other.m_ItersArray64) },
    m_Iters32Filename{ std::move(other.m_Iters32Filename) },
    m_Iters64Filename{ std::move(other.m_Iters64Filename) } {
}

ItersMemoryContainer &ItersMemoryContainer::operator=(const ItersMemoryContainer &other) {
    if (this == &other) {
        return *this;
    }

    m_IterType = other.m_IterType;

    m_Width = other.m_Width;
    m_Height = other.m_Height;
    m_Total = other.m_Total;

    m_OutputWidth = other.m_OutputWidth;
    m_OutputHeight = other.m_OutputHeight;
    m_OutputTotal = other.m_OutputTotal;

    m_RoundedWidth = other.m_RoundedWidth;
    m_RoundedHeight = other.m_RoundedHeight;
    m_RoundedTotal = other.m_RoundedTotal;

    m_RoundedOutputColorWidth = other.m_RoundedOutputColorWidth;
    m_RoundedOutputColorHeight = other.m_RoundedOutputColorHeight;
    m_RoundedOutputColorTotal = other.m_RoundedOutputColorTotal;
    m_RoundedOutputColorMemory = std::make_unique<Color16[]>(m_RoundedOutputColorTotal);
    memcpy(
        m_RoundedOutputColorMemory.get(),
        other.m_RoundedOutputColorMemory.get(),
        m_RoundedOutputColorTotal * sizeof(Color16));

    m_Antialiasing = other.m_Antialiasing;

    m_ItersMemory32 = {};
    m_ItersArray32.clear();
    m_Iters32Filename = L"";

    m_ItersMemory64 = {};
    m_ItersArray64.clear();
    m_Iters64Filename = L"";

    if (m_IterType == IterTypeEnum::Bits32) {
        m_Iters32Filename = GetTempFilename(32);
        m_ItersMemory32 = GrowableVector<uint32_t>{
            AddPointOptions::EnableWithoutSave,
            m_Iters32Filename
        };

        m_ItersMemory32.MutableResize(other.m_RoundedTotal);
        m_ItersArray32.resize(other.m_RoundedHeight);
        for (size_t i = 0; i < other.m_RoundedHeight; i++) {
            m_ItersArray32[i] = &m_ItersMemory32[i * other.m_RoundedWidth];
        }

        memcpy(
            m_ItersMemory32.GetData(),
            other.m_ItersMemory32.GetData(),
            other.m_RoundedTotal * sizeof(uint32_t));
    } else {
        m_Iters64Filename = GetTempFilename(64);
        m_ItersMemory64 = GrowableVector<uint64_t>{
            AddPointOptions::EnableWithoutSave,
            m_Iters64Filename
        };

        m_ItersMemory64.MutableResize(other.m_RoundedTotal);
        m_ItersArray64.resize(other.m_RoundedHeight);
        for (size_t i = 0; i < other.m_RoundedHeight; i++) {
            m_ItersArray64[i] = &m_ItersMemory64[i * other.m_RoundedWidth];
        }

        memcpy(
            m_ItersMemory64.GetData(),
            other.m_ItersMemory64.GetData(),
            other.m_RoundedTotal * sizeof(uint64_t));
    }

    return *this;
}

ItersMemoryContainer &ItersMemoryContainer::operator=(ItersMemoryContainer &&other) noexcept {
    if (this == &other) {
        return *this;
    }

    m_IterType = other.m_IterType;

    m_ItersMemory32 = std::move(other.m_ItersMemory32);
    m_ItersMemory64 = std::move(other.m_ItersMemory64);

    m_ItersArray32 = std::move(other.m_ItersArray32);
    m_ItersArray64 = std::move(other.m_ItersArray64);

    m_Iters32Filename = std::move(other.m_Iters32Filename);
    m_Iters64Filename = std::move(other.m_Iters64Filename);

    m_Width = other.m_Width;
    m_Height = other.m_Height;
    m_Total = other.m_Total;

    m_OutputWidth = other.m_OutputWidth;
    m_OutputHeight = other.m_OutputHeight;
    m_OutputTotal = other.m_OutputTotal;

    m_RoundedWidth = other.m_RoundedWidth;
    m_RoundedHeight = other.m_RoundedHeight;
    m_RoundedTotal = other.m_RoundedTotal;

    m_RoundedOutputColorWidth = other.m_RoundedOutputColorWidth;
    m_RoundedOutputColorHeight = other.m_RoundedOutputColorHeight;
    m_RoundedOutputColorTotal = other.m_RoundedOutputColorTotal;
    m_RoundedOutputColorMemory = std::move(other.m_RoundedOutputColorMemory);

    m_Antialiasing = other.m_Antialiasing;

    return *this;
}

ItersMemoryContainer::~ItersMemoryContainer() {
    m_RoundedOutputColorMemory = nullptr;
}

IterTypeFull ItersMemoryContainer::GetItersArrayValSlow(size_t x, size_t y) const {
    if (m_IterType == IterTypeEnum::Bits32) {
        return m_ItersArray32[y][x];
    } else {
        return m_ItersArray64[y][x];
    }
}

void ItersMemoryContainer::SetItersArrayValSlow(size_t x, size_t y, uint64_t val) {
    if (m_IterType == IterTypeEnum::Bits32) {
        m_ItersArray32[y][x] = (uint32_t)val;
    } else {
        m_ItersArray64[y][x] = val;
    }
}

void ItersMemoryContainer::GetReductionResults(ReductionResults &results) const {
    results = {};
    results.Min = std::numeric_limits<IterTypeFull>::max();
    results.Max = 0;
    results.Sum = 0;

    if (m_IterType == IterTypeEnum::Bits32) {
        for (size_t i = 0; i < m_Height; i++) {
            for (size_t j = 0; j < m_Width; j++) {
                results.Max = std::max(results.Max, static_cast<uint64_t>(m_ItersArray32[i][j]));
                results.Min = std::min(results.Min, static_cast<uint64_t>(m_ItersArray32[i][j]));
                results.Sum += m_ItersArray32[i][j];
            }
        }
    } else {
        for (size_t i = 0; i < m_Height; i++) {
            for (size_t j = 0; j < m_Width; j++) {
                results.Max = std::max(results.Max, m_ItersArray64[i][j]);
                results.Min = std::min(results.Min, m_ItersArray64[i][j]);
                results.Sum += m_ItersArray64[i][j];
            }
        }
    }
}