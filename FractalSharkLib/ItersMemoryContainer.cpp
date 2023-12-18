#include "stdafx.h"
#include "ItersMemoryContainer.h"
#include <algorithm>

ItersMemoryContainer::ItersMemoryContainer(
    IterTypeEnum type,
    size_t width,
    size_t height,
    size_t total_antialiasing)
    : m_IterType(type),
    m_ItersMemory32(nullptr),
    m_ItersArray32(nullptr),
    m_ItersMemory64(nullptr),
    m_ItersArray64(nullptr),
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

    if (m_IterType == IterTypeEnum::Bits32) {
        m_ItersMemory32 = std::make_unique<uint32_t[]>(m_RoundedTotal);
        m_ItersArray32 = new uint32_t * [m_RoundedHeight];
        for (size_t i = 0; i < m_RoundedHeight; i++) {
            m_ItersArray32[i] = &m_ItersMemory32[i * m_RoundedWidth];
        }
    }
    else {
        m_ItersMemory64 = std::make_unique<uint64_t[]>(m_RoundedTotal);
        m_ItersArray64 = new uint64_t * [m_RoundedHeight];
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

ItersMemoryContainer::ItersMemoryContainer(ItersMemoryContainer&& other) noexcept {
    *this = std::move(other);
}

ItersMemoryContainer& ItersMemoryContainer::operator=(ItersMemoryContainer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    m_IterType = other.m_IterType;

    m_ItersMemory32 = std::move(other.m_ItersMemory32);
    m_ItersMemory64 = std::move(other.m_ItersMemory64);

    m_ItersArray32 = other.m_ItersArray32;
    other.m_ItersArray32 = nullptr;

    m_ItersArray64 = other.m_ItersArray64;
    other.m_ItersArray64 = nullptr;

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
    m_ItersMemory32 = nullptr;
    m_ItersMemory64 = nullptr;
    m_RoundedOutputColorMemory = nullptr;

    if (m_ItersArray32) {
        delete[] m_ItersArray32;
        m_ItersArray32 = nullptr;
    }

    if (m_ItersArray64) {
        delete[] m_ItersArray64;
        m_ItersArray64 = nullptr;
    }
}

IterTypeFull ItersMemoryContainer::GetItersArrayValSlow(size_t x, size_t y) const {
    if (m_IterType == IterTypeEnum::Bits32) {
        return m_ItersArray32[y][x];
    }
    else {
        return m_ItersArray64[y][x];
    }
}

void ItersMemoryContainer::SetItersArrayValSlow(size_t x, size_t y, uint64_t val) {
    if (m_IterType == IterTypeEnum::Bits32) {
        m_ItersArray32[y][x] = (uint32_t)val;
    }
    else {
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
    }
    else {
        for (size_t i = 0; i < m_Height; i++) {
            for (size_t j = 0; j < m_Width; j++) {
                results.Max = std::max(results.Max, m_ItersArray64[i][j]);
                results.Min = std::min(results.Min, m_ItersArray64[i][j]);
                results.Sum += m_ItersArray64[i][j];
            }
        }
    }
}