#include "stdafx.h"
#include "FractalPalette.h"

#include <thread>
#include <vector>

FractalPalette::FractalPalette()
    : m_WhichPalette{FractalPaletteType::Default},
      m_PaletteRotate{0},
      m_PaletteDepthIndex{2},
      m_PaletteAuxDepth{0}
{
}

void
FractalPalette::SetDefaults()
{
    m_PaletteRotate = 0;
    m_PaletteDepthIndex = 2;
    m_PaletteAuxDepth = 0;
    m_WhichPalette = FractalPaletteType::Default;
}

void
FractalPalette::InitializeAllPalettes()
{
    auto DefaultPaletteGen = [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        SetThreadDescription(GetCurrentThread(), L"Fractal::DefaultPaletteGen");
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    auto PatrioticPaletteGen= [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        SetThreadDescription(GetCurrentThread(), L"Fractal::PatrioticPaletteGen");
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;

        // R=0xBB G=0x13 B=0x3E
        // R=0xB3 G=0x19 B=0x42
        // R=0xBF G=0x0A B=0x30
        const auto RR = (int)(((double)0xB3 / (double)0xFF) * max_val);
        const auto RG = (int)(((double)0x19 / (double)0xFF) * max_val);
        const auto RB = (int)(((double)0x42 / (double)0xFF) * max_val);

        // R=0x00 G=0x21 B=0x47
        // R=0x0A G=0x31 B=0x61
        // R=0x00 G=0x28 B=0x68
        const auto BR = (int)(((double)0x0A / (double)0xFF) * max_val);
        const auto BG = (int)(((double)0x31 / (double)0xFF) * max_val);
        const auto BB = (int)(((double)0x61 / (double)0xFF) * max_val);

        m_PalInterleaved[WhichPalette][PaletteIndex].push_back(
            {static_cast<uint16_t>(max_val), static_cast<uint16_t>(max_val),
             static_cast<uint16_t>(max_val), 0});

        PalTransition(WhichPalette, PaletteIndex, depth_total, RR, RG, RB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, BR, BG, BB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    auto SummerPaletteGen = [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        SetThreadDescription(GetCurrentThread(), L"Fractal::SummerPaletteGen");
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;

        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val / 2, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val / 2, max_val / 2, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val * 2 / 3, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    for (size_t i = 0; i < FractalPaletteType::Num; i++) {
        m_PalIters[i].resize(NumBitDepths);
    }

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 0, 5));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 1, 6));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 2, 8));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 3, 12));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 4, 16));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPaletteType::Default, 5, 20));

    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 0, 5));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 1, 6));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 2, 8));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 3, 12));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 4, 16));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPaletteType::Patriotic, 5, 20));

    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 0, 5));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 1, 6));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 2, 8));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 3, 12));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 4, 16));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPaletteType::Summer, 5, 20));

    for (auto &it : threads) {
        it->join();
    }

    // Set up random palette.
    CreateNewRandomPalette();
}

//////////////////////////////////////////////////////////////////////////////

// Given an empty array, a range of indexes to iterate over, and a start number
// and end number, this function will smoothly transition from val1 to val2
// over the indexes specified.
// length must be > 0
// total_length = number of elements in pal.
// e.g. unsigned char pal[256];
//   total_length == 256
// Transitions to the color specified.
// Allows for nice smooth palettes.
// length must be > 0
void
FractalPalette::PalTransition(size_t WhichPalette, size_t PaletteIndex, int length, int r, int g, int b)
{
    int curR, curG, curB;
    auto &pal = m_PalInterleaved[WhichPalette][PaletteIndex];
    if (!pal.empty()) {
        curR = pal.back().r;
        curG = pal.back().g;
        curB = pal.back().b;
    } else {
        curR = 0;
        curG = 0;
        curB = 0;
    }

    double deltaR = (double)(r - curR) / length;
    double deltaG = (double)(g - curG) / length;
    double deltaB = (double)(b - curB) / length;

    for (int i = 0; i < length; i++) {
        Color16 c;
        c.r = static_cast<uint16_t>(curR + deltaR * (i + 1));
        c.g = static_cast<uint16_t>(curG + deltaG * (i + 1));
        c.b = static_cast<uint16_t>(curB + deltaB * (i + 1));
        c.a = 0;
        pal.push_back(c);
    }
}

void
FractalPalette::UsePaletteType(FractalPaletteType type)
{
    m_WhichPalette = type;
}

FractalPaletteType
FractalPalette::GetPaletteType() const
{
    return m_WhichPalette;
}

uint32_t
FractalPalette::GetPaletteDepthFromIndex(size_t index) const
{
    switch (index) {
        case 0:
            return 5;
        case 1:
            return 6;
        case 2:
            return 8;
        case 3:
            return 12;
        case 4:
            return 16;
        case 5:
            return 20;
        default:
            return 8;
    }
}

void
FractalPalette::UsePalette(int depth)
{
    switch (depth) {
        case 5:
            m_PaletteDepthIndex = 0;
            break;
        case 6:
            m_PaletteDepthIndex = 1;
            break;
        case 8:
            m_PaletteDepthIndex = 2;
            break;
        case 12:
            m_PaletteDepthIndex = 3;
            break;
        case 16:
            m_PaletteDepthIndex = 4;
            break;
        case 20:
            m_PaletteDepthIndex = 5;
            break;
        default:
            m_PaletteDepthIndex = 0;
            break;
    }
}

void
FractalPalette::UseNextPaletteDepth()
{
    m_PaletteDepthIndex = (m_PaletteDepthIndex + 1) % 6;
}

void
FractalPalette::SetPaletteAuxDepth(int32_t depth)
{
    if (depth < 0 || depth > 16) {
        return;
    }

    m_PaletteAuxDepth = depth;
}

void
FractalPalette::UseNextPaletteAuxDepth(int32_t inc)
{
    if (inc < -5 || inc > 5 || inc == 0) {
        return;
    }

    if (inc < 0) {
        if (m_PaletteAuxDepth == 0) {
            m_PaletteAuxDepth = 17 + inc;
        } else {
            m_PaletteAuxDepth += inc;
        }
    } else {
        if (m_PaletteAuxDepth >= 16) {
            m_PaletteAuxDepth = -1 + inc;
        } else {
            m_PaletteAuxDepth += inc;
        }
    }
}

uint32_t
FractalPalette::GetPaletteDepth() const
{
    return GetPaletteDepthFromIndex(m_PaletteDepthIndex);
}

void
FractalPalette::ResetPaletteRotation()
{
    m_PaletteRotate = 0;
}

void
FractalPalette::RotatePalette(int delta, IterTypeFull maxIters)
{
    m_PaletteRotate += delta;
    if (m_PaletteRotate >= maxIters) {
        m_PaletteRotate = 0;
    }
}

void
FractalPalette::CreateNewRandomPalette()
{
    size_t rtime = __rdtsc();

    auto genNextColor = [](int m) -> int {
        const int max_val = 65535 / (m - 1);
        auto val = (rand() % m) * max_val;
        return val;
    };

    auto RandomPaletteGen = [&](size_t PaletteIndex, size_t Depth) {
        SetThreadDescription(GetCurrentThread(), L"Random Palette Gen");
        int depth_total = (int)(1 << Depth);

        srand((unsigned int)rtime);

        // Force a reallocation to trigger re-initialization in the GPU
        std::vector<Color16>{}.swap(m_PalInterleaved[FractalPaletteType::Random][PaletteIndex]);

        const int m = 5;
        auto firstR = genNextColor(m);
        auto firstG = genNextColor(m);
        auto firstB = genNextColor(m);
        PalTransition(FractalPaletteType::Random, PaletteIndex, depth_total, firstR, firstG, firstB);
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPaletteType::Random, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[FractalPaletteType::Random][PaletteIndex] =
            (uint32_t)m_PalInterleaved[FractalPaletteType::Random][PaletteIndex].size();
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 0, 5));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 1, 6));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 2, 8));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 3, 12));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 4, 16));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 5, 20));

    for (auto &it : threads) {
        it->join();
    }
}

IterTypeFull
FractalPalette::GetPaletteRotation() const
{
    return m_PaletteRotate;
}

int
FractalPalette::GetPaletteDepthIndex() const
{
    return m_PaletteDepthIndex;
}

int32_t
FractalPalette::GetAuxDepth() const
{
    return m_PaletteAuxDepth;
}

const Color16 *
FractalPalette::GetCurrentPalInterleaved() const
{
    return m_PalInterleaved[m_WhichPalette][m_PaletteDepthIndex].data();
}

uint32_t
FractalPalette::GetCurrentNumColors() const
{
    return m_PalIters[m_WhichPalette][m_PaletteDepthIndex];
}

const std::vector<Color16> *
FractalPalette::GetPalInterleaved(size_t whichPalette) const
{
    return m_PalInterleaved[whichPalette];
}

const std::vector<uint32_t> &
FractalPalette::GetPalIters(size_t whichPalette) const
{
    return m_PalIters[whichPalette];
}
