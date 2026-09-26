#include "stdafx.h"
#include "Environment.h"
#include "Exceptions.h"
#include "FractalPalette.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t MaxCustomPaletteEntries = size_t{1} << FractalPalette::PaletteDepths.back();

[[noreturn]] void
ThrowCustomPaletteError(const std::filesystem::path &path, size_t lineNumber, const std::string &message)
{
    std::ostringstream error;
    error << "Custom palette \"" << path.string() << "\"";
    if (lineNumber != 0) {
        error << " line " << lineNumber;
    }
    error << ": " << message;
    throw FractalSharkSeriousException(error.str());
}

int32_t
ParseCustomPaletteComponent(const std::string &token,
                            const std::filesystem::path &path,
                            size_t lineNumber)
{
    const char *begin = token.data();
    if (!token.empty() && *begin == '+') {
        ++begin;
    }

    int32_t component = 0;
    const auto [end, error] = std::from_chars(begin, token.data() + token.size(), component, 10);
    if (error != std::errc{} || end != token.data() + token.size()) {
        ThrowCustomPaletteError(path, lineNumber, "components must be signed 32-bit decimal integers");
    }
    return component;
}

uint16_t
ConvertCustomPaletteComponent(int32_t component)
{
    constexpr int32_t MinComponent = 0;
    constexpr int32_t MaxComponent = 255;
    constexpr int32_t Color16Scale = 257;
    return static_cast<uint16_t>(std::clamp(component, MinComponent, MaxComponent) * Color16Scale);
}

uint16_t
InterpolateCustomPaletteComponent(uint16_t first,
                                  uint16_t second,
                                  uint64_t fraction,
                                  uint64_t denominator)
{
    const uint64_t weighted = static_cast<uint64_t>(first) * (denominator - fraction) +
                              static_cast<uint64_t>(second) * fraction;
    return static_cast<uint16_t>(weighted / denominator);
}

Color16
InterpolateCustomPaletteColor(const Color16 &first,
                              const Color16 &second,
                              uint64_t fraction,
                              uint64_t denominator)
{
    return {InterpolateCustomPaletteComponent(first.r, second.r, fraction, denominator),
            InterpolateCustomPaletteComponent(first.g, second.g, fraction, denominator),
            InterpolateCustomPaletteComponent(first.b, second.b, fraction, denominator),
            0};
}

} // namespace

FractalPalette::FractalPalette()
    : m_WhichPalette{FractalPaletteType::Default}, m_PaletteRotate{0},
      m_PaletteDepthIndex{static_cast<int>(DefaultPaletteDepthIndex)}, m_PaletteAuxDepth{0}
{
}

void
FractalPalette::SetDefaults()
{
    m_PaletteRotate = 0;
    m_PaletteDepthIndex = static_cast<int>(DefaultPaletteDepthIndex);
    m_PaletteAuxDepth = 0;
    m_WhichPalette = FractalPaletteType::Default;
}

void
FractalPalette::InitializeAllPalettes()
{
    auto DefaultPaletteGen = [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        Environment::SetCurrentThreadName(L"Fractal::DefaultPaletteGen");
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] =
            (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    auto PatrioticPaletteGen = [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        Environment::SetCurrentThreadName(L"Fractal::PatrioticPaletteGen");
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

        m_PalInterleaved[WhichPalette][PaletteIndex].push_back({static_cast<uint16_t>(max_val),
                                                                static_cast<uint16_t>(max_val),
                                                                static_cast<uint16_t>(max_val),
                                                                0});

        PalTransition(WhichPalette, PaletteIndex, depth_total, RR, RG, RB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, BR, BG, BB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);

        m_PalIters[WhichPalette][PaletteIndex] =
            (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    auto SummerPaletteGen = [&](FractalPaletteType WhichPalette, size_t PaletteIndex, size_t Depth) {
        Environment::SetCurrentThreadName(L"Fractal::SummerPaletteGen");
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;

        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val / 2, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val / 2, max_val / 2, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val * 2 / 3, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] =
            (uint32_t)m_PalInterleaved[WhichPalette][PaletteIndex].size();
    };

    for (size_t i = 0; i < FractalPaletteType::Num; i++) {
        m_PalIters[i].resize(NumBitDepths);
    }

    std::vector<std::thread> threads;
    threads.reserve(FractalPaletteType::Num * NumBitDepths);
    auto launchPaletteGenerators = [&](auto &generator, FractalPaletteType paletteType) {
        for (size_t paletteIndex = 0; paletteIndex < NumBitDepths; ++paletteIndex) {
            threads.emplace_back(generator, paletteType, paletteIndex, PaletteDepths[paletteIndex]);
        }
    };
    launchPaletteGenerators(DefaultPaletteGen, FractalPaletteType::Default);
    launchPaletteGenerators(PatrioticPaletteGen, FractalPaletteType::Patriotic);
    launchPaletteGenerators(SummerPaletteGen, FractalPaletteType::Summer);

    for (auto &it : threads) {
        it.join();
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
    if (type == FractalPaletteType::Custom && !m_HasCustomPalette) {
        throw FractalSharkSeriousException("No custom palette has been loaded");
    }

    m_WhichPalette = type;
}

FractalPaletteType
FractalPalette::GetPaletteType() const
{
    return m_WhichPalette;
}

bool
FractalPalette::HasCustomPalette() const
{
    return m_HasCustomPalette;
}

void
FractalPalette::LoadCustomPalette(const std::filesystem::path &path)
{
    std::ifstream input(path);
    if (!input) {
        ThrowCustomPaletteError(path, 0, "could not open file");
    }

    std::vector<Color16> sourceColors;
    sourceColors.reserve(1u << DefaultPaletteDepth);

    std::string line;
    size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;

        std::istringstream fields(line);
        std::string redToken;
        if (!(fields >> redToken)) {
            continue;
        }
        if (redToken.starts_with('#') || redToken.starts_with("//")) {
            continue;
        }

        std::string greenToken;
        std::string blueToken;
        if (!(fields >> greenToken >> blueToken)) {
            ThrowCustomPaletteError(path, lineNumber, "expected at least three RGB components");
        }

        if (sourceColors.size() == MaxCustomPaletteEntries) {
            ThrowCustomPaletteError(
                path, lineNumber, "contains more entries than the maximum supported palette depth");
        }

        sourceColors.push_back(
            {ConvertCustomPaletteComponent(ParseCustomPaletteComponent(redToken, path, lineNumber)),
             ConvertCustomPaletteComponent(ParseCustomPaletteComponent(greenToken, path, lineNumber)),
             ConvertCustomPaletteComponent(ParseCustomPaletteComponent(blueToken, path, lineNumber)),
             0});
    }

    if (input.bad()) {
        ThrowCustomPaletteError(path, 0, "failed while reading file");
    }
    if (sourceColors.empty()) {
        ThrowCustomPaletteError(path, 0, "contains no RGB entries");
    }

    std::array<std::vector<Color16>, NumBitDepths> generatedPalettes;
    std::array<uint32_t, NumBitDepths> generatedCounts;
    for (size_t paletteIndex = 0; paletteIndex < NumBitDepths; ++paletteIndex) {
        const size_t targetCount = size_t{1} << PaletteDepths[paletteIndex];
        auto &target = generatedPalettes[paletteIndex];
        target.reserve(targetCount);

        if (sourceColors.size() == targetCount) {
            target = sourceColors;
        } else {
            for (size_t targetIndex = 0; targetIndex < targetCount; ++targetIndex) {
                const uint64_t sourcePosition = static_cast<uint64_t>(targetIndex) * sourceColors.size();
                const size_t sourceIndex = static_cast<size_t>(sourcePosition / targetCount);
                const uint64_t fraction = sourcePosition % targetCount;
                const Color16 &first = sourceColors[sourceIndex];
                const Color16 &second = sourceColors[(sourceIndex + 1) % sourceColors.size()];
                target.push_back(InterpolateCustomPaletteColor(first, second, fraction, targetCount));
            }
        }

        generatedCounts[paletteIndex] = static_cast<uint32_t>(target.size());
    }

    for (size_t paletteIndex = 0; paletteIndex < NumBitDepths; ++paletteIndex) {
        m_PalInterleaved[FractalPaletteType::Custom][paletteIndex] =
            std::move(generatedPalettes[paletteIndex]);
    }
    m_PalIters[FractalPaletteType::Custom].assign(generatedCounts.begin(), generatedCounts.end());
    m_HasCustomPalette = true;
    ++m_PaletteGeneration;
}

uint32_t
FractalPalette::GetPaletteDepthFromIndex(size_t index) const
{
    if (index < PaletteDepths.size()) {
        return PaletteDepths[index];
    }

    return DefaultPaletteDepth;
}

void
FractalPalette::UsePalette(int depth)
{
    m_PaletteDepthIndex = 0;
    for (size_t i = 0; i < PaletteDepths.size(); ++i) {
        if (depth >= 0 && PaletteDepths[i] == static_cast<uint32_t>(depth)) {
            m_PaletteDepthIndex = static_cast<int>(i);
            break;
        }
    }
}

void
FractalPalette::UseNextPaletteDepth()
{
    m_PaletteDepthIndex = (m_PaletteDepthIndex + 1) % static_cast<int>(PaletteDepths.size());
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
        Environment::SetCurrentThreadName(L"Random Palette Gen");
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

    std::vector<std::thread> threads;
    threads.reserve(NumBitDepths);
    for (size_t paletteIndex = 0; paletteIndex < NumBitDepths; ++paletteIndex) {
        threads.emplace_back(RandomPaletteGen, paletteIndex, PaletteDepths[paletteIndex]);
    }

    for (auto &it : threads) {
        it.join();
    }

    ++m_PaletteGeneration;
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

uint64_t
FractalPalette::GetPaletteGeneration() const
{
    return m_PaletteGeneration;
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
