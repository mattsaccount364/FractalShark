#pragma once

#include "FractalSharkGpuLib/NvencMovieEncoder.h"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

class Fractal;

namespace FractalShark {

inline constexpr std::string_view MovieDefaultZoomPerFrame = "1.02";
inline constexpr std::array<std::string_view, 2> MovieRenderOrders{"wide-to-deep", "deep-to-wide"};

struct MovieVideoProfile {
    std::string_view Name;
    std::string_view Description;
    MovieCodec Codec;
    unsigned Preset;
    unsigned Quality;
    MovieHevcTier Tier;
};

inline constexpr MovieVideoProfile DefaultMovieVideoProfile{
    "hevc-high", "P5, High tier, NV12, VBR quality 16", MovieCodec::Hevc, 5, 16, MovieHevcTier::High};
inline constexpr std::array MovieVideoProfiles{
    DefaultMovieVideoProfile,
    MovieVideoProfile{
        "hevc-low", "P5, Main tier, NV12, VBR quality 22", MovieCodec::Hevc, 5, 22, MovieHevcTier::Main},
    MovieVideoProfile{"hevc-rgb-lossless",
                      "P4, lossless RGB8 after rounding",
                      MovieCodec::HevcRgbLossless,
                      4,
                      0,
                      MovieHevcTier::Main},
    MovieVideoProfile{
        "h264-high", "P5, NV12, VBR quality 18", MovieCodec::H264, 5, 18, MovieHevcTier::Main},
    MovieVideoProfile{
        "h264-low", "P1, NV12, VBR quality 30", MovieCodec::H264, 1, 30, MovieHevcTier::Main},
    MovieVideoProfile{
        "png-only", "Numbered RGB16 PNG frames", MovieCodec::PngOnly, 0, 0, MovieHevcTier::Main}};

enum class MovieCommandShell { PowerShell, Posix };

struct MovieCommandExport {
    std::filesystem::path ConfigurationPath;
    std::filesystem::path JobDirectory;
    std::string Command;
};

// Shared by GUI exports and CLI snapshots. Comments are excluded from job identity.
std::string MovieIniComments(std::string_view key);
std::string MovieMegabits(uint32_t bitsPerSecond);

// Capture runs in the render pool's serialized mutation queue without rendering
// or changing the view. The returned INI owns all coordinate digits; it contains
// no palette settings and therefore uses the CLI's default coloring.
std::string CaptureMovieConfiguration(Fractal &fractal);

// Recognize both installed sibling executables and CMake's per-target folders.
std::optional<std::filesystem::path> FindMovieCliExecutable(const std::filesystem::path &guiExecutable,
                                                            MovieCommandShell shell);

MovieCommandExport BuildMovieCommandExport(const std::filesystem::path &configuration,
                                           const std::filesystem::path &cliExecutable,
                                           MovieCommandShell shell);

// Use the chosen INI name as the stem of a new .movie directory with movie.ini inside.
void SaveMovieConfiguration(const MovieCommandExport &output, std::string_view configuration);
bool CopyMovieCommandToClipboard(const MovieCommandExport &output);
std::string DescribeMovieCommandExport(const MovieCommandExport &output, bool copied);

} // namespace FractalShark
