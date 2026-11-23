#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

class Fractal;

namespace FractalShark {

enum class MovieCommandShell { PowerShell, Posix };

struct MovieCommandExport {
    std::filesystem::path ConfigurationPath;
    std::filesystem::path JobDirectory;
    std::string Command;
};

// Shared by exported INIs and the CLI. Examples are comments, not job identity.
std::string MovieEncoderExamples();

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
