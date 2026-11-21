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
    std::filesystem::path FfmpegExecutable;
    std::string Command;
};

// Capture runs in the render pool's serialized mutation queue without rendering
// or changing the view. The returned INI owns all coordinate digits; it contains
// no palette settings and therefore uses the CLI's default coloring.
std::string CaptureMovieConfiguration(Fractal &fractal);

// Recognize both installed sibling executables and CMake's per-target folders.
std::optional<std::filesystem::path> FindMovieCliExecutable(const std::filesystem::path &guiExecutable,
                                                            MovieCommandShell shell);

MovieCommandExport BuildMovieCommandExport(const std::filesystem::path &configuration,
                                           const std::filesystem::path &cliExecutable,
                                           const std::filesystem::path &ffmpegExecutable,
                                           MovieCommandShell shell);

// The chosen INI name supplies the stem of a new .movie directory; movie.ini
// and ffmpeg.txt live inside it. Existing jobs are never replaced by an export.
// All dialogs must finish before publication. The clipboard is updated only
// after both files have been published successfully.
void SaveMovieConfiguration(const MovieCommandExport &output, std::string_view configuration);
bool CopyMovieCommandToClipboard(const MovieCommandExport &output);
std::string DescribeMovieCommandExport(const MovieCommandExport &output, bool copied);

} // namespace FractalShark
