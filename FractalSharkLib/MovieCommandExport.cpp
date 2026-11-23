#include "stdafx.h"

#include "MovieCommandExport.h"

#include "ConsoleLog.h"
#include "Environment.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "SaveFileTransaction.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <locale>
#include <sstream>

namespace FractalShark {
namespace {

constexpr uint64_t MovieWideIterations = 8192;

std::string
PathUtf8(const std::filesystem::path &path)
{
    return Environment::WideToUtf8(Environment::FromFsPath(path));
}

std::string
QuoteMovieArgument(const std::filesystem::path &path, MovieCommandShell shell)
{
    const auto text = PathUtf8(path);
    if (text.empty() || text.find_first_of("\r\n") != std::string::npos ||
        text.find('\0') != std::string::npos) {
        throw FractalSharkSeriousException("Movie command paths must be nonempty single-line paths");
    }
    std::string quoted = "'";
    for (const char ch : text) {
        if (ch == '\'') {
            // PowerShell doubles an apostrophe inside a literal string. POSIX
            // shells close the literal, append an escaped apostrophe, and reopen.
            quoted += shell == MovieCommandShell::PowerShell ? "''" : "'\\''";
        } else {
            quoted += ch;
        }
    }
    return quoted + "'";
}

std::string
FormatMovieZoom(const HighPrecision &zoom)
{
    // Two 64-bit limbs correspond to about 39 decimal digits. Use the decimal
    // exponent directly so the format works beyond the range of native floats.
    constexpr size_t SignificantDigits = 39;
    std::array<char, SignificantDigits + 2> digits{};
    mp_exp_t exponent = 0;
    mpf_get_str(digits.data(), &exponent, 10, SignificantDigits, zoom.backend());
    std::string text(digits.data());
    if (text.size() > 1) {
        text.insert(1, 1, '.');
    }
    if (exponent != 1) {
        text += 'e' + std::to_string(exponent - 1);
    }
    return text;
}

void
WriteMovieSetting(std::ostream &output, std::string_view key, const auto &value)
{
    output << MovieIniComments(key) << key << '=' << value << '\n';
}

std::string
SerializeMovieConfiguration(Fractal &fractal)
{
    const auto &view = fractal.GetPtz();
    if (view.GetZoomFactor() < HighPrecision(1)) {
        throw FractalSharkSeriousException("A zoom movie target must have zoom at least one");
    }
    const auto iterations = fractal.GetNumIterations<IterTypeFull>();
    const auto width = fractal.GetRenderWidth();
    const auto height = fractal.GetRenderHeight();
    if (iterations == 0 || width == 0 || height == 0 ||
        width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw FractalSharkSeriousException("Invalid movie render dimensions or iteration limit");
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    // The center requires its full absolute precision at deep zoom. The
    // viewport scale only needs relative precision.
    output << "[movie]\n";
    WriteMovieSetting(output, "center-x", view.GetPtX());
    WriteMovieSetting(output, "center-y", view.GetPtY());
    WriteMovieSetting(output, "zoom", FormatMovieZoom(view.GetZoomFactor()));
    WriteMovieSetting(output, "zoom-per-frame", MovieDefaultZoomPerFrame);
    WriteMovieSetting(output, "iterations", iterations);
    WriteMovieSetting(output, "width", width);
    WriteMovieSetting(output, "height", height);
    WriteMovieSetting(output, "antialiasing", fractal.GetGpuAntialiasing());
    WriteMovieSetting(output, "wide-iterations", std::min<uint64_t>(MovieWideIterations, iterations));
    WriteMovieSetting(output, "la-slope-ratio", 8);
    WriteMovieSetting(output, "video-gop-frames", MovieVideoSettings{}.GopFrames);
    WriteMovieSetting(output, "video-fps", MovieVideoSettings::DefaultFps);
    WriteMovieSetting(output, "render-algorithm", "AutoSelect");
    WriteMovieSetting(output, "perturbation-alg", "Auto");
    WriteMovieSetting(output, "render-order", MovieRenderOrders.front());
    WriteMovieSetting(output, "video-profile", DefaultMovieVideoProfile.Name);
    WriteMovieSetting(output, "video-preset", DefaultMovieVideoProfile.Preset);
    WriteMovieSetting(output, "video-cq", DefaultMovieVideoProfile.Quality);
    const MovieHevcOptions options;
    const auto rate = ResolveMovieHevcRateControl(static_cast<unsigned>(width),
                                                  static_cast<unsigned>(height),
                                                  MovieVideoSettings::DefaultFps,
                                                  DefaultMovieVideoProfile.Tier,
                                                  options);
    WriteMovieSetting(output, "video-max-bitrate-mbps", options.PeakMbps);
    WriteMovieSetting(output, "video-average-bitrate-mbps", MovieMegabits(rate.AverageBitsPerSecond));
    WriteMovieSetting(output, "video-vbv-buffer-ms", options.BufferMilliseconds);
    WriteMovieSetting(output, "video-vbv-initial-ms", options.InitialMilliseconds);
    return output.str();
}

} // namespace

std::string
MovieIniComments(std::string_view key)
{
    std::ostringstream comments;
    comments << '\n';
    const auto choice = [&](std::string_view name) { comments << "# " << key << '=' << name; };
    if (key == "center-x") {
        comments << "# Real (horizontal) coordinate of the point the movie zooms around.\n"
                    "# Deep zooms need the full coordinate digits to stay centered.\n";
    } else if (key == "center-y") {
        comments << "# Imaginary (vertical) coordinate of the point the movie zooms around.\n"
                    "# Deep zooms need the full coordinate digits to stay centered.\n";
    } else if (key == "zoom") {
        comments
            << "# Target magnification, at least 1x. This is the deepest view in either render order.\n"
               "# For large values, use scientific notation such as 1e30.\n";
    } else if (key == "zoom-per-frame") {
        comments << "# Zoom multiplier per frame, greater than 1. At 1.02, zoom increases by 2%\n"
                    "# per step going in and divides by 1.02 going out. A larger step produces\n"
                    "# fewer frames for the same zoom range.\n";
    } else if (key == "iterations") {
        comments << "# Maximum Mandelbrot iterations per sample at the deepest view. Higher values\n"
                    "# can reveal more detail and take longer to render.\n";
    } else if (key == "width") {
        comments << "# Output frame width in pixels. NV12 video requires an even width.\n";
    } else if (key == "height") {
        comments << "# Output frame height in pixels. NV12 video requires an even height.\n";
    } else if (key == "antialiasing") {
        comments << "# Supersampling factor per dimension. Values 1, 2, 3, and 4 give 1, 4, 9,\n"
                    "# and 16 samples per output pixel. Higher values take more rendering work.\n";
    } else if (key == "wide-iterations") {
        comments << "# Maximum iterations per sample at 1x zoom. The default is 8192, capped at\n"
                    "# iterations. The cap rises toward iterations as zoom increases.\n";
    } else if (key == "la-slope-ratio") {
        comments << "# Ratio of the late slope to the early slope, measured per doubling of zoom.\n"
                    "# At 8, the late ramp is eight times as steep as the early ramp. GPU Auto\n"
                    "# switches at 1e9x zoom, and CPU Auto at 1e34x. A larger ratio delays the\n"
                    "# buildup of iterations and reaches the same cap at the target. A switch\n"
                    "# at either endpoint or outside the zoom range gives a single slope.\n";
    } else if (key == "video-gop-frames") {
        comments << "# Regular keyframe interval in frames. Frame 0 and the first frame after a\n"
                    "# resume are keyframes. An interval of 1000 is about 33 seconds at 30 fps.\n";
    } else if (key == "video-fps") {
        comments << "# Playback frames per second. Duration in seconds is frame count divided by\n"
                    "# FPS. Frame size and FPS also determine the HEVC level.\n";
    } else if (key == "palette-map") {
        comments << "# Optional path to an RGB8 palette map, one red, green, blue triplet per row.\n"
                    "# Relative paths are interpreted from this INI's directory.\n";
    } else if (key == "palette-depth") {
        comments << "# Palette lookup resolution in bits. Valid values are 5, 6, 8, 12, 16, and 20.\n";
    } else if (key == "render-algorithm") {
        comments << "# Algorithm for computing pixels. AutoSelect chooses by zoom depth and GPU "
                    "availability.\n";
        for (const auto &algorithm : RenderAlgorithms) {
            if (FractalShark::IsCliRenderAlgorithm(algorithm)) {
                choice(algorithm.AlgorithmStr);
                comments << '\n';
            }
        }
    } else if (key == "perturbation-alg") {
        comments << "# Algorithm for generating reference orbits when the renderer uses perturbation.\n"
                    "# Auto chooses based on required precision and available hardware.\n";
        for (const auto &algorithm : PerturbationAlgorithms) {
            choice(algorithm.Name);
            comments << '\n';
        }
    } else if (key == "render-order") {
        comments << "# wide-to-deep zooms in from 1x. deep-to-wide zooms out from the target.\n";
        for (const auto order : MovieRenderOrders) {
            choice(order);
            comments << '\n';
        }
    } else if (key == "video-profile") {
        comments
            << "# Select the video codec and its defaults, or choose png-only for individual frames.\n"
               "# Active tuning lines override profile defaults. Comment them out to use the "
               "profile defaults.\n";
        for (const auto &profile : MovieVideoProfiles) {
            choice(profile.Name);
            comments << "  # " << profile.Description << '\n';
        }
    } else if (key == "video-preset") {
        comments << "# NVENC preset P1 (fastest) through P7 (slowest). Slower presets can\n"
                    "# compress more efficiently. Comment out for png-only.\n";
    } else if (key == "video-cq") {
        comments << "# VBR quality target (1..51). Lower values ask for more detail. The bitrate\n"
                    "# budget can limit how much detail the encoder retains. HEVC accepts fractions,\n"
                    "# H.264 takes integers. Comment out for lossless and png-only.\n";
    } else if (key == "video-max-bitrate-mbps") {
        comments
            << "# Lossy HEVC peak bitrate for short bursts, in whole Mb/s (1..800).\n"
               "# The HEVC level describes what the decoder must handle. FractalShark chooses\n"
               "# it from frame size and FPS. The profile's Main or High tier sets the bitrate\n"
               "# ceiling for that level. The effective peak is the smaller of the request and\n"
               "# that ceiling. The log uses level_idc=120 for Level 4 and 180 for Level 6.\n"
               "# Examples for hevc-high at 30 fps with a 180 Mb/s peak request:\n"
               "#   1920x1080: Level 4, 30 Mb/s ceiling and peak, 20 Mb/s default average.\n"
               "#   5120x2160: Level 6, 240 Mb/s ceiling, 180 Mb/s peak, 120 Mb/s default average.\n"
               "# Comment out this peak, the average and both VBV settings for other profiles.\n";
    } else if (key == "video-average-bitrate-mbps") {
        comments
            << "# Lossy HEVC sustained bitrate target in Mb/s, capped at the effective peak.\n"
               "# The exported number is an explicit override. If you change the dimensions,\n"
               "# FPS, profile, or requested peak, update this value or comment out this line\n"
               "# before starting. An omitted value defaults to two-thirds of the effective peak.\n";
    } else if (key == "video-vbv-buffer-ms") {
        comments << "# Lossy HEVC decoder buffer for short bursts. Size in milliseconds of data\n"
                    "# at the effective peak bitrate (1..1000).\n";
    } else if (key == "video-vbv-initial-ms") {
        comments << "# Lossy HEVC starting buffer fill in milliseconds at the effective peak.\n"
                    "# The initial fill must fit in the buffer (1 through video-vbv-buffer-ms).\n";
    } else if (key == "video-output") {
        comments << "# Absolute .ts path. Set before starting. PNG-only jobs use the frames folder.\n";
    } else {
        return {};
    }
    return std::move(comments).str();
}

std::string
MovieMegabits(uint32_t bitsPerSecond)
{
    constexpr uint32_t BitsPerMegabit = 1'000'000;
    const auto whole = std::to_string(bitsPerSecond / BitsPerMegabit);
    const auto remainder = bitsPerSecond % BitsPerMegabit;
    if (remainder == 0) {
        return whole;
    }
    auto fraction = std::to_string(BitsPerMegabit + remainder).substr(1);
    fraction.erase(fraction.find_last_not_of('0') + 1);
    return whole + '.' + fraction;
}

std::string
CaptureMovieConfiguration(Fractal &fractal)
{
    if (fractal.GetRenderPool() == nullptr) {
        return SerializeMovieConfiguration(fractal);
    }
    // Workers dequeue in order but contend for the calculation lock separately.
    // Finish earlier work before enqueueing capture, so a fast capture cannot
    // overtake a pending navigation/settings command on another worker.
    fractal.GetRenderPool()->Drain();
    // packaged_task transports capture errors to the UI thread without letting
    // validation failures escape into the render worker's fatal-error boundary.
    // Waiting keeps this stack-owned task alive until the callback has returned.
    std::packaged_task<std::string(Fractal &)> capture(SerializeMovieConfiguration);
    auto result = capture.get_future();
    fractal
        .EnqueueMutation("capture zoom movie settings",
                         [&capture](Fractal &current) { capture(current); })
        .Wait();
    if (result.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        throw FractalSharkSeriousException("Movie export was cancelled during renderer shutdown");
    }
    return result.get();
}

std::optional<std::filesystem::path>
FindMovieCliExecutable(const std::filesystem::path &guiExecutable, MovieCommandShell shell)
{
    const auto directory = std::filesystem::absolute(guiExecutable).parent_path();
    const auto name = shell == MovieCommandShell::PowerShell ? "FractalSharkCli.exe" : "FractalSharkCli";
    const auto sibling = directory / name;
    if (std::filesystem::is_regular_file(sibling)) {
        return sibling;
    }
    const auto cmakeTarget = directory.parent_path() / "FractalSharkCli" / name;
    if (std::filesystem::is_regular_file(cmakeTarget)) {
        return cmakeTarget;
    }
    return std::nullopt;
}

MovieCommandExport
BuildMovieCommandExport(const std::filesystem::path &configuration,
                        const std::filesystem::path &cliExecutable,
                        MovieCommandShell shell)
{
    if (configuration.empty() || cliExecutable.empty()) {
        throw FractalSharkSeriousException("Movie export requires configuration and CLI paths");
    }
    MovieCommandExport output;
    const auto chosen = std::filesystem::absolute(configuration).lexically_normal();
    output.JobDirectory =
        chosen.parent_path() / (chosen.stem().native() + std::filesystem::path(".movie").native());
    output.ConfigurationPath = output.JobDirectory / "movie.ini";
    const auto cliPath = std::filesystem::absolute(cliExecutable).lexically_normal();
    output.Command = shell == MovieCommandShell::PowerShell ? "& " : "";
    output.Command += QuoteMovieArgument(cliPath, shell) + " --movie " +
                      QuoteMovieArgument(output.JobDirectory, shell);
    return output;
}

void
SaveMovieConfiguration(const MovieCommandExport &output, std::string_view configuration)
{
    std::error_code error;
    if (!std::filesystem::create_directory(output.JobDirectory, error)) {
        throw FractalSharkSeriousException("Could not create new movie job folder: " +
                                           (error ? error.message() : "destination already exists"));
    }
    struct UnfinishedExport {
        const MovieCommandExport &Output;
        bool Complete = false;
        bool ConfigurationWritten = false;
        ~UnfinishedExport()
        {
            if (!Complete) {
                // Own only these newly created files. Never recursively delete
                // a directory that another process might have started using.
                std::error_code ignored;
                if (ConfigurationWritten) {
                    std::filesystem::remove(Output.ConfigurationPath, ignored);
                }
                std::filesystem::remove(Output.JobDirectory, ignored);
            }
        }
    } cleanup{output};
    const auto write = [](const std::filesystem::path &path, std::string_view text) {
        const auto result = SaveFileTransaction::Write(Environment::FromFsPath(path),
                                                       SaveOverwritePolicy::SkipExisting,
                                                       [text](std::ostream &stream) { stream << text; });
        if (!result.Succeeded() || result.Outcome != SaveOutcome::Written) {
            throw FractalSharkSeriousException("Could not save movie job: " + result.Diagnostic);
        }
    };
    std::ostringstream contents;
    contents << configuration;
    if (!configuration.ends_with('\n')) {
        contents << '\n';
    }
    const auto video = PathUtf8(output.ConfigurationPath.parent_path() / "movie.ts");
    if (video.find_first_of("\r\n") != std::string::npos || video.find('\0') != std::string::npos) {
        throw FractalSharkSeriousException("Movie output path must fit on one INI line");
    }
    WriteMovieSetting(contents, "video-output", video);
    write(output.ConfigurationPath, contents.str());
    cleanup.ConfigurationWritten = true;
    cleanup.Complete = true;
}

std::string
DescribeMovieCommandExport(const MovieCommandExport &output, bool copied)
{
    return std::string(copied ? "Command copied to clipboard.\n\n"
                              : "Could not copy to clipboard. Copy this command manually.\n\n") +
           output.Command + "\n\nConfiguration: " + PathUtf8(output.ConfigurationPath) +
           "\nMovie job: " + PathUtf8(output.JobDirectory) +
           "\n\nEdit movie.ini inside the job folder before starting to change the defaults:\n" +
           std::string(MovieDefaultZoomPerFrame) + "x zoom per frame; " +
           std::to_string(MovieVideoSettings::DefaultFps) +
           " fps; 1000-frame IDR interval.\n"
           "Current size, antialiasing, and final iteration limit; Auto algorithms; default palette.\n"
           "Run the command to render, or rerun it to resume.\n"
           "Video encoding requires a compatible NVIDIA driver and GPU.";
}

bool
CopyMovieCommandToClipboard(const MovieCommandExport &output)
{
    // The INI is already published. Clipboard ownership can fail (or the Linux
    // provider can throw); preserve the manual-copy recovery path in either case.
    try {
        if (Environment::SetClipboardText(output.Command)) {
            return true;
        }
    } catch (const std::exception &exception) {
        FractalSharkLog::WriteException(
            "Could not copy zoom movie command", exception, __FILE__, __LINE__);
        return false;
    }
    FractalSharkLog::Write("Could not copy zoom movie command to clipboard", __FILE__, __LINE__);
    return false;
}

} // namespace FractalShark
