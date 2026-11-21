#include "stdafx.h"

#include "MovieCommandExport.h"

#include "ConsoleLog.h"
#include "Environment.h"
#include "Exceptions.h"
#include "Fractal.h"
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
    // Coordinates stay in the INI at their full stored precision. In particular,
    // do not use the shortened zoom string from the rendering-details dialog.
    output << "[movie]\ncenter-x=" << view.GetPtX() << "\ncenter-y=" << view.GetPtY()
           << "\nzoom=" << view.GetZoomFactor() << "\nzoom-per-frame=1.01\niterations=" << iterations
           << "\nwidth=" << width << "\nheight=" << height
           << "\nantialiasing=" << fractal.GetGpuAntialiasing()
           << "\nrender-algorithm=AutoSelect\nperturbation-alg=Auto\nwide-iterations="
           << std::min<uint64_t>(MovieWideIterations, iterations)
           << "\nla-slope-ratio=8\nvideo-chunk-frames=1000\nvideo-fps=60\n"
           << "render-order=wide-to-deep\nvideo-encoder-args=" << DefaultMovieEncoderArguments() << '\n'
           << MovieEncoderExamples();
    return output.str();
}

} // namespace

std::string_view
DefaultMovieEncoderArguments()
{
    return "-c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 16 -profile:v main -pix_fmt nv12 "
           "-colorspace bt709 -color_range tv";
}

std::string_view
LosslessMovieEncoderArguments()
{
    return "-c:v hevc_nvenc -preset p4 -tune lossless -profile:v rext -pix_fmt gbrp "
           "-colorspace rgb -color_range pc";
}

std::string
MovieEncoderExamples()
{
    return "\n# Choose one video-encoder-args line before starting a new job.\n"
           "# Lossless HEVC (RGB8 after deliberate RGB16 rounding):\n"
           "# video-encoder-args=" +
           std::string(LosslessMovieEncoderArguments()) +
           "\n# HEVC high quality (default, QP 16):\n"
           "# video-encoder-args=" +
           std::string(DefaultMovieEncoderArguments()) +
           "\n# HEVC low quality (smaller files, QP 22):\n"
           "# video-encoder-args=-c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 22 -profile:v "
           "main -pix_fmt nv12 -colorspace bt709 -color_range tv\n"
           "# H.264 high quality (3840x2160 works; check your encoder's size limits):\n"
           "# video-encoder-args=-c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 18 -b:v 0 -profile:v "
           "high -pix_fmt nv12\n"
           "# H.264 low quality / faster:\n"
           "# video-encoder-args=-c:v h264_nvenc -preset p1 -tune hq -rc vbr -cq 30 -b:v 0 -profile:v "
           "high -pix_fmt nv12\n"
           "# NV12/yuv420p output uses BT.709 limited range. RGB uses full range.\n";
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
                        const std::filesystem::path &ffmpegExecutable,
                        MovieCommandShell shell)
{
    if (configuration.empty() || cliExecutable.empty() || ffmpegExecutable.empty()) {
        throw FractalSharkSeriousException("Movie export requires configuration, CLI, and FFmpeg paths");
    }
    MovieCommandExport output;
    const auto chosen = std::filesystem::absolute(configuration).lexically_normal();
    output.JobDirectory =
        chosen.parent_path() / (chosen.stem().native() + std::filesystem::path(".movie").native());
    output.ConfigurationPath = output.JobDirectory / "movie.ini";
    const auto cliPath = std::filesystem::absolute(cliExecutable).lexically_normal();
    output.FfmpegExecutable = std::filesystem::absolute(ffmpegExecutable).lexically_normal();
    // The saved encoder path must obey the same single-line contract as command
    // arguments even though it no longer appears in the generated command.
    QuoteMovieArgument(output.FfmpegExecutable, shell);
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
        bool EncoderWritten = false;
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
                if (EncoderWritten) {
                    std::filesystem::remove(Output.JobDirectory / "ffmpeg.txt", ignored);
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
    write(output.JobDirectory / "ffmpeg.txt", PathUtf8(output.FfmpegExecutable));
    cleanup.EncoderWritten = true;
    // Publish the config last: a folder-only CLI invocation requires it.
    write(output.ConfigurationPath, configuration);
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
           "\nFFmpeg: " + PathUtf8(output.FfmpegExecutable) +
           "\n\nEdit movie.ini inside the job folder before starting to change the defaults:\n"
           "1.01x zoom per frame; 60 fps; 1000-frame chunks.\n"
           "Current size, antialiasing, and final iteration limit; Auto algorithms; default palette.\n"
           "Run the command to render, or rerun it to resume.\n"
           "FFmpeg must support the selected NVENC encoder (HEVC by default).";
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
