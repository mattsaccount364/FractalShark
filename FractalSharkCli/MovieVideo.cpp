#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieHash.h"
#include "MovieVideo.h"

#include <atomic>
#include <chrono>
#include <sstream>

namespace FractalSharkCli {
namespace {
std::filesystem::path
UniqueTemporaryMovie(const std::filesystem::path &directory)
{
    static std::atomic<uint64_t> sequence{0};
    for (;;) {
        const auto name = ".fractalshark-video-" + std::to_string(Environment::CurrentProcessId()) +
                          "-" + std::to_string(sequence.fetch_add(1)) + ".tmp";
        const auto path = directory / name;
        if (!std::filesystem::exists(path)) {
            return path;
        }
    }
}
std::string
PathArgument(const std::filesystem::path &path)
{
    // Process arguments are UTF-8 on the portable boundary; do not feed a
    // Windows code-page path into the UTF-8 -> UTF-16 process conversion.
    return Environment::WideToUtf8(Environment::FromFsPath(path));
}
class TemporaryMovie {
public:
    explicit TemporaryMovie(std::filesystem::path path) : m_Path(std::move(path)) {}
    ~TemporaryMovie()
    {
        std::error_code ignored;
        std::filesystem::remove(m_Path, ignored);
    }

private:
    std::filesystem::path m_Path;
};
void
RequireEncoderSuccess(const Environment::MovieProcessResult &result)
{
    if (result.ExitCode != 0) {
        FractalSharkLog::Write(result.ErrorOutput, __FILE__, __LINE__);
        throw FractalSharkSeriousException("FFmpeg failed; source PNGs were retained");
    }
}
} // namespace

MovieVideo::MovieVideo(const std::filesystem::path &ffmpeg,
                       unsigned width,
                       unsigned height,
                       unsigned fps,
                       const std::function<bool()> &stop)
    : m_Ffmpeg(ffmpeg), m_Width(width), m_Height(height), m_Fps(fps), m_Stop(stop)
{
    if (width == 0 || height == 0 || uint64_t{width} * height > SIZE_MAX / 6) {
        throw FractalSharkSeriousException("Invalid movie dimensions");
    }
}
void
MovieVideo::CheckStop() const
{
    if (m_Stop()) {
        throw MovieInterrupted();
    }
}
std::string
MovieVideo::Encode(const std::filesystem::path &path,
                   uint64_t count,
                   const std::function<std::vector<unsigned char>(uint64_t)> &frame) const
{
    const auto keyInterval = std::to_string(count);
    const std::vector<std::string> arguments{"-hide_banner",
                                             "-nostats",
                                             "-loglevel",
                                             "error",
                                             "-y",
                                             "-f",
                                             "rawvideo",
                                             "-pixel_format",
                                             "rgb24",
                                             "-video_size",
                                             std::to_string(m_Width) + "x" + std::to_string(m_Height),
                                             "-framerate",
                                             std::to_string(m_Fps),
                                             "-i",
                                             "pipe:0",
                                             "-an",
                                             "-c:v",
                                             "libaom-av1",
                                             "-profile:v",
                                             "1",
                                             "-usage",
                                             "good",
                                             "-cpu-used",
                                             "4",
                                             "-row-mt",
                                             "1",
                                             "-crf",
                                             "0",
                                             "-b:v",
                                             "0",
                                             "-aom-params",
                                             "lossless=1",
                                             "-pix_fmt",
                                             "gbrp",
                                             "-colorspace",
                                             "rgb",
                                             "-color_range",
                                             "pc",
                                             "-auto-alt-ref",
                                             "0",
                                             "-lag-in-frames",
                                             "0",
                                             "-g",
                                             keyInterval,
                                             "-keyint_min",
                                             keyInterval,
                                             "-fps_mode",
                                             "passthrough",
                                             "-f",
                                             "matroska",
                                             PathArgument(path)};
    MovieHash source;
    const auto result = Environment::RunMovieProcess(
        m_Ffmpeg,
        arguments,
        [&](const std::function<void(std::span<const unsigned char>)> &write) {
            for (uint64_t i = 0; i < count; ++i) {
                CheckStop();
                const auto pixels = frame(i);
                if (pixels.size() != uint64_t{m_Width} * m_Height * 3) {
                    throw FractalSharkSeriousException("Movie encoder received an invalid RGB8 frame");
                }
                source.Update(pixels);
                write(pixels);
            }
        },
        [](std::span<const unsigned char>) {},
        {});
    RequireEncoderSuccess(result);
    return source.Finish();
}
void
MovieVideo::Verify(const std::filesystem::path &path,
                   uint64_t count,
                   const std::string &expectedHash) const
{
    MovieHash decoded;
    uint64_t bytes = 0, frames = 0, keyframes = 0;
    std::string pending;
    const auto sizeMarker = " s:" + std::to_string(m_Width) + "x" + std::to_string(m_Height) + " ";
    const std::vector<std::string> arguments{"-hide_banner",
                                             "-nostdin",
                                             "-nostats",
                                             "-loglevel",
                                             "info",
                                             "-i",
                                             PathArgument(path),
                                             "-map",
                                             "0:v:0",
                                             "-an",
                                             "-vf",
                                             "format=rgb24,showinfo",
                                             "-pix_fmt",
                                             "rgb24",
                                             "-fps_mode",
                                             "passthrough",
                                             "-f",
                                             "rawvideo",
                                             "pipe:1"};
    const auto result = Environment::RunMovieProcess(
        m_Ffmpeg,
        arguments,
        [](const std::function<void(std::span<const unsigned char>)> &) {},
        [&](std::span<const unsigned char> data) {
            CheckStop();
            const auto frameBytes = uint64_t{m_Width} * m_Height * 3;
            if (count > UINT64_MAX / frameBytes || bytes > count * frameBytes ||
                data.size() > count * frameBytes - bytes) {
                throw FractalSharkSeriousException("Movie decoded too many pixels");
            }
            decoded.Update(data);
            bytes += data.size();
        },
        [&](std::string_view text) {
            pending += text;
            size_t newline;
            while ((newline = pending.find('\n')) != std::string::npos) {
                const auto line = pending.substr(0, newline);
                pending.erase(0, newline + 1);
                if (line.find("Parsed_showinfo_") != std::string::npos &&
                    line.find(" iskey:") != std::string::npos) {
                    if (line.find(sizeMarker) == std::string::npos) {
                        throw FractalSharkSeriousException("Movie decoded dimensions changed");
                    }
                    const bool key = line.find(" iskey:1") != std::string::npos;
                    if (frames == 0 && !key) {
                        throw FractalSharkSeriousException("Movie chunk does not start with a keyframe");
                    }
                    ++frames;
                    keyframes += key ? 1 : 0;
                }
            }
            if (pending.size() > 65536) {
                throw FractalSharkSeriousException("Unexpected FFmpeg verification output");
            }
        });
    RequireEncoderSuccess(result);
    if (frames != count || keyframes != 1 || bytes != uint64_t{m_Width} * m_Height * 3 * count ||
        decoded.Finish() != expectedHash) {
        throw FractalSharkSeriousException(
            "Movie verification failed (RGB8, count, or keyframe mismatch)");
    }
}
void
MovieVideo::Preflight(const std::filesystem::path &directory) const
{
    // Abrupt color changes exercise keyframe suppression and RGB identity.
    // Use a small independent instance instead of allocating production frames.
    MovieVideo probe(m_Ffmpeg, 32, 32, m_Fps, m_Stop);
    const auto path = UniqueTemporaryMovie(directory);
    TemporaryMovie cleanup(path);
    const auto hash = probe.Encode(path, 4, [](uint64_t frame) {
        std::vector<unsigned char> pixels(32 * 32 * 3);
        for (size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<unsigned char>((i * 17 + frame * 73) & 255);
        }
        return pixels;
    });
    probe.Verify(path, 4, hash);
}
MovieChunk
MovieVideo::Archive(const MovieJob &job, uint64_t first, uint64_t count) const
{
    const auto started = std::chrono::steady_clock::now();
    const auto target = job.VideoPath(first);
    // After a renderer crash the old child can briefly remain alive while its
    // stdin reaches EOF. Never let a resumed encoder share that child's output.
    const auto temporary = UniqueTemporaryMovie(target.parent_path());
    TemporaryMovie cleanup(temporary);
    uint64_t pngBytes = 0;
    const auto hash = Encode(temporary, count, [&](uint64_t offset) {
        const auto png = job.FramePath(first + offset);
        pngBytes += std::filesystem::file_size(png);
        return ReadMovieRgb8(png, m_Width, m_Height);
    });
    Verify(temporary, count, hash);
    CheckStop();
    // A published orphan still has its PNGs. Replacing it with this newly
    // verified encode is safe and avoids trusting an unregistered movie.
    Environment::PublishMovieFile(temporary, target, true);
    const auto videoBytes = std::filesystem::file_size(target);
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie chunk " << first << " frames=" << count << " png_bytes=" << pngBytes
        << " video_bytes=" << videoBytes << " seconds=" << seconds;
    return {first, count, pngBytes, videoBytes, hash};
}
} // namespace FractalSharkCli
