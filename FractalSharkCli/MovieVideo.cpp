#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieHash.h"
#include "MovieVideo.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <iomanip>
#include <limits>
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
    // The process boundary uses UTF-8, including on Windows.
    return Environment::WideToUtf8(Environment::FromFsPath(path));
}
std::string
SequencePattern(const std::string &prefix, const std::string &suffix)
{
    std::string result;
    for (const char ch : prefix) {
        result += ch;
        if (ch == '%') {
            result += '%';
        }
    }
    // Indices are at most INT_MAX (ten digits). FFmpeg 6 truncates a %020d
    // expansion to 19 characters internally; a literal ten-zero prefix plus
    // %010d addresses the identical 20-digit files on old and new versions.
    return result + "0000000000%010d" + suffix;
}
class TemporaryMovies {
public:
    std::vector<std::filesystem::path> Paths;
    ~TemporaryMovies()
    {
        for (const auto &path : Paths) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    }
};
class TemporaryMovieDirectory {
public:
    explicit TemporaryMovieDirectory(const std::filesystem::path &parent)
        : m_Path(UniqueTemporaryMovie(parent))
    {
        if (!std::filesystem::create_directory(m_Path)) {
            throw FractalSharkSeriousException("Cannot create private movie input sequence");
        }
    }
    TemporaryMovieDirectory(const TemporaryMovieDirectory &) = delete;
    TemporaryMovieDirectory &operator=(const TemporaryMovieDirectory &) = delete;
    ~TemporaryMovieDirectory()
    {
        // Only this uniquely created, reserved directory is owned here. Hard
        // links share PNG data; removing them never removes the source names.
        std::error_code ignored;
        std::filesystem::remove_all(m_Path, ignored);
    }
    const std::filesystem::path &
    Path() const
    {
        return m_Path;
    }

private:
    std::filesystem::path m_Path;
};
void
RequireEncoderSuccess(const Environment::MovieProcessResult &result)
{
    if (result.Cancelled) {
        throw MovieInterrupted();
    }
    if (result.ExitCode != 0) {
        FractalSharkLog::Write(result.ErrorOutput, __FILE__, __LINE__);
        throw FractalSharkSeriousException("FFmpeg failed; source PNGs were retained");
    }
}
class PixelDigest {
public:
    void
    Consume(std::span<const unsigned char> bytes)
    {
        constexpr size_t MaximumDigestOutput = 128;
        if (bytes.size() > MaximumDigestOutput - m_Text.size()) {
            throw FractalSharkSeriousException("Unexpected FFmpeg hash output");
        }
        m_Text.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
    std::string
    Finish() const
    {
        constexpr std::string_view Prefix = "SHA256=";
        auto text = std::string_view(m_Text);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
            text.remove_suffix(1);
        }
        if (!text.starts_with(Prefix) || text.size() != Prefix.size() + 64) {
            throw FractalSharkSeriousException("FFmpeg did not return one SHA-256 digest");
        }
        const auto hash = text.substr(Prefix.size());
        if (hash.find_first_not_of("0123456789abcdef") != std::string_view::npos) {
            throw FractalSharkSeriousException("Invalid FFmpeg SHA-256 digest");
        }
        return std::string(hash);
    }

private:
    std::string m_Text;
};
class FrameInformation {
public:
    FrameInformation(unsigned width, unsigned height)
        : m_SizeMarker(" s:" + std::to_string(width) + "x" + std::to_string(height) + " ")
    {
    }
    void
    Consume(std::string_view text)
    {
        m_Pending += text;
        size_t newline;
        while ((newline = m_Pending.find('\n')) != std::string::npos) {
            const auto line = m_Pending.substr(0, newline);
            m_Pending.erase(0, newline + 1);
            if (line.find("showinfo") != std::string::npos &&
                line.find(" iskey:") != std::string::npos) {
                if (line.find(m_SizeMarker) == std::string::npos) {
                    throw FractalSharkSeriousException("Movie frame dimensions changed");
                }
                const bool key = line.find(" iskey:1") != std::string::npos;
                if (Frames == 0) {
                    FirstKey = key;
                }
                ++Frames;
                Keyframes += key ? 1 : 0;
            }
        }
        constexpr size_t MaximumDiagnosticLine = 65536;
        if (m_Pending.size() > MaximumDiagnosticLine) {
            throw FractalSharkSeriousException("Unexpected FFmpeg frame information");
        }
    }
    uint64_t Frames = 0;
    uint64_t Keyframes = 0;
    bool FirstKey = false;

private:
    std::string m_SizeMarker;
    std::string m_Pending;
};
} // namespace

MovieTileGrid
PlanMovieTiles(unsigned width, unsigned height, unsigned encoderThreads)
{
    if (width == 0 || height == 0 || encoderThreads == 0) {
        throw FractalSharkSeriousException("Invalid movie tile planning dimensions or threads");
    }
    // AV1 limits tile width to 4096 pixels and area to 4096*2304 pixels.
    // Round to the largest possible superblock (128) before deriving the
    // mandatory grid, matching FFmpeg's conservative area calculation.
    constexpr uint64_t Superblock = 128;
    constexpr uint64_t MaximumTileWidth = 4096;
    constexpr uint64_t MaximumTileArea = MaximumTileWidth * 2304;
    constexpr unsigned MaximumAxisTiles = 64;
    constexpr unsigned MaximumTiles = 512;
    const auto alignedWidth = ((uint64_t{width} + Superblock - 1) / Superblock) * Superblock;
    const auto alignedHeight = ((uint64_t{height} + Superblock - 1) / Superblock) * Superblock;
    const auto columns = std::bit_ceil((alignedWidth + MaximumTileWidth - 1) / MaximumTileWidth);
    const auto tileWidth = ((alignedWidth / columns + Superblock - 1) / Superblock) * Superblock;
    const auto rows = std::bit_ceil((tileWidth * alignedHeight + MaximumTileArea - 1) / MaximumTileArea);
    if (columns > MaximumAxisTiles || rows > MaximumAxisTiles || columns * rows > MaximumTiles) {
        throw FractalSharkSeriousException("Movie dimensions exceed AV1 tile limits");
    }
    MovieTileGrid grid{static_cast<unsigned>(columns), static_cast<unsigned>(rows)};
    // Optional parallel tiles must each have room for a full 128-pixel block.
    // Tiny images stay 1x1; do not force the production grid onto preflight.
    const auto maxColumns = std::min(MaximumAxisTiles, std::max(1u, width / unsigned{Superblock}));
    const auto maxRows = std::min(MaximumAxisTiles, std::max(1u, height / unsigned{Superblock}));
    const auto target = std::bit_ceil(std::min(encoderThreads, MaximumTiles));
    bool columnsNext = true;
    while (grid.Columns * grid.Rows < target) {
        const bool canColumns = grid.Columns * 2 <= maxColumns;
        const bool canRows = grid.Rows * 2 <= maxRows;
        if (!canColumns && !canRows) {
            break;
        }
        if ((columnsNext && canColumns) || !canRows) {
            grid.Columns *= 2;
            columnsNext = false;
        } else {
            grid.Rows *= 2;
            columnsNext = true;
        }
    }
    return grid;
}

std::vector<MovieExecutionSettings>
PlanMovieExecution(unsigned width, unsigned height, unsigned logicalProcessors)
{
    constexpr unsigned MaximumWorkers = 4;
    constexpr unsigned MaximumEncoderThreads = 64; // libaom's FFmpeg thread limit.
    const auto budget = logicalProcessors > 1 ? logicalProcessors - 1 : 1;
    const auto slots = std::min(MaximumWorkers, budget);
    const auto encoderThreads = std::min(budget, MaximumEncoderThreads);
    const auto tiles = PlanMovieTiles(width, height, encoderThreads);
    std::vector<MovieExecutionSettings> execution;
    for (unsigned slot = 0; slot < slots; ++slot) {
        const auto decodeThreads = budget / slots + (slot < budget % slots ? 1 : 0);
        // Only decode/filter reservations are shared. Dividing the encoder
        // allowance as well recreates the serialization this policy avoids.
        execution.push_back({slot, decodeThreads, decodeThreads, encoderThreads, tiles});
    }
    return execution;
}

MovieVideo::MovieVideo(const std::filesystem::path &ffmpeg,
                       unsigned width,
                       unsigned height,
                       unsigned fps,
                       const MovieExecutionSettings &execution,
                       const std::function<bool()> &stop)
    : m_Ffmpeg(ffmpeg), m_Width(width), m_Height(height), m_Fps(fps), m_Execution(execution),
      m_Stop(stop)
{
    if (width == 0 || height == 0 || uint64_t{width} * height > SIZE_MAX / 6 ||
        execution.DecoderThreads == 0 || execution.FilterThreads == 0 || execution.EncoderThreads == 0 ||
        execution.Tiles.Columns == 0 || execution.Tiles.Rows == 0) {
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
                   const std::string &pattern,
                   uint64_t first,
                   uint64_t count) const
{
    CheckStop();
    const auto keyInterval = std::to_string(count);
    const auto decoderThreads = std::to_string(std::min<uint64_t>(count, m_Execution.DecoderThreads));
    // q*256 is deliberate: the following bit-exact 16->8 conversion discards
    // the low byte. q*257 would round a second time and change RGB8 identity.
    const std::string conversion =
        "[0:v]trim=end_frame=" + std::to_string(count) +
        ",showinfo=checksum=0,format=gbrp16le,"
        "lutrgb=r='floor((val+128)/257)*256':g='floor((val+128)/257)*256':"
        "b='floor((val+128)/257)*256',"
        "scale=iw:ih:flags=bitexact:sws_dither=none,format=rgb24,split=2[video][hash]";
    const std::vector<std::string> arguments{
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-xerror",
        "-y",
        "-filter_complex_threads",
        std::to_string(m_Execution.FilterThreads),
        "-f",
        "image2",
        "-pattern_type",
        "sequence",
        "-c:v",
        "png",
        "-threads:v",
        decoderThreads,
        "-err_detect",
        "crccheck+explode",
        "-reinit_filter",
        "0",
        "-framerate",
        std::to_string(m_Fps),
        "-start_number",
        std::to_string(first),
        "-start_number_range",
        "1",
        "-i",
        pattern,
        "-filter_complex",
        conversion,
        "-map",
        "[video]",
        "-an",
        "-frames:v",
        keyInterval,
        "-c:v",
        "libaom-av1",
        "-threads:v",
        std::to_string(m_Execution.EncoderThreads),
        "-tiles",
        std::to_string(m_Execution.Tiles.Columns) + "x" + std::to_string(m_Execution.Tiles.Rows),
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
        PathArgument(path),
        "-map",
        "[hash]",
        "-an",
        "-frames:v",
        keyInterval,
        "-c:v",
        "rawvideo",
        "-threads:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-fps_mode",
        "passthrough",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "pipe:1"};
    PixelDigest source;
    FrameInformation frames(m_Width, m_Height);
    const auto result = Environment::RunMovieProcess(
        m_Ffmpeg,
        arguments,
        {},
        [&](std::span<const unsigned char> bytes) { source.Consume(bytes); },
        [&](std::string_view text) { frames.Consume(text); },
        m_Stop);
    RequireEncoderSuccess(result);
    if (frames.Frames != count) {
        throw FractalSharkSeriousException("Movie PNG sequence frame count mismatch");
    }
    return source.Finish();
}
void
MovieVideo::Verify(const std::filesystem::path &path,
                   uint64_t count,
                   const std::string &expectedHash) const
{
    CheckStop();
    PixelDigest decoded;
    FrameInformation frames(m_Width, m_Height);
    const auto threads = std::to_string(std::min<uint64_t>(count, m_Execution.DecoderThreads));
    const std::vector<std::string> arguments{"-hide_banner",
                                             "-nostdin",
                                             "-nostats",
                                             "-loglevel",
                                             "info",
                                             "-xerror",
                                             "-threads:v",
                                             threads,
                                             "-err_detect",
                                             "crccheck+explode",
                                             "-i",
                                             PathArgument(path),
                                             "-map",
                                             "0:v:0",
                                             "-an",
                                             "-filter_threads",
                                             std::to_string(m_Execution.FilterThreads),
                                             "-vf",
                                             "format=rgb24,showinfo=checksum=0",
                                             "-pix_fmt",
                                             "rgb24",
                                             "-c:v",
                                             "rawvideo",
                                             "-threads:v",
                                             "1",
                                             "-fps_mode",
                                             "passthrough",
                                             "-f",
                                             "hash",
                                             "-hash",
                                             "sha256",
                                             "pipe:1"};
    const auto result = Environment::RunMovieProcess(
        m_Ffmpeg,
        arguments,
        {},
        [&](std::span<const unsigned char> bytes) { decoded.Consume(bytes); },
        [&](std::string_view text) { frames.Consume(text); },
        m_Stop);
    RequireEncoderSuccess(result);
    if (frames.Frames != count || frames.Keyframes != 1 || !frames.FirstKey ||
        decoded.Finish() != expectedHash) {
        throw FractalSharkSeriousException(
            "Movie verification failed (RGB8, count, or keyframe mismatch)");
    }
}
void
MovieVideo::Preflight(const std::filesystem::path &directory) const
{
    CheckStop();
    // Exercise the same PNG input, rounding, and dual-output graph used by real
    // chunks. Only four tiny synthetic frames are generated/hashed in C++.
    constexpr unsigned ProbeSize = 32;
    constexpr uint64_t ProbeFrames = 4;
    auto execution = m_Execution;
    execution.Tiles = PlanMovieTiles(ProbeSize, ProbeSize, execution.EncoderThreads);
    MovieVideo probe(m_Ffmpeg, ProbeSize, ProbeSize, m_Fps, execution, m_Stop);
    const auto path = UniqueTemporaryMovie(directory);
    TemporaryMovies cleanup{{path}};
    const TemporaryMovieDirectory inputs(directory);
    const auto prefix = PathArgument(path.parent_path() / path.stem()) + "-";
    MovieHash expected;
    for (uint64_t frame = 0; frame < ProbeFrames; ++frame) {
        std::vector<unsigned char> pixels(ProbeSize * ProbeSize * 6);
        std::vector<unsigned char> rgb8(pixels.size() / 2);
        for (size_t i = 0; i < rgb8.size(); ++i) {
            const auto value = static_cast<unsigned>((i * 419 + frame * 977) & 65535);
            pixels[2 * i] = static_cast<unsigned char>(value >> 8);
            pixels[2 * i + 1] = static_cast<unsigned char>(value);
            rgb8[i] = static_cast<unsigned char>((value + 128) / 257);
        }
        expected.Update(rgb8);
        std::vector<unsigned char> png;
        if (lodepng::encode(png, pixels, ProbeSize, ProbeSize, LCT_RGB, 16) != 0) {
            throw FractalSharkSeriousException("Cannot create movie preflight PNG");
        }
        std::ostringstream number;
        number << std::setw(20) << std::setfill('0') << frame;
        const std::filesystem::path pngPath(
            Environment::ToFsPath(Environment::Utf8ToWide(prefix + number.str() + ".tmp")));
        cleanup.Paths.push_back(pngPath);
        WriteMovieText(pngPath, std::string(reinterpret_cast<const char *>(png.data()), png.size()));
        std::filesystem::create_hard_link(pngPath, inputs.Path() / pngPath.filename());
    }
    const auto inputPrefix = PathArgument(inputs.Path() / path.stem()) + "-";
    const auto hash = probe.Encode(path, SequencePattern(inputPrefix, ".tmp"), 0, ProbeFrames);
    if (hash != expected.Finish()) {
        throw FractalSharkSeriousException("FFmpeg RGB16 conversion differs from movie RGB8 rounding");
    }
    probe.Verify(path, ProbeFrames, hash);
}
MovieChunk
MovieVideo::Archive(const MovieJob &job, uint64_t first, uint64_t count) const
{
    CheckStop();
    if (count == 0 || first > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        count - 1 > static_cast<uint64_t>(std::numeric_limits<int>::max()) - first ||
        job.FramePath(first).parent_path() != job.FramePath(first + count - 1).parent_path()) {
        throw FractalSharkSeriousException("Movie chunk exceeds the numbered PNG sequence limits");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto target = job.VideoPath(first);
    const auto temporary = UniqueTemporaryMovie(target.parent_path());
    TemporaryMovies cleanup{{temporary}};
    const TemporaryMovieDirectory inputs(target.parent_path());
    uint64_t pngBytes = 0;
    for (uint64_t offset = 0; offset < count; ++offset) {
        CheckStop();
        const auto source = job.FramePath(first + offset);
        pngBytes += std::filesystem::file_size(source);
        // image2 probes/prefetches beyond output -frames:v and filter trim.
        // A private directory supplies a real EOF at this chunk's boundary,
        // so another worker may safely delete neighboring source PNGs. These
        // are metadata-only hard links, never image copies or CPU decodes.
        std::filesystem::create_hard_link(source, inputs.Path() / source.filename());
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie encoding slot=" << m_Execution.Slot << " chunk=" << first << " frames=" << count
        << " decoder_threads=" << std::min<uint64_t>(count, m_Execution.DecoderThreads)
        << " filter_threads=" << m_Execution.FilterThreads
        << " encoder_threads=" << m_Execution.EncoderThreads << " tiles=" << m_Execution.Tiles.Columns
        << 'x' << m_Execution.Tiles.Rows;
    const auto pattern = SequencePattern(PathArgument(inputs.Path()) + "/", ".png");
    const auto hash = Encode(temporary, pattern, first, count);
    const auto encoded = std::chrono::steady_clock::now();
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie verifying slot=" << m_Execution.Slot << " chunk=" << first << " frames=" << count;
    Verify(temporary, count, hash);
    const auto verified = std::chrono::steady_clock::now();
    CheckStop();
    // The registry is published by the caller only after durable publication.
    // Until then retained PNGs remain authoritative, including on resume.
    Environment::PublishMovieFile(temporary, target, true);
    const auto videoBytes = std::filesystem::file_size(target);
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie chunk " << first << " frames=" << count << " png_bytes=" << pngBytes
        << " video_bytes=" << videoBytes
        << " encode_seconds=" << std::chrono::duration<double>(encoded - started).count()
        << " verify_seconds=" << std::chrono::duration<double>(verified - encoded).count() << " seconds="
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return {first, count, pngBytes, videoBytes, hash};
}
} // namespace FractalSharkCli
