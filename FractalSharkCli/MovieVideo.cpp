#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieVideo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

namespace FractalSharkCli {
namespace {
// I/P mode requires an initial I slot and a potential P slot. Singleton movies
// still contain only the initial keyframe because their output limit is one.
constexpr uint64_t MinimumNvencGopLength = 2;
constexpr size_t TransportPacketBytes = 188;
constexpr unsigned TransportVideoPid = 256;

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
    // Ten digits cover the supported signed-int sequence indices, including
    // FFmpeg versions whose image2 formatter cannot expand twenty digits.
    return result + "%010d" + suffix;
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
RequireFfmpegSuccess(const Environment::MovieProcessResult &result, const char *failure)
{
    if (result.Cancelled) {
        throw MovieInterrupted();
    }
    if (result.ExitCode != 0) {
        FractalSharkLog::Write(result.ErrorOutput, __FILE__, __LINE__);
        throw FractalSharkSeriousException(failure);
    }
}
// FFmpeg's final progress block confirms output count without decoding the
// resulting movie. Pipe reads may split lines anywhere; retain only one line.
class EncodeProgress {
public:
    void
    Consume(std::span<const unsigned char> bytes)
    {
        for (const unsigned char byte : bytes) {
            if (byte == '\n') {
                const auto line = std::string_view(m_Line);
                if (line.starts_with("frame=")) {
                    const auto value = line.substr(6);
                    uint64_t frames = 0;
                    const auto parsed =
                        std::from_chars(value.data(), value.data() + value.size(), frames);
                    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
                        throw FractalSharkSeriousException("Invalid FFmpeg frame progress");
                    }
                    Frames = frames;
                } else if (line == "progress=end") {
                    Ended = true;
                }
                m_Line.clear();
            } else if (byte != '\r') {
                constexpr size_t MaximumProgressLine = 1024;
                if (m_Line.size() == MaximumProgressLine) {
                    throw FractalSharkSeriousException("Unexpected FFmpeg progress output");
                }
                m_Line += static_cast<char>(byte);
            }
        }
    }
    uint64_t Frames = 0;
    bool Ended = false;

private:
    std::string m_Line;
};
// FFmpeg 6.1 omits frame= from stream-copy progress. The MPEG-TS muxer
// emits one PES start for each copied video packet. Count those starts at
// the chosen PID while writing, without decoding or rereading the stream.
class TransportFrameCount {
public:
    void
    Consume(std::span<const unsigned char> bytes)
    {
        // A pipe read can cut a packet in two. Only those packets need copying
        // into scratch space. Full packets only require their three header bytes.
        if (m_PendingSize != 0) {
            const auto copied = std::min(bytes.size(), TransportPacketBytes - m_PendingSize);
            std::copy_n(bytes.begin(), copied, m_Pending.begin() + m_PendingSize);
            bytes = bytes.subspan(copied);
            m_PendingSize += copied;
            if (m_PendingSize != TransportPacketBytes) {
                return;
            }
            CountPacket(m_Pending.data());
            m_PendingSize = 0;
        }
        while (bytes.size() >= TransportPacketBytes) {
            CountPacket(bytes.data());
            bytes = bytes.subspan(TransportPacketBytes);
        }
        std::copy(bytes.begin(), bytes.end(), m_Pending.begin());
        m_PendingSize = bytes.size();
    }

    uint64_t Frames = 0;

private:
    void
    CountPacket(const unsigned char *header)
    {
        constexpr unsigned char SyncByte = 0x47;
        constexpr unsigned char PayloadStartMask = 0x40;
        constexpr unsigned char PidHighMask = 0x1f;
        if (header[0] != SyncByte) {
            throw FractalSharkSeriousException("Invalid MPEG-TS packet in movie assembly");
        }
        const auto pid = (static_cast<unsigned>(header[1] & PidHighMask) << 8) | header[2];
        Frames += pid == TransportVideoPid && (header[1] & PayloadStartMask) != 0;
    }

    std::array<unsigned char, TransportPacketBytes> m_Pending{};
    size_t m_PendingSize = 0;
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
                ++Frames;
            }
        }
        constexpr size_t MaximumDiagnosticLine = 65536;
        if (m_Pending.size() > MaximumDiagnosticLine) {
            throw FractalSharkSeriousException("Unexpected FFmpeg frame information");
        }
    }
    uint64_t Frames = 0;

private:
    std::string m_SizeMarker;
    std::string m_Pending;
};
} // namespace

void
AssembleMovie(MovieJob &job, const std::filesystem::path &ffmpeg, const std::function<bool()> &stop)
{
    const auto checkStop = [&] {
        job.CheckTiming();
        if (stop()) {
            throw MovieInterrupted();
        }
    };
    checkStop();
    const auto partial = job.Directory() / "movie.ts.partial";
    const auto target = job.Directory() / "movie.ts";
    const auto partialStatus = std::filesystem::symlink_status(partial);
    const auto targetStatus = std::filesystem::symlink_status(target);
    const bool hasPartial = std::filesystem::exists(partialStatus);
    const bool hasTarget = std::filesystem::exists(targetStatus);
    const auto chunks = job.ChunkCount();
    auto progress = job.ReadAssembly();
    if (!progress) {
        if (hasPartial || hasTarget) {
            throw FractalSharkSeriousException(
                "Movie assembly output already exists without a checkpoint");
        }
        if (ffmpeg.empty() || !job.HasAllChunks()) {
            throw FractalSharkSeriousException(
                "Movie assembly requires all committed chunks and FFmpeg");
        }
        // Keep the complete output within one 33-bit MPEG timestamp interval
        // at 90 kHz, so seeking does not depend on wrap handling in the player.
        constexpr uint64_t TimestampWrap = uint64_t{1} << 33;
        constexpr uint64_t TimestampTicksPerSecond = 90000;
        if (job.FrameCount() * TimestampTicksPerSecond >= TimestampWrap * job.Config().Fps) {
            throw FractalSharkSeriousException(
                "Movie exceeds the MPEG-TS timestamp span; chunks retained");
        }
        progress = MovieAssemblyProgress{};
        // A zero-byte journal makes creation of the partial file retryable.
        job.SaveAssembly(*progress);
    }

    // Validate the committed prefix before touching any source. A successfully
    // published file also covers a crash after publication but before exit.
    if (progress->Bytes % TransportPacketBytes != 0 ||
        (hasTarget &&
         (progress->NextChunk != chunks || !std::filesystem::is_regular_file(targetStatus) ||
          std::filesystem::file_size(target) != progress->Bytes)) ||
        (hasPartial && !std::filesystem::is_regular_file(partialStatus)) ||
        (hasPartial && hasTarget && !std::filesystem::equivalent(partial, target)) ||
        (!hasTarget &&
         (hasPartial ? std::filesystem::file_size(partial) < progress->Bytes : progress->Bytes != 0))) {
        throw FractalSharkSeriousException("Movie assembly output does not match its checkpoint");
    }
    if (progress->NextChunk < chunks && ffmpeg.empty()) {
        throw FractalSharkSeriousException("Movie assembly requires FFmpeg; supply --ffmpeg PATH");
    }
    if (!hasPartial && !hasTarget) {
        // Publish the partial name durably before it can become the sole copy
        // of an appended chunk. The zero-byte journal allows retrying creation.
        const auto temporary = UniqueTemporaryMovie(job.Directory());
        TemporaryMovies cleanup{{temporary}};
        std::ofstream empty(temporary, std::ios::binary);
        empty.close();
        if (!empty) {
            throw FractalSharkSeriousException("Cannot create movie assembly output");
        }
        Environment::PublishMovieFile(temporary, partial, false);
    }
    // The manifest no longer describes the archive once a source is retired.
    std::filesystem::remove(job.Directory() / "movie.ffconcat");
    if (!hasTarget && hasPartial && std::filesystem::file_size(partial) > progress->Bytes) {
        std::filesystem::resize_file(partial, progress->Bytes);
    }
    for (uint64_t ordinal = 0; ordinal < progress->NextChunk; ++ordinal) {
        job.DeleteAssembledChunk(ordinal);
    }
    while (progress->NextChunk < chunks) {
        checkStop();
        const auto ordinal = progress->NextChunk;
        const auto first = job.PlaybackChunkStart(ordinal);
        const auto chunk = job.FindChunk(first);
        if (!chunk) {
            throw FractalSharkSeriousException("Movie assembly source chunk is missing");
        }
        job.DeleteChunkPngs(*chunk);
        // Both schedules start each chunk near zero on disk. Use its absolute
        // playback frame offset to avoid accumulating timestamp rounding.
        const auto frames = job.Config().WideToDeep ? first : job.FrameCount() - first - chunk->Count;
        std::ostringstream offset;
        offset.imbue(std::locale::classic());
        offset << std::fixed << std::setprecision(9)
               << static_cast<long double>(frames) / job.Config().Fps;
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Movie joining chunk " << ordinal + 1 << '/' << chunks << " source=" << first;

        // The renderer alone owns the partial file. FFmpeg writes through a
        // pipe, so an orphaned child cannot append to a resumed transaction.
        std::ofstream output(partial, std::ios::binary | std::ios::app);
        if (!output) {
            throw FractalSharkSeriousException("Cannot open movie assembly output");
        }
        TransportFrameCount copied;
        const auto result = Environment::RunMovieProcess(
            ffmpeg,
            {"-hide_banner",
             "-nostdin",
             "-nostats",
             "-loglevel",
             "error",
             "-xerror",
             "-copyts",
             "-start_at_zero",
             "-i",
             PathArgument(job.VideoPath(first)),
             "-map",
             "0:v:0",
             "-c:v",
             "copy",
             "-output_ts_offset",
             offset.str(),
             "-muxdelay",
             "0",
             "-muxpreload",
             "0",
             "-mpegts_copyts",
             "1",
             "-mpegts_flags",
             "+initial_discontinuity",
             "-mpegts_start_pid",
             std::to_string(TransportVideoPid),
             "-mpegts_m2ts_mode",
             "0",
             "-f",
             "mpegts",
             "pipe:1"},
            {},
            [&](std::span<const unsigned char> bytes) {
                copied.Consume(bytes);
                output.write(reinterpret_cast<const char *>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
                if (!output) {
                    throw FractalSharkSeriousException("Cannot write movie assembly output");
                }
            },
            {},
            [&] {
                job.CheckTiming();
                return stop();
            });
        output.close();
        RequireFfmpegSuccess(result, "FFmpeg stream copy failed; the source chunk was retained");
        const auto bytes = std::filesystem::file_size(partial);
        if (!output || copied.Frames != chunk->Count || bytes <= progress->Bytes ||
            bytes % TransportPacketBytes != 0) {
            throw FractalSharkSeriousException(
                "Movie assembly output or frame count mismatch; chunk retained");
        }
        checkStop();
        // Flush payload before advancing the journal. Until the journal is
        // durable the original MKV remains authoritative for this chunk.
        Environment::FlushMovieFile(partial);
        *progress = {ordinal + 1, bytes};
        job.SaveAssembly(*progress);
        job.DeleteAssembledChunk(ordinal);
    }

    checkStop();
    if (!hasTarget) {
        Environment::PublishMovieFile(partial, target, false);
    } else if (hasPartial) {
        // A no-replace POSIX publication may leave both names after a crash.
        Environment::FlushMovieFile(target);
        std::filesystem::remove(partial);
    }
    if (std::filesystem::is_empty(job.Directory() / "frames")) {
        std::filesystem::remove(job.Directory() / "frames");
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie complete: " << Environment::FromFsPath(target);
}

std::vector<MovieExecutionSettings>
PlanMovieExecution(unsigned logicalProcessors)
{
    constexpr unsigned MaximumWorkers = 4;
    const auto budget = logicalProcessors > 1 ? logicalProcessors - 1 : 1;
    const auto slots = std::min(MaximumWorkers, budget);
    std::vector<MovieExecutionSettings> execution;
    for (unsigned slot = 0; slot < slots; ++slot) {
        const auto decodeThreads = budget / slots + (slot < budget % slots ? 1 : 0);
        execution.push_back({slot, decodeThreads, decodeThreads});
    }
    return execution;
}

MovieVideo::MovieVideo(const std::filesystem::path &ffmpeg,
                       unsigned width,
                       unsigned height,
                       const MovieConfig &config,
                       const MovieExecutionSettings &execution,
                       const std::function<bool()> &stop)
    : m_Ffmpeg(ffmpeg), m_Width(width), m_Height(height), m_Fps(config.Fps),
      m_GopFrames(std::max(config.ChunkFrames, MinimumNvencGopLength)), m_WideToDeep(config.WideToDeep),
      m_PixelFormat(config.PixelFormat), m_EncoderArguments(config.EncoderArguments),
      m_Execution(execution), m_Stop(stop)
{
    if (width == 0 || height == 0 || uint64_t{width} * height > SIZE_MAX / 6 ||
        execution.DecoderThreads == 0 || execution.FilterThreads == 0) {
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
void
MovieVideo::Encode(const std::filesystem::path &path,
                   const std::string &pattern,
                   uint64_t first,
                   uint64_t count) const
{
    CheckStop();
    const auto frameLimit = std::to_string(count);
    const auto keyInterval = std::to_string(m_GopFrames);
    const auto decoderThreads = std::to_string(std::min<uint64_t>(count, m_Execution.DecoderThreads));
    // q*256 is deliberate: the following bit-exact 16->8 conversion discards
    // the low byte. q*257 would round a second time and change RGB8 identity.
    std::string conversion = "[0:v]trim=end_frame=" + std::to_string(count) +
                             ",showinfo=checksum=0,format=gbrp16le,"
                             "lutrgb=r='floor((val+128)/257)*256':g='floor((val+128)/257)*256':"
                             "b='floor((val+128)/257)*256',"
                             "scale=iw:ih:flags=bitexact:sws_dither=none,format=gbrp";
    if (m_PixelFormat != "gbrp") {
        conversion +=
            ",scale=iw:ih:in_range=pc:out_range=tv:out_color_matrix=bt709,format=" + m_PixelFormat;
    }
    conversion += "[video]";
    // The optional RGB lossless profile uses the documented NVENC tuning mode:
    // https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html#encoder-tuning-info-and-preset-configurations
    // FFmpeg maps -tune lossless to NV_ENC_TUNING_INFO_LOSSLESS:
    // https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/nvenc_hevc.c
    // Planar full-range RGB avoids chroma subsampling and color conversion.
    // Encoding preserves these RGB8 samples; the preceding 16->8 rounding is
    // deliberate precision reduction. Do not hash/redecode each archived movie.
    std::vector<std::string> arguments{"-hide_banner",
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
                                       frameLimit,
                                       "-bf",
                                       "0",
                                       "-rc-lookahead",
                                       "0",
                                       "-no-scenecut",
                                       "1",
                                       "-g",
                                       keyInterval,
                                       "-fps_mode",
                                       "passthrough",
                                       "-f",
                                       "matroska",
                                       PathArgument(path),
                                       "-progress",
                                       "pipe:1"};
    // Place the user-selected encoder options before the managed output options.
    const auto output = std::find(arguments.begin(), arguments.end(), "-bf");
    arguments.insert(output, m_EncoderArguments.begin(), m_EncoderArguments.end());
    EncodeProgress progress;
    FrameInformation frames(m_Width, m_Height);
    const auto result = Environment::RunMovieProcess(
        m_Ffmpeg,
        arguments,
        {},
        [&](std::span<const unsigned char> bytes) { progress.Consume(bytes); },
        [&](std::string_view text) { frames.Consume(text); },
        m_Stop);
    RequireFfmpegSuccess(result, "FFmpeg failed; source PNGs were retained");
    if (frames.Frames != count || progress.Frames != count || !progress.Ended ||
        std::filesystem::file_size(path) == 0) {
        throw FractalSharkSeriousException("Movie encode frame count or output mismatch; PNGs retained");
    }
}
void
MovieVideo::Preflight(const std::filesystem::path &directory) const
{
    CheckStop();
    // Device limits depend on the driver/GPU. Check the actual requested size
    // with a generated frame before spending time on fractal rendering.
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie preflight encoder preflight dimensions=" << m_Width << 'x' << m_Height;
    std::string probe = "color=black:s=" + std::to_string(m_Width) + "x" + std::to_string(m_Height) +
                        ":r=" + std::to_string(m_Fps) + ",format=gbrp";
    if (m_PixelFormat != "gbrp") {
        probe += ",scale=iw:ih:in_range=pc:out_range=tv:out_color_matrix=bt709,format=" + m_PixelFormat;
    }
    std::vector<std::string> arguments{"-hide_banner",
                                       "-nostdin",
                                       "-v",
                                       "verbose",
                                       "-filter_threads",
                                       std::to_string(m_Execution.FilterThreads),
                                       "-f",
                                       "lavfi",
                                       "-i",
                                       probe,
                                       "-frames:v",
                                       "1"};
    arguments.insert(arguments.end(), m_EncoderArguments.begin(), m_EncoderArguments.end());
    arguments.insert(arguments.end(),
                     {"-bf", "0", "-g", std::to_string(m_GopFrames), "-f", "null", "pipe:1"});
    const auto capabilities = Environment::RunMovieProcess(m_Ffmpeg, arguments, {}, {}, {}, m_Stop);
    RequireFfmpegSuccess(capabilities, "FFmpeg failed; source PNGs were retained");
    // Check hard-link support without touching an existing frame or copying PNGs.
    if (!m_WideToDeep) {
        const TemporaryMovieDirectory inputs(directory);
        const auto source = inputs.Path() / "source.tmp";
        WriteMovieText(source, "movie input link probe");
        std::filesystem::create_hard_link(source, inputs.Path() / "linked.tmp");
    }
}
MovieChunk
MovieVideo::Archive(const MovieJob &job, uint64_t first, uint64_t count) const
{
    CheckStop();
    if (count == 0 || first % job.Config().ChunkFrames != 0 || count > job.Config().ChunkFrames ||
        first > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        count - 1 > static_cast<uint64_t>(std::numeric_limits<int>::max()) - first ||
        job.FramePath(first).parent_path() != job.FramePath(first + count - 1).parent_path()) {
        throw FractalSharkSeriousException("Movie chunk exceeds the numbered PNG sequence limits");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto target = job.VideoPath(first);
    const auto temporary = UniqueTemporaryMovie(target.parent_path());
    TemporaryMovies cleanup{{temporary}};
    std::unique_ptr<TemporaryMovieDirectory> inputs;
    if (!m_WideToDeep) {
        inputs = std::make_unique<TemporaryMovieDirectory>(target.parent_path());
    }
    uint64_t pngBytes = 0;
    for (uint64_t offset = 0; offset < count; ++offset) {
        CheckStop();
        // New jobs read their own forward-only frame folder directly. Legacy
        // jobs need reversed hard links; both layouts isolate image2 prefetch
        // from neighboring chunks that another worker may delete.
        const auto source = job.FramePath(m_WideToDeep ? first + offset : first + count - 1 - offset);
        pngBytes += std::filesystem::file_size(source);
        std::ostringstream number;
        number << std::setw(10) << std::setfill('0') << first + offset << ".png";
        if (inputs) {
            std::filesystem::create_hard_link(source, inputs->Path() / number.str());
        }
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie encoding slot=" << m_Execution.Slot << " chunk=" << first / job.Config().ChunkFrames
        << " first_frame=" << first << " frames=" << count
        << " decoder_threads=" << std::min<uint64_t>(count, m_Execution.DecoderThreads)
        << " filter_threads=" << m_Execution.FilterThreads << " encoder_args=" <<
        [&] {
            std::string text;
            for (const auto &argument : m_EncoderArguments) {
                text += argument + " ";
            }
            return text;
        }()
        << "";
    const auto pattern =
        inputs ? SequencePattern(PathArgument(inputs->Path()) + "/", ".png")
               : SequencePattern(PathArgument(job.FramePath(first).parent_path()) + "/frame_", ".png");
    Encode(temporary, pattern, first, count);
    const auto encoded = std::chrono::steady_clock::now();
    CheckStop();
    // The registry is published by the caller only after durable publication.
    // Until then retained PNGs remain authoritative, including on resume.
    Environment::PublishMovieFile(temporary, target, true);
    const auto videoBytes = std::filesystem::file_size(target);
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Movie chunk " << first / job.Config().ChunkFrames << " first_frame=" << first
        << " frames=" << count << " png_bytes=" << pngBytes << " video_bytes=" << videoBytes
        << " encode_seconds=" << std::chrono::duration<double>(encoded - started).count() << " seconds="
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return {first, count, pngBytes, videoBytes};
}
} // namespace FractalSharkCli
