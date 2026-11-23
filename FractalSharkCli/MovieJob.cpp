#include "stdafx.h"

#include "ConsoleLog.h"
#include "Environment.h"
#include "MovieCommandExport.h"
#include "MovieJob.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <locale>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace FractalSharkCli {
namespace {
constexpr std::string_view MovieFormat = "FractalSharkNativeMovie3";
constexpr auto MovieStateName = "movie-state.ini";

struct MovieStaging {
    std::string Filename;
    std::string Owner;
};

struct MovieCheckpoint {
    uint64_t TotalFrames = 0;
    uint64_t CheckpointedFrames = 0;
    uint64_t VideoBytes = 0;
    uint64_t ActiveTimeMs = 0;
    bool GpuBypassed = false;
    long double LaDepth = 0;
    std::optional<MovieStaging> Staging;
};

std::string
Trim(std::string_view text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    return std::string(text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}
uint64_t
ParseUnsigned(const std::string &text, const char *name)
{
    uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || result == 0) {
        throw FractalSharkSeriousException(std::string("Invalid positive movie integer: ") + name);
    }
    return result;
}

double
ParseMovieDecimal(const std::string &text, const char *key, double minimum, double maximum)
{
    double value = 0;
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::fixed);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !std::isfinite(value) ||
        value < minimum || value > maximum) {
        throw FractalSharkSeriousException(std::string("Invalid movie option: ") + key);
    }
    return value;
}

std::string
MoviePathUtf8(const std::filesystem::path &path)
{
    return Environment::WideToUtf8(Environment::FromFsPath(path));
}

std::filesystem::path
ResolveVideoOutput(const std::string &value)
{
    if (value.find('\0') != std::string::npos || value.find_first_of("\r\n") != std::string::npos) {
        throw FractalSharkSeriousException("Invalid movie video-output path");
    }
    const std::filesystem::path path = Environment::ToFsPath(Environment::Utf8ToWide(value));
    const auto extension = MoviePathUtf8(path.extension());
    if (!path.is_absolute() ||
        (extension != ".ts" && extension != ".TS" && extension != ".Ts" && extension != ".tS")) {
        throw FractalSharkSeriousException("Movie video-output must be an absolute .ts file path");
    }
    std::error_code error;
    const auto parent = std::filesystem::canonical(path.parent_path(), error);
    if (error || !std::filesystem::is_directory(parent)) {
        throw FractalSharkSeriousException("Movie video-output directory is unavailable: " + value);
    }
    return parent / path.filename();
}

void
MaterializeMovieOutputDefaults(MovieConfig &config,
                               const std::filesystem::path &jobDirectory,
                               const MovieConfig *snapshot)
{
    const auto add = [&](const char *key, std::string value) {
        if (!snapshot || snapshot->Values.contains(key)) {
            config.Values.try_emplace(key, std::move(value));
        }
    };
    if (config.Video.Codec != FractalShark::MovieCodec::PngOnly) {
        add("video-output", MoviePathUtf8(std::filesystem::canonical(jobDirectory) / "movie.ts"));
    }
    if (config.Video.Codec == FractalShark::MovieCodec::Hevc) {
        const auto &rate = config.Video.Hevc;
        // All HEVC peak rates are multiples of 1,000 b/s, so this reverses
        // the integer conversion from buffer milliseconds exactly.
        const auto bitsPerMillisecond = rate.PeakBitsPerSecond / 1000;
        add("video-average-bitrate-mbps", FractalShark::MovieMegabits(rate.AverageBitsPerSecond));
        add("video-vbv-buffer-ms", std::to_string(rate.BufferBits / bitsPerMillisecond));
        add("video-vbv-initial-ms", std::to_string(rate.InitialBufferBits / bitsPerMillisecond));
    }
}

std::string
Padded(uint64_t number)
{
    std::ostringstream text;
    // INT_MAX is the frame-count limit, so ten decimal digits cover every index.
    text << std::setw(10) << std::setfill('0') << number;
    return text.str();
}
std::string
MoviePercentComplete(const MovieCheckpoint &state)
{
    if (state.TotalFrames == 0) {
        return "unknown";
    }
    // Round to hundredths of a percent. Reserve 100.00 for a full checkpoint.
    constexpr uint64_t Complete = 10'000;
    const auto rounded =
        (state.CheckpointedFrames * Complete + state.TotalFrames / 2) / state.TotalFrames;
    const auto percent =
        state.CheckpointedFrames == state.TotalFrames ? Complete : std::min(rounded, Complete - 1);
    return std::to_string(percent / 100) + '.' + std::to_string(100 + percent % 100).substr(1);
}

std::string_view
TakeMovieStateLine(std::string_view &input)
{
    const auto newline = input.find('\n');
    if (newline == std::string_view::npos) {
        throw FractalSharkSeriousException("Invalid movie-state.ini line ending");
    }
    const auto line = input.substr(0, newline);
    input.remove_prefix(newline + 1);
    return line;
}

std::string_view
TakeMovieStateField(std::string_view &input, std::string_view key)
{
    const auto line = TakeMovieStateLine(input);
    if (!line.starts_with(key) || line.size() == key.size()) {
        throw FractalSharkSeriousException("Invalid movie-state.ini field: " + std::string(key));
    }
    return line.substr(key.size());
}

template <typename T>
T
TakeMovieStateNumber(std::string_view &input, std::string_view key)
{
    const auto value = TakeMovieStateField(input, key);
    T number{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw FractalSharkSeriousException("Invalid movie-state.ini number: " + std::string(key));
    }
    return number;
}

MovieCheckpoint
ReadMovieCheckpoint(const std::filesystem::path &path,
                    std::string_view configurationChecksum,
                    bool pngOnly)
{
    const auto text = ReadMovieText(path);
    constexpr std::string_view ChecksumField = "state-crc32=";
    const auto checksumLine = text.rfind("\nstate-crc32=");
    if (checksumLine == std::string::npos) {
        throw FractalSharkSeriousException("Invalid movie-state.ini checksum line");
    }
    std::string_view payload(text.data(), checksumLine + 1);
    std::string_view checksum(text.data() + payload.size(), text.size() - payload.size());
    if (TakeMovieStateField(checksum, ChecksumField) != MovieMetadataChecksum(payload) ||
        !checksum.empty()) {
        throw FractalSharkSeriousException("Invalid movie-state.ini checksum");
    }
    if (TakeMovieStateLine(payload) != "[movie-state]" ||
        TakeMovieStateField(payload, "format=") != MovieFormat) {
        throw FractalSharkSeriousException(
            "Incompatible movie state format; create a new native movie folder");
    }

    MovieCheckpoint state;
    state.TotalFrames = TakeMovieStateNumber<uint64_t>(payload, "total-frames=");
    state.CheckpointedFrames = TakeMovieStateNumber<uint64_t>(payload, "checkpointed-frames=");
    // The display percentage is derived; restore the exact counts instead.
    (void)TakeMovieStateField(payload, "percent-complete=");
    state.VideoBytes = TakeMovieStateNumber<uint64_t>(payload, "video-bytes=");
    state.ActiveTimeMs = TakeMovieStateNumber<uint64_t>(payload, "active-time-ms=");
    if (TakeMovieStateField(payload, "configuration-crc32=") != configurationChecksum) {
        throw FractalSharkSeriousException("Movie settings differ from movie-state.ini");
    }
    if (state.TotalFrames != 0) {
        const auto bypassed = TakeMovieStateField(payload, "gpu-bypassed=");
        if (bypassed != "true" && bypassed != "false") {
            throw FractalSharkSeriousException("Invalid movie GPU mode in movie-state.ini");
        }
        state.GpuBypassed = bypassed == "true";
        state.LaDepth = TakeMovieStateNumber<long double>(payload, "la-switch-depth-bits=");
    }
    if (!payload.empty()) {
        const auto filename = TakeMovieStateField(payload, "staging-filename=");
        const auto owner = TakeMovieStateField(payload, "staging-owner=");
        if (pngOnly || state.TotalFrames == 0 || !filename.starts_with(".fractalshark-") ||
            !filename.ends_with(".ts.partial") ||
            filename.find_first_not_of(".abcdefghijklmnopqrstuvwxyz-0123456789") !=
                std::string_view::npos) {
            throw FractalSharkSeriousException("Invalid movie staging reservation in movie-state.ini");
        }
        state.Staging = MovieStaging{std::string(filename), std::string(owner)};
    }
    if (!payload.empty() || state.TotalFrames > INT_MAX ||
        state.CheckpointedFrames > state.TotalFrames || !std::isfinite(state.LaDepth) ||
        state.LaDepth < 0 ||
        (pngOnly ? state.VideoBytes != 0 : (state.CheckpointedFrames == 0) != (state.VideoBytes == 0))) {
        throw FractalSharkSeriousException("Invalid checkpoint values in movie-state.ini");
    }
    return state;
}

std::filesystem::path
PrepareDirectory(const std::filesystem::path &path)
{
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    // Check the retired formats before opening or creating the job lock.
    for (const auto *name : {"profile.txt",
                             "progress.txt",
                             "time.txt",
                             "video-stage.txt",
                             "ffmpeg.txt",
                             "assembly.txt",
                             "chunks"}) {
        if (std::filesystem::exists(std::filesystem::symlink_status(absolute / name))) {
            throw FractalSharkSeriousException(
                "Earlier movie metadata format; create a new native movie folder");
        }
    }
    std::filesystem::create_directories(absolute);
    return absolute;
}
HighPrecision
ScalePower(const HighPrecision &base, uint64_t exponent)
{
    HighPrecision factor{HighPrecision::SetPrecision::True, 256};
    HighPrecision power{base};
    mpf_set_ui(factor.backend(), 1);
    // mpf_pow_ui has a 32-bit exponent on Windows; the movie index is 64-bit.
    while (exponent != 0) {
        if (exponent & 1) {
            mpf_mul(factor.backend(), factor.backend(), power.backend());
        }
        exponent >>= 1;
        if (exponent != 0) {
            mpf_mul(power.backend(), power.backend(), power.backend());
        }
    }
    return factor;
}
} // namespace

class MovieJob::State {
public:
    State(std::filesystem::path path, std::string configurationChecksum, MovieCheckpoint checkpoint)
        : m_Path(std::move(path)), m_ConfigurationChecksum(std::move(configurationChecksum)),
          m_Started(std::chrono::steady_clock::now()), m_PreviousMilliseconds(checkpoint.ActiveTimeMs),
          m_Checkpoint(std::move(checkpoint))
    {
        SaveLocked();
        m_Thread = std::jthread([this](std::stop_token stop) {
            // Exceptions cannot escape a worker. The movie event loop observes
            // a failed state write at the next checkpoint or timing check.
            try {
                constexpr auto Heartbeat = std::chrono::seconds(1);
                std::unique_lock lock(m_Mutex);
                while (!stop.stop_requested()) {
                    const auto lastWrite = m_LastWrite;
                    m_Wake.wait_until(
                        lock, stop, lastWrite + Heartbeat, [&] { return m_LastWrite != lastWrite; });
                    // Each frame checkpoint already includes time. Only a quiet
                    // period needs a separate write, with the same durable counts.
                    if (!stop.stop_requested() && m_LastWrite == lastWrite) {
                        SaveLocked();
                    }
                }
            } catch (...) {
                std::lock_guard lock(m_Mutex);
                m_Error = std::current_exception();
            }
        });
    }
    ~State() { Finish(); }

    auto
    Inspect(const auto &read) const
    {
        std::lock_guard lock(m_Mutex);
        return read(m_Checkpoint);
    }

    void
    Update(const auto &change)
    {
        // Hold the lock across mutation and publication. A timing-only write
        // cannot overtake a frame checkpoint or reuse its previous frontier.
        std::lock_guard lock(m_Mutex);
        CheckLocked();
        if (change(m_Checkpoint)) {
            SaveLocked();
        }
    }

    uint64_t
    Elapsed() const
    {
        std::lock_guard lock(m_Mutex);
        return ElapsedLocked();
    }
    void
    Check() const
    {
        std::lock_guard lock(m_Mutex);
        CheckLocked();
    }
    bool
    Finish()
    {
        if (m_Finished) {
            return m_Succeeded;
        }
        m_Finished = true;
        m_Thread.request_stop();
        m_Wake.notify_all();
        if (m_Thread.joinable()) {
            m_Thread.join();
        }
        // This is the persistence/error boundary, including destructor fallback.
        // A disk error is reported and changes a normal CLI exit to failure.
        try {
            std::lock_guard lock(m_Mutex);
            m_FinalMilliseconds = ElapsedLocked();
            SaveLocked();
            CheckLocked();
        } catch (const std::exception &exception) {
            m_Succeeded = false;
            FractalSharkLog::WriteException(
                "Could not persist movie state", exception, __FILE__, __LINE__);
        }
        const auto total = Elapsed();
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Movie active wall time: total_seconds=" << static_cast<double>(total) / 1000
            << " session_seconds=" << static_cast<double>(total - m_PreviousMilliseconds) / 1000;
        return m_Succeeded;
    }

private:
    uint64_t
    ElapsedLocked() const
    {
        if (m_FinalMilliseconds) {
            return *m_FinalMilliseconds;
        }
        return m_PreviousMilliseconds +
               static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - m_Started)
                                         .count());
    }

    void
    CheckLocked() const
    {
        if (m_Error) {
            std::rethrow_exception(m_Error);
        }
    }

    void
    SaveLocked()
    {
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << "[movie-state]\nformat=" << MovieFormat << '\n'
               << "total-frames=" << m_Checkpoint.TotalFrames << '\n'
               << "checkpointed-frames=" << m_Checkpoint.CheckpointedFrames << '\n'
               << "percent-complete=" << MoviePercentComplete(m_Checkpoint) << '\n'
               << "video-bytes=" << m_Checkpoint.VideoBytes << '\n'
               << "active-time-ms=" << ElapsedLocked() << '\n'
               << "configuration-crc32=" << m_ConfigurationChecksum << '\n';
        if (m_Checkpoint.TotalFrames != 0) {
            output << "gpu-bypassed=" << std::boolalpha << m_Checkpoint.GpuBypassed << '\n'
                   << "la-switch-depth-bits="
                   << std::setprecision(std::numeric_limits<long double>::max_digits10)
                   << m_Checkpoint.LaDepth << '\n';
        }
        if (m_Checkpoint.Staging) {
            output << "staging-filename=" << m_Checkpoint.Staging->Filename << '\n'
                   << "staging-owner=" << m_Checkpoint.Staging->Owner << '\n';
        }
        auto payload = std::move(output).str();
        const auto checksum = MovieMetadataChecksum(payload);
        payload += "state-crc32=" + checksum + '\n';
        WriteMovieText(m_Path, payload);
        m_LastWrite = std::chrono::steady_clock::now();
        m_Wake.notify_all();
    }

    const std::filesystem::path m_Path;
    const std::string m_ConfigurationChecksum;
    const std::chrono::steady_clock::time_point m_Started;
    const uint64_t m_PreviousMilliseconds;
    MovieCheckpoint m_Checkpoint;
    std::optional<uint64_t> m_FinalMilliseconds;
    std::chrono::steady_clock::time_point m_LastWrite;
    mutable std::mutex m_Mutex;
    std::condition_variable_any m_Wake;
    std::exception_ptr m_Error;
    std::jthread m_Thread;
    bool m_Finished = false;
    bool m_Succeeded = true;
};

std::string
MovieMetadataChecksum(std::string_view text)
{
    std::ostringstream checksum;
    checksum << std::hex << std::setw(8) << std::setfill('0')
             << lodepng_crc32(reinterpret_cast<const unsigned char *>(text.data()), text.size());
    return checksum.str();
}

std::string
ReadMovieText(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw FractalSharkSeriousException("Cannot open movie job file: " + path.string());
    }
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad()) {
        throw FractalSharkSeriousException("Cannot read movie job file");
    }
    return text;
}
void
WriteMovieText(const std::filesystem::path &path, const std::string &text)
{
    auto temporary = path;
    temporary += ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.close();
    if (!output) {
        throw FractalSharkSeriousException("Cannot write movie job file: " + path.string());
    }
    Environment::PublishMovieFile(temporary, path, true);
}

MovieConfig
MovieConfig::Load(const std::filesystem::path &path)
{
    MovieConfig config;
    std::istringstream input(ReadMovieText(path));
    std::string line;
    const std::set<std::string> known{"center-x",
                                      "center-y",
                                      "zoom",
                                      "iterations",
                                      "zoom-per-frame",
                                      "width",
                                      "height",
                                      "antialiasing",
                                      "render-algorithm",
                                      "perturbation-alg",
                                      "palette-map",
                                      "palette-depth",
                                      "wide-iterations",
                                      "la-slope-ratio",
                                      "video-gop-frames",
                                      "video-fps",
                                      "render-order",
                                      "video-profile",
                                      "video-preset",
                                      "video-cq",
                                      "video-max-bitrate-mbps",
                                      "video-average-bitrate-mbps",
                                      "video-vbv-buffer-ms",
                                      "video-vbv-initial-ms",
                                      "video-output"};
    while (std::getline(input, line)) {
        if (line.starts_with("\xef\xbb\xbf")) {
            line.erase(0, 3);
        }
        line = Trim(line);
        if (line.empty() || line.front() == '#' || line.front() == ';' || line == "[movie]") {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw FractalSharkSeriousException("Movie config expects key = value");
        }
        auto key = Trim(std::string_view(line).substr(0, separator));
        auto value = Trim(std::string_view(line).substr(separator + 1));
        if (!known.contains(key) || value.empty() || !config.Values.emplace(key, value).second) {
            throw FractalSharkSeriousException("Unknown, empty, or duplicate movie option: " + key);
        }
    }
    if (config.Values.contains("render-order")) {
        const auto &order = config.Values.at("render-order");
        if (std::ranges::find(FractalShark::MovieRenderOrders, std::string_view(order)) ==
            FractalShark::MovieRenderOrders.end()) {
            throw FractalSharkSeriousException("Invalid movie render-order");
        }
        config.WideToDeep = order == FractalShark::MovieRenderOrders.front();
    }
    const auto &profileName =
        config.Values.try_emplace("video-profile", FractalShark::DefaultMovieVideoProfile.Name)
            .first->second;
    const auto profile = std::ranges::find(FractalShark::MovieVideoProfiles,
                                           std::string_view(profileName),
                                           &FractalShark::MovieVideoProfile::Name);
    if (profile == FractalShark::MovieVideoProfiles.end()) {
        throw FractalSharkSeriousException("Unknown movie video-profile: " + profileName);
    }
    auto &video = config.Video;
    video.Codec = profile->Codec;
    video.Preset = profile->Preset;
    video.Quality = profile->Quality;

    const auto option = [&](const char *key, uint64_t fallback, uint64_t minimum, uint64_t maximum) {
        const auto found = config.Values.find(key);
        if (found == config.Values.end()) {
            return fallback;
        }
        uint64_t value = 0;
        const auto &text = found->second;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < minimum ||
            value > maximum) {
            throw FractalSharkSeriousException(std::string("Invalid movie option: ") + key);
        }
        return value;
    };
    video.Fps = static_cast<unsigned>(option("video-fps", video.Fps, 1, 1000));
    video.GopFrames = static_cast<uint32_t>(option("video-gop-frames", video.GopFrames, 1, INT32_MAX));
    if (video.Codec != FractalShark::MovieCodec::PngOnly) {
        video.Preset = static_cast<unsigned>(option("video-preset", video.Preset, 1, 7));
        config.Values["video-preset"] = std::to_string(video.Preset);
    } else if (config.Values.contains("video-preset")) {
        throw FractalSharkSeriousException("PNG-only movies do not have an encoder preset");
    }
    if (video.Codec == FractalShark::MovieCodec::Hevc) {
        const auto quality = config.Values.find("video-cq");
        constexpr unsigned QualityUnits = 256;
        const auto fixedQuality =
            quality == config.Values.end()
                ? video.Quality * QualityUnits
                : static_cast<unsigned>(
                      std::round(ParseMovieDecimal(quality->second, "video-cq", 1, 51) * QualityUnits));
        video.Quality = fixedQuality / QualityUnits;
        video.QualityFraction = static_cast<uint8_t>(fixedQuality % QualityUnits);
        std::ostringstream formatted;
        formatted.imbue(std::locale::classic());
        // All 1/256 steps have an exact representation with eight decimal places.
        formatted << std::fixed << std::setprecision(8)
                  << static_cast<double>(fixedQuality) / QualityUnits;
        auto text = formatted.str();
        text.erase(text.find_last_not_of('0') + 1);
        if (text.back() == '.') {
            text.pop_back();
        }
        config.Values["video-cq"] = std::move(text);
    } else if (video.Codec == FractalShark::MovieCodec::H264) {
        video.Quality = static_cast<unsigned>(option("video-cq", video.Quality, 1, 51));
        config.Values["video-cq"] = std::to_string(video.Quality);
    } else if (config.Values.contains("video-cq")) {
        throw FractalSharkSeriousException("Movie quality override does not match its video profile");
    }
    config.Values.try_emplace("render-order", FractalShark::MovieRenderOrders.front());
    const std::array required{"center-x", "center-y", "zoom", "iterations"};
    for (const auto *key : required) {
        if (!config.Values.contains(key)) {
            throw FractalSharkSeriousException(std::string("Missing movie option: ") + key);
        }
    }
    const auto targetIterations = ParseUnsigned(config.Values.at("iterations"), "iterations");
    if (targetIterations > static_cast<uint64_t>(INT64_MAX - 1)) {
        throw FractalSharkSeriousException("Movie iterations exceed the renderer limit");
    }
    if (config.Values.contains("wide-iterations")) {
        config.WideIterations = ParseUnsigned(config.Values.at("wide-iterations"), "wide-iterations");
    }
    config.WideIterations = std::min(config.WideIterations, targetIterations);
    if (config.Values.contains("la-slope-ratio")) {
        const auto &value = config.Values.at("la-slope-ratio");
        const auto parsed =
            std::from_chars(value.data(), value.data() + value.size(), config.SlopeRatio);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
            throw FractalSharkSeriousException("Invalid movie la-slope-ratio");
        }
    }
    if (!std::isfinite(config.SlopeRatio) || config.SlopeRatio < 1) {
        throw FractalSharkSeriousException("Movie la-slope-ratio must be finite and at least one");
    }
    // Materialize movie defaults in the immutable snapshot. A resumed job must
    // not inherit different defaults from a later executable.
    config.Values.try_emplace("zoom-per-frame", FractalShark::MovieDefaultZoomPerFrame);
    config.Values["video-gop-frames"] = std::to_string(video.GopFrames);
    config.Values["video-fps"] = std::to_string(video.Fps);
    config.Values["wide-iterations"] = std::to_string(config.WideIterations);
    std::ostringstream ratio;
    ratio << std::setprecision(std::numeric_limits<double>::max_digits10) << config.SlopeRatio;
    config.Values["la-slope-ratio"] = ratio.str();
    config.Values.try_emplace("width", "1024");
    config.Values.try_emplace("height", "768");
    constexpr auto peakOption = "video-max-bitrate-mbps";
    constexpr auto averageOption = "video-average-bitrate-mbps";
    constexpr auto bufferOption = "video-vbv-buffer-ms";
    constexpr auto initialOption = "video-vbv-initial-ms";
    if (video.Codec == FractalShark::MovieCodec::Hevc) {
        FractalShark::MovieHevcOptions hevc;
        const auto defaultPeak = profile->Tier == FractalShark::MovieHevcTier::High
                                     ? hevc.DefaultHighPeakMbps
                                     : hevc.DefaultLowPeakMbps;
        hevc.PeakMbps = static_cast<unsigned>(option(peakOption, defaultPeak, 1, hevc.MaximumPeakMbps));
        if (const auto average = config.Values.find(averageOption); average != config.Values.end()) {
            constexpr unsigned BitsPerMegabit = 1'000'000;
            hevc.AverageBitsPerSecond = static_cast<uint32_t>(std::round(
                BitsPerMegabit *
                ParseMovieDecimal(
                    average->second, averageOption, 1.0 / BitsPerMegabit, hevc.MaximumPeakMbps)));
            average->second = FractalShark::MovieMegabits(*hevc.AverageBitsPerSecond);
        }
        hevc.BufferMilliseconds = static_cast<unsigned>(
            option(bufferOption, hevc.BufferMilliseconds, 1, hevc.MaximumBufferMilliseconds));
        hevc.InitialMilliseconds =
            static_cast<unsigned>(option(initialOption,
                                         std::min(hevc.InitialMilliseconds, hevc.BufferMilliseconds),
                                         1,
                                         hevc.BufferMilliseconds));
        for (const auto &entry : {std::pair{bufferOption, hevc.BufferMilliseconds},
                                  std::pair{initialOption, hevc.InitialMilliseconds}}) {
            if (const auto found = config.Values.find(entry.first); found != config.Values.end()) {
                found->second = std::to_string(entry.second);
            }
        }
        const auto width = static_cast<unsigned>(option("width", 1024, 1, INT_MAX));
        const auto height = static_cast<unsigned>(option("height", 768, 1, INT_MAX));
        video.Hevc =
            FractalShark::ResolveMovieHevcRateControl(width, height, video.Fps, profile->Tier, hevc);
        config.Values[peakOption] = std::to_string(hevc.PeakMbps);
    } else if (config.Values.contains(peakOption) || config.Values.contains(averageOption) ||
               config.Values.contains(bufferOption) || config.Values.contains(initialOption)) {
        throw FractalSharkSeriousException(
            "Movie bitrate and buffer overrides require a lossy HEVC profile");
    }
    if (const auto output = config.Values.find("video-output");
        output != config.Values.end() && video.Codec != FractalShark::MovieCodec::PngOnly) {
        config.VideoOutput = ResolveVideoOutput(output->second);
        output->second = MoviePathUtf8(config.VideoOutput);
    }
    config.Values.try_emplace("render-algorithm", "AutoSelect");
    config.Values.try_emplace("perturbation-alg", "Auto");
    if (config.Values.contains("palette-map")) {
        config.Values["palette-map"] =
            std::filesystem::absolute(path.parent_path() / config.Values.at("palette-map"))
                .lexically_normal()
                .string();
    }
    return config;
}
std::vector<std::string>
MovieConfig::RenderArguments() const
{
    std::vector<std::string> arguments;
    for (const auto &[key, value] : Values) {
        if (key == "zoom-per-frame" || key == "wide-iterations" || key == "la-slope-ratio" ||
            key.starts_with("video-") || key == "render-order") {
            continue;
        }
        arguments.push_back("--" + key);
        arguments.push_back(value);
    }
    return arguments;
}
std::string
MovieConfig::Serialize(bool includeComments) const
{
    std::string result = "[movie]\n";
    for (const auto &[key, value] : Values) {
        if (includeComments) {
            result += FractalShark::MovieIniComments(key);
        }
        result += key + "=" + value + "\n";
    }
    return result;
}

long double
MovieSchedule::Log2(const HighPrecision &value)
{
    if (mpf_sgn(value.backend()) <= 0) {
        throw FractalSharkSeriousException("Movie zoom must be positive");
    }
    if (mpf_cmp_ui(value.backend(), 1) >= 0 && mpf_cmp_ui(value.backend(), 2) < 0) {
        HighPrecision delta{HighPrecision::SetPrecision::True, value.precisionInBits()};
        mpf_sub_ui(delta.backend(), value.backend(), 1);
        return std::log1p(static_cast<long double>(mpf_get_d(delta.backend()))) / std::log(2.0L);
    }
    double mantissa;
    const auto exponent = mpf_get_2exp_d(&mantissa, value.backend());
    return static_cast<long double>(exponent) + std::log2(static_cast<long double>(mantissa));
}
MovieSchedule::MovieSchedule(const HighPrecision &targetZoom,
                             const std::string &step,
                             uint64_t wideIterations,
                             uint64_t targetIterations,
                             long double laDepth,
                             double slopeRatio,
                             bool wideToDeep)
    : m_WideToDeep(wideToDeep), m_Target(HighPrecision::SetPrecision::True, 256),
      m_Step(HighPrecision::SetPrecision::True, 256), m_LaDepth(laDepth), m_SlopeRatio(slopeRatio),
      m_WideIterations(wideIterations), m_TargetIterations(targetIterations)
{
    mpf_set(m_Target.backend(), targetZoom.backend());
    if (mpf_set_str(m_Step.backend(), step.c_str(), 10) != 0 || mpf_cmp_ui(m_Step.backend(), 1) <= 0 ||
        mpf_cmp_ui(m_Target.backend(), 1) < 0 || !std::isfinite(slopeRatio) || slopeRatio < 1 ||
        wideIterations == 0 || wideIterations > targetIterations) {
        throw FractalSharkSeriousException("Invalid movie zoom schedule or iteration limits");
    }
    m_Depth = Log2(m_Target);
    // Leave room for the existing precision calculator's guard/reuse bits.
    constexpr long double MaximumDepthBits = 63'000'000;
    if (m_Depth > MaximumDepthBits) {
        throw FractalSharkSeriousException("Movie target exceeds supported rendering precision");
    }
    m_LaDepth = std::clamp(m_LaDepth, 0.0L, m_Depth);
    m_StepDepth = Log2(m_Step);
    // Keep steps above scale-rounding error during 64-bit exponentiation.
    // This is vastly smaller than useful movie zoom rates.
    constexpr int MinimumStepDepthExponent = -128;
    if (m_StepDepth < std::ldexp(1.0L, MinimumStepDepthExponent)) {
        throw FractalSharkSeriousException("Movie zoom-per-frame exceeds supported scale precision");
    }
    const auto intervals = std::ceil(m_Depth / m_StepDepth);
    // Keep movie indexing within the supported render schedule range.
    if (!std::isfinite(intervals) || intervals > std::numeric_limits<int>::max() || m_StepDepth <= 0) {
        throw FractalSharkSeriousException("Movie frame count exceeds INT_MAX");
    }
    auto intervalCount = static_cast<uint64_t>(intervals);
    HighPrecision tolerance{m_Target};
    constexpr unsigned ScaleRoundingGuardBits = 192;
    mpf_div_2exp(tolerance.backend(), tolerance.backend(), ScaleRoundingGuardBits);
    const auto reachesTarget = [&](uint64_t count) {
        if (count == 0) {
            return mpf_cmp_ui(m_Target.backend(), 1) == 0;
        }
        auto power = ScalePower(m_Step, count);
        mpf_add(power.backend(), power.backend(), tolerance.backend());
        return mpf_cmp(power.backend(), m_Target.backend()) >= 0;
    };
    // Refine the logarithm's integer estimate at scale precision. Exact-power
    // endpoints must not acquire an extra frame due to host log rounding.
    while (intervalCount > 0 && reachesTarget(intervalCount - 1)) {
        --intervalCount;
    }
    while (!reachesTarget(intervalCount)) {
        if (intervalCount >= static_cast<uint64_t>(std::numeric_limits<int>::max()) - 1) {
            throw FractalSharkSeriousException("Movie frame count exceeds INT_MAX");
        }
        ++intervalCount;
    }
    m_FrameCount = intervalCount + 1;
    if (m_FrameCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw FractalSharkSeriousException("Movie frame count exceeds INT_MAX");
    }
}
uint64_t
MovieSchedule::FrameCount() const
{
    return m_FrameCount;
}
long double
MovieSchedule::DepthAt(uint64_t index) const
{
    if (index >= m_FrameCount) {
        throw FractalSharkSeriousException("Movie frame index out of range");
    }
    if (m_WideToDeep) {
        return std::min(m_Depth, index * m_StepDepth);
    }
    return index == m_FrameCount - 1 ? 0 : std::max(0.0L, m_Depth - index * m_StepDepth);
}
HighPrecision
MovieSchedule::ZoomAt(uint64_t index) const
{
    if (m_WideToDeep) {
        if (index >= m_FrameCount) {
            throw FractalSharkSeriousException("Movie frame index out of range");
        }
        return index == m_FrameCount - 1 ? HighPrecision{m_Target} : ScalePower(m_Step, index);
    }

    DepthAt(index);
    HighPrecision result{HighPrecision::SetPrecision::True, 256};
    if (index == m_FrameCount - 1 && m_FrameCount > 1) {
        mpf_set_ui(result.backend(), 1);
        return result;
    }
    const auto factor = ScalePower(m_Step, index);
    mpf_div(result.backend(), m_Target.backend(), factor.backend());
    return result;
}
uint64_t
MovieSchedule::IterationsAt(uint64_t index) const
{
    const auto depth = DepthAt(index);
    if (m_FrameCount == 1 || index == (m_WideToDeep ? m_FrameCount - 1 : 0)) {
        return m_TargetIterations;
    }
    if (index == (m_WideToDeep ? 0 : m_FrameCount - 1)) {
        return m_WideIterations;
    }
    const auto weight = [&](long double d) {
        // Scale both segment weights before normalizing to avoid overflowing
        // for a finite but very large slope ratio.
        return std::min(d, m_LaDepth) / m_SlopeRatio + std::max(0.0L, d - m_LaDepth);
    };
    const auto fraction =
        m_LaDepth == 0 || m_LaDepth == m_Depth ? depth / m_Depth : weight(depth) / weight(m_Depth);
    const auto value = std::ceil(m_WideIterations + (m_TargetIterations - m_WideIterations) * fraction);
    if (value >= static_cast<long double>(m_TargetIterations)) {
        return m_TargetIterations;
    }
    return std::clamp(static_cast<uint64_t>(value), m_WideIterations, m_TargetIterations);
}

MovieJob::MovieJob(const std::filesystem::path &directory, const std::filesystem::path &configuration)
    : m_Directory(PrepareDirectory(directory)), m_Lock(m_Directory / "movie.lock")
{
    const auto statePath = m_Directory / MovieStateName;
    const bool existing = std::filesystem::exists(std::filesystem::symlink_status(statePath));
    if (!existing) {
        for (const auto *name : {"frames", "movie.ts", "movie.ts.partial"}) {
            if (std::filesystem::exists(std::filesystem::symlink_status(m_Directory / name))) {
                throw FractalSharkSeriousException(
                    "Movie output has no movie-state.ini; create a new native movie folder");
            }
        }
    }

    const auto snapshot = m_Directory / "movie.ini";
    if (std::filesystem::exists(snapshot)) {
        m_Config = MovieConfig::Load(snapshot);
        if (!configuration.empty()) {
            auto candidate = MovieConfig::Load(configuration);
            MaterializeMovieOutputDefaults(candidate, m_Directory, &m_Config);
            if (candidate.Values.contains("palette-map") && m_Config.Values.contains("palette-map")) {
                if (ReadMovieText(candidate.Values.at("palette-map")) !=
                    ReadMovieText(m_Config.Values.at("palette-map"))) {
                    throw FractalSharkSeriousException("Movie palette differs from the job snapshot");
                }
                candidate.Values["palette-map"] = m_Config.Values.at("palette-map");
            }
            if (candidate.Serialize(false) != m_Config.Serialize(false)) {
                throw FractalSharkSeriousException("Movie settings differ from the existing job");
            }
        }
    } else {
        if (existing || configuration.empty()) {
            throw FractalSharkSeriousException(
                "Movie has no configuration; supply --movie-config for a new job");
        }
        m_Config = MovieConfig::Load(configuration);
        MaterializeMovieOutputDefaults(m_Config, m_Directory, nullptr);
        if (m_Config.Values.contains("palette-map")) {
            WriteMovieText(m_Directory / "palette.map",
                           ReadMovieText(m_Config.Values.at("palette-map")));
            m_Config.Values["palette-map"] = "palette.map";
        }
        WriteMovieText(snapshot, m_Config.Serialize(true));
        m_Config = MovieConfig::Load(snapshot);
    }
    auto portableConfig = m_Config;
    if (portableConfig.Values.contains("palette-map")) {
        portableConfig.Values["palette-map"] = "palette.map";
    }
    auto configurationText = std::string(MovieFormat) + "\n" + portableConfig.Serialize(false);
    if (m_Config.Video.Codec == FractalShark::MovieCodec::Hevc) {
        // Seal the derived limits as well as the user's inputs. A level-table or buffer
        // policy change must never append a different encoder configuration to a job.
        const auto &rate = m_Config.Video.Hevc;
        configurationText +=
            "resolved-hevc=" + std::to_string(static_cast<unsigned>(rate.Tier)) + ',' +
            std::to_string(rate.LevelIdc) + ',' + std::to_string(rate.PeakBitsPerSecond) + ',' +
            std::to_string(rate.AverageBitsPerSecond) + ',' + std::to_string(rate.BufferBits) + ',' +
            std::to_string(rate.InitialBufferBits) + '\n';
    }
    if (m_Config.Values.contains("palette-map")) {
        configurationText += ReadMovieText(m_Config.Values.at("palette-map"));
    }
    const auto configurationChecksum = MovieMetadataChecksum(configurationText);
    const bool pngOnly = m_Config.Video.Codec == FractalShark::MovieCodec::PngOnly;
    auto checkpoint =
        existing ? ReadMovieCheckpoint(statePath, configurationChecksum, pngOnly) : MovieCheckpoint{};
    const auto canonicalJob = std::filesystem::canonical(m_Directory);
    const auto defaultOutput = canonicalJob / "movie.ts";
    m_VideoOutput = m_Config.VideoOutput.empty() ? defaultOutput : m_Config.VideoOutput;
    const bool hasCustomOutput = m_VideoOutput != defaultOutput;
    if ((checkpoint.Staging &&
         (!hasCustomOutput || checkpoint.Staging->Owner != MoviePathUtf8(canonicalJob))) ||
        (hasCustomOutput && checkpoint.VideoBytes != 0 && !checkpoint.Staging)) {
        throw FractalSharkSeriousException(
            "Movie staging reservation is missing or job directory changed");
    }
    if (!existing && !pngOnly &&
        std::filesystem::exists(std::filesystem::symlink_status(m_VideoOutput))) {
        throw FractalSharkSeriousException("Movie video-output already exists: " +
                                           MoviePathUtf8(m_VideoOutput));
    }
    m_State = std::make_unique<State>(statePath, configurationChecksum, std::move(checkpoint));
}
MovieJob::~MovieJob() = default;

bool
MovieJob::FinishTiming()
{
    return m_State->Finish();
}

void
MovieJob::CheckTiming() const
{
    m_State->Check();
}

uint64_t
MovieJob::ElapsedMilliseconds() const
{
    return m_State->Elapsed();
}

const MovieConfig &
MovieJob::Config() const
{
    return m_Config;
}
const std::filesystem::path &
MovieJob::Directory() const
{
    return m_Directory;
}

const std::filesystem::path &
MovieJob::VideoOutputPath() const
{
    return m_VideoOutput;
}

std::filesystem::path
MovieJob::StagingVideoPath() const
{
    const auto canonicalJob = std::filesystem::canonical(m_Directory);
    if (m_VideoOutput == canonicalJob / "movie.ts") {
        return canonicalJob / "movie.ts.partial";
    }

    // A separate reservation per job permits independent checkpoints even when
    // two jobs choose the same destination. Publication decides which can use it.
    const auto owner = MoviePathUtf8(canonicalJob);
    if (owner.find_first_of("\r\n") != std::string::npos) {
        throw FractalSharkSeriousException("Movie job directory must fit on one staging record line");
    }
    std::filesystem::path partial;
    m_State->Update([&](MovieCheckpoint &checkpoint) {
        if (checkpoint.Staging) {
            partial = m_VideoOutput.parent_path() / checkpoint.Staging->Filename;
            return false;
        }
        if (checkpoint.TotalFrames == 0 || checkpoint.VideoBytes != 0 ||
            std::filesystem::exists(std::filesystem::symlink_status(m_VideoOutput))) {
            throw FractalSharkSeriousException("Cannot reserve movie staging path for this output");
        }
        partial = Environment::ReserveMoviePartialFile(m_VideoOutput.parent_path());
        checkpoint.Staging = MovieStaging{partial.filename().string(), owner};
        return true;
    });
    return partial;
}

uint64_t
MovieJob::FrameCount() const
{
    return m_State->Inspect([](const MovieCheckpoint &checkpoint) { return checkpoint.TotalFrames; });
}

uint64_t
MovieJob::Remaining() const
{
    return m_State->Inspect([](const MovieCheckpoint &checkpoint) {
        return checkpoint.TotalFrames - checkpoint.CheckpointedFrames;
    });
}

uint64_t
MovieJob::NextFrame() const
{
    return m_State->Inspect(
        [](const MovieCheckpoint &checkpoint) { return checkpoint.CheckpointedFrames; });
}

bool
MovieJob::GpuBypassed() const
{
    return m_State->Inspect([](const MovieCheckpoint &checkpoint) { return checkpoint.GpuBypassed; });
}

long double
MovieJob::LaDepth() const
{
    return m_State->Inspect([](const MovieCheckpoint &checkpoint) { return checkpoint.LaDepth; });
}

void
MovieJob::InitializeProgress(uint64_t frameCount, bool gpuBypassed, long double laDepth)
{
    if (frameCount == 0 || frameCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw FractalSharkSeriousException("Movie frame count must be between one and INT_MAX");
    }
    m_State->Update([&](MovieCheckpoint &checkpoint) {
        if (checkpoint.TotalFrames != 0) {
            if (frameCount != checkpoint.TotalFrames || gpuBypassed != checkpoint.GpuBypassed) {
                throw FractalSharkSeriousException("Movie schedule or CPU/GPU availability changed");
            }
            return false;
        }
        checkpoint.TotalFrames = frameCount;
        checkpoint.GpuBypassed = gpuBypassed;
        checkpoint.LaDepth = laDepth;
        return true;
    });
}

void
MovieJob::CompleteFrame(uint64_t index)
{
    m_State->Update([&](MovieCheckpoint &checkpoint) {
        if (checkpoint.CheckpointedFrames == checkpoint.TotalFrames ||
            index != checkpoint.CheckpointedFrames) {
            throw FractalSharkSeriousException("Movie frames must commit in ascending render order");
        }
        ++checkpoint.CheckpointedFrames;
        return true;
    });
}

void
MovieJob::Rewind(uint64_t nextFrame)
{
    m_State->Update([&](MovieCheckpoint &checkpoint) {
        if (nextFrame > checkpoint.CheckpointedFrames) {
            throw FractalSharkSeriousException("Invalid movie recovery frontier");
        }
        checkpoint.CheckpointedFrames = nextFrame;
        return true;
    });
}

uint64_t
MovieJob::VideoBytes() const
{
    return m_State->Inspect([](const MovieCheckpoint &checkpoint) { return checkpoint.VideoBytes; });
}

void
MovieJob::CommitVideo(uint64_t nextFrame, uint64_t bytes)
{
    m_State->Update([&](MovieCheckpoint &checkpoint) {
        if (nextFrame <= checkpoint.CheckpointedFrames || nextFrame > checkpoint.TotalFrames ||
            bytes <= checkpoint.VideoBytes) {
            throw FractalSharkSeriousException("Invalid movie video checkpoint");
        }
        checkpoint.CheckpointedFrames = nextFrame;
        checkpoint.VideoBytes = bytes;
        return true;
    });
}

std::filesystem::path
MovieJob::FramePath(uint64_t index) const
{
    return m_Directory / "frames" / ("chunk_" + Padded(index / m_Config.Video.GopFrames)) /
           ("frame_" + Padded(index) + ".png");
}

std::vector<unsigned char>
ReadMovieRgb8(const std::filesystem::path &path, unsigned width, unsigned height)
{
    const auto encoded = ReadMovieText(path);
    std::vector<unsigned char> decoded;
    unsigned actualWidth = 0, actualHeight = 0;
    const auto error = lodepng::decode(decoded,
                                       actualWidth,
                                       actualHeight,
                                       reinterpret_cast<const unsigned char *>(encoded.data()),
                                       encoded.size(),
                                       LCT_RGB,
                                       16);
    if (error != 0 || actualWidth != width || actualHeight != height) {
        throw FractalSharkSeriousException("Invalid movie PNG or dimensions: " + path.string());
    }
    std::vector<unsigned char> result(decoded.size() / 2);
    for (size_t i = 0; i < result.size(); ++i) {
        const uint32_t value = (uint32_t{decoded[2 * i]} << 8) | decoded[2 * i + 1];
        result[i] = static_cast<unsigned char>((value + 128) / 257);
    }
    return result;
}
} // namespace FractalSharkCli
