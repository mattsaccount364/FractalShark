#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieHash.h"
#include "MovieJob.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>

namespace FractalSharkCli {
namespace {
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
std::string
Padded(uint64_t number)
{
    std::ostringstream text;
    text << std::setw(20) << std::setfill('0') << number;
    return text.str();
}
std::string
Seal(const std::string &payload)
{
    return MovieHash::Digest(payload) + "\n" + payload;
}
std::optional<std::string>
Unseal(const std::filesystem::path &path)
{
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto contents = ReadMovieText(path);
    if (contents.size() < 65 || contents[64] != '\n' ||
        MovieHash::Digest(std::string_view(contents).substr(65)) != contents.substr(0, 64)) {
        return std::nullopt;
    }
    return contents.substr(65);
}
std::filesystem::path
PrepareDirectory(const std::filesystem::path &path)
{
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
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
void
RemoveMovieTemporary(const std::filesystem::path &path)
{
    std::error_code error;
    std::filesystem::remove(path, error);
    if (error) {
        // An orphaned encoder may still hold this file. Unique output names
        // prevent collisions, and the next resume retries the cleanup.
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Retaining unfinished movie temporary for later cleanup: " << path.string() << ": "
            << error.message();
    }
}
void
RemoveUnfinishedMovies(const std::filesystem::path &directory)
{
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with(".fractalshark-video-") && entry.path().extension() == ".tmp") {
            RemoveMovieTemporary(entry.path());
        }
    }
}
} // namespace

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
                                      "video-chunk-frames",
                                      "video-fps"};
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
    for (const auto key : {"center-x", "center-y", "zoom", "iterations", "zoom-per-frame"}) {
        if (!config.Values.contains(key)) {
            throw FractalSharkSeriousException(std::string("Missing movie option: ") + key);
        }
    }
    if (config.Values.contains("video-chunk-frames")) {
        config.ChunkFrames = ParseUnsigned(config.Values.at("video-chunk-frames"), "video-chunk-frames");
    }
    if (config.Values.contains("video-fps")) {
        const auto fps = ParseUnsigned(config.Values.at("video-fps"), "video-fps");
        if (fps > 1000) {
            throw FractalSharkSeriousException("Movie video-fps exceeds 1000");
        }
        config.Fps = static_cast<unsigned>(fps);
    }
    if (config.ChunkFrames > INT32_MAX) {
        throw FractalSharkSeriousException(
            "Movie chunk length exceeds FFmpeg's keyframe interval limit");
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
    config.Values["video-chunk-frames"] = std::to_string(config.ChunkFrames);
    config.Values["video-fps"] = std::to_string(config.Fps);
    config.Values["wide-iterations"] = std::to_string(config.WideIterations);
    std::ostringstream ratio;
    ratio << std::setprecision(std::numeric_limits<double>::max_digits10) << config.SlopeRatio;
    config.Values["la-slope-ratio"] = ratio.str();
    config.Values.try_emplace("width", "1024");
    config.Values.try_emplace("height", "768");
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
            key == "video-chunk-frames" || key == "video-fps") {
            continue;
        }
        arguments.push_back("--" + key);
        arguments.push_back(value);
    }
    return arguments;
}
std::string
MovieConfig::Serialize() const
{
    std::string result = "[movie]\n";
    for (const auto &[key, value] : Values) {
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
                             double slopeRatio)
    : m_Target(HighPrecision::SetPrecision::True, 256), m_Step(HighPrecision::SetPrecision::True, 256),
      m_LaDepth(laDepth), m_SlopeRatio(slopeRatio), m_WideIterations(wideIterations),
      m_TargetIterations(targetIterations)
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
    // Conservatively stay below 2^64: MSVC long double has double precision.
    if (!std::isfinite(intervals) || intervals >= std::ldexp(1.0L, 64) - 4096 || m_StepDepth <= 0) {
        throw FractalSharkSeriousException("Movie frame count overflows 64 bits");
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
        if (intervalCount == UINT64_MAX - 1) {
            throw FractalSharkSeriousException("Movie frame count overflows 64 bits");
        }
        ++intervalCount;
    }
    m_FrameCount = intervalCount + 1;
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
    return index == 0 ? 0 : std::max(0.0L, m_Depth - (m_FrameCount - 1 - index) * m_StepDepth);
}
HighPrecision
MovieSchedule::ZoomAt(uint64_t index) const
{
    DepthAt(index);
    HighPrecision result{HighPrecision::SetPrecision::True, 256};
    if (index == 0 && m_FrameCount > 1) {
        mpf_set_ui(result.backend(), 1);
        return result;
    }
    const auto factor = ScalePower(m_Step, m_FrameCount - 1 - index);
    mpf_div(result.backend(), m_Target.backend(), factor.backend());
    return result;
}
uint64_t
MovieSchedule::IterationsAt(uint64_t index) const
{
    const auto depth = DepthAt(index);
    if (index == m_FrameCount - 1) {
        return m_TargetIterations;
    }
    if (index == 0) {
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
    const auto snapshot = m_Directory / "movie.ini";
    std::filesystem::create_directories(m_Directory / "frames");
    std::filesystem::create_directories(m_Directory / "chunks");
    if (std::filesystem::exists(snapshot)) {
        m_Config = MovieConfig::Load(snapshot);
        if (!configuration.empty()) {
            auto candidate = MovieConfig::Load(configuration);
            if (candidate.Values.contains("palette-map")) {
                if (ReadMovieText(candidate.Values.at("palette-map")) !=
                    ReadMovieText(m_Config.Values.at("palette-map"))) {
                    throw FractalSharkSeriousException("Movie palette differs from the job snapshot");
                }
                candidate.Values["palette-map"] = m_Config.Values.at("palette-map");
            }
            if (candidate.Serialize() != m_Config.Serialize()) {
                throw FractalSharkSeriousException("Movie settings differ from the existing job");
            }
        }
    } else {
        if (configuration.empty()) {
            throw FractalSharkSeriousException("New movie job requires --movie-config");
        }
        m_Config = MovieConfig::Load(configuration);
        if (m_Config.Values.contains("palette-map")) {
            WriteMovieText(m_Directory / "palette.map",
                           ReadMovieText(m_Config.Values.at("palette-map")));
            m_Config.Values["palette-map"] = "palette.map";
        }
        WriteMovieText(snapshot, m_Config.Serialize());
        m_Config = MovieConfig::Load(snapshot);
    }
    auto portableConfig = m_Config;
    if (portableConfig.Values.contains("palette-map")) {
        portableConfig.Values["palette-map"] = "palette.map";
    }
    m_ConfigHash = MovieHash::Digest(portableConfig.Serialize());
    if (m_Config.Values.contains("palette-map")) {
        m_ConfigHash =
            MovieHash::Digest(m_ConfigHash + ReadMovieText(m_Config.Values.at("palette-map")));
    }
    // The job lock is held and no new workers exist. Only reserved temporary
    // names are removed; directory enumeration retains no frame-path list.
    RemoveUnfinishedMovies(m_Directory);
    RemoveUnfinishedMovies(m_Directory / "chunks");
    for (const auto &entry : std::filesystem::recursive_directory_iterator(m_Directory / "frames")) {
        const auto name = entry.path().filename().string();
        if (name.starts_with(".fractalshark-save-") && entry.path().extension() == ".tmp") {
            RemoveMovieTemporary(entry.path());
        }
    }
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
uint64_t
MovieJob::Remaining() const
{
    return m_Remaining;
}
bool
MovieJob::GpuBypassed() const
{
    return m_GpuBypassed;
}
long double
MovieJob::LaDepth() const
{
    return m_LaDepth;
}
void
MovieJob::InitializeProgress(uint64_t frameCount, bool gpuBypassed, long double laDepth)
{
    const auto profilePath = m_Directory / "profile.txt";
    if (std::filesystem::exists(profilePath)) {
        const auto profile = Unseal(profilePath);
        std::string identity;
        if (!profile) {
            throw FractalSharkSeriousException("Movie profile checksum failed");
        }
        std::istringstream input(*profile);
        input >> identity >> m_FrameCount >> m_GpuBypassed >> m_LaDepth;
        if (!input || identity != m_ConfigHash || frameCount != m_FrameCount ||
            gpuBypassed != m_GpuBypassed) {
            throw FractalSharkSeriousException("Movie profile or CPU/GPU availability changed");
        }
    } else {
        m_FrameCount = frameCount;
        m_GpuBypassed = gpuBypassed;
        m_LaDepth = laDepth;
        std::ostringstream profile;
        profile << m_ConfigHash << '\n'
                << frameCount << ' ' << gpuBypassed << ' '
                << std::setprecision(std::numeric_limits<long double>::max_digits10) << laDepth << '\n';
        WriteMovieText(profilePath, Seal(profile.str()));
    }
    m_Remaining = m_FrameCount;
    for (const auto name : {"progress-0.txt", "progress-1.txt"}) {
        if (const auto record = Unseal(m_Directory / name)) {
            std::istringstream input(*record);
            std::string identity;
            uint64_t sequence = 0, remaining = 0;
            input >> identity >> sequence >> remaining;
            if (input && identity == m_ConfigHash && remaining <= m_FrameCount &&
                sequence > m_Sequence) {
                m_Sequence = sequence;
                m_Remaining = remaining;
            }
        }
    }
}
void
MovieJob::SaveProgress()
{
    ++m_Sequence;
    const auto payload =
        m_ConfigHash + "\n" + std::to_string(m_Sequence) + " " + std::to_string(m_Remaining) + "\n";
    WriteMovieText(m_Directory / ("progress-" + std::to_string(m_Sequence % 2) + ".txt"), Seal(payload));
}
void
MovieJob::CompleteFrame(uint64_t index)
{
    if (m_Remaining == 0 || index != m_Remaining - 1) {
        throw FractalSharkSeriousException("Movie frames must commit in descending order");
    }
    m_Remaining = index;
    SaveProgress();
}
void
MovieJob::Rewind(uint64_t remaining)
{
    if (remaining > m_FrameCount || remaining < m_Remaining) {
        throw FractalSharkSeriousException("Invalid movie recovery frontier");
    }
    m_Remaining = remaining;
    SaveProgress();
}
void
MovieJob::SkipArchivedChunk(const MovieChunk &chunk)
{
    if (m_Remaining > chunk.First && m_Remaining <= chunk.First + chunk.Count) {
        m_Remaining = chunk.First;
        SaveProgress();
    }
}
std::filesystem::path
MovieJob::FramePath(uint64_t index) const
{
    return m_Directory / "frames" / Padded(index / 10000) / (Padded(index) + ".png");
}
std::filesystem::path
MovieJob::VideoPath(uint64_t first) const
{
    return m_Directory / "chunks" / (Padded(first) + ".mkv");
}
std::optional<MovieChunk>
MovieJob::FindChunk(uint64_t first) const
{
    auto path = VideoPath(first);
    path += ".complete";
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto record = Unseal(path);
    if (!record) {
        throw FractalSharkSeriousException("Movie chunk registry checksum failed");
    }
    std::istringstream input(*record);
    std::string identity;
    MovieChunk chunk{};
    input >> identity >> chunk.First >> chunk.Count >> chunk.PngBytes >> chunk.VideoBytes >>
        chunk.RgbHash;
    const auto expectedCount = std::min(m_Config.ChunkFrames, m_FrameCount - first);
    if (!input || identity != m_ConfigHash || chunk.First != first || chunk.Count != expectedCount ||
        chunk.RgbHash.size() != 64 || !std::filesystem::exists(VideoPath(first)) ||
        std::filesystem::file_size(VideoPath(first)) != chunk.VideoBytes) {
        throw FractalSharkSeriousException("Movie chunk registry or movie file is invalid");
    }
    return chunk;
}
void
MovieJob::CommitChunk(const MovieChunk &chunk)
{
    std::ostringstream record;
    record << m_ConfigHash << '\n'
           << chunk.First << ' ' << chunk.Count << ' ' << chunk.PngBytes << ' ' << chunk.VideoBytes
           << ' ' << chunk.RgbHash << '\n';
    auto path = VideoPath(chunk.First);
    path += ".complete";
    WriteMovieText(path, Seal(record.str()));
}
void
MovieJob::DeleteChunkPngs(const MovieChunk &chunk)
{
    // Registry publication precedes every deletion. Retrying after a crash is
    // idempotent, including a crash halfway through the loop.
    if (!FindChunk(chunk.First)) {
        throw FractalSharkSeriousException("Cannot delete uncommitted movie PNGs");
    }
    for (uint64_t i = 0; i < chunk.Count; ++i) {
        std::filesystem::remove(FramePath(chunk.First + i));
    }
}
void
MovieJob::WriteConcatManifest(uint64_t frameCount)
{
    std::ostringstream output;
    output << "ffconcat version 1.0\n";
    for (uint64_t first = 0; first < frameCount;) {
        if (const auto chunk = FindChunk(first)) {
            output << "file 'chunks/" << VideoPath(first).filename().string() << "'\n";
            // Explicit duration avoids accumulating Matroska's millisecond
            // timestamp rounding over thousands of concatenated chunks.
            output << "duration " << std::fixed << std::setprecision(9)
                   << static_cast<long double>(chunk->Count) / m_Config.Fps << '\n';
        }
        const auto count = std::min(m_Config.ChunkFrames, frameCount - first);
        first += count;
    }
    WriteMovieText(m_Directory / "movie.ffconcat", output.str());
}
std::filesystem::path
MovieJob::Encoder(const std::filesystem::path &overridePath)
{
    const auto hint = m_Directory / "ffmpeg.txt";
    if (!overridePath.empty()) {
        const auto absolute = std::filesystem::absolute(overridePath).lexically_normal();
        if (!std::filesystem::is_regular_file(absolute)) {
            throw FractalSharkSeriousException("FFmpeg executable does not exist");
        }
        WriteMovieText(hint, absolute.string());
        return absolute;
    }
    if (std::filesystem::exists(hint)) {
        const std::filesystem::path saved(ReadMovieText(hint));
        if (!std::filesystem::is_regular_file(saved)) {
            throw FractalSharkSeriousException("Saved FFmpeg path is unavailable; supply --ffmpeg PATH");
        }
        return saved;
    }
    return {};
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
