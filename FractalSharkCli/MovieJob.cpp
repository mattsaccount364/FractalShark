#include "stdafx.h"

#include "ConsoleLog.h"
#include "MovieCommandExport.h"
#include "MovieJob.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

namespace FractalSharkCli {
namespace {
constexpr std::string_view MovieFormat = "FractalSharkNativeMovie1";
constexpr auto ProgressName = "progress.txt";
constexpr auto TimingName = "time.txt";

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
    // INT_MAX is the frame-count limit, so ten decimal digits cover every index.
    text << std::setw(10) << std::setfill('0') << number;
    return text.str();
}
std::string
Seal(const std::string &payload)
{
    return MovieMetadataChecksum(payload) + "\n" + payload;
}
std::optional<std::string>
Unseal(const std::filesystem::path &path)
{
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto contents = ReadMovieText(path);
    const auto newline = contents.find('\n');
    constexpr size_t CrcDigits = 8;
    if (newline == CrcDigits && MovieMetadataChecksum(std::string_view(contents).substr(newline + 1)) ==
                                    contents.substr(0, newline)) {
        return contents.substr(newline + 1);
    }
    return std::nullopt;
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
} // namespace

class MovieJob::Timing {
public:
    Timing(const std::filesystem::path &directory, const std::string &identity, bool existing)
        : m_Directory(directory), m_Identity(identity), m_Started(std::chrono::steady_clock::now())
    {
        if (const auto saved = Unseal(directory / TimingName)) {
            std::istringstream input(*saved);
            std::string savedIdentity;
            uint64_t sequence = 0, elapsed = 0;
            if (input >> savedIdentity >> sequence >> elapsed && savedIdentity == identity) {
                m_Sequence = sequence;
                m_PreviousMilliseconds = elapsed;
            }
        }
        if (existing && m_Sequence == 0) {
            FractalSharkLog::Write("Movie timing resumes without an earlier record", __FILE__, __LINE__);
        }
        Save();
        m_Thread = std::jthread([this](std::stop_token stop) {
            // Exceptions cannot escape a worker. The movie event loop observes
            // the saved exception and stops rather than pretending time is saved.
            try {
                std::unique_lock lock(m_Mutex);
                while (!stop.stop_requested()) {
                    m_Wake.wait_for(lock, stop, std::chrono::seconds(1), [] { return false; });
                    if (stop.stop_requested()) {
                        break;
                    }
                    lock.unlock();
                    Save();
                    lock.lock();
                }
            } catch (...) {
                std::lock_guard lock(m_Mutex);
                m_Error = std::current_exception();
            }
        });
    }
    ~Timing() { Finish(); }
    uint64_t
    Elapsed() const
    {
        const auto final = m_FinalMilliseconds.load();
        if (final != UINT64_MAX) {
            return final;
        }
        return m_PreviousMilliseconds +
               static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::steady_clock::now() - m_Started)
                                         .count());
    }
    void
    Check() const
    {
        std::lock_guard lock(m_Mutex);
        if (m_Error) {
            std::rethrow_exception(m_Error);
        }
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
        m_FinalMilliseconds.store(Elapsed());
        // This is the persistence/error boundary, including destructor fallback.
        // A disk error is reported and changes a normal CLI exit to failure.
        try {
            Save();
            Check();
        } catch (const std::exception &exception) {
            m_Succeeded = false;
            FractalSharkLog::WriteException(
                "Could not persist movie active time", exception, __FILE__, __LINE__);
        }
        const auto total = Elapsed();
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Movie active wall time: total_seconds=" << static_cast<double>(total) / 1000
            << " session_seconds=" << static_cast<double>(total - m_PreviousMilliseconds) / 1000;
        return m_Succeeded;
    }

private:
    void
    Save()
    {
        ++m_Sequence;
        const auto payload =
            m_Identity + "\n" + std::to_string(m_Sequence) + " " + std::to_string(Elapsed()) + "\n";
        WriteMovieText(m_Directory / TimingName, Seal(payload));
    }
    std::filesystem::path m_Directory;
    std::string m_Identity;
    const std::chrono::steady_clock::time_point m_Started;
    uint64_t m_PreviousMilliseconds = 0;
    uint64_t m_Sequence = 0;
    std::atomic<uint64_t> m_FinalMilliseconds{UINT64_MAX};
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
    const std::set<std::string> known{"center-x",         "center-y",       "zoom",
                                      "iterations",       "zoom-per-frame", "width",
                                      "height",           "antialiasing",   "render-algorithm",
                                      "perturbation-alg", "palette-map",    "palette-depth",
                                      "wide-iterations",  "la-slope-ratio", "video-gop-frames",
                                      "video-fps",        "render-order",   "video-profile",
                                      "video-preset",     "video-qp",       "video-cq"};
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
        if (order != "wide-to-deep" && order != "deep-to-wide") {
            throw FractalSharkSeriousException("Invalid movie render-order");
        }
        config.WideToDeep = order == "wide-to-deep";
    }
    const auto profile = config.Values.try_emplace("video-profile", "hevc-high").first->second;
    auto &video = config.Video;
    if (profile == "hevc-low") {
        video.Quality = 22;
    } else if (profile == "hevc-rgb-lossless") {
        video.Codec = FractalShark::MovieCodec::HevcRgbLossless;
        video.Preset = 4;
        video.Quality = 0;
    } else if (profile == "h264-high" || profile == "h264-low") {
        video.Codec = FractalShark::MovieCodec::H264;
        video.Preset = profile == "h264-high" ? 5 : 1;
        video.Quality = profile == "h264-high" ? 18 : 30;
    } else if (profile == "png-only") {
        video.Codec = FractalShark::MovieCodec::PngOnly;
    } else if (profile != "hevc-high") {
        throw FractalSharkSeriousException("Unknown movie video-profile: " + profile);
    }

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
    if (video.Codec == FractalShark::MovieCodec::Hevc && !config.Values.contains("video-cq")) {
        video.Quality = static_cast<unsigned>(option("video-qp", video.Quality, 0, 51));
        config.Values["video-qp"] = std::to_string(video.Quality);
    } else if (video.Codec == FractalShark::MovieCodec::H264 && !config.Values.contains("video-qp")) {
        video.Quality = static_cast<unsigned>(option("video-cq", video.Quality, 1, 51));
        config.Values["video-cq"] = std::to_string(video.Quality);
    } else if (config.Values.contains("video-qp") || config.Values.contains("video-cq")) {
        throw FractalSharkSeriousException("Movie quality override does not match its video profile");
    }
    config.Values.try_emplace("render-order", "wide-to-deep");
    const std::array required{"center-x", "center-y", "zoom", "iterations", "zoom-per-frame"};
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
    config.Values["video-gop-frames"] = std::to_string(video.GopFrames);
    config.Values["video-fps"] = std::to_string(video.Fps);
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
            key.starts_with("video-") || key == "render-order") {
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
    const auto profilePath = m_Directory / "profile.txt";
    const bool existing = std::filesystem::exists(profilePath);
    const auto savedProfile = Unseal(profilePath);
    if (existing) {
        std::string format, identity;
        std::istringstream profile(savedProfile.value_or(""));
        profile >> identity >> format;
        if (!profile || format != MovieFormat) {
            throw FractalSharkSeriousException(
                "Incompatible movie job format; create a new native movie folder");
        }
    } else {
        for (const auto *name : {"ffmpeg.txt",
                                 "assembly.txt",
                                 "chunks",
                                 "frames",
                                 "movie.ts",
                                 "movie.ts.partial",
                                 ProgressName}) {
            if (std::filesystem::exists(m_Directory / name)) {
                throw FractalSharkSeriousException(
                    "Movie state has no native profile; create a new native movie folder");
            }
        }
    }

    const auto snapshot = m_Directory / "movie.ini";
    if (std::filesystem::exists(snapshot)) {
        m_Config = MovieConfig::Load(snapshot);
        if (!configuration.empty()) {
            auto candidate = MovieConfig::Load(configuration);
            if (candidate.Values.contains("palette-map") && m_Config.Values.contains("palette-map")) {
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
        if (existing || configuration.empty()) {
            throw FractalSharkSeriousException(
                "Movie has no configuration; supply --movie-config for a new job");
        }
        m_Config = MovieConfig::Load(configuration);
        if (m_Config.Values.contains("palette-map")) {
            WriteMovieText(m_Directory / "palette.map",
                           ReadMovieText(m_Config.Values.at("palette-map")));
            m_Config.Values["palette-map"] = "palette.map";
        }
        WriteMovieText(snapshot, m_Config.Serialize() + FractalShark::MovieEncoderExamples());
        m_Config = MovieConfig::Load(snapshot);
    }
    auto portableConfig = m_Config;
    if (portableConfig.Values.contains("palette-map")) {
        portableConfig.Values["palette-map"] = "palette.map";
    }
    auto configurationText = std::string(MovieFormat) + "\n" + portableConfig.Serialize();
    if (m_Config.Values.contains("palette-map")) {
        configurationText += ReadMovieText(m_Config.Values.at("palette-map"));
    }
    m_Identity = m_ConfigChecksum = MovieMetadataChecksum(configurationText);
    if (savedProfile) {
        std::string identity, format, configurationChecksum;
        std::istringstream input(*savedProfile);
        input >> identity >> format >> m_FrameCount >> m_GpuBypassed >> m_LaDepth >>
            configurationChecksum;
        if (!input || identity != m_Identity || configurationChecksum != m_ConfigChecksum ||
            m_FrameCount == 0 || m_FrameCount > INT_MAX || !std::isfinite(m_LaDepth) || m_LaDepth < 0) {
            throw FractalSharkSeriousException("Movie settings differ from the job profile");
        }
        RestoreProgress();
    }
    m_Timing = std::make_unique<Timing>(m_Directory, m_Identity, existing);
}
MovieJob::~MovieJob() = default;
bool
MovieJob::FinishTiming()
{
    return !m_Timing || m_Timing->Finish();
}
void
MovieJob::CheckTiming() const
{
    if (m_Timing) {
        m_Timing->Check();
    }
}
uint64_t
MovieJob::ElapsedMilliseconds() const
{
    return m_Timing ? m_Timing->Elapsed() : 0;
}

void
MovieJob::SaveProfile()
{
    std::ostringstream profile;
    profile << m_Identity << '\n'
            << MovieFormat << '\n'
            << m_FrameCount << ' ' << m_GpuBypassed << ' '
            << std::setprecision(std::numeric_limits<long double>::max_digits10) << m_LaDepth << '\n'
            << m_ConfigChecksum << '\n';
    WriteMovieText(m_Directory / "profile.txt", Seal(profile.str()));
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
MovieJob::FrameCount() const
{
    return m_FrameCount;
}
uint64_t
MovieJob::Remaining() const
{
    return m_FrameCount - m_NextFrame;
}
uint64_t
MovieJob::NextFrame() const
{
    return m_NextFrame;
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
    if (frameCount == 0 || frameCount > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw FractalSharkSeriousException("Movie frame count must be between one and INT_MAX");
    }
    if (m_FrameCount != 0) {
        if (frameCount != m_FrameCount || gpuBypassed != m_GpuBypassed) {
            throw FractalSharkSeriousException("Movie profile or CPU/GPU availability changed");
        }
    } else {
        m_FrameCount = frameCount;
        m_GpuBypassed = gpuBypassed;
        m_LaDepth = laDepth;
        SaveProfile();
    }
}
void
MovieJob::RestoreProgress()
{
    const auto path = m_Directory / ProgressName;
    if (!std::filesystem::exists(path)) {
        return;
    }
    const auto saved = Unseal(path);
    std::istringstream input(saved.value_or(""));
    std::string identity;
    input >> identity >> m_Sequence >> m_NextFrame >> m_VideoBytes;
    if (!input || identity != m_Identity || m_Sequence == 0 || m_NextFrame > m_FrameCount) {
        throw FractalSharkSeriousException("Invalid native movie checkpoint");
    }
    input >> std::ws;
    if (!input.eof() ||
        (m_Config.Video.Codec == FractalShark::MovieCodec::PngOnly && m_VideoBytes != 0)) {
        throw FractalSharkSeriousException("Invalid native movie checkpoint");
    }
}
void
MovieJob::SaveProgress()
{
    ++m_Sequence;
    const auto payload = m_Identity + "\n" + std::to_string(m_Sequence) + " " +
                         std::to_string(m_NextFrame) + " " + std::to_string(m_VideoBytes) + "\n";
    WriteMovieText(m_Directory / ProgressName, Seal(payload));
}
void
MovieJob::CompleteFrame(uint64_t index)
{
    if (m_NextFrame == m_FrameCount || index != m_NextFrame) {
        throw FractalSharkSeriousException("Movie frames must commit in ascending render order");
    }
    ++m_NextFrame;
    SaveProgress();
}
void
MovieJob::Rewind(uint64_t nextFrame)
{
    if (nextFrame > m_NextFrame) {
        throw FractalSharkSeriousException("Invalid movie recovery frontier");
    }
    m_NextFrame = nextFrame;
    SaveProgress();
}
uint64_t
MovieJob::VideoBytes() const
{
    return m_VideoBytes;
}
void
MovieJob::CommitVideo(uint64_t nextFrame, uint64_t bytes)
{
    if (nextFrame <= m_NextFrame || nextFrame > m_FrameCount || bytes <= m_VideoBytes) {
        throw FractalSharkSeriousException("Invalid movie video checkpoint");
    }
    m_NextFrame = nextFrame;
    m_VideoBytes = bytes;
    SaveProgress();
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
