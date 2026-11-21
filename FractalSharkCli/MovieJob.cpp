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
// Format identity also salts every checkpoint and chunk record. Old playback-
// indexed AV1 jobs cannot be mistaken for ascending HEVC render-order jobs.
constexpr std::string_view MovieFormat = "FractalSharkMovie3-hevc-nvenc-render-order";
constexpr std::string_view PreviousMovieFormat = "FractalSharkMovie2-hevc-nvenc-render-order";
constexpr std::string_view AssemblyFormat = "FractalSharkAssembly1-mpegts";
constexpr std::array ProgressNames{"progress.txt", "progress-0.txt", "progress-1.txt"};
constexpr std::array TimingNames{"time.txt", "time-0.txt", "time-1.txt"};

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
Unseal(const std::filesystem::path &path, bool allowPrevious)
{
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto contents = ReadMovieText(path);
    const auto newline = contents.find('\n');
    constexpr size_t CrcDigits = 8;
    constexpr size_t PreviousChecksumDigits = 64;
    if (newline == CrcDigits && MovieMetadataChecksum(std::string_view(contents).substr(newline + 1)) ==
                                    contents.substr(0, newline)) {
        return contents.substr(newline + 1);
    }
    // Only the one-time HEVC upgrade accepts obsolete checksum headers. Their
    // payloads are checked structurally against the locked job and movie sizes;
    // no cryptographic implementation is retained for these small old records.
    if (allowPrevious && newline == PreviousChecksumDigits &&
        contents.substr(0, newline).find_first_not_of("0123456789abcdef") == std::string::npos) {
        return contents.substr(newline + 1);
    }
    return std::nullopt;
}

// Both running checkpoints use the same sealed identity, sequence and value.
// The numbered files are accepted only as migration inputs. A .tmp file is
// never authoritative, even if all its bytes reached storage before a crash.
struct MovieCheckpoint {
    uint64_t Sequence = 0;
    uint64_t Value = 0;
    bool HasLegacyFiles = false;
};

MovieCheckpoint
ReadCheckpoint(const std::filesystem::path &directory,
               const std::array<const char *, 3> &names,
               std::string_view identity,
               uint64_t maximum)
{
    MovieCheckpoint best;
    for (size_t index = 0; index < names.size(); ++index) {
        const auto path = directory / names[index];
        if (!std::filesystem::exists(path)) {
            continue;
        }
        best.HasLegacyFiles |= index != 0;
        if (const auto record = Unseal(path, false)) {
            std::istringstream input(*record);
            std::string savedIdentity;
            uint64_t sequence = 0, value = 0;
            input >> savedIdentity >> sequence >> value;
            if (input && savedIdentity == identity && sequence > best.Sequence && value <= maximum) {
                input >> std::ws;
                if (input.eof()) {
                    best.Sequence = sequence;
                    best.Value = value;
                }
            }
        }
    }
    return best;
}

void
RemoveLegacyCheckpoints(const std::filesystem::path &directory, const std::array<const char *, 3> &names)
{
    // Only retire the migration inputs after the canonical record is durable.
    for (size_t index = 1; index < names.size(); ++index) {
        std::filesystem::remove(directory / names[index]);
    }
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
    // Reserved encoder temporaries include private hard-link input directories.
    // The exclusive job lock is held; no new worker can own these stale names.
    std::filesystem::remove_all(path, error);
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

// Only this worker writes timing metadata. Rendering, frame checkpoints and
// archive commits stay on their existing threads. Finalization joins first,
// then publishes the exact final duration while the job lock is still owned.
class MovieJob::Timing {
public:
    Timing(const std::filesystem::path &directory, const std::string &identity, bool existing)
        : m_Directory(directory), m_Identity(identity), m_Started(std::chrono::steady_clock::now())
    {
        const auto saved = ReadCheckpoint(directory, TimingNames, identity, UINT64_MAX);
        m_Sequence = saved.Sequence;
        m_PreviousMilliseconds = saved.Value;
        if (existing && m_Sequence == 0) {
            FractalSharkLog::Write(
                "Movie timing begins with this session; earlier active time was not recorded",
                __FILE__,
                __LINE__);
        }
        Save();
        if (saved.HasLegacyFiles) {
            RemoveLegacyCheckpoints(directory, TimingNames);
        }
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
        WriteMovieText(m_Directory / TimingNames.front(), Seal(payload));
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
                                      "video-fps",
                                      "render-order",
                                      "video-encoder-args"};
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
    const auto encoderText = config.Values.contains("video-encoder-args")
                                 ? config.Values.at("video-encoder-args")
                                 : std::string(FractalShark::LosslessMovieEncoderArguments());
    // Legacy snapshots omit the key and retain RGB lossless. MovieJob writes
    // the current default explicitly before starting a new job.
    // Deliberately no shell expansion. Double quotes group option values;
    // filenames, filter graphs and global/process options are not encoder knobs.
    bool quoted = false;
    std::string token;
    for (const char ch : encoderText) {
        if (ch == '"') {
            quoted = !quoted;
        } else if (!quoted && (ch == ' ' || ch == '\t')) {
            if (!token.empty()) {
                config.EncoderArguments.push_back(std::move(token));
                token.clear();
            }
        } else {
            token += ch;
        }
    }
    if (quoted) {
        throw FractalSharkSeriousException("Unterminated movie encoder argument quote");
    }
    if (!token.empty()) {
        config.EncoderArguments.push_back(std::move(token));
    }
    const std::set<std::string> encoderOptions{
        "-c:v",        "-preset",      "-tune",        "-profile:v",  "-pix_fmt",     "-rc",
        "-qp",         "-cq",          "-b:v",         "-maxrate:v",  "-bufsize:v",   "-multipass",
        "-spatial-aq", "-temporal-aq", "-aq-strength", "-colorspace", "-color_range", "-color_primaries",
        "-color_trc",  "-level:v"};
    std::map<std::string, std::string> selected;
    if (config.EncoderArguments.size() % 2 != 0) {
        throw FractalSharkSeriousException("Movie encoder arguments require option/value pairs");
    }
    for (size_t i = 0; i < config.EncoderArguments.size(); i += 2) {
        const auto &option = config.EncoderArguments[i];
        if (!encoderOptions.contains(option) ||
            !selected.emplace(option, config.EncoderArguments[i + 1]).second) {
            throw FractalSharkSeriousException("Unsupported or duplicate movie encoder option: " +
                                               option);
        }
    }
    if (!selected.contains("-c:v") ||
        (selected.at("-c:v") != "hevc_nvenc" && selected.at("-c:v") != "h264_nvenc")) {
        throw FractalSharkSeriousException("Movie encoder must be hevc_nvenc or h264_nvenc");
    }
    if (!selected.contains("-pix_fmt") ||
        (selected.at("-pix_fmt") != "gbrp" && selected.at("-pix_fmt") != "nv12" &&
         selected.at("-pix_fmt") != "yuv420p")) {
        throw FractalSharkSeriousException("Movie pixel format must be gbrp, nv12 or yuv420p");
    }
    config.PixelFormat = selected.at("-pix_fmt");
    const bool rgb = config.PixelFormat == "gbrp";
    for (const auto &[name, value] : std::map<std::string, std::string>{
             {"-colorspace", rgb ? "rgb" : "bt709"}, {"-color_range", rgb ? "pc" : "tv"}}) {
        if (selected.contains(name) && selected.at(name) != value) {
            throw FractalSharkSeriousException("Movie color settings disagree with its pixel format");
        }
        if (!selected.contains(name)) {
            config.EncoderArguments.push_back(name);
            config.EncoderArguments.push_back(value);
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
            key == "video-chunk-frames" || key == "video-fps" || key == "render-order" ||
            key == "video-encoder-args") {
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
    // FFmpeg image2 accepts a signed-int starting index. Excessive frame counts
    // are configuration errors, rather than requiring another input protocol.
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
    // Recognize only the current layout and its immediately preceding HEVC
    // metadata. Incompatible playback-indexed jobs are rejected before cleanup.
    const auto profilePath = m_Directory / "profile.txt";
    std::optional<std::string> savedProfile;
    bool upgrade = false;
    if (std::filesystem::exists(profilePath)) {
        savedProfile = Unseal(profilePath, true);
        std::string format;
        std::istringstream input(savedProfile ? *savedProfile : std::string{});
        input >> m_Identity >> format;
        if (!input || (format != MovieFormat && format != PreviousMovieFormat) ||
            (format == MovieFormat && !Unseal(profilePath, false))) {
            throw FractalSharkSeriousException("Incompatible movie job format; create a new job folder");
        }
        upgrade = format == PreviousMovieFormat;
    } else {
        const auto hasFiles = [&](const char *name) {
            const auto path = m_Directory / name;
            return std::filesystem::exists(path) && !std::filesystem::is_empty(path);
        };
        const bool hasProgress = std::any_of(ProgressNames.begin(), ProgressNames.end(), [&](auto name) {
            return std::filesystem::exists(m_Directory / name);
        });
        if (hasProgress || std::filesystem::exists(m_Directory / "assembly.txt") || hasFiles("frames") ||
            hasFiles("chunks")) {
            throw FractalSharkSeriousException(
                "Movie state has no current profile; create a new job folder");
        }
    }
    const auto snapshot = m_Directory / "movie.ini";
    std::filesystem::create_directories(m_Directory / "frames");
    std::filesystem::create_directories(m_Directory / "chunks");
    const bool snapshotExists = std::filesystem::exists(snapshot);
    if (snapshotExists) {
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
            // Older external INIs may omit the new-job defaults materialized in
            // this snapshot. Normalize only those keys the snapshot already has;
            // legacy snapshots remain byte-for-byte identical for fingerprinting.
            if (m_Config.Values.contains("render-order")) {
                candidate.Values.try_emplace("render-order", "wide-to-deep");
            }
            if (m_Config.Values.contains("video-encoder-args")) {
                candidate.Values.try_emplace("video-encoder-args",
                                             m_Config.Values.at("video-encoder-args"));
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
    }
    if (!snapshotExists || (!savedProfile && (!m_Config.Values.contains("render-order") ||
                                              !m_Config.Values.contains("video-encoder-args")))) {
        m_Config.Values.try_emplace("render-order", "wide-to-deep");
        m_Config.Values.try_emplace("video-encoder-args",
                                    std::string(FractalShark::DefaultMovieEncoderArguments()));
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
    m_ConfigChecksum = MovieMetadataChecksum(configurationText);
    if (!savedProfile) {
        m_Identity = m_ConfigChecksum;
    } else if (upgrade) {
        UpgradeMetadata(*savedProfile);
    } else {
        std::string identity, format, configurationChecksum;
        std::istringstream input(*savedProfile);
        input >> identity >> format >> m_FrameCount >> m_GpuBypassed >> m_LaDepth >>
            configurationChecksum;
        if (!input || configurationChecksum != m_ConfigChecksum || m_FrameCount == 0 ||
            m_FrameCount > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
            !std::isfinite(m_LaDepth) || m_LaDepth < 0) {
            throw FractalSharkSeriousException("Movie settings differ from the job profile");
        }
    }
    if (savedProfile) {
        RestoreProgress();
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
    m_Timing = std::make_unique<Timing>(m_Directory, m_Identity, savedProfile.has_value());
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
MovieJob::UpgradeMetadata(const std::string &profile)
{
    std::string identity, format;
    std::istringstream input(profile);
    input >> identity >> format >> m_FrameCount >> m_GpuBypassed >> m_LaDepth;
    if (!input || identity != m_Identity || format != PreviousMovieFormat || m_FrameCount == 0 ||
        m_FrameCount > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        !std::isfinite(m_LaDepth) || m_LaDepth < 0) {
        throw FractalSharkSeriousException("Cannot upgrade invalid movie profile");
    }
    // Validate every record before changing any. Keep the identifier unchanged,
    // so CRC records written before an interruption remain readable on retry.
    std::vector<std::pair<std::filesystem::path, std::string>> records;
    for (const auto name : {"progress-0.txt", "progress-1.txt"}) {
        const auto path = m_Directory / name;
        if (const auto payload = Unseal(path, true)) {
            std::istringstream progress(*payload);
            uint64_t sequence = 0, nextFrame = 0;
            progress >> identity >> sequence >> nextFrame;
            if (progress && identity == m_Identity && sequence > 0 && nextFrame <= m_FrameCount) {
                progress >> std::ws;
                if (progress.eof()) {
                    records.emplace_back(path, *payload);
                }
            }
        }
    }
    for (const auto &entry : std::filesystem::directory_iterator(m_Directory / "chunks")) {
        if (entry.path().extension() != ".complete") {
            continue;
        }
        const auto payload = Unseal(entry.path(), true);
        std::istringstream record(payload ? *payload : std::string{});
        MovieChunk chunk{};
        record >> identity >> chunk.First >> chunk.Count >> chunk.PngBytes >> chunk.VideoBytes;
        auto completionPath = VideoPath(chunk.First);
        completionPath += ".complete";
        if (!record || identity != m_Identity || chunk.First >= m_FrameCount ||
            chunk.First % m_Config.ChunkFrames != 0 ||
            chunk.Count != std::min(m_Config.ChunkFrames, m_FrameCount - chunk.First) ||
            chunk.VideoBytes == 0 || !std::filesystem::is_regular_file(VideoPath(chunk.First)) ||
            std::filesystem::file_size(VideoPath(chunk.First)) != chunk.VideoBytes ||
            entry.path() != completionPath) {
            throw FractalSharkSeriousException("Cannot upgrade invalid movie completion record");
        }
        record >> std::ws;
        if (!record.eof()) {
            // An already-converted record has no trailing pixel digest. Older
            // records may have exactly one obsolete 64-digit field to discard.
            std::string obsolete;
            record >> obsolete;
            if (obsolete.size() != 64 ||
                obsolete.find_first_not_of("0123456789abcdef") != std::string::npos) {
                throw FractalSharkSeriousException("Invalid obsolete movie digest field");
            }
            record >> std::ws;
            if (!record.eof()) {
                throw FractalSharkSeriousException("Unexpected movie completion fields");
            }
        }
        std::ostringstream converted;
        converted << m_Identity << '\n'
                  << chunk.First << ' ' << chunk.Count << ' ' << chunk.PngBytes << ' '
                  << chunk.VideoBytes << '\n';
        records.emplace_back(entry.path(), converted.str());
    }
    for (const auto &[path, payload] : records) {
        WriteMovieText(path, Seal(payload));
    }
    // Publishing this marker LAST commits the upgrade. A crash beforehand leaves
    // the old profile and a mix of old/CRC records, accepted by this method only.
    SaveProfile();
    FractalSharkLog::Write(
        "Movie metadata upgraded; saved frames and movies retained", __FILE__, __LINE__);
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
    const auto saved = ReadCheckpoint(m_Directory, ProgressNames, m_Identity, m_FrameCount);
    m_NextFrame = saved.Value;
    m_Sequence = saved.Sequence;
    if (saved.HasLegacyFiles) {
        SaveProgress();
        RemoveLegacyCheckpoints(m_Directory, ProgressNames);
    }
}
void
MovieJob::SaveProgress()
{
    ++m_Sequence;
    const auto payload =
        m_Identity + "\n" + std::to_string(m_Sequence) + " " + std::to_string(m_NextFrame) + "\n";
    WriteMovieText(m_Directory / ProgressNames.front(), Seal(payload));
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
void
MovieJob::SkipArchivedChunk(const MovieChunk &chunk)
{
    if (m_NextFrame >= chunk.First && m_NextFrame < chunk.First + chunk.Count) {
        m_NextFrame = chunk.First + chunk.Count;
        SaveProgress();
    }
}
std::filesystem::path
MovieJob::FramePath(uint64_t index) const
{
    return m_Directory / "frames" / ("chunk_" + Padded(index / m_Config.ChunkFrames)) /
           ("frame_" + Padded(index) + ".png");
}
std::filesystem::path
MovieJob::VideoPath(uint64_t first) const
{
    return m_Directory / "chunks" / ("chunk_" + Padded(first / m_Config.ChunkFrames) + ".mkv");
}
std::optional<MovieChunk>
MovieJob::FindChunk(uint64_t first) const
{
    if (first >= m_FrameCount || first % m_Config.ChunkFrames != 0) {
        throw FractalSharkSeriousException("Invalid movie chunk start");
    }
    auto path = VideoPath(first);
    path += ".complete";
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto record = Unseal(path, false);
    if (!record) {
        throw FractalSharkSeriousException("Movie chunk registry checksum failed");
    }
    std::istringstream input(*record);
    std::string identity;
    MovieChunk chunk{};
    input >> identity >> chunk.First >> chunk.Count >> chunk.PngBytes >> chunk.VideoBytes;
    const auto expectedCount = std::min(m_Config.ChunkFrames, m_FrameCount - first);
    if (!input || identity != m_Identity || chunk.First != first || chunk.Count != expectedCount ||
        chunk.VideoBytes == 0 || !std::filesystem::is_regular_file(VideoPath(first)) ||
        std::filesystem::file_size(VideoPath(first)) != chunk.VideoBytes) {
        throw FractalSharkSeriousException("Movie chunk registry or movie file is invalid");
    }
    return chunk;
}
void
MovieJob::CommitChunk(const MovieChunk &chunk)
{
    std::ostringstream record;
    record << m_Identity << '\n'
           << chunk.First << ' ' << chunk.Count << ' ' << chunk.PngBytes << ' ' << chunk.VideoBytes
           << '\n';
    auto path = VideoPath(chunk.First);
    path += ".complete";
    WriteMovieText(path, Seal(record.str()));
}
void
MovieJob::DeleteChunkPngs(const MovieChunk &chunk)
{
    // Each folder belongs to exactly one immutable chunk. Only a durable,
    // validated completion record permits deleting that entire folder.
    const auto registered = FindChunk(chunk.First);
    if (!registered || registered->Count != chunk.Count) {
        throw FractalSharkSeriousException("Cannot delete uncommitted movie PNGs");
    }
    if (std::filesystem::remove_all(FramePath(chunk.First).parent_path()) != 0) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Movie removed frame folder chunk=" << chunk.First / m_Config.ChunkFrames
            << " frames=" << chunk.Count;
    }
}
uint64_t
MovieJob::ChunkCount() const
{
    return m_FrameCount == 0 ? 0 : 1 + (m_FrameCount - 1) / m_Config.ChunkFrames;
}
uint64_t
MovieJob::PlaybackChunkStart(uint64_t ordinal) const
{
    const auto chunks = ChunkCount();
    if (ordinal >= chunks) {
        throw FractalSharkSeriousException("Invalid movie playback chunk");
    }
    // Legacy chunks are internally in playback order already, but their file
    // ordinals run from the deep end towards the wide end.
    return (m_Config.WideToDeep ? ordinal : chunks - 1 - ordinal) * m_Config.ChunkFrames;
}
bool
MovieJob::HasAllChunks() const
{
    const auto chunks = ChunkCount();
    for (uint64_t ordinal = 0; ordinal < chunks; ++ordinal) {
        if (!FindChunk(PlaybackChunkStart(ordinal))) {
            return false;
        }
    }
    return chunks != 0;
}
std::optional<MovieAssemblyProgress>
MovieJob::ReadAssembly() const
{
    const auto path = m_Directory / "assembly.txt";
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    const auto record = Unseal(path, false);
    std::istringstream input(record ? *record : std::string{});
    std::string identity, format;
    uint64_t frameCount = 0;
    MovieAssemblyProgress progress{};
    input >> identity >> format >> frameCount >> progress.NextChunk >> progress.Bytes;
    if (!input || identity != m_Identity || format != AssemblyFormat || frameCount != m_FrameCount ||
        frameCount == 0 || progress.NextChunk > ChunkCount() ||
        ((progress.NextChunk == 0) != (progress.Bytes == 0))) {
        throw FractalSharkSeriousException("Movie assembly checkpoint is invalid");
    }
    input >> std::ws;
    if (!input.eof()) {
        throw FractalSharkSeriousException("Unexpected movie assembly checkpoint fields");
    }
    return progress;
}
void
MovieJob::SaveAssembly(const MovieAssemblyProgress &progress)
{
    // Every source chunk has already been registered at assembly entry, even
    // if a previous shutdown left the render frontier behind those records.
    if (m_NextFrame != m_FrameCount) {
        m_NextFrame = m_FrameCount;
        SaveProgress();
    }
    std::ostringstream record;
    record << m_Identity << '\n'
           << AssemblyFormat << '\n'
           << m_FrameCount << ' ' << progress.NextChunk << ' ' << progress.Bytes << '\n';
    WriteMovieText(m_Directory / "assembly.txt", Seal(record.str()));
}
void
MovieJob::DeleteAssembledChunk(uint64_t ordinal)
{
    // Caller holds the job lock and has verified the durable output prefix
    // before retiring any source name. Each unlink may be repeated on resume.
    auto path = VideoPath(PlaybackChunkStart(ordinal));
    std::filesystem::remove(path);
    path += ".complete";
    std::filesystem::remove(path);
}
void
MovieJob::WriteConcatManifest()
{
    std::ostringstream output;
    output << "ffconcat version 1.0\n";
    // Forward jobs already match playback order. Legacy jobs reverse both the
    // frames inside their movies and the order of their completed chunks.
    const auto chunks = ChunkCount();
    for (uint64_t ordinal = 0; ordinal < chunks; ++ordinal) {
        const auto first = PlaybackChunkStart(ordinal);
        if (const auto chunk = FindChunk(first)) {
            output << "file 'chunks/" << VideoPath(first).filename().string() << "'\n";
            // Explicit duration avoids accumulating Matroska's millisecond
            // timestamp rounding over thousands of concatenated chunks.
            output << "duration " << std::fixed << std::setprecision(9)
                   << static_cast<long double>(chunk->Count) / m_Config.Fps << '\n';
        }
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
        WriteMovieText(hint, Environment::WideToUtf8(Environment::FromFsPath(absolute)));
        return absolute;
    }
    if (std::filesystem::exists(hint)) {
        const auto saved = Environment::ToFsPath(Environment::Utf8ToWide(ReadMovieText(hint)));
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
