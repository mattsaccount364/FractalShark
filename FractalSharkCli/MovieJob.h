#pragma once

#include "FractalSharkGpuLib/NvencMovieEncoder.h"
#include "HighPrecision.h"
#include "MoviePlatform.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace FractalSharkCli {

struct MovieConfig {
    std::map<std::string, std::string> Values;
    FractalShark::MovieVideoSettings Video;
    std::filesystem::path VideoOutput;
    uint64_t WideIterations = 8192;
    double SlopeRatio = 8;
    bool WideToDeep = true;

    static MovieConfig Load(const std::filesystem::path &path);
    std::vector<std::string> RenderArguments() const;
    std::string Serialize(bool includeComments) const;
};

class MovieSchedule {
public:
    MovieSchedule(const HighPrecision &targetZoom,
                  const std::string &step,
                  uint64_t wideIterations,
                  uint64_t targetIterations,
                  long double laDepth,
                  double slopeRatio,
                  bool wideToDeep);
    uint64_t FrameCount() const;
    long double DepthAt(uint64_t index) const;
    HighPrecision ZoomAt(uint64_t index) const;
    uint64_t IterationsAt(uint64_t index) const;
    static long double Log2(const HighPrecision &value);

private:
    // Zoom scale needs relative precision, not millions of fractional digits.
    // Keep the center at full precision separately; 256 scale bits leave ample
    // margin beyond the renderer's per-pixel precision requirements.
    bool m_WideToDeep;
    HighPrecision m_Target;
    HighPrecision m_Step;
    long double m_Depth;
    long double m_StepDepth;
    long double m_LaDepth;
    double m_SlopeRatio;
    uint64_t m_WideIterations;
    uint64_t m_TargetIterations;
    uint64_t m_FrameCount;
};

// The job owns its checkpoint and staging path. The OS lock remains owned until
// all render/save/encoding workers have stopped, including exceptional exits.
class MovieJob {
public:
    MovieJob(const std::filesystem::path &directory, const std::filesystem::path &configuration);
    ~MovieJob();
    bool FinishTiming();
    void CheckTiming() const;
    uint64_t ElapsedMilliseconds() const;
    const MovieConfig &Config() const;
    const std::filesystem::path &Directory() const;
    const std::filesystem::path &VideoOutputPath() const;
    std::filesystem::path StagingVideoPath() const;
    void InitializeProgress(uint64_t frameCount, bool gpuBypassed, long double laDepth);
    uint64_t FrameCount() const;
    uint64_t Remaining() const;
    uint64_t NextFrame() const;
    void CompleteFrame(uint64_t index);
    void Rewind(uint64_t nextFrame);
    uint64_t VideoBytes() const;
    void CommitVideo(uint64_t nextFrame, uint64_t bytes);
    bool GpuBypassed() const;
    long double LaDepth() const;
    std::filesystem::path FramePath(uint64_t index) const;

private:
    std::filesystem::path m_Directory;
    Environment::MovieJobLock m_Lock;
    MovieConfig m_Config;
    std::filesystem::path m_VideoOutput;
    class State;
    // Stop the state heartbeat before releasing the job lock above.
    std::unique_ptr<State> m_State;
};

// Validate a PNG left between publication and its PNG-only checkpoint.
std::vector<unsigned char> ReadMovieRgb8(const std::filesystem::path &path,
                                         unsigned width,
                                         unsigned height);
void WriteMovieText(const std::filesystem::path &path, const std::string &text);
std::string ReadMovieText(const std::filesystem::path &path);
// CRC32 is only for small metadata records, never image or movie payloads.
std::string MovieMetadataChecksum(std::string_view text);
} // namespace FractalSharkCli
