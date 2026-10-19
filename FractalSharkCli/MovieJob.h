#pragma once

#include "HighPrecision.h"
#include "MoviePlatform.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace FractalSharkCli {

struct MovieConfig {
    std::map<std::string, std::string> Values;
    uint64_t ChunkFrames = 1000;
    uint64_t WideIterations = 8192;
    double SlopeRatio = 8;
    unsigned Fps = 60;

    static MovieConfig Load(const std::filesystem::path &path);
    std::vector<std::string> RenderArguments() const;
    std::string Serialize() const;
};

class MovieSchedule {
public:
    MovieSchedule(const HighPrecision &targetZoom,
                  const std::string &step,
                  uint64_t wideIterations,
                  uint64_t targetIterations,
                  long double laDepth,
                  double slopeRatio);
    uint64_t FrameCount() const;
    long double DepthAt(uint64_t index) const;
    HighPrecision ZoomAt(uint64_t index) const;
    uint64_t IterationsAt(uint64_t index) const;
    static long double Log2(const HighPrecision &value);

private:
    // Zoom scale needs relative precision, not millions of fractional digits.
    // Keep the center at full precision separately; 256 scale bits leave ample
    // margin beyond the renderer's per-pixel precision requirements.
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

struct MovieChunk {
    uint64_t First;
    uint64_t Count;
    uint64_t PngBytes;
    uint64_t VideoBytes;
    std::string RgbHash;
};

// All tracked paths are relative to this job. The OS lock remains owned until
// all render/save/encoding workers have stopped, including exceptional exits.
class MovieJob {
public:
    MovieJob(const std::filesystem::path &directory, const std::filesystem::path &configuration);
    const MovieConfig &Config() const;
    const std::filesystem::path &Directory() const;
    void InitializeProgress(uint64_t frameCount, bool gpuBypassed, long double laDepth);
    uint64_t Remaining() const;
    void CompleteFrame(uint64_t index);
    void Rewind(uint64_t remaining);
    void SkipArchivedChunk(const MovieChunk &chunk);
    bool GpuBypassed() const;
    long double LaDepth() const;
    std::filesystem::path FramePath(uint64_t index) const;
    std::filesystem::path VideoPath(uint64_t first) const;
    std::optional<MovieChunk> FindChunk(uint64_t first) const;
    void CommitChunk(const MovieChunk &chunk);
    void DeleteChunkPngs(const MovieChunk &chunk);
    void WriteConcatManifest(uint64_t frameCount);
    std::filesystem::path Encoder(const std::filesystem::path &overridePath);

private:
    void SaveProgress();
    std::filesystem::path m_Directory;
    Environment::MovieJobLock m_Lock;
    MovieConfig m_Config;
    std::string m_ConfigHash;
    uint64_t m_FrameCount = 0;
    uint64_t m_Remaining = 0;
    uint64_t m_Sequence = 0;
    bool m_GpuBypassed = false;
    long double m_LaDepth = 0;
};

// Validate the complete PNG and dimensions before adopting a frame or encoding
// it. RGB16 conversion is nearest rounding without gamma changes or dithering.
std::vector<unsigned char> ReadMovieRgb8(const std::filesystem::path &path,
                                         unsigned width,
                                         unsigned height);
void WriteMovieText(const std::filesystem::path &path, const std::string &text);
std::string ReadMovieText(const std::filesystem::path &path);
} // namespace FractalSharkCli
