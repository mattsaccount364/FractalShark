#pragma once

#include "MovieJob.h"

#include <filesystem>
#include <functional>
#include <vector>

namespace FractalSharkCli {

// A distinct recoverable outcome lets the movie boundary report an orderly
// pause after all worker/resource owners have unwound normally.
struct MovieInterrupted : FractalSharkSeriousException {
    MovieInterrupted() : FractalSharkSeriousException("Movie interrupted; saved work retained") {}
};

// Runtime policy only: these values never participate in job identity or
// checkpoints. Each slot reserves PNG decode capacity during encoding.
struct MovieExecutionSettings {
    unsigned Slot;
    unsigned DecoderThreads;
    unsigned FilterThreads;
};

std::vector<MovieExecutionSettings> PlanMovieExecution(unsigned logicalProcessors);

// The job lock covers the whole append transaction and retirement of sources.
void AssembleMovie(MovieJob &job,
                   const std::filesystem::path &ffmpeg,
                   const std::function<bool()> &stop);

// Encoder instances never own renderer state. FFmpeg owns PNG decoding,
// conversion, and encoding; only small progress/diagnostic messages cross pipes.
class MovieVideo {
public:
    MovieVideo(const std::filesystem::path &ffmpeg,
               unsigned width,
               unsigned height,
               const MovieConfig &config,
               const MovieExecutionSettings &execution,
               const std::function<bool()> &stop);
    void Preflight(const std::filesystem::path &directory) const;
    MovieChunk Archive(const MovieJob &job, uint64_t first, uint64_t count) const;

private:
    void Encode(const std::filesystem::path &path,
                const std::string &pattern,
                uint64_t first,
                uint64_t count) const;
    void CheckStop() const;
    std::filesystem::path m_Ffmpeg;
    unsigned m_Width;
    unsigned m_Height;
    unsigned m_Fps;
    uint64_t m_GopFrames;
    bool m_WideToDeep;
    std::string m_PixelFormat;
    std::vector<std::string> m_EncoderArguments;
    MovieExecutionSettings m_Execution;
    std::function<bool()> m_Stop;
};
} // namespace FractalSharkCli
