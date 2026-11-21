#pragma once

#include "MovieJob.h"

#include <filesystem>
#include <functional>
#include <vector>

namespace FractalSharkCli {

// A distinct recoverable outcome lets the movie boundary report an orderly
// pause after all worker/resource owners have unwound normally.
struct MovieInterrupted : FractalSharkSeriousException {
    MovieInterrupted() : FractalSharkSeriousException("Movie encoding interrupted; PNGs retained") {}
};

struct MovieTileGrid {
    unsigned Columns;
    unsigned Rows;
};

// Runtime policy only: these values never participate in job identity or
// checkpoints. Each slot reserves decode capacity across encode AND verify.
struct MovieExecutionSettings {
    unsigned Slot;
    unsigned DecoderThreads;
    unsigned FilterThreads;
    unsigned EncoderThreads;
    MovieTileGrid Tiles;
};

MovieTileGrid PlanMovieTiles(unsigned width, unsigned height, unsigned encoderThreads);
std::vector<MovieExecutionSettings> PlanMovieExecution(unsigned width,
                                                       unsigned height,
                                                       unsigned logicalProcessors);

// Encoder instances never own renderer state. FFmpeg owns PNG decoding,
// conversion, and pixel hashing; only small digests/metadata cross the pipes.
class MovieVideo {
public:
    MovieVideo(const std::filesystem::path &ffmpeg,
               unsigned width,
               unsigned height,
               unsigned fps,
               const MovieExecutionSettings &execution,
               const std::function<bool()> &stop);
    void Preflight(const std::filesystem::path &directory) const;
    MovieChunk Archive(const MovieJob &job, uint64_t first, uint64_t count) const;
    void Verify(const std::filesystem::path &path,
                uint64_t count,
                const std::string &expectedHash) const;

private:
    std::string Encode(const std::filesystem::path &path,
                       const std::string &pattern,
                       uint64_t first,
                       uint64_t count) const;
    void CheckStop() const;
    std::filesystem::path m_Ffmpeg;
    unsigned m_Width;
    unsigned m_Height;
    unsigned m_Fps;
    MovieExecutionSettings m_Execution;
    std::function<bool()> m_Stop;
};
} // namespace FractalSharkCli
