#pragma once

#include "MovieJob.h"

#include <filesystem>
#include <functional>

namespace FractalSharkCli {

// A distinct recoverable outcome lets the movie boundary report an orderly
// pause after all worker/resource owners have unwound normally.
struct MovieInterrupted : FractalSharkSeriousException {
    MovieInterrupted() : FractalSharkSeriousException("Movie encoding interrupted; PNGs retained") {}
};

// Encoder instances never own renderer state. The frame reader is invoked for
// one PNG at a time by the encoding worker, keeping staging memory bounded.
class MovieVideo {
public:
    MovieVideo(const std::filesystem::path &ffmpeg,
               unsigned width,
               unsigned height,
               unsigned fps,
               const std::function<bool()> &stop);
    void Preflight(const std::filesystem::path &directory) const;
    MovieChunk Archive(const MovieJob &job, uint64_t first, uint64_t count) const;
    void Verify(const std::filesystem::path &path,
                uint64_t count,
                const std::string &expectedHash) const;

private:
    std::string Encode(const std::filesystem::path &path,
                       uint64_t count,
                       const std::function<std::vector<unsigned char>(uint64_t)> &frame) const;
    void CheckStop() const;
    std::filesystem::path m_Ffmpeg;
    unsigned m_Width;
    unsigned m_Height;
    unsigned m_Fps;
    std::function<bool()> m_Stop;
};
} // namespace FractalSharkCli
