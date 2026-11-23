#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

struct Color16;

namespace FractalShark {

enum class MovieCodec { Hevc, HevcRgbLossless, H264, PngOnly };

struct MovieVideoSettings {
    MovieCodec Codec = MovieCodec::Hevc;
    unsigned Preset = 5;
    unsigned Quality = 16;
    unsigned Fps = 60;
    uint32_t GopFrames = 1000;
};

// The movie queue owns each slot from render submission through consumption
// of its compressed access unit. Surface writes may occur on render threads.
class NvencMovieEncoder {
public:
    static constexpr size_t SurfaceCount = 4;

    NvencMovieEncoder(unsigned width, unsigned height, const MovieVideoSettings &settings);
    ~NvencMovieEncoder();
    NvencMovieEncoder(const NvencMovieEncoder &) = delete;
    NvencMovieEncoder &operator=(const NvencMovieEncoder &) = delete;

    uint32_t WriteGpu(size_t slot, const Color16 *colors, void *stream);
    uint32_t WriteCpu(size_t slot, std::span<const Color16> colors);
    std::vector<unsigned char> Encode(size_t slot, uint64_t frame, bool keyframe);

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace FractalShark
