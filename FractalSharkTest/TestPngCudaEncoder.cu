#include "stdafx.h"

#include <cuda_runtime.h>

#include "FractalPalette.h"
#include "FractalSharkGpuLib/PngCompressionKernels.cuh"
#include "FractalSharkGpuLib/PngEncoder.cuh"
#include "GPU_Render.h"
#include "TestFramework.h"
#include "WPngImage/WPngImage.hh"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <random>

namespace {

class PngTestStream {
public:
    PngTestStream()
    {
        ASSERT_EQ(cudaStreamCreateWithFlags(&m_Stream, cudaStreamNonBlocking), cudaSuccess);
    }
    ~PngTestStream() { cudaStreamDestroy(m_Stream); }
    cudaStream_t
    Get() const
    {
        return m_Stream;
    }

private:
    cudaStream_t m_Stream{};
};

template <class T> struct PngAllocationDeleter {
    void
    operator()(T *allocation) const
    {
        cudaFree(allocation);
    }
};

template <class T> class PngDeviceBuffer {
public:
    explicit PngDeviceBuffer(size_t elements)
    {
        T *allocation = nullptr;
        ASSERT_EQ(cudaMalloc(&allocation, elements * sizeof(T)), cudaSuccess);
        m_Data.reset(allocation);
    }
    ~PngDeviceBuffer() = default;
    PngDeviceBuffer(const PngDeviceBuffer &) = delete;
    PngDeviceBuffer &operator=(const PngDeviceBuffer &) = delete;
    T *
    Get() const
    {
        return m_Data.get();
    }

private:
    std::unique_ptr<T, PngAllocationDeleter<T>> m_Data;
};

enum class Pattern { Black, Repeated, Gradient, Transparent, Noise };

std::vector<Color16>
MakePixels(uint32_t width, uint32_t height, size_t pitch, Pattern pattern)
{
    std::vector<Color16> pixels(pitch * height, Color16{0xdeaf, 0xbeef, 0x1234, 0});
    std::mt19937 random{23917};
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            Color16 pixel{0, 0, 0, 65535};
            if (pattern == Pattern::Repeated) {
                pixel = (x + y) % 2 == 0 ? Color16{0x1111, 0x7777, 0xeeee, 65535}
                                         : Color16{0x3333, 0xaaaa, 0x5555, 65535};
            } else if (pattern == Pattern::Gradient || pattern == Pattern::Transparent) {
                pixel = {
                    static_cast<uint16_t>(0x0102 + x * 257 + y * 13),
                    static_cast<uint16_t>(0x2345 + x * 19 + y * 511),
                    static_cast<uint16_t>(0xabcd - x * 31 - y * 127),
                    static_cast<uint16_t>(pattern == Pattern::Transparent ? x * 4093 + y * 17 : 65535)};
            } else if (pattern == Pattern::Noise) {
                pixel = {static_cast<uint16_t>(random()),
                         static_cast<uint16_t>(random()),
                         static_cast<uint16_t>(random()),
                         65535};
            }
            pixels[y * pitch + x] = pixel;
        }
    }
    return pixels;
}

void
AppendChannel(std::vector<unsigned char> &bytes, uint16_t channel)
{
    bytes.push_back(static_cast<unsigned char>(channel >> 8));
    bytes.push_back(static_cast<unsigned char>(channel));
}

std::vector<unsigned char>
ExpectedPixels(const std::vector<Color16> &pixels, uint32_t width, uint32_t height, size_t pitch)
{
    std::vector<unsigned char> expected;
    expected.reserve(static_cast<size_t>(width) * height * 8);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto pixel = pixels[y * pitch + x];
            AppendChannel(expected, pixel.r);
            AppendChannel(expected, pixel.g);
            AppendChannel(expected, pixel.b);
            AppendChannel(expected, pixel.a);
        }
    }
    return expected;
}

WPngImage
CpuImage(const std::vector<Color16> &pixels, uint32_t width, uint32_t height, size_t pitch)
{
    WPngImage image{static_cast<int>(width), static_cast<int>(height), WPngImage::Pixel16{0, 0, 0}};
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const auto pixel = pixels[y * pitch + x];
            image.set(x, y, WPngImage::Pixel16{pixel.r, pixel.g, pixel.b, pixel.a});
        }
    }
    return image;
}

std::vector<unsigned char>
CpuEncode(const WPngImage &image)
{
    std::vector<unsigned char> encoded;
    const WPngImage::PngEncodingOptions options{false, false, 32};
    ASSERT_EQ(image.SaveImageToRAM(encoded, WPngImage::kPngFileFormat_RGBA16, options),
              WPngImage::kIOStatus_Ok);
    return encoded;
}

uint32_t
ReadBigEndian(const unsigned char *bytes)
{
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | bytes[3];
}

std::vector<size_t>
CheckStructure(const std::vector<unsigned char> &png, unsigned colorType)
{
    ASSERT_TRUE(png.size() >= 57);
    ASSERT_EQ(ReadBigEndian(png.data()), 0x89504e47u);
    ASSERT_EQ(ReadBigEndian(png.data() + 4), 0x0d0a1a0au);
    ASSERT_EQ(png[24], 16);
    ASSERT_EQ(png[25], colorType);
    ASSERT_EQ(png[28], 0);
    std::vector<size_t> idats;
    bool ended = false;
    size_t offset = 8;
    while (offset < png.size()) {
        ASSERT_TRUE(png.size() - offset >= 12);
        const auto length = ReadBigEndian(png.data() + offset);
        ASSERT_TRUE(length <= png.size() - offset - 12);
        ASSERT_EQ(lodepng_chunk_check_crc(png.data() + offset), 0u);
        if (offset == 8) {
            ASSERT_TRUE(lodepng_chunk_type_equals(png.data() + offset, "IHDR"));
            ASSERT_EQ(length, 13u);
        } else if (lodepng_chunk_type_equals(png.data() + offset, "IDAT")) {
            ASSERT_FALSE(ended);
            ASSERT_TRUE(length > 0 && length <= Png::IdatPayloadBytes);
            idats.push_back(offset);
        } else {
            ASSERT_TRUE(lodepng_chunk_type_equals(png.data() + offset, "IEND"));
            ASSERT_EQ(length, 0u);
            ASSERT_TRUE(!idats.empty());
            ended = true;
        }
        offset += length + 12;
        if (ended) {
            ASSERT_EQ(offset, png.size());
        }
    }
    ASSERT_TRUE(ended);
    return idats;
}

void
CompareDecoded(const std::vector<unsigned char> &png,
               const std::vector<unsigned char> &expected,
               uint32_t expectedWidth,
               uint32_t expectedHeight)
{
    std::vector<unsigned char> decoded;
    unsigned width = 0;
    unsigned height = 0;
    ASSERT_EQ(lodepng::decode(decoded, width, height, png, LCT_RGBA, 16), 0u);
    ASSERT_EQ(width, expectedWidth);
    ASSERT_EQ(height, expectedHeight);
    ASSERT_TRUE(decoded == expected);
}

std::vector<unsigned char> DecodeZlib(const std::vector<unsigned char> &bytes, unsigned &error);

std::vector<unsigned char>
FilteredImageData(const std::vector<unsigned char> &png)
{
    std::vector<unsigned char> zlib;
    for (size_t offset = 8; offset < png.size();) {
        ASSERT_TRUE(png.size() - offset >= 12);
        const size_t length = ReadBigEndian(png.data() + offset);
        ASSERT_TRUE(length <= png.size() - offset - 12);
        if (lodepng_chunk_type_equals(png.data() + offset, "IDAT")) {
            zlib.insert(zlib.end(), png.begin() + offset + 8, png.begin() + offset + 8 + length);
        }
        offset += length + 12;
    }
    unsigned error = 0;
    auto filtered = DecodeZlib(zlib, error);
    ASSERT_EQ(error, 0u);
    return filtered;
}

std::vector<unsigned char>
CheckImage(Png::GpuPngEncoder &encoder, uint32_t width, uint32_t height, size_t padding, Pattern pattern)
{
    const size_t pitch = static_cast<size_t>(width) + padding;
    const auto pixels = MakePixels(width, height, pitch, pattern);
    const auto expected = ExpectedPixels(pixels, width, height, pitch);
    const auto cpu = CpuEncode(CpuImage(pixels, width, height, pitch));
    PngTestStream stream;
    PngDeviceBuffer<Color16> input{pixels.size()};
    ASSERT_EQ(cudaMemcpyAsync(input.Get(),
                              pixels.data(),
                              pixels.size() * sizeof(Color16),
                              cudaMemcpyHostToDevice,
                              stream.Get()),
              cudaSuccess);
    std::vector<unsigned char> gpu;
    ASSERT_EQ(encoder.Encode(input.Get(), width, height, pitch * sizeof(Color16), stream.Get(), gpu),
              cudaSuccess);
    CheckStructure(gpu, pattern == Pattern::Transparent ? 6 : 2);
    CompareDecoded(gpu, expected, width, height);
    CompareDecoded(cpu, expected, width, height);
    // The CPU encoder independently selects filters. Compare decompressed scanlines
    // before unfiltering to verify scoring, tie-breaking, and serialization as well.
    ASSERT_TRUE(FilteredImageData(gpu) == FilteredImageData(cpu));
    const auto filtered = FilteredImageData(gpu);
    PngDeviceBuffer<unsigned char> compressionInput{filtered.size()};
    ASSERT_EQ(cudaMemcpyAsync(compressionInput.Get(),
                              filtered.data(),
                              filtered.size(),
                              cudaMemcpyHostToDevice,
                              stream.Get()),
              cudaSuccess);
    std::vector<unsigned char> baseline;
    std::vector<unsigned char> improved;
    ASSERT_EQ(Png::EncodeZlibFixed(compressionInput.Get(), filtered.size(), stream.Get(), baseline),
              cudaSuccess);
    ASSERT_EQ(Png::EncodeZlib(compressionInput.Get(), filtered.size(), stream.Get(), improved),
              cudaSuccess);
    ASSERT_TRUE(improved.size() <= baseline.size());
    std::vector<unsigned char> repeated;
    ASSERT_EQ(
        encoder.Encode(input.Get(), width, height, pitch * sizeof(Color16), stream.Get(), repeated),
        cudaSuccess);
    ASSERT_TRUE(gpu == repeated);
    return gpu;
}

std::vector<unsigned char>
DecodeZlib(const std::vector<unsigned char> &bytes, unsigned &error)
{
    unsigned char *output = nullptr;
    size_t size = 0;
    error = lodepng_zlib_decompress(
        &output, &size, bytes.data(), bytes.size(), &lodepng_default_decompress_settings);
    std::unique_ptr<unsigned char, decltype(&std::free)> allocation{output, &std::free};
    if (error != 0) {
        return {};
    }
    return {output, output + size};
}

class DeflateTree {
public:
    explicit DeflateTree(const std::vector<unsigned> &lengths)
        : m_Counts(16, 0), m_FirstCodes(16, 0), m_FirstSymbols(16, 0)
    {
        for (const unsigned length : lengths) {
            ASSERT_TRUE(length <= 15);
            if (length != 0) {
                ++m_Counts[length];
            }
        }
        unsigned code = 0;
        unsigned first = 0;
        for (unsigned length = 1; length <= 15; ++length) {
            code = (code + m_Counts[length - 1]) * 2;
            ASSERT_TRUE(code + m_Counts[length] <= (1u << length));
            m_FirstCodes[length] = code;
            m_FirstSymbols[length] = first;
            for (unsigned symbol = 0; symbol < lengths.size(); ++symbol) {
                if (lengths[symbol] == length) {
                    m_Symbols.push_back(symbol);
                }
            }
            first += m_Counts[length];
        }
    }
    std::vector<unsigned> m_Counts;
    std::vector<unsigned> m_FirstCodes;
    std::vector<unsigned> m_FirstSymbols;
    std::vector<unsigned> m_Symbols;
};

class DeflateReader {
public:
    explicit DeflateReader(const std::vector<unsigned char> &bytes) : m_Bytes{bytes} {}
    unsigned
    Read(unsigned count)
    {
        unsigned value = 0;
        for (unsigned bit = 0; bit < count; ++bit) {
            ASSERT_TRUE(m_Bit / 8 < m_Bytes.size() - 4);
            value |= ((m_Bytes[m_Bit / 8] >> (m_Bit % 8)) & 1u) << bit;
            ++m_Bit;
        }
        return value;
    }
    unsigned
    Symbol(const DeflateTree &tree)
    {
        unsigned code = 0;
        for (unsigned length = 1; length <= 15; ++length) {
            code = (code << 1) | Read(1);
            if (code >= tree.m_FirstCodes[length] &&
                code - tree.m_FirstCodes[length] < tree.m_Counts[length]) {
                return tree.m_Symbols[tree.m_FirstSymbols[length] + code - tree.m_FirstCodes[length]];
            }
        }
        TestFramework::Fail(__FILE__, __LINE__, "invalid Huffman symbol");
    }
    void
    Align()
    {
        m_Bit = (m_Bit + 7) & ~size_t{7};
    }
    size_t
    Position() const
    {
        return m_Bit;
    }

private:
    const std::vector<unsigned char> &m_Bytes;
    size_t m_Bit{16};
};

struct DeflateInfo {
    size_t m_StoredBlocks{};
    size_t m_FixedBlocks{};
    size_t m_DynamicBlocks{};
    std::vector<unsigned> m_Lengths;
    std::vector<unsigned> m_Distances;
};

DeflateInfo
InspectDeflate(const std::vector<unsigned char> &bytes, size_t expectedBytes)
{
    ASSERT_TRUE(bytes.size() >= 11);
    ASSERT_EQ((static_cast<unsigned>(bytes[0]) * 256 + bytes[1]) % 31, 0u);
    ASSERT_EQ(bytes[0] & 15, 8);
    ASSERT_EQ(bytes[1] & 32, 0);
    DeflateReader reader{bytes};
    DeflateInfo info;
    size_t decoded = 0;
    bool final = false;
    while (!final) {
        final = reader.Read(1) != 0;
        const unsigned type = reader.Read(2);
        if (type == 0) {
            ++info.m_StoredBlocks;
            reader.Align();
            const unsigned length = reader.Read(16);
            ASSERT_EQ(reader.Read(16), length ^ 65535u);
            if (final) {
                ASSERT_EQ(length, 0u);
            }
            for (unsigned byte = 0; byte < length; ++byte) {
                reader.Read(8);
            }
            decoded += length;
        } else {
            ASSERT_TRUE(type == 1 || type == 2);
            ASSERT_FALSE(final);
            std::vector<unsigned> literalLengths;
            std::vector<unsigned> distanceLengths;
            if (type == 1) {
                ++info.m_FixedBlocks;
                for (unsigned symbol = 0; symbol < 288; ++symbol) {
                    literalLengths.push_back(symbol <= 143   ? 8
                                             : symbol <= 255 ? 9
                                             : symbol <= 279 ? 7
                                                             : 8);
                }
                distanceLengths.assign(32, 5);
            } else {
                ++info.m_DynamicBlocks;
                const unsigned literals = reader.Read(5) + 257;
                const unsigned distances = reader.Read(5) + 1;
                const unsigned headers = reader.Read(4) + 4;
                ASSERT_TRUE(literals <= 286 && distances <= 30);
                const std::vector<unsigned> order{
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
                std::vector<unsigned> headerLengths(19, 0);
                for (unsigned index = 0; index < headers; ++index) {
                    headerLengths[order[index]] = reader.Read(3);
                }
                const DeflateTree headerTree{headerLengths};
                std::vector<unsigned> combined;
                while (combined.size() < literals + distances) {
                    const unsigned symbol = reader.Symbol(headerTree);
                    if (symbol < 16) {
                        combined.push_back(symbol);
                    } else {
                        ASSERT_TRUE(symbol <= 18);
                        ASSERT_TRUE(symbol != 16 || !combined.empty());
                        const unsigned value = symbol == 16 ? combined.back() : 0;
                        const unsigned count = symbol == 16   ? reader.Read(2) + 3
                                               : symbol == 17 ? reader.Read(3) + 3
                                                              : reader.Read(7) + 11;
                        ASSERT_TRUE(combined.size() + count <= literals + distances);
                        combined.insert(combined.end(), count, value);
                    }
                }
                literalLengths.assign(combined.begin(), combined.begin() + literals);
                distanceLengths.assign(combined.begin() + literals, combined.end());
                ASSERT_TRUE(literalLengths[256] != 0);
            }
            const DeflateTree literalTree{literalLengths};
            const DeflateTree distanceTree{distanceLengths};
            for (;;) {
                const unsigned symbol = reader.Symbol(literalTree);
                if (symbol == 256) {
                    break;
                }
                if (symbol < 256) {
                    ++decoded;
                    continue;
                }
                ASSERT_TRUE(symbol <= 285);
                unsigned length = 258;
                if (symbol != 285) {
                    unsigned base = 3;
                    for (unsigned current = 257; current < symbol; ++current) {
                        base += 1u << (current <= 264 ? 0 : (current - 261) / 4);
                    }
                    length = base + reader.Read(symbol <= 264 ? 0 : (symbol - 261) / 4);
                }
                const unsigned distanceCode = reader.Symbol(distanceTree);
                ASSERT_TRUE(distanceCode < 30);
                unsigned base = 1;
                for (unsigned code = 0; code < distanceCode; ++code) {
                    base += 1u << (code < 4 ? 0 : code / 2 - 1);
                }
                const unsigned distance =
                    base + reader.Read(distanceCode < 4 ? 0 : distanceCode / 2 - 1);
                ASSERT_TRUE(distance > 0 && distance <= 32768 && distance <= decoded);
                ASSERT_TRUE(distance <= decoded % Png::DeflateRegionBytes);
                ASSERT_TRUE(length <= Png::DeflateRegionBytes - decoded % Png::DeflateRegionBytes);
                info.m_Lengths.push_back(length);
                info.m_Distances.push_back(distance);
                decoded += length;
            }
        }
        ASSERT_TRUE(decoded <= expectedBytes);
    }
    reader.Align();
    ASSERT_EQ(reader.Position() / 8, bytes.size() - 4);
    ASSERT_EQ(decoded, expectedBytes);
    return info;
}

std::vector<unsigned char>
CheckZlib(const std::vector<unsigned char> &input)
{
    PngTestStream stream;
    PngDeviceBuffer<unsigned char> device{input.size()};
    ASSERT_EQ(
        cudaMemcpyAsync(device.Get(), input.data(), input.size(), cudaMemcpyHostToDevice, stream.Get()),
        cudaSuccess);
    std::vector<unsigned char> bytes;
    ASSERT_EQ(Png::EncodeZlib(device.Get(), input.size(), stream.Get(), bytes), cudaSuccess);
    std::vector<unsigned char> baseline;
    ASSERT_EQ(Png::EncodeZlibFixed(device.Get(), input.size(), stream.Get(), baseline), cudaSuccess);
    ASSERT_TRUE(bytes.size() <= baseline.size());
    unsigned error = 0;
    ASSERT_TRUE(DecodeZlib(baseline, error) == input);
    ASSERT_EQ(error, 0u);
    ASSERT_EQ(InspectDeflate(baseline, input.size()).m_DynamicBlocks, size_t{0});
    ASSERT_TRUE(DecodeZlib(bytes, error) == input);
    ASSERT_EQ(error, 0u);
    uint32_t sum = 1;
    uint32_t weighted = 0;
    for (const unsigned char value : input) {
        sum = (sum + value) % 65521;
        weighted = (weighted + sum) % 65521;
    }
    ASSERT_EQ(ReadBigEndian(bytes.data() + bytes.size() - 4), (weighted << 16) | sum);
    const size_t regions = (input.size() - 1) / Png::DeflateRegionBytes + 1;
    ASSERT_TRUE(bytes.size() <= input.size() + 5 * regions + 11);
    InspectDeflate(bytes, input.size());
    return bytes;
}

void
CheckShapes()
{
    Png::GpuPngEncoder encoder;
    CheckImage(encoder, 1, 1, 0, Pattern::Gradient);
    CheckImage(encoder, 1, 37, 3, Pattern::Transparent);
    CheckImage(encoder, 37, 1, 2, Pattern::Gradient);
    for (const auto pattern :
         {Pattern::Black, Pattern::Repeated, Pattern::Gradient, Pattern::Transparent}) {
        CheckImage(encoder, 17, 9, 5, pattern);
        CheckImage(encoder, 257, 129, 0, pattern);
    }
}

void
CheckNoiseAndChunks()
{
    Png::GpuPngEncoder encoder;
    const auto noise = CheckImage(encoder, 257, 129, 7, Pattern::Noise);
    const auto chunks = CheckStructure(noise, 2);
    ASSERT_TRUE(chunks.size() > 1);
    std::vector<unsigned char> zlib;
    for (const size_t offset : chunks) {
        const size_t size = ReadBigEndian(noise.data() + offset);
        zlib.insert(zlib.end(), noise.begin() + offset + 8, noise.begin() + offset + 8 + size);
    }
    const auto info = InspectDeflate(zlib, (257 * 6 + 1) * 129);
    ASSERT_TRUE(info.m_StoredBlocks > 1);
    const auto black = CheckImage(encoder, 257, 129, 0, Pattern::Black);
    ASSERT_TRUE(black.size() < static_cast<size_t>(257) * 129 * 6 / 20);
    const auto repeated = CheckImage(encoder, 257, 129, 0, Pattern::Repeated);
    ASSERT_TRUE(repeated.size() < static_cast<size_t>(257) * 129 * 6 / 10);
}

void
CheckFilters()
{
    PngTestStream stream;
    for (const unsigned channels : {3u, 4u}) {
        constexpr unsigned width = 17;
        constexpr unsigned height = 3;
        const size_t rowBytes = width * channels * 2;
        std::vector<unsigned char> raw(rowBytes * height);
        std::mt19937 random{8128};
        for (auto &value : raw) {
            value = static_cast<unsigned char>(random());
        }
        PngDeviceBuffer<unsigned char> source{raw.size()};
        PngDeviceBuffer<unsigned char> selected{height};
        PngDeviceBuffer<unsigned char> output{raw.size() + height};
        ASSERT_EQ(
            cudaMemcpyAsync(source.Get(), raw.data(), raw.size(), cudaMemcpyHostToDevice, stream.Get()),
            cudaSuccess);
        for (unsigned filter = 0; filter <= 4; ++filter) {
            std::vector<unsigned char> filters(height, static_cast<unsigned char>(filter));
            ASSERT_EQ(cudaMemcpyAsync(selected.Get(),
                                      filters.data(),
                                      filters.size(),
                                      cudaMemcpyHostToDevice,
                                      stream.Get()),
                      cudaSuccess);
            ASSERT_EQ(
                Png::LaunchFilterScanlines(
                    source.Get(), width, height, channels, selected.Get(), output.Get(), stream.Get()),
                cudaSuccess);
            std::vector<unsigned char> filtered(raw.size() + height);
            ASSERT_EQ(cudaMemcpyAsync(filtered.data(),
                                      output.Get(),
                                      filtered.size(),
                                      cudaMemcpyDeviceToHost,
                                      stream.Get()),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream.Get()), cudaSuccess);
            // Independently invert each filter and compare original serialized bytes.
            std::vector<unsigned char> restored(raw.size());
            for (unsigned y = 0; y < height; ++y) {
                ASSERT_EQ(filtered[y * (rowBytes + 1)], filter);
                for (size_t x = 0; x < rowBytes; ++x) {
                    const size_t index = y * rowBytes + x;
                    const int left = x >= channels * 2 ? restored[index - channels * 2] : 0;
                    const int above = y != 0 ? restored[index - rowBytes] : 0;
                    const int corner =
                        y != 0 && x >= channels * 2 ? restored[index - rowBytes - channels * 2] : 0;
                    int predictor = 0;
                    if (filter == 1) {
                        predictor = left;
                    }
                    if (filter == 2) {
                        predictor = above;
                    }
                    if (filter == 3) {
                        predictor = (left + above) / 2;
                    }
                    if (filter == 4) {
                        const int prediction = left + above - corner;
                        const int leftDistance = std::abs(prediction - left);
                        const int aboveDistance = std::abs(prediction - above);
                        const int cornerDistance = std::abs(prediction - corner);
                        predictor = leftDistance <= aboveDistance && leftDistance <= cornerDistance
                                        ? left
                                    : aboveDistance <= cornerDistance ? above
                                                                      : corner;
                    }
                    restored[index] =
                        static_cast<unsigned char>(filtered[y * (rowBytes + 1) + x + 1] + predictor);
                }
            }
            ASSERT_TRUE(restored == raw);
        }
    }
}

void
CheckDeflateBoundaries()
{
    for (const size_t bytes : {size_t{1},
                               size_t{2},
                               size_t{3},
                               size_t{32767},
                               size_t{32768},
                               size_t{32769},
                               size_t{65536},
                               size_t{65537}}) {
        CheckZlib(std::vector<unsigned char>(bytes, 0));
    }
    for (const unsigned length :
         {3u, 4u, 10u, 11u, 18u, 19u, 34u, 35u, 66u, 67u, 130u, 131u, 257u, 258u}) {
        std::vector<unsigned char> input(length + 3 + 1024, 0);
        for (size_t i = 0; i < length + 3; ++i) {
            input[i] = static_cast<unsigned char>('A' + i % 3);
        }
        const auto bytes = CheckZlib(input);
        const auto info = InspectDeflate(bytes, input.size());
        ASSERT_TRUE(std::find(info.m_Lengths.begin(), info.m_Lengths.end(), length) !=
                    info.m_Lengths.end());
        ASSERT_TRUE(std::find(info.m_Distances.begin(), info.m_Distances.end(), 3u) !=
                    info.m_Distances.end());
    }
    for (const unsigned distance : {4u,
                                    5u,
                                    8u,
                                    9u,
                                    16u,
                                    17u,
                                    32u,
                                    33u,
                                    128u,
                                    129u,
                                    1024u,
                                    4096u,
                                    16384u,
                                    24577u,
                                    32765u,
                                    32768u}) {
        std::vector<unsigned char> input(distance + 3 + 1024, 0);
        input[0] = 0xa1;
        input[1] = 0xb2;
        input[2] = 0xc3;
        input[distance] = 0xa1;
        input[distance + 1] = 0xb2;
        input[distance + 2] = 0xc3;
        const auto bytes = CheckZlib(input);
        const auto info = InspectDeflate(bytes, input.size());
        if (distance < 32768) {
            ASSERT_TRUE(std::find(info.m_Distances.begin(), info.m_Distances.end(), distance) !=
                        info.m_Distances.end());
        }
    }
    std::mt19937 random{1977};
    std::vector<unsigned char> noise(100003);
    for (auto &value : noise) {
        value = static_cast<unsigned char>(random());
    }
    const auto bytes = CheckZlib(noise);
    ASSERT_TRUE(InspectDeflate(bytes, noise.size()).m_StoredBlocks > 1);
    std::vector<unsigned char> mixed(Png::DeflateRegionBytes, 0);
    mixed.insert(mixed.end(), noise.begin(), noise.end());
    const auto mixedBytes = CheckZlib(mixed);
    const auto mixedInfo = InspectDeflate(mixedBytes, mixed.size());
    ASSERT_TRUE(mixedInfo.m_DynamicBlocks > 0);
    ASSERT_TRUE(mixedInfo.m_StoredBlocks > 1);
    for (const unsigned distance : {1u, 2u}) {
        std::vector<unsigned char> overlapping(1027);
        for (size_t i = 0; i < overlapping.size(); ++i) {
            overlapping[i] = static_cast<unsigned char>('A' + i % distance);
        }
        const auto overlapBytes = CheckZlib(overlapping);
        const auto overlapInfo = InspectDeflate(overlapBytes, overlapping.size());
        ASSERT_TRUE(std::find(overlapInfo.m_Distances.begin(),
                              overlapInfo.m_Distances.end(),
                              distance) != overlapInfo.m_Distances.end());
    }
}

void
CheckCorruption()
{
    Png::GpuPngEncoder encoder;
    const auto png = CheckImage(encoder, 257, 129, 0, Pattern::Noise);
    const auto idats = CheckStructure(png, 2);
    std::vector<unsigned char> decoded;
    unsigned width = 0;
    unsigned height = 0;
    auto damaged = png;
    damaged[29] ^= 1;
    ASSERT_TRUE(lodepng::decode(decoded, width, height, damaged, LCT_RGBA, 16) != 0);
    damaged = png;
    const size_t last = idats.back();
    const size_t payload = ReadBigEndian(damaged.data() + last);
    damaged[last + 8 + payload - 1] ^= 1; // Adler-32, with a repaired PNG chunk CRC.
    lodepng_chunk_generate_crc(damaged.data() + last);
    ASSERT_TRUE(lodepng::decode(decoded, width, height, damaged, LCT_RGBA, 16) != 0);
    damaged = png;
    damaged.resize(damaged.size() / 2);
    ASSERT_TRUE(lodepng::decode(decoded, width, height, damaged, LCT_RGBA, 16) != 0);
    auto zlib = CheckZlib(std::vector<unsigned char>(1000, 123));
    zlib[2] |= 6; // Reserved DEFLATE block type.
    unsigned error = 0;
    DecodeZlib(zlib, error);
    ASSERT_TRUE(error != 0);
}

void
CheckHuffmanLengths()
{
    PngTestStream stream;
    std::mt19937 random{1951};
    for (const auto specification : std::vector<std::pair<unsigned, unsigned>>{
             {286, 15}, {286, 9}, {30, 15}, {19, 7}, {19, 5}, {2, 1}}) {
        const unsigned count = specification.first;
        const unsigned maximum = specification.second;
        for (unsigned fixture = 0; fixture < 24; ++fixture) {
            std::vector<unsigned> frequencies(count, 0);
            if (fixture == 1) {
                frequencies[count - 1] = 32768;
            } else if (fixture == 2) {
                frequencies[0] = 1000;
                frequencies[count - 1] = 1;
            } else if (fixture == 3) {
                std::fill(frequencies.begin(), frequencies.end(), 1);
            } else if (fixture == 4) {
                unsigned previous = 1;
                unsigned current = 1;
                for (unsigned symbol = 0; symbol < count && symbol < 20; ++symbol) {
                    frequencies[symbol] = previous;
                    const unsigned next = previous + current;
                    previous = current;
                    current = next;
                }
            } else if (fixture > 4) {
                for (auto &frequency : frequencies) {
                    frequency = random() % 113;
                }
            }
            PngDeviceBuffer<unsigned> input{count};
            ASSERT_EQ(cudaMemcpyAsync(input.Get(),
                                      frequencies.data(),
                                      count * sizeof(unsigned),
                                      cudaMemcpyHostToDevice,
                                      stream.Get()),
                      cudaSuccess);
            std::vector<unsigned> gpu;
            ASSERT_EQ(Png::BuildHuffmanCodeLengths(input.Get(), count, maximum, stream.Get(), gpu),
                      cudaSuccess);
            std::vector<unsigned> cpu(count);
            ASSERT_EQ(lodepng_huffman_code_lengths(cpu.data(), frequencies.data(), count, maximum), 0u);
            uint64_t gpuCost = 0;
            uint64_t cpuCost = 0;
            unsigned capacity = 0;
            for (unsigned symbol = 0; symbol < count; ++symbol) {
                ASSERT_TRUE(gpu[symbol] <= maximum);
                ASSERT_TRUE(frequencies[symbol] == 0 || gpu[symbol] != 0);
                if (gpu[symbol] != 0) {
                    capacity += 1u << (maximum - gpu[symbol]);
                }
                gpuCost += static_cast<uint64_t>(frequencies[symbol]) * gpu[symbol];
                cpuCost += static_cast<uint64_t>(frequencies[symbol]) * cpu[symbol];
            }
            ASSERT_EQ(capacity, 1u << maximum);
            ASSERT_EQ(gpuCost, cpuCost);
            ASSERT_TRUE(gpu == cpu); // Stable frequency/symbol ordering matches the CPU oracle.
            if (fixture == 4 && count > 19) {
                ASSERT_EQ(*std::max_element(gpu.begin(), gpu.end()), maximum);
            }
        }
    }
    std::vector<unsigned> output{1};
    ASSERT_EQ(Png::BuildHuffmanCodeLengths(nullptr, 286, 15, stream.Get(), output),
              cudaErrorInvalidValue);
    ASSERT_TRUE(output.empty());
}

void
CheckHeaderRuns()
{
    PngTestStream stream;
    bool seen16 = false;
    bool seen17 = false;
    bool seen18 = false;
    for (const unsigned value : {0u, 1u, 7u, 15u}) {
        for (const unsigned repeated : {1u, 2u, 3u, 4u, 6u, 7u, 10u, 11u, 138u, 139u, 255u}) {
            std::vector<unsigned> lengths{2};
            lengths.insert(lengths.end(), repeated, value);
            lengths.push_back(3);
            PngDeviceBuffer<unsigned> input{lengths.size()};
            ASSERT_EQ(cudaMemcpyAsync(input.Get(),
                                      lengths.data(),
                                      lengths.size() * sizeof(unsigned),
                                      cudaMemcpyHostToDevice,
                                      stream.Get()),
                      cudaSuccess);
            std::vector<Png::CodeLengthRun> runs;
            ASSERT_EQ(Png::EncodeCodeLengthRuns(
                          input.Get(), static_cast<unsigned>(lengths.size()), stream.Get(), runs),
                      cudaSuccess);
            std::vector<unsigned> restored;
            for (const auto run : runs) {
                if (run.m_Symbol < 16) {
                    ASSERT_EQ(run.m_ExtraBits, 0u);
                    restored.push_back(run.m_Symbol);
                } else if (run.m_Symbol == 16) {
                    seen16 = true;
                    ASSERT_FALSE(restored.empty());
                    ASSERT_EQ(run.m_ExtraBits, 2u);
                    ASSERT_TRUE(run.m_Extra <= 3);
                    restored.insert(restored.end(), run.m_Extra + 3, restored.back());
                } else {
                    ASSERT_TRUE(run.m_Symbol == 17 || run.m_Symbol == 18);
                    const bool shortRun = run.m_Symbol == 17;
                    seen17 |= shortRun;
                    seen18 |= !shortRun;
                    ASSERT_EQ(run.m_ExtraBits, shortRun ? 3u : 7u);
                    ASSERT_TRUE(run.m_Extra < (1u << run.m_ExtraBits));
                    restored.insert(restored.end(), run.m_Extra + (shortRun ? 3 : 11), 0);
                }
            }
            ASSERT_TRUE(restored == lengths);
        }
    }
    ASSERT_TRUE(seen16 && seen17 && seen18);
    PngDeviceBuffer<unsigned> invalid{1};
    const unsigned invalidLength = 16;
    ASSERT_EQ(
        cudaMemcpyAsync(
            invalid.Get(), &invalidLength, sizeof(invalidLength), cudaMemcpyHostToDevice, stream.Get()),
        cudaSuccess);
    std::vector<Png::CodeLengthRun> runs;
    ASSERT_EQ(Png::EncodeCodeLengthRuns(invalid.Get(), 1, stream.Get(), runs), cudaErrorInvalidValue);
    ASSERT_TRUE(runs.empty());
}

void
CheckBlockSelection()
{
    constexpr size_t regionBytes = Png::DeflateRegionBytes;
    std::mt19937 random{32195};
    std::vector<unsigned char> mixed(regionBytes * 2 + 20, 0);
    for (size_t index = regionBytes; index < regionBytes * 2; ++index) {
        mixed[index] = static_cast<unsigned char>(random());
    }
    const auto bytes = CheckZlib(mixed);
    const auto info = InspectDeflate(bytes, mixed.size());
    ASSERT_TRUE(info.m_DynamicBlocks > 0);
    ASSERT_TRUE(info.m_FixedBlocks > 0);
    ASSERT_TRUE(info.m_StoredBlocks > 0);
    for (unsigned fixture = 0; fixture < 20; ++fixture) {
        std::vector<unsigned char> input(200 + random() % 70000);
        for (auto &value : input) {
            value = static_cast<unsigned char>(random() % (fixture + 1));
        }
        CheckZlib(input);
    }
    // An Eulerian traversal generates a de Bruijn sequence over 16 symbols. Every
    // three-byte substring is unique, so the parser must emit only literals.
    std::vector<unsigned> nextEdge(256, 0);
    std::vector<unsigned> vertices{0};
    std::vector<unsigned char> literals;
    while (!vertices.empty()) {
        const unsigned vertex = vertices.back();
        if (nextEdge[vertex] < 16) {
            vertices.push_back((vertex % 16) * 16 + nextEdge[vertex]++);
        } else {
            literals.push_back(static_cast<unsigned char>(vertex % 16));
            vertices.pop_back();
        }
    }
    std::reverse(literals.begin(), literals.end());
    literals.resize(4096);
    const auto literalBytes = CheckZlib(literals);
    const auto literalInfo = InspectDeflate(literalBytes, literals.size());
    ASSERT_TRUE(literalInfo.m_Lengths.empty());
    ASSERT_EQ(literalInfo.m_DynamicBlocks, size_t{1});
}

void
CheckReuseAndConcurrency()
{
    Png::GpuPngEncoder encoder;
    CheckImage(encoder, 257, 129, 0, Pattern::Noise);
    const size_t capacity = encoder.GetWorkspaceBytes();
    CheckImage(encoder, 1, 1, 7, Pattern::Transparent);
    ASSERT_EQ(encoder.GetWorkspaceBytes(), capacity);
    CheckImage(encoder, 33, 17, 2, Pattern::Gradient);
    ASSERT_EQ(encoder.GetWorkspaceBytes(), capacity);
    int device = -1;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    auto first = std::async(std::launch::async, [device] {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        Png::GpuPngEncoder separate;
        CheckImage(separate, 129, 65, 3, Pattern::Transparent);
    });
    auto second = std::async(std::launch::async, [device] {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        Png::GpuPngEncoder separate;
        CheckImage(separate, 65, 129, 1, Pattern::Noise);
    });
    first.get();
    second.get();
}

void
CheckInvalidInputs()
{
    Png::GpuPngEncoder encoder;
    PngTestStream stream;
    PngDeviceBuffer<Color16> pixels{1};
    std::vector<unsigned char> output{1, 2, 3};
    ASSERT_EQ(encoder.Encode(nullptr, 1, 1, 8, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_TRUE(output.empty());
    ASSERT_EQ(encoder.Encode(pixels.Get(), 0, 1, 8, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_EQ(encoder.Encode(pixels.Get(), 1, 0, 8, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_EQ(encoder.Encode(pixels.Get(), 1, 1, 7, stream.Get(), output), cudaErrorInvalidValue);
    const auto *unaligned =
        reinterpret_cast<const Color16 *>(reinterpret_cast<const unsigned char *>(pixels.Get()) + 1);
    ASSERT_EQ(encoder.Encode(unaligned, 1, 1, 8, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_EQ(encoder.Encode(pixels.Get(), 1, 2, 9, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_EQ(
        encoder.Encode(pixels.Get(), 1, 2, std::numeric_limits<size_t>::max() - 1, stream.Get(), output),
        cudaErrorInvalidValue);
    ASSERT_EQ(encoder.Encode(
                  pixels.Get(), 1, 2, std::numeric_limits<size_t>::max() - 15, stream.Get(), output),
              cudaErrorInvalidValue);
    ASSERT_EQ(encoder.Encode(pixels.Get(), 0xffffffffu, 1, 8, stream.Get(), output),
              cudaErrorInvalidValue);
    ASSERT_EQ(Png::EncodeZlib(nullptr, 1, stream.Get(), output), cudaErrorInvalidValue);
    ASSERT_EQ(Png::EncodeZlib(reinterpret_cast<unsigned char *>(pixels.Get()), 0, stream.Get(), output),
              cudaErrorInvalidValue);
    ASSERT_EQ(encoder.GetWorkspaceBytes(), size_t{0});
    CheckImage(encoder, 1, 1, 0, Pattern::Gradient);
}

std::vector<Color16>
RenderBenchmarkFractal(uint32_t width, uint32_t height)
{
    FractalPalette palette;
    palette.InitializeAllPalettes();
    palette.SetDefaults();
    GPURenderer renderer;
    ASSERT_EQ(renderer.InitializeMemory<uint32_t>(width,
                                                  height,
                                                  1,
                                                  palette.GetCurrentPalInterleaved(),
                                                  palette.GetCurrentNumColors(),
                                                  palette.GetAuxDepth(),
                                                  palette.GetPaletteRotation(),
                                                  static_cast<uint64_t>(INT32_MAX - 1),
                                                  palette.GetPaletteGeneration(),
                                                  false),
              0u);
    ASSERT_EQ(
        (renderer.Render<uint32_t, float>(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32),
                                          -2.0f,
                                          -1.5f,
                                          3.0f / width,
                                          3.0f / height,
                                          1000,
                                          1)),
        0u);
    ASSERT_EQ(renderer.SyncComputeStream(), 0u);
    std::vector<Color16> colors(static_cast<size_t>(width) * height);
    ReductionResults reduction;
    ASSERT_EQ(renderer.RenderCurrent<uint32_t>(1000, nullptr, colors.data(), &reduction, false), 0u);
    ASSERT_EQ(renderer.SyncComputeStream(), 0u);
    return colors;
}

void
Benchmark4K()
{
    constexpr uint32_t width = 3840;
    constexpr uint32_t height = 2160;
    for (const std::string label : {"uniform", "gradient", "noise", "fractal"}) {
        const auto pattern = label == "uniform" ? Pattern::Black
                             : label == "noise" ? Pattern::Noise
                                                : Pattern::Gradient;
        const auto pixels = label == "fractal" ? RenderBenchmarkFractal(width, height)
                                               : MakePixels(width, height, width, pattern);
        const auto image = CpuImage(pixels, width, height, width);
        PngTestStream stream;
        PngDeviceBuffer<Color16> input{pixels.size()};
        ASSERT_EQ(cudaMemcpyAsync(input.Get(),
                                  pixels.data(),
                                  pixels.size() * sizeof(Color16),
                                  cudaMemcpyHostToDevice,
                                  stream.Get()),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream.Get()), cudaSuccess);
        Png::GpuPngEncoder encoder;
        std::vector<unsigned char> gpu;
        const auto coldStart = std::chrono::steady_clock::now();
        ASSERT_EQ(encoder.Encode(input.Get(), width, height, width * sizeof(Color16), stream.Get(), gpu),
                  cudaSuccess);
        const double coldMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - coldStart)
                .count();
        std::vector<double> gpuTimes;
        std::vector<double> cpuTimes;
        std::vector<unsigned char> cpu;
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            auto start = std::chrono::steady_clock::now();
            ASSERT_EQ(
                encoder.Encode(input.Get(), width, height, width * sizeof(Color16), stream.Get(), gpu),
                cudaSuccess);
            gpuTimes.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
            start = std::chrono::steady_clock::now();
            cpu = CpuEncode(image);
            cpuTimes.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
        }
        std::sort(gpuTimes.begin(), gpuTimes.end());
        std::sort(cpuTimes.begin(), cpuTimes.end());
        const auto expected = ExpectedPixels(pixels, width, height, width);
        CheckStructure(gpu, 2);
        CompareDecoded(gpu, expected, width, height);
        CompareDecoded(cpu, expected, width, height);
        // Compare sizes against the old GPU coding policy on identical filtered bytes.
        // This work is outside every measured encoding interval.
        const auto filtered = FilteredImageData(gpu);
        PngDeviceBuffer<unsigned char> compressionInput{filtered.size()};
        ASSERT_EQ(cudaMemcpyAsync(compressionInput.Get(),
                                  filtered.data(),
                                  filtered.size(),
                                  cudaMemcpyHostToDevice,
                                  stream.Get()),
                  cudaSuccess);
        std::vector<unsigned char> fixedZlib;
        ASSERT_EQ(Png::EncodeZlibFixed(compressionInput.Get(), filtered.size(), stream.Get(), fixedZlib),
                  cudaSuccess);
        const size_t fixedChunks = (fixedZlib.size() - 1) / Png::IdatPayloadBytes + 1;
        const size_t fixedPngBytes = fixedZlib.size() + fixedChunks * 12 + 45;
        ASSERT_TRUE(gpu.size() <= fixedPngBytes);
        if (label != "noise") {
            ASSERT_TRUE(gpu.size() < fixedPngBytes);
        }
        std::cout << "PNG benchmark " << label << ": GPU cold ms=" << coldMs
                  << " GPU warm median ms=" << gpuTimes[1] << " CPU median ms=" << cpuTimes[1]
                  << " GPU bytes=" << gpu.size() << " CPU bytes=" << cpu.size()
                  << " fixed GPU bytes=" << fixedPngBytes
                  << " workspace bytes=" << encoder.GetWorkspaceBytes() << '\n';
    }
}

const bool registered = [] {
    TestFramework::RegisterCase("CudaPng_ShapesAnd16BitPixels", CheckShapes, true, "", false);
    TestFramework::RegisterCase(
        "CudaPng_NoiseCompressionAndIdatChunks", CheckNoiseAndChunks, true, "", false);
    TestFramework::RegisterCase("CudaPng_AllScanlineFilters", CheckFilters, true, "", false);
    TestFramework::RegisterCase("CudaPng_DeflateBoundaries", CheckDeflateBoundaries, true, "", false);
    TestFramework::RegisterCase("CudaPng_HuffmanLengths", CheckHuffmanLengths, true, "", false);
    TestFramework::RegisterCase("CudaPng_HeaderRuns", CheckHeaderRuns, true, "", false);
    TestFramework::RegisterCase("CudaPng_BlockSelection", CheckBlockSelection, true, "", false);
    TestFramework::RegisterCase("CudaPng_CorruptionRejected", CheckCorruption, true, "", false);
    TestFramework::RegisterCase(
        "CudaPng_WorkspaceReuseAndConcurrentStreams", CheckReuseAndConcurrency, true, "", false);
    TestFramework::RegisterCase("CudaPng_InvalidInputs", CheckInvalidInputs, true, "", false);
    TestFramework::RegisterCase("CudaPngBenchmark_4K", Benchmark4K, true, "", false);
    return true;
}();

} // namespace
