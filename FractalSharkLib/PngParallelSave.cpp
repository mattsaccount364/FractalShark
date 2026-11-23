#include "stdafx.h"

#include "ConsoleLog.h"
#include "Environment.h"
#include "FractalSaveThreadPool.h"
#include "FractalSharkGpuLib/GpuSaveEncoder.h"
#include "PngParallelSave.h"
#include "PngSaveLayout.h"
#include "SaveFileTransaction.h"
#include "Utilities.h"
#include "WPngImage/lodepng.h"
#include <algorithm>

#include <cinttypes>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace {

bool
PathHasFilenameExtension(const std::wstring &path)
{
    const size_t lastSeparator = path.find_last_of(L"/\\");
    const size_t filenameStart = lastSeparator == std::wstring::npos ? 0 : lastSeparator + 1;
    const size_t lastDot = path.find_last_of(L'.');
    return lastDot != std::wstring::npos && lastDot > filenameStart;
}

std::wstring
WidenForLog(std::string_view str)
{
    return {str.begin(), str.end()};
}

void
ReportSaveError(const std::wstring &filename, const std::wstring &message, const char *file, int line)
{
    auto log = FractalSharkLog::LogLine(file, line);
    log << L"Failed to save " << filename << L": " << message;
}

std::wstring
DefaultFilename(uint64_t index, const std::wstring &extension)
{
    std::wostringstream name;
    name << L"output" << std::setfill(L'0') << std::setw(5) << index << extension;
    return name.str();
}

// Offline PNG and video use identical palette saturation and antialiasing.
template <typename IterType>
void
ForEachCpuColor(const ItersMemoryContainer &frame,
                uint64_t numIterations,
                uint64_t rotation,
                uint32_t depth,
                uint64_t maxIterations,
                const std::vector<Color16> &palette,
                const auto &consume)
{
    const auto *iterations = frame.GetIters<IterType>();
    const size_t aa = frame.m_Antialiasing;
    const auto samples = static_cast<uint32_t>(aa * aa);
    const auto limit = maxIterations - 1;
    for (size_t y = 0; y < frame.m_OutputHeight; ++y) {
        for (size_t x = 0; x < frame.m_OutputWidth; ++x) {
            uint32_t red = 0, green = 0, blue = 0;
            for (size_t sy = 0; sy < aa; ++sy) {
                const size_t row = (y * aa + sy) * frame.m_RoundedWidth;
                for (size_t sx = 0; sx < aa; ++sx) {
                    const uint64_t count = iterations[row + x * aa + sx];
                    if (count < numIterations) {
                        const uint64_t rotated =
                            count >= limit || rotation >= limit - count ? limit : count + rotation;
                        const auto &color = palette[(rotated >> depth) % palette.size()];
                        red += color.r;
                        green += color.g;
                        blue += color.b;
                    }
                }
            }
            consume({static_cast<uint16_t>(red / samples),
                     static_cast<uint16_t>(green / samples),
                     static_cast<uint16_t>(blue / samples),
                     UINT16_MAX});
        }
    }
}

} // namespace

PngParallelSave::PngParallelSave(Type type,
                                 EncoderBackend backend,
                                 std::wstring filenameBase,
                                 SaveOverwritePolicy overwrite,
                                 ItersMemoryContainer &iterations,
                                 bool copyTheIters,
                                 uint64_t numIterations,
                                 uint64_t paletteRotation,
                                 uint32_t paletteAuxDepth,
                                 uint64_t maxPossibleIterations,
                                 std::vector<Color16> paletteColors,
                                 FractalSaveThreadPool &pool,
                                 ReturnIterationsFunction returnIterations)
    : m_Type(type), m_Backend(backend), m_FilenameBase(std::move(filenameBase)), m_Overwrite(overwrite),
      m_NumIterations(numIterations), m_PaletteRotate(paletteRotation),
      m_PaletteAuxDepth(paletteAuxDepth), m_MaxPossibleIters(maxPossibleIterations),
      m_PaletteColors(std::move(paletteColors)), m_Pool(pool),
      m_ReturnIterations(std::move(returnIterations))
{
    // Capture under the submitting thread's existing render-state ownership. A factory
    // reservation bounds these buffers before any copy or move takes place.
    if (copyTheIters) {
        m_CurIters.CopyIterationsFrom(iterations);
    } else {
        m_CurIters = std::move(iterations);
    }
}

PngParallelSave::~PngParallelSave()
{
    if (m_ReturnIterations) {
        m_ReturnIterations(std::move(m_CurIters));
    }
}

std::wstring
PngParallelSave::ResolveFilename(uint64_t index) const
{
    const std::wstring extension = m_Type == Type::PngImg ? L".png" : L".txt";
    if (!m_FilenameBase.empty()) {
        return PathHasFilenameExtension(m_FilenameBase) ? m_FilenameBase : m_FilenameBase + extension;
    }
    return DefaultFilename(index, extension);
}

bool
PngParallelSave::Validate(std::wstring &error) const
{
    const auto aa = m_CurIters.m_Antialiasing;
    size_t bytes = 0;
    const bool bits32 = m_CurIters.GetIterType() == IterTypeEnum::Bits32;
    const bool validType = bits32 || m_CurIters.GetIterType() == IterTypeEnum::Bits64;
    const auto maximumCount = bits32 ? static_cast<uint64_t>(UINT32_MAX) : UINT64_MAX;
    // Check every product before accessing rows; padded storage remains authoritative.
    if (!validType || aa < 1 || aa > 4 ||
        !Png::Rgb16ByteCount(m_CurIters.m_OutputWidth, m_CurIters.m_OutputHeight, bytes) ||
        m_CurIters.m_OutputWidth > SIZE_MAX / aa || m_CurIters.m_OutputHeight > SIZE_MAX / aa ||
        m_CurIters.m_Width != m_CurIters.m_OutputWidth * aa ||
        m_CurIters.m_Height != m_CurIters.m_OutputHeight * aa ||
        m_CurIters.m_RoundedWidth < m_CurIters.m_Width ||
        m_CurIters.m_RoundedHeight < m_CurIters.m_Height ||
        m_CurIters.m_RoundedWidth > SIZE_MAX / m_CurIters.m_RoundedHeight ||
        m_CurIters.m_RoundedTotal < m_CurIters.m_RoundedWidth * m_CurIters.m_RoundedHeight ||
        m_CurIters.m_RoundedTotal > m_CurIters.GetIterationCount() ||
        (bits32 ? m_CurIters.GetIters<uint32_t>() == nullptr
                : m_CurIters.GetIters<uint64_t>() == nullptr) ||
        m_NumIterations == 0 || m_NumIterations > maximumCount || m_MaxPossibleIters < 2 ||
        m_NumIterations > m_MaxPossibleIters || m_PaletteAuxDepth >= 64 ||
        (m_Type == Type::PngImg && m_PaletteColors.empty())) {
        error = L"invalid save dimensions, iteration storage, or palette settings";
        return false;
    }
    return true;
}

template <typename IterType>
int
PngParallelSave::EncodeGpuPng(std::vector<unsigned char> &pngBytes, std::wstring &error)
{
    if (m_CurIters.m_OutputWidth > std::numeric_limits<uint32_t>::max() ||
        m_CurIters.m_OutputHeight > std::numeric_limits<uint32_t>::max() ||
        m_PaletteColors.size() > std::numeric_limits<uint32_t>::max()) {
        error = L"GPU PNG dimensions or palette exceed supported limits";
        return 1;
    }
    // The pool owns the reusable context. The lease covers encoding, never file I/O.
    const auto lease = m_Pool.AcquireGpuEncoding();
    const auto result = lease.GetEncoder().Encode(m_CurIters.GetIters<IterType>(),
                                                  m_CurIters.m_RoundedWidth,
                                                  static_cast<uint32_t>(m_CurIters.m_OutputWidth),
                                                  static_cast<uint32_t>(m_CurIters.m_OutputHeight),
                                                  static_cast<uint32_t>(m_CurIters.m_Antialiasing),
                                                  static_cast<IterType>(m_NumIterations),
                                                  m_PaletteColors.data(),
                                                  static_cast<uint32_t>(m_PaletteColors.size()),
                                                  m_PaletteAuxDepth,
                                                  m_PaletteRotate,
                                                  m_MaxPossibleIters,
                                                  pngBytes);
    if (result != 0) {
        error = L"GPU PNG encoder failed: " + WidenForLog(Png::GpuSaveEncoder::ErrorString(result));
    }
    return static_cast<int>(result);
}

int
PngParallelSave::EncodeCpuPng(std::vector<unsigned char> &pngBytes, std::wstring &error)
{
    return m_CurIters.GetIterType() == IterTypeEnum::Bits32
               ? EncodeCpuPngTyped<uint32_t>(pngBytes, error)
               : EncodeCpuPngTyped<uint64_t>(pngBytes, error);
}

template <typename IterType>
int
PngParallelSave::EncodeCpuPngTyped(std::vector<unsigned char> &pngBytes, std::wstring &error)
{
    size_t byteCount = 0;
    if (!Png::Rgb16ByteCount(m_CurIters.m_OutputWidth, m_CurIters.m_OutputHeight, byteCount)) {
        error = L"RGB16 byte count exceeds supported limits";
        return 1;
    }
    std::vector<unsigned char> raw(byteCount);
    size_t offset = 0;
    ForEachCpuColor<IterType>(m_CurIters,
                              m_NumIterations,
                              m_PaletteRotate,
                              m_PaletteAuxDepth,
                              m_MaxPossibleIters,
                              m_PaletteColors,
                              [&](Color16 color) {
                                  for (const auto value : {color.r, color.g, color.b}) {
                                      raw[offset++] = static_cast<unsigned char>(value >> 8);
                                      raw[offset++] = static_cast<unsigned char>(value);
                                  }
                              });
    // Preserve the CPU encoder policy while removing the intermediate Color16
    // image and the wrapper's 32-bit raw-buffer size arithmetic.
    lodepng::State state;
    state.info_raw.colortype = LCT_RGB;
    state.info_raw.bitdepth = 16;
    state.info_png.color.colortype = LCT_RGB;
    state.info_png.color.bitdepth = 16;
    state.encoder.auto_convert = false;
    state.encoder.zlibsettings.lazymatching = false;
    state.encoder.zlibsettings.nicematch = 32;
    const unsigned code = lodepng::encode(pngBytes,
                                          raw,
                                          static_cast<unsigned>(m_CurIters.m_OutputWidth),
                                          static_cast<unsigned>(m_CurIters.m_OutputHeight),
                                          state);
    if (code != 0) {
        error = L"PNG encoder failed: " + WidenForLog(lodepng_error_text(code));
    }
    return static_cast<int>(code);
}

std::vector<Color16>
PngParallelSave::ColorCpuFrame(const ItersMemoryContainer &frame,
                               uint64_t numIterations,
                               uint64_t rotation,
                               uint32_t depth,
                               uint64_t maxIterations,
                               const std::vector<Color16> &palette)
{
    std::vector<Color16> colors;
    colors.reserve(frame.m_OutputWidth * frame.m_OutputHeight);
    const auto append = [&](Color16 color) { colors.push_back(color); };
    if (frame.GetIterType() == IterTypeEnum::Bits32) {
        ForEachCpuColor<uint32_t>(frame, numIterations, rotation, depth, maxIterations, palette, append);
    } else {
        ForEachCpuColor<uint64_t>(frame, numIterations, rotation, depth, maxIterations, palette, append);
    }
    return colors;
}

void
PngParallelSave::WriteIterations(std::ostream &out) const
{
    out << "# x, y, and iteration counts are decimal.\n";
    for (size_t y = 0; y < m_CurIters.m_Height; ++y) {
        for (size_t x = 0; x < m_CurIters.m_Width; ++x) {
            out << "(x=" << x << ",y=" << y << "):iters=" << m_CurIters.GetItersArrayValSlow(x, y)
                << " ";
        }
        out << "\n";
    }
}

SaveResult
PngParallelSave::Run()
{
    uint64_t index = 0;
    auto filename = ResolveFilename(index);
    const bool unnamed = m_FilenameBase.empty();
    const auto overwrite = unnamed ? SaveOverwritePolicy::SkipExisting : m_Overwrite;
    if (!unnamed && overwrite == SaveOverwritePolicy::SkipExisting &&
        Utilities::FileExists(filename.c_str())) {
        return {SaveOutcome::SkippedExisting, filename, 0, {}};
    }
    std::wstring error;
    int code = Validate(error) ? 0 : 1;
    std::vector<unsigned char> pngBytes;
    if (code == 0 && m_Type == Type::PngImg) {
        code = m_Backend == EncoderBackend::Cpu ? EncodeCpuPng(pngBytes, error)
               : m_CurIters.GetIterType() == IterTypeEnum::Bits32
                   ? EncodeGpuPng<uint32_t>(pngBytes, error)
                   : EncodeGpuPng<uint64_t>(pngBytes, error);
    }
    if (code != 0) {
        ReportSaveError(filename, error, __FILE__, __LINE__);
        return {SaveOutcome::Failed, filename, code, Environment::WideToUtf8(error)};
    }
    for (;;) {
        // Avoid rewriting the encoded image for every previously occupied numbered
        // name. This is only a hint: publication still arbitrates concurrent saves.
        if (unnamed && Utilities::FileExists(filename.c_str())) {
            filename = ResolveFilename(++index);
            continue;
        }
        auto result = SaveFileTransaction::Write(filename, overwrite, [&](std::ostream &out) {
            if (m_Type == Type::ItersText) {
                WriteIterations(out);
            } else {
                // Avoid narrowing a huge encoded vector to streamsize.
                constexpr size_t writeChunkBytes = 1024 * 1024;
                for (size_t offset = 0; offset < pngBytes.size() && out;) {
                    const size_t bytes = std::min(writeChunkBytes, pngBytes.size() - offset);
                    out.write(reinterpret_cast<const char *>(pngBytes.data() + offset),
                              static_cast<std::streamsize>(bytes));
                    offset += bytes;
                }
            }
        });
        if (unnamed && result.Outcome == SaveOutcome::SkippedExisting) {
            filename = ResolveFilename(++index);
            continue;
        }
        if (!result.Succeeded()) {
            ReportSaveError(filename, WidenForLog(result.Diagnostic), __FILE__, __LINE__);
        }
        return result;
    }
}
