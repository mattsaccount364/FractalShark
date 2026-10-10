#include "stdafx.h"

#include "ConsoleLog.h"
#include "Environment.h"
#include "Fractal.h"
#include "FractalSaveThreadPool.h"
#include "PngParallelSave.h"

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
DefaultFilename(int index, const std::wstring &extension)
{
    std::wostringstream name;
    name << L"output" << std::setfill(L'0') << std::setw(5) << index << extension;
    return name.str();
}

bool
WriteBinaryFile(const std::filesystem::path &path, const std::vector<unsigned char> &bytes)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    }

    out.close();
    return out.good();
}

} // namespace

PngParallelSave::PngParallelSave(
    Type typ, EncoderBackend backend, std::wstring filenameBase, bool copyTheIters, Fractal &fractal)
    : m_Type(typ), m_Backend(backend), m_Fractal(fractal), m_ScrnWidth(fractal.m_ScrnWidth),
      m_ScrnHeight(fractal.m_ScrnHeight), m_GpuAntialiasing(fractal.m_GpuAntialiasing),
      m_NumIterations(fractal.m_NumIterations),
      m_PaletteRotate(fractal.GetPalette().GetPaletteRotation()),
      m_PaletteDepthIndex(fractal.GetPalette().GetPaletteDepthIndex()),
      m_PaletteAuxDepth(fractal.GetPalette().GetAuxDepth()),
      m_MaxPossibleIters(fractal.GetMaxIterationsRT()),
      m_WhichPalette(fractal.GetPalette().GetPaletteType()), m_PaletteColors{}, m_NumPaletteColors(),
      m_PaletteGeneration(fractal.GetPalette().GetPaletteGeneration()),
      m_IterType(fractal.GetIterType()), m_CurIters{}, m_CopyTheIters(copyTheIters),
      m_FilenameBase(std::move(filenameBase))
{

    const std::vector<Color16> *palInterleaved = fractal.GetPalette().GetPalInterleaved(m_WhichPalette);
    m_PaletteColors = palInterleaved[m_PaletteDepthIndex];
    m_NumPaletteColors = static_cast<uint32_t>(m_PaletteColors.size());

    if (m_CopyTheIters) {
        m_CurIters = fractal.m_CurIters;
    } else {
        m_CurIters = std::move(fractal.m_CurIters);
        fractal.SetCurItersMemory();
    }
}

PngParallelSave::~PngParallelSave()
{
    if (!m_CopyTheIters) {
        m_Fractal.ReturnIterMemory(std::move(m_CurIters));
    }
}

template <typename IterType>
uint32_t
PngParallelSave::EncodeGpuPng(std::vector<unsigned char> &pngBytes)
{
    if (m_CurIters.m_Width > std::numeric_limits<uint32_t>::max() ||
        m_CurIters.m_Height > std::numeric_limits<uint32_t>::max()) {
        return 1;
    }
    // The lease protects both lazy construction and all GPU workspace use. Its destructor
    // releases the encoder before Run writes the host PNG bytes to disk.
    const auto lease = m_Fractal.m_SavePool->AcquireGpuEncoding();
    if (!m_Fractal.m_SaveGpuRenderer) {
        m_Fractal.m_SaveGpuRenderer = std::make_unique<GPURenderer>();
    }
    auto &renderer = *m_Fractal.m_SaveGpuRenderer;
    auto result = renderer.InitializeMemory<IterType>(static_cast<uint32_t>(m_CurIters.m_Width),
                                                      static_cast<uint32_t>(m_CurIters.m_Height),
                                                      m_GpuAntialiasing,
                                                      m_PaletteColors.data(),
                                                      m_NumPaletteColors,
                                                      m_PaletteAuxDepth,
                                                      m_PaletteRotate,
                                                      m_MaxPossibleIters,
                                                      m_PaletteGeneration,
                                                      true);
    if (result == 0) {
        result = renderer.EncodePng(m_CurIters.GetIters<IterType>(),
                                    m_CurIters.m_RoundedWidth,
                                    static_cast<IterType>(m_NumIterations),
                                    pngBytes);
    }
    return result;
}

int
PngParallelSave::Run()
{
    if (m_Backend == EncoderBackend::Cpu) {
        Environment::SetCurrentThreadName(L"PngParallelSave::Run");
    }

    std::wstring finalFilename;

    std::wstring ext;
    if (m_Type == Type::PngImg) {
        ext = L".png";
    } else {
        ext = L".txt";
    }

    if (!m_FilenameBase.empty()) {
        finalFilename = m_FilenameBase;
        if (!PathHasFilenameExtension(finalFilename)) {
            finalFilename += ext;
        }
        if (Utilities::FileExists(finalFilename.c_str())) {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << L"Not saving, file exists";
            return 0;
        }
    } else {
        int i = 0;
        do {
            finalFilename = DefaultFilename(i, ext);
            i++;
        } while (Utilities::FileExists(finalFilename.c_str()));
    }

    const std::filesystem::path finalPath(finalFilename);

    if (m_Type == Type::PngImg) {
        if (m_NumPaletteColors == 0) {
            ReportSaveError(finalFilename, L"selected palette has no colors", __FILE__, __LINE__);
            return 1;
        }

        std::vector<unsigned char> pngBytes;
        if (m_Backend == EncoderBackend::Gpu) {
            const auto result = m_IterType == IterTypeEnum::Bits32 ? EncodeGpuPng<uint32_t>(pngBytes)
                                                                   : EncodeGpuPng<uint64_t>(pngBytes);
            if (result != 0) {
                ReportSaveError(
                    finalFilename,
                    L"GPU PNG encoder failed: " + WidenForLog(GPURenderer::ConvertErrorToString(result)),
                    __FILE__,
                    __LINE__);
                return static_cast<int>(result);
            }
        } else {
            double accR, accB, accG;
            size_t inputX, inputY;
            size_t outputX, outputY;
            IterTypeFull numIters;

            WPngImage image((int)m_ScrnWidth, (int)m_ScrnHeight, WPngImage::Pixel16(0, 0, 0));

            for (outputY = 0; outputY < m_ScrnHeight; outputY++) {
                for (outputX = 0; outputX < m_ScrnWidth; outputX++) {
                    accR = 0;
                    accG = 0;
                    accB = 0;

                    for (inputX = outputX * m_GpuAntialiasing;
                         inputX < (outputX + 1) * m_GpuAntialiasing;
                         inputX++) {
                        for (inputY = outputY * m_GpuAntialiasing;
                             inputY < (outputY + 1) * m_GpuAntialiasing;
                             inputY++) {

                            numIters = m_CurIters.GetItersArrayValSlow(inputX, inputY);
                            if (numIters < m_NumIterations) {
                                numIters += m_PaletteRotate;
                                if (numIters >= m_MaxPossibleIters) {
                                    numIters = m_MaxPossibleIters - 1;
                                }

                                auto shiftedIters = (numIters >> m_PaletteAuxDepth);
                                auto palIndex = shiftedIters % m_NumPaletteColors;

                                accR += m_PaletteColors[palIndex].r;
                                accG += m_PaletteColors[palIndex].g;
                                accB += m_PaletteColors[palIndex].b;
                            }
                        }
                    }

                    accR /= m_GpuAntialiasing * m_GpuAntialiasing;
                    accG /= m_GpuAntialiasing * m_GpuAntialiasing;
                    accB /= m_GpuAntialiasing * m_GpuAntialiasing;

                    image.set((int)outputX,
                              (int)outputY,
                              WPngImage::Pixel16((uint16_t)accR, (uint16_t)accG, (uint16_t)accB));
                }
            }

            const WPngImage::PngEncodingOptions encodingOptions{false, false, 32};
            const auto status = image.SaveImageToRAM(
                pngBytes, WPngImage::PngFileFormat::kPngFileFormat_RGBA16, encodingOptions);
            if (status != WPngImage::kIOStatus_Ok) {
                std::wstring message = L"PNG encoder failed";
                if (!status.pngLibErrorMsg.empty()) {
                    message += L": ";
                    message += WidenForLog(status.pngLibErrorMsg);
                }
                ReportSaveError(finalFilename, message, __FILE__, __LINE__);
                return 1;
            }
        }

        if (!WriteBinaryFile(finalPath, pngBytes)) {
            ReportSaveError(finalFilename, L"could not write PNG file", __FILE__, __LINE__);
            return 1;
        }
    } else {
        std::ofstream out(finalPath);
        if (!out) {
            ReportSaveError(finalFilename, L"could not open text file", __FILE__, __LINE__);
            return 1;
        }

        out << "# x, y, and iteration counts are decimal.\n";

        for (size_t outputY = 0; outputY < m_ScrnHeight * m_GpuAntialiasing; outputY++) {
            for (size_t outputX = 0; outputX < m_ScrnWidth * m_GpuAntialiasing; outputX++) {
                IterTypeFull numIters = m_CurIters.GetItersArrayValSlow(outputX, outputY);
                out << "(x=" << outputX << ",y=" << outputY << "):iters=" << numIters << " ";
            }

            out << "\n";
        }

        if (!out) {
            ReportSaveError(finalFilename, L"could not write text file", __FILE__, __LINE__);
            return 1;
        }
    }
    return 0;
}
