#include "stdafx.h"

#include "Environment.h"
#include "Fractal.h"
#include "PngParallelSave.h"

#include <cinttypes>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

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
ReportSaveError(const std::wstring &filename, const std::wstring &message)
{
    std::wcerr << L"Failed to save " << filename << L": " << message << std::endl;
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

    return out.good();
}

} // namespace

//////////////////////////////////////////////////////////////////////////////
// Saves the current fractal as a bitmap to the given file.
// If halfImage is true, a bitmap with half the dimensions of the current
// fractal is saved instead.  Thus, 1024x768 is resized to 512x384.
//////////////////////////////////////////////////////////////////////////////

PngParallelSave::PngParallelSave(enum Type typ,
                                 std::wstring filename_base,
                                 bool copy_the_iters,
                                 Fractal &fractal)
    : m_Type(typ), m_Fractal(fractal), m_ScrnWidth(fractal.m_ScrnWidth),
      m_ScrnHeight(fractal.m_ScrnHeight), m_GpuAntialiasing(fractal.m_GpuAntialiasing),
      m_NumIterations(fractal.m_NumIterations),
      m_PaletteRotate(fractal.GetPalette().GetPaletteRotation()),
      m_PaletteDepthIndex(fractal.GetPalette().GetPaletteDepthIndex()),
      m_PaletteAuxDepth(fractal.GetPalette().GetAuxDepth()),
      m_MaxPossibleIters(fractal.GetMaxIterationsRT()),
      m_WhichPalette(fractal.GetPalette().GetPaletteType()), m_PaletteColors{}, m_NumPaletteColors(),
      m_CurIters{}, m_CopyTheIters(copy_the_iters), m_FilenameBase(filename_base), m_Thread(nullptr),
      m_Destructable(false)
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
    if (m_Thread) {
        m_Thread->join();
    }
}

void
PngParallelSave::StartThread()
{
    assert(m_Thread == nullptr);
    m_Thread = std::unique_ptr<std::thread>(DEBUG_NEW std::thread(&PngParallelSave::Run, this));
}

void
PngParallelSave::Run()
{
    // Ensure m_Destructable is set on every exit path (early return, normal
    // completion, or exception). Fractal::CleanupThreads waits for this flag;
    // missing it on any path hangs the process at exit.
    struct DestructableGuard {
        PngParallelSave *self;
        ~DestructableGuard() { self->m_Destructable.store(true, std::memory_order_release); }
    } destructableGuard{this};

    struct IterMemoryReturnGuard {
        PngParallelSave *self;
        bool active;
        ~IterMemoryReturnGuard()
        {
            if (active) {
                self->m_Fractal.ReturnIterMemory(std::move(self->m_CurIters));
            }
        }
    } iterMemoryReturnGuard{this, !m_CopyTheIters};

    try {
        Environment::SetCurrentThreadName(L"PngParallelSave::Run");

        std::wstring final_filename;

        std::wstring ext;
        if (m_Type == Type::PngImg) {
            ext = L".png";
        } else {
            ext = L".txt";
        }

        if (m_FilenameBase != L"") {
            final_filename = m_FilenameBase;
            if (!PathHasFilenameExtension(final_filename)) {
                final_filename += ext;
            }
            if (Utilities::FileExists(final_filename.c_str())) {
                std::wcerr << L"Not saving, file exists" << std::endl;
                return;
            }
        } else {
            int i = 0;
            do {
                final_filename = DefaultFilename(i, ext);
                i++;
            } while (Utilities::FileExists(final_filename.c_str()));
        }

        const std::filesystem::path finalPath(final_filename);

        if (m_Type == Type::PngImg) {
            if (m_NumPaletteColors == 0) {
                ReportSaveError(final_filename, L"selected palette has no colors");
                return;
            }

            double acc_r, acc_b, acc_g;
            size_t input_x, input_y;
            size_t output_x, output_y;
            IterTypeFull numIters;

            WPngImage image((int)m_ScrnWidth, (int)m_ScrnHeight, WPngImage::Pixel16(0, 0, 0));

            for (output_y = 0; output_y < m_ScrnHeight; output_y++) {
                for (output_x = 0; output_x < m_ScrnWidth; output_x++) {
                    acc_r = 0;
                    acc_g = 0;
                    acc_b = 0;

                    for (input_x = output_x * m_GpuAntialiasing;
                         input_x < (output_x + 1) * m_GpuAntialiasing;
                         input_x++) {
                        for (input_y = output_y * m_GpuAntialiasing;
                             input_y < (output_y + 1) * m_GpuAntialiasing;
                             input_y++) {

                            numIters = m_CurIters.GetItersArrayValSlow(input_x, input_y);
                            if (numIters < m_NumIterations) {
                                numIters += m_PaletteRotate;
                                if (numIters >= m_MaxPossibleIters) {
                                    numIters = m_MaxPossibleIters - 1;
                                }

                                auto shiftedIters = (numIters >> m_PaletteAuxDepth);
                                auto palIndex = shiftedIters % m_NumPaletteColors;

                                acc_r += m_PaletteColors[palIndex].r;
                                acc_g += m_PaletteColors[palIndex].g;
                                acc_b += m_PaletteColors[palIndex].b;
                            }
                        }
                    }

                    acc_r /= m_GpuAntialiasing * m_GpuAntialiasing;
                    acc_g /= m_GpuAntialiasing * m_GpuAntialiasing;
                    acc_b /= m_GpuAntialiasing * m_GpuAntialiasing;

                    image.set((int)output_x,
                              (int)output_y,
                              WPngImage::Pixel16((uint16_t)acc_r, (uint16_t)acc_g, (uint16_t)acc_b));
                }
            }

            std::vector<unsigned char> pngBytes;
            const auto status =
                image.saveImageToRAM(pngBytes, WPngImage::PngFileFormat::kPngFileFormat_RGBA16);
            if (status != WPngImage::kIOStatus_Ok) {
                std::wstring message = L"PNG encoder failed";
                if (!status.pngLibErrorMsg.empty()) {
                    message += L": ";
                    message += WidenForLog(status.pngLibErrorMsg);
                }
                ReportSaveError(final_filename, message);
                return;
            }

            if (!WriteBinaryFile(finalPath, pngBytes)) {
                ReportSaveError(final_filename, L"could not write PNG file");
                return;
            }
        } else {
            std::ofstream out(finalPath);
            if (!out) {
                ReportSaveError(final_filename, L"could not open text file");
                return;
            }

            out << "# x, y, and iteration counts are decimal.\n";

            for (size_t output_y = 0; output_y < m_ScrnHeight * m_GpuAntialiasing; output_y++) {
                for (size_t output_x = 0; output_x < m_ScrnWidth * m_GpuAntialiasing; output_x++) {
                    IterTypeFull numiters = m_CurIters.GetItersArrayValSlow(output_x, output_y);
                    out << "(x=" << output_x << ",y=" << output_y << "):iters=" << numiters << " ";
                }

                out << "\n";
            }

            if (!out) {
                ReportSaveError(final_filename, L"could not write text file");
                return;
            }
        }
    } catch (const std::exception &ex) {
        std::cerr << "PngParallelSave failed: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "PngParallelSave failed with an unknown exception" << std::endl;
    }
}
