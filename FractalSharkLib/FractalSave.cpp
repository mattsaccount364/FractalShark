#include "stdafx.h"
#include "FractalSave.h"

#include "Fractal.h"

//////////////////////////////////////////////////////////////////////////////
// Saves the current fractal as a bitmap to the given file.
// If halfImage is true, a bitmap with half the dimensions of the current
// fractal is saved instead.  Thus, 1024x768 is resized to 512x384.
//////////////////////////////////////////////////////////////////////////////

CurrentFractalSave::CurrentFractalSave(
    enum Type typ,
    std::wstring filename_base,
    Fractal& fractal)
    : m_Type(typ),
    m_FilenameBase(filename_base),
    m_Fractal(fractal),
    m_ScrnWidth(fractal.m_ScrnWidth),
    m_ScrnHeight(fractal.m_ScrnHeight),
    m_GpuAntialiasing(fractal.m_GpuAntialiasing),
    m_NumIterations(fractal.m_NumIterations),
    m_PaletteRotate(fractal.m_PaletteRotate),
    m_PaletteDepthIndex(fractal.m_PaletteDepthIndex),
    m_PaletteAuxDepth(fractal.m_PaletteAuxDepth),
    m_WhichPalette(fractal.m_WhichPalette),
    m_CurIters(std::move(fractal.m_CurIters)) {

    //
    // TODO Note we pass off ownership of m_CurIters.
    // Implication is that if you save multiple copies of the same bit map, it's not
    // going to work sensibly.  This is a bug.
    //

    fractal.SetCurItersMemory();

    for (size_t i = 0; i < PaletteStyle::Num; i++) {
        m_PalR[i] = fractal.m_PalR[i];
        m_PalG[i] = fractal.m_PalG[i];
        m_PalB[i] = fractal.m_PalB[i];

        m_PalIters[i] = fractal.m_PalIters[i];
    }

    m_Thread = nullptr;
    m_Destructable = false;
}

CurrentFractalSave::~CurrentFractalSave() {
    if (m_Thread) {
        m_Thread->join();
    }
}

void CurrentFractalSave::StartThread() {
    assert(m_Thread == nullptr);
    m_Thread = std::unique_ptr<std::thread>(new std::thread(&CurrentFractalSave::Run, this));
}

void CurrentFractalSave::Run() {
    int ret;
    std::wstring final_filename;

    std::wstring ext;
    if (m_Type == Type::PngImg) {
        ext = L".png";
    }
    else {
        ext = L".txt";
    }

    if (m_FilenameBase != L"") {
        wchar_t temp[512];
        wsprintf(temp, L"%s", m_FilenameBase.c_str());
        final_filename = std::wstring(temp) + ext;
        if (Fractal::FileExists(final_filename.c_str())) {
            ::MessageBox(nullptr, L"Not saving, file exists", L"", MB_OK | MB_APPLMODAL);
            return;
        }
    }
    else {
        size_t i = 0;
        do {
            wchar_t temp[512];
            wsprintf(temp, L"output%05d", i);
            final_filename = std::wstring(temp) + ext;
            i++;
        } while (Fractal::FileExists(final_filename.c_str()));
    }

    // TODO racy bug, changing iteration type while save in progress.
    IterTypeFull maxPossibleIters = m_Fractal.GetMaxIterationsRT();

    //setup converter deprecated
    //using convert_type = std::codecvt_utf8<wchar_t>;
    //std::wstring_convert<convert_type, wchar_t> converter;
    ////use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    //const std::string filename_c = converter.to_bytes(final_filename);

    std::string filename_c;
    std::transform(final_filename.begin(), final_filename.end(), std::back_inserter(filename_c), [](wchar_t c) {
        return (char)c;
        });

    if (m_Type == Type::PngImg) {
        double acc_r, acc_b, acc_g;
        size_t input_x, input_y;
        size_t output_x, output_y;
        size_t numIters;

        WPngImage image((int)m_ScrnWidth, (int)m_ScrnHeight, WPngImage::Pixel16(0, 0, 0));

        for (output_y = 0; output_y < m_ScrnHeight; output_y++)
        {
            for (output_x = 0; output_x < m_ScrnWidth; output_x++)
            {
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
                        if (numIters < m_NumIterations)
                        {
                            numIters += m_PaletteRotate;
                            if (numIters >= maxPossibleIters) {
                                numIters = maxPossibleIters - 1;
                            }

                            auto palIndex = (numIters >> m_Fractal.m_PaletteAuxDepth) % m_PalIters[m_WhichPalette][m_PaletteDepthIndex];

                            acc_r += m_PalR[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                            acc_g += m_PalG[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                            acc_b += m_PalB[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                        }
                    }
                }

                acc_r /= m_GpuAntialiasing * m_GpuAntialiasing;
                acc_g /= m_GpuAntialiasing * m_GpuAntialiasing;
                acc_b /= m_GpuAntialiasing * m_GpuAntialiasing;

                //if (index > GetMaxIterations<IterType>()) {
                //    index = GetMaxIterations<IterType>() - 1;
                //}

                //data[i] = (unsigned char)acc_r;
                //i++;
                //data[i] = (unsigned char)acc_g;
                //i++;
                //data[i] = (unsigned char)acc_b;
                //i++;

                image.set((int)output_x,
                    (int)output_y,
                    WPngImage::Pixel16((uint16_t)acc_r, (uint16_t)acc_g, (uint16_t)acc_b));
            }
        }

        m_Fractal.ReturnIterMemory(std::move(m_CurIters));

        ret = image.saveImage(filename_c, WPngImage::PngFileFormat::kPngFileFormat_RGBA16);
    }
    else {
        constexpr size_t buf_size = 128;
        char one_val[buf_size];
        std::string out_str;

        for (uint32_t output_y = 0; output_y < m_ScrnHeight * m_GpuAntialiasing; output_y++) {
            for (uint32_t output_x = 0; output_x < m_ScrnWidth * m_GpuAntialiasing; output_x++) {
                IterTypeFull numiters = m_CurIters.GetItersArrayValSlow(output_x, output_y);
                memset(one_val, ' ', sizeof(one_val));

                //static_assert(sizeof(IterType) == 8, "!");
                //char(*__kaboom1)[sizeof(IterType)] = 1;
                sprintf(one_val, "(%u,%u):%llu ", output_x, output_y, (IterTypeFull)numiters);

                // Wow what a kludge
                //size_t orig_len = strlen(one_val);
                //one_val[orig_len] = ' ';
                //one_val[orig_len + 1] = 0;

                out_str += one_val;
            }

            out_str += "\n";
        }

        std::ofstream out(filename_c);
        out << out_str;
        out.close();

        m_Fractal.ReturnIterMemory(std::move(m_CurIters));
    }

    m_Destructable = true;
    return;
}