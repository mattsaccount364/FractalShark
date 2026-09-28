#pragma once

#include "GPU_Types.h"
#include "HighPrecision.h"
#include "ItersMemoryContainer.h"
#include <stdint.h>
#include <vector>

class Fractal;

// The palette!
enum FractalPaletteType : size_t { Basic = 0, Default, Patriotic, Summer, Random, Custom, Num };

class PngParallelSave {
public:
    enum class Type { ItersText, PngImg };

    PngParallelSave(enum Type typ, std::wstring filenameBase, bool copyTheIters, Fractal &fractal);
    ~PngParallelSave();
    void Run();

    PngParallelSave(PngParallelSave &&) = delete;

    Type m_Type;
    Fractal &m_Fractal;
    size_t m_ScrnWidth;
    size_t m_ScrnHeight;
    uint32_t m_GpuAntialiasing;
    IterTypeFull m_NumIterations;
    IterTypeFull m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepthIndex;      // 0, 1, 2
    int m_PaletteAuxDepth;
    IterTypeFull m_MaxPossibleIters;
    FractalPaletteType m_WhichPalette;
    std::vector<Color16> m_PaletteColors;
    uint32_t m_NumPaletteColors;
    ItersMemoryContainer m_CurIters;
    bool m_CopyTheIters;
    std::wstring m_FilenameBase;
};
