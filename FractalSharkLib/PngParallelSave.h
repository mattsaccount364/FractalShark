#pragma once

#include "HighPrecision.h"
#include "ItersMemoryContainer.h"
#include <stdint.h>

class Fractal;

// The palette!
enum FractalPalette : size_t {
    Basic = 0,
    Default,
    Patriotic,
    Summer,
    Random,
    Num
};

class PngParallelSave {
public:
    enum class Type {
        ItersText,
        PngImg
    };

    PngParallelSave(
        enum Type typ,
        std::wstring filename_base,
        bool copy_the_iters,
        Fractal& fractal);
    ~PngParallelSave();
    void Run();
    void StartThread();

    PngParallelSave(PngParallelSave&&) = default;

    Type m_Type;
    Fractal& m_Fractal;
    size_t m_ScrnWidth;
    size_t m_ScrnHeight;
    uint32_t m_GpuAntialiasing;
    IterTypeFull m_NumIterations;
    IterTypeFull m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepthIndex; // 0, 1, 2
    int m_PaletteAuxDepth;
    std::vector<uint16_t>* m_PalR[FractalPalette::Num], * m_PalG[FractalPalette::Num], * m_PalB[FractalPalette::Num];
    FractalPalette m_WhichPalette;
    std::vector<uint32_t> m_PalIters[FractalPalette::Num];
    ItersMemoryContainer m_CurIters;
    bool m_CopyTheIters;
    std::wstring m_FilenameBase;
    std::unique_ptr<std::thread> m_Thread;
    bool m_Destructable;
};