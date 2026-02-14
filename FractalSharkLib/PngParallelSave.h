#pragma once

#include "HighPrecision.h"
#include "ItersMemoryContainer.h"
#include <stdint.h>
#include <thread>

class Fractal;

// The palette!
enum FractalPaletteType : size_t {
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
        Fractal &fractal);
    ~PngParallelSave();
    void Run();
    void StartThread();

    PngParallelSave(PngParallelSave &&) = default;

    Type m_Type;
    Fractal &m_Fractal;
    size_t m_ScrnWidth;
    size_t m_ScrnHeight;
    uint32_t m_GpuAntialiasing;
    IterTypeFull m_NumIterations;
    IterTypeFull m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepthIndex; // 0, 1, 2
    int m_PaletteAuxDepth;
    std::vector<uint16_t> *m_PalR[FractalPaletteType::Num], *m_PalG[FractalPaletteType::Num], *m_PalB[FractalPaletteType::Num];
    FractalPaletteType m_WhichPalette;
    std::vector<uint32_t> m_PalIters[FractalPaletteType::Num];
    ItersMemoryContainer m_CurIters;
    bool m_CopyTheIters;
    std::wstring m_FilenameBase;
    std::unique_ptr<std::thread> m_Thread;
    bool m_Destructable;
};