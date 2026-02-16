#pragma once

#include "PngParallelSave.h"
#include "GPU_Types.h"

#include <cstdint>
#include <vector>

class FractalPalette {
public:
    FractalPalette();

    void InitializeAllPalettes();
    void CreateNewRandomPalette();

    void UsePaletteType(FractalPaletteType type);
    FractalPaletteType GetPaletteType() const;
    void UsePalette(int depth);
    void UseNextPaletteDepth();
    uint32_t GetPaletteDepth() const;
    uint32_t GetPaletteDepthFromIndex(size_t index) const;
    void SetPaletteAuxDepth(int32_t depth);
    void UseNextPaletteAuxDepth(int32_t inc);

    void ResetPaletteRotation();
    void RotatePalette(int delta, IterTypeFull maxIters);

    IterTypeFull GetPaletteRotation() const;
    int GetPaletteDepthIndex() const;
    int32_t GetAuxDepth() const;

    // Data access for GPU upload and drawing
    const Color16 *GetCurrentPalInterleaved() const;
    uint32_t GetCurrentNumColors() const;

    // Access palette arrays for PngParallelSave
    const std::vector<Color16> *GetPalInterleaved(size_t whichPalette) const;
    const std::vector<uint32_t> &GetPalIters(size_t whichPalette) const;

    void SetDefaults();

private:
    void PalTransition(size_t WhichPalette, size_t paletteIndex, int length, int r, int g, int b);

    static constexpr size_t NumBitDepths = 6;

    std::vector<Color16> m_PalInterleaved[FractalPaletteType::Num][NumBitDepths];
    std::vector<uint32_t> m_PalIters[FractalPaletteType::Num];
    FractalPaletteType m_WhichPalette;

    IterTypeFull m_PaletteRotate;
    int m_PaletteDepthIndex;
    int m_PaletteAuxDepth;
};
