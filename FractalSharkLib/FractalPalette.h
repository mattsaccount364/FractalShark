#pragma once

#include "PngParallelSave.h"

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
    const uint16_t *GetCurrentPalR() const;
    const uint16_t *GetCurrentPalG() const;
    const uint16_t *GetCurrentPalB() const;
    uint32_t GetCurrentNumColors() const;

    // Access palette arrays for PngParallelSave
    std::vector<uint16_t> *GetPalR(size_t whichPalette);
    std::vector<uint16_t> *GetPalG(size_t whichPalette);
    std::vector<uint16_t> *GetPalB(size_t whichPalette);
    const std::vector<uint32_t> &GetPalIters(size_t whichPalette) const;

    void SetDefaults();

private:
    static void PalIncrease(std::vector<uint16_t> &pal, int length, int val1, int val2);
    void PalTransition(size_t WhichPalette, size_t paletteIndex, int length, int r, int g, int b);

    static constexpr size_t NumBitDepths = 6;

    std::vector<uint16_t> m_PalR[FractalPaletteType::Num][NumBitDepths];
    std::vector<uint16_t> m_PalG[FractalPaletteType::Num][NumBitDepths];
    std::vector<uint16_t> m_PalB[FractalPaletteType::Num][NumBitDepths];
    std::vector<uint32_t> m_PalIters[FractalPaletteType::Num];
    FractalPaletteType m_WhichPalette;

    IterTypeFull m_PaletteRotate;
    int m_PaletteDepthIndex;
    int m_PaletteAuxDepth;
};
