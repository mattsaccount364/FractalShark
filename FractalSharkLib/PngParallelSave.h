#pragma once

#include "ItersMemoryContainer.h"
#include "SaveResult.h"
#include <functional>
#include <ostream>
#include <string>
#include <vector>

class FractalSaveThreadPool;

// A queued save owns its iterations and captured palette. Only the optional buffer
// recycler refers back to Fractal; encoding never reads live render or palette state.
class PngParallelSave {
public:
    enum class Type { ItersText, PngImg };
    enum class EncoderBackend { Cpu, Gpu };
    using ReturnIterationsFunction = std::function<void(ItersMemoryContainer &&)>;

    PngParallelSave(Type type,
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
                    ReturnIterationsFunction returnIterations);
    ~PngParallelSave();
    PngParallelSave(const PngParallelSave &) = delete;
    PngParallelSave &operator=(const PngParallelSave &) = delete;
    PngParallelSave(PngParallelSave &&) = delete;
    SaveResult Run();

private:
    std::wstring ResolveFilename(uint64_t index) const;
    bool Validate(std::wstring &error) const;
    int EncodeCpuPng(std::vector<unsigned char> &pngBytes, std::wstring &error);
    template <typename IterType>
    int EncodeCpuPngTyped(std::vector<unsigned char> &pngBytes, std::wstring &error);
    template <typename IterType>
    int EncodeGpuPng(std::vector<unsigned char> &pngBytes, std::wstring &error);
    void WriteIterations(std::ostream &out) const;

    Type m_Type;
    EncoderBackend m_Backend;
    std::wstring m_FilenameBase;
    SaveOverwritePolicy m_Overwrite;
    ItersMemoryContainer m_CurIters;
    uint64_t m_NumIterations;
    uint64_t m_PaletteRotate;
    uint32_t m_PaletteAuxDepth;
    uint64_t m_MaxPossibleIters;
    std::vector<Color16> m_PaletteColors;
    FractalSaveThreadPool &m_Pool;
    ReturnIterationsFunction m_ReturnIterations;
};
