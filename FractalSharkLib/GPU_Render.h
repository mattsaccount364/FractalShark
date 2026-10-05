#pragma once
//
// Created by dany on 22/05/19.
//

#ifndef GPGPU_RENDER_GPU_HPP
#define GPGPU_RENDER_GPU_HPP

#include "BLA.h"
#include "BLAS.h"
#include "LAstep.h"

#include "GPU_Types.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace FractalShark::Png {
class GpuPngEncoder;
}

// This is the main class that does the rendering on the GPU
class GPURenderer {
public:
    GPURenderer();
    ~GPURenderer();

    static uint32_t TestCudaIsWorking();

    template <typename IterType, class T>
    uint32_t Render(RenderAlgorithm algorithm,
                    T cx,
                    T cy,
                    T dx,
                    T dy,
                    IterType n_iterations,
                    int iteration_precision);

    template <typename IterType, class T>
    uint32_t RenderPerturbBLAScaled(
        RenderAlgorithm algorithm,
        const GPUPerturbResults<IterType, T, PerturbExtras::Bad> *double_perturb,
        const GPUPerturbResults<IterType, float, PerturbExtras::Bad> *float_perturb,
        T cx,
        T cy,
        T dx,
        T dy,
        T centerX,
        T centerY,
        IterType n_iterations,
        int iteration_precision);

    template <typename IterType>
    uint32_t RenderPerturbBLA(
        RenderAlgorithm algorithm,
        const GPUPerturbResults<IterType, MattDblflt, PerturbExtras::Disable> *results,
        BLAS<IterType, MattDblflt> *blas,
        MattDblflt cx,
        MattDblflt cy,
        MattDblflt dx,
        MattDblflt dy,
        MattDblflt centerX,
        MattDblflt centerY,
        IterType n_iterations,
        int iteration_precision);

    template <typename IterType, class T>
    uint32_t RenderPerturbBLA(RenderAlgorithm algorithm,
                              const GPUPerturbResults<IterType, T, PerturbExtras::Disable> *results,
                              BLAS<IterType, T> *blas,
                              T cx,
                              T cy,
                              T dx,
                              T dy,
                              T centerX,
                              T centerY,
                              IterType n_iterations,
                              int iteration_precision);

    template <typename IterType, class T, class SubType, LAv2Mode Mode, PerturbExtras PExtras>
    uint32_t RenderPerturbLAv2(
        RenderAlgorithm algorithm, T cx, T cy, T dx, T dy, T centerX, T centerY, IterType n_iterations);

    // Side effect is this initializes CUDA the first time it's run
    template <typename IterType>
    uint32_t InitializeMemory(uint32_t w,            // original width * antialiasing
                              uint32_t h,            // original height * antialiasing
                              uint32_t antialiasing, // w and h are ech scaled up by this amt
                              const Color16 *palInterleaved,
                              uint32_t palIters,
                              uint32_t paletteAuxDepth,
                              uint64_t paletteRotation,
                              uint64_t maxPossibleIterations,
                              uint64_t paletteGeneration,
                              bool expectedReuse);

    template <typename IterType, class T1, class SubType, PerturbExtras PExtras, class T2>
    uint32_t InitializePerturb(size_t GenerationNumber1,
                               const GPUPerturbResults<IterType, T1, PExtras> *Perturb1,
                               size_t GenerationNumber2,
                               const GPUPerturbResults<IterType, T2, PExtras> *Perturb2,
                               const LAReference<IterType, T1, SubType, PExtras> *LaReferenceHost);

    template <typename IterType> void ClearMemory();

    static const char *ConvertErrorToString(uint32_t err);

    // Match in Fractal.cpp
    static const int32_t NB_THREADS_W = 16; // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H = 8;

    static const int32_t NB_THREADS_W_AA = 16; // W=16, H=8 previously seemed OK
    static const int32_t NB_THREADS_H_AA = 8;

public:
    template <typename IterType>
    uint32_t RenderCurrent(IterType numIterations,
                           IterType *iterBuffer,
                           Color16 *colorBuffer,
                           ReductionResults *reductionResults,
                           bool progressive);

    // Synchronous recoloring of the authoritative host iteration buffer; counts remain unchanged.
    template <typename IterType>
    uint32_t RecolorFromHostIterations(const IterType *hostIters,
                                       size_t rowStrideElements,
                                       IterType numIterations,
                                       FractalShark::ColoringMode coloringMode,
                                       Color16 *hostColors,
                                       size_t hostColorCapacity);

    uint32_t SyncComputeStream();

    // The caller must drain rendering before uploading a save snapshot. Encoding is synchronous.
    template <typename IterType>
    uint32_t EncodePng(const IterType *hostIters,
                       size_t rowStrideElements,
                       IterType numIterations,
                       std::vector<unsigned char> &pngBytes);
    uint32_t SyncDisplayStream();
    uint32_t QueryComputeStream();
    uint32_t EnqueueComputeDoneCallback();

    void
    ResetComputeDoneFlag()
    {
        m_ComputeDoneFlag.store(false, std::memory_order_release);
    }

    bool
    IsComputeDone() const
    {
        return m_ComputeDoneFlag.load(std::memory_order_acquire);
    }

    void
    SignalComputeDone()
    {
        m_ComputeDoneFlag.store(true, std::memory_order_release);
        if (m_ComputeDoneMutex && m_ComputeDoneCV) {
            std::lock_guard lk(*m_ComputeDoneMutex);
            m_ComputeDoneCV->notify_all();
        }
    }

    void
    SetComputeDoneNotification(std::mutex *mutex, std::condition_variable *cv)
    {
        m_ComputeDoneMutex = mutex;
        m_ComputeDoneCV = cv;
    }

    uint32_t
    GetWidth() const
    {
        return m_Width;
    }
    uint32_t
    GetHeight() const
    {
        return m_Height;
    }

private:
    bool MemoryInitialized() const;
    void ResetPalettesOnly();

    enum class ResetLocals { Yes, No };
    enum class ResetPalettes { Yes, No };

    enum class ResetPerturb { Yes, No };

    enum class ResetStreams { No, Destroy };

    void ResetMemory(ResetLocals locals,
                     ResetPalettes palettes,
                     ResetPerturb perturb,
                     ResetStreams streams);
    void ClearLocals();

    template <typename IterType>
    uint32_t RunAntialiasing(IterType numIterations,
                             cudaStream_t stream,
                             FractalShark::ColoringMode coloringMode);

    template <typename IterType>
    uint32_t UploadHostIterations(const IterType *hostIters, size_t rowStrideElements);

    template <typename IterType>
    uint32_t ExtractItersAndColors(IterType *iter_buffer,
                                   Color16 *color_buffer,
                                   ReductionResults *reduction_results,
                                   cudaStream_t stream);

    void *OutputIterMatrix;
    ReductionResults *OutputReductionResults;
    AntialiasedColors OutputColorMatrix;

    Palette Pals;

    uint32_t m_Width;
    uint32_t m_Height;
    uint32_t m_ColorWidth;
    uint32_t m_ColorHeight;
    uint32_t m_Antialiasing;
    uint32_t m_IterTypeSize;
    uint32_t w_block;
    uint32_t h_block;
    uint32_t m_ColorWidthBlocks;
    uint32_t m_ColorHeightBlocks;
    size_t N_cu;
    size_t N_color_cu;

    cudaStream_t m_ComputeStream;
    cudaStream_t m_DisplayStream;

    std::atomic<bool> m_ComputeDoneFlag{false};
    std::mutex *m_ComputeDoneMutex{nullptr};
    std::condition_variable *m_ComputeDoneCV{nullptr};

    PerturbResultsCollection m_PerturbResults;
    std::unique_ptr<FractalShark::Png::GpuPngEncoder> m_PngEncoder;
};

#endif // GPGPU_RENDER_GPU_HPP
