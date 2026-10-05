// TODO Remove old build tools

// TODO: 2x32 perturb is busted, do git diff
// Re-run  profile on current default view

#include <iostream>
#include <stdio.h>

#include "GPU_Render.h"
#include "PngEncoder.cuh"
#include "QuadDouble/gqd_basic.cuh"
#include "QuadFloat/gqf_basic.cuh"
#include "dbldbl.cuh"
#include "dblflt.cuh"

#include "CudaDblflt.h"

#include "GPU_BLAS.h"

#include "BLA.h"
#include "HDRFloat.h"
#include "HDRFloatComplex.h"

#include "GPU_LAReference.h"

#include "GPU_LAInfoDeep.h"
#include "LAReference.h"

#include <limits>
#include <stdint.h>
#include <type_traits>

namespace {
// Fence queued uploads and kernels before host buffers or a renderer lease can be released.
class PendingRendererWork {
public:
    explicit PendingRendererWork(cudaStream_t stream) : m_Stream(stream) {}
    ~PendingRendererWork()
    {
        if (m_Stream != nullptr) {
            cudaStreamSynchronize(m_Stream);
        }
    }

    uint32_t
    Finish(uint32_t result)
    {
        const auto syncResult = m_Stream != nullptr ? cudaStreamSynchronize(m_Stream) : cudaSuccess;
        m_Stream = nullptr;
        return result == cudaSuccess ? syncResult : result;
    }

private:
    cudaStream_t m_Stream;
};
} // namespace
// #include <cuda/pipeline>
// #include <cuda_pipeline.h>

enum FractalSharkError : int32_t {
    Error1 = 10000,
    Error2,
    Error3,
    Error4,
    Error5,
    Error6,
    Error7,
    Error8,
    Error9,
};

constexpr static bool Default = true;
constexpr static bool ForceEnable = true;

constexpr static bool EnableGpu1x32 = ForceEnable;
constexpr static bool EnableGpu2x32 = Default;
constexpr static bool EnableGpu4x32 = Default;
constexpr static bool EnableGpu1x64 = Default;
constexpr static bool EnableGpu2x64 = Default;
constexpr static bool EnableGpu4x64 = Default;
constexpr static bool EnableGpuHDRx32 = Default;

constexpr static bool EnableGpu1x32PerturbedScaled = Default;
constexpr static bool EnableGpu2x32PerturbedScaled = Default;
constexpr static bool EnableGpuHDRx32PerturbedScaled = Default;

constexpr static bool EnableGpu1x64PerturbedBLA = Default;
constexpr static bool EnableGpuHDRx32PerturbedBLA = Default;
constexpr static bool EnableGpuHDRx64PerturbedBLA = Default;

constexpr static bool EnableGpu1x32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpu2x32PerturbedLAv2 = Default;
constexpr static bool EnableGpu1x64PerturbedLAv2 = Default;
constexpr static bool EnableGpuHDRx32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpuHDRx2x32PerturbedLAv2 = Default;
constexpr static bool EnableGpuHDRx64PerturbedLAv2 = Default;

#define DEFAULT_KERNEL_LAUNCH_PARAMS nb_blocks, threads_per_block, 0, m_ComputeStream

__device__ size_t
ConvertLocToIndex(size_t X, size_t Y, size_t OriginalWidth)
{
    auto RoundedBlocks =
        OriginalWidth / GPURenderer::NB_THREADS_W + (OriginalWidth % GPURenderer::NB_THREADS_W != 0);
    auto RoundedWidth = RoundedBlocks * GPURenderer::NB_THREADS_W;
    return Y * RoundedWidth + X;
}

#include "AntialiasingKernel.cuh"
#include "BLA.cuh"
#include "BLAKernels.cuh"
#include "DisabledKernels.cuh"
#include "LAKernel.cuh"
#include "LowPrecisionKernels.cuh"
#include "Perturb.cuh"
#include "PerturbResultsCollection.cuh"
#include "ReductionKernels.cuh"
#include "ScaledKernels.cuh"

GPURenderer::GPURenderer() { ClearLocals(); }

GPURenderer::~GPURenderer()
{
    ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
}

uint32_t
GPURenderer::TestCudaIsWorking()
{
    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
        return false;

    if (deviceCount <= 0)
        return false;

    // Try to actually select device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess)
        return false;

    // Force a lightweight runtime interaction
    err = cudaFree(nullptr);
    if (err != cudaSuccess)
        return false;

    return true;
}

void
GPURenderer::ResetPalettesOnly()
{
    if (Pals.local_pal != nullptr) {
        cudaFreeAsync(Pals.local_pal, m_ComputeStream);
        Pals.local_pal = nullptr;
    }
}

void
GPURenderer::ResetMemory(ResetLocals locals,
                         ResetPalettes palettes,
                         ResetPerturb perturb,
                         ResetStreams streams)
{

    const cudaStream_t displayStream = m_DisplayStream;
    const cudaStream_t computeStream = m_ComputeStream;

    if (OutputIterMatrix != nullptr) {
        cudaFreeAsync(OutputIterMatrix, m_ComputeStream);
        OutputIterMatrix = nullptr;
    }

    if (OutputReductionResults != nullptr) {
        cudaFreeAsync(OutputReductionResults, m_ComputeStream);
        OutputReductionResults = nullptr;
    }

    if (OutputColorMatrix.aa_colors != nullptr) {
        cudaFreeAsync(OutputColorMatrix.aa_colors, m_ComputeStream);
        OutputColorMatrix.aa_colors = nullptr;
    }

    if (palettes == ResetPalettes::Yes) {
        ResetPalettesOnly();
    }

    if (perturb == ResetPerturb::Yes) {
        m_PerturbResults.DeleteAll();
    }

    if (streams == ResetStreams::Destroy && computeStream != nullptr) {
        cudaStreamSynchronize(computeStream);
    }

    if (streams == ResetStreams::Destroy && displayStream != nullptr) {
        cudaStreamSynchronize(displayStream);
    }

    if (streams == ResetStreams::Destroy && displayStream != nullptr) {
        cudaStreamDestroy(displayStream);
    }

    if (streams == ResetStreams::Destroy && computeStream != nullptr) {
        cudaStreamDestroy(computeStream);
    }

    if (locals == ResetLocals::Yes) {
        ClearLocals();
    }
}

void
GPURenderer::ClearLocals()
{
    // This function assumes memory is freed!
    OutputIterMatrix = nullptr;
    OutputReductionResults = nullptr;
    OutputColorMatrix = {};

    m_Width = 0;
    m_Height = 0;
    m_ColorWidth = 0;
    m_ColorHeight = 0;
    m_Antialiasing = 0;
    m_IterTypeSize = 0;
    w_block = 0;
    h_block = 0;
    m_ColorWidthBlocks = 0;
    m_ColorHeightBlocks = 0;
    N_cu = 0;
    N_color_cu = 0;

    m_ComputeStream = nullptr;
    m_DisplayStream = nullptr;

    Pals = {};

    m_PerturbResults = {};
}

template <typename IterType>
void
GPURenderer::ClearMemory()
{
    if (OutputIterMatrix != nullptr) {
        cudaMemsetAsync(OutputIterMatrix, 0, N_cu * sizeof(IterType), m_ComputeStream);
    }

    if (OutputReductionResults != nullptr) {
        cudaMemsetAsync(OutputReductionResults, 0, sizeof(IterType), m_ComputeStream);
    }

    if (OutputColorMatrix.aa_colors != nullptr) {
        cudaMemsetAsync(OutputColorMatrix.aa_colors, 0, N_color_cu * sizeof(Color16), m_ComputeStream);
    }
}

template void GPURenderer::ClearMemory<uint32_t>();
template void GPURenderer::ClearMemory<uint64_t>();

template <typename IterType>
uint32_t
GPURenderer::InitializeMemory(uint32_t antialiasWidth,  // screen width
                              uint32_t antialiasHeight, // screen height
                              uint32_t antialiasing,
                              const Color16 *palInterleaved,
                              uint32_t palIters,
                              uint32_t paletteAuxDepth,
                              uint64_t paletteRotation,
                              uint64_t maxPossibleIterations,
                              uint64_t paletteGeneration,
                              bool expectedReuse)
{
    if (maxPossibleIterations < 2 || paletteAuxDepth >= 64) {
        return cudaErrorInvalidValue;
    }

    // Ensure compute stream exists before any cudaMallocAsync/cudaFreeAsync calls
    if (m_ComputeStream == nullptr) {
        int streamPriorityLow;
        int streamPriorityHigh;
        cudaError_t err = cudaDeviceGetStreamPriorityRange(&streamPriorityLow, &streamPriorityHigh);
        if (err != cudaSuccess) {
            return err;
        }

        // Compute gets low priority; display gets high priority so
        // progressive frame extraction preempts compute work.
        err = cudaStreamCreateWithPriority(&m_ComputeStream, cudaStreamNonBlocking, streamPriorityLow);
        if (err != cudaSuccess) {
            return err;
        }

        err = cudaStreamCreateWithPriority(&m_DisplayStream, cudaStreamNonBlocking, streamPriorityHigh);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }
    }

    // Re-do palettes if pointer or generation changed.
    if (Pals.cached_hostPalInterleaved != palInterleaved ||
        Pals.cached_paletteGeneration != paletteGeneration) {

        ResetPalettesOnly();

        Pals = Palette(nullptr, palIters, paletteAuxDepth, palInterleaved);

        // Palettes:
        cudaError_t err =
            cudaMallocAsync(&Pals.local_pal, Pals.local_palIters * sizeof(Color16), m_ComputeStream);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }

        // Host data is already in interleaved Color16 format — direct memcpy
        err = cudaMemcpyAsync(Pals.local_pal,
                              palInterleaved,
                              Pals.local_palIters * sizeof(Color16),
                              cudaMemcpyHostToDevice,
                              m_ComputeStream);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }

        Pals.cached_paletteGeneration = paletteGeneration;
    }

    // Refresh scalar coloring settings even when the palette and allocation are reused.
    Pals.palette_aux_depth = paletteAuxDepth;
    Pals.m_PaletteRotation = paletteRotation;
    Pals.m_MaxPossibleIterations = maxPossibleIterations;

    if ((m_Width == antialiasWidth) && (m_Height == antialiasHeight) &&
        (m_Antialiasing == antialiasing) && (m_IterTypeSize == sizeof(IterType)) && expectedReuse) {
        return 0;
    }

    // if (w % NB_THREADS_W != 0) {
    //     return FractalSharkError::Error1;
    // }

    // if (h % NB_THREADS_H != 0) {
    //     return FractalSharkError::Error2;
    // }

    if (antialiasing > 4 || antialiasing < 1) {
        return FractalSharkError::Error3;
    }

    if (antialiasWidth % antialiasing != 0) {
        return FractalSharkError::Error4;
    }

    if (antialiasHeight % antialiasing != 0) {
        return FractalSharkError::Error5;
    }

    w_block =
        antialiasWidth / GPURenderer::NB_THREADS_W + (antialiasWidth % GPURenderer::NB_THREADS_W != 0);
    h_block =
        antialiasHeight / GPURenderer::NB_THREADS_H + (antialiasHeight % GPURenderer::NB_THREADS_H != 0);
    m_Width = antialiasWidth;
    m_Height = antialiasHeight;
    m_Antialiasing = antialiasing;
    m_IterTypeSize = sizeof(IterType);
    N_cu = static_cast<decltype(N_cu)>(w_block) * NB_THREADS_W * h_block * NB_THREADS_H;

    const auto colorWidth = antialiasWidth / antialiasing;
    const auto colorHeight = antialiasHeight / antialiasing;
    m_ColorWidthBlocks =
        colorWidth / GPURenderer::NB_THREADS_W_AA + (colorWidth % GPURenderer::NB_THREADS_W_AA != 0);
    m_ColorHeightBlocks =
        colorHeight / GPURenderer::NB_THREADS_H_AA + (colorHeight % GPURenderer::NB_THREADS_H_AA != 0);
    m_ColorWidth = colorWidth;
    m_ColorHeight = colorHeight;
    N_color_cu = static_cast<decltype(N_color_cu)>(m_ColorWidthBlocks) * NB_THREADS_W_AA *
                 m_ColorHeightBlocks * NB_THREADS_H_AA;

    ResetMemory(ResetLocals::No, ResetPalettes::No, ResetPerturb::Yes, ResetStreams::No);

    {
        IterType *tempiter = nullptr;
        cudaError_t err = cudaMallocAsync(&tempiter, N_cu * sizeof(IterType), m_ComputeStream);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }

        OutputIterMatrix = tempiter;
    }

    {
        // Unconditionally allocate uint64_t
        ReductionResults *tempreduction = nullptr;
        cudaError_t err = cudaMallocAsync(&tempreduction, sizeof(ReductionResults), m_ComputeStream);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }

        OutputReductionResults = tempreduction;
    }

    {
        Color16 *tempaa = nullptr;

        cudaError_t err = cudaMallocAsync(&tempaa, N_color_cu * sizeof(Color16), m_ComputeStream);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return err;
        }

        OutputColorMatrix.aa_colors = tempaa;
    }

    ClearMemory<IterType>();

    return 0;
}

template uint32_t GPURenderer::InitializeMemory<uint32_t>(uint32_t antialiasWidth,
                                                          uint32_t antialiasHeight,
                                                          uint32_t antialiasing,
                                                          const Color16 *palInterleaved,
                                                          uint32_t palIters,
                                                          uint32_t paletteAuxDepth,
                                                          uint64_t paletteRotation,
                                                          uint64_t maxPossibleIterations,
                                                          uint64_t paletteGeneration,
                                                          bool expectedReuse);

template uint32_t GPURenderer::InitializeMemory<uint64_t>(uint32_t antialiasWidth,
                                                          uint32_t antialiasHeight,
                                                          uint32_t antialiasing,
                                                          const Color16 *palInterleaved,
                                                          uint32_t palIters,
                                                          uint32_t paletteAuxDepth,
                                                          uint64_t paletteRotation,
                                                          uint64_t maxPossibleIterations,
                                                          uint64_t paletteGeneration,
                                                          bool expectedReuse);

template <typename IterType, class T1, class SubType, PerturbExtras PExtras, class T2>
uint32_t
GPURenderer::InitializePerturb(size_t GenerationNumber1,
                               const GPUPerturbResults<IterType, T1, PExtras> *Perturb1,
                               size_t GenerationNumber2,
                               const GPUPerturbResults<IterType, T2, PExtras> *Perturb2,
                               const LAReference<IterType, T1, SubType, PExtras> *LaReferenceHost)
{
    bool InstallLA = false;

    if (GenerationNumber1 != m_PerturbResults.GetHostGenerationNumber1() ||
        GenerationNumber2 != m_PerturbResults.GetHostGenerationNumber2()) {
        m_PerturbResults.DeleteAll();
    }

    if (GenerationNumber1 != m_PerturbResults.GetHostGenerationNumber1()) {
        auto *CudaResults1 =
            new GPUPerturbSingleResults<IterType, T1, PExtras>{Perturb1->GetCompressedSize(),
                                                               Perturb1->GetUncompressedSize(),
                                                               Perturb1->GetPeriodMaybeZero(),
                                                               Perturb1->GetOrbitXLow(),
                                                               Perturb1->GetOrbitYLow(),
                                                               Perturb1->GetFullOrbit(),
                                                               m_ComputeStream};

        auto result = CudaResults1->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return result;
        }

        m_PerturbResults.SetPtr1(GenerationNumber1, CudaResults1);

        InstallLA = true;
    }

    if (GenerationNumber2 != m_PerturbResults.GetHostGenerationNumber2()) {
        auto *CudaResults2 =
            new GPUPerturbSingleResults<IterType, T2, PExtras>{Perturb2->GetCompressedSize(),
                                                               Perturb2->GetUncompressedSize(),
                                                               Perturb2->GetPeriodMaybeZero(),
                                                               Perturb2->GetOrbitXLow(),
                                                               Perturb2->GetOrbitYLow(),
                                                               Perturb2->GetFullOrbit(),
                                                               m_ComputeStream};

        auto result = CudaResults2->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return result;
        }

        m_PerturbResults.SetPtr2(GenerationNumber2, CudaResults2);

        InstallLA = true;
    }

    if (InstallLA && LaReferenceHost != nullptr) {
        auto *LaReferenceCuda =
            new GPU_LAReference<IterType, T1, SubType>{*LaReferenceHost, m_ComputeStream};
        auto result = LaReferenceCuda->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes, ResetStreams::Destroy);
            return result;
        }

        m_PerturbResults.SetLaReference1(GenerationNumber1, LaReferenceCuda);
    }

    return cudaSuccess;
}

#define DefineInitializePerturb(IterType, T1, SubType, PExtras, T2)                                     \
    template uint32_t GPURenderer::InitializePerturb<IterType, T1, SubType, PExtras, T2>(               \
        size_t GenerationNumber1,                                                                       \
        const GPUPerturbResults<IterType, T1, PExtras> *Perturb1,                                       \
        size_t GenerationNumber2,                                                                       \
        const GPUPerturbResults<IterType, T2, PExtras> *Perturb2,                                       \
        const LAReference<IterType, T1, SubType, PExtras> *LaReferenceHost);

DefineInitializePerturb(uint32_t, float, float, PerturbExtras::Disable, float);
DefineInitializePerturb(uint32_t, double, double, PerturbExtras::Disable, double);
DefineInitializePerturb(uint32_t,
                        CudaDblflt<MattDblflt>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::Disable,
                        CudaDblflt<MattDblflt>);
DefineInitializePerturb(uint32_t, class HDRFloat<float>, float, PerturbExtras::Disable, HDRFloat<float>);
DefineInitializePerturb(
    uint32_t, class HDRFloat<double>, double, PerturbExtras::Disable, HDRFloat<double>);
DefineInitializePerturb(uint32_t,
                        class HDRFloat<CudaDblflt<MattDblflt>>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::Disable,
                        HDRFloat<CudaDblflt<MattDblflt>>);

DefineInitializePerturb(uint32_t, float, float, PerturbExtras::SimpleCompression, float);
DefineInitializePerturb(uint32_t, double, double, PerturbExtras::SimpleCompression, double);
DefineInitializePerturb(uint32_t,
                        CudaDblflt<MattDblflt>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::SimpleCompression,
                        CudaDblflt<MattDblflt>);
DefineInitializePerturb(
    uint32_t, class HDRFloat<float>, float, PerturbExtras::SimpleCompression, HDRFloat<float>);
DefineInitializePerturb(
    uint32_t, class HDRFloat<double>, double, PerturbExtras::SimpleCompression, HDRFloat<double>);
DefineInitializePerturb(uint32_t,
                        class HDRFloat<CudaDblflt<MattDblflt>>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::SimpleCompression,
                        HDRFloat<CudaDblflt<MattDblflt>>);

DefineInitializePerturb(uint64_t, float, float, PerturbExtras::Disable, float);
DefineInitializePerturb(uint64_t, double, double, PerturbExtras::Disable, double);
DefineInitializePerturb(uint64_t,
                        CudaDblflt<MattDblflt>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::Disable,
                        CudaDblflt<MattDblflt>);
DefineInitializePerturb(uint64_t, class HDRFloat<float>, float, PerturbExtras::Disable, HDRFloat<float>);
DefineInitializePerturb(
    uint64_t, class HDRFloat<double>, double, PerturbExtras::Disable, HDRFloat<double>);
DefineInitializePerturb(uint64_t,
                        class HDRFloat<CudaDblflt<MattDblflt>>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::Disable,
                        HDRFloat<CudaDblflt<MattDblflt>>);

DefineInitializePerturb(uint64_t, float, float, PerturbExtras::SimpleCompression, float);
DefineInitializePerturb(uint64_t, double, double, PerturbExtras::SimpleCompression, double);
DefineInitializePerturb(uint64_t,
                        CudaDblflt<MattDblflt>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::SimpleCompression,
                        CudaDblflt<MattDblflt>);
DefineInitializePerturb(
    uint64_t, class HDRFloat<float>, float, PerturbExtras::SimpleCompression, HDRFloat<float>);
DefineInitializePerturb(
    uint64_t, class HDRFloat<double>, double, PerturbExtras::SimpleCompression, HDRFloat<double>);
DefineInitializePerturb(uint64_t,
                        class HDRFloat<CudaDblflt<MattDblflt>>,
                        CudaDblflt<MattDblflt>,
                        PerturbExtras::SimpleCompression,
                        HDRFloat<CudaDblflt<MattDblflt>>);

bool
GPURenderer::MemoryInitialized() const
{
    if (OutputIterMatrix == nullptr) {
        return false;
    }

    if (OutputReductionResults == nullptr) {
        return false;
    }

    if (OutputColorMatrix.aa_colors == nullptr) {
        return false;
    }

    return true;
}

template <typename IterType>
uint32_t
GPURenderer::RenderCurrent(IterType numIterations,
                           IterType *iterBuffer,
                           Color16 *colorBuffer,
                           ReductionResults *reductionResults,
                           bool progressive)
{

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    cudaStream_t stream = progressive ? m_DisplayStream : m_ComputeStream;

    uint32_t result = RunAntialiasing(numIterations, stream, FractalShark::ColoringMode::PaletteLookup);

    if (!result) {
        result = ExtractItersAndColors<IterType>(iterBuffer, colorBuffer, reductionResults, stream);
    }

    return result;
}

template uint32_t GPURenderer::RenderCurrent(uint32_t n_iterations,
                                             uint32_t *iter_buffer,
                                             Color16 *color_buffer,
                                             ReductionResults *reduction_results,
                                             bool progressive);
template uint32_t GPURenderer::RenderCurrent(uint64_t n_iterations,
                                             uint64_t *iter_buffer,
                                             Color16 *color_buffer,
                                             ReductionResults *reduction_results,
                                             bool progressive);

template <typename IterType>
uint32_t
GPURenderer::UploadHostIterations(const IterType *hostIters, size_t rowStrideElements)
{
    if (!MemoryInitialized() || hostIters == nullptr || m_Width == 0 || m_Height == 0 ||
        m_IterTypeSize != sizeof(IterType) || rowStrideElements < m_Width ||
        rowStrideElements > std::numeric_limits<size_t>::max() / sizeof(IterType) / m_Height) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpy2DAsync(OutputIterMatrix,
                             static_cast<size_t>(w_block) * NB_THREADS_W * sizeof(IterType),
                             hostIters,
                             rowStrideElements * sizeof(IterType),
                             static_cast<size_t>(m_Width) * sizeof(IterType),
                             m_Height,
                             cudaMemcpyHostToDevice,
                             m_ComputeStream);
}

template <typename IterType>
uint32_t
GPURenderer::RecolorFromHostIterations(const IterType *hostIters,
                                       size_t rowStrideElements,
                                       IterType numIterations,
                                       FractalShark::ColoringMode coloringMode,
                                       Color16 *hostColors,
                                       size_t hostColorCapacity)
{
    PendingRendererWork pendingWork(m_ComputeStream);
    const size_t colorCount = static_cast<size_t>(m_ColorWidth) * m_ColorHeight;
    if (hostColors == nullptr || hostColorCapacity < colorCount ||
        colorCount > std::numeric_limits<size_t>::max() / sizeof(Color16) || numIterations == 0 ||
        (coloringMode != FractalShark::ColoringMode::PaletteLookup &&
         coloringMode != FractalShark::ColoringMode::BasicGrayscale)) {
        return pendingWork.Finish(cudaErrorInvalidValue);
    }
    uint32_t result = UploadHostIterations(hostIters, rowStrideElements);
    if (result == cudaSuccess) {
        result = RunAntialiasing(numIterations, m_ComputeStream, coloringMode);
    }
    if (result == cudaSuccess) {
        result = cudaMemcpyAsync(hostColors,
                                 OutputColorMatrix.aa_colors,
                                 colorCount * sizeof(Color16),
                                 cudaMemcpyDeviceToHost,
                                 m_ComputeStream);
    }
    return pendingWork.Finish(result);
}

template uint32_t GPURenderer::RecolorFromHostIterations<uint32_t>(
    const uint32_t *, size_t, uint32_t, FractalShark::ColoringMode, Color16 *, size_t);
template uint32_t GPURenderer::RecolorFromHostIterations<uint64_t>(
    const uint64_t *, size_t, uint64_t, FractalShark::ColoringMode, Color16 *, size_t);

template <typename IterType>
uint32_t
GPURenderer::EncodePng(const IterType *hostIters,
                       size_t rowStrideElements,
                       IterType numIterations,
                       std::vector<unsigned char> &pngBytes)
{
    PendingRendererWork pendingWork(m_ComputeStream);
    pngBytes.clear();
    // Save palettes are snapshot-owned; their host address must not outlive the snapshot as a cache key.
    Pals.cached_hostPalInterleaved = nullptr;
    uint32_t result =
        numIterations == 0 ? cudaErrorInvalidValue : UploadHostIterations(hostIters, rowStrideElements);
    if (result == cudaSuccess) {
        result =
            RunAntialiasing(numIterations, m_ComputeStream, FractalShark::ColoringMode::PaletteLookup);
    }
    if (result == cudaSuccess) {
        if (!m_PngEncoder) {
            m_PngEncoder = std::make_unique<FractalShark::Png::GpuPngEncoder>();
        }
    }
    if (result == cudaSuccess) {
        result = m_PngEncoder->Encode(OutputColorMatrix.aa_colors,
                                      m_ColorWidth,
                                      m_ColorHeight,
                                      static_cast<size_t>(m_ColorWidth) * sizeof(Color16),
                                      m_ComputeStream,
                                      pngBytes);
    }
    // The snapshot and palette may be released as soon as this method returns, even on failure.
    return pendingWork.Finish(result);
}

template uint32_t GPURenderer::EncodePng<uint32_t>(const uint32_t *hostIters,
                                                   size_t rowStrideElements,
                                                   uint32_t numIterations,
                                                   std::vector<unsigned char> &pngBytes);
template uint32_t GPURenderer::EncodePng<uint64_t>(const uint64_t *hostIters,
                                                   size_t rowStrideElements,
                                                   uint64_t numIterations,
                                                   std::vector<unsigned char> &pngBytes);

uint32_t
GPURenderer::SyncComputeStream()
{
    return cudaStreamSynchronize(m_ComputeStream);
}

uint32_t
GPURenderer::SyncDisplayStream()
{
    return cudaStreamSynchronize(m_DisplayStream);
}

uint32_t
GPURenderer::QueryComputeStream()
{
    return cudaStreamQuery(m_ComputeStream);
}

static void CUDART_CB
ComputeDoneCallback(void *userData)
{
    auto *renderer = static_cast<GPURenderer *>(userData);
    renderer->SignalComputeDone();
}

uint32_t
GPURenderer::EnqueueComputeDoneCallback()
{
    return cudaLaunchHostFunc(m_ComputeStream, ::ComputeDoneCallback, this);
}

template <typename IterType, class T>
uint32_t
GPURenderer::Render(
    RenderAlgorithm algorithm, T cx, T cy, T dx, T dy, IterType n_iterations, int iteration_precision)
{
    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithmEnum::Gpu1x64) {
        // all are doubleOnly
        if constexpr (EnableGpu1x64 && std::is_same<T, double>::value) {
            switch (iteration_precision) {
                case 1:
                    mandel_1x_double<IterType, 1>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 4:
                    mandel_1x_double<IterType, 4>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 8:
                    mandel_1x_double<IterType, 8>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 16:
                    mandel_1x_double<IterType, 16>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                default:
                    break;
            }
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu2x64) {
        if constexpr (EnableGpu2x64 && std::is_same<T, MattDbldbl>::value) {
            dbldbl cx2{cx.head, cx.tail};
            dbldbl cy2{cy.head, cy.tail};
            dbldbl dx2{dx.head, dx.tail};
            dbldbl dy2{dy.head, dy.tail};

            mandel_2x_double<IterType>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   m_Width,
                                                   m_Height,
                                                   cx2,
                                                   cy2,
                                                   dx2,
                                                   dy2,
                                                   n_iterations);
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu4x64) {
        // qdbl
        if constexpr (EnableGpu4x64 && std::is_same<T, MattQDbldbl>::value) {
            GQD::gqd_real cx2;
            cx2 = GQD::make_qd(cx.x, cx.y, cx.z, cx.w);

            GQD::gqd_real cy2;
            cy2 = GQD::make_qd(cy.x, cy.y, cy.z, cy.w);

            GQD::gqd_real dx2;
            dx2 = GQD::make_qd(dx.x, dx.y, dx.z, dx.w);

            GQD::gqd_real dy2;
            dy2 = GQD::make_qd(dy.x, dy.y, dy.z, dy.w);

            mandel_4x_double<IterType>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   m_Width,
                                                   m_Height,
                                                   cx2,
                                                   cy2,
                                                   dx2,
                                                   dy2,
                                                   n_iterations);
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu1x32) {
        if constexpr (EnableGpu1x32 && std::is_same<T, float>::value) {
            // floatOnly
            switch (iteration_precision) {
                case 1:
                    mandel_1x_float<IterType, 1>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 4:
                    mandel_1x_float<IterType, 4>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 8:
                    mandel_1x_float<IterType, 8>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                case 16:
                    mandel_1x_float<IterType, 16>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx,
                                                           cy,
                                                           dx,
                                                           dy,
                                                           n_iterations);
                    break;
                default:
                    break;
            }
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu2x32) {
        // flt
        if constexpr (EnableGpu2x32 && std::is_same<T, MattDblflt>::value) {
            dblflt cx2{cx.head, cx.tail};
            dblflt cy2{cy.head, cy.tail};
            dblflt dx2{dx.head, dx.tail};
            dblflt dy2{dy.head, dy.tail};

            switch (iteration_precision) {
                case 1:
                    mandel_2x_float<IterType, 1>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 4:
                    mandel_2x_float<IterType, 4>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 8:
                    mandel_2x_float<IterType, 8>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 16:
                    mandel_2x_float<IterType, 16>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                default:
                    break;
            }
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu4x32) {
        // qflt
        if constexpr (EnableGpu4x32 && std::is_same<T, MattQFltflt>::value) {
            GQF::gqf_real cx2;
            cx2 = GQF::make_qf(cx.x, cx.y, cx.z, cx.w);

            GQF::gqf_real cy2;
            cy2 = GQF::make_qf(cy.x, cy.y, cy.z, cy.w);

            GQF::gqf_real dx2;
            dx2 = GQF::make_qf(dx.x, dx.y, dx.z, dx.w);

            GQF::gqf_real dy2;
            dy2 = GQF::make_qf(dy.x, dy.y, dy.z, dy.w);

            mandel_4x_float<IterType>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   m_Width,
                                                   m_Height,
                                                   cx2,
                                                   cy2,
                                                   dx2,
                                                   dy2,
                                                   n_iterations);
        }
    } else if (algorithm == RenderAlgorithmEnum::GpuHDRx32) {
        if constexpr (EnableGpuHDRx32 && std::is_same<T, HDRFloat<double>>::value) {
            HDRFloat<CudaDblflt<dblflt>> cx2{cx};
            HDRFloat<CudaDblflt<dblflt>> cy2{cy};
            HDRFloat<CudaDblflt<dblflt>> dx2{dx};
            HDRFloat<CudaDblflt<dblflt>> dy2{dy};

            switch (iteration_precision) {
                case 1:
                    mandel_hdr_float<IterType, 1>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 4:
                    mandel_hdr_float<IterType, 4>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 8:
                    mandel_hdr_float<IterType, 8>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                case 16:
                    mandel_hdr_float<IterType, 16>
                        <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                           OutputColorMatrix,
                                                           m_Width,
                                                           m_Height,
                                                           cx2,
                                                           cy2,
                                                           dx2,
                                                           dy2,
                                                           n_iterations);
                    break;
                default:
                    break;
            }
        }
    }

    return cudaSuccess;
}

//////////////////////////////////////////////////
template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      double cx,
                                      double cy,
                                      double dx,
                                      double dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      float cx,
                                      float cy,
                                      float dx,
                                      float dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattDbldbl cx,
                                      MattDbldbl cy,
                                      MattDbldbl dx,
                                      MattDbldbl dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattQDbldbl cx,
                                      MattQDbldbl cy,
                                      MattQDbldbl dx,
                                      MattQDbldbl dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattDblflt cx,
                                      MattDblflt cy,
                                      MattDblflt dx,
                                      MattDblflt dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattQFltflt cx,
                                      MattQFltflt cy,
                                      MattQFltflt dx,
                                      MattQFltflt dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      CudaDblflt<MattDblflt> cx,
                                      CudaDblflt<MattDblflt> cy,
                                      CudaDblflt<MattDblflt> dx,
                                      CudaDblflt<MattDblflt> dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      HDRFloat<double> cx,
                                      HDRFloat<double> cy,
                                      HDRFloat<double> dx,
                                      HDRFloat<double> dy,
                                      uint32_t n_iterations,
                                      int iteration_precision);
//////////////////////////////////////////////////
template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      double cx,
                                      double cy,
                                      double dx,
                                      double dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      float cx,
                                      float cy,
                                      float dx,
                                      float dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattDbldbl cx,
                                      MattDbldbl cy,
                                      MattDbldbl dx,
                                      MattDbldbl dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattQDbldbl cx,
                                      MattQDbldbl cy,
                                      MattQDbldbl dx,
                                      MattQDbldbl dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattDblflt cx,
                                      MattDblflt cy,
                                      MattDblflt dx,
                                      MattDblflt dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      MattQFltflt cx,
                                      MattQFltflt cy,
                                      MattQFltflt dx,
                                      MattQFltflt dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      CudaDblflt<MattDblflt> cx,
                                      CudaDblflt<MattDblflt> cy,
                                      CudaDblflt<MattDblflt> dx,
                                      CudaDblflt<MattDblflt> dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);

template uint32_t GPURenderer::Render(RenderAlgorithm algorithm,
                                      HDRFloat<double> cx,
                                      HDRFloat<double> cy,
                                      HDRFloat<double> dx,
                                      HDRFloat<double> dy,
                                      uint64_t n_iterations,
                                      int iteration_precision);
/////////////////////////////////////////////////////////

template <typename IterType, class T, class SubType, LAv2Mode Mode, PerturbExtras PExtras>
uint32_t
GPURenderer::RenderPerturbLAv2(
    RenderAlgorithm algorithm, T cx, T cy, T dx, T dy, T centerX, T centerY, IterType n_iterations)
{
    uint32_t result = cudaSuccess;

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    auto *cudaResults = m_PerturbResults.GetPtr1<IterType, T, PExtras>();
    if (!cudaResults) {
        return FractalSharkError::Error6;
    }

    auto *laReferenceCuda = m_PerturbResults.GetLaReference1<IterType, T, SubType>();
    if (!cudaResults) {
        return FractalSharkError::Error7;
    }

    GPU_LAReference<IterType, T, SubType> local_la_copy{
        laReferenceCuda != nullptr ? *laReferenceCuda : GPU_LAReference<IterType, T, SubType>{}};

    if ((algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedLAv2) ||
        (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO) ||
        (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO) ||
        (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2) ||
        (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO) ||
        (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpu1x32PerturbedLAv2 && std::is_same<float, T>::value) {

            mandel_1xHDR_float_perturb_lav2<IterType, float, float, Mode>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   *cudaResults,
                                                   local_la_copy,
                                                   m_Width,
                                                   m_Height,
                                                   m_Antialiasing,
                                                   cx,
                                                   cy,
                                                   dx,
                                                   dy,
                                                   centerX,
                                                   centerY,
                                                   n_iterations);
        }
    } else if ((algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2) ||
               (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO) ||
               (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2) ||
               (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpu2x32PerturbedLAv2 && std::is_same<CudaDblflt<MattDblflt>, T>::value) {

            CudaDblflt<dblflt> cx2{cx};
            CudaDblflt<dblflt> cy2{cy};
            CudaDblflt<dblflt> dx2{dx};
            CudaDblflt<dblflt> dy2{dy};

            CudaDblflt<dblflt> centerX2{centerX};
            CudaDblflt<dblflt> centerY2{centerY};

            mandel_1xHDR_float_perturb_lav2<IterType,
                                            CudaDblflt<dblflt>,
                                            CudaDblflt<dblflt>,
                                            Mode,
                                            PExtras><<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(

                static_cast<IterType *>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults,
                local_la_copy,
                m_Width,
                m_Height,
                m_Antialiasing,
                cx2,
                cy2,
                dx2,
                dy2,
                centerX2,
                centerY2,
                n_iterations);
        }
    } else if ((algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedLAv2) ||
               (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO) ||
               (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2) ||
               (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpu1x64PerturbedLAv2 && std::is_same<double, T>::value) {

            mandel_1xHDR_float_perturb_lav2<IterType, double, double, Mode, PExtras>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(

                    static_cast<IterType *>(OutputIterMatrix),
                    OutputColorMatrix,
                    *cudaResults,
                    local_la_copy,
                    m_Width,
                    m_Height,
                    m_Antialiasing,
                    cx,
                    cy,
                    dx,
                    dy,
                    centerX,
                    centerY,
                    n_iterations);
        }
    } else if ((algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpuHDRx32PerturbedLAv2 && std::is_same<HDRFloat<float>, T>::value) {

            mandel_1xHDR_float_perturb_lav2<IterType, HDRFloat<float>, float, Mode, PExtras>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(

                    static_cast<IterType *>(OutputIterMatrix),
                    OutputColorMatrix,
                    *cudaResults,
                    local_la_copy,
                    m_Width,
                    m_Height,
                    m_Antialiasing,
                    cx,
                    cy,
                    dx,
                    dy,
                    centerX,
                    centerY,
                    n_iterations);
        }
    } else if ((algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpuHDRx64PerturbedLAv2 && std::is_same<HDRFloat<double>, T>::value) {

            mandel_1xHDR_float_perturb_lav2<IterType, HDRFloat<double>, double, Mode, PExtras>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(

                    static_cast<IterType *>(OutputIterMatrix),
                    OutputColorMatrix,
                    *cudaResults,
                    local_la_copy,
                    m_Width,
                    m_Height,
                    m_Antialiasing,
                    cx,
                    cy,
                    dx,
                    dy,
                    centerX,
                    centerY,
                    n_iterations);
        }
    } else if ((algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO) ||
               (algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO)) {

        if constexpr (EnableGpuHDRx2x32PerturbedLAv2 &&
                      std::is_same<HDRFloat<CudaDblflt<MattDblflt>>, T>::value) {

            HDRFloat<CudaDblflt<dblflt>> cx2{cx};
            HDRFloat<CudaDblflt<dblflt>> cy2{cy};
            HDRFloat<CudaDblflt<dblflt>> dx2{dx};
            HDRFloat<CudaDblflt<dblflt>> dy2{dy};

            HDRFloat<CudaDblflt<dblflt>> centerX2{centerX};
            HDRFloat<CudaDblflt<dblflt>> centerY2{centerY};

            mandel_1xHDR_float_perturb_lav2<IterType,
                                            HDRFloat<CudaDblflt<dblflt>>,
                                            CudaDblflt<dblflt>,
                                            Mode,
                                            PExtras><<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(

                static_cast<IterType *>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults,
                local_la_copy,
                m_Width,
                m_Height,
                m_Antialiasing,
                cx2,
                cy2,
                dx2,
                dy2,
                centerX2,
                centerY2,
                n_iterations);
        }
    }

    return result;
}

////////////////////////////////////////////////////////

#define InitializeRenderPerturbLAv2(IterType, T, SubType, Mode, PExtras)                                \
    template uint32_t GPURenderer::RenderPerturbLAv2<IterType, T, SubType, Mode, PExtras>(              \
        RenderAlgorithm algorithm,                                                                      \
        T cx,                                                                                           \
        T cy,                                                                                           \
        T dx,                                                                                           \
        T dy,                                                                                           \
        T centerX,                                                                                      \
        T centerY,                                                                                      \
        IterType n_iterations);

InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(
    uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(
    uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(
    uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint32_t, HDRFloat<float>, float, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, HDRFloat<float>, float, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, HDRFloat<float>, float, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint32_t, HDRFloat<double>, double, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, HDRFloat<double>, double, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t, HDRFloat<double>, double, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(
    uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(
    uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(
    uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint64_t, HDRFloat<float>, float, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, HDRFloat<float>, float, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, HDRFloat<float>, float, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint64_t, HDRFloat<double>, double, LAv2Mode::Full, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, HDRFloat<double>, double, LAv2Mode::PO, PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t, HDRFloat<double>, double, LAv2Mode::LAO, PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::Disable);
InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::Disable);

InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t, float, float, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t, double, double, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint32_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<float>, float, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<float>, float, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<float>, float, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<double>, double, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<double>, double, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint32_t, HDRFloat<double>, double, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint32_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t, float, float, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t, double, double, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint64_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t,
                            CudaDblflt<MattDblflt>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<float>, float, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<float>, float, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<float>, float, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<double>, double, LAv2Mode::Full, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<double>, double, LAv2Mode::PO, PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(
    uint64_t, HDRFloat<double>, double, LAv2Mode::LAO, PerturbExtras::SimpleCompression);

InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::Full,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::PO,
                            PerturbExtras::SimpleCompression);
InitializeRenderPerturbLAv2(uint64_t,
                            HDRFloat<CudaDblflt<MattDblflt>>,
                            CudaDblflt<MattDblflt>,
                            LAv2Mode::LAO,
                            PerturbExtras::SimpleCompression);

template <typename IterType, class T>
uint32_t
GPURenderer::RenderPerturbBLAScaled(
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
    int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    GPUPerturbSingleResults<IterType, float, PerturbExtras::Bad> cudaResults(
        float_perturb->GetCompressedSize(),
        float_perturb->GetUncompressedSize(),
        float_perturb->GetPeriodMaybeZero(),
        float_perturb->GetOrbitXLow(),
        float_perturb->GetOrbitYLow(),
        float_perturb->GetFullOrbit(),
        m_ComputeStream);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    GPUPerturbSingleResults<IterType, T, PerturbExtras::Bad> cudaResultsDouble(
        double_perturb->GetCompressedSize(),
        double_perturb->GetUncompressedSize(),
        float_perturb->GetPeriodMaybeZero(),
        double_perturb->GetOrbitXLow(),
        double_perturb->GetOrbitYLow(),
        double_perturb->GetFullOrbit(),
        m_ComputeStream);

    result = cudaResultsDouble.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithmEnum::Gpu1x32PerturbedScaled) {
        if constexpr (EnableGpu1x32PerturbedScaled && std::is_same<T, double>::value) {
            // doubleOnly
            mandel_1x_float_perturb_scaled<IterType, T>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   cudaResults,
                                                   cudaResultsDouble,
                                                   m_Width,
                                                   m_Height,
                                                   cx,
                                                   cy,
                                                   dx,
                                                   dy,
                                                   centerX,
                                                   centerY,
                                                   n_iterations);
        }
    } else if (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedScaled) {
        if constexpr (EnableGpuHDRx32PerturbedScaled && std::is_same<T, HDRFloat<float>>::value) {
            mandel_1x_float_perturb_scaled<IterType, T>
                <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                   OutputColorMatrix,
                                                   cudaResults,
                                                   cudaResultsDouble,
                                                   m_Width,
                                                   m_Height,
                                                   cx,
                                                   cy,
                                                   dx,
                                                   dy,
                                                   centerX,
                                                   centerY,
                                                   n_iterations);
        }
    }

    return result;
}

//////////////////////////////////////////////////////////////////

template uint32_t GPURenderer::RenderPerturbBLAScaled<uint32_t, double>(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint32_t, double, PerturbExtras::Bad> *double_perturb,
    const GPUPerturbResults<uint32_t, float, PerturbExtras::Bad> *float_perturb,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/
);

template uint32_t GPURenderer::RenderPerturbBLAScaled<uint32_t, HDRFloat<float>>(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint32_t, HDRFloat<float>, PerturbExtras::Bad> *double_perturb,
    const GPUPerturbResults<uint32_t, float, PerturbExtras::Bad> *float_perturb,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/
);

//////////////////////////////////////////////////////////////////

template uint32_t GPURenderer::RenderPerturbBLAScaled<uint64_t, double>(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint64_t, double, PerturbExtras::Bad> *double_perturb,
    const GPUPerturbResults<uint64_t, float, PerturbExtras::Bad> *float_perturb,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/
);

template uint32_t GPURenderer::RenderPerturbBLAScaled<uint64_t, HDRFloat<float>>(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint64_t, HDRFloat<float>, PerturbExtras::Bad> *double_perturb,
    const GPUPerturbResults<uint64_t, float, PerturbExtras::Bad> *float_perturb,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/
);

//////////////////////////////////////////////////////////////////

template <typename IterType, class T>
uint32_t
GPURenderer::RenderPerturbBLA(RenderAlgorithm algorithm,
                              const GPUPerturbResults<IterType, T, PerturbExtras::Disable> *perturb,
                              BLAS<IterType, T> *blas,
                              T cx,
                              T cy,
                              T dx,
                              T dy,
                              T centerX,
                              T centerY,
                              IterType n_iterations,
                              int /*iteration_precision*/)
{
    uint32_t result = cudaSuccess;

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithmEnum::GpuHDRx32PerturbedBLA) {
        if constexpr (EnableGpuHDRx32PerturbedBLA && std::is_same<T, HDRFloat<float>>::value) {
            GPUPerturbSingleResults<IterType, HDRFloat<float>, PerturbExtras::Disable> cudaResults(
                perturb->GetCompressedSize(),
                perturb->GetUncompressedSize(),
                perturb->GetPeriodMaybeZero(),
                perturb->GetOrbitXLow(),
                perturb->GetOrbitYLow(),
                perturb->GetFullOrbit(),
                m_ComputeStream);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, HDRFloat<float>, BLA<HDRFloat<float>>, LM2> gpu_blas(blas->m_B,
                                                                                        m_ComputeStream);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_float_perturb_bla<IterType, HDRFloat<float>, LM2>
                    <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                       OutputColorMatrix,
                                                       cudaResults,
                                                       gpu_blas,
                                                       m_Width,
                                                       m_Height,
                                                       cx,
                                                       cy,
                                                       dx,
                                                       dy,
                                                       centerX,
                                                       centerY,
                                                       n_iterations);
                return result;
            };

            LargeSwitch
        }
    } else if (algorithm == RenderAlgorithmEnum::GpuHDRx64PerturbedBLA) {
        if constexpr (EnableGpuHDRx64PerturbedBLA && std::is_same<T, HDRFloat<double>>::value) {
            GPUPerturbSingleResults<IterType, HDRFloat<double>, PerturbExtras::Disable> cudaResults(
                perturb->GetCompressedSize(),
                perturb->GetUncompressedSize(),
                perturb->GetPeriodMaybeZero(),
                perturb->GetOrbitXLow(),
                perturb->GetOrbitYLow(),
                perturb->GetFullOrbit(),
                m_ComputeStream);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, HDRFloat<double>, BLA<HDRFloat<double>>, LM2> gpu_blas(
                    blas->m_B, m_ComputeStream);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_float_perturb_bla<IterType, HDRFloat<double>, LM2>
                    <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                       OutputColorMatrix,
                                                       cudaResults,
                                                       gpu_blas,
                                                       m_Width,
                                                       m_Height,
                                                       cx,
                                                       cy,
                                                       dx,
                                                       dy,
                                                       centerX,
                                                       centerY,
                                                       n_iterations);
                return result;
            };

            LargeSwitch
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu1x64PerturbedBLA) {
        if constexpr (EnableGpu1x64PerturbedBLA && std::is_same<T, double>::value) {
            GPUPerturbSingleResults<IterType, double, PerturbExtras::Disable> cudaResults(
                perturb->GetCompressedSize(),
                perturb->GetUncompressedSize(),
                perturb->GetPeriodMaybeZero(),
                perturb->GetOrbitXLow(),
                perturb->GetOrbitYLow(),
                perturb->GetFullOrbit(),
                m_ComputeStream);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, double, BLA<double>, LM2> gpu_blas(blas->m_B, m_ComputeStream);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                // doubleOnly
                mandel_1x_double_perturb_bla<IterType, LM2>
                    <<<DEFAULT_KERNEL_LAUNCH_PARAMS>>>(static_cast<IterType *>(OutputIterMatrix),
                                                       OutputColorMatrix,
                                                       cudaResults,
                                                       gpu_blas,
                                                       m_Width,
                                                       m_Height,
                                                       cx,
                                                       cy,
                                                       dx,
                                                       dy,
                                                       centerX,
                                                       centerY,
                                                       n_iterations);
                return result;
            };

            LargeSwitch
        }
    } else if (algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedScaled) {
        if constexpr (EnableGpu2x32PerturbedScaled && std::is_same<T, dblflt>::value) {
            // GPUPerturbSingleResults<IterType, dblflt> cudaResults(
            //     Perturb->GetCountOrbitEntries(),
            //     Perturb->GetPeriodMaybeZero(),
            //     Perturb->GetFullOrbit());

            // result = cudaResults.CheckValid();
            // if (result != 0) {
            //     return result;
            // }

            // GPUPerturbSingleResults<IterType, double> cudaResultsDouble(
            //     Perturb->GetCountOrbitEntries(),
            //     Perturb->GetPeriodMaybeZero(),
            //     Perturb->GetFullOrbit());

            // result = cudaResultsDouble.CheckValid();
            // if (result != 0) {
            //     return result;
            // }

            //// doubleOnly
            // mandel_2x_float_perturb_setup << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (cudaResults);

            // mandel_2x_float_perturb_scaled<IterType> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
            //     static_cast<IterType*>(OutputIterMatrix),
            //     OutputColorMatrix,
            //     cudaResults, cudaResultsDouble,
            //     m_Width, m_Height, cx, cy, dx, dy,
            //     centerX, centerY,
            //     n_iterations);
        }
    }

    return result;
}

//////////////////////////////////////////////////////////
template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> *perturb,
    BLAS<uint32_t, HDRFloat<float>> *blas,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint32_t, HDRFloat<double>, PerturbExtras::Disable> *perturb,
    BLAS<uint32_t, HDRFloat<double>> *blas,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint32_t, double, PerturbExtras::Disable> *perturb,
    BLAS<uint32_t, double> *blas,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);
//////////////////////////////////////////////////////////
template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint64_t, HDRFloat<float>, PerturbExtras::Disable> *perturb,
    BLAS<uint64_t, HDRFloat<float>> *blas,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint64_t, HDRFloat<double>, PerturbExtras::Disable> *perturb,
    BLAS<uint64_t, HDRFloat<double>> *blas,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);

template uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    const GPUPerturbResults<uint64_t, double, PerturbExtras::Disable> *perturb,
    BLAS<uint64_t, double> *blas,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);
//////////////////////////////////////////////////////////

template <typename IterType>
__host__ uint32_t
GPURenderer::RunAntialiasing(IterType numIterations,
                             cudaStream_t stream,
                             FractalShark::ColoringMode coloringMode)
{
    if (numIterations == 0) {
        return cudaErrorInvalidValue;
    }
    dim3 aaBlocks(m_ColorWidthBlocks, m_ColorHeightBlocks, 1);
    dim3 aaThreadsPerBlock(NB_THREADS_W_AA, NB_THREADS_H_AA, 1);

    switch (m_Antialiasing) {
        case 1:
            antialiasing_kernel<IterType, 1, true>
                <<<aaBlocks, aaThreadsPerBlock, 0, stream>>>(static_cast<IterType *>(OutputIterMatrix),
                                                             m_Width,
                                                             m_Height,
                                                             OutputColorMatrix,
                                                             Pals,
                                                             m_ColorWidth,
                                                             m_ColorHeight,
                                                             numIterations,
                                                             coloringMode);
            break;
        case 2:
            antialiasing_kernel<IterType, 2, true>
                <<<aaBlocks, aaThreadsPerBlock, 0, stream>>>(static_cast<IterType *>(OutputIterMatrix),
                                                             m_Width,
                                                             m_Height,
                                                             OutputColorMatrix,
                                                             Pals,
                                                             m_ColorWidth,
                                                             m_ColorHeight,
                                                             numIterations,
                                                             coloringMode);
            break;
        case 3:
            antialiasing_kernel<IterType, 3, true>
                <<<aaBlocks, aaThreadsPerBlock, 0, stream>>>(static_cast<IterType *>(OutputIterMatrix),
                                                             m_Width,
                                                             m_Height,
                                                             OutputColorMatrix,
                                                             Pals,
                                                             m_ColorWidth,
                                                             m_ColorHeight,
                                                             numIterations,
                                                             coloringMode);
            break;
        case 4:
        default:
            antialiasing_kernel<IterType, 4, true>
                <<<aaBlocks, aaThreadsPerBlock, 0, stream>>>(static_cast<IterType *>(OutputIterMatrix),
                                                             m_Width,
                                                             m_Height,
                                                             OutputColorMatrix,
                                                             Pals,
                                                             m_ColorWidth,
                                                             m_ColorHeight,
                                                             numIterations,
                                                             coloringMode);
            break;
    }

    const auto launchResult = cudaGetLastError();
    if (launchResult != cudaSuccess) {
        return launchResult;
    }
    // Reset before launching any reduction block; block-local barriers cannot order this globally.
    auto resetResult = cudaMemsetAsync(OutputReductionResults, 0, sizeof(ReductionResults), stream);
    if (resetResult != cudaSuccess) {
        return resetResult;
    }
    resetResult = cudaMemsetAsync(&OutputReductionResults->Min, 0xff, sizeof(uint64_t), stream);
    if (resetResult != cudaSuccess) {
        return resetResult;
    }
    dim3 maxBlocks(16, 16, 1);
    max_kernel<IterType><<<maxBlocks, aaThreadsPerBlock, 0, stream>>>(
        static_cast<IterType *>(OutputIterMatrix), m_Width, m_Height, OutputReductionResults);
    return cudaGetLastError();
}

template <typename IterType>
uint32_t
GPURenderer::ExtractItersAndColors(IterType *iter_buffer,
                                   Color16 *color_buffer,
                                   ReductionResults *reduction_results,
                                   cudaStream_t stream)
{

    cudaError_t result = cudaSuccess;

    if (iter_buffer) {
        result = cudaMemcpyAsync(iter_buffer,
                                 static_cast<IterType *>(OutputIterMatrix),
                                 sizeof(IterType) * N_cu,
                                 cudaMemcpyDefault,
                                 stream);
        if (result != cudaSuccess) {
            return result;
        }
    }

    if (color_buffer) {
        result = cudaMemcpyAsync(color_buffer,
                                 OutputColorMatrix.aa_colors,
                                 sizeof(Color16) * N_color_cu,
                                 cudaMemcpyDefault,
                                 stream);
        if (result != cudaSuccess) {
            return result;
        }
    }

    if (reduction_results != nullptr) {
        result = cudaMemcpyAsync(reduction_results,
                                 OutputReductionResults,
                                 sizeof(ReductionResults),
                                 cudaMemcpyDefault,
                                 stream);
        if (result != cudaSuccess) {
            return result;
        }
    }

    return cudaSuccess;
}

template uint32_t GPURenderer::ExtractItersAndColors<uint32_t>(uint32_t *iter_buffer,
                                                               Color16 *color_buffer,
                                                               ReductionResults *reduction_results,
                                                               cudaStream_t stream);
template uint32_t GPURenderer::ExtractItersAndColors<uint64_t>(uint64_t *iter_buffer,
                                                               Color16 *color_buffer,
                                                               ReductionResults *reduction_results,
                                                               cudaStream_t stream);

const char *
GPURenderer::ConvertErrorToString(uint32_t err)
{
    auto typeNotExposedOutSideHere = static_cast<cudaError_t>(err);
    return cudaGetErrorString(typeNotExposedOutSideHere);
}
