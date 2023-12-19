// TODO: 2x32 perturb is busted, do git diff
// Re-run  profile on current default view
#include <stdio.h>
#include <iostream>

#include "GPU_Render.h"
#include "dbldbl.cuh"
#include "dblflt.cuh"
#include "QuadDouble/gqd_basic.cuh"
#include "QuadFloat/gqf_basic.cuh"

#include "CudaDblflt.h"

#include "GPU_BLAS.h"

#include "HDRFloatComplex.h"
#include "BLA.h"
#include "HDRFloat.h"

#include "GPU_LAReference.h"

#include "GPU_LAInfoDeep.h"
#include "LAReference.h"

#include <type_traits>
#include <stdint.h>
//#include <cuda/pipeline>
//#include <cuda_pipeline.h>

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

constexpr static bool EnableGpu1x32 = Default;
constexpr static bool EnableGpu2x32 = Default;
constexpr static bool EnableGpu4x32 = Default;
constexpr static bool EnableGpu1x64 = Default;
constexpr static bool EnableGpu2x64 = Default;
constexpr static bool EnableGpu4x64 = Default;
constexpr static bool EnableGpuHDRx32 = Default;

constexpr static bool EnableGpu1x32PerturbedScaled = Default;
constexpr static bool EnableGpu2x32PerturbedScaled = Default;
constexpr static bool EnableGpuHDRx32PerturbedScaled = Default;

constexpr static bool EnableGpu1x32PerturbedScaledBLA = Default;
constexpr static bool EnableGpu1x64PerturbedBLA = Default;
constexpr static bool EnableGpuHDRx32PerturbedBLA = Default;
constexpr static bool EnableGpuHDRx64PerturbedBLA = Default;

constexpr static bool EnableGpu1x32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpu1x32PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpu1x32PerturbedLAv2LAO = ForceEnable;
constexpr static bool EnableGpu2x32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpu2x32PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpu2x32PerturbedLAv2LAO = ForceEnable;
constexpr static bool EnableGpu1x64PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpu1x64PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpu1x64PerturbedLAv2LAO = ForceEnable;
constexpr static bool EnableGpuHDRx32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpuHDRx32PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpuHDRx32PerturbedLAv2LAO = ForceEnable;
constexpr static bool EnableGpuHDRx2x32PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpuHDRx2x32PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpuHDRx2x32PerturbedLAv2LAO = ForceEnable;
constexpr static bool EnableGpuHDRx64PerturbedLAv2 = ForceEnable;
constexpr static bool EnableGpuHDRx64PerturbedLAv2PO = ForceEnable;
constexpr static bool EnableGpuHDRx64PerturbedLAv2LAO = ForceEnable;

//#define DEFAULT_KERNEL_LAUNCH_PARAMS nb_blocks, threads_per_block, 0, cudaStreamPerThread
#define DEFAULT_KERNEL_LAUNCH_PARAMS nb_blocks, threads_per_block

__device__
size_t
ConvertLocToIndex(size_t X, size_t Y, size_t OriginalWidth) {
    auto RoundedBlocks = OriginalWidth / GPURenderer::NB_THREADS_W + (OriginalWidth % GPURenderer::NB_THREADS_W != 0);
    auto RoundedWidth = RoundedBlocks * GPURenderer::NB_THREADS_W;
    return Y * RoundedWidth + X;
}

#include "InitStatics.cuh"
#include "BLA.cuh"
#include "Perturb.cuh"
#include "PerturbResultsCollection.cuh"
#include "LowPrecisionKernels.cuh"
#include "BLAKernels.cuh"
#include "ReductionKernels.cuh"
#include "AntialiasingKernel.cuh"
#include "LAKernel.cuh"
#include "DisabledKernels.cuh"
#include "ScaledKernels.cuh"

GPURenderer::GPURenderer() {
    ClearLocals();
}

GPURenderer::~GPURenderer() {
    ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);

    if (m_Stream1Initialized) {
        cudaStreamDestroy(m_Stream1);
    }
}

void GPURenderer::ResetPalettesOnly() {
    if (Pals.local_pal != nullptr) {
        cudaFree(Pals.local_pal);
        Pals.local_pal = nullptr;
    }
}

void GPURenderer::ResetMemory(
    ResetLocals locals,
    ResetPalettes palettes,
    ResetPerturb perturb) {

    if (OutputIterMatrix != nullptr) {
        cudaFree(OutputIterMatrix);
        OutputIterMatrix = nullptr;
    }

    if (OutputReductionResults != nullptr) {
        cudaFree(OutputReductionResults);
        OutputReductionResults = nullptr;
    }

    if (OutputColorMatrix.aa_colors != nullptr) {
        cudaFree(OutputColorMatrix.aa_colors);
        OutputColorMatrix.aa_colors = nullptr;
    }

    if (palettes == ResetPalettes::Yes) {
        ResetPalettesOnly();
    }

    if (perturb == ResetPerturb::Yes) {
        m_PerturbResults.DeleteAll();
    }

    if (locals == ResetLocals::Yes) {
        ClearLocals();
    }
}

void GPURenderer::ClearLocals() {
    // This function assumes memory is freed!
    OutputIterMatrix = nullptr;
    OutputReductionResults = nullptr;
    OutputColorMatrix = {};

    m_Width = 0;
    m_Height = 0;
    local_color_width = 0;
    local_color_height = 0;
    m_Antialiasing = 0;
    m_IterTypeSize = 0;
    w_block = 0;
    h_block = 0;
    w_color_block = 0;
    h_color_block = 0;
    N_cu = 0;
    N_color_cu = 0;
    
    m_Stream1Initialized = false;

    Pals = {};

    m_PerturbResults = {};
}

template<typename IterType>
void GPURenderer::ClearMemory() {
    if (OutputIterMatrix != nullptr) {
        cudaMemset(OutputIterMatrix, 0, N_cu * sizeof(IterType));
    }

    if (OutputReductionResults != nullptr) {
        cudaMemset(OutputReductionResults, 0, sizeof(IterType));
    }

    if (OutputColorMatrix.aa_colors != nullptr) {
        cudaMemset(OutputColorMatrix.aa_colors, 0, N_color_cu * sizeof(Color16));
    }
}

template
void GPURenderer::ClearMemory<uint32_t>();
template
void GPURenderer::ClearMemory<uint64_t>();

template<typename IterType>
uint32_t GPURenderer::InitializeMemory(
    uint32_t antialias_width, // screen width
    uint32_t antialias_height, // screen height
    uint32_t antialiasing,
    const uint16_t* palR,
    const uint16_t* palG,
    const uint16_t* palB,
    uint32_t palIters,
    uint32_t paletteAuxDepth)
{
    if (Pals.palette_aux_depth != paletteAuxDepth) {
        Pals.palette_aux_depth = paletteAuxDepth;
    }

    // Re-do palettes only.
    if ((Pals.cached_hostPalR != palR) ||
        (Pals.cached_hostPalG != palG) ||
        (Pals.cached_hostPalB != palB)) {

        ResetPalettesOnly();

        Pals = Palette(
            nullptr,
            palIters,
            paletteAuxDepth,
            palR,
            palG,
            palB);

        // Palettes:
        cudaError_t err = cudaMallocManaged(
            &Pals.local_pal,
            Pals.local_palIters * sizeof(Color16),
            cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        // TODO the incoming data should be rearranged so we can memcpy
        // Copy in palettes
        for (uint32_t i = 0; i < Pals.local_palIters; i++) {
            Pals.local_pal[i].r = palR[i];
            Pals.local_pal[i].g = palG[i];
            Pals.local_pal[i].b = palB[i];
        }
    }

    if ((m_Width == antialias_width) &&
        (m_Height == antialias_height) &&
        (m_Antialiasing == antialiasing) &&
        (m_IterTypeSize == sizeof(IterType))) {
        return 0;
    }

    //if (w % NB_THREADS_W != 0) {
    //    return FractalSharkError::Error1;
    //}

    //if (h % NB_THREADS_H != 0) {
    //    return FractalSharkError::Error2;
    //}

    if (antialiasing > 4 || antialiasing < 1) {
        return FractalSharkError::Error3;
    }

    if (antialias_width % antialiasing != 0) {
        return FractalSharkError::Error4;
    }

    if (antialias_height % antialiasing != 0) {
        return FractalSharkError::Error5;
    }

    w_block =
        antialias_width / GPURenderer::NB_THREADS_W +
        (antialias_width % GPURenderer::NB_THREADS_W != 0);
    h_block =
        antialias_height / GPURenderer::NB_THREADS_H +
        (antialias_height % GPURenderer::NB_THREADS_H != 0);
    m_Width = antialias_width;
    m_Height = antialias_height;
    m_Antialiasing = antialiasing;
    m_IterTypeSize = sizeof(IterType);
    N_cu = w_block * NB_THREADS_W * h_block * NB_THREADS_H;

    const auto no_antialias_width = antialias_width / antialiasing;
    const auto no_antialias_height = antialias_height / antialiasing;
    w_color_block =
        no_antialias_width / GPURenderer::NB_THREADS_W_AA +
        (no_antialias_width % GPURenderer::NB_THREADS_W_AA != 0);
    h_color_block =
        no_antialias_height / GPURenderer::NB_THREADS_H_AA +
        (no_antialias_height % GPURenderer::NB_THREADS_H_AA != 0);
    local_color_width = no_antialias_width;
    local_color_height = no_antialias_height;
    N_color_cu = w_color_block * NB_THREADS_W_AA * h_color_block * NB_THREADS_H_AA;

    ResetMemory(ResetLocals::No, ResetPalettes::No, ResetPerturb::No);

    {
        IterType* tempiter = nullptr;
        cudaError_t err = cudaMallocManaged(
            &tempiter,
            N_cu * sizeof(IterType),
            cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        OutputIterMatrix = tempiter;
    }

    {
        // Unconditionally allocate uint64_t
        ReductionResults* tempreduction = nullptr;
        cudaError_t err = cudaMallocManaged(
            &tempreduction,
            sizeof(ReductionResults));
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        OutputReductionResults = tempreduction;
    }

    {
        Color16* tempaa = nullptr;

        cudaError_t err = cudaMallocManaged(
            &tempaa,
            N_color_cu * sizeof(Color16),
            cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        OutputColorMatrix.aa_colors = tempaa;
    }

    ClearMemory<IterType>();

    if (m_Stream1Initialized == false) {
        cudaError_t err = cudaDeviceGetStreamPriorityRange(&m_StreamPriorityLow, &m_StreamPriorityHigh);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        // cudaStreamNonBlocking
        err = cudaStreamCreateWithPriority(&m_Stream1, cudaStreamNonBlocking, m_StreamPriorityHigh);
        if (err != cudaSuccess) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return err;
        }

        m_Stream1Initialized = true;
    }

    return 0;
}

template
uint32_t GPURenderer::InitializeMemory<uint32_t>(
    uint32_t antialias_width, // screen width
    uint32_t antialias_height, // screen height
    uint32_t antialiasing,
    const uint16_t* palR,
    const uint16_t* palG,
    const uint16_t* palB,
    uint32_t palIters,
    uint32_t paletteAuxDepth);

template
uint32_t GPURenderer::InitializeMemory<uint64_t>(
    uint32_t antialias_width, // screen width
    uint32_t antialias_height, // screen height
    uint32_t antialiasing,
    const uint16_t* palR,
    const uint16_t* palG,
    const uint16_t* palB,
    uint32_t palIters,
    uint32_t paletteAuxDepth);

template<typename IterType, class T1, class SubType, CalcBad Bad, class T2>
uint32_t GPURenderer::InitializePerturb(
    size_t GenerationNumber1,
    const MattPerturbResults<IterType, T1, Bad>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<IterType, T2, Bad>* Perturb2,
    const LAReference<IterType, T1, SubType>* LaReferenceHost,
    size_t LaGenerationNumber)
{
    if (GenerationNumber1 != m_PerturbResults.GetHostGenerationNumber1() ||
        GenerationNumber2 != m_PerturbResults.GetHostGenerationNumber2() ||
        LaGenerationNumber != m_PerturbResults.GetHostLaGenerationNumber1()) {
        m_PerturbResults.DeleteAll();
    }

    if (GenerationNumber1 != m_PerturbResults.GetHostGenerationNumber1()) {
        auto *CudaResults1 = new MattPerturbSingleResults<IterType, T1, Bad>{
            Perturb1->size,
            Perturb1->PeriodMaybeZero,
            Perturb1->iters
        };

        auto result = CudaResults1->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return result;
        }

        m_PerturbResults.SetPtr1(GenerationNumber1, CudaResults1);
    }

    if (GenerationNumber2 != m_PerturbResults.GetHostGenerationNumber2()) {
        auto* CudaResults2 = new MattPerturbSingleResults<IterType, T2, Bad>{
            Perturb2->size,
            Perturb2->PeriodMaybeZero,
            Perturb2->iters
        };

        auto result = CudaResults2->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return result;
        }

        m_PerturbResults.SetPtr2(GenerationNumber2, CudaResults2);
    }

    if (LaGenerationNumber != m_PerturbResults.GetHostLaGenerationNumber1()) {
        auto* LaReferenceCuda = new GPU_LAReference<IterType, T1, SubType>{ *LaReferenceHost };
        auto result = LaReferenceCuda->CheckValid();
        if (result != 0) {
            ResetMemory(ResetLocals::Yes, ResetPalettes::Yes, ResetPerturb::Yes);
            return result;
        }

        m_PerturbResults.SetLaReference1(LaGenerationNumber, LaReferenceCuda);
    }

    return cudaSuccess;
}

template
uint32_t GPURenderer::InitializePerturb<uint32_t, float, float, CalcBad::Disable, float>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, float, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, float, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, float, float>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint32_t, double, double, CalcBad::Disable, double>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, double, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, double, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, double, double>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, CalcBad::Disable, CudaDblflt<MattDblflt>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, CudaDblflt<MattDblflt>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, CudaDblflt<MattDblflt>, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint32_t, class HDRFloat<float>, float, CalcBad::Disable, HDRFloat<float>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, HDRFloat<float>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, HDRFloat<float>, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, HDRFloat<float>, float>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint32_t, class HDRFloat<double>, double, CalcBad::Disable, HDRFloat<double>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, HDRFloat<double>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, HDRFloat<double>, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, HDRFloat<double>, double>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint32_t, class HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, CalcBad::Disable, HDRFloat<CudaDblflt<MattDblflt>>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* Perturb2,
    const LAReference<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>* LaReferenceHost,
    size_t LaGenerationNumber1);

template
uint32_t GPURenderer::InitializePerturb<uint64_t, float, float, CalcBad::Disable, float>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, float, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, float, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, float, float>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint64_t, double, double, CalcBad::Disable, double>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, double, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, double, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, double, double>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, CalcBad::Disable, CudaDblflt<MattDblflt>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, CudaDblflt<MattDblflt>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, CudaDblflt<MattDblflt>, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint64_t, class HDRFloat<float>, float, CalcBad::Disable, HDRFloat<float>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, HDRFloat<float>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, HDRFloat<float>, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, HDRFloat<float>, float>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint64_t, class HDRFloat<double>, double, CalcBad::Disable, HDRFloat<double>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, HDRFloat<double>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, HDRFloat<double>, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, HDRFloat<double>, double>* LaReferenceHost,
    size_t LaGenerationNumber1);
template
uint32_t GPURenderer::InitializePerturb<uint64_t, class HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, CalcBad::Disable, HDRFloat<CudaDblflt<MattDblflt>>>(
    size_t GenerationNumber1,
    const MattPerturbResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* Perturb1,
    size_t GenerationNumber2,
    const MattPerturbResults<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CalcBad::Disable>* Perturb2,
    const LAReference<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>* LaReferenceHost,
    size_t LaGenerationNumber1);

bool GPURenderer::MemoryInitialized() const {
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

// Not the same as OnlyAA
template<typename IterType>
uint32_t GPURenderer::RenderAsNeeded(
    IterType n_iterations,
    IterType* iter_buffer,
    Color16* color_buffer) {

    uint32_t result = cudaSuccess;

    // TODO
    //result = RunAntialiasing(n_iterations, cudaStreamDefault);
    //if (!result) {
    //    result = ExtractItersAndColors<IterType, false>(iter_buffer, color_buffer);
    //}

    return result;
}

template<typename IterType>
void GPURenderer::RenderAsNeeded(
    uint32_t &result,
    IterType n_iterations,
    IterType* iter_buffer,
    Color16* color_buffer) {

    result = RenderAsNeeded(n_iterations, iter_buffer, color_buffer);
}

template<typename IterType>
uint32_t GPURenderer::RenderCurrent(
    IterType n_iterations,
    IterType* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results) {

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    uint32_t result = cudaSuccess;

    if (iter_buffer == nullptr) {
        result = RunAntialiasing(n_iterations, &m_Stream1);

        if (!result) {
            result = ExtractItersAndColors<IterType, true>(
                iter_buffer,
                color_buffer,
                reduction_results);
        }
    }
    else {
        cudaStream_t stream = 0;
        result = RunAntialiasing(n_iterations, &stream);

        if (!result) {
            result = ExtractItersAndColors<IterType, false>(
                iter_buffer,
                color_buffer,
                reduction_results);
        }
    }

    return result;
}

template uint32_t GPURenderer::RenderCurrent(
    uint32_t n_iterations,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);
template uint32_t GPURenderer::RenderCurrent(
    uint64_t n_iterations,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);

uint32_t GPURenderer::SyncStream(bool altStream) {
    if (altStream) {
        return cudaStreamSynchronize(m_Stream1);
    }
    else {
        return cudaStreamSynchronize(cudaStreamDefault);
    }
}

template<typename IterType, class T>
uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    IterType* iter_buffer,
    Color16* color_buffer,
    T cx,
    T cy,
    T dx,
    T dy,
    IterType n_iterations,
    int iteration_precision)
{
    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    if (algorithm == RenderAlgorithm::Gpu1x64) {
        // all are doubleOnly
        if constexpr (EnableGpu1x64 && std::is_same<T, double>::value) {
            switch (iteration_precision) {
            case 1:
                mandel_1x_double<IterType, 1> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 4:
                mandel_1x_double<IterType, 4> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 8:
                mandel_1x_double<IterType, 8> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 16:
                mandel_1x_double<IterType, 16> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            default:
                break;
            }
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu2x64) {
        if constexpr (EnableGpu2x64 && std::is_same<T, MattDbldbl>::value) {
            dbldbl cx2{ cx.head, cx.tail };
            dbldbl cy2{ cy.head, cy.tail };
            dbldbl dx2{ dx.head, dx.tail };
            dbldbl dy2{ dy.head, dy.tail };

            mandel_2x_double<IterType> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                m_Width, m_Height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu4x64) {
        // qdbl
        if constexpr (EnableGpu4x64 && std::is_same<T, MattQDbldbl>::value) {
            using namespace GQD;
            gqd_real cx2;
            cx2 = make_qd(cx.x, cx.y, cx.z, cx.w);

            gqd_real cy2;
            cy2 = make_qd(cy.x, cy.y, cy.z, cy.w);

            gqd_real dx2;
            dx2 = make_qd(dx.x, dx.y, dx.z, dx.w);

            gqd_real dy2;
            dy2 = make_qd(dy.x, dy.y, dy.z, dy.w);

            mandel_4x_double<IterType> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                m_Width, m_Height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32) {
        if constexpr (EnableGpu1x32 && std::is_same<T, float>::value) {
            // floatOnly
            switch (iteration_precision) {
            case 1:
                mandel_1x_float<IterType, 1> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 4:
                mandel_1x_float<IterType, 4> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 8:
                mandel_1x_float<IterType, 8> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            case 16:
                mandel_1x_float<IterType, 16> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx, cy, dx, dy,
                    n_iterations);
                break;
            default:
                break;
            }
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32) {
        // flt
        if constexpr (EnableGpu2x32 && std::is_same<T, MattDblflt>::value) {
            dblflt cx2{ cx.head, cx.tail };
            dblflt cy2{ cy.head, cy.tail };
            dblflt dx2{ dx.head, dx.tail };
            dblflt dy2{ dy.head, dy.tail };

            switch (iteration_precision) {
            case 1:
                mandel_2x_float<IterType, 1> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx2, cy2, dx2, dy2,
                    n_iterations);
                break;
            case 4:
                mandel_2x_float<IterType, 4> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx2, cy2, dx2, dy2,
                    n_iterations);
                break;
            case 8:
                mandel_2x_float<IterType, 8> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx2, cy2, dx2, dy2,
                    n_iterations);
                break;
            case 16:
                mandel_2x_float<IterType, 16> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    m_Width, m_Height, cx2, cy2, dx2, dy2,
                    n_iterations);
                break;
            default:
                break;
            }
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu4x32) {
        // qflt
        if constexpr (EnableGpu4x32 && std::is_same<T, MattQFltflt>::value) {
            using namespace GQF;
            gqf_real cx2;
            cx2 = make_qf(cx.x, cx.y, cx.z, cx.w);

            gqf_real cy2;
            cy2 = make_qf(cy.x, cy.y, cy.z, cy.w);

            gqf_real dx2;
            dx2 = make_qf(dx.x, dx.y, dx.z, dx.w);

            gqf_real dy2;
            dy2 = make_qf(dy.x, dy.y, dy.z, dy.w);

            mandel_4x_float<IterType> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                m_Width, m_Height, cx2, cy2, dx2, dy2,
                n_iterations);
        }
    }
    else if (algorithm == RenderAlgorithm::GpuHDRx32) {
        if constexpr (EnableGpuHDRx32 && std::is_same<T, HDRFloat<double>>::value) {
            HDRFloat<CudaDblflt<dblflt>> cx2{ cx };
            HDRFloat<CudaDblflt<dblflt>> cy2{ cy };
            HDRFloat<CudaDblflt<dblflt>> dx2{ dx };
            HDRFloat<CudaDblflt<dblflt>> dy2{ dy };

            mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

            switch(iteration_precision) {
                case 1:
                    mandel_hdr_float<IterType, 1> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                        static_cast<IterType*>(OutputIterMatrix),
                        OutputColorMatrix,
                        m_Width, m_Height, cx2, cy2, dx2, dy2,
                        n_iterations);
                    break;
                case 4:
                    mandel_hdr_float<IterType, 4> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                        static_cast<IterType*>(OutputIterMatrix),
                        OutputColorMatrix,
                        m_Width, m_Height, cx2, cy2, dx2, dy2,
                        n_iterations);
                    break;
                case 8:
                    mandel_hdr_float<IterType, 8> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                        static_cast<IterType*>(OutputIterMatrix),
                        OutputColorMatrix,
                        m_Width, m_Height, cx2, cy2, dx2, dy2,
                        n_iterations);
                    break;
                case 16:
                    mandel_hdr_float<IterType, 16> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                        static_cast<IterType*>(OutputIterMatrix),
                        OutputColorMatrix,
                        m_Width, m_Height, cx2, cy2, dx2, dy2,
                        n_iterations);
                    break;
                default:
                    break;
            }
        }
    }
    else {
        return cudaSuccess;
    }

    return RenderAsNeeded(n_iterations, iter_buffer, color_buffer);
}

//////////////////////////////////////////////////
template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    MattDbldbl cx,
    MattDbldbl cy,
    MattDbldbl dx,
    MattDbldbl dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    MattQDbldbl cx,
    MattQDbldbl cy,
    MattQDbldbl dx,
    MattQDbldbl dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    MattDblflt cx,
    MattDblflt cy,
    MattDblflt dx,
    MattDblflt dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    MattQFltflt cx,
    MattQFltflt cy,
    MattQFltflt dx,
    MattQFltflt dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    uint32_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    uint32_t n_iterations,
    int iteration_precision);
//////////////////////////////////////////////////
template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    MattDbldbl cx,
    MattDbldbl cy,
    MattDbldbl dx,
    MattDbldbl dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    MattQDbldbl cx,
    MattQDbldbl cy,
    MattQDbldbl dx,
    MattQDbldbl dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    MattDblflt cx,
    MattDblflt cy,
    MattDblflt dx,
    MattDblflt dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    MattQFltflt cx,
    MattQFltflt cy,
    MattQFltflt dx,
    MattQFltflt dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    uint64_t n_iterations,
    int iteration_precision);

template uint32_t GPURenderer::Render(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    uint64_t n_iterations,
    int iteration_precision);
/////////////////////////////////////////////////////////


template<typename IterType, class T, class SubType, LAv2Mode Mode>
uint32_t GPURenderer::RenderPerturbLAv2(
    RenderAlgorithm algorithm,
    IterType* iter_buffer,
    Color16* color_buffer,
    T cx,
    T cy,
    T dx,
    T dy,
    T centerX,
    T centerY,
    IterType n_iterations)
{
    uint32_t result = cudaSuccess;

    if (!MemoryInitialized()) {
        return cudaSuccess;
    }

    dim3 nb_blocks(w_block, h_block, 1);
    dim3 threads_per_block(NB_THREADS_W, NB_THREADS_H, 1);

    auto* cudaResults = m_PerturbResults.GetPtr1<IterType, T>();
    if (!cudaResults) {
        return FractalSharkError::Error6;
    }

    auto* laReferenceCuda = m_PerturbResults.GetLaReference1<IterType, T, SubType>();
    if (!cudaResults) {
        return FractalSharkError::Error7;
    }

    if ((algorithm == RenderAlgorithm::Gpu1x32PerturbedLAv2) ||
        (algorithm == RenderAlgorithm::Gpu1x32PerturbedLAv2PO) ||
        (algorithm == RenderAlgorithm::Gpu1x32PerturbedLAv2LAO)) {
        if constexpr (
            (EnableGpu1x32PerturbedLAv2 || EnableGpu1x32PerturbedLAv2PO || EnableGpu1x32PerturbedLAv2LAO)
            && std::is_same<float, T>::value) {
            // hdrflt
            mandel_1xHDR_float_perturb_lav2<IterType, float, float, Mode> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    } else if ((algorithm == RenderAlgorithm::Gpu2x32PerturbedLAv2) ||
        (algorithm == RenderAlgorithm::Gpu2x32PerturbedLAv2PO) ||
        (algorithm == RenderAlgorithm::Gpu2x32PerturbedLAv2LAO)) {
        if constexpr (
            (EnableGpu2x32PerturbedLAv2 || EnableGpu2x32PerturbedLAv2PO || EnableGpu2x32PerturbedLAv2LAO)
            && std::is_same<CudaDblflt<MattDblflt>, T>::value) {
            // hdrflt

            CudaDblflt<dblflt> cx2{ cx };
            CudaDblflt<dblflt> cy2{ cy };
            CudaDblflt<dblflt> dx2{ dx };
            CudaDblflt<dblflt> dy2{ dy };

            CudaDblflt<dblflt> centerX2{ centerX };
            CudaDblflt<dblflt> centerY2{ centerY };

            mandel_1xHDR_float_perturb_lav2<IterType, CudaDblflt<dblflt>, CudaDblflt<dblflt>, Mode> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx2, cy2, dx2, dy2,
                centerX2, centerY2,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    } else if ((algorithm == RenderAlgorithm::Gpu1x64PerturbedLAv2) ||
        (algorithm == RenderAlgorithm::Gpu1x64PerturbedLAv2PO) ||
        (algorithm == RenderAlgorithm::Gpu1x64PerturbedLAv2LAO)) {
        if constexpr (
            (EnableGpu1x64PerturbedLAv2 || EnableGpu1x64PerturbedLAv2PO || EnableGpu1x64PerturbedLAv2LAO)
            && std::is_same<double, T>::value) {
            // hdrflt
            mandel_1xHDR_float_perturb_lav2<IterType, double, double, Mode> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    }
    else if ((algorithm == RenderAlgorithm::GpuHDRx32PerturbedLAv2) ||
        (algorithm == RenderAlgorithm::GpuHDRx32PerturbedLAv2PO) ||
        (algorithm == RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO)) {
        if constexpr (
            (EnableGpuHDRx32PerturbedLAv2 || EnableGpuHDRx32PerturbedLAv2PO || EnableGpuHDRx32PerturbedLAv2LAO)
            && std::is_same<HDRFloat<float>, T>::value) {
            mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

            // hdrflt
            mandel_1xHDR_float_perturb_lav2<IterType, HDRFloat<float>, float, Mode> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    } else if ((algorithm == RenderAlgorithm::GpuHDRx64PerturbedLAv2) ||
               (algorithm == RenderAlgorithm::GpuHDRx64PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO)) {
        // hdrdbl
        if constexpr (
            (EnableGpuHDRx64PerturbedLAv2 || EnableGpuHDRx64PerturbedLAv2PO || EnableGpuHDRx64PerturbedLAv2LAO)
            && std::is_same<HDRFloat<double>, T>::value) {
            mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

            mandel_1xHDR_float_perturb_lav2<IterType, HDRFloat<double>, double, Mode> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    } else if ((algorithm == RenderAlgorithm::GpuHDRx2x32PerturbedLAv2) ||
               (algorithm == RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO) ||
               (algorithm == RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO)) {
        if constexpr (
            (EnableGpuHDRx2x32PerturbedLAv2 || EnableGpuHDRx2x32PerturbedLAv2PO || EnableGpuHDRx2x32PerturbedLAv2LAO) &&
            std::is_same<HDRFloat<CudaDblflt<MattDblflt>>, T>::value) {
            HDRFloat<CudaDblflt<dblflt>> cx2{ cx };
            HDRFloat<CudaDblflt<dblflt>> cy2{ cy };
            HDRFloat<CudaDblflt<dblflt>> dx2{ dx };
            HDRFloat<CudaDblflt<dblflt>> dy2{ dy };

            HDRFloat<CudaDblflt<dblflt>> centerX2{ centerX };
            HDRFloat<CudaDblflt<dblflt>> centerY2{ centerY };

            mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

            // hdrflt
            mandel_1xHDR_float_perturb_lav2<IterType, HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>, Mode> << < DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                *cudaResults, *laReferenceCuda,
                m_Width, m_Height, m_Antialiasing, cx2, cy2, dx2, dy2,
                centerX2, centerY2,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    }

    return result;
}

////////////////////////////////////////////////////////

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, float, float, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, float, float, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, float, float, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, double, double, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, double, double, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, double, double, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<float>, float, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t * iter_buffer,
    Color16 * color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<float>, float, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t * iter_buffer,
    Color16 * color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<float>, float, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t * iter_buffer,
    Color16 * color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<double>, double, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<double>, double, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint32_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint32_t, HDRFloat<double>, double, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint32_t n_iterations);

////////////////////////////////////////////////////////

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, float, float, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, float, float, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, float, float, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    float cx,
    float cy,
    float dx,
    float dy,
    float centerX,
    float centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, double, double, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, double, double, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, double, double, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    CudaDblflt<MattDblflt> cx,
    CudaDblflt<MattDblflt> cy,
    CudaDblflt<MattDblflt> dx,
    CudaDblflt<MattDblflt> dy,
    CudaDblflt<MattDblflt> centerX,
    CudaDblflt<MattDblflt> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<float>, float, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<float>, float, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<float>, float, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<CudaDblflt<MattDblflt>> cx,
    HDRFloat<CudaDblflt<MattDblflt>> cy,
    HDRFloat<CudaDblflt<MattDblflt>> dx,
    HDRFloat<CudaDblflt<MattDblflt>> dy,
    HDRFloat<CudaDblflt<MattDblflt>> centerX,
    HDRFloat<CudaDblflt<MattDblflt>> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<double>, double, LAv2Mode::Full>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<double>, double, LAv2Mode::PO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint64_t n_iterations);

template
uint32_t GPURenderer::RenderPerturbLAv2<uint64_t, HDRFloat<double>, double, LAv2Mode::LAO>(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint64_t n_iterations);

////////////////////////////////////////////////////////

template<typename IterType, class T>
uint32_t GPURenderer::RenderPerturbBLAScaled(
    RenderAlgorithm algorithm,
    IterType* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<IterType, T, CalcBad::Enable>* double_perturb,
    const MattPerturbResults<IterType, float, CalcBad::Enable>* float_perturb,
    BLAS<IterType, T, CalcBad::Enable>* blas,
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

    MattPerturbSingleResults<IterType, float, CalcBad::Enable> cudaResults(
        float_perturb->size,
        float_perturb->PeriodMaybeZero,
        float_perturb->iters);

    result = cudaResults.CheckValid();
    if (result != 0) {
        return result;
    }

    MattPerturbSingleResults<IterType, T, CalcBad::Enable> cudaResultsDouble(
        double_perturb->size,
        float_perturb->PeriodMaybeZero,
        double_perturb->iters);

    result = cudaResultsDouble.CheckValid();
    if (result != 0) {
        return result;
    }

    if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaled) {
        if constexpr (EnableGpu1x32PerturbedScaled && std::is_same<T, double>::value) {
            // doubleOnly
            mandel_1x_float_perturb_scaled<IterType, T> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                cudaResults, cudaResultsDouble,
                m_Width, m_Height, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    } else if (algorithm == RenderAlgorithm::GpuHDRx32PerturbedScaled) {
        if constexpr (EnableGpuHDRx32PerturbedScaled && std::is_same<T, HDRFloat<float>>::value) {
            // hdrflt
            mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

            mandel_1x_float_perturb_scaled<IterType, T> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                static_cast<IterType*>(OutputIterMatrix),
                OutputColorMatrix,
                cudaResults, cudaResultsDouble,
                m_Width, m_Height, cx, cy, dx, dy,
                centerX, centerY,
                n_iterations);

            RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu1x32PerturbedScaledBLA) {
        if constexpr (EnableGpu1x32PerturbedScaledBLA && std::is_same<T, double>::value) {
            // doubleOnly
            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, double, BLA<double>, LM2> doubleGpuBlas(blas->m_B);
                result = doubleGpuBlas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

                mandel_1x_float_perturb_scaled_bla<IterType, LM2> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    cudaResults, cudaResultsDouble, doubleGpuBlas,
                    m_Width, m_Height, cx, cy, dx, dy,
                    centerX, centerY,
                    n_iterations);

                RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
                return result;
            };

            LargeSwitch
        }
    }

    return result;
}

//////////////////////////////////////////////////////////////////

template uint32_t GPURenderer::RenderPerturbBLAScaled<uint32_t, double>(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint32_t, double, CalcBad::Enable>* double_perturb,
    const MattPerturbResults<uint32_t, float, CalcBad::Enable>* float_perturb,
    BLAS<uint32_t, double, CalcBad::Enable>* blas,
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
    uint32_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint32_t, HDRFloat<float>, CalcBad::Enable>* double_perturb,
    const MattPerturbResults<uint32_t, float, CalcBad::Enable>* float_perturb,
    BLAS<uint32_t, HDRFloat<float>, CalcBad::Enable>* blas,
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
    uint64_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint64_t, double, CalcBad::Enable>* double_perturb,
    const MattPerturbResults<uint64_t, float, CalcBad::Enable>* float_perturb,
    BLAS<uint64_t, double, CalcBad::Enable>* blas,
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
    uint64_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint64_t, HDRFloat<float>, CalcBad::Enable>* double_perturb,
    const MattPerturbResults<uint64_t, float, CalcBad::Enable>* float_perturb,
    BLAS<uint64_t, HDRFloat<float>, CalcBad::Enable>* blas,
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

template<typename IterType, class T>
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    IterType* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<IterType, T>* perturb,
    BLAS<IterType, T>* blas,
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

    if (algorithm == RenderAlgorithm::GpuHDRx32PerturbedBLA) {
        if constexpr (EnableGpuHDRx32PerturbedBLA && std::is_same<T, HDRFloat<float>>::value) {
            MattPerturbSingleResults<IterType, HDRFloat<float>> cudaResults(
                perturb->size,
                perturb->PeriodMaybeZero,
                perturb->iters);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, HDRFloat<float>, BLA<HDRFloat<float>>, LM2> gpu_blas(blas->m_B);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

                // hdrflt
                mandel_1xHDR_float_perturb_bla<IterType, HDRFloat<float>, LM2> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    cudaResults,
                    gpu_blas,
                    m_Width, m_Height, cx, cy, dx, dy,
                    centerX, centerY,
                    n_iterations);

                RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
                return result;
            };

            LargeSwitch
        }
    }
    else if (algorithm == RenderAlgorithm::GpuHDRx64PerturbedBLA) {
        if constexpr (EnableGpuHDRx64PerturbedBLA && std::is_same<T, HDRFloat<double>>::value) {
            MattPerturbSingleResults<IterType, HDRFloat<double>> cudaResults(
                perturb->size,
                perturb->PeriodMaybeZero,
                perturb->iters);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, HDRFloat<double>, BLA<HDRFloat<double>>, LM2> gpu_blas(blas->m_B);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

                // hdrflt -- that looks like a bug and probably should be hdrdbl
                mandel_1xHDR_float_perturb_bla<IterType, HDRFloat<double>, LM2> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    cudaResults,
                    gpu_blas,
                    m_Width, m_Height, cx, cy, dx, dy,
                    centerX, centerY,
                    n_iterations);

                RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
                return result;
            };

            LargeSwitch
        }
    } else if (algorithm == RenderAlgorithm::Gpu1x64PerturbedBLA) {
        if constexpr (EnableGpu1x64PerturbedBLA && std::is_same<T, double>::value) {
            MattPerturbSingleResults<IterType, double> cudaResults(
                perturb->size,
                perturb->PeriodMaybeZero,
                perturb->iters);

            result = cudaResults.CheckValid();
            if (result != 0) {
                return result;
            }

            auto Run = [&]<int32_t LM2>() -> uint32_t {
                GPU_BLAS<IterType, double, BLA<double>, LM2> gpu_blas(blas->m_B);
                result = gpu_blas.CheckValid();
                if (result != 0) {
                    return result;
                }

                mandel_1xHDR_InitStatics << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > ();

                // doubleOnly
                mandel_1x_double_perturb_bla<IterType, LM2> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
                    static_cast<IterType*>(OutputIterMatrix),
                    OutputColorMatrix,
                    cudaResults,
                    gpu_blas,
                    m_Width, m_Height, cx, cy, dx, dy,
                    centerX, centerY,
                    n_iterations);

                RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
                return result;
            };

            LargeSwitch
        }
    }
    else if (algorithm == RenderAlgorithm::Gpu2x32PerturbedScaled) {
        if constexpr (EnableGpu2x32PerturbedScaled && std::is_same<T, dblflt>::value) {
            //MattPerturbSingleResults<IterType, dblflt> cudaResults(
            //    Perturb->size,
            //    Perturb->PeriodMaybeZero,
            //    Perturb->iters);

            //result = cudaResults.CheckValid();
            //if (result != 0) {
            //    return result;
            //}

            //MattPerturbSingleResults<IterType, double> cudaResultsDouble(
            //    Perturb->size,
            //    Perturb->PeriodMaybeZero,
            //    Perturb->iters);

            //result = cudaResultsDouble.CheckValid();
            //if (result != 0) {
            //    return result;
            //}

            //// doubleOnly
            //mandel_2x_float_perturb_setup << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (cudaResults);

            //mandel_2x_float_perturb_scaled<IterType> << <DEFAULT_KERNEL_LAUNCH_PARAMS >> > (
            //    static_cast<IterType*>(OutputIterMatrix),
            //    OutputColorMatrix,
            //    cudaResults, cudaResultsDouble,
            //    m_Width, m_Height, cx, cy, dx, dy,
            //    centerX, centerY,
            //    n_iterations);

            // RenderAsNeeded(result, n_iterations, iter_buffer, color_buffer);
        }
    }

    return result;
}

//////////////////////////////////////////////////////////
template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint32_t, HDRFloat<float>>* perturb,
    BLAS<uint32_t, HDRFloat<float>>* blas,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint32_t, HDRFloat<double>>* perturb,
    BLAS<uint32_t, HDRFloat<double>>* blas,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);

template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint32_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint32_t, double>* perturb,
    BLAS<uint32_t, double>* blas,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint32_t n_iterations,
    int /*iteration_precision*/);
//////////////////////////////////////////////////////////
template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint64_t, HDRFloat<float>>* perturb,
    BLAS<uint64_t, HDRFloat<float>>* blas,
    HDRFloat<float> cx,
    HDRFloat<float> cy,
    HDRFloat<float> dx,
    HDRFloat<float> dy,
    HDRFloat<float> centerX,
    HDRFloat<float> centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);

template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint64_t, HDRFloat<double>>* perturb,
    BLAS<uint64_t, HDRFloat<double>>* blas,
    HDRFloat<double> cx,
    HDRFloat<double> cy,
    HDRFloat<double> dx,
    HDRFloat<double> dy,
    HDRFloat<double> centerX,
    HDRFloat<double> centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);

template
uint32_t GPURenderer::RenderPerturbBLA(
    RenderAlgorithm algorithm,
    uint64_t* iter_buffer,
    Color16* color_buffer,
    const MattPerturbResults<uint64_t, double>* perturb,
    BLAS<uint64_t, double>* blas,
    double cx,
    double cy,
    double dx,
    double dy,
    double centerX,
    double centerY,
    uint64_t n_iterations,
    int /*iteration_precision*/);
//////////////////////////////////////////////////////////

//// Not the same as RenderAsNeeded
//template<typename IterType>
//__host__
//uint32_t
//GPURenderer::OnlyAA(
//    Color16* color_buffer,
//    IterType n_iterations) {
//
//    auto result = RunAntialiasing(n_iterations, cudaStreamDefault);
//    if (result != cudaSuccess) {
//        return result;
//    }
//
//    result = ExtractItersAndColors<IterType, false>(nullptr, color_buffer);
//    if (result != cudaSuccess) {
//        return result;
//    }
//
//    return cudaSuccess;
//}

//template
//__host__
//uint32_t
//GPURenderer::OnlyAA<uint32_t>(
//    Color16* color_buffer,
//    uint32_t n_iterations);
//
//template
//__host__
//uint32_t
//GPURenderer::OnlyAA(
//    Color16* color_buffer,
//    uint64_t n_iterations);

template<typename IterType>
__host__
uint32_t
GPURenderer::RunAntialiasing(IterType n_iterations, cudaStream_t *stream) {
    dim3 aa_blocks(w_color_block, h_color_block, 1);
    dim3 aa_threads_per_block(NB_THREADS_W_AA, NB_THREADS_H_AA, 1);
    
    switch (m_Antialiasing) {
    case 1:
        antialiasing_kernel<IterType, 1, true> << <aa_blocks, aa_threads_per_block, 0, *stream >> > (
            static_cast<IterType*>(OutputIterMatrix),
            m_Width,
            m_Height,
            OutputColorMatrix,
            Pals,
            local_color_width,
            local_color_height,
            n_iterations);
        break;
    case 2:
        antialiasing_kernel<IterType, 2, true> << <aa_blocks, aa_threads_per_block, 0, *stream >> > (
            static_cast<IterType*>(OutputIterMatrix),
            m_Width,
            m_Height,
            OutputColorMatrix,
            Pals,
            local_color_width,
            local_color_height,
            n_iterations);
        break;
    case 3:
        antialiasing_kernel<IterType, 3, true> << <aa_blocks, aa_threads_per_block, 0, *stream >> > (
            static_cast<IterType*>(OutputIterMatrix),
            m_Width,
            m_Height,
            OutputColorMatrix,
            Pals,
            local_color_width,
            local_color_height,
            n_iterations);
        break;
    case 4:
    default:
        antialiasing_kernel<IterType, 4, true> << <aa_blocks, aa_threads_per_block, 0, *stream >> > (
            static_cast<IterType*>(OutputIterMatrix),
            m_Width,
            m_Height,
            OutputColorMatrix,
            Pals,
            local_color_width,
            local_color_height,
            n_iterations);
        break;
    }

    dim3 max_blocks(16, 16, 1);
    max_kernel<IterType> << <max_blocks, aa_threads_per_block, 0, *stream >> > (
        static_cast<IterType*>(OutputIterMatrix),
        m_Width,
        m_Height,
        OutputReductionResults);
    return cudaSuccess;
}

template<typename IterType, bool Async>
uint32_t GPURenderer::ExtractItersAndColors(
    IterType* iter_buffer,
    Color16 *color_buffer,
    ReductionResults* reduction_results) {

    const size_t ERROR_COLOR = 255;
    cudaError_t result = cudaSuccess;

    if (!Async) {
        result = cudaStreamSynchronize(cudaStreamDefault);
        if (result != cudaSuccess) {
            if (iter_buffer) {
                cudaMemset(iter_buffer, ERROR_COLOR, sizeof(IterType) * m_Width * m_Height);
            }

            if (color_buffer) {
                cudaMemset(color_buffer, ERROR_COLOR, sizeof(Color16) * local_color_width * local_color_height);
            }
            return result;
        }
    }

    if (iter_buffer) {
        if constexpr (Async) {
            result = cudaMemcpyAsync(
                iter_buffer,
                static_cast<IterType*>(OutputIterMatrix),
                sizeof(IterType) * N_cu,
                cudaMemcpyDefault,
                m_Stream1);
            if (result != cudaSuccess) {
                return result;
            }
        } else {
            result = cudaMemcpy(
                iter_buffer,
                static_cast<IterType*>(OutputIterMatrix),
                sizeof(IterType) * N_cu,
                cudaMemcpyDefault);
            if (result != cudaSuccess) {
                return result;
            }
        }
    }

    if (color_buffer) {
        if constexpr (Async) {
            result = cudaMemcpyAsync(
                color_buffer,
                OutputColorMatrix.aa_colors,
                sizeof(Color16) * N_color_cu,
                cudaMemcpyDefault,
                m_Stream1);
            if (result != cudaSuccess) {
                return result;
            }
        } else {
            result = cudaMemcpy(
                color_buffer,
                OutputColorMatrix.aa_colors,
                sizeof(Color16) * N_color_cu,
                cudaMemcpyDefault);
            if (result != cudaSuccess) {
                return result;
            }
        }
    }

    if (reduction_results != nullptr) {
        if constexpr (Async) {
            result = cudaMemcpyAsync(
                reduction_results,
                OutputReductionResults,
                sizeof(ReductionResults),
                cudaMemcpyDefault,
                m_Stream1);
            if (result != cudaSuccess) {
                return result;
            }
        }
        else {
            result = cudaMemcpy(
                reduction_results,
                OutputReductionResults,
                sizeof(ReductionResults),
                cudaMemcpyDefault);
            if (result != cudaSuccess) {
                return result;
            }
        }
    }

    return cudaSuccess;
}

template
uint32_t GPURenderer::ExtractItersAndColors<uint32_t, false>(
    uint32_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);
template
uint32_t GPURenderer::ExtractItersAndColors<uint64_t, false>(
    uint64_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);

template
uint32_t GPURenderer::ExtractItersAndColors<uint32_t, true>(
    uint32_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);
template
uint32_t GPURenderer::ExtractItersAndColors<uint64_t, true>(
    uint64_t* iter_buffer,
    Color16* color_buffer,
    ReductionResults* reduction_results);

const char* GPURenderer::ConvertErrorToString(uint32_t err) {
    auto typeNotExposedOutSideHere = static_cast<cudaError_t>(err);
    return cudaGetErrorString(typeNotExposedOutSideHere);
}