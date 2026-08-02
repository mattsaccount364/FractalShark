#pragma once

#include <cuda_runtime.h>

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelHpSharkReferenceOrbit2.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <sstream>

namespace HpShark {

template <class SharkFloatParams> class Reference2PreparedTables {
public:
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;

private:
    Workspace *m_Descriptor{};
    void *m_Storage{};
    size_t m_StorageBytes{};
    Workspace m_HostDescriptor{};
    uint64_t m_Id{};
    inline static std::atomic<uint64_t> s_NextId{1};

public:
    Reference2PreparedTables(Workspace *descriptor,
                             void *storage,
                             size_t storageBytes,
                             const Workspace &hostDescriptor)
        : m_Descriptor{descriptor}, m_Storage{storage}, m_StorageBytes{storageBytes},
          m_HostDescriptor{hostDescriptor}, m_Id{s_NextId.fetch_add(1, std::memory_order_relaxed)}
    {
    }

    ~Reference2PreparedTables()
    {
        if (m_Descriptor != nullptr)
            cudaFree(m_Descriptor);
        if (m_Storage != nullptr)
            cudaFree(m_Storage);
    }

    Reference2PreparedTables(const Reference2PreparedTables &) = delete;
    Reference2PreparedTables &operator=(const Reference2PreparedTables &) = delete;
    Reference2PreparedTables(Reference2PreparedTables &&) = delete;
    Reference2PreparedTables &operator=(Reference2PreparedTables &&) = delete;

    Workspace *
    GetDeviceDescriptor() const
    {
        return m_Descriptor;
    }

    const Workspace &
    GetHostDescriptor() const
    {
        return m_HostDescriptor;
    }

    size_t
    GetStorageBytes() const
    {
        return m_StorageBytes;
    }

    uint64_t
    GetId() const
    {
        return m_Id;
    }

    Workspace *
    ReleaseDescriptor()
    {
        Workspace *result = m_Descriptor;
        m_Descriptor = nullptr;
        return result;
    }

    void *
    ReleaseStorage()
    {
        void *result = m_Storage;
        m_Storage = nullptr;
        m_StorageBytes = 0;
        return result;
    }
};

namespace Reference2SetupDetail {

inline void
CheckCuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return;

    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error) << " (code "
            << static_cast<int>(error) << ")";
    throw FractalSharkSeriousException(message.str());
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
AllocatePreparedTables(uint32_t actualPrecisionLimbs)
{
    using PreparedTables = Reference2PreparedTables<SharkFloatParams>;
    using Workspace = typename PreparedTables::Workspace;
    constexpr uint32_t StoragePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    constexpr size_t WorkingSpectrumCount = 4u + (SharkFloatParams::EnableNewtonRaphson ? 4u : 0u);
    constexpr size_t ConstantArenaCount = 2u + (SharkFloatParams::EnableNewtonRaphson ? 1u : 0u);
    constexpr size_t LimbCount = SharkFloatParams::EnableNewtonRaphson ? 4u : 2u;
    constexpr size_t WorkspaceAlignment = 16u;

    if (actualPrecisionLimbs <= StoragePrecisionLimbs / 2u ||
        actualPrecisionLimbs > StoragePrecisionLimbs) {
        throw FractalSharkSeriousException(
            "Ref2 actual precision is outside the storage precision bucket");
    }

    const SharkNTT::PlanPrime precisionPlan =
        SharkNTT::BuildPlanPrime2(static_cast<int>(actualPrecisionLimbs));
    if (!precisionPlan.ok)
        throw FractalSharkSeriousException("Unable to build the Ref2 precision plan");

    const auto alignWorkspace = [](size_t value, size_t alignment) {
        return (value + alignment - 1u) & ~(alignment - 1u);
    };
    size_t workspaceBytes = 0;
    const auto addAllocation = [&](size_t count, size_t elementSize, size_t alignment) {
        workspaceBytes = alignWorkspace(workspaceBytes, alignment);
        workspaceBytes += count * elementSize;
    };
    addAllocation(WorkingSpectrumCount, Workspace::MaxFusedN * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(ConstantArenaCount, Workspace::PsiArenaSize * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(LimbCount, Workspace::MaxFusedLimbs * sizeof(int64_t), WorkspaceAlignment);
    addAllocation(2u, Workspace::MaxFusedLimbs * sizeof(uint32_t), WorkspaceAlignment);
    addAllocation(1u, Workspace::MaxFusedLimbs * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(Workspace::MaxCarryPrefixParts,
                  sizeof(HpSharkReference2CarryPrefixDescriptor),
                  WorkspaceAlignment);
    addAllocation(Workspace::CarryPrefixControlCount, sizeof(uint32_t), WorkspaceAlignment);
    addAllocation(1u, Workspace::MaxFusedStages * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(1u, Workspace::MaxFusedStages * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(2u, Workspace::PsiArenaSize * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(2u, Workspace::MaxFusedN * sizeof(uint64_t), WorkspaceAlignment);

    void *workspaceStorage = nullptr;
    Workspace *workspaceGpu = nullptr;
    try {
        CheckCuda(cudaMalloc(&workspaceStorage, workspaceBytes),
                  "cudaMalloc(Reference2 workspace storage)");
        CheckCuda(cudaMemset(workspaceStorage, 0, workspaceBytes),
                  "cudaMemset(Reference2 workspace storage)");

        auto *workspaceBase = static_cast<uint8_t *>(workspaceStorage);
        size_t workspaceOffset = 0;
        const auto allocateWorkspace = [&](size_t count, size_t elementSize, size_t alignment) {
            workspaceOffset = alignWorkspace(workspaceOffset, alignment);
            void *result = workspaceBase + workspaceOffset;
            workspaceOffset += count * elementSize;
            return result;
        };
        const auto allocateSpectrum = [&] {
            return static_cast<uint64_t *>(
                allocateWorkspace(Workspace::MaxFusedN, sizeof(uint64_t), WorkspaceAlignment));
        };
        const auto allocateConstantArena = [&] {
            return static_cast<uint64_t *>(
                allocateWorkspace(Workspace::PsiArenaSize, sizeof(uint64_t), WorkspaceAlignment));
        };
        const auto allocateLimbs = [&] {
            return static_cast<int64_t *>(
                allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(int64_t), WorkspaceAlignment));
        };

        Workspace workspace{};
        workspace.ZReal = allocateSpectrum();
        workspace.ZImag = allocateSpectrum();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            workspace.DzdcReal = allocateSpectrum();
            workspace.DzdcImag = allocateSpectrum();
        }
        workspace.RealOutput = allocateSpectrum();
        workspace.ImagOutput = allocateSpectrum();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            workspace.DzdcRealOutput = allocateSpectrum();
            workspace.DzdcImagOutput = allocateSpectrum();
        }
        workspace.CRealArena = allocateConstantArena();
        workspace.CImagArena = allocateConstantArena();
        if constexpr (SharkFloatParams::EnableNewtonRaphson)
            workspace.OneArena = allocateConstantArena();
        workspace.RealLimbs = allocateLimbs();
        workspace.ImagLimbs = allocateLimbs();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            workspace.DzdcRealLimbs = allocateLimbs();
            workspace.DzdcImagLimbs = allocateLimbs();
        }
        workspace.MagnitudeDigits = static_cast<uint32_t *>(
            allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint32_t), WorkspaceAlignment));
        workspace.Magnitude = static_cast<uint32_t *>(
            allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint32_t), WorkspaceAlignment));
        workspace.CarryPrefixTransforms = static_cast<uint64_t *>(
            allocateWorkspace(Workspace::MaxFusedLimbs, sizeof(uint64_t), WorkspaceAlignment));
        workspace.CarryPrefixDescriptors = static_cast<HpSharkReference2CarryPrefixDescriptor *>(
            allocateWorkspace(Workspace::MaxCarryPrefixParts,
                              sizeof(HpSharkReference2CarryPrefixDescriptor),
                              WorkspaceAlignment));
        workspace.CarryPrefixControl = static_cast<uint32_t *>(
            allocateWorkspace(Workspace::CarryPrefixControlCount, sizeof(uint32_t), WorkspaceAlignment));
        workspace.StageOmegas = static_cast<uint64_t *>(
            allocateWorkspace(Workspace::MaxFusedStages, sizeof(uint64_t), WorkspaceAlignment));
        workspace.StageOmegasInverse = static_cast<uint64_t *>(
            allocateWorkspace(Workspace::MaxFusedStages, sizeof(uint64_t), WorkspaceAlignment));
        workspace.PsiPowersArena = static_cast<uint64_t *>(
            allocateWorkspace(Workspace::PsiArenaSize, sizeof(uint64_t), WorkspaceAlignment));
        workspace.PsiInversePowersArena = static_cast<uint64_t *>(
            allocateWorkspace(Workspace::PsiArenaSize, sizeof(uint64_t), WorkspaceAlignment));
        workspace.ForwardTwiddles = allocateSpectrum();
        workspace.InverseTwiddles = allocateSpectrum();
        workspace.ActualPrecisionLimbs = actualPrecisionLimbs;
        workspace.IgnoredPrecisionBits = (StoragePrecisionLimbs - actualPrecisionLimbs) * 32u;

        for (uint32_t slot = 0; slot < Workspace::PlanCacheEntryCount; ++slot) {
            const uint32_t stages = Workspace::MinFusedStages + slot;
            const uint32_t n = 1u << stages;
            const uint32_t arenaOffset = n - Workspace::MinFusedN;
            workspace.Plans[slot] = {precisionPlan.n32,
                                     precisionPlan.b,
                                     precisionPlan.L,
                                     static_cast<int>(n),
                                     static_cast<int>(stages),
                                     precisionPlan.ok};
            workspace.PlanRoots[slot] = {static_cast<int32_t>(stages),
                                         workspace.StageOmegas,
                                         workspace.StageOmegasInverse,
                                         static_cast<int32_t>(n),
                                         workspace.PsiPowersArena + arenaOffset,
                                         workspace.PsiInversePowersArena + arenaOffset,
                                         0,
                                         workspace.ForwardTwiddles,
                                         workspace.InverseTwiddles,
                                         n - 1u};
            workspace.ConstantSpectra[slot] = {
                workspace.CRealArena + arenaOffset,
                workspace.CImagArena + arenaOffset,
                SharkFloatParams::EnableNewtonRaphson ? workspace.OneArena + arenaOffset : nullptr};
        }

        if (workspaceOffset != workspaceBytes)
            throw FractalSharkSeriousException("Reference2 workspace size does not match its layout");

        CheckCuda(cudaMalloc(&workspaceGpu, sizeof(Workspace)),
                  "cudaMalloc(Reference2 workspace descriptor)");
        CheckCuda(cudaMemcpy(workspaceGpu, &workspace, sizeof(Workspace), cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 workspace descriptor H2D)");
        return std::make_unique<PreparedTables>(
            workspaceGpu, workspaceStorage, workspaceBytes, workspace);
    } catch (...) {
        if (workspaceGpu != nullptr)
            cudaFree(workspaceGpu);
        if (workspaceStorage != nullptr)
            cudaFree(workspaceStorage);
        throw;
    }
}

} // namespace Reference2SetupDetail

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                               const HpSharkFloat<SharkFloatParams> &cReal,
                               const HpSharkFloat<SharkFloatParams> &cImag,
                               uint32_t actualPrecisionLimbs)
{
    auto prepared =
        Reference2SetupDetail::AllocatePreparedTables<SharkFloatParams>(actualPrecisionLimbs);
    HpSharkFloat<SharkFloatParams> *inputsGpu = nullptr;
    uint64_t *tempData = nullptr;
    try {
        constexpr size_t InputCount = SharkFloatParams::EnableNewtonRaphson ? 3u : 2u;
        Reference2SetupDetail::CheckCuda(cudaMalloc(&inputsGpu, InputCount * sizeof(*inputsGpu)),
                                         "cudaMalloc(Reference2 setup inputs)");
        Reference2SetupDetail::CheckCuda(
            cudaMemcpy(inputsGpu, &cReal, sizeof(cReal), cudaMemcpyHostToDevice),
            "cudaMemcpy(Reference2 CReal setup input H2D)");
        Reference2SetupDetail::CheckCuda(
            cudaMemcpy(inputsGpu + 1, &cImag, sizeof(cImag), cudaMemcpyHostToDevice),
            "cudaMemcpy(Reference2 CImag setup input H2D)");
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            HpSharkFloat<SharkFloatParams> one;
            one.template FromHDRFloat<typename SharkFloatParams::SubType>(
                HDRFloat<typename SharkFloatParams::SubType>{typename SharkFloatParams::SubType(1.0)});
            Reference2SetupDetail::CheckCuda(
                cudaMemcpy(inputsGpu + 2, &one, sizeof(one), cudaMemcpyHostToDevice),
                "cudaMemcpy(Reference2 One setup input H2D)");
        }

        constexpr size_t TempBytes = HpShark::AdditionalUInt64Global * sizeof(uint64_t);
        Reference2SetupDetail::CheckCuda(cudaMalloc(&tempData, TempBytes),
                                         "cudaMalloc(Reference2 setup debug scratch)");
        Reference2SetupDetail::CheckCuda(cudaMemset(tempData, 0, TempBytes),
                                         "cudaMemset(Reference2 setup debug scratch)");

        auto *workspace = prepared->GetDeviceDescriptor();
        const auto *cRealGpu = inputsGpu;
        const auto *cImagGpu = inputsGpu + 1;
        const auto *oneGpu = SharkFloatParams::EnableNewtonRaphson ? inputsGpu + 2 : nullptr;
        void *kernelArgs[] = {&workspace, &cRealGpu, &cImagGpu, &oneGpu, &tempData};
        cudaStream_t stream{};
        ComputeHpSharkReference2Setup<SharkFloatParams>(launchParams, stream, kernelArgs);

        Reference2SetupDetail::CheckCuda(cudaFree(tempData), "cudaFree(Reference2 setup debug scratch)");
        tempData = nullptr;
        Reference2SetupDetail::CheckCuda(cudaFree(inputsGpu), "cudaFree(Reference2 setup inputs)");
        inputsGpu = nullptr;
        return prepared;
    } catch (...) {
        if (tempData != nullptr)
            cudaFree(tempData);
        if (inputsGpu != nullptr)
            cudaFree(inputsGpu);
        throw;
    }
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                               const mpf_t cReal,
                               const mpf_t cImag,
                               uint32_t actualPrecisionLimbs)
{
    auto inputReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    inputReal->MpfToHpGpu(
        cReal, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputImag->MpfToHpGpu(
        cImag, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    return PrepareHpSharkReference2Tables<SharkFloatParams>(
        launchParams, *inputReal, *inputImag, actualPrecisionLimbs);
}

} // namespace HpShark
