#pragma once

#include <cuda_runtime.h>

#include "Exceptions.h"
#include "HpSharkFloat.h"
#include "KernelHpSharkReferenceOrbit.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <sstream>

namespace HpShark {

template <class SharkFloatParams> class ReferencePreparedTables {
public:
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;

private:
    Workspace *m_Descriptor{};
    void *m_Storage{};
    size_t m_StorageBytes{};
    Workspace m_HostDescriptor{};
    uint64_t m_Id{};
    inline static std::atomic<uint64_t> s_NextId{1};

public:
    ReferencePreparedTables(Workspace *descriptor,
                             void *storage,
                             size_t storageBytes,
                             const Workspace &hostDescriptor)
        : m_Descriptor{descriptor}, m_Storage{storage}, m_StorageBytes{storageBytes},
          m_HostDescriptor{hostDescriptor}, m_Id{s_NextId.fetch_add(1, std::memory_order_relaxed)}
    {
    }

    ~ReferencePreparedTables()
    {
        if (m_Descriptor != nullptr)
            cudaFree(m_Descriptor);
        if (m_Storage != nullptr)
            cudaFree(m_Storage);
    }

    ReferencePreparedTables(const ReferencePreparedTables &) = delete;
    ReferencePreparedTables &operator=(const ReferencePreparedTables &) = delete;
    ReferencePreparedTables(ReferencePreparedTables &&) = delete;
    ReferencePreparedTables &operator=(ReferencePreparedTables &&) = delete;

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

    void
    UpdateHostDescriptor(const Workspace &descriptor)
    {
        m_HostDescriptor = descriptor;
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

namespace ReferenceSetupDetail {

template <class SharkFloatParams>
void
ValidateStageRange(uint32_t minFusedStages, uint32_t maxFusedStages)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    if (minFusedStages < Workspace::MinFusedStages || maxFusedStages > Workspace::MaxFusedStages ||
        minFusedStages > maxFusedStages) {
        throw FractalSharkSeriousException(
            "Reference fused stage range is outside the workspace limits");
    }
}

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
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
AllocatePreparedTables(uint32_t actualPrecisionLimbs, uint32_t minFusedStages, uint32_t maxFusedStages)
{
    using PreparedTables = ReferencePreparedTables<SharkFloatParams>;
    using Workspace = typename PreparedTables::Workspace;
    constexpr uint32_t StoragePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    constexpr size_t WorkingSpectrumCount = 4u + (SharkFloatParams::EnableNewtonRaphson ? 4u : 0u);
    constexpr size_t LimbCount = SharkFloatParams::EnableNewtonRaphson ? 4u : 2u;
    constexpr size_t WorkspaceAlignment = 16u;

    ValidateStageRange<SharkFloatParams>(minFusedStages, maxFusedStages);
    const uint32_t activeMinFusedN = 1u << minFusedStages;
    const uint32_t activeMaxFusedN = 1u << maxFusedStages;
    const uint32_t activeMaxFusedLimbs = (activeMaxFusedN * 16u) / 32u + 4u;
    const uint32_t activeMaxCarryPrefixParts = (activeMaxFusedLimbs + 31u) / 32u;
    const uint32_t activePlanCacheEntryCount = maxFusedStages - minFusedStages + 1u;

    if (actualPrecisionLimbs <= StoragePrecisionLimbs / 2u ||
        actualPrecisionLimbs > StoragePrecisionLimbs) {
        throw FractalSharkSeriousException(
            "Reference actual precision is outside the storage precision bucket");
    }

    const SharkNTT::Plan precisionPlan =
        SharkNTT::BuildPlan(static_cast<int>(actualPrecisionLimbs));
    if (!precisionPlan.ok)
        throw FractalSharkSeriousException("Unable to build the reference precision plan");

    const auto alignWorkspace = [](size_t value, size_t alignment) {
        return (value + alignment - 1u) & ~(alignment - 1u);
    };
    size_t workspaceBytes = 0;
    const auto addAllocation = [&](size_t count, size_t elementSize, size_t alignment) {
        workspaceBytes = alignWorkspace(workspaceBytes, alignment);
        workspaceBytes += count * elementSize;
    };
    // Keep this allocation sequence in exact lockstep with the Workspace pointer assignments below.
    // Any added, removed, or reordered workspace field must be changed in both places.
    addAllocation(WorkingSpectrumCount, activeMaxFusedN * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(LimbCount, activeMaxFusedLimbs * sizeof(int64_t), WorkspaceAlignment);
    addAllocation(2u, activeMaxFusedLimbs * sizeof(uint32_t), WorkspaceAlignment);
    addAllocation(1u, maxFusedStages * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(1u, maxFusedStages * sizeof(uint64_t), WorkspaceAlignment);
    addAllocation(2u, activeMaxFusedN * sizeof(uint64_t), WorkspaceAlignment);

    void *workspaceStorage = nullptr;
    Workspace *workspaceGpu = nullptr;
    try {
        CheckCuda(cudaMalloc(&workspaceStorage, workspaceBytes),
                  "cudaMalloc(Reference workspace storage)");
        CheckCuda(cudaMemset(workspaceStorage, 0, workspaceBytes),
                  "cudaMemset(Reference workspace storage)");

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
                allocateWorkspace(activeMaxFusedN, sizeof(uint64_t), WorkspaceAlignment));
        };
        const auto allocateLimbs = [&] {
            return static_cast<int64_t *>(
                allocateWorkspace(activeMaxFusedLimbs, sizeof(int64_t), WorkspaceAlignment));
        };

        // Keep this assignment sequence in exact lockstep with the workspaceBytes sequence above.
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
        workspace.RealLimbs = allocateLimbs();
        workspace.ImagLimbs = allocateLimbs();
        if constexpr (SharkFloatParams::EnableNewtonRaphson) {
            workspace.DzdcRealLimbs = allocateLimbs();
            workspace.DzdcImagLimbs = allocateLimbs();
        }
        workspace.MagnitudeDigits = static_cast<uint32_t *>(
            allocateWorkspace(activeMaxFusedLimbs, sizeof(uint32_t), WorkspaceAlignment));
        workspace.Magnitude = static_cast<uint32_t *>(
            allocateWorkspace(activeMaxFusedLimbs, sizeof(uint32_t), WorkspaceAlignment));
        workspace.StageOmegas = static_cast<uint64_t *>(
            allocateWorkspace(maxFusedStages, sizeof(uint64_t), WorkspaceAlignment));
        workspace.StageOmegasInverse = static_cast<uint64_t *>(
            allocateWorkspace(maxFusedStages, sizeof(uint64_t), WorkspaceAlignment));
        workspace.ForwardTwiddles = allocateSpectrum();
        workspace.InverseTwiddles = allocateSpectrum();
        workspace.ActualPrecisionLimbs = actualPrecisionLimbs;
        workspace.IgnoredPrecisionBits = (StoragePrecisionLimbs - actualPrecisionLimbs) * 32u;
        workspace.ActiveMinFusedN = activeMinFusedN;
        workspace.ActiveMaxFusedN = activeMaxFusedN;
        workspace.ActiveMinFusedStages = minFusedStages;
        workspace.ActiveMaxFusedStages = maxFusedStages;
        workspace.ActiveMaxFusedLimbs = activeMaxFusedLimbs;
        workspace.ActiveMaxCarryPrefixParts = activeMaxCarryPrefixParts;
        workspace.ActivePlanCacheEntryCount = activePlanCacheEntryCount;
        workspace.GeneratedStages = 0u;

        workspace.Plans[0] = {precisionPlan.n32,
                              precisionPlan.b,
                              precisionPlan.L,
                              static_cast<int>(Workspace::MinFusedN),
                              static_cast<int>(Workspace::MinFusedStages),
                              precisionPlan.ok};

        for (uint32_t stages = minFusedStages; stages <= maxFusedStages; ++stages) {
            const uint32_t slot = stages - Workspace::MinFusedStages;
            const uint32_t n = 1u << stages;
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
                                         nullptr,
                                         0,
                                         0,
                                         workspace.ForwardTwiddles,
                                         workspace.InverseTwiddles,
                                         n - 1u,
                                         SharkNTT::ReferenceInputScaleR(stages)};
        }

        if (workspaceOffset != workspaceBytes)
            throw FractalSharkSeriousException("Reference workspace size does not match its layout");

        CheckCuda(cudaMalloc(&workspaceGpu, sizeof(Workspace)),
                  "cudaMalloc(Reference workspace descriptor)");
        CheckCuda(cudaMemcpy(workspaceGpu, &workspace, sizeof(Workspace), cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference workspace descriptor H2D)");
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

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
AllocatePreparedTables(uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return AllocatePreparedTables<SharkFloatParams>(
        actualPrecisionLimbs, Workspace::MinFusedStages, Workspace::MaxFusedStages);
}

} // namespace ReferenceSetupDetail

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                               const HpSharkFloat<SharkFloatParams> &cReal,
                               const HpSharkFloat<SharkFloatParams> &cImag,
                               uint32_t actualPrecisionLimbs,
                               uint32_t minFusedStages,
                               uint32_t maxFusedStages)
{
    (void)cReal;
    (void)cImag;
    auto prepared = ReferenceSetupDetail::AllocatePreparedTables<SharkFloatParams>(
        actualPrecisionLimbs, minFusedStages, maxFusedStages);
    uint64_t *tempData = nullptr;
    try {
        constexpr size_t TempBytes = HpShark::AdditionalUInt64Global * sizeof(uint64_t);
        ReferenceSetupDetail::CheckCuda(cudaMalloc(&tempData, TempBytes),
                                         "cudaMalloc(Reference setup debug scratch)");
        ReferenceSetupDetail::CheckCuda(cudaMemset(tempData, 0, TempBytes),
                                         "cudaMemset(Reference setup debug scratch)");

        auto *workspace = prepared->GetDeviceDescriptor();
        void *kernelArgs[] = {&workspace, &tempData};
        cudaStream_t stream{};
        ComputeHpSharkReferenceSetup<SharkFloatParams>(launchParams, stream, kernelArgs);

        ReferenceSetupDetail::CheckCuda(cudaFree(tempData), "cudaFree(Reference setup debug scratch)");
        tempData = nullptr;
        return prepared;
    } catch (...) {
        if (tempData != nullptr)
            cudaFree(tempData);
        throw;
    }
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                               const HpSharkFloat<SharkFloatParams> &cReal,
                               const HpSharkFloat<SharkFloatParams> &cImag,
                               uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return PrepareHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                            cReal,
                                                            cImag,
                                                            actualPrecisionLimbs,
                                                            Workspace::MinFusedStages,
                                                            Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                               const mpf_t cReal,
                               const mpf_t cImag,
                               uint32_t actualPrecisionLimbs,
                               uint32_t minFusedStages,
                               uint32_t maxFusedStages)
{
    auto inputReal = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    auto inputImag = std::make_unique<HpSharkFloat<SharkFloatParams>>();
    inputReal->MpfToHpGpu(
        cReal, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    inputImag->MpfToHpGpu(
        cImag, HpSharkFloat<SharkFloatParams>::DefaultMpirBits, InjectNoiseInLowOrder::Enable);
    return PrepareHpSharkReferenceTables<SharkFloatParams>(
        launchParams, *inputReal, *inputImag, actualPrecisionLimbs, minFusedStages, maxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                               const mpf_t cReal,
                               const mpf_t cImag,
                               uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return PrepareHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                            cReal,
                                                            cImag,
                                                            actualPrecisionLimbs,
                                                            Workspace::MinFusedStages,
                                                            Workspace::MaxFusedStages);
}

} // namespace HpShark
