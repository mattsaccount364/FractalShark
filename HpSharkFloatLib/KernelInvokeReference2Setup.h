#pragma once

#include <cuda_runtime.h>

#include "Exceptions.h"
#include "HpSharkFloat.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

namespace HpShark::Reference2HostSetup {

template <class SharkFloatParams> struct WorkspaceAllocation {
    HpSharkReference2Workspace<SharkFloatParams> *Descriptor;
    void *Storage;
    size_t StorageBytes;
};

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
uint64_t
ReadBits(const HpSharkFloat<SharkFloatParams> &value, int64_t bitIndex, int bitCount)
{
    constexpr int64_t TotalBits = static_cast<int64_t>(SharkFloatParams::GlobalNumUint32) * 32;
    if (bitIndex < 0 || bitIndex >= TotalBits)
        return 0;

    uint64_t result = 0;
    int remaining = bitCount;
    int outputBit = 0;
    while (remaining > 0 && bitIndex < TotalBits) {
        const int64_t word = bitIndex / 32;
        const int offset = static_cast<int>(bitIndex % 32);
        const uint32_t limb = value.Digits[static_cast<int>(word)];
        const uint32_t chunk = offset == 0 ? limb : limb >> offset;
        const int take = std::min(32 - offset, remaining);
        const uint32_t mask = take == 32 ? 0xffff'ffffu : (1u << take) - 1u;
        result |= static_cast<uint64_t>(chunk & mask) << outputBit;
        outputBit += take;
        remaining -= take;
        bitIndex += take;
    }
    return result;
}

inline uint64_t
AddP(uint64_t left, uint64_t right)
{
    const uint64_t sum = left + right;
    return sum < left || sum >= SharkNTT::MagicPrime ? sum - SharkNTT::MagicPrime : sum;
}

inline uint64_t
SubP(uint64_t left, uint64_t right)
{
    return left >= right ? left - right : left + SharkNTT::MagicPrime - right;
}

inline uint32_t
ReverseBits(uint32_t value, uint32_t bitCount)
{
    uint32_t reversed = 0;
    for (uint32_t bit = 0; bit < bitCount; ++bit) {
        reversed = (reversed << 1u) | (value & 1u);
        value >>= 1u;
    }
    return reversed;
}

inline void
BitReverse(std::vector<uint64_t> &values, uint32_t n, uint32_t stages)
{
    for (uint32_t index = 0; index < n; ++index) {
        const uint32_t reversed = ReverseBits(index, stages) & (n - 1u);
        if (reversed > index)
            std::swap(values[index], values[reversed]);
    }
}

template <class SharkFloatParams>
void
ForwardNtt(std::vector<uint64_t> &values,
           uint32_t n,
           uint32_t stages,
           const SharkNTT::RootTables &maximumRoots)
{
    for (uint32_t stage = 1; stage <= stages; ++stage) {
        const uint32_t width = 1u << stage;
        const uint32_t half = width >> 1u;
        const uint32_t twiddleOffset = half - 1u;
        for (uint32_t base = 0; base < n; base += width) {
            for (uint32_t index = 0; index < half; ++index) {
                const uint32_t leftIndex = base + index;
                const uint32_t rightIndex = leftIndex + half;
                const uint64_t left = values[leftIndex];
                const uint64_t right = SharkNTT::MontgomeryMul<SharkFloatParams>(
                    values[rightIndex], maximumRoots.stage_twiddles_fwd[twiddleOffset + index]);
                values[leftIndex] = AddP(left, right);
                values[rightIndex] = SubP(left, right);
            }
        }
    }
}

template <class SharkFloatParams>
void
BuildInvariantSpectrum(const HpSharkFloat<SharkFloatParams> &value,
                       const SharkNTT::PlanPrime &plan,
                       const SharkNTT::RootTables &maximumRoots,
                       uint32_t psiStride,
                       uint32_t inputBitOffset,
                       std::vector<uint64_t> &staging)
{
    const uint32_t n = static_cast<uint32_t>(plan.N);
    for (uint32_t index = 0; index < n; ++index) {
        const uint64_t coefficient =
            index < static_cast<uint32_t>(plan.L)
                ? ReadBits(value,
                           static_cast<int64_t>(inputBitOffset) + static_cast<int64_t>(index) * plan.b,
                           plan.b)
                : 0;
        const uint64_t coefficientMont =
            SharkNTT::ToMontgomery<SharkFloatParams>(coefficient % SharkNTT::MagicPrime);
        staging[index] = SharkNTT::MontgomeryMul<SharkFloatParams>(
            coefficientMont, maximumRoots.psi_pows[static_cast<size_t>(index) * psiStride]);
    }

    BitReverse(staging, n, static_cast<uint32_t>(plan.stages));
    ForwardNtt<SharkFloatParams>(staging, n, static_cast<uint32_t>(plan.stages), maximumRoots);
}

template <class SharkFloatParams>
WorkspaceAllocation<SharkFloatParams>
CreateWorkspace(const HpSharkFloat<SharkFloatParams> &cReal,
                const HpSharkFloat<SharkFloatParams> &cImag,
                const HpSharkFloat<SharkFloatParams> *one,
                uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
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
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        if (one == nullptr)
            throw FractalSharkSeriousException("Ref2 NR workspace requires the constant one");
    }

    const SharkNTT::PlanPrime precisionPlan =
        SharkNTT::BuildPlanPrime2(static_cast<int>(actualPrecisionLimbs));
    if (!precisionPlan.ok)
        throw FractalSharkSeriousException("Unable to build the Ref2 precision plan");
    const uint32_t ignoredPrecisionBits = (StoragePrecisionLimbs - actualPrecisionLimbs) * 32u;
    const auto alignWorkspace = [](size_t value, size_t alignment) {
        return (value + alignment - 1u) & ~(alignment - 1u);
    };

    size_t workspaceBytes = 0;
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes +=
        WorkingSpectrumCount * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes +=
        ConstantArenaCount * static_cast<size_t>(Workspace::PsiArenaSize) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += LimbCount * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(int64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += 2u * static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint32_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += static_cast<size_t>(Workspace::MaxFusedLimbs) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += static_cast<size_t>(Workspace::MaxCarryPrefixParts) *
                      sizeof(HpSharkReference2CarryPrefixDescriptor);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += static_cast<size_t>(Workspace::CarryPrefixControlCount) * sizeof(uint32_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += static_cast<size_t>(Workspace::MaxFusedStages) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += static_cast<size_t>(Workspace::MaxFusedStages) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += 2u * static_cast<size_t>(Workspace::PsiArenaSize) * sizeof(uint64_t);
    workspaceBytes = alignWorkspace(workspaceBytes, WorkspaceAlignment);
    workspaceBytes += 2u * static_cast<size_t>(Workspace::MaxFusedN) * sizeof(uint64_t);

    void *workspaceStorage = nullptr;
    Workspace *workspaceGpu = nullptr;
    SharkNTT::RootTables maximumRoots{};
    bool rootsBuilt = false;
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
        workspace.ForwardTwiddles = allocateSpectrum();
        workspace.InverseTwiddles = allocateSpectrum();
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
        workspace.ActualPrecisionLimbs = actualPrecisionLimbs;
        workspace.IgnoredPrecisionBits = ignoredPrecisionBits;

        if (workspaceOffset != workspaceBytes)
            throw FractalSharkSeriousException("Reference2 workspace size does not match its layout");

        SharkNTT::BuildRoots<SharkFloatParams>(
            Workspace::MaxFusedN, Workspace::MaxFusedStages, maximumRoots);
        rootsBuilt = true;

        CheckCuda(cudaMemcpy(workspace.StageOmegas,
                             maximumRoots.stage_omegas,
                             Workspace::MaxFusedStages * sizeof(uint64_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 stage omegas H2D)");
        CheckCuda(cudaMemcpy(workspace.StageOmegasInverse,
                             maximumRoots.stage_omegas_inv,
                             Workspace::MaxFusedStages * sizeof(uint64_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 inverse stage omegas H2D)");
        CheckCuda(cudaMemcpy(workspace.ForwardTwiddles,
                             maximumRoots.stage_twiddles_fwd,
                             maximumRoots.total_twiddles * sizeof(uint64_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 forward twiddles H2D)");
        CheckCuda(cudaMemcpy(workspace.InverseTwiddles,
                             maximumRoots.stage_twiddles_inv,
                             maximumRoots.total_twiddles * sizeof(uint64_t),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 inverse twiddles H2D)");

        std::vector<uint64_t> staging(Workspace::MaxFusedN);
        const uint64_t inverseTwo =
            SharkNTT::ToMontgomery<SharkFloatParams>((SharkNTT::MagicPrime + 1ull) >> 1u);
        const uint64_t oneMont = SharkNTT::ToMontgomery<SharkFloatParams>(1);
        for (uint32_t slot = 0; slot < Workspace::PlanCacheEntryCount; ++slot) {
            const uint32_t stages = Workspace::MinFusedStages + slot;
            const uint32_t n = 1u << stages;
            const uint32_t arenaOffset = n - Workspace::MinFusedN;
            const uint32_t psiStride = Workspace::MaxFusedN / n;
            workspace.Plans[slot] = {precisionPlan.n32,
                                     precisionPlan.b,
                                     precisionPlan.L,
                                     static_cast<int>(n),
                                     static_cast<int>(stages),
                                     precisionPlan.ok};

            uint64_t nInverse = oneMont;
            for (uint32_t stage = 0; stage < stages; ++stage)
                nInverse = SharkNTT::MontgomeryMul<SharkFloatParams>(nInverse, inverseTwo);
            workspace.PlanRoots[slot] = {static_cast<int32_t>(stages),
                                         workspace.StageOmegas,
                                         workspace.StageOmegasInverse,
                                         static_cast<int32_t>(n),
                                         workspace.PsiPowersArena + arenaOffset,
                                         workspace.PsiInversePowersArena + arenaOffset,
                                         nInverse,
                                         workspace.ForwardTwiddles,
                                         workspace.InverseTwiddles,
                                         n - 1u};
            workspace.ConstantSpectra[slot] = {
                workspace.CRealArena + arenaOffset,
                workspace.CImagArena + arenaOffset,
                SharkFloatParams::EnableNewtonRaphson ? workspace.OneArena + arenaOffset : nullptr};

            for (uint32_t index = 0; index < n; ++index)
                staging[index] = maximumRoots.psi_pows[static_cast<size_t>(index) * psiStride];
            CheckCuda(cudaMemcpy(workspace.PlanRoots[slot].psi_pows,
                                 staging.data(),
                                 static_cast<size_t>(n) * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(Reference2 psi powers H2D)");
            for (uint32_t index = 0; index < n; ++index)
                staging[index] = maximumRoots.psi_inv_pows[static_cast<size_t>(index) * psiStride];
            CheckCuda(cudaMemcpy(workspace.PlanRoots[slot].psi_inv_pows,
                                 staging.data(),
                                 static_cast<size_t>(n) * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(Reference2 inverse psi powers H2D)");

            BuildInvariantSpectrum(
                cReal, workspace.Plans[slot], maximumRoots, psiStride, ignoredPrecisionBits, staging);
            CheckCuda(cudaMemcpy(workspace.ConstantSpectra[slot].CReal,
                                 staging.data(),
                                 static_cast<size_t>(n) * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(Reference2 CReal spectrum H2D)");
            BuildInvariantSpectrum(
                cImag, workspace.Plans[slot], maximumRoots, psiStride, ignoredPrecisionBits, staging);
            CheckCuda(cudaMemcpy(workspace.ConstantSpectra[slot].CImag,
                                 staging.data(),
                                 static_cast<size_t>(n) * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(Reference2 CImag spectrum H2D)");
            if constexpr (SharkFloatParams::EnableNewtonRaphson) {
                BuildInvariantSpectrum(
                    *one, workspace.Plans[slot], maximumRoots, psiStride, ignoredPrecisionBits, staging);
                CheckCuda(cudaMemcpy(workspace.ConstantSpectra[slot].One,
                                     staging.data(),
                                     static_cast<size_t>(n) * sizeof(uint64_t),
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy(Reference2 One spectrum H2D)");
            }
        }

        SharkNTT::DestroyRoots<SharkFloatParams>(false, maximumRoots);
        rootsBuilt = false;
        CheckCuda(cudaMalloc(&workspaceGpu, sizeof(Workspace)),
                  "cudaMalloc(Reference2 workspace descriptor)");
        CheckCuda(cudaMemcpy(workspaceGpu, &workspace, sizeof(Workspace), cudaMemcpyHostToDevice),
                  "cudaMemcpy(Reference2 workspace descriptor H2D)");
        return {workspaceGpu, workspaceStorage, workspaceBytes};
    } catch (...) {
        if (rootsBuilt)
            SharkNTT::DestroyRoots<SharkFloatParams>(false, maximumRoots);
        if (workspaceGpu != nullptr)
            cudaFree(workspaceGpu);
        if (workspaceStorage != nullptr)
            cudaFree(workspaceStorage);
        throw;
    }
}

} // namespace HpShark::Reference2HostSetup
