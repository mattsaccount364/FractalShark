#pragma once

#include "Environment.h"
#include "KernelInvokeReference2Setup.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace HpShark {

class Reference2MappedCacheFile {
public:
    static std::unique_ptr<Reference2MappedCacheFile> CreateWrite(const wchar_t *path, size_t bytes);

    static std::unique_ptr<Reference2MappedCacheFile> OpenRead(const wchar_t *path);

    ~Reference2MappedCacheFile();

    Reference2MappedCacheFile(const Reference2MappedCacheFile &) = delete;
    Reference2MappedCacheFile &operator=(const Reference2MappedCacheFile &) = delete;

    uint8_t *Data();

    const uint8_t *Data() const;

    size_t Size() const;

    void Flush();

private:
    explicit Reference2MappedCacheFile(std::unique_ptr<Environment::MappedFile> mappedFile);

    std::unique_ptr<Environment::MappedFile> m_MappedFile;
};

#pragma pack(push, 1)
struct Reference2CacheHeader {
    char Magic[8];
    uint32_t Version;
    uint32_t HeaderBytes;
    uint32_t StoragePrecisionLimbs;
    uint32_t ActualPrecisionLimbs;
    uint32_t EnableNewtonRaphson;
    uint32_t MinFusedN;
    uint32_t MaxFusedN;
    uint32_t MinFusedStages;
    uint32_t MaxFusedStages;
    uint32_t PlanCacheEntryCount;
    uint64_t PayloadBytes;
    int64_t TestNumber;
    uint32_t Sequence;
    uint32_t Reserved;
};
#pragma pack(pop)

static constexpr uint32_t Reference2CacheVersion = 4;

namespace Reference2CacheDetail {

inline constexpr std::array<char, 8> CacheMagic{'F', 'S', 'R', '2', 'C', 'A', 'C', 'H'};

inline constexpr size_t
Align(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

template <class SharkFloatParams> struct CacheLayout {
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;

    static constexpr size_t PayloadAlignment = 16u;
    size_t StageBytes;
    size_t PsiBytes;
    size_t TwiddleBytes;
    size_t NinvmBytes;
    size_t StageOmegasOffset;
    size_t StageOmegasInverseOffset;
    size_t PsiPowersOffset;
    size_t PsiInversePowersOffset;
    size_t ForwardTwiddlesOffset;
    size_t InverseTwiddlesOffset;
    size_t CRealOffset;
    size_t CImagOffset;
    size_t OneOffset;
    size_t NinvmOffset;
    size_t PayloadBytes;
    size_t PayloadOffset;
    size_t FileBytes;
    uint32_t MinFusedN;
    uint32_t MaxFusedN;
    uint32_t MinFusedStages;
    uint32_t MaxFusedStages;
    uint32_t PlanCacheEntryCount;

    CacheLayout(uint32_t minFusedStages, uint32_t maxFusedStages)
        : StageBytes(maxFusedStages * sizeof(uint64_t)),
          PsiBytes((2u * (static_cast<size_t>(1u) << maxFusedStages) -
                    (static_cast<size_t>(1u) << minFusedStages)) *
                   sizeof(uint64_t)),
          TwiddleBytes((static_cast<size_t>(1u) << maxFusedStages) * sizeof(uint64_t)),
          NinvmBytes((maxFusedStages - minFusedStages + 1u) * sizeof(uint64_t)), StageOmegasOffset(0),
          StageOmegasInverseOffset(StageOmegasOffset + StageBytes),
          PsiPowersOffset(StageOmegasInverseOffset + StageBytes),
          PsiInversePowersOffset(PsiPowersOffset + PsiBytes),
          ForwardTwiddlesOffset(PsiInversePowersOffset + PsiBytes),
          InverseTwiddlesOffset(ForwardTwiddlesOffset + TwiddleBytes),
          CRealOffset(InverseTwiddlesOffset + TwiddleBytes), CImagOffset(CRealOffset + PsiBytes),
          OneOffset(CImagOffset + PsiBytes),
          NinvmOffset(OneOffset + (SharkFloatParams::EnableNewtonRaphson ? PsiBytes : 0u)),
          PayloadBytes(NinvmOffset + NinvmBytes),
          PayloadOffset(Align(sizeof(Reference2CacheHeader), PayloadAlignment)),
          FileBytes(PayloadOffset + PayloadBytes), MinFusedN(1u << minFusedStages),
          MaxFusedN(1u << maxFusedStages), MinFusedStages(minFusedStages),
          MaxFusedStages(maxFusedStages), PlanCacheEntryCount(maxFusedStages - minFusedStages + 1u)
    {
    }
};

template <class SharkFloatParams>
uint32_t
PlanMask(uint32_t minFusedStages, uint32_t maxFusedStages)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const uint32_t firstSlot = minFusedStages - Workspace::MinFusedStages;
    const uint32_t count = maxFusedStages - minFusedStages + 1u;
    const uint32_t rangeMask = count == 32u ? ~0u : (1u << count) - 1u;
    return rangeMask << firstSlot;
}

template <class SharkFloatParams>
std::wstring
CachePath(int64_t testNumber,
          uint32_t sequence,
          uint32_t actualPrecisionLimbs,
          uint32_t minFusedStages,
          uint32_t maxFusedStages)
{
    std::wostringstream name;
    name << L"Ref2Prepared-v" << Reference2CacheVersion << L"-p" << SharkFloatParams::GlobalNumUint32
         << L"-nr" << (SharkFloatParams::EnableNewtonRaphson ? 1 : 0) << L"-a" << actualPrecisionLimbs
         << L"-test" << testNumber << L"-iter" << sequence << L"-stage" << minFusedStages << L"-"
         << maxFusedStages << L".r2cache";
    return name.str();
}

template <class SharkFloatParams>
Reference2CacheHeader
MakeHeader(int64_t testNumber,
           uint32_t sequence,
           uint32_t actualPrecisionLimbs,
           const CacheLayout<SharkFloatParams> &layout)
{
    Reference2CacheHeader header{};
    std::copy(CacheMagic.begin(), CacheMagic.end(), std::begin(header.Magic));
    header.Version = Reference2CacheVersion;
    header.HeaderBytes = static_cast<uint32_t>(layout.PayloadOffset);
    header.StoragePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    header.ActualPrecisionLimbs = actualPrecisionLimbs;
    header.EnableNewtonRaphson = SharkFloatParams::EnableNewtonRaphson ? 1u : 0u;
    header.MinFusedN = layout.MinFusedN;
    header.MaxFusedN = layout.MaxFusedN;
    header.MinFusedStages = layout.MinFusedStages;
    header.MaxFusedStages = layout.MaxFusedStages;
    header.PlanCacheEntryCount = layout.PlanCacheEntryCount;
    header.PayloadBytes = layout.PayloadBytes;
    header.TestNumber = testNumber;
    header.Sequence = sequence;
    return header;
}

template <class SharkFloatParams>
void
ValidateHeader(const Reference2CacheHeader &header,
               int64_t testNumber,
               uint32_t sequence,
               uint32_t actualPrecisionLimbs,
               const CacheLayout<SharkFloatParams> &layout,
               size_t mappedBytes)
{
    const auto magic = std::array<char, 8>{header.Magic[0],
                                           header.Magic[1],
                                           header.Magic[2],
                                           header.Magic[3],
                                           header.Magic[4],
                                           header.Magic[5],
                                           header.Magic[6],
                                           header.Magic[7]};
    if (magic != CacheMagic || header.Version != Reference2CacheVersion ||
        header.HeaderBytes != layout.PayloadOffset ||
        header.StoragePrecisionLimbs != SharkFloatParams::GlobalNumUint32 ||
        header.ActualPrecisionLimbs != actualPrecisionLimbs ||
        header.EnableNewtonRaphson != (SharkFloatParams::EnableNewtonRaphson ? 1u : 0u) ||
        header.MinFusedN != layout.MinFusedN || header.MaxFusedN != layout.MaxFusedN ||
        header.MinFusedStages != layout.MinFusedStages ||
        header.MaxFusedStages != layout.MaxFusedStages ||
        header.PlanCacheEntryCount != layout.PlanCacheEntryCount ||
        header.PayloadBytes != layout.PayloadBytes || header.TestNumber != testNumber ||
        header.Sequence != sequence || mappedBytes != layout.FileBytes) {
        throw FractalSharkSeriousException("Ref2 prepared-table cache header is incompatible");
    }
}

template <class SharkFloatParams>
void
CopyDeviceToCache(uint8_t *destination, const void *source, size_t bytes, const char *operation)
{
    Reference2SetupDetail::CheckCuda(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
                                     operation);
}

template <class SharkFloatParams>
void
CopyCacheToDevice(void *destination, const uint8_t *source, size_t bytes, const char *operation)
{
    Reference2SetupDetail::CheckCuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                                     operation);
}

template <class SharkFloatParams>
void
CopyPreparedPayloadToCache(
    const typename Reference2PreparedTables<SharkFloatParams>::Workspace &workspace,
    const CacheLayout<SharkFloatParams> &layout,
    uint8_t *payload)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    CopyDeviceToCache<SharkFloatParams>(payload + layout.StageOmegasOffset,
                                        workspace.StageOmegas,
                                        layout.StageBytes,
                                        "cudaMemcpy(Ref2 cache stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.StageOmegasInverseOffset,
                                        workspace.StageOmegasInverse,
                                        layout.StageBytes,
                                        "cudaMemcpy(Ref2 cache inverse stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.PsiPowersOffset,
                                        workspace.PsiPowersArena,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache psi powers D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.PsiInversePowersOffset,
                                        workspace.PsiInversePowersArena,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache inverse psi powers D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.ForwardTwiddlesOffset,
                                        workspace.ForwardTwiddles,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache forward twiddles D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.InverseTwiddlesOffset,
                                        workspace.InverseTwiddles,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache inverse twiddles D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.CRealOffset,
                                        workspace.CRealArena,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache CReal spectra D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.CImagOffset,
                                        workspace.CImagArena,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache CImag spectra D2H)");
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        CopyDeviceToCache<SharkFloatParams>(payload + layout.OneOffset,
                                            workspace.OneArena,
                                            layout.PsiBytes,
                                            "cudaMemcpy(Ref2 cache One spectra D2H)");
    }

    std::vector<uint64_t> ninvm(layout.PlanCacheEntryCount);
    const uint32_t firstSlot = layout.MinFusedStages - Workspace::MinFusedStages;
    for (uint32_t index = 0; index < ninvm.size(); ++index)
        ninvm[index] = workspace.PlanRoots[firstSlot + index].Ninvm_mont;
    std::memcpy(payload + layout.NinvmOffset, ninvm.data(), layout.NinvmBytes);
}

template <class SharkFloatParams>
void
CopyCachePayloadToPrepared(const uint8_t *payload,
                           Reference2PreparedTables<SharkFloatParams> &prepared,
                           uint32_t actualPrecisionLimbs,
                           const CacheLayout<SharkFloatParams> &layout)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    Workspace workspace{};
    Reference2SetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Ref2 cache descriptor D2H)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegas,
                                        payload + layout.StageOmegasOffset,
                                        layout.StageBytes,
                                        "cudaMemcpy(Ref2 cache stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegasInverse,
                                        payload + layout.StageOmegasInverseOffset,
                                        layout.StageBytes,
                                        "cudaMemcpy(Ref2 cache inverse stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.PsiPowersArena,
                                        payload + layout.PsiPowersOffset,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache psi powers H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.PsiInversePowersArena,
                                        payload + layout.PsiInversePowersOffset,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache inverse psi powers H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.ForwardTwiddles,
                                        payload + layout.ForwardTwiddlesOffset,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache forward twiddles H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.InverseTwiddles,
                                        payload + layout.InverseTwiddlesOffset,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache inverse twiddles H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.CRealArena,
                                        payload + layout.CRealOffset,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache CReal spectra H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.CImagArena,
                                        payload + layout.CImagOffset,
                                        layout.PsiBytes,
                                        "cudaMemcpy(Ref2 cache CImag spectra H2D)");
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        CopyCacheToDevice<SharkFloatParams>(workspace.OneArena,
                                            payload + layout.OneOffset,
                                            layout.PsiBytes,
                                            "cudaMemcpy(Ref2 cache One spectra H2D)");
    }

    std::vector<uint64_t> ninvm(layout.PlanCacheEntryCount);
    std::memcpy(ninvm.data(), payload + layout.NinvmOffset, layout.NinvmBytes);
    const uint32_t firstSlot = layout.MinFusedStages - Workspace::MinFusedStages;
    for (uint32_t index = 0; index < layout.PlanCacheEntryCount; ++index)
        workspace.PlanRoots[firstSlot + index].Ninvm_mont = ninvm[index];
    workspace.ValidPlanMask = PlanMask<SharkFloatParams>(layout.MinFusedStages, layout.MaxFusedStages);
    workspace.GeneratedStages = layout.MaxFusedStages;
    workspace.ActualPrecisionLimbs = actualPrecisionLimbs;
    workspace.IgnoredPrecisionBits = (SharkFloatParams::GlobalNumUint32 - actualPrecisionLimbs) * 32u;
    Reference2SetupDetail::CheckCuda(
        cudaMemcpy(
            prepared.GetDeviceDescriptor(), &workspace, sizeof(workspace), cudaMemcpyHostToDevice),
        "cudaMemcpy(Ref2 cache descriptor H2D)");
    prepared.UpdateHostDescriptor(workspace);
}

} // namespace Reference2CacheDetail

template <class SharkFloatParams>
void
SaveHpSharkReference2Tables(const Reference2PreparedTables<SharkFloatParams> &prepared,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs,
                            uint32_t minFusedStages,
                            uint32_t maxFusedStages)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const Reference2CacheDetail::CacheLayout<SharkFloatParams> layout(minFusedStages, maxFusedStages);
    const auto target = Reference2CacheDetail::CachePath<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    const auto temporary = target + L".tmp";
    Environment::FileDelete(temporary.c_str());

    auto mapped = Reference2MappedCacheFile::CreateWrite(temporary.c_str(), layout.FileBytes);
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Unable to create Ref2 prepared-table cache");
    uint8_t *payload = mapped->Data() + layout.PayloadOffset;
    Workspace workspace{};
    Reference2SetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Ref2 cache save descriptor D2H)");
    const uint32_t fullMask =
        Reference2CacheDetail::PlanMask<SharkFloatParams>(minFusedStages, maxFusedStages);
    if (workspace.ValidPlanMask != fullMask)
        throw FractalSharkSeriousException("Cannot cache incomplete Ref2 prepared tables");
    if (workspace.ActiveMinFusedStages != minFusedStages ||
        workspace.ActiveMaxFusedStages != maxFusedStages)
        throw FractalSharkSeriousException("Ref2 prepared-table cache stage range is incompatible");
    Reference2CacheDetail::CopyPreparedPayloadToCache<SharkFloatParams>(workspace, layout, payload);
    auto header = Reference2CacheDetail::MakeHeader<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, layout);
    std::memcpy(mapped->Data(), &header, sizeof(header));
    mapped->Flush();
    mapped.reset();
    if (!Environment::FileRename(temporary.c_str(), target.c_str(), true))
        throw FractalSharkSeriousException("Unable to publish Ref2 prepared-table cache");
}

template <class SharkFloatParams>
void
SaveHpSharkReference2Tables(const Reference2PreparedTables<SharkFloatParams> &prepared,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    SaveHpSharkReference2Tables<SharkFloatParams>(prepared,
                                                  testNumber,
                                                  sequence,
                                                  actualPrecisionLimbs,
                                                  Workspace::MinFusedStages,
                                                  Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
LoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs,
                            uint32_t minFusedStages,
                            uint32_t maxFusedStages)
{
    (void)launchParams;
    const Reference2CacheDetail::CacheLayout<SharkFloatParams> layout(minFusedStages, maxFusedStages);
    const auto path = Reference2CacheDetail::CachePath<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    auto mapped = Reference2MappedCacheFile::OpenRead(path.c_str());
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Ref2 prepared-table cache is unavailable");
    if (mapped->Size() < sizeof(Reference2CacheHeader))
        throw FractalSharkSeriousException("Ref2 prepared-table cache is truncated");
    Reference2CacheHeader header{};
    std::memcpy(&header, mapped->Data(), sizeof(header));
    Reference2CacheDetail::ValidateHeader<SharkFloatParams>(
        header, testNumber, sequence, actualPrecisionLimbs, layout, mapped->Size());
    const auto *payload = mapped->Data() + layout.PayloadOffset;

    auto prepared = Reference2SetupDetail::AllocatePreparedTables<SharkFloatParams>(
        actualPrecisionLimbs, minFusedStages, maxFusedStages);
    Reference2CacheDetail::CopyCachePayloadToPrepared<SharkFloatParams>(
        payload, *prepared, actualPrecisionLimbs, layout);
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
LoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    return LoadHpSharkReference2Tables<SharkFloatParams>(launchParams,
                                                         testNumber,
                                                         sequence,
                                                         actualPrecisionLimbs,
                                                         Workspace::MinFusedStages,
                                                         Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                                     const HpSharkFloat<SharkFloatParams> &cReal,
                                     const HpSharkFloat<SharkFloatParams> &cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence,
                                     uint32_t minFusedStages,
                                     uint32_t maxFusedStages)
{
    try {
        return LoadHpSharkReference2Tables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReference2Tables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    try {
        SaveHpSharkReference2Tables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache save failed for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                                     const HpSharkFloat<SharkFloatParams> &cReal,
                                     const HpSharkFloat<SharkFloatParams> &cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence = 0)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    return PrepareOrLoadHpSharkReference2Tables<SharkFloatParams>(launchParams,
                                                                  cReal,
                                                                  cImag,
                                                                  actualPrecisionLimbs,
                                                                  testNumber,
                                                                  sequence,
                                                                  Workspace::MinFusedStages,
                                                                  Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                                     const mpf_t cReal,
                                     const mpf_t cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence,
                                     uint32_t minFusedStages,
                                     uint32_t maxFusedStages)
{
    try {
        return LoadHpSharkReference2Tables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReference2Tables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    try {
        SaveHpSharkReference2Tables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache save failed for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                                     const mpf_t cReal,
                                     const mpf_t cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence = 0)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    return PrepareOrLoadHpSharkReference2Tables<SharkFloatParams>(launchParams,
                                                                  cReal,
                                                                  cImag,
                                                                  actualPrecisionLimbs,
                                                                  testNumber,
                                                                  sequence,
                                                                  Workspace::MinFusedStages,
                                                                  Workspace::MaxFusedStages);
}

} // namespace HpShark
