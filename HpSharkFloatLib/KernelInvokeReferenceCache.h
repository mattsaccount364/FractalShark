#pragma once

// Persistent cache support for prepared reference-NTT tables.

#include "Environment.h"
#include "KernelInvokeReferenceSetup.h"

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

class ReferenceMappedCacheFile {
public:
    static std::unique_ptr<ReferenceMappedCacheFile> CreateWrite(const wchar_t *path, size_t bytes);

    static std::unique_ptr<ReferenceMappedCacheFile> OpenRead(const wchar_t *path);

    ~ReferenceMappedCacheFile();

    ReferenceMappedCacheFile(const ReferenceMappedCacheFile &) = delete;
    ReferenceMappedCacheFile &operator=(const ReferenceMappedCacheFile &) = delete;

    uint8_t *Data();

    const uint8_t *Data() const;

    size_t Size() const;

    void Flush();

private:
    explicit ReferenceMappedCacheFile(std::unique_ptr<Environment::MappedFile> mappedFile);

    std::unique_ptr<Environment::MappedFile> m_MappedFile;
};

#pragma pack(push, 1)
struct ReferenceCacheHeader {
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

static constexpr uint32_t ReferenceCacheVersion = 6;

namespace ReferenceCacheDetail {

inline constexpr std::array<char, 8> CacheMagic{'F', 'S', 'R', '2', 'C', 'A', 'C', 'H'};
inline constexpr wchar_t CacheDirectoryName[] = L"ReferencePreparedTemp";

inline constexpr size_t
Align(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

template <class SharkFloatParams> struct CacheLayout {
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;

    static constexpr size_t PayloadAlignment = 16u;
    size_t StageBytes;
    size_t TwiddleBytes;
    size_t NinvBytes;
    size_t StageOmegasOffset;
    size_t StageOmegasInverseOffset;
    size_t ForwardTwiddlesOffset;
    size_t InverseTwiddlesOffset;
    size_t NinvOffset;
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
          TwiddleBytes((static_cast<size_t>(1u) << maxFusedStages) * sizeof(uint64_t)),
          NinvBytes((maxFusedStages - minFusedStages + 1u) * sizeof(uint64_t)), StageOmegasOffset(0),
          StageOmegasInverseOffset(StageOmegasOffset + StageBytes),
          ForwardTwiddlesOffset(StageOmegasInverseOffset + StageBytes),
          InverseTwiddlesOffset(ForwardTwiddlesOffset + TwiddleBytes),
          NinvOffset(InverseTwiddlesOffset + TwiddleBytes), PayloadBytes(NinvOffset + NinvBytes),
          PayloadOffset(Align(sizeof(ReferenceCacheHeader), PayloadAlignment)),
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
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
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
    name << L"ReferencePrepared-v" << ReferenceCacheVersion << L"-p" << SharkFloatParams::GlobalNumUint32
         << L"-nr" << (SharkFloatParams::EnableNewtonRaphson ? 1 : 0) << L"-a" << actualPrecisionLimbs
         << L"-test" << testNumber << L"-iter" << sequence << L"-stage" << minFusedStages << L"-"
         << maxFusedStages << L".r2cache";
    return std::wstring{CacheDirectoryName} + L"/" + name.str();
}

template <class SharkFloatParams>
ReferenceCacheHeader
MakeHeader(int64_t testNumber,
           uint32_t sequence,
           uint32_t actualPrecisionLimbs,
           const CacheLayout<SharkFloatParams> &layout)
{
    ReferenceCacheHeader header{};
    std::copy(CacheMagic.begin(), CacheMagic.end(), std::begin(header.Magic));
    header.Version = ReferenceCacheVersion;
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
ValidateHeader(const ReferenceCacheHeader &header,
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
    if (magic != CacheMagic || header.Version != ReferenceCacheVersion ||
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
        throw FractalSharkSeriousException("Reference prepared-table cache header is incompatible");
    }
}

template <class SharkFloatParams>
void
CopyDeviceToCache(uint8_t *destination, const void *source, size_t bytes, const char *operation)
{
    ReferenceSetupDetail::CheckCuda(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
                                     operation);
}

template <class SharkFloatParams>
void
CopyCacheToDevice(void *destination, const uint8_t *source, size_t bytes, const char *operation)
{
    ReferenceSetupDetail::CheckCuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                                     operation);
}

template <class SharkFloatParams>
void
CopyPreparedPayloadToCache(
    const typename ReferencePreparedTables<SharkFloatParams>::Workspace &workspace,
    const CacheLayout<SharkFloatParams> &layout,
    uint8_t *payload)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    CopyDeviceToCache<SharkFloatParams>(payload + layout.StageOmegasOffset,
                                        workspace.StageOmegas,
                                        layout.StageBytes,
                                        "cudaMemcpy(Reference cache stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.StageOmegasInverseOffset,
                                        workspace.StageOmegasInverse,
                                        layout.StageBytes,
                                        "cudaMemcpy(Reference cache inverse stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.ForwardTwiddlesOffset,
                                        workspace.ForwardTwiddles,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Reference cache forward twiddles D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + layout.InverseTwiddlesOffset,
                                        workspace.InverseTwiddles,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Reference cache inverse twiddles D2H)");
    std::vector<uint64_t> ninv(layout.PlanCacheEntryCount);
    const uint32_t firstSlot = layout.MinFusedStages - Workspace::MinFusedStages;
    for (uint32_t index = 0; index < ninv.size(); ++index)
        ninv[index] = workspace.PlanRoots[firstSlot + index].Ninv;
    std::memcpy(payload + layout.NinvOffset, ninv.data(), layout.NinvBytes);
}

template <class SharkFloatParams>
void
CopyCachePayloadToPrepared(const uint8_t *payload,
                           ReferencePreparedTables<SharkFloatParams> &prepared,
                           uint32_t actualPrecisionLimbs,
                           const CacheLayout<SharkFloatParams> &layout)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    Workspace workspace{};
    ReferenceSetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Reference cache descriptor D2H)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegas,
                                        payload + layout.StageOmegasOffset,
                                        layout.StageBytes,
                                        "cudaMemcpy(Reference cache stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegasInverse,
                                        payload + layout.StageOmegasInverseOffset,
                                        layout.StageBytes,
                                        "cudaMemcpy(Reference cache inverse stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.ForwardTwiddles,
                                        payload + layout.ForwardTwiddlesOffset,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Reference cache forward twiddles H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.InverseTwiddles,
                                        payload + layout.InverseTwiddlesOffset,
                                        layout.TwiddleBytes,
                                        "cudaMemcpy(Reference cache inverse twiddles H2D)");
    std::vector<uint64_t> ninv(layout.PlanCacheEntryCount);
    std::memcpy(ninv.data(), payload + layout.NinvOffset, layout.NinvBytes);
    const uint32_t firstSlot = layout.MinFusedStages - Workspace::MinFusedStages;
    for (uint32_t index = 0; index < layout.PlanCacheEntryCount; ++index)
        workspace.PlanRoots[firstSlot + index].Ninv = ninv[index];
    workspace.ValidPlanMask = PlanMask<SharkFloatParams>(layout.MinFusedStages, layout.MaxFusedStages);
    workspace.GeneratedStages = layout.MaxFusedStages;
    workspace.ActualPrecisionLimbs = actualPrecisionLimbs;
    workspace.IgnoredPrecisionBits = (SharkFloatParams::GlobalNumUint32 - actualPrecisionLimbs) * 32u;
    ReferenceSetupDetail::CheckCuda(
        cudaMemcpy(
            prepared.GetDeviceDescriptor(), &workspace, sizeof(workspace), cudaMemcpyHostToDevice),
        "cudaMemcpy(Reference cache descriptor H2D)");
    prepared.UpdateHostDescriptor(workspace);
}

} // namespace ReferenceCacheDetail

template <class SharkFloatParams>
void
SaveHpSharkReferenceTables(const ReferencePreparedTables<SharkFloatParams> &prepared,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs,
                            uint32_t minFusedStages,
                            uint32_t maxFusedStages)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    const ReferenceCacheDetail::CacheLayout<SharkFloatParams> layout(minFusedStages, maxFusedStages);
    if (!Environment::DirectoryCreate(ReferenceCacheDetail::CacheDirectoryName) ||
        !Environment::DirectoryExists(ReferenceCacheDetail::CacheDirectoryName))
        throw FractalSharkSeriousException("Unable to create Reference prepared-table cache directory");
    const auto target = ReferenceCacheDetail::CachePath<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    const auto temporary = target + L".tmp";
    Environment::FileDelete(temporary.c_str());

    auto mapped = ReferenceMappedCacheFile::CreateWrite(temporary.c_str(), layout.FileBytes);
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Unable to create Reference prepared-table cache");
    uint8_t *payload = mapped->Data() + layout.PayloadOffset;
    Workspace workspace{};
    ReferenceSetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Reference cache save descriptor D2H)");
    const uint32_t fullMask =
        ReferenceCacheDetail::PlanMask<SharkFloatParams>(minFusedStages, maxFusedStages);
    if (workspace.ValidPlanMask != fullMask)
        throw FractalSharkSeriousException("Cannot cache incomplete Reference prepared tables");
    if (workspace.ActiveMinFusedStages != minFusedStages ||
        workspace.ActiveMaxFusedStages != maxFusedStages)
        throw FractalSharkSeriousException("Reference prepared-table cache stage range is incompatible");
    ReferenceCacheDetail::CopyPreparedPayloadToCache<SharkFloatParams>(workspace, layout, payload);
    auto header = ReferenceCacheDetail::MakeHeader<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, layout);
    std::memcpy(mapped->Data(), &header, sizeof(header));
    mapped->Flush();
    mapped.reset();
    if (!Environment::FileRename(temporary.c_str(), target.c_str(), true))
        throw FractalSharkSeriousException("Unable to publish Reference prepared-table cache");
}

template <class SharkFloatParams>
void
SaveHpSharkReferenceTables(const ReferencePreparedTables<SharkFloatParams> &prepared,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    SaveHpSharkReferenceTables<SharkFloatParams>(prepared,
                                                  testNumber,
                                                  sequence,
                                                  actualPrecisionLimbs,
                                                  Workspace::MinFusedStages,
                                                  Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
LoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs,
                            uint32_t minFusedStages,
                            uint32_t maxFusedStages)
{
    (void)launchParams;
    const ReferenceCacheDetail::CacheLayout<SharkFloatParams> layout(minFusedStages, maxFusedStages);
    const auto path = ReferenceCacheDetail::CachePath<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    auto mapped = ReferenceMappedCacheFile::OpenRead(path.c_str());
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Reference prepared-table cache is unavailable");
    if (mapped->Size() < sizeof(ReferenceCacheHeader))
        throw FractalSharkSeriousException("Reference prepared-table cache is truncated");
    ReferenceCacheHeader header{};
    std::memcpy(&header, mapped->Data(), sizeof(header));
    ReferenceCacheDetail::ValidateHeader<SharkFloatParams>(
        header, testNumber, sequence, actualPrecisionLimbs, layout, mapped->Size());
    const auto *payload = mapped->Data() + layout.PayloadOffset;

    auto prepared = ReferenceSetupDetail::AllocatePreparedTables<SharkFloatParams>(
        actualPrecisionLimbs, minFusedStages, maxFusedStages);
    ReferenceCacheDetail::CopyCachePayloadToPrepared<SharkFloatParams>(
        payload, *prepared, actualPrecisionLimbs, layout);
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
LoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return LoadHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                         testNumber,
                                                         sequence,
                                                         actualPrecisionLimbs,
                                                         Workspace::MinFusedStages,
                                                         Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                                     const HpSharkFloat<SharkFloatParams> &cReal,
                                     const HpSharkFloat<SharkFloatParams> &cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence,
                                     uint32_t minFusedStages,
                                     uint32_t maxFusedStages)
{
    try {
        return LoadHpSharkReferenceTables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Reference cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReferenceTables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    try {
        SaveHpSharkReferenceTables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Reference cache save failed for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                                     const HpSharkFloat<SharkFloatParams> &cReal,
                                     const HpSharkFloat<SharkFloatParams> &cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence = 0)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                                  cReal,
                                                                  cImag,
                                                                  actualPrecisionLimbs,
                                                                  testNumber,
                                                                  sequence,
                                                                  Workspace::MinFusedStages,
                                                                  Workspace::MaxFusedStages);
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                                     const mpf_t cReal,
                                     const mpf_t cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence,
                                     uint32_t minFusedStages,
                                     uint32_t maxFusedStages)
{
    try {
        return LoadHpSharkReferenceTables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Reference cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReferenceTables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    try {
        SaveHpSharkReferenceTables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs, minFusedStages, maxFusedStages);
    } catch (const std::exception &error) {
        std::cout << "Reference cache save failed for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }
    return prepared;
}

template <class SharkFloatParams>
std::unique_ptr<ReferencePreparedTables<SharkFloatParams>>
PrepareOrLoadHpSharkReferenceTables(const HpShark::LaunchParams &launchParams,
                                     const mpf_t cReal,
                                     const mpf_t cImag,
                                     uint32_t actualPrecisionLimbs,
                                     int64_t testNumber,
                                     uint32_t sequence = 0)
{
    using Workspace = HpSharkReferenceWorkspace<SharkFloatParams>;
    return PrepareOrLoadHpSharkReferenceTables<SharkFloatParams>(launchParams,
                                                                  cReal,
                                                                  cImag,
                                                                  actualPrecisionLimbs,
                                                                  testNumber,
                                                                  sequence,
                                                                  Workspace::MinFusedStages,
                                                                  Workspace::MaxFusedStages);
}

} // namespace HpShark
