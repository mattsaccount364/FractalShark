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
    uint32_t MaxFusedStages;
    uint32_t PlanCacheEntryCount;
    uint64_t PayloadBytes;
    uint64_t PayloadChecksum;
    int64_t TestNumber;
    uint32_t Sequence;
    uint32_t Reserved;
};
#pragma pack(pop)

static constexpr uint32_t Reference2CacheVersion = 1;

namespace Reference2CacheDetail {

inline constexpr std::array<char, 8> CacheMagic{'F', 'S', 'R', '2', 'C', 'A', 'C', 'H'};

inline constexpr size_t
Align(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

template <class SharkFloatParams> struct CacheLayout {
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;

    static constexpr size_t StageBytes = Workspace::MaxFusedStages * sizeof(uint64_t);
    static constexpr size_t PsiBytes = Workspace::PsiArenaSize * sizeof(uint64_t);
    static constexpr size_t TwiddleBytes = Workspace::MaxFusedN * sizeof(uint64_t);
    static constexpr size_t NinvmBytes = Workspace::PlanCacheEntryCount * sizeof(uint64_t);
    static constexpr size_t PayloadAlignment = 16u;
    static constexpr size_t StageOmegasOffset = 0;
    static constexpr size_t StageOmegasInverseOffset = StageOmegasOffset + StageBytes;
    static constexpr size_t PsiPowersOffset = StageOmegasInverseOffset + StageBytes;
    static constexpr size_t PsiInversePowersOffset = PsiPowersOffset + PsiBytes;
    static constexpr size_t ForwardTwiddlesOffset = PsiInversePowersOffset + PsiBytes;
    static constexpr size_t InverseTwiddlesOffset = ForwardTwiddlesOffset + TwiddleBytes;
    static constexpr size_t CRealOffset = InverseTwiddlesOffset + TwiddleBytes;
    static constexpr size_t CImagOffset = CRealOffset + PsiBytes;
    static constexpr size_t OneOffset = CImagOffset + PsiBytes;
    static constexpr size_t NinvmOffset =
        OneOffset + (SharkFloatParams::EnableNewtonRaphson ? PsiBytes : 0u);
    static constexpr size_t PayloadBytes = NinvmOffset + NinvmBytes;
    static constexpr size_t PayloadOffset = Align(sizeof(Reference2CacheHeader), PayloadAlignment);
    static constexpr size_t FileBytes = PayloadOffset + PayloadBytes;
};

inline uint64_t
Checksum(const uint8_t *data, size_t bytes)
{
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

template <class SharkFloatParams>
std::wstring
CachePath(int64_t testNumber, uint32_t sequence, uint32_t actualPrecisionLimbs)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    std::wostringstream name;
    name << L"Ref2Prepared-v" << Reference2CacheVersion << L"-p" << SharkFloatParams::GlobalNumUint32
         << L"-nr" << (SharkFloatParams::EnableNewtonRaphson ? 1 : 0) << L"-a" << actualPrecisionLimbs
         << L"-test" << testNumber << L"-iter" << sequence << L"-min" << Workspace::MinFusedN << L".bin";
    return name.str();
}

template <class SharkFloatParams>
Reference2CacheHeader
MakeHeader(int64_t testNumber, uint32_t sequence, uint32_t actualPrecisionLimbs, uint64_t checksum)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    Reference2CacheHeader header{};
    std::copy(CacheMagic.begin(), CacheMagic.end(), std::begin(header.Magic));
    header.Version = Reference2CacheVersion;
    header.HeaderBytes = static_cast<uint32_t>(CacheLayout<SharkFloatParams>::PayloadOffset);
    header.StoragePrecisionLimbs = SharkFloatParams::GlobalNumUint32;
    header.ActualPrecisionLimbs = actualPrecisionLimbs;
    header.EnableNewtonRaphson = SharkFloatParams::EnableNewtonRaphson ? 1u : 0u;
    header.MinFusedN = Workspace::MinFusedN;
    header.MaxFusedN = Workspace::MaxFusedN;
    header.MaxFusedStages = Workspace::MaxFusedStages;
    header.PlanCacheEntryCount = Workspace::PlanCacheEntryCount;
    header.PayloadBytes = CacheLayout<SharkFloatParams>::PayloadBytes;
    header.PayloadChecksum = checksum;
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
               size_t mappedBytes)
{
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const auto magic = std::array<char, 8>{header.Magic[0],
                                           header.Magic[1],
                                           header.Magic[2],
                                           header.Magic[3],
                                           header.Magic[4],
                                           header.Magic[5],
                                           header.Magic[6],
                                           header.Magic[7]};
    if (magic != CacheMagic || header.Version != Reference2CacheVersion ||
        header.HeaderBytes != CacheLayout<SharkFloatParams>::PayloadOffset ||
        header.StoragePrecisionLimbs != SharkFloatParams::GlobalNumUint32 ||
        header.ActualPrecisionLimbs != actualPrecisionLimbs ||
        header.EnableNewtonRaphson != (SharkFloatParams::EnableNewtonRaphson ? 1u : 0u) ||
        header.MinFusedN != Workspace::MinFusedN || header.MaxFusedN != Workspace::MaxFusedN ||
        header.MaxFusedStages != Workspace::MaxFusedStages ||
        header.PlanCacheEntryCount != Workspace::PlanCacheEntryCount ||
        header.PayloadBytes != CacheLayout<SharkFloatParams>::PayloadBytes ||
        header.TestNumber != testNumber || header.Sequence != sequence ||
        mappedBytes != CacheLayout<SharkFloatParams>::FileBytes) {
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
    const typename Reference2PreparedTables<SharkFloatParams>::Workspace &workspace, uint8_t *payload)
{
    using Layout = CacheLayout<SharkFloatParams>;
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::StageOmegasOffset,
                                        workspace.StageOmegas,
                                        Layout::StageBytes,
                                        "cudaMemcpy(Ref2 cache stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::StageOmegasInverseOffset,
                                        workspace.StageOmegasInverse,
                                        Layout::StageBytes,
                                        "cudaMemcpy(Ref2 cache inverse stage omegas D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::PsiPowersOffset,
                                        workspace.PsiPowersArena,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache psi powers D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::PsiInversePowersOffset,
                                        workspace.PsiInversePowersArena,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache inverse psi powers D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::ForwardTwiddlesOffset,
                                        workspace.ForwardTwiddles,
                                        Layout::TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache forward twiddles D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::InverseTwiddlesOffset,
                                        workspace.InverseTwiddles,
                                        Layout::TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache inverse twiddles D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::CRealOffset,
                                        workspace.CRealArena,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache CReal spectra D2H)");
    CopyDeviceToCache<SharkFloatParams>(payload + Layout::CImagOffset,
                                        workspace.CImagArena,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache CImag spectra D2H)");
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        CopyDeviceToCache<SharkFloatParams>(payload + Layout::OneOffset,
                                            workspace.OneArena,
                                            Layout::PsiBytes,
                                            "cudaMemcpy(Ref2 cache One spectra D2H)");
    }

    std::vector<uint64_t> ninvm(Workspace::PlanCacheEntryCount);
    for (uint32_t slot = 0; slot < ninvm.size(); ++slot)
        ninvm[slot] = workspace.PlanRoots[slot].Ninvm_mont;
    std::memcpy(payload + Layout::NinvmOffset, ninvm.data(), Layout::NinvmBytes);
}

template <class SharkFloatParams>
void
CopyCachePayloadToPrepared(const uint8_t *payload,
                           Reference2PreparedTables<SharkFloatParams> &prepared,
                           uint32_t actualPrecisionLimbs)
{
    using Layout = CacheLayout<SharkFloatParams>;
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    Workspace workspace{};
    Reference2SetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Ref2 cache descriptor D2H)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegas,
                                        payload + Layout::StageOmegasOffset,
                                        Layout::StageBytes,
                                        "cudaMemcpy(Ref2 cache stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.StageOmegasInverse,
                                        payload + Layout::StageOmegasInverseOffset,
                                        Layout::StageBytes,
                                        "cudaMemcpy(Ref2 cache inverse stage omegas H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.PsiPowersArena,
                                        payload + Layout::PsiPowersOffset,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache psi powers H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.PsiInversePowersArena,
                                        payload + Layout::PsiInversePowersOffset,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache inverse psi powers H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.ForwardTwiddles,
                                        payload + Layout::ForwardTwiddlesOffset,
                                        Layout::TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache forward twiddles H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.InverseTwiddles,
                                        payload + Layout::InverseTwiddlesOffset,
                                        Layout::TwiddleBytes,
                                        "cudaMemcpy(Ref2 cache inverse twiddles H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.CRealArena,
                                        payload + Layout::CRealOffset,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache CReal spectra H2D)");
    CopyCacheToDevice<SharkFloatParams>(workspace.CImagArena,
                                        payload + Layout::CImagOffset,
                                        Layout::PsiBytes,
                                        "cudaMemcpy(Ref2 cache CImag spectra H2D)");
    if constexpr (SharkFloatParams::EnableNewtonRaphson) {
        CopyCacheToDevice<SharkFloatParams>(workspace.OneArena,
                                            payload + Layout::OneOffset,
                                            Layout::PsiBytes,
                                            "cudaMemcpy(Ref2 cache One spectra H2D)");
    }

    std::vector<uint64_t> ninvm(Workspace::PlanCacheEntryCount);
    std::memcpy(ninvm.data(), payload + Layout::NinvmOffset, Layout::NinvmBytes);
    for (uint32_t slot = 0; slot < Workspace::PlanCacheEntryCount; ++slot)
        workspace.PlanRoots[slot].Ninvm_mont = ninvm[slot];
    workspace.ValidPlanMask =
        Workspace::PlanCacheEntryCount == 32u ? ~0u : (1u << Workspace::PlanCacheEntryCount) - 1u;
    workspace.GeneratedStages = Workspace::MaxFusedStages;
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
                            uint32_t actualPrecisionLimbs)
{
    using Layout = Reference2CacheDetail::CacheLayout<SharkFloatParams>;
    using Workspace = HpSharkReference2Workspace<SharkFloatParams>;
    const auto target =
        Reference2CacheDetail::CachePath<SharkFloatParams>(testNumber, sequence, actualPrecisionLimbs);
    const auto temporary = target + L".tmp";
    Environment::FileDelete(temporary.c_str());

    auto mapped = Reference2MappedCacheFile::CreateWrite(temporary.c_str(), Layout::FileBytes);
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Unable to create Ref2 prepared-table cache");
    uint8_t *payload = mapped->Data() + Layout::PayloadOffset;
    Workspace workspace{};
    Reference2SetupDetail::CheckCuda(
        cudaMemcpy(
            &workspace, prepared.GetDeviceDescriptor(), sizeof(workspace), cudaMemcpyDeviceToHost),
        "cudaMemcpy(Ref2 cache save descriptor D2H)");
    const uint32_t fullMask =
        Workspace::PlanCacheEntryCount == 32u ? ~0u : (1u << Workspace::PlanCacheEntryCount) - 1u;
    if (workspace.ValidPlanMask != fullMask)
        throw FractalSharkSeriousException("Cannot cache incomplete Ref2 prepared tables");
    Reference2CacheDetail::CopyPreparedPayloadToCache<SharkFloatParams>(workspace, payload);
    const uint64_t checksum = Reference2CacheDetail::Checksum(payload, Layout::PayloadBytes);
    auto header = Reference2CacheDetail::MakeHeader<SharkFloatParams>(
        testNumber, sequence, actualPrecisionLimbs, checksum);
    std::memcpy(mapped->Data(), &header, sizeof(header));
    mapped->Flush();
    mapped.reset();
    if (!Environment::FileRename(temporary.c_str(), target.c_str(), true))
        throw FractalSharkSeriousException("Unable to publish Ref2 prepared-table cache");
}

template <class SharkFloatParams>
std::unique_ptr<Reference2PreparedTables<SharkFloatParams>>
LoadHpSharkReference2Tables(const HpShark::LaunchParams &launchParams,
                            int64_t testNumber,
                            uint32_t sequence,
                            uint32_t actualPrecisionLimbs)
{
    (void)launchParams;
    using Layout = Reference2CacheDetail::CacheLayout<SharkFloatParams>;
    const auto path =
        Reference2CacheDetail::CachePath<SharkFloatParams>(testNumber, sequence, actualPrecisionLimbs);
    auto mapped = Reference2MappedCacheFile::OpenRead(path.c_str());
    if (mapped == nullptr)
        throw FractalSharkSeriousException("Ref2 prepared-table cache is unavailable");
    if (mapped->Size() < sizeof(Reference2CacheHeader))
        throw FractalSharkSeriousException("Ref2 prepared-table cache is truncated");
    Reference2CacheHeader header{};
    std::memcpy(&header, mapped->Data(), sizeof(header));
    Reference2CacheDetail::ValidateHeader<SharkFloatParams>(
        header, testNumber, sequence, actualPrecisionLimbs, mapped->Size());
    const auto *payload = mapped->Data() + Layout::PayloadOffset;
    if (Reference2CacheDetail::Checksum(payload, Layout::PayloadBytes) != header.PayloadChecksum)
        throw FractalSharkSeriousException("Ref2 prepared-table cache checksum mismatch");

    auto prepared =
        Reference2SetupDetail::AllocatePreparedTables<SharkFloatParams>(actualPrecisionLimbs);
    Reference2CacheDetail::CopyCachePayloadToPrepared<SharkFloatParams>(
        payload, *prepared, actualPrecisionLimbs);
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
    try {
        return LoadHpSharkReference2Tables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReference2Tables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs);
    try {
        SaveHpSharkReference2Tables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs);
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
    try {
        return LoadHpSharkReference2Tables<SharkFloatParams>(
            launchParams, testNumber, sequence, actualPrecisionLimbs);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache miss for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }

    auto prepared = PrepareHpSharkReference2Tables<SharkFloatParams>(
        launchParams, cReal, cImag, actualPrecisionLimbs);
    try {
        SaveHpSharkReference2Tables<SharkFloatParams>(
            *prepared, testNumber, sequence, actualPrecisionLimbs);
    } catch (const std::exception &error) {
        std::cout << "Ref2 cache save failed for test " << testNumber << " sequence " << sequence << ": "
                  << error.what() << std::endl;
    }
    return prepared;
}

} // namespace HpShark
