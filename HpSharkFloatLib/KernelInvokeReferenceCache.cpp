#include "KernelInvokeReferenceCache.h"

#include "HpSharkFloat.h"
#include "KernelInvoke.h"

#include <cuda_runtime.h>

namespace HpShark {

bool
SupportsReferenceSharedOnlyMemory(uint32_t requestedBlocks)
{
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;

    int maxOptinSharedBytes = 0;
    if (cudaDeviceGetAttribute(&maxOptinSharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) !=
        cudaSuccess) {
        return false;
    }
    const int requiredSharedBytes =
        static_cast<int>(ReferenceSharedOnlyDynamicBytes + ReferenceCarryPrefixSharedBytes);
    if (maxOptinSharedBytes < requiredSharedBytes)
        return false;

    int maxSharedBytesPerMultiprocessor = 0;
    if (cudaDeviceGetAttribute(&maxSharedBytesPerMultiprocessor,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               device) != cudaSuccess) {
        return false;
    }
    if (maxSharedBytesPerMultiprocessor < requiredSharedBytes) {
        return false;
    }

    if (requestedBlocks == 0)
        return true;

    int multiprocessorCount = 0;
    if (cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device) !=
        cudaSuccess) {
        return false;
    }
    return requestedBlocks <= static_cast<uint32_t>(multiprocessorCount);
}

ReferenceMappedCacheFile::ReferenceMappedCacheFile(std::unique_ptr<Environment::MappedFile> mappedFile)
    : m_MappedFile{std::move(mappedFile)}
{
}

ReferenceMappedCacheFile::~ReferenceMappedCacheFile() = default;

std::unique_ptr<ReferenceMappedCacheFile>
ReferenceMappedCacheFile::CreateWrite(const wchar_t *path, size_t bytes)
{
    auto mappedFile = Environment::MappedFile::CreateWrite(path, bytes);
    if (mappedFile == nullptr)
        return nullptr;
    return std::unique_ptr<ReferenceMappedCacheFile>(
        new ReferenceMappedCacheFile{std::move(mappedFile)});
}

std::unique_ptr<ReferenceMappedCacheFile>
ReferenceMappedCacheFile::OpenRead(const wchar_t *path)
{
    auto mappedFile = Environment::MappedFile::OpenRead(path);
    if (mappedFile == nullptr)
        return nullptr;
    return std::unique_ptr<ReferenceMappedCacheFile>(
        new ReferenceMappedCacheFile{std::move(mappedFile)});
}

uint8_t *
ReferenceMappedCacheFile::Data()
{
    return m_MappedFile->Data();
}

const uint8_t *
ReferenceMappedCacheFile::Data() const
{
    return m_MappedFile->Data();
}

size_t
ReferenceMappedCacheFile::Size() const
{
    return m_MappedFile->Size();
}

void
ReferenceMappedCacheFile::Flush()
{
    if (!m_MappedFile->Flush())
        throw FractalSharkSeriousException("Unable to flush Reference prepared-table cache");
}

} // namespace HpShark
