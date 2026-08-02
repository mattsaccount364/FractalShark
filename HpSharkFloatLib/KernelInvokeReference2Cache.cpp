#include "KernelInvokeReference2Cache.h"

namespace HpShark {

Reference2MappedCacheFile::Reference2MappedCacheFile(std::unique_ptr<Environment::MappedFile> mappedFile)
    : m_MappedFile{std::move(mappedFile)}
{
}

Reference2MappedCacheFile::~Reference2MappedCacheFile() = default;

std::unique_ptr<Reference2MappedCacheFile>
Reference2MappedCacheFile::CreateWrite(const wchar_t *path, size_t bytes)
{
    auto mappedFile = Environment::MappedFile::CreateWrite(path, bytes);
    if (mappedFile == nullptr)
        return nullptr;
    return std::unique_ptr<Reference2MappedCacheFile>(
        new Reference2MappedCacheFile{std::move(mappedFile)});
}

std::unique_ptr<Reference2MappedCacheFile>
Reference2MappedCacheFile::OpenRead(const wchar_t *path)
{
    auto mappedFile = Environment::MappedFile::OpenRead(path);
    if (mappedFile == nullptr)
        return nullptr;
    return std::unique_ptr<Reference2MappedCacheFile>(
        new Reference2MappedCacheFile{std::move(mappedFile)});
}

uint8_t *
Reference2MappedCacheFile::Data()
{
    return m_MappedFile->Data();
}

const uint8_t *
Reference2MappedCacheFile::Data() const
{
    return m_MappedFile->Data();
}

size_t
Reference2MappedCacheFile::Size() const
{
    return m_MappedFile->Size();
}

void
Reference2MappedCacheFile::Flush()
{
    if (!m_MappedFile->Flush())
        throw FractalSharkSeriousException("Unable to flush Ref2 prepared-table cache");
}

} // namespace HpShark
