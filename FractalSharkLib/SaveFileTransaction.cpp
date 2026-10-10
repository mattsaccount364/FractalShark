#include "stdafx.h"

#include "Environment.h"
#include "SaveFileTransaction.h"

#include <atomic>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace {
class TemporaryFile {
public:
    std::filesystem::path m_Path;
    std::ofstream m_Stream;
    bool m_Owned = false;

    TemporaryFile() = default;
    TemporaryFile(const TemporaryFile &) = delete;
    TemporaryFile &operator=(const TemporaryFile &) = delete;
    ~TemporaryFile()
    {
        // Close before removal (Windows cannot remove an open output stream).
        if (m_Stream.is_open()) {
            m_Stream.close();
        }
        if (m_Owned) {
            std::error_code ignored;
            std::filesystem::remove(m_Path, ignored);
        }
    }
};

SaveResult
Failure(const std::wstring &destination, const std::error_code &error, const char *operation)
{
    return {SaveOutcome::Failed,
            destination,
            error.value(),
            std::string(operation) + ": " + error.message()};
}
} // namespace

SaveResult
SaveFileTransaction::Write(const std::wstring &destination,
                           SaveOverwritePolicy overwrite,
                           const std::function<void(std::ostream &)> &writer)
{
    const std::filesystem::path target(Environment::ToFsPath(destination));
    TemporaryFile temporary;
    static std::atomic<uint64_t> sequence{0};
    for (;;) {
        temporary.m_Path = target.parent_path() /
                           (L".fractalshark-save-" + std::to_wstring(Environment::CurrentProcessId()) +
                            L"-" + std::to_wstring(sequence.fetch_add(1)) + L".tmp");
        errno = 0;
        temporary.m_Stream.open(temporary.m_Path,
                                std::ios::binary | std::ios::out | std::ios::noreplace);
        if (temporary.m_Stream.is_open()) {
            temporary.m_Owned = true;
            break;
        }
        const int openError = errno;
        if (openError != EEXIST) {
            return Failure(destination,
                           std::error_code(openError != 0 ? openError : EIO, std::generic_category()),
                           "create temporary file");
        }
        // Only name collisions are retried. Storage and encoder failures propagate.
        temporary.m_Stream.clear();
    }
    writer(temporary.m_Stream);
    if (!temporary.m_Stream) {
        return Failure(destination, std::make_error_code(std::errc::io_error), "write temporary file");
    }
    temporary.m_Stream.close();
    if (!temporary.m_Stream) {
        return Failure(destination, std::make_error_code(std::errc::io_error), "close temporary file");
    }
    std::error_code error;
    const auto publication = Environment::PublishFile(Environment::FromFsPath(temporary.m_Path).c_str(),
                                                      destination.c_str(),
                                                      overwrite == SaveOverwritePolicy::ReplaceExisting,
                                                      error);
    if (publication == Environment::FilePublication::Failed) {
        return Failure(destination, error, "publish output");
    }
    return {publication == Environment::FilePublication::AlreadyExists ? SaveOutcome::SkippedExisting
                                                                       : SaveOutcome::Written,
            destination,
            0,
            {}};
}
