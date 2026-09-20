#include "ConsoleLog.h"
#include "JobObject.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace Environment {

class JobObject::JobObjectImpl {
public:
    JobObjectImpl();
    ~JobObjectImpl() noexcept;

    JobObjectImpl &operator=(const JobObjectImpl &) = delete;
    JobObjectImpl(const JobObjectImpl &) = delete;

    uint64_t GetCommitLimitInBytes() const;

private:
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION m_ExtendedLimitInfo{};
    HANDLE m_JobHandle = nullptr;
};

JobObject::JobObjectImpl::JobObjectImpl()
{
    // Use a Win32 job object to limit virtual memory used by this process.
    m_JobHandle = CreateJobObject(nullptr, nullptr);
    if (m_JobHandle == nullptr) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << L"Failed to create job object";
        return;
    }

    ULONGLONG totalPhysicalMemoryInKb = 0;
    GetPhysicallyInstalledSystemMemory(&totalPhysicalMemoryInKb);

    constexpr uint64_t EightGbInBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const uint64_t limit1InBytes = totalPhysicalMemoryInKb * 1024ULL / 2ULL;

    uint64_t limit2InBytes = 0;
    if (totalPhysicalMemoryInKb >= EightGbInBytes / 1024ULL) {
        limit2InBytes = totalPhysicalMemoryInKb * 1024ULL - EightGbInBytes;
    }

    const uint64_t limit = limit1InBytes < limit2InBytes ? limit1InBytes : limit2InBytes;

    m_ExtendedLimitInfo.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
    m_ExtendedLimitInfo.JobMemoryLimit = limit;
    if (!SetInformationJobObject(m_JobHandle,
                                 JobObjectExtendedLimitInformation,
                                 &m_ExtendedLimitInfo,
                                 sizeof(m_ExtendedLimitInfo))) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << L"Failed to set job object information";
        return;
    }

    if (!AssignProcessToJobObject(m_JobHandle, GetCurrentProcess())) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << L"Failed to assign process to job object";
        return;
    }
}

JobObject::JobObjectImpl::~JobObjectImpl() noexcept
{
    if (m_JobHandle != nullptr) {
        CloseHandle(m_JobHandle);
    }
}

uint64_t
JobObject::JobObjectImpl::GetCommitLimitInBytes() const
{
    return m_ExtendedLimitInfo.JobMemoryLimit;
}

JobObject::JobObject() : m_Impl{std::make_unique<JobObjectImpl>()} {}

JobObject::~JobObject() noexcept = default;

uint64_t
JobObject::GetCommitLimitInBytes() const noexcept
{
    return m_Impl->GetCommitLimitInBytes();
}

} // namespace Environment
