#include "StdAfx.h"

#include "JobObject.h"

class JobObject::JobObjectImpl {
public:
    JobObjectImpl();
    ~JobObjectImpl();

    JobObjectImpl &operator=(const JobObjectImpl &) = delete;
    JobObjectImpl(const JobObjectImpl &) = delete;

    uint64_t GetCommitLimitInBytes() const;

private:
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli;
    HANDLE hJob;
};

JobObject::JobObjectImpl::JobObjectImpl() :
    jeli{},
    hJob{}
{
    // Use a Win32 job object to limit virtual memory used
    // by this process.

    hJob = CreateJobObject(nullptr, nullptr);
    if (hJob == nullptr) {
        ::MessageBox(nullptr, L"Failed to create job object", L"Error", MB_OK | MB_APPLMODAL);
        return;
    }

    // Get the system's total physical memory using
    // GetPhysicallyInstalledSystemMemory
    ULONGLONG totalPhysicalMemoryInKb = 0;
    GetPhysicallyInstalledSystemMemory(&totalPhysicalMemoryInKb);

    // One option:  limit to half the physical memory
    constexpr size_t EightGbInBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const size_t Limit1InBytes = totalPhysicalMemoryInKb * 1024ULL / 2ULL;

    // Another option:  limit to physical memory minus 8 GB
    size_t Limit2InBytes;
    if (totalPhysicalMemoryInKb < EightGbInBytes / 1024) {
        Limit2InBytes = 0;
    } else {
        Limit2InBytes = totalPhysicalMemoryInKb * 1024ULL - EightGbInBytes;
    }

    // Choose whichever limit is smaller
    const size_t Limit = Limit1InBytes < Limit2InBytes ? Limit1InBytes : Limit2InBytes;

    jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY;
    jeli.JobMemoryLimit = Limit;
    if (!SetInformationJobObject(hJob, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli))) {
        ::MessageBox(nullptr, L"Failed to set job object information", L"Error", MB_OK | MB_APPLMODAL);
        CloseHandle(hJob);
        return;
    }

    if (!AssignProcessToJobObject(hJob, GetCurrentProcess())) {
        ::MessageBox(nullptr, L"Failed to assign process to job object", L"Error", MB_OK | MB_APPLMODAL);
        CloseHandle(hJob);
        return;
    }
}

JobObject::JobObjectImpl::~JobObjectImpl() {
    if (hJob != nullptr) {
        CloseHandle(hJob);
    }
}

uint64_t JobObject::JobObjectImpl::GetCommitLimitInBytes() const {
    return jeli.JobMemoryLimit;
}

JobObject::JobObject() : impl{std::make_unique<JobObjectImpl>()} {
}

JobObject::~JobObject() {
}

uint64_t JobObject::GetCommitLimitInBytes() const {
    return impl->GetCommitLimitInBytes();
}