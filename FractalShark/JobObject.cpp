#include "StdAfx.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "JobObject.h"
#include <iostream>

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

JobObject::JobObjectImpl::JobObjectImpl() : jeli{}, hJob{}
{
    // Use a Win32 job object to limit virtual memory used
    // by this process.

    hJob = CreateJobObject(nullptr, nullptr);
    if (hJob == nullptr) {
        std::wcerr << L"Failed to create job object" << std::endl;
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
        std::wcerr << L"Failed to set job object information" << std::endl;
        CloseHandle(hJob);
        return;
    }

    if (!AssignProcessToJobObject(hJob, GetCurrentProcess())) {
        std::wcerr << L"Failed to assign process to job object" << std::endl;
        CloseHandle(hJob);
        return;
    }
}

JobObject::JobObjectImpl::~JobObjectImpl()
{
    if (hJob != nullptr) {
        CloseHandle(hJob);
    }
}

uint64_t
JobObject::JobObjectImpl::GetCommitLimitInBytes() const
{
    return jeli.JobMemoryLimit;
}

#else // !_WIN32

#include "JobObject.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <sys/resource.h>

class JobObject::JobObjectImpl {
public:
    JobObjectImpl();
    ~JobObjectImpl() = default;

    JobObjectImpl &operator=(const JobObjectImpl &) = delete;
    JobObjectImpl(const JobObjectImpl &) = delete;

    uint64_t GetCommitLimitInBytes() const;

private:
    uint64_t m_LimitBytes = UINT64_MAX;
};

static uint64_t
ReadTotalPhysicalMemory()
{
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.compare(0, 9, "MemTotal:") == 0) {
            // Format: "MemTotal:     12345678 kB"
            uint64_t kb = 0;
            const char *p = line.c_str() + 9;
            while (*p == ' ') {
                ++p;
            }
            while (*p >= '0' && *p <= '9') {
                kb = kb * 10 + (*p - '0');
                ++p;
            }
            return kb * 1024ULL;
        }
    }
    return 0;
}

JobObject::JobObjectImpl::JobObjectImpl()
{
    uint64_t totalBytes = ReadTotalPhysicalMemory();
    if (totalBytes == 0) {
        return;
    }

    constexpr uint64_t EightGbInBytes = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const uint64_t limit1 = totalBytes / 2ULL;
    const uint64_t limit2 = totalBytes > EightGbInBytes ? totalBytes - EightGbInBytes : 0ULL;
    m_LimitBytes = limit1 < limit2 ? limit1 : limit2;

    struct rlimit rl{};
    rl.rlim_cur = m_LimitBytes;
    rl.rlim_max = m_LimitBytes;
    setrlimit(RLIMIT_AS, &rl);
}

uint64_t
JobObject::JobObjectImpl::GetCommitLimitInBytes() const
{
    return m_LimitBytes;
}

#endif // _WIN32

JobObject::JobObject() : impl{std::make_unique<JobObjectImpl>()} {}

JobObject::~JobObject() {}

uint64_t
JobObject::GetCommitLimitInBytes() const
{
    return impl->GetCommitLimitInBytes();
}