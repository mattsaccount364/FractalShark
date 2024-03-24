#include "StdAfx.h"

#include "JobObject.h"

JobObject::JobObject()
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
    constexpr size_t EightGb = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    const size_t Limit1 = totalPhysicalMemoryInKb * 1024ULL / 2ULL;

    // Another option:  limit to physical memory minus 8 GB
    size_t Limit2;
    if (totalPhysicalMemoryInKb < EightGb / 1024) {
        Limit2 = 0;
    }
    else {
        Limit2 = totalPhysicalMemoryInKb * 1024ULL - EightGb;
    }

    // Choose whichever limit is smaller
    const size_t Limit = Limit1 < Limit2 ? Limit1 : Limit2;

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = { 0 };
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

JobObject::~JobObject()
{
    if (hJob != nullptr) {
        CloseHandle(hJob);
    }
}
