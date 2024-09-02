#pragma once

#include <memory>

class JobObject {
public:
    JobObject();
    ~JobObject();

    JobObject &operator=(const JobObject &) = delete;
    JobObject(const JobObject &) = delete;

    uint64_t GetCommitLimitInBytes() const;

private:
    // PIMPL
    class JobObjectImpl;
    std::unique_ptr<JobObjectImpl> impl;
};