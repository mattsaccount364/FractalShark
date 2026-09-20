#pragma once

#include <cstdint>
#include <memory>

namespace Environment {

class JobObject {
public:
    JobObject();
    ~JobObject() noexcept;

    JobObject &operator=(const JobObject &) = delete;
    JobObject(const JobObject &) = delete;
    JobObject &operator=(JobObject &&) = delete;
    JobObject(JobObject &&) = delete;

    uint64_t GetCommitLimitInBytes() const noexcept;

private:
    class JobObjectImpl;
    std::unique_ptr<JobObjectImpl> m_Impl;
};

} // namespace Environment
