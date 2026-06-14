#include "JobObject.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <sys/resource.h>

namespace Environment {

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

namespace {

uint64_t
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
                kb = kb * 10 + static_cast<uint64_t>(*p - '0');
                ++p;
            }
            return kb * 1024ULL;
        }
    }
    return 0;
}

} // namespace

JobObject::JobObjectImpl::JobObjectImpl()
{
    const uint64_t totalBytes = ReadTotalPhysicalMemory();
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

JobObject::JobObject() : m_Impl{std::make_unique<JobObjectImpl>()} {}

JobObject::~JobObject() = default;

uint64_t
JobObject::GetCommitLimitInBytes() const
{
    return m_Impl->GetCommitLimitInBytes();
}

} // namespace Environment
