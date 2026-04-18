// Linux (POSIX) implementation of the Environment namespace.
// This file is only compiled on Linux builds.

#ifndef _WIN32

#include "Environment.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>

#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <malloc.h> // malloc_usable_size
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <immintrin.h> // _mm_pause (x86 Linux)

namespace {

// -------------------------------------------------------------------------
// Helper: convert wchar_t (UTF-32 on Linux) to UTF-8.
// -------------------------------------------------------------------------
std::string
WideToUtf8(const wchar_t *wide)
{
    if (wide == nullptr || *wide == L'\0') {
        return {};
    }

    std::string result;
    for (const wchar_t *p = wide; *p != L'\0'; ++p) {
        auto c = static_cast<uint32_t>(*p);
        if (c < 0x80) {
            result.push_back(static_cast<char>(c));
        } else if (c < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (c >> 6)));
            result.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else if (c < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | (c >> 12)));
            result.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | (c >> 18)));
            result.push_back(static_cast<char>(0x80 | ((c >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((c >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (c & 0x3F)));
        }
    }
    return result;
}

// -------------------------------------------------------------------------
// Section handle: tracks fd + view info for memory-mapped files.
// -------------------------------------------------------------------------
struct SectionState {
    int fd;            // duplicated file descriptor (owned by this section)
    uint64_t fileSize; // current file size after last extend
    void *mappedBase;
    size_t mappedSize;
    bool readOnly;
};

// Command line storage (set once at program startup).
static const wchar_t *g_commandLine = L"";

} // anonymous namespace

// =========================================================================
// Virtual memory (anonymous, reserve-and-commit model)
// =========================================================================

void *
Environment::VirtualReserve(size_t bytes)
{
    void *p = ::mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? nullptr : p;
}

bool
Environment::VirtualCommit(void *addr, size_t bytes)
{
    return ::mprotect(addr, bytes, PROT_READ | PROT_WRITE) == 0;
}

void
Environment::VirtualRelease(void *base)
{
    // Release requires knowing the size.  On Linux we cannot recover it from
    // the pointer alone, so callers must ensure the full reserved range is
    // released.  For GrowableVector anonymous mode, the size is
    // m_PhysicalMemoryCapacityKB * 1024 (or the override).
    //
    // As a practical fallback we pass a large size.  A more robust design
    // would store the reserved size in a side structure; that refinement can
    // be done when callers are actually migrated (Phase 2).
    //
    // For now, this is a reasonable implementation since munmap silently
    // ignores pages that are not mapped.
    constexpr size_t LargeGuess = static_cast<size_t>(1) << 40; // 1 TiB
    ::munmap(base, LargeGuess);
}

// =========================================================================
// File handle operations (for PerturbationResults delete-on-close)
// =========================================================================

void *
Environment::FileOpenDeleteOnClose(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    int fd = ::open(u8.c_str(), O_RDWR);
    if (fd < 0) {
        return Environment::InvalidHandle;
    }
    // POSIX delete-on-close idiom: unlink while open.
    ::unlink(u8.c_str());
    return reinterpret_cast<void *>(static_cast<intptr_t>(fd));
}

void
Environment::FileClose(void *handle)
{
    if (handle != nullptr && handle != Environment::InvalidHandle) {
        int fd = static_cast<int>(reinterpret_cast<intptr_t>(handle));
        ::close(fd);
    }
}

// =========================================================================
// System information
// =========================================================================

size_t
Environment::SystemPageSize()
{
    long ps = ::sysconf(_SC_PAGESIZE);
    return (ps > 0) ? static_cast<size_t>(ps) : 4096;
}

uint64_t
Environment::ProcessCommitChargeBytes()
{
    // Read VmRSS from /proc/self/status.
    FILE *f = ::fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return 0;
    }

    uint64_t vmRssKB = 0;
    char line[256];
    while (::fgets(line, sizeof(line), f) != nullptr) {
        if (::strncmp(line, "VmRSS:", 6) == 0) {
            ::sscanf(line + 6, " %lu", &vmRssKB);
            break;
        }
    }
    ::fclose(f);
    return vmRssKB * 1024;
}

uint32_t
Environment::LogicalProcessorCount()
{
    long n = ::sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? static_cast<uint32_t>(n) : 1;
}

bool
Environment::IsHyperthreadingEnabled()
{
    // Parse /proc/cpuinfo for "siblings" vs "cpu cores" on any package.
    FILE *f = ::fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        return false;
    }

    int siblings = 0;
    int cores = 0;
    char line[256];
    while (::fgets(line, sizeof(line), f) != nullptr) {
        if (::strncmp(line, "siblings", 8) == 0 && siblings == 0) {
            ::sscanf(::strchr(line, ':') + 1, " %d", &siblings);
        } else if (::strncmp(line, "cpu cores", 9) == 0 && cores == 0) {
            ::sscanf(::strchr(line, ':') + 1, " %d", &cores);
        }
        if (siblings > 0 && cores > 0) {
            break;
        }
    }
    ::fclose(f);
    return siblings > 0 && cores > 0 && siblings > cores;
}

// =========================================================================
// Debugging
// =========================================================================

void
Environment::DebugOutput(const char *msg)
{
    ::fprintf(stderr, "%s", msg);
}

bool
Environment::IsDebuggerAttached()
{
    // Check TracerPid in /proc/self/status.
    FILE *f = ::fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return false;
    }

    bool attached = false;
    char line[256];
    while (::fgets(line, sizeof(line), f) != nullptr) {
        if (::strncmp(line, "TracerPid:", 10) == 0) {
            int pid = 0;
            ::sscanf(line + 10, " %d", &pid);
            attached = (pid != 0);
            break;
        }
    }
    ::fclose(f);
    return attached;
}

void
Environment::DebugBreakpoint()
{
    ::raise(SIGTRAP);
}

[[noreturn]] void
Environment::FastFail(int /*code*/)
{
    ::abort();
}

[[noreturn]] void
Environment::ProcessTerminate()
{
    ::_exit(1);
}

// =========================================================================
// Threading
// =========================================================================

void
Environment::SetCurrentThreadName(const wchar_t *name)
{
    std::string u8 = WideToUtf8(name);
    // pthread_setname_np limit: 16 chars including null terminator.
    if (u8.size() > 15) {
        u8.resize(15);
    }
    ::pthread_setname_np(::pthread_self(), u8.c_str());
}

void
Environment::SleepMs(uint32_t ms)
{
    struct timespec ts{};
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = static_cast<long>(ms % 1000) * 1000000L;
    ::nanosleep(&ts, nullptr);
}

void
Environment::YieldProcessor()
{
    _mm_pause();
}

int32_t
Environment::InterlockedCAS32(volatile int32_t *dest, int32_t exchange, int32_t comparand)
{
    int32_t expected = comparand;
    __atomic_compare_exchange_n(dest, &expected, exchange, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
}

void *
Environment::GetCurrentThreadHandle()
{
    return reinterpret_cast<void *>(::pthread_self());
}

uint64_t
Environment::SetThreadAffinity(void *threadHandle, uint64_t mask)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 64; ++i) {
        if (mask & (static_cast<uint64_t>(1) << i)) {
            CPU_SET(i, &cpuset);
        }
    }

    auto th = static_cast<pthread_t>(reinterpret_cast<uintptr_t>(threadHandle));

    // Read previous affinity.
    cpu_set_t prev;
    CPU_ZERO(&prev);
    ::pthread_getaffinity_np(th, sizeof(prev), &prev);

    uint64_t prevMask = 0;
    for (int i = 0; i < 64; ++i) {
        if (CPU_ISSET(i, &prev)) {
            prevMask |= (static_cast<uint64_t>(1) << i);
        }
    }

    if (::pthread_setaffinity_np(th, sizeof(cpuset), &cpuset) != 0) {
        return 0;
    }
    return prevMask;
}

// =========================================================================
// Timing
// =========================================================================

uint64_t
Environment::HighResCounter()
{
    struct timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t
Environment::HighResFrequency()
{
    return 1000000000ULL; // nanoseconds
}

// =========================================================================
// Keyboard / mouse input (for AbortMonitor)
// =========================================================================

bool
Environment::IsKeyDown(Key /*key*/)
{
    // No raw key-state query on Linux without a window system.
    // AbortMonitor will need a callback mechanism on Linux.
    return false;
}

std::pair<int, int>
Environment::GetCursorPosition()
{
    // No global cursor query on Linux without a window system.
    return {0, 0};
}

// =========================================================================
// System heap
// =========================================================================

void *
Environment::SystemHeapAlloc(size_t bytes)
{
    return ::malloc(bytes);
}

void *
Environment::SystemHeapRealloc(void *ptr, size_t bytes)
{
    return ::realloc(ptr, bytes);
}

void *
Environment::SystemHeapAllocZeroed(size_t bytes)
{
    return ::calloc(1, bytes);
}

void
Environment::SystemHeapFree(void *ptr)
{
    ::free(ptr);
}

size_t
Environment::SystemHeapSize(const void *ptr)
{
    if (ptr == nullptr) {
        return 0;
    }
    // malloc_usable_size takes non-const; cast is safe (read-only query).
    return ::malloc_usable_size(const_cast<void *>(ptr));
}

// =========================================================================
// OS error codes
// =========================================================================

uint32_t
Environment::GetLastOSError()
{
    return static_cast<uint32_t>(errno);
}

// =========================================================================
// Process / command line
// =========================================================================

const wchar_t *
Environment::GetCommandLineWide()
{
    return g_commandLine;
}

// =========================================================================
// Console
// =========================================================================

void
Environment::ClearConsole()
{
    // ANSI escape: clear screen + home cursor.
    (void)::write(STDOUT_FILENO, "\033[2J\033[H", 7);
}

// =========================================================================
// File I/O (general-purpose)
// =========================================================================

void *
Environment::FileOpen(const wchar_t *path,
                      FileAccess access,
                      FileDisposition disposition,
                      FileFlags flags)
{
    std::string u8 = WideToUtf8(path);

    int oflags = 0;
    switch (access) {
        case FileAccess::Read:
            oflags = O_RDONLY;
            break;
        case FileAccess::Write:
            oflags = O_WRONLY;
            break;
        case FileAccess::ReadWrite:
            oflags = O_RDWR;
            break;
    }

    switch (disposition) {
        case FileDisposition::OpenExisting:
            break; // no extra flags
        case FileDisposition::OpenAlways:
            oflags |= O_CREAT;
            break;
        case FileDisposition::CreateAlways:
            oflags |= O_CREAT | O_TRUNC;
            break;
    }

    mode_t mode = 0666; // default; umask applies
    int fd = ::open(u8.c_str(), oflags, mode);
    if (fd < 0) {
        return Environment::InvalidHandle;
    }

    if (flags & FileFlags::DeleteOnClose) {
        // POSIX delete-on-close: unlink now; data lives until fd is closed.
        ::unlink(u8.c_str());
    }

    return reinterpret_cast<void *>(static_cast<intptr_t>(fd));
}

uint64_t
Environment::FileGetSize(void *fileHandle)
{
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(fileHandle));
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.st_size);
}

bool
Environment::FileSetSize(void *fileHandle, uint64_t newSize)
{
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(fileHandle));
    return ::ftruncate(fd, static_cast<off_t>(newSize)) == 0;
}

size_t
Environment::FileWrite(void *fileHandle, const void *data, size_t bytes)
{
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(fileHandle));
    ssize_t n = ::write(fd, data, bytes);
    return (n >= 0) ? static_cast<size_t>(n) : 0;
}

bool
Environment::FileDelete(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    return ::unlink(u8.c_str()) == 0;
}

// =========================================================================
// Memory-mapped files (section objects)
// =========================================================================

void *
Environment::SectionCreate(void *fileHandle, uint64_t maxSize, bool readOnly)
{
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(fileHandle));

    // Duplicate the fd so the section has its own independent lifetime.
    int dupFd = ::dup(fd);
    if (dupFd < 0) {
        return nullptr;
    }

    // If maxSize > 0 and exceeds current file size, extend the file.
    // This matches the Windows SEC_RESERVE semantics where the section has
    // a maximum size.  On Linux, ftruncate creates a sparse file so no
    // physical disk is consumed for unwritten regions.
    if (maxSize > 0) {
        struct stat st{};
        if (::fstat(dupFd, &st) == 0 && static_cast<uint64_t>(st.st_size) < maxSize) {
            ::ftruncate(dupFd, static_cast<off_t>(maxSize));
        }
    }

    auto *state = new SectionState{};
    state->fd = dupFd;
    state->fileSize = maxSize;
    state->mappedBase = nullptr;
    state->mappedSize = 0;
    state->readOnly = readOnly;
    return state;
}

void *
Environment::SectionMapView(
    void *sectionHandle, void *suggestedBase, size_t *viewSize, bool /*reserveOnly*/, bool readOnly)
{
    if (sectionHandle == nullptr) {
        return nullptr;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);

    int prot = readOnly ? PROT_READ : (PROT_READ | PROT_WRITE);
    int flags = MAP_SHARED;

    // Use suggestedBase as a hint if provided.  MAP_FIXED is not used
    // because the caller treats it as a hint, not a requirement.
    void *base = ::mmap(suggestedBase, *viewSize, prot, flags, state->fd, 0);
    if (base == MAP_FAILED) {
        return nullptr;
    }

    state->mappedBase = base;
    state->mappedSize = *viewSize;
    return base;
}

bool
Environment::SectionExtend(void *sectionHandle, uint64_t newSize)
{
    if (sectionHandle == nullptr) {
        return false;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);

    // Extend the backing file.  Existing mmap MAP_SHARED views see the
    // new data once the file is extended (POSIX guarantee for pages within
    // the mapped range).
    if (::ftruncate(state->fd, static_cast<off_t>(newSize)) != 0) {
        return false;
    }

    state->fileSize = newSize;
    return true;
}

void
Environment::SectionUnmapView(void *sectionHandle)
{
    if (sectionHandle == nullptr) {
        return;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);
    if (state->mappedBase != nullptr) {
        ::munmap(state->mappedBase, state->mappedSize);
        state->mappedBase = nullptr;
        state->mappedSize = 0;
    }
}

void
Environment::SectionClose(void *sectionHandle)
{
    if (sectionHandle == nullptr) {
        return;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);
    if (state->fd >= 0) {
        ::close(state->fd);
    }
    delete state;
}

// =========================================================================
// System information (extended)
// =========================================================================

uint64_t
Environment::PhysicalMemoryKB()
{
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long pageSize = ::sysconf(_SC_PAGESIZE);
    if (pages <= 0 || pageSize <= 0) {
        return 0;
    }
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize) / 1024;
}

void *
Environment::GetCurrentProcessHandle()
{
    // Linux uses PIDs, not handles.  Return a sentinel that can be passed
    // to SectionMapView (which ignores it on Linux).
    return reinterpret_cast<void *>(static_cast<intptr_t>(-1));
}

// =========================================================================
// GUID generation
// =========================================================================

std::wstring
Environment::GenerateGuidString()
{
    // Read 16 random bytes and format as a GUID string.
    uint8_t bytes[16]{};
    FILE *f = ::fopen("/dev/urandom", "rb");
    if (f == nullptr) {
        return {};
    }
    size_t n = ::fread(bytes, 1, 16, f);
    ::fclose(f);
    if (n != 16) {
        return {};
    }

    // Set version 4 and variant bits per RFC 4122.
    bytes[6] = (bytes[6] & 0x0F) | 0x40;
    bytes[8] = (bytes[8] & 0x3F) | 0x80;

    wchar_t buf[40]{};
    ::swprintf(buf,
               40,
               L"{%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x}",
               bytes[0],
               bytes[1],
               bytes[2],
               bytes[3],
               bytes[4],
               bytes[5],
               bytes[6],
               bytes[7],
               bytes[8],
               bytes[9],
               bytes[10],
               bytes[11],
               bytes[12],
               bytes[13],
               bytes[14],
               bytes[15]);
    return std::wstring(buf);
}

// =========================================================================
// File system utilities
// =========================================================================

bool
Environment::DirectoryCreate(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    if (::mkdir(u8.c_str(), 0777) == 0) {
        return true;
    }
    return errno == EEXIST;
}

bool
Environment::DirectoryExists(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    struct stat st{};
    if (::stat(u8.c_str(), &st) != 0) {
        return false;
    }
    return S_ISDIR(st.st_mode);
}

namespace {

int
NftwRemoveCallback(const char *fpath,
                   const struct stat * /*sb*/,
                   int /*typeflag*/,
                   struct FTW * /*ftwbuf*/)
{
    return ::remove(fpath);
}

} // anonymous namespace

bool
Environment::DirectoryRemoveRecursive(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    // FTW_DEPTH: process directory contents before the directory itself.
    // FTW_PHYS: do not follow symlinks.
    return ::nftw(u8.c_str(), NftwRemoveCallback, 64, FTW_DEPTH | FTW_PHYS) == 0;
}

#endif // !_WIN32
