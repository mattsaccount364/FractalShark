// Linux (POSIX) implementation of the Environment namespace.
// This file is only compiled on Linux builds.

#include "Environment.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <iostream>
#include <string>

#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <malloc.h> // malloc_usable_size
#include <pthread.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <termios.h>
#include <unistd.h>

#include <immintrin.h> // _mm_pause (x86 Linux)

namespace Environment {

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

// Command line storage (set once at program startup).
static const wchar_t *g_commandLine = L"";

class X11KeyState {
public:
    ~X11KeyState()
    {
        if (m_Display) {
            ::XCloseDisplay(m_Display);
        }
    }

    bool
    IsKeyDown(Environment::Key key)
    {
        if (!EnsureDisplay()) {
            return false;
        }

        char keyMap[32]{};
        ::XQueryKeymap(m_Display, keyMap);

        switch (key) {
            case Environment::Key::Control:
                return IsKeySymDown(keyMap, XK_Control_L) || IsKeySymDown(keyMap, XK_Control_R);
            case Environment::Key::Alt:
                return IsKeySymDown(keyMap, XK_Alt_L) || IsKeySymDown(keyMap, XK_Alt_R);
            case Environment::Key::Escape:
                return IsKeySymDown(keyMap, XK_Escape);
        }
        return false;
    }

private:
    bool
    EnsureDisplay()
    {
        if (!m_Initialized) {
            m_Display = ::XOpenDisplay(nullptr);
            m_Initialized = true;
        }
        return m_Display != nullptr;
    }

    bool
    IsKeySymDown(const char *keyMap, KeySym keySym) const
    {
        const KeyCode keyCode = ::XKeysymToKeycode(m_Display, keySym);
        if (keyCode == 0) {
            return false;
        }
        return (static_cast<unsigned char>(keyMap[keyCode >> 3]) & (1U << (keyCode & 7))) != 0;
    }

    Display *m_Display = nullptr;
    bool m_Initialized = false;
};

thread_local X11KeyState g_X11KeyState;

struct ConsoleRawGuard {
    termios original{};
    bool valid = false;

    ConsoleRawGuard()
    {
        if (!::isatty(STDIN_FILENO)) {
            return;
        }
        if (::tcgetattr(STDIN_FILENO, &original) != 0) {
            return;
        }

        termios raw = original;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        valid = (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0);
    }

    ~ConsoleRawGuard()
    {
        if (valid) {
            ::tcsetattr(STDIN_FILENO, TCSANOW, &original);
        }
    }
};

ConsoleRawGuard &
GetConsoleRawGuard()
{
    static ConsoleRawGuard guard;
    return guard;
}

int g_ConsolePending = -1;

} // anonymous namespace

} // namespace Environment

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
    // Read VmSize from /proc/self/status — total committed virtual memory,
    // which is the closest Linux equivalent to Win32 PagefileUsage.
    FILE *f = ::fopen("/proc/self/status", "r");
    if (f == nullptr) {
        return 0;
    }

    uint64_t vmSizeKB = 0;
    char line[256];
    while (::fgets(line, sizeof(line), f) != nullptr) {
        if (::strncmp(line, "VmSize:", 7) == 0) {
            ::sscanf(line + 7, " %lu", &vmSizeKB);
            break;
        }
    }
    ::fclose(f);
    return vmSizeKB * 1024;
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
            const char *colon = ::strchr(line, ':');
            if (colon != nullptr) {
                ::sscanf(colon + 1, " %d", &siblings);
            }
        } else if (::strncmp(line, "cpu cores", 9) == 0 && cores == 0) {
            const char *colon = ::strchr(line, ':');
            if (colon != nullptr) {
                ::sscanf(colon + 1, " %d", &cores);
            }
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
    if (msg == nullptr) {
        msg = "<null>";
    }

    size_t len = 0;
    while (msg[len] != '\0') {
        ++len;
    }

    while (len != 0) {
        const ssize_t written = ::write(STDERR_FILENO, msg, len);
        if (written <= 0) {
            return;
        }

        msg += written;
        len -= static_cast<size_t>(written);
    }
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
Environment::IsKeyDown(Key key)
{
    return g_X11KeyState.IsKeyDown(key);
}

std::pair<int, int>
Environment::GetCursorPosition()
{
    Display *display = ::XOpenDisplay(nullptr);
    if (!display) {
        return {0, 0};
    }

    Window rootReturn = 0;
    Window childReturn = 0;
    int rootX = 0;
    int rootY = 0;
    int windowX = 0;
    int windowY = 0;
    unsigned int mask = 0;

    const Window root = DefaultRootWindow(display);
    const Bool ok = ::XQueryPointer(
        display, root, &rootReturn, &childReturn, &rootX, &rootY, &windowX, &windowY, &mask);
    ::XCloseDisplay(display);

    if (!ok) {
        return {0, 0};
    }
    return {rootX, rootY};
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

bool
Environment::ConsoleKeyAvailable()
{
    auto &guard = GetConsoleRawGuard();
    if (!guard.valid) {
        return false;
    }

    if (g_ConsolePending != -1) {
        return true;
    }

    unsigned char c = 0;
    const ssize_t n = ::read(STDIN_FILENO, &c, 1);
    if (n == 1) {
        g_ConsolePending = c;
        return true;
    }

    return false;
}

int
Environment::ConsoleReadCharBlocking()
{
    GetConsoleRawGuard();

    if (g_ConsolePending != -1) {
        const int c = g_ConsolePending;
        g_ConsolePending = -1;
        return c;
    }

    unsigned char c = 0;
    termios current{};
    bool changed = false;
    if (::tcgetattr(STDIN_FILENO, &current) == 0) {
        termios blocking = current;
        blocking.c_cc[VMIN] = 1;
        blocking.c_cc[VTIME] = 0;
        changed = (::tcsetattr(STDIN_FILENO, TCSANOW, &blocking) == 0);
    }

    const ssize_t n = ::read(STDIN_FILENO, &c, 1);

    if (changed) {
        ::tcsetattr(STDIN_FILENO, TCSANOW, &current);
    }

    return (n == 1) ? static_cast<int>(c) : -1;
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

std::optional<uint64_t>
Environment::FileSizeBytes(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    struct stat st{};
    if (::stat(u8.c_str(), &st) != 0) {
        return std::nullopt;
    }
    if (!S_ISREG(st.st_mode)) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(st.st_size);
}

// =========================================================================
// Process information
// =========================================================================

uint64_t
Environment::CurrentProcessId()
{
    return static_cast<uint64_t>(::getpid());
}

std::wstring
Environment::TempDirectoryPath()
{
    const char *env = ::getenv("TMPDIR");
    std::string narrow = (env != nullptr && *env != '\0') ? std::string(env) : std::string("/tmp");
    if (narrow.back() != '/') {
        narrow.push_back('/');
    }
    std::wstring wide;
    wide.reserve(narrow.size());
    for (char c : narrow) {
        wide.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    }
    return wide;
}

// =========================================================================
// System information (extended)
// =========================================================================

uint32_t
Environment::GetMemoryLoad()
{
    // Parse /proc/meminfo to compute memory usage percentage.
    FILE *f = ::fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }

    uint64_t memTotal = 0;
    uint64_t memAvailable = 0;
    bool gotTotal = false;
    bool gotAvailable = false;

    char line[256];
    while (::fgets(line, sizeof(line), f)) {
        if (!gotTotal && ::strncmp(line, "MemTotal:", 9) == 0) {
            memTotal = ::strtoull(line + 9, nullptr, 10);
            gotTotal = true;
        } else if (!gotAvailable && ::strncmp(line, "MemAvailable:", 13) == 0) {
            memAvailable = ::strtoull(line + 13, nullptr, 10);
            gotAvailable = true;
        }
        if (gotTotal && gotAvailable) {
            break;
        }
    }
    ::fclose(f);

    if (!gotTotal || memTotal == 0) {
        return 0;
    }

    uint64_t used = (memTotal > memAvailable) ? (memTotal - memAvailable) : 0;
    return static_cast<uint32_t>(used * 100 / memTotal);
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

namespace Environment {

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

} // namespace Environment

bool
Environment::DirectoryRemoveRecursive(const wchar_t *path)
{
    std::string u8 = WideToUtf8(path);
    // FTW_DEPTH: process directory contents before the directory itself.
    // FTW_PHYS: do not follow symlinks.
    return ::nftw(u8.c_str(), NftwRemoveCallback, 64, FTW_DEPTH | FTW_PHYS) == 0;
}

// =========================================================================
// UI helpers
// =========================================================================

void
Environment::ShowWarning(const wchar_t *message)
{
    std::wcerr << L"Warning: " << message << std::endl;
}

bool
Environment::SetClipboardText(std::string_view text)
{
    (void)text;
    return true;
}

void
Environment::PumpUIEvents()
{
    // TODO: Linux, should we be checking the state of things etc here at all
}
