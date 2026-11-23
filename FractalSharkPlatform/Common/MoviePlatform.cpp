#include "MoviePlatform.h"
#include "../../HpSharkFloatLib/Exceptions.h"
#include "ConsoleLog.h"
#include "Environment.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <optional>
#include <random>
#include <system_error>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace Environment {
namespace {
std::atomic<bool> Interrupted{false};
std::atomic<bool> InterruptOwned{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void
InterruptMovie(int)
{
    Interrupted.store(true, std::memory_order_relaxed);
}

void
WatchMovieInterrupt(std::stop_token stop)
{
    // Match the GUI's three-second hold, with a monotonic clock so a slow poll
    // does not change the hold threshold. The latch is shared with Ctrl+C.
    constexpr auto holdDuration = std::chrono::seconds(3);
    constexpr auto pollInterval = std::chrono::milliseconds(100);
    std::optional<std::chrono::steady_clock::time_point> holdStarted;
    while (!stop.stop_requested() && !Interrupted.load(std::memory_order_relaxed)) {
        if (IsKeyDown(Key::Control) && IsKeyDown(Key::Alt)) {
            const auto now = std::chrono::steady_clock::now();
            if (!holdStarted) {
                holdStarted = now;
            }
            if (now - *holdStarted >= holdDuration) {
                InterruptMovie(0);
                break;
            }
        } else {
            holdStarted.reset();
        }
        std::this_thread::sleep_for(pollInterval);
    }
    if (Interrupted.load(std::memory_order_relaxed)) {
        FractalSharkLog::Write("Movie pause requested. Finishing active work.", __FILE__, __LINE__);
    }
}

#ifdef _WIN32
BOOL WINAPI
ConsoleInterrupt(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
        InterruptMovie(0);
        return TRUE;
    }
    return FALSE;
}
class NativeFile {
public:
    HANDLE Value = INVALID_HANDLE_VALUE;
    ~NativeFile() { Close(); }
    void
    Close()
    {
        if (Value != INVALID_HANDLE_VALUE && Value != nullptr) {
            CloseHandle(Value);
            Value = INVALID_HANDLE_VALUE;
        }
    }
};
[[noreturn]] void
SystemFailure(const char *operation)
{
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}
#else
class NativeFile {
public:
    int Value = -1;
    ~NativeFile() { Close(); }
    void
    Close()
    {
        if (Value >= 0) {
            close(Value);
            Value = -1;
        }
    }
};
[[noreturn]] void
SystemFailure(const char *operation)
{
    throw std::system_error(errno, std::generic_category(), operation);
}
#endif
} // namespace

MovieDriverLibrary::MovieDriverLibrary(MovieDriver driver)
{
#ifdef _WIN32
    const auto *name = driver == MovieDriver::Cuda ? L"nvcuda.dll" : L"nvEncodeAPI64.dll";
    m_Library = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    const auto *name = driver == MovieDriver::Cuda ? "libcuda.so.1" : "libnvidia-encode.so.1";
    m_Library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
    if (m_Library == nullptr) {
        throw FractalSharkSeriousException(driver == MovieDriver::Cuda
                                               ? "Cannot load the NVIDIA CUDA driver"
                                               : "Cannot load the NVIDIA NVENC driver");
    }
}

MovieDriverLibrary::~MovieDriverLibrary()
{
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(m_Library));
#else
    dlclose(m_Library);
#endif
}

void *
MovieDriverLibrary::Symbol(const char *name) const
{
#ifdef _WIN32
    const auto symbol = reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(m_Library), name));
#else
    const auto symbol = dlsym(m_Library, name);
#endif
    if (symbol == nullptr) {
        throw FractalSharkSeriousException(std::string("NVIDIA driver entry point is missing: ") + name);
    }
    return symbol;
}

struct MovieInterruptScope::Impl {
    ~Impl();

    bool m_OwnsHandlers = false;
    std::jthread m_KeyMonitor;
#ifdef _WIN32
    decltype(&InterruptMovie) m_Previous = nullptr;
#else
    struct sigaction m_PreviousInt{};
    struct sigaction m_PreviousTerm{};
#endif
};

MovieInterruptScope::MovieInterruptScope() : m_Impl(std::make_unique<Impl>())
{
    if (InterruptOwned.exchange(true)) {
        throw std::system_error(std::make_error_code(std::errc::device_or_resource_busy),
                                "movie interrupt subscription already owned");
    }
    Interrupted.store(false, std::memory_order_relaxed);
#ifdef _WIN32
    if (!SetConsoleCtrlHandler(ConsoleInterrupt, TRUE)) {
        InterruptOwned.store(false);
        SystemFailure("install movie console handler");
    }
    m_Impl->m_Previous = std::signal(SIGINT, InterruptMovie);
    if (m_Impl->m_Previous == SIG_ERR) {
        SetConsoleCtrlHandler(ConsoleInterrupt, FALSE);
        InterruptOwned.store(false);
        SystemFailure("install movie interrupt handler");
    }
#else
    // sigaction retains the subscription across repeated Ctrl+C requests.
    struct sigaction action{};
    action.sa_handler = InterruptMovie;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &m_Impl->m_PreviousInt) != 0) {
        InterruptOwned.store(false);
        SystemFailure("install movie interrupt handler");
    }
    if (sigaction(SIGTERM, &action, &m_Impl->m_PreviousTerm) != 0) {
        const auto error = errno;
        sigaction(SIGINT, &m_Impl->m_PreviousInt, nullptr);
        InterruptOwned.store(false);
        throw std::system_error(error, std::generic_category(), "install movie termination handler");
    }
#endif
    // Impl also restores the handlers if creating the monitor thread throws.
    m_Impl->m_OwnsHandlers = true;
    m_Impl->m_KeyMonitor = std::jthread(WatchMovieInterrupt);
}

MovieInterruptScope::~MovieInterruptScope() = default;

MovieInterruptScope::Impl::~Impl()
{
    if (!m_OwnsHandlers) {
        return;
    }
    // Join before releasing the process-wide latch so the outgoing monitor
    // cannot request a pause for the next movie subscription.
    m_KeyMonitor.request_stop();
    if (m_KeyMonitor.joinable()) {
        m_KeyMonitor.join();
    }
#ifdef _WIN32
    std::signal(SIGINT, m_Previous);
    SetConsoleCtrlHandler(ConsoleInterrupt, FALSE);
#else
    sigaction(SIGTERM, &m_PreviousTerm, nullptr);
    sigaction(SIGINT, &m_PreviousInt, nullptr);
#endif
    InterruptOwned.store(false);
}

bool
MovieInterruptScope::StopRequested() const
{
    return Interrupted.load(std::memory_order_relaxed);
}

struct MovieJobLock::Impl {
    NativeFile File;
};

MovieJobLock::MovieJobLock(const std::filesystem::path &path) : m_Impl(std::make_unique<Impl>())
{
#ifdef _WIN32
    m_Impl->File.Value = CreateFileW(path.c_str(),
                                     GENERIC_READ | GENERIC_WRITE,
                                     0,
                                     nullptr,
                                     OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL,
                                     nullptr);
    if (m_Impl->File.Value == INVALID_HANDLE_VALUE) {
        SystemFailure("lock movie job (another process may own it)");
    }
#else
    m_Impl->File.Value = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (m_Impl->File.Value < 0 || flock(m_Impl->File.Value, LOCK_EX | LOCK_NB) != 0) {
        SystemFailure("lock movie job (another process may own it)");
    }
#endif
}
MovieJobLock::~MovieJobLock() = default;

std::filesystem::path
ReserveMoviePartialFile(const std::filesystem::path &directory)
{
    std::random_device random;
    constexpr unsigned ReservationAttempts = 16;
    for (unsigned attempt = 0; attempt < ReservationAttempts; ++attempt) {
        const auto path = directory / (".fractalshark-" + std::to_string(random()) + "-" +
                                       std::to_string(random()) + ".ts.partial");
        NativeFile file;
#ifdef _WIN32
        file.Value = CreateFileW(
            path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file.Value == INVALID_HANDLE_VALUE) {
            const auto error = GetLastError();
            if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
                continue;
            }
            SystemFailure("reserve movie staging file");
        }
#else
        file.Value = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (file.Value < 0) {
            if (errno == EEXIST) {
                continue;
            }
            SystemFailure("reserve movie staging file");
        }
#endif
        file.Close();
        FlushMovieFile(path);
        return path;
    }
    throw FractalSharkSeriousException("Unable to reserve a unique movie staging file");
}

void
FlushMovieFile(const std::filesystem::path &path)
{
    NativeFile file;
#ifdef _WIN32
    file.Value = CreateFileW(path.c_str(),
                             GENERIC_WRITE,
                             FILE_SHARE_READ,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (file.Value == INVALID_HANDLE_VALUE || !FlushFileBuffers(file.Value)) {
        SystemFailure("flush movie output");
    }
#else
    file.Value = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (file.Value < 0 || fsync(file.Value) != 0) {
        SystemFailure("flush movie output");
    }
    NativeFile directory;
    directory.Value = open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory.Value < 0 || fsync(directory.Value) != 0) {
        SystemFailure("flush movie frame directory");
    }
#endif
}

void
PublishMovieFile(const std::filesystem::path &temporary,
                 const std::filesystem::path &target,
                 bool replace)
{
    FlushMovieFile(temporary);
#ifdef _WIN32
    const DWORD flags = MOVEFILE_WRITE_THROUGH | (replace ? MOVEFILE_REPLACE_EXISTING : 0);
    if (!MoveFileExW(temporary.c_str(), target.c_str(), flags)) {
        SystemFailure("publish movie output");
    }
#else
    NativeFile directory;
    directory.Value = open(target.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory.Value < 0) {
        SystemFailure("open movie output directory");
    }
    if (replace) {
        if (rename(temporary.c_str(), target.c_str()) != 0) {
            SystemFailure("publish movie output");
        }
    } else {
        if (link(temporary.c_str(), target.c_str()) != 0) {
            SystemFailure("publish movie output");
        }
        // The assembled movie may be the only payload copy. Make the new
        // name durable before unlinking the old one, even on a power loss.
        if (fsync(directory.Value) != 0) {
            SystemFailure("flush published movie link");
        }
        if (unlink(temporary.c_str()) != 0) {
            SystemFailure("remove published movie temporary");
        }
    }
    if (fsync(directory.Value) != 0) {
        SystemFailure("flush movie output directory");
    }
#endif
}

} // namespace Environment
