//
// CrashHandlerLinux.cpp - Linux signal-based crash handler.
//
// Mirrors the Win32 SEH/MiniDump path in CrashHandlerWin32.cpp at the same
// public seam (CrashHandler::Install). Differences are unavoidable:
//   * No MiniDump - when std::stacktrace is available, we print a best-effort
//     stacktrace, then re-raise the signal so the kernel can drop a real core
//     dump if ulimit -c allows.
//   * No SetUnhandledExceptionFilter / structured exceptions - we install
//     sigaction handlers for the synchronous fault signals.
//   * No vectored handler / abort filter - SA_RESETHAND lets re-raised
//     signals run the default disposition (terminate + core).
//
// Handler is async-signal-safe by construction: it touches only stack-local
// data and file descriptors via low-level POSIX I/O. The optional
// std::stacktrace path is best-effort diagnostics only.
//

#include "CrashHandler.h"

#ifndef _WIN32

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if FRACTALSHARK_HAS_STACKTRACE
#include <iostream>
#include <stacktrace>
#endif
#include <unistd.h>

namespace {

std::atomic<bool> g_handlingFatal{false};

// async-signal-safe write of a NUL-terminated string to stderr.
void
SafeWriteStderr(const char *s) noexcept
{
    if (s == nullptr) {
        return;
    }
    const size_t len = std::strlen(s);
    [[maybe_unused]] auto ignored = ::write(STDERR_FILENO, s, len);
}

void
SafeWriteSignalHeader(int sig) noexcept
{
    SafeWriteStderr("\nFractalShark: fatal signal ");
    char buf[16];
    int n = std::snprintf(buf, sizeof(buf), "%d", sig);
    if (n > 0) {
        [[maybe_unused]] auto ignored = ::write(STDERR_FILENO, buf, static_cast<size_t>(n));
    }
    SafeWriteStderr("\n");
}

extern "C" void
OnFatalSignal(int sig, siginfo_t * /*si*/, void * /*uctx*/) noexcept
{
    bool expected = false;
    if (!g_handlingFatal.compare_exchange_strong(expected, true)) {
        // Re-entered the handler from itself; let the default disposition
        // (set up by SA_RESETHAND) actually terminate the process.
        ::raise(sig);
        return;
    }

    SafeWriteSignalHeader(sig);

#if FRACTALSHARK_HAS_STACKTRACE
    // std::stacktrace + ostream<< is not strictly async-signal-safe, but this
    // is best-effort fatal diagnostics when the backend is available.
    try {
        auto trace = std::stacktrace::current();
        std::cerr << trace << std::endl;
    } catch (...) {
        SafeWriteStderr("(stacktrace unavailable)\n");
    }
#else
    SafeWriteStderr("(stacktrace disabled)\n");
#endif

    // Re-raise so the kernel produces a core dump if ulimit -c allows.
    // SA_RESETHAND means our handler is no longer installed, so the default
    // disposition (terminate + core) runs.
    ::raise(sig);
}

} // namespace

void
CrashHandler::Install()
{
    struct sigaction sa{};
    sa.sa_sigaction = &OnFatalSignal;
    sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
    ::sigemptyset(&sa.sa_mask);

    constexpr int kFatalSignals[] = {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS};
    for (int sig : kFatalSignals) {
        ::sigaction(sig, &sa, nullptr);
    }
}

#endif // !_WIN32
