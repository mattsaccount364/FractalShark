//
// CrashHandlerLinux.cpp - Linux signal-based crash handler.
//
// Linux crash handling intentionally stays minimal. The handler writes a small
// fatal-signal header, then re-raises the signal so the kernel can produce a
// normal core dump if ulimit -c allows.
//

#include "CrashHandler.h"

#include <csignal>
#include <cstddef>
#include <unistd.h>

namespace {

volatile std::sig_atomic_t g_handlingFatal = 0;

template <size_t N>
void
SafeWriteLiteral(const char (&s)[N]) noexcept
{
    [[maybe_unused]] auto ignored = ::write(STDERR_FILENO, s, N - 1);
}

void
SafeWriteSignalNumber(int sig) noexcept
{
    char buf[16];
    int idx = 0;

    if (sig < 0) {
        buf[idx++] = '-';
        sig = -sig;
    }

    char reversed[16];
    int reversedIdx = 0;
    do {
        reversed[reversedIdx++] = static_cast<char>('0' + (sig % 10));
        sig /= 10;
    } while (sig > 0 && reversedIdx < static_cast<int>(sizeof(reversed)));

    while (reversedIdx > 0 && idx < static_cast<int>(sizeof(buf))) {
        buf[idx++] = reversed[--reversedIdx];
    }

    [[maybe_unused]] auto ignored = ::write(STDERR_FILENO, buf, static_cast<size_t>(idx));
}

extern "C" void
OnFatalSignal(int sig, siginfo_t * /*si*/, void * /*uctx*/) noexcept
{
    if (g_handlingFatal != 0) {
        ::raise(sig);
        return;
    }
    g_handlingFatal = 1;

    SafeWriteLiteral("\nFractalShark: fatal signal ");
    SafeWriteSignalNumber(sig);
    SafeWriteLiteral("\n");

    // SA_RESETHAND means this handler is removed before control reaches here.
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
