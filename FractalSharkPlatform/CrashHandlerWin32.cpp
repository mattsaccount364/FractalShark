#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "CrashHandler.h"

#include <cstdlib>
#include <exception>
#include <minidumpapiset.h>

// ---------------------------------------------------------------------------
// File-static state — initialised once by Install(), used by all handlers.
// ---------------------------------------------------------------------------

using MiniDumpWriteDumpFn = BOOL(WINAPI *)(HANDLE hProcess,
                                           DWORD dwPid,
                                           HANDLE hFile,
                                           MINIDUMP_TYPE DumpType,
                                           CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
                                           CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
                                           CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

static HMODULE s_DbgHelpModule = nullptr;
static MiniDumpWriteDumpFn s_MiniDumpWriteDump = nullptr;

static constexpr MINIDUMP_TYPE DumpFlags = static_cast<MINIDUMP_TYPE>(
    MiniDumpNormal | MiniDumpWithDataSegs | MiniDumpWithIndirectlyReferencedMemory);

// ---------------------------------------------------------------------------
// Minidump writer — called from every handler path.
// Avoids heap allocation; uses only stack-local data and pre-loaded pointers.
// ---------------------------------------------------------------------------

static void
WriteMiniDump(EXCEPTION_POINTERS *exceptionInfo)
{
    if (s_MiniDumpWriteDump == nullptr) {
        return;
    }

    HANDLE hFile = ::CreateFileW(L"core.dmp",
                                 GENERIC_WRITE,
                                 FILE_SHARE_WRITE,
                                 nullptr,
                                 CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL,
                                 nullptr);

    if (hFile == INVALID_HANDLE_VALUE) {
        return;
    }

    MINIDUMP_EXCEPTION_INFORMATION exInfo{};
    exInfo.ThreadId = ::GetCurrentThreadId();
    exInfo.ExceptionPointers = exceptionInfo;
    exInfo.ClientPointers = FALSE;

    s_MiniDumpWriteDump(::GetCurrentProcess(),
                        ::GetCurrentProcessId(),
                        hFile,
                        DumpFlags,
                        exceptionInfo ? &exInfo : nullptr,
                        nullptr,
                        nullptr);

    ::CloseHandle(hFile);
}

// ---------------------------------------------------------------------------
// Helper: capture context and write dump when no EXCEPTION_POINTERS exist
// (used by CRT handlers that don't receive one).
// ---------------------------------------------------------------------------

static void
WriteMiniDumpFromCurrentContext()
{
    // GetExceptionInformation() is only valid inside the __except filter
    // expression, so the dump must be written there.  Copy the structures
    // to stack-locals in the filter so WriteMiniDump operates on data that
    // outlives the filter evaluation.
    EXCEPTION_RECORD record{};
    CONTEXT ctx{};
    EXCEPTION_POINTERS captured{&record, &ctx};

    __try {
        ::RaiseException(0xE0000001, 0, 0, nullptr);
    } __except (record = *GetExceptionInformation()->ExceptionRecord,
                ctx = *GetExceptionInformation()->ContextRecord,
                EXCEPTION_EXECUTE_HANDLER) {
        WriteMiniDump(&captured);
    }
}

// ---------------------------------------------------------------------------
// SEH unhandled-exception filter
// ---------------------------------------------------------------------------

static LONG WINAPI
UnhandledExceptionHandler(EXCEPTION_POINTERS *exceptionInfo)
{
    WriteMiniDump(exceptionInfo);
    return EXCEPTION_CONTINUE_SEARCH;
}

// ---------------------------------------------------------------------------
// CRT invalid-parameter handler
// ---------------------------------------------------------------------------

static void
InvalidParameterHandler(const wchar_t * /*expression*/,
                        const wchar_t * /*function*/,
                        const wchar_t * /*file*/,
                        unsigned int /*line*/,
                        uintptr_t /*reserved*/)
{
    WriteMiniDumpFromCurrentContext();
    ::TerminateProcess(::GetCurrentProcess(), 1);
}

// ---------------------------------------------------------------------------
// Pure-virtual-call handler
// ---------------------------------------------------------------------------

static void
PureCallHandler()
{
    WriteMiniDumpFromCurrentContext();
    ::TerminateProcess(::GetCurrentProcess(), 1);
}

// ---------------------------------------------------------------------------
// std::terminate handler (unhandled C++ exceptions, failed noexcept, etc.)
// ---------------------------------------------------------------------------

static void
TerminateHandler()
{
    WriteMiniDumpFromCurrentContext();
    ::TerminateProcess(::GetCurrentProcess(), 1);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void
CrashHandler::Install()
{
    // Must be called exactly once, on the main thread, before any worker
    // threads are created.
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;

    // Eagerly load dbghelp.dll and resolve MiniDumpWriteDump so the crash
    // path doesn't have to call LoadLibrary (which touches the heap).
    s_DbgHelpModule = ::LoadLibraryW(L"dbghelp.dll");
    if (s_DbgHelpModule != nullptr) {
        s_MiniDumpWriteDump = reinterpret_cast<MiniDumpWriteDumpFn>(
            ::GetProcAddress(s_DbgHelpModule, "MiniDumpWriteDump"));
    }

    // SEH top-level filter.
    ::SetUnhandledExceptionFilter(UnhandledExceptionHandler);

    // CRT handlers for crashes that bypass SEH.
    _set_invalid_parameter_handler(InvalidParameterHandler);
    _set_purecall_handler(PureCallHandler);
    std::set_terminate(TerminateHandler);

    // Prevent abort() from showing a dialog.
    _set_abort_behavior(0, _WRITE_ABORT_MSG);

    // Reserve extra stack space so the crash handler can execute even
    // during a stack overflow.  32 KB is generous for our dump-only handler.
    ULONG extraStack = 32 * 1024;
    ::SetThreadStackGuarantee(&extraStack);
}

#else // !_WIN32

// Linux crash handler: signal-based, zero heap allocation in handlers.
// Uses backtrace() + backtrace_symbols_fd() which write into stack-local
// buffers and directly to file descriptors — safe even during heap corruption.

#include "CrashHandler.h"

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>

static void
WriteStacktraceToFd(int fd)
{
    void *frames[64];
    int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, fd);
}

// Async-signal-safe write helper (no heap, no printf).
static void
SafeWrite(int fd, const char *msg)
{
    // write() is async-signal-safe per POSIX.
    size_t len = 0;
    const char *p = msg;
    while (*p++) {
        ++len;
    }
    (void)write(fd, msg, len);
}

static void
SignalHandler(int sig)
{
    SafeWrite(STDERR_FILENO, "\n=== FractalShark crash (signal ");

    // Convert signal number to string without sprintf (async-signal-safe).
    char sigBuf[16];
    int idx = 0;
    int s = sig;
    if (s < 0) {
        sigBuf[idx++] = '-';
        s = -s;
    }
    char tmp[16];
    int tmpIdx = 0;
    do {
        tmp[tmpIdx++] = '0' + (s % 10);
        s /= 10;
    } while (s > 0);
    while (tmpIdx > 0) {
        sigBuf[idx++] = tmp[--tmpIdx];
    }
    sigBuf[idx] = '\0';
    SafeWrite(STDERR_FILENO, sigBuf);

    SafeWrite(STDERR_FILENO, ") ===\n");
    WriteStacktraceToFd(STDERR_FILENO);
    SafeWrite(STDERR_FILENO, "=== end stack trace ===\n");

    // Re-raise with default handler to produce a core dump.
    signal(sig, SIG_DFL);
    raise(sig);
}

static void
TerminateHandler()
{
    SafeWrite(STDERR_FILENO, "\n=== FractalShark std::terminate ===\n");
    WriteStacktraceToFd(STDERR_FILENO);
    SafeWrite(STDERR_FILENO, "=== end stack trace ===\n");
    _exit(1);
}

void
CrashHandler::Install()
{
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;

    struct sigaction sa{};
    sa.sa_handler = SignalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND; // One-shot: re-raise uses default handler.

    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);

    std::set_terminate(TerminateHandler);
}

#endif // _WIN32
