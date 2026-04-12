// Win32 implementation of the Environment namespace.
// This is the ONLY file in the library projects that should include <windows.h>.

#include "Environment.h"

#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// clang-format off
#include <psapi.h> // must follow windows.h
// clang-format on

// =========================================================================
// Virtual memory
// =========================================================================

void *
Environment::VirtualReserve(size_t bytes)
{
    return ::VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
}

bool
Environment::VirtualCommit(void *addr, size_t bytes)
{
    return ::VirtualAlloc(addr, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

void
Environment::VirtualRelease(void *base)
{
    ::VirtualFree(base, 0, MEM_RELEASE);
}

// =========================================================================
// File handle operations
// =========================================================================

void *
Environment::FileOpenDeleteOnClose(const wchar_t *path)
{
    HANDLE h = ::CreateFileW(path,
                             0, // no read or write access
                             FILE_SHARE_DELETE,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
                             nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return Environment::InvalidHandle;
    }
    return static_cast<void *>(h);
}

void
Environment::FileClose(void *handle)
{
    if (handle && handle != Environment::InvalidHandle) {
        ::CloseHandle(static_cast<HANDLE>(handle));
    }
}

// =========================================================================
// System information
// =========================================================================

uint64_t
Environment::ProcessCommitChargeBytes()
{
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (::GetProcessMemoryInfo(
            ::GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc), sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.PagefileUsage);
    }
    return 0;
}

uint32_t
Environment::LogicalProcessorCount()
{
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
}

bool
Environment::IsHyperthreadingEnabled()
{
    DWORD returnLength = 0;
    ::GetLogicalProcessorInformation(nullptr, &returnLength);
    if (::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return false;
    }

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
        returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!::GetLogicalProcessorInformation(buffer.data(), &returnLength)) {
        return false;
    }

    for (const auto &info : buffer) {
        if (info.Relationship == RelationProcessorCore) {
            if (info.ProcessorCore.Flags == LTP_PC_SMT) {
                return true;
            }
        }
    }
    return false;
}

// =========================================================================
// Debugging
// =========================================================================

void
Environment::DebugOutput(const char *msg)
{
    ::OutputDebugStringA(msg);
}

bool
Environment::IsDebuggerAttached()
{
    return ::IsDebuggerPresent() != FALSE;
}

void
Environment::DebugBreakpoint()
{
    __debugbreak();
}

[[noreturn]] void
Environment::FastFail(int code)
{
    __fastfail(static_cast<unsigned int>(code));
}

[[noreturn]] void
Environment::ProcessTerminate()
{
    ::TerminateProcess(::GetCurrentProcess(), 1);
    __assume(false); // unreachable, silence warning
}

// =========================================================================
// Threading
// =========================================================================

void
Environment::SetCurrentThreadName(const wchar_t *name)
{
    ::SetThreadDescription(::GetCurrentThread(), name);
}

void
Environment::SleepMs(uint32_t ms)
{
    ::Sleep(static_cast<DWORD>(ms));
}

#undef YieldProcessor
#include <immintrin.h> // _mm_pause

void
Environment::YieldProcessor()
{
    _mm_pause();
}

int32_t
Environment::InterlockedCAS32(volatile int32_t *dest, int32_t exchange, int32_t comparand)
{
    return static_cast<int32_t>(
        ::InterlockedCompareExchange(reinterpret_cast<volatile LONG *>(dest), exchange, comparand));
}

// =========================================================================
// Timing
// =========================================================================

uint64_t
Environment::HighResCounter()
{
    LARGE_INTEGER li;
    ::QueryPerformanceCounter(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

uint64_t
Environment::HighResFrequency()
{
    LARGE_INTEGER li;
    ::QueryPerformanceFrequency(&li);
    return static_cast<uint64_t>(li.QuadPart);
}

// =========================================================================
// Keyboard / mouse input
// =========================================================================

bool
Environment::IsKeyDown(Key key)
{
    return (::GetAsyncKeyState(static_cast<int>(key)) & 0x8000) != 0;
}

std::pair<int, int>
Environment::GetCursorPosition()
{
    POINT pt;
    ::GetCursorPos(&pt);
    return {pt.x, pt.y};
}

// =========================================================================
// System heap
// =========================================================================

void *
Environment::SystemHeapAlloc(size_t bytes)
{
    return ::HeapAlloc(::GetProcessHeap(), 0, bytes);
}

void *
Environment::SystemHeapRealloc(void *ptr, size_t bytes)
{
    return ::HeapReAlloc(::GetProcessHeap(), 0, ptr, bytes);
}

void *
Environment::SystemHeapAllocZeroed(size_t bytes)
{
    return ::HeapAlloc(::GetProcessHeap(), HEAP_ZERO_MEMORY, bytes);
}

void
Environment::SystemHeapFree(void *ptr)
{
    ::HeapFree(::GetProcessHeap(), 0, ptr);
}

size_t
Environment::SystemHeapSize(const void *ptr)
{
    SIZE_T s = ::HeapSize(::GetProcessHeap(), 0, ptr);
    if (s == static_cast<SIZE_T>(-1)) {
        return 0;
    }
    return static_cast<size_t>(s);
}

// =========================================================================
// OS error codes
// =========================================================================

uint32_t
Environment::GetLastOSError()
{
    return ::GetLastError();
}

// =========================================================================
// Process / command line
// =========================================================================

const wchar_t *
Environment::GetCommandLineWide()
{
    return ::GetCommandLineW();
}

// =========================================================================
// Console
// =========================================================================

void
Environment::ClearConsole()
{
    HANDLE hOut = ::GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) {
        return;
    }

    CONSOLE_SCREEN_BUFFER_INFO csbi{};
    if (!::GetConsoleScreenBufferInfo(hOut, &csbi)) {
        return;
    }

    const DWORD cells = static_cast<DWORD>(csbi.dwSize.X) * csbi.dwSize.Y;
    DWORD written = 0;
    const COORD home{0, 0};

    ::FillConsoleOutputCharacterA(hOut, ' ', cells, home, &written);
    ::FillConsoleOutputAttribute(hOut, csbi.wAttributes, cells, home, &written);
    ::SetConsoleCursorPosition(hOut, home);
}
