// Win32 implementation of the Environment namespace.
// This is the ONLY file in the library projects that should include <windows.h>.

#include "Environment.h"

#include <mutex>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// clang-format off
#include <combaseapi.h> // CoCreateGuid, StringFromGUID2
#include <psapi.h>      // must follow windows.h
#include <shellapi.h>   // SHFileOperationW
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

size_t
Environment::SystemPageSize()
{
    SYSTEM_INFO si;
    ::GetSystemInfo(&si);
    return static_cast<size_t>(si.dwPageSize);
}

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

void *
Environment::GetCurrentThreadHandle()
{
    return ::GetCurrentThread();
}

uint64_t
Environment::SetThreadAffinity(void *threadHandle, uint64_t mask)
{
    DWORD_PTR result =
        ::SetThreadAffinityMask(static_cast<HANDLE>(threadHandle), static_cast<DWORD_PTR>(mask));
    return static_cast<uint64_t>(result);
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

// =========================================================================
// File I/O (general-purpose)
// =========================================================================

void *
Environment::FileOpen(const wchar_t *path,
                      FileAccess access,
                      FileDisposition disposition,
                      FileFlags flags)
{
    DWORD desiredAccess = 0;
    if (static_cast<uint32_t>(access) & static_cast<uint32_t>(FileAccess::Read)) {
        desiredAccess |= GENERIC_READ;
    }
    if (static_cast<uint32_t>(access) & static_cast<uint32_t>(FileAccess::Write)) {
        desiredAccess |= GENERIC_WRITE;
    }

    DWORD creationDisp = 0;
    switch (disposition) {
        case FileDisposition::OpenExisting:
            creationDisp = OPEN_EXISTING;
            break;
        case FileDisposition::OpenAlways:
            creationDisp = OPEN_ALWAYS;
            break;
        case FileDisposition::CreateAlways:
            creationDisp = CREATE_ALWAYS;
            break;
    }

    DWORD attributes = FILE_ATTRIBUTE_NORMAL;
    DWORD shareMode = 0;

    if (flags & FileFlags::DeleteOnClose) {
        attributes |= FILE_FLAG_DELETE_ON_CLOSE;
    }
    if (flags & FileFlags::Temporary) {
        attributes |= FILE_ATTRIBUTE_TEMPORARY;
    }
    if (flags & FileFlags::ShareRead) {
        shareMode |= FILE_SHARE_READ;
    }

    HANDLE h = ::CreateFileW(path, desiredAccess, shareMode, nullptr, creationDisp, attributes, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return Environment::InvalidHandle;
    }
    return static_cast<void *>(h);
}

uint64_t
Environment::FileGetSize(void *fileHandle)
{
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(static_cast<HANDLE>(fileHandle), &size)) {
        return 0;
    }
    return static_cast<uint64_t>(size.QuadPart);
}

bool
Environment::FileSetSize(void *fileHandle, uint64_t newSize)
{
    LARGE_INTEGER dist{};
    dist.QuadPart = static_cast<LONGLONG>(newSize);
    if (!::SetFilePointerEx(static_cast<HANDLE>(fileHandle), dist, nullptr, FILE_BEGIN)) {
        return false;
    }
    return ::SetEndOfFile(static_cast<HANDLE>(fileHandle)) != FALSE;
}

size_t
Environment::FileWrite(void *fileHandle, const void *data, size_t bytes)
{
    DWORD written = 0;
    if (!::WriteFile(
            static_cast<HANDLE>(fileHandle), data, static_cast<DWORD>(bytes), &written, nullptr)) {
        return 0;
    }
    return static_cast<size_t>(written);
}

bool
Environment::FileDelete(const wchar_t *path)
{
    return ::DeleteFileW(path) != FALSE;
}

// =========================================================================
// Memory-mapped files (section objects) — NT native API
// =========================================================================

namespace {

// NT function pointer types (not in standard SDK headers).
using NtCreateSectionFn = LONG(NTAPI *)(PHANDLE SectionHandle,
                                        ULONG DesiredAccess,
                                        PVOID ObjectAttributes,
                                        PLARGE_INTEGER MaximumSize,
                                        ULONG PageAttributes,
                                        ULONG SectionAttributes,
                                        HANDLE FileHandle);

using NtMapViewOfSectionFn = LONG(NTAPI *)(HANDLE SectionHandle,
                                           HANDLE ProcessHandle,
                                           PVOID *BaseAddress,
                                           ULONG_PTR ZeroBits,
                                           SIZE_T CommitSize,
                                           PLARGE_INTEGER SectionOffset,
                                           PSIZE_T ViewSize,
                                           DWORD InheritDisposition,
                                           ULONG AllocationType,
                                           ULONG Win32Protect);

using NtExtendSectionFn = LONG(NTAPI *)(HANDLE SectionHandle, PLARGE_INTEGER NewSectionSize);

static NtCreateSectionFn g_NtCreateSection;
static NtMapViewOfSectionFn g_NtMapViewOfSection;
static NtExtendSectionFn g_NtExtendSection;
static std::once_flag g_ntdllInitFlag;

enum : DWORD { ViewUnmap_e = 2 };

void
InitNtDll()
{
    HMODULE hNtDll = ::GetModuleHandleW(L"ntdll.dll");
    if (hNtDll != nullptr) {
        g_NtCreateSection =
            reinterpret_cast<NtCreateSectionFn>(::GetProcAddress(hNtDll, "NtCreateSection"));
        g_NtMapViewOfSection =
            reinterpret_cast<NtMapViewOfSectionFn>(::GetProcAddress(hNtDll, "NtMapViewOfSection"));
        g_NtExtendSection =
            reinterpret_cast<NtExtendSectionFn>(::GetProcAddress(hNtDll, "NtExtendSection"));
    }
}

void
EnsureNtDll()
{
    std::call_once(g_ntdllInitFlag, InitNtDll);
}

// Per-section state wrapping the native HANDLE and current view info.
struct SectionState {
    HANDLE ntSection;
    void *mappedBase;
    SIZE_T mappedSize;
};

} // anonymous namespace

void *
Environment::SectionCreate(void *fileHandle, uint64_t maxSize, bool readOnly)
{
    EnsureNtDll();
    if (g_NtCreateSection == nullptr) {
        return nullptr;
    }

    LARGE_INTEGER initialSize{};
    initialSize.QuadPart = static_cast<LONGLONG>(maxSize);

    ULONG protection = readOnly ? PAGE_READONLY : PAGE_READWRITE;
    ULONG access = readOnly ? (SECTION_MAP_READ | SECTION_EXTEND_SIZE) : SECTION_ALL_ACCESS;

    HANDLE section = nullptr;
    LONG status = g_NtCreateSection(&section,
                                    access,
                                    nullptr,
                                    &initialSize,
                                    protection,
                                    SEC_RESERVE,
                                    static_cast<HANDLE>(fileHandle));
    if (status != 0) {
        return nullptr;
    }

    auto *state = new SectionState{section, nullptr, 0};
    return state;
}

void *
Environment::SectionMapView(
    void *sectionHandle, void *suggestedBase, size_t *viewSize, bool reserveOnly, bool readOnly)
{
    EnsureNtDll();
    if (g_NtMapViewOfSection == nullptr || sectionHandle == nullptr) {
        return nullptr;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);

    ULONG protection = readOnly ? PAGE_READONLY : PAGE_READWRITE;
    ULONG allocationType = reserveOnly ? MEM_RESERVE : 0;

    void *base = suggestedBase;
    SIZE_T size = static_cast<SIZE_T>(*viewSize);

    LONG status = g_NtMapViewOfSection(state->ntSection,
                                       ::GetCurrentProcess(),
                                       &base,
                                       0,
                                       0,
                                       nullptr,
                                       &size,
                                       ViewUnmap_e,
                                       allocationType,
                                       protection);
    if (status != 0) {
        return nullptr;
    }

    state->mappedBase = base;
    state->mappedSize = size;
    *viewSize = static_cast<size_t>(size);
    return base;
}

bool
Environment::SectionExtend(void *sectionHandle, uint64_t newSize)
{
    EnsureNtDll();
    if (g_NtExtendSection == nullptr || sectionHandle == nullptr) {
        return false;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);
    LARGE_INTEGER li{};
    li.QuadPart = static_cast<LONGLONG>(newSize);
    LONG status = g_NtExtendSection(state->ntSection, &li);
    return status == 0;
}

void
Environment::SectionUnmapView(void *sectionHandle)
{
    if (sectionHandle == nullptr) {
        return;
    }

    auto *state = static_cast<SectionState *>(sectionHandle);
    if (state->mappedBase != nullptr) {
        ::UnmapViewOfFile(state->mappedBase);
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
    if (state->ntSection != nullptr && state->ntSection != INVALID_HANDLE_VALUE) {
        ::CloseHandle(state->ntSection);
    }
    delete state;
}

// =========================================================================
// System information (extended)
// =========================================================================

uint64_t
Environment::PhysicalMemoryKB()
{
    ULONGLONG memKB = 0;
    if (::GetPhysicallyInstalledSystemMemory(&memKB)) {
        return static_cast<uint64_t>(memKB);
    }
    return 0;
}

void *
Environment::GetCurrentProcessHandle()
{
    return ::GetCurrentProcess();
}

// =========================================================================
// GUID generation
// =========================================================================

std::wstring
Environment::GenerateGuidString()
{
    GUID guid{};
    HRESULT hr = ::CoCreateGuid(&guid);
    if (FAILED(hr)) {
        return {};
    }

    wchar_t buf[40]{};
    int len = ::StringFromGUID2(guid, buf, 40);
    if (len == 0) {
        return {};
    }
    return std::wstring(buf, static_cast<size_t>(len - 1)); // exclude null terminator
}

// =========================================================================
// File system utilities
// =========================================================================

bool
Environment::DirectoryCreate(const wchar_t *path)
{
    if (::CreateDirectoryW(path, nullptr)) {
        return true;
    }
    return ::GetLastError() == ERROR_ALREADY_EXISTS;
}

bool
Environment::DirectoryExists(const wchar_t *path)
{
    DWORD attr = ::GetFileAttributesW(path);
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool
Environment::DirectoryRemoveRecursive(const wchar_t *path)
{
    // SHFileOperationW requires a double-null-terminated string.
    size_t len = wcslen(path);
    std::vector<wchar_t> from(len + 2, L'\0');
    wcsncpy_s(from.data(), from.size(), path, len);

    SHFILEOPSTRUCTW fileOp{};
    fileOp.wFunc = FO_DELETE;
    fileOp.pFrom = from.data();
    fileOp.fFlags = FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;

    return ::SHFileOperationW(&fileOp) == 0;
}
