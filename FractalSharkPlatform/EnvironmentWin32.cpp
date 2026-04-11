// Win32 implementation of the Environment namespace.
// This is the ONLY file in the library projects that should include <windows.h>.

#include "Environment.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <combaseapi.h>
#include <psapi.h>
#include <windows.h>

#pragma comment(lib, "ntdll")

// NTSTATUS is a LONG (typedef'd in ntdef.h which we don't include directly)
typedef long NTSTATUS_e;
#ifndef NTAPI
#define NTAPI __stdcall
#endif

// =========================================================================
// NtDll native API types and function pointers (for file-backed sections)
// =========================================================================

typedef enum { ViewShare_e = 1, ViewUnmap_e = 2 } SECTION_INHERIT_e;

typedef struct {
    USHORT Length;
    USHORT MaximumLength;
    PWSTR Buffer;
} UNICODE_STRING_e, *PUNICODE_STRING_e;

typedef struct {
    ULONG Length;
    HANDLE RootDirectory;
    PUNICODE_STRING_e ObjectName;
    ULONG Attributes;
    PVOID SecurityDescriptor;
    PVOID SecurityQualityOfService;
} OBJECT_ATTRIBUTES_e, *POBJECT_ATTRIBUTES_e;

using NtCreateSectionFn = NTSTATUS_e(NTAPI *)(OUT PHANDLE SectionHandle,
                                              IN ULONG DesiredAccess,
                                              IN POBJECT_ATTRIBUTES_e ObjectAttributes OPTIONAL,
                                              IN PLARGE_INTEGER MaximumSize OPTIONAL,
                                              IN ULONG PageAttributes,
                                              IN ULONG SectionAttributes,
                                              IN HANDLE FileHandle OPTIONAL);
using NtMapViewOfSectionFn = NTSTATUS_e(NTAPI *)(HANDLE SectionHandle,
                                                 HANDLE ProcessHandle,
                                                 PVOID *BaseAddress,
                                                 ULONG_PTR ZeroBits,
                                                 SIZE_T CommitSize,
                                                 PLARGE_INTEGER SectionOffset,
                                                 PSIZE_T ViewSize,
                                                 DWORD InheritDisposition,
                                                 ULONG AllocationType,
                                                 ULONG Win32Protect);
using NtExtendSectionFn = NTSTATUS_e(NTAPI *)(IN HANDLE SectionHandle, IN PLARGE_INTEGER NewSectionSize);

static NtCreateSectionFn s_NtCreateSection;
static NtMapViewOfSectionFn s_NtMapViewOfSection;
static NtExtendSectionFn s_NtExtendSection;

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
// File-backed memory mapping
// =========================================================================

void
Environment::FileMappingStaticInit()
{
    HMODULE hNtDll = ::GetModuleHandleW(L"ntdll.dll");
    if (!hNtDll) {
        return;
    }
    s_NtCreateSection = reinterpret_cast<NtCreateSectionFn>(::GetProcAddress(hNtDll, "NtCreateSection"));
    s_NtMapViewOfSection =
        reinterpret_cast<NtMapViewOfSectionFn>(::GetProcAddress(hNtDll, "NtMapViewOfSection"));
    s_NtExtendSection = reinterpret_cast<NtExtendSectionFn>(::GetProcAddress(hNtDll, "NtExtendSection"));
}

uint32_t
Environment::FileOpen(
    FileMapping &fm, const wchar_t *path, int openMode, int desiredAccess, bool deleteOnClose)
{
    DWORD access = (desiredAccess == 1) ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
    DWORD creation = (openMode == 1) ? OPEN_EXISTING : OPEN_ALWAYS;
    DWORD flags = FILE_ATTRIBUTE_NORMAL;
    if (deleteOnClose) {
        flags |= FILE_FLAG_DELETE_ON_CLOSE;
    }

    HANDLE h = ::CreateFileW(path, access, FILE_SHARE_READ, nullptr, creation, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        fm.fileHandle = 0;
        return ::GetLastError();
    }

    fm.fileHandle = reinterpret_cast<uintptr_t>(h);

    // Return ERROR_ALREADY_EXISTS if the file existed, 0 otherwise.
    DWORD lastErr = ::GetLastError();
    return lastErr;
}

uint64_t
Environment::FileGetSize(const FileMapping &fm)
{
    LARGE_INTEGER size{};
    ::GetFileSizeEx(reinterpret_cast<HANDLE>(fm.fileHandle), &size);
    return static_cast<uint64_t>(size.QuadPart);
}

void
Environment::SectionCreate(FileMapping &fm, uint64_t initialSizeBytes, bool readOnly)
{
    LARGE_INTEGER maxSize;
    maxSize.QuadPart = static_cast<LONGLONG>(initialSizeBytes);

    ULONG protection = readOnly ? PAGE_READONLY : PAGE_READWRITE;
    ULONG attributes = SEC_RESERVE;
    ULONG access = SECTION_ALL_ACCESS;

    HANDLE section = nullptr;
    s_NtCreateSection(&section,
                      access,
                      nullptr,
                      &maxSize,
                      protection,
                      attributes,
                      reinterpret_cast<HANDLE>(fm.fileHandle));
    fm.sectionHandle = reinterpret_cast<uintptr_t>(section);
}

void *
Environment::SectionMapView(FileMapping &fm, void *desiredData, size_t &viewSizeBytes, bool readOnly)
{
    ULONG protection = readOnly ? PAGE_READONLY : PAGE_READWRITE;
    void *base = desiredData;
    SIZE_T viewSize = viewSizeBytes;

    s_NtMapViewOfSection(reinterpret_cast<HANDLE>(fm.sectionHandle),
                         ::GetCurrentProcess(),
                         &base,
                         0,
                         0,
                         nullptr,
                         &viewSize,
                         ViewUnmap_e,
                         MEM_RESERVE,
                         protection);

    viewSizeBytes = viewSize;
    return base;
}

void
Environment::SectionExtend(FileMapping &fm, uint64_t newSizeBytes)
{
    LARGE_INTEGER newSize;
    newSize.QuadPart = static_cast<LONGLONG>(newSizeBytes);
    s_NtExtendSection(reinterpret_cast<HANDLE>(fm.sectionHandle), &newSize);
}

void
Environment::SectionUnmapView(void *addr)
{
    ::UnmapViewOfFile(addr);
}

void
Environment::SectionClose(FileMapping &fm)
{
    if (fm.sectionHandle) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fm.sectionHandle));
        fm.sectionHandle = 0;
    }
}

void
Environment::FileTruncateAndClose(FileMapping &fm, uint64_t newSizeBytes)
{
    if (fm.fileHandle) {
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(newSizeBytes);
        ::SetFilePointerEx(reinterpret_cast<HANDLE>(fm.fileHandle), li, nullptr, FILE_BEGIN);
        ::SetEndOfFile(reinterpret_cast<HANDLE>(fm.fileHandle));
        ::CloseHandle(reinterpret_cast<HANDLE>(fm.fileHandle));
        fm.fileHandle = 0;
    }
}

void
Environment::FileClose(FileMapping &fm)
{
    if (fm.fileHandle) {
        ::CloseHandle(reinterpret_cast<HANDLE>(fm.fileHandle));
        fm.fileHandle = 0;
    }
}

// =========================================================================
// System information
// =========================================================================

uint64_t
Environment::PhysicalMemoryKB()
{
    ULONGLONG kb = 0;
    ::GetPhysicallyInstalledSystemMemory(&kb);
    return static_cast<uint64_t>(kb);
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

uintptr_t
Environment::CurrentProcessHandle()
{
    return reinterpret_cast<uintptr_t>(::GetCurrentProcess());
}

// =========================================================================
// GUID generation
// =========================================================================

int
Environment::GenerateGuidString(wchar_t *buf, int bufLen)
{
    GUID guid;
    if (::CoCreateGuid(&guid) != S_OK) {
        return 0;
    }
    return ::StringFromGUID2(guid, buf, bufLen);
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
