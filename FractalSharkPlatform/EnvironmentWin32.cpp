// Win32 implementation of the Environment namespace.
// This is the ONLY file in the library projects that should include <windows.h>.

#include "Environment.h"

#include <array>
#include <cstring>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
// clang-format off
#include <psapi.h>      // must follow windows.h
#include <shellapi.h>   // SHFileOperationW
// clang-format on

#include "../FractalShark/resource.h"

namespace {

struct ImaginaResourceEntry {
    Environment::ImaginaFixtureInfo info;
    int resourceId;
};

constexpr std::array<Environment::ImaginaFixtureInfo, 6> kImaginaFixtureInfos{{
    {L"View0.im", size_t{0}},
    {L"View5.im", size_t{5}},
    {L"View14.im", size_t{14}},
    {L"View15.im", size_t{15}},
    {L"View19.im", size_t{19}},
    {L"ViewEasy1.im", std::nullopt},
}};

constexpr std::array<ImaginaResourceEntry, 6> kImaginaResources{{
    {kImaginaFixtureInfos[0], IDR_IMAGINA_VIEW0},
    {kImaginaFixtureInfos[1], IDR_IMAGINA_VIEW5},
    {kImaginaFixtureInfos[2], IDR_IMAGINA_VIEW14},
    {kImaginaFixtureInfos[3], IDR_IMAGINA_VIEW15},
    {kImaginaFixtureInfos[4], IDR_IMAGINA_VIEW19},
    {kImaginaFixtureInfos[5], IDR_IMAGINA_VIEWEASY1},
}};

} // namespace

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
                             FILE_FLAG_DELETE_ON_CLOSE,
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

    DWORD attributes = 0;
    DWORD shareMode = 0;

    if (flags & FileFlags::DeleteOnClose) {
        attributes |= FILE_FLAG_DELETE_ON_CLOSE;
    }
    if (flags & FileFlags::Temporary) {
        attributes |= FILE_ATTRIBUTE_TEMPORARY;
    }
    if (attributes == 0) {
        attributes = FILE_ATTRIBUTE_NORMAL;
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

std::optional<uint64_t>
Environment::FileSizeBytes(const wchar_t *path)
{
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!::GetFileAttributesExW(path, GetFileExInfoStandard, &fad)) {
        return std::nullopt;
    }
    if ((fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return std::nullopt;
    }
    LARGE_INTEGER size;
    size.LowPart = fad.nFileSizeLow;
    size.HighPart = static_cast<LONG>(fad.nFileSizeHigh);
    return static_cast<uint64_t>(size.QuadPart);
}

std::span<const Environment::ImaginaFixtureInfo>
Environment::GetEmbeddedImaginaFixtureInfos()
{
    return kImaginaFixtureInfos;
}

std::optional<Environment::EmbeddedResourceView>
Environment::FindEmbeddedImaginaFixture(std::wstring_view name)
{
    for (const auto &entry : kImaginaResources) {
        if (entry.info.name != name) {
            continue;
        }

        auto hInst = ::GetModuleHandleW(nullptr);
        auto hRes = ::FindResourceW(hInst, MAKEINTRESOURCEW(entry.resourceId), L"SHARK_DATA");
        if (hRes == nullptr) {
            return std::nullopt;
        }

        auto hGlobal = ::LoadResource(hInst, hRes);
        if (hGlobal == nullptr) {
            return std::nullopt;
        }

        auto *resourceData = ::LockResource(hGlobal);
        if (resourceData == nullptr) {
            return std::nullopt;
        }

        const auto resourceSize = ::SizeofResource(hInst, hRes);
        if (resourceSize == 0) {
            return std::nullopt;
        }

        return EmbeddedResourceView{
            entry.info.name,
            std::span<const std::byte>{static_cast<const std::byte *>(resourceData),
                                       static_cast<size_t>(resourceSize)}};
    }

    return std::nullopt;
}

// =========================================================================
// Process information
// =========================================================================

uint64_t
Environment::CurrentProcessId()
{
    return static_cast<uint64_t>(::GetCurrentProcessId());
}

std::wstring
Environment::TempDirectoryPath()
{
    wchar_t buf[MAX_PATH + 1] = {};
    DWORD len = ::GetTempPathW(MAX_PATH, buf);
    if (len == 0 || len > MAX_PATH) {
        return L".\\";
    }
    return std::wstring(buf, len);
}

// =========================================================================
// System information (extended)
// =========================================================================

uint32_t
Environment::GetMemoryLoad()
{
    MEMORYSTATUSEX statex{};
    statex.dwLength = sizeof(statex);
    if (::GlobalMemoryStatusEx(&statex)) {
        return static_cast<uint32_t>(statex.dwMemoryLoad);
    }
    return 0;
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

// =========================================================================
// UI helpers
// =========================================================================

void
Environment::ShowWarning(const wchar_t *message)
{
    ::MessageBoxW(nullptr, message, L"Warning", MB_OK | MB_APPLMODAL | MB_ICONWARNING);
}

bool
Environment::SetClipboardText(std::string_view text)
{
    if (!::OpenClipboard(nullptr)) {
        return false;
    }

    if (!::EmptyClipboard()) {
        ::CloseClipboard();
        return false;
    }

    HGLOBAL textHandle = ::GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
    if (textHandle == nullptr) {
        ::CloseClipboard();
        return false;
    }

    auto *clipboardBytes = ::GlobalLock(textHandle);
    if (clipboardBytes == nullptr) {
        ::GlobalFree(textHandle);
        ::CloseClipboard();
        return false;
    }

    std::memcpy(clipboardBytes, text.data(), text.size());
    static_cast<char *>(clipboardBytes)[text.size()] = '\0';
    ::GlobalUnlock(textHandle);

    if (::SetClipboardData(CF_TEXT, textHandle) == nullptr) {
        ::GlobalFree(textHandle);
        ::CloseClipboard();
        return false;
    }

    ::CloseClipboard();
    return true;
}

void
Environment::PumpUIEvents()
{
    MSG msg;
    ::PeekMessageW(&msg, nullptr, 0, 0, PM_NOREMOVE);
}
