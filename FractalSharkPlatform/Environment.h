#pragma once

// Platform-agnostic environment interface.
// All platform-specific calls (Win32 today, Linux in the future)
// are routed through this namespace. No windows.h types appear here.
//
// Implementation: EnvironmentWin32.cpp (or EnvironmentLinux.cpp in future).

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace Environment {

// =========================================================================
// Virtual memory (anonymous, reserve-and-commit model)
// =========================================================================

// Reserve a contiguous virtual address range without committing physical pages.
void *VirtualReserve(size_t bytes);

// Commit physical pages within a previously reserved range.
bool VirtualCommit(void *addr, size_t bytes);

// Release an entire reserved region (must pass the base from VirtualReserve).
void VirtualRelease(void *base);

// =========================================================================
// File-backed memory mapping (section objects)
// =========================================================================

// Opaque handle pair for a file-backed mapping.
struct FileMapping {
    uintptr_t fileHandle = 0;    // OS file handle
    uintptr_t sectionHandle = 0; // OS section/mapping handle
};

// One-time initialization of native section APIs (e.g. NtDll on Win32).
void FileMappingStaticInit();

// Open or create a file for mapping.  Returns the file handle inside fm.
// openMode: 0 = OPEN_ALWAYS, 1 = OPEN_EXISTING
// desiredAccess: 0 = read+write, 1 = read-only
// deleteOnClose: if true, file is deleted when handle is closed.
// Returns OS error code (0 = success, platform-specific nonzero on failure).
uint32_t FileOpen(FileMapping &fm, const wchar_t *path,
                  int openMode, int desiredAccess, bool deleteOnClose);

// Get the size of the opened file in bytes.
uint64_t FileGetSize(const FileMapping &fm);

// Create a section (memory-mapped file) from an open file handle.
// initialSizeBytes: initial mapping size.
// readOnly: if true, section is read-only.
void SectionCreate(FileMapping &fm, uint64_t initialSizeBytes, bool readOnly);

// Map a view of the section into the process address space.
// desiredData: suggested base address (nullptr for any).
// viewSizeBytes: in/out — requested size, may be adjusted.
// readOnly: if true, mapping is read-only.
// Returns pointer to mapped memory.
void *SectionMapView(FileMapping &fm, void *desiredData,
                     size_t &viewSizeBytes, bool readOnly);

// Extend an existing section to a new (larger) size.
void SectionExtend(FileMapping &fm, uint64_t newSizeBytes);

// Unmap a previously mapped view.
void SectionUnmapView(void *addr);

// Close the section handle.
void SectionClose(FileMapping &fm);

// Truncate the underlying file to the given size and close the file handle.
void FileTruncateAndClose(FileMapping &fm, uint64_t newSizeBytes);

// Close just the file handle (no truncation).
void FileClose(FileMapping &fm);

// =========================================================================
// System information
// =========================================================================

// Total physical RAM in kilobytes.
uint64_t PhysicalMemoryKB();

// Current process commit charge (pagefile usage) in bytes.
uint64_t ProcessCommitChargeBytes();

// Get the current process pseudo-handle (as uintptr_t).
uintptr_t CurrentProcessHandle();

// =========================================================================
// GUID generation
// =========================================================================

// Generate a random GUID and write it as a wide string (e.g. "{xxxxxxxx-...}").
// Returns the number of characters written, or 0 on failure.
int GenerateGuidString(wchar_t *buf, int bufLen);

// =========================================================================
// Debugging
// =========================================================================

// Write a message to the platform debug output (e.g. OutputDebugString).
void DebugOutput(const char *msg);

// Returns true if a debugger is attached to the current process.
bool IsDebuggerAttached();

// Break into the debugger (clean stack).
void DebugBreakpoint();

// Terminate the process immediately with a fast-fail code.
[[noreturn]] void FastFail(int code);

// Terminate the process with exit code 1.
[[noreturn]] void ProcessTerminate();

// =========================================================================
// Threading
// =========================================================================

// Set the name of the current thread (for debugger display).
void SetCurrentThreadName(const wchar_t *name);

// Sleep the current thread for the given number of milliseconds.
void SleepMs(uint32_t ms);

// Atomic compare-and-swap on a 32-bit integer.
// Returns the original value of *dest.
int32_t InterlockedCAS32(volatile int32_t *dest, int32_t exchange, int32_t comparand);

// =========================================================================
// Timing
// =========================================================================

// High-resolution performance counter value.
uint64_t HighResCounter();

// Frequency of the high-resolution counter (ticks per second).
uint64_t HighResFrequency();

// =========================================================================
// Keyboard / mouse input (for AbortMonitor)
// =========================================================================

// Platform-agnostic virtual key codes.
enum class Key : int {
    Control = 0x11,  // VK_CONTROL
    Alt = 0x12,      // VK_MENU
    Escape = 0x1B,   // VK_ESCAPE
};

// Returns true if the specified key is currently held down.
bool IsKeyDown(Key key);

// Returns the current cursor position in screen coordinates.
std::pair<int, int> GetCursorPosition();

// =========================================================================
// System heap (fallback for HeapCpp bootstrap)
// =========================================================================

// Allocate from the system default heap.
void *SystemHeapAlloc(size_t bytes);

// Reallocate from the system default heap.
void *SystemHeapRealloc(void *ptr, size_t bytes);

// Allocate zeroed memory from the system default heap.
void *SystemHeapAllocZeroed(size_t bytes);

// Free to the system default heap.
void SystemHeapFree(void *ptr);

// =========================================================================
// OS error codes
// =========================================================================

// Get the last OS error code for the current thread.
uint32_t GetLastOSError();

} // namespace Environment
