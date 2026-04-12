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
// File handle operations (for PerturbationResults delete-on-close)
// =========================================================================

// Open a file for delete-on-close.  Returns opaque handle, or InvalidHandle on failure.
void *FileOpenDeleteOnClose(const wchar_t *path);

// Close an opaque file handle (e.g. from FileOpenDeleteOnClose).
void FileClose(void *handle);

// =========================================================================
// System information
// =========================================================================

// Current process commit charge (pagefile usage) in bytes.
uint64_t ProcessCommitChargeBytes();

// Number of logical processors (cores × HT).
uint32_t LogicalProcessorCount();

// Returns true if any physical core has SMT (hyperthreading) enabled.
bool IsHyperthreadingEnabled();

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

// Yield the processor (spin-wait hint, e.g. x86 PAUSE instruction).
void YieldProcessor();

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
    Control = 0x11, // VK_CONTROL
    Alt = 0x12,     // VK_MENU
    Escape = 0x1B,  // VK_ESCAPE
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

// Query the usable size of a system-heap allocation.  Returns 0 on failure.
size_t SystemHeapSize(const void *ptr);

// =========================================================================
// OS error codes
// =========================================================================

// Get the last OS error code for the current thread.
uint32_t GetLastOSError();

// =========================================================================
// Process / command line
// =========================================================================

// Return the raw wide-character command line.  No allocation; returns a
// pointer into process memory that is valid for the lifetime of the process.
const wchar_t *GetCommandLineWide();

// =========================================================================
// Console
// =========================================================================

// Clear the console screen and move the cursor to the top-left corner.
void ClearConsole();

// =========================================================================
// Opaque handle sentinel
// =========================================================================

// Platform-independent invalid-handle constant (maps to INVALID_HANDLE_VALUE
// on Windows, i.e. (void*)(intptr_t)-1).
inline void *const InvalidHandle = reinterpret_cast<void *>(static_cast<intptr_t>(-1));

} // namespace Environment
