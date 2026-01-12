#pragma once

// This header is intentionally CRT-minimal.
// Do NOT include <string>, <vector>, <atomic>, etc.

#include <stdbool.h> // bool (C-compatible)
#include <stddef.h>  // size_t
#define NOMINMAX
#include <windows.h> // HMODULE, wchar_t, WinAPI types

enum class FancyHeap : int { Unknown = 0, Enable = 1, Disable = 2 };

// -----------------------------------------------------------------------------
// No-CRT command-line scanning helpers
// These must NOT allocate or call CRT routines.
// -----------------------------------------------------------------------------

bool HasSafeModeFlag_NoCRT();

// -----------------------------------------------------------------------------
// CRT allocator forwarding resolution
// Must be called before using any g_crt_* function pointers.
// -----------------------------------------------------------------------------

void ResolveCrtAllocators(void);


extern "C" void __cdecl EarlyInit_SafeMode_NoCRT();
