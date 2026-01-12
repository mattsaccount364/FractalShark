#include "include/HeapPanic.h"

#define NOMINMAX
#include <windows.h>

__declspec(noreturn) void
HeapPanic(const char *msg)
{
    // Best-effort debug output; does not allocate.
    OutputDebugStringA("FractalShark Heap panic: ");
    OutputDebugStringA(msg);
    OutputDebugStringA("\n");

    if (IsDebuggerPresent()) {
        __debugbreak(); // stop *here* with a clean stack
    }

    // Fail-fast: no unwinding, no handlers, no allocations.
    // FAST_FAIL_FATAL_APP_EXIT is 7, but any code is fine.
    __fastfail(7);

    // In case __fastfail is unavailable in some config:
    TerminateProcess(GetCurrentProcess(), 0xDEAD);
    __assume(0);
}
