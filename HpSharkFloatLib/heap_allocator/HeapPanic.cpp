#include "include/HeapPanic.h"
#include "Environment.h"

__declspec(noreturn) void
HeapPanic(const char *msg)
{
    // Best-effort debug output; does not allocate.
    Environment::DebugOutput("FractalShark Heap panic: ");
    Environment::DebugOutput(msg);
    Environment::DebugOutput("\n");

    if (Environment::IsDebuggerAttached()) {
        Environment::DebugBreakpoint();
    }

    // Fail-fast: no unwinding, no handlers, no allocations.
    Environment::FastFail(7);
}
