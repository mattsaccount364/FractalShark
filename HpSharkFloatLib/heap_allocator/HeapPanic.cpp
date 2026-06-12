#include "include/HeapPanic.h"
#include "Environment.h"

[[noreturn]] void
HeapPanic(const char *msg)
{
    // Best-effort debug output; DebugOutput is required to be heap-safe.
    Environment::DebugOutput("FractalShark Heap panic: ");
    Environment::DebugOutput(msg);
    Environment::DebugOutput("\n");

#ifdef _WIN32
    if (Environment::IsDebuggerAttached()) {
        Environment::DebugBreakpoint();
    }
#endif

    // Fail-fast: no unwinding, no handlers, no allocations.
    Environment::FastFail(7);
}
