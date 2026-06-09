#include "include/HeapPanic.h"
#include "Environment.h"

#ifndef _WIN32
#include <unistd.h>

namespace {

void
WriteCStringNoAlloc(const char *msg)
{
    if (msg == nullptr) {
        msg = "<null>";
    }

    size_t len = 0;
    while (msg[len] != '\0') {
        ++len;
    }

    while (len != 0) {
        const ssize_t written = ::write(STDERR_FILENO, msg, len);
        if (written <= 0) {
            return;
        }
        msg += written;
        len -= static_cast<size_t>(written);
    }
}

} // namespace
#endif

[[noreturn]] void
HeapPanic(const char *msg)
{
#ifdef _WIN32
    // Best-effort debug output; does not allocate.
    Environment::DebugOutput("FractalShark Heap panic: ");
    Environment::DebugOutput(msg);
    Environment::DebugOutput("\n");

    if (Environment::IsDebuggerAttached()) {
        Environment::DebugBreakpoint();
    }
#else
    WriteCStringNoAlloc("FractalShark Heap panic: ");
    WriteCStringNoAlloc(msg);
    WriteCStringNoAlloc("\n");
#endif

    // Fail-fast: no unwinding, no handlers, no allocations.
    Environment::FastFail(7);
}
