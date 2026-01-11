#pragma once

__declspec(noreturn) void HeapPanic(const char *msg);

#define HEAP_ASSERT(expr, msg)                                                                          \
    do {                                                                                                \
        if (!(expr))                                                                                    \
            HeapPanic(msg);                                                                             \
    } while (0)
