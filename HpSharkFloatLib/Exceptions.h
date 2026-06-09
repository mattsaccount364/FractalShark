#pragma once

#include <cstring>
#include <exception>
#include <string>
#include <string_view>

#ifndef FRACTALSHARK_HAS_STACKTRACE
#define FRACTALSHARK_HAS_STACKTRACE 1
#endif

#if FRACTALSHARK_HAS_STACKTRACE
#include <stacktrace>
using FractalSharkStacktrace = std::stacktrace;
std::string GetCallStack(const std::stacktrace &stack);
#else
struct FractalSharkStacktrace {};
#endif

// This is a C++23 feature. It's a simple exception class that includes a callstack.
// It's a simple wrapper around std::stacktrace::current(1) to get the callstack.
// When captureStack is false, the stacktrace is skipped to avoid heap allocation
// (critical for error paths in GrowableVector/HeapCpp where heap allocation would
// cause infinite recursion).

struct Cpp23ExceptionWithCallstack : std::exception {
    static constexpr size_t MsgBufSize = 256;

    Cpp23ExceptionWithCallstack(const char *msg, bool captureStack = true);

    const char *
    what() const noexcept override
    {
        return m_msg;
    }

    char m_msg[MsgBufSize];
    FractalSharkStacktrace m_stacktrace;
    std::string GetCallstack(std::string extraMsg) const;
};

struct FractalSharkSeriousException : public Cpp23ExceptionWithCallstack {
    FractalSharkSeriousException(const char *msg, bool captureStack = true)
        : Cpp23ExceptionWithCallstack{msg, captureStack}
    {
    }

    ~FractalSharkSeriousException() throw() {}
};
