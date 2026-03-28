#pragma once

#include <cstring>
#include <exception>
#include <stacktrace>
#include <string>
#include <string_view>

// This is a C++23 feature. It's a simple exception class that includes a callstack.
// It's a simple wrapper around std::stacktrace::current(1) to get the callstack.
// When captureStack is false, the stacktrace is skipped to avoid heap allocation
// (critical for error paths in GrowableVector/HeapCpp where heap allocation would
// cause infinite recursion).

std::string GetCallStack(const std::stacktrace &stack);

struct Cpp23ExceptionWithCallstack : std::exception {
    static constexpr size_t MsgBufSize = 256;

    Cpp23ExceptionWithCallstack(const char *msg, bool captureStack = true);

    const char *what() const noexcept override { return m_msg; }

    char m_msg[MsgBufSize];
    std::stacktrace m_stacktrace;
    std::string GetCallstack(std::string extraMsg) const;
};

struct FractalSharkSeriousException : public Cpp23ExceptionWithCallstack {
    FractalSharkSeriousException(const char *msg, bool captureStack = true)
        : Cpp23ExceptionWithCallstack{msg, captureStack} {}

    ~FractalSharkSeriousException() throw() {}
};
