#pragma once

#include <exception>
#include <string>
#include <stacktrace>
#include <stdexcept>
#include <string>
#include <string_view>

// This is a C++23 feature. It's a simple exception class that includes a callstack.
// It's a simple wrapper around std::stacktrace::current(1) to get the callstack.

std::string GetCallStack(const std::stacktrace &stack);

struct Cpp23ExceptionWithCallstack : std::runtime_error {
    Cpp23ExceptionWithCallstack(const char *msg);

    std::stacktrace m_stacktrace;
    std::string GetCallstack(std::string extraMsg) const;
};

struct FractalSharkSeriousException : public Cpp23ExceptionWithCallstack {
    FractalSharkSeriousException(std::string ss) : Cpp23ExceptionWithCallstack{ ss.c_str() } {
    }

    ~FractalSharkSeriousException() throw () {
    }
};

