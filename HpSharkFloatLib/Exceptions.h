#pragma once

#include <cstring>
#include <exception>

struct FractalSharkSeriousException : std::exception {
    static constexpr size_t MsgBufSize = 256;

    FractalSharkSeriousException(const char *msg);

    const char *
    what() const noexcept override
    {
        return m_msg;
    }

    char m_msg[MsgBufSize];

    ~FractalSharkSeriousException() throw() {}
};
