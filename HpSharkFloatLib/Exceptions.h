#pragma once

#include <cstring>
#include <exception>
#include <string>

struct FractalSharkSeriousException : std::exception {
    static constexpr size_t MsgBufSize = 256;

    FractalSharkSeriousException(const char *msg);
    FractalSharkSeriousException(const std::string &msg);

    const char *
    what() const noexcept override
    {
        return m_msg;
    }

    char m_msg[MsgBufSize];

    ~FractalSharkSeriousException() throw() {}
};
