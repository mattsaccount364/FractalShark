#pragma once

#include <cstddef>
#include <exception>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace FractalSharkLog {

std::string WideToUtf8(std::wstring_view value);

void Write(std::string_view message, const char *file, int line) noexcept;

void WriteException(std::string_view context,
                    const std::exception &exception,
                    const char *file,
                    int line) noexcept;

class ConsoleLogLine {
public:
    ConsoleLogLine(const char *file, int line) noexcept;

    ~ConsoleLogLine() noexcept;

    ConsoleLogLine(const ConsoleLogLine &) = delete;
    ConsoleLogLine &operator=(const ConsoleLogLine &) = delete;

    template <std::size_t Size>
    ConsoleLogLine &
    operator<<(const wchar_t (&value)[Size])
    {
        return operator<<(std::wstring_view{value});
    }

    template <typename Value>
    ConsoleLogLine &
    operator<<(const Value &value)
    {
        m_Message << value;
        return *this;
    }

    ConsoleLogLine &operator<<(std::wstring_view value);

    ConsoleLogLine &operator<<(const std::wstring &value);

    ConsoleLogLine &operator<<(const wchar_t *value);

    ConsoleLogLine &operator<<(std::ostream &(*manipulator)(std::ostream &));

    ConsoleLogLine &operator<<(std::ios_base &(*manipulator)(std::ios_base &));

private:
    std::ostringstream m_Message;
    const char *m_File;
    int m_Line;
};

ConsoleLogLine LogLine(const char *file, int line);

} // namespace FractalSharkLog
