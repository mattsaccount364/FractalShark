#include "ConsoleLog.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

namespace {

void
AppendUtf8(std::string &result, uint32_t codePoint)
{
    if (codePoint > 0x10FFFF || (codePoint >= 0xD800 && codePoint <= 0xDFFF)) {
        codePoint = 0xFFFD;
    }

    if (codePoint <= 0x7F) {
        result.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7FF) {
        result.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
        result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else if (codePoint <= 0xFFFF) {
        result.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
        result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    } else {
        result.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
        result.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        result.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
}

std::mutex &
LogMutex()
{
    static std::mutex mutex;
    return mutex;
}

} // namespace

namespace FractalSharkLog {

std::string
WideToUtf8(std::wstring_view value)
{
    std::string result;
    result.reserve(value.size());

    for (std::size_t i = 0; i < value.size(); ++i) {
        uint32_t codePoint = static_cast<uint32_t>(value[i]);

        if constexpr (sizeof(wchar_t) == 2) {
            if (codePoint >= 0xD800 && codePoint <= 0xDBFF && i + 1 < value.size()) {
                const uint32_t low = static_cast<uint32_t>(value[i + 1]);
                if (low >= 0xDC00 && low <= 0xDFFF) {
                    codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                    ++i;
                }
            }
        }

        AppendUtf8(result, codePoint);
    }

    return result;
}

void
Write(std::string_view message, const char *file, int line) noexcept
{
    try {
        const char *sourceFile = file == nullptr ? "<unknown>" : file;
        std::lock_guard lock{LogMutex()};
        std::cerr << '[' << sourceFile << ':' << line << "] " << message << '\n';
    } catch (...) {
    }
}

void
WriteException(std::string_view context,
               const std::exception &exception,
               const char *file,
               int line) noexcept
{
    try {
        std::string message{context};
        if (!message.empty()) {
            message += ": ";
        }
        const char *what = exception.what();
        message += what == nullptr ? "<unknown exception>" : what;
        Write(message, file, line);
    } catch (...) {
    }
}

ConsoleLogLine::ConsoleLogLine(const char *file, int line) noexcept : m_File(file), m_Line(line) {}

ConsoleLogLine::~ConsoleLogLine() noexcept
{
    try {
        Write(m_Message.str(), m_File, m_Line);
    } catch (...) {
    }
}

ConsoleLogLine &
ConsoleLogLine::operator<<(std::wstring_view value)
{
    m_Message << WideToUtf8(value);
    return *this;
}

ConsoleLogLine &
ConsoleLogLine::operator<<(const std::wstring &value)
{
    return operator<<(std::wstring_view{value});
}

ConsoleLogLine &
ConsoleLogLine::operator<<(const wchar_t *value)
{
    m_Message << WideToUtf8(value == nullptr ? std::wstring_view{} : std::wstring_view{value});
    return *this;
}

ConsoleLogLine &
ConsoleLogLine::operator<<(std::ostream &(*manipulator)(std::ostream &))
{
    m_Message << manipulator;
    return *this;
}

ConsoleLogLine &
ConsoleLogLine::operator<<(std::ios_base &(*manipulator)(std::ios_base &))
{
    m_Message << manipulator;
    return *this;
}

ConsoleLogLine
LogLine(const char *file, int line)
{
    return ConsoleLogLine{file, line};
}

} // namespace FractalSharkLog
