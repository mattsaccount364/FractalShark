#include "Exceptions.h"

std::string
GetCallStack(const std::stacktrace &stack)
{
    std::string result;
    for (const std::stacktrace_entry &entry : stack) {
        const std::string file = entry.source_file();
        const std::string sview = file.empty() ? "unknown" : file;
        result += entry.description();
        result += ", ";
        result += sview;
        result += "(";
        result += std::to_string(entry.source_line());
        result += ")\n";
    }

    return result;
}

Cpp23ExceptionWithCallstack::Cpp23ExceptionWithCallstack(const char *msg, bool captureStack)
    : m_stacktrace{captureStack ? std::stacktrace::current(1) : std::stacktrace{}}
{
    strncpy_s(m_msg, msg, _TRUNCATE);
}

std::string
Cpp23ExceptionWithCallstack::GetCallstack(std::string extraMsg) const
{
    using namespace std; // without using std ns, the sv literal fails to compile. Strange.

    std::string result = GetCallStack(m_stacktrace);

    if (extraMsg != "") {
        result += "\n";
        result += extraMsg;
    }

    return result + "\n" + what();
}
