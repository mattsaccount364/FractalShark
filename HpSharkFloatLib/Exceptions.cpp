#include "Exceptions.h"

#if FRACTALSHARK_HAS_STACKTRACE
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
#endif

namespace {

FractalSharkStacktrace
CaptureStacktrace(bool captureStack)
{
#if FRACTALSHARK_HAS_STACKTRACE
    return captureStack ? std::stacktrace::current(1) : std::stacktrace{};
#else
    (void)captureStack;
    return {};
#endif
}

std::string
FormatStacktrace(const FractalSharkStacktrace &stack)
{
#if FRACTALSHARK_HAS_STACKTRACE
    return GetCallStack(stack);
#else
    (void)stack;
    return "(stacktrace disabled)\n";
#endif
}

} // namespace

Cpp23ExceptionWithCallstack::Cpp23ExceptionWithCallstack(const char *msg, bool captureStack)
    : m_stacktrace{CaptureStacktrace(captureStack)}
{
#ifdef _MSC_VER
    strncpy_s(m_msg, msg, _TRUNCATE);
#else
    if (msg == nullptr) {
        m_msg[0] = '\0';
    } else {
        std::strncpy(m_msg, msg, MsgBufSize - 1);
        m_msg[MsgBufSize - 1] = '\0';
    }
#endif
}

std::string
Cpp23ExceptionWithCallstack::GetCallstack(std::string extraMsg) const
{
    using namespace std; // without using std ns, the sv literal fails to compile. Strange.

    std::string result = FormatStacktrace(m_stacktrace);

    if (extraMsg != "") {
        result += "\n";
        result += extraMsg;
    }

    return result + "\n" + what();
}
