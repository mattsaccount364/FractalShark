#include "Exceptions.h"

std::string GetCallStack(const std::stacktrace &stack) {
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

Cpp23ExceptionWithCallstack::Cpp23ExceptionWithCallstack(const char *msg) :
    runtime_error{ msg },
    m_stacktrace{ std::stacktrace::current(1 /*skipped frames*/) } {
}

std::string Cpp23ExceptionWithCallstack::GetCallstack(std::string extraMsg) const {
    using namespace std; // without using std ns, the sv literal fails to compile. Strange.

    std::string result = GetCallStack(m_stacktrace);

    if (extraMsg != "") {
        result += "\n";
        result += extraMsg;
    }

    return result + "\n" + what();
}
