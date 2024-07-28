#pragma once

#include <exception>
#include <string>
#include <stacktrace>
#include <stdexcept>
#include <string>
#include <string_view>

struct Cpp23ExceptionWithCallstack : std::runtime_error {
    Cpp23ExceptionWithCallstack(const char *msg) :
        runtime_error{ msg },
        m_stacktrace{ std::stacktrace::current(1 /*skipped frames*/) } {
    }

    std::stacktrace m_stacktrace;

    std::string GetCallstack(std::string extraMsg) const {
        using namespace std; // without using std ns, the sv literal fails to compile. Strange.
        std::string result;
        for (const std::stacktrace_entry &entry : m_stacktrace) {
            const std::string file = entry.source_file();
            const std::string_view sview = file.empty() ? "unknown"sv : file;
            result += entry.description();
            result += ", ";
            result += sview;
            result += "(";
            result += std::to_string(entry.source_line());
            result += ")\n";
        }

        if (extraMsg != "") {
            result += "\n";
            result += extraMsg;
        }

        return result + "\n" + what();
    }
};

struct FractalSharkSeriousException : public Cpp23ExceptionWithCallstack {
    FractalSharkSeriousException(std::string ss) : Cpp23ExceptionWithCallstack{ ss.c_str() } {
    }

    ~FractalSharkSeriousException() throw () {
    }
};

