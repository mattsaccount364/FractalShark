#pragma once

#include <exception>
#include <string>

struct FractalSharkSeriousException : public std::exception {
    std::string s;
    FractalSharkSeriousException(std::string ss) : s(ss) {}
    ~FractalSharkSeriousException() throw () {
    }

    const char *what() const throw() { return s.c_str(); }
};

