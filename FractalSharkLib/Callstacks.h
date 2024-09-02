#pragma once

#include <mutex>
#include <string>
#include <vector>
#include <fstream>

struct CallstackDetails {
    std::string m_Callstack;
    size_t m_Bytes;
    const void *m_Ptr;
};

namespace Callstacks {
    // Initialize some heap tracking.
    void InitCallstacks();

    // Output a callstack to a file.
    void OutputToFile(std::ofstream &file, const CallstackDetails &callstack);

    // Log a callstack for a "reserve" call (reserved address space).
    void LogReserveCallstack(size_t bytes, const void *ptr);

    // Log a callstack for an "alloc" call (actual committed memory allocation).
    void LogAllocCallstack(size_t bytes, const void *ptr);

    // Releases memory.  Doesn't log a callstack - it simply removes
    // the callstack from the list of allocated callstacks. 
    void LogDeallocCallstack(const void *ptr);

    // Log all callstacks to a file.
    void LogAllCallstacks();

    // Free all callstacks, check for leaks.
    void FreeCallstacks();
}