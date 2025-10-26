#include "Exceptions.h"
#include "Callstacks.h"

#include <limits>

std::unique_ptr<CallStacks> GlobalCallstacks;

void CallStacks::InitCallstacks() {
    //_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_WNDW);

    GlobalCallstacks = std::make_unique<CallStacks>();
}

void CallStacks::OutputToFile(std::ofstream &file, const CallstackDetails &callstack) {
    file << callstack.m_Callstack;
    file << "Bytes: " << callstack.m_Bytes << "\n";
    file << "Ptr: " << callstack.m_Ptr << "\n";
    file << "\n";
}

void CallStacks::LogReserveCallstack(size_t bytes, const void *ptr) {
    std::stacktrace stack = std::stacktrace::current(1);
    auto callstack = GetCallStack(stack);

    std::lock_guard<std::mutex> lock(CallstacksMutex);
    auto newCallstack = CallstackDetails{ callstack, bytes, ptr };
    ReserveCallstacks.push_back(newCallstack);

    if (ReserveCallstacks.size() > MaxCallstacks) {
        // Remove first element
        ReserveCallstacks.erase(ReserveCallstacks.begin());
    }
}

void CallStacks::LogAllocCallstack(size_t bytes, const void *ptr) {
    std::stacktrace stack = std::stacktrace::current(1);
    auto callstack = GetCallStack(stack);

    std::lock_guard<std::mutex> lock(CallstacksMutex);
    auto newCallstack = CallstackDetails{ callstack, bytes, ptr };
    AllocCallstacks.push_back(newCallstack);

    if (AllocCallstacks.size() > MaxCallstacks) {
        // Remove first element
        AllocCallstacks.erase(AllocCallstacks.begin());
    }
}

void CallStacks::LogDeallocCallstack(const void *ptr) {
    std::lock_guard<std::mutex> lock(CallstacksMutex);
    for (auto it = ReserveCallstacks.begin(); it != ReserveCallstacks.end(); ++it) {
        if (it->m_Ptr == ptr) {
            ReserveCallstacks.erase(it);
            break;
        }
    }
}

void CallStacks::LogAllCallstacks() {
    if (!ReserveCallstacks.empty()) {
        if (!ReserveCallstacksFile.is_open()) {
            ReserveCallstacksFile.open("DebugReserveCallstacks.txt");
        }

        std::lock_guard<std::mutex> lock(CallstacksMutex);
        for (const auto &callstack : ReserveCallstacks) {
            OutputToFile(ReserveCallstacksFile, callstack);
        }
    }
}

void CallStacks::FreeCallstacks() {
    LogAllCallstacks();

    if (ReserveCallstacksFile.is_open()) {
        ReserveCallstacksFile.close();
    }

    ReserveCallstacks.clear();

    if (AllocCallstacksFile.is_open()) {
        AllocCallstacksFile.close();
    }

    AllocCallstacks.clear();

    if constexpr (Debug) {
        // We have a 16-byte leak thanks to Windows no matter what we do.
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
        _CrtDumpMemoryLeaks();
    }

    GlobalCallstacks.reset();
}
