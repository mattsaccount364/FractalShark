#pragma once

#include <cstdint>
#include <string>

enum class CommandLineValueKind {
    Omitted,
    Auto,
    Production,
    Explicit,
};

template <typename T> struct CommandLineOptionValue {
    CommandLineValueKind m_Kind = CommandLineValueKind::Omitted;
    T m_Value{};
};

struct CommandLineOptions {
    CommandLineOptionValue<int> m_Mode;
    CommandLineOptionValue<int> m_Verbose;
    CommandLineOptionValue<int> m_CudaIterations;
    CommandLineOptionValue<int> m_NumIters;
    CommandLineOptionValue<int> m_NumBlocks;
    CommandLineOptionValue<int> m_NumThreads;
    CommandLineOptionValue<int> m_MpirThreading;
    CommandLineOptionValue<int> m_View;
    CommandLineOptionValue<uint32_t> m_StorageLimbs;
    CommandLineOptionValue<uint32_t> m_EffectiveLimbs;
    // 0 = force global storage, 1 = force SharedOnly; Auto keeps capability-based selection.
    CommandLineOptionValue<int> m_SharedOnly;
};

struct CommandLineParseResult {
    CommandLineOptions m_Options;
    std::string m_Error;
    bool m_ShowHelp = false;
};

CommandLineParseResult ParseCommandLine(int argc, char **argv);
const char *CommandLineUsage();
