#include "Conversion.h"
#include "DbgHeap.h"
#include "HpSharkFloat.h"
#include "ShowMostEfficientSizes.h"
#include "TestTracker.h"
#include "TestVerbose.h"
#include "Tests.h"

#include "Callstacks.h"
#include "HDRFloat.h"
#include "MainTestCuda.h"
#include "heap_allocator/include/HeapCpp.h"

#include <cuda_runtime.h>

#include <chrono>
#include <conio.h> // _getch, _kbhit
#include <iostream>
#include <limits>
#include <mpir.h>
#include <sstream>
#include <stdarg.h>
#include <string>

#define NOMINMAX
#include <windows.h> // Sleep()

// -----------------------------------------------------------------------------
// Assumed defined elsewhere
// -----------------------------------------------------------------------------
enum class BasicCorrectnessMode : int;

const char *
BasicCorrectnessModeToString(BasicCorrectnessMode mode)
{
    switch (mode) {
        case BasicCorrectnessMode::Correctness_P1:
            return "Correctness (Params1)";
        case BasicCorrectnessMode::PerfSweep:
            return "Performance Sweep";
        case BasicCorrectnessMode::PerfSingle:
            return "Performance Single";
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            return "Correctness (Params1..5)";
        default:
            return "Unknown";
    }
}

// -----------------------------------------------------------------------------
// Test base IDs (remove magic numbers)
// -----------------------------------------------------------------------------
namespace TestIds {
constexpr int kConversion = 0;

constexpr int kAddCorrectness = 4000;
constexpr int kMultiplyCorrectness = 6000;
constexpr int kFullCorrectness = 2000;

constexpr int kAddPerf = 10000;
constexpr int kMultiplyPerf = 13000;
constexpr int kFullPerf = 14000;

constexpr int kPerfView30 = 16020;
constexpr int kPerfView5 = 16010;

constexpr int kPerfSweepStart = 1000;
} // namespace TestIds

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
static char
PressKey()
{
    std::cout << "Press any key to continue...";
    return (char)_getch();
}

static bool
ContinueAfterFailure(bool res)
{
    if (res)
        return true;
    return PressKey() != 'q';
}

/// Robust console line input using _getch/_kbhit (no std::getline mixing).
/// - If user provides any input (even invalid), interactive mode becomes true.
/// - If interactive mode is true, subsequent prompts wait indefinitely.
/// - On timeout:
///     * if buffer empty -> returns defaultValue
///     * if buffer non-empty (user started typing but didn't press Enter) -> accepts buffer
struct PromptResult {
    int value = 0;
    bool gotAnyInput = false; // user typed something (even if invalid)
};

static PromptResult
PromptIntWithTimeout(const std::string &promptText,
                     int defaultValue,
                     int timeoutSec,
                     bool &interactiveMode,
                     int sleepIntervalMs = 50)
{
    std::cout << promptText << " " << std::flush;

    // If user has interacted once, make prompts "sticky" (wait indefinitely).
    const bool waitForever = interactiveMode;
    auto deadline =
        std::chrono::steady_clock::now() + (waitForever ? std::chrono::hours(24 * 365 * 100) // ~100y
                                                        : std::chrono::seconds(timeoutSec));

    std::string buf;
    bool pressedEnter = false;

    while (std::chrono::steady_clock::now() < deadline) {
        if (_kbhit()) {
            int ch = _getch();
            if (ch == '\r' || ch == '\n') {
                pressedEnter = true;
                std::cout << "\n";
                break;
            }

            // Backspace
            if (ch == 8) {
                if (!buf.empty()) {
                    buf.pop_back();
                    // Erase last char from console
                    std::cout << "\b \b" << std::flush;
                }
                continue;
            }

            // Ctrl+C etc: treat as "no-op" (don’t inject weird chars)
            if (ch < 32) {
                continue;
            }

            buf.push_back((char)ch);
            std::cout << (char)ch << std::flush;
        } else {
            Sleep(sleepIntervalMs);
        }
    }

    PromptResult out;
    out.gotAnyInput = !buf.empty();

    // If user typed anything, consider them "interactive" from now on.
    if (out.gotAnyInput) {
        interactiveMode = true;
    }

    // Timeout cases
    if (!pressedEnter && !waitForever && std::chrono::steady_clock::now() >= deadline) {
        if (buf.empty()) {
            std::cout << "\n(no input in " << timeoutSec << "s, defaulting to " << defaultValue << ")\n";
            out.value = defaultValue;
            return out;
        }
        // User started typing but didn't hit Enter before timeout; accept what we have.
        std::cout << "\n(timeout; accepting partial input)\n";
    }

    if (buf.empty()) {
        out.value = defaultValue;
        return out;
    }

    try {
        size_t idx = 0;
        int v = std::stoi(buf, &idx, 10);
        // Reject trailing non-space garbage
        for (; idx < buf.size(); ++idx) {
            if (buf[idx] != ' ' && buf[idx] != '\t') {
                throw std::runtime_error("trailing garbage");
            }
        }
        out.value = v;
        return out;
    } catch (...) {
        std::cout << "(could not parse \"" << buf << "\", defaulting to " << defaultValue << ")\n";
        out.value = defaultValue;
        return out;
    }
}

// -----------------------------------------------------------------------------
// Correctness tests
// -----------------------------------------------------------------------------
template <typename TestSharkParams>
static bool
CorrectnessTests()
{
    bool res = true;

    res = TestConversion<TestSharkParams>(TestIds::kConversion);
    if (!ContinueAfterFailure(res))
        return false;

    res = TestAllBinaryOp<TestSharkParams, Operator::Add>(TestIds::kAddCorrectness);
    if (!ContinueAfterFailure(res))
        return false;

    res = TestAllBinaryOp<TestSharkParams, Operator::MultiplyNTT>(TestIds::kMultiplyCorrectness);
    if (!ContinueAfterFailure(res))
        return false;

    res = TestAllBinaryOp<TestSharkParams, Operator::ReferenceOrbit>(TestIds::kFullCorrectness);
    if (!ContinueAfterFailure(res))
        return false;

    return true;
}

static int
RunCorrectnessTest(BasicCorrectnessMode mode)
{
    // Only run for correctness modes.
    // (Assumes these enum values exist elsewhere)
    if (mode != BasicCorrectnessMode::Correctness_P1 &&
        mode != BasicCorrectnessMode::Correctness_P1_to_P5) {
        return 1;
    }

    do {
        if (!CorrectnessTests<TestCorrectnessSharkParams1>())
            return 0;

        if (mode == BasicCorrectnessMode::Correctness_P1_to_P5) {
            if (!CorrectnessTests<TestCorrectnessSharkParams2>())
                return 0;
            if (!CorrectnessTests<TestCorrectnessSharkParams3>())
                return 0;
            if (!CorrectnessTests<TestCorrectnessSharkParams4>())
                return 0;
            if (!CorrectnessTests<TestCorrectnessSharkParams5>())
                return 0;
        }

        ComicalCorrectness();

    } while (HpShark::TestInfiniteCorrectness);

    return PressKey() != 'q';
}

// -----------------------------------------------------------------------------
// Performance modes (split into smaller dispatchable functions)
// -----------------------------------------------------------------------------
static int
RunPerfSubtests(int numIters, int internalTestLoopCount, BasicCorrectnessMode mode)
{
    bool res = true;

    // Add / Multiply / Full perf (delegates per-mode behavior to TestBinaryOperatorPerf)
    res =
        TestBinaryOperatorPerf<Operator::Add>(TestIds::kAddPerf, numIters, internalTestLoopCount, mode);
    if (!ContinueAfterFailure(res))
        return 0;

    res = TestBinaryOperatorPerf<Operator::MultiplyNTT>(
        TestIds::kMultiplyPerf, numIters, internalTestLoopCount, mode);
    if (!ContinueAfterFailure(res))
        return 0;

    res = TestBinaryOperatorPerf<Operator::ReferenceOrbit>(
        TestIds::kFullPerf, numIters, internalTestLoopCount, mode);
    if (!ContinueAfterFailure(res))
        return 0;

    return 1;
}

static int
RunPerfFullSingle(int timeoutInSec, bool &interactiveMode, int numIters, int internalTestLoopCount)
{
    bool res = true;
    TestTracker Tests;

    auto nb = PromptIntWithTimeout("NumBlocks? Default 65", 65, timeoutInSec, interactiveMode);
    auto nt = PromptIntWithTimeout("NumThreads? Default 256", 256, timeoutInSec, interactiveMode);

    int numBlocks = nb.value;
    int numThreads = nt.value;

    res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
        Tests, numBlocks, numThreads, TestIds::kPerfView30, numIters, internalTestLoopCount);
    if (!ContinueAfterFailure(res))
        return 0;

    PressKey();

    res = TestFullReferencePerfView5<Operator::ReferenceOrbit>(
        Tests, numBlocks, numThreads, TestIds::kPerfView5, numIters, internalTestLoopCount);
    if (!ContinueAfterFailure(res))
        return 0;

    return 1;
}

static int
RunPerfFullSweep(int numIters, int internalTestLoopCount)
{
    bool res = true;
    TestTracker Tests;

    int testBaseLocal = TestIds::kPerfSweepStart;

    for (int numBlocks = 16; numBlocks <= 256; numBlocks *= 2) {
        for (int numThreads = 64; numThreads <= 256; numThreads *= 2) {
            res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
                Tests, numBlocks, numThreads, testBaseLocal, numIters, internalTestLoopCount);
            if (!ContinueAfterFailure(res))
                return 0;
            testBaseLocal += 100;

            res = TestFullReferencePerfView5<Operator::ReferenceOrbit>(
                Tests, numBlocks, numThreads, testBaseLocal, numIters, internalTestLoopCount);
            if (!ContinueAfterFailure(res))
                return 0;
            testBaseLocal += 100;
        }
    }

    return Tests.CheckAllTestsPassed();
}

static int
RunPerfModes(BasicCorrectnessMode mode, int timeoutInSec, bool &interactiveMode)
{
    // Only run for perf modes.
    if (mode != BasicCorrectnessMode::PerfSub && mode != BasicCorrectnessMode::PerfSweep &&
        mode != BasicCorrectnessMode::PerfSingle) {
        return 1;
    }

    auto iters = PromptIntWithTimeout("NumIters? Default 5", 5, timeoutInSec, interactiveMode);
    auto loops =
        PromptIntWithTimeout("CUDA iteration count? Default 1000", 1000, timeoutInSec, interactiveMode);

    const int numIters = iters.value;
    const int internalTestLoopCount = loops.value;

    // If PerfSub is selected, run the operator perf suite first.
    if (mode == BasicCorrectnessMode::PerfSub) {
        if (!RunPerfSubtests(numIters, internalTestLoopCount, mode))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingle) {
        if (!RunPerfFullSingle(timeoutInSec, interactiveMode, numIters, internalTestLoopCount))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSweep) {
        if (!RunPerfFullSweep(numIters, internalTestLoopCount))
            return 0;
    }

    return 1;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int
main(int, char **)
{
    GlobalCallstacks->InitCallstacks();

    {
        auto plateaus =
            SharkNTT::BuildPrecisionPlateaus(1048576, HpShark::NTTBHint, HpShark::NTTNumBitsMargin);
        SharkNTT::PrintPlateauTable(plateaus);
        SharkNTT::PrintPrecisionTiers(plateaus);
    }

    constexpr int kTimeoutInSec = 3;
    bool interactiveMode = false; // becomes true after any user input, making later prompts wait forever

    // Verbose
    {
        auto v =
            PromptIntWithTimeout("Verbose? Default=0 (0=No, 1=Yes):", 0, kTimeoutInSec, interactiveMode);
        SetVerboseMode(v.value ? VerboseMode::Debug : VerboseMode::None);
    }

    // Mode prompt: keep default consistent with the enum value
    const int defaultModeInt = static_cast<int>(BasicCorrectnessMode::PerfSingle);
    std::ostringstream modePrompt;
    modePrompt << "Mode? Default=" << defaultModeInt << " "
               << "(0=Correctness(P1), 1=PerfSub, 2=PerfSweep, 3=PerfSingle, 4=Correctness(P1..P5)):";

    int rawMode =
        PromptIntWithTimeout(modePrompt.str(), defaultModeInt, kTimeoutInSec, interactiveMode).value;

    BasicCorrectnessMode mode = BasicCorrectnessMode::PerfSingle;
    switch (rawMode) {
        case 0:
            mode = BasicCorrectnessMode::Correctness_P1;
            break;
        case 1:
            mode = BasicCorrectnessMode::PerfSub;
            break;
        case 2:
            mode = BasicCorrectnessMode::PerfSweep;
            break;
        case 3:
            mode = BasicCorrectnessMode::PerfSingle;
            break;
        case 4:
            mode = BasicCorrectnessMode::Correctness_P1_to_P5;
            break;
        default:
            std::cout << "Invalid mode " << rawMode << " (valid: 0..4). "
                      << "Defaulting to " << defaultModeInt << ".\n";
            mode = BasicCorrectnessMode::PerfSingle;
            break;
    }

    std::cout << "Selected mode: " << static_cast<int>(mode) << " ("
              << BasicCorrectnessModeToString(mode) << ")\n";

    // Explicit dispatch (don’t “call both and early-out”)
    switch (mode) {
        case BasicCorrectnessMode::Correctness_P1:
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            if (!RunCorrectnessTest(mode)) {
                GlobalCallstacks->FreeCallstacks();
                return 0;
            }
            break;

        case BasicCorrectnessMode::PerfSub:
        case BasicCorrectnessMode::PerfSweep:
        case BasicCorrectnessMode::PerfSingle:
            if (!RunPerfModes(mode, kTimeoutInSec, interactiveMode)) {
                GlobalCallstacks->FreeCallstacks();
                return 0;
            }
            break;

        default:
            break;
    }

    GlobalCallstacks->FreeCallstacks();
    return 0;
}
