#include "Conversion.h"
#include "DbgHeap.h"
#include "HpSharkFloat.h"
#include "ShowMostEfficientSizes.h"
#include "TestTracker.h"
#include "TestVerbose.h"
#include "Tests.h"

#include "Callstacks.h"
#include "LaunchParams.h"
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
        case BasicCorrectnessMode::Error:
            return "Error";
        case BasicCorrectnessMode::Correctness_P1:
            return "Correctness (Params1)";
        case BasicCorrectnessMode::PerfSub:
            return "Performance Sub-kernels";
        case BasicCorrectnessMode::PerfSweep:
            return "Performance Sweep";
        case BasicCorrectnessMode::PerfSingleView30:
            return "Performance Single View30";
        case BasicCorrectnessMode::PerfSingleView32:
            return "Performance Single View32";
        case BasicCorrectnessMode::PerfSingleView5:
            return "Performance Single View5";
        case BasicCorrectnessMode::PerfSingleNRView5:
            return "Performance Single NR View5";
        case BasicCorrectnessMode::PerfSingleNRView30:
            return "Performance Single NR View30";
        case BasicCorrectnessMode::PerfSingleNRAdd:
            return "Performance Single NR Add";
        case BasicCorrectnessMode::PerfSingleNRMultiply:
            return "Performance Single NR Multiply";
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
constexpr int kPerfView32 = 16030;
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

    bool waitForever = interactiveMode;

    auto deadline = std::chrono::steady_clock::now() + (waitForever ? std::chrono::hours(24 * 365 * 100)
                                                                    : std::chrono::seconds(timeoutSec));

    std::string buf;
    bool pressedEnter = false;
    bool sawEditingKey = false; // <-- new

    while (waitForever || std::chrono::steady_clock::now() < deadline) {
        if (_kbhit()) {
            int ch = _getch();

            if (ch == '\r' || ch == '\n') {
                pressedEnter = true;
                std::cout << "\n";
                break;
            }

            // Backspace
            if (ch == 8) {
                sawEditingKey = true;
                waitForever = true;     // <-- new: don't timeout mid-edit
                interactiveMode = true; // optional: make prompts sticky globally now

                if (!buf.empty()) {
                    buf.pop_back();
                    std::cout << "\b \b" << std::flush;
                }
                continue;
            }

            // Ctrl+C etc: ignore
            if (ch < 32) {
                continue;
            }

            sawEditingKey = true;
            waitForever = true;     // <-- new
            interactiveMode = true; // optional (see note below)

            buf.push_back((char)ch);
            std::cout << (char)ch << std::flush;
        } else {
            Sleep(sleepIntervalMs);
        }
    }

    PromptResult out;
    out.gotAnyInput = !buf.empty();

    // Timeout case: only possible if we never went waitForever
    if (!pressedEnter && !waitForever && std::chrono::steady_clock::now() >= deadline) {
        std::cout << "\n(no input in " << timeoutSec << "s, defaulting to " << defaultValue << ")\n";
        out.value = defaultValue;
        return out;
    }

    if (buf.empty()) {
        out.value = defaultValue;
        return out;
    }

    try {
        size_t idx = 0;
        int v = std::stoi(buf, &idx, 10);
        for (; idx < buf.size(); ++idx) {
            if (buf[idx] != ' ' && buf[idx] != '\t') {
                throw std::runtime_error("trailing garbage");
            }
        }
        out.value = v;
    } catch (...) {
        std::cout << "(could not parse \"" << buf << "\", defaulting to " << defaultValue << ")\n";
        out.value = defaultValue;
    }
    return out;
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
RunPerfFullSweep(int numIters, int internalTestLoopCount)
{
    bool res = true;
    TestTracker Tests;

    int testBaseLocal = TestIds::kPerfSweepStart;
    constexpr std::pair<int, int> blockThreadPairs[] = {
        {64, 128},
        {64, 256},
        {64, 512},

        {65, 256},

        {128, 128},
        {128, 256},
        {128, 512},

        {129, 256},

        {256, 128},
        {256, 256},

        {170, 128},
        {170, 256},
        {170, 512},

        {340, 128},
        {340, 256},
    };

    for (const auto &[numBlocks, numThreads] : blockThreadPairs) {
        res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
            Tests, numBlocks, numThreads, testBaseLocal, numIters, internalTestLoopCount);
        if (!ContinueAfterFailure(res))
            return 0;
        testBaseLocal += 100;

        res = TestFullReferencePerfView32<Operator::ReferenceOrbit>(
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

    return Tests.CheckAllTestsPassed();
}

static int
RunPerfModes(BasicCorrectnessMode mode, int timeoutInSec, bool &interactiveMode)
{
    // Only run for perf modes.
    if (mode != BasicCorrectnessMode::PerfSub && mode != BasicCorrectnessMode::PerfSweep &&
        mode != BasicCorrectnessMode::PerfSingleView30 &&
        mode != BasicCorrectnessMode::PerfSingleView32 &&
        mode != BasicCorrectnessMode::PerfSingleView5) {
        return 1;
    }

    auto iters = PromptIntWithTimeout("NumIters? Default 5", 5, timeoutInSec, interactiveMode);
    auto loops =
        PromptIntWithTimeout("CUDA iteration count? Default 1000", 1000, timeoutInSec, interactiveMode);
    auto numBlocks = PromptIntWithTimeout("NumBlocks? Default 65, 0 for auto", 65, timeoutInSec, interactiveMode);
    auto numThreads =
        PromptIntWithTimeout("NumThreads? Default 256, 0 for auto", 256, timeoutInSec, interactiveMode);
    const HpShark::LaunchParams launchParams{numBlocks.value, numThreads.value};

    const int numIters = iters.value;
    const int internalTestLoopCount = loops.value;

    // If PerfSub is selected, run the operator perf suite first.
    if (mode == BasicCorrectnessMode::PerfSub) {
        bool res = true;

        // Add / Multiply / Full perf (delegates per-mode behavior to TestBinaryOperatorPerf)
        res = TestBinaryOperatorPerf<Operator::Add>(
            launchParams, TestIds::kAddPerf, numIters, internalTestLoopCount, mode);
        if (!ContinueAfterFailure(res))
            return 0;

        res = TestBinaryOperatorPerf<Operator::MultiplyNTT>(
            launchParams, TestIds::kMultiplyPerf, numIters, internalTestLoopCount, mode);
        if (!ContinueAfterFailure(res))
            return 0;

        res = TestBinaryOperatorPerf<Operator::ReferenceOrbit>(
            launchParams, TestIds::kFullPerf, numIters, internalTestLoopCount, mode);
        if (!ContinueAfterFailure(res))
            return 0;

        return 1;
    }

    if (mode == BasicCorrectnessMode::PerfSingleView30) {
        TestTracker Tests;
        auto res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
            Tests, launchParams.NumBlocks, launchParams.ThreadsPerBlock,
            TestIds::kPerfView30, numIters, internalTestLoopCount);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingleView5) {
        TestTracker Tests;
        auto res = TestFullReferencePerfView5<Operator::ReferenceOrbit>(
            Tests, launchParams.NumBlocks, launchParams.ThreadsPerBlock,
            TestIds::kPerfView5, numIters, internalTestLoopCount);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingleView32) {
        TestTracker Tests;
        auto res = TestFullReferencePerfView32<Operator::ReferenceOrbit>(
            Tests, launchParams.NumBlocks, launchParams.ThreadsPerBlock,
            TestIds::kPerfView32, numIters, internalTestLoopCount);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSweep) {
        if (!RunPerfFullSweep(numIters, internalTestLoopCount))
            return 0;
    }

    return 1;
}

template <Operator op>
static int
RunPerfBasicOp(int testBase, BasicCorrectnessMode mode, int timeoutInSec, bool &interactiveMode)     {
    auto iters = PromptIntWithTimeout("NumIters? Default 5", 5, timeoutInSec, interactiveMode);
    auto loops =
        PromptIntWithTimeout("CUDA iteration count? Default 1000", 1000, timeoutInSec, interactiveMode);
    auto numBlocks =
        PromptIntWithTimeout("NumBlocks? Default 65, 0 for auto", 65, timeoutInSec, interactiveMode);
    auto numThreads =
        PromptIntWithTimeout("NumThreads? Default 256, 0 for auto", 256, timeoutInSec, interactiveMode);
    const HpShark::LaunchParams launchParams{numBlocks.value, numThreads.value};

    const int numIters = iters.value;
    const int internalTestLoopCount = loops.value;

    auto res = TestBinaryOperatorPerf<op>(launchParams, testBase, numIters, internalTestLoopCount, mode);
    if (!ContinueAfterFailure(res))
        return 0;

    return 1;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int
main(int, char **)
{
    RegisterHeapCleanup();
    GlobalCallstacks->InitCallstacks();

    {
        auto plateaus =
            SharkNTT::BuildPrecisionPlateaus(1048576, HpShark::NTTBHint, HpShark::NTTNumBitsMargin);
        SharkNTT::PrintPlateauTable(plateaus);
        SharkNTT::PrintPrecisionTiers(plateaus);
    }

    constexpr int kTimeoutInSec = 3;
    bool interactiveMode = false; // becomes true after any user input, making later prompts wait forever

    // Mode prompt: keep default consistent with the enum value
    const int defaultModeInt = static_cast<int>(BasicCorrectnessMode::PerfSingleView30);
    std::ostringstream modePrompt;

    modePrompt << "Mode? Default=" << defaultModeInt << " "
               << "1=Correctness(P1)" << std::endl
               << "2=PerfSub" << std::endl
               << "3=PerfSweep" << std::endl
               << "4=PerfSingle View30" << std::endl
               << "5=PerfSingle View32" << std::endl
               << "6=PerfSingle View5" << std::endl
               << "7=PerfSingleAdd" << std::endl
               << "8=PerfSingleMultiply" << std::endl
               << "9=PerfSingleNRAdd" << std::endl
               << "10=PerfSingleNRMultiply" << std::endl
               << "11=PerfSingleRef (broken currently)" << std::endl
               << "12=Correctness(P1..P5)" << std::endl
               << "13=NR View5" << std::endl
               << "14=NR View30" << std::endl
               << "anything else=Exit" << std::endl
               << "Enter choice:";

    int rawMode =
        PromptIntWithTimeout(modePrompt.str(), defaultModeInt, kTimeoutInSec, interactiveMode).value;

    BasicCorrectnessMode mode = BasicCorrectnessMode::PerfSingleView30;
    switch (rawMode) {
        case 1:
            mode = BasicCorrectnessMode::Correctness_P1;
            break;
        case 2:
            mode = BasicCorrectnessMode::PerfSub;
            break;
        case 3:
            mode = BasicCorrectnessMode::PerfSweep;
            break;
        case 4:
            mode = BasicCorrectnessMode::PerfSingleView30;
            break;
        case 5:
            mode = BasicCorrectnessMode::PerfSingleView32;
            break;
        case 6:
            mode = BasicCorrectnessMode::PerfSingleView5;
            break;
        case 7:
            mode = BasicCorrectnessMode::PerfSingleAdd;
            break;
        case 8:
            mode = BasicCorrectnessMode::PerfSingleMultiply;
            break;
        case 9:
            mode = BasicCorrectnessMode::PerfSingleNRAdd;
            break;
        case 10:
            mode = BasicCorrectnessMode::PerfSingleNRMultiply;
            break;
        case 11:
            mode = BasicCorrectnessMode::PerfSingleRef;
            break;
        case 12:
            mode = BasicCorrectnessMode::Correctness_P1_to_P5;
            break;
        case 13:
            mode = BasicCorrectnessMode::PerfSingleNRView5;
            break;
        case 14:
            mode = BasicCorrectnessMode::PerfSingleNRView30;
            break;
        default:
            std::cout << "Invalid mode " << rawMode << " (valid: 0..4). "
                      << "Exiting.\n";
            mode = BasicCorrectnessMode::Error;
            break;
    }

    std::cout << "Selected mode: " << static_cast<int>(mode) << " ("
              << BasicCorrectnessModeToString(mode) << ")\n";

    // Verbose
    if (mode != BasicCorrectnessMode::Error) {
        auto v =
            PromptIntWithTimeout("Verbose? Default=0 (0=No, 1=Yes):", 0, kTimeoutInSec, interactiveMode);
        SetVerboseMode(v.value ? VerboseMode::Debug : VerboseMode::None);
    }

    // Explicit dispatch (don’t “call both and early-out”)
    switch (mode) {
        case BasicCorrectnessMode::Correctness_P1:
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            RunCorrectnessTest(mode);
            break;

        case BasicCorrectnessMode::PerfSub:
        case BasicCorrectnessMode::PerfSweep:
        case BasicCorrectnessMode::PerfSingleView30:
        case BasicCorrectnessMode::PerfSingleView32:
        case BasicCorrectnessMode::PerfSingleView5:
            RunPerfModes(mode, kTimeoutInSec, interactiveMode);
            break;

        case BasicCorrectnessMode::PerfSingleAdd:
            RunPerfBasicOp<Operator::Add>(TestIds::kAddPerf, mode, kTimeoutInSec, interactiveMode);
            break;
        case BasicCorrectnessMode::PerfSingleMultiply:
            RunPerfBasicOp<Operator::MultiplyNTT>(
                TestIds::kMultiplyPerf, mode, kTimeoutInSec, interactiveMode);
            break;
        case BasicCorrectnessMode::PerfSingleRef:
            RunPerfBasicOp<Operator::ReferenceOrbit>(
                TestIds::kFullPerf, mode, kTimeoutInSec, interactiveMode);
            break;

        case BasicCorrectnessMode::PerfSingleNRAdd: {
            TestTracker Tests;
            auto res = TestSingleNRAdd<SharkParamsNR7>(Tests, 0);
            if (!ContinueAfterFailure(res))
                return 0;
            break;
        }

        case BasicCorrectnessMode::PerfSingleNRMultiply: {
            TestTracker Tests;
            auto res = TestSingleNRMultiply<SharkParamsNR7>(Tests, 0);
            if (!ContinueAfterFailure(res))
                return 0;
            break;
        }

        case BasicCorrectnessMode::PerfSingleNRView5: {
            TestTracker Tests;
            auto res = TestNewtonRaphsonView5<SharkParamsNR7>(Tests, 0);
            if (!ContinueAfterFailure(res))
                return 0;
            break;
        }

        case BasicCorrectnessMode::PerfSingleNRView30: {
            TestTracker Tests;
            auto res = TestNewtonRaphsonView30<SharkParamsNR7>(Tests, 0);
            if (!ContinueAfterFailure(res))
                return 0;
            break;
        }

        default:
            break;
    }

    GlobalCallstacks->FreeCallstacks();
    return 0;
}
