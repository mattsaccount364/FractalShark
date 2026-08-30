#include "CommandLineOptions.h"
#include "Conversion.h"
#include "DbgHeap.h"
#include "GpuPrecisionDispatch.h"
#include "HpSharkFloat.h"
#include "HpSharkTestConfig.h"
#include "KernelInvoke.h"
#include "TestTracker.h"
#include "TestVerbose.h"
#include "Tests.h"

#include "HDRFloat.h"
#include "LaunchParams.h"
#include "MainTestCuda.h"
#include "heap_allocator/include/HeapCpp.h"

#include <cuda_runtime.h>

#include "Environment.h"
#include <charconv>
#include <chrono>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

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
        case BasicCorrectnessMode::Correctness_NR:
            return "Correctness NR";
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
        case BasicCorrectnessMode::PerfSingleNRView32:
            return "Performance Single NR View32";
        case BasicCorrectnessMode::PerfSingleRef:
            return "Performance Single Reference Orbit";
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            return "Correctness (Params1..5)";
        case BasicCorrectnessMode::PerfSingleViewAny:
            return "Performance Single View (any)";
        default:
            return "Unknown";
    }
}

// -----------------------------------------------------------------------------
// Test base IDs (remove magic numbers)
// -----------------------------------------------------------------------------
namespace TestIds {
constexpr int kConversion = 0;

constexpr int kFullCorrectness = 2000;

constexpr int kFullPerf = 14000;

constexpr int kPerfView30 = 16020;
constexpr int kPerfView32 = 16030;
constexpr int kPerfView5 = 16010;
constexpr int kPerfViewAny = 16100;

constexpr int kPerfSweepStart = 1000;
} // namespace TestIds

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
static char
PressKey()
{
    std::cout << "Press any key to continue...";
    return static_cast<char>(Environment::ConsoleReadCharBlocking());
}

static bool
ContinueAfterFailure(bool res)
{
    if (res)
        return true;
    return PressKey() != 'q';
}

/// Robust console line input using Environment console input (no std::getline mixing).
/// - If user provides any input (even invalid), interactive mode becomes true.
/// - If interactive mode is true, subsequent prompts wait indefinitely.
/// - On timeout:
///     * if buffer empty -> returns defaultValue
///     * if buffer non-empty (user started typing but didn't press Enter) -> accepts buffer
struct PromptResult {
    int value = 0;
    bool gotAnyInput = false; // user typed something (even if invalid)
    bool parsed = false;
    std::string input;
};

static std::string_view
TrimPromptText(std::string_view text)
{
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
        text.remove_prefix(1);
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) {
        text.remove_suffix(1);
    }
    return text;
}

static std::optional<CommandLineValueKind>
TryParseLimbSelectionKeyword(std::string_view text)
{
    text = TrimPromptText(text);
    if (text == "auto") {
        return CommandLineValueKind::Auto;
    }
    if (text == "production") {
        return CommandLineValueKind::Production;
    }
    return std::nullopt;
}

static bool
TryParsePromptInt(std::string_view text, int &value)
{
    if (text.empty()) {
        return false;
    }

    const char *first = text.data();
    const char *last = first + text.size();
    while (first != last && (*first == ' ' || *first == '\t')) {
        ++first;
    }

    if (first == last) {
        return false;
    }

    if (*first == '+') {
        ++first;
        if (first == last || *first == '-') {
            return false;
        }
    }

    int parsedValue = 0;
    const auto parseResult = std::from_chars(first, last, parsedValue, 10);
    if (parseResult.ec != std::errc{} || parseResult.ptr == first) {
        return false;
    }

    for (const char *ptr = parseResult.ptr; ptr != last; ++ptr) {
        if (*ptr != ' ' && *ptr != '\t') {
            return false;
        }
    }

    value = parsedValue;
    return true;
}

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
    while (waitForever || std::chrono::steady_clock::now() < deadline) {
        if (Environment::ConsoleKeyAvailable()) {
            int ch = Environment::ConsoleReadCharBlocking();

            if (ch == '\r' || ch == '\n') {
                pressedEnter = true;
                std::cout << "\n";
                break;
            }

            // Backspace
            if (ch == '\b' || ch == 0x7f) {
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

            waitForever = true;     // <-- new
            interactiveMode = true; // optional (see note below)

            buf.push_back((char)ch);
            std::cout << (char)ch << std::flush;
        } else {
            Environment::SleepMs(sleepIntervalMs);
        }
    }

    PromptResult out;
    out.gotAnyInput = !buf.empty();
    out.input = buf;

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

    int value = 0;
    if (TryParsePromptInt(buf, value)) {
        out.value = value;
        out.parsed = true;
    } else if (TryParseLimbSelectionKeyword(buf)) {
        out.value = defaultValue;
    } else {
        std::cout << "(could not parse \"" << buf << "\", defaulting to " << defaultValue << ")\n";
        out.value = defaultValue;
    }
    return out;
}

template <typename T>
static bool
IsCommandLineSupplied(const CommandLineOptionValue<T> &option)
{
    return option.m_Kind != CommandLineValueKind::Omitted;
}

static int
ResolveIntOption(const CommandLineOptionValue<int> &option,
                 const std::string &promptText,
                 int defaultValue,
                 int timeoutInSec,
                 bool &interactiveMode)
{
    if (option.m_Kind == CommandLineValueKind::Explicit) {
        return option.m_Value;
    }
    if (option.m_Kind == CommandLineValueKind::Auto) {
        return defaultValue;
    }
    return PromptIntWithTimeout(promptText, defaultValue, timeoutInSec, interactiveMode).value;
}

static bool
IsRunPerfModesMode(BasicCorrectnessMode mode)
{
    return mode == BasicCorrectnessMode::PerfSweep || mode == BasicCorrectnessMode::PerfSingleView30 ||
           mode == BasicCorrectnessMode::PerfSingleView32 ||
           mode == BasicCorrectnessMode::PerfSingleView5 ||
           mode == BasicCorrectnessMode::PerfSingleNRView5 ||
           mode == BasicCorrectnessMode::PerfSingleNRView30 ||
           mode == BasicCorrectnessMode::PerfSingleNRView32 ||
           mode == BasicCorrectnessMode::PerfSingleViewAny;
}

static bool
IsRunPerfBasicOpMode(BasicCorrectnessMode mode)
{
    return mode == BasicCorrectnessMode::PerfSingleRef;
}

static bool
IsFullReferenceViewMode(BasicCorrectnessMode mode)
{
    return mode == BasicCorrectnessMode::PerfSingleView30 ||
           mode == BasicCorrectnessMode::PerfSingleView32 ||
           mode == BasicCorrectnessMode::PerfSingleView5 ||
           mode == BasicCorrectnessMode::PerfSingleViewAny;
}

static bool
ValidateCommandLineApplicability(const CommandLineOptions &options, BasicCorrectnessMode mode)
{
    const bool isPerfModesMode = IsRunPerfModesMode(mode);
    const bool isBasicOpMode = IsRunPerfBasicOpMode(mode);
    const bool isSweepMode = mode == BasicCorrectnessMode::PerfSweep;
    const bool isViewMode = IsFullReferenceViewMode(mode) || isSweepMode;

    if ((IsCommandLineSupplied(options.m_CudaIterations) || IsCommandLineSupplied(options.m_NumIters) ||
         IsCommandLineSupplied(options.m_NumBlocks) || IsCommandLineSupplied(options.m_NumThreads)) &&
        !isPerfModesMode && !isBasicOpMode) {
        std::cerr << "Performance options require a performance mode.\n";
        return false;
    }

    if (IsCommandLineSupplied(options.m_MpirThreading) && !isPerfModesMode) {
        std::cerr << "--mpir-threading is only valid for the view/operator performance modes.\n";
        return false;
    }

    if (IsCommandLineSupplied(options.m_View) && mode != BasicCorrectnessMode::PerfSingleViewAny) {
        std::cerr << "--view is only valid with mode 12 (PerfSingle View Any).\n";
        return false;
    }

    const bool isNrView5 = mode == BasicCorrectnessMode::PerfSingleNRView5;
    if ((IsCommandLineSupplied(options.m_StorageLimbs) ||
         IsCommandLineSupplied(options.m_EffectiveLimbs)) &&
        !isViewMode && !isNrView5) {
        std::cerr
            << "Limb options require a full-reference view mode, NR view 5, or the performance sweep.\n";
        return false;
    }

    return true;
}

struct FullReferencePerfLimbOptions {
    CommandLineOptionValue<uint32_t> m_StorageLimbs;
    CommandLineOptionValue<uint32_t> m_EffectiveLimbs;
};

static bool
IsProductionLimbSelectionRequested(const FullReferencePerfLimbOptions &options)
{
    return options.m_StorageLimbs.m_Kind == CommandLineValueKind::Production ||
           options.m_EffectiveLimbs.m_Kind == CommandLineValueKind::Production;
}

static FullReferencePerfLimbSelection
ResolveAutomaticLimbSelection(Operator referenceOperator,
                              const FullReferencePerfPrecision &precision,
                              const FullReferencePerfLimbOptions &options)
{
    const FullReferencePerfLimbSelection &defaultSelection = IsProductionLimbSelectionRequested(options)
                                                                 ? precision.m_ProductionSelection
                                                                 : precision.m_DefaultSelection;
    FullReferencePerfLimbSelection selection = defaultSelection;

    if (options.m_StorageLimbs.m_Kind == CommandLineValueKind::Explicit) {
        selection.m_StorageLimbs = options.m_StorageLimbs.m_Value;
        if (options.m_EffectiveLimbs.m_Kind != CommandLineValueKind::Explicit) {
            selection.m_EffectiveLimbs = GetFullReferencePerfEffectiveLimbs(
                referenceOperator, precision.m_RequestedPrecisionLimbs, selection.m_StorageLimbs);
        }
    }
    if (options.m_EffectiveLimbs.m_Kind == CommandLineValueKind::Explicit) {
        selection.m_EffectiveLimbs = options.m_EffectiveLimbs.m_Value;
    }

    return selection;
}

static bool
PromptSupportedLimbCount(const std::string &promptText,
                         uint32_t defaultValue,
                         int timeoutInSec,
                         bool &interactiveMode,
                         CommandLineOptionValue<uint32_t> &option)
{
    for (;;) {
        const PromptResult prompt = PromptIntWithTimeout(
            promptText, static_cast<int>(defaultValue), timeoutInSec, interactiveMode);
        if (!prompt.gotAnyInput) {
            option.m_Kind = CommandLineValueKind::Auto;
            option.m_Value = defaultValue;
            return true;
        }

        if (const auto keyword = TryParseLimbSelectionKeyword(prompt.input)) {
            option.m_Kind = *keyword;
            option.m_Value = defaultValue;
            return true;
        }

        if (prompt.parsed && prompt.value >= 0 &&
            IsSupportedLimbCount(static_cast<uint32_t>(prompt.value))) {
            option.m_Kind = CommandLineValueKind::Explicit;
            option.m_Value = static_cast<uint32_t>(prompt.value);
            return true;
        }

        std::cout << "Storage limbs must be one of 256, 512, 1024, 2048, 4096, 8192, "
                     "16384, 32768, 65536, 131072, 262144, or 524288; enter auto or "
                     "production for a preset.\n";
    }
}

static bool
PromptEffectiveLimbCount(const std::string &promptText,
                         uint32_t defaultValue,
                         uint32_t minimumValue,
                         uint32_t maximumValue,
                         int timeoutInSec,
                         bool &interactiveMode,
                         CommandLineOptionValue<uint32_t> &option)
{
    for (;;) {
        const PromptResult prompt = PromptIntWithTimeout(
            promptText, static_cast<int>(defaultValue), timeoutInSec, interactiveMode);
        if (!prompt.gotAnyInput) {
            option.m_Kind = CommandLineValueKind::Auto;
            option.m_Value = defaultValue;
            return true;
        }

        if (const auto keyword = TryParseLimbSelectionKeyword(prompt.input)) {
            option.m_Kind = *keyword;
            option.m_Value = defaultValue;
            return true;
        }

        if (prompt.parsed && prompt.value >= 0 && static_cast<uint32_t>(prompt.value) >= minimumValue &&
            static_cast<uint32_t>(prompt.value) <= maximumValue) {
            option.m_Kind = CommandLineValueKind::Explicit;
            option.m_Value = static_cast<uint32_t>(prompt.value);
            return true;
        }

        std::cout << "Effective limbs must be in the range " << minimumValue << ".." << maximumValue
                  << "; enter auto or production for a preset.\n";
    }
}

static bool
PromptSweepLimbCount(const std::string &promptText,
                     bool storageLimbs,
                     int timeoutInSec,
                     bool &interactiveMode,
                     CommandLineOptionValue<uint32_t> &option)
{
    for (;;) {
        const PromptResult prompt = PromptIntWithTimeout(promptText, 0, timeoutInSec, interactiveMode);
        if (!prompt.gotAnyInput) {
            option.m_Kind = CommandLineValueKind::Auto;
            option.m_Value = 0;
            return true;
        }

        if (const auto keyword = TryParseLimbSelectionKeyword(prompt.input)) {
            option.m_Kind = *keyword;
            option.m_Value = 0;
            return true;
        }

        if (prompt.parsed && prompt.value > 0 &&
            (!storageLimbs || IsSupportedLimbCount(static_cast<uint32_t>(prompt.value)))) {
            option.m_Kind = CommandLineValueKind::Explicit;
            option.m_Value = static_cast<uint32_t>(prompt.value);
            return true;
        }

        if (storageLimbs) {
            std::cout << "Enter a supported storage limb count, auto, or production.\n";
        } else {
            std::cout << "Enter a positive effective limb count, auto, or production.\n";
        }
    }
}

template <Operator referenceOperator>
static bool
ResolveSingleViewLimbSelection(const CommandLineOptions &options,
                               size_t view,
                               int timeoutInSec,
                               bool &interactiveMode,
                               FullReferencePerfLimbSelection &selection)
{
    const auto precision = GetFullReferencePerfPrecision(referenceOperator, view);
    FullReferencePerfLimbOptions limbOptions{options.m_StorageLimbs, options.m_EffectiveLimbs};

    const bool productionDefault = IsProductionLimbSelectionRequested(limbOptions);
    const auto &defaultSelection =
        productionDefault ? precision.m_ProductionSelection : precision.m_DefaultSelection;
    if (limbOptions.m_StorageLimbs.m_Kind == CommandLineValueKind::Omitted) {
        std::ostringstream prompt;
        prompt << "Storage limbs? Default " << defaultSelection.m_StorageLimbs;
        if (!productionDefault && precision.m_DefaultIsBenchmarkPreset) {
            prompt << " (benchmark preset; production-derived storage "
                   << precision.m_ProductionSelection.m_StorageLimbs << ", effective "
                   << precision.m_ProductionSelection.m_EffectiveLimbs << ')';
        } else {
            prompt << " (production-derived)";
        }
        prompt << " (supported powers of two 256..524288):";
        if (view == 5)
            prompt << " view 5 low-limb=256..1024, high-limb=2048+";
        PromptSupportedLimbCount(prompt.str(),
                                 defaultSelection.m_StorageLimbs,
                                 timeoutInSec,
                                 interactiveMode,
                                 limbOptions.m_StorageLimbs);
    }

    if (limbOptions.m_EffectiveLimbs.m_Kind == CommandLineValueKind::Omitted) {
        FullReferencePerfLimbOptions automaticOptions = limbOptions;
        automaticOptions.m_EffectiveLimbs.m_Kind = CommandLineValueKind::Auto;
        const auto automaticSelection =
            ResolveAutomaticLimbSelection(referenceOperator, precision, automaticOptions);
        std::ostringstream prompt;
        prompt << "Effective limbs? Default " << automaticSelection.m_EffectiveLimbs << " (range "
               << GetMinimumFullReferencePerfEffectiveLimbs(referenceOperator,
                                                            automaticSelection.m_StorageLimbs)
               << ".." << automaticSelection.m_StorageLimbs;
        const bool effectiveProductionDefault = IsProductionLimbSelectionRequested(limbOptions);
        if (!effectiveProductionDefault && precision.m_DefaultIsBenchmarkPreset &&
            limbOptions.m_StorageLimbs.m_Kind != CommandLineValueKind::Explicit) {
            prompt << "; benchmark preset; production-derived "
                   << precision.m_ProductionSelection.m_EffectiveLimbs;
        } else if (effectiveProductionDefault) {
            prompt << "; production-derived";
        }
        prompt << "):";
        PromptEffectiveLimbCount(prompt.str(),
                                 automaticSelection.m_EffectiveLimbs,
                                 GetMinimumFullReferencePerfEffectiveLimbs(
                                     referenceOperator, automaticSelection.m_StorageLimbs),
                                 automaticSelection.m_StorageLimbs,
                                 timeoutInSec,
                                 interactiveMode,
                                 limbOptions.m_EffectiveLimbs);
    }

    selection = ResolveAutomaticLimbSelection(referenceOperator, precision, limbOptions);
    if (!IsValidFullReferencePerfLimbSelection(referenceOperator, selection)) {
        std::cerr << "Invalid limb selection for the selected reference implementation: storage="
                  << selection.m_StorageLimbs << ", effective=" << selection.m_EffectiveLimbs << ".\n";
        return false;
    }
    return true;
}

static bool
ResolveSweepLimbOverride(const CommandLineOptions &options,
                         int timeoutInSec,
                         bool &interactiveMode,
                         FullReferencePerfLimbOptions &limbOptions)
{
    limbOptions = FullReferencePerfLimbOptions{options.m_StorageLimbs, options.m_EffectiveLimbs};
    if (limbOptions.m_StorageLimbs.m_Kind == CommandLineValueKind::Omitted) {
        PromptSweepLimbCount(
            "Storage limbs override for all sweep views? Default auto (saved benchmark presets):",
            true,
            timeoutInSec,
            interactiveMode,
            limbOptions.m_StorageLimbs);
    }
    if (limbOptions.m_EffectiveLimbs.m_Kind == CommandLineValueKind::Omitted) {
        PromptSweepLimbCount(
            "Effective limbs override for all sweep views? Default auto (saved benchmark presets):",
            false,
            timeoutInSec,
            interactiveMode,
            limbOptions.m_EffectiveLimbs);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Correctness tests
// -----------------------------------------------------------------------------
template <typename TestSharkParams>
static bool
CorrectnessTests()
{
    bool res = true;

    res = TestAllBinaryOp<TestSharkParams, Operator::ReferenceOrbit2>(TestIds::kFullCorrectness);
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

    } while (HpShark::TestInfiniteCorrectness);

    return PressKey() != 'q';
}

// -----------------------------------------------------------------------------
// Performance modes (split into smaller dispatchable functions)
// -----------------------------------------------------------------------------

template <Operator referenceOperator>
static int
RunPerfFullSweep(int numIters,
                 int internalTestLoopCount,
                 const FullReferencePerfLimbOptions &limbOptions)
{
    static_assert(IsReferenceOrbitOperator<referenceOperator>);

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
        const auto precision30 = GetFullReferencePerfPrecision(referenceOperator, 30);
        const FullReferencePerfLimbSelection selection30 =
            ResolveAutomaticLimbSelection(referenceOperator, precision30, limbOptions);
        if (!IsValidFullReferencePerfLimbSelection(referenceOperator, selection30)) {
            std::cerr << "Invalid sweep limb selection for view 30: storage="
                      << selection30.m_StorageLimbs << ", effective=" << selection30.m_EffectiveLimbs
                      << ".\n";
            return 0;
        }
        res = TestFullReferencePerfView<referenceOperator>(Tests,
                                                           numBlocks,
                                                           numThreads,
                                                           testBaseLocal,
                                                           numIters,
                                                           internalTestLoopCount,
                                                           true,
                                                           30,
                                                           selection30);
        if (!ContinueAfterFailure(res))
            return 0;
        testBaseLocal += 100;

        const auto precision32 = GetFullReferencePerfPrecision(referenceOperator, 32);
        const FullReferencePerfLimbSelection selection32 =
            ResolveAutomaticLimbSelection(referenceOperator, precision32, limbOptions);
        if (!IsValidFullReferencePerfLimbSelection(referenceOperator, selection32)) {
            std::cerr << "Invalid sweep limb selection for view 32: storage="
                      << selection32.m_StorageLimbs << ", effective=" << selection32.m_EffectiveLimbs
                      << ".\n";
            return 0;
        }
        res = TestFullReferencePerfView<referenceOperator>(Tests,
                                                           numBlocks,
                                                           numThreads,
                                                           testBaseLocal,
                                                           numIters,
                                                           internalTestLoopCount,
                                                           true,
                                                           32,
                                                           selection32);
        if (!ContinueAfterFailure(res))
            return 0;
        testBaseLocal += 100;

        const auto precision5 = GetFullReferencePerfPrecision(referenceOperator, 5);
        const FullReferencePerfLimbSelection selection5 =
            ResolveAutomaticLimbSelection(referenceOperator, precision5, limbOptions);
        if (!IsValidFullReferencePerfLimbSelection(referenceOperator, selection5)) {
            std::cerr << "Invalid sweep limb selection for view 5: storage=" << selection5.m_StorageLimbs
                      << ", effective=" << selection5.m_EffectiveLimbs << ".\n";
            return 0;
        }
        res = TestFullReferencePerfView<referenceOperator>(Tests,
                                                           numBlocks,
                                                           numThreads,
                                                           testBaseLocal,
                                                           numIters,
                                                           internalTestLoopCount,
                                                           true,
                                                           5,
                                                           selection5);
        if (!ContinueAfterFailure(res))
            return 0;
        testBaseLocal += 100;
    }

    return Tests.CheckAllTestsPassed();
}

template <Operator referenceOperator>
static int
RunPerfModes(BasicCorrectnessMode mode,
             int timeoutInSec,
             bool &interactiveMode,
             const CommandLineOptions &options)
{
    static_assert(IsReferenceOrbitOperator<referenceOperator>);

    // Only run for perf modes.
    if (mode != BasicCorrectnessMode::PerfSweep && mode != BasicCorrectnessMode::PerfSingleView30 &&
        mode != BasicCorrectnessMode::PerfSingleView32 &&
        mode != BasicCorrectnessMode::PerfSingleView5 &&
        mode != BasicCorrectnessMode::PerfSingleNRView5 &&
        mode != BasicCorrectnessMode::PerfSingleNRView30 &&
        mode != BasicCorrectnessMode::PerfSingleNRView32 &&
        mode != BasicCorrectnessMode::PerfSingleViewAny) {
        return 1;
    }

    const bool isNRMode = (mode == BasicCorrectnessMode::PerfSingleNRView5 ||
                           mode == BasicCorrectnessMode::PerfSingleNRView30 ||
                           mode == BasicCorrectnessMode::PerfSingleNRView32);

    const int internalTestLoopCount =
        ResolveIntOption(options.m_CudaIterations,
                         "CUDA iteration count? Default 1000 (NR: 0=convergence)",
                         1000,
                         timeoutInSec,
                         interactiveMode);

    // NumIters: skip for NR convergence mode (internalTestLoopCount == 0)
    int numIters = 1;
    if (!(isNRMode && internalTestLoopCount == 0)) {
        numIters = ResolveIntOption(
            options.m_NumIters, "NumIters? Default 5", 5, timeoutInSec, interactiveMode);
    }

    const int numBlocks = ResolveIntOption(
        options.m_NumBlocks, "NumBlocks? Default 65, 0 for auto", 65, timeoutInSec, interactiveMode);
    const int numThreads = ResolveIntOption(
        options.m_NumThreads, "NumThreads? Default 256, 0 for auto", 256, timeoutInSec, interactiveMode);
    const HpShark::LaunchParams launchParams{numBlocks, numThreads};

    // MPIR threading option for view modes
    const int mpirThreading = ResolveIntOption(options.m_MpirThreading,
                                               "MPIR threading? 0=MT(default), 1=ST:",
                                               0,
                                               timeoutInSec,
                                               interactiveMode);
    const bool useMT = (mpirThreading == 0);

    size_t selectedView = 0;
    FullReferencePerfLimbSelection selectedLimbSelection;
    if (IsFullReferenceViewMode(mode)) {
        if (mode == BasicCorrectnessMode::PerfSingleView30) {
            selectedView = 30;
        } else if (mode == BasicCorrectnessMode::PerfSingleView32) {
            selectedView = 32;
        } else if (mode == BasicCorrectnessMode::PerfSingleView5) {
            selectedView = 5;
        } else {
            const int view = ResolveIntOption(options.m_View,
                                              "View number (1..34)?  5/30/32 use verified baselines",
                                              5,
                                              timeoutInSec,
                                              interactiveMode);
            if (view < 1 || view > 34) {
                std::cerr << "View number must be in the range 1..34.\n";
                return 0;
            }
            selectedView = static_cast<size_t>(view);
        }

        if (!ResolveSingleViewLimbSelection<referenceOperator>(
                options, selectedView, timeoutInSec, interactiveMode, selectedLimbSelection)) {
            return 0;
        }
    } else if (mode == BasicCorrectnessMode::PerfSingleNRView5) {
        selectedView = 5;
        if (!ResolveSingleViewLimbSelection<referenceOperator>(
                options, selectedView, timeoutInSec, interactiveMode, selectedLimbSelection)) {
            return 0;
        }
    }

    FullReferencePerfLimbOptions sweepLimbOptions;
    if (mode == BasicCorrectnessMode::PerfSweep &&
        !ResolveSweepLimbOverride(options, timeoutInSec, interactiveMode, sweepLimbOptions)) {
        return 0;
    }
    if (mode == BasicCorrectnessMode::PerfSweep) {
        const auto precision5 = GetFullReferencePerfPrecision(referenceOperator, 5);
        const FullReferencePerfLimbSelection selection5 =
            ResolveAutomaticLimbSelection(referenceOperator, precision5, sweepLimbOptions);
        if (!IsValidFullReferencePerfLimbSelection(referenceOperator, selection5)) {
            std::cerr << "Invalid sweep limb selection for view 5: storage=" << selection5.m_StorageLimbs
                      << ", effective=" << selection5.m_EffectiveLimbs << ".\n";
            return 0;
        }
    }

    if (IsFullReferenceViewMode(mode)) {
        TestTracker Tests;
        auto res = TestFullReferencePerfView<referenceOperator>(
            Tests,
            launchParams.NumBlocks,
            launchParams.ThreadsPerBlock,
            static_cast<int>(mode == BasicCorrectnessMode::PerfSingleView30   ? TestIds::kPerfView30
                             : mode == BasicCorrectnessMode::PerfSingleView32 ? TestIds::kPerfView32
                             : mode == BasicCorrectnessMode::PerfSingleView5  ? TestIds::kPerfView5
                                                                              : TestIds::kPerfViewAny),
            numIters,
            internalTestLoopCount,
            useMT,
            selectedView,
            selectedLimbSelection);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingleNRView5) {
        TestTracker Tests;
        bool res = true;
        auto runNrView5 = [&]<class SharkFloatParams>() {
            res = TestNewtonRaphsonView5<SharkFloatParams, referenceOperator>(
                Tests, 0, launchParams, static_cast<uint64_t>(internalTestLoopCount), useMT, numIters);
        };
        DispatchByLimbCount<SharkParamsNRFamily>(selectedLimbSelection.m_StorageLimbs, runNrView5);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingleNRView30) {
        TestTracker Tests;
        auto res = TestNewtonRaphsonView30<SharkParamsNR7, referenceOperator>(
            Tests, 0, launchParams, static_cast<uint64_t>(internalTestLoopCount), useMT, numIters);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingleNRView32) {
        TestTracker Tests;
        auto res = TestNewtonRaphsonView32<SharkParamsNR9, referenceOperator>(
            Tests, 0, launchParams, static_cast<uint64_t>(internalTestLoopCount), useMT, numIters);
        if (!ContinueAfterFailure(res))
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSweep) {
        if (!RunPerfFullSweep<referenceOperator>(numIters, internalTestLoopCount, sweepLimbOptions))
            return 0;
    }

    return 1;
}

template <Operator op>
static int
RunPerfBasicOp(int testBase,
               BasicCorrectnessMode mode,
               int timeoutInSec,
               bool &interactiveMode,
               const CommandLineOptions &options)
{
    const int numIters =
        ResolveIntOption(options.m_NumIters, "NumIters? Default 5", 5, timeoutInSec, interactiveMode);
    const int internalTestLoopCount = ResolveIntOption(options.m_CudaIterations,
                                                       "CUDA iteration count? Default 1000",
                                                       1000,
                                                       timeoutInSec,
                                                       interactiveMode);
    const int numBlocks = ResolveIntOption(
        options.m_NumBlocks, "NumBlocks? Default 65, 0 for auto", 65, timeoutInSec, interactiveMode);
    const int numThreads = ResolveIntOption(
        options.m_NumThreads, "NumThreads? Default 256, 0 for auto", 256, timeoutInSec, interactiveMode);
    const HpShark::LaunchParams launchParams{numBlocks, numThreads};

    auto res = TestBinaryOperatorPerf<op>(launchParams, testBase, numIters, internalTestLoopCount, mode);
    if (!ContinueAfterFailure(res))
        return 0;

    return 1;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int
main(int argc, char **argv)
{
    const CommandLineParseResult commandLine = ParseCommandLine(argc, argv);
    if (commandLine.m_ShowHelp) {
        std::cout << CommandLineUsage();
        return 0;
    }
    if (!commandLine.m_Error.empty()) {
        std::cerr << commandLine.m_Error << "\n\n" << CommandLineUsage();
        return 1;
    }

    const CommandLineOptions &options = commandLine.m_Options;
    Environment::RegisterHeapCleanup();

    constexpr int kTimeoutInSec = 3;
    bool interactiveMode = false; // becomes true after any user input, making later prompts wait forever

    // Mode prompt: keep default consistent with the enum value
    const int defaultModeInt = static_cast<int>(BasicCorrectnessMode::PerfSingleView5);
    std::ostringstream modePrompt;

    modePrompt << "Mode? Default=" << defaultModeInt << " "
               << "1=Correctness(P1)" << std::endl
               << "2=Correctness NR" << std::endl
               << "3=PerfSweep" << std::endl
               << "--- Non-NR Perf Views ---" << std::endl
               << "4=PerfSingle View5" << std::endl
               << "5=PerfSingle View30" << std::endl
               << "6=PerfSingle View32" << std::endl
               << "--- NR Perf Views ---" << std::endl
               << "7=NR View5" << std::endl
               << "8=NR View30" << std::endl
               << "9=NR View32" << std::endl
               << "10=Correctness(P1..P5)" << std::endl
               << "11=PerfSingle Reference Orbit" << std::endl
               << "12=PerfSingle View (pick 1-34)" << std::endl
               << "anything else=Exit" << std::endl
               << "Enter choice:";

    const int rawMode = ResolveIntOption(
        options.m_Mode, modePrompt.str(), defaultModeInt, kTimeoutInSec, interactiveMode);

    BasicCorrectnessMode mode = BasicCorrectnessMode::PerfSingleView5;
    switch (rawMode) {
        case 1:
            mode = BasicCorrectnessMode::Correctness_P1;
            break;
        case 2:
            mode = BasicCorrectnessMode::Correctness_NR;
            break;
        case 3:
            mode = BasicCorrectnessMode::PerfSweep;
            break;
        case 4:
            mode = BasicCorrectnessMode::PerfSingleView5;
            break;
        case 5:
            mode = BasicCorrectnessMode::PerfSingleView30;
            break;
        case 6:
            mode = BasicCorrectnessMode::PerfSingleView32;
            break;
        case 7:
            mode = BasicCorrectnessMode::PerfSingleNRView5;
            break;
        case 8:
            mode = BasicCorrectnessMode::PerfSingleNRView30;
            break;
        case 9:
            mode = BasicCorrectnessMode::PerfSingleNRView32;
            break;
        case 10:
            mode = BasicCorrectnessMode::Correctness_P1_to_P5;
            break;
        case 11:
            mode = BasicCorrectnessMode::PerfSingleRef;
            break;
        case 12:
            mode = BasicCorrectnessMode::PerfSingleViewAny;
            break;
        default:
            std::cout << "Invalid mode " << rawMode << " (valid range: 1..12). "
                      << "Exiting.\n";
            mode = BasicCorrectnessMode::Error;
            break;
    }

    if (mode == BasicCorrectnessMode::Error && IsCommandLineSupplied(options.m_Mode)) {
        std::cerr << "Invalid command-line mode " << rawMode << " (valid range: 1..12).\n";
        return 1;
    }
    if (mode != BasicCorrectnessMode::Error && !ValidateCommandLineApplicability(options, mode)) {
        return 1;
    }

    std::cout << "Selected mode: " << static_cast<int>(mode) << " ("
              << BasicCorrectnessModeToString(mode) << ")\n";

    // Verbose
    if (mode != BasicCorrectnessMode::Error) {
        const int verbose = ResolveIntOption(
            options.m_Verbose, "Verbose? Default=0 (0=No, 1=Yes):", 0, kTimeoutInSec, interactiveMode);
        SetVerboseMode(verbose ? VerboseMode::Debug : VerboseMode::None);
    }

    // Explicit dispatch (don’t “call both and early-out”)
    switch (mode) {
        case BasicCorrectnessMode::Correctness_P1:
        case BasicCorrectnessMode::Correctness_P1_to_P5:
            RunCorrectnessTest(mode);
            break;

        case BasicCorrectnessMode::Correctness_NR: {
            do {
                if (!CorrectnessTests<SharkParamsNR1>()) {
                    if (!ContinueAfterFailure(false))
                        return 0;
                }
            } while (HpShark::TestInfiniteCorrectness);
            break;
        }

        case BasicCorrectnessMode::PerfSweep:
        case BasicCorrectnessMode::PerfSingleView30:
        case BasicCorrectnessMode::PerfSingleView32:
        case BasicCorrectnessMode::PerfSingleView5:
        case BasicCorrectnessMode::PerfSingleNRView5:
        case BasicCorrectnessMode::PerfSingleNRView30:
        case BasicCorrectnessMode::PerfSingleNRView32:
        case BasicCorrectnessMode::PerfSingleViewAny:
            RunPerfModes<Operator::ReferenceOrbit2>(mode, kTimeoutInSec, interactiveMode, options);
            break;
        case BasicCorrectnessMode::PerfSingleRef:
            RunPerfBasicOp<Operator::ReferenceOrbit2>(
                TestIds::kFullPerf, mode, kTimeoutInSec, interactiveMode, options);
            break;

        default:
            break;
    }

    return 0;
}
