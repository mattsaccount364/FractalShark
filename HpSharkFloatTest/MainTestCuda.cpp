#include "Conversion.h"
#include "DbgHeap.h"
#include "HpSharkFloat.h"
#include "ShowMostEfficientSizes.h"
#include "TestTracker.h"
#include "TestVerbose.h"
#include "Tests.h"

#include <cuda_runtime.h>

#include "MainTestCuda.h"
#include <conio.h>
#include <iostream>
#include <mpir.h>
#include <stdarg.h>

#include <chrono>  // steady_clock
#include <conio.h> // _kbhit()
#include <iostream>
#include <string>

#include "Callstacks.h"

#define NOMINMAX
#include <windows.h> // Sleep()

#include "HDRFloat.h"

#include "heap_allocator\include\HeapCpp.h"

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
char
PressKey()
{
    std::cout << "Press any key to continue...";
    return (char)_getch();
}

int
PromptIntWithTimeout(const std::string &promptText,
                     int defaultValue,
                     int timeoutSec,
                     int sleepIntervalMs = 50)
{
    std::cout << promptText << " " << std::flush;

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);

    std::string line;

    while (std::chrono::steady_clock::now() < deadline) {
        if (_kbhit()) {
            std::getline(std::cin, line);
            break;
        }
        Sleep(sleepIntervalMs);
    }

    if (line.empty()) {
        std::cout << "\n(no input in " << timeoutSec << "s, defaulting to " << defaultValue << ")\n";
        return defaultValue;
    }

    try {
        return std::stoi(line);
    } catch (...) {
        std::cout << "(could not parse input, defaulting to " << defaultValue << ")\n";
        return defaultValue;
    }
}

// -----------------------------------------------------------------------------
// Correctness tests
// -----------------------------------------------------------------------------
template <typename TestSharkParams>
bool
CorrectnessTests()
{
    int testBase = 0;
    bool res;

    res = TestConversion<TestSharkParams>(0);
    if (!res && PressKey() == 'q')
        return false;

    testBase = 4000;
    res = TestAllBinaryOp<TestSharkParams, Operator::Add>(testBase);
    if (!res && PressKey() == 'q')
        return false;

    testBase = 6000;
    res = TestAllBinaryOp<TestSharkParams, Operator::MultiplyNTT>(testBase);
    if (!res && PressKey() == 'q')
        return false;

    testBase = 2000;
    res = TestAllBinaryOp<TestSharkParams, Operator::ReferenceOrbit>(testBase);
    if (!res && PressKey() == 'q')
        return false;

    return true;
}

int
RunCorrectnessTest(BasicCorrectnessMode mode)
{
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
// Performance modes
// -----------------------------------------------------------------------------
int
RunPerfModes(BasicCorrectnessMode mode, int timeoutInSec)
{
    if (mode != BasicCorrectnessMode::PerfSub &&
        mode != BasicCorrectnessMode::PerfSweep &&
        mode != BasicCorrectnessMode::PerfSingle) {
        return 1;
    }

    bool res;

    int numIters = PromptIntWithTimeout("NumIters? Default 5", 5, timeoutInSec);
    int internalTestLoopCount =
        PromptIntWithTimeout("CUDA iteration count? Default 1000", 1000, timeoutInSec);

    int testBase = 0;

    if (mode == BasicCorrectnessMode::PerfSub) {
        testBase = 10000;
        res = TestBinaryOperatorPerf<Operator::Add>(testBase, numIters, internalTestLoopCount, mode);
        if (!res && PressKey() == 'q')
            return 0;

        testBase = 13000;
        res = TestBinaryOperatorPerf<Operator::MultiplyNTT>(
            testBase, numIters, internalTestLoopCount, mode);
        if (!res && PressKey() == 'q')
            return 0;

        testBase = 14000;
        res = TestBinaryOperatorPerf<Operator::ReferenceOrbit>(
            testBase, numIters, internalTestLoopCount, mode);
        if (!res && PressKey() == 'q')
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSingle) {
        TestTracker Tests;

        int numBlocks = PromptIntWithTimeout("NumBlocks? Default 65", 65, timeoutInSec);
        int numThreads = PromptIntWithTimeout("NumThreads? Default 256", 256, timeoutInSec);

        testBase = 16020;
        res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
            Tests, numBlocks, numThreads, testBase, numIters, internalTestLoopCount);
        if (!res && PressKey() == 'q')
            return 0;

        PressKey();

        testBase = 16010;
        res = TestFullReferencePerfView5<Operator::ReferenceOrbit>(
            Tests, numBlocks, numThreads, testBase, numIters, internalTestLoopCount);
        if (!res && PressKey() == 'q')
            return 0;
    }

    if (mode == BasicCorrectnessMode::PerfSweep) {
        TestTracker Tests;

        int testBaseLocal = 1000;

        for (int numBlocks = 16; numBlocks <= 256; numBlocks *= 2) {
            for (int numThreads = 64; numThreads <= 256; numThreads *= 2) {
                res = TestFullReferencePerfView30<Operator::ReferenceOrbit>(
                    Tests, numBlocks, numThreads, testBaseLocal, numIters, internalTestLoopCount);
                if (!res && PressKey() == 'q')
                    return 0;
                testBaseLocal += 100;

                res = TestFullReferencePerfView5<Operator::ReferenceOrbit>(
                    Tests, numBlocks, numThreads, testBaseLocal, numIters, internalTestLoopCount);
                if (!res && PressKey() == 'q')
                    return 0;
                testBaseLocal += 100;
            }
        }

        Tests.CheckAllTestsPassed();
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

    constexpr int timeoutInSec = 3;

    int verboseInput = PromptIntWithTimeout("Verbose? Default=0 (0=No, 1=Yes):", 0, timeoutInSec);
    SetVerboseMode(verboseInput ? VerboseMode::Debug : VerboseMode::None);

    int rawMode =
        PromptIntWithTimeout("Mode? Default=2 "
                             "(0=Correctness(P1), 1=PerfSub, 2=PerfSweep, 3=PerfSingle, 4=Correctness(P1..P5)):",
                             static_cast<int>(BasicCorrectnessMode::PerfSingle),
                             timeoutInSec);

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
            std::cout << "Invalid mode, defaulting to PerfSingle\n";
            break;
    }

    std::cout << "Selected mode: " << BasicCorrectnessModeToString(mode) << "\n";

    if (!RunCorrectnessTest(mode)) {
        return 0;
    }

    if (!RunPerfModes(mode, timeoutInSec)) {
        return 0;
    }

    GlobalCallstacks->FreeCallstacks();
    return 0;
}
