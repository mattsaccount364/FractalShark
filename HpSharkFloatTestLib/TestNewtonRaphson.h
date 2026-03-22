#pragma once

#include "HpSharkFloat.h"
#include "PerfTimingResult.h"

class TestTracker;

template <class SharkFloatParams>
bool TestNewtonRaphsonView5(TestTracker &Tests, int testBase,
                            const HpShark::LaunchParams &launchParams = {2, 32},
                            uint64_t iterCountOverride = 0,
                            bool useMT = true,
                            int numRepeats = 1);

template <class SharkFloatParams>
bool TestNewtonRaphsonView30(TestTracker &Tests, int testBase,
                             const HpShark::LaunchParams &launchParams = {2, 32},
                             uint64_t iterCountOverride = 0,
                             bool useMT = true,
                             int numRepeats = 1);

template <class SharkFloatParams>
bool TestNewtonRaphsonView32(TestTracker &Tests, int testBase,
                             const HpShark::LaunchParams &launchParams = {2, 32},
                             uint64_t iterCountOverride = 0,
                             bool useMT = true,
                             int numRepeats = 1);

template <class SharkFloatParams>
bool TestSingleNRMultiply(TestTracker &Tests, int testBase);

template <class SharkFloatParams>
bool TestSingleNRAdd(TestTracker &Tests, int testBase);
