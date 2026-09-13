#pragma once

#include "HpSharkFloat.h"
#include "PerfTimingResult.h"

class TestTracker;
enum class Operator;

template <class SharkFloatParams, Operator referenceOperator>
bool TestNewtonRaphsonView5(TestTracker &Tests,
                            int testBase,
                            const HpShark::LaunchParams &launchParams,
                            uint64_t iterCountOverride,
                            int numRepeats);

template <class SharkFloatParams, Operator referenceOperator>
bool TestNewtonRaphsonView30(TestTracker &Tests,
                             int testBase,
                             const HpShark::LaunchParams &launchParams,
                             uint64_t iterCountOverride,
                             int numRepeats);

template <class SharkFloatParams, Operator referenceOperator>
bool TestNewtonRaphsonView32(TestTracker &Tests,
                             int testBase,
                             const HpShark::LaunchParams &launchParams,
                             uint64_t iterCountOverride,
                             int numRepeats);
