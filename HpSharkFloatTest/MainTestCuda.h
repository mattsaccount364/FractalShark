#pragma once

#include "TestParams.h"

enum class BasicCorrectnessMode : int;

bool RunCorrectnessTest(BasicCorrectnessMode mode, const HpShark::TestParams &testParams);
