#pragma once

enum class VerboseMode { None = 0, Debug = 1 };

extern VerboseMode SharkVerbose;

void SetVerboseMode(VerboseMode mode);
