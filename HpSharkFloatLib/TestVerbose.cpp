#include "TestVerbose.h"

VerboseMode SharkVerbose{ VerboseMode::None };

void SetVerboseMode(VerboseMode mode) {
    SharkVerbose = mode;
}
