#include "Environment.h"
#include "HighPrecision.h"
#include "TestFramework.h"

int
main()
{
    Environment::RegisterHeapCleanup();

    // Set a reasonable default precision for MPIR operations used by tests.
    HighPrecision::defaultPrecisionInBits(256);

    return TestFramework::RunAllTests();
}
