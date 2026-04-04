#include "HighPrecision.h"
#include "TestFramework.h"

void RegisterHeapCleanup();

int
main()
{
    RegisterHeapCleanup();

    // Set a reasonable default precision for MPIR operations used by tests.
    HighPrecision::defaultPrecisionInBits(256);

    return TestFramework::RunAllTests();
}
