#include "TestFramework.h"
#include "HighPrecision.h"

int
main()
{
    // Set a reasonable default precision for MPIR operations used by tests.
    HighPrecision::defaultPrecisionInBits(256);

    return TestFramework::RunAllTests();
}
