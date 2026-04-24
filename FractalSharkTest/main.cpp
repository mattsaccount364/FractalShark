#include "Callstacks.h"
#include "Environment.h"
#include "HighPrecision.h"
#include "TestFramework.h"

int
main()
{
    Environment::RegisterHeapCleanup();

    // GrowableVector and other HpSharkFloatLib allocation paths assume
    // GlobalCallstacks has been initialized by the application.
    CallStacks::InitCallstacks();

    // Set a reasonable default precision for MPIR operations used by tests.
    HighPrecision::defaultPrecisionInBits(256);

    return TestFramework::RunAllTests();
}
