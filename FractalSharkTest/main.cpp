#include "Environment.h"
#include "GpuTestRuntime.h"
#include "HighPrecision.h"
#include "TestFramework.h"
#include "heap_allocator/include/HeapCpp.h"

int
main(int argc, char **argv)
{
    TestFramework::RunOptions options;
    std::string error;
    if (!TestFramework::ParseArguments(argc, argv, options, error)) {
        std::cerr << "error: " << error << "\n";
        return 2;
    }
    if (options.Help) {
        TestFramework::PrintHelp(std::cout);
        return 0;
    }
    if (options.ListTests) {
        return TestFramework::RunTests(TestFramework::Registry(), options, std::cout, std::cerr);
    }

    if (options.UseGpu) {
        std::string description;
        if (!TestFramework::CheckGpuRuntime(description, error)) {
            std::cerr << "error: --use-gpu requires working CUDA hardware: " << error << '\n';
            return 1;
        }
        std::cout << "CUDA: " << description << '\n';
    }

    Environment::RegisterHeapCleanup();

    // Set a reasonable default precision for MPIR operations used by tests.
    HighPrecision::defaultPrecisionInBits(256);

    return TestFramework::RunTests(TestFramework::Registry(), options, std::cout, std::cerr);
}
