#include "ConsoleLog.h"
#include "Environment.h"
#include "GpuTestRuntime.h"
#include "HighPrecision.h"
#include "MoviePlatform.h"
#include "TestFramework.h"
#include "heap_allocator/include/HeapCpp.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace TestFramework {
int
MovieTestEncoder(int argc, char **argv, const char *controlPath)
{
    const std::filesystem::path control(Environment::ToFsPath(Environment::Utf8ToWide(controlPath)));
    std::ifstream encoderFile(control / "ffmpeg.txt");
    std::string encoder;
    std::getline(encoderFile, encoder);
    if (encoder.empty()) {
        std::cerr << "Movie test encoder requires a real FFmpeg path\n";
        return 2;
    }
    bool pngSequence = false;
    std::string first;
    std::string joinChunk;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string_view(argv[i]) == "-i") {
            const std::string_view source(argv[i + 1]);
            pngSequence = source.ends_with(".png");
            if (source.ends_with(".mkv")) {
                const std::filesystem::path input(
                    Environment::ToFsPath(Environment::Utf8ToWide(source)));
                joinChunk = input.filename().string();
            }
        }
        if (std::string_view(argv[i]) == "-start_number") {
            first = argv[i + 1];
        }
    }
    if (!joinChunk.empty()) {
#ifdef _WIN32
        // The remuxer sends binary transport packets through stdout.
        if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
            return 2;
        }
#endif
        const auto hold = control / ("hold-join-" + joinChunk);
        if (std::filesystem::exists(hold) ||
            std::filesystem::exists(control / ("fail-join-" + joinChunk))) {
            std::cout << "unfinished transport bytes" << std::flush;
            std::ofstream(control / ("started-join-" + joinChunk)).put('\n');
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::filesystem::exists(hold) && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            FractalSharkLog::Write("Intentional stream-copy failure", __FILE__, __LINE__);
            return 42;
        }
    }
    if (pngSequence && std::filesystem::exists(control / "hold")) {
        // Per-range gates prove concurrent starts and out-of-order harvesting.
        // Preflight uses .tmp PNGs, so it bypasses all archive gates.
        std::ofstream(control / "started").put('\n');
        std::ofstream(control / ("started-" + first)).put('\n');
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (!std::filesystem::exists(control / "release") &&
               !std::filesystem::exists(control / ("release-" + first))) {
            if (std::chrono::steady_clock::now() >= deadline) {
                std::cerr << "Movie test archive gate timed out before rendering resumed\n";
                return 124;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (std::filesystem::exists(control / "fail") ||
            std::filesystem::exists(control / ("fail-" + first))) {
            std::cerr << "Intentional movie test encoder failure\n";
            return 42;
        }
    }
    // Reuse Environment's pipe, priority, and job/group ownership abstractions.
    // Preserve FFmpeg's stdout and stderr, including binary transport packets.
    const auto result = Environment::RunMovieProcess(
        Environment::ToFsPath(Environment::Utf8ToWide(encoder)),
        std::vector<std::string>(argv + 1, argv + argc),
        {},
        [](std::span<const unsigned char> bytes) {
            std::cout
                .write(reinterpret_cast<const char *>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()))
                .flush();
        },
        [](std::string_view text) { std::cerr << text << std::flush; },
        {});
    return result.ExitCode;
}

int
MovieProcessTestChild(int argc, char **argv)
{
    const std::string_view mode(argv[2]);
    if (mode == "priority") {
#ifdef _WIN32
        BOOL inJob = FALSE;
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        return GetPriorityClass(GetCurrentProcess()) == IDLE_PRIORITY_CLASS &&
                       IsProcessInJob(GetCurrentProcess(), nullptr, &inJob) && inJob &&
                       QueryInformationJobObject(nullptr,
                                                 JobObjectExtendedLimitInformation,
                                                 &limits,
                                                 sizeof(limits),
                                                 nullptr) &&
                       (limits.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
                   ? 0
                   : 1;
#else
        return getpriority(PRIO_PROCESS, 0) == 19 && getpgrp() == getpid() ? 0 : 1;
#endif
    }
    if (mode == "quiet") {
        std::cout << Environment::CurrentProcessId() << '\n' << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(30));
        return 0;
    }
    if (mode == "owner" && argc == 4) {
        const auto result = Environment::RunMovieProcess(
            Environment::ExecutablePath(),
            {"--movie-process-test-child", "quiet"},
            {},
            [&](std::span<const unsigned char> bytes) {
                // The outer test kills this owner while its encoder is alive.
                // Its private job handle must close during that hard kill.
                std::ofstream output(Environment::ToFsPath(Environment::Utf8ToWide(argv[3])),
                                     std::ios::binary);
                output.write(reinterpret_cast<const char *>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()));
            },
            {},
            {});
        return result.ExitCode;
    }
    return 2;
}
} // namespace TestFramework

int
main(int argc, char **argv)
{
    if (const auto control = std::getenv("FRACTALSHARK_TEST_MOVIE_PROXY");
        control && argc > 1 && std::string_view(argv[1]) == "-hide_banner") {
        return TestFramework::MovieTestEncoder(argc, argv, control);
    }
    // Native process lifecycle tests execute this same binary in a tiny child
    // mode, avoiding shell-specific helper scripts or a second test harness.
    if (argc >= 3 && std::string_view(argv[1]) == "--movie-process-test-child") {
        return TestFramework::MovieProcessTestChild(argc, argv);
    }
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
