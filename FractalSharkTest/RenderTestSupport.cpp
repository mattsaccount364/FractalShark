#include "RenderTestSupport.h"

#include "Crc64.h"
#include "Environment.h"
#include "Fractal.h"
#include "GoldenChecksums.h"
#include "GpuTestRuntime.h"
#include "RenderThreadPool.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {
std::string
PathUtf8(const std::filesystem::path &path)
{
    const auto encoded = path.u8string();
    return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}
} // namespace

std::string
RenderTests::Profile()
{
#ifdef _WIN32
    const std::string platform = "windows";
#else
    const std::string platform = "linux";
#endif
#if defined(_DEBUG) || (!defined(_WIN32) && !defined(NDEBUG))
    return platform + "-Debug";
#else
    return platform + "-Release";
#endif
}

const std::filesystem::path &
RenderTests::OutputDirectory()
{
    static const auto directory = [] {
        const auto &options = TestFramework::CurrentOptions();
        const auto root = options.OutputDirectory.empty()
                              ? std::filesystem::current_path() / "validation-render-goldens"
                              : std::filesystem::absolute(options.OutputDirectory);
        const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
        const auto result =
            root / Profile() /
            (std::to_string(stamp) + "-" + std::to_string(Environment::CurrentProcessId()));
        std::filesystem::create_directories(result);
        std::ofstream metadata(result / "provenance.txt");
        metadata << "profile=" << Profile() << "\nbuilt=" << __DATE__ << ' ' << __TIME__
                 << "\ngeneration=" << options.GenerateGoldens << "\nmatrix-pixels=" << MatrixDimension()
                 << "\ngolden-source=windows-Release\n";
        if (options.UseGpu) {
            std::string description, error;
            ASSERT_TRUE(TestFramework::CheckGpuRuntime(description, error));
            metadata << "gpu=" << description << '\n';
        }
        ASSERT_TRUE(metadata.good());
        std::cout << "Render artifacts: " << PathUtf8(result) << std::endl;
        return result;
    }();
    return directory;
}

uint32_t
RenderTests::MatrixDimension()
{
    return 256;
}

RenderTests::ScopedDirectory::ScopedDirectory(std::string_view caseId)
    : m_Previous{std::filesystem::current_path()}
{
    const auto suffix = Crc64::ToHex(Crc64::Compute(caseId.data(), caseId.size())).substr(0, 8);
    const auto directory = OutputDirectory() / (std::string{caseId.substr(0, 60)} + "_" + suffix);
    std::filesystem::create_directories(directory);
    std::ofstream identity(directory / "case-id.txt");
    identity << caseId << '\n';
    ASSERT_TRUE(identity.good());
    std::filesystem::current_path(directory);
}

RenderTests::ScopedDirectory::~ScopedDirectory()
{
    std::error_code error;
    std::filesystem::current_path(m_Previous, error);
}

void
RenderTests::CheckChecksum(std::string_view caseId, std::span<const uint8_t> bytes)
{
    const auto actual = Crc64::ToHex(Crc64::Compute(bytes.data(), bytes.size()));
    const auto profile = Profile();
    std::ofstream results(OutputDirectory() / "checksums.tsv", std::ios::app);
    results << profile << '\t' << caseId << '\t' << actual << '\n';
    ASSERT_TRUE(results.good());
    std::cout << "  CRC " << caseId << ' ' << actual << std::endl;
    if (TestFramework::CurrentOptions().GenerateGoldens) {
        return;
    }
    for (const auto &golden : GoldenChecksums) {
        if (golden.CaseId == caseId) {
            if (golden.Crc != actual) {
                TestFramework::Fail(__FILE__,
                                    __LINE__,
                                    "CRC mismatch for " + std::string{caseId} + ": expected " +
                                        std::string{golden.Crc} + ", got " + actual +
                                        "; artifacts: " + PathUtf8(OutputDirectory()));
            }
            return;
        }
    }
    TestFramework::Fail(__FILE__, __LINE__, "missing golden for " + profile + "/" + std::string{caseId});
}

bool
RenderTests::CheckPng(std::string_view caseId,
                      const std::filesystem::path &path,
                      unsigned expectedWidth,
                      unsigned expectedHeight)
{
    std::ofstream manifest(OutputDirectory() / "artifacts.tsv", std::ios::app);
    manifest << caseId << '\t' << PathUtf8(std::filesystem::absolute(path)) << '\n';
    ASSERT_TRUE(manifest.good());
    std::ifstream input(path, std::ios::binary);
    ASSERT_TRUE(input.good());
    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>{input},
                                           std::istreambuf_iterator<char>{}};
    std::vector<unsigned char> pixels;
    unsigned width = 0, height = 0;
    const auto error = lodepng::decode(pixels, width, height, bytes, LCT_RGBA, 16);
    if (error != 0) {
        TestFramework::Fail(
            __FILE__,
            __LINE__,
            "PNG decode failed: " + std::string{lodepng_error_text(error)} + " at " + PathUtf8(path));
    }
    ASSERT_EQ(width, expectedWidth);
    ASSERT_EQ(height, expectedHeight);
    ASSERT_EQ(pixels.size(), static_cast<size_t>(width) * height * 8);
    CheckChecksum(caseId, pixels);
    ASSERT_TRUE(pixels.size() >= 8);
    return !std::equal(pixels.begin() + 8, pixels.end(), pixels.begin());
}

bool
RenderTests::SaveAndCheck(Fractal &fractal, std::string_view caseId)
{
    fractal.GetRenderPool()->Drain();
    const auto width = static_cast<unsigned>(fractal.GetRenderWidth());
    const auto height = static_cast<unsigned>(fractal.GetRenderHeight());
    std::ifstream identity("case-id.txt");
    std::string parentCaseId;
    ASSERT_TRUE(static_cast<bool>(std::getline(identity, parentCaseId)));
    const auto suffix = caseId.starts_with(parentCaseId + "_")
                            ? std::string{caseId.substr(parentCaseId.size() + 1)}
                            : "render";
    const auto basename = std::filesystem::current_path() /
                          (suffix + "-" + Crc64::ToHex(Crc64::Compute(caseId.data(), caseId.size())));
    ASSERT_EQ(fractal.SaveCurrentFractal(basename.wstring(), false), 0);
    fractal.CleanupThreads(true);
    auto path = basename;
    path += ".png";
    return CheckPng(caseId, path, width, height);
}
