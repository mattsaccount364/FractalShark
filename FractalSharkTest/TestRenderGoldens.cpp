#include "RenderTestSupport.h"
#include "RenderToPng.h"
#include "TestFramework.h"

#include <cstring>
#include <fstream>
#include <iterator>

namespace {
const RenderAlgorithm *
LookupAlgorithm(const char *name)
{
    for (const auto &algorithm : RenderAlgorithms) {
        if (algorithm.AlgorithmStr && std::strcmp(algorithm.AlgorithmStr, name) == 0) {
            return &algorithm;
        }
    }
    return nullptr;
}

std::vector<uint8_t>
ReadFileBytes(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

struct GoldenCase {
    const char *Name;
    size_t BuiltinView;
    const char *AlgorithmName;
    uint32_t Antialiasing;
};
const GoldenCase kCases[] = {
    {"view0-cpu64", 0, "Cpu64", 1},
    {"view0-cpu64-aa4", 0, "Cpu64", 4},
    {"view1-cpu-bla", 1, "Cpu64PerturbedBLAHDR", 1},
    {"view0-cpuhdr", 0, "CpuHDR32", 1},
    {"view5-cpu-bla-v2", 5, "Cpu32PerturbedBLAV2HDR", 1},
    {"view0-cpuhdr64", 0, "CpuHDR64", 1},
    {"view5-cpu-perturbed-bla", 5, "Cpu64PerturbedBLA", 1},
    {"view5-cpu32-bla-hdr", 5, "Cpu32PerturbedBLAHDR", 1},
    {"view5-cpu64-bla-hdr", 5, "Cpu64PerturbedBLAHDR", 1},
    {"view5-cpu64-bla-v2", 5, "Cpu64PerturbedBLAV2HDR", 1},
    {"view5-cpu32-rc-bla-v2", 5, "Cpu32PerturbedRCBLAV2HDR", 1},
    {"view5-cpu64-rc-bla-v2", 5, "Cpu64PerturbedRCBLAV2HDR", 1},
};

void
RunGoldenCase(const GoldenCase &test)
{
    const std::string caseId = "RenderGolden_Legacy_" + std::string{test.Name};
    RenderTests::ScopedDirectory directory{caseId};
    const auto *algorithm = LookupAlgorithm(test.AlgorithmName);
    ASSERT_TRUE(algorithm != nullptr);
    RenderRequest request;
    request.Width = 256;
    request.Height = 256;
    request.ViewSource = RenderRequest::ViewSourceKind::Builtin;
    request.BuiltinView = test.BuiltinView;
    request.Algorithm = *algorithm;
    request.Antialiasing = test.Antialiasing;
    request.OutPngBasename = L"legacy";
    request.Quiet = true;
    Fractal fractal{256, 256, nullptr, false, request.CommitCapBytes, true, GpuMode::Disabled};
    std::string error;
    ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
    RenderTests::CheckPng(caseId, "legacy.png", 256, 256);
}

const bool registered = [] {
    for (const auto &test : kCases) {
        TestFramework::RegisterCase(
            "RenderGolden_Legacy_" + std::string{test.Name},
            [test] { RunGoldenCase(test); },
            false,
            "",
            true);
    }
    return true;
}();

void
RenderSmallOutputPathCase(const std::filesystem::path &basename,
                          const std::filesystem::path &expectedPngPath,
                          std::string_view caseId)
{
    const RenderAlgorithm *alg = LookupAlgorithm("Cpu64");
    if (!alg) {
        TestFramework::Fail(__FILE__, __LINE__, "unknown algorithm: Cpu64");
    }

    std::error_code ec;
    std::filesystem::remove(expectedPngPath, ec);

    RenderRequest req;
    req.Width = 32;
    req.Height = 32;
    req.ViewSource = RenderRequest::ViewSourceKind::Builtin;
    req.BuiltinView = 0;
    req.Algorithm = *alg;
    req.Antialiasing = 1;
    req.OutPngBasename = basename.wstring();
    req.Quiet = true;

    std::string err;
    Fractal fractal(req.Width,
                    req.Height,
                    /*nativeWindow=*/nullptr,
                    /*UseSensoCursor=*/false,
                    req.CommitCapBytes,
                    /*hostOwnedGlPresentation=*/true,
                    GpuMode::Disabled);
    int rc = RenderToPng(req, fractal, &err);
    if (rc != 0) {
        std::ostringstream oss;
        oss << "RenderToPng failed for output path case (rc=" << rc << "): " << err;
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }

    auto bytes = ReadFileBytes(expectedPngPath);
    if (bytes.empty()) {
        std::ostringstream oss;
        oss << "rendered PNG missing or empty for output path case: " << expectedPngPath.string();
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }
    RenderTests::CheckPng(caseId, expectedPngPath, req.Width, req.Height);
}

void
RunOutputPathHandlingCases()
{
    auto outDir = RenderTests::OutputDirectory() / "path.cases";
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);

    auto dottedDir = outDir / "dotted.dir";
    std::filesystem::create_directories(dottedDir, ec);

    auto dottedBasename = dottedDir / "no-extension";
    auto dottedExpected = dottedBasename;
    dottedExpected += ".png";
    RenderSmallOutputPathCase(dottedBasename, dottedExpected, "RenderGolden_OutputPaths_Dotted");

    auto existingExtension = outDir / "already.png";
    RenderSmallOutputPathCase(
        existingExtension, existingExtension, "RenderGolden_OutputPaths_Extension");

#ifdef _WIN32
    auto unicodeBasename = outDir / std::filesystem::path(L"unicode-\u00e9-\u6d4b");
    auto unicodeExpected = unicodeBasename;
    unicodeExpected += ".png";
    RenderSmallOutputPathCase(unicodeBasename, unicodeExpected, "RenderGolden_OutputPaths_Unicode");
#endif
}

} // namespace

namespace {
const bool pathCasesRegistered =
    TestFramework::RegisterCase("RenderGolden_OutputPaths", RunOutputPathHandlingCases, false, "", true);
}
