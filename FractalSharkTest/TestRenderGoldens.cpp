// Golden-CRC regression tests for the headless render path.
//
// Each case uses RenderToPng (FractalSharkLib) to render a fixed view + CPU
// algorithm to a PNG, then CRC-64s decoded RGBA16 pixels and compares against a
// hardcoded value. The CRC only catches drift; baseline correctness must be
// confirmed by visually inspecting the PNG when seeding a new case (run with
// FRACTALSHARK_UPDATE_GOLDENS=1, look at the kept PNG, then bake the printed
// CRC into the table).

#include "Crc64.h"
#include "RenderAlgorithm.h"
#include "RenderToPng.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const RenderAlgorithm *
LookupAlgorithm(const char *name)
{
    for (const auto &alg : RenderAlgorithms) {
        if (alg.AlgorithmStr && std::strcmp(alg.AlgorithmStr, name) == 0) {
            return &alg;
        }
    }
    return nullptr;
}

bool
IsUpdateMode()
{
    const char *env = std::getenv("FRACTALSHARK_UPDATE_GOLDENS");
    return env && env[0] && std::strcmp(env, "0") != 0;
}

std::filesystem::path
GoldensOutDir()
{
    auto dir = std::filesystem::temp_directory_path() / "fractalshark-goldens";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

std::vector<uint8_t>
ReadFileBytes(const std::filesystem::path &p)
{
    std::ifstream in(p, std::ios::binary);
    std::vector<uint8_t> buf;
    if (!in)
        return buf;
    in.seekg(0, std::ios::end);
    auto sz = in.tellg();
    in.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(sz));
    if (sz > 0) {
        in.read(reinterpret_cast<char *>(buf.data()), sz);
    }
    return buf;
}

// Hardcoded golden CRCs. The CPU render paths covered here are expected to
// produce identical decoded RGBA16 pixels across supported platforms. Channel
// values use PNG's big-endian byte order, independently of the host byte order.
// "PENDING" =
// unbaked; first run will print actual + pass.
struct GoldenCase {
    const char *Name;
    size_t BuiltinView;
    const char *AlgorithmName;
    uint32_t Antialiasing; // 0 = use view default
    const char *ExpectedCrc;
};

constexpr int kGoldenWidth = 256;
constexpr int kGoldenHeight = 256;

const GoldenCase kCases[] = {
    {"view0-cpu64", 0, "Cpu64", 1, "8eee9f298276208d"},
    {"view0-cpu64-aa4", 0, "Cpu64", 4, "ccb0a5811b8c13b8"},
    {"view1-cpu-bla", 1, "Cpu64PerturbedBLAHDR", 1, "408ab08e4feb17f1"},
    {"view0-cpuhdr", 0, "CpuHDR32", 1, "accf6b90d46ad55e"},
    {"view5-cpu-bla-v2", 5, "Cpu32PerturbedBLAV2HDR", 1, "00579b31ca3aa41e"},
    {"view0-cpuhdr64", 0, "CpuHDR64", 1, "8eee9f298276208d"},
    {"view5-cpu-perturbed-bla", 5, "Cpu64PerturbedBLA", 1, "b7221b9cad319c8f"},
    {"view5-cpu32-bla-hdr", 5, "Cpu32PerturbedBLAHDR", 1, "8e2e1d1730c2e8f4"},
    {"view5-cpu64-bla-hdr", 5, "Cpu64PerturbedBLAHDR", 1, "f64dad356c7780ba"},
    {"view5-cpu64-bla-v2", 5, "Cpu64PerturbedBLAV2HDR", 1, "88184a0787aa341c"},
    {"view5-cpu32-rc-bla-v2", 5, "Cpu32PerturbedRCBLAV2HDR", 1, "3a8313703a7428b6"},
    {"view5-cpu64-rc-bla-v2", 5, "Cpu64PerturbedRCBLAV2HDR", 1, "2258571d542f1530"},
};

void
RunGoldenCase(const GoldenCase &c)
{
    const RenderAlgorithm *alg = LookupAlgorithm(c.AlgorithmName);
    if (!alg) {
        std::ostringstream oss;
        oss << "unknown algorithm: " << c.AlgorithmName;
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }

    auto outDir = GoldensOutDir();
    auto basename = outDir / c.Name;
    auto pngPath = basename;
    pngPath += ".png";

    // Always start clean — the saver refuses to overwrite.
    std::error_code ec;
    std::filesystem::remove(pngPath, ec);

    RenderRequest req;
    req.Width = kGoldenWidth;
    req.Height = kGoldenHeight;
    req.ViewSource = RenderRequest::ViewSourceKind::Builtin;
    req.BuiltinView = c.BuiltinView;
    req.Algorithm = *alg;
    req.Antialiasing = c.Antialiasing;
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
        oss << "RenderToPng failed (rc=" << rc << "): " << err;
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }

    auto bytes = ReadFileBytes(pngPath);
    if (bytes.empty()) {
        std::ostringstream oss;
        oss << "rendered PNG missing or empty: " << pngPath.string();
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }

    std::vector<unsigned char> pixels;
    unsigned width = 0;
    unsigned height = 0;
    const unsigned decodeError = lodepng::decode(pixels, width, height, bytes, LCT_RGBA, 16);
    if (decodeError != 0) {
        std::ostringstream message;
        message << "could not decode golden PNG: " << lodepng_error_text(decodeError) << " (png kept at "
                << pngPath.string() << ")";
        TestFramework::Fail(__FILE__, __LINE__, message.str());
    }
    ASSERT_EQ(width, static_cast<unsigned>(kGoldenWidth));
    ASSERT_EQ(height, static_cast<unsigned>(kGoldenHeight));
    ASSERT_EQ(pixels.size(), static_cast<size_t>(width) * height * 8);

    auto actualCrc = Crc64::ToHex(Crc64::Compute(pixels.data(), pixels.size()));
    const char *expected = c.ExpectedCrc;

    bool updateMode = IsUpdateMode();
    bool pending = std::strcmp(expected, "PENDING") == 0;

    if (updateMode || pending) {
        std::cout << "  GOLDEN " << c.Name << " decoded RGBA16 CRC(base16) " << actualCrc
                  << "    (png: " << pngPath.string() << ")\n";
        if (pending && !updateMode) {
            std::cout << "         (PENDING placeholder — passing; bake CRC after visual check)\n";
        }
        return;
    }

    // Keep the PNG around on mismatch for inspection; otherwise clean up.
    if (actualCrc != expected) {
        std::ostringstream oss;
        oss << "decoded RGBA16 CRC(base16) mismatch for " << c.Name << ": expected " << expected
            << ", got " << actualCrc << " (png kept at " << pngPath.string() << ")";
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }
    std::cout << "  GOLDEN " << c.Name << " decoded RGBA16 CRC(base16) " << actualCrc << " OK"
              << "    (png: " << pngPath.string() << ")\n";
}

void
RenderSmallOutputPathCase(const std::filesystem::path &basename,
                          const std::filesystem::path &expectedPngPath)
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
}

void
RunOutputPathHandlingCases()
{
    auto outDir = GoldensOutDir() / "path.cases";
    std::error_code ec;
    std::filesystem::create_directories(outDir, ec);

    auto dottedDir = outDir / "dotted.dir";
    std::filesystem::create_directories(dottedDir, ec);

    auto dottedBasename = dottedDir / "no-extension";
    auto dottedExpected = dottedBasename;
    dottedExpected += ".png";
    RenderSmallOutputPathCase(dottedBasename, dottedExpected);

    auto existingExtension = outDir / "already.png";
    RenderSmallOutputPathCase(existingExtension, existingExtension);

#ifdef _WIN32
    auto unicodeBasename = outDir / std::filesystem::path(L"unicode-\u00e9-\u6d4b");
    auto unicodeExpected = unicodeBasename;
    unicodeExpected += ".png";
    RenderSmallOutputPathCase(unicodeBasename, unicodeExpected);
#endif
}

} // namespace

TEST(RenderGolden_view0_cpu64) { RunGoldenCase(kCases[0]); }

TEST(RenderGolden_view0_cpu64_aa4) { RunGoldenCase(kCases[1]); }

TEST(RenderGolden_view1_cpu_bla) { RunGoldenCase(kCases[2]); }

TEST(RenderGolden_view0_cpuhdr) { RunGoldenCase(kCases[3]); }

TEST(RenderGolden_view5_cpu_bla_v2) { RunGoldenCase(kCases[4]); }

TEST(RenderGolden_view0_cpuhdr64) { RunGoldenCase(kCases[5]); }

TEST(RenderGolden_view5_cpu_perturbed_bla) { RunGoldenCase(kCases[6]); }

TEST(RenderGolden_view5_cpu32_bla_hdr) { RunGoldenCase(kCases[7]); }

TEST(RenderGolden_view5_cpu64_bla_hdr) { RunGoldenCase(kCases[8]); }

TEST(RenderGolden_view5_cpu64_bla_v2) { RunGoldenCase(kCases[9]); }

TEST(RenderGolden_view5_cpu32_rc_bla_v2) { RunGoldenCase(kCases[10]); }

TEST(RenderGolden_view5_cpu64_rc_bla_v2) { RunGoldenCase(kCases[11]); }

TEST(RenderOutputPathHandling) { RunOutputPathHandlingCases(); }
