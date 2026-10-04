#include "Crc64.h"
#include "Environment.h"
#include "RenderThreadPool.h"
#include "RenderToPng.h"
#include "TestFramework.h"

#include <filesystem>
#include <string_view>
#include <vector>

namespace {

class RenderDirectory {
    std::filesystem::path m_Previous = std::filesystem::current_path();
    std::filesystem::path m_Directory =
        std::filesystem::temp_directory_path() /
        ("fractalshark-cuda-render-" + std::to_string(Environment::CurrentProcessId()));

public:
    RenderDirectory()
    {
        ASSERT_TRUE(std::filesystem::create_directory(m_Directory));
        std::filesystem::current_path(m_Directory);
    }
    ~RenderDirectory()
    {
        std::error_code error;
        std::filesystem::current_path(m_Previous, error);
        std::filesystem::remove(m_Directory, error);
    }
};

void
CheckRender(std::string_view algorithmName, const char *broadViewCrc)
{
    // Fractal discovers native caches in the working directory during construction.
    RenderDirectory directory;
    const RenderAlgorithm *algorithm = nullptr;
    for (const auto &candidate : RenderAlgorithms) {
        if (algorithmName == candidate.AlgorithmStr) {
            algorithm = &candidate;
            break;
        }
    }
    ASSERT_TRUE(algorithm != nullptr);
    for (size_t view : {size_t{0}, size_t{5}}) {
        RenderRequest request;
        request.Width = 128;
        request.Height = 128;
        request.ViewSource = RenderRequest::ViewSourceKind::Builtin;
        request.BuiltinView = view;
        request.Iterations = 10000;
        request.Antialiasing = 1;
        request.Algorithm = *algorithm;
        Fractal fractal{128, 128, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
        fractal.GetRenderPool()->Drain();
        fractal.SetResultsAutosave(AddPointOptions::DontSave);
        std::string error;
        ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
        for (IterTypeEnum bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            fractal.GetRenderPool()->Drain();
            fractal.SetIterType(bits);
            fractal.ResetDimensions(128, 128, 1);
            fractal.SetNumIterations<uint64_t>(10000);
            fractal.CalcFractal(true);
            ASSERT_TRUE(fractal.GetRenderAlgorithm().Algorithm == algorithm->Algorithm);
            std::vector<uint64_t> iterations;
            iterations.reserve(128 * 128);
            for (size_t y = 0; y < 128; ++y) {
                for (size_t x = 0; x < 128; ++x) {
                    iterations.push_back(fractal.GetCurIters().GetItersArrayValSlow(x, y));
                }
            }
            // Raw iteration baselines captured before compacting the LA rows.
            const std::string expected = view == 0 ? broadViewCrc : "54719d198d67aced";
            ASSERT_EQ(
                Crc64::ToHex(Crc64::Compute(iterations.data(), iterations.size() * sizeof(uint64_t))),
                expected);
        }
    }
}

} // namespace

TEST(CudaLARender_Full) { CheckRender("GpuHDRx32PerturbedLAv2", "fce327cad6f3d50d"); }

TEST(CudaLARender_LAOnly) { CheckRender("GpuHDRx32PerturbedLAv2LAO", "0000000000000000"); }

TEST(CudaLARender_CompressedFull) { CheckRender("GpuHDRx32PerturbedRCLAv2", "fce327cad6f3d50d"); }

TEST(CudaLARender_CompressedLAOnly) { CheckRender("GpuHDRx32PerturbedRCLAv2LAO", "0000000000000000"); }
