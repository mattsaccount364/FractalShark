#include "Environment.h"
#include "PngParallelSave.h"
#include "RenderThreadPool.h"
#include "RenderToPng.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

class PngSaveDirectory {
public:
    PngSaveDirectory()
        : m_Path(std::filesystem::temp_directory_path() /
                 ("fractalshark-png-save-" + std::to_string(Environment::CurrentProcessId())))
    {
        std::filesystem::create_directory(m_Path);
    }
    ~PngSaveDirectory()
    {
        std::error_code error;
        for (const auto &file : m_Files) {
            std::filesystem::remove(file, error);
        }
        std::filesystem::remove(m_Path, error);
    }
    std::filesystem::path
    File(const std::wstring &name)
    {
        const auto path = m_Path / name;
        std::error_code error;
        std::filesystem::remove(path, error);
        m_Files.push_back(path);
        return path;
    }

private:
    std::filesystem::path m_Path;
    std::vector<std::filesystem::path> m_Files;
};

std::vector<unsigned char>
DecodeSavedPng(const std::filesystem::path &path, unsigned expectedWidth, unsigned expectedHeight)
{
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    std::ifstream input(path, std::ios::binary);
    const std::vector<unsigned char> encoded{std::istreambuf_iterator<char>(input),
                                             std::istreambuf_iterator<char>()};
    std::vector<unsigned char> decoded;
    unsigned width = 0;
    unsigned height = 0;
    ASSERT_EQ(lodepng::decode(decoded, width, height, encoded, LCT_RGBA, 16), 0u);
    ASSERT_EQ(width, expectedWidth);
    ASSERT_EQ(height, expectedHeight);
    return decoded;
}

std::vector<unsigned char>
SaveCpuOracle(Fractal &fractal, const std::filesystem::path &path)
{
    PngParallelSave save(PngParallelSave::Type::PngImg,
                         PngParallelSave::EncoderBackend::Cpu,
                         path.wstring(),
                         true,
                         fractal);
    ASSERT_EQ(save.Run(), 0);
    return DecodeSavedPng(path,
                          static_cast<unsigned>(fractal.GetRenderWidth()),
                          static_cast<unsigned>(fractal.GetRenderHeight()));
}

void
CheckSnapshotsAndResize()
{
    PngSaveDirectory directory;
    Fractal fractal{17, 11, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
    fractal.GetRenderPool()->Drain();
    fractal.View(0, false);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32)));
    ASSERT_FALSE(fractal.RequiresUseLocalColor());
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (uint32_t aa = 1; aa <= 4; ++aa) {
            fractal.GetRenderPool()->Drain();
            fractal.SetIterType(bits);
            const unsigned width = 17 + 2 * aa;
            const unsigned height = 11 + 2 * aa;
            fractal.ResetDimensions(width, height, aa);
            fractal.SetNumIterations<uint64_t>(64);
            fractal.CalcFractal(true);
            const auto expected = SaveCpuOracle(fractal, directory.File(L"oracle.png"));
            const auto copied = directory.File(L"copied.png");
            ASSERT_EQ(fractal.SaveCurrentFractal(copied.wstring(), true), 0);
            ASSERT_TRUE(DecodeSavedPng(copied, width, height) == expected);
            ASSERT_FALSE(fractal.CleanupThreads(false));
            const auto moved = directory.File(L"moved.png");
            ASSERT_EQ(fractal.SaveCurrentFractal(moved.wstring(), false), 0);
            ASSERT_TRUE(DecodeSavedPng(moved, width, height) == expected);
            ASSERT_FALSE(fractal.CleanupThreads(true));
        }
    }
}

void
CheckRenderPoolSnapshot()
{
    PngSaveDirectory directory;
    Fractal fractal{23, 15, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
    fractal.GetRenderPool()->Drain();
    fractal.View(0, false);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32)));
    fractal.SetNumIterations<uint32_t>(128);
    fractal.CalcFractal(true);
    const auto original = SaveCpuOracle(fractal, directory.File(L"original.png"));
    fractal.EnqueueCommand("PNG save fixture", [](Fractal &target) {
        target.View(0, false);
        target.SetNumIterations<uint32_t>(16);
    });
    fractal.GetRenderPool()->Drain();
    const auto expected = SaveCpuOracle(fractal, directory.File(L"pool-oracle.png"));
    ASSERT_TRUE(expected != original);
    const auto output = directory.File(L"pool.png");
    ASSERT_EQ(fractal.SaveCurrentFractal(output.wstring(), true), 0);
    ASSERT_TRUE(DecodeSavedPng(output, 23, 15) == expected);
    ASSERT_FALSE(fractal.CleanupThreads(true));
}

void
CheckBackgroundAndWriteErrors()
{
    PngSaveDirectory directory;
    Fractal fractal{19, 13, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
    RenderRequest request;
    request.Width = 19;
    request.Height = 13;
    request.ViewSource = RenderRequest::ViewSourceKind::Builtin;
    request.BuiltinView = 0;
    request.Iterations = 64;
    request.Antialiasing = 2;
    request.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO);
    request.PngCompletion = PngCompletionMode::Background;
    const auto output = directory.File(L"frame.with-dots-\u03bb.png");
    request.OutPngBasename = output.wstring();
    std::string error;
    ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
    ASSERT_FALSE(fractal.RequiresUseLocalColor());
    const auto expected = DecodeSavedPng(output, 19, 13);
    ASSERT_FALSE(fractal.CleanupThreads(true));
    ASSERT_EQ(fractal.SaveCurrentFractal(output.wstring(), true), 0);
    ASSERT_TRUE(DecodeSavedPng(output, 19, 13) == expected);
    request.OutPngBasename = directory.File(L"missing-parent").wstring() + L"/frame.png";
    ASSERT_NE(RenderToPng(request, fractal, &error), 0);
    ASSERT_FALSE(error.empty());
    ASSERT_FALSE(fractal.CleanupThreads(true));

    // Selecting a CPU algorithm on this same instance still queues the CPU save worker.
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.CalcFractal(true);
    const auto cpuOutput = directory.File(L"cpu.png");
    ASSERT_EQ(fractal.SaveCurrentFractal(cpuOutput.wstring(), true), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    DecodeSavedPng(cpuOutput, 19, 13);
}

const bool registered = [] {
    TestFramework::RegisterCase(
        "CudaPngSave_SnapshotsAndResize", CheckSnapshotsAndResize, true, "", false);
    TestFramework::RegisterCase(
        "CudaPngSave_RenderPoolSnapshot", CheckRenderPoolSnapshot, true, "", false);
    TestFramework::RegisterCase(
        "CudaPngSave_BackgroundAndWriteErrors", CheckBackgroundAndWriteErrors, true, "", false);
    return true;
}();

} // namespace
