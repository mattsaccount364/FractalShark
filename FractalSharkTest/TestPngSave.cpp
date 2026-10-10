#include "Environment.h"
#include "FractalSaveThreadPool.h"
#include "PngParallelSave.h"
#include "RenderThreadPool.h"
#include "RenderToPng.h"
#include "TestFramework.h"
#include "WPngImage/lodepng.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

// Only the integration fixture needs to hold the encoder lease before submitting a save.
// This keeps scheduling controls out of Fractal's public interface.
class FractalSaveTestAccess {
public:
    static FractalSaveThreadPool &
    GetPool(Fractal &fractal)
    {
        return *fractal.m_SavePool;
    }
};

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
            ASSERT_TRUE(fractal.CleanupThreads(true));
            ASSERT_TRUE(DecodeSavedPng(copied, width, height) == expected);
            ASSERT_FALSE(fractal.CleanupThreads(false));
            const auto moved = directory.File(L"moved.png");
            ASSERT_EQ(fractal.SaveCurrentFractal(moved.wstring(), false), 0);
            ASSERT_TRUE(fractal.CleanupThreads(true));
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
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_TRUE(DecodeSavedPng(output, 23, 15) == expected);
    ASSERT_FALSE(fractal.CleanupThreads(true));
}

void
CheckRotatedSnapshots()
{
    PngSaveDirectory directory;
    const auto customPath = directory.File(L"palette.map");
    {
        std::ofstream output(customPath);
        output << "12 231 44\n255 7 193\n0 41 255\n";
        ASSERT_TRUE(static_cast<bool>(output));
    }
    Fractal fractal{23, 15, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
    fractal.GetRenderPool()->Drain();
    fractal.View(0, false);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32)));
    fractal.LoadCustomPalette(customPath);
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (uint32_t aa = 1; aa <= 4; ++aa) {
            fractal.GetRenderPool()->Drain();
            fractal.SetIterType(bits);
            fractal.ResetDimensions(23, 15, aa);
            fractal.SetNumIterations<uint64_t>(97);
            fractal.CalcFractal(true);
            for (const auto type : {FractalPaletteType::Default, FractalPaletteType::Custom}) {
                fractal.UsePaletteType(type);
                fractal.ResetFractalPalette();
                fractal.SetPaletteAuxDepth(0);
                const auto unrotated = SaveCpuOracle(fractal, directory.File(L"unrotated.png"));
                fractal.RotateFractalPalette(37);
                fractal.SetPaletteAuxDepth(2);
                const auto cpuPath = directory.File(L"rotated-cpu.png");
                const auto gpuPath = directory.File(L"rotated-gpu.png");
                PngParallelSave cpu(PngParallelSave::Type::PngImg,
                                    PngParallelSave::EncoderBackend::Cpu,
                                    cpuPath.wstring(),
                                    true,
                                    fractal);
                PngParallelSave gpu(PngParallelSave::Type::PngImg,
                                    PngParallelSave::EncoderBackend::Gpu,
                                    gpuPath.wstring(),
                                    true,
                                    fractal);
                // Both saves must use captured settings even if the renderer currently has newer ones.
                fractal.RotateFractalPalette(11);
                fractal.SetPaletteAuxDepth(1);
                ASSERT_EQ(cpu.Run(), 0);
                ASSERT_EQ(gpu.Run(), 0);
                const auto expected = DecodeSavedPng(cpuPath, 23, 15);
                ASSERT_TRUE(expected != unrotated);
                ASSERT_TRUE(DecodeSavedPng(gpuPath, 23, 15) == expected);
            }
        }
    }
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
    ASSERT_TRUE(fractal.CleanupThreads(true));
    const auto expected = DecodeSavedPng(output, 19, 13);
    ASSERT_FALSE(fractal.CleanupThreads(true));
    ASSERT_EQ(fractal.SaveCurrentFractal(output.wstring(), true), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_TRUE(DecodeSavedPng(output, 19, 13) == expected);
    request.OutPngBasename = directory.File(L"missing-parent").wstring() + L"/frame.png";
    ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
    ASSERT_TRUE(error.empty());
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_FALSE(std::filesystem::exists(std::filesystem::path(request.OutPngBasename)));

    // A failed worker write must not retain the encoder lease. Wait mode returns only after
    // the following valid save has been written, using the same reusable GPU workspace.
    const auto recovered = directory.File(L"recovered.png");
    request.OutPngBasename = recovered.wstring();
    request.PngCompletion = PngCompletionMode::Wait;
    ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
    ASSERT_TRUE(DecodeSavedPng(recovered, 19, 13) == expected);
    ASSERT_FALSE(fractal.CleanupThreads(true));

    // Selecting a CPU algorithm on this same instance still queues the CPU save worker.
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.CalcFractal(true);
    const auto cpuOutput = directory.File(L"cpu.png");
    ASSERT_EQ(fractal.SaveCurrentFractal(cpuOutput.wstring(), true), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    DecodeSavedPng(cpuOutput, 19, 13);
}

void
CheckAsyncSubmissionAndRenderOverlap()
{
    PngSaveDirectory directory;
    Fractal fractal{31, 19, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
    fractal.GetRenderPool()->Drain();
    fractal.View(0, false);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32)));
    fractal.SetNumIterations<uint32_t>(128);
    fractal.CalcFractal(true);
    const auto original = SaveCpuOracle(fractal, directory.File(L"original.png"));
    const auto copied = directory.File(L"queued-copy.png");

    std::packaged_task<int()> submit([&] { return fractal.SaveCurrentFractal(copied.wstring(), true); });
    auto submitted = submit.get_future();
    std::jthread submitter;
    bool returnedWhileBlocked = false;
    bool pendingWhileBlocked = false;
    {
        const auto lease = FractalSaveTestAccess::GetPool(fractal).AcquireGpuEncoding();
        submitter = std::jthread(std::move(submit));
        returnedWhileBlocked = submitted.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        if (returnedWhileBlocked) {
            pendingWhileBlocked = !std::filesystem::exists(copied) && !fractal.CleanupThreads(false);
            // The queued save owns its iterations and palette, so render-pool work can publish
            // a different frame while its GPU encoding is deliberately held back.
            fractal.EnqueueCommand("render while PNG is queued", [](Fractal &target) {
                target.RotateFractalPalette(37);
                target.SetPaletteAuxDepth(2);
                target.SetNumIterations<uint32_t>(16);
            });
            fractal.GetRenderPool()->Drain();
        }
    }
    // Always release the gate and join before asserting, including a synchronous regression.
    submitter.join();
    ASSERT_EQ(submitted.get(), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_TRUE(returnedWhileBlocked);
    ASSERT_TRUE(pendingWhileBlocked);
    ASSERT_TRUE(DecodeSavedPng(copied, 31, 19) == original);

    const auto next = SaveCpuOracle(fractal, directory.File(L"next-oracle.png"));
    ASSERT_TRUE(next != original);
    const auto moved = directory.File(L"queued-move.png");
    const auto secondCopy = directory.File(L"second-copy.png");
    ASSERT_EQ(fractal.SaveCurrentFractal(secondCopy.wstring(), true), 0);
    ASSERT_EQ(fractal.SaveCurrentFractal(moved.wstring(), false), 0);
    fractal.CalcFractal(true);
    // Resize uses the existing CPU save drain, including returning the moved buffer.
    fractal.ResetDimensions(29, 17, 2);
    fractal.CalcFractal(true);
    fractal.CleanupThreads(true);
    ASSERT_TRUE(DecodeSavedPng(secondCopy, 31, 19) == next);
    ASSERT_TRUE(DecodeSavedPng(moved, 31, 19) == next);
    const auto resized = directory.File(L"resized.png");
    const auto resizedOracle = SaveCpuOracle(fractal, directory.File(L"resized-oracle.png"));
    ASSERT_EQ(fractal.SaveCurrentFractal(resized.wstring(), false), 0);
    ASSERT_TRUE(fractal.CleanupThreads(true));
    ASSERT_TRUE(DecodeSavedPng(resized, 29, 17) == resizedOracle);
}

void
CheckShutdownCompletesSave()
{
    PngSaveDirectory directory;
    const auto output = directory.File(L"shutdown.png");
    std::vector<unsigned char> expected;
    {
        Fractal fractal{23, 15, nullptr, false, UINT64_MAX, true, GpuMode::Auto};
        fractal.GetRenderPool()->Drain();
        fractal.View(0, false);
        ASSERT_TRUE(
            fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32)));
        fractal.SetNumIterations<uint32_t>(64);
        fractal.CalcFractal(true);
        expected = SaveCpuOracle(fractal, directory.File(L"oracle.png"));
        ASSERT_EQ(fractal.SaveCurrentFractal(output.wstring(), false), 0);
    }
    ASSERT_TRUE(DecodeSavedPng(output, 23, 15) == expected);
}

const bool registered = [] {
    TestFramework::RegisterCase(
        "CudaPngSave_SnapshotsAndResize", CheckSnapshotsAndResize, true, "", false);
    TestFramework::RegisterCase(
        "CudaPngSave_RenderPoolSnapshot", CheckRenderPoolSnapshot, true, "", false);
    TestFramework::RegisterCase(
        "CudaPngSave_BackgroundAndWriteErrors", CheckBackgroundAndWriteErrors, true, "", false);
    TestFramework::RegisterCase("CudaPngSave_RotatedSnapshots", CheckRotatedSnapshots, true, "", false);
    TestFramework::RegisterCase("CudaPngSave_AsyncSubmissionAndRenderOverlap",
                                CheckAsyncSubmissionAndRenderOverlap,
                                true,
                                "",
                                false);
    TestFramework::RegisterCase(
        "CudaPngSave_ShutdownCompletesSave", CheckShutdownCompletesSave, true, "", false);
    return true;
}();

} // namespace
