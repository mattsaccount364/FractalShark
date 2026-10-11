#include "Environment.h"
#include "Fractal.h"
#include "FractalSaveThreadPool.h"
#include "PngParallelSave.h"
#include "PngSaveLayout.h"
#include "RenderToPng.h"
#include "SaveFileTransaction.h"
#include "TestFramework.h"
#include "WPngImage/WPngImage.hh"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <sstream>
#include <thread>

namespace {
class SaveDirectory {
public:
    std::filesystem::path Path;
    SaveDirectory()
    {
        static std::atomic<unsigned> sequence{0};
        Path = std::filesystem::temp_directory_path() /
               ("fractalshark-transactions-" + std::to_string(Environment::CurrentProcessId()) + "-" +
                std::to_string(sequence.fetch_add(1)));
        std::filesystem::create_directory(Path);
    }
    ~SaveDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(Path, ignored);
    }
};

std::vector<unsigned char>
ReadBytes(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

SaveResult
WriteText(const std::filesystem::path &path, SaveOverwritePolicy overwrite, const std::string &text)
{
    return SaveFileTransaction::Write(
        Environment::FromFsPath(path), overwrite, [&](std::ostream &out) { out << text; });
}
} // namespace

TEST(PngSave_AsyncResultsReportFailureAndRecovery)
{
    SaveDirectory directory;
    const auto invalid = directory.Path / "missing" / "failed.png";
    const auto valid = directory.Path / "valid.png";
    Fractal fractal{16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled};
    fractal.GetRenderPool()->Drain();
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    fractal.SetNumIterations<uint32_t>(64);
    fractal.CalcFractal(true);
    auto failed =
        fractal.SaveCurrentFractalWithResult(invalid.wstring(), true, SaveOverwritePolicy::SkipExisting);
    auto succeeded =
        fractal.SaveCurrentFractalWithResult(valid.wstring(), true, SaveOverwritePolicy::SkipExisting);
    // Drain with both futures still retained: result ownership must not retain snapshots.
    ASSERT_TRUE(fractal.CleanupThreads(true));
    const auto failure = failed.get();
    ASSERT_TRUE(failure.Outcome == SaveOutcome::Failed);
    ASSERT_TRUE(failure.OutputPath == invalid.wstring());
    ASSERT_NE(failure.ErrorCode, 0);
    ASSERT_FALSE(failure.Diagnostic.empty());
    ASSERT_FALSE(std::filesystem::exists(invalid));
    ASSERT_TRUE(succeeded.get().Outcome == SaveOutcome::Written);
    WPngImage image;
    ASSERT_TRUE(image.loadImage(valid.string()) == WPngImage::kIOStatus_Ok);
    ASSERT_EQ(image.width(), 16);
    ASSERT_EQ(image.height(), 16);
    const auto original = ReadBytes(valid);
    auto skipped =
        fractal.SaveCurrentFractalWithResult(valid.wstring(), true, SaveOverwritePolicy::SkipExisting);
    fractal.CleanupThreads(true);
    ASSERT_TRUE(skipped.get().Outcome == SaveOutcome::SkippedExisting);
    ASSERT_TRUE(ReadBytes(valid) == original);
}

TEST(PngSave_CompletionResultOutlivesFractal)
{
    SaveDirectory directory;
    const auto output = directory.Path / "moved.png";
    std::future<SaveResult> completion;
    {
        Fractal fractal{16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled};
        fractal.GetRenderPool()->Drain();
        ASSERT_TRUE(
            fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
        fractal.SetNumIterations<uint32_t>(64);
        fractal.CalcFractal(true);
        completion = fractal.SaveCurrentFractalWithResult(
            output.wstring(), false, SaveOverwritePolicy::SkipExisting);
        // Shutdown must destroy the moved-buffer recycler before the Fractal owner dies,
        // even when nobody has consumed the completion future yet.
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(output));
    ASSERT_TRUE(completion.get().Outcome == SaveOutcome::Written);
}

TEST(PngSave_RenderToPngChecksWaitAndExposesBackgroundResults)
{
    SaveDirectory directory;
    Fractal fractal{16, 16, nullptr, false, UINT64_MAX, true, GpuMode::Disabled};
    RenderRequest request;
    request.Width = 16;
    request.Height = 16;
    request.ViewSource = RenderRequest::ViewSourceKind::Builtin;
    request.BuiltinView = 0;
    request.Iterations = 64;
    request.Antialiasing = 1;
    request.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);
    const auto invalid = directory.Path / "missing" / "failed.png";
    request.OutPngBasename = invalid.wstring();
    std::string error;
    ASSERT_EQ(RenderToPng(request, fractal, &error), 1);
    ASSERT_TRUE(error.find(invalid.string()) != std::string::npos);
    ASSERT_TRUE(error.find("could not write PNG file") != std::string::npos);
    ASSERT_FALSE(fractal.CleanupThreads(true));

    request.PngCompletion = PngCompletionMode::Background;
    std::future<SaveResult> completion;
    std::ostringstream output;
    error.clear();
    ASSERT_EQ(RenderToPng(request, fractal, &error, output, &completion), 0);
    ASSERT_TRUE(error.empty());
    ASSERT_TRUE(completion.valid());
    fractal.CleanupThreads(true);
    ASSERT_TRUE(completion.get().Outcome == SaveOutcome::Failed);

    const auto valid = directory.Path / "recovered.png";
    request.OutPngBasename = valid.wstring();
    request.PngCompletion = PngCompletionMode::Wait;
    ASSERT_EQ(RenderToPng(request, fractal, &error, output, &completion), 0);
    ASSERT_FALSE(completion.valid());
    ASSERT_TRUE(std::filesystem::is_regular_file(valid));
    const auto original = ReadBytes(valid);
    ASSERT_EQ(RenderToPng(request, fractal, &error), 0);
    ASSERT_TRUE(ReadBytes(valid) == original);

    request.PngCompletion = PngCompletionMode::Background;
    ASSERT_EQ(RenderToPng(request, fractal, &error, output, &completion), 0);
    fractal.CleanupThreads(true);
    ASSERT_TRUE(completion.valid());
    // An empty-output render resets a previously valid handle rather than exposing
    // the preceding image's outcome as this render's save completion.
    request.OutPngBasename.clear();
    ASSERT_EQ(RenderToPng(request, fractal, &error, output, &completion), 0);
    ASSERT_FALSE(completion.valid());
}

TEST(PngSave_AtomicPublicationAcrossPools)
{
    SaveDirectory directory;
    const auto path = directory.Path / "race.txt";
    FractalSaveThreadPool first(1, [] { return uint32_t{0}; });
    FractalSaveThreadPool second(1, [] { return uint32_t{0}; });
    std::barrier ready(2);
    const auto submit = [&](FractalSaveThreadPool &pool, std::string payload, SaveResult &result) {
        pool.Submit([&, payload = std::move(payload)] {
            return FractalSaveThreadPool::Task([&, payload] {
                result = SaveFileTransaction::Write(Environment::FromFsPath(path),
                                                    SaveOverwritePolicy::SkipExisting,
                                                    [&](std::ostream &out) {
                                                        out << payload;
                                                        ready.arrive_and_wait();
                                                    });
            });
        });
    };
    SaveResult ar, br;
    submit(first, std::string(4096, 'a'), ar);
    submit(second, std::string(4096, 'b'), br);
    // Draining both pools publishes worker writes to these test-local results.
    first.Cleanup(true);
    second.Cleanup(true);
    ASSERT_TRUE(ar.Succeeded() && br.Succeeded());
    ASSERT_TRUE((ar.Outcome == SaveOutcome::Written) != (br.Outcome == SaveOutcome::Written));
    const auto bytes = ReadBytes(path);
    ASSERT_EQ(bytes.size(), size_t{4096});
    ASSERT_TRUE(std::all_of(bytes.begin(), bytes.end(), [&](auto byte) { return byte == bytes[0]; }));
    ASSERT_EQ(std::distance(std::filesystem::directory_iterator(directory.Path),
                            std::filesystem::directory_iterator()),
              1);
}

TEST(PngSave_ReplacementVisibilityAndUnicode)
{
    SaveDirectory directory;
    const auto path = directory.Path / Environment::ToFsPath(L"image-\u03bb-\u65e5.png");
    ASSERT_TRUE(WriteText(path, SaveOverwritePolicy::SkipExisting, "old").Succeeded());
    ASSERT_TRUE(WriteText(path, SaveOverwritePolicy::SkipExisting, "skipped").Outcome ==
                SaveOutcome::SkippedExisting);
    const auto result = SaveFileTransaction::Write(
        Environment::FromFsPath(path), SaveOverwritePolicy::ReplaceExisting, [&](std::ostream &out) {
            out << "new";
            ASSERT_TRUE(ReadBytes(path) == std::vector<unsigned char>({'o', 'l', 'd'}));
        });
    ASSERT_TRUE(result.Outcome == SaveOutcome::Written);
    ASSERT_TRUE(ReadBytes(path) == std::vector<unsigned char>({'n', 'e', 'w'}));
    ASSERT_TRUE(result.OutputPath == Environment::FromFsPath(path));
}

TEST(PngSave_PartialWriteAndCloseFailuresPreserveDestination)
{
    SaveDirectory directory;
    const auto path = directory.Path / "retained.png";
    ASSERT_TRUE(WriteText(path, SaveOverwritePolicy::SkipExisting, "original").Succeeded());
    for (bool closeEarly : {false, true}) {
        const auto result = SaveFileTransaction::Write(
            Environment::FromFsPath(path), SaveOverwritePolicy::ReplaceExisting, [&](std::ostream &out) {
                out << "partial";
                if (closeEarly) {
                    // Force the transaction's close to fail with an already-closed filebuf.
                    dynamic_cast<std::ofstream &>(out).close();
                } else {
                    out.setstate(std::ios::badbit);
                }
            });
        ASSERT_TRUE(result.Outcome == SaveOutcome::Failed);
        ASSERT_TRUE(result.ErrorCode != 0);
        ASSERT_TRUE(result.Diagnostic.find(closeEarly ? "close" : "write") != std::string::npos);
        ASSERT_EQ(ReadBytes(path).size(), size_t{8});
        ASSERT_EQ(std::distance(std::filesystem::directory_iterator(directory.Path),
                                std::filesystem::directory_iterator()),
                  1);
    }
}

TEST(PngSave_PublicationAndWriterFailuresCleanTemporaryFiles)
{
    SaveDirectory directory;
    const auto path = directory.Path / "directory.png";
    std::filesystem::create_directory(path);
    const auto result = WriteText(path, SaveOverwritePolicy::ReplaceExisting, "new");
    ASSERT_TRUE(result.Outcome == SaveOutcome::Failed);
    ASSERT_TRUE(std::filesystem::is_directory(path));
    ASSERT_THROWS(SaveFileTransaction::Write((directory.Path / "throw.png").wstring(),
                                             SaveOverwritePolicy::ReplaceExisting,
                                             [](std::ostream &out) {
                                                 out << "partial";
                                                 throw std::runtime_error("writer failed");
                                             }),
                  std::runtime_error);
    ASSERT_TRUE(
        WriteText(directory.Path / "missing" / "image.png", SaveOverwritePolicy::SkipExisting, "new")
            .Outcome == SaveOutcome::Failed);
    ASSERT_EQ(std::distance(std::filesystem::directory_iterator(directory.Path),
                            std::filesystem::directory_iterator()),
              1);
}

TEST(PngSave_DrainReleasesCapturesAfterExceptions)
{
    struct Capture {
        std::atomic<bool> &Destroyed;
        explicit Capture(std::atomic<bool> &destroyed) : Destroyed(destroyed) {}
        ~Capture() { Destroyed.store(true); }
    };
    std::atomic<bool> destroyed{false};
    FractalSaveThreadPool pool(1, [] { return uint32_t{0}; });
    pool.Submit([&] {
        return FractalSaveThreadPool::Task([capture = std::make_unique<Capture>(destroyed)] {
            throw std::runtime_error("intentional encode exception");
        });
    });
    pool.Cleanup(true);
    ASSERT_TRUE(destroyed.load());
    std::atomic<bool> completed{false};
    pool.Submit([&] { return FractalSaveThreadPool::Task([&] { completed.store(true); }); });
    pool.Shutdown();
    ASSERT_TRUE(completed.load());
}

TEST(PngSave_CheckedLargeRgb16ByteCounts)
{
    size_t bytes = 0;
    ASSERT_TRUE(Png::Rgb16ByteCount(32768, 32768, bytes));
    ASSERT_EQ(bytes, size_t{6} * 1024 * 1024 * 1024);
    ASSERT_FALSE(Png::Rgb16ByteCount(0, 1, bytes));
    ASSERT_FALSE(Png::Rgb16ByteCount(SIZE_MAX, 1, bytes));
    ASSERT_FALSE(Png::Rgb16ByteCount(0x80000000, 1, bytes));
    ASSERT_FALSE(Png::Rgb16ByteCount(0x7fffffff, 0x7fffffff, bytes));
}

TEST(PngSave_IterationOnlyCopiesRetainRowsAndFullCopiesRetainColors)
{
    for (auto type : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        ItersMemoryContainer original(type, 3, 5, 4);
        original.SetItersArrayValSlow(11, 19, 12345);
        original.m_RoundedOutputColorMemory[0] = Color16{17, 19, 23, 65535};
        ItersMemoryContainer iterations;
        iterations.CopyIterationsFrom(original);
        ASSERT_FALSE(iterations.m_RoundedOutputColorMemory);
        ASSERT_EQ(iterations.GetItersArrayValSlow(11, 19), uint64_t{12345});
        original.SetItersArrayValSlow(11, 19, 0);
        ASSERT_EQ(iterations.GetItersArrayValSlow(11, 19), uint64_t{12345});
        ItersMemoryContainer full;
        full = original;
        ASSERT_TRUE(full.m_RoundedOutputColorMemory != nullptr);
        ASSERT_EQ(full.m_RoundedOutputColorMemory[0].r, uint16_t{17});
    }
}

TEST(PngSave_CpuTypedSamplesAndBitstreamsMatchIndependentOracle)
{
    SaveDirectory directory;
    FractalSaveThreadPool pool(2, [] { return uint32_t{0}; });
    const std::vector<Color16> palette{
        {1000, 2000, 3000, 65535}, {4000, 5000, 6000, 65535}, {7000, 8000, 9000, 65535}};
    for (auto type : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (size_t aa = 1; aa <= 4; ++aa) {
            for (uint64_t rotation : {uint64_t{7}, UINT64_MAX}) {
                ItersMemoryContainer iterations(type, 3, 5, aa);
                WPngImage oracle(3, 5, WPngImage::Pixel16(0, 0, 0));
                for (size_t y = 0; y < 5; ++y) {
                    for (size_t x = 0; x < 3; ++x) {
                        uint32_t r = 0, g = 0, b = 0;
                        for (size_t sy = 0; sy < aa; ++sy) {
                            for (size_t sx = 0; sx < aa; ++sx) {
                                const uint64_t count = (x + y + sx + sy) % 5;
                                iterations.SetItersArrayValSlow(x * aa + sx, y * aa + sy, count);
                                if (count < 4) {
                                    // Rotation seven never saturates; maximal rotation always
                                    // selects the terminal count (99), independently of overflow.
                                    const auto &color =
                                        palette[((rotation == 7 ? count + 7 : 99) >> 1) % 3];
                                    r += color.r;
                                    g += color.g;
                                    b += color.b;
                                }
                            }
                        }
                        oracle.set(static_cast<int>(x),
                                   static_cast<int>(y),
                                   WPngImage::Pixel16(static_cast<uint16_t>(r / (aa * aa)),
                                                      static_cast<uint16_t>(g / (aa * aa)),
                                                      static_cast<uint16_t>(b / (aa * aa))));
                    }
                }
                const auto path = directory.Path / "typed.png";
                PngParallelSave save(PngParallelSave::Type::PngImg,
                                     PngParallelSave::EncoderBackend::Cpu,
                                     Environment::FromFsPath(path),
                                     SaveOverwritePolicy::ReplaceExisting,
                                     iterations,
                                     true,
                                     4,
                                     rotation,
                                     1,
                                     100,
                                     palette,
                                     pool,
                                     {});
                ASSERT_TRUE(save.Run().Outcome == SaveOutcome::Written);
                std::vector<unsigned char> expected;
                ASSERT_TRUE(oracle.SaveImageToRAM(expected,
                                                  WPngImage::PngFileFormat::kPngFileFormat_RGBA16,
                                                  WPngImage::PngEncodingOptions{false, false, 32}) ==
                            WPngImage::kIOStatus_Ok);
                ASSERT_TRUE(ReadBytes(path) == expected);
            }
        }
    }
}

TEST(PngSave_UnnamedJobsAdvanceOnAtomicCollision)
{
    SaveDirectory directory;
    const auto oldDirectory = std::filesystem::current_path();
    struct RestoreDirectory {
        std::filesystem::path Path;
        ~RestoreDirectory() { std::filesystem::current_path(Path); }
    } restore{oldDirectory};
    std::filesystem::current_path(directory.Path);
    // Occupied names include directories. Skip them before creating and filling a
    // temporary file, while the publication operation still resolves actual races.
    std::filesystem::create_directory("output00000.txt");
    ASSERT_TRUE(WriteText("output00001.txt", SaveOverwritePolicy::SkipExisting, "retained").Succeeded());
    FractalSaveThreadPool first(1, [] { return uint32_t{0}; });
    FractalSaveThreadPool second(1, [] { return uint32_t{0}; });
    ItersMemoryContainer iterations(IterTypeEnum::Bits32, 1, 1, 1);
    iterations.SetItersArrayValSlow(0, 0, 1);
    std::barrier ready(2);
    const auto submit = [&](FractalSaveThreadPool &pool, SaveResult &result) {
        pool.Submit([&] {
            auto save = std::make_unique<PngParallelSave>(PngParallelSave::Type::ItersText,
                                                          PngParallelSave::EncoderBackend::Cpu,
                                                          L"",
                                                          SaveOverwritePolicy::SkipExisting,
                                                          iterations,
                                                          true,
                                                          4,
                                                          0,
                                                          0,
                                                          100,
                                                          std::vector<Color16>{},
                                                          pool,
                                                          PngParallelSave::ReturnIterationsFunction{});
            return FractalSaveThreadPool::Task([&, save = std::move(save)] {
                ready.arrive_and_wait();
                result = save->Run();
            });
        });
    };
    SaveResult ar, br;
    submit(first, ar);
    submit(second, br);
    first.Cleanup(true);
    second.Cleanup(true);
    ASSERT_TRUE(ar.Outcome == SaveOutcome::Written && br.Outcome == SaveOutcome::Written);
    ASSERT_TRUE(ar.OutputPath != br.OutputPath);
    ASSERT_TRUE(std::filesystem::is_directory("output00000.txt"));
    ASSERT_EQ(ReadBytes("output00001.txt").size(), size_t{8});
    ASSERT_TRUE(std::filesystem::is_regular_file("output00002.txt"));
    ASSERT_TRUE(std::filesystem::is_regular_file("output00003.txt"));
    ASSERT_TRUE(ReadBytes("output00002.txt") == ReadBytes("output00003.txt"));
}
