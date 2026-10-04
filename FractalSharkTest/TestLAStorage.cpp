#include "GPU_LAInfoDeep.h"
#include "LAReference.h"
#include "PerturbationResults.h"
#include "TestFramework.h"

#include <filesystem>
#include <fstream>
#include <type_traits>

namespace {

class StorageFiles {
public:
    std::filesystem::path m_Base =
        std::filesystem::temp_directory_path() /
        ("fractalshark-la-storage-" + std::to_string(Environment::CurrentProcessId()));
    ~StorageFiles()
    {
        for (const char *suffix :
             {".LAs", ".LAStages", ".FullOrbit", ".met", ".parameters", ".construction"}) {
            std::error_code error;
            std::filesystem::remove(m_Base.string() + suffix, error);
        }
    }
};

template <typename IterType, class Float, class SubType>
void
CheckLayout()
{
    FractalShark::LA::CheckRowLayout<LAInfoDeep<IterType, Float, SubType, PerturbExtras::Disable>,
                                     GPU_LAInfoDeep<IterType, Float, SubType>>();
    FractalShark::LA::CheckRowLayout<
        LAInfoDeep<IterType, Float, SubType, PerturbExtras::SimpleCompression>,
        GPU_LAInfoDeep<IterType, Float, SubType>>();
}

template <typename IterType>
void
CheckGeneration()
{
    PerturbationResults<IterType, HDRFloat<float>, PerturbExtras::Disable> orbit{
        AddPointOptions::DontSave, 0};
    HDRFloatComplex<float> z{0.0f, 0.0f};
    const HDRFloatComplex<float> c{-0.12256f, 0.74486f};
    // Enough entries to exercise multiple construction workers and boundary merging.
    for (size_t i = 0; i <= 100003; ++i) {
        orbit.AddUncompressedIteration({z.getRe(), z.getIm()});
        z = z * z + c;
        z.Reduce();
    }
    LAParameters parameters;
    parameters.SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    LAReference<IterType, HDRFloat<float>, float, PerturbExtras::Disable> single{
        parameters, AddPointOptions::DontSave, L"", L""};
    single.GenerateApproximationData(orbit, HDRFloat<float>{0.00001f}, false);
    parameters.SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
    LAReference<IterType, HDRFloat<float>, float, PerturbExtras::Disable> parallel{
        parameters, AddPointOptions::DontSave, L"", L""};
    parallel.GenerateApproximationData(orbit, HDRFloat<float>{0.00001f}, false);
    ASSERT_TRUE(single.ValidateTables());
    ASSERT_TRUE(parallel.ValidateTables());
    ASSERT_EQ(single.GetLAStageCount(), parallel.GetLAStageCount());
    ASSERT_EQ(single.GetLAs().GetSize(), parallel.GetLAs().GetSize());
    for (size_t i = 0; i < single.GetLAs().GetSize(); ++i) {
        const auto &left = single.GetLAs()[i];
        const auto &right = parallel.GetLAs()[i];
        ASSERT_TRUE(left.Ref == right.Ref);
        ASSERT_TRUE(left.ZCoeff == right.ZCoeff);
        ASSERT_TRUE(left.CCoeff == right.CCoeff);
        ASSERT_TRUE(left.LAThreshold == right.LAThreshold);
        ASSERT_EQ(left.LAi.StepLength, right.LAi.StepLength);
        ASSERT_EQ(left.LAi.NextStageLAIndex, right.LAi.NextStageLAIndex);
    }
    for (IterType stage = 0; stage < single.GetLAStageCount(); ++stage) {
        const auto &left = single.GetLAStages()[stage];
        const auto &right = parallel.GetLAStages()[stage];
        ASSERT_EQ(left.LAIndex, right.LAIndex);
        ASSERT_EQ(left.MacroItCount, right.MacroItCount);
        ASSERT_TRUE(left.LAThresholdC == right.LAThresholdC);
        ASSERT_TRUE(
            single.GetLAs()[left.LAIndex + left.MacroItCount].Ref ==
            (HDRFloatComplex<float>{orbit.GetOrbitData()[100003].x, orbit.GetOrbitData()[100003].y}));
    }
}

} // namespace

TEST(LAStorage_LayoutCompatibilityAcrossTypes)
{
    CheckLayout<uint32_t, float, float>();
    CheckLayout<uint64_t, float, float>();
    CheckLayout<uint32_t, double, double>();
    CheckLayout<uint64_t, double, double>();
    CheckLayout<uint32_t, HDRFloat<float>, float>();
    CheckLayout<uint64_t, HDRFloat<float>, float>();
    CheckLayout<uint32_t, HDRFloat<double>, double>();
    CheckLayout<uint64_t, HDRFloat<double>, double>();
    CheckLayout<uint32_t, CudaDblflt<MattDblflt>, CudaDblflt<MattDblflt>>();
    CheckLayout<uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>();
    ASSERT_EQ((sizeof(LAInfoDeep<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable>)),
              size_t{52});
    ASSERT_EQ((sizeof(LAConstructionInfo<HDRFloat<float>>)), size_t{16});
    ASSERT_EQ((sizeof(LAStageInfo<uint32_t, HDRFloat<float>>)), size_t{16});
}

TEST(LAStorage_ConstructionMethodsKeepSeparateMetadata)
{
    LAParameters settings;
    LAParametersRuntime<double> parameters{settings};
    for (int method : {0, 1}) {
        parameters.m_DetectionMethod = method;
        LAConstructionEntry<uint32_t, double, double, PerturbExtras::Disable> entry{
            parameters, FloatComplex<double>{0, 0}};
        auto stepped = entry.Step(parameters, FloatComplex<double>{0.5, 0.25});
        ASSERT_NEAR(stepped.ZCoeff.getRe(), 1.0, 0.0);
        ASSERT_NEAR(stepped.ZCoeff.getIm(), 0.5, 0.0);
        ASSERT_NEAR(stepped.CCoeff.getRe(), 2.0, 0.0);
        ASSERT_NEAR(stepped.CCoeff.getIm(), 0.5, 0.0);
        ASSERT_NEAR(stepped.LAThreshold, 0.5 * parameters.m_LAThresholdScale, 0.0);
        ASSERT_NEAR(stepped.m_Construction.LAThresholdC, 0.5 * parameters.m_LAThresholdCScale, 0.0);
        if (method == 1) {
            ASSERT_NEAR(stepped.m_Construction.MinMag, 0.5, 0.0);
        }
        auto composed = stepped.Composite(parameters, entry);
        ASSERT_TRUE(composed.m_Construction.LAThresholdC <= stepped.m_Construction.LAThresholdC);
        ATInfo<uint32_t, double, double> at;
        stepped.CreateAT(at, entry, false, 0.00001);
        ASSERT_NEAR(at.ThresholdC, 0.00001, 0.0);
    }
}

TEST(LAStorage_ParallelGenerationMatchesSingleThreaded)
{
    CheckGeneration<uint32_t>();
    CheckGeneration<uint64_t>();
}

TEST(LAStorage_StageThresholdIsIndependentOfCurrentRow)
{
    LAReference<uint32_t, double, double, PerturbExtras::Disable> reference{
        AddPointOptions::DontSave, L"", L""};
    reference.GetLAs().MutableResize(4);
    reference.GetLAStages().MutableResize(1);
    reference.GetLAStages()[0] = LAStageInfo<uint32_t, double>{0, 3, 0.01};
    reference.GetLAs()[2].LAi.NextStageLAIndex = 7;
    reference.GetLAs()[2].LAThreshold = 0;
    ASSERT_FALSE(reference.IsLAStageInvalid(0, {0.005, 0}));
    ASSERT_TRUE(reference.IsLAStageInvalid(0, {0.01, 0}));
    ASSERT_EQ(reference.getLA(0, {}, 2, 0, 100).nextStageLAindex, uint32_t{7});
}

TEST(LAStorage_NativeCacheRoundTripAndRangeValidation)
{
    StorageFiles files;
    const std::wstring rows = files.m_Base.wstring() + L".LAs";
    const std::wstring stages = files.m_Base.wstring() + L".LAStages";
    PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> orbit{
        AddPointOptions::DontSave, 0};
    HDRFloatComplex<float> z{0.0f, 0.0f};
    for (size_t i = 0; i <= 100; ++i) {
        orbit.AddUncompressedIteration({z.getRe(), z.getIm()});
        z = z * z + HDRFloatComplex<float>{-0.12256f, 0.74486f};
        z.Reduce();
    }
    std::vector<LAStageInfo<uint32_t, HDRFloat<float>>> expected;
    size_t rowCount = 0;
    {
        LAReference<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable> reference{
            LAParameters{}, AddPointOptions::EnableWithSave, rows, stages};
        reference.GenerateApproximationData(orbit, HDRFloat<float>{0.00001f}, false);
        rowCount = reference.GetLAs().GetSize();
        for (uint32_t stage = 0; stage < reference.GetLAStageCount(); ++stage) {
            expected.push_back(reference.GetLAStages()[stage]);
        }
        std::ofstream metadata(files.m_Base.string() + ".met");
        ASSERT_TRUE(reference.WriteMetadata(metadata));
        ASSERT_FALSE(std::filesystem::exists(rows + L".construction"));
    }
    ASSERT_EQ(std::filesystem::file_size(rows), rowCount * 52);
    ASSERT_EQ(std::filesystem::file_size(stages), expected.size() * 16);
    LAReference<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable> loaded{
        AddPointOptions::OpenExistingWithSave, rows, stages};
    std::ifstream metadata(files.m_Base.string() + ".met");
    ASSERT_TRUE(loaded.ReadMetadata(metadata));
    ASSERT_EQ(loaded.GetLAs().GetSize(), rowCount);
    ASSERT_EQ(loaded.GetLAStageCount(), expected.size());
    for (size_t stage = 0; stage < expected.size(); ++stage) {
        ASSERT_TRUE(loaded.GetLAStages()[stage].LAThresholdC == expected[stage].LAThresholdC);
        ASSERT_EQ(loaded.GetLAStages()[stage].LAIndex, expected[stage].LAIndex);
        ASSERT_EQ(loaded.GetLAStages()[stage].MacroItCount, expected[stage].MacroItCount);
    }
    ASSERT_FALSE(expected.empty());
    loaded.GetLAStages()[0].MacroItCount = static_cast<uint32_t>(rowCount);
    ASSERT_FALSE(loaded.ValidateTables());
}

TEST(LAStorage_OldNativeVersionRejectedBeforeMapping)
{
    StorageFiles files;
    for (const char *version : {"0.46", "0.47"}) {
        {
            std::ofstream metadata(files.m_Base.string() + ".met");
            metadata << version << '\n';
        }
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> old{
            files.m_Base.wstring(), AddPointOptions::OpenExistingWithSave, 0};
        ASSERT_FALSE(old.ReadMetadata());
        ASSERT_TRUE(old.GetLaReference() == nullptr);
        ASSERT_EQ(old.GetCompressedOrUncompressedOrbitSize(), size_t{0});
    }
}

TEST(LAStorage_NativeVersionAndMalformedTables)
{
    StorageFiles files;
    {
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> orbit{
            files.m_Base.wstring(), AddPointOptions::EnableWithSave, 0};
        HDRFloatComplex<float> z{0.0f, 0.0f};
        for (size_t i = 0; i <= 100; ++i) {
            orbit.AddUncompressedIteration({z.getRe(), z.getIm()});
            z = z * z + HDRFloatComplex<float>{-0.12256f, 0.74486f};
            z.Reduce();
        }
        auto reference =
            std::make_unique<LAReference<uint32_t, HDRFloat<float>, float, PerturbExtras::Disable>>(
                LAParameters{},
                AddPointOptions::EnableWithSave,
                files.m_Base.wstring() + L".LAs",
                files.m_Base.wstring() + L".LAStages");
        reference->GenerateApproximationData(orbit, HDRFloat<float>{0.00001f}, false);
        ASSERT_TRUE(reference->IsValid());
        orbit.SetLaReference(std::move(reference));
        orbit.WriteMetadata();
    }
    std::string metadata;
    {
        std::ifstream input(files.m_Base.string() + ".met");
        metadata.assign(std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{});
    }
    ASSERT_TRUE(metadata.starts_with("0.545\n"));
    {
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> loaded{
            files.m_Base.wstring(), AddPointOptions::OpenExistingWithSave, 0};
        ASSERT_TRUE(loaded.ReadMetadata());
        ASSERT_EQ(loaded.GetCountOrbitEntries(), uint32_t{101});
        ASSERT_TRUE(loaded.GetLaReference() != nullptr);
        ASSERT_TRUE(loaded.GetLaReference()->ValidateTables());
    }
    // Reject counts before narrowing to the specialization's iteration type.
    std::string corrupt = metadata;
    const size_t begin = corrupt.find("LAStageCount: ");
    ASSERT_TRUE(begin != std::string::npos);
    corrupt.replace(begin, corrupt.find('\n', begin) - begin, "LAStageCount: 4294967296");
    {
        std::ofstream output(files.m_Base.string() + ".met", std::ios::binary);
        output << corrupt;
    }
    {
        PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> loaded{
            files.m_Base.wstring(), AddPointOptions::OpenExistingWithSave, 0};
        ASSERT_FALSE(loaded.ReadMetadata());
        ASSERT_TRUE(loaded.GetLaReference() == nullptr);
    }
    {
        std::ofstream output(files.m_Base.string() + ".met", std::ios::binary);
        output << metadata;
    }
    const auto rows = files.m_Base.string() + ".LAs";
    std::filesystem::resize_file(rows, std::filesystem::file_size(rows) - 1);
    PerturbationResults<uint32_t, HDRFloat<float>, PerturbExtras::Disable> malformed{
        files.m_Base.wstring(), AddPointOptions::OpenExistingWithSave, 0};
    ASSERT_THROWS(malformed.ReadMetadata(), FractalSharkSeriousException);
}

TEST(LAStorage_EmptyOrbitIsInvalid)
{
    PerturbationResults<uint32_t, double, PerturbExtras::Disable> orbit{AddPointOptions::DontSave, 0};
    LAReference<uint32_t, double, double, PerturbExtras::Disable> reference{
        LAParameters{}, AddPointOptions::DontSave, L"", L""};
    reference.GenerateApproximationData(orbit, 0.00001, false);
    ASSERT_FALSE(reference.IsValid());
    ASSERT_EQ(reference.GetLAStageCount(), uint32_t{0});
    ASSERT_EQ(reference.GetLAs().GetSize(), size_t{0});
}
