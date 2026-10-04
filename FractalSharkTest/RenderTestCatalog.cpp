#include "RenderTestCatalog.h"

#include "Environment.h"
#include "RenderTestSupport.h"

#include <algorithm>
#include <array>
#include <set>

namespace RenderTests {
namespace {
struct AlgorithmViews {
    RenderAlgorithmEnum Algorithm;
    std::vector<size_t> Basic;
    std::vector<size_t> Reference;
    bool Reuse;
};

const AlgorithmViews OriginalViews[] = {
#include "RenderAlgorithmCases.inc"
};

std::string
CaseName(const RenderCase &test, std::string_view family)
{
    return "RenderGolden_" + std::string{family} + "_" + test.Algorithm.AlgorithmStr + "_View" +
           std::to_string(test.View) + "_Bits" + (test.Bits == IterTypeEnum::Bits32 ? "32" : "64") +
           "_Store" + std::to_string(static_cast<int>(test.Storage)) + "_Ref" +
           std::to_string(static_cast<int>(test.Reference)) + "_AA" + std::to_string(test.Antialiasing) +
           "_Step" + std::to_string(test.Step) + "_Comp" + std::to_string(test.Compression) + "_LA" +
           std::to_string(static_cast<int>(test.LA)) + "_Threads" +
           std::to_string(static_cast<int>(test.Threading)) + "_Load" +
           std::to_string(static_cast<int>(test.LoadSettings)) + (test.AutoGpu ? "_AutoGpu" : "") +
           (test.IterationLimit != 0 ? "_Iters" + std::to_string(test.IterationLimit) : "");
}

void
Append(std::vector<RenderCase> &cases, RenderCase test, std::string_view family)
{
    test.Name = CaseName(test, family);
    if (test.View == 10 || test.View == 27) {
        test.DisabledReason = "INCOMPLETE: slow view remains disabled; placeholder CRC";
    }
    if (test.Algorithm.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedScaled) {
        test.DisabledReason = "INCOMPLETE: Gpu2x32PerturbedScaled dispatch is not implemented";
    }
    if (test.Reference == RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4) {
        test.DisabledReason = "INCOMPLETE: MTMed4 reuse backend is documented as broken";
    }
    cases.push_back(std::move(test));
}

std::vector<RenderCase>
BuildCases()
{
    std::vector<RenderCase> cases;
    for (const auto &entry : OriginalViews) {
        auto views = entry.Basic;
        const auto algorithm = GetRenderAlgorithmTupleEntry(entry.Algorithm);
        const std::string name = algorithm.AlgorithmStr;
        if (name.find("LAv2") != std::string::npos || name.find("BLAV2") != std::string::npos) {
            const size_t meaningfulView = name.starts_with("Gpu1x32") || name.starts_with("Gpu2x32") ? 9
                                          : name.starts_with("GpuHDR")                               ? 11
                                                                                                     : 5;
            if (std::find(views.begin(), views.end(), meaningfulView) == views.end()) {
                views.push_back(meaningfulView);
            }
        }
        for (const size_t view : views) {
            for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
                for (const auto storage :
                     {AddPointOptions::DontSave, AddPointOptions::EnableWithoutSave}) {
                    RenderCase test;
                    test.Algorithm = algorithm;
                    test.AutoGpu = entry.Algorithm == RenderAlgorithmEnum::AUTO;
                    test.View = view;
                    test.Bits = bits;
                    test.Storage = storage;
                    if (view == 9 &&
                        std::find(entry.Basic.begin(), entry.Basic.end(), view) == entry.Basic.end()) {
                        // Added float LA mode probes retain the deep location without a stress-length
                        // loop.
                        test.IterationLimit = 50'000;
                    }
                    Append(cases, test, "Basic");
                }
            }
        }
        for (const size_t view : entry.Reference) {
            for (const auto settings : {ImaginaSettings::ConvertToCurrent, ImaginaSettings::UseSaved}) {
                RenderCase test;
                test.Kind = Scenario::ReferenceSave;
                test.Algorithm = algorithm;
                test.View = view;
                test.Bits = IterTypeEnum::Bits64;
                test.LoadSettings = settings;
                Append(cases, test, "ReferenceSave");
            }
            if (view == 5) {
                for (int32_t compression = 1; compression <= 20; ++compression) {
                    RenderCase test;
                    test.Kind = Scenario::Compression;
                    test.Algorithm = algorithm;
                    test.View = view;
                    test.Bits = IterTypeEnum::Bits64;
                    test.Compression = compression;
                    Append(cases, test, "Compression");
                }
            }
        }
        if (entry.Reuse) {
            RenderCase test;
            test.Kind = Scenario::Reuse;
            test.Algorithm = algorithm;
            test.View = 14;
            test.Antialiasing = 4;
            test.Reference = RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3;
            Append(cases, test, "PerturbedPerturb");
        }
    }

    for (const size_t view : {size_t{0}, size_t{5}}) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            for (const auto settings : {ImaginaSettings::ConvertToCurrent, ImaginaSettings::UseSaved}) {
                RenderCase test;
                test.Kind = Scenario::ReferenceSave;
                test.Algorithm =
                    GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
                test.View = view;
                test.Bits = bits;
                test.LoadSettings = settings;
                Append(cases, test, "CpuReferenceSave");
            }
        }
    }
    for (const bool useGpu : {false, true}) {
        for (const size_t view : {size_t{0}, size_t{1}, size_t{9}, size_t{5}}) {
            for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
                RenderCase test;
                test.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO);
                test.AutoGpu = useGpu;
                test.View = view;
                test.Bits = bits;
                Append(cases, test, "AutoSelection");
            }
        }
    }
    for (const auto algorithm : {RenderAlgorithmEnum::Gpu1x32,
                                 RenderAlgorithmEnum::Gpu1x64,
                                 RenderAlgorithmEnum::Gpu2x32,
                                 RenderAlgorithmEnum::GpuHDRx32}) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            for (const uint32_t step : {1u, 4u, 8u, 16u}) {
                RenderCase test;
                test.Algorithm = GetRenderAlgorithmTupleEntry(algorithm);
                test.Bits = bits;
                test.Step = step;
                Append(cases, test, "Precision");
            }
        }
    }
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (const uint32_t aa : {1u, 2u, 3u, 4u}) {
            RenderCase test;
            test.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32);
            test.Bits = bits;
            test.Antialiasing = aa;
            Append(cases, test, "Antialiasing");
        }
    }
    for (int backend = 0; backend <= static_cast<int>(RefOrbitCalc::PerturbationAlg::Auto); ++backend) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            RenderCase test;
            test.Kind = Scenario::ReferenceBackend;
            test.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
            test.View = 5;
            test.Bits = bits;
            test.Reference = static_cast<RefOrbitCalc::PerturbationAlg>(backend);
            Append(cases, test, "ReferenceBackend");
        }
    }
    for (const auto algorithm : {RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2,
                                 RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2,
                                 RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2}) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            RenderCase test;
            test.Kind = Scenario::ReferenceBackend;
            test.Algorithm = GetRenderAlgorithmTupleEntry(algorithm);
            test.View = 11;
            test.Bits = bits;
            test.Reference = RefOrbitCalc::PerturbationAlg::GPU;
            Append(cases, test, "GpuReference");
        }
    }
    for (const auto preset : {LAParameters::LADefaults::MaxAccuracy,
                              LAParameters::LADefaults::MaxPerf,
                              LAParameters::LADefaults::MinMemory}) {
        for (const auto threading : {LAParameters::LAThreadingAlgorithm::SingleThreaded,
                                     LAParameters::LAThreadingAlgorithm::MultiThreaded}) {
            RenderCase test;
            test.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
            test.View = 5;
            test.Bits = IterTypeEnum::Bits64;
            test.LA = preset;
            test.Threading = threading;
            Append(cases, test, "LASettings");
        }
    }
    for (const auto &fixture : Environment::GetEmbeddedImaginaFixtureInfos()) {
        for (const bool useGpu : {false, true}) {
            RenderCase test;
            test.Kind = Scenario::Imagina;
            test.Algorithm =
                GetRenderAlgorithmTupleEntry(useGpu ? RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2
                                                    : RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
            test.View = fixture.presetView.value_or(0);
            test.Bits = IterTypeEnum::Bits64;
            test.Fixture = fixture.name;
            Append(cases, test, "Imagina");
            cases.back().Name += "_" + std::filesystem::path{fixture.name}.string();
        }
    }
    for (int32_t compression = 18; compression <= 22; ++compression) {
        RenderCase test;
        test.Kind = Scenario::HardView;
        test.Algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2);
        test.View = 27;
        test.Bits = IterTypeEnum::Bits64;
        test.Compression = compression;
        test.LA = LAParameters::LADefaults::MaxPerf;
        Append(cases, test, "HardView");
    }
    return cases;
}
} // namespace

const std::vector<RenderCase> &
Cases()
{
    static const auto cases = BuildCases();
    return cases;
}

namespace {
const bool Registered = [] {
    for (const auto &test : Cases()) {
        const bool gpu = test.AutoGpu ||
                         (test.Algorithm.Algorithm != RenderAlgorithmEnum::AUTO &&
                          test.Algorithm.Gpu == RequiresGpu::Yes) ||
                         test.Reference == RefOrbitCalc::PerturbationAlg::GPU;
        TestFramework::RegisterCase(
            test.Name, [test] { Execute(test); }, gpu, test.DisabledReason, true);
    }
    return true;
}();
} // namespace
} // namespace RenderTests

TEST(RenderCoverage_Inventory)
{
    const auto requireCase = [](auto predicate) {
        ASSERT_TRUE(std::any_of(RenderTests::Cases().begin(), RenderTests::Cases().end(), predicate));
    };
    std::set<std::string> names;
    std::set<RenderAlgorithmEnum> algorithms;
    for (const auto &test : RenderTests::Cases()) {
        ASSERT_TRUE(names.insert(test.Name).second);
        algorithms.insert(test.Algorithm.Algorithm);
        if (test.View == 10 || test.View == 27) {
            ASSERT_FALSE(test.DisabledReason.empty());
        }
    }
    for (const auto &algorithm : RenderAlgorithms) {
        if (algorithm.Algorithm != RenderAlgorithmEnum::MAX) {
            ASSERT_TRUE(algorithms.contains(algorithm.Algorithm));
            for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
                for (const auto storage :
                     {AddPointOptions::DontSave, AddPointOptions::EnableWithoutSave}) {
                    ASSERT_TRUE(std::any_of(
                        RenderTests::Cases().begin(), RenderTests::Cases().end(), [&](const auto &test) {
                            return test.Kind == RenderTests::Scenario::Basic &&
                                   test.Algorithm.Algorithm == algorithm.Algorithm &&
                                   test.Bits == bits && test.Storage == storage;
                        }));
                }
            }
        }
    }
    for (int backend = 0; backend <= static_cast<int>(RefOrbitCalc::PerturbationAlg::Auto); ++backend) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            requireCase([&](const auto &test) {
                return test.Kind == RenderTests::Scenario::ReferenceBackend && test.Bits == bits &&
                       static_cast<int>(test.Reference) == backend;
            });
        }
    }
    for (const auto algorithm : {RenderAlgorithmEnum::Gpu1x32,
                                 RenderAlgorithmEnum::Gpu1x64,
                                 RenderAlgorithmEnum::Gpu2x32,
                                 RenderAlgorithmEnum::GpuHDRx32}) {
        for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
            for (const uint32_t step : {1u, 4u, 8u, 16u}) {
                requireCase([&](const auto &test) {
                    return test.Name.starts_with("RenderGolden_Precision_") &&
                           test.Algorithm.Algorithm == algorithm && test.Bits == bits &&
                           test.Step == step;
                });
            }
        }
    }
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        for (const uint32_t aa : {1u, 2u, 3u, 4u}) {
            requireCase([&](const auto &test) {
                return test.Name.starts_with("RenderGolden_Antialiasing_") && test.Bits == bits &&
                       test.Antialiasing == aa;
            });
        }
        for (const bool useGpu : {false, true}) {
            for (const size_t view : {size_t{0}, size_t{1}, size_t{9}, size_t{5}}) {
                requireCase([&](const auto &test) {
                    return test.Name.starts_with("RenderGolden_AutoSelection_") && test.Bits == bits &&
                           test.View == view && test.AutoGpu == useGpu;
                });
            }
        }
        for (const size_t view : {size_t{0}, size_t{5}}) {
            for (const auto mode : {ImaginaSettings::ConvertToCurrent, ImaginaSettings::UseSaved}) {
                requireCase([&](const auto &test) {
                    return test.Name.starts_with("RenderGolden_CpuReferenceSave_") &&
                           test.Bits == bits && test.View == view && test.LoadSettings == mode;
                });
            }
        }
    }
    for (const auto preset : {LAParameters::LADefaults::MaxAccuracy,
                              LAParameters::LADefaults::MaxPerf,
                              LAParameters::LADefaults::MinMemory}) {
        for (const auto threading : {LAParameters::LAThreadingAlgorithm::SingleThreaded,
                                     LAParameters::LAThreadingAlgorithm::MultiThreaded}) {
            requireCase([&](const auto &test) {
                return test.Name.starts_with("RenderGolden_LASettings_") && test.LA == preset &&
                       test.Threading == threading;
            });
        }
    }
    for (const auto &fixture : Environment::GetEmbeddedImaginaFixtureInfos()) {
        for (const bool useGpu : {false, true}) {
            requireCase([&](const auto &test) {
                return test.Kind == RenderTests::Scenario::Imagina && test.Fixture == fixture.name &&
                       (test.Algorithm.Gpu == RequiresGpu::Yes) == useGpu;
            });
        }
    }
    for (const auto &entry : RenderTests::OriginalViews) {
        for (const size_t view : entry.Reference) {
            for (const auto mode : {ImaginaSettings::ConvertToCurrent, ImaginaSettings::UseSaved}) {
                requireCase([&](const auto &test) {
                    return test.Kind == RenderTests::Scenario::ReferenceSave &&
                           test.Algorithm.Algorithm == entry.Algorithm && test.View == view &&
                           test.LoadSettings == mode;
                });
            }
            if (view == 5) {
                for (int32_t compression = 1; compression <= 20; ++compression) {
                    requireCase([&](const auto &test) {
                        return test.Kind == RenderTests::Scenario::Compression &&
                               test.Algorithm.Algorithm == entry.Algorithm &&
                               test.Compression == compression;
                    });
                }
            }
        }
        if (entry.Reuse) {
            requireCase([&](const auto &test) {
                return test.Kind == RenderTests::Scenario::Reuse &&
                       test.Algorithm.Algorithm == entry.Algorithm;
            });
        }
    }
    size_t bucketCases = 0, colorCases = 0;
    for (const auto &test : TestFramework::Registry()) {
        if (test.name.starts_with("RenderGolden_GpuReferenceBucket_")) {
            ++bucketCases;
            ASSERT_TRUE(test.RequiresGpu);
        }
        if (test.name.starts_with("RenderGolden_GpuColors_")) {
            ++colorCases;
            ASSERT_TRUE(test.RequiresGpu);
        }
    }
    ASSERT_EQ(bucketCases, size_t{36});
    ASSERT_EQ(colorCases, size_t{8});
    std::cout << "Coverage: " << algorithms.size() << " algorithm selections, "
              << RenderTests::Cases().size() << " render scenarios, 36 reference buckets, "
              << "8 GPU coloring/reduction modes; disabled entries are reported separately.\n";
}
