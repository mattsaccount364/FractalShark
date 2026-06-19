// CommandCatalogTest.cpp
//
// Mechanical contract test for FractalShark::ExecuteCommand: for every
// FractalCommand value that the production switch handles, calling
// ExecuteCommand against a recording host must invoke exactly the matching
// On*() hook.  Catches typos in the Phase 0c giant switch on both Win32
// and Linux without requiring a human to click the menu.
//
// We also exercise:
//   - The algorithm-family fast path (FractalCommand::Alg* → OnSetAlgorithm
//     with the correct RenderAlgorithmEnum payload).
//   - The closed-world contract: unknown FractalCommand values throw instead
//     of silently routing back to the platform shell.

#include "CommandCatalog.h"
#include "AlgCmds.h"
#include "Exceptions.h"
#include "MenuTree.h"
#include "RenderAlgorithm.h"
#include "TestFramework.h"

#include <span>
#include <string>

namespace {

struct RecordingHost : FractalShark::ExecuteCommandHost {
    std::string lastMethod;
    ::RenderAlgorithmEnum lastAlg = ::RenderAlgorithmEnum::AUTO;

    void
    OnSetAlgorithm(::RenderAlgorithmEnum alg) override
    {
        lastMethod = "OnSetAlgorithm";
        lastAlg = alg;
    }

#define FRACTALSHARK_TEST_RECORD(method)                                                                \
    void method() override { lastMethod = #method; }

    FRACTALSHARK_TEST_RECORD(OnAutoZoomFeatureAtPoint)
    FRACTALSHARK_TEST_RECORD(OnAutoZoomDefaultAtPoint)
    FRACTALSHARK_TEST_RECORD(OnCenterViewClearPerturbation)
    FRACTALSHARK_TEST_RECORD(OnResetCompressionDefaults)
    FRACTALSHARK_TEST_RECORD(OnLaThresholdScaleIncrease)
    FRACTALSHARK_TEST_RECORD(OnLaThresholdScaleDecrease)
    FRACTALSHARK_TEST_RECORD(OnLaPeriodDetectionIncrease)
    FRACTALSHARK_TEST_RECORD(OnLaPeriodDetectionDecrease)
    FRACTALSHARK_TEST_RECORD(OnRecalcCurrentCopyDetails)
    FRACTALSHARK_TEST_RECORD(OnRecalcClearMediumCopyDetails)
    FRACTALSHARK_TEST_RECORD(OnRecalcClearAllCopyDetails)
    FRACTALSHARK_TEST_RECORD(OnRecalcClearLaCopyDetails)
    FRACTALSHARK_TEST_RECORD(OnIntermediateCompressionIncrease)
    FRACTALSHARK_TEST_RECORD(OnIntermediateCompressionDecrease)
    FRACTALSHARK_TEST_RECORD(OnLowCompressionIncrease)
    FRACTALSHARK_TEST_RECORD(OnLowCompressionDecrease)
    FRACTALSHARK_TEST_RECORD(OnPaletteAuxDepthNext)
    FRACTALSHARK_TEST_RECORD(OnPaletteAuxDepthPrevious)
    FRACTALSHARK_TEST_RECORD(OnPaletteDepthNext)
    FRACTALSHARK_TEST_RECORD(OnRecalcClearAllSquareView)

    FRACTALSHARK_TEST_RECORD(OnShowHotkeys)
    FRACTALSHARK_TEST_RECORD(OnViewsHelp)
    FRACTALSHARK_TEST_RECORD(OnHelpAlg)
    FRACTALSHARK_TEST_RECORD(OnSquareView)
    FRACTALSHARK_TEST_RECORD(OnRepainting)
    FRACTALSHARK_TEST_RECORD(OnWindowed)
    FRACTALSHARK_TEST_RECORD(OnWindowedSq)
    FRACTALSHARK_TEST_RECORD(OnMinimize)
    FRACTALSHARK_TEST_RECORD(OnCurPos)
    FRACTALSHARK_TEST_RECORD(OnExit)

    FRACTALSHARK_TEST_RECORD(OnBack)
    FRACTALSHARK_TEST_RECORD(OnCenterView)
    FRACTALSHARK_TEST_RECORD(OnZoomIn)
    FRACTALSHARK_TEST_RECORD(OnZoomOut)
    FRACTALSHARK_TEST_RECORD(OnAutoZoomDefault)
    FRACTALSHARK_TEST_RECORD(OnAutoZoomMax)
    FRACTALSHARK_TEST_RECORD(OnAutoZoomFilament)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderDirect)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderDirectScan)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderPt)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderPtScan)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderLa)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderLaScan)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderZoom)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderClear)
    FRACTALSHARK_TEST_RECORD(OnFeatureFinderResume)
    FRACTALSHARK_TEST_RECORD(OnNrInnerLoopGpu)
    FRACTALSHARK_TEST_RECORD(OnNrInnerLoopCpu)
    FRACTALSHARK_TEST_RECORD(OnNrInnerLoopCpuSt)

    FRACTALSHARK_TEST_RECORD(OnStandardView)

    size_t lastViewIndex = static_cast<size_t>(-1);
    void
    OnSelectBuiltInView(size_t oneBasedIndex) override
    {
        lastMethod = "OnSelectBuiltInView";
        lastViewIndex = oneBasedIndex;
    }

    FRACTALSHARK_TEST_RECORD(OnGpuAntialiasing1x)
    FRACTALSHARK_TEST_RECORD(OnGpuAntialiasing4x)
    FRACTALSHARK_TEST_RECORD(OnGpuAntialiasing9x)
    FRACTALSHARK_TEST_RECORD(OnGpuAntialiasing16x)

    FRACTALSHARK_TEST_RECORD(OnResetIterations)
    FRACTALSHARK_TEST_RECORD(OnIncreaseIterations1p5x)
    FRACTALSHARK_TEST_RECORD(OnIncreaseIterations6x)
    FRACTALSHARK_TEST_RECORD(OnIncreaseIterations24x)
    FRACTALSHARK_TEST_RECORD(OnDecreaseIterations)
    FRACTALSHARK_TEST_RECORD(OnIterations32Bit)
    FRACTALSHARK_TEST_RECORD(OnIterations64Bit)

    FRACTALSHARK_TEST_RECORD(OnIterationPrecision1x)
    FRACTALSHARK_TEST_RECORD(OnIterationPrecision2x)
    FRACTALSHARK_TEST_RECORD(OnIterationPrecision3x)
    FRACTALSHARK_TEST_RECORD(OnIterationPrecision4x)

    FRACTALSHARK_TEST_RECORD(OnPerturbClearAll)
    FRACTALSHARK_TEST_RECORD(OnPerturbClearMed)
    FRACTALSHARK_TEST_RECORD(OnPerturbClearHigh)
    FRACTALSHARK_TEST_RECORD(OnPerturbationAuto)
    FRACTALSHARK_TEST_RECORD(OnPerturbationSinglethread)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMultithread)
    FRACTALSHARK_TEST_RECORD(OnPerturbationSinglethreadPeriodicity)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMultithread2Periodicity)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMt2PerturbMthighStmed)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMt2PerturbMthighMtmed1)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMt2PerturbMthighMtmed2)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMt2PerturbMthighMtmed3)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMt2PerturbMthighMtmed4)
    FRACTALSHARK_TEST_RECORD(OnPerturbationMultithread5Periodicity)
    FRACTALSHARK_TEST_RECORD(OnPerturbationGpu)
    FRACTALSHARK_TEST_RECORD(OnPerturbationLoad)
    FRACTALSHARK_TEST_RECORD(OnPerturbationSave)

    FRACTALSHARK_TEST_RECORD(OnPerturbAutosaveOnDelete)
    FRACTALSHARK_TEST_RECORD(OnPerturbAutosaveOn)
    FRACTALSHARK_TEST_RECORD(OnPerturbAutosaveOff)

    FRACTALSHARK_TEST_RECORD(OnPaletteType0)
    FRACTALSHARK_TEST_RECORD(OnPaletteType1)
    FRACTALSHARK_TEST_RECORD(OnPaletteType2)
    FRACTALSHARK_TEST_RECORD(OnPaletteType3)
    FRACTALSHARK_TEST_RECORD(OnPaletteType4)
    FRACTALSHARK_TEST_RECORD(OnCreateNewPalette)
    FRACTALSHARK_TEST_RECORD(OnPalette5)
    FRACTALSHARK_TEST_RECORD(OnPalette6)
    FRACTALSHARK_TEST_RECORD(OnPalette8)
    FRACTALSHARK_TEST_RECORD(OnPalette12)
    FRACTALSHARK_TEST_RECORD(OnPalette16)
    FRACTALSHARK_TEST_RECORD(OnPalette20)
    FRACTALSHARK_TEST_RECORD(OnPaletteRotate)

    FRACTALSHARK_TEST_RECORD(OnSaveLocation)
    FRACTALSHARK_TEST_RECORD(OnSaveHiResBmp)
    FRACTALSHARK_TEST_RECORD(OnSaveItersText)
    FRACTALSHARK_TEST_RECORD(OnSaveBmp)
    FRACTALSHARK_TEST_RECORD(OnSaveRefOrbitText)
    FRACTALSHARK_TEST_RECORD(OnSaveRefOrbitTextSimple)
    FRACTALSHARK_TEST_RECORD(OnSaveRefOrbitTextMax)
    FRACTALSHARK_TEST_RECORD(OnSaveRefOrbitImagMax)
    FRACTALSHARK_TEST_RECORD(OnDiffRefOrbitImagMax)
    FRACTALSHARK_TEST_RECORD(OnLoadLocation)
    FRACTALSHARK_TEST_RECORD(OnLoadEnterLocation)
    FRACTALSHARK_TEST_RECORD(OnLoadRefOrbitImagMax)
    FRACTALSHARK_TEST_RECORD(OnLoadRefOrbitImagMaxSaved)

    FRACTALSHARK_TEST_RECORD(OnBasicTest)
    FRACTALSHARK_TEST_RECORD(OnTest27)
    FRACTALSHARK_TEST_RECORD(OnBenchmarkFull)
    FRACTALSHARK_TEST_RECORD(OnBenchmarkInt)

    FRACTALSHARK_TEST_RECORD(OnLaMultithreaded)
    FRACTALSHARK_TEST_RECORD(OnLaSinglethreaded)
    FRACTALSHARK_TEST_RECORD(OnLaSettings1)
    FRACTALSHARK_TEST_RECORD(OnLaSettings2)
    FRACTALSHARK_TEST_RECORD(OnLaSettings3)

#undef FRACTALSHARK_TEST_RECORD
};

void
ExpectHook(RecordingHost &host, FractalShark::FractalCommand cmd, const char *expectedHook)
{
    host.lastMethod.clear();
    FractalShark::ExecuteCommand(cmd, host);
    if (host.lastMethod != expectedHook) {
        std::ostringstream oss;
        oss << "ExecuteCommand(" << static_cast<unsigned>(cmd) << ") routed to '" << host.lastMethod
            << "', expected '" << expectedHook << "'";
        TestFramework::Fail(__FILE__, __LINE__, oss.str());
    }
}

void
ExpectHotKey(FractalShark::HotKey hotkey, wchar_t key, bool shift, bool ctrl = false, bool alt = false)
{
    hotkey = FractalShark::NormalizeHotKey(hotkey);
    ASSERT_TRUE(hotkey.key == key);
    ASSERT_TRUE(hotkey.shift == shift);
    ASSERT_TRUE(hotkey.ctrl == ctrl);
    ASSERT_TRUE(hotkey.alt == alt);
}

} // namespace

#define EXPECT(cmd, hook) ExpectHook(host, FractalShark::FractalCommand::cmd, #hook)

TEST(CommandCatalog_HotKeyFromCharacterBuildsCatalogShape)
{
    using FractalShark::FractalCommand;
    using FractalShark::HotKeyFromCharacter;

    ExpectHotKey(HotKeyFromCharacter(L'a', false, false, false), L'a', false);
    ExpectHotKey(HotKeyFromCharacter(L'A', true, false, false), L'a', true);
    ExpectHotKey(HotKeyFromCharacter(L'A', false, false, false), L'a', false);
    ExpectHotKey(HotKeyFromCharacter(L'a', false, true, true), L'a', false, true, true);
    ExpectHotKey(HotKeyFromCharacter(L'+', false, false, false), L'=', true);
    ExpectHotKey(HotKeyFromCharacter(L'<', false, false, false), L',', true);
    ExpectHotKey(HotKeyFromCharacter(L'>', false, false, false), L'.', true);
    ASSERT_FALSE(HotKeyFromCharacter(L'_', true, false, false).HasKey());

    const FractalShark::Command *plus =
        FractalShark::FindCommandByHotKey(HotKeyFromCharacter(L'+', false, false, false));
    ASSERT_TRUE(plus != nullptr);
    ASSERT_TRUE(plus->id == FractalCommand::IncreaseIterations24x);

    const FractalShark::Command *less =
        FractalShark::FindCommandByHotKey(HotKeyFromCharacter(L'<', false, false, false));
    ASSERT_TRUE(less != nullptr);
    ASSERT_TRUE(less->id == FractalCommand::FeatureFinderLaScan);

    const FractalShark::Command *greater =
        FractalShark::FindCommandByHotKey(HotKeyFromCharacter(L'>', false, false, false));
    ASSERT_TRUE(greater != nullptr);
    ASSERT_TRUE(greater->id == FractalCommand::FeatureFinderClear);
}

TEST(CommandCatalog_RoutesAllMigratedHooks)
{
    RecordingHost host;

    // Synthetic shortcut commands
    EXPECT(AutoZoomFeatureAtPoint, OnAutoZoomFeatureAtPoint);
    EXPECT(AutoZoomDefaultAtPoint, OnAutoZoomDefaultAtPoint);
    EXPECT(CenterViewClearPerturbation, OnCenterViewClearPerturbation);
    EXPECT(ResetCompressionDefaults, OnResetCompressionDefaults);
    EXPECT(LaThresholdScaleIncrease, OnLaThresholdScaleIncrease);
    EXPECT(LaThresholdScaleDecrease, OnLaThresholdScaleDecrease);
    EXPECT(LaPeriodDetectionIncrease, OnLaPeriodDetectionIncrease);
    EXPECT(LaPeriodDetectionDecrease, OnLaPeriodDetectionDecrease);
    EXPECT(RecalcCurrentCopyDetails, OnRecalcCurrentCopyDetails);
    EXPECT(RecalcClearMediumCopyDetails, OnRecalcClearMediumCopyDetails);
    EXPECT(RecalcClearAllCopyDetails, OnRecalcClearAllCopyDetails);
    EXPECT(RecalcClearLaCopyDetails, OnRecalcClearLaCopyDetails);
    EXPECT(IntermediateCompressionIncrease, OnIntermediateCompressionIncrease);
    EXPECT(IntermediateCompressionDecrease, OnIntermediateCompressionDecrease);
    EXPECT(LowCompressionIncrease, OnLowCompressionIncrease);
    EXPECT(LowCompressionDecrease, OnLowCompressionDecrease);
    EXPECT(PaletteAuxDepthNext, OnPaletteAuxDepthNext);
    EXPECT(PaletteAuxDepthPrevious, OnPaletteAuxDepthPrevious);
    EXPECT(PaletteDepthNext, OnPaletteDepthNext);
    EXPECT(RecalcClearAllSquareView, OnRecalcClearAllSquareView);

    // Help / Window
    EXPECT(ShowHotkeys, OnShowHotkeys);
    EXPECT(ViewsHelp, OnViewsHelp);
    EXPECT(HelpAlg, OnHelpAlg);
    EXPECT(SquareView, OnSquareView);
    EXPECT(Repainting, OnRepainting);
    EXPECT(Windowed, OnWindowed);
    EXPECT(WindowedSq, OnWindowedSq);
    EXPECT(Minimize, OnMinimize);
    EXPECT(CurPos, OnCurPos);
    EXPECT(Exit, OnExit);

    // Navigate
    EXPECT(Back, OnBack);
    EXPECT(CenterView, OnCenterView);
    EXPECT(ZoomIn, OnZoomIn);
    EXPECT(ZoomOut, OnZoomOut);
    EXPECT(AutoZoomDefault, OnAutoZoomDefault);
    EXPECT(AutoZoomMax, OnAutoZoomMax);
    EXPECT(AutoZoomFilament, OnAutoZoomFilament);
    EXPECT(FeatureFinderDirect, OnFeatureFinderDirect);
    EXPECT(FeatureFinderDirectScan, OnFeatureFinderDirectScan);
    EXPECT(FeatureFinderPt, OnFeatureFinderPt);
    EXPECT(FeatureFinderPtScan, OnFeatureFinderPtScan);
    EXPECT(FeatureFinderLa, OnFeatureFinderLa);
    EXPECT(FeatureFinderLaScan, OnFeatureFinderLaScan);
    EXPECT(FeatureFinderZoom, OnFeatureFinderZoom);
    EXPECT(FeatureFinderClear, OnFeatureFinderClear);
    EXPECT(FeatureFinderResume, OnFeatureFinderResume);
    EXPECT(NrInnerLoopGpu, OnNrInnerLoopGpu);
    EXPECT(NrInnerLoopCpu, OnNrInnerLoopCpu);
    EXPECT(NrInnerLoopCpuSt, OnNrInnerLoopCpuSt);

    // Built-in views (point entry only; View1..40 ranges tested separately)
    EXPECT(StandardView, OnStandardView);

    // Antialiasing
    EXPECT(GpuAntialiasing1x, OnGpuAntialiasing1x);
    EXPECT(GpuAntialiasing4x, OnGpuAntialiasing4x);
    EXPECT(GpuAntialiasing9x, OnGpuAntialiasing9x);
    EXPECT(GpuAntialiasing16x, OnGpuAntialiasing16x);

    // Iterations
    EXPECT(ResetIterations, OnResetIterations);
    EXPECT(IncreaseIterations1p5x, OnIncreaseIterations1p5x);
    EXPECT(IncreaseIterations6x, OnIncreaseIterations6x);
    EXPECT(IncreaseIterations24x, OnIncreaseIterations24x);
    EXPECT(DecreaseIterations, OnDecreaseIterations);
    EXPECT(Iterations32Bit, OnIterations32Bit);
    EXPECT(Iterations64Bit, OnIterations64Bit);

    // Iteration precision
    EXPECT(IterationPrecision1x, OnIterationPrecision1x);
    EXPECT(IterationPrecision2x, OnIterationPrecision2x);
    EXPECT(IterationPrecision3x, OnIterationPrecision3x);
    EXPECT(IterationPrecision4x, OnIterationPrecision4x);

    // Perturbation
    EXPECT(PerturbClearAll, OnPerturbClearAll);
    EXPECT(PerturbClearMed, OnPerturbClearMed);
    EXPECT(PerturbClearHigh, OnPerturbClearHigh);
    EXPECT(PerturbationAuto, OnPerturbationAuto);
    EXPECT(PerturbationSinglethread, OnPerturbationSinglethread);
    EXPECT(PerturbationMultithread, OnPerturbationMultithread);
    EXPECT(PerturbationSinglethreadPeriodicity, OnPerturbationSinglethreadPeriodicity);
    EXPECT(PerturbationMultithread2Periodicity, OnPerturbationMultithread2Periodicity);
    EXPECT(PerturbationMultithread2PeriodicityPerturbMthighStmed, OnPerturbationMt2PerturbMthighStmed);
    EXPECT(PerturbationMultithread2PeriodicityPerturbMthighMtmed1, OnPerturbationMt2PerturbMthighMtmed1);
    EXPECT(PerturbationMultithread2PeriodicityPerturbMthighMtmed2, OnPerturbationMt2PerturbMthighMtmed2);
    EXPECT(PerturbationMultithread2PeriodicityPerturbMthighMtmed3, OnPerturbationMt2PerturbMthighMtmed3);
    EXPECT(PerturbationMultithread2PeriodicityPerturbMthighMtmed4, OnPerturbationMt2PerturbMthighMtmed4);
    EXPECT(PerturbationMultithread5Periodicity, OnPerturbationMultithread5Periodicity);
    EXPECT(PerturbationGpu, OnPerturbationGpu);
    EXPECT(PerturbationLoad, OnPerturbationLoad);
    EXPECT(PerturbationSave, OnPerturbationSave);

    // Memory / Autosave
    EXPECT(PerturbAutosaveOnDelete, OnPerturbAutosaveOnDelete);
    EXPECT(PerturbAutosaveOn, OnPerturbAutosaveOn);
    EXPECT(PerturbAutosaveOff, OnPerturbAutosaveOff);

    // Palette
    EXPECT(PaletteType0, OnPaletteType0);
    EXPECT(PaletteType1, OnPaletteType1);
    EXPECT(PaletteType2, OnPaletteType2);
    EXPECT(PaletteType3, OnPaletteType3);
    EXPECT(PaletteType4, OnPaletteType4);
    EXPECT(CreateNewPalette, OnCreateNewPalette);
    EXPECT(Palette5, OnPalette5);
    EXPECT(Palette6, OnPalette6);
    EXPECT(Palette8, OnPalette8);
    EXPECT(Palette12, OnPalette12);
    EXPECT(Palette16, OnPalette16);
    EXPECT(Palette20, OnPalette20);
    EXPECT(PaletteRotate, OnPaletteRotate);

    // Save / Load
    EXPECT(SaveLocation, OnSaveLocation);
    EXPECT(SaveHiResBmp, OnSaveHiResBmp);
    EXPECT(SaveItersText, OnSaveItersText);
    EXPECT(SaveBmp, OnSaveBmp);
    EXPECT(SaveRefOrbitText, OnSaveRefOrbitText);
    EXPECT(SaveRefOrbitTextSimple, OnSaveRefOrbitTextSimple);
    EXPECT(SaveRefOrbitTextMax, OnSaveRefOrbitTextMax);
    EXPECT(SaveRefOrbitImagMax, OnSaveRefOrbitImagMax);
    EXPECT(DiffRefOrbitImagMax, OnDiffRefOrbitImagMax);
    EXPECT(LoadLocation, OnLoadLocation);
    EXPECT(LoadEnterLocation, OnLoadEnterLocation);
    EXPECT(LoadRefOrbitImagMax, OnLoadRefOrbitImagMax);
    EXPECT(LoadRefOrbitImagMaxSaved, OnLoadRefOrbitImagMaxSaved);

    // Tests / Benchmarks
    EXPECT(BasicTest, OnBasicTest);
    EXPECT(Test27, OnTest27);
    EXPECT(BenchmarkFull, OnBenchmarkFull);
    EXPECT(BenchmarkInt, OnBenchmarkInt);

    // LA
    EXPECT(LaMultithreaded, OnLaMultithreaded);
    EXPECT(LaSinglethreaded, OnLaSinglethreaded);
    EXPECT(LaSettings1, OnLaSettings1);
    EXPECT(LaSettings2, OnLaSettings2);
    EXPECT(LaSettings3, OnLaSettings3);
}

#undef EXPECT

TEST(CommandCatalog_AlgorithmFamilyRoutesViaOnSetAlgorithm)
{
    RecordingHost host;

    // Sample one command per family of RenderAlgorithmEnum entries.  The
    // catalog discovers them via kAlgCmds; it must always reach
    // OnSetAlgorithm with the correct enum value.
    for (const auto &entry : FractalShark::kAlgCmds) {
        host.lastMethod.clear();
        host.lastAlg = ::RenderAlgorithmEnum::AUTO;

        const auto cmd = FractalShark::CommandFromIdm(static_cast<uint32_t>(entry.id));
        FractalShark::ExecuteCommand(cmd, host);

        ASSERT_EQ(host.lastMethod, std::string("OnSetAlgorithm"));
        ASSERT_TRUE(host.lastAlg == entry.alg);
    }
}

TEST(CommandCatalog_RangeCommandsRouteToSelectBuiltInView)
{
    RecordingHost host;

    // View1..View40 must route through OnSelectBuiltInView with a 1-based
    // index so Linux can implement the saved-view menu without re-deriving
    // native command-id range arithmetic.
    for (uint32_t v = static_cast<uint32_t>(FractalShark::FractalCommand::View1);
         v <= static_cast<uint32_t>(FractalShark::FractalCommand::View40);
         ++v) {
        host.lastMethod.clear();
        host.lastViewIndex = static_cast<size_t>(-1);

        FractalShark::ExecuteCommand(static_cast<FractalShark::FractalCommand>(v), host);

        ASSERT_EQ(host.lastMethod, std::string("OnSelectBuiltInView"));
        const size_t expected =
            static_cast<size_t>(v - static_cast<uint32_t>(FractalShark::FractalCommand::View1)) + 1u;
        ASSERT_EQ(host.lastViewIndex, expected);
    }
}

TEST(CommandCatalog_NoneCommandThrows)
{
    RecordingHost host;

    ASSERT_THROWS(FractalShark::ExecuteCommand(FractalShark::FractalCommand::None, host),
                  FractalSharkSeriousException);
}

namespace {

void
WalkAndCheckLeaves(std::span<const FractalShark::Node> nodes, RecordingHost &host)
{
    using FractalShark::Kind;
    for (const auto &node : nodes) {
        if (node.kind == Kind::Popup) {
            WalkAndCheckLeaves(node.kids, host);
            continue;
        }
        if (node.kind == Kind::Separator) {
            continue;
        }

        // Item / Toggle / Radio carry an id and must dispatch.
        ASSERT_TRUE(node.id != 0);

        host.lastMethod.clear();
        try {
            FractalShark::ExecuteCommand(FractalShark::CommandFromIdm(node.id), host);
        } catch (const std::exception &e) {
            std::ostringstream oss;
            oss << "Menu leaf id=" << node.id << " threw from ExecuteCommand: " << e.what();
            TestFramework::Fail(__FILE__, __LINE__, oss.str());
        }

        if (host.lastMethod.empty()) {
            std::ostringstream oss;
            oss << "Menu leaf id=" << node.id << " dispatched via ExecuteCommand but no host hook fired";
            TestFramework::Fail(__FILE__, __LINE__, oss.str());
        }
    }
}

} // namespace

TEST(CommandCatalog_MenuTreeLeavesAllRouteToHooks)
{
    using FractalShark::FractalCommand;
    using FractalShark::Item;
    using FractalShark::Node;
    using FractalShark::Popup;
    using FractalShark::Radio;
    using FractalShark::RadioGroup;
    using FractalShark::Rule;
    using FractalShark::Sep;
    using FractalShark::Toggle;

#include "MenuTreeDef.h"

    RecordingHost host;
    WalkAndCheckLeaves(std::span<const Node>{menu}, host);
}
