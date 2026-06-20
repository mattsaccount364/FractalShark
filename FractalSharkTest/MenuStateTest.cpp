#include "TestFramework.h"

#include "AlgCmds.h"
#include "Fractal.h"
#include "MenuState.h"
#include "MenuTree.h"
#include "RenderAlgorithm.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>

namespace {

const FractalShark::Node *
FindMenuNode(std::span<const FractalShark::Node> nodes, uint32_t commandId)
{
    for (const auto &node : nodes) {
        if (node.kind == FractalShark::Kind::Popup) {
            if (const auto *child = FindMenuNode(node.kids, commandId)) {
                return child;
            }
        } else if (node.kind == FractalShark::Kind::Item && node.id == commandId) {
            return &node;
        }
    }
    return nullptr;
}

uint32_t
CommandForAlgorithm(RenderAlgorithmEnum algorithm)
{
    for (const auto &entry : FractalShark::kAlgCmds) {
        if (entry.alg == algorithm) {
            return static_cast<uint32_t>(entry.id);
        }
    }
    return 0;
}

} // namespace

TEST(MenuState_EnablementRulesUseSharedRuntimeState)
{
    Fractal fractal(32, 32, nullptr, false, std::numeric_limits<uint64_t>::max());
    FractalShark::MenuState state(fractal);

    ASSERT_TRUE(state.IsEnabled(FractalShark::Rule::Always));
    ASSERT_TRUE(state.IsEnabled(FractalShark::Rule::EnableIfCpuActive));
    ASSERT_FALSE(state.IsEnabled(FractalShark::Rule::EnableIfPerturbationAvailable));
    ASSERT_EQ(state.IsEnabled(FractalShark::Rule::EnableIfGpuActive), !fractal.GpuBypassed());

    const std::filesystem::path checkpoint{"nr_checkpoint.txt"};
    const bool checkpointInitiallyExists = std::filesystem::exists(checkpoint);
    ASSERT_EQ(state.IsEnabled(FractalShark::Rule::EnableIfNRCheckpointExists),
              checkpointInitiallyExists);

    if (!checkpointInitiallyExists) {
        {
            std::ofstream output(checkpoint);
            output << "menu-state test\n";
        }
        ASSERT_TRUE(state.IsEnabled(FractalShark::Rule::EnableIfNRCheckpointExists));
        std::error_code error;
        std::filesystem::remove(checkpoint, error);
        ASSERT_FALSE(error);
    }
}

TEST(MenuState_RepaintingReflectsFractalState)
{
    Fractal fractal(32, 32, nullptr, false, std::numeric_limits<uint64_t>::max());
    FractalShark::MenuState state(fractal);

    fractal.SetRepaint(false);
    ASSERT_FALSE(state.IsChecked(IDM_REPAINTING));
    fractal.SetRepaint(true);
    ASSERT_TRUE(state.IsChecked(IDM_REPAINTING));

    ASSERT_FALSE(state.IsChecked(IDM_WINDOWED));
    ASSERT_FALSE(state.IsChecked(IDM_WINDOWED_SQ));
}

TEST(MenuState_RadioGroupsFollowFractalState)
{
    Fractal fractal(32, 32, nullptr, false, std::numeric_limits<uint64_t>::max());
    FractalShark::MenuState state(fractal);

    const auto cpu64 = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);
    (void)fractal.SetRenderAlgorithm(cpu64);
    fractal.ResetDimensions(SIZE_MAX, SIZE_MAX, 4);
    fractal.SetIterationPrecision(8);
    fractal.GetLAParameters().SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity);
    fractal.UsePaletteType(FractalPaletteType::Summer);
    fractal.UsePalette(12);
    fractal.SetResultsAutosave(AddPointOptions::DontSave);
    fractal.SetIterType(IterTypeEnum::Bits64);
    fractal.SetNRInnerLoopBackend(NRInnerLoopBackend::CpuST);

    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::RenderAlgorithm),
              CommandForAlgorithm(RenderAlgorithmEnum::Cpu64));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::GpuAntialiasing),
              static_cast<uint32_t>(IDM_GPUANTIALIASING_16X));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::IterationPrecision),
              static_cast<uint32_t>(IDM_ITERATIONPRECISION_3X));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::LaThreading),
              static_cast<uint32_t>(IDM_LA_SINGLETHREADED));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::PerturbationMode),
              static_cast<uint32_t>(IDM_PERTURBATION_SINGLETHREAD_PERIODICITY));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::PaletteType),
              static_cast<uint32_t>(IDM_PALETTE_TYPE_3));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::PaletteBitDepth),
              static_cast<uint32_t>(IDM_PALETTE_12));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::MemoryAutosave),
              static_cast<uint32_t>(IDM_PERTURB_AUTOSAVE_OFF));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::IterationsWidth),
              static_cast<uint32_t>(IDM_64BIT_ITERATIONS));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::NRInnerLoopBackend),
              static_cast<uint32_t>(IDM_NR_INNERLOOP_CPUST));
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::None), 0u);
}

TEST(MenuState_WindowSizeCommandsAreActions)
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

    const auto nodes = std::span<const Node>{menu};
    const Node *windowed = FindMenuNode(nodes, IDM_WINDOWED);
    const Node *windowedSquare = FindMenuNode(nodes, IDM_WINDOWED_SQ);

    ASSERT_TRUE(windowed != nullptr);
    ASSERT_TRUE(windowedSquare != nullptr);
    ASSERT_TRUE(windowed->checkKind == FractalShark::CheckKind::None);
    ASSERT_TRUE(windowedSquare->checkKind == FractalShark::CheckKind::None);
}
