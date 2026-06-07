//
// MenuTreeDef.h - canonical menu definition.
//
// This header is INCLUDED INSIDE a function body that has already brought
// the FractalShark::Menu namespace's factories (Sep, Item, Toggle, Radio,
// Popup) and types (Node, Rule, RadioGroup) into scope. It defines a single
// `static const Node menu[] = {...}` array describing the application's
// context menu.
//
// Important detail: popup children must have *stable storage*.
// Braced lists like `{ Item(...), ... }` produce temporary arrays, and
// std::initializer_list does not own that memory. If we stored an
// initializer_list inside a Node and recursed later, we could end up
// iterating dangling memory.
//
// To keep the "giant definition" style while making kids stable, each
// Popup is created via a small wrapper (FS_POPUP / FS_POPUP0) that:
//   1) hoists the children into a hidden `static const Node _kids[]` array,
//   2) passes `std::span{_kids}` into Popup(...),
//   3) returns a normal Node that simply *views* those children.
//
// Result: `menu[]` stays readable and static, and every submenu's children
// live for the entire program lifetime, so recursive menu walking is safe.
//

#include <span>

using RG = RadioGroup;
using R = Rule;

#define FS_POPUP(label, enableRule, adornGroup, ...)                                                    \
    []() -> Node {                                                                                      \
        static const Node _kids[] = {__VA_ARGS__};                                                      \
        return Popup(label, std::span<const Node>{_kids}, enableRule, adornGroup);                      \
    }()

#define FS_POPUP0(label, ...) FS_POPUP(label, R::Always, RG::None, __VA_ARGS__)

// -----------------------------------------------------------------------------
// Menu definition (structure preserved; only Popup(...) replaced)
// -----------------------------------------------------------------------------

static const Node menu[] = {
    Item(L"Show Help", FractalCommand::ShowHotkeys),
    Sep(),

    FS_POPUP0(
        L"Navigate",
        Item(L"Back", FractalCommand::Back),
        Sep(),
        Item(L"Center View Here", FractalCommand::CenterView),
        Item(L"Zoom In Here", FractalCommand::ZoomIn),
        Item(L"Zoom Out", FractalCommand::ZoomOut),
        Sep(),
        Item(L"Autozoom Default", FractalCommand::AutoZoomDefault),
        Item(L"Autozoom Max", FractalCommand::AutoZoomMax),
        Item(L"Autozoom Filament Tip (S)", FractalCommand::AutoZoomFilament),
        Sep(),
        FS_POPUP0(L"Feature Finder",
                  Item(L"Direct (n)", FractalCommand::FeatureFinderDirect),
                  Item(L"DirectScan (N)", FractalCommand::FeatureFinderDirectScan),
                  Item(L"PT (m)", FractalCommand::FeatureFinderPt),
                  Item(L"PTScan (M)", FractalCommand::FeatureFinderPtScan),
                  Item(L"LA (,)", FractalCommand::FeatureFinderLa),
                  Item(L"LAScan (<)", FractalCommand::FeatureFinderLaScan),
                  Sep(),
                  Item(L"Zoom to Found Feature (.)", FractalCommand::FeatureFinderZoom),
                  Item(L"Resume NR Refinement", FractalCommand::FeatureFinderResume, R::EnableIfNRCheckpointExists),
                  Item(L"Clear All Found Features (>)", FractalCommand::FeatureFinderClear),
                  Sep(),
                  Radio(L"NR Inner Loop: GPU",
                        FractalCommand::NrInnerLoopGpu,
                        RG::NRInnerLoopBackend,
                        R::EnableIfGpuActive),
                  Radio(L"NR Inner Loop: CPU MT", FractalCommand::NrInnerLoopCpu, RG::NRInnerLoopBackend),
                  Radio(L"NR Inner Loop: CPU ST", FractalCommand::NrInnerLoopCpuSt, RG::NRInnerLoopBackend))),
    FS_POPUP0(L"Built-In Views",
              Item(L"Help", FractalCommand::ViewsHelp),
              Sep(),
              Item(L"&Standard View", FractalCommand::StandardView),
              Item(L"#1 - 4x64 GPU Limit", FractalCommand::View1),
              Item(L"#2 - 4x32 GPU Limit", FractalCommand::View2),
              Item(L"#3 - 32-bit BLA Limit", FractalCommand::View3),
              Item(L"#4 - 32-bit Scaled Limit", FractalCommand::View4),
              Item(L"#5 - 32-bit Scaled Limit, 4xAA", FractalCommand::View5),
              Item(L"#6 - Scale float with pixellation", FractalCommand::View6),
              Item(L"#7 - Scaled float limit with square", FractalCommand::View7),
              Item(L"#8 - ~10^700 or so, 0.5b iterations", FractalCommand::View8),
              Item(L"#9 - bug: scaled/BLA/32-bit failure", FractalCommand::View9),
              Item(L"#10 - ~10^130, ~2b iterations", FractalCommand::View10),
              Item(L"#11 - ~10^700 or so, ~3m iterations", FractalCommand::View11),
              Item(L"#12 - Nearby #14", FractalCommand::View12),
              Item(L"#13 - ~10^4000, ~100m iterations, 16xAA", FractalCommand::View13),
              Item(L"#14 - ~10^6000, ~2b iterations, 16xAA", FractalCommand::View14),
              Item(L"#15 - ~10^140, high period", FractalCommand::View15),
              Item(L"#16 - LAv2 test spot 1 (quick)", FractalCommand::View16),
              Item(L"#17 - LAv2 test spot 2 (quick)", FractalCommand::View17),
              Item(L"#18 - #5 with point + magnification", FractalCommand::View18),
              Item(L"#19 - LAv2 test spot 3 (quick)", FractalCommand::View19),
              Item(L"#20 - ~10 minutes ref orbit, 50x10^9 iters", FractalCommand::View20),
              Item(L"#21 - ~10^11000, 2b iterations", FractalCommand::View21),
              Item(L"#22 - high period", FractalCommand::View22),
              Item(L"#23 - high period, 50b iterations", FractalCommand::View23),
              Item(L"#24", FractalCommand::View24),
              Item(L"#25 - Claude - low period, 32-bit, hard GPU", FractalCommand::View25),
              Item(L"#26 - low period, 1x32 busted, 1x64 OK, hard GPU", FractalCommand::View26),
              Item(L"#27", FractalCommand::View27),
              Item(L"#28", FractalCommand::View28),
              Item(L"#29 - Fast HDRx32 broke, HDRx64 works", FractalCommand::View29),
              Item(L"#30 - 1e114514 - low period (70s rtx5090 reforbit)", FractalCommand::View30),
              Item(L"#31 - 1e715 - feature finder test", FractalCommand::View31),
              Item(L"#32 - 1e244240 - period 27,209,300", FractalCommand::View32)),

    Item(L"Recalculate, Reuse Reference", FractalCommand::SquareView),
    Sep(),

    // These read their check state from IMenuState::IsChecked(commandId)
    Toggle(L"Toggle Repainting", FractalCommand::Repainting),
    Toggle(L"Toggle Window Size", FractalCommand::Windowed),
    Toggle(L"Toggle Window Size (Square)", FractalCommand::WindowedSq),

    Item(L"Minimize Window", FractalCommand::Minimize),
    Sep(),

    FS_POPUP(L"Antialiasing",
             R::Always,
             /*adornment*/ RG::GpuAntialiasing,
             Radio(L"1x (fast)", FractalCommand::GpuAntialiasing1x, RG::GpuAntialiasing),
             Radio(L"4x", FractalCommand::GpuAntialiasing4x, RG::GpuAntialiasing),
             Radio(L"9x", FractalCommand::GpuAntialiasing9x, RG::GpuAntialiasing),
             Radio(L"16x (better quality)", FractalCommand::GpuAntialiasing16x, RG::GpuAntialiasing)),

    FS_POPUP(
        L"Choose Render Algorithm",
        R::Always,
        /*adornment*/ RG::RenderAlgorithm,

        Item(L"Help", FractalCommand::HelpAlg),

        // All algorithms are one big mutually-exclusive group
        Radio(L"Auto (Default)", FractalCommand::AlgAuto, RG::RenderAlgorithm),

        FS_POPUP0(L"LA Parameters",
                  Radio(L"Multithreaded (Default)", FractalCommand::LaMultithreaded, RG::LaThreading),
                  Radio(L"Single Threaded", FractalCommand::LaSinglethreaded, RG::LaThreading),
                  Sep(),
                  Item(L"Max Accuracy (Default)", FractalCommand::LaSettings1),
                  Item(L"Max Performance (Accuracy Loss)", FractalCommand::LaSettings2),
                  Item(L"Min Memory", FractalCommand::LaSettings3)),

        Sep(),

        FS_POPUP0(
            L"CPU-Only",
            Radio(L"Very High Precision CPU", FractalCommand::AlgCpuHigh, RG::RenderAlgorithm),
            Sep(),
            Radio(L"1x64 CPU", FractalCommand::AlgCpu1x64, RG::RenderAlgorithm),
            Radio(L"HDRx32 CPU", FractalCommand::AlgCpu1x32Hdr, RG::RenderAlgorithm),
            Radio(L"HDRx64 CPU", FractalCommand::AlgCpu1x64Hdr, RG::RenderAlgorithm),
            Sep(),
            Radio(L"1x64 CPU - Perturbation BLA", FractalCommand::AlgCpu1x64PerturbBla, RG::RenderAlgorithm),
            Radio(
                L"HDRx32 CPU - Perturbation BLA", FractalCommand::AlgCpu1x32PerturbBlaHdr, RG::RenderAlgorithm),
            Radio(
                L"HDRx64 CPU - Perturbation BLA", FractalCommand::AlgCpu1x64PerturbBlaHdr, RG::RenderAlgorithm),
            Sep(),
            Radio(L"HDRx32 CPU - Perturbation LAv2",
                  FractalCommand::AlgCpu1x32PerturbBlav2Hdr,
                  RG::RenderAlgorithm),
            Radio(L"HDRx64 CPU - Perturbation LAv2",
                  FractalCommand::AlgCpu1x64PerturbBlav2Hdr,
                  RG::RenderAlgorithm),
            Radio(L"HDRx32 RC CPU - Perturbation LAv2",
                  FractalCommand::AlgCpu1x32PerturbRcBlav2Hdr,
                  RG::RenderAlgorithm),
            Radio(L"HDRx64 RC CPU - Perturbation LAv2",
                  FractalCommand::AlgCpu1x64PerturbRcBlav2Hdr,
                  RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Low-Zoom Depth",

            FS_POPUP(L"Iteration Precision",
                     R::Always,
                     /*adornment*/ RG::IterationPrecision,
                     Radio(L"4x (fast)", FractalCommand::IterationPrecision4x, RG::IterationPrecision),
                     Radio(L"3x", FractalCommand::IterationPrecision3x, RG::IterationPrecision),
                     Radio(L"2x", FractalCommand::IterationPrecision2x, RG::IterationPrecision),
                     Radio(L"1x (better quality)", FractalCommand::IterationPrecision1x, RG::IterationPrecision)),

            Sep(),

            Radio(L"1x32 GPU", FractalCommand::AlgGpu1x32, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"2x32 GPU", FractalCommand::AlgGpu2x32, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"4x32 GPU", FractalCommand::AlgGpu4x32, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"1x64 GPU", FractalCommand::AlgGpu1x64, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"2x64 GPU", FractalCommand::AlgGpu2x64, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"4x64 GPU", FractalCommand::AlgGpu4x64, RG::RenderAlgorithm, R::EnableIfGpuActive),
            Radio(L"HDRx32 GPU", FractalCommand::AlgGpu2x32Hdr, RG::RenderAlgorithm, R::EnableIfGpuActive)),

        FS_POPUP0(L"Scaled",
                  Radio(L"1x32 GPU - Perturbation Scaled",
                        FractalCommand::AlgGpu1x32PerturbScaled,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"2x32 GPU - Perturbation Scaled (broken)",
                        FractalCommand::AlgGpu2x32PerturbScaled,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx32 GPU - Perturbation Scaled",
                        FractalCommand::AlgGpuHdr32PerturbScaled,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive)),

        FS_POPUP0(L"Bilinear Approximation V1",
                  Radio(L"1x64 GPU - Perturbation BLA",
                        FractalCommand::AlgGpu1x64PerturbBla,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx32 GPU - Perturbation BLA",
                        FractalCommand::AlgGpuHdr32PerturbBla,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx64 GPU - Perturbation BLA",
                        FractalCommand::AlgGpuHdr64PerturbBla,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive)),

        FS_POPUP0(L"LA Only (for testing)",
                  Radio(L"1x32 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpu1x32PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"2x32 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpu2x32PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"1x64 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpu1x64PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx32 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpuHdr32PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx2x32 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpuHdr2x32PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx64 GPU - LAv2 - LA only",
                        FractalCommand::AlgGpuHdr64PerturbLav2Lao,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive)),

        FS_POPUP0(L"Perturbation Only",
                  Radio(L"1x32 GPU - Perturb only",
                        FractalCommand::AlgGpu1x32PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"2x32 GPU - Perturb only",
                        FractalCommand::AlgGpu2x32PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"1x64 GPU - Perturb only",
                        FractalCommand::AlgGpu1x64PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx32 GPU - Perturb only",
                        FractalCommand::AlgGpuHdr32PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx2x32 GPU - Perturb only",
                        FractalCommand::AlgGpuHdr2x32PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive),
                  Radio(L"HDRx64 GPU - Perturb only",
                        FractalCommand::AlgGpuHdr64PerturbLav2Po,
                        RG::RenderAlgorithm,
                        R::EnableIfGpuActive)),

        FS_POPUP0(
            L"Reference Compression",

            Radio(L"1x32 GPU - RC LAv2",
                  FractalCommand::AlgGpu1x32PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),
            Radio(L"2x32 GPU - RC LAv2",
                  FractalCommand::AlgGpu2x32PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),
            Radio(L"1x64 GPU - RC LAv2",
                  FractalCommand::AlgGpu1x64PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),
            Radio(L"HDRx32 GPU - RC LAv2",
                  FractalCommand::AlgGpuHdr32PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),
            Radio(L"HDRx2x32 GPU - RC LAv2",
                  FractalCommand::AlgGpuHdr2x32PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),
            Radio(L"HDRx64 GPU - RC LAv2",
                  FractalCommand::AlgGpuHdr64PerturbRcLav2,
                  RG::RenderAlgorithm,
                  R::EnableIfGpuActive),

            FS_POPUP0(L"Perturbation Only",
                      Radio(L"1x32 GPU - RC Perturb Only",
                            FractalCommand::AlgGpu1x32PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"2x32 GPU - RC Perturb Only",
                            FractalCommand::AlgGpu2x32PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"1x64 GPU - RC Perturb Only",
                            FractalCommand::AlgGpu1x64PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx32 GPU - RC Perturb Only",
                            FractalCommand::AlgGpuHdr32PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx2x32 GPU - RC Perturb Only",
                            FractalCommand::AlgGpuHdr2x32PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx64 GPU - RC Perturb Only",
                            FractalCommand::AlgGpuHdr64PerturbRcLav2Po,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive)),

            FS_POPUP0(L"LA Only",
                      Radio(L"1x32 GPU - RC LAv2",
                            FractalCommand::AlgGpu1x32PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"2x32 GPU - RC LAv2",
                            FractalCommand::AlgGpu2x32PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"1x64 GPU - RC LAv2",
                            FractalCommand::AlgGpu1x64PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx32 GPU - RC LAv2",
                            FractalCommand::AlgGpuHdr32PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx2x32 GPU - RC LAv2",
                            FractalCommand::AlgGpuHdr2x32PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive),
                      Radio(L"HDRx64 GPU - RC LAv2",
                            FractalCommand::AlgGpuHdr64PerturbRcLav2Lao,
                            RG::RenderAlgorithm,
                            R::EnableIfGpuActive))),

        Sep(),

        Radio(L"1x32 GPU - LAv2",
              FractalCommand::AlgGpu1x32PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),
        Radio(L"2x32 GPU - LAv2",
              FractalCommand::AlgGpu2x32PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),
        Radio(L"1x64 GPU - LAv2",
              FractalCommand::AlgGpu1x64PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),
        Radio(L"HDRx32 GPU - LAv2",
              FractalCommand::AlgGpuHdr32PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),
        Radio(L"HDRx2x32 GPU - LAv2",
              FractalCommand::AlgGpuHdr2x32PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),
        Radio(L"HDRx64 GPU - LAv2",
              FractalCommand::AlgGpuHdr64PerturbLav2,
              RG::RenderAlgorithm,
              R::EnableIfGpuActive),

        Sep(),

        FS_POPUP0(L"Tests",
                  Item(L"Run Basic Test (saves files in local dir)", FractalCommand::BasicTest),
                  Item(L"Run View #27 test", FractalCommand::Test27))),

    FS_POPUP0(L"Iterations",
              Item(L"De&fault Iterations", FractalCommand::ResetIterations),
              Item(L"+1.5x", FractalCommand::IncreaseIterations1p5x),
              Item(L"+6x", FractalCommand::IncreaseIterations6x),
              Item(L"+24x", FractalCommand::IncreaseIterations24x),
              Item(L"&Decrease Iterations", FractalCommand::DecreaseIterations),
              Sep(),
              Radio(L"32-Bit Iterations", FractalCommand::Iterations32Bit, RG::IterationsWidth),
              Radio(L"64-Bit Iterations", FractalCommand::Iterations64Bit, RG::IterationsWidth)),

    FS_POPUP(
        L"Perturbation",
        R::Always,
        /*adornment*/ RG::PerturbationMode,

        Item(L"Clear Perturbation References - All", FractalCommand::PerturbClearAll),
        Item(L"Clear Perturbation References - Med", FractalCommand::PerturbClearMed),
        Item(L"Clear Perturbation References - High", FractalCommand::PerturbClearHigh),
        Item(L"Show Perturbation Results", FractalCommand::PerturbResults),

        Sep(),

        Radio(L"Auto (default)", FractalCommand::PerturbationAuto, RG::PerturbationMode),
        Radio(L"Single Thread (ST)", FractalCommand::PerturbationSinglethread, RG::PerturbationMode),
        Radio(L"Multi Thread (MT)", FractalCommand::PerturbationMultithread, RG::PerturbationMode),
        Radio(L"ST + Periodicity", FractalCommand::PerturbationSinglethreadPeriodicity, RG::PerturbationMode),
        Radio(L"MT2 + Periodicity", FractalCommand::PerturbationMultithread2Periodicity, RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb ST",
              FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighStmed,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v1",
              FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed1,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v2",
              FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed2,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v3",
              FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed3,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v4 (broken)",
              FractalCommand::PerturbationMultithread2PeriodicityPerturbMthighMtmed4,
              RG::PerturbationMode),

        // Formerly Inactive(). Replace with a rule your IMenuState returns false for.
        Item(L"MT5 + Periodicity",
             FractalCommand::PerturbationMultithread5Periodicity,
             R::EnableIfPerturbationAvailable),

        // If this is also a mode selection you can convert it to Radio() in RG::PerturbationMode.
        Radio(L"GPU-Accelerated (see README)",
              FractalCommand::PerturbationGpu,
              RG::PerturbationMode,
              R::EnableIfGpuActive),

        Sep(),
        Item(L"Clear and Reload Reference Orbits", FractalCommand::PerturbationLoad),
        Item(L"Save Reference Orbits", FractalCommand::PerturbationSave)),

    Sep(),

    FS_POPUP(L"Palette Color Depth",
             R::Always,
             /*adornment*/ RG::PaletteBitDepth,

             Radio(L"Basic", FractalCommand::PaletteType0, RG::PaletteType),
             Radio(L"Default", FractalCommand::PaletteType1, RG::PaletteType),

             Sep(),

             Radio(L"Patriotic", FractalCommand::PaletteType2, RG::PaletteType),
             Radio(L"Summer", FractalCommand::PaletteType3, RG::PaletteType),
             Radio(L"Random", FractalCommand::PaletteType4, RG::PaletteType),

             Sep(),

             Item(L"&Create Random Palette", FractalCommand::CreateNewPalette),

             Sep(),

             Radio(L"5-bit", FractalCommand::Palette5, RG::PaletteBitDepth),
             Radio(L"6-bit", FractalCommand::Palette6, RG::PaletteBitDepth),
             Radio(L"8-bit", FractalCommand::Palette8, RG::PaletteBitDepth),
             Radio(L"12-bit", FractalCommand::Palette12, RG::PaletteBitDepth),
             Radio(L"16-bit", FractalCommand::Palette16, RG::PaletteBitDepth),
             Radio(L"20-bit", FractalCommand::Palette20, RG::PaletteBitDepth),

             Sep(),

             Item(L"Pa&lette Rotation", FractalCommand::PaletteRotate)),

    Sep(),

    FS_POPUP0(L"Memory Management",
              Radio(L"Enable Auto-Save Orbit (delete when done, default)",
                    FractalCommand::PerturbAutosaveOnDelete,
                    RG::MemoryAutosave),
              Radio(L"Enable Auto-Save Orbit (keep files)", FractalCommand::PerturbAutosaveOn, RG::MemoryAutosave),
              Radio(L"Disable Auto-Save Orbit", FractalCommand::PerturbAutosaveOff, RG::MemoryAutosave),

              Sep(),

              Radio(L"Remove Memory Limits", FractalCommand::MemoryLimit0, RG::MemoryLimit),
              Radio(L"Leave max of (1/2*ram, 8GB) free (default)", FractalCommand::MemoryLimit1, RG::MemoryLimit)),

    Item(L"Show Rendering Details", FractalCommand::CurPos),

    FS_POPUP0(L"Save",
              Item(L"Save Location...", FractalCommand::SaveLocation),
              Item(L"Save High Res Bitmap", FractalCommand::SaveHiResBmp),
              Item(L"Save Iterations as Text", FractalCommand::SaveItersText),
              Item(L"Save Bitmap Ima&ge", FractalCommand::SaveBmp),
              Sep(),
              Item(L"Save Reference Orbit as Text", FractalCommand::SaveRefOrbitText),
              Item(L"Save Compressed Orbit as Text (simple)", FractalCommand::SaveRefOrbitTextSimple),
              Item(L"Save Compressed Orbit as Text (max)", FractalCommand::SaveRefOrbitTextMax),
              Item(L"Save Compressed Orbit (Imagina/max)", FractalCommand::SaveRefOrbitImagMax),
              Sep(),
              Item(L"Benchmark (5x, full recalc)", FractalCommand::BenchmarkFull),
              Item(L"Benchmark (5x, intermediate only)", FractalCommand::BenchmarkInt),
              Sep(),
              Item(L"Diff Imagina Orbits (choose two)", FractalCommand::DiffRefOrbitImagMax)),

    FS_POPUP0(L"Load",
              Item(L"Load Location...", FractalCommand::LoadLocation),
              Item(L"Enter Location", FractalCommand::LoadEnterLocation),
              Sep(),
              Item(L"Load Imagina Orbit (Match)...", FractalCommand::LoadRefOrbitImagMax),
              Item(L"Load Imagina Orbit (Use Saved)...", FractalCommand::LoadRefOrbitImagMaxSaved)),

    Sep(),
    Item(L"E&xit", FractalCommand::Exit),
};

#undef FS_POPUP0
#undef FS_POPUP
