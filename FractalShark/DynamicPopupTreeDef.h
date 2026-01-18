//
// The menu tree below is written as a single static `Node menu[]` array.
//
// Important detail: popup children must have *stable storage*.
// Braced lists like `{ Item(...), ... }` produce temporary arrays, and
// `std::initializer_list` does not own that memory. If you store an
// initializer_list inside a Node and recurse later, you can end up iterating
// dangling memory.
//
// To keep the "giant definition" style while making kids stable, each Popup is
// created via a small wrapper (FS_POPUP / FS_POPUP0) that:
//   1) hoists the children into a hidden `static const Node _kids[]` array,
//   2) passes `std::span{_kids}` into Popup(...),
//   3) returns a normal Node that simply *views* those children.
//
// Result: `menu[]` stays readable and static, and every submenu’s children live
// for the entire program lifetime, so recursive menu building is safe.
//

#include <span>

using RG = DynamicPopupMenu::RadioGroup;
using R = DynamicPopupMenu::Rule;

#define FS_POPUP(label, enableRule, adornGroup, ...)                                                    \
    []() -> DynamicPopupMenu::Node {                                                                    \
        static const DynamicPopupMenu::Node _kids[] = {__VA_ARGS__};                                    \
        return DynamicPopupMenu::Popup(                                                                 \
            label, std::span<const DynamicPopupMenu::Node>{_kids}, enableRule, adornGroup);             \
    }()

#define FS_POPUP0(label, ...) FS_POPUP(label, R::Always, RG::None, __VA_ARGS__)

// -----------------------------------------------------------------------------
// Menu definition (structure preserved; only Popup(...) replaced)
// -----------------------------------------------------------------------------

static const DynamicPopupMenu::Node menu[] = {
    Item(L"Show Help", IDM_SHOWHOTKEYS),
    Sep(),

    FS_POPUP0(L"Navigate",
              Item(L"Back", IDM_BACK),
              Sep(),
              Item(L"Center View Here", IDM_CENTERVIEW),
              Item(L"Zoom In Here", IDM_ZOOMIN),
              Item(L"Zoom Out", IDM_ZOOMOUT),
              Sep(),
              Item(L"Autozoom Default", IDM_AUTOZOOM_DEFAULT),
              Item(L"Autozoom Max", IDM_AUTOZOOM_MAX)),

    FS_POPUP0(L"Built-In Views",
              Item(L"Help", IDM_VIEWS_HELP),
              Sep(),
              Item(L"&Standard View", IDM_STANDARDVIEW),
              Item(L"#1 - 4x64 GPU Limit", IDM_VIEW1),
              Item(L"#2 - 4x32 GPU Limit", IDM_VIEW2),
              Item(L"#3 - 32-bit BLA Limit", IDM_VIEW3),
              Item(L"#4 - 32-bit Scaled Limit", IDM_VIEW4),
              Item(L"#5 - 32-bit Scaled Limit, 4xAA", IDM_VIEW5),
              Item(L"#6 - Scale float with pixellation", IDM_VIEW6),
              Item(L"#7 - Scaled float limit with square", IDM_VIEW7),
              Item(L"#8 - ~10^700 or so, 0.5b iterations", IDM_VIEW8),
              Item(L"#9 - bug: scaled/BLA/32-bit failure", IDM_VIEW9),
              Item(L"#10 - ~10^130, ~2b iterations", IDM_VIEW10),
              Item(L"#11 - ~10^700 or so, ~3m iterations", IDM_VIEW11),
              Item(L"#12 - Nearby #14", IDM_VIEW12),
              Item(L"#13 - ~10^4000, ~100m iterations, 16xAA", IDM_VIEW13),
              Item(L"#14 - ~10^6000, ~2b iterations, 16xAA", IDM_VIEW14),
              Item(L"#15 - ~10^140, high period", IDM_VIEW15),
              Item(L"#16 - LAv2 test spot 1 (quick)", IDM_VIEW16),
              Item(L"#17 - LAv2 test spot 2 (quick)", IDM_VIEW17),
              Item(L"#18 - #5 with point + magnification", IDM_VIEW18),
              Item(L"#19 - LAv2 test spot 3 (quick)", IDM_VIEW19),
              Item(L"#20 - ~10 minutes ref orbit, 50x10^9 iters", IDM_VIEW20),
              Item(L"#21 - ~10^11000, 2b iterations", IDM_VIEW21),
              Item(L"#22 - high period", IDM_VIEW22),
              Item(L"#23 - high period, 50b iterations", IDM_VIEW23),
              Item(L"#24", IDM_VIEW24),
              Item(L"#25 - Claude - low period, 32-bit, hard GPU", IDM_VIEW25),
              Item(L"#26 - low period, 1x32 busted, 1x64 OK, hard GPU", IDM_VIEW26),
              Item(L"#27", IDM_VIEW27),
              Item(L"#28", IDM_VIEW28),
              Item(L"#29 - Fast HDRx32 broke, HDRx64 works", IDM_VIEW29),
              Item(L"#30 - 1e100000 - low period", IDM_VIEW30)),

    Item(L"Recalculate, Reuse Reference", IDM_SQUAREVIEW),
    Sep(),

    // These read their check state from IMenuState::IsChecked(commandId)
    Toggle(L"Toggle Repainting", IDM_REPAINTING),
    Toggle(L"Toggle Window Size", IDM_WINDOWED),
    Toggle(L"Toggle Window Size (Square)", IDM_WINDOWED_SQ),

    Item(L"Minimize Window", IDM_MINIMIZE),
    Sep(),

    FS_POPUP(L"GPU Antialiasing",
             R::Always,
             /*adornment*/ RG::GpuAntialiasing,
             Radio(L"1x (fast)", IDM_GPUANTIALIASING_1X, RG::GpuAntialiasing),
             Radio(L"4x", IDM_GPUANTIALIASING_4X, RG::GpuAntialiasing),
             Radio(L"9x", IDM_GPUANTIALIASING_9X, RG::GpuAntialiasing),
             Radio(L"16x (better quality)", IDM_GPUANTIALIASING_16X, RG::GpuAntialiasing)),

    FS_POPUP(
        L"Choose Render Algorithm",
        R::Always,
        /*adornment*/ RG::RenderAlgorithm,

        Item(L"Help", IDM_HELP_ALG),

        // All algorithms are one big mutually-exclusive group
        Radio(L"Auto (Default)", IDM_ALG_AUTO, RG::RenderAlgorithm),

        FS_POPUP0(L"LA Parameters",
                  Radio(L"Multithreaded (Default)", IDM_LA_MULTITHREADED, RG::LaThreading),
                  Radio(L"Single Threaded", IDM_LA_SINGLETHREADED, RG::LaThreading),
                  Sep(),
                  Item(L"Max Accuracy (Default)", IDM_LA_SETTINGS_1),
                  Item(L"Max Performance (Accuracy Loss)", IDM_LA_SETTINGS_2),
                  Item(L"Min Memory", IDM_LA_SETTINGS_3)),

        Sep(),

        FS_POPUP0(
            L"CPU-Only",
            Radio(L"Very High Precision CPU", IDM_ALG_CPU_HIGH, RG::RenderAlgorithm),
            Sep(),
            Radio(L"1x64 CPU", IDM_ALG_CPU_1_64, RG::RenderAlgorithm),
            Radio(L"HDRx32 CPU", IDM_ALG_CPU_1_32_HDR, RG::RenderAlgorithm),
            Radio(L"HDRx64 CPU", IDM_ALG_CPU_1_64_HDR, RG::RenderAlgorithm),
            Sep(),
            Radio(L"1x64 CPU - Perturbation BLA", IDM_ALG_CPU_1_64_PERTURB_BLA, RG::RenderAlgorithm),
            Radio(
                L"HDRx32 CPU - Perturbation BLA", IDM_ALG_CPU_1_32_PERTURB_BLA_HDR, RG::RenderAlgorithm),
            Radio(
                L"HDRx64 CPU - Perturbation BLA", IDM_ALG_CPU_1_64_PERTURB_BLA_HDR, RG::RenderAlgorithm),
            Sep(),
            Radio(L"HDRx32 CPU - Perturbation LAv2",
                  IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR,
                  RG::RenderAlgorithm),
            Radio(L"HDRx64 CPU - Perturbation LAv2",
                  IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR,
                  RG::RenderAlgorithm),
            Radio(L"HDRx32 RC CPU - Perturbation LAv2",
                  IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR,
                  RG::RenderAlgorithm),
            Radio(L"HDRx64 RC CPU - Perturbation LAv2",
                  IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR,
                  RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Low-Zoom Depth",

            FS_POPUP(L"Iteration Precision",
                     R::Always,
                     /*adornment*/ RG::IterationPrecision,
                     Radio(L"4x (fast)", IDM_ITERATIONPRECISION_4X, RG::IterationPrecision),
                     Radio(L"3x", IDM_ITERATIONPRECISION_3X, RG::IterationPrecision),
                     Radio(L"2x", IDM_ITERATIONPRECISION_2X, RG::IterationPrecision),
                     Radio(L"1x (better quality)", IDM_ITERATIONPRECISION_1X, RG::IterationPrecision)),

            Sep(),

            Radio(L"1x32 GPU", IDM_ALG_GPU_1_32, RG::RenderAlgorithm),
            Radio(L"2x32 GPU", IDM_ALG_GPU_2_32, RG::RenderAlgorithm),
            Radio(L"4x32 GPU", IDM_ALG_GPU_4_32, RG::RenderAlgorithm),
            Radio(L"1x64 GPU", IDM_ALG_GPU_1_64, RG::RenderAlgorithm),
            Radio(L"2x64 GPU", IDM_ALG_GPU_2_64, RG::RenderAlgorithm),
            Radio(L"4x64 GPU", IDM_ALG_GPU_4_64, RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU", IDM_ALG_GPU_2X32_HDR, RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Scaled",
            Radio(
                L"1x32 GPU - Perturbation Scaled", IDM_ALG_GPU_1_32_PERTURB_SCALED, RG::RenderAlgorithm),
            Radio(L"2x32 GPU - Perturbation Scaled (broken)",
                  IDM_ALG_GPU_2_32_PERTURB_SCALED,
                  RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU - Perturbation Scaled",
                  IDM_ALG_GPU_HDR_32_PERTURB_SCALED,
                  RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Bilinear Approximation V1",
            Radio(L"1x64 GPU - Perturbation BLA", IDM_ALG_GPU_1_64_PERTURB_BLA, RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU - Perturbation BLA", IDM_ALG_GPU_HDR_32_PERTURB_BLA, RG::RenderAlgorithm),
            Radio(
                L"HDRx64 GPU - Perturbation BLA", IDM_ALG_GPU_HDR_64_PERTURB_BLA, RG::RenderAlgorithm)),

        FS_POPUP0(
            L"LA Only (for testing)",
            Radio(L"1x32 GPU - LAv2 - LA only", IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO, RG::RenderAlgorithm),
            Radio(L"2x32 GPU - LAv2 - LA only", IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO, RG::RenderAlgorithm),
            Radio(L"1x64 GPU - LAv2 - LA only", IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO, RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU - LAv2 - LA only",
                  IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO,
                  RG::RenderAlgorithm),
            Radio(L"HDRx2x32 GPU - LAv2 - LA only",
                  IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO,
                  RG::RenderAlgorithm),
            Radio(L"HDRx64 GPU - LAv2 - LA only",
                  IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO,
                  RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Perturbation Only",
            Radio(L"1x32 GPU - Perturb only", IDM_ALG_GPU_1_32_PERTURB_LAV2_PO, RG::RenderAlgorithm),
            Radio(L"2x32 GPU - Perturb only", IDM_ALG_GPU_2_32_PERTURB_LAV2_PO, RG::RenderAlgorithm),
            Radio(L"1x64 GPU - Perturb only", IDM_ALG_GPU_1_64_PERTURB_LAV2_PO, RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU - Perturb only", IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO, RG::RenderAlgorithm),
            Radio(L"HDRx2x32 GPU - Perturb only",
                  IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO,
                  RG::RenderAlgorithm),
            Radio(
                L"HDRx64 GPU - Perturb only", IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO, RG::RenderAlgorithm)),

        FS_POPUP0(
            L"Reference Compression",

            Radio(L"1x32 GPU - RC LAv2", IDM_ALG_GPU_1_32_PERTURB_RC_LAV2, RG::RenderAlgorithm),
            Radio(L"2x32 GPU - RC LAv2", IDM_ALG_GPU_2_32_PERTURB_RC_LAV2, RG::RenderAlgorithm),
            Radio(L"1x64 GPU - RC LAv2", IDM_ALG_GPU_1_64_PERTURB_RC_LAV2, RG::RenderAlgorithm),
            Radio(L"HDRx32 GPU - RC LAv2", IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2, RG::RenderAlgorithm),
            Radio(L"HDRx2x32 GPU - RC LAv2", IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2, RG::RenderAlgorithm),
            Radio(L"HDRx64 GPU - RC LAv2", IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2, RG::RenderAlgorithm),

            FS_POPUP0(L"Perturbation Only",
                      Radio(L"1x32 GPU - RC Perturb Only",
                            IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm),
                      Radio(L"2x32 GPU - RC Perturb Only",
                            IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm),
                      Radio(L"1x64 GPU - RC Perturb Only",
                            IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm),
                      Radio(L"HDRx32 GPU - RC Perturb Only",
                            IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm),
                      Radio(L"HDRx2x32 GPU - RC Perturb Only",
                            IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm),
                      Radio(L"HDRx64 GPU - RC Perturb Only",
                            IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_PO,
                            RG::RenderAlgorithm)),

            FS_POPUP0(
                L"LA Only",
                Radio(L"1x32 GPU - RC LAv2", IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_LAO, RG::RenderAlgorithm),
                Radio(L"2x32 GPU - RC LAv2", IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_LAO, RG::RenderAlgorithm),
                Radio(L"1x64 GPU - RC LAv2", IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_LAO, RG::RenderAlgorithm),
                Radio(L"HDRx32 GPU - RC LAv2",
                      IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_LAO,
                      RG::RenderAlgorithm),
                Radio(L"HDRx2x32 GPU - RC LAv2",
                      IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_LAO,
                      RG::RenderAlgorithm),
                Radio(L"HDRx64 GPU - RC LAv2",
                      IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_LAO,
                      RG::RenderAlgorithm))),

        Sep(),

        Radio(L"1x32 GPU - LAv2", IDM_ALG_GPU_1_32_PERTURB_LAV2, RG::RenderAlgorithm),
        Radio(L"2x32 GPU - LAv2", IDM_ALG_GPU_2_32_PERTURB_LAV2, RG::RenderAlgorithm),
        Radio(L"1x64 GPU - LAv2", IDM_ALG_GPU_1_64_PERTURB_LAV2, RG::RenderAlgorithm),
        Radio(L"HDRx32 GPU - LAv2", IDM_ALG_GPU_HDR_32_PERTURB_LAV2, RG::RenderAlgorithm),
        Radio(L"HDRx2x32 GPU - LAv2", IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2, RG::RenderAlgorithm),
        Radio(L"HDRx64 GPU - LAv2", IDM_ALG_GPU_HDR_64_PERTURB_LAV2, RG::RenderAlgorithm),

        Sep(),

        FS_POPUP0(L"Tests",
                  Item(L"Run Basic Test (saves files in local dir)", IDM_BASICTEST),
                  Item(L"Run View #27 test", IDM_TEST_27))),

    FS_POPUP0(L"Iterations",
              Item(L"De&fault Iterations", IDM_RESETITERATIONS),
              Item(L"+1.5x", IDM_INCREASEITERATIONS_1P5X),
              Item(L"+6x", IDM_INCREASEITERATIONS_6X),
              Item(L"+24x", IDM_INCREASEITERATIONS_24X),
              Item(L"&Decrease Iterations", IDM_DECREASEITERATIONS),
              Sep(),
              Radio(L"32-Bit Iterations", IDM_32BIT_ITERATIONS, RG::IterationsWidth),
              Radio(L"64-Bit Iterations", IDM_64BIT_ITERATIONS, RG::IterationsWidth)),

    FS_POPUP(
        L"Perturbation",
        R::Always,
        /*adornment*/ RG::PerturbationMode,

        Item(L"Clear Perturbation References - All", IDM_PERTURB_CLEAR_ALL),
        Item(L"Clear Perturbation References - Med", IDM_PERTURB_CLEAR_MED),
        Item(L"Clear Perturbation References - High", IDM_PERTURB_CLEAR_HIGH),
        Item(L"Show Perturbation Results", IDM_PERTURB_RESULTS),

        Sep(),

        Radio(L"Auto (default)", IDM_PERTURBATION_AUTO, RG::PerturbationMode),
        Radio(L"Single Thread (ST)", IDM_PERTURBATION_SINGLETHREAD, RG::PerturbationMode),
        Radio(L"Multi Thread (MT)", IDM_PERTURBATION_MULTITHREAD, RG::PerturbationMode),
        Radio(L"ST + Periodicity", IDM_PERTURBATION_SINGLETHREAD_PERIODICITY, RG::PerturbationMode),
        Radio(L"MT2 + Periodicity", IDM_PERTURBATION_MULTITHREAD2_PERIODICITY, RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb ST",
              IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v1",
              IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v2",
              IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v3",
              IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3,
              RG::PerturbationMode),
        Radio(L"MT2 + Periodicity + Perturb MT v4 (broken)",
              IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4,
              RG::PerturbationMode),

        // Formerly Inactive(). Replace with a rule your IMenuState returns false for.
        Item(L"MT5 + Periodicity",
             IDM_PERTURBATION_MULTITHREAD5_PERIODICITY,
             R::EnableIfPerturbationAvailable),

        Sep(),

        // If this is also a mode selection you can convert it to Radio() in RG::PerturbationMode.
        Item(L"GPU-Accelerated (see README)", IDM_PERTURBATION_GPU),

        Sep(),
        Item(L"Clear and Reload Reference Orbits", IDM_PERTURBATION_LOAD),
        Item(L"Save Reference Orbits", IDM_PERTURBATION_SAVE)),

    Sep(),

    FS_POPUP(L"Palette Color Depth",
             R::Always,
             /*adornment*/ RG::PaletteBitDepth,

             // Formerly Inactive(). Replace with a rule your IMenuState returns false for.
             Radio(L"Basic", IDM_PALETTE_TYPE_0, RG::PaletteType, R::EnableIfPaletteRotationSupported),
             Radio(L"Default", IDM_PALETTE_TYPE_1, RG::PaletteType),

             Sep(),

             Radio(L"Patriotic", IDM_PALETTE_TYPE_2, RG::PaletteType),
             Radio(L"Summer", IDM_PALETTE_TYPE_3, RG::PaletteType),
             Radio(L"Random", IDM_PALETTE_TYPE_4, RG::PaletteType),

             Sep(),

             Item(L"&Create Random Palette", IDM_CREATENEWPALETTE),

             Sep(),

             Radio(L"5-bit", IDM_PALETTE_5, RG::PaletteBitDepth),
             Radio(L"6-bit", IDM_PALETTE_6, RG::PaletteBitDepth),
             Radio(L"8-bit", IDM_PALETTE_8, RG::PaletteBitDepth),
             Radio(L"12-bit", IDM_PALETTE_12, RG::PaletteBitDepth),
             Radio(L"16-bit", IDM_PALETTE_16, RG::PaletteBitDepth),
             Radio(L"20-bit", IDM_PALETTE_20, RG::PaletteBitDepth),

             Sep(),

             // Formerly Inactive(). Probably really a Toggle; leaving as Item with a rule for now.
             // If it IS a toggle, change to Toggle(...).
             Item(L"Pa&lette Rotation", IDM_PALETTEROTATE, R::EnableIfPaletteRotationSupported)),

    Sep(),

    FS_POPUP0(L"Memory Management",
              Radio(L"Enable Auto-Save Orbit (delete when done, default)",
                    IDM_PERTURB_AUTOSAVE_ON_DELETE,
                    RG::MemoryAutosave),
              Radio(L"Enable Auto-Save Orbit (keep files)", IDM_PERTURB_AUTOSAVE_ON, RG::MemoryAutosave),
              Radio(L"Disable Auto-Save Orbit", IDM_PERTURB_AUTOSAVE_OFF, RG::MemoryAutosave),

              Sep(),

              Radio(L"Remove Memory Limits", IDM_MEMORY_LIMIT_0, RG::MemoryLimit),
              Radio(L"Leave max of (1/2*ram, 8GB) free (default)", IDM_MEMORY_LIMIT_1, RG::MemoryLimit)),

    Item(L"Show Rendering Details", IDM_CURPOS),

    FS_POPUP0(L"Save",
              Item(L"Save Location...", IDM_SAVELOCATION),
              Item(L"Save High Res Bitmap", IDM_SAVEHIRESBMP),
              Item(L"Save Iterations as Text", IDM_SAVE_ITERS_TEXT),
              Item(L"Save Bitmap Ima&ge", IDM_SAVEBMP),
              Sep(),
              Item(L"Save Reference Orbit as Text", IDM_SAVE_REFORBIT_TEXT),
              Item(L"Save Compressed Orbit as Text (simple)", IDM_SAVE_REFORBIT_TEXT_SIMPLE),
              Item(L"Save Compressed Orbit as Text (max)", IDM_SAVE_REFORBIT_TEXT_MAX),
              Item(L"Save Compressed Orbit (Imagina/max)", IDM_SAVE_REFORBIT_IMAG_MAX),
              Sep(),
              Item(L"Benchmark (5x, full recalc)", IDM_BENCHMARK_FULL),
              Item(L"Benchmark (5x, intermediate only)", IDM_BENCHMARK_INT),
              Sep(),
              Item(L"Diff Imagina Orbits (choose two)", IDM_DIFF_REFORBIT_IMAG_MAX)),

    FS_POPUP0(L"Load",
              Item(L"Load Location...", IDM_LOADLOCATION),
              Item(L"Enter Location", IDM_LOAD_ENTERLOCATION),
              Sep(),
              Item(L"Load Imagina Orbit (Match)...", IDM_LOAD_REFORBIT_IMAG_MAX),
              Item(L"Load Imagina Orbit (Use Saved)...", IDM_LOAD_REFORBIT_IMAG_MAX_SAVED)),

    Sep(),
    Item(L"E&xit", IDM_EXIT),
};

#undef FS_POPUP0
#undef FS_POPUP
