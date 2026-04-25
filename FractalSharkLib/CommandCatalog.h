//
// CommandCatalog.h - portable command catalog.
//
// Defines FractalCommand: a strongly-typed enum that mirrors the legacy
// IDM_* values 1:1. Every menu-driven and hotkey-driven action in the
// application has exactly one entry here, with the underlying value equal
// to its IDM_* numeric ID. That makes the catalog forward- and backward-
// compatible with the existing Win32 WM_COMMAND switch: the int dispatched
// by Windows is bit-for-bit the same as static_cast<uint32_t>(cmd).
//
// HotKey describes a single keyboard shortcut (key + modifiers). The
// kCommands array is the single source of truth that the GUIs walk to:
//   * decorate menu items with hotkey hints (e.g. "Zoom In Here\tZ"),
//   * dispatch keypresses to the right FractalCommand,
//   * drive the help-modal listing.
//
// The Win32 GUI initially keeps its existing WM_COMMAND switch; ExecuteCommand
// is defined per-platform and is allowed to forward via SendMessage during
// the migration. The Linux GUI calls ExecuteCommand directly.
//

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace FractalShark {

// Mirrors IDM_* from FractalSharkGUILib/AlgCmds.h. Values are exactly the
// IDM_* numeric IDs so existing Win32 dispatch paths remain unaffected.
enum class FractalCommand : uint32_t {
    None = 0,

    // ---- General / Help / Window ----
    ShowHotkeys = 40000,         // IDM_SHOWHOTKEYS
    ViewsHelp = 40001,           // IDM_VIEWS_HELP
    HelpAlg = 40002,             // IDM_HELP_ALG

    SquareView = 40010,          // IDM_SQUAREVIEW
    Repainting = 40011,          // IDM_REPAINTING
    Windowed = 40012,            // IDM_WINDOWED
    WindowedSq = 40013,          // IDM_WINDOWED_SQ
    Minimize = 40014,            // IDM_MINIMIZE
    CurPos = 40015,              // IDM_CURPOS

    Exit = 40020,                // IDM_EXIT

    // ---- Navigate ----
    Back = 40100,                // IDM_BACK
    CenterView = 40101,          // IDM_CENTERVIEW
    ZoomIn = 40102,              // IDM_ZOOMIN
    ZoomOut = 40103,             // IDM_ZOOMOUT
    AutoZoomDefault = 40104,     // IDM_AUTOZOOM_DEFAULT
    AutoZoomMax = 40105,         // IDM_AUTOZOOM_MAX
    FeatureFinderDirect = 40106, // IDM_FEATUREFINDER_DIRECT
    FeatureFinderPt = 40107,     // IDM_FEATUREFINDER_PT
    FeatureFinderLa = 40108,     // IDM_FEATUREFINDER_LA
    FeatureFinderDirectScan = 40109, // IDM_FEATUREFINDER_DIRECTSCAN
    FeatureFinderPtScan = 40110, // IDM_FEATUREFINDER_PTSCAN
    FeatureFinderLaScan = 40111, // IDM_FEATUREFINDER_LASCAN
    FeatureFinderZoom = 40112,   // IDM_FEATUREFINDER_ZOOM
    FeatureFinderClear = 40113,  // IDM_FEATUREFINDER_CLEAR
    AutoZoomFilament = 40114,    // IDM_AUTOZOOM_FILAMENT
    FeatureFinderResume = 40115, // IDM_FEATUREFINDER_RESUME
    NrInnerLoopGpu = 40116,      // IDM_NR_INNERLOOP_GPU
    NrInnerLoopCpu = 40117,      // IDM_NR_INNERLOOP_CPU
    NrInnerLoopCpuSt = 40118,    // IDM_NR_INNERLOOP_CPUST

    // ---- Built-In Views ----
    StandardView = 40200,        // IDM_STANDARDVIEW
    View1 = 40201,
    View2 = 40202,
    View3 = 40203,
    View4 = 40204,
    View5 = 40205,
    View6 = 40206,
    View7 = 40207,
    View8 = 40208,
    View9 = 40209,
    View10 = 40210,
    View11 = 40211,
    View12 = 40212,
    View13 = 40213,
    View14 = 40214,
    View15 = 40215,
    View16 = 40216,
    View17 = 40217,
    View18 = 40218,
    View19 = 40219,
    View20 = 40220,
    View21 = 40221,
    View22 = 40222,
    View23 = 40223,
    View24 = 40224,
    View25 = 40225,
    View26 = 40226,
    View27 = 40227,
    View28 = 40228,
    View29 = 40229,
    View30 = 40230,
    View31 = 40231,
    View32 = 40232,
    View33 = 40233,
    View34 = 40234,
    View35 = 40235,
    View36 = 40236,
    View37 = 40237,
    View38 = 40238,
    View39 = 40239,
    View40 = 40240,

    // ---- Antialiasing ----
    GpuAntialiasing1x = 40300,   // IDM_GPUANTIALIASING_1X
    GpuAntialiasing4x = 40301,   // IDM_GPUANTIALIASING_4X
    GpuAntialiasing9x = 40302,   // IDM_GPUANTIALIASING_9X
    GpuAntialiasing16x = 40303,  // IDM_GPUANTIALIASING_16X

    // ---- Iterations ----
    ResetIterations = 40400,         // IDM_RESETITERATIONS
    IncreaseIterations1p5x = 40401,  // IDM_INCREASEITERATIONS_1P5X
    IncreaseIterations6x = 40402,    // IDM_INCREASEITERATIONS_6X
    IncreaseIterations24x = 40403,   // IDM_INCREASEITERATIONS_24X
    DecreaseIterations = 40404,      // IDM_DECREASEITERATIONS
    Iterations32Bit = 40405,         // IDM_32BIT_ITERATIONS
    Iterations64Bit = 40406,         // IDM_64BIT_ITERATIONS

    // ---- Perturbation ----
    PerturbClearAll = 40500,
    PerturbClearMed = 40501,
    PerturbClearHigh = 40502,
    PerturbResults = 40503,

    PerturbationAuto = 40510,
    PerturbationSinglethread = 40511,
    PerturbationMultithread = 40512,
    PerturbationSinglethreadPeriodicity = 40513,
    PerturbationMultithread2Periodicity = 40514,
    PerturbationMultithread2PeriodicityPerturbMthighStmed = 40515,
    PerturbationMultithread2PeriodicityPerturbMthighMtmed1 = 40516,
    PerturbationMultithread2PeriodicityPerturbMthighMtmed2 = 40517,
    PerturbationMultithread2PeriodicityPerturbMthighMtmed3 = 40518,
    PerturbationMultithread2PeriodicityPerturbMthighMtmed4 = 40519,
    PerturbationMultithread5Periodicity = 40520,
    PerturbationGpu = 40521,

    PerturbationLoad = 40530,
    PerturbationSave = 40531,

    // ---- Palette ----
    PaletteType0 = 40600,
    PaletteType1 = 40601,
    PaletteType2 = 40602,
    PaletteType3 = 40603,
    PaletteType4 = 40604,

    CreateNewPalette = 40610,

    Palette5 = 40620,
    Palette6 = 40621,
    Palette8 = 40622,
    Palette12 = 40623,
    Palette16 = 40624,
    Palette20 = 40625,

    PaletteRotate = 40630,

    // ---- Memory Management ----
    PerturbAutosaveOnDelete = 40700,
    PerturbAutosaveOn = 40701,
    PerturbAutosaveOff = 40702,

    MemoryLimit0 = 40710,
    MemoryLimit1 = 40711,

    // ---- Save / Load ----
    SaveLocation = 40800,
    SaveHiResBmp = 40801,
    SaveItersText = 40802,
    SaveBmp = 40803,

    SaveRefOrbitText = 40810,
    SaveRefOrbitTextSimple = 40811,
    SaveRefOrbitTextMax = 40812,
    SaveRefOrbitImagMax = 40813,

    BenchmarkFull = 40820,
    BenchmarkInt = 40821,
    DiffRefOrbitImagMax = 40822,

    LoadLocation = 40830,
    LoadEnterLocation = 40831,
    LoadRefOrbitImagMax = 40832,
    LoadRefOrbitImagMaxSaved = 40833,

    // ---- Tests ----
    BasicTest = 40900,
    Test27 = 40901,

    // ---- Algorithm Selection ----
    AlgAuto = 41000,

    LaMultithreaded = 41010,
    LaSinglethreaded = 41011,
    LaSettings1 = 41012,
    LaSettings2 = 41013,
    LaSettings3 = 41014,

    AlgCpuHigh = 41100,
    AlgCpu1x64 = 41101,         // IDM_ALG_CPU_1_64
    AlgCpu1x32Hdr = 41102,      // IDM_ALG_CPU_1_32_HDR
    AlgCpu1x64Hdr = 41103,      // IDM_ALG_CPU_1_64_HDR

    AlgCpu1x64PerturbBla = 41110,        // IDM_ALG_CPU_1_64_PERTURB_BLA
    AlgCpu1x32PerturbBlaHdr = 41111,     // IDM_ALG_CPU_1_32_PERTURB_BLA_HDR
    AlgCpu1x64PerturbBlaHdr = 41112,     // IDM_ALG_CPU_1_64_PERTURB_BLA_HDR

    AlgCpu1x32PerturbBlav2Hdr = 41120,   // IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR
    AlgCpu1x64PerturbBlav2Hdr = 41121,   // IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR
    AlgCpu1x32PerturbRcBlav2Hdr = 41122, // IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR
    AlgCpu1x64PerturbRcBlav2Hdr = 41123, // IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR

    IterationPrecision4x = 41200,
    IterationPrecision3x = 41201,
    IterationPrecision2x = 41202,
    IterationPrecision1x = 41203,

    AlgGpu1x32 = 41210,         // IDM_ALG_GPU_1_32
    AlgGpu2x32 = 41211,         // IDM_ALG_GPU_2_32
    AlgGpu4x32 = 41212,         // IDM_ALG_GPU_4_32
    AlgGpu1x64 = 41213,         // IDM_ALG_GPU_1_64
    AlgGpu2x64 = 41214,         // IDM_ALG_GPU_2_64
    AlgGpu4x64 = 41215,         // IDM_ALG_GPU_4_64
    AlgGpu2x32Hdr = 41216,      // IDM_ALG_GPU_2X32_HDR

    AlgGpu1x32PerturbScaled = 41300,
    AlgGpu2x32PerturbScaled = 41301,
    AlgGpuHdr32PerturbScaled = 41302,

    AlgGpu1x64PerturbBla = 41310,
    AlgGpuHdr32PerturbBla = 41311,
    AlgGpuHdr64PerturbBla = 41312,

    AlgGpu1x32PerturbLav2Lao = 41320,
    AlgGpu2x32PerturbLav2Lao = 41321,
    AlgGpu1x64PerturbLav2Lao = 41322,
    AlgGpuHdr32PerturbLav2Lao = 41323,
    AlgGpuHdr2x32PerturbLav2Lao = 41324,
    AlgGpuHdr64PerturbLav2Lao = 41325,

    AlgGpu1x32PerturbLav2Po = 41330,
    AlgGpu2x32PerturbLav2Po = 41331,
    AlgGpu1x64PerturbLav2Po = 41332,
    AlgGpuHdr32PerturbLav2Po = 41333,
    AlgGpuHdr2x32PerturbLav2Po = 41334,
    AlgGpuHdr64PerturbLav2Po = 41335,

    AlgGpu1x32PerturbRcLav2 = 41340,
    AlgGpu2x32PerturbRcLav2 = 41341,
    AlgGpu1x64PerturbRcLav2 = 41342,
    AlgGpuHdr32PerturbRcLav2 = 41343,
    AlgGpuHdr2x32PerturbRcLav2 = 41344,
    AlgGpuHdr64PerturbRcLav2 = 41345,

    AlgGpu1x32PerturbRcLav2Po = 41350,
    AlgGpu2x32PerturbRcLav2Po = 41351,
    AlgGpu1x64PerturbRcLav2Po = 41352,
    AlgGpuHdr32PerturbRcLav2Po = 41353,
    AlgGpuHdr2x32PerturbRcLav2Po = 41354,
    AlgGpuHdr64PerturbRcLav2Po = 41355,

    AlgGpu1x32PerturbRcLav2Lao = 41360,
    AlgGpu2x32PerturbRcLav2Lao = 41361,
    AlgGpu1x64PerturbRcLav2Lao = 41362,
    AlgGpuHdr32PerturbRcLav2Lao = 41363,
    AlgGpuHdr2x32PerturbRcLav2Lao = 41364,
    AlgGpuHdr64PerturbRcLav2Lao = 41365,

    AlgGpu1x32PerturbLav2 = 41400,
    AlgGpu2x32PerturbLav2 = 41401,
    AlgGpu1x64PerturbLav2 = 41402,
    AlgGpuHdr32PerturbLav2 = 41403,
    AlgGpuHdr2x32PerturbLav2 = 41404,
    AlgGpuHdr64PerturbLav2 = 41405,
};

// Convert FractalCommand to its underlying IDM_* numeric value.
constexpr uint32_t
IdmFromCommand(FractalCommand c) noexcept
{
    return static_cast<uint32_t>(c);
}

// Convert IDM_* numeric value back to FractalCommand. Caller asserts it is valid.
constexpr FractalCommand
CommandFromIdm(uint32_t idm) noexcept
{
    return static_cast<FractalCommand>(idm);
}

// A single keyboard shortcut. ASCII 'A'..'Z', '0'..'9' or printable punctuation
// in `key` (lower-case for letters; the dispatcher applies shift). Modifiers
// are explicit so the same key can map differently with shift held.
struct HotKey {
    wchar_t key = 0;     // 0 means "no hotkey"
    bool shift = false;
    bool ctrl = false;
    bool alt = false;

    constexpr bool HasKey() const noexcept { return key != 0; }
};

// Catalog row tying a command to an optional label and hotkey. Label is
// purely for the help/hotkey-listing UI; menu labels still come from
// MenuTreeDef.h so menu structure can group entries that share an id.
struct Command {
    FractalCommand id = FractalCommand::None;
    std::wstring_view label = {};
    HotKey hotkey = {};
};

// kCommands is populated incrementally as Phase 0c migrates each menu.
// It need not be exhaustive; absence here just means "no first-class hotkey
// metadata yet" and the GUI falls back to the legacy WM_CHAR/HandleKeyDown
// handler on Win32.
inline constexpr std::array<Command, 0> kCommands{};

// ---------------------------------------------------------------------------
// ExecuteCommand — per-platform host forwarder (Phase 0c).
// ---------------------------------------------------------------------------
//
// Every catalog FractalCommand maps to one On*() hook on the host.  The
// algorithm-selection family (~55 RenderAlgorithmEnum-driven commands) is
// funnelled through the single OnSetAlgorithm hook; the catalog→alg lookup
// happens in CommandCatalog.cpp via kAlgCmds.  Range-based commands (View
// 1..40, dynamic orbit/imag slots) fall through to DispatchByIdm so the
// legacy CommandDispatcher picks them up.

// Forward declaration so we don't drag RenderAlgorithm.h into this header.
// The actual definition in RenderAlgorithm.h fixes the underlying type to
// uint32_t to match.
} // namespace FractalShark

enum class RenderAlgorithmEnum : uint32_t;

namespace FractalShark {

struct ExecuteCommandHost {
    virtual ~ExecuteCommandHost() = default;

    // Catalog→legacy bridge for unmigrated / range-based commands.
    virtual void DispatchByIdm(int wmId) = 0;

    // ---- Algorithm selection ------------------------------------------
    virtual void OnSetAlgorithm(::RenderAlgorithmEnum alg) = 0;

    // ---- Help / Window ------------------------------------------------
    virtual void OnShowHotkeys() = 0;
    virtual void OnViewsHelp() = 0;
    virtual void OnHelpAlg() = 0;
    virtual void OnSquareView() = 0;
    virtual void OnRepainting() = 0;
    virtual void OnWindowed() = 0;
    virtual void OnWindowedSq() = 0;
    virtual void OnMinimize() = 0;
    virtual void OnCurPos() = 0;
    virtual void OnExit() = 0;

    // ---- Navigate -----------------------------------------------------
    virtual void OnBack() = 0;
    virtual void OnCenterView() = 0;
    virtual void OnZoomIn() = 0;
    virtual void OnZoomOut() = 0;
    virtual void OnAutoZoomDefault() = 0;
    virtual void OnAutoZoomMax() = 0;
    virtual void OnAutoZoomFilament() = 0;
    virtual void OnFeatureFinderDirect() = 0;
    virtual void OnFeatureFinderDirectScan() = 0;
    virtual void OnFeatureFinderPt() = 0;
    virtual void OnFeatureFinderPtScan() = 0;
    virtual void OnFeatureFinderLa() = 0;
    virtual void OnFeatureFinderLaScan() = 0;
    virtual void OnFeatureFinderZoom() = 0;
    virtual void OnFeatureFinderClear() = 0;
    virtual void OnFeatureFinderResume() = 0;
    virtual void OnNrInnerLoopGpu() = 0;
    virtual void OnNrInnerLoopCpu() = 0;
    virtual void OnNrInnerLoopCpuSt() = 0;

    // ---- Built-In Views (point entry; View1..40 are range-dispatched) -
    virtual void OnStandardView() = 0;

    // ---- Antialiasing -------------------------------------------------
    virtual void OnGpuAntialiasing1x() = 0;
    virtual void OnGpuAntialiasing4x() = 0;
    virtual void OnGpuAntialiasing9x() = 0;
    virtual void OnGpuAntialiasing16x() = 0;

    // ---- Iterations ---------------------------------------------------
    virtual void OnResetIterations() = 0;
    virtual void OnIncreaseIterations1p5x() = 0;
    virtual void OnIncreaseIterations6x() = 0;
    virtual void OnIncreaseIterations24x() = 0;
    virtual void OnDecreaseIterations() = 0;
    virtual void OnIterations32Bit() = 0;
    virtual void OnIterations64Bit() = 0;

    // ---- Iteration Precision -----------------------------------------
    virtual void OnIterationPrecision1x() = 0;
    virtual void OnIterationPrecision2x() = 0;
    virtual void OnIterationPrecision3x() = 0;
    virtual void OnIterationPrecision4x() = 0;

    // ---- Perturbation -------------------------------------------------
    virtual void OnPerturbResults() = 0;
    virtual void OnPerturbClearAll() = 0;
    virtual void OnPerturbClearMed() = 0;
    virtual void OnPerturbClearHigh() = 0;
    virtual void OnPerturbationAuto() = 0;
    virtual void OnPerturbationSinglethread() = 0;
    virtual void OnPerturbationMultithread() = 0;
    virtual void OnPerturbationSinglethreadPeriodicity() = 0;
    virtual void OnPerturbationMultithread2Periodicity() = 0;
    virtual void OnPerturbationMt2PerturbMthighStmed() = 0;
    virtual void OnPerturbationMt2PerturbMthighMtmed1() = 0;
    virtual void OnPerturbationMt2PerturbMthighMtmed2() = 0;
    virtual void OnPerturbationMt2PerturbMthighMtmed3() = 0;
    virtual void OnPerturbationMt2PerturbMthighMtmed4() = 0;
    virtual void OnPerturbationMultithread5Periodicity() = 0;
    virtual void OnPerturbationGpu() = 0;
    virtual void OnPerturbationLoad() = 0;
    virtual void OnPerturbationSave() = 0;

    // ---- Memory / Autosave -------------------------------------------
    virtual void OnPerturbAutosaveOnDelete() = 0;
    virtual void OnPerturbAutosaveOn() = 0;
    virtual void OnPerturbAutosaveOff() = 0;
    virtual void OnMemoryLimit0() = 0;
    virtual void OnMemoryLimit1() = 0;

    // ---- Palette ------------------------------------------------------
    virtual void OnPaletteType0() = 0;
    virtual void OnPaletteType1() = 0;
    virtual void OnPaletteType2() = 0;
    virtual void OnPaletteType3() = 0;
    virtual void OnPaletteType4() = 0;
    virtual void OnCreateNewPalette() = 0;
    virtual void OnPalette5() = 0;
    virtual void OnPalette6() = 0;
    virtual void OnPalette8() = 0;
    virtual void OnPalette12() = 0;
    virtual void OnPalette16() = 0;
    virtual void OnPalette20() = 0;
    virtual void OnPaletteRotate() = 0;

    // ---- Save / Load --------------------------------------------------
    virtual void OnSaveLocation() = 0;
    virtual void OnSaveHiResBmp() = 0;
    virtual void OnSaveItersText() = 0;
    virtual void OnSaveBmp() = 0;
    virtual void OnSaveRefOrbitText() = 0;
    virtual void OnSaveRefOrbitTextSimple() = 0;
    virtual void OnSaveRefOrbitTextMax() = 0;
    virtual void OnSaveRefOrbitImagMax() = 0;
    virtual void OnDiffRefOrbitImagMax() = 0;
    virtual void OnLoadLocation() = 0;
    virtual void OnLoadEnterLocation() = 0;
    virtual void OnLoadRefOrbitImagMax() = 0;
    virtual void OnLoadRefOrbitImagMaxSaved() = 0;

    // ---- Tests / Benchmarks ------------------------------------------
    virtual void OnBasicTest() = 0;
    virtual void OnTest27() = 0;
    virtual void OnBenchmarkFull() = 0;
    virtual void OnBenchmarkInt() = 0;

    // ---- LA ----------------------------------------------------------
    virtual void OnLaMultithreaded() = 0;
    virtual void OnLaSinglethreaded() = 0;
    virtual void OnLaSettings1() = 0;
    virtual void OnLaSettings2() = 0;
    virtual void OnLaSettings3() = 0;
};

// Defined in CommandCatalog.cpp — does the FractalCommand→host hook
// dispatch, including the alg-family lookup.
void ExecuteCommand(FractalCommand cmd, ExecuteCommandHost &host);

} // namespace FractalShark
