//
// CommandCatalog.h - portable command catalog.
//
// Defines FractalCommand: a strongly-typed enum whose native menu-command
// entries intentionally mirror IDM_* values 1:1. Every menu-driven and
// hotkey-driven action in the application has exactly one entry here, with
// native menu-command entries keeping their original numeric IDs. That makes
// the catalog forward- and backward-compatible with the existing Win32
// WM_COMMAND path: the int dispatched by Windows is bit-for-bit the same as
// static_cast<uint32_t>(cmd).
//
// HotKey describes a single keyboard shortcut (key + modifiers). The
// kCommands array is the single source of truth that the GUIs walk to:
//   * decorate menu items with hotkey hints (e.g. "Zoom In Here\tZ"),
//   * dispatch keypresses to the right FractalCommand,
//   * drive the help-modal listing.
//
// Win32 and Linux route catalog commands through ExecuteCommand. GUI-specific
// input code is responsible only for translating native events into HotKey or
// FractalCommand values.
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

// X11 (Xlib.h / X.h) defines several short identifiers as preprocessor
// macros (None, Always, Above, Below, etc.) that collide with our enum
// member names below.  Undefine the ones that clash here so any TU that
// includes both headers (in either order) compiles.  Code that genuinely
// needs the X11 constants can spell them as their integer literals or
// re-include <X11/X.h> *after* this header.
#ifdef None
#undef None
#endif
#ifdef Always
#undef Always
#endif
#ifdef Above
#undef Above
#endif
#ifdef Below
#undef Below
#endif

namespace FractalShark {

// Commands in the native menu-command range mirror IDM_* values from
// FractalSharkGuiWin32/AlgCmds.h. Their numeric compatibility is intentional:
// a Win32 WM_COMMAND id can be cast to FractalCommand, and Linux can reuse the
// same catalog ids without carrying Win32 menu resources. Synthetic shortcut
// commands live outside that native range when a shortcut variant needs its own
// catalog identity.
enum class FractalCommand : uint32_t {
    None = 0,

    // ---- General / Help / Window ----
    ShowHotkeys = 40000, // IDM_SHOWHOTKEYS
    ViewsHelp = 40001,   // IDM_VIEWS_HELP
    HelpAlg = 40002,     // IDM_HELP_ALG

    SquareView = 40010, // IDM_SQUAREVIEW
    Repainting = 40011, // IDM_REPAINTING
    Windowed = 40012,   // IDM_WINDOWED
    WindowedSq = 40013, // IDM_WINDOWED_SQ
    Minimize = 40014,   // IDM_MINIMIZE
    CurPos = 40015,     // IDM_CURPOS

    Exit = 40020, // IDM_EXIT

    // ---- Navigate ----
    Back = 40100,                    // IDM_BACK
    CenterView = 40101,              // IDM_CENTERVIEW
    ZoomIn = 40102,                  // IDM_ZOOMIN
    ZoomOut = 40103,                 // IDM_ZOOMOUT
    AutoZoomDefault = 40104,         // IDM_AUTOZOOM_DEFAULT
    AutoZoomMax = 40105,             // IDM_AUTOZOOM_MAX
    FeatureFinderDirect = 40106,     // IDM_FEATUREFINDER_DIRECT
    FeatureFinderPt = 40107,         // IDM_FEATUREFINDER_PT
    FeatureFinderLa = 40108,         // IDM_FEATUREFINDER_LA
    FeatureFinderDirectScan = 40109, // IDM_FEATUREFINDER_DIRECTSCAN
    FeatureFinderPtScan = 40110,     // IDM_FEATUREFINDER_PTSCAN
    FeatureFinderLaScan = 40111,     // IDM_FEATUREFINDER_LASCAN
    FeatureFinderZoom = 40112,       // IDM_FEATUREFINDER_ZOOM
    FeatureFinderClear = 40113,      // IDM_FEATUREFINDER_CLEAR
    AutoZoomFilament = 40114,        // IDM_AUTOZOOM_FILAMENT
    FeatureFinderResume = 40115,     // IDM_FEATUREFINDER_RESUME
    NrInnerLoopGpu = 40116,          // IDM_NR_INNERLOOP_GPU
    NrInnerLoopCpu = 40117,          // IDM_NR_INNERLOOP_CPU
    NrInnerLoopCpuSt = 40118,        // IDM_NR_INNERLOOP_CPUST

    // ---- Built-In Views ----
    StandardView = 40200, // IDM_STANDARDVIEW
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
    GpuAntialiasing1x = 40300,  // IDM_GPUANTIALIASING_1X
    GpuAntialiasing4x = 40301,  // IDM_GPUANTIALIASING_4X
    GpuAntialiasing9x = 40302,  // IDM_GPUANTIALIASING_9X
    GpuAntialiasing16x = 40303, // IDM_GPUANTIALIASING_16X

    // ---- Iterations ----
    ResetIterations = 40400,        // IDM_RESETITERATIONS
    IncreaseIterations1p5x = 40401, // IDM_INCREASEITERATIONS_1P5X
    IncreaseIterations6x = 40402,   // IDM_INCREASEITERATIONS_6X
    IncreaseIterations24x = 40403,  // IDM_INCREASEITERATIONS_24X
    DecreaseIterations = 40404,     // IDM_DECREASEITERATIONS
    Iterations32Bit = 40405,        // IDM_32BIT_ITERATIONS
    Iterations64Bit = 40406,        // IDM_64BIT_ITERATIONS

    // ---- Perturbation ----
    PerturbClearAll = 40500,
    PerturbClearMed = 40501,
    PerturbClearHigh = 40502,

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
    AlgCpu1x64 = 41101,    // IDM_ALG_CPU_1_64
    AlgCpu1x32Hdr = 41102, // IDM_ALG_CPU_1_32_HDR
    AlgCpu1x64Hdr = 41103, // IDM_ALG_CPU_1_64_HDR

    AlgCpu1x64PerturbBla = 41110,    // IDM_ALG_CPU_1_64_PERTURB_BLA
    AlgCpu1x32PerturbBlaHdr = 41111, // IDM_ALG_CPU_1_32_PERTURB_BLA_HDR
    AlgCpu1x64PerturbBlaHdr = 41112, // IDM_ALG_CPU_1_64_PERTURB_BLA_HDR

    AlgCpu1x32PerturbBlav2Hdr = 41120,   // IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR
    AlgCpu1x64PerturbBlav2Hdr = 41121,   // IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR
    AlgCpu1x32PerturbRcBlav2Hdr = 41122, // IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR
    AlgCpu1x64PerturbRcBlav2Hdr = 41123, // IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR

    IterationPrecision4x = 41200,
    IterationPrecision3x = 41201,
    IterationPrecision2x = 41202,
    IterationPrecision1x = 41203,

    AlgGpu1x32 = 41210,    // IDM_ALG_GPU_1_32
    AlgGpu2x32 = 41211,    // IDM_ALG_GPU_2_32
    AlgGpu4x32 = 41212,    // IDM_ALG_GPU_4_32
    AlgGpu1x64 = 41213,    // IDM_ALG_GPU_1_64
    AlgGpu2x64 = 41214,    // IDM_ALG_GPU_2_64
    AlgGpu4x64 = 41215,    // IDM_ALG_GPU_4_64
    AlgGpu2x32Hdr = 41216, // IDM_ALG_GPU_2X32_HDR

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

    // ---- Synthetic Shortcut Commands ----
    AutoZoomFeatureAtPoint = 42000,
    AutoZoomDefaultAtPoint = 42001,
    CenterViewClearPerturbation = 42002,
    ResetCompressionDefaults = 42003,
    LaThresholdScaleIncrease = 42004,
    LaThresholdScaleDecrease = 42005,
    LaPeriodDetectionIncrease = 42006,
    LaPeriodDetectionDecrease = 42007,
    RecalcCurrentCopyDetails = 42008,
    RecalcClearMediumCopyDetails = 42009,
    RecalcClearAllCopyDetails = 42010,
    RecalcClearLaCopyDetails = 42011,
    IntermediateCompressionIncrease = 42012,
    IntermediateCompressionDecrease = 42013,
    LowCompressionIncrease = 42014,
    LowCompressionDecrease = 42015,
    PaletteAuxDepthNext = 42016,
    PaletteAuxDepthPrevious = 42017,
    PaletteDepthNext = 42018,
    RecalcClearAllSquareView = 42019,
};

// Convert FractalCommand to the native command-id value used by menu dispatch.
constexpr uint32_t
IdmFromCommand(FractalCommand c) noexcept
{
    return static_cast<uint32_t>(c);
}

// Convert a native command-id value back to FractalCommand. Caller asserts it is valid.
constexpr FractalCommand
CommandFromIdm(uint32_t idm) noexcept
{
    return static_cast<FractalCommand>(idm);
}

// A single keyboard shortcut. ASCII 'A'..'Z', '0'..'9' or printable punctuation
// in `key` (lower-case for letters; the dispatcher applies shift). Modifiers
// are explicit so the same key can map differently with shift held.
struct HotKey {
    wchar_t key = 0; // 0 means "no hotkey"
    bool shift = false;
    bool ctrl = false;
    bool alt = false;

    constexpr bool
    HasKey() const noexcept
    {
        return key != 0;
    }
};

constexpr wchar_t
NormalizeHotKeyKey(wchar_t key) noexcept
{
    return (key >= L'A' && key <= L'Z') ? static_cast<wchar_t>(key - L'A' + L'a') : key;
}

constexpr HotKey
NormalizeHotKey(HotKey hotkey) noexcept
{
    hotkey.key = NormalizeHotKeyKey(hotkey.key);
    return hotkey;
}

constexpr HotKey
HotKeyFromCharacter(wchar_t ch, bool shift, bool ctrl, bool alt) noexcept
{
    HotKey hotkey{0, shift, ctrl, alt};

    if (ch >= L'A' && ch <= L'Z') {
        hotkey.key = static_cast<wchar_t>(ch - L'A' + L'a');
        return hotkey;
    }
    if (ch >= L'a' && ch <= L'z') {
        hotkey.key = ch;
        return hotkey;
    }
    if (ch >= L'0' && ch <= L'9') {
        hotkey.key = ch;
        return hotkey;
    }

    switch (ch) {
        case L'+':
            hotkey.key = L'=';
            hotkey.shift = true;
            break;
        case L'<':
            hotkey.key = L',';
            hotkey.shift = true;
            break;
        case L'>':
            hotkey.key = L'.';
            hotkey.shift = true;
            break;
        case L'=':
        case L',':
        case L'.':
        case L'-':
            hotkey.key = ch;
            break;
        default:
            break;
    }

    return hotkey;
}

constexpr bool
HotKeysEqual(HotKey lhs, HotKey rhs) noexcept
{
    lhs = NormalizeHotKey(lhs);
    rhs = NormalizeHotKey(rhs);
    return lhs.key == rhs.key && lhs.shift == rhs.shift && lhs.ctrl == rhs.ctrl && lhs.alt == rhs.alt;
}

// Catalog row tying a command to an optional label and hotkey. Label is
// purely for the help/hotkey-listing UI; menu labels still come from
// MenuTreeDef.h so menu structure can group entries that share an id.
struct Command {
    FractalCommand id = FractalCommand::None;
    std::wstring_view label = {};
    HotKey hotkey = {};
};

inline constexpr auto kCommands = std::to_array<Command>({
    {FractalCommand::AutoZoomFeatureAtPoint, L"Autozoom feature at cursor", {L'a'}},
    {FractalCommand::AutoZoomDefaultAtPoint, L"Autozoom default from cursor", {L'a', true}},
    {FractalCommand::AutoZoomFilament, L"Autozoom filament tip", {L's', true}},
    {FractalCommand::Back, L"Go back to the previous view", {L'b'}},
    {FractalCommand::CenterView, L"Center view at cursor", {L'c'}},
    {FractalCommand::CenterViewClearPerturbation,
     L"Center view at cursor and clear perturbation",
     {L'c', true}},
    {FractalCommand::ResetCompressionDefaults, L"Reset compression defaults", {L'e'}},
    {FractalCommand::ResetCompressionDefaults, L"Reset compression defaults", {L'e', true}},
    {FractalCommand::FeatureFinderDirect, L"Find periodic point: direct", {L'n'}},
    {FractalCommand::FeatureFinderDirectScan, L"Find periodic point: direct scan", {L'n', true}},
    {FractalCommand::FeatureFinderPt, L"Find periodic point: PT", {L'm'}},
    {FractalCommand::FeatureFinderPtScan, L"Find periodic point: PT scan", {L'm', true}},
    {FractalCommand::FeatureFinderLa, L"Find periodic point: LA", {L','}},
    {FractalCommand::FeatureFinderLaScan, L"Find periodic point: LA scan", {L',', true}},
    {FractalCommand::FeatureFinderZoom, L"Zoom to found feature", {L'.'}},
    {FractalCommand::FeatureFinderClear, L"Clear all found features", {L'.', true}},
    {FractalCommand::LaThresholdScaleIncrease, L"Increase LA threshold scale exponents", {L'h'}},
    {FractalCommand::LaThresholdScaleDecrease, L"Decrease LA threshold scale exponents", {L'h', true}},
    {FractalCommand::LaPeriodDetectionIncrease, L"Increase LA period detection exponents", {L'j'}},
    {FractalCommand::LaPeriodDetectionDecrease, L"Decrease LA period detection exponents", {L'j', true}},
    {FractalCommand::RecalcCurrentCopyDetails, L"Recalculate and copy render details", {L'i'}},
    {FractalCommand::RecalcClearMediumCopyDetails,
     L"Clear medium perturbation, recalculate, copy details",
     {L'i', true}},
    {FractalCommand::RecalcCurrentCopyDetails, L"Recalculate and copy render details", {L'o'}},
    {FractalCommand::RecalcClearAllCopyDetails,
     L"Clear all perturbation, recalculate, copy details",
     {L'o', true}},
    {FractalCommand::RecalcCurrentCopyDetails, L"Recalculate and copy render details", {L'p'}},
    {FractalCommand::RecalcClearLaCopyDetails,
     L"Clear LA perturbation, recalculate, copy details",
     {L'p', true}},
    {FractalCommand::IntermediateCompressionIncrease,
     L"Increase intermediate compression error",
     {L'q'}},
    {FractalCommand::IntermediateCompressionDecrease,
     L"Decrease intermediate compression error",
     {L'q', true}},
    {FractalCommand::SquareView, L"Recalculate, reusing reference", {L'r'}},
    {FractalCommand::RecalcClearAllSquareView, L"Clear all perturbation and recalculate", {L'r', true}},
    {FractalCommand::PaletteAuxDepthNext, L"Use next auxiliary palette depth", {L't'}},
    {FractalCommand::PaletteAuxDepthPrevious, L"Use prior auxiliary palette depth", {L't', true}},
    {FractalCommand::LowCompressionIncrease, L"Increase reference compression error", {L'w'}},
    {FractalCommand::LowCompressionDecrease, L"Decrease reference compression error", {L'w', true}},
    {FractalCommand::ZoomIn, L"Zoom in at cursor", {L'z'}},
    {FractalCommand::ZoomOut, L"Zoom out at cursor", {L'z', true}},
    {FractalCommand::PaletteDepthNext, L"Use next palette lookup depth", {L'd'}},
    {FractalCommand::CreateNewPalette, L"Create and use random palette", {L'd', true}},
    {FractalCommand::IncreaseIterations24x, L"Multiply max iterations by 24", {L'='}},
    {FractalCommand::IncreaseIterations24x, L"Multiply max iterations by 24", {L'=', true}},
    {FractalCommand::DecreaseIterations, L"Multiply max iterations by 2/3", {L'-'}},
});

template <size_t N>
constexpr bool
ValidateCommandCatalog(const std::array<Command, N> &commands) noexcept
{
    for (size_t i = 0; i < N; ++i) {
        if (commands[i].id == FractalCommand::None || commands[i].label.empty() ||
            !commands[i].hotkey.HasKey()) {
            return false;
        }
        for (size_t j = i + 1; j < N; ++j) {
            if (HotKeysEqual(commands[i].hotkey, commands[j].hotkey)) {
                return false;
            }
        }
    }
    return true;
}

static_assert(ValidateCommandCatalog(kCommands), "Invalid FractalShark command hotkey catalog");

const Command *FindCommandByHotKey(HotKey hotkey) noexcept;
std::wstring FormatHotKey(HotKey hotkey);
std::string FormatHotKeyUtf8(HotKey hotkey);

// ---------------------------------------------------------------------------
// ExecuteCommand — per-platform host forwarder.
// ---------------------------------------------------------------------------
//
// Every catalog FractalCommand maps to one On*() hook on the host.  The
// algorithm-selection family (~55 RenderAlgorithmEnum-driven commands) is
// funnelled through the single OnSetAlgorithm hook; the catalog→alg lookup
// happens in CommandCatalog.cpp via kAlgCmds.  Native-only command ids must be
// handled by the platform shell before entering this catalog dispatcher.

// Forward declaration so we don't drag RenderAlgorithm.h into this header.
// The actual definition in RenderAlgorithm.h fixes the underlying type to
// uint32_t to match.
} // namespace FractalShark

enum class RenderAlgorithmEnum : uint32_t;

namespace FractalShark {

struct ExecuteCommandHost {
    virtual ~ExecuteCommandHost() = default;

    // ---- Algorithm selection ------------------------------------------
    virtual void OnSetAlgorithm(::RenderAlgorithmEnum alg) = 0;

    // ---- Synthetic shortcut command hooks ----------------------------
    virtual void OnAutoZoomFeatureAtPoint() = 0;
    virtual void OnAutoZoomDefaultAtPoint() = 0;
    virtual void OnCenterViewClearPerturbation() = 0;
    virtual void OnResetCompressionDefaults() = 0;
    virtual void OnLaThresholdScaleIncrease() = 0;
    virtual void OnLaThresholdScaleDecrease() = 0;
    virtual void OnLaPeriodDetectionIncrease() = 0;
    virtual void OnLaPeriodDetectionDecrease() = 0;
    virtual void OnRecalcCurrentCopyDetails() = 0;
    virtual void OnRecalcClearMediumCopyDetails() = 0;
    virtual void OnRecalcClearAllCopyDetails() = 0;
    virtual void OnRecalcClearLaCopyDetails() = 0;
    virtual void OnIntermediateCompressionIncrease() = 0;
    virtual void OnIntermediateCompressionDecrease() = 0;
    virtual void OnLowCompressionIncrease() = 0;
    virtual void OnLowCompressionDecrease() = 0;
    virtual void OnPaletteAuxDepthNext() = 0;
    virtual void OnPaletteAuxDepthPrevious() = 0;
    virtual void OnPaletteDepthNext() = 0;
    virtual void OnRecalcClearAllSquareView() = 0;

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
    // ---- Built-in views (range commands, 1-based) ---------------------
    virtual void OnStandardView() = 0;
    virtual void OnSelectBuiltInView(size_t oneBasedIndex) = 0;

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
