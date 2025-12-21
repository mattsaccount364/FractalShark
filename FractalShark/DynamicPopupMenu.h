#pragma once

#include <windows.h>

#include <cstdint>
#include <initializer_list>
#include <string_view>

//
// ---------------------------------------------------------------------------
// General / Help / Window
// ---------------------------------------------------------------------------
#define IDM_SHOWHOTKEYS 40000
#define IDM_VIEWS_HELP 40001
#define IDM_HELP_ALG 40002

#define IDM_SQUAREVIEW 40010
#define IDM_REPAINTING 40011
#define IDM_WINDOWED 40012
#define IDM_WINDOWED_SQ 40013
#define IDM_MINIMIZE 40014
#define IDM_CURPOS 40015

#define IDM_EXIT 40020

//
// ---------------------------------------------------------------------------
// Navigate
// ---------------------------------------------------------------------------
#define IDM_BACK 40100
#define IDM_CENTERVIEW 40101
#define IDM_ZOOMIN 40102
#define IDM_ZOOMOUT 40103
#define IDM_AUTOZOOM_DEFAULT 40104
#define IDM_AUTOZOOM_MAX 40105

//
// ---------------------------------------------------------------------------
// Built-In Views
// ---------------------------------------------------------------------------
#define IDM_STANDARDVIEW 40200
#define IDM_VIEW1 40201
#define IDM_VIEW2 40202
#define IDM_VIEW3 40203
#define IDM_VIEW4 40204
#define IDM_VIEW5 40205
#define IDM_VIEW6 40206
#define IDM_VIEW7 40207
#define IDM_VIEW8 40208
#define IDM_VIEW9 40209
#define IDM_VIEW10 40210
#define IDM_VIEW11 40211
#define IDM_VIEW12 40212
#define IDM_VIEW13 40213
#define IDM_VIEW14 40214
#define IDM_VIEW15 40215
#define IDM_VIEW16 40216
#define IDM_VIEW17 40217
#define IDM_VIEW18 40218
#define IDM_VIEW19 40219
#define IDM_VIEW20 40220
#define IDM_VIEW21 40221
#define IDM_VIEW22 40222
#define IDM_VIEW23 40223
#define IDM_VIEW24 40224
#define IDM_VIEW25 40225
#define IDM_VIEW26 40226
#define IDM_VIEW27 40227
#define IDM_VIEW28 40228
#define IDM_VIEW29 40229
#define IDM_VIEW30 40230
#define IDM_VIEW31 40231
#define IDM_VIEW32 40232
#define IDM_VIEW33 40233
#define IDM_VIEW34 40234
#define IDM_VIEW35 40235
#define IDM_VIEW36 40236
#define IDM_VIEW37 40237
#define IDM_VIEW38 40238
#define IDM_VIEW39 40239
#define IDM_VIEW40 40240

//
// ---------------------------------------------------------------------------
// GPU Antialiasing
// ---------------------------------------------------------------------------
#define IDM_GPUANTIALIASING_1X 40300
#define IDM_GPUANTIALIASING_4X 40301
#define IDM_GPUANTIALIASING_9X 40302
#define IDM_GPUANTIALIASING_16X 40303

//
// ---------------------------------------------------------------------------
// Iterations
// ---------------------------------------------------------------------------
#define IDM_RESETITERATIONS 40400
#define IDM_INCREASEITERATIONS_1P5X 40401
#define IDM_INCREASEITERATIONS_6X 40402
#define IDM_INCREASEITERATIONS_24X 40403
#define IDM_DECREASEITERATIONS 40404
#define IDM_32BIT_ITERATIONS 40405
#define IDM_64BIT_ITERATIONS 40406

//
// ---------------------------------------------------------------------------
// Perturbation
// ---------------------------------------------------------------------------
#define IDM_PERTURB_CLEAR_ALL 40500
#define IDM_PERTURB_CLEAR_MED 40501
#define IDM_PERTURB_CLEAR_HIGH 40502
#define IDM_PERTURB_RESULTS 40503

#define IDM_PERTURBATION_AUTO 40510
#define IDM_PERTURBATION_SINGLETHREAD 40511
#define IDM_PERTURBATION_MULTITHREAD 40512
#define IDM_PERTURBATION_SINGLETHREAD_PERIODICITY 40513
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY 40514
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED 40515
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1 40516
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2 40517
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3 40518
#define IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4 40519
#define IDM_PERTURBATION_MULTITHREAD5_PERIODICITY 40520
#define IDM_PERTURBATION_GPU 40521

#define IDM_PERTURBATION_LOAD 40530
#define IDM_PERTURBATION_SAVE 40531

//
// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------
#define IDM_PALETTE_TYPE_0 40600
#define IDM_PALETTE_TYPE_1 40601
#define IDM_PALETTE_TYPE_2 40602
#define IDM_PALETTE_TYPE_3 40603
#define IDM_PALETTE_TYPE_4 40604

#define IDM_CREATENEWPALETTE 40610

#define IDM_PALETTE_5 40620
#define IDM_PALETTE_6 40621
#define IDM_PALETTE_8 40622
#define IDM_PALETTE_12 40623
#define IDM_PALETTE_16 40624
#define IDM_PALETTE_20 40625

#define IDM_PALETTEROTATE 40630

//
// ---------------------------------------------------------------------------
// Memory Management
// ---------------------------------------------------------------------------
#define IDM_PERTURB_AUTOSAVE_ON_DELETE 40700
#define IDM_PERTURB_AUTOSAVE_ON 40701
#define IDM_PERTURB_AUTOSAVE_OFF 40702

#define IDM_MEMORY_LIMIT_0 40710
#define IDM_MEMORY_LIMIT_1 40711

//
// ---------------------------------------------------------------------------
// Save / Load
// ---------------------------------------------------------------------------
#define IDM_SAVELOCATION 40800
#define IDM_SAVEHIRESBMP 40801
#define IDM_SAVE_ITERS_TEXT 40802
#define IDM_SAVEBMP 40803

#define IDM_SAVE_REFORBIT_TEXT 40810
#define IDM_SAVE_REFORBIT_TEXT_SIMPLE 40811
#define IDM_SAVE_REFORBIT_TEXT_MAX 40812
#define IDM_SAVE_REFORBIT_IMAG_MAX 40813

#define IDM_BENCHMARK_FULL 40820
#define IDM_BENCHMARK_INT 40821
#define IDM_DIFF_REFORBIT_IMAG_MAX 40822

#define IDM_LOADLOCATION 40830
#define IDM_LOAD_ENTERLOCATION 40831
#define IDM_LOAD_REFORBIT_IMAG_MAX 40832
#define IDM_LOAD_REFORBIT_IMAG_MAX_SAVED 40833

//
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#define IDM_BASICTEST 40900
#define IDM_TEST_27 40901

//
// ---------------------------------------------------------------------------
// Algorithm Selection
// ---------------------------------------------------------------------------
#define IDM_ALG_AUTO 41000

//
// LA Parameters
//
#define IDM_LA_MULTITHREADED 41010
#define IDM_LA_SINGLETHREADED 41011
#define IDM_LA_SETTINGS_1 41012
#define IDM_LA_SETTINGS_2 41013
#define IDM_LA_SETTINGS_3 41014

//
// CPU-Only
//
#define IDM_ALG_CPU_HIGH 41100
#define IDM_ALG_CPU_1_64 41101
#define IDM_ALG_CPU_1_32_HDR 41102
#define IDM_ALG_CPU_1_64_HDR 41103

#define IDM_ALG_CPU_1_64_PERTURB_BLA 41110
#define IDM_ALG_CPU_1_32_PERTURB_BLA_HDR 41111
#define IDM_ALG_CPU_1_64_PERTURB_BLA_HDR 41112

#define IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR 41120
#define IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR 41121
#define IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR 41122
#define IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR 41123

//
// Iteration Precision
//
#define IDM_ITERATIONPRECISION_4X 41200
#define IDM_ITERATIONPRECISION_3X 41201
#define IDM_ITERATIONPRECISION_2X 41202
#define IDM_ITERATIONPRECISION_1X 41203

//
// Low-Zoom GPU (non-perturb)
//
#define IDM_ALG_GPU_1_32 41210
#define IDM_ALG_GPU_2_32 41211
#define IDM_ALG_GPU_4_32 41212
#define IDM_ALG_GPU_1_64 41213
#define IDM_ALG_GPU_2_64 41214
#define IDM_ALG_GPU_4_64 41215
#define IDM_ALG_GPU_2X32_HDR 41216

//
// Scaled
//
#define IDM_ALG_GPU_1_32_PERTURB_SCALED 41300
#define IDM_ALG_GPU_2_32_PERTURB_SCALED 41301
#define IDM_ALG_GPU_HDR_32_PERTURB_SCALED 41302

//
// Bilinear Approximation V1
//
#define IDM_ALG_GPU_1_64_PERTURB_BLA 41310
#define IDM_ALG_GPU_HDR_32_PERTURB_BLA 41311
#define IDM_ALG_GPU_HDR_64_PERTURB_BLA 41312

//
// LA-only (testing)
//
#define IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO 41320
#define IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO 41321
#define IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO 41322
#define IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO 41323
#define IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO 41324
#define IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO 41325

//
// Perturbation-only
//
#define IDM_ALG_GPU_1_32_PERTURB_LAV2_PO 41330
#define IDM_ALG_GPU_2_32_PERTURB_LAV2_PO 41331
#define IDM_ALG_GPU_1_64_PERTURB_LAV2_PO 41332
#define IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO 41333
#define IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO 41334
#define IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO 41335

//
// Reference Compression
//
#define IDM_ALG_GPU_1_32_PERTURB_RC_LAV2 41340
#define IDM_ALG_GPU_2_32_PERTURB_RC_LAV2 41341
#define IDM_ALG_GPU_1_64_PERTURB_RC_LAV2 41342
#define IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2 41343
#define IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2 41344
#define IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2 41345

//
// RC → Perturbation-only
//
#define IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_PO 41350
#define IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_PO 41351
#define IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_PO 41352
#define IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_PO 41353
#define IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_PO 41354
#define IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_PO 41355

//
// RC → LA-only
//
#define IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_LAO 41360
#define IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_LAO 41361
#define IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_LAO 41362
#define IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_LAO 41363
#define IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_LAO 41364
#define IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_LAO 41365

//
// Main LAv2 entries
//
#define IDM_ALG_GPU_1_32_PERTURB_LAV2 41400
#define IDM_ALG_GPU_2_32_PERTURB_LAV2 41401
#define IDM_ALG_GPU_1_64_PERTURB_LAV2 41402
#define IDM_ALG_GPU_HDR_32_PERTURB_LAV2 41403
#define IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2 41404
#define IDM_ALG_GPU_HDR_64_PERTURB_LAV2 41405


namespace FractalShark {

class UniqueHMenu;

class DynamicPopupMenu final {
public:
    static UniqueHMenu Create();
    static HMENU GetPopup(HMENU rootMenu) noexcept;

private:
    enum class Kind : uint8_t { Item, Separator, Popup };

    // IMPORTANT:
    //  - First 5 fields match your existing aggregate initializers:
    //      {Kind::Item, L"...", IDM_..., Enabled(), {}}
    //  - Extra metadata fields come AFTER kids so existing code compiles unchanged.
    //  - No in-class initializers: avoids MSVC C2797 with initializer_list.
    struct Node final {
        Kind kind;
        std::wstring_view text;           // Item/Popup label
        UINT id;                          // Item: command id, Popup: unused
        UINT stateFlags;                  // MF_ENABLED / MF_GRAYED
        std::initializer_list<Node> kids; // Popup children

        // Extended metadata (all optional; default to "off/zero" via aggregate init).
        bool checked;       // -> MFS_CHECKED
        bool radio;         // -> MFT_RADIOCHECK
        bool isDefault;     // -> MFS_DEFAULT
        bool ownerDraw;     // -> MFT_OWNERDRAW (requires WM_DRAWITEM/WM_MEASUREITEM)
        HBITMAP hbmpItem;   // -> MIIM_BITMAP
        ULONG_PTR itemData; // -> MIIM_DATA (dwItemData)
    };

    static constexpr UINT
    Enabled() noexcept
    {
        return MF_ENABLED;
    }
    static constexpr UINT
    Inactive() noexcept
    {
        return MF_GRAYED;
    }

    static bool InsertNodeAtEnd(HMENU menu, const Node &n);
    static bool InsertSeparatorAtEnd(HMENU menu);
    static bool InsertItemAtEnd(HMENU menu, const Node &n);
    static bool InsertPopupAtEnd(HMENU menu, const Node &n, HMENU popup);

    static bool BuildMenuTree(HMENU parent, std::initializer_list<Node> nodes);
    static bool BuildPopupContents(HMENU popup);
};

} // namespace FractalShark
