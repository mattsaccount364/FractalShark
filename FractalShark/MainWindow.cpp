#include "StdAfx.h"

#include "CrummyTest.h"
#include "DynamicPopupMenu.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "OpenGLContext.h"
#include "PngParallelSave.h"
#include "RecommendedSettings.h"
#include "WaitCursor.h"
#include "MainWindowSavedLocation.h"
#include "resource.h"

#include <cstdio>
#include <commdlg.h>
#include <minidumpapiset.h>
#include <random>
#include <unordered_map>

namespace {

constexpr int kMaxDynamic = 30;

// ---- One source of truth for ALL algorithm command mappings ----
// Add/remove entries ONLY here. Everything else derives from this.
#define FS_ALG_CMD_LIST(X)                                                                              \
    X(IDM_ALG_AUTO, RenderAlgorithmEnum::AUTO)                                                          \
    X(IDM_ALG_CPU_HIGH, RenderAlgorithmEnum::CpuHigh)                                                   \
    X(IDM_ALG_CPU_1_32_HDR, RenderAlgorithmEnum::CpuHDR32)                                              \
    X(IDM_ALG_CPU_1_32_PERTURB_BLA_HDR, RenderAlgorithmEnum::Cpu32PerturbedBLAHDR)                      \
    X(IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR, RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR)                  \
    X(IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR, RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR)             \
    X(IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR, RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR)                  \
    X(IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR, RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR)             \
    X(IDM_ALG_CPU_1_64, RenderAlgorithmEnum::Cpu64)                                                     \
    X(IDM_ALG_CPU_1_64_HDR, RenderAlgorithmEnum::CpuHDR64)                                              \
    X(IDM_ALG_CPU_1_64_PERTURB_BLA, RenderAlgorithmEnum::Cpu64PerturbedBLA)                             \
    X(IDM_ALG_CPU_1_64_PERTURB_BLA_HDR, RenderAlgorithmEnum::Cpu64PerturbedBLAHDR)                      \
    X(IDM_ALG_GPU_1_64, RenderAlgorithmEnum::Gpu1x64)                                                   \
    X(IDM_ALG_GPU_1_64_PERTURB_BLA, RenderAlgorithmEnum::Gpu1x64PerturbedBLA)                           \
    X(IDM_ALG_GPU_2_64, RenderAlgorithmEnum::Gpu2x64)                                                   \
    X(IDM_ALG_GPU_4_64, RenderAlgorithmEnum::Gpu4x64)                                                   \
    X(IDM_ALG_GPU_2X32_HDR, RenderAlgorithmEnum::GpuHDRx32)                                             \
    X(IDM_ALG_GPU_1_32, RenderAlgorithmEnum::Gpu1x32)                                                   \
    X(IDM_ALG_GPU_1_32_PERTURB_SCALED, RenderAlgorithmEnum::Gpu1x32PerturbedScaled)                     \
    X(IDM_ALG_GPU_HDR_32_PERTURB_SCALED, RenderAlgorithmEnum::GpuHDRx32PerturbedScaled)                 \
    X(IDM_ALG_GPU_2_32, RenderAlgorithmEnum::Gpu2x32)                                                   \
    X(IDM_ALG_GPU_2_32_PERTURB_SCALED, RenderAlgorithmEnum::Gpu2x32PerturbedScaled)                     \
    X(IDM_ALG_GPU_4_32, RenderAlgorithmEnum::Gpu4x32)                                                   \
    X(IDM_ALG_GPU_HDR_32_PERTURB_BLA, RenderAlgorithmEnum::GpuHDRx32PerturbedBLA)                       \
    X(IDM_ALG_GPU_HDR_64_PERTURB_BLA, RenderAlgorithmEnum::GpuHDRx64PerturbedBLA)                       \
    /* ------------------------- LAv2 family ------------------------- */                               \
    X(IDM_ALG_GPU_1_32_PERTURB_LAV2, RenderAlgorithmEnum::Gpu1x32PerturbedLAv2)                         \
    X(IDM_ALG_GPU_1_32_PERTURB_LAV2_PO, RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO)                    \
    X(IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO, RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO)                  \
    X(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2, RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2)                    \
    X(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO)               \
    X(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO)             \
    X(IDM_ALG_GPU_2_32_PERTURB_LAV2, RenderAlgorithmEnum::Gpu2x32PerturbedLAv2)                         \
    X(IDM_ALG_GPU_2_32_PERTURB_LAV2_PO, RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO)                    \
    X(IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO, RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO)                  \
    X(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2, RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2)                    \
    X(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO)               \
    X(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO)             \
    X(IDM_ALG_GPU_1_64_PERTURB_LAV2, RenderAlgorithmEnum::Gpu1x64PerturbedLAv2)                         \
    X(IDM_ALG_GPU_1_64_PERTURB_LAV2_PO, RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO)                    \
    X(IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO, RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO)                  \
    X(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2, RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2)                    \
    X(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO)               \
    X(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO)             \
    X(IDM_ALG_GPU_HDR_32_PERTURB_LAV2, RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2)                     \
    X(IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO, RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO)                \
    X(IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO)              \
    X(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2, RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2)                \
    X(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO)           \
    X(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO)         \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2, RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2)                 \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO, RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO)            \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO)          \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2, RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2)            \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO)       \
    X(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO)     \
    X(IDM_ALG_GPU_HDR_64_PERTURB_LAV2, RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2)                     \
    X(IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO, RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO)                \
    X(IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO)              \
    X(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2, RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2)                \
    X(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_PO, RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO)           \
    X(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_LAO, RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO)

struct AlgCmd {
    int id;
    RenderAlgorithmEnum alg;
};

#define FS_MAKE_ALG_CMD(id_, alg_) AlgCmd{(id_), (alg_)},

static constexpr std::array<AlgCmd,
                            []() consteval {
                                size_t n = 0;
#define FS_COUNT_ALG_CMD(id_, alg_) ++n;
                                FS_ALG_CMD_LIST(FS_COUNT_ALG_CMD)
#undef FS_COUNT_ALG_CMD
                                return n;
                            }()>
    kAlgCmds = {FS_ALG_CMD_LIST(FS_MAKE_ALG_CMD)};

#undef FS_MAKE_ALG_CMD

// ---- constexpr checks ----
template <typename T, size_t N, typename Proj>
consteval bool
all_unique(const std::array<T, N> &a, Proj proj)
{
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (proj(a[i]) == proj(a[j])) {
                return false;
            }
        }
    }
    return true;
}

consteval bool
ids_unique()
{
    return all_unique(kAlgCmds, [](const AlgCmd &e) { return e.id; });
}

consteval bool
enums_unique()
{
    return all_unique(kAlgCmds, [](const AlgCmd &e) { return static_cast<int>(e.alg); });
}

static_assert(enums_unique(), "Duplicate RenderAlgorithmEnum in kAlgCmds (did you copy/paste?)");

static_assert(ids_unique(), "Duplicate IDM_ALG_* ID in kAlgCmds (did you copy/paste?)");

// Optional: constexpr lookup helper
constexpr const AlgCmd *
FindAlgForCmd(int wmId)
{
    for (const auto &e : kAlgCmds) {
        if (e.id == wmId)
            return &e;
    }
    return nullptr;
}

} // namespace

MainWindow::MainWindow(HINSTANCE hInstance, int nCmdShow)
{
    gJobObj = std::make_unique<JobObject>();
    HighPrecision::defaultPrecisionInBits(256);

    // Create a dump file whenever the gateway crashes only on windows
    SetUnhandledExceptionFilter(unhandled_handler);

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // Perform application initialization:
    hWnd = InitInstance(hInstance, nCmdShow);
    if (!hWnd) {
        throw FractalSharkSeriousException("Failed to create window.");
    }

    ImaginaMenu = nullptr;
    LoadSubMenu = nullptr;
}

MainWindow::~MainWindow()
{
    // Cleanup
    UnInit();

    gJobObj.reset();
}

void
MainWindow::MainWindow::DrawFractalShark()
{
    auto glContext = std::make_unique<OpenGlContext>(hWnd);
    if (!glContext->IsValid()) {
        return;
    }

    glContext->DrawFractalShark(hWnd);
}

void
MainWindow::MainWindow::DrawFractalSharkGdi(int nCmdShow)
{
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_int_distribution<int> dist(0, 2);

    const int choice = dist(rng);

    LPCWSTR resStr = nullptr;
    switch (choice) {
        case 0:
            resStr = MAKEINTRESOURCE(IDB_PNG_SPLASH1);
            break;
        case 1:
            resStr = MAKEINTRESOURCE(IDB_PNG_SPLASH2);
            break;
        case 2:
            resStr = MAKEINTRESOURCE(IDB_PNG_SPLASH3);
            break;
    }

    HRSRC hRes = FindResource(hInst, resStr, L"PNG");
    if (hRes == nullptr) {
        return;
    }

    //  Convert the HRSRC into a pointer to the actual data
    HGLOBAL hResData = LoadResource(hInst, hRes);
    if (hResData == nullptr) {
        return;
    }

    void *pResData = LockResource(hResData);
    if (pResData == nullptr) {
        return;
    }

    //  Get the size of the resource data
    DWORD dwSize = SizeofResource(hInst, hRes);
    if (dwSize == 0) {
        return;
    }

    WPngImage image{};
    image.loadImageFromRAM(pResData, dwSize, WPngImage::PixelFormat::kPixelFormat_RGBA8);

    std::vector<uint8_t> imageBytes;
    imageBytes.resize(image.width() * image.height() * 4);

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            auto pixel = image.get8(x, y);
            imageBytes[(y * image.width() + x) * 4 + 0] = pixel.b;
            imageBytes[(y * image.width() + x) * 4 + 1] = pixel.g;
            imageBytes[(y * image.width() + x) * 4 + 2] = pixel.r;
            imageBytes[(y * image.width() + x) * 4 + 3] = pixel.a;
        }
    }

    RECT windowDimensions;
    GetClientRect(hWnd, &windowDimensions);

    // Create a bitmap and render it to hWnd
    HDC hdc = GetDC(hWnd);
    HDC hdcMem = CreateCompatibleDC(hdc);
    HBITMAP hBitmap = CreateBitmap(image.width(), image.height(), 1, 32, imageBytes.data());

    // Render the bitmap to the window, scaling the bitmap down if needed.
    // If it needs to be scaled up, just leave it at its original size.
    SelectObject(hdcMem, hBitmap);

    // Find the min width and height between the window and the bitmap, and render it up to that size
    const int windowWidth = (int)windowDimensions.right;
    const int windowHeight = (int)windowDimensions.bottom;

    SetStretchBltMode(hdc, HALFTONE);
    SetBrushOrgEx(hdc, 0, 0, nullptr);

    // Clear the window with black
    RECT rt;
    GetClientRect(hWnd, &rt);
    FillRect(hdc, &rt, (HBRUSH)GetStockObject(BLACK_BRUSH));

    // Display!
    ShowWindow(hWnd, nCmdShow);

    // Given the image width and window dimensions, calculate the starting point for the image
    // such that it ends up centered.
    int startX = (windowWidth - image.width()) / 2;
    int startY = (windowHeight - image.height()) / 2;

    if (windowWidth < image.width() || windowHeight < image.height()) {
        StretchBlt(
            hdc, 0, 0, windowWidth, windowHeight, hdcMem, 0, 0, image.width(), image.height(), SRCCOPY);
    } else {
        // Center the image
        BitBlt(hdc, startX, startY, image.width(), image.height(), hdcMem, 0, 0, SRCCOPY);
    }

    // Set the window as opaque
    SetLayeredWindowAttributes(hWnd, RGB(0, 0, 0), 255, LWA_ALPHA);

    // ShowWindow(hWnd, nCmdShow);
    // ShowWindow(hWnd, SW_RESTORE);

    // Clean up
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(hWnd, hdc);
}

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM
MainWindow::MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)StaticWndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = sizeof(MainWindow *);
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, (LPCTSTR)IDI_FRACTALS);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, (LPCTSTR)IDI_SMALL);

    return RegisterClassEx(&wcex);
}

//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//     Here we create the main window and return its handle,
//     and we perform other initialization.
//
HWND
MainWindow::InitInstance(HINSTANCE hInstance, int nCmdShow)
{ // Store instance handle in our global variable
    hInst = hInstance;

    constexpr bool startWindowed = true;
    constexpr bool finishWindowed = true;
    constexpr DWORD forceStartWidth = 0;
    constexpr DWORD forceStartHeight = 0;

    const auto scrnWidth = GetSystemMetrics(SM_CXSCREEN);
    const auto scrnHeight = GetSystemMetrics(SM_CYSCREEN);

    DWORD startX, startY;
    DWORD width, height;

    if constexpr (startWindowed) {
        width = std::min(scrnWidth / 2, scrnHeight / 2);
        height = width;
        startX = scrnWidth / 2 - width / 2;
        startY = scrnHeight / 2 - width / 2;

        // Uncomment to start in smaller window
        gWindowed = true;
        // MenuWindowed(hWnd, true);
    } else {
        startX = 0;
        startY = 0;
        width = scrnWidth;
        height = scrnHeight;

        gWindowed = false;
    }

    if constexpr (forceStartWidth) {
        width = forceStartWidth;
    }

    if constexpr (forceStartHeight) {
        height = forceStartHeight;
    }

    DWORD wndFlags = WS_POPUP | WS_THICKFRAME;

    if (!startWindowed) {
        wndFlags |= WS_MAXIMIZE;
    }

    // Create the window
    hWnd = CreateWindow(szWindowClass,
                        L"",
                        wndFlags,
                        startX,
                        startY,
                        width,
                        height,
                        nullptr,
                        nullptr,
                        hInstance,
                        nullptr);

    if (!hWnd) {
        return nullptr;
    }

    // Initialize the 8 bytes after the window handle to point to this object
    SetWindowLongPtrA(hWnd, 0, (LONG_PTR)this);

    // Use  SetWindowLong to make the window layered
    SetWindowLongPtr(hWnd, GWL_EXSTYLE, GetWindowLongPtr(hWnd, GWL_EXSTYLE) | WS_EX_LAYERED);

    // Set the window as transparent
    SetLayeredWindowAttributes(hWnd, RGB(0, 0, 0), 0, LWA_ALPHA);

    // DrawFractalShark(hWnd);
    DrawFractalSharkGdi(nCmdShow);

    // Put us on top
    // SetWindowPos (hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    // Create the menu
    gPopupMenu = FractalShark::DynamicPopupMenu::Create();
    FractalShark::DynamicPopupMenu::SetCurrentRenderAlgorithmId(IDM_ALG_AUTO);

    // Create the fractal
    RECT rt;
    GetClientRect(hWnd, &rt);

    gFractal =
        std::make_unique<Fractal>(rt.right, rt.bottom, hWnd, false, gJobObj->GetCommitLimitInBytes());

    if constexpr (finishWindowed == false) {
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);
        gWindowed = false;
    }

    return hWnd;
}

//
// Performs all cleanup operations
//
void
MainWindow::UnInit()
{
    ClearMenu(LoadSubMenu);
    ClearMenu(ImaginaMenu);

    DestroyWindow(hWnd);
    UnregisterClass(szWindowClass, hInst);
    gFractal.reset();
    gJobObj.reset();
}

void
MainWindow::HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM /*lParam*/)
{
    POINT mousePt;
    GetCursorPos(&mousePt);
    if (ScreenToClient(hWnd, &mousePt) == 0) {
        return;
    }

    SHORT nState = GetAsyncKeyState(VK_SHIFT);
    bool shiftDown = (nState & 0x8000) != 0;

    switch (wParam) {
        case 'A':
        case 'a':
            if (!shiftDown) {
                MenuCenterView(mousePt.x, mousePt.y);
                gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
            } else {
                MenuCenterView(mousePt.x, mousePt.y);
                gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>();
            }
            break;

        case 'b':
            MenuGoBack();
            break;

        case 'C':
        case 'c':
            if (shiftDown) {
                gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            }
            MenuCenterView(mousePt.x, mousePt.y);
            break;

        case 'E':
        case 'e':
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            gFractal->DefaultCompressionErrorExp(Fractal::CompressionError::Low);
            gFractal->DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
            PaintAsNecessary();
            break;

        case 'H':
        case 'h': {
            auto &laParameters = gFractal->GetLAParameters();
            if (shiftDown) {
                laParameters.AdjustLAThresholdScaleExponent(-1);
                laParameters.AdjustLAThresholdCScaleExponent(-1);
            } else {
                laParameters.AdjustLAThresholdScaleExponent(1);
                laParameters.AdjustLAThresholdCScaleExponent(1);
            }
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
            gFractal->ForceRecalc();
            PaintAsNecessary();
            break;
        }

        case 'J':
        case 'j': {
            auto &laParameters = gFractal->GetLAParameters();
            if (shiftDown) {
                laParameters.AdjustPeriodDetectionThreshold2Exponent(-1);
                laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(-1);
            } else {
                laParameters.AdjustPeriodDetectionThreshold2Exponent(1);
                laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(1);
            }
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
            gFractal->ForceRecalc();
            PaintAsNecessary();
            break;
        }

        case 'I':
        case 'i':
            if (shiftDown) {
                gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
            }
            gFractal->ForceRecalc();
            PaintAsNecessary();
            MenuGetCurPos();
            break;

        case 'O':
        case 'o':
            if (shiftDown) {
                gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            }
            gFractal->ForceRecalc();
            PaintAsNecessary();
            MenuGetCurPos();
            break;

        case 'P':
        case 'p':
            if (shiftDown) {
                gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
            }
            gFractal->ForceRecalc();
            PaintAsNecessary();
            MenuGetCurPos();
            break;

        case 'q':
        case 'Q':
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            if (shiftDown) {
                gFractal->DecCompressionError(Fractal::CompressionError::Intermediate, 10);
            } else {
                gFractal->IncCompressionError(Fractal::CompressionError::Intermediate, 10);
            }
            PaintAsNecessary();
            break;

        case 'R':
        case 'r':
            if (shiftDown) {
                gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            }
            MenuSquareView();
            break;

        case 'T':
        case 't':
            if (shiftDown) {
                gFractal->UseNextPaletteAuxDepth(-1);
            } else {
                gFractal->UseNextPaletteAuxDepth(1);
            }
            gFractal->DrawFractal(false);
            break;

        case 'W':
        case 'w':
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            if (shiftDown) {
                gFractal->DecCompressionError(Fractal::CompressionError::Low, 1);
            } else {
                gFractal->IncCompressionError(Fractal::CompressionError::Low, 1);
            }
            PaintAsNecessary();
            break;

        case 'Z':
        case 'z':
            if (shiftDown) {
                MenuZoomOut(mousePt);
            } else {
                MenuZoomIn(mousePt);
            }
            break;

        case 'D':
        case 'd': {
            if (shiftDown) {
                gFractal->CreateNewFractalPalette();
                gFractal->UsePaletteType(FractalPalette::Random);
            } else {
                gFractal->UseNextPaletteDepth();
            }
            gFractal->DrawFractal(false);
            break;
        }

        case '=':
            MenuMultiplyIterations(24.0);
            break;
        case '-':
            MenuMultiplyIterations(2.0 / 3.0);
            break;

        default:
            break;
    }
}

LRESULT
MainWindow::StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    // Get the window class from the hWnd
    MainWindow *pThis = reinterpret_cast<MainWindow *>(GetWindowLongPtr(hWnd, 0));

    if (pThis == nullptr) {
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    auto copyToClipboard = [](std::string str) {
        if (OpenClipboard(nullptr)) {
            HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, str.size() + 1);
            if (hMem != nullptr) {
                char *pMem = (char *)GlobalLock(hMem);
                if (pMem != nullptr) {
                    memcpy(pMem, str.c_str(), str.size() + 1);
                    GlobalUnlock(hMem);
                    EmptyClipboard();
                    SetClipboardData(CF_TEXT, hMem);
                }
            }
            CloseClipboard();
        }
    };

    // And invoke WndProc
    if (IsDebuggerPresent()) {
        return pThis->WndProc(message, wParam, lParam);
    } else {
        try {
            return pThis->WndProc(message, wParam, lParam);
        } catch (const FractalSharkSeriousException &e) {
            const auto msg = e.GetCallstack("Message copied to clipboard.  CTRL-V to paste.");
            copyToClipboard(msg);
            MessageBoxA(hWnd, msg.c_str(), "Error", MB_OK);
            return 0;
        } catch (const std::exception &e) {
            copyToClipboard(e.what());
            MessageBoxA(hWnd, e.what(), "Error", MB_OK);
            return 0;
        }
    }
}

std::wstring
MainWindow::OpenFileDialog(OpenBoxType type)
{
    OPENFILENAME ofn;    // common dialog box structure
    wchar_t szFile[260]; // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = L"All\0*.*\0Imagina\0*.im\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;

    if (type == OpenBoxType::Open) {
        // Display the Open dialog box.
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        if (GetOpenFileName(&ofn) == TRUE) {
            return std::wstring(ofn.lpstrFile);
        } else {
            return std::wstring();
        }
    } else {
        ofn.Flags = 0;
        if (GetSaveFileName(&ofn) == TRUE) {
            return std::wstring(ofn.lpstrFile);
        } else {
            return std::wstring();
        }
    }
}

bool
MainWindow::HasLastMenuPtClient() const noexcept
{
    return lastMenuPtClient_.x >= 0 && lastMenuPtClient_.y >= 0;
}

POINT
MainWindow::GetSafeMenuPtClient() const
{
    // If user hasn’t opened the context menu yet, fall back to cursor pos.
    POINT pt = lastMenuPtClient_;

    if (!HasLastMenuPtClient()) {
        ::GetCursorPos(&pt);
        ::ScreenToClient(hWnd, &pt);
    }
    return pt;
}

bool
MainWindow::HandleCommandTable(int wmId)
{
    auto doCenter = [](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuCenterView(pt.x, pt.y);
    };
    auto doZoomIn = [](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuZoomIn(pt);
    };
    auto doZoomOut = [](MainWindow &w) {
        const POINT pt = w.GetSafeMenuPtClient();
        w.MenuZoomOut(pt);
    };

    using Fn = std::function<void(MainWindow &)>;

    // NOTE: This table is intended to fully subsume the remaining WM_COMMAND switch cases,
    // except for:
    //   - ranges handled by HandleCommandRange()
    //   - algorithm IDs handled by HandleAlgCommand()
    static const std::unordered_map<int, Fn> table = {
        // Navigation / view
        {IDM_BACK, [](MainWindow &w) { w.MenuGoBack(); }},
        {IDM_STANDARDVIEW, [](MainWindow &w) { w.MenuStandardView(0); }},
        {IDM_SQUAREVIEW, [](MainWindow &w) { w.MenuSquareView(); }},
        {IDM_VIEWS_HELP, [](MainWindow &w) { w.MenuViewsHelp(); }},
        {IDM_CENTERVIEW, doCenter},
        {IDM_ZOOMIN, doZoomIn},
        {IDM_ZOOMOUT, doZoomOut},

        {IDM_AUTOZOOM_DEFAULT,
         [](MainWindow &w) { w.gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>(); }},
        {IDM_AUTOZOOM_MAX,
         [](MainWindow &w) { w.gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>(); }},

        {IDM_REPAINTING, [](MainWindow &w) { w.MenuRepainting(); }},
        {IDM_WINDOWED, [](MainWindow &w) { w.MenuWindowed(false); }},
        {IDM_WINDOWED_SQ, [](MainWindow &w) { w.MenuWindowed(true); }},
        {IDM_MINIMIZE, [](MainWindow &w) { PostMessage(w.hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0); }},

        // GPU AA
        {IDM_GPUANTIALIASING_1X,
         [](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1); }},
        {IDM_GPUANTIALIASING_4X,
         [](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2); }},
        {IDM_GPUANTIALIASING_9X,
         [](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 3); }},
        {IDM_GPUANTIALIASING_16X,
         [](MainWindow &w) { w.gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4); }},

        // Iteration precision
        {IDM_ITERATIONPRECISION_1X, [](MainWindow &w) { w.gFractal->SetIterationPrecision(1); }},
        {IDM_ITERATIONPRECISION_2X, [](MainWindow &w) { w.gFractal->SetIterationPrecision(4); }},
        {IDM_ITERATIONPRECISION_3X, [](MainWindow &w) { w.gFractal->SetIterationPrecision(8); }},
        {IDM_ITERATIONPRECISION_4X, [](MainWindow &w) { w.gFractal->SetIterationPrecision(16); }},

        // Algorithm help
        {IDM_HELP_ALG, [](MainWindow &w) { w.MenuAlgHelp(); }},

        // LA toggles / presets
        {IDM_LA_SINGLETHREADED,
         [](MainWindow &w) {
             auto &p = w.gFractal->GetLAParameters();
             p.SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
         }},
        {IDM_LA_MULTITHREADED,
         [](MainWindow &w) {
             auto &p = w.gFractal->GetLAParameters();
             p.SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
         }},
        {IDM_LA_SETTINGS_1,
         [](MainWindow &w) {
             auto &p = w.gFractal->GetLAParameters();
             p.SetDefaults(LAParameters::LADefaults::MaxAccuracy);
         }},
        {IDM_LA_SETTINGS_2,
         [](MainWindow &w) {
             auto &p = w.gFractal->GetLAParameters();
             p.SetDefaults(LAParameters::LADefaults::MaxPerf);
         }},
        {IDM_LA_SETTINGS_3,
         [](MainWindow &w) {
             auto &p = w.gFractal->GetLAParameters();
             p.SetDefaults(LAParameters::LADefaults::MinMemory);
         }},

        // Tests / benchmarks
        {IDM_BASICTEST,
         [](MainWindow &w) {
             CrummyTest t{*w.gFractal};
             t.TestAll();
         }},
        {IDM_TEST_27,
         [](MainWindow &w) {
             CrummyTest t{*w.gFractal};
             t.TestReallyHardView27();
         }},
        {IDM_BENCHMARK_FULL,
         [](MainWindow &w) {
             CrummyTest t{*w.gFractal};
             t.Benchmark(RefOrbitCalc::PerturbationResultType::All);
         }},
        {IDM_BENCHMARK_INT,
         [](MainWindow &w) {
             CrummyTest t{*w.gFractal};
             t.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
         }},

        // Iterations
        {IDM_INCREASEITERATIONS_1P5X, [](MainWindow &w) { w.MenuMultiplyIterations(1.5); }},
        {IDM_INCREASEITERATIONS_6X, [](MainWindow &w) { w.MenuMultiplyIterations(6.0); }},
        {IDM_INCREASEITERATIONS_24X, [](MainWindow &w) { w.MenuMultiplyIterations(24.0); }},
        {IDM_DECREASEITERATIONS, [](MainWindow &w) { w.MenuMultiplyIterations(2.0 / 3.0); }},
        {IDM_RESETITERATIONS, [](MainWindow &w) { w.MenuResetIterations(); }},
        {IDM_32BIT_ITERATIONS, [](MainWindow &w) { w.gFractal->SetIterType(IterTypeEnum::Bits32); }},
        {IDM_64BIT_ITERATIONS, [](MainWindow &w) { w.gFractal->SetIterType(IterTypeEnum::Bits64); }},

        // Perturbation UI
        {IDM_PERTURB_RESULTS,
         [](MainWindow &w) {
             ::MessageBox(w.hWnd,
                          L"TODO.  By default these are shown as white pixels overlayed on the image. "
                          L"It'd be nice to have an option that shows them as white pixels against a "
                          L"black screen so they're location is obvious.",
                          L"TODO",
                          MB_OK | MB_APPLMODAL);
         }},
        {IDM_PERTURB_CLEAR_ALL,
         [](MainWindow &w) {
             w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
         }},
        {IDM_PERTURB_CLEAR_MED,
         [](MainWindow &w) {
             w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
         }},
        {IDM_PERTURB_CLEAR_HIGH,
         [](MainWindow &w) {
             w.gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes);
         }},

        {IDM_PERTURBATION_AUTO,
         [](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto); }},
        {IDM_PERTURBATION_SINGLETHREAD,
         [](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST); }},
        {IDM_PERTURBATION_MULTITHREAD,
         [](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT); }},
        {IDM_PERTURBATION_SINGLETHREAD_PERIODICITY,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(
                 RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(
                 RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(
                 RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(
                 RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
         }},
        {IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(
                 RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
         }},
        {IDM_PERTURBATION_MULTITHREAD5_PERIODICITY,
         [](MainWindow &w) {
             w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5);
         }},
        {IDM_PERTURBATION_GPU,
         [](MainWindow &w) { w.gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU); }},

        {IDM_PERTURBATION_SAVE, [](MainWindow &w) { w.gFractal->SavePerturbationOrbits(); }},
        {IDM_PERTURBATION_LOAD, [](MainWindow &w) { w.gFractal->LoadPerturbationOrbits(); }},

        {IDM_PERTURB_AUTOSAVE_ON,
         [](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::EnableWithSave); }},
        {IDM_PERTURB_AUTOSAVE_ON_DELETE,
         [](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::EnableWithoutSave); }},
        {IDM_PERTURB_AUTOSAVE_OFF,
         [](MainWindow &w) { w.gFractal->SetResultsAutosave(AddPointOptions::DontSave); }},

        // Memory limit toggle
        {IDM_MEMORY_LIMIT_0, [](MainWindow &w) { w.gJobObj = nullptr; }},
        {IDM_MEMORY_LIMIT_1, [](MainWindow &w) { w.gJobObj = std::make_unique<JobObject>(); }},

        // Palettes
        {IDM_PALETTEROTATE, [](MainWindow &w) { w.MenuPaletteRotation(); }},
        {IDM_CREATENEWPALETTE, [](MainWindow &w) { w.MenuCreateNewPalette(); }},

        {IDM_PALETTE_TYPE_0, [](MainWindow &w) { w.MenuPaletteType(FractalPalette::Basic); }},
        {IDM_PALETTE_TYPE_1, [](MainWindow &w) { w.MenuPaletteType(FractalPalette::Default); }},
        {IDM_PALETTE_TYPE_2, [](MainWindow &w) { w.MenuPaletteType(FractalPalette::Patriotic); }},
        {IDM_PALETTE_TYPE_3, [](MainWindow &w) { w.MenuPaletteType(FractalPalette::Summer); }},
        {IDM_PALETTE_TYPE_4, [](MainWindow &w) { w.MenuPaletteType(FractalPalette::Random); }},

        {IDM_PALETTE_5, [](MainWindow &w) { w.MenuPaletteDepth(5); }},
        {IDM_PALETTE_6, [](MainWindow &w) { w.MenuPaletteDepth(6); }},
        {IDM_PALETTE_8, [](MainWindow &w) { w.MenuPaletteDepth(8); }},
        {IDM_PALETTE_12, [](MainWindow &w) { w.MenuPaletteDepth(12); }},
        {IDM_PALETTE_16, [](MainWindow &w) { w.MenuPaletteDepth(16); }},
        {IDM_PALETTE_20, [](MainWindow &w) { w.MenuPaletteDepth(20); }},

        // Location / IO
        {IDM_CURPOS, [](MainWindow &w) { w.MenuGetCurPos(); }},
        {IDM_SAVELOCATION, [](MainWindow &w) { w.MenuSaveCurrentLocation(); }},
        {IDM_LOADLOCATION, [](MainWindow &w) { w.MenuLoadCurrentLocation(); }},
        {IDM_LOAD_ENTERLOCATION, [](MainWindow &w) { w.MenuLoadEnterLocation(); }},

        {IDM_SAVEBMP, [](MainWindow &w) { w.MenuSaveBMP(); }},
        {IDM_SAVEHIRESBMP, [](MainWindow &w) { w.MenuSaveHiResBMP(); }},
        {IDM_SAVE_ITERS_TEXT, [](MainWindow &w) { w.MenuSaveItersAsText(); }},

        {IDM_SAVE_REFORBIT_TEXT, [](MainWindow &w) { w.MenuSaveImag(CompressToDisk::Disable); }},
        {IDM_SAVE_REFORBIT_TEXT_SIMPLE,
         [](MainWindow &w) { w.MenuSaveImag(CompressToDisk::SimpleCompression); }},
        {IDM_SAVE_REFORBIT_TEXT_MAX,
         [](MainWindow &w) { w.MenuSaveImag(CompressToDisk::MaxCompression); }},
        {IDM_SAVE_REFORBIT_IMAG_MAX,
         [](MainWindow &w) { w.MenuSaveImag(CompressToDisk::MaxCompressionImagina); }},
        {IDM_DIFF_REFORBIT_IMAG_MAX, [](MainWindow &w) { w.MenuDiffImag(); }},

        {IDM_LOAD_REFORBIT_IMAG_MAX,
         [](MainWindow &w) { w.MenuLoadImagDyn(ImaginaSettings::ConvertToCurrent); }},
        {IDM_LOAD_REFORBIT_IMAG_MAX_SAVED,
         [](MainWindow &w) { w.MenuLoadImagDyn(ImaginaSettings::UseSaved); }},

        {IDM_LOAD_IMAGINA_DLG,
         [](MainWindow &w) {
             w.MenuLoadImag(ImaginaSettings::ConvertToCurrent, CompressToDisk::MaxCompressionImagina);
         }},
        {IDM_LOAD_IMAGINA_DLG_SAVED,
         [](MainWindow &w) {
             w.MenuLoadImag(ImaginaSettings::UseSaved, CompressToDisk::MaxCompressionImagina);
         }},

        // Help / exit
        {IDM_SHOWHOTKEYS, [](MainWindow &w) { w.MenuShowHotkeys(); }},
        {IDM_EXIT, [](MainWindow &w) { DestroyWindow(w.hWnd); }},
    };

    if (auto it = table.find(wmId); it != table.end()) {
        it->second(*this);
        return true;
    }

    return false;
}

bool
MainWindow::HandleCommandRange(int wmId)
{
    // ---- Views 1..40 (contiguous) ----
    if (wmId >= IDM_VIEW1 && wmId <= IDM_VIEW40) {
        static_assert(IDM_VIEW40 == IDM_VIEW1 + 39, "IDM_VIEW range must be contiguous");
        MenuStandardView(static_cast<size_t>(wmId - IDM_VIEW1 + 1));
        return true;
    }

    // ---- Dynamic orbit slots (0..29) ----
    if (wmId >= IDM_VIEW_DYNAMIC_ORBIT && wmId < IDM_VIEW_DYNAMIC_ORBIT + kMaxDynamic) {
        const size_t index = static_cast<size_t>(wmId - IDM_VIEW_DYNAMIC_ORBIT);
        if (index < gSavedLocations.size()) {
            ActivateSavedOrbit(index);
        }
        return true;
    }

    // ---- Dynamic imag slots (0..29) ----
    if (wmId >= IDM_VIEW_DYNAMIC_IMAG && wmId < IDM_VIEW_DYNAMIC_IMAG + kMaxDynamic) {
        const size_t index = static_cast<size_t>(wmId - IDM_VIEW_DYNAMIC_IMAG);
        if (index < gImaginaLocations.size()) {
            ActivateImagina(index);
        }
        return true;
    }

    return false;
}

void
MainWindow::ActivateSavedOrbit(size_t index)
{
    ClearMenu(LoadSubMenu);

    const auto ptz = gSavedLocations[index].ptz;
    const auto num_iterations = gSavedLocations[index].num_iterations;
    const auto antialiasing = gSavedLocations[index].antialiasing;

    gFractal->RecenterViewCalc(ptz);
    gFractal->SetNumIterations<IterTypeFull>(num_iterations);
    gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, antialiasing);
    PaintAsNecessary();
}

void
MainWindow::ActivateImagina(size_t index)
{
    const auto &entry = gImaginaLocations[index];

    LoadRefOrbit(CompressToDisk::MaxCompressionImagina, entry.Settings, entry.Filename);
    ClearMenu(ImaginaMenu);
}

// Top-level router expanded in Phase 3
bool
MainWindow::HandleCommand(int wmId)
{
    if (HandleCommandRange(wmId))
        return true;
    if (HandleAlgCommand(wmId))
        return true;
    if (HandleCommandTable(wmId))
        return true;
    return false;
}

bool
MainWindow::HandleAlgCommand(int wmId)
{
    if (const auto *e = FindAlgForCmd(wmId)) {
        gFractal->SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(e->alg));
        FractalShark::DynamicPopupMenu::SetCurrentRenderAlgorithmId(wmId);

        if (gPopupMenu) {
            auto popup = FractalShark::DynamicPopupMenu::GetPopup(gPopupMenu.get());
            FractalShark::DynamicPopupMenu::ApplyRenderAlgorithmRadioChecks(popup, wmId);
        }
        return true;
    }
    return false;
}

LRESULT
MainWindow::WndProc(UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) {
        case WM_COMMAND: {
            const int wmId = LOWORD(wParam);
            const int wmEvent = HIWORD(wParam);
            (void)wmEvent;

            if (HandleCommand(wmId)) {
                return 0;
            }

            wchar_t buf[256];
            swprintf_s(buf,
                       L"Unknown WM_COMMAND.\n\nwmId = %d (0x%X)\nwmEvent = %d (0x%X)\n",
                       wmId,
                       wmId,
                       wmEvent,
                       wmEvent);

            ::MessageBoxW(hWnd, buf, L"Unknown menu item", MB_OK | MB_APPLMODAL);

            return 0;
        }


        case WM_SIZE: {
            if (gFractal) {
                gFractal->ResetDimensions(LOWORD(lParam), HIWORD(lParam));
                PaintAsNecessary();
            }
            return 0;
        }

        case WM_CONTEXTMENU: {
            // WM_CONTEXTMENU:
            //  - If invoked by mouse: lParam contains screen coords (x,y)
            //  - If invoked by keyboard (Shift+F10 / menu key): lParam == -1
            POINT ptScreen{};

            if (lParam == static_cast<LPARAM>(-1)) {
                // Keyboard invocation: use current cursor position as the popup anchor.
                ::GetCursorPos(&ptScreen);
            } else {
                // Mouse invocation: lParam is already in SCREEN coordinates.
                ptScreen.x = GET_X_LPARAM(lParam);
                ptScreen.y = GET_Y_LPARAM(lParam);
            }

            // Show popup (screen coords)
            auto popup = FractalShark::DynamicPopupMenu::GetPopup(gPopupMenu.get());

            // Sync radio checks on the existing menu instance
            FractalShark::DynamicPopupMenu::ApplyRenderAlgorithmRadioChecks(
                popup, FractalShark::DynamicPopupMenu::GetCurrentRenderAlgorithmId());

            ::TrackPopupMenu(popup, 0, ptScreen.x, ptScreen.y, 0, hWnd, nullptr);

            // Persist menu location as CLIENT coords on the instance (used by Center/Zoom commands).
            POINT ptClient = ptScreen;
            ::ScreenToClient(hWnd, &ptClient);
            lastMenuPtClient_ = ptClient;

            return 0;
        }

        case WM_LBUTTONDOWN: {
            if (gWindowed && IsDownAlt()) {
                ::PostMessage(hWnd, WM_NCLBUTTONDOWN, HTCAPTION, lParam);
                return 0;
            }

            if (lButtonDown)
                return 0;

            lButtonDown = true;
            dragBoxX1 = GET_X_LPARAM(lParam);
            dragBoxY1 = GET_Y_LPARAM(lParam);
            prevX1 = prevY1 = -1;

            ::SetCapture(hWnd);
            return 0;
        }

        case WM_LBUTTONUP: {
            if (!lButtonDown || IsDownAlt()) {
                if (::GetCapture() == hWnd)
                    ::ReleaseCapture();
                return 0;
            }

            // release capture early so we don’t get stuck captured on exceptions/returns
            if (::GetCapture() == hWnd)
                ::ReleaseCapture();

            lButtonDown = false;
            prevX1 = prevY1 = -1;

            RECT newView{};
            const bool maintainAspect = !IsDownShift();

            if (maintainAspect) {
                RECT windowRect;
                ::GetClientRect(hWnd, &windowRect);
                const double ratio = double(windowRect.right) / double(windowRect.bottom);

                newView.left = dragBoxX1;
                newView.top = dragBoxY1;
                newView.bottom = GET_Y_LPARAM(lParam);
                newView.right =
                    LONG(double(newView.left) + ratio * (double(newView.bottom) - double(newView.top)));
            } else {
                newView.left = dragBoxX1;
                newView.top = dragBoxY1;
                newView.right = GET_X_LPARAM(lParam);
                newView.bottom = GET_Y_LPARAM(lParam);
            }

            if (gFractal && gFractal->RecenterViewScreen(newView)) {
                if (maintainAspect)
                    gFractal->SquareCurrentView();
                PaintAsNecessary();
            }

            return 0;
        }

        case WM_CANCELMODE:
        case WM_CAPTURECHANGED: {
            if (!lButtonDown)
                return 0;

            // erase any existing inverted rect
            if (prevX1 != -1 || prevY1 != -1) {
                HDC dc = ::GetDC(hWnd);
                RECT rect{dragBoxX1, dragBoxY1, prevX1, prevY1};
                ::InvertRect(dc, &rect);
                ::ReleaseDC(hWnd, dc);
            }

            lButtonDown = false;
            prevX1 = prevY1 = -1;

            if (::GetCapture() == hWnd)
                ::ReleaseCapture();

            return 0;
        }

        case WM_MOUSEMOVE: {
            if (lButtonDown == false) {
                return 0;
            }

            HDC dc = GetDC(hWnd);
            RECT rect;

            // Erase the previous rectangle.
            if (prevX1 != -1 || prevY1 != -1) {
                rect.left = dragBoxX1;
                rect.top = dragBoxY1;
                rect.right = prevX1;
                rect.bottom = prevY1;

                InvertRect(dc, &rect);
            }

            if (IsDownShift() == false) {
                RECT windowRect;
                GetClientRect(hWnd, &windowRect);
                double ratio = (double)windowRect.right / (double)windowRect.bottom;

                // Note order is important.
                rect.left = dragBoxX1;
                rect.top = dragBoxY1;
                rect.bottom = GET_Y_LPARAM(lParam);
                rect.right = (long)((double)rect.left +
                                    (double)ratio * (double)((double)rect.bottom - (double)rect.top));

                prevX1 = rect.right;
                prevY1 = rect.bottom;
            } else {
                rect.left = dragBoxX1;
                rect.top = dragBoxY1;
                rect.right = GET_X_LPARAM(lParam);
                rect.bottom = GET_Y_LPARAM(lParam);

                prevX1 = GET_X_LPARAM(lParam);
                prevY1 = GET_Y_LPARAM(lParam);
            }

            InvertRect(dc, &rect);

            ReleaseDC(hWnd, dc);
            break;
        }

        case WM_MOUSEWHEEL: {
            // Zoom in or out when mouse wheel is scrolled.
            POINT pt;
            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);

            // convert to client coordinates
            ScreenToClient(hWnd, &pt);

            if (GET_WHEEL_DELTA_WPARAM(wParam) > 0) {
                gFractal->Zoom2(pt.x, pt.y, -.3);
            } else {
                // gFractal->Zoom(pt.x, pt.y, 0.3);
                gFractal->Zoom(0.3);
            }

            PaintAsNecessary();
            return 0;
        }

        case WM_PAINT: {
            PaintAsNecessary();

            PAINTSTRUCT ps;
            BeginPaint(hWnd, &ps);
            EndPaint(hWnd, &ps);
            return 0;
        }

        case WM_DESTROY: {
            PostQuitMessage(0);
            return 0;
        }

        case WM_CHAR: {
            HandleKeyDown(message, wParam, lParam);
            PaintAsNecessary();
            return 0;
        }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

void
MainWindow::MenuGoBack()
{
    if (gFractal->Back() == true) {
        PaintAsNecessary();
    }
}

void
MainWindow::MenuStandardView(size_t i)
{
    gFractal->View(i);
    PaintAsNecessary();
}

void
MainWindow::MenuSquareView()
{
    gFractal->SquareCurrentView();
    PaintAsNecessary();
}

void
MainWindow::MenuCenterView(int x, int y)
{
    gFractal->CenterAtPoint(x, y);
    PaintAsNecessary();
}

void
MainWindow::MenuZoomIn(POINT mousePt)
{
    gFractal->Zoom(mousePt.x, mousePt.y, -.45);
    PaintAsNecessary();
}

void
MainWindow::MenuZoomOut(POINT mousePt)
{
    gFractal->Zoom(mousePt.x, mousePt.y, 1);
    PaintAsNecessary();
}

void
MainWindow::MenuRepainting()
{
    gFractal->ToggleRepainting();
    PaintAsNecessary();
}

void
MainWindow::MenuWindowed(bool square)
{
    if (gWindowed == false) {
        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true) {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SendMessage(hWnd, WM_SYSCOMMAND, SC_RESTORE, 0);

        if (temporaryChange == true) {
            gFractal->SetRepaint(true);
        }

        RECT rect;
        GetWindowRect(hWnd, &rect);

        if (square) {
            auto width = std::min((rect.right + rect.left) / 2, (rect.bottom + rect.top) / 2);
            // width /= 2;
            SetWindowPos(hWnd,
                         HWND_NOTOPMOST,
                         (rect.right + rect.left) / 2 - width / 2,
                         (rect.bottom + rect.top) / 2 - width / 2,
                         width,
                         width,
                         SWP_SHOWWINDOW);
        } else {
            SetWindowPos(hWnd,
                         HWND_NOTOPMOST,
                         (rect.right - rect.left) / 4,
                         (rect.bottom - rect.top) / 4,
                         (rect.right - rect.left) / 2,
                         (rect.bottom - rect.top) / 2,
                         SWP_SHOWWINDOW);
        }
        gWindowed = true;

        if (gFractal) {
            RECT rt;
            GetClientRect(hWnd, &rt);
            gFractal->ResetDimensions(rt.right, rt.bottom);
        }
    } else {
        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true) {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SetWindowPos(hWnd, HWND_NOTOPMOST, 0, 0, width, height, SWP_SHOWWINDOW);
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

        if (temporaryChange == true) {
            gFractal->SetRepaint(true);
        }

        gWindowed = false;

        if (gFractal) {
            RECT rt;
            GetClientRect(hWnd, &rt);
            gFractal->ResetDimensions(rt.right, rt.bottom);
        }
    }

    PaintAsNecessary();
}

void
MainWindow::MenuMultiplyIterations(double factor)
{
    if (gFractal->GetIterType() == IterTypeEnum::Bits32) {
        uint64_t curIters = gFractal->GetNumIterations<uint32_t>();
        curIters = (uint64_t)((double)curIters * (double)factor);
        gFractal->SetNumIterations<uint32_t>(curIters);
    } else {
        uint64_t curIters = gFractal->GetNumIterations<uint64_t>();
        curIters = (uint64_t)((double)curIters * (double)factor);
        gFractal->SetNumIterations<uint64_t>(curIters);
    }

    PaintAsNecessary();
}

void
MainWindow::MenuResetIterations()
{
    gFractal->ResetNumIterations();
    PaintAsNecessary();
}

void
MainWindow::MenuGetCurPos()
{
    constexpr size_t numBytes = 4 * 1024 * 1024;

    BOOL ret = OpenClipboard(hWnd);
    if (ret == 0) {
        MessageBox(hWnd,
                   L"Opening the clipboard failed.  Another program must be using it.",
                   L"",
                   MB_OK | MB_APPLMODAL);
        return;
    }

    ret = EmptyClipboard();
    if (ret == 0) {
        MessageBox(hWnd,
                   L"Emptying the clipboard of its current contents failed.  Make sure no other "
                   L"programs are using it.",
                   L"",
                   MB_OK | MB_APPLMODAL);
        CloseClipboard();
        return;
    }

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, numBytes);
    if (hData == nullptr) {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK | MB_APPLMODAL);
        CloseClipboard();
        return;
    }

    char *mem = (char *)GlobalLock(hData);
    if (mem == nullptr) {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK | MB_APPLMODAL);
        CloseClipboard();
        return;
    }

    std::string shortStr, longStr;
    gFractal->GetRenderDetails(shortStr, longStr);

    // Append temp2 to mem without overrunning the buffer
    // using strncat.
    mem[0] = 0;
    strncpy(mem, longStr.data(), numBytes - 1);

    GlobalUnlock(hData);

    //
    // This is not a memory leak - we don't "free" hData.
    //

    HANDLE clpData = SetClipboardData(CF_TEXT, hData);
    if (clpData == nullptr) {
        MessageBox(hWnd,
                   L"Adding the data to the clipboard failed.  You are probably very low on memory.  "
                   L"Try closing other programs or restarting your computer.",
                   L"",
                   MB_OK | MB_APPLMODAL);
        CloseClipboard();
        return;
    }

    CloseClipboard();

    if (shortStr.length() < 5000) {
        ::MessageBoxA(hWnd, shortStr.c_str(), "", MB_OK | MB_APPLMODAL);
    } else {
        ::MessageBoxA(hWnd, "Location copied to clipboard.", "", MB_OK | MB_APPLMODAL);
    }
}

void
MainWindow::MenuPaletteRotation()
{
    POINT OrgPos, CurPos;
    GetCursorPos(&OrgPos);

    for (;;) {
        gFractal->RotateFractalPalette(10);
        gFractal->DrawFractal(false);
        GetCursorPos(&CurPos);
        if (abs(CurPos.x - OrgPos.x) > 5 || abs(CurPos.y - OrgPos.y) > 5) {
            break;
        }
    }

    gFractal->ResetFractalPalette();
    gFractal->DrawFractal(false);
}

void
MainWindow::MenuPaletteType(FractalPalette type)
{
    gFractal->UsePaletteType(type);
    if (type == FractalPalette::Default) {
        gFractal->UsePalette(8);
        gFractal->SetPaletteAuxDepth(0);
    }
    gFractal->DrawFractal(false);
}

void
MainWindow::MenuPaletteDepth(int depth)
{
    gFractal->UsePalette(depth);
    gFractal->DrawFractal(false);
}

void
MainWindow::MenuCreateNewPalette()
{
    gFractal->CreateNewFractalPalette();
    gFractal->UsePaletteType(FractalPalette::Random);
    gFractal->DrawFractal(false);
}

void
MainWindow::MenuSaveCurrentLocation()
{
    int response = ::MessageBox(hWnd, L"Scale dimensions to maximum?", L"Choose!", MB_YESNO);
    char filename[256];
    SYSTEMTIME time_struct;
    GetLocalTime(&time_struct);
    sprintf(filename,
            "output_%d_%d_%d_%d_%d_%d.bmp",
            time_struct.wYear,
            time_struct.wMonth,
            time_struct.wDay,
            time_struct.wHour,
            time_struct.wMinute,
            time_struct.wSecond);

    size_t x, y;
    if (response == IDYES) {
        x = gFractal->GetRenderWidth();
        y = gFractal->GetRenderHeight();
        if (x > y) {
            y = (int)((double)16384.0 / (double)((double)x / (double)y));
            x = 16384;
        } else if (x < y) {
            x = (int)((double)16384.0 / (double)((double)y / (double)x));
            y = 16384;
        }
    } else {
        x = gFractal->GetRenderWidth();
        y = gFractal->GetRenderHeight();
    }

    std::stringstream ss;
    ss << x << " ";
    ss << y << " ";
    ss << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
    ss << gFractal->GetMinX() << " ";
    ss << gFractal->GetMinY() << " ";
    ss << gFractal->GetMaxX() << " ";
    ss << gFractal->GetMaxY() << " ";
    ss << gFractal->GetNumIterations<IterTypeFull>() << " ";
    ss << gFractal->GetGpuAntialiasing() << " ";
    // ss << gFractal->GetIterationPrecision() << " ";
    ss << "FractalTrayDestination";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());

    MessageBox(nullptr, ws.c_str(), L"location", MB_OK | MB_APPLMODAL);

    FILE *file = fopen("locations.txt", "at+");
    fprintf(file, "%s\r\n", s.c_str());
    fclose(file);
}

void
MainWindow::MenuLoadCurrentLocation()
{
    std::ifstream infile("locations.txt");
    HMENU hSubMenu = CreatePopupMenu();

    size_t index = 0;

    gSavedLocations.clear();

    for (;;) {
        SavedLocation loc(infile);
        if (infile.rdstate() != std::ios_base::goodbit) {
            break;
        }

        // Convert loc.description to a wstring:
        std::string s = loc.description;
        const std::wstring ws(s.begin(), s.end());

        gSavedLocations.push_back(loc);
        AppendMenu(hSubMenu, MF_STRING, IDM_VIEW_DYNAMIC_ORBIT + index, ws.c_str());
        index++;

        // Limit the number of locations we show.
        if (index > 30) {
            break;
        }
    }

    POINT point;
    GetCursorPos(&point);
    TrackPopupMenu(hSubMenu, 0, point.x, point.y, 0, hWnd, nullptr);

    DestroyMenu(hSubMenu);
}

// Subclass procedure for the edit controls
LRESULT
MainWindow::EditSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    if (uMsg == WM_KEYDOWN) {
        if ((wParam == 'A') && (GetKeyState(VK_CONTROL) & 0x8000)) {
            // CTRL+A pressed, select all text
            SendMessage(hwnd, EM_SETSEL, 0, -1);
            return 0;
        }
    }
    // Call the original window procedure for the edit control
    WNDPROC originalProc = (WNDPROC)GetWindowLongPtr(hwnd, GWLP_USERDATA);
    return CallWindowProc(originalProc, hwnd, uMsg, wParam, lParam);
}

void
MainWindow::MenuLoadEnterLocation()
{
    // Create a window with three text boxes for entering the location.
    // The text boxes are for the real, imaginary, and zoom values.
    // Store the results from the text boxes in three strings.
    // Set the fractal to the new location and repaint.
    // Include OK and Cancel buttons.
    // Use the IDD_DIALOG_LOCATION resource

    // Define EnterLocationDialogProc:
    // This is a dialog box procedure that handles the dialog box messages.
    // It should handle WM_COMMAND, and WM_CLOSE.
    // WM_COMMAND should handle the OK and Cancel buttons.
    // If the OK button is pressed, it should store the values in the text boxes
    // in the strings, and then call EndDialog(hWnd, 0).
    // If the Cancel button is pressed, it should call EndDialog(hWnd, 1).
    // WM_CLOSE should call EndDialog(hWnd, 1).

    struct Values {
        Values() : real(""), imag(""), zoom(""), num_iterations(0) {}

        std::string real, imag, zoom;
        IterTypeFull num_iterations;

        std::string
        ItersToString() const
        {
            return std::to_string(num_iterations);
        }

        void
        StringToIters(std::string new_iters)
        {
            num_iterations = std::stoull(new_iters);
        }
    };

    Values values;

    // Store existing location in the strings.
    HighPrecision minX = gFractal->GetMinX();
    HighPrecision minY = gFractal->GetMinY();
    HighPrecision maxX = gFractal->GetMaxX();
    HighPrecision maxY = gFractal->GetMaxY();

    PointZoomBBConverter pz{minX, minY, maxX, maxY};
    values.real = pz.GetPtX().str();
    values.imag = pz.GetPtY().str();
    values.zoom = pz.GetZoomFactor().str();
    values.num_iterations = gFractal->GetNumIterations<IterTypeFull>();

    auto EnterLocationDialogProc = [](HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) -> INT_PTR {
        // TODO: static?  This is surely not the right way to do this.
        static Values *values = nullptr;

        switch (message) {
            case WM_INITDIALOG: {
                // Get the pointer to the Values struct from lParam.
                values = (Values *)lParam;

                RECT rcDlg, rcScreen;
                int x, y;

                // Get the dimensions of the dialog box
                GetWindowRect(hDlg, &rcDlg);

                // Get the dimensions of the screen
                GetClientRect(GetDesktopWindow(), &rcScreen);

                // Calculate the position to center the dialog box
                x = (rcScreen.right - (rcDlg.right - rcDlg.left)) / 2;
                y = (rcScreen.bottom - (rcDlg.bottom - rcDlg.top)) / 2;

                // Move the dialog box to the calculated position
                SetWindowPos(hDlg, NULL, x, y, 0, 0, SWP_NOZORDER | SWP_NOSIZE);

                // Set the text in the text boxes to the values in the strings.
                SetDlgItemTextA(hDlg, IDC_EDIT_REAL, values->real.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_IMAG, values->imag.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_ZOOM, values->zoom.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_ITERATIONS, values->ItersToString().c_str());

                // Subclass the edit controls
                HWND hEditReal = GetDlgItem(hDlg, IDC_EDIT_REAL);
                HWND hEditImag = GetDlgItem(hDlg, IDC_EDIT_IMAG);
                HWND hEditZoom = GetDlgItem(hDlg, IDC_EDIT_ZOOM);
                HWND hEditIterations = GetDlgItem(hDlg, IDC_EDIT_ITERATIONS);

                auto OriginalEditProcReal =
                    (WNDPROC)SetWindowLongPtr(hEditReal, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditReal, GWLP_USERDATA, (LONG_PTR)OriginalEditProcReal);

                auto OriginalEditProcImag =
                    (WNDPROC)SetWindowLongPtr(hEditImag, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditImag, GWLP_USERDATA, (LONG_PTR)OriginalEditProcImag);

                auto OriginalEditProcZoom =
                    (WNDPROC)SetWindowLongPtr(hEditZoom, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditZoom, GWLP_USERDATA, (LONG_PTR)OriginalEditProcZoom);

                auto OriginalEditProcIterations =
                    (WNDPROC)SetWindowLongPtr(hEditIterations, GWLP_WNDPROC, (LONG_PTR)EditSubclassProc);
                SetWindowLongPtr(hEditIterations, GWLP_USERDATA, (LONG_PTR)OriginalEditProcIterations);

                break;
            }

            case WM_COMMAND: {
                if (LOWORD(wParam) == IDOK) {
                    // Get the text from the text boxes.
                    // Store the text in the strings.
                    // Call EndDialog(hDlg, 0);

                    // First, figure out how many bytes are needed
                    // to store the text in the text boxes.
                    int len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_REAL));
                    values->real.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_REAL), &values->real[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_IMAG));
                    values->imag.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_IMAG), &values->imag[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_ZOOM));
                    values->zoom.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_ZOOM), &values->zoom[0], len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_ITERATIONS));
                    std::string new_iters;
                    new_iters.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_ITERATIONS), &new_iters[0], len + 1);
                    values->StringToIters(new_iters);

                    EndDialog(hDlg, 0);
                    return TRUE;
                } else if (LOWORD(wParam) == IDCANCEL) {
                    // Call EndDialog(hDlg, 1);
                    EndDialog(hDlg, 1);
                    return TRUE;
                }
                break;
            }
            case WM_CLOSE: {
                // Call EndDialog(hDlg, 1);
                EndDialog(hDlg, 1);
                return TRUE;
            }
        }

        return FALSE;
    };

    // (hInst, MAKEINTRESOURCE(IDD_DIALOG_LOCATION), hWnd, Dlgproc);
    LPARAM lParam = reinterpret_cast<LPARAM>(&values);
    auto OkOrCancel = DialogBoxParam(
        hInst, MAKEINTRESOURCE(IDD_DIALOG_LOCATION), hWnd, EnterLocationDialogProc, lParam);

    if (values.real.empty() || values.imag.empty() || values.zoom.empty()) {
        return;
    }

    // If OkOrCancel is 1, return.
    if (OkOrCancel == 1) {
        return;
    }

    // Convert the strings to HighPrecision and set the fractal to the new location.
    HighPrecision::defaultPrecisionInBits(Fractal::MaxPrecisionLame);
    HighPrecision realHP(values.real);
    HighPrecision imagHP(values.imag);
    HighPrecision zoomHP(values.zoom);

    PointZoomBBConverter pz2{realHP, imagHP, zoomHP};
    gFractal->RecenterViewCalc(pz2);
    gFractal->SetNumIterations<IterTypeFull>(values.num_iterations);
    PaintAsNecessary();
}

void
MainWindow::MenuSaveBMP()
{
    gFractal->SaveCurrentFractal(L"", true);
}

void
MainWindow::MenuSaveHiResBMP()
{
    gFractal->SaveHiResFractal(L"");
}

void
MainWindow::MenuSaveItersAsText()
{
    gFractal->SaveItersAsText(L"");
}

void
MainWindow::BenchmarkMessage(size_t milliseconds)
{
    std::stringstream ss;
    ss << std::string("Time taken in ms: ") << milliseconds << ".";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());
    MessageBox(hWnd, ws.c_str(), L"", MB_OK | MB_APPLMODAL);
}

void
MainWindow::ClearMenu(HMENU &menu)
{
    if (menu != nullptr) {
        DestroyMenu(menu);
        menu = nullptr;
        gImaginaLocations.clear();
    }
}

void
MainWindow::LoadRefOrbit(CompressToDisk compressToDisk,
                         ImaginaSettings loadSettings,
                         std::wstring filename)
{

    RecommendedSettings settings{};
    gFractal->LoadRefOrbit(&settings, compressToDisk, loadSettings, filename);

    PaintAsNecessary();

    // Restore only "Auto".  If the savefile changes our iteration type
    // to 64-bit, just leave it.  The "Auto" concept is kind of weird in
    // this context.
    if (settings.GetRenderAlgorithm() == RenderAlgorithmEnum::AUTO) {
        gFractal->SetRenderAlgorithm(settings.GetRenderAlgorithm());
    }
}

void
MainWindow::MenuLoadImagDyn(ImaginaSettings loadSettings)
{
    ClearMenu(ImaginaMenu);

    ImaginaMenu = CreatePopupMenu();
    size_t index = 0;

    std::vector<std::wstring> imagFiles;
    // Find all files with the extension .im in current directory.
    // Add filenames to imagFiles.

    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile(L"*.im", &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            imagFiles.push_back(FindFileData.cFileName);
        } while (FindNextFile(hFind, &FindFileData) != 0);

        FindClose(hFind);
    }

    // Even if no files found, just run the following so we get
    // an empty menu.

    for (const auto &imagFile : imagFiles) {

        gImaginaLocations.push_back({imagFile, loadSettings});

        AppendMenu(ImaginaMenu, MF_STRING, IDM_VIEW_DYNAMIC_IMAG + index, imagFile.c_str());
        index++;

        if (index > 30) {
            break;
        }
    }

    if (loadSettings == ImaginaSettings::ConvertToCurrent) {
        AppendMenu(ImaginaMenu, MF_STRING, IDM_LOAD_IMAGINA_DLG, L"Load from file (Match)...");
    } else if (loadSettings == ImaginaSettings::UseSaved) {
        AppendMenu(ImaginaMenu, MF_STRING, IDM_LOAD_IMAGINA_DLG_SAVED, L"Load from file (Use Saved)...");
    } else {
    }

    POINT point;
    GetCursorPos(&point);
    TrackPopupMenu(ImaginaMenu, 0, point.x, point.y, 0, hWnd, nullptr);

    DestroyMenu(ImaginaMenu);
}

void
MainWindow::MenuSaveImag(CompressToDisk compression)
{
    std::wstring filename = OpenFileDialog(OpenBoxType::Save);
    if (filename.empty()) {
        return;
    }

    gFractal->SaveRefOrbit(compression, filename);
}

void
MainWindow::MenuDiffImag()
{

    std::wstring outFile = OpenFileDialog(OpenBoxType::Save);
    if (outFile.empty()) {
        return;
    }

    // Open two files, both must exist
    std::wstring filename1 = OpenFileDialog(OpenBoxType::Open);
    if (filename1.empty()) {
        return;
    }

    std::wstring filename2 = OpenFileDialog(OpenBoxType::Open);
    if (filename2.empty()) {
        return;
    }

    gFractal->DiffRefOrbits(CompressToDisk::MaxCompressionImagina, outFile, filename1, filename2);
}

void
MainWindow::MenuLoadImag(ImaginaSettings loadSettings, CompressToDisk compression)
{

    std::wstring filename = OpenFileDialog(OpenBoxType::Open);
    if (filename.empty()) {
        return;
    }

    LoadRefOrbit(compression, loadSettings, filename);
}

void
MainWindow::MenuAlgHelp()
{
    // This message box shows some help related to the algorithms.
    ::MessageBox(
        nullptr,
        L"Algorithms\r\n"
        L"\r\n"
        L"- As a general recommendation, choose AUTO.  Auto will render the fractal using "
        L"direct 32-bit evaluation at the lowest zoom depths. "
        L"From 1e4 to 1e9, it uses perturbation + 32-bit floating point. "
        L"From 1e9 to 1e34, it uses perturbation + 32-bit + linear approximation.  "
        L"Past that, it uses perturbation a 32-bit \"high dynamic range\" implementation, "
        L"which simply stores the exponent in a separate integer.\r\n"
        L"\r\n"
        L"- If you try rendering \"hard\" points, you may find that the 32-bit implementations "
        L"are not accurate enough.  In this case, you can try the 64-bit implementations.  "
        L"You may also find the 2x32 implementations to be faster than the 1x64."
        L"Generally, it's probably easiest to use the 32-bit implementations, and only "
        L"switch to the 64-bit implementations when you need to.\r\n"
        L"\r\n"
        L"Note that professional/high-end chips offer superior 64-bit performance, so if you have one "
        L"of those, you may find that the 64-bit implementations work well.  Most consumer GPUs offer"
        L"poor 64-bit performance\r\n",
        L"Algorithms",
        MB_OK);
}

void
MainWindow::MenuViewsHelp()
{
    ::MessageBox(nullptr,
                 L"Views\r\n"
                 L"\r\n"
                 L"The purpose of these is simply to make it easy to navigate to\r\n"
                 L"some interesting locations.\r\n",
                 L"Views",
                 MB_OK);
}

void
MainWindow::MenuShowHotkeys()
{
    // Shows some basic help + hotkeys as defined in HandleKeyDown
    ::MessageBox(
        nullptr,
        L"Hotkeys\r\n"
        L"\r\n"
        L"Navigation\r\n"
        L"a - Autozoom using averaging heuristic.  Buggy.  Hold CTRL to abort.\r\n"
        L"A - Autozoom by zooming in on the highest iteration count point.  Buggy.  Hold CTRL to "
        L"abort.\r\n"
        L"b - Go back to the previous view\r\n"
        L"c - Center the view at the current mouse position\r\n"
        L"C - Center the view at the current mouse position + recalculate reference orbit\r\n"
        L"z - Zoom in predefined amount\r\n"
        L"Z - Zoom out predefined amount\r\n"
        L"Left click/drag - Zoom in\r\n"
        L"\r\n"
        L"Recaluating and Benchmarking\r\n"
        L"I - Clear medium-res perturbation results, recalculate, and benchmark\r\n"
        L"i - Recalculate and benchmark current display, reusing perturbation results\r\n"
        L"O - Clear high-res perturbation results, recalculate, and benchmark\r\n"
        L"o - Recalculate and benchmark current display, reusing perturbation results\r\n"
        L"P - Clear all perturbation results and recalculate\r\n"
        L"p - Recalculate current display, reusing perturbation results\r\n"
        L"R - Clear all perturbation results and recalculate\r\n"
        L"r - Recalculate current display, reusing perturbation results\r\n"
        L"\r\n"
        L"Reference Compression\r\n"
        L"e - Clear all perturbation results, reset error exponent to 19 (default).  Recalculate.\r\n"
        L"q - Decrease intermediate orbit compression: less error, more memory. Recalculate.\r\n"
        L"Q - Increase intermediate orbit compression: more error, less memory. Recalculate.\r\n"
        L"w - Decrease reference compression: less error, more memory. Recalculate.\r\n"
        L"W - Increase reference compression: more error, less memory. Recalculate.\r\n"
        L"\r\n"
        L"Linear Approximation parameters, adjustments by powers of two\r\n"
        L"H - Decrease LA Threshold Scale exponents.  More accurate/slower per-pixel\r\n"
        L"h - Increase LA Threshold Scale exponents.  Less accurate/faster per-pixel\r\n"
        L"J - Decrease LA period detection exponents.  Less memory/slower per-pixel\r\n"
        L"j - Increase LA period detection exponents.  More memory/faster per-pixel\r\n"
        L"\r\n"
        L"Palettes\r\n"
        L"T - Use prior auxiliary palette depth (mul/div iteration count by 2)\r\n"
        L"t - Use next auxiliary palette depth (mul/div iteration count by 2)\r\n"
        L"D - Create and use new random palette\r\n"
        L"d - Use next palette lookup table depth\r\n"
        L"\r\n"
        L"Iterations\r\n"
        L"Use these keys to increase/decrease the number of iterations used to calculate the "
        L"fractal.\r\n"
        L"= - Multiply max iterations by 24\r\n"
        L"- - Multiply max iterations by 2/3\r\n"
        L"\r\n"
        L"Misc\r\n"
        L"CTRL - Press and hold to abort autozoom\r\n"
        L"ALT - Press, click/drag to move window when in windowed mode\r\n"
        L"Right click - popup menu\r\n",
        L"",
        MB_OK);
}

void
MainWindow::PaintAsNecessary()
{
    RECT rt;
    GetClientRect(hWnd, &rt);

    if (rt.left == 0 && rt.right == 0 && rt.top == 0 && rt.bottom == 0) {
        return;
    }

    if (gFractal != nullptr) {
        gFractal->CalcFractal(false);
    }
}

// These functions are used to create a minidump when the program crashes.
typedef BOOL(WINAPI *MINIDUMPWRITEDUMP)(HANDLE hProcess,
                                        DWORD dwPid,
                                        HANDLE hFile,
                                        MINIDUMP_TYPE DumpType,
                                        CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
                                        CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
                                        CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

void
MainWindow::create_minidump(struct _EXCEPTION_POINTERS *apExceptionInfo)
{
    HMODULE mhLib = ::LoadLibrary(_T("dbghelp.dll"));
    if (mhLib == nullptr) {
        return;
    }

    MINIDUMPWRITEDUMP pDump = (MINIDUMPWRITEDUMP)::GetProcAddress(mhLib, "MiniDumpWriteDump");

    HANDLE hFile = ::CreateFile(_T("core.dmp"),
                                GENERIC_WRITE,
                                FILE_SHARE_WRITE,
                                nullptr,
                                CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);

    _MINIDUMP_EXCEPTION_INFORMATION ExInfo;
    ExInfo.ThreadId = ::GetCurrentThreadId();
    ExInfo.ExceptionPointers = apExceptionInfo;
    ExInfo.ClientPointers = FALSE;

    pDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &ExInfo, nullptr, nullptr);
    ::CloseHandle(hFile);
}

LONG WINAPI
MainWindow::unhandled_handler(struct _EXCEPTION_POINTERS *apExceptionInfo)
{
    create_minidump(apExceptionInfo);
    return EXCEPTION_CONTINUE_SEARCH;
}

