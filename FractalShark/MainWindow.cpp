#include "StdAfx.h"

#include "MainWindow.h"
#include "resource.h"
#include "Fractal.h"
#include "FractalTest.h"
#include "JobObject.h"
#include "OpenGLContext.h"
#include "PngParallelSave.h"

#include <minidumpapiset.h>
#include <mpir.h>

struct MainWindow::SavedLocation
{
    SavedLocation(std::ifstream &infile) {
        // Read minX, minY, maxX, maxY, num_iterations, antialiasing from infile
        // To read minX, read a string and convert to HighPrecision

        infile >> width;
        infile >> height;

        infile >> minX;
        infile >> minY;
        infile >> maxX;
        infile >> maxY;

        infile >> num_iterations;
        infile >> antialiasing;

        std::getline(infile, description);
    }

    HighPrecision minX, maxX, minY, maxY;
    size_t width, height;
    IterTypeFull num_iterations;
    uint32_t antialiasing;
    std::string description;
};

MainWindow::MainWindow(HINSTANCE hInstance, int nCmdShow) {
    gJobObj = std::make_unique<JobObject>();
    mpf_set_default_prec(256);

    // Create a dump file whenever the gateway crashes only on windows
    SetUnhandledExceptionFilter(unhandled_handler);

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // Perform application initialization:
    hWnd = InitInstance(hInstance, nCmdShow);
    if (!hWnd) {
        throw std::runtime_error("Failed to create window");
    }
}

MainWindow::~MainWindow() {
    // Cleanup
    UnInit();

    gJobObj.reset();

#ifdef _DEBUG
    // We have a 16-byte leak thanks to Windows no matter what we do.

    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif
}

void MainWindow::MainWindow::DrawFractalShark() {
    auto glContext = std::make_unique<OpenGlContext>(hWnd);
    if (!glContext->IsValid()) {
        return;
    }

    glContext->DrawFractalShark(hWnd);
}

void MainWindow::MainWindow::DrawFractalSharkGdi(int nCmdShow) {
    //image.loadImage("FractalShark.png");
    // Get a pointer to the binary data in the IDB_PNG1 resource
    HRSRC hRes = FindResource(hInst, MAKEINTRESOURCE(IDB_PNG_SPLASH), L"PNG");
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
        StretchBlt(hdc, 0, 0, windowWidth, windowHeight, hdcMem, 0, 0, image.width(), image.height(), SRCCOPY);
    } else {
        // Center the image
        BitBlt(hdc, startX, startY, image.width(), image.height(), hdcMem, 0, 0, SRCCOPY);
    }

    // Set the window as opaque
    SetLayeredWindowAttributes(hWnd, RGB(0, 0, 0), 255, LWA_ALPHA);

    //ShowWindow(hWnd, nCmdShow);
    //ShowWindow(hWnd, SW_RESTORE);

    // Clean up
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(hWnd, hdc);
}

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM MainWindow::MyRegisterClass(HINSTANCE hInstance) {
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
HWND MainWindow::InitInstance(HINSTANCE hInstance, int nCmdShow) { // Store instance handle in our global variable
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
        //MenuWindowed(hWnd, true);
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
    hWnd = CreateWindow(szWindowClass, L"", wndFlags,
        startX, startY, width, height,
        nullptr, nullptr, hInstance, nullptr);

    if (!hWnd) {
        return nullptr;
    }
    
    // Initialize the 8 bytes after the window handle to point to this object
    SetWindowLongPtrA(hWnd, 0, (LONG_PTR)this);

    // Use  SetWindowLong to make the window layered
    SetWindowLong(hWnd, GWL_EXSTYLE, GetWindowLong(hWnd, GWL_EXSTYLE) | WS_EX_LAYERED);

    // Set the window as transparent
    SetLayeredWindowAttributes(hWnd, RGB(0, 0, 0), 0, LWA_ALPHA);

    // DrawFractalShark(hWnd);
    DrawFractalSharkGdi(nCmdShow);

    // Put us on top
    //SetWindowPos (hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    // Create the menu
    gPopupMenu = LoadMenu(hInst, MAKEINTRESOURCE(IDR_MENU_POPUP));

    // Create the fractal
    RECT rt;
    GetClientRect(hWnd, &rt);

    gFractal = std::make_unique<Fractal>(rt.right, rt.bottom, hWnd, false);

    if constexpr (finishWindowed == false) {
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);
        gWindowed = false;
    }

    return hWnd;
}

//
// Performs all cleanup operations
//
void MainWindow::UnInit() {
    DestroyMenu(gPopupMenu);
    DestroyWindow(hWnd);
    UnregisterClass(szWindowClass, hInst);
    gFractal.reset();
    gJobObj.reset();
}

void MainWindow::HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM /*lParam*/) {
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
    case 'h':
    {
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
    case 'j':
    {
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
    case 'd':
    {
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

LRESULT MainWindow::StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    // Get the window class from the hWnd
    MainWindow *pThis = reinterpret_cast<MainWindow *>(GetWindowLongPtr(hWnd, 0));

    if (pThis == nullptr) {
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    // And invoke WndProc
    return pThis->WndProc(message, wParam, lParam);
}

LRESULT MainWindow::WndProc(UINT message, WPARAM wParam, LPARAM lParam) {
    static bool lButtonDown = false;
    static int dragBoxX1, dragBoxY1;

    // Used for drawing the inverted rectangle properly.
    static int prevX1 = -1, prevY1 = -1;

    // Used for keeping track of where the menu was located
    static int menuX = -1, menuY = -1;

#define MapMenuItemToAlg(wmId, renderAlg) \
    case wmId: \
    { \
        gFractal->SetRenderAlgorithm(renderAlg); \
        break; \
    }

    switch (message) {
    case WM_COMMAND:
    {
        int wmId, wmEvent;
        wmId = LOWORD(wParam);
        wmEvent = HIWORD(wParam);

        switch (wmId) { // Go back to the previous location
        case IDM_BACK:
        {
            MenuGoBack();
            break;
        }

        // Reset the view of the fractal to standard
        case IDM_STANDARDVIEW:
        {
            MenuStandardView(0);
            break;
        }

        case IDM_VIEWS_HELP:
        {
            MenuViewsHelp();
            break;
        }

        case IDM_VIEW1:
        case IDM_VIEW2:
        case IDM_VIEW3:
        case IDM_VIEW4:
        case IDM_VIEW5:
        case IDM_VIEW6:
        case IDM_VIEW7:
        case IDM_VIEW8:
        case IDM_VIEW9:
        case IDM_VIEW10:
        case IDM_VIEW11:
        case IDM_VIEW12:
        case IDM_VIEW13:
        case IDM_VIEW14:
        case IDM_VIEW15:
        case IDM_VIEW16:
        case IDM_VIEW17:
        case IDM_VIEW18:
        case IDM_VIEW19:
        case IDM_VIEW20:
        case IDM_VIEW21:
        case IDM_VIEW22:
        case IDM_VIEW23:
        case IDM_VIEW24:
        case IDM_VIEW25:
        case IDM_VIEW26:
        case IDM_VIEW27:
        case IDM_VIEW28:
        case IDM_VIEW29:
        case IDM_VIEW30:
        case IDM_VIEW31:
        case IDM_VIEW32:
        case IDM_VIEW33:
        case IDM_VIEW34:
        case IDM_VIEW35:
        case IDM_VIEW36:
        case IDM_VIEW37:
        case IDM_VIEW38:
        case IDM_VIEW39:
        case IDM_VIEW40:
        {
            static_assert(IDM_VIEW2 == IDM_VIEW1 + 1, "!");
            static_assert(IDM_VIEW3 == IDM_VIEW1 + 2, "!");
            static_assert(IDM_VIEW4 == IDM_VIEW1 + 3, "!");
            static_assert(IDM_VIEW5 == IDM_VIEW1 + 4, "!");
            static_assert(IDM_VIEW6 == IDM_VIEW1 + 5, "!");
            static_assert(IDM_VIEW7 == IDM_VIEW1 + 6, "!");
            static_assert(IDM_VIEW8 == IDM_VIEW1 + 7, "!");
            static_assert(IDM_VIEW9 == IDM_VIEW1 + 8, "!");
            static_assert(IDM_VIEW10 == IDM_VIEW1 + 9, "!");
            static_assert(IDM_VIEW11 == IDM_VIEW1 + 10, "!");
            static_assert(IDM_VIEW12 == IDM_VIEW1 + 11, "!");
            static_assert(IDM_VIEW13 == IDM_VIEW1 + 12, "!");
            static_assert(IDM_VIEW14 == IDM_VIEW1 + 13, "!");
            static_assert(IDM_VIEW15 == IDM_VIEW1 + 14, "!");
            static_assert(IDM_VIEW16 == IDM_VIEW1 + 15, "!");
            static_assert(IDM_VIEW17 == IDM_VIEW1 + 16, "!");
            static_assert(IDM_VIEW18 == IDM_VIEW1 + 17, "!");
            static_assert(IDM_VIEW19 == IDM_VIEW1 + 18, "!");
            static_assert(IDM_VIEW20 == IDM_VIEW1 + 19, "!");
            static_assert(IDM_VIEW21 == IDM_VIEW1 + 20, "!");
            static_assert(IDM_VIEW22 == IDM_VIEW1 + 21, "!");
            static_assert(IDM_VIEW23 == IDM_VIEW1 + 22, "!");
            static_assert(IDM_VIEW24 == IDM_VIEW1 + 23, "!");
            static_assert(IDM_VIEW25 == IDM_VIEW1 + 24, "!");
            static_assert(IDM_VIEW26 == IDM_VIEW1 + 25, "!");
            static_assert(IDM_VIEW27 == IDM_VIEW1 + 26, "!");
            static_assert(IDM_VIEW28 == IDM_VIEW1 + 27, "!");
            static_assert(IDM_VIEW29 == IDM_VIEW1 + 28, "!");
            static_assert(IDM_VIEW30 == IDM_VIEW1 + 29, "!");
            static_assert(IDM_VIEW31 == IDM_VIEW1 + 30, "!");
            static_assert(IDM_VIEW32 == IDM_VIEW1 + 31, "!");
            static_assert(IDM_VIEW33 == IDM_VIEW1 + 32, "!");
            static_assert(IDM_VIEW34 == IDM_VIEW1 + 33, "!");
            static_assert(IDM_VIEW35 == IDM_VIEW1 + 34, "!");
            static_assert(IDM_VIEW36 == IDM_VIEW1 + 35, "!");
            static_assert(IDM_VIEW37 == IDM_VIEW1 + 36, "!");
            static_assert(IDM_VIEW38 == IDM_VIEW1 + 37, "!");
            static_assert(IDM_VIEW39 == IDM_VIEW1 + 38, "!");
            static_assert(IDM_VIEW40 == IDM_VIEW1 + 39, "!");

            MenuStandardView(wmId - IDM_VIEW1 + 1);
            break;
        }

        // Reset the view of the fractal to "square", taking into
        // account window aspect ratio.  Eliminates distortion.
        case IDM_SQUAREVIEW:
        {
            MenuSquareView();
            break;
        }

        // Recenter the current view at the point where the menu was
        // created, not the current mouse position or some bs like that.
        case IDM_CENTERVIEW:
        {
            MenuCenterView(menuX, menuY);
            break;
        }

        case IDM_ZOOMIN:
        {
            MenuZoomIn({ menuX, menuY });
            break;
        }

        case IDM_ZOOMOUT:
        {
            MenuZoomOut({ menuX, menuY });
            break;
        }

        case IDM_AUTOZOOM_DEFAULT:
        {
            gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
            break;
        }

        case IDM_AUTOZOOM_MAX:
        {
            gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>();
            break;
        }

        case IDM_REPAINTING:
        {
            MenuRepainting();
            break;
        }

        // Make the fractal window a "window" instead of fullscreen
        case IDM_WINDOWED:
        {
            MenuWindowed(false);
            break;
        }

        case IDM_WINDOWED_SQ:
        {
            MenuWindowed(true);
            break;
        }

        // Minimize the window
        case IDM_MINIMIZE:
        {
            PostMessage(hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0);
            break;
        }

        case IDM_GPUANTIALIASING_1X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1);
            break;
        }
        case IDM_GPUANTIALIASING_4X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2);
            break;
        }
        case IDM_GPUANTIALIASING_9X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 3);
            break;
        }

        case IDM_GPUANTIALIASING_16X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;
        }

        // Iteration precision
        case IDM_ITERATIONPRECISION_1X:
        {
            gFractal->SetIterationPrecision(1);
            break;
        }

        case IDM_ITERATIONPRECISION_2X:
        {
            gFractal->SetIterationPrecision(4);
            break;
        }

        case IDM_ITERATIONPRECISION_3X:
        {
            gFractal->SetIterationPrecision(8);
            break;
        }

        case IDM_ITERATIONPRECISION_4X:
        {
            gFractal->SetIterationPrecision(16);
            break;
        }

        // Change rendering algorithm
        case IDM_HELP_ALG:
        {
            MenuAlgHelp();
            break;
        }

        MapMenuItemToAlg(IDM_ALG_AUTO, RenderAlgorithm::AUTO);
        MapMenuItemToAlg(IDM_ALG_CPU_HIGH, RenderAlgorithm::CpuHigh);
        MapMenuItemToAlg(IDM_ALG_CPU_1_32_HDR, RenderAlgorithm::CpuHDR32);
        MapMenuItemToAlg(IDM_ALG_CPU_1_32_PERTURB_BLA_HDR, RenderAlgorithm::Cpu32PerturbedBLAHDR);
        MapMenuItemToAlg(IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR, RenderAlgorithm::Cpu32PerturbedBLAV2HDR);
        MapMenuItemToAlg(IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR, RenderAlgorithm::Cpu32PerturbedRCBLAV2HDR);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR, RenderAlgorithm::Cpu64PerturbedBLAV2HDR);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR, RenderAlgorithm::Cpu64PerturbedRCBLAV2HDR);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64, RenderAlgorithm::Cpu64);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64_HDR, RenderAlgorithm::CpuHDR64);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64_PERTURB_BLA, RenderAlgorithm::Cpu64PerturbedBLA);
        MapMenuItemToAlg(IDM_ALG_CPU_1_64_PERTURB_BLA_HDR, RenderAlgorithm::Cpu64PerturbedBLAHDR);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64, RenderAlgorithm::Gpu1x64);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_BLA, RenderAlgorithm::Gpu1x64PerturbedBLA);
        MapMenuItemToAlg(IDM_ALG_GPU_2_64, RenderAlgorithm::Gpu2x64);
        MapMenuItemToAlg(IDM_ALG_GPU_4_64, RenderAlgorithm::Gpu4x64);
        MapMenuItemToAlg(IDM_ALG_GPU_2X32_HDR, RenderAlgorithm::GpuHDRx32);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32, RenderAlgorithm::Gpu1x32);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_SCALED, RenderAlgorithm::Gpu1x32PerturbedScaled);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_SCALED, RenderAlgorithm::GpuHDRx32PerturbedScaled);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32, RenderAlgorithm::Gpu2x32);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_SCALED, RenderAlgorithm::Gpu2x32PerturbedScaled);
        MapMenuItemToAlg(IDM_ALG_GPU_4_32, RenderAlgorithm::Gpu4x32);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_BLA, RenderAlgorithm::GpuHDRx32PerturbedBLA);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_BLA, RenderAlgorithm::GpuHDRx64PerturbedBLA);

        /////////////////////////// Begin LAV2 ///////////////////////////
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_LAV2, RenderAlgorithm::Gpu1x32PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_LAV2_PO, RenderAlgorithm::Gpu1x32PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO, RenderAlgorithm::Gpu1x32PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2, RenderAlgorithm::Gpu1x32PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_PO, RenderAlgorithm::Gpu1x32PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_LAO, RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO);

        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_LAV2, RenderAlgorithm::Gpu2x32PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_LAV2_PO, RenderAlgorithm::Gpu2x32PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO, RenderAlgorithm::Gpu2x32PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2, RenderAlgorithm::Gpu2x32PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_PO, RenderAlgorithm::Gpu2x32PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_LAO, RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO);

        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_LAV2, RenderAlgorithm::Gpu1x64PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_LAV2_PO, RenderAlgorithm::Gpu1x64PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO, RenderAlgorithm::Gpu1x64PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2, RenderAlgorithm::Gpu1x64PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_PO, RenderAlgorithm::Gpu1x64PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_LAO, RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO);

        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_LAV2, RenderAlgorithm::GpuHDRx32PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO, RenderAlgorithm::GpuHDRx32PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO, RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2, RenderAlgorithm::GpuHDRx32PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_PO, RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_LAO, RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO);

        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2, RenderAlgorithm::GpuHDRx2x32PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO, RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO, RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2, RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_PO, RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_LAO, RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO);

        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_LAV2, RenderAlgorithm::GpuHDRx64PerturbedLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO, RenderAlgorithm::GpuHDRx64PerturbedLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO, RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2, RenderAlgorithm::GpuHDRx64PerturbedRCLAv2);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_PO, RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO);
        MapMenuItemToAlg(IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_LAO, RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO);

        case IDM_LA_SINGLETHREADED:
        {
            auto &parameters = gFractal->GetLAParameters();
            parameters.SetThreading(LAParameters::LAThreadingAlgorithm::SingleThreaded);
            break;
        }

        case IDM_LA_MULTITHREADED:
        {
            auto &parameters = gFractal->GetLAParameters();
            parameters.SetThreading(LAParameters::LAThreadingAlgorithm::MultiThreaded);
            break;
        }

        case IDM_LA_SETTINGS_1:
        {
            auto &parameters = gFractal->GetLAParameters();
            parameters.SetDefaults(LAParameters::LADefaults::MaxAccuracy);
            break;
        }

        case IDM_LA_SETTINGS_2:
        {
            auto &parameters = gFractal->GetLAParameters();
            parameters.SetDefaults(LAParameters::LADefaults::MaxPerf);
            break;
        }

        case IDM_LA_SETTINGS_3:
        {
            auto &parameters = gFractal->GetLAParameters();
            parameters.SetDefaults(LAParameters::LADefaults::MinMemory);
            break;
        }

        case IDM_BASICTEST:
        {
            FractalTest test{ *gFractal };
            test.BasicTest();
            break;
        }

        // Increase the number of iterations we are using.
        // This will slow down rendering, but image quality
        // will be improved.
        case IDM_INCREASEITERATIONS_1P5X:
        {
            MenuMultiplyIterations(1.5);
            break;
        }
        case IDM_INCREASEITERATIONS_6X:
        {
            MenuMultiplyIterations(6.0);
            break;
        }

        case IDM_INCREASEITERATIONS_24X:
        {
            MenuMultiplyIterations(24.0);
            break;
        }
        // Decrease the number of iterations we are using
        case IDM_DECREASEITERATIONS:
        {
            MenuMultiplyIterations(2.0 / 3.0);
            break;
        }
        // Reset the number of iterations to the default
        case IDM_RESETITERATIONS:
        {
            MenuResetIterations();
            break;
        }

        case IDM_32BIT_ITERATIONS:
        {
            gFractal->SetIterType(IterTypeEnum::Bits32);
            break;
        }

        case IDM_64BIT_ITERATIONS:
        {
            gFractal->SetIterType(IterTypeEnum::Bits64);
            break;
        }

        case IDM_PERTURB_RESULTS:
        {
            //gFractal->DrawAllPerturbationResults(false);
            ::MessageBox(hWnd,
                L"TODO.  By default these are shown as white pixels overlayed on the image. "
                L"It'd be nice to have an option that shows them as white pixels against a "
                L"black screen so they're location is obvious.", L"TODO", MB_OK | MB_APPLMODAL);
            break;
        }
        case IDM_PERTURB_CLEAR_ALL:
        {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            break;
        }

        case IDM_PERTURB_CLEAR_MED:
        {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
            break;
        }

        case IDM_PERTURB_CLEAR_HIGH:
        {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::HighRes);
            break;
        }

        case IDM_PERTURBATION_AUTO:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto);
            break;
        }

        case IDM_PERTURBATION_SINGLETHREAD:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::ST);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT);
            break;
        }

        case IDM_PERTURBATION_SINGLETHREAD_PERIODICITY:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_STMED:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED1:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED2:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED3:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED4:
        {
            // Broken
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD5_PERIODICITY:
        {
            gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5);
            break;
        }

        case IDM_PERTURBATION_SAVE:
        {
            gFractal->SavePerturbationOrbits();
            break;
        }

        case IDM_PERTURBATION_LOAD:
        {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            gFractal->LoadPerturbationOrbits();
            break;
        }

        case IDM_PERTURB_AUTOSAVE_ON:
        {
            gFractal->SetResultsAutosave(AddPointOptions::EnableWithSave);
            break;
        }

        case IDM_PERTURB_AUTOSAVE_ON_DELETE:
        {
            gFractal->SetResultsAutosave(AddPointOptions::EnableWithoutSave);
            break;
        }

        case IDM_PERTURB_AUTOSAVE_OFF:
        {
            gFractal->SetResultsAutosave(AddPointOptions::DontSave);
            break;
        }

        case IDM_MEMORY_LIMIT_0:
        {
            gJobObj = nullptr;
            break;
        }

        case IDM_MEMORY_LIMIT_1:
        {
            gJobObj = std::make_unique<JobObject>();
            break;
        }

        case IDM_PALETTEROTATE:
        {
            MenuPaletteRotation();
            break;
        }
        case IDM_CREATENEWPALETTE:
        {
            MenuCreateNewPalette();
            break;
        }

        case IDM_PALETTE_TYPE_0:
        {
            MenuPaletteType(FractalPalette::Basic);
            break;
        }

        case IDM_PALETTE_TYPE_1:
        {
            MenuPaletteType(FractalPalette::Default);
            break;
        }

        case IDM_PALETTE_TYPE_2:
        {
            MenuPaletteType(FractalPalette::Patriotic);
            break;
        }

        case IDM_PALETTE_TYPE_3:
        {
            MenuPaletteType(FractalPalette::Summer);
            break;
        }

        case IDM_PALETTE_TYPE_4:
        {
            MenuPaletteType(FractalPalette::Random);
            break;
        }

        case IDM_PALETTE_5:
        {
            MenuPaletteDepth(5);
            break;
        }

        case IDM_PALETTE_6:
        {
            MenuPaletteDepth(6);
            break;
        }

        case IDM_PALETTE_8:
        {
            MenuPaletteDepth(8);
            break;
        }

        case IDM_PALETTE_12:
        {
            MenuPaletteDepth(12);
            break;
        }

        case IDM_PALETTE_16:
        {
            MenuPaletteDepth(16);
            break;
        }

        case IDM_PALETTE_20:
        {
            MenuPaletteDepth(20);
            break;
        }

        // Put the current window position on
        // the clipboard.  The coordinates put
        // on the clipboard are the "calculator"
        // coordinates, not screen coordinates.
        case IDM_CURPOS:
        {
            MenuGetCurPos();
            break;
        }
        case IDM_BENCHMARK_FULL:
        {
            FractalTest test{ *gFractal };
            test.Benchmark(RefOrbitCalc::PerturbationResultType::All);
            break;
        }
        case IDM_BENCHMARK_INT:
        {
            FractalTest test{ *gFractal };
            test.Benchmark(RefOrbitCalc::PerturbationResultType::MediumRes);
            break;
        }

        // Save/load current location
        case (IDM_SAVELOCATION):
        {
            MenuSaveCurrentLocation();
            break;
        }
        case (IDM_LOADLOCATION):
        {
            MenuLoadCurrentLocation();
            break;
        }
        case IDM_SAVEBMP:
        {
            MenuSaveBMP();
            break;
        }
        case IDM_SAVEHIRESBMP:
        {
            MenuSaveHiResBMP();
            break;
        }
        case IDM_SAVE_ITERS_TEXT:
        {
            MenuSaveItersAsText();
            break;
        }
        case IDM_SAVE_REFORBIT_TEXT:
        {
            gFractal->SaveRefOrbitAsText(CompressToDisk::Disable);
            break;
        }
        case IDM_SAVE_REFORBIT_TEXT_SIMPLE:
        {
            gFractal->SaveRefOrbitAsText(CompressToDisk::SimpleCompression);
            break;
        }
        case IDM_SAVE_REFORBIT_TEXT_MAX:
        {
            gFractal->SaveRefOrbitAsText(CompressToDisk::MaxCompression);
            break;
        }
        case IDM_SHOWHOTKEYS:
        {
            MenuShowHotkeys();
            break;
        }
        // Exit the program
        case IDM_EXIT:
        {
            DestroyWindow(hWnd);
            break;
        }

        case IDM_VIEW_DYNAMIC + 0:
        case IDM_VIEW_DYNAMIC + 1:
        case IDM_VIEW_DYNAMIC + 2:
        case IDM_VIEW_DYNAMIC + 3:
        case IDM_VIEW_DYNAMIC + 4:
        case IDM_VIEW_DYNAMIC + 5:
        case IDM_VIEW_DYNAMIC + 6:
        case IDM_VIEW_DYNAMIC + 7:
        case IDM_VIEW_DYNAMIC + 8:
        case IDM_VIEW_DYNAMIC + 9:
        case IDM_VIEW_DYNAMIC + 10:
        case IDM_VIEW_DYNAMIC + 11:
        case IDM_VIEW_DYNAMIC + 12:
        case IDM_VIEW_DYNAMIC + 13:
        case IDM_VIEW_DYNAMIC + 14:
        case IDM_VIEW_DYNAMIC + 15:
        case IDM_VIEW_DYNAMIC + 16:
        case IDM_VIEW_DYNAMIC + 17:
        case IDM_VIEW_DYNAMIC + 18:
        case IDM_VIEW_DYNAMIC + 19:
        case IDM_VIEW_DYNAMIC + 20:
        case IDM_VIEW_DYNAMIC + 21:
        case IDM_VIEW_DYNAMIC + 22:
        case IDM_VIEW_DYNAMIC + 23:
        case IDM_VIEW_DYNAMIC + 24:
        case IDM_VIEW_DYNAMIC + 25:
        case IDM_VIEW_DYNAMIC + 26:
        case IDM_VIEW_DYNAMIC + 27:
        case IDM_VIEW_DYNAMIC + 28:
        case IDM_VIEW_DYNAMIC + 29:
        {
            auto index = wmId - IDM_VIEW_DYNAMIC;
            auto minX = gSavedLocations[index].minX;
            auto minY = gSavedLocations[index].minY;
            auto maxX = gSavedLocations[index].maxX;
            auto maxY = gSavedLocations[index].maxY;
            auto num_iterations = gSavedLocations[index].num_iterations;
            auto antialiasing = gSavedLocations[index].antialiasing;

            gFractal->RecenterViewCalc(minX, minY, maxX, maxY);
            gFractal->SetNumIterations<IterTypeFull>(num_iterations);
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, antialiasing);
            PaintAsNecessary();
            break;
        }

        // Catch-all
        default:
        {
            ::MessageBox(hWnd, L"Unknown menu item", L"Error", MB_OK | MB_APPLMODAL);
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        }

        break;
    }

    case WM_SIZE:
    {
        if (gFractal != nullptr) {
            gFractal->ResetDimensions(LOWORD(lParam), HIWORD(lParam));
            PaintAsNecessary();
        }
        break;
    }
    // Display the popup menu with options
    case WM_CONTEXTMENU:
    {
        menuX = GET_X_LPARAM(lParam);
        menuY = GET_Y_LPARAM(lParam);

        TrackPopupMenu(GetSubMenu(gPopupMenu, 0), 0, menuX, menuY, 0, hWnd, nullptr);

        // Store the menu location as client coordinates
        // not screen coordinates.
        POINT temp;
        temp.x = menuX;
        temp.y = menuY;
        ScreenToClient(hWnd, &temp);
        menuX = temp.x;
        menuY = temp.y;
        break;
    }

    // Begin dragging a box for zooming in.
    case WM_LBUTTONDOWN:
    {
        if (gWindowed == true && IsDownAlt() == true) { // Don't drag a box if we are pressing the ALT key and this is windowed
            // mode.  Instead, move the window!
            PostMessage(hWnd, WM_NCLBUTTONDOWN, HTCAPTION, lParam);
        } else {
            if (lButtonDown == true) {
                break;
            }
            lButtonDown = true;

            dragBoxX1 = GET_X_LPARAM(lParam);
            dragBoxY1 = GET_Y_LPARAM(lParam);
        }
        break;
    }

    // Zoom in
    case WM_LBUTTONUP:
    {
        if (lButtonDown == false || IsDownAlt() == true) {
            break;
        }
        lButtonDown = false;

        prevX1 = -1;
        prevY1 = -1;

        RECT newView;
        bool MaintainAspectRatio = !IsDownShift();
        if (MaintainAspectRatio == true) // Maintain aspect ratio
        { // Get the aspect ratio of the window.
            RECT windowRect;
            GetClientRect(hWnd, &windowRect);
            double ratio = (double)windowRect.right / (double)windowRect.bottom;

            // Note order is important.
            newView.left = dragBoxX1;
            newView.top = dragBoxY1;
            newView.bottom = GET_Y_LPARAM(lParam);
            newView.right = (long)((double)newView.left + (double)ratio * (double)((double)newView.bottom - (double)newView.top));
        } else // Do anything
        {
            newView.left = dragBoxX1;
            newView.right = GET_X_LPARAM(lParam);
            newView.top = dragBoxY1;
            newView.bottom = GET_Y_LPARAM(lParam);
        }

        if (gFractal->RecenterViewScreen(newView) == true) {
            if (MaintainAspectRatio == true) {
                gFractal->SquareCurrentView();
            }

            RECT rect;
            GetClientRect(hWnd, &rect);

            PaintAsNecessary();
        }
        break;
    }

    case WM_MOUSEMOVE:
    {
        if (lButtonDown == false) {
            break;
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
            rect.right = (long)((double)rect.left + (double)ratio * (double)((double)rect.bottom - (double)rect.top));

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

    case WM_MOUSEWHEEL:
    {
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
        break;
    }

    // Repaint the screen
    case WM_PAINT:
    {
        PaintAsNecessary();

        PAINTSTRUCT ps;
        BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
        break;
    }

    // Exit
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        break;
    }

    case WM_CHAR:
    {
        HandleKeyDown(message, wParam, lParam);
        PaintAsNecessary();
        break;
    }

    // Catch-all.
    default:
    {
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    }

    return 0;
}

void MainWindow::MenuGoBack() {
    if (gFractal->Back() == true) {
        PaintAsNecessary();
    }
}

void MainWindow::MenuStandardView(size_t i) {
    gFractal->View(i);
    PaintAsNecessary();
}

void MainWindow::MenuSquareView() {
    gFractal->SquareCurrentView();
    PaintAsNecessary();
}

void MainWindow::MenuCenterView(int x, int y) {
    gFractal->CenterAtPoint(x, y);
    PaintAsNecessary();
}

void MainWindow::MenuZoomIn(POINT mousePt) {
    gFractal->Zoom(mousePt.x, mousePt.y, -.45);
    PaintAsNecessary();
}

void MainWindow::MenuZoomOut(POINT mousePt) {
    gFractal->Zoom(mousePt.x, mousePt.y, 1);
    PaintAsNecessary();
}

void MainWindow::MenuRepainting() {
    gFractal->ToggleRepainting();
    PaintAsNecessary();
}

void MainWindow::MenuWindowed(bool square) {
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
            auto width = std::min(
                (rect.right + rect.left) / 2,
                (rect.bottom + rect.top) / 2);
            //width /= 2;
            SetWindowPos(hWnd, HWND_NOTOPMOST,
                (rect.right + rect.left) / 2 - width / 2,
                (rect.bottom + rect.top) / 2 - width / 2,
                width,
                width,
                SWP_SHOWWINDOW);
        } else {
            SetWindowPos(hWnd, HWND_NOTOPMOST,
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

void MainWindow::MenuMultiplyIterations(double factor) {
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

void MainWindow::MenuResetIterations() {
    gFractal->ResetNumIterations();
    PaintAsNecessary();
}

void MainWindow::MenuGetCurPos() {
    constexpr size_t numBytes = 32768;

    BOOL ret = OpenClipboard(hWnd);
    if (ret == 0) {
        MessageBox(hWnd, L"Opening the clipboard failed.  Another program must be using it.", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    ret = EmptyClipboard();
    if (ret == 0) {
        MessageBox(hWnd, L"Emptying the clipboard of its current contents failed.  Make sure no other programs are using it.", L"", MB_OK | MB_APPLMODAL);
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

    HighPrecision minX, minY;
    HighPrecision maxX, maxY;

    auto prec = gFractal->GetPrecision();

    minX = gFractal->GetMinX();
    minY = gFractal->GetMinY();
    maxX = gFractal->GetMaxX();
    maxY = gFractal->GetMaxY();

    std::stringstream ss;
    std::string s;

    auto setupSS = [&](const HighPrecision &num) -> std::string {
        ss.str("");
        ss.clear();
        ss << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
        ss << num;
        return ss.str();
        };

    s = setupSS(minX);
    auto sminX = std::string(s.begin(), s.end());

    s = setupSS(minY);
    auto sminY = std::string(s.begin(), s.end());

    s = setupSS(maxX);
    auto smaxX = std::string(s.begin(), s.end());

    s = setupSS(maxY);
    auto smaxY = std::string(s.begin(), s.end());

    PointZoomBBConverter pz{ minX, minY, maxX, maxY };
    s = setupSS(pz.GetPtX());
    auto ptXStr = std::string(s.begin(), s.end());

    s = setupSS(pz.GetPtY());
    auto ptYStr = std::string(s.begin(), s.end());

    auto reducedPrecZF = pz.GetZoomFactor();
    reducedPrecZF.precisionInBits(50);
    s = setupSS(reducedPrecZF);
    auto zoomFactorStr = std::string(s.begin(), s.end());

    RefOrbitDetails details;
    gFractal->GetSomeDetails(details);

    auto ActualPeriodIfAny = (details.InternalPeriodMaybeZero > 0) ? (details.InternalPeriodMaybeZero - 1) : 0;

    const auto additionalDetailsStr =
        std::string("PerturbationAlg = ") + details.PerturbationAlg + "\r\n" +
        std::string("InternalPeriodIfAny = ") + std::to_string(details.InternalPeriodMaybeZero) + "\r\n" +
        std::string("ActualPeriodIfAny = ") + std::to_string(ActualPeriodIfAny) + "\r\n" +
        std::string("CompressedIters = ") + std::to_string(details.CompressedIters) + "\r\n" +
        std::string("UncompressedIters = ") + std::to_string(details.UncompressedIters) + "\r\n" +
        std::string("Compression ratio = ") + std::to_string((double)details.UncompressedIters / (double)details.CompressedIters) + "\r\n" +
        std::string("Compression error exp = ") + std::to_string(details.CompressionErrorExp) + "\r\n" +
        std::string("CompressedIntermediateIters = ") + std::to_string(details.CompressedIntermediateIters) + "\r\n" +
        std::string("Reuse compression error exp = ") + std::to_string(details.IntermediateCompressionErrorExp) + "\r\n" +
        std::string("Reuse compression ratio = ") + std::to_string((double)details.UncompressedIters / (double)details.CompressedIntermediateIters) + "\r\n" +
        std::string("DeltaIntermediatePrecision = ") + std::to_string(details.DeltaIntermediatePrecision) + "\r\n" +
        std::string("ExtraIntermediatePrecision = ") + std::to_string(details.ExtraIntermediatePrecision) + "\r\n";

    const auto &laParameters = gFractal->GetLAParameters();
    const auto threadingVal = laParameters.GetThreading();
    std::string threadingStr;
    if (threadingVal == LAParameters::LAThreadingAlgorithm::SingleThreaded) {
        threadingStr = "Single threaded";
    } else if (threadingVal == LAParameters::LAThreadingAlgorithm::MultiThreaded) {
        threadingStr = "Multi threaded";
    } else {
        threadingStr = "Unknown";
    }

    const auto laParametersStr =
        std::string("Detection method = ")
        + std::to_string(laParameters.GetDetectionMethod()) + "\r\n" +
        std::string("Threshold scale = ")
        + std::to_string(laParameters.GetLAThresholdScaleExp()) + "\r\n" +
        std::string("Threshold C scale = ")
        + std::to_string(laParameters.GetLAThresholdCScaleExp()) + "\r\n" +
        std::string("Stage 0 period detection threshold 2 = ")
        + std::to_string(laParameters.GetStage0PeriodDetectionThreshold2Exp()) + "\r\n" +
        std::string("Period detection threshold 2 = ")
        + std::to_string(laParameters.GetPeriodDetectionThreshold2Exp()) + "\r\n" +
        std::string("Stage 0 period detection threshold = ")
        + std::to_string(laParameters.GetStage0PeriodDetectionThresholdExp()) + "\r\n" +
        std::string("Period detection threshold = ")
        + std::to_string(laParameters.GetPeriodDetectionThresholdExp()) + "\r\n" +
        std::string("LA Threading: ")
        + threadingStr + "\r\n" +
        std::string("LA size: ")
        + std::to_string(details.LASize) + "\r\n";

    const auto benchmarkData =
        std::string("Overall time (ms) = ") + std::to_string(gFractal->GetBenchmarkOverall().GetDeltaInMs()) + "\r\n" +
        std::string("Per pixel (ms) = ") + std::to_string(gFractal->GetBenchmarkPerPixel().GetDeltaInMs()) + "\r\n" +
        std::string("RefOrbit time (ms) = ") + std::to_string(details.OrbitMilliseconds) + "\r\n" +
        std::string("LA generation time (ms) = ") + std::to_string(details.LAMilliseconds) + "\r\n";

    snprintf(
        mem,
        numBytes,
        "This text is copied to clipboard.  Using \"%s\"\r\n"
        "Antialiasing: %u\r\n"
        "Palette depth: %u\r\n"
        "Coordinate precision = %zu;\r\n"
        "Center X: \"%s\"\r\n"
        "Center Y: \"%s\"\r\n"
        "zoomFactor \"%s\"\r\n"
        "\r\n"
        "LA parameters:\r\n"
        "%s\r\n"
        "Benchmark data:\r\n"
        "%s\r\n"
        "\r\n"
        "Additional details:\r\n"
        "%s\r\n"
        "SetNumIterations<IterTypeFull>(%zu);\r\n",
        gFractal->GetRenderAlgorithmName(),
        gFractal->GetGpuAntialiasing(),
        gFractal->GetPaletteDepth(),
        prec,
        ptXStr.c_str(),
        ptYStr.c_str(),
        zoomFactorStr.c_str(),
        laParametersStr.c_str(),
        benchmarkData.c_str(),
        additionalDetailsStr.c_str(),
        gFractal->GetNumIterations<IterTypeFull>());

    const std::string stringCopy = mem;
    char temp2[numBytes];

    // Put some extra information on the clipboard.
    snprintf(temp2, numBytes,
        "Bounding box:\r\n"
        "minX = HighPrecision{ \"%s\" };\r\n"
        "minY = HighPrecision{ \"%s\" };\r\n"
        "maxX = HighPrecision{ \"%s\" };\r\n"
        "maxY = HighPrecision{ \"%s\" };\r\n",
        sminX.c_str(), sminY.c_str(),
        smaxX.c_str(), smaxY.c_str());

    // Append temp2 to mem without overrunning the buffer 
    // using strncat.
    strncat(mem, temp2, numBytes - stringCopy.size() - 1);
    GlobalUnlock(hData);

    //
    // This is not a memory leak - we don't "free" hData.
    //

    HANDLE clpData = SetClipboardData(CF_TEXT, hData);
    if (clpData == nullptr) {
        MessageBox(hWnd, L"Adding the data to the clipboard failed.  You are probably very low on memory.  Try closing other programs or restarting your computer.", L"", MB_OK | MB_APPLMODAL);
        CloseClipboard();
        return;
    }

    CloseClipboard();

    ::MessageBoxA(hWnd, stringCopy.c_str(), "", MB_OK | MB_APPLMODAL);
}

void MainWindow::MenuPaletteRotation() {
    POINT OrgPos, CurPos;
    GetCursorPos(&OrgPos);

    for (;;) {
        gFractal->RotateFractalPalette(10);
        gFractal->DrawFractal(false);
        GetCursorPos(&CurPos);
        if (abs(CurPos.x - OrgPos.x) > 5 ||
            abs(CurPos.y - OrgPos.y) > 5) {
            break;
        }
    }

    gFractal->ResetFractalPalette();
    gFractal->DrawFractal(false);
}

void MainWindow::MenuPaletteType(FractalPalette type) {
    gFractal->UsePaletteType(type);
    if (type == FractalPalette::Default) {
        gFractal->UsePalette(8);
        gFractal->SetPaletteAuxDepth(0);
    }
    gFractal->DrawFractal(false);
}

void MainWindow::MenuPaletteDepth(int depth) {
    gFractal->UsePalette(depth);
    gFractal->DrawFractal(false);
}

void MainWindow::MenuCreateNewPalette() {
    gFractal->CreateNewFractalPalette();
    gFractal->UsePaletteType(FractalPalette::Random);
    gFractal->DrawFractal(false);
}

void MainWindow::MenuSaveCurrentLocation() {
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
    //ss << gFractal->GetIterationPrecision() << " ";
    ss << "FractalTrayDestination";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());

    MessageBox(nullptr, ws.c_str(), L"location", MB_OK | MB_APPLMODAL);

    FILE *file = fopen("locations.txt", "at+");
    fprintf(file, "%s\r\n", s.c_str());
    fclose(file);
}

void MainWindow::MenuLoadCurrentLocation() {
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
        AppendMenu(hSubMenu, MF_STRING, IDM_VIEW_DYNAMIC + index, ws.c_str());
        index++;

        // Limit the number of locations we show.
        if (index > 30) {
            break;
        }
    }

    POINT point;
    GetCursorPos(&point);
    TrackPopupMenu(hSubMenu, 0, point.x, point.y, 0, hWnd, nullptr);
}

void MainWindow::MenuSaveBMP() {
    gFractal->SaveCurrentFractal(L"", true);
}

void MainWindow::MenuSaveHiResBMP() {
    gFractal->SaveHiResFractal(L"");
}

void MainWindow::MenuSaveItersAsText() {
    gFractal->SaveItersAsText(L"");
}

void MainWindow::BenchmarkMessage(size_t milliseconds) {
    std::stringstream ss;
    ss << std::string("Time taken in ms: ") << milliseconds << ".";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());
    MessageBox(hWnd, ws.c_str(), L"", MB_OK | MB_APPLMODAL);

}

void MainWindow::MenuAlgHelp() {
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
        L"poor 64-bit performance\r\n"
        , L"Algorithms",
        MB_OK
    );
}

void MainWindow::MenuViewsHelp() {
    ::MessageBox(
        nullptr,
        L"Views\r\n"
        L"\r\n"
        L"The purpose of these is simply to make it easy to navigate to\r\n"
        L"some interesting locations.\r\n"
        , L"Views",
        MB_OK);
}

void MainWindow::MenuShowHotkeys() {
    // Shows some basic help + hotkeys as defined in HandleKeyDown
    ::MessageBox(
        nullptr,
        L"Hotkeys\r\n"
        L"\r\n"
        L"Navigation\r\n"
        L"a - Autozoom using averaging heuristic.  Buggy.  Hold CTRL to abort.\r\n"
        L"A - Autozoom by zooming in on the highest iteration count point.  Buggy.  Hold CTRL to abort.\r\n"
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
        L"Use these keys to increase/decrease the number of iterations used to calculate the fractal.\r\n"
        L"= - Multiply max iterations by 24\r\n"
        L"- - Multiply max iterations by 2/3\r\n"
        L"\r\n"
        L"Misc\r\n"
        L"CTRL - Press and hold to abort autozoom\r\n"
        L"ALT - Press, click/drag to move window when in windowed mode\r\n"
        L"Right click - popup menu\r\n"
        , L"",
        MB_OK);
}

void MainWindow::PaintAsNecessary() {
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
typedef BOOL(WINAPI *MINIDUMPWRITEDUMP)(HANDLE hProcess, DWORD dwPid, HANDLE hFile, MINIDUMP_TYPE DumpType, CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam, CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam, CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

void MainWindow::create_minidump(struct _EXCEPTION_POINTERS *apExceptionInfo) {
    HMODULE mhLib = ::LoadLibrary(_T("dbghelp.dll"));
    MINIDUMPWRITEDUMP pDump = (MINIDUMPWRITEDUMP)::GetProcAddress(mhLib, "MiniDumpWriteDump");

    HANDLE  hFile = ::CreateFile(_T("core.dmp"), GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    _MINIDUMP_EXCEPTION_INFORMATION ExInfo;
    ExInfo.ThreadId = ::GetCurrentThreadId();
    ExInfo.ExceptionPointers = apExceptionInfo;
    ExInfo.ClientPointers = FALSE;

    pDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &ExInfo, nullptr, nullptr);
    ::CloseHandle(hFile);
}

LONG WINAPI MainWindow::unhandled_handler(struct _EXCEPTION_POINTERS *apExceptionInfo) {
    create_minidump(apExceptionInfo);
    return EXCEPTION_CONTINUE_SEARCH;
}