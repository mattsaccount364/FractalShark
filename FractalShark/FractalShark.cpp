#include "stdafx.h"
#include "resource.h"
#include "Fractal.h"

#include <Dbghelp.h>

// Global Variables:
HINSTANCE hInst;                // current instance
LPCWSTR szWindowClass = L"FractalWindow";
HMENU gPopupMenu;
bool gWindowed; // Says whether we are in windowed mode or not.
HDC gHDC;

// Fractal:
Fractal *gFractal = NULL;

// Foward declarations of functions included in this code module:
ATOM              MyRegisterClass(HINSTANCE hInstance);
HWND              InitInstance(HINSTANCE, int);
LRESULT CALLBACK  WndProc(HWND, UINT, WPARAM, LPARAM);
void              UnInit(void);

LONG WINAPI unhandled_handler(struct _EXCEPTION_POINTERS* apExceptionInfo);

// Controlling functions
void MenuGoBack(HWND hWnd);
void MenuStandardView(HWND hWnd, size_t i);
void MenuSquareView(HWND hWnd);
void MenuCenterView(HWND hWnd, int x, int y);
void MenuZoomIn(HWND hWnd, POINT mousePt);
void MenuZoomOut(HWND hWnd, POINT mousePt);
void MenuRepainting(HWND hWnd);
void MenuWindowed(HWND hWnd, bool square);
void MenuMultiplyIterations(HWND hWnd, double factor);
void MenuResetIterations(HWND hWnd);
void MenuPaletteType(Fractal::Palette type);
void MenuPaletteDepth(int depth);
void MenuPaletteRotation(HWND hWnd);
void MenuCreateNewPalette(HWND hWnd);
void MenuGetCurPos(HWND hWnd);
void MenuSaveCurrentLocation(HWND hWnd);
void MenuSaveBMP(HWND hWnd);
void MenuSaveHiResBMP(HWND hWnd);
void MenuSaveItersAsText();
void MenuBenchmark(HWND hWnd, bool fastbenchmark);
void MenuBenchmarkRefPtDouble(HWND hWnd);
void MenuBenchmarkRefPtHDRFloat(HWND hWnd);
void MenuBenchmarkThis(HWND hWnd);
void MenuAlgHelp(HWND hWnd);
void MenuViewsHelp(HWND hWnd);
void MenuShowHotkeys(HWND hWnd);
void PaintAsNecessary(HWND hWnd);

bool IsDownControl() { return (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000; };
bool IsDownShift() { return (GetAsyncKeyState(VK_SHIFT) & 0x8000) == 0x8000; };
bool IsDownAlt() { return (GetAsyncKeyState(VK_MENU) & 0x8000) == 0x8000; };

int APIENTRY WinMain(HINSTANCE hInstance,
    HINSTANCE /*hPrevInstance*/,
    LPSTR     /*lpCmdLine*/,
    int       nCmdShow)
{
    WSADATA info;
    if (WSAStartup(MAKEWORD(1, 1), &info) != 0)
    {
        MessageBox(NULL, L"Cannot initialize WinSock!", L"WSAStartup", MB_OK);
    }

    // Create a dump file whenever the gateway crashes only on windows
    SetUnhandledExceptionFilter(unhandled_handler);

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // Perform application initialization:
    HWND hWnd = InitInstance(hInstance, nCmdShow);
    if (hWnd == NULL)
    {
        return 1;
    }

    // Main message loop:
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Cleanup
    UnInit();

    return (int)msg.wParam;
}

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, (LPCTSTR)IDI_FRACTALS);
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = NULL;
    wcex.lpszMenuName = NULL;
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
HWND InitInstance(HINSTANCE hInstance, int nCmdShow)
{ // Store instance handle in our global variable
    hInst = hInstance;

    constexpr bool startWindowed = true;

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
    }
    else {
        startX = 0;
        startY = 0;
        width = scrnWidth;
        height = scrnHeight;

        gWindowed = false;
    }

    // Create the window
    HWND hWnd;
    hWnd = CreateWindow(szWindowClass, L"", WS_POPUP | WS_THICKFRAME,
        startX, startY, width, height,
        NULL, NULL, hInstance, NULL);

    if (!hWnd)
    {
        return NULL;
    }

    // Put us on top
    //SetWindowPos (hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    // Create the menu
    gPopupMenu = LoadMenu(hInst, MAKEINTRESOURCE(IDR_MENU_POPUP));

    // Create the fractal
    RECT rt;
    GetClientRect(hWnd, &rt);

    FractalSetupData setupData;
    setupData.Load();
    gFractal = new Fractal(&setupData, rt.right, rt.bottom, NULL, hWnd, false);

    // Display!
    ShowWindow(hWnd, nCmdShow);

    if constexpr (startWindowed == false) {
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);
    }

    return hWnd;
}

//
// Performs all cleanup operations
//
void UnInit(void)
{
    DestroyMenu(gPopupMenu);
    delete gFractal;
    WSACleanup();
}

void HandleKeyDown(HWND hWnd, UINT /*message*/, WPARAM wParam, LPARAM /*lParam*/) {
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
            MenuCenterView(hWnd, mousePt.x, mousePt.y);
            gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
        }
        else {
            MenuCenterView(hWnd, mousePt.x, mousePt.y);
            gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Max>();
        }
        break;

    case 'b':
        MenuGoBack(hWnd);
        break;

    case 'C':
    case 'c':
        if (shiftDown) {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        }
        MenuCenterView(hWnd, mousePt.x, mousePt.y);
        break;

    case 'I':
    case 'i':
        if (shiftDown) {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
        }
        MenuBenchmarkThis(hWnd);
        break;

    case 'O':
    case 'o':
        if (shiftDown) {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        }
        MenuBenchmarkThis(hWnd);
        break;

    case 'P':
    case 'p':
        if (shiftDown) {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        }
        MenuSquareView(hWnd);
        break;

    case 'R':
    case 'r':
        if (shiftDown) {
            gFractal->ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        }
        MenuSquareView(hWnd);
        break;

    case 'T':
    case 't':
        if (shiftDown) {
            gFractal->UseNextPaletteAuxDepth(-1);
        }
        else {
            gFractal->UseNextPaletteAuxDepth(1);
        }
        gFractal->DrawFractal(false);
        break;

    case 'Z':
    case 'z':
        if (shiftDown) {
            MenuZoomOut(hWnd, mousePt);
        }
        else {
            MenuZoomIn(hWnd, mousePt);
        }
        break;

    case 'D':
    case 'd':
    {
        if (shiftDown) {
            gFractal->CreateNewFractalPalette();
            gFractal->UsePaletteType(Fractal::Palette::Random);
        }
        else {
            gFractal->UseNextPaletteDepth();
        }
        gFractal->DrawFractal(false);
        break;
    }

    case '=':
        MenuMultiplyIterations(hWnd, 24.0);
        break;
    case '-':
        MenuMultiplyIterations(hWnd, 2.0 / 3.0);
        break;

    default:
        break;
    }
}

//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT  - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{ // Used for processing click-and-drag properly:
    static bool lButtonDown = false;
    static int dragBoxX1, dragBoxY1;

    // Used for drawing the inverted rectangle properly.
    static int prevX1 = -1, prevY1 = -1;

    // Used for keeping track of where the menu was located
    static int menuX = -1, menuY = -1;

    switch (message)
    {
    case WM_COMMAND:
    {
        int wmId, wmEvent;
        wmId = LOWORD(wParam);
        wmEvent = HIWORD(wParam);

        switch (wmId)
        { // Go back to the previous location
            case IDM_BACK:
            {
                MenuGoBack(hWnd);
                break;
            }

            // Reset the view of the fractal to standard
            case IDM_STANDARDVIEW:
            {
                MenuStandardView(hWnd, 0);
                break;
            }

            case IDM_VIEWS_HELP:
            {
                MenuViewsHelp(hWnd);
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

                MenuStandardView(hWnd, wmId - IDM_VIEW1 + 1);
                break;
            }

            // Reset the view of the fractal to "square", taking into
            // account window aspect ratio.  Eliminates distortion.
            case IDM_SQUAREVIEW:
            {
                MenuSquareView(hWnd);
                break;
            }

            // Recenter the current view at the point where the menu was
            // created, not the current mouse position or some bs like that.
            case IDM_CENTERVIEW:
            {
                MenuCenterView(hWnd, menuX, menuY);
                break;
            }

            case IDM_ZOOMIN:
            {
                MenuZoomIn(hWnd, { menuX, menuY });
                break;
            }

            case IDM_ZOOMOUT:
            {
                MenuZoomOut(hWnd, {menuX, menuY});
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
                MenuRepainting(hWnd);
                break;
            }

            // Make the fractal window a "window" instead of fullscreen
            case IDM_WINDOWED:
            {
                MenuWindowed(hWnd, false);
                break;
            }

            case IDM_WINDOWED_SQ:
            {
                MenuWindowed(hWnd, true);
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
                MenuAlgHelp(hWnd);
                break;
            }

            case IDM_ALG_AUTO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::AUTO);
                break;
            }

            case IDM_ALG_CPU_HIGH:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::CpuHigh);
                break;
            }

            case IDM_ALG_CPU_1_32_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::CpuHDR32);
                break;
            }

            case IDM_ALG_CPU_1_32_PERTURB_BLA_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu32PerturbedBLAHDR);
                break;
            }

            case IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu32PerturbedBLAV2HDR);
                break;
            }

            case IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLAV2HDR);
                break;
            }

            case IDM_ALG_CPU_1_64:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu64);
                break;
            }

            case IDM_ALG_CPU_1_64_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::CpuHDR64);
                break;
            }
    
            case IDM_ALG_CPU_1_64_PERTURB_BLA:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLA);
                break;
            }

            case IDM_ALG_CPU_1_64_PERTURB_BLA_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLAHDR);
                break;
            }

            case IDM_ALG_GPU_1_64:
            { 
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64);
                break;
            }

            case IDM_ALG_GPU_1_64_PERTURB_BLA:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64PerturbedBLA);
                break;
            }

            case IDM_ALG_GPU_2_64:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x64);
                break;
            }
            case IDM_ALG_GPU_4_64:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu4x64);
                break;
            }
            case IDM_ALG_GPU_2X32_HDR:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32);
                break;
            }
            case IDM_ALG_GPU_1_32:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32);
                break;
            }
            case IDM_ALG_GPU_1_32_PERTURB_SCALED:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedScaled);
                break;
            }
            case IDM_ALG_GPU_HDR_32_PERTURB_SCALED:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedScaled);
                break;
            }
            case IDM_ALG_GPU_1_32_PERTURB_SCALED_BLA:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedScaledBLA);
                break;
            }
            case IDM_ALG_GPU_2_32:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32);
                break;
            }
            case IDM_ALG_GPU_2_32_PERTURB_SCALED:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32PerturbedScaled);
                break;
            }
            case IDM_ALG_GPU_4_32:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu4x32);
                break;
            }

            case IDM_ALG_GPU_HDR_32_PERTURB_BLA:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedBLA);
                break;
            }

            case IDM_ALG_GPU_HDR_64_PERTURB_BLA:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx64PerturbedBLA);
                break;
            }

            case IDM_ALG_GPU_1_32_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_1_32_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedLAv2PO);
                break;
            }

            case IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedLAv2LAO);
                break;
            }

            case IDM_ALG_GPU_2_32_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_2_32_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32PerturbedLAv2PO);
                break;
            }

            case IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32PerturbedLAv2LAO);
                break;
            }

            case IDM_ALG_GPU_1_64_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_1_64_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64PerturbedLAv2PO);
                break;
            }

            case IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64PerturbedLAv2LAO);
                break;
            }

            case IDM_ALG_GPU_HDR_32_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedLAv2PO);
                break;
            }

            case IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO);
                break;
            }

            case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx2x32PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO);
                break;
            }

            case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO);
                break;
            }

            case IDM_ALG_GPU_HDR_64_PERTURB_LAV2:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx64PerturbedLAv2);
                break;
            }

            case IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx64PerturbedLAv2PO);
                break;
            }
            
            case IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO:
            {
                gFractal->SetRenderAlgorithm(RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO);
                break;
            }

            case IDM_BASICTEST:
            {
                gFractal->BasicTest();
                break;
            }

            // Increase the number of iterations we are using.
            // This will slow down rendering, but image quality
            // will be improved.
            case IDM_INCREASEITERATIONS_1P5X:
            {
                MenuMultiplyIterations(hWnd, 1.5);
                break;
            }
            case IDM_INCREASEITERATIONS_6X:
            {
                MenuMultiplyIterations(hWnd, 6.0);
                break;
            }

            case IDM_INCREASEITERATIONS_24X:
            {
                MenuMultiplyIterations(hWnd, 24.0);
                break;
            }
            // Decrease the number of iterations we are using
            case IDM_DECREASEITERATIONS:
            {
                MenuMultiplyIterations(hWnd, 2.0/3.0);
                break;
            }
            // Reset the number of iterations to the default
            case IDM_RESETITERATIONS:
            {
                MenuResetIterations(hWnd);
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
                    L"black screen so they're location is obvious.", L"TODO", MB_OK);
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

            case IDM_PERTURBATION_MULTITHREAD2_PERIODICITY_PERTURB_MTHIGH_MTMED:
            {
                gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed);
                break;
            }

            case IDM_PERTURBATION_MULTITHREAD5_PERIODICITY:
            {
                gFractal->SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity5);
                break;
            }

            case IDM_PERTURBATION_SAVE:
            {
                gFractal->SavePerturbationOrbit();
                break;
            }

            case IDM_PERTURBATION_LOAD:
            {
                gFractal->LoadPerturbationOrbit();
                break;
            }

            case IDM_PALETTEROTATE:
            {
                MenuPaletteRotation(hWnd);
                break;
            }
            case IDM_CREATENEWPALETTE:
            {
                MenuCreateNewPalette(hWnd);
                break;
            }

            case IDM_PALETTE_TYPE_0:
            {
                MenuPaletteType(Fractal::Palette::Basic);
                break;
            }

            case IDM_PALETTE_TYPE_1:
            {
                MenuPaletteType(Fractal::Palette::Default);
                break;
            }

            case IDM_PALETTE_TYPE_2:
            {
                MenuPaletteType(Fractal::Palette::Patriotic);
                break;
            }

            case IDM_PALETTE_TYPE_3:
            {
                MenuPaletteType(Fractal::Palette::Summer);
                break;
            }

            case IDM_PALETTE_TYPE_4:
            {
                MenuPaletteType(Fractal::Palette::Random);
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
                MenuGetCurPos(hWnd);
                break;
            }
            // Save/load current location
            case (IDM_SAVELOCATION):
            {
                MenuSaveCurrentLocation(hWnd);
                break;
            }
            //case (IDM_LOADLOCATION):
            //{
            //}
            case IDM_SAVEBMP:
            {
                MenuSaveBMP(hWnd);
                break;
            }
            case IDM_SAVEHIRESBMP:
            {
                MenuSaveHiResBMP(hWnd);
                break;
            }
            case IDM_SAVE_ITERS_TEXT:
            {
                MenuSaveItersAsText();
                break;
            }
            case IDM_BENCHMARK_ACCURATE:
            {
                MenuBenchmark(hWnd, false);
                break;
            }
            case IDM_BENCHMARK_QUICK:
            {
                MenuBenchmark(hWnd, true);
                break;
            }
            case IDM_BENCHMARK_ACCURATE_REFPT_DOUBLE:
            {
                MenuBenchmarkRefPtDouble(hWnd);
                break;
            }
            case IDM_BENCHMARK_ACCURATE_REFPT_HDRFLOAT:
            {
                MenuBenchmarkRefPtHDRFloat(hWnd);
                break;
            }
            case IDM_BENCHMARK_THIS:
            {
                MenuBenchmarkThis(hWnd);
                break;
            }
            case IDM_SHOWHOTKEYS:
            {
                MenuShowHotkeys(hWnd);
                break;
            }
            // Exit the program
            case IDM_EXIT:
            {
                DestroyWindow(hWnd);
                break;
            }
            // Catch-all
            default:
            {
                return DefWindowProc(hWnd, message, wParam, lParam); }
            }

            break;
        }

    case WM_SIZE:
    {
        if (gFractal != NULL)
        {
            gFractal->ResetDimensions(LOWORD(lParam), HIWORD(lParam));
            PaintAsNecessary(hWnd);
        }
        break;
    }
    // Display the popup menu with options
    case WM_CONTEXTMENU:
    {
        menuX = GET_X_LPARAM(lParam);
        menuY = GET_Y_LPARAM(lParam);

        TrackPopupMenu(GetSubMenu(gPopupMenu, 0), 0, menuX, menuY, 0, hWnd, NULL);

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
        if (gWindowed == true && IsDownAlt() == true)
        { // Don't drag a box if we are pressing the ALT key and this is windowed
            // mode.  Instead, move the window!
            PostMessage(hWnd, WM_NCLBUTTONDOWN, HTCAPTION, lParam);
        }
        else
        {
            if (lButtonDown == true)
            {
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
        if (lButtonDown == false || IsDownAlt() == true)
        {
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
        }
        else // Do anything
        {
            newView.left = dragBoxX1;
            newView.right = GET_X_LPARAM(lParam);
            newView.top = dragBoxY1;
            newView.bottom = GET_Y_LPARAM(lParam);
        }

        if (gFractal->RecenterViewScreen(newView) == true)
        {
            if (MaintainAspectRatio == true)
            {
                gFractal->SquareCurrentView();
            }

            RECT rect;
            GetClientRect(hWnd, &rect);

            PaintAsNecessary(hWnd);
        }
        break;
    }

    case WM_MOUSEMOVE:
    {
        if (lButtonDown == false)
        {
            break;
        }

        HDC dc = GetDC(hWnd);
        RECT rect;

        // Erase the previous rectangle.
        if (prevX1 != -1 || prevY1 != -1)
        {
            rect.left = dragBoxX1;
            rect.top = dragBoxY1;
            rect.right = prevX1;
            rect.bottom = prevY1;

            InvertRect(dc, &rect);
        }

        if (IsDownShift() == false)
        {
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
        }
        else
        {
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

    // Repaint the screen
    case WM_PAINT:
    {
        PaintAsNecessary(hWnd);

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
        HandleKeyDown(hWnd, message, wParam, lParam);
        PaintAsNecessary(hWnd);
        break;
    }

    // Catch-all.
    default:
    {
        return DefWindowProc(hWnd, message, wParam, lParam); }
    }

    return 0;
}

void MenuGoBack(HWND hWnd)
{
    if (gFractal->Back() == true)
    {
        PaintAsNecessary(hWnd);
    }
}

void MenuStandardView(HWND hWnd, size_t i)
{
    gFractal->View(i);
    PaintAsNecessary(hWnd);
}

void MenuSquareView(HWND hWnd){
    gFractal->SquareCurrentView();
    PaintAsNecessary(hWnd);
}

void MenuCenterView(HWND hWnd, int x, int y)
{
    gFractal->CenterAtPoint(x, y);
    PaintAsNecessary(hWnd);
}

void MenuZoomIn(HWND hWnd, POINT mousePt)
{
    gFractal->Zoom(mousePt.x, mousePt.y, -.45);
    PaintAsNecessary(hWnd);
}

void MenuZoomOut(HWND hWnd, POINT mousePt)
{
    gFractal->Zoom(mousePt.x, mousePt.y, 1);
    PaintAsNecessary(hWnd);
}

void MenuRepainting(HWND hWnd)
{
    gFractal->ToggleRepainting();
    PaintAsNecessary(hWnd);
}

void MenuWindowed(HWND hWnd, bool square)
{
    if (gWindowed == false)
    {
        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true)
        {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SendMessage(hWnd, WM_SYSCOMMAND, SC_RESTORE, 0);

        if (temporaryChange == true)
        {
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
        }
        else {
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
    }
    else
    {
        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        bool temporaryChange = false;
        if (gFractal->GetRepaint() == true)
        {
            gFractal->SetRepaint(false);
            temporaryChange = true;
        }

        SetWindowPos(hWnd, HWND_NOTOPMOST, 0, 0, width, height, SWP_SHOWWINDOW);
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

        if (temporaryChange == true)
        {
            gFractal->SetRepaint(true);
        }

        gWindowed = false;

        if (gFractal) {
            RECT rt;
            GetClientRect(hWnd, &rt);
            gFractal->ResetDimensions(rt.right, rt.bottom);
        }
    }
}

void MenuMultiplyIterations(HWND hWnd, double factor)
{
    if (gFractal->GetIterType() == IterTypeEnum::Bits32) {
        uint64_t curIters = gFractal->GetNumIterations<uint32_t>();
        curIters = (uint64_t)((double)curIters * (double)factor);
        gFractal->SetNumIterations<uint32_t>(curIters);
    }
    else {
        uint64_t curIters = gFractal->GetNumIterations<uint64_t>();
        curIters = (uint64_t)((double)curIters * (double)factor);
        gFractal->SetNumIterations<uint64_t>(curIters);
    }

    PaintAsNecessary(hWnd);
}

void MenuResetIterations(HWND hWnd)
{
    gFractal->ResetNumIterations();
    PaintAsNecessary(hWnd);
}

void MenuGetCurPos(HWND hWnd)
{
    constexpr size_t numBytes = 32768;

    BOOL ret = OpenClipboard(hWnd);
    if (ret == 0)
    {
        MessageBox(hWnd, L"Opening the clipboard failed.  Another program must be using it.", L"", MB_OK);
        return;
    }

    ret = EmptyClipboard();
    if (ret == 0)
    {
        MessageBox(hWnd, L"Emptying the clipboard of its current contents failed.  Make sure no other programs are using it.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, numBytes);
    if (hData == NULL)
    {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    char* mem = (char*)GlobalLock(hData);
    if (mem == NULL)
    {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    HighPrecision minX, minY;
    HighPrecision maxX, maxY;

    size_t prec = gFractal->GetPrecision();

    minX = gFractal->GetMinX();
    minY = gFractal->GetMinY();
    maxX = gFractal->GetMaxX();
    maxY = gFractal->GetMaxY();

    std::stringstream ss;
    std::string s;

    auto setupSS = [&](const HighPrecision& num) -> std::string {
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
    s = setupSS(pz.ptX);
    auto ptXStr = std::string(s.begin(), s.end());

    s = setupSS(pz.ptY);
    auto ptYStr = std::string(s.begin(), s.end());

    auto reducedPrecZF = pz.zoomFactor;
    reducedPrecZF.precision(10);
    s = setupSS(reducedPrecZF);
    auto zoomFactorStr = std::string(s.begin(), s.end());

    snprintf(
        mem,
        numBytes,
        "Using \"%s\"\r\n"
        "Antialiasing: %u\r\n"
        "Palette depth: %u\r\n"
        "Coordinate precision = %zu;\r\n"
        "Center X: \"%s\"\r\n"
        "Center Y: \"%s\"\r\n"
        "zoomFactor \"%s\"\r\n"
        "\r\n"
        "Bounding box:\r\n"
        "minX = HighPrecision{ \"%s\" };\r\n"
        "minY = HighPrecision{ \"%s\" };\r\n"
        "maxX = HighPrecision{ \"%s\" };\r\n"
        "maxY = HighPrecision{ \"%s\" };\r\n"
        "SetNumIterations(%zu);\r\n",
        gFractal->GetRenderAlgorithmName().c_str(),
        gFractal->GetGpuAntialiasing(),
        gFractal->GetPaletteDepth(),
        prec,
        ptXStr.c_str(),
        ptYStr.c_str(),
        zoomFactorStr.c_str(),
        sminX.c_str(), sminY.c_str(),
        smaxX.c_str(), smaxY.c_str(),
        gFractal->GetNumIterations<IterTypeFull>());
    ::MessageBoxA(hWnd, mem, "", MB_OK);

    GlobalUnlock(hData);

    //
    // This is not a memory leak - we don't "free" hData.
    //

    HANDLE clpData = SetClipboardData(CF_TEXT, hData);
    if (clpData == NULL)
    {
        MessageBox(hWnd, L"Adding the data to the clipboard failed.  You are probably very low on memory.  Try closing other programs or restarting your computer.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    CloseClipboard();
}

void MenuPaletteRotation(HWND)
{
    POINT OrgPos, CurPos;
    GetCursorPos(&OrgPos);

    for (;;)
    {
        gFractal->RotateFractalPalette(10);
        gFractal->DrawFractal(false);
        //SwapBuffers (gHDC);
        GetCursorPos(&CurPos);
        if (abs(CurPos.x - OrgPos.x) > 5 ||
            abs(CurPos.y - OrgPos.y) > 5)
        {
            break;
        }
    }

    gFractal->ResetFractalPalette();
    gFractal->DrawFractal(false);
}

void MenuPaletteType(Fractal::Palette type) {
    gFractal->UsePaletteType(type);
    if (type == Fractal::Palette::Default) {
        gFractal->UsePalette(8);
        gFractal->SetPaletteAuxDepth(0);
    }
    gFractal->DrawFractal(false);
}

void MenuPaletteDepth(int depth) {
    gFractal->UsePalette(depth);
    gFractal->DrawFractal(false);
}

void MenuCreateNewPalette(HWND) {
    gFractal->CreateNewFractalPalette();
    gFractal->UsePaletteType(Fractal::Palette::Random);
    gFractal->DrawFractal(false);
}

void MenuSaveCurrentLocation(HWND hWnd) {
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
    if (response == IDYES)
    {
        x = gFractal->GetRenderWidth();
        y = gFractal->GetRenderHeight();
        if (x > y)
        {
            y = (int)((double) 16384.0 / (double)((double)x / (double)y));
            x = 16384;
        }
        else if (x < y)
        {
            x = (int)((double) 16384.0 / (double)((double)y / (double)x));
            y = 16384;
        }
    }
    else
    {
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
    ss << gFractal->GetIterationPrecision() << " ";
    ss << filename << std::endl;
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());

    MessageBox(NULL, ws.c_str(), L"location", MB_OK);

    FILE *file = fopen("locations.txt", "at+");
    fprintf(file, "%s\r\n", s.c_str());
    fclose(file);
}

void MenuSaveBMP(HWND) {
    gFractal->SaveCurrentFractal(L"");
}

void MenuSaveHiResBMP(HWND) {
    gFractal->SaveHiResFractal(L"");
}

void MenuSaveItersAsText() {
    gFractal->SaveItersAsText(L"");
}

void BenchmarkMessage(HWND hWnd, HighPrecision megaIters, size_t milliseconds) {
    std::stringstream ss;
    ss << std::string("Your computer calculated ");
    ss << megaIters << " million iterations per second, took " << milliseconds << "ms.";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());
    MessageBox(hWnd, ws.c_str(), L"", MB_OK);

}

void MenuBenchmark(HWND hWnd, bool fastbenchmark)
{
    size_t milliseconds;
    HighPrecision megaIters = gFractal->Benchmark(fastbenchmark ? 5000 : 5000000, milliseconds);
    BenchmarkMessage(hWnd, megaIters, milliseconds);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkRefPtDouble(HWND hWnd)
{
    size_t milliseconds;
    HighPrecision megaIters = gFractal->BenchmarkReferencePoint<double, double>(5000000, milliseconds);
    BenchmarkMessage(hWnd, megaIters, milliseconds);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkRefPtHDRFloat(HWND hWnd)
{
    size_t milliseconds;
    HighPrecision megaIters = gFractal->BenchmarkReferencePoint<HDRFloat<double>, double>(5000000, milliseconds);
    BenchmarkMessage(hWnd, megaIters, milliseconds);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkThis(HWND hWnd) {
    size_t milliseconds;
    HighPrecision megaIters = gFractal->BenchmarkThis(milliseconds);
    BenchmarkMessage(hWnd, megaIters, milliseconds);
    gFractal->DrawFractal(false);
}

void MenuAlgHelp(HWND /*hWnd*/) {
    // This message box shows some help related to the algorithms.
    ::MessageBox(
        NULL,
        L"Algorithms\r\n"
        L"TODO\r\n"
        , L"Algorithms",
        MB_OK
    );
}

void MenuViewsHelp(HWND /*hWnd*/) {
    ::MessageBox(
        NULL,
        L"Views\r\n"
        L"\r\n"
        L"The purpose of these is simply to make it easy to navigate to\r\n"
        L"some interesting locations.\r\n"
        , L"Views",
        MB_OK);
}

void MenuShowHotkeys(HWND /*hWnd*/) {
    // Shows some basic help + hotkeys as defined in HandleKeyDown
    ::MessageBox(
        NULL,
        L"Hotkeys\r\n"
        L"\r\n"
        L"Navigation\r\n"
        L"a - Autozoom using averaging heuristic\r\n"
        L"A - Autozoom by zooming in on the highest iteration count point\r\n"
        L"b - Go back to the previous view\r\n"
        L"c - Center the view at the current mouse position\r\n"
        L"Z - Zoom out\r\n"
        L"z - Zoom in\r\n"
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
        L"Palettes\r\n"
        L"T - Use prior auxiliary palette depth (mul/div iteration count by 2)\r\n"
        L"t - Use next auxiliary palette depth (mul/div iteration count by 2)\r\n"
        L"D - Create and use new random palette\r\n"
        L"d - Use next palette lookup table depth\r\n"
        L"\r\n"
        L"Iterations\r\n"
        L"= - Multiply max iterations by 24\r\n"
        L"- - Multiply max iterations by 2/3\r\n"
        L"\r\n"
        L"Misc\r\n"
        L"CTRL - Press and hold to abort autozoom\r\n"
        L"ALT - Press, click/drag to move window when in windowed mode\r\n"
        L"Left click/drag - zoom in\r\n"
        L"Right click - popup menu\r\n"
        , L"",
        MB_OK);
}

void PaintAsNecessary(HWND hWnd)
{
    RECT rt;
    GetClientRect(hWnd, &rt);

    if (rt.left == 0 && rt.right == 0 && rt.top == 0 && rt.bottom == 0) {
        return;
    }

    if (gFractal != NULL)
    {
        gFractal->CalcFractal(false);
    }
}

// These functions are used to create a minidump when the program crashes.
typedef BOOL(WINAPI* MINIDUMPWRITEDUMP)(HANDLE hProcess, DWORD dwPid, HANDLE hFile, MINIDUMP_TYPE DumpType, CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam, CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam, CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

void create_minidump(struct _EXCEPTION_POINTERS* apExceptionInfo)
{
    HMODULE mhLib = ::LoadLibrary(_T("dbghelp.dll"));
    MINIDUMPWRITEDUMP pDump = (MINIDUMPWRITEDUMP)::GetProcAddress(mhLib, "MiniDumpWriteDump");

    HANDLE  hFile = ::CreateFile(_T("core.dmp"), GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL, NULL);

    _MINIDUMP_EXCEPTION_INFORMATION ExInfo;
    ExInfo.ThreadId = ::GetCurrentThreadId();
    ExInfo.ExceptionPointers = apExceptionInfo;
    ExInfo.ClientPointers = FALSE;

    pDump(GetCurrentProcess(), GetCurrentProcessId(), hFile, MiniDumpNormal, &ExInfo, NULL, NULL);
    ::CloseHandle(hFile);
}

LONG WINAPI unhandled_handler(struct _EXCEPTION_POINTERS* apExceptionInfo)
{
    create_minidump(apExceptionInfo);
    return EXCEPTION_CONTINUE_SEARCH;
}
