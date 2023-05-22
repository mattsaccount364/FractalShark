#include "stdafx.h"
#include "resource.h"
#include "..\Fractal.h"
#include <GL/gl.h>			/* OpenGL header file */
#include <GL/glu.h>			/* OpenGL utilities header file */

// Global Variables:
HINSTANCE hInst;                // current instance
LPCWSTR szWindowClass = L"FractalWindow";
HMENU gPopupMenu;
bool gWindowed; // Says whether we are in windowed mode or not.
bool gRepainting;
bool gAltDraw;
HDC gHDC;

// Fractal:
Fractal *gFractal = NULL;

// Foward declarations of functions included in this code module:
ATOM              MyRegisterClass(HINSTANCE hInstance);
HWND              InitInstance(HINSTANCE);
LRESULT CALLBACK  WndProc(HWND, UINT, WPARAM, LPARAM);
void              UnInit(void);

// Controlling functions
void MenuGoBack(HWND hWnd);
void MenuStandardView(HWND hWnd, size_t i);
void MenuSquareView(HWND hWnd);
void MenuCenterView(HWND hWnd, int x, int y);
void MenuZoomOut(HWND hWnd);
void MenuRepainting(HWND hWnd);
void MenuWindowed(HWND hWnd);
void MenuIncreaseIterations(HWND hWnd, double factor);
void MenuDecreaseIterations(HWND hWnd);
void MenuResetIterations(HWND hWnd);
void MenuPaletteDepth(int depth);
void MenuPaletteRotation(HWND hWnd);
void MenuCreateNewPalette(HWND hWnd);
void MenuGetCurPos(HWND hWnd);
void MenuSaveCurrentLocation(HWND hWnd);
void MenuSaveBMP(HWND hWnd);
void MenuSaveHiResBMP(HWND hWnd);
void MenuBenchmark(HWND hWnd, bool fastbenchmark);
void MenuBenchmarkRefPtDouble(HWND hWnd);
void MenuBenchmarkRefPtHDRFloat(HWND hWnd);
void MenuBenchmarkThis(HWND hWnd);
void PaintAsNecessary(HWND hWnd);
void glResetView(HWND hWnd);
void glResetViewDim(int width, int height);

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

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // Perform application initialization:
    HWND hWnd = InitInstance(hInstance);
    if (hWnd == NULL)
    {
        return 1;
    }

    // More opengl initialization
    gHDC = GetDC(hWnd); // Grab on and don't let go until the program is done.
    HGLRC hRC = wglCreateContext(gHDC);
    wglMakeCurrent(gHDC, hRC);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Display!
    ShowWindow(hWnd, nCmdShow);
    SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

    // Main message loop:
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // OpenGL de-initialization
    wglMakeCurrent(NULL, NULL);
    ReleaseDC(hWnd, gHDC);
    wglDeleteContext(hRC);

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
HWND InitInstance(HINSTANCE hInstance)
{ // Store instance handle in our global variable
    hInst = hInstance;

    // Create the window
    HWND hWnd;
    hWnd = CreateWindow(szWindowClass, L"", WS_POPUP | WS_THICKFRAME,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        NULL, NULL, hInstance, NULL);

    if (!hWnd)
    {
        return NULL;
    }

    /////////////////////////////
    // opengl initialization
    /////////////////////////////
    HDC hDC = GetDC(hWnd);
    PIXELFORMATDESCRIPTOR pfd;
    int pf;

    /* there is no guarantee that the contents of the stack that become
       the pfd are zeroed, therefore _make sure_ to clear these bits. */
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; // | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;

    pf = ChoosePixelFormat(hDC, &pfd);
    if (pf == 0)
    {
        MessageBox(NULL, L"ChoosePixelFormat() failed: Cannot find a suitable pixel format.", L"Error", MB_OK);
        return NULL;
    }

    if (SetPixelFormat(hDC, pf, &pfd) == FALSE)
    {
        MessageBox(NULL, L"SetPixelFormat() failed:  Cannot set format specified.", L"Error", MB_OK);
        return NULL;
    }

    DescribePixelFormat(hDC, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

    ReleaseDC(hWnd, hDC);
    ////////////////////////////
    // end opengl initialization
    ////////////////////////////

    // Put us on top
    //SetWindowPos (hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);

    // Create the menu
    gPopupMenu = LoadMenu(hInst, MAKEINTRESOURCE(IDR_MENU_POPUP));

    // Initialize some global variables
    gWindowed = false;
    gRepainting = true;

    // Create the fractal
    RECT rt;
    GetClientRect(hWnd, &rt);

    FractalSetupData setupData;
    setupData.Load();
    gFractal = new Fractal(&setupData, rt.right, rt.bottom, NULL, hWnd, false);

    if (setupData.m_AltDraw == 'y')
    {
        gAltDraw = true;
    }
    else
    {
        gAltDraw = false;
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
    { int wmId, wmEvent;
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
        {
            assert(IDM_VIEW2 == IDM_VIEW1 + 1);
            assert(IDM_VIEW3 == IDM_VIEW1 + 2);
            assert(IDM_VIEW4 == IDM_VIEW1 + 3);
            assert(IDM_VIEW5 == IDM_VIEW1 + 4);
            assert(IDM_VIEW6 == IDM_VIEW1 + 5);
            assert(IDM_VIEW7 == IDM_VIEW1 + 6);
            assert(IDM_VIEW8 == IDM_VIEW1 + 7);
            assert(IDM_VIEW9 == IDM_VIEW1 + 8);
            assert(IDM_VIEW10 == IDM_VIEW1 + 9);
            assert(IDM_VIEW11 == IDM_VIEW1 + 10);
            assert(IDM_VIEW12 == IDM_VIEW1 + 11);
            assert(IDM_VIEW13 == IDM_VIEW1 + 12);

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

        case IDM_ZOOMOUT:
        {
            MenuZoomOut(hWnd);
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
            MenuWindowed(hWnd);
            break;
        }

        // Minimize the window
        case IDM_MINIMIZE:
        {
            PostMessage(hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0);
            break;
        }

        case IDM_ITERATION_ANTIALIASING_1X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1, UINT32_MAX);
            break;
        }

        case IDM_ITERATION_ANTIALIASING_4X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2, UINT32_MAX);
            break;
        }

        case IDM_ITERATION_ANTIALIASING_9X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 3, UINT32_MAX);
            break;
        }
        case IDM_ITERATION_ANTIALIASING_16X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4, UINT32_MAX);
            break;
        }

        case IDM_GPUANTIALIASING_1X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, UINT32_MAX, 1);
            break;
        }
        case IDM_GPUANTIALIASING_4X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, UINT32_MAX, 2);
            break;
        }
        case IDM_GPUANTIALIASING_9X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, UINT32_MAX, 3);
            break;
        }

        case IDM_GPUANTIALIASING_16X:
        {
            gFractal->ResetDimensions(MAXSIZE_T, MAXSIZE_T, UINT32_MAX, 4);
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
        case IDM_ALG_CPU_HIGH:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::CpuHigh);
            break;
        }
    
        case IDM_ALG_CPU_1_64:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Cpu64);
            break;
        }

        case IDM_ALG_CPU_1_64_HDR:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::CpuHDR);
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

        case IDM_ALG_GPU_1_64_PERTURB:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x64Perturbed);
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
        case IDM_ALG_GPU_1_32:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32);
            break;
        }
        case IDM_ALG_GPU_1_32_PERTURB:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32Perturbed);
            break;
        }
        case IDM_ALG_GPU_1_32_PERTURB_SCALED:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedScaled);
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
        case IDM_ALG_GPU_2_32_PERTURB:
        {
            gFractal->SetRenderAlgorithm(RenderAlgorithm::Gpu2x32Perturbed);
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

        // Increase the number of iterations we are using.
        // This will slow down rendering, but image quality
        // will be improved.
        case IDM_INCREASEITERATIONS_1P5X:
        {
            MenuIncreaseIterations(hWnd, 1.5);
            break;
        }
        case IDM_INCREASEITERATIONS_6X:
        { MenuIncreaseIterations(hWnd, 6.0);
            break;
        }

        case IDM_INCREASEITERATIONS_24X:
        {
            MenuIncreaseIterations(hWnd, 24.0);
            break;
        }
        // Decrease the number of iterations we are using
        case IDM_DECREASEITERATIONS:
        {
            MenuDecreaseIterations(hWnd);
            break;
        }
        // Reset the number of iterations to the default
        case IDM_RESETITERATIONS:
        {
            MenuResetIterations(hWnd);
            break;
        }

        case IDM_PERTURB_RESULTS:
        {
            gFractal->DrawPerturbationResults<double>(false);
            gFractal->DrawPerturbationResults<HDRFloat>(false);
            break;
        }
        case IDM_PERTURB_CLEAR:
        {
            gFractal->ClearPerturbationResults();
            break;
        }

        case IDM_PERTURBATION_SINGLETHREAD:
        {
            gFractal->SetPerturbationAlg(Fractal::PerturbationAlg::ST);
            break;
        }

        case IDM_PERTURBATION_MULTITHREAD:
        {
            gFractal->SetPerturbationAlg(Fractal::PerturbationAlg::MT);
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
            glResetViewDim(LOWORD(lParam), HIWORD(lParam));
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

void MenuZoomOut(HWND)
{
    gFractal->ZoomOut(1);
}

void MenuRepainting(HWND hWnd)
{
    gRepainting = !gRepainting;
    PaintAsNecessary(hWnd);
}

void MenuWindowed(HWND hWnd)
{
    if (gWindowed == false)
    {
        bool temporaryChange = false;
        if (gRepainting == true)
        {
            gRepainting = false;
            temporaryChange = true;
        }

        SendMessage(hWnd, WM_SYSCOMMAND, SC_RESTORE, 0);

        if (temporaryChange == true)
        {
            gRepainting = true;
        }

        RECT rect;
        GetWindowRect(hWnd, &rect);
        SetWindowPos(hWnd, HWND_NOTOPMOST,
            (rect.right - rect.left) / 4,
            (rect.bottom - rect.top) / 4,
            (rect.right - rect.left) / 2,
            (rect.bottom - rect.top) / 2,
            SWP_SHOWWINDOW);
        gWindowed = true;

        RECT rt;
        GetClientRect(hWnd, &rt);
        gFractal->ResetDimensions(rt.right, rt.bottom);

        glResetView(hWnd);
    }
    else
    {
        int width = GetSystemMetrics(SM_CXSCREEN);
        int height = GetSystemMetrics(SM_CYSCREEN);

        bool temporaryChange = false;
        if (gRepainting == true)
        {
            gRepainting = false;
            temporaryChange = true;
        }

        SetWindowPos(hWnd, HWND_NOTOPMOST, 0, 0, width, height, SWP_SHOWWINDOW);
        SendMessage(hWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

        if (temporaryChange == true)
        {
            gRepainting = true;
        }

        gWindowed = false;

        RECT rt;
        GetClientRect(hWnd, &rt);
        gFractal->ResetDimensions(rt.right, rt.bottom);

        glResetView(hWnd);
    }
}

void MenuIncreaseIterations(HWND hWnd, double factor)
{
    size_t curIters = gFractal->GetNumIterations();
    curIters = (size_t)((double)curIters * (double)factor);
    gFractal->SetNumIterations(curIters);
    PaintAsNecessary(hWnd);
}

void MenuDecreaseIterations(HWND)
{
    size_t curIters = gFractal->GetNumIterations();
    curIters = (curIters * 2) / 3;
    gFractal->SetNumIterations(curIters);
}

void MenuResetIterations(HWND hWnd)
{
    gFractal->ResetNumIterations();
    PaintAsNecessary(hWnd);
}

void MenuGetCurPos(HWND hWnd)
{
    wchar_t temp[256];

    double minX, minY;
    double maxX, maxY;

    minX = Convert<HighPrecision, double>(gFractal->GetMinX());
    minY = Convert<HighPrecision, double>(gFractal->GetMinY());
    maxX = Convert<HighPrecision, double>(gFractal->GetMaxX());
    maxY = Convert<HighPrecision, double>(gFractal->GetMaxY());

    wsprintf(temp, L"int ScreenWidth = %d;\r\nint ScreenHeight = %d;\r\ndouble MinX = %.15f;\r\ndouble MinY = %.15f;\r\ndouble MaxX = %.15f;\r\ndouble MaxY = %.15f;\r\nint numIters = %d;",
        gFractal->GetRenderWidth(), gFractal->GetRenderHeight(),
        minX, minY,
        maxX, maxY,
        gFractal->GetNumIterations());
    ::MessageBox(hWnd, temp, L"", MB_OK);

    wsprintf(temp, L"%.15f %.15f %.15f %.15f %d",
        //gFractal->GetRenderWidth (), gFractal->GetRenderHeight (),
        minX, minY,
        maxX, maxY,
        gFractal->GetNumIterations());

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

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, 256);
    if (hData == NULL)
    {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    char *mem = (char *)GlobalLock(hData);
    if (mem == NULL)
    {
        MessageBox(hWnd, L"Insufficient memory.", L"", MB_OK);
        CloseClipboard();
        return;
    }

    memcpy(mem, temp, 256);
    mem[255] = 0;

    GlobalUnlock(hData);

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

void MenuPaletteDepth(int depth)
{
    gFractal->UsePalette(depth);
}

void MenuCreateNewPalette(HWND)
{
    gFractal->CreateNewFractalPalette();
}

void MenuSaveCurrentLocation(HWND hWnd)
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
    ss << gFractal->GetNumIterations() << " ";
    ss << gFractal->GetIterationAntialiasing() << " ";
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

void MenuSaveBMP(HWND)
{
    gFractal->SaveCurrentFractal(L"");
}

void MenuSaveHiResBMP(HWND)
{
    gFractal->SaveHiResFractal(L"");
}

void BenchmarkMessage(HWND hWnd, HighPrecision megaIters) {
    std::stringstream ss;
    ss << std::string("Your computer calculated ");
    ss << megaIters << " million iterations per second";
    std::string s = ss.str();
    const std::wstring ws(s.begin(), s.end());
    MessageBox(hWnd, ws.c_str(), L"", MB_OK);

}

void MenuBenchmark(HWND hWnd, bool fastbenchmark)
{
    HighPrecision megaIters = gFractal->Benchmark(fastbenchmark ? 5000 : 5000000);
    BenchmarkMessage(hWnd, megaIters);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkRefPtDouble(HWND hWnd)
{
    HighPrecision megaIters = gFractal->BenchmarkReferencePoint<double>(5000000);
    BenchmarkMessage(hWnd, megaIters);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkRefPtHDRFloat(HWND hWnd)
{
    HighPrecision megaIters = gFractal->BenchmarkReferencePoint<HDRFloat>(5000000);
    BenchmarkMessage(hWnd, megaIters);
    gFractal->DrawFractal(false);
}

void MenuBenchmarkThis(HWND hWnd) {
    HighPrecision megaIters = gFractal->BenchmarkThis();
    BenchmarkMessage(hWnd, megaIters);
    gFractal->DrawFractal(false);
}

void PaintAsNecessary(HWND hWnd)
{
    RECT rt;
    GetClientRect(hWnd, &rt);

    if (gRepainting == false)
    {
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_LINES);

        glColor3f(1.0, 1.0, 1.0);
        glVertex2i(0, 0);
        glVertex2i(rt.right, rt.bottom);

        glVertex2i(rt.right, 0);
        glVertex2i(0, rt.bottom);

        glEnd();
        glFlush();
        return;
    }

    if (gFractal != NULL)
    {
        if (gAltDraw == true)
        {
            gFractal->CalcFractal(true);
            gFractal->DrawFractal(false);
        }
        else
        {
            gFractal->CalcFractal(false);
        }
        //SwapBuffers (gHDC);
    }
}

void glResetView(HWND hWnd)
{
    RECT rt;
    GetClientRect(hWnd, &rt);

    glResetViewDim(rt.right, rt.bottom);
}

void glResetViewDim(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, width, 0.0, height);
    glMatrixMode(GL_MODELVIEW);
}