// todo:
// Make the screen saver use a separate thread for the fractal rendering.
//   This way, the windows message pump can continue working.

#include "stdafx.h"
#include "resource.h"
#include "time.h"

#include "Fractal.h"

#include <GL/gl.h>			/* OpenGL header file */
#include <GL/glu.h>			/* OpenGL utilities header file */

// Global Variables:
HINSTANCE hInst;                // current instance
LPCWSTR szWindowClass = L"FractalSaverWindow";

HWND gHWnd;
volatile bool gTimeToExit;

// Foward declarations of functions included in this code module:
ATOM        MyRegisterClass(HINSTANCE hInstance);
BOOL        InitInstance(HINSTANCE);
LRESULT CALLBACK  WndProc(HWND, UINT, WPARAM, LPARAM);

HANDLE gDrawingThreadHandle;
unsigned long WINAPI DrawingThread(void *);
void glResetView(void);
void glResetViewDim(int width, int height);

int APIENTRY WinMain(HINSTANCE hInstance,
    HINSTANCE /*hPrevInstance*/,
    LPSTR     lpCmdLine,
    int       nCmdShow) {
    if (_strcmpi(lpCmdLine, "-s") != 0 &&
        _strcmpi(lpCmdLine, "/s") != 0 &&
        _strcmpi(lpCmdLine, "s") != 0) {
        return 0;
    }

    // Initialize the random number generator.
    srand((unsigned int)time(nullptr));

    // Hide the cursor
    ShowCursor(FALSE);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // Create the window
    if (InitInstance(hInstance) == FALSE) {
        return 1;
    }

    // Initialize the secondary drawing thread.
    gTimeToExit = false;

    DWORD threadID;
    gDrawingThreadHandle = (HANDLE)CreateThread(nullptr, 0, DrawingThread, nullptr, 0, &threadID);

    // Display!
    ShowWindow(gHWnd, nCmdShow);
    SendMessage(gHWnd, WM_SYSCOMMAND, SC_MAXIMIZE, 0);

    // Main message loop:
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Show the cursor
    ShowCursor(TRUE);

    // Cleanup the secondary drawing thread.
    gTimeToExit = true;
    WaitForSingleObject(gDrawingThreadHandle, INFINITE);

    return 0;
}

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM MyRegisterClass(HINSTANCE hInstance) {
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, (LPCTSTR)IDI_FRACTALSAVER);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, (LPCTSTR)IDI_SMALL);

    return RegisterClassEx(&wcex);
}

//
//   FUNCTION: InitInstance (HANDLE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//     Here we create the main window and return its handle,
//     and we perform other initialization.
//
BOOL InitInstance(HINSTANCE hInstance) { // Lower our priority
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
    SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);

    // Store instance handle in our global variable
    hInst = hInstance;

    // Create the window
    gHWnd = CreateWindow(szWindowClass, L"", WS_POPUP | WS_THICKFRAME,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, hInstance, nullptr);

    if (!gHWnd) {
        return FALSE;
    }

    return TRUE;
}

//
//  FUNCTION: WndProc (HWND, unsigned, WORD, LONG)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT  - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        BeginPaint(hWnd, &ps);
        EndPaint(hWnd, &ps);
        break;
    }
    case WM_MOUSEMOVE:
    {
        static int counter = 0;
        counter++;
        if (counter > 10) {
            DestroyWindow(hWnd);
        }
        break;
    }
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        break;
    }
    case WM_SIZE:
    {
        break;
    }
    default:
    { return DefWindowProc(hWnd, message, wParam, lParam); }
    }

    return 0;
}

unsigned long WINAPI DrawingThread(void *) { /////////////////////////////
    // opengl initialization
    /////////////////////////////
    HDC hDC = GetDC(gHWnd);
    PIXELFORMATDESCRIPTOR pfd;
    int pf;

    /* there is no guarantee that the contents of the stack that become
       the pfd are zeroed, therefore _make sure_ to clear these bits. */
    memset(&pfd, 0, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;

    pf = ChoosePixelFormat(hDC, &pfd);
    if (pf == 0) {
        MessageBox(nullptr, L"ChoosePixelFormat() failed: Cannot find a suitable pixel format.", L"Error", MB_OK | MB_APPLMODAL);
        return FALSE;
    }

    if (SetPixelFormat(hDC, pf, &pfd) == FALSE) {
        MessageBox(nullptr, L"SetPixelFormat() failed:  Cannot set format specified.", L"Error", MB_OK | MB_APPLMODAL);
        return FALSE;
    }

    DescribePixelFormat(hDC, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

    ReleaseDC(gHWnd, hDC);

    // More opengl initialization.  Set the current thread's context
    hDC = GetDC(gHWnd); // Grab on and don't let go until the thread is done.
    HGLRC hRC = wglCreateContext(hDC);
    wglMakeCurrent(hDC, hRC);
    ////////////////////////////
    // end opengl initialization
    ////////////////////////////

    // Create the fractal
    RECT rt;
    GetClientRect(gHWnd, &rt);

    glResetViewDim(rt.right, rt.bottom);

    Fractal *gFractal = nullptr;
    gFractal = DEBUG_NEW Fractal(rt.right, rt.bottom, gHWnd, true, 0);

    // Autozoom
    bool gAutoZoomDone = false;

    RECT nextView;
    nextView.top = -1;
    nextView.left = -1;
    nextView.bottom = -1;
    nextView.right = -1;

    for (;;) {
        if (gAutoZoomDone == false) {
            gFractal->ApproachTarget();
            gAutoZoomDone = true;
        }

        // Render the fractal and draw it after rendering is complete.
        gFractal->EnqueueRender().Wait();

        // Find a new target
        gFractal->FindInterestingLocation(&nextView);

        // Recenter the view there
        gFractal->RecenterViewScreen(nextView);
        gFractal->SquareCurrentView();

        // Increase iterations to ensure a high quality image
        gFractal->SetNumIterations<uint32_t>(3 * gFractal->GetNumIterations<uint32_t>() / 2);

        if (gTimeToExit == true) {
            break;
        }
    }

    // Deallocate fractal memory
    delete gFractal;

    // OpenGL de-initialization.
    wglMakeCurrent(nullptr, nullptr);
    ReleaseDC(gHWnd, hDC);
    wglDeleteContext(hRC);

    return 0;
}

void glResetView(void) {
    RECT rt;
    GetClientRect(gHWnd, &rt);

    glResetViewDim(rt.right, rt.bottom);
}

void glResetViewDim(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, width, 0.0, height);
    glMatrixMode(GL_MODELVIEW);
}