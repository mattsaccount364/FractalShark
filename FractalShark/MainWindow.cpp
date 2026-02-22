#include "StdAfx.h"

#include "CommandDispatcher.h"
#include "ConsoleWindow.h"
#include "CrummyTest.h"
#include "DynamicPopupMenu.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "MainWindowMenuState.h"
#include "MainWindowSavedLocation.h"
#include "OpenGLContext.h"
#include "PngParallelSave.h"
#include "RecommendedSettings.h"
#include "SplashWindow.h"
#include "WaitCursor.h"
#include "resource.h"

#include <commdlg.h>
#include <cstdio>
#include <minidumpapiset.h>
#include <random>

namespace {

constexpr bool startWindowed = true;
constexpr bool finishWindowed = false;
constexpr DWORD forceStartWidth = 0;
constexpr DWORD forceStartHeight = 0;

} // namespace

MainWindow::MainWindow(HINSTANCE hInstance, int nCmdShow) : commandDispatcher(*this), hWnd{}
{
    gJobObj = std::make_unique<JobObject>();
    HighPrecision::defaultPrecisionInBits(256);

    // Create a dump file whenever the gateway crashes only on windows
    SetUnhandledExceptionFilter(unhandled_handler);

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // --- Splash (separate UI thread) ---
    Splash.Start(hInstance);

    auto startupBackgroundConsole = []() { AttachBackgroundConsole(true); };
    auto threadConsole = std::thread(startupBackgroundConsole);

    // Perform application initialization:
    hWnd = InitInstance(hInstance, nCmdShow);
    if (!hWnd) {
        throw FractalSharkSeriousException("Failed to create window.");
    }

    threadConsole.join();

    // Main window is ready; close splash now.
    Splash.Stop();

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

//
// Registers the window class
// Note CS_OWNDC.  This is important for OpenGL.
//
ATOM
MainWindow::MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(wcex);

    wcex.style = CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)StaticWndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = sizeof(MainWindow *);
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIconW(hInstance, (LPCWSTR)IDI_FRACTALS);
    wcex.hCursor = LoadCursorW(nullptr, IDC_ARROW);

    // IMPORTANT: avoid the default white erase/paint before your first draw.
    // This makes the client area reliably black from the first frame.
    wcex.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIconW(hInstance, (LPCWSTR)IDI_SMALL);

    return RegisterClassExW(&wcex);
}

HWND
MainWindow::InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    (void)nCmdShow; // ignore debugger-dependent startup show state
    hInst = hInstance;

    // Use *virtual* screen metrics (multi-monitor correct) and signed ints.
    const int vLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vTop = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int vWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    const int vHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    // Default "start windowed" geometry (you can tweak)
    const int size = std::min(vWidth / 2, vHeight / 2);
    const int startX = vLeft + (vWidth - size) / 2;
    const int startY = vTop + (vHeight - size) / 2;
    int width = size;
    int height = size;

    if constexpr (forceStartWidth)
        width = (int)forceStartWidth;
    if constexpr (forceStartHeight)
        height = (int)forceStartHeight;

    // Create MAIN WINDOW hidden (do NOT show until gFractal is ready)
    DWORD style = WS_OVERLAPPEDWINDOW; // no WS_VISIBLE
    DWORD exStyle = WS_EX_APPWINDOW;

    hWnd = CreateWindowExW(exStyle,
                           szWindowClass,
                           L"FractalShark",
                           style,
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

    // Store "this" in the extra bytes immediately (StaticWndProc needs it)
    SetWindowLongPtrW(hWnd, 0, (LONG_PTR)this);

    // Don't set default algorithm on menu yet; wait until gFractal exists.

    // Apply your preferred mode while still hidden
    SetModeWindowed(startWindowed);

    // If you want to end fullscreen immediately, do it here (still hidden)
    if constexpr (finishWindowed == false) {

        SetModeWindowed(false);

        // Pick monitor nearest current window
        HMONITOR mon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
        MONITORINFO mi{};
        mi.cbSize = sizeof(mi);
        GetMonitorInfoW(mon, &mi);
        RECT r = mi.rcMonitor;

        // Force "normal" placement so Windows stops restoring elsewhere
        WINDOWPLACEMENT wp{};
        wp.length = sizeof(wp);
        wp.showCmd = 0;
        wp.flags = 0;
        wp.ptMinPosition = {0, 0};
        wp.ptMaxPosition = {0, 0};
        wp.rcNormalPosition = r;
        SetWindowPlacement(hWnd, &wp);

        SetWindowPos(hWnd,
                     HWND_NOTOPMOST,
                     r.left,
                     r.top,
                     r.right - r.left,
                     r.bottom - r.top,
                     SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE);
    }

    // Create fractal using FINAL client size
    RECT rt{};
    GetClientRect(hWnd, &rt);

    gFractal =
        std::make_unique<Fractal>(rt.right, rt.bottom, hWnd, false, gJobObj->GetCommitLimitInBytes());

    // Create menu / popup (doesn't require the window to be shown)
    MainWindowMenuState menuState(*this);
    gPopupMenu = FractalShark::DynamicPopupMenu::Create(menuState);

    // SetRenderAlgorithm: TODO kind of gross but it works, reset now that
    // gFractal exists.  If CPU-only is enforced, this will show the radio
    // button the menu properly.  Without this, the menu is out of sync until
    // the user changes algorithm manually.
    commandDispatcher.Dispatch(IDM_ALG_AUTO);
    //commandDispatcher.Dispatch(IDM_ALG_GPU_HDR_64_PERTURB_LAV2);

    // Optional: force an initial black fill before first show (prevents any flash)
    {
        HDC dc = GetDC(hWnd);
        RECT rc{};
        GetClientRect(hWnd, &rc);
        FillRect(dc, &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));
        ReleaseDC(hWnd, dc);
    }

    // Now show exactly once
    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    return hWnd;
}

void
MainWindow::ApplyBorderlessWindowedStyle()
{
    LONG_PTR style = GetWindowLongPtrW(hWnd, GWL_STYLE);
    style &= ~(WS_CAPTION | WS_SYSMENU | WS_MAXIMIZEBOX);
    style |= (WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX); // THICKFRAME is the resize enabler
    SetWindowLongPtrW(hWnd, GWL_STYLE, style);

    SetWindowPos(hWnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
}

void
MainWindow::ApplyBorderlessFullscreenStyle()
{
    LONG_PTR style = GetWindowLongPtrW(hWnd, GWL_STYLE);
    style &= ~(WS_THICKFRAME | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);
    style |= WS_POPUP;
    SetWindowLongPtrW(hWnd, GWL_STYLE, style);

    SetWindowPos(hWnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
}

void
MainWindow::SetModeWindowed(bool enableWindowed)
{
    gWindowed = enableWindowed;
    if (enableWindowed) {
        ApplyBorderlessWindowedStyle();
        // SetWindowPos to some desired restored rect
    } else {
        ApplyBorderlessFullscreenStyle();
        // SetWindowPos to monitor rect
    }
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
                gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Feature>();
            } else {
                MenuCenterView(mousePt.x, mousePt.y);
                gFractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
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

        case 'n': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::Direct);
            break;
        }

        case 'N': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::DirectScan);
            break;
        }

        case 'm': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::PT);
            break;
        }

        case 'M': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::PTScan);
            break;
        }

        case ',': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::LA);
            break;
        }

        case '<': {
            POINT pt;
            ::GetCursorPos(&pt);
            ::ScreenToClient(hWnd, &pt);
            gFractal->TryFindPeriodicPoint(pt.x, pt.y, FeatureFinderMode::LAScan);
            break;
        }

        case '.': {
            bool ret = gFractal->ZoomToFoundFeature();
            if (ret) {
                PaintAsNecessary();
            }
        }

        case '>': {
            gFractal->ClearAllFoundFeatures();
            PaintAsNecessary();
            break;
        }

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
            gFractal->EnqueueRender();
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
                gFractal->UsePaletteType(FractalPaletteType::Random);
            } else {
                gFractal->UseNextPaletteDepth();
            }
            gFractal->EnqueueRender();
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
            std::cerr << msg.c_str() << std::endl;
            return 0;
        } catch (const std::exception &e) {
            copyToClipboard(e.what());
            std::cerr << e.what() << std::endl;
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

LRESULT
MainWindow::WndProc(UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message) {
        case WM_NCACTIVATE: {
            if (gWindowed) {
                // Prevent Windows from drawing the standard non-client border
                // when activating/deactivating (focus changes).
                return TRUE;
            }
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

        case WM_NCPAINT: {
            if (gWindowed) {
                // Suppress non-client painting entirely in borderless windowed mode.
                return 0;
            }
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

        case WM_NCCALCSIZE:
            if (gWindowed) {
                return 0; // Remove all standard non-client area
            }
            break;

        case WM_NCHITTEST: {
            if (!gWindowed)
                break;

            constexpr int resizeBorder = 8; // pixels
            POINT pt{GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
            ScreenToClient(hWnd, &pt);

            RECT rc;
            GetClientRect(hWnd, &rc);

            const bool left = pt.x < resizeBorder;
            const bool right = pt.x >= rc.right - resizeBorder;
            const bool top = pt.y < resizeBorder;
            const bool bottom = pt.y >= rc.bottom - resizeBorder;

            if (top && left)
                return HTTOPLEFT;
            if (top && right)
                return HTTOPRIGHT;
            if (bottom && left)
                return HTBOTTOMLEFT;
            if (bottom && right)
                return HTBOTTOMRIGHT;
            if (left)
                return HTLEFT;
            if (right)
                return HTRIGHT;
            if (top)
                return HTTOP;
            if (bottom)
                return HTBOTTOM;

            return HTCLIENT;
        }

        case WM_COMMAND: {
            const int wmId = LOWORD(wParam);
            const int wmEvent = HIWORD(wParam);
            (void)wmEvent;

            if (commandDispatcher.Dispatch(wmId)) {
                return 0;
            }

            wchar_t buf[256];
            swprintf_s(buf,
                       L"Unknown WM_COMMAND.\n\nwmId = %d (0x%X)\nwmEvent = %d (0x%X)\n",
                       wmId,
                       wmId,
                       wmEvent,
                       wmEvent);

            std::wcerr << buf << L" Unknown menu item" << std::endl;

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

            // Rebuild the menu each time so dynamic state (radio/toggles/enabled) is fresh.
            MainWindowMenuState menuState(*this);
            gPopupMenu = FractalShark::DynamicPopupMenu::Create(menuState);

            HMENU popup = FractalShark::DynamicPopupMenu::GetPopup(gPopupMenu.get());
            if (!popup) {
                return 0;
            }

            ::TrackPopupMenu(popup,
                             TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN,
                             ptScreen.x,
                             ptScreen.y,
                             0,
                             hWnd,
                             nullptr);

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
            // Mouse wheel zoom control
            //
            // Windows convention:
            //   GET_WHEEL_DELTA_WPARAM(wParam) > 0  => wheel rotated FORWARD (away from user)
            //   GET_WHEEL_DELTA_WPARAM(wParam) < 0  => wheel rotated BACKWARD (toward user)
            //
            // UI convention (FractalShark):
            //   Wheel FORWARD  = zoom IN
            //   Wheel BACKWARD = zoom OUT

            POINT pt;
            pt.x = GET_X_LPARAM(lParam);
            pt.y = GET_Y_LPARAM(lParam);

            // Convert to client coordinates
            ScreenToClient(hWnd, &pt);

            if (GET_WHEEL_DELTA_WPARAM(wParam) > 0) {
                // Wheel FORWARD → ZOOM IN
                // Negative factor shrinks the bounding box in ZoomTowardPoint()
                gFractal->ZoomTowardPoint(pt.x, pt.y, -0.3);
            } else {
                // Wheel BACKWARD → ZOOM OUT
                // Smaller zoom factor expands the view
                gFractal->ZoomAtCenter(0.3);
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
            break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
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
    gFractal->ZoomRecentered(mousePt.x, mousePt.y, -.45);
    PaintAsNecessary();
}

void
MainWindow::MenuZoomOut(POINT mousePt)
{
    gFractal->ZoomRecentered(mousePt.x, mousePt.y, 1);
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

        SetModeWindowed(true);

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

        SetModeWindowed(false);

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
        std::wcerr << L"Opening the clipboard failed.  Another program must be using it." << std::endl;
        return;
    }

    ret = EmptyClipboard();
    if (ret == 0) {
        std::wcerr << L"Emptying the clipboard of its current contents failed.  Make sure no other "
                      L"programs are using it."
                   << std::endl;
        CloseClipboard();
        return;
    }

    HGLOBAL hData = GlobalAlloc(GMEM_MOVEABLE, numBytes);
    if (hData == nullptr) {
        std::wcerr << L"Insufficient memory." << std::endl;
        CloseClipboard();
        return;
    }

    char *mem = (char *)GlobalLock(hData);
    if (mem == nullptr) {
        std::wcerr << L"Insufficient memory 2." << std::endl;
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
        std::wcerr << L"Adding the data to the clipboard failed.  You are probably very low on memory.  "
                      L"Try closing other programs or restarting your computer."
                   << std::endl;
        CloseClipboard();
        return;
    }

    CloseClipboard();

    if (shortStr.length() < 5000) {
        ::MessageBoxA(hWnd, shortStr.c_str(), "Current Position", MB_OK | MB_APPLMODAL);
    } else {
        std::wcerr << L"Location copied to clipboard." << std::endl;
    }
}

void
MainWindow::MenuPaletteRotation()
{
    POINT OrgPos, CurPos;
    GetCursorPos(&OrgPos);

    for (;;) {
        gFractal->RotateFractalPalette(10);
        gFractal->EnqueueRender();
        GetCursorPos(&CurPos);
        if (abs(CurPos.x - OrgPos.x) > 5 || abs(CurPos.y - OrgPos.y) > 5) {
            break;
        }
    }

    gFractal->ResetFractalPalette();
    gFractal->EnqueueRender();
}

void
MainWindow::MenuPaletteType(FractalPaletteType type)
{
    gFractal->UsePaletteType(type);
    if (type == FractalPaletteType::Default) {
        gFractal->UsePalette(8);
        gFractal->SetPaletteAuxDepth(0);
    }
    gFractal->EnqueueRender();
}

void
MainWindow::MenuPaletteDepth(int depth)
{
    gFractal->UsePalette(depth);
    gFractal->EnqueueRender();
}

void
MainWindow::MenuCreateNewPalette()
{
    gFractal->CreateNewFractalPalette();
    gFractal->UsePaletteType(FractalPaletteType::Random);
    gFractal->EnqueueRender();
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
    TrackPopupMenu(
        hSubMenu, TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN, point.x, point.y, 0, hWnd, nullptr);

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

    PointZoomBBConverter pz{minX, minY, maxX, maxY, PointZoomBBConverter::TestMode::Enabled};
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

    PointZoomBBConverter pz2{realHP, imagHP, zoomHP, PointZoomBBConverter::TestMode::Enabled};
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
        const bool success = gFractal->SetRenderAlgorithm(settings.GetRenderAlgorithm());
        if (!success) {
            std::wcerr << L"Warning: Could not set render algorithm to AUTO." << std::endl;
        }
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
    TrackPopupMenu(
        ImaginaMenu, TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN, point.x, point.y, 0, hWnd, nullptr);

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
        L"poor 64-bit performance (even RTX 4090, 5090 etc)\r\n",
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
        L"a - Autozoom using feature heuristic (zooms toward perturbation reference point).  Hold CTRL to abort.\r\n"
        L"A - Autozoom using default heuristic (weighted geometric mean of iteration counts).  Hold CTRL to "
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
        L"Feature Finder\r\n"
        L"n - Find periodic point at cursor (Direct mode)\r\n"
        L"N - Find periodic point at cursor (DirectScan mode)\r\n"
        L"m - Find periodic point at cursor (PT mode)\r\n"
        L"M - Find periodic point at cursor (PTScan mode)\r\n"
        L", - Find periodic point at cursor (LA mode)\r\n"
        L"< - Find periodic point at cursor (LAScan mode)\r\n"
        L". - Zoom to found feature\r\n"
        L"> - Clear all found features\r\n"
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
        gFractal->EnqueueRender();
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
