#include "StdAfx.h"

#include "CommandDispatcher.h"
#include "ConsoleWindow.h"
#include "CrashHandler.h"
#include "DynamicPopupMenu.h"
#include "Environment.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "GuiFileOperations.h"
#include "GuiHelp.h"
#include "JobObject.h"
#include "MainWindow.h"
#include "MenuState.h"
#include "PngParallelSave.h"
#include "SavedLocation.h"

#include "SplashWindow.h"
#include "resource.h"
#include <Windowsx.h>

#include <commdlg.h>
#include <string>

namespace {

constexpr bool startWindowed = true;
constexpr bool finishWindowed = false;
constexpr DWORD forceStartWidth = 0;
constexpr DWORD forceStartHeight = 0;
constexpr const wchar_t *kOrbitFileFilter = L"All\0*.*\0Imagina\0*.im\0";
constexpr const wchar_t *kPngFileFilter = L"PNG Image\0*.png\0All\0*.*\0";
constexpr const wchar_t *kTextFileFilter = L"Text File\0*.txt\0All\0*.*\0";

bool
IsNumpadAddSubtractCharacter(WPARAM wParam, LPARAM lParam) noexcept
{
    constexpr UINT numpadAddScanCode = 0x4e;
    constexpr UINT numpadSubtractScanCode = 0x4a;
    const UINT scanCode = static_cast<UINT>((static_cast<DWORD_PTR>(lParam) >> 16) & 0xffu);

    return (wParam == L'+' && scanCode == numpadAddScanCode) ||
           (wParam == L'-' && scanCode == numpadSubtractScanCode);
}

} // namespace

namespace FractalShark::Win32 {

MainWindow::MainWindow(HINSTANCE hInstance, int nCmdShow) : commandDispatcher(*this), hWnd{}
{
    gJobObj = std::make_unique<Environment::JobObject>();
    HighPrecision::defaultPrecisionInBits(256);

    Environment::CrashHandler::Install();

    // SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    // Initialize global strings
    MyRegisterClass(hInstance);

    // --- Splash (separate UI thread) ---
    Splash.Start(hInstance);

    auto startupBackgroundConsole = []() { AttachBackgroundConsole(true); };
    auto threadConsole = std::thread(startupBackgroundConsole);

    // Join before InitInstance since that might print stuff
    threadConsole.join();

    // Perform application initialization:
    hWnd = InitInstance(hInstance, nCmdShow);
    if (!hWnd) {
        throw FractalSharkSeriousException("Failed to create window.");
    }

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

// ---- Per-menu host hook implementation -------------------------------------

// ---- Synthetic shortcut command hooks --------------------------------------
void
MainWindow::OnAutoZoomFeatureAtPoint()
{
    const POINT pt = GetSafeMenuPtClient();
    gFractal->AutoZoomFeatureAtPoint(pt.x, pt.y);
}

// ---- Help / Window ---------------------------------------------------------
void
MainWindow::OnShowHotkeys()
{
    MenuShowHotkeys();
}
void
MainWindow::OnViewsHelp()
{
    MenuViewsHelp();
}
void
MainWindow::OnHelpAlg()
{
    MenuAlgHelp();
}
void
MainWindow::OnWindowed()
{
    MenuWindowed(false);
}
void
MainWindow::OnWindowedSq()
{
    MenuWindowed(true);
}
void
MainWindow::OnMinimize()
{
    ::PostMessage(hWnd, WM_SYSCOMMAND, SC_MINIMIZE, 0);
}
void
MainWindow::OnCurPos()
{
    MenuGetCurPos();
}
void
MainWindow::OnExit()
{
    ::DestroyWindow(hWnd);
}

// ---- Navigate --------------------------------------------------------------

// ---- Built-In Views (point entry) -----------------------------------------
// ---- Antialiasing ---------------------------------------------------------
// ---- Iterations -----------------------------------------------------------
// ---- Iteration precision --------------------------------------------------
// ---- Perturbation ---------------------------------------------------------
// ---- Memory / Autosave ---------------------------------------------------
// ---- Palette --------------------------------------------------------------
// ---- Save / Load ----------------------------------------------------------
void
MainWindow::OnSaveLocation()
{
    MenuSaveCurrentLocation();
}
void
MainWindow::OnSaveHiResBmp()
{
    MenuSaveHiResBMP();
}
void
MainWindow::OnSaveItersText()
{
    MenuSaveItersAsText();
}
void
MainWindow::OnSaveBmp()
{
    MenuSaveBMP();
}
void
MainWindow::OnSaveRefOrbitText()
{
    MenuSaveImag(CompressToDisk::Disable);
}
void
MainWindow::OnSaveRefOrbitTextSimple()
{
    MenuSaveImag(CompressToDisk::SimpleCompression);
}
void
MainWindow::OnSaveRefOrbitTextMax()
{
    MenuSaveImag(CompressToDisk::MaxCompression);
}
void
MainWindow::OnSaveRefOrbitImagMax()
{
    MenuSaveImag(CompressToDisk::MaxCompressionImagina);
}
void
MainWindow::OnDiffRefOrbitImagMax()
{
    MenuDiffImag();
}
void
MainWindow::OnLoadLocation()
{
    MenuLoadCurrentLocation();
}
void
MainWindow::OnLoadEnterLocation()
{
    MenuLoadEnterLocation();
}
void
MainWindow::OnLoadRefOrbitImagMax()
{
    MenuLoadImagDyn(ImaginaSettings::ConvertToCurrent);
}
void
MainWindow::OnLoadRefOrbitImagMaxSaved()
{
    MenuLoadImagDyn(ImaginaSettings::UseSaved);
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

    gFractal = std::make_unique<Fractal>(
        rt.right, rt.bottom, static_cast<void *>(hWnd), false, gJobObj->GetCommitLimitInBytes());

    // Create menu / popup (doesn't require the window to be shown)
    MenuState menuState(*gFractal);
    gPopupMenu = DynamicPopupMenu::Create(menuState);

    // SetRenderAlgorithm: TODO kind of gross but it works, reset now that
    // gFractal exists.  If CPU-only is enforced, this will show the radio
    // button the menu properly.  Without this, the menu is out of sync until
    // the user changes algorithm manually.
    ExecuteCommand(FractalShark::CommandFromIdm(IDM_ALG_AUTO));
    // commandDispatcher.Dispatch(IDM_ALG_GPU_HDR_64_PERTURB_LAV2);

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
MainWindow::HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM lParam)
{
    if (IsNumpadAddSubtractCharacter(wParam, lParam)) {
        return;
    }

    POINT mousePt;
    if (::GetCursorPos(&mousePt) == 0 || ::ScreenToClient(hWnd, &mousePt) == 0) {
        return;
    }
    lastMenuPtClient_ = mousePt;

    const FractalShark::HotKey hotkey = FractalShark::HotKeyFromCharacter(
        static_cast<wchar_t>(wParam), IsDownShift(), IsDownControl(), IsDownAlt());
    if (!hotkey.HasKey()) {
        return;
    }

    if (const FractalShark::Command *command = FractalShark::FindCommandByHotKey(hotkey);
        command != nullptr) {
        ExecuteCommand(command->id);
    }
}

void
MainWindow::HandleArrowAndZoomKeys(WPARAM vk)
{
    bool shiftDown = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    bool ctrlDown = (GetAsyncKeyState(VK_CONTROL) & 0x8000) != 0;

    // Pan fraction: 10% (Shift), 50% (Ctrl), 25% (default)
    double frac = 0.25;
    if (shiftDown) {
        frac = 0.10;
    } else if (ctrlDown) {
        frac = 0.50;
    }

    switch (vk) {
        case VK_LEFT:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(-frac, 0.0); });
            break;
        case VK_RIGHT:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(frac, 0.0); });
            break;
        case VK_UP:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(0.0, frac); });
            break;
        case VK_DOWN:
            gFractal->EnqueueCommand([frac](Fractal &f) { f.PanByFraction(0.0, -frac); });
            break;
        case VK_ADD:
            gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(-0.3); });
            break;
        case VK_SUBTRACT:
            gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); });
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
                    if (SetClipboardData(CF_TEXT, hMem) == nullptr) {
                        GlobalFree(hMem);
                    }
                } else {
                    GlobalFree(hMem);
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
            const std::string msg =
                std::string{"Message copied to clipboard.  CTRL-V to paste.\n"} + e.what();
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
MainWindow::OpenFileDialog(OpenBoxType type,
                           const wchar_t *filter,
                           const wchar_t *defaultExtension,
                           unsigned long saveFlags)
{
    OPENFILENAME ofn;    // common dialog box structure
    wchar_t szFile[260]; // buffer for file name

    // Initialize OPENFILENAME
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFile = szFile;
    ofn.lpstrFile[0] = '\0';
    ofn.nMaxFile = sizeof(szFile) / sizeof(szFile[0]);
    ofn.lpstrFilter = filter != nullptr ? filter : kOrbitFileFilter;
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.lpstrDefExt = defaultExtension;

    if (type == OpenBoxType::Open) {
        // Display the Open dialog box.
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        if (GetOpenFileName(&ofn) == TRUE) {
            return std::wstring(ofn.lpstrFile);
        } else {
            return std::wstring();
        }
    } else {
        ofn.Flags = saveFlags;
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

MenuPoint
MainWindow::GetMenuMousePos() const
{
    const POINT point = GetSafeMenuPtClient();
    return {point.x, point.y};
}

void
MainWindow::ActivateSavedOrbit(size_t index)
{
    ClearMenu(LoadSubMenu);
    EnqueueSavedLocation(*gFractal, gSavedLocations.at(index));
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

            // Native-only dynamic/dialog IDs sit below the portable catalog
            // range and must keep their raw command id payload.
            if (wmId < static_cast<int>(FractalShark::FractalCommand::ShowHotkeys)) {
                if (commandDispatcher.Dispatch(wmId)) {
                    return 0;
                }
            } else {
                // Static catalog commands go through the portable handlers.
                const auto cmd = FractalShark::CommandFromIdm(static_cast<uint32_t>(wmId));
                ExecuteCommand(cmd);
                return 0;
            }

            wchar_t buf[256];
            swprintf_s(buf,
                       L"Unknown WM_COMMAND.\n\nwmId = %d, wmId(base16) = 0x%X\n"
                       L"wmEvent = %d, wmEvent(base16) = 0x%X\n",
                       wmId,
                       wmId,
                       wmEvent,
                       wmEvent);

            std::wcerr << buf << L" Unknown menu item" << std::endl;

            return 0;
        }

        case WM_SIZE: {
            if (gFractal) {
                auto w = LOWORD(lParam);
                auto h = HIWORD(lParam);
                gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
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
            MenuState menuState(*gFractal);
            gPopupMenu = DynamicPopupMenu::Create(menuState);

            HMENU popup = DynamicPopupMenu::GetPopup(gPopupMenu.get());
            if (!popup) {
                return 0;
            }

            // Persist menu location as CLIENT coords before command dispatch.
            POINT ptClient = ptScreen;
            ::ScreenToClient(hWnd, &ptClient);
            lastMenuPtClient_ = ptClient;

            ::TrackPopupMenu(popup,
                             TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN,
                             ptScreen.x,
                             ptScreen.y,
                             0,
                             hWnd,
                             nullptr);

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

            // Clear the GL drag-rect overlay.
            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(false, 0, 0, 0, 0);
            }

            RECT newViewWin{};
            const bool maintainAspect = !IsDownShift();

            if (maintainAspect) {
                RECT windowRect;
                ::GetClientRect(hWnd, &windowRect);
                const double ratio = double(windowRect.right) / double(windowRect.bottom);

                newViewWin.left = dragBoxX1;
                newViewWin.top = dragBoxY1;
                newViewWin.bottom = GET_Y_LPARAM(lParam);
                newViewWin.right = LONG(double(newViewWin.left) +
                                        ratio * (double(newViewWin.bottom) - double(newViewWin.top)));
            } else {
                newViewWin.left = dragBoxX1;
                newViewWin.top = dragBoxY1;
                newViewWin.right = GET_X_LPARAM(lParam);
                newViewWin.bottom = GET_Y_LPARAM(lParam);
            }

            Environment::ScreenRect newView{
                newViewWin.left, newViewWin.top, newViewWin.right, newViewWin.bottom};

            if (gFractal) {
                gFractal->EnqueueCommand([newView, maintainAspect](Fractal &f) {
                    if (f.RecenterViewScreen(newView)) {
                        if (maintainAspect)
                            f.SquareCurrentView();
                    }
                });
            }

            return 0;
        }

        case WM_CANCELMODE:
        case WM_CAPTURECHANGED: {
            if (!lButtonDown)
                return 0;

            // Clear the GL drag-rect overlay.
            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(false, 0, 0, 0, 0);
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

            int rectX1, rectY1;

            if (IsDownShift() == false) {
                RECT windowRect;
                GetClientRect(hWnd, &windowRect);
                double ratio = (double)windowRect.right / (double)windowRect.bottom;

                rectY1 = GET_Y_LPARAM(lParam);
                rectX1 = (int)((double)dragBoxX1 + (double)ratio * (double)(rectY1 - dragBoxY1));

                prevX1 = rectX1;
                prevY1 = rectY1;
            } else {
                rectX1 = GET_X_LPARAM(lParam);
                rectY1 = GET_Y_LPARAM(lParam);

                prevX1 = rectX1;
                prevY1 = rectY1;
            }

            if (gFractal && gFractal->GetRenderPool()) {
                gFractal->GetRenderPool()->SetDragRect(true, dragBoxX1, dragBoxY1, rectX1, rectY1);
            }

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
                gFractal->EnqueueCommand(
                    [x = pt.x, y = pt.y](Fractal &f) { f.ZoomTowardPoint(x, y, -0.3); }, false);
            } else {
                gFractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); }, false);
            }

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
            return 0;
        }

        case WM_KEYDOWN: {
            HandleArrowAndZoomKeys(wParam);
            break;
        }

        default:
            break;
    }

    return DefWindowProc(hWnd, message, wParam, lParam);
}

void
MainWindow::MenuStandardView(size_t i)
{
    gFractal->EnqueueCommand([i](Fractal &f) { f.View(i); });
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
            auto w = rt.right;
            auto h = rt.bottom;
            gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
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
            auto w = rt.right;
            auto h = rt.bottom;
            gFractal->EnqueueCommand([w, h](Fractal &f) { f.ResetDimensions(w, h); });
        }
    }
}

void
MainWindow::MenuGetCurPos()
{
    std::string shortStr, longStr;
    gFractal->GetRenderDetails(shortStr, longStr);

    if (!Environment::SetClipboardText(longStr)) {
        std::wcerr << L"Copying the current position to the clipboard failed." << std::endl;
        return;
    }

    if (shortStr.length() < 5000) {
        ::MessageBoxA(hWnd, shortStr.c_str(), "Current Position", MB_OK | MB_APPLMODAL);
    } else {
        std::wcerr << L"Location copied to clipboard." << std::endl;
    }
}

void
MainWindow::MenuSaveCurrentLocation()
{
    const int response = ::MessageBox(hWnd, L"Scale dimensions to maximum?", L"Choose!", MB_YESNO);
    const std::string serialized = AppendSavedLocation(*gFractal, response == IDYES);
    const std::wstring message(serialized.begin(), serialized.end());
    ::MessageBox(nullptr, message.c_str(), L"Location", MB_OK | MB_APPLMODAL);
}

void
MainWindow::MenuLoadCurrentLocation()
{
    gSavedLocations = ReadSavedLocationsFile(kSavedLocationsFilename, 30);
    std::vector<std::string> labels = BuildSavedLocationLabels(gSavedLocations);
    HMENU submenu = ::CreatePopupMenu();
    for (size_t index = 0; index < labels.size(); ++index) {
        const std::string &description = labels[index];
        const std::wstring label(description.begin(), description.end());
        ::AppendMenu(submenu, MF_STRING, IDM_VIEW_DYNAMIC_ORBIT + index, label.c_str());
    }

    POINT point;
    ::GetCursorPos(&point);
    ::TrackPopupMenu(
        submenu, TPM_RIGHTBUTTON | TPM_LEFTALIGN | TPM_TOPALIGN, point.x, point.y, 0, hWnd, nullptr);
    ::DestroyMenu(submenu);
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

    struct Values : EnteredLocation {

        std::string
        ItersToString() const
        {
            return std::to_string(NumIterations);
        }

        void
        StringToIters(std::string new_iters)
        {
            NumIterations = std::stoull(new_iters);
        }
    };

    Values values;
    static_cast<EnteredLocation &>(values) = CaptureEnteredLocation(*gFractal);

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
                SetDlgItemTextA(hDlg, IDC_EDIT_REAL, values->Real.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_IMAG, values->Imaginary.c_str());
                SetDlgItemTextA(hDlg, IDC_EDIT_ZOOM, values->Zoom.c_str());
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
                    values->Real.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_REAL), values->Real.data(), len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_IMAG));
                    values->Imaginary.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_IMAG), values->Imaginary.data(), len + 1);

                    len = GetWindowTextLength(GetDlgItem(hDlg, IDC_EDIT_ZOOM));
                    values->Zoom.resize(len + 1);
                    GetWindowTextA(GetDlgItem(hDlg, IDC_EDIT_ZOOM), values->Zoom.data(), len + 1);

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

    if (values.Real.empty() || values.Imaginary.empty() || values.Zoom.empty()) {
        return;
    }

    // If OkOrCancel is 1, return.
    if (OkOrCancel == 1) {
        return;
    }

    EnqueueEnteredLocation(*gFractal, values);
}

void
MainWindow::MenuSaveBMP()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kPngFileFilter, L"png", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (!filename.empty()) {
        filename = AppendExtensionIfMissing(std::move(filename), L".png");
        SaveFractalOutput(*gFractal, FractalOutputFile::CurrentImage, std::move(filename));
    }
}

void
MainWindow::MenuSaveHiResBMP()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kPngFileFilter, L"png", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (!filename.empty()) {
        filename = AppendExtensionIfMissing(std::move(filename), L".png");
        SaveFractalOutput(*gFractal, FractalOutputFile::HighResolutionImage, std::move(filename));
    }
}

void
MainWindow::MenuSaveItersAsText()
{
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kTextFileFilter, L"txt", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (!filename.empty()) {
        filename = AppendExtensionIfMissing(std::move(filename), L".txt");
        SaveFractalOutput(*gFractal, FractalOutputFile::IterationsText, std::move(filename));
    }
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
    LoadReferenceOrbit(*gFractal, compressToDisk, loadSettings, std::move(filename));
}

void
MainWindow::MenuLoadImagDyn(ImaginaSettings loadSettings)
{
    ClearMenu(ImaginaMenu);

    ImaginaMenu = CreatePopupMenu();
    const auto imagFiles = FindReferenceOrbitFiles(std::filesystem::current_path(), 30);
    for (size_t index = 0; index < imagFiles.size(); ++index) {
        const std::wstring filename = imagFiles[index].wstring();
        const std::wstring label = imagFiles[index].filename().wstring();
        gImaginaLocations.push_back({filename, loadSettings});
        AppendMenu(ImaginaMenu, MF_STRING, IDM_VIEW_DYNAMIC_IMAG + index, label.c_str());
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
    std::wstring filename = OpenFileDialog(
        OpenBoxType::Save, kOrbitFileFilter, L"im", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (!filename.empty()) {
        filename = AppendExtensionIfMissing(std::move(filename), L".im");
        SaveReferenceOrbit(*gFractal, compression, std::move(filename));
    }
}

void
MainWindow::MenuDiffImag()
{
    std::wstring output = OpenFileDialog(
        OpenBoxType::Save, kOrbitFileFilter, L"im", OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT);
    if (output.empty()) {
        return;
    }
    output = AppendExtensionIfMissing(std::move(output), L".im");
    std::wstring first = OpenFileDialog(OpenBoxType::Open);
    if (first.empty()) {
        return;
    }
    std::wstring second = OpenFileDialog(OpenBoxType::Open);
    if (!second.empty()) {
        DiffReferenceOrbits(*gFractal, std::move(output), std::move(first), std::move(second));
    }
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
    const std::wstring title = GuiHelpTitleWide(GuiHelpTopic::Algorithms);
    const std::wstring body = GuiHelpBodyWide(GuiHelpTopic::Algorithms);
    ::MessageBox(nullptr, body.c_str(), title.c_str(), MB_OK);
}

void
MainWindow::MenuViewsHelp()
{
    const std::wstring title = GuiHelpTitleWide(GuiHelpTopic::Views);
    const std::wstring body = GuiHelpBodyWide(GuiHelpTopic::Views);
    ::MessageBox(nullptr, body.c_str(), title.c_str(), MB_OK);
}

void
MainWindow::MenuShowHotkeys()
{
    const std::wstring body = BuildHotkeysHelpWide();
    ::MessageBox(nullptr, body.c_str(), L"Hotkeys", MB_OK);
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
        if (auto *renderPool = gFractal->GetRenderPool()) {
            renderPool->RequestOverlayRepaint();
        }
    }
}

} // namespace FractalShark::Win32
