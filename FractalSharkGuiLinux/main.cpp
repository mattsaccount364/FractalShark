// FractalSharkGuiLinux - Xlib + Dear ImGui GUI shell.

#include "CommandCatalog.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "FeatureFinderMode.h"
#include "Fractal.h"
#include "LinuxClipboard.h"
#include "LinuxCommandHandlers.h"
#include "LinuxHelpModals.h"
#include "LinuxImGuiOverlay.h"
#include "LinuxMenuState.h"
#include "LinuxX11ContextMenu.h"
// LinuxMenuState.h pulls in MenuTree.h which #undefs the X11 `None` and
// `Always` macros (they collide with FractalShark::Menu enum values).
// X.h has a header guard so it can't be re-included to restore them — just
// redefine the two we use locally.
#ifndef None
#define None 0L
#endif
#ifndef Always
#define Always 2
#endif
#include "OpenGLContext.h"
#include "RecommendedSettings.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "RenderThreadPool.h"
#include "WaitCursor.h"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>

#include <GL/glx.h>

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

struct GlxVisualSelection {
    XVisualInfo *VisualInfo;
};

struct PresentationTickResult {
    bool FreshFrame;
    bool NeedsTick;
};

struct DragEndpoint {
    int X;
    int Y;
};

DragEndpoint
ComputeAspectLockedDragEndpoint(int anchorX, int anchorY, int rawX, int rawY, double aspectRatio)
{
    const int dy = rawY - anchorY;
    const double width = aspectRatio * std::abs(static_cast<double>(dy));

    int horizontalSign = 0;
    if (rawX > anchorX) {
        horizontalSign = 1;
    } else if (rawX < anchorX) {
        horizontalSign = -1;
    } else {
        horizontalSign = dy < 0 ? -1 : 1;
    }

    return {static_cast<int>(static_cast<double>(anchorX) + static_cast<double>(horizontalSign) * width),
            rawY};
}

GlxVisualSelection
ChooseGlxVisual(Display *display, int screen)
{
    int rgbaDoubleBufferedWithAlpha[] = {GLX_RGBA,
                                         GLX_DOUBLEBUFFER,
                                         GLX_RED_SIZE,
                                         8,
                                         GLX_GREEN_SIZE,
                                         8,
                                         GLX_BLUE_SIZE,
                                         8,
                                         GLX_ALPHA_SIZE,
                                         8,
                                         None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaDoubleBufferedWithAlpha)) {
        return {visualInfo};
    }

    int rgbaDoubleBuffered[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaDoubleBuffered)) {
        return {visualInfo};
    }

    int rgbaSingleBuffered[] = {GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaSingleBuffered)) {
        return {visualInfo};
    }

    return {};
}

long long
ElapsedMilliseconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 start)
        .count();
}

std::string
NarrowAscii(std::wstring_view text)
{
    std::string result;
    result.reserve(text.size());
    for (const wchar_t ch : text) {
        result.push_back((ch >= 0 && ch <= 0x7f) ? static_cast<char>(ch) : '?');
    }
    return result;
}

std::string
BuildHotkeysModalBody()
{
    std::string body = "Hotkeys\n\nCommand shortcuts\n";
    for (const FractalShark::Command &command : FractalShark::kCommands) {
        body += FractalShark::FormatHotKeyUtf8(command.hotkey);
        body += " - ";
        body += NarrowAscii(command.label);
        body += "\n";
    }

    body += "\nDirect controls\n"
            "Arrow keys - Pan viewport 25% of the view. Shift+Arrow: 10%, Ctrl+Arrow: 50%\n"
            "Numpad + - Zoom in at center\n"
            "Numpad - - Zoom out at center\n"
            "Left click/drag - Zoom in\n"
            "Right click - popup menu\n"
            "CTRL - Press and hold to abort autozoom\n"
            "ALT - Press, click/drag to move window when in windowed mode\n";
    return body;
}

std::optional<wchar_t>
HotKeyBaseFromKeySym(KeySym keysym)
{
    if (keysym >= XK_A && keysym <= XK_Z) {
        return static_cast<wchar_t>(L'a' + (keysym - XK_A));
    }
    if (keysym >= XK_a && keysym <= XK_z) {
        return static_cast<wchar_t>(L'a' + (keysym - XK_a));
    }
    if (keysym >= XK_0 && keysym <= XK_9) {
        return static_cast<wchar_t>(L'0' + (keysym - XK_0));
    }

    switch (keysym) {
        case XK_comma:
            return L',';
        case XK_less:
            return L',';
        case XK_period:
            return L'.';
        case XK_greater:
            return L'.';
        case XK_equal:
            return L'=';
        case XK_plus:
            return L'=';
        case XK_minus:
            return L'-';
        default:
            return std::nullopt;
    }
}

std::optional<FractalShark::HotKey>
HotKeyFromXKeyEvent(const XKeyEvent &ev)
{
    XKeyEvent mutableEvent = ev;
    const KeySym baseKeySym = XLookupKeysym(&mutableEvent, 0);
    const std::optional<wchar_t> key = HotKeyBaseFromKeySym(baseKeySym);
    if (!key) {
        return std::nullopt;
    }

    return FractalShark::NormalizeHotKey(
        {*key, (ev.state & ShiftMask) != 0, (ev.state & ControlMask) != 0, (ev.state & Mod1Mask) != 0});
}

// Mechanical-contract stub: every catalog command hook the Win32 GUI
// implements in MainWindow must also be overridden here.  Until the Linux
// GUI grows real menu/render plumbing each hook just announces itself so a
// developer wiring up a UI surface can see which commands fire.  A missing
// hook produces a compile error (pure virtual not implemented), which is
// exactly the signal we want from Phase 0c.
#define FRACTALSHARK_LINUX_STUB(method)                                                                 \
    void method() override { std::fprintf(stderr, "TODO LinuxMainWindow: %s\n", #method); }

struct LinuxMainWindow : FractalSharkLinux::LinuxCommandHandlers {
    Display *display = nullptr;
    Window window = 0;
    Colormap colormap = 0;
    XVisualInfo *visualInfo = nullptr;
    Atom wmDeleteWindow = 0;
    Atom wmProtocols = 0;
    Cursor idleCursor = 0;
    Cursor dragCursor = 0;
    int screen = 0;
    bool running = true;
    bool firstExposeSeen = false;
    int lastWidth = kInitialWidth;
    int lastHeight = kInitialHeight;

    // Window-mode (fullscreen) state.  Mirrors Win32 gWindowed in
    // MainWindow.cpp.  Linux uses EWMH _NET_WM_STATE_FULLSCREEN to toggle.
    bool fullscreen = false;
    int savedX = 0;
    int savedY = 0;
    int savedWidth = kInitialWidth;
    int savedHeight = kInitialHeight;
    bool everPresented = false;
    bool exposeRepaintPending = false;

    // Right-click position in client coordinates, matching MainWindow.cpp's
    // m_LastMenuPtClient anchor for menu commands that act on the click point.
    int contextMenuX = 0;
    int contextMenuY = 0;

    // Left-button drag-zoom state.  Mirrors MainWindow's lButtonDown +
    // dragBoxX1/Y1 (anchor) + prevX1/Y1 (last cursor for outline rect).
    // Linux uses XGrabPointer during the gesture to mirror Win32
    // SetCapture/ReleaseCapture behavior.
    bool dragging = false;
    bool pointerGrabbed = false;
    int dragAnchorX = 0;
    int dragAnchorY = 0;
    int dragPrevX = -1;
    int dragPrevY = -1;

    std::optional<FractalShark::LinuxClipboard> clipboard;
    std::unique_ptr<Fractal> fractal;
    std::optional<FractalSharkLinux::LinuxMenuState> menuState;
    // GL context is owned by the GUI thread (host-owned presentation
    // mode).  Created after Fractal so the X window/visual are ready;
    // destroyed before Fractal so any pool teardown that touches the
    // cached frame buffer happens before GL goes away.
    std::unique_ptr<OpenGlContext> glContext;
    std::optional<FractalShark::Linux::ImGuiOverlay> overlay;
    std::optional<FractalShark::Linux::X11ContextMenu> contextMenu;

    LinuxMainWindow();
    ~LinuxMainWindow() override;

    LinuxMainWindow(const LinuxMainWindow &) = delete;
    LinuxMainWindow &operator=(const LinuxMainWindow &) = delete;

    bool
    Valid() const noexcept
    {
        return display != nullptr && window != 0 && clipboard.has_value() && fractal != nullptr &&
               menuState.has_value() && glContext != nullptr && overlay.has_value();
    }
    void RunEventLoop();
    void HandleEvent(const XEvent &ev);
    void HandleKeyPress(const XKeyEvent &ev);
    PresentationTickResult PresentRenderTick();
    void RepaintAfterNativePopupUnmap();
    int GetPresentationPollTimeoutMs(bool needsTick);
    void RunFeatureAutoZoomSynchronously(int mouseX, int mouseY);
    void BeginDragZoom(const XButtonEvent &btn);
    void CancelDragZoom(Time eventTime);
    void SetDragCursorActive(bool active);
    void StartWindowMove(const XButtonEvent &btn);
    void FinishDragZoom(const XButtonEvent &btn);
    void EnterFullscreen(bool square);
    void ExitFullscreen();

    // ---- LinuxCommandHandlers protected accessors -------------------------
    Fractal &
    GetFractal() noexcept override
    {
        return *fractal;
    }
    FractalSharkLinux::MenuPoint
    GetMenuMousePos() const override
    {
        return {contextMenuX, contextMenuY};
    }

    // ---- ExecuteCommandHost stubs (Linux-specific only) -------------------
    void OnAutoZoomFeatureAtPoint() override;
    // Everything else is provided by LinuxCommandHandlers.  These ~25 hooks
    // touch the X server, ImGui modals, the file dialog, the clipboard, or
    // process-wide JobObject state — so they need real Linux implementations
    // (forthcoming Phase 3.3.2 / 3.3.4 / 3.3.6 / 3.3.7) and stay as stubs
    // here until then.
    void OnShowHotkeys() override;
    void OnViewsHelp() override;
    void OnHelpAlg() override;
    void OnWindowed() override;
    void OnWindowedSq() override;
    void OnMinimize() override;
    void OnCurPos() override;
    void OnExit() override;

    // TODO: Wire the Linux JobObject/RLIMIT_AS implementation into the GUI and
    // define reversible unlimited/limited behavior before enabling these menu
    // items.
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit0)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit1)
    void OnPaletteRotate() override;

    void OnSaveLocation() override;
    void OnSaveHiResBmp() override;
    void OnSaveItersText() override;
    void OnSaveBmp() override;
    void OnSaveRefOrbitText() override;
    void OnSaveRefOrbitTextSimple() override;
    void OnSaveRefOrbitTextMax() override;
    void OnSaveRefOrbitImagMax() override;
    void OnDiffRefOrbitImagMax() override;
    void OnLoadLocation() override;
    void OnLoadEnterLocation() override;
    void OnLoadRefOrbitImagMax() override;
    void OnLoadRefOrbitImagMaxSaved() override;

private:
    void SaveLocation(bool scaleToMaximum);
    void DoSaveRefOrbit(::CompressToDisk compression);
    void DoLoadRefOrbitImag(::ImaginaSettings settings);
};

#undef FRACTALSHARK_LINUX_STUB

LinuxMainWindow::LinuxMainWindow()
{
    display = XOpenDisplay(nullptr);
    if (!display) {
        std::fprintf(stderr, "FractalSharkGuiLinux: XOpenDisplay failed (DISPLAY env not set?)\n");
        return;
    }

    screen = DefaultScreen(display);

    // Pick the best GLX-compatible visual available.  OpenGlContext later
    // creates its context from this window's actual visual, keeping fallback
    // selection in one place.
    const GlxVisualSelection visualSelection = ChooseGlxVisual(display, screen);
    visualInfo = visualSelection.VisualInfo;
    if (!visualInfo) {
        std::fprintf(stderr,
                     "FractalSharkGuiLinux: glXChooseVisual failed (tried RGBA double-buffered with "
                     "alpha, RGBA double-buffered without required alpha, and RGBA single-buffered "
                     "without required alpha). Check GLX support with glxinfo.\n");
        XCloseDisplay(display);
        display = nullptr;
        return;
    }

    Window root = RootWindow(display, screen);

    colormap = XCreateColormap(display, root, visualInfo->visual, AllocNone);
    idleCursor = XCreateFontCursor(display, XC_left_ptr);
    dragCursor = XCreateFontCursor(display, XC_crosshair);

    XSetWindowAttributes swa{};
    swa.colormap = colormap;
    swa.background_pixel = BlackPixel(display, screen);
    swa.border_pixel = BlackPixel(display, screen);
    if (idleCursor) {
        swa.cursor = idleCursor;
    }
    // File-drop is not a parity target: neither GUI exposes OS drag/drop as
    // part of the documented FractalShark workflow.
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask | FocusChangeMask;

    window = XCreateWindow(
        display,
        root,
        0,
        0,
        kInitialWidth,
        kInitialHeight,
        0,
        visualInfo->depth,
        InputOutput,
        visualInfo->visual,
        CWColormap | CWBackPixel | CWBorderPixel | CWEventMask | (idleCursor ? CWCursor : 0),
        &swa);

    XStoreName(display, window, kWindowTitle);

    // Class hint — lets WMs recognize and theme the app.
    XClassHint classHint{};
    std::string resName = "fractalshark";
    std::string resClass = "FractalShark";
    classHint.res_name = resName.data();
    classHint.res_class = resClass.data();
    XSetClassHint(display, window, &classHint);

    // Receive WM_DELETE_WINDOW client messages instead of having the WM kill us.
    wmProtocols = XInternAtom(display, "WM_PROTOCOLS", False);
    wmDeleteWindow = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wmDeleteWindow, 1);
    WaitCursor::RegisterLinuxCursorTarget(
        display, static_cast<uintptr_t>(window), static_cast<uintptr_t>(idleCursor));

    // Match Win32 KEYDOWN semantics: only autorepeat KeyPress, never KeyRelease.
    Bool supported = False;
    XkbSetDetectableAutoRepeat(display, True, &supported);

    XMapWindow(display, window);
    XFlush(display);

    clipboard.emplace(display, window);

    // Instantiate Fractal in host-owned GL presentation mode: the render
    // pool will not start its own GL consumer thread; we drive
    // presentation ourselves on the GUI thread below.
    fractal = std::make_unique<Fractal>(kInitialWidth,
                                        kInitialHeight,
                                        reinterpret_cast<void *>(static_cast<uintptr_t>(window)),
                                        /*UseSensoCursor=*/false,
                                        /*commitLimitInBytes=*/UINT64_MAX,
                                        /*hostOwnedGlPresentation=*/true);

    // Create the GLX context on this (the GUI) thread.  OpenGlContext's
    // constructor binds GL to this thread via glXMakeCurrent — every
    // subsequent GL call (RenderFrameToGL, ImGui, SwapBuffers) must
    // happen on this thread.
    glContext =
        std::make_unique<OpenGlContext>(reinterpret_cast<void *>(static_cast<uintptr_t>(window)));
    if (!glContext->IsValid()) {
        std::fprintf(stderr, "FractalSharkGuiLinux: OpenGlContext creation failed; running headless.\n");
    }

    // ImGui overlay: single-threaded, all calls on this thread.  Init
    // requires GL to be current, which OpenGlContext::ctor already
    // ensured.
    menuState.emplace(*fractal, fullscreen);
    overlay.emplace(display, window, &*clipboard);
    contextMenu.emplace(
        display, screen, window, &*menuState, this, [this] { RepaintAfterNativePopupUnmap(); });
    if (glContext && glContext->IsValid()) {
        if (!overlay->Init()) {
            std::fprintf(stderr, "FractalSharkGuiLinux: ImGui backend init failed.\n");
        }
    }
}

LinuxMainWindow::~LinuxMainWindow()
{
    CancelDragZoom(CurrentTime);

    // Destroy in reverse order of creation: overlay first (it owns ImGui
    // backends that need GL current), then GL context, then Fractal.
    contextMenu.reset();
    overlay.reset();
    menuState.reset();
    glContext.reset();
    fractal.reset();

    if (display) {
        WaitCursor::UnregisterLinuxCursorTarget();
        if (window) {
            XDestroyWindow(display, window);
            window = 0;
        }
        if (colormap) {
            XFreeColormap(display, colormap);
            colormap = 0;
        }
        if (dragCursor) {
            XFreeCursor(display, dragCursor);
            dragCursor = 0;
        }
        if (idleCursor) {
            XFreeCursor(display, idleCursor);
            idleCursor = 0;
        }
        if (visualInfo) {
            XFree(visualInfo);
            visualInfo = nullptr;
        }
        XCloseDisplay(display);
        display = nullptr;
    }
}

void
LinuxMainWindow::HandleEvent(const XEvent &ev)
{
    // Native menus own root-level X11 windows and grabs while open.  Route
    // their events before ImGui or the fractal window sees them.
    if (contextMenu && contextMenu->ProcessEvent(ev)) {
        return;
    }

    // Forward to the ImGui overlay first.  Overlay processes the event
    // synchronously on this thread; if ImGui is capturing the input we
    // drop the event rather than dispatching it to the fractal.
    if (overlay && overlay->ProcessEvent(ev)) {
        return;
    }

    switch (ev.type) {
        case ClientMessage: {
            const auto &cm = ev.xclient;
            if (cm.message_type == wmProtocols && static_cast<Atom>(cm.data.l[0]) == wmDeleteWindow) {
                running = false;
            }
            break;
        }

        case DestroyNotify:
            running = false;
            break;

        case Expose:
            // Kick off a single render on the first Expose.  The Fractal's
            // RenderThreadPool computes off-thread; the GL consumer (also a
            // pool thread) presents via glXSwapBuffers when the frame is ready.
            // Later exposes still matter: native X11 popup menus uncover the
            // GL window and require the cached frame to be presented again.
            exposeRepaintPending = true;
            if (!firstExposeSeen && fractal) {
                firstExposeSeen = true;
                fractal->EnqueueRender();
            }
            break;

        case ConfigureNotify: {
            // X sends ConfigureNotify for moves *and* resizes.  Filter to actual
            // dimension changes so we don't thrash the render queue.  Mirrors
            // Win32 WM_SIZE handling at MainWindow.cpp:1190.
            const auto &cfg = ev.xconfigure;
            if (fractal && (cfg.width != lastWidth || cfg.height != lastHeight)) {
                lastWidth = cfg.width;
                lastHeight = cfg.height;
                const int w = cfg.width;
                const int h = cfg.height;
                fractal->EnqueueCommand([w, h](Fractal &f) {
                    f.ResetDimensions(static_cast<size_t>(w), static_cast<size_t>(h));
                });
            }
            break;
        }

        case ButtonPress: {
            const auto &btn = ev.xbutton;
            if (!fractal) {
                break;
            }
            switch (btn.button) {
                case Button1: {
                    // Alt+LMB → window move (mirrors Win32 WM_NCLBUTTONDOWN
                    // HTCAPTION dispatch at MainWindow.cpp:1240).  Otherwise begin
                    // drag-zoom capture.
                    if (btn.state & Mod1Mask) {
                        StartWindowMove(btn);
                        break;
                    }
                    if (dragging) {
                        break;
                    }
                    BeginDragZoom(btn);
                    break;
                }
                case Button3:
                    // Keep client coords for Center/Zoom commands, but open the
                    // native popup at root coords so it may leave the app window.
                    contextMenuX = btn.x;
                    contextMenuY = btn.y;
                    if (contextMenu) {
                        contextMenu->Open(btn.x_root, btn.y_root);
                    }
                    break;
                case Button4: {
                    // Wheel forward → zoom in toward cursor.  Mirrors
                    // MainWindow.cpp:1393 `ZoomTowardPoint(x, y, -0.3)`.
                    const int x = btn.x;
                    const int y = btn.y;
                    fractal->EnqueueCommand([x, y](Fractal &f) { f.ZoomTowardPoint(x, y, -0.3); });
                    break;
                }
                case Button5:
                    // Wheel backward → zoom out at center.  Mirrors
                    // MainWindow.cpp:1397 `ZoomAtCenter(0.3)`.
                    fractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); });
                    break;
                default:
                    // Button2 (MMB) currently unused on Win32; leave as-is.
                    break;
            }
            break;
        }

        case ButtonRelease: {
            const auto &btn = ev.xbutton;
            if (btn.button != Button1 || !dragging) {
                break;
            }
            FinishDragZoom(btn);
            break;
        }

        case MotionNotify: {
            if (!dragging) {
                break;
            }
            const auto &mot = ev.xmotion;
            // Aspect-lock unless Shift is held, mirroring MainWindow.cpp:1345.
            if ((mot.state & ShiftMask) == 0) {
                const double ratio = (lastHeight > 0) ? (double)lastWidth / (double)lastHeight : 1.0;
                const DragEndpoint endpoint =
                    ComputeAspectLockedDragEndpoint(dragAnchorX, dragAnchorY, mot.x, mot.y, ratio);
                dragPrevX = endpoint.X;
                dragPrevY = endpoint.Y;
            } else {
                dragPrevX = mot.x;
                dragPrevY = mot.y;
            }
            // Live outline rect rendered by the overlay's foreground draw list.
            if (overlay) {
                overlay->SetDragRect(true, dragAnchorX, dragAnchorY, dragPrevX, dragPrevY);
            }
            break;
        }

        case FocusOut:
            // Focus loss cancels an in-flight drag-zoom (mirrors
            // WM_CANCELMODE/WM_CAPTURECHANGED at MainWindow.cpp:1305).  Drop
            // anchors silently — no Recenter is enqueued.
            CancelDragZoom(CurrentTime);
            break;

        case KeyPress:
            HandleKeyPress(ev.xkey);
            break;

        default:
            break;
    }
}

void
LinuxMainWindow::RunEventLoop()
{
    if (!display) {
        return;
    }

    const int xfd = ConnectionNumber(display);

    XEvent ev;
    while (running) {
        // Drain any X events already queued on the connection without
        // blocking — they may have been deposited while the previous
        // iteration was rendering.
        while (XPending(display) > 0) {
            XNextEvent(display, &ev);
            if (clipboard->ProcessEvent(ev)) {
                continue;
            }
            HandleEvent(ev);
            if (!running) {
                break;
            }
        }
        if (!running) {
            break;
        }

        const auto presentation = PresentRenderTick();

        // Sleep until either an X event arrives or our 16ms tick budget
        // expires.  No need for an explicit wake from the pool — the next
        // tick will pick the frame up.
        struct pollfd pfd{};
        pfd.fd = xfd;
        pfd.events = POLLIN;
        poll(&pfd, 1, GetPresentationPollTimeoutMs(presentation.NeedsTick));
    }
}

void
LinuxMainWindow::RepaintAfterNativePopupUnmap()
{
    exposeRepaintPending = true;
    (void)PresentRenderTick();
}

PresentationTickResult
LinuxMainWindow::PresentRenderTick()
{
    // Pull any frames the render pool has finished and present them.
    // TryPresentTick skips ready tombstones and uploads at most one
    // visible frame so each completed animation step gets a swap.
    bool freshFrame = false;
    if (fractal && glContext && glContext->IsValid()) {
        auto *pool = fractal->GetRenderPool();
        if (pool) {
            freshFrame = pool->TryPresentTick(*glContext);
        }
    }

    const bool overlayWantsTick = overlay && overlay->WantsTick();
    const bool needsTick =
        freshFrame || overlayWantsTick || dragging || exposeRepaintPending || !everPresented;

    if (needsTick && glContext && glContext->IsValid()) {
        static bool diagnosedFirstFreshFrame = false;
        const bool diagnoseFreshFrame = freshFrame && !diagnosedFirstFreshFrame;
        if (diagnoseFreshFrame) {
            diagnosedFirstFreshFrame = true;
        }

        // If this tick was triggered by overlay motion only, re-blit
        // the cached frame so the overlay has something underneath.
        if (!freshFrame && fractal) {
            if (auto *pool = fractal->GetRenderPool()) {
                pool->RepresentLastFrame(*glContext);
            }
        }
        if (overlay) {
            std::chrono::steady_clock::time_point overlayStart;
            if (diagnoseFreshFrame) {
                GlLog("LinuxMainWindow: first completed-frame ImGui draw begin");
                overlayStart = std::chrono::steady_clock::now();
            }

            overlay->RenderFrame();

            if (diagnoseFreshFrame) {
                char buf[128];
                snprintf(buf,
                         sizeof(buf),
                         "LinuxMainWindow: first completed-frame ImGui draw end, elapsedMs=%lld",
                         ElapsedMilliseconds(overlayStart));
                GlLog(buf);
            }
        } else if (diagnoseFreshFrame) {
            GlLog("LinuxMainWindow: first completed-frame ImGui draw skipped");
        }

        std::chrono::steady_clock::time_point swapStart;
        if (diagnoseFreshFrame) {
            GlLog("LinuxMainWindow: first completed-frame glXSwapBuffers begin");
            swapStart = std::chrono::steady_clock::now();
        }

        glContext->SwapBuffers();

        if (diagnoseFreshFrame) {
            char buf[128];
            snprintf(buf,
                     sizeof(buf),
                     "LinuxMainWindow: first completed-frame glXSwapBuffers end, elapsedMs=%lld",
                     ElapsedMilliseconds(swapStart));
            GlLog(buf);
        }

        everPresented = true;
        exposeRepaintPending = false;
    }

    return {freshFrame, needsTick};
}

int
LinuxMainWindow::GetPresentationPollTimeoutMs(bool needsTick)
{
    int timeoutMs = needsTick ? 16 : 33;
    if (fractal) {
        if (auto *pool = fractal->GetRenderPool()) {
            if (auto pacingDelay = pool->GetTimeUntilNextPresentation()) {
                timeoutMs = std::min(timeoutMs, static_cast<int>(pacingDelay->count()));
            }
        }
    }
    return std::max(timeoutMs, 1);
}

void
LinuxMainWindow::RunFeatureAutoZoomSynchronously(int mouseX, int mouseY)
{
    if (!fractal || !glContext || !glContext->IsValid()) {
        std::fprintf(stderr, "Feature autozoom requires a valid Linux GL context.\n");
        return;
    }

    std::atomic<bool> finished = false;
    std::exception_ptr autoZoomException;
    std::thread autoZoomThread([&, mouseX, mouseY] {
        try {
            fractal->AutoZoomFeatureAtPoint(mouseX, mouseY);
        } catch (...) {
            autoZoomException = std::current_exception();
        }
        finished.store(true, std::memory_order_release);
    });

    for (;;) {
        const auto presentation = PresentRenderTick();
        const auto *pool = fractal->GetRenderPool();
        const auto pacingDelay =
            pool ? pool->GetTimeUntilNextPresentation() : std::optional<std::chrono::milliseconds>{};
        if (finished.load(std::memory_order_acquire) && !presentation.FreshFrame && !pacingDelay) {
            break;
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds(GetPresentationPollTimeoutMs(presentation.NeedsTick)));
    }

    autoZoomThread.join();
    if (autoZoomException) {
        std::rethrow_exception(autoZoomException);
    }
}

void
LinuxMainWindow::OnAutoZoomFeatureAtPoint()
{
    const FractalSharkLinux::MenuPoint pt = GetMenuMousePos();
    RunFeatureAutoZoomSynchronously(pt.X, pt.Y);
}

void
LinuxMainWindow::OnShowHotkeys()
{
    if (overlay) {
        const std::string body = BuildHotkeysModalBody();
        overlay->RequestInfoModal("Hotkeys", body.c_str());
    }
}

void
LinuxMainWindow::OnViewsHelp()
{
    if (overlay) {
        overlay->RequestInfoModal(FractalSharkLinux::kViewsModalTitle,
                                  FractalSharkLinux::kViewsModalBody);
    }
}

void
LinuxMainWindow::OnHelpAlg()
{
    if (overlay) {
        overlay->RequestInfoModal(FractalSharkLinux::kAlgorithmsModalTitle,
                                  FractalSharkLinux::kAlgorithmsModalBody);
    }
}

void
LinuxMainWindow::OnMinimize()
{
    if (display && window) {
        XIconifyWindow(display, window, screen);
        XFlush(display);
    }
}

void
LinuxMainWindow::OnCurPos()
{
    std::string shortStr;
    std::string longStr;
    fractal->GetRenderDetails(shortStr, longStr);
    clipboard->Set(longStr);
    if (shortStr.size() < 5000 && overlay) {
        overlay->RequestInfoModal("Current Position", shortStr.c_str());
    } else {
        std::fprintf(stderr, "Location copied to clipboard.\n");
    }
}

void
LinuxMainWindow::OnPaletteRotate()
{
    // TODO(palette-rotation): This is visually broken because RotateFractalPalette() updates the
    // internal palette state without forcing the displayed image to be re-rendered or recolored,
    // so the command appears to do nothing.
    const auto origin = Environment::GetCursorPosition();

    for (;;) {
        fractal->EnqueueCommand([](Fractal &f) { f.RotateFractalPalette(10); }).Wait();

        const auto current = Environment::GetCursorPosition();
        if (std::abs(current.first - origin.first) > 5 || std::abs(current.second - origin.second) > 5) {
            break;
        }
    }

    fractal->EnqueueCommand([](Fractal &f) { f.ResetFractalPalette(); });
}

void
LinuxMainWindow::OnExit()
{
    running = false;
}

void
LinuxMainWindow::OnWindowed()
{
    if (fullscreen) {
        ExitFullscreen();
    } else {
        EnterFullscreen(false);
    }
}

void
LinuxMainWindow::OnWindowedSq()
{
    if (fullscreen) {
        ExitFullscreen();
    } else {
        EnterFullscreen(true);
    }
}

void
LinuxMainWindow::EnterFullscreen(bool square)
{
    if (!display || !window || fullscreen) {
        return;
    }

    // Cache geometry so ExitFullscreen can restore it.  XGetWindowAttributes
    // x/y are relative to the parent (often a WM frame or root); for restore
    // purposes that's fine since we just XMoveResizeWindow back to the same
    // values.
    XWindowAttributes attrs{};
    if (XGetWindowAttributes(display, window, &attrs)) {
        Window dummyChild = 0;
        int rx = 0, ry = 0;
        XTranslateCoordinates(display, window, RootWindow(display, screen), 0, 0, &rx, &ry, &dummyChild);
        savedX = rx;
        savedY = ry;
        savedWidth = attrs.width;
        savedHeight = attrs.height;
    }

    if (square) {
        // For the square arm, resize *before* requesting fullscreen so the
        // WM lays the window out at our chosen square size.  Most WMs
        // ignore size on a fullscreen toggle (they cover the whole monitor)
        // — that's the documented Win32 behaviour we are mirroring loosely.
        const int side = std::min(savedWidth, savedHeight);
        if (side > 0) {
            XResizeWindow(display, window, side, side);
        }
    }

    Atom wmState = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wmFullscreen = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);
    if (wmState == None || wmFullscreen == None) {
        return;
    }

    XEvent ev{};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = window;
    ev.xclient.message_type = wmState;
    ev.xclient.format = 32;
    ev.xclient.data.l[0] = 1; // _NET_WM_STATE_ADD
    ev.xclient.data.l[1] = static_cast<long>(wmFullscreen);
    ev.xclient.data.l[2] = 0;
    ev.xclient.data.l[3] = 1; // source = normal application
    ev.xclient.data.l[4] = 0;
    XSendEvent(display,
               RootWindow(display, screen),
               False,
               SubstructureRedirectMask | SubstructureNotifyMask,
               &ev);
    XFlush(display);
    fullscreen = true;
}

void
LinuxMainWindow::ExitFullscreen()
{
    if (!display || !window || !fullscreen) {
        return;
    }

    Atom wmState = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wmFullscreen = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);
    if (wmState == None || wmFullscreen == None) {
        return;
    }

    XEvent ev{};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = window;
    ev.xclient.message_type = wmState;
    ev.xclient.format = 32;
    ev.xclient.data.l[0] = 0; // _NET_WM_STATE_REMOVE
    ev.xclient.data.l[1] = static_cast<long>(wmFullscreen);
    ev.xclient.data.l[2] = 0;
    ev.xclient.data.l[3] = 1;
    ev.xclient.data.l[4] = 0;
    XSendEvent(display,
               RootWindow(display, screen),
               False,
               SubstructureRedirectMask | SubstructureNotifyMask,
               &ev);

    if (savedWidth > 0 && savedHeight > 0) {
        XMoveResizeWindow(display, window, savedX, savedY, savedWidth, savedHeight);
    }
    XFlush(display);
    fullscreen = false;
}

namespace {

// Build a "output_YYYY_MM_DD_HH_MM_SS" filename stem (no extension).
std::string
MakeTimestampedStem()
{
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[64];
    std::snprintf(buf,
                  sizeof(buf),
                  "output_%04d_%02d_%02d_%02d_%02d_%02d",
                  tm.tm_year + 1900,
                  tm.tm_mon + 1,
                  tm.tm_mday,
                  tm.tm_hour,
                  tm.tm_min,
                  tm.tm_sec);
    return std::string(buf);
}

std::wstring
Utf8ToWide(const std::string &s)
{
    std::wstring w;
    w.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const auto c0 = static_cast<uint8_t>(s[i]);
        if (c0 < 0x80) {
            w.push_back(static_cast<wchar_t>(c0));
            ++i;
            continue;
        }

        uint32_t codePoint = 0;
        size_t length = 0;
        uint32_t minValue = 0;
        if ((c0 & 0xe0) == 0xc0) {
            codePoint = c0 & 0x1f;
            length = 2;
            minValue = 0x80;
        } else if ((c0 & 0xf0) == 0xe0) {
            codePoint = c0 & 0x0f;
            length = 3;
            minValue = 0x800;
        } else if ((c0 & 0xf8) == 0xf0) {
            codePoint = c0 & 0x07;
            length = 4;
            minValue = 0x10000;
        } else {
            w.push_back(static_cast<wchar_t>(0xfffd));
            ++i;
            continue;
        }

        if (i + length > s.size()) {
            w.push_back(static_cast<wchar_t>(0xfffd));
            break;
        }

        bool valid = true;
        for (size_t j = 1; j < length; ++j) {
            const auto cj = static_cast<uint8_t>(s[i + j]);
            if ((cj & 0xc0) != 0x80) {
                valid = false;
                break;
            }
            codePoint = (codePoint << 6) | (cj & 0x3f);
        }

        if (!valid || codePoint < minValue || (codePoint >= 0xd800 && codePoint <= 0xdfff) ||
            codePoint > 0x10ffff) {
            w.push_back(static_cast<wchar_t>(0xfffd));
            ++i;
            continue;
        }

        w.push_back(static_cast<wchar_t>(codePoint));
        i += length;
    }
    return w;
}

using LinuxFileDialogFilter = FractalShark::Linux::ImGuiOverlay::FileDialogFilter;
using LinuxFileDialogMode = FractalShark::Linux::ImGuiOverlay::FileDialogMode;

std::vector<LinuxFileDialogFilter>
OrbitFileFilters()
{
    return {{"All (*.*)", ""}, {"Imagina (*.im)", ".im"}};
}

std::vector<LinuxFileDialogFilter>
PngFileFilters()
{
    return {{"PNG Image (*.png)", ".png"}, {"All (*.*)", ""}};
}

std::vector<LinuxFileDialogFilter>
TextFileFilters()
{
    return {{"Text File (*.txt)", ".txt"}, {"All (*.*)", ""}};
}

std::string
AppendExtensionIfMissing(const std::string &filename, const char *extension)
{
    if (filename.empty() || extension == nullptr || extension[0] == '\0') {
        return filename;
    }

    try {
        if (std::filesystem::path(filename).extension().empty()) {
            return filename + extension;
        }
    } catch (const std::filesystem::filesystem_error &) {
    }
    return filename;
}

} // namespace

void
LinuxMainWindow::OnSaveLocation()
{
    if (!fractal || !overlay) {
        return;
    }

    overlay->RequestPickFromList("Scale dimensions to maximum?",
                                 {"Scale dimensions to maximum", "Use current dimensions"},
                                 [this](size_t index) {
                                     if (index <= 1) {
                                         SaveLocation(index == 0);
                                     }
                                 });
}

void
LinuxMainWindow::SaveLocation(bool scaleToMaximum)
{
    if (!fractal || !overlay) {
        return;
    }

    size_t width = fractal->GetRenderWidth();
    size_t height = fractal->GetRenderHeight();
    if (scaleToMaximum) {
        if (width > height) {
            height = static_cast<size_t>(16384.0 / (static_cast<double>(width) / height));
            width = 16384;
        } else if (width < height) {
            width = static_cast<size_t>(16384.0 / (static_cast<double>(height) / width));
            height = 16384;
        }
    }

    std::ostringstream record;
    record << width << " ";
    record << height << " ";
    record << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
    record << fractal->GetMinX() << " ";
    record << fractal->GetMinY() << " ";
    record << fractal->GetMaxX() << " ";
    record << fractal->GetMaxY() << " ";
    record << fractal->GetNumIterations<IterTypeFull>() << " ";
    record << fractal->GetGpuAntialiasing() << " ";
    record << "FractalTrayDestination";

    const std::string serialized = record.str();
    std::ofstream file("locations.txt", std::ios::app);
    if (!file) {
        overlay->RequestInfoModal("Save Location", "Could not open locations.txt for append.");
        return;
    }
    file << serialized << "\r\n";
    file.close();
    if (!file) {
        overlay->RequestInfoModal("Save Location", "Could not write the location to locations.txt.");
        return;
    }

    overlay->RequestInfoModal("Location", serialized.c_str());
}

void
LinuxMainWindow::OnSaveHiResBmp()
{
    if (!fractal || !overlay) {
        return;
    }
    overlay->RequestFileDialog("Save high-resolution image",
                               LinuxFileDialogMode::Save,
                               MakeTimestampedStem() + ".png",
                               PngFileFilters(),
                               [this](std::string filename) {
                                   if (!fractal)
                                       return;
                                   filename = AppendExtensionIfMissing(filename, ".png");
                                   if (auto *pool = fractal->GetRenderPool()) {
                                       pool->Drain();
                                   }
                                   fractal->SaveHiResFractal(Utf8ToWide(filename));
                               });
}

void
LinuxMainWindow::OnSaveItersText()
{
    if (!fractal || !overlay) {
        return;
    }
    overlay->RequestFileDialog("Save iterations as text",
                               LinuxFileDialogMode::Save,
                               MakeTimestampedStem() + ".txt",
                               TextFileFilters(),
                               [this](std::string filename) {
                                   if (!fractal)
                                       return;
                                   filename = AppendExtensionIfMissing(filename, ".txt");
                                   if (auto *pool = fractal->GetRenderPool()) {
                                       pool->Drain();
                                   }
                                   fractal->SaveItersAsText(Utf8ToWide(filename));
                               });
}

void
LinuxMainWindow::OnSaveBmp()
{
    if (!fractal || !overlay) {
        return;
    }
    overlay->RequestFileDialog("Save current image",
                               LinuxFileDialogMode::Save,
                               MakeTimestampedStem() + ".png",
                               PngFileFilters(),
                               [this](std::string filename) {
                                   if (!fractal)
                                       return;
                                   filename = AppendExtensionIfMissing(filename, ".png");
                                   if (auto *pool = fractal->GetRenderPool()) {
                                       pool->Drain();
                                   }
                                   fractal->SaveCurrentFractal(Utf8ToWide(filename), true);
                               });
}

void
LinuxMainWindow::DoSaveRefOrbit(::CompressToDisk compression)
{
    if (!fractal || !overlay) {
        return;
    }
    std::string defaultName = MakeTimestampedStem() + ".im";
    overlay->RequestFileDialog("Save reference orbit",
                               LinuxFileDialogMode::Save,
                               defaultName,
                               OrbitFileFilters(),
                               [this, compression](std::string filename) {
                                   if (!fractal)
                                       return;
                                   std::wstring w = Utf8ToWide(filename);
                                   fractal->EnqueueCommand([compression, w = std::move(w)](Fractal &f) {
                                       f.SaveRefOrbit(compression, w);
                                   });
                               });
}

void
LinuxMainWindow::OnSaveRefOrbitText()
{
    DoSaveRefOrbit(::CompressToDisk::Disable);
}

void
LinuxMainWindow::OnSaveRefOrbitTextSimple()
{
    DoSaveRefOrbit(::CompressToDisk::SimpleCompression);
}

void
LinuxMainWindow::OnSaveRefOrbitTextMax()
{
    DoSaveRefOrbit(::CompressToDisk::MaxCompression);
}

void
LinuxMainWindow::OnSaveRefOrbitImagMax()
{
    DoSaveRefOrbit(::CompressToDisk::MaxCompressionImagina);
}

void
LinuxMainWindow::OnDiffRefOrbitImagMax()
{
    if (!fractal || !overlay) {
        return;
    }
    // Chain: 1) prompt for output filename, 2) pick first input,
    // 3) pick second input, 4) enqueue diff.
    overlay->RequestFileDialog(
        "Output (.im) - diff result",
        LinuxFileDialogMode::Save,
        MakeTimestampedStem() + "_diff.im",
        OrbitFileFilters(),
        [this](std::string outFile) mutable {
            if (!fractal || !overlay)
                return;
            overlay->RequestFileDialog(
                "First input (.im)",
                LinuxFileDialogMode::Open,
                "",
                OrbitFileFilters(),
                [this, outFile = std::move(outFile)](std::string in1) mutable {
                    if (!fractal || !overlay)
                        return;
                    overlay->RequestFileDialog(
                        "Second input (.im)",
                        LinuxFileDialogMode::Open,
                        "",
                        OrbitFileFilters(),
                        [this, outFile = std::move(outFile), in1 = std::move(in1)](std::string in2) {
                            if (!fractal)
                                return;
                            std::wstring outW = Utf8ToWide(outFile);
                            std::wstring f1W = Utf8ToWide(in1);
                            std::wstring f2W = Utf8ToWide(in2);
                            fractal->EnqueueCommand([outW = std::move(outW),
                                                     f1W = std::move(f1W),
                                                     f2W = std::move(f2W)](Fractal &f) {
                                f.DiffRefOrbits(::CompressToDisk::MaxCompressionImagina, outW, f1W, f2W);
                            });
                        });
                });
        });
}

void
LinuxMainWindow::OnLoadLocation()
{
    if (!fractal || !overlay) {
        return;
    }
    // Read locations.txt — same format as the Win32 SavedLocation parser.
    std::ifstream infile("locations.txt");
    if (!infile) {
        overlay->RequestInfoModal("Load Location", "No locations.txt found in current directory.");
        return;
    }
    struct Entry {
        size_t Width = 0, Height = 0;
        HighPrecision MinX, MinY, MaxX, MaxY;
        IterTypeFull NumIterations = 0;
        uint32_t Antialiasing = 0;
        std::string Description;
    };
    std::vector<Entry> entries;
    std::vector<std::string> labels;
    while (infile.good() && entries.size() < 30) {
        Entry e;
        infile >> e.Width >> e.Height;
        infile >> e.MinX >> e.MinY >> e.MaxX >> e.MaxY;
        infile >> e.NumIterations >> e.Antialiasing;
        if (!infile.good()) {
            break;
        }
        infile >> std::ws;
        std::getline(infile, e.Description);
        labels.push_back(e.Description.empty() ? "(unnamed)" : e.Description);
        entries.push_back(std::move(e));
    }
    if (entries.empty()) {
        overlay->RequestInfoModal("Load Location", "locations.txt has no entries.");
        return;
    }
    // The pick-from-list callback only gets the index, so move entries
    // into a shared container.  Use a shared_ptr would normally fit but
    // codebase forbids it — instead, leak-free by capturing entries by
    // value into the lambda (move).
    overlay->RequestPickFromList(
        "Load saved location",
        std::move(labels),
        [this, entries = std::move(entries)](size_t index) mutable {
            if (!fractal || index >= entries.size())
                return;
            const auto &e = entries[index];
            PointZoomBBConverter ptz{
                e.MinX, e.MinY, e.MaxX, e.MaxY, PointZoomBBConverter::TestMode::Enabled};
            auto numIters = e.NumIterations;
            auto antialiasing = e.Antialiasing;
            fractal->EnqueueCommand([ptz = std::move(ptz), numIters, antialiasing](Fractal &f) {
                f.RecenterViewCalc(ptz);
                f.SetNumIterations<IterTypeFull>(numIters);
                f.ResetDimensions(SIZE_MAX, SIZE_MAX, antialiasing);
            });
        });
}

void
LinuxMainWindow::OnLoadEnterLocation()
{
    if (!fractal || !overlay) {
        return;
    }
    // Pull current location to prefill the dialog.
    HighPrecision minX = fractal->GetMinX();
    HighPrecision minY = fractal->GetMinY();
    HighPrecision maxX = fractal->GetMaxX();
    HighPrecision maxY = fractal->GetMaxY();
    PointZoomBBConverter pz{minX, minY, maxX, maxY, PointZoomBBConverter::TestMode::Enabled};
    std::string real = pz.GetPtX().str();
    std::string imag = pz.GetPtY().str();
    std::string zoom = pz.GetZoomFactor().str();
    uint64_t iters = fractal->GetNumIterations<IterTypeFull>();
    overlay->RequestEnterLocation(
        std::move(real),
        std::move(imag),
        std::move(zoom),
        iters,
        [this](std::string r, std::string i, std::string z, uint64_t numIters) {
            if (!fractal)
                return;
            HighPrecision::defaultPrecisionInBits(Fractal::MaxPrecisionLame);
            HighPrecision realHP(r);
            HighPrecision imagHP(i);
            HighPrecision zoomHP(z);
            PointZoomBBConverter pz2{realHP, imagHP, zoomHP, PointZoomBBConverter::TestMode::Enabled};
            fractal->EnqueueCommand([pz2 = std::move(pz2), numIters](Fractal &f) {
                f.RecenterViewCalc(pz2);
                f.SetNumIterations<IterTypeFull>(numIters);
            });
        });
}

void
LinuxMainWindow::DoLoadRefOrbitImag(::ImaginaSettings settings)
{
    if (!fractal || !overlay) {
        return;
    }
    overlay->RequestFileDialog("Load reference orbit (.im)",
                               LinuxFileDialogMode::Open,
                               "",
                               OrbitFileFilters(),
                               [this, settings](std::string filename) {
                                   if (!fractal)
                                       return;
                                   std::wstring w = Utf8ToWide(filename);
                                   fractal->EnqueueCommand([settings, w = std::move(w)](Fractal &f) {
                                       RecommendedSettings rs{};
                                       f.LoadRefOrbit(
                                           &rs, ::CompressToDisk::MaxCompressionImagina, settings, w);
                                       if (rs.GetRenderAlgorithm() == RenderAlgorithmEnum::AUTO) {
                                           (void)f.SetRenderAlgorithm(rs.GetRenderAlgorithm());
                                       }
                                   });
                               });
}

void
LinuxMainWindow::OnLoadRefOrbitImagMax()
{
    DoLoadRefOrbitImag(::ImaginaSettings::ConvertToCurrent);
}

void
LinuxMainWindow::OnLoadRefOrbitImagMaxSaved()
{
    DoLoadRefOrbitImag(::ImaginaSettings::UseSaved);
}

void
LinuxMainWindow::SetDragCursorActive(bool active)
{
    if (!display || !window) {
        return;
    }

    Cursor cursor = active ? dragCursor : idleCursor;
    if (cursor) {
        XDefineCursor(display, window, cursor);
    } else {
        XUndefineCursor(display, window);
    }
    XFlush(display);
}

void
LinuxMainWindow::BeginDragZoom(const XButtonEvent &btn)
{
    dragging = true;
    dragAnchorX = btn.x;
    dragAnchorY = btn.y;
    dragPrevX = -1;
    dragPrevY = -1;

    pointerGrabbed = false;
    if (display && window) {
        const unsigned int eventMask = ButtonPressMask | ButtonReleaseMask | PointerMotionMask;
        const int grabResult = XGrabPointer(display,
                                            window,
                                            False,
                                            eventMask,
                                            GrabModeAsync,
                                            GrabModeAsync,
                                            None,
                                            dragCursor ? dragCursor : None,
                                            btn.time);
        pointerGrabbed = grabResult == GrabSuccess;
    }

    SetDragCursorActive(true);
}

void
LinuxMainWindow::CancelDragZoom(Time eventTime)
{
    if (!dragging && !pointerGrabbed) {
        return;
    }

    if (display && pointerGrabbed) {
        XUngrabPointer(display, eventTime);
        pointerGrabbed = false;
    }

    dragging = false;
    dragPrevX = -1;
    dragPrevY = -1;
    if (overlay) {
        overlay->SetDragRect(false, 0, 0, 0, 0);
    }
    SetDragCursorActive(false);
}

void
LinuxMainWindow::StartWindowMove(const XButtonEvent &btn)
{
    // Initiate a window move via the EWMH _NET_WM_MOVERESIZE protocol —
    // this is the X11 equivalent of posting WM_NCLBUTTONDOWN/HTCAPTION on
    // Win32.  The WM does the actual move + tracking.
    constexpr long kMoveDirection = 8; // _NET_WM_MOVERESIZE_MOVE
    Atom moveResize = XInternAtom(display, "_NET_WM_MOVERESIZE", False);
    if (moveResize == None) {
        return;
    }

    // Release any implicit pointer grab so the WM can take over.
    XUngrabPointer(display, btn.time);

    XEvent ev{};
    ev.xclient.type = ClientMessage;
    ev.xclient.window = window;
    ev.xclient.message_type = moveResize;
    ev.xclient.format = 32;
    ev.xclient.data.l[0] = btn.x_root;
    ev.xclient.data.l[1] = btn.y_root;
    ev.xclient.data.l[2] = kMoveDirection;
    ev.xclient.data.l[3] = Button1;
    ev.xclient.data.l[4] = 1; // source = normal application

    XSendEvent(display,
               RootWindow(display, screen),
               False,
               SubstructureRedirectMask | SubstructureNotifyMask,
               &ev);
    XFlush(display);
}

void
LinuxMainWindow::FinishDragZoom(const XButtonEvent &btn)
{
    CancelDragZoom(btn.time);

    if (btn.state & Mod1Mask) {
        // Alt-released over the window — Win32 path also bails before
        // recentering.  Mirrors MainWindow.cpp:1258.
        return;
    }

    Environment::ScreenRect newView{};
    const bool maintainAspect = (btn.state & ShiftMask) == 0;
    if (maintainAspect) {
        const double ratio = (lastHeight > 0) ? (double)lastWidth / (double)lastHeight : 1.0;
        const DragEndpoint endpoint =
            ComputeAspectLockedDragEndpoint(dragAnchorX, dragAnchorY, btn.x, btn.y, ratio);
        newView.left = dragAnchorX;
        newView.top = dragAnchorY;
        newView.right = endpoint.X;
        newView.bottom = endpoint.Y;
    } else {
        newView.left = dragAnchorX;
        newView.top = dragAnchorY;
        newView.right = btn.x;
        newView.bottom = btn.y;
    }

    if (!fractal) {
        return;
    }
    fractal->EnqueueCommand([newView, maintainAspect](Fractal &f) {
        if (f.RecenterViewScreen(newView)) {
            if (maintainAspect) {
                f.SquareCurrentView();
            }
        }
    });
}

void
LinuxMainWindow::HandleKeyPress(const XKeyEvent &ev)
{
    if (!fractal) {
        return;
    }

    XKeyEvent mutableEvent = ev;
    const KeySym keysym = XLookupKeysym(&mutableEvent, 0);
    const bool shiftDown = (ev.state & ShiftMask) != 0;
    const bool ctrlDown = (ev.state & ControlMask) != 0;

    double panFrac = 0.25;
    if (shiftDown) {
        panFrac = 0.10;
    } else if (ctrlDown) {
        panFrac = 0.50;
    }

    switch (keysym) {
        case XK_Left:
            fractal->EnqueueCommand([panFrac](Fractal &f) { f.PanByFraction(-panFrac, 0.0); });
            return;
        case XK_Right:
            fractal->EnqueueCommand([panFrac](Fractal &f) { f.PanByFraction(panFrac, 0.0); });
            return;
        case XK_Up:
            fractal->EnqueueCommand([panFrac](Fractal &f) { f.PanByFraction(0.0, panFrac); });
            return;
        case XK_Down:
            fractal->EnqueueCommand([panFrac](Fractal &f) { f.PanByFraction(0.0, -panFrac); });
            return;
        case XK_KP_Add:
            fractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(-0.3); });
            return;
        case XK_KP_Subtract:
            fractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); });
            return;
        default:
            break;
    }

    const std::optional<FractalShark::HotKey> hotkey = HotKeyFromXKeyEvent(ev);
    if (!hotkey) {
        return;
    }

    contextMenuX = ev.x;
    contextMenuY = ev.y;
    if (const FractalShark::Command *command = FractalShark::FindCommandByHotKey(*hotkey)) {
        FractalShark::ExecuteCommand(command->id, *this);
    }
}

} // namespace

int
main(int /*argc*/, char ** /*argv*/)
{
    Environment::RegisterHeapCleanup();
    CrashHandler::Install();

    // Xlib functions are touched from both the GUI thread (XNextEvent
    // pump) and the GL consumer thread (ImGui overlay callback queries
    // window attributes).  XInitThreads must be called before any other
    // Xlib call, otherwise concurrent access is undefined.
    XInitThreads();

    LinuxMainWindow win;
    if (!win.Valid()) {
        return 1;
    }
    win.RunEventLoop();
    return 0;
}
