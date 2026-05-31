// FractalSharkGuiLinux — Xlib window skeleton.
//
// Phase: fractal-render.  This stage brings up an empty X11 window with a
// GLX-compatible visual, instantiates a Fractal bound to that window, and
// kicks off a single default-view render.  The render thread pool's GL
// consumer presents via glXSwapBuffers or glFlush.  No keyboard / mouse input yet —
// that is the next phase (input-keys / input-mouse / drag-zoom).

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

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <GL/glx.h>

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

struct GlxVisualSelection {
    XVisualInfo *VisualInfo;
    const char *Description;
};

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
        return {visualInfo, "RGBA double-buffered with 8-bit alpha"};
    }

    int rgbaDoubleBuffered[] = {
        GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaDoubleBuffered)) {
        return {visualInfo, "RGBA double-buffered without required alpha"};
    }

    int rgbaSingleBuffered[] = {GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaSingleBuffered)) {
        return {visualInfo, "RGBA single-buffered without required alpha"};
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

    // Set on right-click; consumed when the imgui-menus phase opens the
    // ImGui popup.  Position is window-relative, matching MainWindow.cpp's
    // m_LastMenuPtClient anchor.
    bool showContextMenu = false;
    int contextMenuX = 0;
    int contextMenuY = 0;

    // Left-button drag-zoom state.  Mirrors MainWindow's lButtonDown +
    // dragBoxX1/Y1 (anchor) + prevX1/Y1 (last cursor for outline rect).
    // Outline rendering itself is deferred to imgui-menus phase (ImGui
    // foreground draw list); for now we only capture the box and submit
    // RecenterViewScreen on release.
    bool dragging = false;
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

    LinuxMainWindow();
    ~LinuxMainWindow() override;

    LinuxMainWindow(const LinuxMainWindow &) = delete;
    LinuxMainWindow &operator=(const LinuxMainWindow &) = delete;

    bool
    Valid() const noexcept
    {
        return display != nullptr && window != 0;
    }
    void RunEventLoop();
    void HandleEvent(const XEvent &ev);
    void HandleKeyPress(const XKeyEvent &ev);
    void StartWindowMove(const XButtonEvent &btn);
    void FinishDragZoom(const XButtonEvent &btn);
    void CopyRenderDetailsToClipboard();
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
    // Everything else is provided by LinuxCommandHandlers.  These ~25 hooks
    // touch the X server, ImGui modals, the file dialog, the clipboard, or
    // process-wide JobObject state — so they need real Linux implementations
    // (forthcoming Phase 3.3.2 / 3.3.4 / 3.3.6 / 3.3.7) and stay as stubs
    // here until then.
    void
    DispatchByIdm(int wmId) override
    {
        std::fprintf(stderr, "TODO LinuxMainWindow: DispatchByIdm(%d)\n", wmId);
    }

    void OnShowHotkeys() override;
    void OnViewsHelp() override;
    void OnHelpAlg() override;
    void OnWindowed() override;
    void OnWindowedSq() override;
    void OnMinimize() override;
    void OnCurPos() override;
    void OnExit() override;

    FRACTALSHARK_LINUX_STUB(OnMemoryLimit0)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit1)
    FRACTALSHARK_LINUX_STUB(OnPaletteRotate)

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
    std::fprintf(stderr, "FractalSharkGuiLinux: selected GLX visual: %s\n", visualSelection.Description);

    Window root = RootWindow(display, screen);

    colormap = XCreateColormap(display, root, visualInfo->visual, AllocNone);

    XSetWindowAttributes swa{};
    swa.colormap = colormap;
    swa.background_pixel = BlackPixel(display, screen);
    swa.border_pixel = BlackPixel(display, screen);
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                     PointerMotionMask | StructureNotifyMask | FocusChangeMask;

    window = XCreateWindow(display,
                           root,
                           0,
                           0,
                           kInitialWidth,
                           kInitialHeight,
                           0,
                           visualInfo->depth,
                           InputOutput,
                           visualInfo->visual,
                           CWColormap | CWBackPixel | CWBorderPixel | CWEventMask,
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
    overlay.emplace(display, window, clipboard ? &*clipboard : nullptr);
    overlay->SetExecuteHost(this);
    menuState.emplace(*fractal, fullscreen);
    overlay->SetMenuState(&*menuState);
    if (glContext && glContext->IsValid()) {
        if (!overlay->Init()) {
            std::fprintf(stderr, "FractalSharkGuiLinux: ImGui backend init failed.\n");
        }
    }
}

LinuxMainWindow::~LinuxMainWindow()
{
    // Destroy in reverse order of creation: overlay first (it owns ImGui
    // backends that need GL current), then GL context, then Fractal.
    overlay.reset();
    menuState.reset();
    glContext.reset();
    fractal.reset();

    if (display) {
        if (window) {
            XDestroyWindow(display, window);
            window = 0;
        }
        if (colormap) {
            XFreeColormap(display, colormap);
            colormap = 0;
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
                    dragging = true;
                    dragAnchorX = btn.x;
                    dragAnchorY = btn.y;
                    dragPrevX = -1;
                    dragPrevY = -1;
                    // Pointer motion + release is already in our event mask, and
                    // X11 implicitly delivers all subsequent button-related events
                    // to the window that received the press until release, so an
                    // explicit XGrabPointer is unnecessary.
                    break;
                }
                case Button3:
                    // Right-click anchors the context menu.  Window-relative coords
                    // mirror MainWindow's m_LastMenuPtClient anchor (Win32 receives
                    // them already client-relative; X11 hands them to us already
                    // window-relative in btn.x/btn.y).
                    showContextMenu = true;
                    contextMenuX = btn.x;
                    contextMenuY = btn.y;
                    if (overlay) {
                        overlay->RequestContextMenu(btn.x, btn.y);
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
                dragPrevY = mot.y;
                dragPrevX = static_cast<int>((double)dragAnchorX +
                                             ratio * ((double)dragPrevY - (double)dragAnchorY));
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
            dragging = false;
            dragPrevX = -1;
            dragPrevY = -1;
            if (overlay) {
                overlay->SetDragRect(false, 0, 0, 0, 0);
            }
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
    bool everPresented = false;

    while (running) {
        // Drain any X events already queued on the connection without
        // blocking — they may have been deposited while the previous
        // iteration was rendering.
        while (XPending(display) > 0) {
            XNextEvent(display, &ev);
            if (clipboard && clipboard->ProcessEvent(ev)) {
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

        // Pull any frames the render pool has finished and present them.
        // TryPresentTick drains all ready frames in order, releasing
        // superseded buffers, and returns true if a non-tombstone frame
        // was uploaded.
        bool freshFrame = false;
        if (fractal && glContext && glContext->IsValid()) {
            auto *pool = fractal->GetRenderPool();
            if (pool) {
                freshFrame = pool->TryPresentTick(*glContext);
            }
        }

        const bool overlayWantsTick = overlay && overlay->WantsTick();
        const bool needsTick = freshFrame || overlayWantsTick || dragging || !everPresented;

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
        }

        // Sleep until either an X event arrives or our 16ms tick budget
        // expires.  No need for an explicit wake from the pool — the next
        // tick will pick the frame up.
        struct pollfd pfd{};
        pfd.fd = xfd;
        pfd.events = POLLIN;
        const int timeoutMs = needsTick ? 16 : 33;
        poll(&pfd, 1, timeoutMs);
    }
}

void
LinuxMainWindow::CopyRenderDetailsToClipboard()
{
    if (!fractal || !clipboard) {
        return;
    }
    std::string shortStr;
    std::string longStr;
    fractal->GetRenderDetails(shortStr, longStr);
    // Mirror Win32 MenuGetCurPos: copy the long form (includes coordinates,
    // zoom, iterations, algorithm).  Win32 concatenates short + long; the
    // long form already contains everything useful so copy that.
    clipboard->Set(longStr);
}

void
LinuxMainWindow::OnShowHotkeys()
{
    if (overlay) {
        overlay->RequestInfoModal(FractalSharkLinux::kHotkeysModalTitle,
                                  FractalSharkLinux::kHotkeysModalBody);
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
    if (!fractal) {
        return;
    }
    std::string shortStr;
    std::string longStr;
    fractal->GetRenderDetails(shortStr, longStr);
    if (clipboard) {
        clipboard->Set(longStr);
    }
    if (shortStr.size() < 5000 && overlay) {
        overlay->RequestInfoModal("Current Position", shortStr.c_str());
    } else {
        std::fprintf(stderr, "Location copied to clipboard.\n");
    }
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

// Naive ASCII widening — adequate for filenames we generate ourselves and
// for typical user-typed paths in the dialog.  Matches FractalSharkCli's
// ToWStringUtf8 helper.
std::wstring
WidenAscii(const std::string &s)
{
    std::wstring w;
    w.reserve(s.size());
    for (unsigned char c : s) {
        w.push_back(static_cast<wchar_t>(c));
    }
    return w;
}

// List files in cwd matching the given extension (e.g. ".im").  Returns
// names sorted alphabetically.
std::vector<std::string>
ListFilesByExtension(const char *ext)
{
    std::vector<std::string> out;
    std::error_code ec;
    for (auto &entry : std::filesystem::directory_iterator(".", ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const auto path = entry.path();
        if (path.extension() == ext) {
            out.push_back(path.filename().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

void
LinuxMainWindow::OnSaveLocation()
{
    if (!fractal) {
        return;
    }
    // Mirror the Win32 NO branch (use current dimensions).  Skipping the
    // "scale to maximum" prompt for simplicity; users wanting the high-
    // res variant can use Save Hi-Res BMP instead.
    std::string filename = MakeTimestampedStem() + ".bmp";
    if (auto *pool = fractal->GetRenderPool()) {
        pool->Drain();
    }
    fractal->SaveCurrentFractal(WidenAscii(filename), true);
}

void
LinuxMainWindow::OnSaveHiResBmp()
{
    if (!fractal) {
        return;
    }
    if (auto *pool = fractal->GetRenderPool()) {
        pool->Drain();
    }
    fractal->SaveHiResFractal(L"");
}

void
LinuxMainWindow::OnSaveItersText()
{
    if (!fractal) {
        return;
    }
    if (auto *pool = fractal->GetRenderPool()) {
        pool->Drain();
    }
    fractal->SaveItersAsText(L"");
}

void
LinuxMainWindow::OnSaveBmp()
{
    if (!fractal) {
        return;
    }
    if (auto *pool = fractal->GetRenderPool()) {
        pool->Drain();
    }
    fractal->SaveCurrentFractal(L"", true);
}

void
LinuxMainWindow::DoSaveRefOrbit(::CompressToDisk compression)
{
    if (!fractal || !overlay) {
        return;
    }
    std::string defaultName = MakeTimestampedStem() + ".im";
    overlay->RequestSaveDialog(
        "Save reference orbit", defaultName, [this, compression](std::string filename) {
            if (!fractal)
                return;
            std::wstring w = WidenAscii(filename);
            fractal->EnqueueCommand(
                [compression, w = std::move(w)](Fractal &f) { f.SaveRefOrbit(compression, w); });
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
    auto files = ListFilesByExtension(".im");
    if (files.size() < 2) {
        overlay->RequestInfoModal("Diff Reference Orbit",
                                  "Need at least two *.im files in the current directory.");
        return;
    }
    // Chain: 1) prompt for output filename, 2) pick first input,
    // 3) pick second input, 4) enqueue diff.
    overlay->RequestSaveDialog(
        "Output (.im) — diff result",
        MakeTimestampedStem() + "_diff.im",
        [this, files](std::string outFile) mutable {
            if (!fractal || !overlay)
                return;
            overlay->RequestPickFromList(
                "First input (.im)",
                files,
                [this, files, outFile = std::move(outFile)](size_t idx1) mutable {
                    if (!fractal || !overlay || idx1 >= files.size())
                        return;
                    std::string in1 = files[idx1];
                    overlay->RequestPickFromList(
                        "Second input (.im)",
                        files,
                        [this, outFile = std::move(outFile), in1 = std::move(in1), files](size_t idx2) {
                            if (!fractal || idx2 >= files.size())
                                return;
                            std::wstring outW = WidenAscii(outFile);
                            std::wstring f1W = WidenAscii(in1);
                            std::wstring f2W = WidenAscii(files[idx2]);
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
    auto files = ListFilesByExtension(".im");
    if (files.empty()) {
        overlay->RequestInfoModal("Load Reference Orbit", "No *.im files in current directory.");
        return;
    }
    overlay->RequestPickFromList(
        "Load reference orbit (.im)", files, [this, files, settings](size_t index) {
            if (!fractal || index >= files.size())
                return;
            std::wstring w = WidenAscii(files[index]);
            fractal->EnqueueCommand([settings, w = std::move(w)](Fractal &f) {
                RecommendedSettings rs{};
                f.LoadRefOrbit(&rs, ::CompressToDisk::MaxCompressionImagina, settings, w);
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
    dragging = false;
    dragPrevX = -1;
    dragPrevY = -1;
    if (overlay) {
        overlay->SetDragRect(false, 0, 0, 0, 0);
    }

    if (btn.state & Mod1Mask) {
        // Alt-released over the window — Win32 path also bails before
        // recentering.  Mirrors MainWindow.cpp:1258.
        return;
    }

    Environment::ScreenRect newView{};
    const bool maintainAspect = (btn.state & ShiftMask) == 0;
    if (maintainAspect) {
        const double ratio = (lastHeight > 0) ? (double)lastWidth / (double)lastHeight : 1.0;
        newView.left = dragAnchorX;
        newView.top = dragAnchorY;
        newView.bottom = btn.y;
        newView.right = static_cast<int32_t>((double)newView.left +
                                             ratio * ((double)newView.bottom - (double)newView.top));
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

    // Convert KeyPress → ASCII via XLookupString (already shift-aware:
    // 'A' shifted yields 'A', unshifted yields 'a').  Win32 WM_CHAR has
    // identical semantics; mirroring lets us reuse case literals verbatim.
    char buf[4]{};
    KeySym keysym = NoSymbol;
    int n = XLookupString(const_cast<XKeyEvent *>(&ev), buf, sizeof(buf), &keysym, nullptr);

    // Handle arrow keys and numpad +/- (these don't produce ASCII).
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

    if (n <= 0) {
        return;
    }

    const int mouseX = ev.x;
    const int mouseY = ev.y;
    const char ch = buf[0];

    switch (ch) {
        case 'A':
        case 'a':
            if (!shiftDown) {
                fractal->AutoZoom<Fractal::AutoZoomHeuristic::Feature>();
            } else {
                fractal->EnqueueCommand(
                    [mouseX, mouseY](Fractal &f) { f.CenterAtPoint(mouseX, mouseY); });
                fractal->AutoZoom<Fractal::AutoZoomHeuristic::Default>();
            }
            break;

        case 'S':
            fractal->AutoZoom<Fractal::AutoZoomHeuristic::FilamentTip>();
            break;

        case 'b':
            fractal->EnqueueCommand([](Fractal &f) { f.Back(); });
            break;

        case 'C':
        case 'c':
            if (shiftDown) {
                fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                    f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                    f.CenterAtPoint(mouseX, mouseY);
                });
            } else {
                fractal->EnqueueCommand(
                    [mouseX, mouseY](Fractal &f) { f.CenterAtPoint(mouseX, mouseY); });
            }
            break;

        case 'E':
        case 'e':
            fractal->EnqueueCommand([](Fractal &f) {
                f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                f.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
                f.DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
            });
            break;

        case 'n':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::Direct);
            });
            break;
        case 'N':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::DirectScan);
            });
            break;
        case 'm':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::PT);
            });
            break;
        case 'M':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::PTScan);
            });
            break;
        case ',':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::LA);
            });
            break;
        case '<':
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) {
                f.TryFindPeriodicPoint(mouseX, mouseY, FeatureFinderMode::LAScan);
            });
            break;
        case '.':
            fractal->EnqueueCommand([](Fractal &f) { f.ZoomToFoundFeature(); });
            break;
        case '>':
            fractal->EnqueueCommand([](Fractal &f) { f.ClearAllFoundFeatures(); });
            break;

        case 'H':
        case 'h':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                auto &laParameters = f.GetLAParameters();
                if (shiftDown) {
                    laParameters.AdjustLAThresholdScaleExponent(-1);
                    laParameters.AdjustLAThresholdCScaleExponent(-1);
                } else {
                    laParameters.AdjustLAThresholdScaleExponent(1);
                    laParameters.AdjustLAThresholdCScaleExponent(1);
                }
                f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
                f.ForceRecalc();
            });
            break;

        case 'J':
        case 'j':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                auto &laParameters = f.GetLAParameters();
                if (shiftDown) {
                    laParameters.AdjustPeriodDetectionThreshold2Exponent(-1);
                    laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(-1);
                } else {
                    laParameters.AdjustPeriodDetectionThreshold2Exponent(1);
                    laParameters.AdjustStage0PeriodDetectionThreshold2Exponent(1);
                }
                f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
                f.ForceRecalc();
            });
            break;

        case 'I':
        case 'i':
            fractal
                ->EnqueueCommand([shiftDown](Fractal &f) {
                    if (shiftDown) {
                        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::MediumRes);
                    }
                    f.ForceRecalc();
                })
                .Wait();
            CopyRenderDetailsToClipboard();
            break;

        case 'O':
        case 'o':
            fractal
                ->EnqueueCommand([shiftDown](Fractal &f) {
                    if (shiftDown) {
                        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                    }
                    f.ForceRecalc();
                })
                .Wait();
            CopyRenderDetailsToClipboard();
            break;

        case 'P':
        case 'p':
            fractal
                ->EnqueueCommand([shiftDown](Fractal &f) {
                    if (shiftDown) {
                        f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::LAOnly);
                    }
                    f.ForceRecalc();
                })
                .Wait();
            CopyRenderDetailsToClipboard();
            break;

        case 'q':
        case 'Q':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                if (shiftDown) {
                    f.DecCompressionError(Fractal::CompressionError::Intermediate, 10);
                } else {
                    f.IncCompressionError(Fractal::CompressionError::Intermediate, 10);
                }
            });
            break;

        case 'R':
        case 'r':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                if (shiftDown) {
                    f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                }
                f.SquareCurrentView();
            });
            break;

        case 'T':
        case 't':
            fractal->EnqueueCommand(
                [shiftDown](Fractal &f) { f.UseNextPaletteAuxDepth(shiftDown ? -1 : 1); });
            break;

        case 'W':
        case 'w':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                f.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                if (shiftDown) {
                    f.DecCompressionError(Fractal::CompressionError::Low, 1);
                } else {
                    f.IncCompressionError(Fractal::CompressionError::Low, 1);
                }
            });
            break;

        case 'Z':
        case 'z':
            if (shiftDown) {
                fractal->EnqueueCommand(
                    [mouseX, mouseY](Fractal &f) { f.ZoomRecentered(mouseX, mouseY, 1); });
            } else {
                fractal->EnqueueCommand(
                    [mouseX, mouseY](Fractal &f) { f.ZoomRecentered(mouseX, mouseY, -.45); });
            }
            break;

        case 'D':
        case 'd':
            fractal->EnqueueCommand([shiftDown](Fractal &f) {
                if (shiftDown) {
                    f.CreateNewFractalPalette();
                    f.UsePaletteType(FractalPaletteType::Random);
                } else {
                    f.UseNextPaletteDepth();
                }
            });
            break;

        case '=':
        case '+':
            fractal->EnqueueCommand([](Fractal &f) {
                const double factor = 24.0;
                if (f.GetIterType() == IterTypeEnum::Bits32) {
                    uint64_t cur = f.GetNumIterations<uint32_t>();
                    cur = static_cast<uint64_t>(static_cast<double>(cur) * factor);
                    f.SetNumIterations<uint32_t>(cur);
                } else {
                    uint64_t cur = f.GetNumIterations<uint64_t>();
                    cur = static_cast<uint64_t>(static_cast<double>(cur) * factor);
                    f.SetNumIterations<uint64_t>(cur);
                }
            });
            break;

        case '-':
            fractal->EnqueueCommand([](Fractal &f) {
                const double factor = 2.0 / 3.0;
                if (f.GetIterType() == IterTypeEnum::Bits32) {
                    uint64_t cur = f.GetNumIterations<uint32_t>();
                    cur = static_cast<uint64_t>(static_cast<double>(cur) * factor);
                    f.SetNumIterations<uint32_t>(cur);
                } else {
                    uint64_t cur = f.GetNumIterations<uint64_t>();
                    cur = static_cast<uint64_t>(static_cast<double>(cur) * factor);
                    f.SetNumIterations<uint64_t>(cur);
                }
            });
            break;

        default:
            break;
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
