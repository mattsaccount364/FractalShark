// FractalSharkGuiLinux — Xlib window skeleton.
//
// Phase: fractal-render.  This stage brings up an empty X11 window with a
// GLX-compatible visual, instantiates a Fractal bound to that window, and
// kicks off a single default-view render.  The render thread pool's GL
// consumer presents via glXSwapBuffers.  No keyboard / mouse input yet —
// that is the next phase (input-keys / input-mouse / drag-zoom).

#include "CommandCatalog.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "FeatureFinderMode.h"
#include "Fractal.h"
#include "LinuxClipboard.h"
#include "LinuxImGuiOverlay.h"
#include "OpenGLContext.h"
#include "RefOrbitCalc.h"
#include "RenderThreadPool.h"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <GL/glx.h>

#include <poll.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

// Mechanical-contract stub: every catalog command hook the Win32 GUI
// implements in MainWindow must also be overridden here.  Until the Linux
// GUI grows real menu/render plumbing each hook just announces itself so a
// developer wiring up a UI surface can see which commands fire.  A missing
// hook produces a compile error (pure virtual not implemented), which is
// exactly the signal we want from Phase 0c.
#define FRACTALSHARK_LINUX_STUB(method)                                                               \
    void method() override                                                                            \
    {                                                                                                 \
        std::fprintf(stderr, "TODO LinuxMainWindow: %s\n", #method);                                  \
    }

struct LinuxMainWindow : FractalShark::ExecuteCommandHost {
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

    bool Valid() const noexcept { return display != nullptr && window != 0; }
    void RunEventLoop();
    void HandleEvent(const XEvent &ev);
    void HandleKeyPress(const XKeyEvent &ev);
    void StartWindowMove(const XButtonEvent &btn);
    void FinishDragZoom(const XButtonEvent &btn);
    void CopyRenderDetailsToClipboard();

    // ---- ExecuteCommandHost stubs (one per virtual on the host) -----------
    void DispatchByIdm(int wmId) override
    {
        std::fprintf(stderr, "TODO LinuxMainWindow: DispatchByIdm(%d)\n", wmId);
    }

    void OnSetAlgorithm(::RenderAlgorithmEnum alg) override
    {
        std::fprintf(stderr,
                     "TODO LinuxMainWindow: OnSetAlgorithm(%u)\n",
                     static_cast<unsigned>(alg));
    }

    void OnSelectBuiltInView(size_t oneBasedIndex) override
    {
        std::fprintf(stderr,
                     "TODO LinuxMainWindow: OnSelectBuiltInView(%zu)\n",
                     oneBasedIndex);
    }

    FRACTALSHARK_LINUX_STUB(OnShowHotkeys)
    FRACTALSHARK_LINUX_STUB(OnViewsHelp)
    FRACTALSHARK_LINUX_STUB(OnHelpAlg)
    FRACTALSHARK_LINUX_STUB(OnSquareView)
    FRACTALSHARK_LINUX_STUB(OnRepainting)
    FRACTALSHARK_LINUX_STUB(OnWindowed)
    FRACTALSHARK_LINUX_STUB(OnWindowedSq)
    FRACTALSHARK_LINUX_STUB(OnMinimize)
    FRACTALSHARK_LINUX_STUB(OnCurPos)
    FRACTALSHARK_LINUX_STUB(OnExit)

    FRACTALSHARK_LINUX_STUB(OnBack)
    FRACTALSHARK_LINUX_STUB(OnCenterView)
    FRACTALSHARK_LINUX_STUB(OnZoomIn)
    FRACTALSHARK_LINUX_STUB(OnZoomOut)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomDefault)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomMax)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomFilament)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderDirect)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderDirectScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderPt)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderPtScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderLa)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderLaScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderZoom)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderClear)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderResume)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopGpu)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopCpu)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopCpuSt)

    FRACTALSHARK_LINUX_STUB(OnStandardView)

    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing1x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing4x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing9x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing16x)

    FRACTALSHARK_LINUX_STUB(OnResetIterations)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations1p5x)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations6x)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations24x)
    FRACTALSHARK_LINUX_STUB(OnDecreaseIterations)
    FRACTALSHARK_LINUX_STUB(OnIterations32Bit)
    FRACTALSHARK_LINUX_STUB(OnIterations64Bit)

    FRACTALSHARK_LINUX_STUB(OnIterationPrecision1x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision2x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision3x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision4x)

    FRACTALSHARK_LINUX_STUB(OnPerturbResults)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearAll)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearMed)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearHigh)
    FRACTALSHARK_LINUX_STUB(OnPerturbationAuto)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSinglethread)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSinglethreadPeriodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread2Periodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighStmed)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed1)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed2)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed3)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed4)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread5Periodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationGpu)
    FRACTALSHARK_LINUX_STUB(OnPerturbationLoad)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSave)

    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOnDelete)
    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOn)
    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOff)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit0)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit1)

    FRACTALSHARK_LINUX_STUB(OnPaletteType0)
    FRACTALSHARK_LINUX_STUB(OnPaletteType1)
    FRACTALSHARK_LINUX_STUB(OnPaletteType2)
    FRACTALSHARK_LINUX_STUB(OnPaletteType3)
    FRACTALSHARK_LINUX_STUB(OnPaletteType4)
    FRACTALSHARK_LINUX_STUB(OnCreateNewPalette)
    FRACTALSHARK_LINUX_STUB(OnPalette5)
    FRACTALSHARK_LINUX_STUB(OnPalette6)
    FRACTALSHARK_LINUX_STUB(OnPalette8)
    FRACTALSHARK_LINUX_STUB(OnPalette12)
    FRACTALSHARK_LINUX_STUB(OnPalette16)
    FRACTALSHARK_LINUX_STUB(OnPalette20)
    FRACTALSHARK_LINUX_STUB(OnPaletteRotate)

    FRACTALSHARK_LINUX_STUB(OnSaveLocation)
    FRACTALSHARK_LINUX_STUB(OnSaveHiResBmp)
    FRACTALSHARK_LINUX_STUB(OnSaveItersText)
    FRACTALSHARK_LINUX_STUB(OnSaveBmp)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitText)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitTextSimple)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitTextMax)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnDiffRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnLoadLocation)
    FRACTALSHARK_LINUX_STUB(OnLoadEnterLocation)
    FRACTALSHARK_LINUX_STUB(OnLoadRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnLoadRefOrbitImagMaxSaved)

    FRACTALSHARK_LINUX_STUB(OnBasicTest)
    FRACTALSHARK_LINUX_STUB(OnTest27)
    FRACTALSHARK_LINUX_STUB(OnBenchmarkFull)
    FRACTALSHARK_LINUX_STUB(OnBenchmarkInt)

    FRACTALSHARK_LINUX_STUB(OnLaMultithreaded)
    FRACTALSHARK_LINUX_STUB(OnLaSinglethreaded)
    FRACTALSHARK_LINUX_STUB(OnLaSettings1)
    FRACTALSHARK_LINUX_STUB(OnLaSettings2)
    FRACTALSHARK_LINUX_STUB(OnLaSettings3)
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

    // Pick a GLX-compatible visual at window creation time so the GLX context
    // OpenGlContext later creates can be made current on this window.  The
    // attribute list mirrors OpenGLContext.cpp's glXChooseVisual call to
    // guarantee a match.
    int glxAttribs[] = {GLX_RGBA,
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
    visualInfo = glXChooseVisual(display, screen, glxAttribs);
    if (!visualInfo) {
        std::fprintf(stderr,
                     "FractalSharkGuiLinux: glXChooseVisual failed (no RGBA double-buffered visual)\n");
        XCloseDisplay(display);
        display = nullptr;
        return;
    }

    Window root = RootWindow(display, screen);

    colormap = XCreateColormap(display, root, visualInfo->visual, AllocNone);

    XSetWindowAttributes swa{};
    swa.colormap = colormap;
    swa.background_pixel = BlackPixel(display, screen);
    swa.border_pixel = BlackPixel(display, screen);
    swa.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask
                     | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask
                     | FocusChangeMask;

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
    glContext = std::make_unique<OpenGlContext>(reinterpret_cast<void *>(static_cast<uintptr_t>(window)));
    if (!glContext->IsValid()) {
        std::fprintf(stderr,
                     "FractalSharkGuiLinux: OpenGlContext creation failed; running headless.\n");
    }

    // ImGui overlay: single-threaded, all calls on this thread.  Init
    // requires GL to be current, which OpenGlContext::ctor already
    // ensured.
    overlay.emplace(display, window, clipboard ? &*clipboard : nullptr);
    overlay->SetExecuteHost(this);
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
        if (cm.message_type == wmProtocols
            && static_cast<Atom>(cm.data.l[0]) == wmDeleteWindow) {
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
            fractal->EnqueueCommand(
                [w, h](Fractal &f) { f.ResetDimensions(static_cast<size_t>(w), static_cast<size_t>(h)); });
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
            fractal->EnqueueCommand(
                [x, y](Fractal &f) { f.ZoomTowardPoint(x, y, -0.3); });
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
            const double ratio =
                (lastHeight > 0) ? (double)lastWidth / (double)lastHeight : 1.0;
            dragPrevY = mot.y;
            dragPrevX =
                static_cast<int>((double)dragAnchorX + ratio * ((double)dragPrevY - (double)dragAnchorY));
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
            // If this tick was triggered by overlay motion only, re-blit
            // the cached frame so the overlay has something underneath.
            if (!freshFrame && fractal) {
                if (auto *pool = fractal->GetRenderPool()) {
                    pool->RepresentLastFrame(*glContext);
                }
            }
            if (overlay) {
                overlay->RenderFrame();
            }
            glContext->SwapBuffers();
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
LinuxMainWindow::StartWindowMove(const XButtonEvent &btn)
{
    // Initiate a window move via the EWMH _NET_WM_MOVERESIZE protocol —
    // this is the X11 equivalent of posting WM_NCLBUTTONDOWN/HTCAPTION on
    // Win32.  The WM does the actual move + tracking.
    constexpr long kMoveDirection = 8;  // _NET_WM_MOVERESIZE_MOVE
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
    ev.xclient.data.l[4] = 1;  // source = normal application

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
        newView.right =
            static_cast<int32_t>((double)newView.left + ratio * ((double)newView.bottom - (double)newView.top));
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
    if (n <= 0) {
        return;
    }

    const bool shiftDown = (ev.state & ShiftMask) != 0;
    const int mouseX = ev.x;
    const int mouseY = ev.y;
    const char ch = buf[0];

    switch (ch) {
    case 'A':
    case 'a':
        if (!shiftDown) {
            fractal->AutoZoom<Fractal::AutoZoomHeuristic::Feature>();
        } else {
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) { f.CenterAtPoint(mouseX, mouseY); });
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
            fractal->EnqueueCommand([mouseX, mouseY](Fractal &f) { f.CenterAtPoint(mouseX, mouseY); });
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
