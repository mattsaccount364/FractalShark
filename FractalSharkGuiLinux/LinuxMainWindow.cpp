// LinuxMainWindow - Xlib + Dear ImGui GUI shell.

#include "LinuxMainWindow.h"
#include "CommandCatalog.h"
#include "Environment.h"
#include "Exceptions.h"
#include "FeatureFinderMode.h"
#include "Fractal.h"
#include "GuiFileOperations.h"
#include "GuiHelp.h"
#include "LinuxClipboard.h"
#include "LinuxImGuiOverlay.h"
#include "MenuState.h"
#include "PortableCommandHandlers.h"
// MenuState.h pulls in MenuTree.h which #undefs the X11 `None` and
// `Always` macros (they collide with FractalShark enum values).
// X.h has a header guard so it can't be re-included to restore them — just
// redefine the two we use locally.
#ifndef None
#define None 0L
#endif
#ifndef Always
#define Always 2
#endif
#include "OpenGLContext.h"
#include "RenderThreadPool.h"
#include "SavedLocation.h"
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
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

void
RequireUiState(bool available, const char *operation)
{
    if (!available) {
        throw FractalSharkSeriousException(std::string(operation) +
                                           " requires initialized Linux UI state");
    }
}

bool
SetEnvironmentClipboardText(void *context, std::string_view text)
{
    auto *clipboard = static_cast<FractalShark::Linux::LinuxClipboard *>(context);
    if (clipboard == nullptr) {
        return false;
    }
    clipboard->Set(std::string(text));
    return true;
}

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

    int rgbaWithAlpha[] = {
        GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgbaWithAlpha)) {
        return {visualInfo};
    }

    int rgba[] = {GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
    if (XVisualInfo *visualInfo = glXChooseVisual(display, screen, rgba)) {
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

struct LinuxMainWindow : FractalShark::PortableCommandHandlers {
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
    bool waitCursorRegistered = false;
    bool clipboardTextSetterRegistered = false;

    // Right-click position in client coordinates, matching MainWindow.cpp's
    // m_LastMenuPtClient anchor for menu commands that act on the click point.
    int contextMenuX = 0;
    int contextMenuY = 0;
    bool suppressContextMenuButtonRelease = false;

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

    std::optional<FractalShark::Linux::LinuxClipboard> clipboard;
    std::unique_ptr<Fractal> fractal;
    std::optional<FractalShark::MenuState> menuState;
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

    void RunEventLoop();
    void HandleEvent(const XEvent &ev);
    void HandleKeyPress(const XKeyEvent &ev);
    PresentationTickResult PresentRenderTick();
    int GetPresentationPollTimeoutMs(bool needsTick);
    void RunFeatureAutoZoomSynchronously(int mouseX, int mouseY);
    void BeginDragZoom(const XButtonEvent &btn);
    void CancelDragZoom(Time eventTime);
    void SetDragCursorActive(bool active);
    void StartWindowMove(const XButtonEvent &btn);
    void FinishDragZoom(const XButtonEvent &btn);
    void EnterFullscreen(bool square);
    void ExitFullscreen();

    // ---- PortableCommandHandlers protected accessors ----------------------
    Fractal &
    GetFractal() noexcept override
    {
        return *fractal;
    }
    FractalShark::MenuPoint
    GetMenuMousePos() const override
    {
        return {contextMenuX, contextMenuY};
    }

    // ---- Platform-specific command handlers ------------------------------
    void OnAutoZoomFeatureAtPoint() override;
    // Everything else is provided by PortableCommandHandlers. These hooks
    // touch the X server, ImGui modals, the file dialog, the clipboard, or
    // other Linux-specific UI state.
    void OnShowHotkeys() override;
    void OnViewsHelp() override;
    void OnHelpAlg() override;
    void OnWindowed() override;
    void OnWindowedSq() override;
    void OnMinimize() override;
    void OnCurPos() override;
    void OnExit() override;

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
    using FileDialogCallback = FractalShark::Linux::ImGuiOverlay::FileDialogCallback;
    using FileDialogFilter = FractalShark::Linux::ImGuiOverlay::FileDialogFilter;
    using FileDialogMode = FractalShark::Linux::ImGuiOverlay::FileDialogMode;
    using PickFromListCallback = FractalShark::Linux::ImGuiOverlay::PickFromListCallback;

    void Destroy() noexcept;
    void ShowInfo(const char *title, const char *body);
    void ShowInfo(const char *title, const std::string &body);
    void RequestFileDialog(const char *operation,
                           const char *title,
                           FileDialogMode mode,
                           const std::string &defaultName,
                           std::vector<FileDialogFilter> filters,
                           FileDialogCallback callback);
    void RequestSaveFile(const char *operation,
                         const char *title,
                         const std::string &defaultName,
                         std::vector<FileDialogFilter> filters,
                         FileDialogCallback callback);
    void RequestOpenFile(const char *operation,
                         const char *title,
                         std::vector<FileDialogFilter> filters,
                         FileDialogCallback callback);
    void RequestPick(const char *operation,
                     const char *title,
                     std::vector<std::string> items,
                     PickFromListCallback callback);
    void RequestFractalOutputSave(const char *operation,
                                  const char *title,
                                  FractalShark::FractalOutputFile outputType,
                                  std::string extension,
                                  std::vector<FileDialogFilter> filters);
    void SaveLocation(bool scaleToMaximum);
    void DoSaveRefOrbit(::CompressToDisk compression);
    void DoLoadRefOrbitImag(::ImaginaSettings settings);
    void RequestLoadRefOrbitImagFileDialog(::ImaginaSettings settings);
    void LoadRefOrbitImagFile(::ImaginaSettings settings, std::string filename);
};

LinuxMainWindow::LinuxMainWindow()
{
    const auto cleanup = [this](LinuxMainWindow *) { Destroy(); };
    std::unique_ptr<LinuxMainWindow, decltype(cleanup)> constructionGuard(this, cleanup);

    display = XOpenDisplay(nullptr);
    if (!display) {
        throw FractalSharkSeriousException("XOpenDisplay failed; DISPLAY may be unset");
    }

    screen = DefaultScreen(display);

    // Pick the best GLX-compatible visual available.  OpenGlContext later
    // creates its context from this window's actual visual, keeping fallback
    // selection in one place.
    const GlxVisualSelection visualSelection = ChooseGlxVisual(display, screen);
    visualInfo = visualSelection.VisualInfo;
    if (!visualInfo) {
        throw FractalSharkSeriousException(
            "glXChooseVisual failed after trying supported RGBA configurations; check GLX support");
    }

    Window root = RootWindow(display, screen);

    colormap = XCreateColormap(display, root, visualInfo->visual, AllocNone);
    if (!colormap) {
        throw FractalSharkSeriousException("XCreateColormap failed");
    }
    idleCursor = XCreateFontCursor(display, XC_left_ptr);
    dragCursor = XCreateFontCursor(display, XC_crosshair);
    if (!idleCursor || !dragCursor) {
        throw FractalSharkSeriousException("XCreateFontCursor failed");
    }

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
    if (!window) {
        throw FractalSharkSeriousException("XCreateWindow failed");
    }

    if (XStoreName(display, window, kWindowTitle) == 0) {
        throw FractalSharkSeriousException("XStoreName failed");
    }

    // Class hint — lets WMs recognize and theme the app.
    XClassHint classHint{};
    std::string resName = "fractalshark";
    std::string resClass = "FractalShark";
    classHint.res_name = resName.data();
    classHint.res_class = resClass.data();
    if (XSetClassHint(display, window, &classHint) == 0) {
        throw FractalSharkSeriousException("XSetClassHint failed");
    }

    // Receive WM_DELETE_WINDOW client messages instead of having the WM kill us.
    wmProtocols = XInternAtom(display, "WM_PROTOCOLS", False);
    wmDeleteWindow = XInternAtom(display, "WM_DELETE_WINDOW", False);
    if (wmProtocols == None || wmDeleteWindow == None ||
        XSetWMProtocols(display, window, &wmDeleteWindow, 1) == 0) {
        throw FractalSharkSeriousException("Failed to register the WM_DELETE_WINDOW protocol");
    }
    Environment::WaitCursor::RegisterLinuxCursorTarget(
        display, static_cast<uintptr_t>(window), static_cast<uintptr_t>(idleCursor));
    waitCursorRegistered = true;

    // Match Win32 KEYDOWN semantics: only autorepeat KeyPress, never KeyRelease.
    Bool supported = False;
    XkbSetDetectableAutoRepeat(display, True, &supported);

    XMapWindow(display, window);
    XFlush(display);

    clipboard.emplace(display, window);
    Environment::RegisterClipboardTextSetter(&SetEnvironmentClipboardText, &*clipboard);
    clipboardTextSetterRegistered = true;

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
        throw FractalSharkSeriousException("OpenGL context creation failed");
    }

    // ImGui overlay: single-threaded, all calls on this thread.  Init
    // requires GL to be current, which OpenGlContext::ctor already
    // ensured.
    menuState.emplace(*fractal);
    overlay.emplace(display, window, &*clipboard, &*menuState, this);
    overlay->Init();
    constructionGuard.release();
}

LinuxMainWindow::~LinuxMainWindow() { Destroy(); }

void
LinuxMainWindow::Destroy() noexcept
{
    CancelDragZoom(CurrentTime);

    // Destroy in reverse order of creation: overlay first (it owns ImGui
    // backends that need GL current), then GL context, then Fractal.
    overlay.reset();
    menuState.reset();
    glContext.reset();
    fractal.reset();
    if (clipboardTextSetterRegistered && clipboard) {
        Environment::UnregisterClipboardTextSetter(&SetEnvironmentClipboardText, &*clipboard);
        clipboardTextSetterRegistered = false;
    }
    clipboard.reset();

    if (display) {
        if (waitCursorRegistered) {
            Environment::WaitCursor::UnregisterLinuxCursorTarget();
            waitCursorRegistered = false;
        }
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
    if (ev.type == ButtonRelease && ev.xbutton.button == Button3 && suppressContextMenuButtonRelease) {
        suppressContextMenuButtonRelease = false;
        return;
    }

    if (ev.type == ButtonPress && ev.xbutton.button == Button3) {
        const auto &btn = ev.xbutton;
        if (!fractal) {
            throw FractalSharkSeriousException("Button event received without an initialized fractal");
        }
        // Keep this initiating click out of ImGui's mouse-button state; it is only the popup trigger.
        contextMenuX = btn.x;
        contextMenuY = btn.y;
        suppressContextMenuButtonRelease = true;
        RequireUiState(overlay.has_value(), "Opening the context menu");
        overlay->RequestContextMenu(btn.x, btn.y);
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
            // Later exposes still matter: window manager expose events require
            // the cached frame to be presented again.
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
                throw FractalSharkSeriousException(
                    "Button event received without an initialized fractal");
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
                case Button4: {
                    // Wheel forward → zoom in toward cursor.  Mirrors
                    // MainWindow.cpp:1393 `ZoomTowardPoint(x, y, -0.3)`.
                    const int x = btn.x;
                    const int y = btn.y;
                    fractal->EnqueueCommand([x, y](Fractal &f) { f.ZoomTowardPoint(x, y, -0.3); },
                                            false);
                    break;
                }
                case Button5:
                    // Wheel backward → zoom out at center.  Mirrors
                    // MainWindow.cpp:1397 `ZoomAtCenter(0.3)`.
                    fractal->EnqueueCommand([](Fractal &f) { f.ZoomAtCenter(0.3); }, false);
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
            RequireUiState(overlay.has_value(), "Updating drag zoom");
            overlay->SetDragRect(true, dragAnchorX, dragAnchorY, dragPrevX, dragPrevY);
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
        throw FractalSharkSeriousException("Linux event loop started without an X display");
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
        const int pollResult = poll(&pfd, 1, GetPresentationPollTimeoutMs(presentation.NeedsTick));
        if (pollResult < 0 && errno != EINTR) {
            const int errorCode = errno;
            throw FractalSharkSeriousException(std::string("Linux UI poll failed: ") +
                                               std::generic_category().message(errorCode));
        }
    }
}

PresentationTickResult
LinuxMainWindow::PresentRenderTick()
{
    // Pull any frames the render pool has finished and present them.
    // TryPresentTick skips ready tombstones and uploads at most one
    // visible frame so each completed animation step gets a swap.
    RequireUiState(fractal && glContext && glContext->IsValid() && overlay, "Presenting a render frame");
    auto *pool = fractal->GetRenderPool();
    if (!pool) {
        throw FractalSharkSeriousException("Presenting a render frame requires a render pool");
    }
    const bool freshFrame = pool->TryPresentTick(*glContext);

    const bool overlayWantsTick = overlay->WantsTick();
    const bool needsTick =
        freshFrame || overlayWantsTick || dragging || exposeRepaintPending || !everPresented;

    if (needsTick) {
        static bool diagnosedFirstFreshFrame = false;
        const bool diagnoseFreshFrame = freshFrame && !diagnosedFirstFreshFrame;
        if (diagnoseFreshFrame) {
            diagnosedFirstFreshFrame = true;
        }

        // If this tick was triggered by overlay motion only, re-blit
        // the cached frame so the overlay has something underneath.
        if (!freshFrame) {
            pool->RepresentLastFrame(*glContext);
        }
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
    RequireUiState(fractal != nullptr, "Calculating presentation timing");
    if (auto *pool = fractal->GetRenderPool()) {
        if (auto pacingDelay = pool->GetTimeUntilNextPresentation()) {
            timeoutMs = std::min(timeoutMs, static_cast<int>(pacingDelay->count()));
        }
    } else {
        throw FractalSharkSeriousException("Calculating presentation timing requires a render pool");
    }
    return std::max(timeoutMs, 1);
}

void
LinuxMainWindow::RunFeatureAutoZoomSynchronously(int mouseX, int mouseY)
{
    if (!fractal || !glContext || !glContext->IsValid()) {
        throw FractalSharkSeriousException(
            "Feature autozoom requires an initialized fractal and GL context");
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
    const FractalShark::MenuPoint pt = GetMenuMousePos();
    RunFeatureAutoZoomSynchronously(pt.X, pt.Y);
}

void
LinuxMainWindow::OnShowHotkeys()
{
    const std::string body = FractalShark::BuildHotkeysHelpUtf8();
    ShowInfo("Hotkeys", body);
}

void
LinuxMainWindow::OnViewsHelp()
{
    const auto help = FractalShark::GetGuiHelpContent(FractalShark::GuiHelpTopic::Views);
    ShowInfo(help.Title.data(), help.Body.data());
}

void
LinuxMainWindow::OnHelpAlg()
{
    const auto help = FractalShark::GetGuiHelpContent(FractalShark::GuiHelpTopic::Algorithms);
    ShowInfo(help.Title.data(), help.Body.data());
}

void
LinuxMainWindow::OnMinimize()
{
    RequireUiState(display && window, "Minimizing the window");
    if (XIconifyWindow(display, window, screen) == 0) {
        throw FractalSharkSeriousException("XIconifyWindow failed");
    }
    XFlush(display);
}

void
LinuxMainWindow::OnCurPos()
{
    RequireUiState(fractal && overlay, "Copying the current position");
    std::string shortStr;
    std::string longStr;
    fractal->GetRenderDetails(shortStr, longStr);
    if (!Environment::SetClipboardText(longStr)) {
        std::fprintf(stderr, "Could not copy location to clipboard.\n");
    }
    if (shortStr.size() < 5000) {
        ShowInfo("Current Position", shortStr);
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
    RequireUiState(display && window, "Entering fullscreen");
    if (fullscreen) {
        return;
    }

    // Cache geometry so ExitFullscreen can restore it.  XGetWindowAttributes
    // x/y are relative to the parent (often a WM frame or root); for restore
    // purposes that's fine since we just XMoveResizeWindow back to the same
    // values.
    XWindowAttributes attrs{};
    if (XGetWindowAttributes(display, window, &attrs) == 0) {
        throw FractalSharkSeriousException("XGetWindowAttributes failed before entering fullscreen");
    }
    Window dummyChild = 0;
    int rx = 0, ry = 0;
    if (XTranslateCoordinates(
            display, window, RootWindow(display, screen), 0, 0, &rx, &ry, &dummyChild) == 0) {
        throw FractalSharkSeriousException("XTranslateCoordinates failed before entering fullscreen");
    }
    savedX = rx;
    savedY = ry;
    savedWidth = attrs.width;
    savedHeight = attrs.height;

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
        throw FractalSharkSeriousException("Failed to initialize fullscreen X atoms");
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
    if (XSendEvent(display,
                   RootWindow(display, screen),
                   False,
                   SubstructureRedirectMask | SubstructureNotifyMask,
                   &ev) == 0) {
        throw FractalSharkSeriousException("Failed to send the fullscreen request");
    }
    XFlush(display);
    fullscreen = true;
}

void
LinuxMainWindow::ExitFullscreen()
{
    RequireUiState(display && window, "Exiting fullscreen");
    if (!fullscreen) {
        return;
    }

    Atom wmState = XInternAtom(display, "_NET_WM_STATE", False);
    Atom wmFullscreen = XInternAtom(display, "_NET_WM_STATE_FULLSCREEN", False);
    if (wmState == None || wmFullscreen == None) {
        throw FractalSharkSeriousException("Failed to initialize fullscreen X atoms");
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
    if (XSendEvent(display,
                   RootWindow(display, screen),
                   False,
                   SubstructureRedirectMask | SubstructureNotifyMask,
                   &ev) == 0) {
        throw FractalSharkSeriousException("Failed to send the windowed-mode request");
    }

    if (savedWidth > 0 && savedHeight > 0) {
        XMoveResizeWindow(display, window, savedX, savedY, savedWidth, savedHeight);
    }
    XFlush(display);
    fullscreen = false;
}

namespace {

constexpr size_t kMaxQuickPickImaginaFiles = 30;

using LinuxFileDialogFilter = FractalShark::Linux::ImGuiOverlay::FileDialogFilter;

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

} // namespace

void
LinuxMainWindow::ShowInfo(const char *title, const char *body)
{
    RequireUiState(overlay.has_value(), title);
    overlay->RequestInfoModal(title, body);
}

void
LinuxMainWindow::ShowInfo(const char *title, const std::string &body)
{
    ShowInfo(title, body.c_str());
}

void
LinuxMainWindow::RequestFileDialog(const char *operation,
                                   const char *title,
                                   FileDialogMode mode,
                                   const std::string &defaultName,
                                   std::vector<FileDialogFilter> filters,
                                   FileDialogCallback callback)
{
    RequireUiState(overlay.has_value(), operation);
    overlay->RequestFileDialog(title, mode, defaultName, std::move(filters), std::move(callback));
}

void
LinuxMainWindow::RequestSaveFile(const char *operation,
                                 const char *title,
                                 const std::string &defaultName,
                                 std::vector<FileDialogFilter> filters,
                                 FileDialogCallback callback)
{
    RequestFileDialog(
        operation, title, FileDialogMode::Save, defaultName, std::move(filters), std::move(callback));
}

void
LinuxMainWindow::RequestOpenFile(const char *operation,
                                 const char *title,
                                 std::vector<FileDialogFilter> filters,
                                 FileDialogCallback callback)
{
    RequestFileDialog(
        operation, title, FileDialogMode::Open, "", std::move(filters), std::move(callback));
}

void
LinuxMainWindow::RequestPick(const char *operation,
                             const char *title,
                             std::vector<std::string> items,
                             PickFromListCallback callback)
{
    RequireUiState(overlay.has_value(), operation);
    overlay->RequestPickFromList(title, std::move(items), std::move(callback));
}

void
LinuxMainWindow::RequestFractalOutputSave(const char *operation,
                                          const char *title,
                                          FractalShark::FractalOutputFile outputType,
                                          std::string extension,
                                          std::vector<FileDialogFilter> filters)
{
    RequireUiState(fractal && overlay, operation);
    RequestSaveFile(
        operation,
        title,
        FractalShark::MakeTimestampedOutputStem() + extension,
        std::move(filters),
        [this, operation, outputType, extension = std::move(extension)](std::string filename) {
            RequireUiState(fractal != nullptr, operation);
            filename = FractalShark::AppendExtensionIfMissing(std::move(filename), extension);
            FractalShark::SaveFractalOutput(*fractal, outputType, Environment::Utf8ToWide(filename));
        });
}

void
LinuxMainWindow::OnSaveLocation()
{
    RequireUiState(fractal && overlay, "Saving a location");

    RequestPick("Saving a location",
                "Scale dimensions to maximum?",
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
    RequireUiState(fractal && overlay, "Saving a location");
    const std::string serialized = FractalShark::AppendSavedLocation(*fractal, scaleToMaximum);
    ShowInfo("Location", serialized);
}

void
LinuxMainWindow::OnSaveHiResBmp()
{
    RequestFractalOutputSave("Saving a high-resolution image",
                             "Save high-resolution image",
                             FractalShark::FractalOutputFile::HighResolutionImage,
                             ".png",
                             PngFileFilters());
}

void
LinuxMainWindow::OnSaveItersText()
{
    RequestFractalOutputSave("Saving iterations",
                             "Save iterations as text",
                             FractalShark::FractalOutputFile::IterationsText,
                             ".txt",
                             TextFileFilters());
}

void
LinuxMainWindow::OnSaveBmp()
{
    RequestFractalOutputSave("Saving an image",
                             "Save current image",
                             FractalShark::FractalOutputFile::CurrentImage,
                             ".png",
                             PngFileFilters());
}

void
LinuxMainWindow::DoSaveRefOrbit(::CompressToDisk compression)
{
    RequireUiState(fractal && overlay, "Saving a reference orbit");
    std::string defaultName = FractalShark::MakeTimestampedOutputStem() + ".im";
    RequestSaveFile("Saving a reference orbit",
                    "Save reference orbit",
                    defaultName,
                    OrbitFileFilters(),
                    [this, compression](std::string filename) {
                        RequireUiState(fractal != nullptr, "Saving a reference orbit");
                        filename = FractalShark::AppendExtensionIfMissing(std::move(filename), ".im");
                        FractalShark::SaveReferenceOrbit(
                            *fractal, compression, Environment::Utf8ToWide(filename));
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
    RequireUiState(fractal && overlay, "Diffing reference orbits");
    // Chain: 1) prompt for output filename, 2) pick first input,
    // 3) pick second input, 4) enqueue diff.
    RequestSaveFile(
        "Diffing reference orbits",
        "Output (.im) - diff result",
        FractalShark::MakeTimestampedOutputStem() + "_diff.im",
        OrbitFileFilters(),
        [this](std::string outFile) mutable {
            RequireUiState(fractal && overlay, "Diffing reference orbits");
            outFile = FractalShark::AppendExtensionIfMissing(std::move(outFile), ".im");
            RequestOpenFile(
                "Diffing reference orbits",
                "First input (.im)",
                OrbitFileFilters(),
                [this, outFile = std::move(outFile)](std::string in1) mutable {
                    RequireUiState(fractal && overlay, "Diffing reference orbits");
                    RequestOpenFile(
                        "Diffing reference orbits",
                        "Second input (.im)",
                        OrbitFileFilters(),
                        [this, outFile = std::move(outFile), in1 = std::move(in1)](std::string in2) {
                            RequireUiState(fractal != nullptr, "Diffing reference orbits");
                            std::wstring outW = Environment::Utf8ToWide(outFile);
                            std::wstring f1W = Environment::Utf8ToWide(in1);
                            std::wstring f2W = Environment::Utf8ToWide(in2);
                            FractalShark::DiffReferenceOrbits(
                                *fractal, std::move(outW), std::move(f1W), std::move(f2W));
                        });
                });
        });
}

void
LinuxMainWindow::OnLoadLocation()
{
    RequireUiState(fractal && overlay, "Loading a location");
    std::vector<FractalShark::SavedLocation> locations =
        FractalShark::ReadSavedLocationsFile(FractalShark::kSavedLocationsFilename, 30);
    if (locations.empty()) {
        ShowInfo("Load Location", "locations.txt has no entries.");
        return;
    }

    std::vector<std::string> labels = FractalShark::BuildSavedLocationLabels(locations);
    RequestPick(
        "Loading a location",
        "Load saved location",
        std::move(labels),
        [this, locations = std::move(locations)](size_t index) {
            RequireUiState(fractal != nullptr, "Loading a location");
            if (index >= locations.size()) {
                throw FractalSharkSeriousException("Selected saved-location index is out of range");
            }
            FractalShark::EnqueueSavedLocation(*fractal, locations[index]);
        });
}

void
LinuxMainWindow::OnLoadEnterLocation()
{
    RequireUiState(fractal && overlay, "Entering a location");
    FractalShark::EnteredLocation current = FractalShark::CaptureEnteredLocation(*fractal);
    overlay->RequestEnterLocation(
        std::move(current.Real),
        std::move(current.Imaginary),
        std::move(current.Zoom),
        current.NumIterations,
        [this](std::string real, std::string imaginary, std::string zoom, uint64_t numIterations) {
            RequireUiState(fractal != nullptr, "Entering a location");
            FractalShark::EnqueueEnteredLocation(
                *fractal, {std::move(real), std::move(imaginary), std::move(zoom), numIterations});
        });
}

void
LinuxMainWindow::DoLoadRefOrbitImag(::ImaginaSettings settings)
{
    RequireUiState(fractal && overlay, "Loading a reference orbit");

    std::vector<std::filesystem::path> imagFiles = FractalShark::FindReferenceOrbitFiles(
        std::filesystem::current_path(), kMaxQuickPickImaginaFiles);
    if (imagFiles.empty()) {
        RequestLoadRefOrbitImagFileDialog(settings);
        return;
    }

    std::vector<std::string> labels;
    labels.reserve(imagFiles.size() + 1);
    for (const std::filesystem::path &file : imagFiles) {
        labels.push_back(file.filename().string());
    }
    labels.push_back("Load from file...");

    RequestPick("Loading a reference orbit",
                "Load reference orbit (.im)",
                std::move(labels),
                [this, settings, imagFiles = std::move(imagFiles)](size_t index) {
                    RequireUiState(fractal && overlay, "Loading a reference orbit");
                    if (index < imagFiles.size()) {
                        LoadRefOrbitImagFile(settings, imagFiles[index].string());
                    } else if (index == imagFiles.size()) {
                        RequestLoadRefOrbitImagFileDialog(settings);
                    }
                });
}

void
LinuxMainWindow::RequestLoadRefOrbitImagFileDialog(::ImaginaSettings settings)
{
    RequireUiState(fractal && overlay, "Loading a reference orbit");
    RequestOpenFile(
        "Loading a reference orbit",
        "Load reference orbit (.im)",
        OrbitFileFilters(),
        [this, settings](std::string filename) { LoadRefOrbitImagFile(settings, std::move(filename)); });
}

void
LinuxMainWindow::LoadRefOrbitImagFile(::ImaginaSettings settings, std::string filename)
{
    RequireUiState(fractal != nullptr, "Loading a reference orbit");
    FractalShark::LoadReferenceOrbit(
        *fractal, ::CompressToDisk::MaxCompressionImagina, settings, Environment::Utf8ToWide(filename));
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
    RequireUiState(display && window && overlay, "Beginning drag zoom");
    dragging = true;
    dragAnchorX = btn.x;
    dragAnchorY = btn.y;
    dragPrevX = -1;
    dragPrevY = -1;

    pointerGrabbed = false;
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
    if (!pointerGrabbed) {
        throw FractalSharkSeriousException("XGrabPointer failed while beginning drag zoom");
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
        throw FractalSharkSeriousException("Failed to initialize the window-move X atom");
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

    if (XSendEvent(display,
                   RootWindow(display, screen),
                   False,
                   SubstructureRedirectMask | SubstructureNotifyMask,
                   &ev) == 0) {
        throw FractalSharkSeriousException("Failed to send the window-move request");
    }
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
        throw FractalSharkSeriousException("Drag zoom completed without an initialized fractal");
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
        throw FractalSharkSeriousException("Key event received without an initialized fractal");
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
        ExecuteCommand(command->id);
    }
}

} // namespace

namespace FractalShark::Linux {

int
RunMainWindow(const std::function<void()> &onReady)
{
    LinuxMainWindow win;
    if (onReady) {
        onReady();
    }
    win.RunEventLoop();
    return 0;
}

} // namespace FractalShark::Linux
