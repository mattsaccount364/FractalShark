// FractalSharkGuiLinux — Xlib window skeleton.
//
// Phase: xlib-window-creation.  This stage brings up an empty X11 window with
// a working close button + keyboard auto-repeat tweak, runs an event loop, and
// exits cleanly on WM_DELETE_WINDOW.  No GLX context yet — that arrives with
// the fractal-render task, when OpenGlContext is wired in.

#include "CrashHandler.h"
#include "Environment.h"
#include "LinuxClipboard.h"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

struct LinuxMainWindow {
    Display *display = nullptr;
    Window window = 0;
    Atom wmDeleteWindow = 0;
    Atom wmProtocols = 0;
    int screen = 0;
    bool running = true;
    std::optional<FractalShark::LinuxClipboard> clipboard;

    LinuxMainWindow();
    ~LinuxMainWindow();

    LinuxMainWindow(const LinuxMainWindow &) = delete;
    LinuxMainWindow &operator=(const LinuxMainWindow &) = delete;

    bool Valid() const noexcept { return display != nullptr && window != 0; }
    void RunEventLoop();
    void HandleEvent(const XEvent &ev);
};

LinuxMainWindow::LinuxMainWindow()
{
    display = XOpenDisplay(nullptr);
    if (!display) {
        std::fprintf(stderr, "FractalSharkGuiLinux: XOpenDisplay failed (DISPLAY env not set?)\n");
        return;
    }

    screen = DefaultScreen(display);

    const unsigned long black = BlackPixel(display, screen);
    const unsigned long white = WhitePixel(display, screen);

    window = XCreateSimpleWindow(display,
                                 RootWindow(display, screen),
                                 0,
                                 0,
                                 kInitialWidth,
                                 kInitialHeight,
                                 0,
                                 white,
                                 black);

    XStoreName(display, window, kWindowTitle);

    // Class hint — lets WMs recognize and theme the app.
    XClassHint classHint{};
    std::string resName = "fractalshark";
    std::string resClass = "FractalShark";
    classHint.res_name = resName.data();
    classHint.res_class = resClass.data();
    XSetClassHint(display, window, &classHint);

    // Subscribe to events we'll need throughout the GUI (input + lifecycle).
    XSelectInput(display,
                 window,
                 ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask
                     | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask
                     | FocusChangeMask);

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
}

LinuxMainWindow::~LinuxMainWindow()
{
    if (display) {
        if (window) {
            XDestroyWindow(display, window);
            window = 0;
        }
        XCloseDisplay(display);
        display = nullptr;
    }
}

void
LinuxMainWindow::HandleEvent(const XEvent &ev)
{
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
        // Until OpenGL is bound, we have nothing to draw.  Leave the window
        // black so the framework is visible without confusing flicker.
        break;

    default:
        break;
    }
}

void
LinuxMainWindow::RunEventLoop()
{
    XEvent ev;
    while (running) {
        XNextEvent(display, &ev);
        if (clipboard && clipboard->ProcessEvent(ev)) {
            continue;
        }
        HandleEvent(ev);
    }
}

} // namespace

int
main(int /*argc*/, char ** /*argv*/)
{
    Environment::RegisterHeapCleanup();
    CrashHandler::Install();

    LinuxMainWindow win;
    if (!win.Valid()) {
        return 1;
    }
    win.RunEventLoop();
    return 0;
}
