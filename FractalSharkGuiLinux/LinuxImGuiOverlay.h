// LinuxImGuiOverlay.h
//
// Single-threaded ImGui overlay for the Linux GUI.  All ImGui state and
// every ImGui call happens on the GUI thread (the same thread that runs
// the X event loop and owns the GLX context).
//
// Lifecycle:
//   1. Construct on the GUI thread.  The ImGui context is created here,
//      but the platform/renderer backends are NOT initialized yet (GL
//      may not be current at construction time).
//   2. Call Init() once after the GL context has been made current.
//      Returns false on backend init failure.
//   3. Call ProcessEvent for every XEvent before dispatching it to the
//      app's own handler.  Returns true if ImGui consumed it (caller
//      should drop the event).
//   4. Call RenderFrame() once per main-loop tick (after fractal frame
//      is presented to GL, before SwapBuffers).
//   5. Destroy on the GUI thread before tearing down the GL context.
//
// No mutexes, no event queue, no atomics — everything runs serially on
// one thread.

#pragma once

#include <X11/Xlib.h>

struct ImGuiContext;

namespace FractalShark {
struct ExecuteCommandHost;
struct LinuxClipboard;
namespace Menu {
struct IMenuState;
} // namespace Menu
} // namespace FractalShark

namespace FractalShark::Linux {

class ImGuiOverlay {
public:
    ImGuiOverlay(Display *display, Window window, FractalShark::LinuxClipboard *clipboard);
    ~ImGuiOverlay();

    ImGuiOverlay(const ImGuiOverlay &) = delete;
    ImGuiOverlay &operator=(const ImGuiOverlay &) = delete;

    // Initialize the platform + renderer backends.  Must be called once,
    // on the GUI thread, after GL context is current.  Returns false on
    // backend init failure (rare).
    bool Init();

    // Forward an XEvent to ImGui.  Returns true when ImGui captured it
    // (a popup is up and the cursor / keyboard input is over the popup
    // area), in which case the caller should not run its own dispatch
    // for this event.
    bool ProcessEvent(const XEvent &ev);

    void SetExecuteHost(FractalShark::ExecuteCommandHost *host);
    void SetMenuState(const FractalShark::Menu::IMenuState *state);

    // Open the context menu at (x, y) (window-relative) on the next
    // RenderFrame() call.
    void RequestContextMenu(int x, int y);

    // Configure the drag-zoom rubber band.  active=false hides it.
    void SetDragRect(bool active, int x0, int y0, int x1, int y1);

    // Run NewFrame, build the UI (popup + drag rect), Render, and
    // RenderDrawData.  Caller is responsible for SwapBuffers().
    void RenderFrame();

    // True when ImGui state implies the loop should keep ticking even
    // without external input — e.g. a popup is open and animations
    // need to advance.  Used by the host's poll() loop to choose a
    // tighter timeout.
    bool WantsTick() const;

private:
    Display *display_;
    Window window_;
    FractalShark::LinuxClipboard *clipboard_;
    ImGuiContext *ctx_ = nullptr;
    bool xlibBackendInited_ = false;
    bool oglBackendInited_ = false;

    bool contextMenuRequested_ = false;
    int contextMenuX_ = 0;
    int contextMenuY_ = 0;

    bool dragRectActive_ = false;
    int dragX0_ = 0;
    int dragY0_ = 0;
    int dragX1_ = 0;
    int dragY1_ = 0;

    FractalShark::ExecuteCommandHost *host_ = nullptr;
    const FractalShark::Menu::IMenuState *menuState_ = nullptr;
};

} // namespace FractalShark::Linux
