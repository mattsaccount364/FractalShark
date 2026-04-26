// LinuxImGuiOverlay.h
//
// Owns the ImGui context for the Linux GUI.  All ImGui state lives here;
// access is serialized so the GUI thread (XNextEvent loop) and the GL
// consumer thread (RenderThreadPool) can both touch it without racing.
//
// Thread model:
//
//   GUI thread queues XEvents via QueueEvent() and toggles UI state
//   (e.g. RequestContextMenu).  It never calls ImGui directly.
//
//   GL consumer thread invokes the overlay callback installed via
//   InstallCallback().  Inside the callback we drain the event queue,
//   call ImGui::NewFrame, run the menu builder, ImGui::Render, and feed
//   the resulting DrawData to ImGui_ImplOpenGL2_RenderDrawData.  All of
//   this happens with the GL context already current.
//
// Single-threaded ImGui inside the callback avoids cross-thread
// ImDrawData transfer.  The cost is that menu logic runs on the render
// thread; that is fine because every UI action ultimately enqueues onto
// the Fractal command queue which is itself thread-safe.

#pragma once

#include "OpenGLContext.h"

#include <X11/Xlib.h>

#include <atomic>
#include <deque>
#include <functional>
#include <mutex>

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

    // Install the overlay callback on the supplied OpenGlContext.  The
    // OpenGlContext lives inside the Fractal's RenderThreadPool, owned by
    // the render pool.  Caller must arrange to clear the callback (via
    // OpenGlContext::SetOverlayCallback({})) before destroying the overlay.
    void InstallCallback(OpenGlContext &gl);

    // Called from the GUI thread for each XEvent.  Stored under mutex,
    // drained on the consumer thread's first NewFrame after enqueue.
    // Returns true if ImGui will likely consume the event (caller should
    // skip its own dispatch).  This is a cheap heuristic based on the
    // last-known WantCaptureMouse / WantCaptureKeyboard flags; the
    // authoritative consumption happens inside ImGui_ImplXlib_ProcessEvent
    // on the consumer thread.
    bool QueueEvent(const XEvent &ev);

    // Set the host whose ExecuteCommandHost methods are invoked when the
    // user clicks a menu item.  Lifetime: must outlive the overlay.
    void SetExecuteHost(FractalShark::ExecuteCommandHost *host);

    // Set the menu-state oracle (rule evaluator) used to compute
    // checkmarks / enabled-disabled.  Lifetime: must outlive the overlay.
    void SetMenuState(const FractalShark::Menu::IMenuState *state);

    // Request the context menu open at (x, y) (window-relative) on the
    // next consumer-thread frame.
    void RequestContextMenu(int x, int y);

    // For the drag-zoom rubber band.  GUI thread sets the rect each
    // MotionNotify; consumer reads it under mutex when drawing.
    // Setting active=false hides the rect.
    void SetDragRect(bool active, int x0, int y0, int x1, int y1);

    // Drain queued events, ImGui::NewFrame, build menu, ImGui::Render,
    // RenderDrawData.  Called by the GL consumer thread once per frame
    // (just before SwapBuffers) via the overlay callback registered on
    // RenderThreadPool::SetOverlayCallback.  GL context must be current.
    void Render() noexcept;

private:
    void DrainEvents();

    Display *display_;
    Window window_;
    FractalShark::LinuxClipboard *clipboard_;
    ImGuiContext *ctx_ = nullptr;
    OpenGlContext *installedOn_ = nullptr;

    std::mutex mu_;
    bool xlibBackendInited_ = false;
    bool oglBackendInited_ = false;

    std::deque<XEvent> pendingEvents_;

    // Context menu request from GUI thread.
    bool contextMenuRequested_ = false;
    int contextMenuX_ = 0;
    int contextMenuY_ = 0;
    // Latched: once the consumer opens the popup we clear the request flag.

    // Drag-zoom rubber band (drawn via ImGui foreground draw list).
    bool dragRectActive_ = false;
    int dragX0_ = 0;
    int dragY0_ = 0;
    int dragX1_ = 0;
    int dragY1_ = 0;

    FractalShark::ExecuteCommandHost *host_ = nullptr;
    const FractalShark::Menu::IMenuState *menuState_ = nullptr;

    // Heuristic capture flags published by the consumer thread for the
    // GUI thread's QueueEvent return value.  Atomic because GUI reads
    // without holding mu_.
    std::atomic<bool> wantCaptureMouse_{false};
    std::atomic<bool> wantCaptureKeyboard_{false};
};

} // namespace FractalShark::Linux
