// LinuxX11ContextMenu.h
//
// Native X11 context-menu windows for the Linux GUI.  Unlike an ImGui popup
// rendered into the main GLX surface, these root-level windows can extend
// beyond the application client area.

#pragma once

#include <X11/Xlib.h>

#include <functional>
#include <memory>

namespace FractalShark {
struct ExecuteCommandHost;
namespace Menu {
struct IMenuState;
} // namespace Menu
} // namespace FractalShark

namespace FractalShark::Linux {

class X11ContextMenu final {
public:
    X11ContextMenu(Display *display,
                   int screen,
                   Window owner,
                   const Menu::IMenuState *state,
                   ExecuteCommandHost *host,
                   std::function<void()> repaintOwner);
    ~X11ContextMenu();

    X11ContextMenu(const X11ContextMenu &) = delete;
    X11ContextMenu &operator=(const X11ContextMenu &) = delete;

    void Open(int rootX, int rootY);
    void Close();

    // Returns true when the event belongs to an active native menu and must
    // not be dispatched to ImGui or the fractal window.
    bool ProcessEvent(const XEvent &ev);

    bool IsOpen() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace FractalShark::Linux
