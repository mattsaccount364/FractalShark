// LinuxClipboard — X11 selection clipboard targeting CLIPBOARD (not PRIMARY).
//
// Mirrors the Win32 OpenClipboard / SetClipboardData(CF_TEXT) / GetClipboardData
// usage in MainWindow.cpp.  Only plain-text payloads are exchanged; payloads
// are tiny (coordinate strings, locations files), so INCR transfer is not
// implemented.
//
// Usage:
//     LinuxClipboard clip(display, window);
//     clip.Set("hello");
//     auto v = clip.Get();
//     // In the main event loop, forward every X event to clip.ProcessEvent().

#pragma once

#include <X11/Xlib.h>

#include <chrono>
#include <optional>
#include <string>

namespace FractalShark::Linux {

struct LinuxClipboard {
    LinuxClipboard(Display *display, Window window);

    LinuxClipboard(const LinuxClipboard &) = delete;
    LinuxClipboard &operator=(const LinuxClipboard &) = delete;

    // Take ownership of the CLIPBOARD selection with the given UTF-8 payload.
    void Set(std::string text);

    // Request the current clipboard contents.  Synchronous: pumps X events
    // until SelectionNotify arrives or `timeout` elapses.  Returns nullopt when
    // no client owns the selection.  Throws on timeout or conversion failure.
    std::optional<std::string> Get(std::chrono::milliseconds timeout = std::chrono::milliseconds(250));

    // Forward every X event from the main loop here.  Returns true if the
    // event was consumed (SelectionRequest / SelectionClear / SelectionNotify
    // for our window).  Required so SelectionRequests from other clients can
    // be answered while we hold ownership.
    bool ProcessEvent(const XEvent &ev);

private:
    Display *m_Display;
    Window m_Window;

    // Cached atoms.
    Atom m_AtomClipboard;
    Atom m_AtomTargets;
    Atom m_AtomUtf8String;
    Atom m_AtomString;
    Atom m_AtomText;
    Atom m_AtomIncr;
    Atom m_AtomFractalSharkPaste; // private property used by Get()

    // Owned text — answered to SelectionRequest while we own CLIPBOARD.
    std::string m_OwnedText;
    bool m_OwnsSelection = false;

    // Get-pump state.
    bool m_GetPending = false;
    std::optional<std::string> m_GetResult;

    void RespondSelectionRequest(const XSelectionRequestEvent &req);
};

} // namespace FractalShark::Linux
