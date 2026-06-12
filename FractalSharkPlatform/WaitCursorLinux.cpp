#include "WaitCursor.h"

#include <X11/Xlib.h>
#include <X11/cursorfont.h>

#include <mutex>

namespace {

struct LinuxCursorTarget {
    Display *DisplayHandle = nullptr;
    Window WindowHandle = 0;
    Cursor NormalCursor = 0;
    Cursor WaitCursor = 0;
};

std::mutex g_LinuxCursorMutex;
LinuxCursorTarget g_LinuxCursorTarget;

} // namespace

WaitCursor::WaitCursor() : m_CursorSet{false}
{
    std::lock_guard lock{g_LinuxCursorMutex};
    if (!g_LinuxCursorTarget.DisplayHandle || !g_LinuxCursorTarget.WindowHandle ||
        !g_LinuxCursorTarget.WaitCursor) {
        return;
    }

    XDefineCursor(g_LinuxCursorTarget.DisplayHandle,
                  g_LinuxCursorTarget.WindowHandle,
                  g_LinuxCursorTarget.WaitCursor);
    XFlush(g_LinuxCursorTarget.DisplayHandle);
    m_CursorSet = true;
}

WaitCursor::~WaitCursor() { ResetCursor(); }

void
WaitCursor::ResetCursor()
{
    std::lock_guard lock{g_LinuxCursorMutex};
    if (!m_CursorSet || !g_LinuxCursorTarget.DisplayHandle || !g_LinuxCursorTarget.WindowHandle) {
        return;
    }

    if (g_LinuxCursorTarget.NormalCursor) {
        XDefineCursor(g_LinuxCursorTarget.DisplayHandle,
                      g_LinuxCursorTarget.WindowHandle,
                      g_LinuxCursorTarget.NormalCursor);
    } else {
        XUndefineCursor(g_LinuxCursorTarget.DisplayHandle, g_LinuxCursorTarget.WindowHandle);
    }
    XFlush(g_LinuxCursorTarget.DisplayHandle);
    m_CursorSet = false;
}

void
WaitCursor::RegisterLinuxCursorTarget(void *display, std::uintptr_t window, std::uintptr_t normalCursor)
{
    auto *xDisplay = static_cast<Display *>(display);
    std::lock_guard lock{g_LinuxCursorMutex};

    if (g_LinuxCursorTarget.DisplayHandle && g_LinuxCursorTarget.WaitCursor) {
        XFreeCursor(g_LinuxCursorTarget.DisplayHandle, g_LinuxCursorTarget.WaitCursor);
    }

    g_LinuxCursorTarget = {};
    if (!xDisplay || window == 0) {
        return;
    }

    g_LinuxCursorTarget.DisplayHandle = xDisplay;
    g_LinuxCursorTarget.WindowHandle = static_cast<Window>(window);
    g_LinuxCursorTarget.NormalCursor = static_cast<Cursor>(normalCursor);
    g_LinuxCursorTarget.WaitCursor = XCreateFontCursor(xDisplay, XC_watch);
}

void
WaitCursor::UnregisterLinuxCursorTarget()
{
    std::lock_guard lock{g_LinuxCursorMutex};
    if (g_LinuxCursorTarget.DisplayHandle && g_LinuxCursorTarget.WaitCursor) {
        XFreeCursor(g_LinuxCursorTarget.DisplayHandle, g_LinuxCursorTarget.WaitCursor);
    }
    g_LinuxCursorTarget = {};
}
