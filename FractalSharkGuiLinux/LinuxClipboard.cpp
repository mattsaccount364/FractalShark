#include "LinuxClipboard.h"

#include "Exceptions.h"

#include <X11/Xatom.h>

#include <cstring>
#include <thread>

namespace FractalShark::Linux {

LinuxClipboard::LinuxClipboard(Display *display, Window window) : m_Display(display), m_Window(window)
{
    if (!m_Display || !m_Window) {
        throw FractalSharkSeriousException("LinuxClipboard requires a valid X display and window");
    }
    m_AtomClipboard = XInternAtom(m_Display, "CLIPBOARD", False);
    m_AtomTargets = XInternAtom(m_Display, "TARGETS", False);
    m_AtomUtf8String = XInternAtom(m_Display, "UTF8_STRING", False);
    m_AtomString = XA_STRING;
    m_AtomText = XInternAtom(m_Display, "TEXT", False);
    m_AtomIncr = XInternAtom(m_Display, "INCR", False);
    m_AtomFractalSharkPaste = XInternAtom(m_Display, "_FRACTALSHARK_PASTE", False);
    if (m_AtomClipboard == None || m_AtomTargets == None || m_AtomUtf8String == None ||
        m_AtomText == None || m_AtomIncr == None || m_AtomFractalSharkPaste == None) {
        throw FractalSharkSeriousException("Failed to initialize X clipboard atoms");
    }
}

void
LinuxClipboard::Set(std::string text)
{
    m_OwnedText = std::move(text);
    XSetSelectionOwner(m_Display, m_AtomClipboard, m_Window, CurrentTime);
    m_OwnsSelection = (XGetSelectionOwner(m_Display, m_AtomClipboard) == m_Window);
    XFlush(m_Display);
    if (!m_OwnsSelection) {
        throw FractalSharkSeriousException("Failed to take ownership of the X clipboard selection");
    }
}

std::optional<std::string>
LinuxClipboard::Get(std::chrono::milliseconds timeout)
{
    // Optimization: if we own the selection, just return our copy without a
    // round-trip through the X server.
    if (m_OwnsSelection) {
        return m_OwnedText;
    }

    Window owner = XGetSelectionOwner(m_Display, m_AtomClipboard);
    if (owner == None) {
        return std::nullopt;
    }

    m_GetPending = true;
    m_GetResult.reset();

    XConvertSelection(
        m_Display, m_AtomClipboard, m_AtomUtf8String, m_AtomFractalSharkPaste, m_Window, CurrentTime);
    XFlush(m_Display);

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (m_GetPending && std::chrono::steady_clock::now() < deadline) {
        if (XPending(m_Display) > 0) {
            XEvent ev;
            XNextEvent(m_Display, &ev);
            ProcessEvent(ev);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    if (m_GetPending) {
        m_GetPending = false;
        throw FractalSharkSeriousException("Timed out waiting for the X clipboard selection");
    }
    return std::move(m_GetResult);
}

void
LinuxClipboard::RespondSelectionRequest(const XSelectionRequestEvent &req)
{
    XSelectionEvent reply{};
    reply.type = SelectionNotify;
    reply.display = req.display;
    reply.requestor = req.requestor;
    reply.selection = req.selection;
    reply.target = req.target;
    reply.property = None; // refuse by default; success path overrides
    reply.time = req.time;

    if (req.target == m_AtomTargets) {
        // Tell the requestor which targets we support.
        const Atom targets[] = {m_AtomTargets, m_AtomUtf8String, m_AtomString, m_AtomText};
        XChangeProperty(req.display,
                        req.requestor,
                        req.property,
                        XA_ATOM,
                        32,
                        PropModeReplace,
                        reinterpret_cast<const unsigned char *>(targets),
                        static_cast<int>(sizeof(targets) / sizeof(targets[0])));
        reply.property = req.property;
    } else if (req.target == m_AtomUtf8String || req.target == m_AtomString ||
               req.target == m_AtomText) {
        XChangeProperty(req.display,
                        req.requestor,
                        req.property,
                        req.target,
                        8,
                        PropModeReplace,
                        reinterpret_cast<const unsigned char *>(m_OwnedText.data()),
                        static_cast<int>(m_OwnedText.size()));
        reply.property = req.property;
    }

    if (XSendEvent(req.display, req.requestor, False, NoEventMask, reinterpret_cast<XEvent *>(&reply)) ==
        0) {
        throw FractalSharkSeriousException("Failed to send an X clipboard selection response");
    }
    XFlush(req.display);
}

bool
LinuxClipboard::ProcessEvent(const XEvent &ev)
{
    switch (ev.type) {
        case SelectionRequest: {
            const auto &req = ev.xselectionrequest;
            if (req.owner != m_Window || req.selection != m_AtomClipboard) {
                return false;
            }
            RespondSelectionRequest(req);
            return true;
        }

        case SelectionClear: {
            const auto &clr = ev.xselectionclear;
            if (clr.window != m_Window || clr.selection != m_AtomClipboard) {
                return false;
            }
            m_OwnsSelection = false;
            m_OwnedText.clear();
            return true;
        }

        case SelectionNotify: {
            const auto &sel = ev.xselection;
            if (sel.requestor != m_Window || sel.selection != m_AtomClipboard) {
                return false;
            }
            if (sel.property == None) {
                m_GetPending = false;
                throw FractalSharkSeriousException("X clipboard owner refused UTF-8 conversion");
            }
            if (sel.property != m_AtomFractalSharkPaste || sel.target != m_AtomUtf8String) {
                return false;
            }

            Atom actualType = 0;
            int actualFormat = 0;
            unsigned long nItems = 0;
            unsigned long bytesAfter = 0;
            unsigned char *data = nullptr;
            const int status = XGetWindowProperty(m_Display,
                                                  m_Window,
                                                  m_AtomFractalSharkPaste,
                                                  0,
                                                  (~0L) / 4,
                                                  True, // delete on read
                                                  AnyPropertyType,
                                                  &actualType,
                                                  &actualFormat,
                                                  &nItems,
                                                  &bytesAfter,
                                                  &data);
            if (status != Success || !data) {
                m_GetPending = false;
                throw FractalSharkSeriousException("Failed to read the X clipboard selection property");
            }
            if (actualType == m_AtomIncr) {
                XFree(data);
                m_GetPending = false;
                throw FractalSharkSeriousException(
                    "Incremental X clipboard transfers are not supported");
            }
            if (actualFormat != 8 || bytesAfter != 0) {
                XFree(data);
                m_GetPending = false;
                throw FractalSharkSeriousException("X clipboard selection has an unexpected format");
            }
            m_GetResult = std::string(reinterpret_cast<const char *>(data), nItems);
            XFree(data);
            m_GetPending = false;
            return true;
        }

        default:
            return false;
    }
}

} // namespace FractalShark::Linux
