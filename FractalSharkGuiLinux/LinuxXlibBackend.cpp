// LinuxXlibBackend — implementation.
//
// Translates Xlib events to Dear ImGui inputs.  Modeled on imgui_impl_glfw.cpp
// (upstream) and imgui_impl_x11.cpp (community).  Key/Button mapping is
// hand-derived from <X11/keysymdef.h>.

#include "LinuxXlibBackend.h"

#include "Exceptions.h"
#include "LinuxClipboard.h"

#include "imgui.h"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <cerrno>
#include <cstring>
#include <ctime>
#include <string>
#include <system_error>

namespace {

struct BackendData {
    Display *display = nullptr;
    Window window = 0;
    timespec lastTime = {};
    bool installed = false;
    FractalShark::Linux::LinuxClipboard *clipboard = nullptr;
    std::string clipboardCache; // ImGui's GetClipboardText returns const char*

    int displayWidth = 0;
    int displayHeight = 0;

    bool wantUpdateMonitors = true;
};

BackendData &
Self()
{
    static BackendData data;
    return data;
}

// ---------------------------------------------------------------------------
// KeySym → ImGuiKey table.
// ---------------------------------------------------------------------------

ImGuiKey
KeySymToImGuiKey(KeySym ks)
{
    switch (ks) {
        case XK_Tab:
            return ImGuiKey_Tab;
        case XK_Left:
            return ImGuiKey_LeftArrow;
        case XK_Right:
            return ImGuiKey_RightArrow;
        case XK_Up:
            return ImGuiKey_UpArrow;
        case XK_Down:
            return ImGuiKey_DownArrow;
        case XK_Page_Up:
            return ImGuiKey_PageUp;
        case XK_Page_Down:
            return ImGuiKey_PageDown;
        case XK_Home:
            return ImGuiKey_Home;
        case XK_End:
            return ImGuiKey_End;
        case XK_Insert:
            return ImGuiKey_Insert;
        case XK_Delete:
            return ImGuiKey_Delete;
        case XK_BackSpace:
            return ImGuiKey_Backspace;
        case XK_space:
            return ImGuiKey_Space;
        case XK_Return:
        case XK_KP_Enter:
            return ImGuiKey_Enter;
        case XK_Escape:
            return ImGuiKey_Escape;
        case XK_apostrophe:
            return ImGuiKey_Apostrophe;
        case XK_comma:
            return ImGuiKey_Comma;
        case XK_minus:
            return ImGuiKey_Minus;
        case XK_period:
            return ImGuiKey_Period;
        case XK_slash:
            return ImGuiKey_Slash;
        case XK_semicolon:
            return ImGuiKey_Semicolon;
        case XK_equal:
            return ImGuiKey_Equal;
        case XK_bracketleft:
            return ImGuiKey_LeftBracket;
        case XK_backslash:
            return ImGuiKey_Backslash;
        case XK_bracketright:
            return ImGuiKey_RightBracket;
        case XK_grave:
            return ImGuiKey_GraveAccent;
        case XK_Caps_Lock:
            return ImGuiKey_CapsLock;
        case XK_Scroll_Lock:
            return ImGuiKey_ScrollLock;
        case XK_Num_Lock:
            return ImGuiKey_NumLock;
        case XK_Print:
            return ImGuiKey_PrintScreen;
        case XK_Pause:
            return ImGuiKey_Pause;
        case XK_KP_0:
            return ImGuiKey_Keypad0;
        case XK_KP_1:
            return ImGuiKey_Keypad1;
        case XK_KP_2:
            return ImGuiKey_Keypad2;
        case XK_KP_3:
            return ImGuiKey_Keypad3;
        case XK_KP_4:
            return ImGuiKey_Keypad4;
        case XK_KP_5:
            return ImGuiKey_Keypad5;
        case XK_KP_6:
            return ImGuiKey_Keypad6;
        case XK_KP_7:
            return ImGuiKey_Keypad7;
        case XK_KP_8:
            return ImGuiKey_Keypad8;
        case XK_KP_9:
            return ImGuiKey_Keypad9;
        case XK_KP_Decimal:
            return ImGuiKey_KeypadDecimal;
        case XK_KP_Divide:
            return ImGuiKey_KeypadDivide;
        case XK_KP_Multiply:
            return ImGuiKey_KeypadMultiply;
        case XK_KP_Subtract:
            return ImGuiKey_KeypadSubtract;
        case XK_KP_Add:
            return ImGuiKey_KeypadAdd;
        case XK_KP_Equal:
            return ImGuiKey_KeypadEqual;
        case XK_Shift_L:
            return ImGuiKey_LeftShift;
        case XK_Control_L:
            return ImGuiKey_LeftCtrl;
        case XK_Alt_L:
            return ImGuiKey_LeftAlt;
        case XK_Super_L:
            return ImGuiKey_LeftSuper;
        case XK_Shift_R:
            return ImGuiKey_RightShift;
        case XK_Control_R:
            return ImGuiKey_RightCtrl;
        case XK_Alt_R:
            return ImGuiKey_RightAlt;
        case XK_Super_R:
            return ImGuiKey_RightSuper;
        case XK_Menu:
            return ImGuiKey_Menu;
        case XK_0:
            return ImGuiKey_0;
        case XK_1:
            return ImGuiKey_1;
        case XK_2:
            return ImGuiKey_2;
        case XK_3:
            return ImGuiKey_3;
        case XK_4:
            return ImGuiKey_4;
        case XK_5:
            return ImGuiKey_5;
        case XK_6:
            return ImGuiKey_6;
        case XK_7:
            return ImGuiKey_7;
        case XK_8:
            return ImGuiKey_8;
        case XK_9:
            return ImGuiKey_9;
        case XK_a:
        case XK_A:
            return ImGuiKey_A;
        case XK_b:
        case XK_B:
            return ImGuiKey_B;
        case XK_c:
        case XK_C:
            return ImGuiKey_C;
        case XK_d:
        case XK_D:
            return ImGuiKey_D;
        case XK_e:
        case XK_E:
            return ImGuiKey_E;
        case XK_f:
        case XK_F:
            return ImGuiKey_F;
        case XK_g:
        case XK_G:
            return ImGuiKey_G;
        case XK_h:
        case XK_H:
            return ImGuiKey_H;
        case XK_i:
        case XK_I:
            return ImGuiKey_I;
        case XK_j:
        case XK_J:
            return ImGuiKey_J;
        case XK_k:
        case XK_K:
            return ImGuiKey_K;
        case XK_l:
        case XK_L:
            return ImGuiKey_L;
        case XK_m:
        case XK_M:
            return ImGuiKey_M;
        case XK_n:
        case XK_N:
            return ImGuiKey_N;
        case XK_o:
        case XK_O:
            return ImGuiKey_O;
        case XK_p:
        case XK_P:
            return ImGuiKey_P;
        case XK_q:
        case XK_Q:
            return ImGuiKey_Q;
        case XK_r:
        case XK_R:
            return ImGuiKey_R;
        case XK_s:
        case XK_S:
            return ImGuiKey_S;
        case XK_t:
        case XK_T:
            return ImGuiKey_T;
        case XK_u:
        case XK_U:
            return ImGuiKey_U;
        case XK_v:
        case XK_V:
            return ImGuiKey_V;
        case XK_w:
        case XK_W:
            return ImGuiKey_W;
        case XK_x:
        case XK_X:
            return ImGuiKey_X;
        case XK_y:
        case XK_Y:
            return ImGuiKey_Y;
        case XK_z:
        case XK_Z:
            return ImGuiKey_Z;
        case XK_F1:
            return ImGuiKey_F1;
        case XK_F2:
            return ImGuiKey_F2;
        case XK_F3:
            return ImGuiKey_F3;
        case XK_F4:
            return ImGuiKey_F4;
        case XK_F5:
            return ImGuiKey_F5;
        case XK_F6:
            return ImGuiKey_F6;
        case XK_F7:
            return ImGuiKey_F7;
        case XK_F8:
            return ImGuiKey_F8;
        case XK_F9:
            return ImGuiKey_F9;
        case XK_F10:
            return ImGuiKey_F10;
        case XK_F11:
            return ImGuiKey_F11;
        case XK_F12:
            return ImGuiKey_F12;
        default:
            return ImGuiKey_None;
    }
}

void
UpdateModifiers(unsigned int xstate)
{
    ImGuiIO &io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Ctrl, (xstate & ControlMask) != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (xstate & ShiftMask) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (xstate & Mod1Mask) != 0);
    io.AddKeyEvent(ImGuiMod_Super, (xstate & Mod4Mask) != 0);
}

const char *
ImGuiGetClipboardCb(ImGuiContext * /*ctx*/)
{
    BackendData &data = Self();
    if (!data.clipboard) {
        throw FractalSharkSeriousException("ImGui Xlib clipboard helper is not initialized");
    }
    auto v = data.clipboard->Get();
    data.clipboardCache = v.value_or(std::string{});
    return data.clipboardCache.c_str();
}

void
ImGuiSetClipboardCb(ImGuiContext * /*ctx*/, const char *text)
{
    BackendData &data = Self();
    if (!data.clipboard || !text) {
        throw FractalSharkSeriousException("ImGui Xlib clipboard callback is not initialized");
    }
    data.clipboard->Set(text);
}

double
TimeSeconds(const timespec &ts)
{
    return double(ts.tv_sec) + double(ts.tv_nsec) / 1e9;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void
ImGui_ImplXlib_Init(Display *display, Window window)
{
    if (!display || !window) {
        throw FractalSharkSeriousException("ImGui Xlib backend requires a valid display and window");
    }

    BackendData &data = Self();
    if (data.installed) {
        return;
    }

    timespec initialTime{};
    if (clock_gettime(CLOCK_MONOTONIC, &initialTime) != 0) {
        const int errorCode = errno;
        throw FractalSharkSeriousException(std::string("clock_gettime failed: ") +
                                           std::generic_category().message(errorCode));
    }

    XWindowAttributes attr{};
    if (XGetWindowAttributes(display, window, &attr) == 0) {
        throw FractalSharkSeriousException("XGetWindowAttributes failed during ImGui initialization");
    }

    data.display = display;
    data.window = window;
    data.lastTime = initialTime;
    data.displayWidth = attr.width;
    data.displayHeight = attr.height;

    ImGuiIO &io = ImGui::GetIO();
    io.BackendPlatformName = "imgui_impl_xlib";
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    // Clipboard handlers (1.91+ moved this into ImGuiPlatformIO).
    ImGuiPlatformIO &platformIo = ImGui::GetPlatformIO();
    platformIo.Platform_GetClipboardTextFn = ImGuiGetClipboardCb;
    platformIo.Platform_SetClipboardTextFn = ImGuiSetClipboardCb;

    data.installed = true;
}

void
ImGui_ImplXlib_Shutdown()
{
    BackendData &data = Self();
    data = {};
}

void
ImGui_ImplXlib_NewFrame()
{
    BackendData &data = Self();
    if (!data.installed) {
        throw FractalSharkSeriousException("ImGui Xlib NewFrame called before initialization");
    }

    ImGuiIO &io = ImGui::GetIO();

    // Display size.  Updated lazily on ConfigureNotify; refresh here too in
    // case the host resized between events.
    XWindowAttributes attr{};
    if (XGetWindowAttributes(data.display, data.window, &attr) == 0) {
        throw FractalSharkSeriousException("XGetWindowAttributes failed while starting an ImGui frame");
    }
    data.displayWidth = attr.width;
    data.displayHeight = attr.height;
    io.DisplaySize = ImVec2(float(data.displayWidth), float(data.displayHeight));
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

    // Time delta.
    timespec now{};
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        const int errorCode = errno;
        throw FractalSharkSeriousException(std::string("clock_gettime failed: ") +
                                           std::generic_category().message(errorCode));
    }
    double dt = TimeSeconds(now) - TimeSeconds(data.lastTime);
    if (dt <= 0.0) {
        dt = 1.0 / 60.0;
    }
    io.DeltaTime = float(dt);
    data.lastTime = now;
}

void
ImGui_ImplXlib_SetClipboardHelper(FractalShark::Linux::LinuxClipboard *clipboard)
{
    if (!clipboard) {
        throw FractalSharkSeriousException("ImGui Xlib clipboard helper must not be null");
    }
    Self().clipboard = clipboard;
}

bool
ImGui_ImplXlib_ProcessEvent(const XEvent &ev)
{
    BackendData &data = Self();
    if (!data.installed) {
        throw FractalSharkSeriousException("ImGui Xlib event processed before initialization");
    }

    ImGuiIO &io = ImGui::GetIO();

    switch (ev.type) {
        case MotionNotify: {
            const auto &m = ev.xmotion;
            io.AddMousePosEvent(float(m.x), float(m.y));
            UpdateModifiers(m.state);
            return io.WantCaptureMouseUnlessPopupClose;
        }

        case ButtonPress:
        case ButtonRelease: {
            const auto &b = ev.xbutton;
            UpdateModifiers(b.state);
            const bool down = (ev.type == ButtonPress);
            switch (b.button) {
                case Button1:
                    io.AddMouseButtonEvent(0, down);
                    break;
                case Button2:
                    io.AddMouseButtonEvent(2, down);
                    break; // middle
                case Button3:
                    io.AddMouseButtonEvent(1, down);
                    break; // right
                case Button4:
                    if (down)
                        io.AddMouseWheelEvent(0.0f, 1.0f);
                    break;
                case Button5:
                    if (down)
                        io.AddMouseWheelEvent(0.0f, -1.0f);
                    break;
                case 6:
                    if (down)
                        io.AddMouseWheelEvent(-1.0f, 0.0f);
                    break; // hwheel
                case 7:
                    if (down)
                        io.AddMouseWheelEvent(1.0f, 0.0f);
                    break;
                default:
                    break;
            }
            // Wheel events use the same popup-close-aware capture policy as buttons.
            if (b.button == Button4 || b.button == Button5 || b.button == 6 || b.button == 7) {
                return io.WantCaptureMouseUnlessPopupClose;
            }
            return io.WantCaptureMouseUnlessPopupClose;
        }

        case KeyPress:
        case KeyRelease: {
            // Use a non-const local because XLookupString takes XKeyEvent*.
            XKeyEvent k = ev.xkey;
            UpdateModifiers(k.state);

            KeySym ks = NoSymbol;
            char buf[32]{};
            int len = XLookupString(&k, buf, sizeof(buf) - 1, &ks, nullptr);

            const ImGuiKey ikey = KeySymToImGuiKey(ks);
            if (ikey != ImGuiKey_None) {
                io.AddKeyEvent(ikey, ev.type == KeyPress);
            }

            if (ev.type == KeyPress && len > 0) {
                buf[len] = 0;
                // XLookupString returns Latin-1 / control chars; only forward
                // printable bytes to ImGui's text input.
                for (int i = 0; i < len; ++i) {
                    unsigned char c = static_cast<unsigned char>(buf[i]);
                    if (c >= 32 && c != 127) {
                        io.AddInputCharacter(c);
                    }
                }
            }
            return io.WantCaptureKeyboard;
        }

        case ConfigureNotify: {
            const auto &c = ev.xconfigure;
            data.displayWidth = c.width;
            data.displayHeight = c.height;
            return false;
        }

        case FocusIn:
            io.AddFocusEvent(true);
            return false;

        case FocusOut:
            io.AddFocusEvent(false);
            // Drop modifier state on focus loss to match Win32 behavior.
            io.AddKeyEvent(ImGuiMod_Ctrl, false);
            io.AddKeyEvent(ImGuiMod_Shift, false);
            io.AddKeyEvent(ImGuiMod_Alt, false);
            io.AddKeyEvent(ImGuiMod_Super, false);
            return false;

        case EnterNotify:
            io.AddMousePosEvent(float(ev.xcrossing.x), float(ev.xcrossing.y));
            return false;

        case LeaveNotify:
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
            return false;

        default:
            return false;
    }
}
