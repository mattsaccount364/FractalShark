// LinuxImGuiOverlay.cpp

#include "LinuxImGuiOverlay.h"

#include "LinuxClipboard.h"
#include "LinuxXlibBackend.h"

#include "CommandCatalog.h"
#include "MenuTree.h"

#include "imgui.h"
#include "backends/imgui_impl_opengl2.h"

#include <GL/gl.h>

#include <algorithm>
#include <string>

using namespace FractalShark;
using namespace FractalShark::Menu;

namespace FractalShark::Linux {

namespace {

std::string
WideToUtf8(std::wstring_view in)
{
    // The menu labels in MenuTreeDef.h are L"..." literals containing
    // ASCII characters plus a sprinkling of mathematical / unicode glyphs
    // (ellipses, ranges).  We hand-roll a UTF-16 → UTF-8 conversion
    // because <codecvt> is deprecated and we don't want a libicu dep.
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        uint32_t cp = static_cast<uint32_t>(in[i]);
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < in.size()) {
            uint32_t low = static_cast<uint32_t>(in[i + 1]);
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                ++i;
            }
        }
        if (cp < 0x80) {
            out.push_back(char(cp));
        } else if (cp < 0x800) {
            out.push_back(char(0xC0 | (cp >> 6)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(char(0xE0 | (cp >> 12)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(char(0xF0 | (cp >> 18)));
            out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

void
WalkNodes(std::span<const Node> nodes,
          const IMenuState *state,
          ExecuteCommandHost *host)
{
    for (const Node &n : nodes) {
        const bool enabled = state ? state->IsEnabled(n.enableRule) : true;
        const std::string label = WideToUtf8(n.text);

        switch (n.kind) {
        case Kind::Separator:
            ImGui::Separator();
            break;

        case Kind::Item: {
            bool selected = false;
            if (n.checkKind == CheckKind::Toggle && state) {
                selected = state->IsChecked(n.id);
            } else if (n.checkKind == CheckKind::Radio && state) {
                selected = (state->GetRadioSelection(n.radioGroup) == n.id);
            }
            // ImGui::MenuItem signature: (label, shortcut, selected, enabled).
            // We don't surface shortcut text here — the legacy-bridge
            // keyboard table is documented in PARITY.md and the catalog
            // doesn't carry a per-item shortcut string.
            if (ImGui::MenuItem(label.c_str(), nullptr, selected, enabled)) {
                if (host) {
                    ExecuteCommand(CommandFromIdm(n.id), *host);
                }
            }
            break;
        }

        case Kind::Popup: {
            // BeginMenu also takes an enabled flag.  Disabling a
            // submenu hides its children entirely, matching Win32
            // grayed-out submenu behavior.
            if (ImGui::BeginMenu(label.c_str(), enabled)) {
                WalkNodes(n.kids, state, host);
                ImGui::EndMenu();
            }
            break;
        }
        }
    }
}

void
BuildContextMenuContents(const IMenuState *state, ExecuteCommandHost *host)
{
    // MenuTreeDef.h defines a `static const Node menu[]` when included
    // with the FractalShark::Menu namespace's factories in scope.  Same
    // pattern as DynamicPopupMenu.cpp on the Win32 side — single source
    // of truth for the menu structure.
    using namespace FractalShark::Menu;

#include "MenuTreeDef.h"

    WalkNodes(std::span<const Node>{menu, sizeof(menu) / sizeof(menu[0])}, state, host);
}

} // namespace

ImGuiOverlay::ImGuiOverlay(Display *display, Window window, FractalShark::LinuxClipboard *clipboard)
    : display_(display), window_(window), clipboard_(clipboard)
{
    IMGUI_CHECKVERSION();
    ctx_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(ctx_);

    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Disable .ini persistence — we don't want the GUI scribbling state
    // files into the user's home directory without asking.
    io.IniFilename = nullptr;
}

ImGuiOverlay::~ImGuiOverlay()
{
    if (installedOn_) {
        installedOn_->SetOverlayCallback({});
        installedOn_ = nullptr;
    }
    if (ctx_) {
        // Backend shutdowns must happen on the consumer thread that
        // owns GL.  We can't call them safely here from arbitrary
        // contexts.  Leak the small GL objects on shutdown rather than
        // risk crashing — the process is terminating anyway.
        ImGui::DestroyContext(ctx_);
        ctx_ = nullptr;
    }
}

void
ImGuiOverlay::InstallCallback(OpenGlContext &gl)
{
    installedOn_ = &gl;
    gl.SetOverlayCallback([this]() noexcept { this->Render(); });
}

bool
ImGuiOverlay::QueueEvent(const XEvent &ev)
{
    {
        std::lock_guard lock(mu_);
        pendingEvents_.push_back(ev);
    }
    // Heuristic: if ImGui captured the previous frame, capture this one
    // too.  Authoritative consumption is computed by the consumer thread
    // when it actually feeds the event into ImGui, but the GUI thread
    // needs an immediate signal to know whether to short-circuit its own
    // dispatch.  Mouse-button events that hit a popup will reliably show
    // up as captured because the popup raised the flag on the previous
    // frame.
    switch (ev.type) {
    case ButtonPress:
    case ButtonRelease:
    case MotionNotify:
        return wantCaptureMouse_.load(std::memory_order_relaxed);
    case KeyPress:
    case KeyRelease:
        return wantCaptureKeyboard_.load(std::memory_order_relaxed);
    default:
        return false;
    }
}

void
ImGuiOverlay::SetExecuteHost(ExecuteCommandHost *host)
{
    std::lock_guard lock(mu_);
    host_ = host;
}

void
ImGuiOverlay::SetMenuState(const IMenuState *state)
{
    std::lock_guard lock(mu_);
    menuState_ = state;
}

void
ImGuiOverlay::RequestContextMenu(int x, int y)
{
    std::lock_guard lock(mu_);
    contextMenuRequested_ = true;
    contextMenuX_ = x;
    contextMenuY_ = y;
}

void
ImGuiOverlay::SetDragRect(bool active, int x0, int y0, int x1, int y1)
{
    std::lock_guard lock(mu_);
    dragRectActive_ = active;
    dragX0_ = x0;
    dragY0_ = y0;
    dragX1_ = x1;
    dragY1_ = y1;
}

void
ImGuiOverlay::DrainEvents()
{
    // mu_ already held by Render().
    for (const XEvent &ev : pendingEvents_) {
        ImGui_ImplXlib_ProcessEvent(ev);
    }
    pendingEvents_.clear();
}

void
ImGuiOverlay::Render() noexcept
{
    // GL context is current here (we run inside InvokeOverlayCallback,
    // which is invoked by the consumer just before SwapBuffers).

    std::lock_guard lock(mu_);

    if (!ctx_) {
        return;
    }
    ImGui::SetCurrentContext(ctx_);

    // Lazy-init the platform + renderer backends on first invocation.
    // Both need to happen with GL current.
    if (!xlibBackendInited_) {
        if (!ImGui_ImplXlib_Init(display_, window_)) {
            // Failed init — disable.
            xlibBackendInited_ = false;
            return;
        }
        if (clipboard_) {
            ImGui_ImplXlib_SetClipboardHelper(clipboard_);
        }
        xlibBackendInited_ = true;
    }
    if (!oglBackendInited_) {
        if (!ImGui_ImplOpenGL2_Init()) {
            return;
        }
        oglBackendInited_ = true;
    }

    DrainEvents();

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplXlib_NewFrame();
    ImGui::NewFrame();

    // ---- Build UI ----

    if (contextMenuRequested_) {
        ImGui::SetNextWindowPos(ImVec2(float(contextMenuX_), float(contextMenuY_)),
                                ImGuiCond_Appearing);
        ImGui::OpenPopup("FractalSharkContextMenu");
        contextMenuRequested_ = false;
    }

    if (ImGui::BeginPopup("FractalSharkContextMenu")) {
        BuildContextMenuContents(menuState_, host_);
        ImGui::EndPopup();
    }

    if (dragRectActive_) {
        ImDrawList *fg = ImGui::GetForegroundDrawList();
        if (fg) {
            ImVec2 p0(float(std::min(dragX0_, dragX1_)), float(std::min(dragY0_, dragY1_)));
            ImVec2 p1(float(std::max(dragX0_, dragX1_)), float(std::max(dragY0_, dragY1_)));
            // Bright magenta outline — distinctive against most fractal
            // colorings and matches the convention used by the Win32
            // path's InvertRect (which produces a similar high-contrast
            // outline against typical fractal backgrounds).
            fg->AddRect(p0, p1, IM_COL32(255, 32, 255, 255), 0.0f, 0, 1.5f);
        }
    }

    ImGui::Render();

    // Republish capture flags for the GUI thread's QueueEvent heuristic.
    const ImGuiIO &io = ImGui::GetIO();
    wantCaptureMouse_.store(io.WantCaptureMouse, std::memory_order_relaxed);
    wantCaptureKeyboard_.store(io.WantCaptureKeyboard, std::memory_order_relaxed);

    // Render to GL.  ImGui_ImplOpenGL2_RenderDrawData saves and restores
    // the relevant fixed-function pipeline state, so it doesn't disrupt
    // the surrounding fractal blit.
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

} // namespace FractalShark::Linux
