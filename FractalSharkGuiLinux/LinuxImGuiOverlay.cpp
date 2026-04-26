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
WalkNodes(std::span<const Node> nodes, const IMenuState *state, ExecuteCommandHost *host)
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
            if (ImGui::MenuItem(label.c_str(), nullptr, selected, enabled)) {
                if (host) {
                    ExecuteCommand(CommandFromIdm(n.id), *host);
                }
            }
            break;
        }

        case Kind::Popup: {
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
    io.IniFilename = nullptr;
}

ImGuiOverlay::~ImGuiOverlay()
{
    if (ctx_) {
        ImGui::SetCurrentContext(ctx_);
        if (oglBackendInited_) {
            ImGui_ImplOpenGL2_Shutdown();
            oglBackendInited_ = false;
        }
        if (xlibBackendInited_) {
            ImGui_ImplXlib_Shutdown();
            xlibBackendInited_ = false;
        }
        ImGui::DestroyContext(ctx_);
        ctx_ = nullptr;
    }
}

bool
ImGuiOverlay::Init()
{
    if (!ctx_) {
        return false;
    }
    ImGui::SetCurrentContext(ctx_);

    if (!xlibBackendInited_) {
        if (!ImGui_ImplXlib_Init(display_, window_)) {
            return false;
        }
        if (clipboard_) {
            ImGui_ImplXlib_SetClipboardHelper(clipboard_);
        }
        xlibBackendInited_ = true;
    }
    if (!oglBackendInited_) {
        if (!ImGui_ImplOpenGL2_Init()) {
            return false;
        }
        oglBackendInited_ = true;
    }
    return true;
}

bool
ImGuiOverlay::ProcessEvent(const XEvent &ev)
{
    if (!ctx_ || !xlibBackendInited_) {
        return false;
    }
    ImGui::SetCurrentContext(ctx_);
    ImGui_ImplXlib_ProcessEvent(ev);

    const ImGuiIO &io = ImGui::GetIO();
    switch (ev.type) {
    case ButtonPress:
    case ButtonRelease:
    case MotionNotify:
        return io.WantCaptureMouse;
    case KeyPress:
    case KeyRelease:
        return io.WantCaptureKeyboard;
    default:
        return false;
    }
}

void
ImGuiOverlay::SetExecuteHost(ExecuteCommandHost *host)
{
    host_ = host;
}

void
ImGuiOverlay::SetMenuState(const IMenuState *state)
{
    menuState_ = state;
}

void
ImGuiOverlay::RequestContextMenu(int x, int y)
{
    contextMenuRequested_ = true;
    contextMenuX_ = x;
    contextMenuY_ = y;
}

void
ImGuiOverlay::SetDragRect(bool active, int x0, int y0, int x1, int y1)
{
    dragRectActive_ = active;
    dragX0_ = x0;
    dragY0_ = y0;
    dragX1_ = x1;
    dragY1_ = y1;
}

void
ImGuiOverlay::RenderFrame()
{
    if (!ctx_ || !xlibBackendInited_ || !oglBackendInited_) {
        return;
    }
    ImGui::SetCurrentContext(ctx_);

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplXlib_NewFrame();
    ImGui::NewFrame();

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
            fg->AddRect(p0, p1, IM_COL32(255, 32, 255, 255), 0.0f, 0, 1.5f);
        }
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
}

bool
ImGuiOverlay::WantsTick() const
{
    if (!ctx_) {
        return false;
    }
    return dragRectActive_ || contextMenuRequested_
           || (ImGui::GetCurrentContext() && ImGui::IsPopupOpen("FractalSharkContextMenu"));
}

} // namespace FractalShark::Linux
