// LinuxImGuiOverlay.cpp

#include "LinuxImGuiOverlay.h"

#include "LinuxClipboard.h"
#include "LinuxXlibBackend.h"

#include "backends/imgui_impl_opengl2.h"
#include "imgui.h"

#include <GL/gl.h>

#include <algorithm>
#include <string>

using namespace FractalShark;

namespace FractalShark::Linux {

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
    const bool captured = ImGui_ImplXlib_ProcessEvent(ev);

    switch (ev.type) {
        case ButtonPress:
        case ButtonRelease:
        case MotionNotify:
        case KeyPress:
        case KeyRelease:
        case FocusIn:
        case FocusOut:
        case EnterNotify:
        case LeaveNotify:
            inputPending_ = true;
            break;
        default:
            break;
    }
    return captured;
}

void
ImGuiOverlay::RequestInfoModal(const char *title, const char *body)
{
    infoModalRequested_ = true;
    infoModalTitle_ = title ? title : "";
    infoModalBody_ = body ? body : "";
}

void
ImGuiOverlay::RequestSaveDialog(const char *title,
                                const std::string &defaultName,
                                SaveFilenameCallback cb)
{
    // TODO(linux-parity): Replace this filename-only modal with a browsable Linux file
    // dialog that supports open/save mode and filters. Decide whether last-directory
    // persistence and overwrite confirmation should be added to both GUIs.
    saveDlgRequested_ = true;
    saveDlgTitle_ = title ? title : "Save";
    saveDlgFilename_ = defaultName;
    saveDlgCallback_ = std::move(cb);
}

void
ImGuiOverlay::RequestPickFromList(const char *title,
                                  std::vector<std::string> items,
                                  PickFromListCallback cb)
{
    pickDlgRequested_ = true;
    pickDlgTitle_ = title ? title : "Pick";
    pickDlgItems_ = std::move(items);
    pickDlgSelected_ = -1;
    pickDlgCallback_ = std::move(cb);
}

void
ImGuiOverlay::RequestEnterLocation(std::string real,
                                   std::string imag,
                                   std::string zoom,
                                   uint64_t numIterations,
                                   EnterLocationCallback cb)
{
    locDlgRequested_ = true;
    locDlgReal_ = std::move(real);
    locDlgImag_ = std::move(imag);
    locDlgZoom_ = std::move(zoom);
    locDlgIters_ = std::to_string(numIterations);
    locDlgCallback_ = std::move(cb);
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

    if (infoModalRequested_) {
        ImGui::OpenPopup("FractalSharkInfoModal");
        infoModalRequested_ = false;
        infoModalOpen_ = true;
    }

    if (infoModalOpen_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        if (vp) {
            ImGui::SetNextWindowPos(
                ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(std::min(vp->WorkSize.x * 0.8f, 800.0f), 0.0f),
                                     ImGuiCond_Appearing);
        }
        const char *title = infoModalTitle_.empty() ? "Info" : infoModalTitle_.c_str();
        // Encode the popup-id name independently of the current title so a
        // RequestInfoModal call mid-display doesn't dismiss the popup before
        // the user closes it.
        if (ImGui::BeginPopupModal(
                "FractalSharkInfoModal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(title);
            ImGui::Separator();
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextUnformatted(infoModalBody_.c_str());
            ImGui::PopTextWrapPos();
            ImGui::Separator();
            if (ImGui::Button("Close", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
                infoModalOpen_ = false;
            }
            ImGui::EndPopup();
        } else {
            // Popup got dismissed externally (e.g. by clicking elsewhere).
            infoModalOpen_ = false;
        }
    }

    if (saveDlgRequested_) {
        ImGui::OpenPopup("FractalSharkSaveDialog");
        saveDlgRequested_ = false;
        saveDlgOpen_ = true;
    }

    if (saveDlgOpen_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        if (vp) {
            ImGui::SetNextWindowPos(
                ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(std::min(vp->WorkSize.x * 0.7f, 600.0f), 0.0f),
                                     ImGuiCond_Appearing);
        }
        if (ImGui::BeginPopupModal(
                "FractalSharkSaveDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted(saveDlgTitle_.c_str());
            ImGui::Separator();
            // Resize buffer if needed.
            if (saveDlgFilename_.size() < 512) {
                saveDlgFilename_.resize(512, '\0');
            }
            ImGui::SetNextItemWidth(-1.0f);
            const bool enter = ImGui::InputText("##filename",
                                                saveDlgFilename_.data(),
                                                saveDlgFilename_.size(),
                                                ImGuiInputTextFlags_EnterReturnsTrue);
            ImGui::Separator();
            const bool ok = enter || ImGui::Button("Save", ImVec2(120, 0));
            ImGui::SameLine();
            const bool cancel =
                ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape);
            if (ok) {
                std::string trimmed(saveDlgFilename_.c_str()); // strip trailing NULs
                ImGui::CloseCurrentPopup();
                saveDlgOpen_ = false;
                if (!trimmed.empty() && saveDlgCallback_) {
                    auto cb = std::move(saveDlgCallback_);
                    saveDlgCallback_ = nullptr;
                    cb(std::move(trimmed));
                }
            } else if (cancel) {
                ImGui::CloseCurrentPopup();
                saveDlgOpen_ = false;
                saveDlgCallback_ = nullptr;
            }
            ImGui::EndPopup();
        } else {
            saveDlgOpen_ = false;
            saveDlgCallback_ = nullptr;
        }
    }

    if (pickDlgRequested_) {
        ImGui::OpenPopup("FractalSharkPickDialog");
        pickDlgRequested_ = false;
        pickDlgOpen_ = true;
    }

    if (pickDlgOpen_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        if (vp) {
            ImGui::SetNextWindowPos(
                ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(
                ImVec2(std::min(vp->WorkSize.x * 0.7f, 700.0f), std::min(vp->WorkSize.y * 0.7f, 500.0f)),
                ImGuiCond_Appearing);
        }
        if (ImGui::BeginPopupModal("FractalSharkPickDialog", nullptr)) {
            ImGui::TextUnformatted(pickDlgTitle_.c_str());
            ImGui::Separator();
            if (pickDlgItems_.empty()) {
                ImGui::TextUnformatted("(no entries)");
            } else {
                ImGui::BeginChild(
                    "##items", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() - 4.0f), true);
                for (size_t i = 0; i < pickDlgItems_.size(); ++i) {
                    const bool isSel = (pickDlgSelected_ == int(i));
                    if (ImGui::Selectable(
                            pickDlgItems_[i].c_str(), isSel, ImGuiSelectableFlags_AllowDoubleClick)) {
                        pickDlgSelected_ = int(i);
                        if (ImGui::IsMouseDoubleClicked(0)) {
                            ImGui::CloseCurrentPopup();
                            pickDlgOpen_ = false;
                            if (pickDlgCallback_) {
                                auto cb = std::move(pickDlgCallback_);
                                pickDlgCallback_ = nullptr;
                                cb(size_t(i));
                            }
                            ImGui::EndChild();
                            ImGui::EndPopup();
                            goto pickDone;
                        }
                    }
                }
                ImGui::EndChild();
            }
            ImGui::Separator();
            const bool okEnabled = pickDlgSelected_ >= 0;
            if (!okEnabled)
                ImGui::BeginDisabled();
            const bool ok = ImGui::Button("OK", ImVec2(120, 0));
            if (!okEnabled)
                ImGui::EndDisabled();
            ImGui::SameLine();
            const bool cancel =
                ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape);
            if (ok && okEnabled) {
                ImGui::CloseCurrentPopup();
                pickDlgOpen_ = false;
                if (pickDlgCallback_) {
                    auto cb = std::move(pickDlgCallback_);
                    pickDlgCallback_ = nullptr;
                    cb(size_t(pickDlgSelected_));
                }
            } else if (cancel) {
                ImGui::CloseCurrentPopup();
                pickDlgOpen_ = false;
                pickDlgCallback_ = nullptr;
            }
            ImGui::EndPopup();
        } else {
            pickDlgOpen_ = false;
            pickDlgCallback_ = nullptr;
        }
    }
pickDone:

    if (locDlgRequested_) {
        ImGui::OpenPopup("FractalSharkLocationDialog");
        locDlgRequested_ = false;
        locDlgOpen_ = true;
    }

    if (locDlgOpen_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        if (vp) {
            ImGui::SetNextWindowPos(
                ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(std::min(vp->WorkSize.x * 0.8f, 900.0f), 0.0f),
                                     ImGuiCond_Appearing);
        }
        if (ImGui::BeginPopupModal(
                "FractalSharkLocationDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::TextUnformatted("Enter location:");
            ImGui::Separator();
            auto editField = [](const char *label, std::string &s) {
                if (s.size() < 4096)
                    s.resize(4096, '\0');
                ImGui::SetNextItemWidth(-1.0f);
                ImGui::InputText(label, s.data(), s.size());
            };
            editField("Real (X)##real", locDlgReal_);
            editField("Imag (Y)##imag", locDlgImag_);
            editField("Zoom##zoom", locDlgZoom_);
            editField("Iterations##iters", locDlgIters_);
            ImGui::Separator();
            const bool ok = ImGui::Button("OK", ImVec2(120, 0));
            ImGui::SameLine();
            const bool cancel =
                ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape);
            if (ok) {
                std::string r(locDlgReal_.c_str());
                std::string i(locDlgImag_.c_str());
                std::string z(locDlgZoom_.c_str());
                std::string it(locDlgIters_.c_str());
                ImGui::CloseCurrentPopup();
                locDlgOpen_ = false;
                if (!r.empty() && !i.empty() && !z.empty() && locDlgCallback_) {
                    uint64_t iters = 0;
                    try {
                        iters = std::stoull(it);
                    } catch (...) {
                        iters = 0;
                    }
                    auto cb = std::move(locDlgCallback_);
                    locDlgCallback_ = nullptr;
                    cb(std::move(r), std::move(i), std::move(z), iters);
                }
            } else if (cancel) {
                ImGui::CloseCurrentPopup();
                locDlgOpen_ = false;
                locDlgCallback_ = nullptr;
            }
            ImGui::EndPopup();
        } else {
            locDlgOpen_ = false;
            locDlgCallback_ = nullptr;
        }
    }

    if (dragRectActive_) {
        ImDrawList *fg = ImGui::GetForegroundDrawList();
        if (fg) {
            ImVec2 p0(float(std::min(dragX0_, dragX1_)), float(std::min(dragY0_, dragY1_)));
            ImVec2 p1(float(std::max(dragX0_, dragX1_)), float(std::max(dragY0_, dragY1_)));
            // TODO(linux-parity): Draw the documented thick white outer rectangle plus
            // thin black inner rectangle so the drag outline remains visible on any image.
            fg->AddRect(p0, p1, IM_COL32(255, 32, 255, 255), 0.0f, 0, 1.5f);
        }
    }

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    inputPending_ = false;
}

bool
ImGuiOverlay::WantsTick() const
{
    if (!ctx_) {
        return false;
    }
    return inputPending_ || dragRectActive_ || infoModalRequested_ || infoModalOpen_ ||
           saveDlgRequested_ || saveDlgOpen_ || pickDlgRequested_ || pickDlgOpen_ ||
           locDlgRequested_ || locDlgOpen_;
}

} // namespace FractalShark::Linux
