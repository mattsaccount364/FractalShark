// LinuxImGuiOverlay.cpp

#include "LinuxImGuiOverlay.h"

#include "Exceptions.h"
#include "LinuxClipboard.h"
#include "LinuxXlibBackend.h"
#include "MenuTree.h"
#include "PortableCommandHandlers.h"

#include "backends/imgui_impl_opengl2.h"
#include "imgui.h"

#include <GL/gl.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace FractalShark::Linux {
namespace {

constexpr size_t kFileDialogInputCapacity = 4096;

std::string
InputBufferValue(const std::string &buffer)
{
    return std::string(buffer.c_str());
}

void
AssignInputBuffer(std::string &buffer, const std::string &value)
{
    // ImGui edits std::string storage in place and needs stable spare capacity.  The logical value
    // remains the prefix before the first NUL, as recovered by InputBufferValue().
    buffer = value;
    if (buffer.size() + 1 < kFileDialogInputCapacity) {
        buffer.resize(kFileDialogInputCapacity, '\0');
    } else {
        buffer.push_back('\0');
    }
}

std::string
PathToString(const std::filesystem::path &path)
{
    return path.string();
}

std::string
PathNameToString(const std::filesystem::path &path)
{
    std::string name = PathToString(path.filename());
    if (name.empty()) {
        name = PathToString(path);
    }
    return name;
}

std::string
LowerAscii(std::string value)
{
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::span<const Node>
GetMenuNodes()
{
#include "MenuTreeDef.h"

    return {menu, sizeof(menu) / sizeof(menu[0])};
}

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

std::string
StripMenuMnemonics(std::string label)
{
    std::string out;
    out.reserve(label.size());
    for (std::size_t i = 0; i < label.size(); ++i) {
        if (label[i] == '&' && i + 1 < label.size()) {
            if (label[i + 1] == '&') {
                out.push_back('&');
                ++i;
            }
            continue;
        }
        out.push_back(label[i]);
    }
    return out;
}

std::string
MenuLabel(const Node &node, const IMenuState &state)
{
    std::wstring label(node.text);
    if (node.kind == Kind::Popup && node.adornGroup != RadioGroup::None) {
        uint32_t adornId = state.GetPopupAdornmentCommandId(node.adornGroup);
        if (adornId == 0) {
            adornId = state.GetRadioSelection(node.adornGroup);
        }
        if (adornId != 0) {
            const std::wstring_view selection = state.GetCommandLabel(adornId);
            if (!selection.empty()) {
                label += L" (";
                label += selection;
                label += L")";
            }
        }
    }
    return StripMenuMnemonics(WideToUtf8(label));
}

bool
IsNodeChecked(const Node &node, const IMenuState &state)
{
    if (node.checkKind == CheckKind::Toggle) {
        return state.IsChecked(node.id);
    }
    if (node.checkKind == CheckKind::Radio) {
        return state.GetRadioSelection(node.radioGroup) == node.id;
    }
    return false;
}

bool
RenderContextMenuNodes(std::span<const Node> nodes, const IMenuState &state, FractalCommand &command)
{
    for (const Node &node : nodes) {
        ImGui::PushID(&node);

        bool selected = false;
        switch (node.kind) {
            case Kind::Separator:
                ImGui::Separator();
                break;

            case Kind::Popup: {
                const std::string label = MenuLabel(node, state);
                if (ImGui::BeginMenu(label.c_str(), state.IsEnabled(node.enableRule))) {
                    selected = RenderContextMenuNodes(node.kids, state, command);
                    ImGui::EndMenu();
                }
                break;
            }

            case Kind::Item: {
                const std::string label = MenuLabel(node, state);
                const bool enabled = state.IsEnabled(node.enableRule);
                const bool checked = IsNodeChecked(node, state);
                if (ImGui::MenuItem(label.c_str(), nullptr, checked, enabled)) {
                    command = CommandFromIdm(node.id);
                    ImGui::CloseCurrentPopup();
                    selected = true;
                }
                break;
            }
        }

        ImGui::PopID();
        if (selected) {
            return true;
        }
    }
    return false;
}

} // namespace

ImGuiOverlay::ImGuiOverlay(Display *display,
                           Window window,
                           LinuxClipboard *clipboard,
                           const IMenuState *menuState,
                           PortableCommandHandlers *commandHandlers)
    : display_(display), window_(window), clipboard_(clipboard), menuState_(menuState),
      commandHandlers_(commandHandlers)
{
    if (!display_ || !window_ || !menuState_ || !commandHandlers_) {
        throw FractalSharkSeriousException(
            "ImGuiOverlay requires a valid X display, window, menu state, and command handlers");
    }
    IMGUI_CHECKVERSION();
    ctx_ = ImGui::CreateContext();
    if (!ctx_) {
        throw FractalSharkSeriousException("ImGui::CreateContext failed");
    }
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

void
ImGuiOverlay::Init()
{
    if (!ctx_) {
        throw FractalSharkSeriousException("ImGui overlay has no context");
    }
    ImGui::SetCurrentContext(ctx_);

    if (!xlibBackendInited_) {
        ImGui_ImplXlib_Init(display_, window_);
        if (clipboard_) {
            ImGui_ImplXlib_SetClipboardHelper(clipboard_);
        }
        xlibBackendInited_ = true;
    }
    if (!oglBackendInited_) {
        if (!ImGui_ImplOpenGL2_Init()) {
            throw FractalSharkSeriousException("ImGui OpenGL backend initialization failed");
        }
        oglBackendInited_ = true;
    }
}

bool
ImGuiOverlay::ProcessEvent(const XEvent &ev)
{
    if (!ctx_ || !xlibBackendInited_) {
        throw FractalSharkSeriousException("ImGui overlay processed an event before initialization");
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
    switch (ev.type) {
        case ButtonPress:
        case ButtonRelease:
        case MotionNotify:
        case KeyPress:
        case KeyRelease:
            return captured || contextMenuRequested_ || contextMenuOpen_;
        default:
            return captured;
    }
}

void
ImGuiOverlay::RequestInfoModal(const char *title, const char *body)
{
    infoModalRequested_ = true;
    infoModalTitle_ = title ? title : "";
    infoModalBody_ = body ? body : "";
}

void
ImGuiOverlay::RequestContextMenu(int x, int y)
{
    contextMenuRequested_ = true;
    contextMenuApplyPosition_ = true;
    contextMenuX_ = x;
    contextMenuY_ = y;
}

void
ImGuiOverlay::RequestFileDialog(const char *title,
                                FileDialogMode mode,
                                const std::string &defaultName,
                                std::vector<FileDialogFilter> filters,
                                FileDialogCallback cb)
{
    fileDlgRequested_ = true;
    fileDlgMode_ = mode;
    fileDlgTitle_ = title ? title : (mode == FileDialogMode::Open ? "Open" : "Save");
    fileDlgFilters_ = std::move(filters);
    if (fileDlgFilters_.empty()) {
        fileDlgFilters_.push_back({"All (*.*)", ""});
    }
    fileDlgFilterIndex_ = 0;
    fileDlgSelected_ = -1;
    fileDlgMessage_.clear();
    fileDlgCallback_ = std::move(cb);

    std::filesystem::path defaultPath(defaultName);
    std::filesystem::path initialDir = std::filesystem::current_path();

    std::string filename = defaultName;
    if (defaultPath.has_parent_path()) {
        initialDir = defaultPath.parent_path();
        filename = PathNameToString(defaultPath);
    }

    AssignInputBuffer(fileDlgFilenameBuffer_, filename);
    if (!SetFileDialogDirectory(initialDir)) {
        throw FractalSharkSeriousException("Could not initialize the file-dialog directory");
    }
}

bool
ImGuiOverlay::SetFileDialogDirectory(const std::filesystem::path &path)
{
    std::error_code ec;
    std::filesystem::path dir = path.empty() ? std::filesystem::path(".") : path;
    if (dir.is_relative()) {
        dir = std::filesystem::absolute(dir, ec);
        if (ec) {
            fileDlgMessage_ = "Could not resolve directory.";
            return false;
        }
    }

    const auto canonical = std::filesystem::weakly_canonical(dir, ec);
    if (!ec && !canonical.empty()) {
        dir = canonical;
    } else {
        dir = dir.lexically_normal();
        ec.clear();
    }

    if (!std::filesystem::is_directory(dir, ec) || ec) {
        fileDlgMessage_ = "Selected path is not a directory.";
        return false;
    }

    fileDlgCurrentDir_ = dir;
    AssignInputBuffer(fileDlgDirBuffer_, PathToString(fileDlgCurrentDir_));
    fileDlgSelected_ = -1;
    RefreshFileDialogEntries();
    return true;
}

void
ImGuiOverlay::RefreshFileDialogEntries()
{
    fileDlgEntries_.clear();
    fileDlgSelected_ = -1;
    fileDlgMessage_.clear();

    const std::string wantedExtension =
        fileDlgFilterIndex_ >= 0 && fileDlgFilterIndex_ < static_cast<int>(fileDlgFilters_.size())
            ? LowerAscii(fileDlgFilters_[static_cast<size_t>(fileDlgFilterIndex_)].Extension)
            : std::string();

    std::error_code ec;
    std::filesystem::directory_iterator it(fileDlgCurrentDir_, ec);
    if (ec) {
        fileDlgMessage_ = "Could not read directory.";
        return;
    }

    for (std::filesystem::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) {
            fileDlgMessage_ = "Could not read all directory entries.";
            break;
        }

        const std::filesystem::path entryPath = it->path();
        std::error_code typeEc;
        const bool isDirectory = it->is_directory(typeEc);
        if (typeEc) {
            continue;
        }
        const bool isRegularFile = it->is_regular_file(typeEc);
        if (typeEc || (!isDirectory && !isRegularFile)) {
            continue;
        }
        if (!isDirectory && !wantedExtension.empty() &&
            LowerAscii(PathToString(entryPath.extension())) != wantedExtension) {
            continue;
        }

        std::string name = PathNameToString(entryPath);
        if (!name.empty()) {
            fileDlgEntries_.push_back({std::move(name), entryPath, isDirectory});
        }
    }

    std::sort(fileDlgEntries_.begin(),
              fileDlgEntries_.end(),
              [](const FileDialogEntry &a, const FileDialogEntry &b) {
                  if (a.IsDirectory != b.IsDirectory) {
                      return a.IsDirectory;
                  }
                  return LowerAscii(a.Name) < LowerAscii(b.Name);
              });
}

std::optional<std::string>
ImGuiOverlay::ResolveFileDialogSelection()
{
    const std::string filename = InputBufferValue(fileDlgFilenameBuffer_);
    if (filename.empty()) {
        fileDlgMessage_ = "Choose a file.";
        return std::nullopt;
    }

    std::filesystem::path selected(filename);
    if (selected.is_relative()) {
        selected = fileDlgCurrentDir_ / selected;
    }
    selected = selected.lexically_normal();

    std::error_code ec;
    if (fileDlgMode_ == FileDialogMode::Open) {
        if (!std::filesystem::is_regular_file(selected, ec) || ec) {
            fileDlgMessage_ = "Selected path is not an existing file.";
            return std::nullopt;
        }
    } else {
        const auto parent = selected.parent_path();
        if (!parent.empty() && (!std::filesystem::is_directory(parent, ec) || ec)) {
            fileDlgMessage_ = "Destination directory does not exist.";
            return std::nullopt;
        }
    }

    return PathToString(selected);
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
    locDlgMessage_.clear();
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
        throw FractalSharkSeriousException("ImGui overlay rendered before initialization");
    }
    ImGui::SetCurrentContext(ctx_);

    ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplXlib_NewFrame();
    ImGui::NewFrame();

    RenderInfoModal();
    RenderFileDialog();
    RenderPickDialog();
    RenderEnterLocationDialog();
    RenderContextMenu();
    RenderDragRect();

    ImGui::Render();
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
    inputPending_ = false;
}

void
ImGuiOverlay::RenderInfoModal()
{
    // Requests may arrive between frames.  Split requested/open state so OpenPopup is issued once,
    // while the event loop continues rendering until ImGui reports that the modal has closed.
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
}

void
ImGuiOverlay::RenderFileDialog()
{
    if (fileDlgRequested_) {
        ImGui::OpenPopup("FractalSharkFileDialog");
        fileDlgRequested_ = false;
        fileDlgOpen_ = true;
    }

    if (fileDlgOpen_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        if (vp) {
            ImGui::SetNextWindowPos(
                ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                ImGuiCond_Appearing,
                ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(std::min(vp->WorkSize.x * 0.85f, 900.0f),
                                            std::min(vp->WorkSize.y * 0.85f, 650.0f)),
                                     ImGuiCond_Appearing);
        }
        if (ImGui::BeginPopupModal("FractalSharkFileDialog", nullptr)) {
            ImGui::TextUnformatted(fileDlgTitle_.c_str());
            ImGui::Separator();

            const ImGuiStyle &style = ImGui::GetStyle();
            constexpr float kMinDirectoryInputWidth = 160.0f;
            const float goWidth =
                std::max(48.0f, ImGui::CalcTextSize("Go").x + style.FramePadding.x * 2.0f);
            const float refreshWidth =
                std::max(80.0f, ImGui::CalcTextSize("Refresh").x + style.FramePadding.x * 2.0f);
            const float itemSpacing = style.ItemSpacing.x;
            const float headerAvail = ImGui::GetContentRegionAvail().x;
            const bool allHeaderControlsFit =
                headerAvail >= kMinDirectoryInputWidth + goWidth + refreshWidth + itemSpacing * 2.0f;
            const bool goFitsWithInput = headerAvail >= kMinDirectoryInputWidth + goWidth + itemSpacing;
            const bool buttonsFitTogether = headerAvail >= goWidth + refreshWidth + itemSpacing;
            const float directoryInputWidth =
                allHeaderControlsFit
                    ? std::max(kMinDirectoryInputWidth,
                               headerAvail - goWidth - refreshWidth - itemSpacing * 2.0f)
                : goFitsWithInput
                    ? std::max(kMinDirectoryInputWidth, headerAvail - goWidth - itemSpacing)
                    : headerAvail;

            ImGui::SetNextItemWidth(directoryInputWidth);
            const bool dirEnter = ImGui::InputText("##directory",
                                                   fileDlgDirBuffer_.data(),
                                                   fileDlgDirBuffer_.size(),
                                                   ImGuiInputTextFlags_EnterReturnsTrue);
            if (goFitsWithInput) {
                ImGui::SameLine(0.0f, itemSpacing);
            }
            const bool goDir = ImGui::Button("Go", ImVec2(goWidth, 0));
            if (allHeaderControlsFit || (!goFitsWithInput && buttonsFitTogether)) {
                ImGui::SameLine(0.0f, itemSpacing);
            }
            const bool refresh = ImGui::Button("Refresh", ImVec2(refreshWidth, 0));
            if (dirEnter || goDir) {
                (void)SetFileDialogDirectory(std::filesystem::path(InputBufferValue(fileDlgDirBuffer_)));
            } else if (refresh) {
                RefreshFileDialogEntries();
            }

            const auto parent = fileDlgCurrentDir_.parent_path();
            if (ImGui::Button("Up", ImVec2(80, 0)) && !parent.empty() && parent != fileDlgCurrentDir_) {
                (void)SetFileDialogDirectory(parent);
            }
            if (!fileDlgFilters_.empty()) {
                std::vector<const char *> labels;
                labels.reserve(fileDlgFilters_.size());
                for (const auto &filter : fileDlgFilters_) {
                    labels.push_back(filter.Label.c_str());
                }
                ImGui::SameLine();
                ImGui::SetNextItemWidth(220.0f);
                if (ImGui::Combo("##filter", &fileDlgFilterIndex_, labels.data(), int(labels.size()))) {
                    RefreshFileDialogEntries();
                }
            }

            std::optional<std::filesystem::path> enterDirectory;
            std::optional<std::string> acceptedPath;
            ImGui::BeginChild("##files", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 3.0f), true);
            for (size_t i = 0; i < fileDlgEntries_.size(); ++i) {
                const auto &entry = fileDlgEntries_[i];
                std::string label = entry.IsDirectory ? "[dir] " + entry.Name : entry.Name;
                label += "##file";
                label += std::to_string(i);
                const bool selected = fileDlgSelected_ == static_cast<int>(i);
                if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
                    fileDlgSelected_ = static_cast<int>(i);
                    if (entry.IsDirectory) {
                        if (ImGui::IsMouseDoubleClicked(0)) {
                            enterDirectory = entry.Path;
                            break;
                        }
                    } else {
                        AssignInputBuffer(fileDlgFilenameBuffer_, entry.Name);
                        if (fileDlgMode_ == FileDialogMode::Open && ImGui::IsMouseDoubleClicked(0)) {
                            acceptedPath = ResolveFileDialogSelection();
                            break;
                        }
                    }
                }
            }
            ImGui::EndChild();

            if (enterDirectory.has_value()) {
                (void)SetFileDialogDirectory(*enterDirectory);
            }

            ImGui::SetNextItemWidth(-1.0f);
            const bool filenameEnter = ImGui::InputText("##filename",
                                                        fileDlgFilenameBuffer_.data(),
                                                        fileDlgFilenameBuffer_.size(),
                                                        ImGuiInputTextFlags_EnterReturnsTrue);
            if (!fileDlgMessage_.empty()) {
                ImGui::TextWrapped("%s", fileDlgMessage_.c_str());
            }
            ImGui::Separator();
            const char *okLabel = fileDlgMode_ == FileDialogMode::Open ? "Open" : "Save";
            if (!acceptedPath.has_value() && (filenameEnter || ImGui::Button(okLabel, ImVec2(120, 0)))) {
                acceptedPath = ResolveFileDialogSelection();
            }
            ImGui::SameLine();
            const bool cancel =
                ImGui::Button("Cancel", ImVec2(120, 0)) || ImGui::IsKeyPressed(ImGuiKey_Escape);

            if (acceptedPath.has_value()) {
                ImGui::CloseCurrentPopup();
                fileDlgOpen_ = false;
                if (fileDlgCallback_) {
                    auto cb = std::move(fileDlgCallback_);
                    fileDlgCallback_ = nullptr;
                    cb(std::move(*acceptedPath));
                }
            } else if (cancel) {
                ImGui::CloseCurrentPopup();
                fileDlgOpen_ = false;
                fileDlgCallback_ = nullptr;
            }
            ImGui::EndPopup();
        } else {
            fileDlgOpen_ = false;
            fileDlgCallback_ = nullptr;
        }
    }
}

void
ImGuiOverlay::RenderPickDialog()
{
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
                                // Clear stored ownership before invoking client code: callbacks may
                                // immediately request another overlay dialog reentrantly.
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
}

void
ImGuiOverlay::RenderEnterLocationDialog()
{
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
            if (!locDlgMessage_.empty()) {
                ImGui::TextWrapped("%s", locDlgMessage_.c_str());
            }
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
                if (!r.empty() && !i.empty() && !z.empty() && locDlgCallback_) {
                    uint64_t iters = 0;
                    const auto [end, error] = std::from_chars(it.data(), it.data() + it.size(), iters);
                    if (error != std::errc{} || end != it.data() + it.size()) {
                        locDlgMessage_ = "Iterations must be a non-negative integer.";
                    } else {
                        ImGui::CloseCurrentPopup();
                        locDlgOpen_ = false;
                        locDlgMessage_.clear();
                        auto cb = std::move(locDlgCallback_);
                        locDlgCallback_ = nullptr;
                        cb(std::move(r), std::move(i), std::move(z), iters);
                    }
                } else {
                    locDlgMessage_ = "Real, imaginary, and zoom values are required.";
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
}

void
ImGuiOverlay::RenderContextMenu()
{
    FractalCommand contextCommand = FractalCommand::None;
    if (contextMenuRequested_) {
        ImGui::OpenPopup("FractalSharkContextMenu");
        contextMenuRequested_ = false;
        contextMenuOpen_ = true;
    }

    if (contextMenuOpen_) {
        if (contextMenuApplyPosition_) {
            ImGui::SetNextWindowPos(ImVec2(float(contextMenuX_), float(contextMenuY_)),
                                    ImGuiCond_Always);
            contextMenuApplyPosition_ = false;
        }
        if (ImGui::BeginPopup("FractalSharkContextMenu")) {
            (void)RenderContextMenuNodes(GetMenuNodes(), *menuState_, contextCommand);
            ImGui::EndPopup();
        } else {
            contextMenuOpen_ = false;
        }
    }

    if (contextCommand != FractalCommand::None) {
        contextMenuOpen_ = false;
        if (!commandHandlers_) {
            throw FractalSharkSeriousException("Context-menu command handlers are not initialized");
        }
        commandHandlers_->ExecuteCommand(contextCommand);
    }
}

void
ImGuiOverlay::RenderDragRect()
{
    if (dragRectActive_) {
        ImDrawList *fg = ImGui::GetForegroundDrawList();
        if (fg) {
            ImVec2 p0(float(std::min(dragX0_, dragX1_)), float(std::min(dragY0_, dragY1_)));
            ImVec2 p1(float(std::max(dragX0_, dragX1_)), float(std::max(dragY0_, dragY1_)));
            fg->AddRect(p0, p1, IM_COL32(255, 255, 255, 255), 0.0f, 0, 3.0f);
            fg->AddRect(p0, p1, IM_COL32(0, 0, 0, 255), 0.0f, 0, 1.0f);
        }
    }
}

bool
ImGuiOverlay::WantsTick() const
{
    if (!ctx_) {
        throw FractalSharkSeriousException("ImGui overlay has no context");
    }
    // The X event loop otherwise sleeps until render-pool work arrives.  Keep presentation ticking
    // while an overlay can animate, consume input, or transition a queued request into an open popup.
    return inputPending_ || dragRectActive_ || contextMenuRequested_ || contextMenuOpen_ ||
           infoModalRequested_ || infoModalOpen_ || fileDlgRequested_ || fileDlgOpen_ ||
           pickDlgRequested_ || pickDlgOpen_ || locDlgRequested_ || locDlgOpen_;
}

} // namespace FractalShark::Linux
