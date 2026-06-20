// LinuxImGuiOverlay.h
//
// Single-threaded ImGui overlay for the Linux GUI.  All ImGui state and
// every ImGui call happens on the GUI thread (the same thread that runs
// the X event loop and owns the GLX context).
//
// Lifecycle:
//   1. Construct on the GUI thread.  The ImGui context is created here,
//      but the platform/renderer backends are NOT initialized yet (GL
//      may not be current at construction time).
//   2. Call Init() once after the GL context has been made current.
//      Throws on backend init failure.
//   3. Call ProcessEvent for every XEvent before dispatching it to the
//      app's own handler.  Returns true if ImGui consumed it (caller
//      should drop the event).
//   4. Call RenderFrame() once per main-loop tick (after fractal frame
//      is presented to GL, before SwapBuffers).
//   5. Destroy on the GUI thread before tearing down the GL context.
//
// No mutexes, no event queue, no atomics — everything runs serially on
// one thread.

#pragma once

#include <X11/Xlib.h>

struct ImGuiContext;

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace FractalShark::Linux {

struct LinuxClipboard;

class ImGuiOverlay {
public:
    ImGuiOverlay(Display *display, Window window, LinuxClipboard *clipboard);
    ~ImGuiOverlay();

    ImGuiOverlay(const ImGuiOverlay &) = delete;
    ImGuiOverlay &operator=(const ImGuiOverlay &) = delete;

    // Initialize the platform + renderer backends.  Must be called once,
    // on the GUI thread, after GL context is current.  Throws on backend
    // init failure.
    void Init();

    // Forward an XEvent to ImGui.  Returns true when ImGui captured it
    // (a popup is up and the cursor / keyboard input is over the popup
    // area), in which case the caller should not run its own dispatch
    // for this event.
    bool ProcessEvent(const XEvent &ev);

    // Open a modal info window with the given title + body on the next
    // RenderFrame() call.  Body is rendered as wrapped text with a single
    // "Close" button; this is the Linux equivalent of MessageBox(MB_OK).
    // If a previous modal is still open, the new request supersedes it.
    void RequestInfoModal(const char *title, const char *body);

    enum class FileDialogMode {
        Open,
        Save,
    };

    struct FileDialogFilter {
        std::string Label;
        std::string Extension;
    };

    // Open an in-app file dialog on the next RenderFrame.  Callback is
    // invoked on OK with the selected UTF-8 path; not invoked on Cancel.
    using FileDialogCallback = std::function<void(std::string)>;
    void RequestFileDialog(const char *title,
                           FileDialogMode mode,
                           const std::string &defaultName,
                           std::vector<FileDialogFilter> filters,
                           FileDialogCallback cb);

    // Open a "pick one from a list" modal on the next RenderFrame.  Each
    // entry in `items` is a display string.  Callback is invoked on
    // selection with the chosen index; not invoked on Cancel.
    using PickFromListCallback = std::function<void(size_t)>;
    void RequestPickFromList(const char *title, std::vector<std::string> items, PickFromListCallback cb);

    // Open a "enter location" modal on the next RenderFrame: 4 text
    // inputs (real, imag, zoom, iterations) prefilled with the given
    // values, plus OK/Cancel.  Callback invoked on OK with the four
    // edited values; not invoked on Cancel or when any of the first
    // three strings is empty.
    using EnterLocationCallback = std::function<void(
        std::string real, std::string imag, std::string zoom, uint64_t numIterations)>;
    void RequestEnterLocation(std::string real,
                              std::string imag,
                              std::string zoom,
                              uint64_t numIterations,
                              EnterLocationCallback cb);

    // Configure the drag-zoom rubber band.  active=false hides it.
    void SetDragRect(bool active, int x0, int y0, int x1, int y1);

    // Run NewFrame, build the UI (popup + drag rect), Render, and
    // RenderDrawData.  Caller is responsible for SwapBuffers().
    void RenderFrame();

    // True when ImGui state implies the loop should keep ticking even
    // without external input — e.g. a popup is open and animations
    // need to advance.  Used by the host's poll() loop to choose a
    // tighter timeout.
    bool WantsTick() const;

private:
    struct FileDialogEntry {
        std::string Name;
        std::filesystem::path Path;
        bool IsDirectory = false;
    };

    bool SetFileDialogDirectory(const std::filesystem::path &path);
    void RefreshFileDialogEntries();
    std::optional<std::string> ResolveFileDialogSelection();

    Display *display_;
    Window window_;
    LinuxClipboard *clipboard_;
    ImGuiContext *ctx_ = nullptr;
    bool xlibBackendInited_ = false;
    bool oglBackendInited_ = false;

    bool inputPending_ = false;

    bool infoModalRequested_ = false;
    bool infoModalOpen_ = false;
    std::string infoModalTitle_;
    std::string infoModalBody_;

    // Open/save file dialog state.
    bool fileDlgRequested_ = false;
    bool fileDlgOpen_ = false;
    FileDialogMode fileDlgMode_ = FileDialogMode::Open;
    std::string fileDlgTitle_;
    std::filesystem::path fileDlgCurrentDir_;
    std::string fileDlgDirBuffer_;
    std::string fileDlgFilenameBuffer_;
    std::vector<FileDialogFilter> fileDlgFilters_;
    int fileDlgFilterIndex_ = 0;
    std::vector<FileDialogEntry> fileDlgEntries_;
    int fileDlgSelected_ = -1;
    std::string fileDlgMessage_;
    FileDialogCallback fileDlgCallback_;

    // Pick-from-list dialog state.
    bool pickDlgRequested_ = false;
    bool pickDlgOpen_ = false;
    std::string pickDlgTitle_;
    std::vector<std::string> pickDlgItems_;
    int pickDlgSelected_ = -1;
    PickFromListCallback pickDlgCallback_;

    // Enter-location dialog state.
    bool locDlgRequested_ = false;
    bool locDlgOpen_ = false;
    std::string locDlgReal_;
    std::string locDlgImag_;
    std::string locDlgZoom_;
    std::string locDlgIters_;
    std::string locDlgMessage_;
    EnterLocationCallback locDlgCallback_;

    bool dragRectActive_ = false;
    int dragX0_ = 0;
    int dragY0_ = 0;
    int dragX1_ = 0;
    int dragY1_ = 0;
};

} // namespace FractalShark::Linux
