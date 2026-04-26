// FractalSharkGuiLinux — Xlib window skeleton.
//
// Phase: xlib-window-creation.  This stage brings up an empty X11 window with
// a working close button + keyboard auto-repeat tweak, runs an event loop, and
// exits cleanly on WM_DELETE_WINDOW.  No GLX context yet — that arrives with
// the fractal-render task, when OpenGlContext is wired in.

#include "CommandCatalog.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "LinuxClipboard.h"

#include <X11/XKBlib.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth = 1600;
constexpr int kInitialHeight = 1000;
constexpr const char *kWindowTitle = "FractalShark (Linux)";

// Mechanical-contract stub: every catalog command hook the Win32 GUI
// implements in MainWindow must also be overridden here.  Until the Linux
// GUI grows real menu/render plumbing each hook just announces itself so a
// developer wiring up a UI surface can see which commands fire.  A missing
// hook produces a compile error (pure virtual not implemented), which is
// exactly the signal we want from Phase 0c.
#define FRACTALSHARK_LINUX_STUB(method)                                                               \
    void method() override                                                                            \
    {                                                                                                 \
        std::fprintf(stderr, "TODO LinuxMainWindow: %s\n", #method);                                  \
    }

struct LinuxMainWindow : FractalShark::ExecuteCommandHost {
    Display *display = nullptr;
    Window window = 0;
    Atom wmDeleteWindow = 0;
    Atom wmProtocols = 0;
    int screen = 0;
    bool running = true;
    std::optional<FractalShark::LinuxClipboard> clipboard;

    LinuxMainWindow();
    ~LinuxMainWindow() override;

    LinuxMainWindow(const LinuxMainWindow &) = delete;
    LinuxMainWindow &operator=(const LinuxMainWindow &) = delete;

    bool Valid() const noexcept { return display != nullptr && window != 0; }
    void RunEventLoop();
    void HandleEvent(const XEvent &ev);

    // ---- ExecuteCommandHost stubs (one per virtual on the host) -----------
    void DispatchByIdm(int wmId) override
    {
        std::fprintf(stderr, "TODO LinuxMainWindow: DispatchByIdm(%d)\n", wmId);
    }

    void OnSetAlgorithm(::RenderAlgorithmEnum alg) override
    {
        std::fprintf(stderr,
                     "TODO LinuxMainWindow: OnSetAlgorithm(%u)\n",
                     static_cast<unsigned>(alg));
    }

    void OnSelectBuiltInView(size_t oneBasedIndex) override
    {
        std::fprintf(stderr,
                     "TODO LinuxMainWindow: OnSelectBuiltInView(%zu)\n",
                     oneBasedIndex);
    }

    FRACTALSHARK_LINUX_STUB(OnShowHotkeys)
    FRACTALSHARK_LINUX_STUB(OnViewsHelp)
    FRACTALSHARK_LINUX_STUB(OnHelpAlg)
    FRACTALSHARK_LINUX_STUB(OnSquareView)
    FRACTALSHARK_LINUX_STUB(OnRepainting)
    FRACTALSHARK_LINUX_STUB(OnWindowed)
    FRACTALSHARK_LINUX_STUB(OnWindowedSq)
    FRACTALSHARK_LINUX_STUB(OnMinimize)
    FRACTALSHARK_LINUX_STUB(OnCurPos)
    FRACTALSHARK_LINUX_STUB(OnExit)

    FRACTALSHARK_LINUX_STUB(OnBack)
    FRACTALSHARK_LINUX_STUB(OnCenterView)
    FRACTALSHARK_LINUX_STUB(OnZoomIn)
    FRACTALSHARK_LINUX_STUB(OnZoomOut)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomDefault)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomMax)
    FRACTALSHARK_LINUX_STUB(OnAutoZoomFilament)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderDirect)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderDirectScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderPt)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderPtScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderLa)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderLaScan)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderZoom)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderClear)
    FRACTALSHARK_LINUX_STUB(OnFeatureFinderResume)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopGpu)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopCpu)
    FRACTALSHARK_LINUX_STUB(OnNrInnerLoopCpuSt)

    FRACTALSHARK_LINUX_STUB(OnStandardView)

    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing1x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing4x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing9x)
    FRACTALSHARK_LINUX_STUB(OnGpuAntialiasing16x)

    FRACTALSHARK_LINUX_STUB(OnResetIterations)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations1p5x)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations6x)
    FRACTALSHARK_LINUX_STUB(OnIncreaseIterations24x)
    FRACTALSHARK_LINUX_STUB(OnDecreaseIterations)
    FRACTALSHARK_LINUX_STUB(OnIterations32Bit)
    FRACTALSHARK_LINUX_STUB(OnIterations64Bit)

    FRACTALSHARK_LINUX_STUB(OnIterationPrecision1x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision2x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision3x)
    FRACTALSHARK_LINUX_STUB(OnIterationPrecision4x)

    FRACTALSHARK_LINUX_STUB(OnPerturbResults)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearAll)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearMed)
    FRACTALSHARK_LINUX_STUB(OnPerturbClearHigh)
    FRACTALSHARK_LINUX_STUB(OnPerturbationAuto)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSinglethread)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSinglethreadPeriodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread2Periodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighStmed)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed1)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed2)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed3)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMt2PerturbMthighMtmed4)
    FRACTALSHARK_LINUX_STUB(OnPerturbationMultithread5Periodicity)
    FRACTALSHARK_LINUX_STUB(OnPerturbationGpu)
    FRACTALSHARK_LINUX_STUB(OnPerturbationLoad)
    FRACTALSHARK_LINUX_STUB(OnPerturbationSave)

    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOnDelete)
    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOn)
    FRACTALSHARK_LINUX_STUB(OnPerturbAutosaveOff)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit0)
    FRACTALSHARK_LINUX_STUB(OnMemoryLimit1)

    FRACTALSHARK_LINUX_STUB(OnPaletteType0)
    FRACTALSHARK_LINUX_STUB(OnPaletteType1)
    FRACTALSHARK_LINUX_STUB(OnPaletteType2)
    FRACTALSHARK_LINUX_STUB(OnPaletteType3)
    FRACTALSHARK_LINUX_STUB(OnPaletteType4)
    FRACTALSHARK_LINUX_STUB(OnCreateNewPalette)
    FRACTALSHARK_LINUX_STUB(OnPalette5)
    FRACTALSHARK_LINUX_STUB(OnPalette6)
    FRACTALSHARK_LINUX_STUB(OnPalette8)
    FRACTALSHARK_LINUX_STUB(OnPalette12)
    FRACTALSHARK_LINUX_STUB(OnPalette16)
    FRACTALSHARK_LINUX_STUB(OnPalette20)
    FRACTALSHARK_LINUX_STUB(OnPaletteRotate)

    FRACTALSHARK_LINUX_STUB(OnSaveLocation)
    FRACTALSHARK_LINUX_STUB(OnSaveHiResBmp)
    FRACTALSHARK_LINUX_STUB(OnSaveItersText)
    FRACTALSHARK_LINUX_STUB(OnSaveBmp)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitText)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitTextSimple)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitTextMax)
    FRACTALSHARK_LINUX_STUB(OnSaveRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnDiffRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnLoadLocation)
    FRACTALSHARK_LINUX_STUB(OnLoadEnterLocation)
    FRACTALSHARK_LINUX_STUB(OnLoadRefOrbitImagMax)
    FRACTALSHARK_LINUX_STUB(OnLoadRefOrbitImagMaxSaved)

    FRACTALSHARK_LINUX_STUB(OnBasicTest)
    FRACTALSHARK_LINUX_STUB(OnTest27)
    FRACTALSHARK_LINUX_STUB(OnBenchmarkFull)
    FRACTALSHARK_LINUX_STUB(OnBenchmarkInt)

    FRACTALSHARK_LINUX_STUB(OnLaMultithreaded)
    FRACTALSHARK_LINUX_STUB(OnLaSinglethreaded)
    FRACTALSHARK_LINUX_STUB(OnLaSettings1)
    FRACTALSHARK_LINUX_STUB(OnLaSettings2)
    FRACTALSHARK_LINUX_STUB(OnLaSettings3)
};

#undef FRACTALSHARK_LINUX_STUB

LinuxMainWindow::LinuxMainWindow()
{
    display = XOpenDisplay(nullptr);
    if (!display) {
        std::fprintf(stderr, "FractalSharkGuiLinux: XOpenDisplay failed (DISPLAY env not set?)\n");
        return;
    }

    screen = DefaultScreen(display);

    const unsigned long black = BlackPixel(display, screen);
    const unsigned long white = WhitePixel(display, screen);

    window = XCreateSimpleWindow(display,
                                 RootWindow(display, screen),
                                 0,
                                 0,
                                 kInitialWidth,
                                 kInitialHeight,
                                 0,
                                 white,
                                 black);

    XStoreName(display, window, kWindowTitle);

    // Class hint — lets WMs recognize and theme the app.
    XClassHint classHint{};
    std::string resName = "fractalshark";
    std::string resClass = "FractalShark";
    classHint.res_name = resName.data();
    classHint.res_class = resClass.data();
    XSetClassHint(display, window, &classHint);

    // Subscribe to events we'll need throughout the GUI (input + lifecycle).
    XSelectInput(display,
                 window,
                 ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask
                     | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask
                     | FocusChangeMask);

    // Receive WM_DELETE_WINDOW client messages instead of having the WM kill us.
    wmProtocols = XInternAtom(display, "WM_PROTOCOLS", False);
    wmDeleteWindow = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &wmDeleteWindow, 1);

    // Match Win32 KEYDOWN semantics: only autorepeat KeyPress, never KeyRelease.
    Bool supported = False;
    XkbSetDetectableAutoRepeat(display, True, &supported);

    XMapWindow(display, window);
    XFlush(display);

    clipboard.emplace(display, window);
}

LinuxMainWindow::~LinuxMainWindow()
{
    if (display) {
        if (window) {
            XDestroyWindow(display, window);
            window = 0;
        }
        XCloseDisplay(display);
        display = nullptr;
    }
}

void
LinuxMainWindow::HandleEvent(const XEvent &ev)
{
    switch (ev.type) {
    case ClientMessage: {
        const auto &cm = ev.xclient;
        if (cm.message_type == wmProtocols
            && static_cast<Atom>(cm.data.l[0]) == wmDeleteWindow) {
            running = false;
        }
        break;
    }

    case DestroyNotify:
        running = false;
        break;

    case Expose:
        // Until OpenGL is bound, we have nothing to draw.  Leave the window
        // black so the framework is visible without confusing flicker.
        break;

    default:
        break;
    }
}

void
LinuxMainWindow::RunEventLoop()
{
    XEvent ev;
    while (running) {
        XNextEvent(display, &ev);
        if (clipboard && clipboard->ProcessEvent(ev)) {
            continue;
        }
        HandleEvent(ev);
    }
}

} // namespace

int
main(int /*argc*/, char ** /*argv*/)
{
    Environment::RegisterHeapCleanup();
    CrashHandler::Install();

    LinuxMainWindow win;
    if (!win.Valid()) {
        return 1;
    }
    win.RunEventLoop();
    return 0;
}
