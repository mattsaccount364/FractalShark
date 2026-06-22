#pragma once

#include <memory>
#include <string>
#include <vector>

#include "CommandDispatcher.h"
#include "PortableCommandHandlers.h"
#include "SavedLocation.h"
#include "SplashWindow.h"
#include "UniqueHMenu.h"

class Fractal;

namespace Environment {
class JobObject;
} // namespace Environment

enum FractalPaletteType : size_t;
enum class CompressToDisk;
enum class ImaginaSettings;

namespace FractalShark::Win32 {

class MainWindow : public PortableCommandHandlers {
    friend class CommandDispatcher;

public:
    struct ImaginaSavedLocation {
        std::wstring Filename;
        ImaginaSettings Settings;
    };

    MainWindow(HINSTANCE hInstance, int nCmdShow);
    ~MainWindow();

    // ---- Platform-specific command handlers ----------------------------

    // Synthetic shortcut command hooks
    void OnAutoZoomFeatureAtPoint() override;

    // Help / Window
    void OnShowHotkeys() override;
    void OnViewsHelp() override;
    void OnHelpAlg() override;
    void OnWindowed() override;
    void OnWindowedSq() override;
    void OnMinimize() override;
    void OnCurPos() override;
    void OnExit() override;

    // Navigate

    // Built-In Views

    // Antialiasing

    // Iterations

    // Iteration precision

    // Perturbation

    // Memory / Autosave

    // Palette
    void OnPaletteRotate() override;

    // Save / Load
    void OnSaveLocation() override;
    void OnSaveHiResBmp() override;
    void OnSaveItersText() override;
    void OnSaveBmp() override;
    void OnSaveRefOrbitText() override;
    void OnSaveRefOrbitTextSimple() override;
    void OnSaveRefOrbitTextMax() override;
    void OnSaveRefOrbitImagMax() override;
    void OnDiffRefOrbitImagMax() override;
    void OnLoadLocation() override;
    void OnLoadEnterLocation() override;
    void OnLoadRefOrbitImagMax() override;
    void OnLoadRefOrbitImagMaxSaved() override;

    // Tests / Benchmarks

    // LA

private:
    SplashWindow Splash;

    std::vector<FractalShark::SavedLocation> gSavedLocations;
    std::vector<ImaginaSavedLocation> gImaginaLocations;

    // Global Variables:
    std::unique_ptr<Environment::JobObject> gJobObj;

    HINSTANCE hInst; // current instance
    LPCWSTR szWindowClass = L"FractalWindow";
    UniqueHMenu gPopupMenu;
    bool gWindowed; // Says whether we are in windowed mode or not.
    HWND hWnd;

    HMENU LoadSubMenu;
    HMENU ImaginaMenu;

    // Fractal:
    std::unique_ptr<Fractal> gFractal;

    ATOM MyRegisterClass(HINSTANCE hInstance);
    HWND InitInstance(HINSTANCE, int);
    void ApplyBorderlessWindowedStyle();
    void ApplyBorderlessFullscreenStyle();
    void SetModeWindowed(bool windowed);
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT CALLBACK WndProc(UINT message, WPARAM wParam, LPARAM lParam);
    void UnInit();
    void HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM lParam);
    void HandleArrowAndZoomKeys(WPARAM vk);

    void DrawFractalShark();

    enum class OpenBoxType { Open, Save };

    static std::wstring OpenFileDialog(OpenBoxType type,
                                       const wchar_t *filter = nullptr,
                                       const wchar_t *defaultExtension = nullptr,
                                       unsigned long saveFlags = 0);

    void ActivateSavedOrbit(size_t index);
    void ActivateImagina(size_t index);

    Fractal &
    GetFractal() noexcept override
    {
        return *gFractal;
    }
    MenuPoint GetMenuMousePos() const override;

    // ---- Chunk A: persist menu click location as member state ----
    // Stored in CLIENT coordinates. Set by WM_CONTEXTMENU.
    POINT lastMenuPtClient_{-1, -1};

    // --- Mouse drag/zoom box state (was function-local statics in WndProc) ---
    bool lButtonDown = false;
    int dragBoxX1 = 0;
    int dragBoxY1 = 0;

    // Used for drawing the inverted rectangle properly.
    int prevX1 = -1;
    int prevY1 = -1;

    CommandDispatcher commandDispatcher;

    bool HasLastMenuPtClient() const noexcept;
    POINT GetSafeMenuPtClient() const;
    // ---------------------------------------------------------------------

    // Controlling functions
    void MenuStandardView(size_t i);
    void MenuWindowed(bool square);
    void MenuPaletteRotation();
    void MenuGetCurPos();
    void MenuSaveCurrentLocation();
    void MenuLoadCurrentLocation();
    static LRESULT EditSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    void MenuLoadEnterLocation();
    void MenuSaveBMP();
    void MenuSaveHiResBMP();
    void MenuSaveItersAsText();
    void BenchmarkMessage(size_t milliseconds);
    void MenuAlgHelp();
    void MenuViewsHelp();
    void MenuLoadImagDyn(ImaginaSettings loadSettings);
    void MenuSaveImag(CompressToDisk compression);
    void MenuDiffImag();
    void MenuLoadImag(ImaginaSettings loadSettings, CompressToDisk compression);
    void MenuShowHotkeys();

    void PaintAsNecessary();

    void ClearMenu(HMENU &menu);
    void LoadRefOrbit(CompressToDisk compressToDisk,
                      ImaginaSettings loadSettings,
                      std::wstring filename);

    bool
    IsDownControl()
    {
        return (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000;
    };
    bool
    IsDownShift()
    {
        return (GetAsyncKeyState(VK_SHIFT) & 0x8000) == 0x8000;
    };
    bool
    IsDownAlt()
    {
        return (GetAsyncKeyState(VK_MENU) & 0x8000) == 0x8000;
    };
};

} // namespace FractalShark::Win32
