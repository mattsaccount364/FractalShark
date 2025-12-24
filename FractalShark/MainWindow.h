#pragma once

#include <memory>
#include <vector>
#include <string>

#include "UniqueHMenu.h"

class JobObject;
class Fractal;
enum FractalPalette : size_t;
enum class CompressToDisk;
enum class ImaginaSettings;

namespace FractalShark {
class UniqueHMenu;
} // namespace FractalShark

class MainWindow {
public:
    struct SavedLocation;
    struct ImaginaSavedLocation;

    MainWindow(HINSTANCE hInstance, int nCmdShow);
    ~MainWindow();

private:

    std::vector<SavedLocation> gSavedLocations;
    std::vector<ImaginaSavedLocation> gImaginaLocations;

    // Global Variables:
    std::unique_ptr<JobObject> gJobObj;

    HINSTANCE hInst;                // current instance
    LPCWSTR szWindowClass = L"FractalWindow";
    FractalShark::UniqueHMenu gPopupMenu;
    bool gWindowed; // Says whether we are in windowed mode or not.
    HWND hWnd;

    HMENU LoadSubMenu;
    HMENU ImaginaMenu;

    // Fractal:
    std::unique_ptr<Fractal> gFractal;

    ATOM MyRegisterClass(HINSTANCE hInstance);
    HWND InitInstance(HINSTANCE, int);
    static LRESULT CALLBACK StaticWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
    LRESULT CALLBACK WndProc(UINT message, WPARAM wParam, LPARAM lParam); 
    void UnInit();
    void HandleKeyDown(UINT /*message*/, WPARAM wParam, LPARAM /*lParam*/);

    static void create_minidump(struct _EXCEPTION_POINTERS *apExceptionInfo);
    static LONG WINAPI unhandled_handler(struct _EXCEPTION_POINTERS *apExceptionInfo);

    void DrawFractalShark();
    void DrawFractalSharkGdi(int nCmdShow);

    enum class OpenBoxType {
        Open,
        Save
    };

    static std::wstring OpenFileDialog(OpenBoxType type);

    // ---------------------------------------------------------------------
    // PHASE 3: command router expanded
    // ---------------------------------------------------------------------
    bool HandleCommand(int wmId);

    // Phase 2 pieces
    bool HandleCommandRange(int wmId);
    void ActivateSavedOrbit(size_t index);
    void ActivateImagina(size_t index);

    // Phase 3 pieces
    bool HandleAlgCommand(int wmId);
    bool HandleCommandTable(int wmId);

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

    bool HasLastMenuPtClient() const noexcept;
    POINT GetSafeMenuPtClient() const;
    // ---------------------------------------------------------------------

    // Controlling functions
    void MenuGoBack();
    void MenuStandardView(size_t i);
    void MenuSquareView();
    void MenuCenterView(int x, int y);
    void MenuZoomIn(POINT mousePt);
    void MenuZoomOut(POINT mousePt);
    void MenuRepainting();
    void MenuWindowed(bool square);
    void MenuMultiplyIterations(double factor);
    void MenuResetIterations();
    void MenuPaletteType(FractalPalette type);
    void MenuPaletteDepth(int depth);
    void MenuPaletteRotation();
    void MenuCreateNewPalette();
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
    void MenuLoadImag(
        ImaginaSettings loadSettings,
        CompressToDisk compression);
    void MenuShowHotkeys();

    void PaintAsNecessary();

    void ClearMenu(HMENU &menu);
    void LoadRefOrbit(CompressToDisk compressToDisk, ImaginaSettings loadSettings, std::wstring filename);

    bool IsDownControl() { return (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000; };
    bool IsDownShift() { return (GetAsyncKeyState(VK_SHIFT) & 0x8000) == 0x8000; };
    bool IsDownAlt() { return (GetAsyncKeyState(VK_MENU) & 0x8000) == 0x8000; };
};