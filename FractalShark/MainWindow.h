#pragma once

#include <memory>
#include <vector>

class JobObject;
class Fractal;
enum FractalPalette : size_t;

class MainWindow {
public:
    struct SavedLocation;

    MainWindow(HINSTANCE hInstance, int nCmdShow);
    ~MainWindow();

    std::vector<SavedLocation> gSavedLocations;

    // Global Variables:
    std::unique_ptr<JobObject> gJobObj;

    HINSTANCE hInst;                // current instance
    LPCWSTR szWindowClass = L"FractalWindow";
    HMENU gPopupMenu;
    bool gWindowed; // Says whether we are in windowed mode or not.
    HWND hWnd;

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
    void MenuSaveBMP();
    void MenuSaveHiResBMP();
    void MenuSaveItersAsText();
    void BenchmarkMessage(size_t milliseconds);
    void MenuAlgHelp();
    void MenuViewsHelp();
    void MenuShowHotkeys();

    void PaintAsNecessary();

    bool IsDownControl() { return (GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000; };
    bool IsDownShift() { return (GetAsyncKeyState(VK_SHIFT) & 0x8000) == 0x8000; };
    bool IsDownAlt() { return (GetAsyncKeyState(VK_MENU) & 0x8000) == 0x8000; };

};