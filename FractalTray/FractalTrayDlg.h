#pragma once

#include "TrayIcon.h"
#include <string>
#include <thread>

constexpr UINT WM_ICON_NOTIFY = 1200;
constexpr UINT WM_FINISHED_CALCULATING = 1201;

constexpr double DefaultScaleFactor = 75.0;
constexpr int DefaultWidth = 3840;
constexpr int DefaultHeight = 1600;

class FractalTrayDialog {
public:
    FractalTrayDialog();
    INT_PTR DoModal(HINSTANCE hInst);

private:
    static INT_PTR CALLBACK StaticDlgProc(HWND hDlg, UINT msg,
                                          WPARAM wParam, LPARAM lParam);
    INT_PTR HandleMessage(UINT msg, WPARAM wParam, LPARAM lParam);

    BOOL OnInitDialog();
    void OnPaintIconic();
    void OnClose();
    void OnGenerate();
    LRESULT OnTrayNotification(WPARAM wParam, LPARAM lParam);
    LRESULT OnFinishedCalculating();
    void OnRestore();
    void OnExit();

    void TryLoadDestCoords();
    void StartCalculation();
    void RunCalculation(std::stop_token stopToken);
    int CalculateFrameCount();

    std::wstring GetDlgText(int controlId) const;
    void SetDlgText(int controlId, const std::wstring &text);
    void ReadControlsToMembers();

    HWND m_hDlg = nullptr;
    HINSTANCE m_hInst = nullptr;
    HICON m_hIconActive = nullptr;
    HICON m_hIconIdle = nullptr;
    TrayIcon m_TrayIcon;
    std::jthread m_CalcThread;

    std::wstring m_SourceCoords;
    std::wstring m_DestCoords;
    double m_ScaleFactor = DefaultScaleFactor;
    int m_ResX = DefaultWidth;
    int m_ResY = DefaultHeight;
    std::wstring m_LocationFilename = L"locations.txt";
};
