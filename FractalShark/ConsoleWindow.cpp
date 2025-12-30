#include "stdafx.h"
#include "ConsoleWindow.h"

#include <iostream>

void
AttachBackgroundConsole(bool initiallyVisible)
{
    if (!AllocConsole())
        return;

    HWND hConsole = GetConsoleWindow();
    if (!hConsole)
        return;

    // Hide immediately to avoid the creation flash
    ShowWindow(hConsole, SW_HIDE);

    // Redirect stdio
    FILE *fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
    freopen_s(&fp, "CONIN$", "r", stdin);
    std::ios::sync_with_stdio(true);

    // Alt-Tab visible: APPWINDOW on, TOOLWINDOW off
    LONG_PTR ex = GetWindowLongPtrW(hConsole, GWL_EXSTYLE);
    ex |= WS_EX_APPWINDOW;
    ex &= ~WS_EX_TOOLWINDOW;
    SetWindowLongPtrW(hConsole, GWL_EXSTYLE, ex);

    // Apply the style change
    SetWindowPos(hConsole,
                 nullptr,
                 0,
                 0,
                 0,
                 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER | SWP_FRAMECHANGED);

    if (initiallyVisible) {
        // Show without activation
        ShowWindow(hConsole, SW_SHOWNOACTIVATE);

        // **Force it behind everything** (your previous "OK" behavior)
        SetWindowPos(hConsole, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    }
}
