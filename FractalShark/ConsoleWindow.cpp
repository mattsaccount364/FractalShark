#include "stdafx.h"
#include "ConsoleWindow.h"

#include <iostream>
#include <windows.h>

void
AttachBackgroundConsole(bool initiallyVisible)
{
    // Prefer AllocConsoleWithOptions so the window is never shown until we're ready.
    // (If unavailable, fall back to AllocConsole + immediate hide.)
    using AllocConsoleWithOptionsFn = HRESULT(WINAPI *)(PALLOC_CONSOLE_OPTIONS, PALLOC_CONSOLE_RESULT);

    bool consoleAllocated = false;

    ALLOC_CONSOLE_OPTIONS opts{};
    opts.mode = ALLOC_CONSOLE_MODE_NEW_WINDOW; // ensure we get our own console
    opts.useShowWindow = TRUE;
    opts.showWindow = SW_HIDE; // start hidden: no flash

    ALLOC_CONSOLE_RESULT result = ALLOC_CONSOLE_RESULT_NO_CONSOLE;
    HRESULT hr = AllocConsoleWithOptions(&opts, &result);
    consoleAllocated = SUCCEEDED(hr) && (result == ALLOC_CONSOLE_RESULT_NEW_CONSOLE);

    if (!consoleAllocated) {
        if (!::AllocConsole())
            return;
    }

    HWND hConsole = ::GetConsoleWindow();
    if (!hConsole)
        return;

    // Redundant for the WithOptions path, but harmless (and needed for fallback).
    ::ShowWindow(hConsole, SW_HIDE);

    // Redirect stdio
    FILE *fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    freopen_s(&fp, "CONOUT$", "w", stderr);
    freopen_s(&fp, "CONIN$", "r", stdin);
    std::ios::sync_with_stdio(true);

    // Alt-Tab visible: APPWINDOW on, TOOLWINDOW off
    LONG_PTR ex = ::GetWindowLongPtrW(hConsole, GWL_EXSTYLE);
    ex |= WS_EX_APPWINDOW;
    ex &= ~WS_EX_TOOLWINDOW;
    ::SetWindowLongPtrW(hConsole, GWL_EXSTYLE, ex);

    // Apply the style change
    ::SetWindowPos(hConsole,
                   nullptr,
                   0,
                   0,
                   0,
                   0,
                   SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER | SWP_FRAMECHANGED);

    if (initiallyVisible) {
        // Force it behind everything (your prior behavior)
        ::SetWindowPos(hConsole, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

        // Show without activation
        ::ShowWindow(hConsole, SW_SHOWNOACTIVATE);
    }
}
