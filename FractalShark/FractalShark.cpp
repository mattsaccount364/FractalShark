#include "stdafx.h"

#include "MainWindow.h"

int APIENTRY WinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE /*hPrevInstance*/,
    _In_ LPSTR     /*lpCmdLine*/,
    _In_ int       nCmdShow) {

    MSG msg{};

    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_WNDW);

    {
        MainWindow mainWindow{ hInstance, nCmdShow };

        //// Main message loop:
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

#ifdef _DEBUG
    // We have a 16-byte leak thanks to Windows no matter what we do.

    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif

    return (int)msg.wParam;
}

