#include "stdafx.h"

#include "MainWindow.h"
#include "Callstacks.h"
#include "heap_allocator\include\HeapCpp.h"

#include <iostream>
#include <conio.h>

void
PressAnyKeyToContinue()
{
    std::cout << "\nPress any key to continue . . .";
    std::cout.flush();

    _getch(); // waits for a single key, no Enter required
}


int APIENTRY WinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE /*hPrevInstance*/,
    _In_ LPSTR     /*lpCmdLine*/,
    _In_ int       nCmdShow) {

    MSG msg{};

    RegisterHeapCleanup();
    GlobalCallstacks->InitCallstacks();

    {
        MainWindow mainWindow{ hInstance, nCmdShow };

        //// Main message loop:
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    GlobalCallstacks->FreeCallstacks();
    PressAnyKeyToContinue();

    return (int)msg.wParam;
}

