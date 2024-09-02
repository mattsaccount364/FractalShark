#include "stdafx.h"

#include "MainWindow.h"
#include "Callstacks.h"
#include "heap_allocator\include\HeapCpp.h"

void InitStatics();

int APIENTRY WinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE /*hPrevInstance*/,
    _In_ LPSTR     /*lpCmdLine*/,
    _In_ int       nCmdShow) {

    MSG msg{};

    InitStatics();
    InitGlobalHeap();
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
    ShutdownGlobalHeap();

    return (int)msg.wParam;
}

