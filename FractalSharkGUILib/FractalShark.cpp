#include "stdafx.h"

#include "MainWindow.h"
#include "heap_allocator\include\HeapCpp.h"

#include <chrono>
#include <conio.h>
#include <cstdint>
#include <iostream>
#include <thread>

void
PressAnyKeyToContinue(uint32_t timeoutMs = 5000)
{
    std::cout << "\nPress any key to continue (exits in 5s automatically) . . .";
    std::cout.flush();

    const auto start = std::chrono::steady_clock::now();

    while (true) {
        if (_kbhit()) {
            _getch(); // consume key
            break;
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= timeoutMs) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int APIENTRY
WinMain(_In_ HINSTANCE hInstance,
        _In_opt_ HINSTANCE /*hPrevInstance*/,
        _In_ LPSTR /*lpCmdLine*/,
        _In_ int nCmdShow)
{

    MSG msg{};

    Environment::RegisterHeapCleanup();

    {
        MainWindow mainWindow{hInstance, nCmdShow};

        //// Main message loop:
        while (GetMessage(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    PressAnyKeyToContinue();

    return (int)msg.wParam;
}
