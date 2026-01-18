#pragma once

#define NOMINMAX
#include <windows.h>

#include <atomic>
#include <thread>

class SplashWindow {
public:
    SplashWindow() noexcept = default;
    ~SplashWindow() noexcept;

    SplashWindow(const SplashWindow &) = delete;
    SplashWindow &operator=(const SplashWindow &) = delete;

    bool Start(HINSTANCE hInst);
    void Stop() noexcept;

    [[nodiscard]] HWND
    GetHwnd() const noexcept
    {
        return SplashHwnd.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool
    IsRunning() const noexcept
    {
        return Running.load(std::memory_order_acquire);
    }

private:
    void ThreadMain(std::stop_token stopToken);

    bool RegisterClassOnce();
    HWND CreateSplashWindow();
    void PaintOnce();

    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

private:
    std::jthread Worker;
    std::atomic_bool Running{false};
    std::atomic<HWND> SplashHwnd{nullptr};

    HINSTANCE HInst = nullptr;

    // Manual-reset event used to signal HWND creation.
    // Owned/closed by Start/Stop (or dtor).
    HANDLE CreatedEvent = nullptr;
};
