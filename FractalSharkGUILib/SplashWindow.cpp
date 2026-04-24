#include "stdafx.h"
#include "SplashWindow.h"

#include "WPngImage\WPngImage.hh"
#include "resource.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::wstring_view kSplashClassName = L"FractalSharkSplashWindowClass";

// -------------------- RAII helpers --------------------

struct UniqueHandle {
    HANDLE h = nullptr;
    UniqueHandle() = default;
    explicit UniqueHandle(HANDLE hh) : h(hh) {}
    UniqueHandle(const UniqueHandle &) = delete;
    UniqueHandle &operator=(const UniqueHandle &) = delete;
    UniqueHandle(UniqueHandle &&o) noexcept : h(std::exchange(o.h, nullptr)) {}
    UniqueHandle &
    operator=(UniqueHandle &&o) noexcept
    {
        if (this != &o) {
            reset();
            h = std::exchange(o.h, nullptr);
        }
        return *this;
    }
    ~UniqueHandle() { reset(); }
    void
    reset(HANDLE nh = nullptr) noexcept
    {
        if (h)
            CloseHandle(h);
        h = nh;
    }
    HANDLE
    get() const noexcept
    {
        return h;
    }
    explicit
    operator bool() const noexcept
    {
        return h != nullptr;
    }
};

struct ScopedDC {
    HWND hwnd = nullptr;
    HDC dc = nullptr;
    explicit ScopedDC(HWND w) : hwnd(w), dc(GetDC(w)) {}
    ScopedDC(const ScopedDC &) = delete;
    ScopedDC &operator=(const ScopedDC &) = delete;
    ~ScopedDC()
    {
        if (dc)
            ReleaseDC(hwnd, dc);
    }
    HDC
    get() const noexcept
    {
        return dc;
    }
    explicit
    operator bool() const noexcept
    {
        return dc != nullptr;
    }
};

struct ScopedMemDC {
    HDC dc = nullptr;
    explicit ScopedMemDC(HDC compatibleWith) : dc(CreateCompatibleDC(compatibleWith)) {}
    ScopedMemDC(const ScopedMemDC &) = delete;
    ScopedMemDC &operator=(const ScopedMemDC &) = delete;
    ~ScopedMemDC()
    {
        if (dc)
            DeleteDC(dc);
    }
    HDC
    get() const noexcept
    {
        return dc;
    }
    explicit
    operator bool() const noexcept
    {
        return dc != nullptr;
    }
};

struct ScopedBitmap {
    HBITMAP bm = nullptr;
    ScopedBitmap() = default;
    explicit ScopedBitmap(HBITMAP b) : bm(b) {}
    ScopedBitmap(const ScopedBitmap &) = delete;
    ScopedBitmap &operator=(const ScopedBitmap &) = delete;
    ScopedBitmap(ScopedBitmap &&o) noexcept : bm(std::exchange(o.bm, nullptr)) {}
    ScopedBitmap &
    operator=(ScopedBitmap &&o) noexcept
    {
        if (this != &o) {
            reset();
            bm = std::exchange(o.bm, nullptr);
        }
        return *this;
    }
    ~ScopedBitmap() { reset(); }
    void
    reset(HBITMAP nb = nullptr) noexcept
    {
        if (bm)
            DeleteObject(bm);
        bm = nb;
    }
    HBITMAP
    get() const noexcept
    {
        return bm;
    }
    explicit
    operator bool() const noexcept
    {
        return bm != nullptr;
    }
};

struct ScopedSelectObject {
    HDC dc = nullptr;
    HGDIOBJ old = nullptr;
    ScopedSelectObject(HDC d, HGDIOBJ obj) : dc(d), old(SelectObject(d, obj)) {}
    ScopedSelectObject(const ScopedSelectObject &) = delete;
    ScopedSelectObject &operator=(const ScopedSelectObject &) = delete;
    ~ScopedSelectObject()
    {
        if (dc && old)
            SelectObject(dc, old);
    }
};

inline UniqueHandle
MakeManualResetEvent(bool initialState)
{
    return UniqueHandle(CreateEventW(nullptr, TRUE, initialState ? TRUE : FALSE, nullptr));
}

} // namespace

// ------------------------------------------------------------
// Lifetime
// ------------------------------------------------------------

SplashWindow::~SplashWindow() noexcept { Stop(); }

bool
SplashWindow::Start(HINSTANCE hInst)
{
    if (Running.load(std::memory_order_acquire)) {
        return true;
    }

    HInst = hInst;
    CreatedEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!CreatedEvent)
        return false;

    // jthread = RAII thread + stop token
    Worker = std::jthread([this](std::stop_token st) { this->ThreadMain(st); });

    // wait for window creation (or early failure)
    WaitForSingleObject(CreatedEvent, INFINITE);
    return SplashHwnd.load(std::memory_order_acquire) != nullptr;
}

void
SplashWindow::Stop() noexcept
{
    if (!Running.load(std::memory_order_acquire) && !Worker.joinable()) {
        return;
    }

    // request thread stop (C++20)
    if (Worker.joinable()) {
        Worker.request_stop();
        Worker.join();
    }

    Running.store(false, std::memory_order_release);
    SplashHwnd.store(nullptr, std::memory_order_release);

    if (CreatedEvent) {
        CloseHandle(CreatedEvent);
        CreatedEvent = nullptr;
    }
}

// ------------------------------------------------------------
// Thread + message loop
// ------------------------------------------------------------

void
SplashWindow::ThreadMain(std::stop_token st)
{
    Running.store(true, std::memory_order_release);

    if (!RegisterClassOnce()) {
        SetEvent(CreatedEvent);
        Running.store(false, std::memory_order_release);
        return;
    }

    HWND hwnd = CreateSplashWindow();
    SplashHwnd.store(hwnd, std::memory_order_release);

    // signal creation (success or failure)
    SetEvent(CreatedEvent);

    if (!hwnd) {
        Running.store(false, std::memory_order_release);
        return;
    }

    PaintOnce();

    // Wait for either:
    //  - messages in the queue
    //  - stop requested
    //
    // Use MsgWaitForMultipleObjects to avoid polling.
    for (;;) {
        if (st.stop_requested()) {
            // Politely drop topmost, then close
            SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
            PostMessageW(hwnd, WM_CLOSE, 0, 0);
        }

        // Pump messages
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                Running.store(false, std::memory_order_release);
                return;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        // If stop requested and window is gone, exit
        if (st.stop_requested() && !IsWindow(hwnd)) {
            Running.store(false, std::memory_order_release);
            return;
        }

        // Wait until either new messages arrive or a short timeout passes
        // (timeout keeps stop responsiveness without spinning)
        MsgWaitForMultipleObjects(0,
                                  nullptr,
                                  FALSE,
                                  50, // ms
                                  QS_ALLINPUT);
    }
}

// ------------------------------------------------------------
// Window class / creation
// ------------------------------------------------------------

bool
SplashWindow::RegisterClassOnce()
{
    WNDCLASSEXW wcex{};
    wcex.cbSize = sizeof(wcex);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = &SplashWindow::WndProc;
    wcex.hInstance = HInst;
    wcex.hIcon = LoadIcon(HInst, (LPCTSTR)IDI_FRACTALS);
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = nullptr;
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = kSplashClassName.data();
    wcex.hIconSm = LoadIcon(HInst, (LPCTSTR)IDI_SMALL);

    RegisterClassExW(&wcex);
    return true;
}

HWND
SplashWindow::CreateSplashWindow()
{
    // Virtual screen (multi-monitor safe)
    const int vLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
    const int vTop = GetSystemMetrics(SM_YVIRTUALSCREEN);
    const int vWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    const int vHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);

    const int size = std::min(vWidth / 2, vHeight / 2);
    const int startX = vLeft + (vWidth - size) / 2;
    const int startY = vTop + (vHeight - size) / 2;

    constexpr DWORD style = WS_POPUP;
    constexpr DWORD exStyle = WS_EX_APPWINDOW | WS_EX_LAYERED;

    HWND hwnd = CreateWindowExW(exStyle,
                                kSplashClassName.data(),
                                L"",
                                style,
                                startX,
                                startY,
                                size,
                                size,
                                nullptr,
                                nullptr,
                                HInst,
                                nullptr);

    if (!hwnd) {
        return nullptr;
    }

    // Start transparent
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_ALPHA);

    SetWindowPos(hwnd,
                 HWND_NOTOPMOST,
                 startX,
                 startY,
                 size,
                 size,
                 SWP_NOOWNERZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);

    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    UpdateWindow(hwnd);

    return hwnd;
}

// ------------------------------------------------------------
// Painting
// ------------------------------------------------------------

void
SplashWindow::PaintOnce()
{
    const HWND hwnd = SplashHwnd.load(std::memory_order_acquire);
    if (!hwnd)
        return;

    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_int_distribution<int> dist(0, 2);

    const std::array<LPCWSTR, 3> resources = {
        MAKEINTRESOURCE(IDB_PNG_SPLASH1),
        MAKEINTRESOURCE(IDB_PNG_SPLASH2),
        MAKEINTRESOURCE(IDB_PNG_SPLASH3),
    };

    LPCWSTR resStr = resources[(size_t)dist(rng)];

    HRSRC hRes = FindResource(HInst, resStr, L"PNG");
    if (!hRes)
        return;

    HGLOBAL hResData = LoadResource(HInst, hRes);
    if (!hResData)
        return;

    void *pResData = LockResource(hResData);
    const DWORD bytes = SizeofResource(HInst, hRes);
    if (!pResData || !bytes)
        return;

    WPngImage image{};
    image.loadImageFromRAM(pResData, bytes, WPngImage::PixelFormat::kPixelFormat_RGBA8);

    const int srcW = image.width();
    const int srcH = image.height();
    if (srcW <= 0 || srcH <= 0)
        return;

    std::vector<std::uint8_t> pixels((size_t)srcW * (size_t)srcH * 4);
    for (int y = 0; y < srcH; ++y) {
        for (int x = 0; x < srcW; ++x) {
            auto p = image.get8(x, y);
            const size_t i = ((size_t)y * (size_t)srcW + (size_t)x) * 4;
            pixels[i + 0] = p.b;
            pixels[i + 1] = p.g;
            pixels[i + 2] = p.r;
            pixels[i + 3] = p.a;
        }
    }

    RECT rc{};
    GetClientRect(hwnd, &rc);
    const int dstW = rc.right - rc.left;
    const int dstH = rc.bottom - rc.top;
    if (dstW <= 0 || dstH <= 0)
        return;

    ScopedDC hdc(hwnd);
    if (!hdc)
        return;

    ScopedMemDC mem(hdc.get());
    if (!mem)
        return;

    ScopedBitmap bm(CreateBitmap(srcW, srcH, 1, 32, pixels.data()));
    if (!bm)
        return;

    ScopedSelectObject sel(mem.get(), bm.get());

    // Clear to black first
    FillRect(hdc.get(), &rc, (HBRUSH)GetStockObject(BLACK_BRUSH));

    // Compute "fit inside" rectangle preserving aspect ratio
    // scale = min(dstW/srcW, dstH/srcH)
    const double sx = (double)dstW / (double)srcW;
    const double sy = (double)dstH / (double)srcH;
    const double s = std::min(sx, sy);

    const int drawW = std::max(1, (int)std::lround((double)srcW * s));
    const int drawH = std::max(1, (int)std::lround((double)srcH * s));

    const int dx = (dstW - drawW) / 2;
    const int dy = (dstH - drawH) / 2;

    // High-quality stretch (for both downscale and upscale)
    SetStretchBltMode(hdc.get(), HALFTONE);
    SetBrushOrgEx(hdc.get(), 0, 0, nullptr);

    StretchBlt(hdc.get(), dx, dy, drawW, drawH, mem.get(), 0, 0, srcW, srcH, SRCCOPY);

    // Make opaque after first paint
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 255, LWA_ALPHA);
}

// ------------------------------------------------------------
// Window proc
// ------------------------------------------------------------

LRESULT CALLBACK
SplashWindow::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
        case WM_NCHITTEST:
            return HTCLIENT;

        case WM_ERASEBKGND:
            return 1;

        case WM_CLOSE:
            DestroyWindow(hWnd);
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        default:
            return DefWindowProcW(hWnd, msg, wParam, lParam);
    }
}
