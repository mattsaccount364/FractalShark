#include "LinuxSplashWindow.h"

#include "LinuxEmbeddedSplashImages.h"
#include "WPngImage/WPngImage.hh"

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <poll.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char *kSplashTitle = "FractalShark";

bool
LoadRandomSplashImage(WPngImage &image)
{
    const std::span<const FractalShark::Linux::EmbeddedSplashImage> images =
        FractalShark::Linux::GetEmbeddedSplashImages();
    if (images.empty()) {
        return false;
    }

    std::vector<std::size_t> imageOrder(images.size());
    for (std::size_t i = 0; i < imageOrder.size(); ++i) {
        imageOrder[i] = i;
    }

    std::random_device randomDevice;
    std::mt19937 rng(randomDevice());
    std::shuffle(imageOrder.begin(), imageOrder.end(), rng);

    for (const std::size_t imageIndex : imageOrder) {
        WPngImage candidate;
        const FractalShark::Linux::EmbeddedSplashImage &embeddedImage = images[imageIndex];
        const WPngImage::IOStatus status = candidate.loadImageFromRAM(
            embeddedImage.Bytes.data(), embeddedImage.Bytes.size(), WPngImage::kPixelFormat_RGBA8);
        if (status == WPngImage::kIOStatus_Ok && candidate.width() > 0 && candidate.height() > 0) {
            image = std::move(candidate);
            return true;
        }
    }

    return false;
}

unsigned long
ScaleColorComponentToMask(std::uint8_t value, unsigned long mask)
{
    if (mask == 0) {
        return 0;
    }

    int shift = 0;
    while (((mask >> shift) & 1UL) == 0) {
        ++shift;
    }

    const unsigned long maskMax = mask >> shift;
    const unsigned long scaled = (static_cast<unsigned long>(value) * maskMax + 127UL) / 255UL;
    return (scaled << shift) & mask;
}

unsigned long
PixelToXPixel(const Visual *visual, const WPngImage::Pixel8 &pixel)
{
    const auto blendAgainstBlack = [](std::uint8_t component, std::uint8_t alpha) {
        return static_cast<std::uint8_t>(
            (static_cast<unsigned int>(component) * static_cast<unsigned int>(alpha) + 127U) / 255U);
    };

    return ScaleColorComponentToMask(blendAgainstBlack(pixel.r, pixel.a), visual->red_mask) |
           ScaleColorComponentToMask(blendAgainstBlack(pixel.g, pixel.a), visual->green_mask) |
           ScaleColorComponentToMask(blendAgainstBlack(pixel.b, pixel.a), visual->blue_mask);
}

WPngImage::Pixel8
SampleBilinear(const WPngImage &image, int destX, int destY, int destWidth, int destHeight)
{
    const double srcX = (static_cast<double>(destX) + 0.5) * static_cast<double>(image.width()) /
                            static_cast<double>(destWidth) -
                        0.5;
    const double srcY = (static_cast<double>(destY) + 0.5) * static_cast<double>(image.height()) /
                            static_cast<double>(destHeight) -
                        0.5;

    const double clampedX = std::clamp(srcX, 0.0, static_cast<double>(image.width() - 1));
    const double clampedY = std::clamp(srcY, 0.0, static_cast<double>(image.height() - 1));
    const int x0 = static_cast<int>(std::floor(clampedX));
    const int y0 = static_cast<int>(std::floor(clampedY));
    const int x1 = std::min(x0 + 1, image.width() - 1);
    const int y1 = std::min(y0 + 1, image.height() - 1);
    const double tx = clampedX - static_cast<double>(x0);
    const double ty = clampedY - static_cast<double>(y0);

    double red = 0.0;
    double green = 0.0;
    double blue = 0.0;
    const auto addPixel = [&](int x, int y, double weight) {
        const WPngImage::Pixel8 pixel = image.get8(x, y);
        const double alpha = static_cast<double>(pixel.a) / 255.0;
        red += weight * static_cast<double>(pixel.r) * alpha;
        green += weight * static_cast<double>(pixel.g) * alpha;
        blue += weight * static_cast<double>(pixel.b) * alpha;
    };

    addPixel(x0, y0, (1.0 - tx) * (1.0 - ty));
    addPixel(x1, y0, tx * (1.0 - ty));
    addPixel(x0, y1, (1.0 - tx) * ty);
    addPixel(x1, y1, tx * ty);

    const auto toByte = [](double value) {
        return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    };

    return WPngImage::Pixel8(toByte(red), toByte(green), toByte(blue));
}

class SplashSession {
public:
    SplashSession() = default;
    ~SplashSession() { Destroy(); }

    SplashSession(const SplashSession &) = delete;
    SplashSession &operator=(const SplashSession &) = delete;

    bool
    Create()
    {
        DisplayHandle = XOpenDisplay(nullptr);
        if (!DisplayHandle) {
            return false;
        }

        if (!LoadRandomSplashImage(Image)) {
            return false;
        }

        Screen = DefaultScreen(DisplayHandle);
        Root = RootWindow(DisplayHandle, Screen);
        VisualHandle = DefaultVisual(DisplayHandle, Screen);
        Depth = DefaultDepth(DisplayHandle, Screen);

        const int displayWidth = DisplayWidth(DisplayHandle, Screen);
        const int displayHeight = DisplayHeight(DisplayHandle, Screen);
        SplashSize = std::max(1, std::min(displayWidth, displayHeight) / 2);
        const int x = std::max(0, (displayWidth - SplashSize) / 2);
        const int y = std::max(0, (displayHeight - SplashSize) / 2);

        XSetWindowAttributes attributes{};
        attributes.background_pixel = BlackPixel(DisplayHandle, Screen);
        attributes.border_pixel = BlackPixel(DisplayHandle, Screen);
        attributes.event_mask = ExposureMask | StructureNotifyMask;
        attributes.override_redirect = True;

        WindowHandle = XCreateWindow(DisplayHandle,
                                     Root,
                                     x,
                                     y,
                                     static_cast<unsigned int>(SplashSize),
                                     static_cast<unsigned int>(SplashSize),
                                     0,
                                     Depth,
                                     InputOutput,
                                     VisualHandle,
                                     CWBackPixel | CWBorderPixel | CWEventMask | CWOverrideRedirect,
                                     &attributes);
        if (!WindowHandle) {
            return false;
        }

        XStoreName(DisplayHandle, WindowHandle, kSplashTitle);
        SetClassHint();
        SetWindowHints(x, y);
        SetWindowType();

        GraphicsContext = XCreateGC(DisplayHandle, WindowHandle, 0, nullptr);
        if (!GraphicsContext) {
            return false;
        }

        XMapRaised(DisplayHandle, WindowHandle);
        Draw();
        XFlush(DisplayHandle);
        return true;
    }

    void
    Run(std::stop_token stopToken)
    {
        const int connection = ConnectionNumber(DisplayHandle);
        while (!stopToken.stop_requested()) {
            while (XPending(DisplayHandle) > 0) {
                XEvent event{};
                XNextEvent(DisplayHandle, &event);
                if (event.type == Expose && event.xexpose.count == 0) {
                    Draw();
                } else if (event.type == ConfigureNotify) {
                    Draw();
                }
            }

            XFlush(DisplayHandle);
            pollfd pollInfo{connection, POLLIN, 0};
            const int pollResult = poll(&pollInfo, 1, 50);
            (void)pollResult;
        }

        if (WindowHandle) {
            XUnmapWindow(DisplayHandle, WindowHandle);
            XFlush(DisplayHandle);
        }
    }

private:
    void
    SetClassHint()
    {
        XClassHint classHint{};
        std::string resName = "fractalshark-splash";
        std::string resClass = "FractalShark";
        classHint.res_name = resName.data();
        classHint.res_class = resClass.data();
        XSetClassHint(DisplayHandle, WindowHandle, &classHint);
    }

    void
    SetWindowHints(int x, int y)
    {
        XSizeHints sizeHints{};
        sizeHints.flags = PPosition | PMinSize | PMaxSize;
        sizeHints.x = x;
        sizeHints.y = y;
        sizeHints.min_width = SplashSize;
        sizeHints.max_width = SplashSize;
        sizeHints.min_height = SplashSize;
        sizeHints.max_height = SplashSize;
        XSetWMNormalHints(DisplayHandle, WindowHandle, &sizeHints);
    }

    void
    SetWindowType()
    {
        const Atom windowType = XInternAtom(DisplayHandle, "_NET_WM_WINDOW_TYPE", True);
        const Atom splashType = XInternAtom(DisplayHandle, "_NET_WM_WINDOW_TYPE_SPLASH", True);
        if (windowType != None && splashType != None) {
            XChangeProperty(DisplayHandle,
                            WindowHandle,
                            windowType,
                            XA_ATOM,
                            32,
                            PropModeReplace,
                            reinterpret_cast<const unsigned char *>(&splashType),
                            1);
        }
    }

    void
    Draw()
    {
        XWindowAttributes attrs{};
        int width = SplashSize;
        int height = SplashSize;
        if (XGetWindowAttributes(DisplayHandle, WindowHandle, &attrs) != 0) {
            width = attrs.width;
            height = attrs.height;
        }

        XSetForeground(DisplayHandle, GraphicsContext, BlackPixel(DisplayHandle, Screen));
        XFillRectangle(DisplayHandle,
                       WindowHandle,
                       GraphicsContext,
                       0,
                       0,
                       static_cast<unsigned int>(width),
                       static_cast<unsigned int>(height));

        const double scale = std::min(static_cast<double>(width) / static_cast<double>(Image.width()),
                                      static_cast<double>(height) / static_cast<double>(Image.height()));
        const int drawWidth = std::max(1, static_cast<int>(std::lround(Image.width() * scale)));
        const int drawHeight = std::max(1, static_cast<int>(std::lround(Image.height() * scale)));
        const int destX = (width - drawWidth) / 2;
        const int destY = (height - drawHeight) / 2;
        PutScaledImage(destX, destY, drawWidth, drawHeight);
    }

    void
    PutScaledImage(int destX, int destY, int drawWidth, int drawHeight)
    {
        XImage *xImage = XCreateImage(DisplayHandle,
                                      VisualHandle,
                                      static_cast<unsigned int>(Depth),
                                      ZPixmap,
                                      0,
                                      nullptr,
                                      static_cast<unsigned int>(drawWidth),
                                      static_cast<unsigned int>(drawHeight),
                                      32,
                                      0);
        if (!xImage || xImage->bytes_per_line <= 0) {
            if (xImage) {
                XDestroyImage(xImage);
            }
            return;
        }

        std::vector<char> imageData(static_cast<std::size_t>(xImage->bytes_per_line) *
                                    static_cast<std::size_t>(drawHeight));
        xImage->data = imageData.data();

        for (int y = 0; y < drawHeight; ++y) {
            for (int x = 0; x < drawWidth; ++x) {
                const WPngImage::Pixel8 pixel = SampleBilinear(Image, x, y, drawWidth, drawHeight);
                XPutPixel(xImage, x, y, PixelToXPixel(VisualHandle, pixel));
            }
        }

        XPutImage(DisplayHandle,
                  WindowHandle,
                  GraphicsContext,
                  xImage,
                  0,
                  0,
                  destX,
                  destY,
                  static_cast<unsigned int>(drawWidth),
                  static_cast<unsigned int>(drawHeight));

        xImage->data = nullptr;
        XDestroyImage(xImage);
    }

    void
    Destroy()
    {
        if (!DisplayHandle) {
            return;
        }

        if (GraphicsContext) {
            XFreeGC(DisplayHandle, GraphicsContext);
            GraphicsContext = nullptr;
        }
        if (WindowHandle) {
            XDestroyWindow(DisplayHandle, WindowHandle);
            WindowHandle = 0;
        }
        XCloseDisplay(DisplayHandle);
        DisplayHandle = nullptr;
    }

    Display *DisplayHandle = nullptr;
    Window Root = 0;
    Window WindowHandle = 0;
    GC GraphicsContext = nullptr;
    Visual *VisualHandle = nullptr;
    int Screen = 0;
    int Depth = 0;
    int SplashSize = 1;
    WPngImage Image;
};

} // namespace

namespace FractalShark::Linux {

SplashWindow::~SplashWindow() noexcept { Stop(); }

bool
SplashWindow::Start()
{
    if (m_Running.load(std::memory_order_acquire)) {
        return true;
    }

    Stop();

    {
        std::lock_guard<std::mutex> lock(m_StartMutex);
        m_StartCompleted = false;
        m_StartSucceeded = false;
    }

    try {
        m_Worker = std::jthread([this](std::stop_token stopToken) { ThreadMain(stopToken); });
    } catch (...) {
        return false;
    }

    std::unique_lock<std::mutex> lock(m_StartMutex);
    m_StartCondition.wait(lock, [this] { return m_StartCompleted; });
    return m_StartSucceeded;
}

void
SplashWindow::Stop() noexcept
{
    if (m_Worker.joinable()) {
        try {
            m_Worker.request_stop();
            m_Worker.join();
        } catch (...) {
        }
    }
    m_Running.store(false, std::memory_order_release);
}

void
SplashWindow::ThreadMain(std::stop_token stopToken)
{
    SplashSession session;
    const bool created = session.Create();
    if (created) {
        m_Running.store(true, std::memory_order_release);
    }
    SignalStarted(created);
    if (!created) {
        return;
    }

    session.Run(stopToken);
    m_Running.store(false, std::memory_order_release);
}

void
SplashWindow::SignalStarted(bool succeeded) noexcept
{
    {
        std::lock_guard<std::mutex> lock(m_StartMutex);
        m_StartSucceeded = succeeded;
        m_StartCompleted = true;
    }
    m_StartCondition.notify_all();
}

} // namespace FractalShark::Linux
