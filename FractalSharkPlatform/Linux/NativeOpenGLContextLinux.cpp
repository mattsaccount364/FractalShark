#include "NativeOpenGLContext.h"

#include "ConsoleLog.h"

// clang-format off
#include "GlIncludes.h"
// clang-format on

#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace {

Display *g_display = nullptr;

Display *
GetX11Display()
{
    if (!g_display) {
        g_display = XOpenDisplay(nullptr);
    }
    return g_display;
}

bool
IsKnownSoftwareRendererName(std::string_view rendererName)
{
    constexpr std::array softwareRendererTokens{
        "llvmpipe", "softpipe", "swrast", "lavapipe", "software rasterizer"};

    return std::any_of(
        softwareRendererTokens.begin(), softwareRendererTokens.end(), [&](std::string_view token) {
            return std::search(rendererName.begin(),
                               rendererName.end(),
                               token.begin(),
                               token.end(),
                               [](char lhs, char rhs) {
                                   return std::tolower(static_cast<unsigned char>(lhs)) ==
                                          std::tolower(static_cast<unsigned char>(rhs));
                               }) != rendererName.end();
        });
}

struct SharedWindowGlState {
    GLXContext shareRoot = nullptr;
    bool doubleBuffered = false;
};

std::mutex g_windowMutex;
std::unordered_map<Window, SharedWindowGlState> g_windowState;

struct XVisualInfoDeleter {
    void
    operator()(XVisualInfo *visualInfo) const noexcept
    {
        if (visualInfo) {
            XFree(visualInfo);
        }
    }
};

using XVisualInfoPtr = std::unique_ptr<XVisualInfo, XVisualInfoDeleter>;

void
RegisterWindowGlState(Window win, GLXContext ctx, bool doubleBuffered)
{
    std::scoped_lock lock(g_windowMutex);
    auto &st = g_windowState[win];
    if (!st.shareRoot)
        st.shareRoot = ctx;
    st.doubleBuffered = doubleBuffered;
}

void
UnregisterShareRoot(Window win, GLXContext ctx)
{
    std::scoped_lock lock(g_windowMutex);
    auto it = g_windowState.find(win);
    if (it != g_windowState.end() && it->second.shareRoot == ctx)
        it->second.shareRoot = nullptr;
}

GLXContext
GetShareRoot(Window win)
{
    std::scoped_lock lock(g_windowMutex);
    auto it = g_windowState.find(win);
    if (it == g_windowState.end())
        return nullptr;
    return it->second.shareRoot;
}

bool
IsWindowDoubleBuffered(Window win)
{
    std::scoped_lock lock(g_windowMutex);
    auto it = g_windowState.find(win);
    if (it == g_windowState.end())
        return false;
    return it->second.doubleBuffered;
}

XVisualInfoPtr
GetWindowVisualInfo(Display *dpy, Window win)
{
    XWindowAttributes attributes{};
    if (!XGetWindowAttributes(dpy, win, &attributes)) {
        return nullptr;
    }

    XVisualInfo visualTemplate{};
    visualTemplate.visualid = XVisualIDFromVisual(attributes.visual);
    visualTemplate.screen = XScreenNumberOfScreen(attributes.screen);

    int visualCount = 0;
    return XVisualInfoPtr{
        XGetVisualInfo(dpy, VisualIDMask | VisualScreenMask, &visualTemplate, &visualCount)};
}

} // namespace

namespace Environment {

NativeOpenGLContext::NativeOpenGLContext(void *nativeWindow) : m_NativeWindow(nativeWindow)
{
    if (!m_NativeWindow) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "OpenGlContext: null Window";
        return;
    }

    Display *dpy = GetX11Display();
    if (!dpy) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "OpenGlContext: failed to open X11 display";
        return;
    }

    Window win = reinterpret_cast<Window>(m_NativeWindow);

    XVisualInfoPtr visualInfo = GetWindowVisualInfo(dpy, win);
    if (!visualInfo) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "OpenGlContext: failed to query window visual";
        return;
    }

    int supportsGl = False;
    int rgba = False;
    int doubleBuffered = False;
    if (glXGetConfig(dpy, visualInfo.get(), GLX_USE_GL, &supportsGl) != 0 || supportsGl != True ||
        glXGetConfig(dpy, visualInfo.get(), GLX_RGBA, &rgba) != 0 || rgba != True ||
        glXGetConfig(dpy, visualInfo.get(), GLX_DOUBLEBUFFER, &doubleBuffered) != 0) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "OpenGlContext: window visual is not a usable GLX RGBA visual";
        return;
    }

    GLXContext shareCtx = GetShareRoot(win);
    GLXContext ctx = glXCreateContext(dpy, visualInfo.get(), shareCtx, GL_TRUE);

    if (!ctx) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "OpenGlContext: glXCreateContext failed";
        return;
    }

    m_RenderContext = reinterpret_cast<void *>(ctx);
    RegisterWindowGlState(win, ctx, doubleBuffered == True);

    if (!MakeCurrent()) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "OpenGlContext: MakeCurrent failed";
        return;
    }

    const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const char *version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_MaxTextureSize);
    const bool isDirect = glXIsDirect(dpy, ctx) == True;
    m_IsSoftwareRenderer = !isDirect || IsKnownSoftwareRendererName(renderer ? renderer : "");

    char buf[512];
    snprintf(buf,
             sizeof(buf),
             "OpenGlContext: renderer=%s, version=%s, maxTex=%d, direct(bool)=%d, "
             "software(bool)=%d, doubleBuffered(bool)=%d",
             renderer ? renderer : "(null)",
             version ? version : "(null)",
             m_MaxTextureSize,
             isDirect ? 1 : 0,
             m_IsSoftwareRenderer ? 1 : 0,
             doubleBuffered == True ? 1 : 0);
    FractalSharkLog::LogLine(__FILE__, __LINE__) << buf;

    m_Valid = true;
}

NativeOpenGLContext::~NativeOpenGLContext()
{
    Display *dpy = g_display;
    GLXContext ctx = reinterpret_cast<GLXContext>(m_RenderContext);

    if (dpy && ctx && glXGetCurrentContext() == ctx) {
        glXMakeCurrent(dpy, None, nullptr);
    }

    if (dpy && ctx) {
        if (m_NativeWindow) {
            Window win = reinterpret_cast<Window>(m_NativeWindow);
            UnregisterShareRoot(win, ctx);
        }
        glXDestroyContext(dpy, ctx);
        m_RenderContext = nullptr;
    }
}

bool
NativeOpenGLContext::IsValid() const noexcept
{
    return m_Valid;
}

bool
NativeOpenGLContext::MakeCurrent() noexcept
{
    Display *dpy = GetX11Display();
    if (!dpy || !m_NativeWindow || !m_RenderContext)
        return false;

    Window win = reinterpret_cast<Window>(m_NativeWindow);
    GLXContext ctx = reinterpret_cast<GLXContext>(m_RenderContext);
    return glXMakeCurrent(dpy, win, ctx) == True;
}

void
NativeOpenGLContext::SwapBuffers() noexcept
{
    Display *dpy = GetX11Display();
    if (!dpy || !m_NativeWindow) {
        glFlush();
        return;
    }

    Window win = reinterpret_cast<Window>(m_NativeWindow);
    if (IsWindowDoubleBuffered(win)) {
        glXSwapBuffers(dpy, win);
    } else {
        glFlush();
    }
}

std::optional<ScreenRect>
NativeOpenGLContext::GetClientRect() const noexcept
{
    Display *dpy = GetX11Display();
    if (!dpy || !m_NativeWindow)
        return std::nullopt;

    Window win = reinterpret_cast<Window>(m_NativeWindow);
    XWindowAttributes attr{};
    if (!XGetWindowAttributes(dpy, win, &attr))
        return std::nullopt;

    return ScreenRect{0, 0, attr.width, attr.height};
}

bool
NativeOpenGLContext::IsSoftwareRenderer() const noexcept
{
    return m_IsSoftwareRenderer;
}

int
NativeOpenGLContext::GetMaxTextureSize() const noexcept
{
    return m_MaxTextureSize;
}

} // namespace Environment
