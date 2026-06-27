#include "NativeOpenGLContext.h"

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
#include <iostream>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace {

void
GlLog(const char *msg)
{
    std::cerr << msg << std::endl;
}

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
};

std::mutex g_windowMutex;
std::unordered_map<Window, SharedWindowGlState> g_windowState;

void
RegisterShareRoot(Window win, GLXContext ctx)
{
    std::scoped_lock lock(g_windowMutex);
    auto &st = g_windowState[win];
    if (!st.shareRoot)
        st.shareRoot = ctx;
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

XVisualInfo *
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
    return XGetVisualInfo(dpy, VisualIDMask | VisualScreenMask, &visualTemplate, &visualCount);
}

} // namespace

namespace Environment {

NativeOpenGLContext::NativeOpenGLContext(void *nativeWindow) : m_NativeWindow(nativeWindow)
{
    if (!m_NativeWindow) {
        GlLog("OpenGlContext: null Window");
        return;
    }

    Display *dpy = GetX11Display();
    if (!dpy) {
        GlLog("OpenGlContext: failed to open X11 display");
        return;
    }

    Window win = reinterpret_cast<Window>(m_NativeWindow);

    XVisualInfo *vi = GetWindowVisualInfo(dpy, win);
    if (!vi) {
        GlLog("OpenGlContext: failed to query window visual");
        return;
    }

    int supportsGl = False;
    int rgba = False;
    int doubleBuffered = False;
    if (glXGetConfig(dpy, vi, GLX_USE_GL, &supportsGl) != 0 || supportsGl != True ||
        glXGetConfig(dpy, vi, GLX_RGBA, &rgba) != 0 || rgba != True ||
        glXGetConfig(dpy, vi, GLX_DOUBLEBUFFER, &doubleBuffered) != 0) {
        GlLog("OpenGlContext: window visual is not a usable GLX RGBA visual");
        XFree(vi);
        return;
    }
    m_IsDoubleBuffered = doubleBuffered == True;

    GLXContext shareCtx = GetShareRoot(win);
    GLXContext ctx = glXCreateContext(dpy, vi, shareCtx, GL_TRUE);
    XFree(vi);

    if (!ctx) {
        GlLog("OpenGlContext: glXCreateContext failed");
        return;
    }

    m_RenderContext = reinterpret_cast<void *>(ctx);
    RegisterShareRoot(win, ctx);

    if (!MakeCurrent()) {
        GlLog("OpenGlContext: MakeCurrent failed");
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
             m_IsDoubleBuffered ? 1 : 0);
    GlLog(buf);

    m_Valid = true;
}

NativeOpenGLContext::~NativeOpenGLContext()
{
    Display *dpy = GetX11Display();
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
    if (dpy && m_NativeWindow) {
        if (m_IsDoubleBuffered) {
            Window win = reinterpret_cast<Window>(m_NativeWindow);
            glXSwapBuffers(dpy, win);
        } else {
            glFlush();
        }
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

bool
NativeOpenGLContext::IsDoubleBuffered() const noexcept
{
    return m_IsDoubleBuffered;
}

int
NativeOpenGLContext::GetMaxTextureSize() const noexcept
{
    return m_MaxTextureSize;
}

} // namespace Environment
