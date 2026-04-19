// OpenGLContext.cpp
#include "stdafx.h"

// clang-format off
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
// clang-format on

#include "OpenGLContext.h"
#include "WPngImage/WPngImage.hh"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

void
GlLog(const char *msg)
{
    std::cerr << msg << std::endl;
}

// ============================================================================
// Platform-specific helpers: querying the window client area.
// ============================================================================

#ifdef _WIN32

struct NativeRect {
    long left, top, right, bottom;
};

static bool
GetNativeClientRect(void *nativeWindow, NativeRect &out)
{
    RECT rt{};
    if (!GetClientRect(static_cast<HWND>(nativeWindow), &rt))
        return false;
    out = {rt.left, rt.top, rt.right, rt.bottom};
    return true;
}

#else // Linux / X11

#include <X11/Xlib.h>

// Process-global X11 display connection.  Opened once, never closed (normal X11 practice).
static Display *g_display = nullptr;

static Display *
GetX11Display()
{
    if (!g_display) {
        g_display = XOpenDisplay(nullptr);
    }
    return g_display;
}

struct NativeRect {
    long left, top, right, bottom;
};

static bool
GetNativeClientRect(void *nativeWindow, NativeRect &out)
{
    Display *dpy = GetX11Display();
    if (!dpy || !nativeWindow)
        return false;
    Window win = reinterpret_cast<Window>(nativeWindow);
    XWindowAttributes attr{};
    if (!XGetWindowAttributes(dpy, win, &attr))
        return false;
    out = {0, 0, attr.width, attr.height};
    return true;
}

#endif // _WIN32

// ============================================================================
// Win32/WGL-specific context management
// ============================================================================

#ifdef _WIN32

// Cast helpers for opaque handle members
static inline HWND
AsHWND(void *p)
{
    return static_cast<HWND>(p);
}
static inline HDC
AsHDC(void *p)
{
    return static_cast<HDC>(p);
}
static inline HGLRC
AsHGLRC(void *p)
{
    return static_cast<HGLRC>(p);
}

namespace {

// One-time (per HWND) pixel format setup + optional context sharing.
struct SharedWindowGlState {
    int pixelFormat = 0;
    HGLRC shareRoot = nullptr;
};

std::mutex g_hwndMutex;
std::unordered_map<HWND, SharedWindowGlState> g_hwndState;

bool
EnsurePixelFormatSet(HWND hWnd, HDC hdc, int &outPixelFormat)
{
    if (!hWnd || !hdc)
        return false;

    std::scoped_lock lock(g_hwndMutex);
    auto &st = g_hwndState[hWnd];

    if (st.pixelFormat != 0) {
        outPixelFormat = st.pixelFormat;
        return true;
    }

    const int existing = GetPixelFormat(hdc);
    if (existing != 0) {
        st.pixelFormat = existing;
        outPixelFormat = existing;
        return true;
    }

    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 0;
    pfd.cStencilBits = 0;
    pfd.iLayerType = PFD_MAIN_PLANE;

    const int pf = ChoosePixelFormat(hdc, &pfd);
    if (pf == 0)
        return false;

    if (SetPixelFormat(hdc, pf, &pfd) == FALSE) {
        const int nowPf = GetPixelFormat(hdc);
        if (nowPf == 0)
            return false;
        st.pixelFormat = nowPf;
        outPixelFormat = nowPf;
        return true;
    }

    st.pixelFormat = pf;
    outPixelFormat = pf;
    return true;
}

void
RegisterShareRoot(HWND hWnd, HGLRC rc)
{
    std::scoped_lock lock(g_hwndMutex);
    auto &st = g_hwndState[hWnd];
    if (!st.shareRoot)
        st.shareRoot = rc;
}

void
UnregisterShareRoot(HWND hWnd, HGLRC rc)
{
    std::scoped_lock lock(g_hwndMutex);
    auto it = g_hwndState.find(hWnd);
    if (it != g_hwndState.end() && it->second.shareRoot == rc)
        it->second.shareRoot = nullptr;
}

HGLRC
GetShareRoot(HWND hWnd)
{
    std::scoped_lock lock(g_hwndMutex);
    auto it = g_hwndState.find(hWnd);
    if (it == g_hwndState.end())
        return nullptr;
    return it->second.shareRoot;
}

void
MaybeShareWithRoot(HWND hWnd, HGLRC newRc)
{
    HGLRC root = GetShareRoot(hWnd);
    if (!root || root == newRc)
        return;

    if (wglShareLists(root, newRc) == FALSE) {
    }
}

} // namespace

OpenGlContext::OpenGlContext(void *nativeWindow) : m_hWnd(nativeWindow)
{
    m_Valid = false;

    if (!m_hWnd) {
        GlLog("OpenGlContext: null HWND");
        return;
    }

    m_hDC = GetDC(AsHWND(m_hWnd));
    if (!m_hDC) {
        GlLog("OpenGlContext: GetDC failed");
        return;
    }

    int pf = 0;
    if (!EnsurePixelFormatSet(AsHWND(m_hWnd), AsHDC(m_hDC), pf)) {
        GlLog("OpenGlContext: EnsurePixelFormatSet failed");
        return;
    }

    m_hRC = wglCreateContext(AsHDC(m_hDC));
    if (!m_hRC) {
        char buf[128];
        snprintf(buf, sizeof(buf), "OpenGlContext: wglCreateContext failed, error=%lu", GetLastError());
        GlLog(buf);
        return;
    }

    MaybeShareWithRoot(AsHWND(m_hWnd), AsHGLRC(m_hRC));
    RegisterShareRoot(AsHWND(m_hWnd), AsHGLRC(m_hRC));

    if (!MakeCurrent()) {
        char buf[128];
        snprintf(buf, sizeof(buf), "OpenGlContext: MakeCurrent failed, error=%lu", GetLastError());
        GlLog(buf);
        return;
    }

    // Detect software-only renderer (e.g. Microsoft GDI Generic on RDP).
    {
        PIXELFORMATDESCRIPTOR actualPfd{};
        actualPfd.nSize = sizeof(actualPfd);
        DescribePixelFormat(AsHDC(m_hDC), pf, sizeof(actualPfd), &actualPfd);
        m_IsSoftwareRenderer = ((actualPfd.dwFlags & PFD_GENERIC_FORMAT) != 0) &&
                               ((actualPfd.dwFlags & PFD_GENERIC_ACCELERATED) == 0);

        const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        const char *version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_MaxTextureSize);

        char buf[512];
        snprintf(buf,
                 sizeof(buf),
                 "OpenGlContext: renderer=%s, version=%s, maxTex=%d, pfdFlags=0x%lx, "
                 "software=%d, thread=%lu",
                 renderer ? renderer : "(null)",
                 version ? version : "(null)",
                 m_MaxTextureSize,
                 actualPfd.dwFlags,
                 m_IsSoftwareRenderer ? 1 : 0,
                 GetCurrentThreadId());
        GlLog(buf);
    }

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    NativeRect rt{};
    GetNativeClientRect(m_hWnd, rt);
    m_CachedRect = {static_cast<int32_t>(rt.left),
                    static_cast<int32_t>(rt.top),
                    static_cast<int32_t>(rt.right),
                    static_cast<int32_t>(rt.bottom)};
    glResetViewDim((size_t)rt.right, (size_t)rt.bottom);

    m_Valid = true;
}

OpenGlContext::~OpenGlContext()
{
    if (wglGetCurrentContext() == AsHGLRC(m_hRC)) {
        wglMakeCurrent(nullptr, nullptr);
    }

    if (m_hRC) {
        if (m_hWnd)
            UnregisterShareRoot(AsHWND(m_hWnd), AsHGLRC(m_hRC));

        wglDeleteContext(AsHGLRC(m_hRC));
        m_hRC = nullptr;
    }

    if (m_hWnd && m_hDC) {
        ReleaseDC(AsHWND(m_hWnd), AsHDC(m_hDC));
        m_hDC = nullptr;
    }
}

bool
OpenGlContext::MakeCurrent() noexcept
{
    if (!m_hDC || !m_hRC)
        return false;

    return wglMakeCurrent(AsHDC(m_hDC), AsHGLRC(m_hRC)) == TRUE;
}

void
OpenGlContext::SwapBuffers()
{
    if (m_hDC) {
        ::SwapBuffers(AsHDC(m_hDC));
    }
}

#else // !_WIN32 — GLX implementation

#include <GL/glx.h>

namespace {

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

} // namespace

OpenGlContext::OpenGlContext(void *nativeWindow) : m_hWnd(nativeWindow)
{
    m_Valid = false;

    if (!m_hWnd) {
        GlLog("OpenGlContext: null Window");
        return;
    }

    Display *dpy = GetX11Display();
    if (!dpy) {
        GlLog("OpenGlContext: failed to open X11 display");
        return;
    }

    Window win = reinterpret_cast<Window>(m_hWnd);

    // Choose a visual with RGBA, double-buffered, depth=0.
    int attribs[] = {GLX_RGBA,
                     GLX_DOUBLEBUFFER,
                     GLX_RED_SIZE,
                     8,
                     GLX_GREEN_SIZE,
                     8,
                     GLX_BLUE_SIZE,
                     8,
                     GLX_ALPHA_SIZE,
                     8,
                     None};

    XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attribs);
    if (!vi) {
        GlLog("OpenGlContext: glXChooseVisual failed");
        return;
    }

    GLXContext shareCtx = GetShareRoot(win);
    GLXContext ctx = glXCreateContext(dpy, vi, shareCtx, GL_TRUE);
    XFree(vi);

    if (!ctx) {
        GlLog("OpenGlContext: glXCreateContext failed");
        return;
    }

    m_hRC = reinterpret_cast<void *>(ctx);
    RegisterShareRoot(win, ctx);

    if (!MakeCurrent()) {
        GlLog("OpenGlContext: MakeCurrent failed");
        return;
    }

    // Detect software renderer via glXIsDirect.
    m_IsSoftwareRenderer = !glXIsDirect(dpy, ctx);

    const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const char *version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_MaxTextureSize);

    char buf[512];
    snprintf(buf,
             sizeof(buf),
             "OpenGlContext: renderer=%s, version=%s, maxTex=%d, direct=%d",
             renderer ? renderer : "(null)",
             version ? version : "(null)",
             m_MaxTextureSize,
             m_IsSoftwareRenderer ? 0 : 1);
    GlLog(buf);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    NativeRect rt{};
    GetNativeClientRect(m_hWnd, rt);
    m_CachedRect = {static_cast<int32_t>(rt.left),
                    static_cast<int32_t>(rt.top),
                    static_cast<int32_t>(rt.right),
                    static_cast<int32_t>(rt.bottom)};
    glResetViewDim((size_t)rt.right, (size_t)rt.bottom);

    m_Valid = true;
}

OpenGlContext::~OpenGlContext()
{
    Display *dpy = GetX11Display();
    GLXContext ctx = reinterpret_cast<GLXContext>(m_hRC);

    if (dpy && ctx && glXGetCurrentContext() == ctx) {
        glXMakeCurrent(dpy, None, nullptr);
    }

    if (dpy && ctx) {
        if (m_hWnd) {
            Window win = reinterpret_cast<Window>(m_hWnd);
            UnregisterShareRoot(win, ctx);
        }
        glXDestroyContext(dpy, ctx);
        m_hRC = nullptr;
    }
}

bool
OpenGlContext::MakeCurrent() noexcept
{
    Display *dpy = GetX11Display();
    if (!dpy || !m_hWnd || !m_hRC)
        return false;

    Window win = reinterpret_cast<Window>(m_hWnd);
    GLXContext ctx = reinterpret_cast<GLXContext>(m_hRC);
    return glXMakeCurrent(dpy, win, ctx) == True;
}

void
OpenGlContext::SwapBuffers()
{
    Display *dpy = GetX11Display();
    if (dpy && m_hWnd) {
        Window win = reinterpret_cast<Window>(m_hWnd);
        glXSwapBuffers(dpy, win);
    }
}

#endif // _WIN32

// ============================================================================
// Platform-independent GL methods
// ============================================================================

void
OpenGlContext::glResetView()
{
    if (!m_hWnd)
        return;

    NativeRect rt{};
    if (!GetNativeClientRect(m_hWnd, rt))
        return;

    if (rt.right != m_CachedRect.right || rt.bottom != m_CachedRect.bottom ||
        rt.left != m_CachedRect.left || rt.top != m_CachedRect.top) {

        glResetViewDim((size_t)rt.right, (size_t)rt.bottom);
        m_CachedRect = {static_cast<int32_t>(rt.left),
                        static_cast<int32_t>(rt.top),
                        static_cast<int32_t>(rt.right),
                        static_cast<int32_t>(rt.bottom)};
    }
}

void
OpenGlContext::glResetViewDim(size_t width, size_t height)
{
    if (!m_hWnd)
        return;

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

bool
OpenGlContext::IsValid() const
{
    return m_Valid;
}

bool
OpenGlContext::IsSoftwareRenderer() const
{
    return m_IsSoftwareRenderer;
}

GLint
OpenGlContext::GetMaxTextureSize() const
{
    return m_MaxTextureSize;
}

void
OpenGlContext::DrawGlBox()
{
    if (!m_hWnd || !MakeCurrent())
        return;

    glResetView();

    NativeRect rt{};
    GetNativeClientRect(m_hWnd, rt);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_LINES);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2i(0, 0);
    glVertex2i((int)rt.right, (int)rt.bottom);

    glVertex2i((int)rt.right, 0);
    glVertex2i(0, (int)rt.bottom);
    glEnd();

    glFlush();
    SwapBuffers();
}

void
OpenGlContext::SetRepaint(bool repaint)
{
    m_Repainting = repaint;
}

bool
OpenGlContext::GetRepaint() const
{
    return m_Repainting;
}

void
OpenGlContext::ToggleRepaint()
{
    m_Repainting = !m_Repainting;
}

void
OpenGlContext::DrawFractalShark(void *nativeWindow)
{
    if (!nativeWindow || !MakeCurrent())
        return;

    glResetView();

    WPngImage image{};
    image.loadImage("FractalShark.png");

    std::vector<uint8_t> imageBytes;
    imageBytes.resize((size_t)image.width() * (size_t)image.height() * 4);

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            auto pixel = image.get8(x, y);
            const size_t idx = ((size_t)y * (size_t)image.width() + (size_t)x) * 4;
            imageBytes[idx + 0] = pixel.r;
            imageBytes[idx + 1] = pixel.g;
            imageBytes[idx + 2] = pixel.b;
            imageBytes[idx + 3] = pixel.a;
        }
    }

    NativeRect windowDimensions{};
    GetNativeClientRect(nativeWindow, windowDimensions);

    const GLint scrnHeight = (GLint)windowDimensions.bottom;
    const GLint scrnWidth = (GLint)windowDimensions.right;

    GLuint texid = 0;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 (GLsizei)image.width(),
                 (GLsizei)image.height(),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 imageBytes.data());

    glColor4f(1.f, 1.f, 1.f, 1.f);

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2i(0, scrnHeight);
    glTexCoord2i(0, 1);
    glVertex2i(0, 0);
    glTexCoord2i(1, 1);
    glVertex2i(scrnWidth, 0);
    glTexCoord2i(1, 0);
    glVertex2i(scrnWidth, scrnHeight);
    glEnd();

    glFlush();
    glFinish();
    SwapBuffers();

    glDeleteTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
