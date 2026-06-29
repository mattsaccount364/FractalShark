#include "NativeOpenGLContext.h"

// clang-format off
#include "GlIncludes.h"
// clang-format on

#include <cstdio>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace {

void
GlLog(const char *msg)
{
    std::cerr << msg << std::endl;
}

HWND
AsHWND(void *p)
{
    return static_cast<HWND>(p);
}

HDC
AsHDC(void *p)
{
    return static_cast<HDC>(p);
}

HGLRC
AsHGLRC(void *p) { return static_cast<HGLRC>(p); }

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
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
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

namespace Environment {

NativeOpenGLContext::NativeOpenGLContext(void *nativeWindow) : m_NativeWindow(nativeWindow)
{
    if (!m_NativeWindow) {
        GlLog("OpenGlContext: null HWND");
        return;
    }

    m_DeviceContext = GetDC(AsHWND(m_NativeWindow));
    if (!m_DeviceContext) {
        GlLog("OpenGlContext: GetDC failed");
        return;
    }

    int pixelFormat = 0;
    if (!EnsurePixelFormatSet(AsHWND(m_NativeWindow), AsHDC(m_DeviceContext), pixelFormat)) {
        GlLog("OpenGlContext: EnsurePixelFormatSet failed");
        return;
    }

    m_RenderContext = wglCreateContext(AsHDC(m_DeviceContext));
    if (!m_RenderContext) {
        char buf[128];
        snprintf(buf, sizeof(buf), "OpenGlContext: wglCreateContext failed, error=%lu", GetLastError());
        GlLog(buf);
        return;
    }

    MaybeShareWithRoot(AsHWND(m_NativeWindow), AsHGLRC(m_RenderContext));
    RegisterShareRoot(AsHWND(m_NativeWindow), AsHGLRC(m_RenderContext));

    if (!MakeCurrent()) {
        char buf[128];
        snprintf(buf, sizeof(buf), "OpenGlContext: MakeCurrent failed, error=%lu", GetLastError());
        GlLog(buf);
        return;
    }

    PIXELFORMATDESCRIPTOR actualPfd{};
    actualPfd.nSize = sizeof(actualPfd);
    DescribePixelFormat(AsHDC(m_DeviceContext), pixelFormat, sizeof(actualPfd), &actualPfd);
    m_IsSoftwareRenderer = ((actualPfd.dwFlags & PFD_GENERIC_FORMAT) != 0) &&
                           ((actualPfd.dwFlags & PFD_GENERIC_ACCELERATED) == 0);

    const char *renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
    const char *version = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &m_MaxTextureSize);

    char buf[512];
    snprintf(buf,
             sizeof(buf),
             "OpenGlContext: renderer=%s, version=%s, maxTex=%d, pfdFlags(base16)=0x%lx, "
             "software(bool)=%d, thread=%lu",
             renderer ? renderer : "(null)",
             version ? version : "(null)",
             m_MaxTextureSize,
             actualPfd.dwFlags,
             m_IsSoftwareRenderer ? 1 : 0,
             GetCurrentThreadId());
    GlLog(buf);

    m_Valid = true;
}

NativeOpenGLContext::~NativeOpenGLContext()
{
    if (wglGetCurrentContext() == AsHGLRC(m_RenderContext)) {
        wglMakeCurrent(nullptr, nullptr);
    }

    if (m_RenderContext) {
        if (m_NativeWindow)
            UnregisterShareRoot(AsHWND(m_NativeWindow), AsHGLRC(m_RenderContext));

        wglDeleteContext(AsHGLRC(m_RenderContext));
        m_RenderContext = nullptr;
    }

    if (m_NativeWindow && m_DeviceContext) {
        ReleaseDC(AsHWND(m_NativeWindow), AsHDC(m_DeviceContext));
        m_DeviceContext = nullptr;
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
    if (!m_DeviceContext || !m_RenderContext)
        return false;

    return wglMakeCurrent(AsHDC(m_DeviceContext), AsHGLRC(m_RenderContext)) == TRUE;
}

void
NativeOpenGLContext::SwapBuffers() noexcept
{
    glFlush();
}

std::optional<ScreenRect>
NativeOpenGLContext::GetClientRect() const noexcept
{
    RECT rt{};
    if (!m_NativeWindow || !::GetClientRect(AsHWND(m_NativeWindow), &rt))
        return std::nullopt;

    return ScreenRect{static_cast<int32_t>(rt.left),
                      static_cast<int32_t>(rt.top),
                      static_cast<int32_t>(rt.right),
                      static_cast<int32_t>(rt.bottom)};
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
