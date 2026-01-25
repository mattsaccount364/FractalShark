// OpenGLContext.cpp
#include "stdafx.h"
#include "OpenGLContext.h"
#include "WPngImage\WPngImage.hh"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// One-time (per HWND) pixel format setup + optional context sharing.
// This lets you create TWO OpenGlContext instances for the same HWND
// (e.g., one per thread) without calling SetPixelFormat twice.
struct SharedWindowGlState {
    int pixelFormat = 0;       // GetPixelFormat(hdc)
    HGLRC shareRoot = nullptr; // First context created for this HWND
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

    // If already known, trust it.
    if (st.pixelFormat != 0) {
        outPixelFormat = st.pixelFormat;
        return true;
    }

    // If the DC already has a pixel format, keep it.
    const int existing = GetPixelFormat(hdc);
    if (existing != 0) {
        st.pixelFormat = existing;
        outPixelFormat = existing;
        return true;
    }

    // Otherwise set it once.
    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; // (no double buffer per your code)
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
        // If SetPixelFormat fails, it may already be set on that DC; try reading it again.
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
    // Share objects (textures, display lists, etc.) with the first-created context for this HWND.
    // This is optional, but is usually what people want for "two threads, two contexts".
    HGLRC root = GetShareRoot(hWnd);
    if (!root || root == newRc)
        return;

    // wglShareLists returns FALSE on failure; not fatal for correctness of simple line drawing.
    if (wglShareLists(root, newRc) == FALSE) {
        // Keep it quiet; you can log if you want.
        // std::wcerr << L"wglShareLists failed.\n";
    }
}

} // namespace

OpenGlContext::OpenGlContext(HWND hWnd) : m_hWnd(hWnd)
{
    m_Valid = false;

    if (!m_hWnd)
        return;

    m_hDC = GetDC(m_hWnd);
    if (!m_hDC)
        return;

    int pf = 0;
    if (!EnsurePixelFormatSet(m_hWnd, m_hDC, pf)) {
        std::wcerr << L"EnsurePixelFormatSet failed.\n";
        return;
    }

    // Create a dedicated context for this instance (so you can have one per thread).
    m_hRC = wglCreateContext(m_hDC);
    if (!m_hRC)
        return;

    // Optional sharing with first context created for this HWND.
    // This is safe even if it fails; it just means no shared textures/lists.
    MaybeShareWithRoot(m_hWnd, m_hRC);
    RegisterShareRoot(m_hWnd, m_hRC);

    if (!MakeCurrent())
        return;

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    RECT rt{};
    GetClientRect(m_hWnd, &rt);
    m_CachedRect = rt;
    glResetViewDim((size_t)rt.right, (size_t)rt.bottom);

    m_Valid = true;
}

OpenGlContext::~OpenGlContext()
{
    // Only detach if THIS context is current on this thread.
    if (wglGetCurrentContext() == m_hRC) {
        wglMakeCurrent(nullptr, nullptr);
    }

    if (m_hRC) {
        wglDeleteContext(m_hRC);
        m_hRC = nullptr;
    }

    if (m_hWnd && m_hDC) {
        ReleaseDC(m_hWnd, m_hDC);
        m_hDC = nullptr;
    }
}

bool
OpenGlContext::MakeCurrent() noexcept
{
    if (!m_hDC || !m_hRC)
        return false;

    // If another context is current on this thread, this will switch it.
    return wglMakeCurrent(m_hDC, m_hRC) == TRUE;
}

void
OpenGlContext::glResetView()
{
    if (!m_hWnd)
        return;

    RECT rt{};
    GetClientRect(m_hWnd, &rt);

    if (rt.right != m_CachedRect.right || rt.bottom != m_CachedRect.bottom ||
        rt.left != m_CachedRect.left || rt.top != m_CachedRect.top) {

        glResetViewDim((size_t)rt.right, (size_t)rt.bottom);
        m_CachedRect = rt;
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
    // Bottom-left origin in GL coords:
    gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity(); // IMPORTANT: your old code omitted this.
}

bool
OpenGlContext::IsValid() const
{
    return m_Valid;
}

void
OpenGlContext::DrawGlBox()
{
    if (!m_hWnd || !MakeCurrent())
        return;

    glResetView();

    RECT rt{};
    GetClientRect(m_hWnd, &rt);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_LINES);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2i(0, 0);
    glVertex2i(rt.right, rt.bottom);

    glVertex2i(rt.right, 0);
    glVertex2i(0, rt.bottom);
    glEnd();

    glFlush();
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
OpenGlContext::DrawFractalShark(HWND hWnd)
{
    if (!hWnd || !MakeCurrent())
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

    RECT windowDimensions{};
    GetClientRect(hWnd, &windowDimensions);

    const GLint scrnHeight = windowDimensions.bottom;
    const GLint scrnWidth = windowDimensions.right;

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

    // In fixed function, make sure vertex color doesn't tint the texture.
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

    glDeleteTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
