#include "stdafx.h"
#include "OpenGLContext.h"
#include "WPngImage\WPngImage.hh"

OpenGlContext::OpenGlContext(HWND hWnd)
    : m_hWnd(hWnd)
    , m_hRC(nullptr)
    , m_hDC(nullptr)
    , m_Valid(false)
    , m_Repainting(true)
    , m_CachedRect{}
{
    if (m_hWnd) {
        m_hDC = GetDC(m_hWnd);
        PIXELFORMATDESCRIPTOR pfd;
        int pf;

        /* there is no guarantee that the contents of the stack that become
            the pfd are zeroed, therefore _make sure_ to clear these bits. */
        memset(&pfd, 0, sizeof(pfd));
        pfd.nSize = sizeof(pfd);
        pfd.nVersion = 1;
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; // | PFD_DOUBLEBUFFER;
        pfd.iPixelType = PFD_TYPE_RGBA;
        pfd.cColorBits = 32;

        pf = ChoosePixelFormat(m_hDC, &pfd);
        if (pf == 0)
        {
            MessageBox(nullptr, L"ChoosePixelFormat() failed: Cannot find a suitable pixel format.", L"Error", MB_OK | MB_APPLMODAL);
            m_Valid = false;
            return;
        }

        if (SetPixelFormat(m_hDC, pf, &pfd) == FALSE)
        {
            MessageBox(nullptr, L"SetPixelFormat() failed:  Cannot set format specified.", L"Error", MB_OK | MB_APPLMODAL);
            m_Valid = false;
            return;
        }

        DescribePixelFormat(m_hDC, pf, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

        m_hRC = wglCreateContext(m_hDC);
        if (m_hRC == nullptr) {
            m_Valid = false;
            return;
        }

        auto ret = wglMakeCurrent(m_hDC, m_hRC);
        if (ret == FALSE) {
            m_Valid = false;
            return;
        }

        glClearColor(0.0, 0.0, 0.0, 0.0);
        glShadeModel(GL_FLAT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        RECT rt;
        GetClientRect(m_hWnd, &rt);

        glResetViewDim(rt.right, rt.bottom);
    }

    m_Valid = true;
}

OpenGlContext::~OpenGlContext() {
    if (m_hWnd) {
        wglMakeCurrent(nullptr, nullptr);
        ReleaseDC(m_hWnd, m_hDC);
        wglDeleteContext(m_hRC);
    }
}

void OpenGlContext::glResetView()
{
    if (m_hWnd) {
        RECT rt;
        GetClientRect(m_hWnd, &rt);

        if (rt.right != m_CachedRect.right ||
            rt.bottom != m_CachedRect.bottom ||
            rt.left != m_CachedRect.left ||
            rt.top != m_CachedRect.top) {
            glResetViewDim(rt.right, rt.bottom);
            m_CachedRect = rt;
        }
    }
}

void OpenGlContext::glResetViewDim(size_t width, size_t height) {
    if (m_hWnd) {
        glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, static_cast<GLsizei>(width), 0.0, static_cast<GLsizei>(height));
        glMatrixMode(GL_MODELVIEW);
    }
}

bool OpenGlContext::IsValid() const {
    return m_Valid;
}

void OpenGlContext::DrawGlBox() {
    if (m_hWnd) {
        RECT rt;
        GetClientRect(m_hWnd, &rt);

        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_LINES);

        glColor3f(1.0, 1.0, 1.0);
        glVertex2i(0, 0);
        glVertex2i(rt.right, rt.bottom);

        glVertex2i(rt.right, 0);
        glVertex2i(0, rt.bottom);

        glEnd();
        glFlush();
    }
}

void OpenGlContext::SetRepaint(bool repaint) {
    m_Repainting = repaint;
}

bool OpenGlContext::GetRepaint() const {
    return m_Repainting;
}

void OpenGlContext::ToggleRepaint() {
    m_Repainting = !m_Repainting;
}

void OpenGlContext::DrawFractalShark(HWND hWnd) {
    WPngImage image{};
    image.loadImage("FractalShark.png");

    std::vector<uint8_t> imageBytes;
    imageBytes.resize(image.width() * image.height() * 4);

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            auto pixel = image.get8(x, y);
            imageBytes[(y * image.width() + x) * 4 + 0] = pixel.r;
            imageBytes[(y * image.width() + x) * 4 + 1] = pixel.g;
            imageBytes[(y * image.width() + x) * 4 + 2] = pixel.b;
            imageBytes[(y * image.width() + x) * 4 + 3] = pixel.a;
        }
    }

    RECT windowDimensions;
    GetClientRect(hWnd, &windowDimensions);

    GLuint texid;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap levels of a texture.
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // Change m_DrawOutBytes size if GL_RGBA is changed
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8,
        (GLsizei)image.width(), (GLsizei)image.height(), 0,
        GL_RGBA, GL_UNSIGNED_BYTE, imageBytes.data());

    const GLint scrnHeight = windowDimensions.bottom;
    const GLint scrnWidth = windowDimensions.right;

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, scrnHeight);
    glTexCoord2i(0, 1); glVertex2i(0, 0);
    glTexCoord2i(1, 1); glVertex2i(scrnWidth, 0);
    glTexCoord2i(1, 0); glVertex2i(scrnWidth, scrnHeight);
    glEnd();
    glFlush();
    glFinish();
    glDeleteTextures(1, &texid);
}
