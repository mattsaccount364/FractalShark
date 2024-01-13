#pragma once

#include <GL/gl.h>			/* OpenGL header file */
#include <GL/glu.h>			/* OpenGL utilities header file */

struct OpenGlContext {
    HWND m_hWnd;
    HGLRC m_hRC;
    HDC m_hDC;
    bool m_Valid;
    bool m_Repainting;

    RECT m_CachedRect;

    OpenGlContext(HWND hWnd)
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

    ~OpenGlContext() {
        if (m_hWnd) {
            wglMakeCurrent(nullptr, nullptr);
            ReleaseDC(m_hWnd, m_hDC);
            wglDeleteContext(m_hRC);
        }
    }

    void glResetView()
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

    void glResetViewDim(size_t width, size_t height) {
        if (m_hWnd) {
            glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluOrtho2D(0.0, static_cast<GLsizei>(width), 0.0, static_cast<GLsizei>(height));
            glMatrixMode(GL_MODELVIEW);
        }
    }

    bool IsValid() const {
        return m_Valid;
    }

    void DrawGlBox() {
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

    void SetRepaint(bool repaint) {
        m_Repainting = repaint;
    }

    bool GetRepaint() const {
        return m_Repainting;
    }

    void ToggleRepaint() {
        m_Repainting = !m_Repainting;
    }
};