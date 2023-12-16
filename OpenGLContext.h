#pragma once

#include <GL/gl.h>			/* OpenGL header file */
#include <GL/glu.h>			/* OpenGL utilities header file */

struct OpenGlContext {
    HWND m_hWnd;
    HGLRC m_hRC;
    HDC m_hDC;
    bool m_Valid;

    void glResetViewDim(int width, int height) {
        glViewport(0, 0, width, height);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, width, 0.0, height);
        glMatrixMode(GL_MODELVIEW);
    }

    OpenGlContext(HWND hWnd)
        : m_hWnd(hWnd)
        , m_hRC(nullptr)
        , m_hDC(nullptr)
        , m_Valid(false)
    {
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
            MessageBox(NULL, L"ChoosePixelFormat() failed: Cannot find a suitable pixel format.", L"Error", MB_OK);
            m_Valid = false;
            return;
        }

        if (SetPixelFormat(m_hDC, pf, &pfd) == FALSE)
        {
            MessageBox(NULL, L"SetPixelFormat() failed:  Cannot set format specified.", L"Error", MB_OK);
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
        m_Valid = true;
    }

    bool IsValid() const {
        return m_Valid;
    }

    ~OpenGlContext() {
        wglMakeCurrent(NULL, NULL);
        ReleaseDC(m_hWnd, m_hDC);
        wglDeleteContext(m_hRC);
    }
};