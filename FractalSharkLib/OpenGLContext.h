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

public:
    OpenGlContext(HWND hWnd);
    ~OpenGlContext();

    void glResetView();
    void glResetViewDim(size_t width, size_t height);
    bool IsValid() const;
    void DrawGlBox();
    void SetRepaint(bool repaint);
    bool GetRepaint() const;
    void ToggleRepaint();
    void DrawFractalShark(HWND hWnd);
};