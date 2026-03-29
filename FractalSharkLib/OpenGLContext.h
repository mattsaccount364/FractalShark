// OpenGLContext.h
#pragma once

#include <GL/gl.h>
#include <GL/glu.h>

// Append a diagnostic line to FractalShark_gl.log.
void GlLog(const char *msg);

struct OpenGlContext {
    HWND m_hWnd{};
    HGLRC m_hRC{};
    HDC m_hDC{};
    bool m_Valid{};
    bool m_IsSoftwareRenderer{};
    bool m_Repainting{true};
    GLint m_MaxTextureSize{};
    RECT m_CachedRect{};

public:
    OpenGlContext(HWND hWnd);
    ~OpenGlContext();

    bool MakeCurrent() noexcept;
    void SwapBuffers();

    void glResetView();
    void glResetViewDim(size_t width, size_t height);
    bool IsValid() const;
    bool IsSoftwareRenderer() const;
    GLint GetMaxTextureSize() const;
    void DrawGlBox();
    void SetRepaint(bool repaint);
    bool GetRepaint() const;
    void ToggleRepaint();
    void DrawFractalShark(HWND hWnd);
};
