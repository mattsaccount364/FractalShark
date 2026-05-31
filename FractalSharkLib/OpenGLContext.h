// OpenGLContext.h
#pragma once

#include "Environment.h"

#include <string_view>

// GL types needed in the header (avoids including GL/gl.h which needs windows.h on Win32)
using GLint = int;

// Append a diagnostic line to FractalShark_gl.log.
void GlLog(const char *msg);

struct OpenGlContext {
    void *m_hWnd{};
    void *m_hRC{};
    void *m_hDC{};
    bool m_Valid{};
    bool m_IsSoftwareRenderer{};
    bool m_IsDoubleBuffered{true};
    bool m_Repainting{true};
    GLint m_MaxTextureSize{};
    Environment::ScreenRect m_CachedRect{};

public:
    OpenGlContext(void *nativeWindow);
    ~OpenGlContext();

    bool MakeCurrent() noexcept;
    void SwapBuffers();

    void glResetView();
    void glResetViewDim(size_t width, size_t height);
    bool IsValid() const;
    static bool IsKnownSoftwareRendererName(std::string_view rendererName);
    bool IsSoftwareRenderer() const;
    GLint GetMaxTextureSize() const;
    void DrawGlBox();
    void SetRepaint(bool repaint);
    bool GetRepaint() const;
    void ToggleRepaint();
    void DrawFractalShark(void *nativeWindow);
};
