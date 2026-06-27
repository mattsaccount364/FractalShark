// OpenGLContext.h
#pragma once

#include "Environment.h"
#include "NativeOpenGLContext.h"

#include <memory>
#include <string_view>

// GL types needed in the header (avoids including GL/gl.h which needs windows.h on Win32)
using GLint = int;

// Append a diagnostic line to FractalShark_gl.log.
void GlLog(const char *msg);

struct OpenGlContext {
    void *m_NativeWindow{};
    std::unique_ptr<Environment::NativeOpenGLContext> m_NativeContext;
    bool m_Valid{};
    bool m_IsSoftwareRenderer{};
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
    void DrawGlBox(bool swapBuffers = true);
    void DrawFractalShark(void *nativeWindow);
};
