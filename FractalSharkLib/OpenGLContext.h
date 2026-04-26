// OpenGLContext.h
#pragma once

#include "Environment.h"

#include <functional>
#include <mutex>

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
    bool m_Repainting{true};
    GLint m_MaxTextureSize{};
    Environment::ScreenRect m_CachedRect{};

    // Overlay callback invoked from the GL consumer thread just before
    // SwapBuffers, while the GL context is current.  Used by the Linux
    // GUI to render an ImGui overlay (menus, modals, drag-zoom outline)
    // on top of the fractal canvas.  Default is empty (Win32 does not
    // install one and pays no overhead beyond a null check).
    using OverlayCallback = std::function<void()>;
    std::mutex m_OverlayMutex;
    OverlayCallback m_OverlayCallback;

public:
    OpenGlContext(void *nativeWindow);
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
    void DrawFractalShark(void *nativeWindow);

    // Install / remove the overlay callback.  Pass an empty std::function
    // to clear.  Safe to call from any thread.
    void SetOverlayCallback(OverlayCallback cb);
    // Invoke the overlay callback if one is installed.  Caller must
    // already have the GL context current.  No-op if no callback.
    void InvokeOverlayCallback() noexcept;
};
