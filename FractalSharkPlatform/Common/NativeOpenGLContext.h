#pragma once

#include "Environment.h"

#include <optional>

namespace Environment {

class NativeOpenGLContext {
public:
    explicit NativeOpenGLContext(void *nativeWindow);
    ~NativeOpenGLContext();

    NativeOpenGLContext &operator=(const NativeOpenGLContext &) = delete;
    NativeOpenGLContext(const NativeOpenGLContext &) = delete;
    NativeOpenGLContext &operator=(NativeOpenGLContext &&) = delete;
    NativeOpenGLContext(NativeOpenGLContext &&) = delete;

    bool IsValid() const noexcept;
    bool MakeCurrent() noexcept;
    void SwapBuffers() noexcept;

    std::optional<ScreenRect> GetClientRect() const noexcept;
    bool IsSoftwareRenderer() const noexcept;
    bool IsDoubleBuffered() const noexcept;
    int GetMaxTextureSize() const noexcept;

private:
    void *m_NativeWindow{};
    void *m_DeviceContext{};
    void *m_RenderContext{};
    bool m_Valid{};
    bool m_IsSoftwareRenderer{};
    bool m_IsDoubleBuffered{true};
    int m_MaxTextureSize{};
};

} // namespace Environment
