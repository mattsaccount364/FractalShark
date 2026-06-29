// OpenGLContext.cpp
#include "stdafx.h"

// clang-format off
#include "GlIncludes.h"
// clang-format on

#include "OpenGLContext.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <string_view>

bool
OpenGlContext::IsKnownSoftwareRendererName(std::string_view rendererName)
{
    constexpr std::array softwareRendererTokens{
        "llvmpipe", "softpipe", "swrast", "lavapipe", "software rasterizer"};

    return std::any_of(
        softwareRendererTokens.begin(), softwareRendererTokens.end(), [&](std::string_view token) {
            return std::search(rendererName.begin(),
                               rendererName.end(),
                               token.begin(),
                               token.end(),
                               [](char lhs, char rhs) {
                                   return std::tolower(static_cast<unsigned char>(lhs)) ==
                                          std::tolower(static_cast<unsigned char>(rhs));
                               }) != rendererName.end();
        });
}

void
GlLog(const char *msg)
{
    std::cerr << msg << std::endl;
}

OpenGlContext::OpenGlContext(void *nativeWindow)
    : m_NativeWindow(nativeWindow),
      m_NativeContext(std::make_unique<Environment::NativeOpenGLContext>(nativeWindow))
{
    if (!m_NativeContext || !m_NativeContext->IsValid())
        return;

    if (!MakeCurrent()) {
        GlLog("OpenGlContext: MakeCurrent failed after native context creation");
        return;
    }

    m_IsSoftwareRenderer = m_NativeContext->IsSoftwareRenderer();
    m_MaxTextureSize = static_cast<GLint>(m_NativeContext->GetMaxTextureSize());

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    auto rt = m_NativeContext->GetClientRect();
    if (!rt)
        return;

    m_CachedRect = *rt;
    glResetViewDim(static_cast<size_t>(rt->right), static_cast<size_t>(rt->bottom));

    m_Valid = true;
}

OpenGlContext::~OpenGlContext() = default;

bool
OpenGlContext::MakeCurrent() noexcept
{
    return m_NativeContext && m_NativeContext->MakeCurrent();
}

void
OpenGlContext::SwapBuffers()
{
    if (m_NativeContext) {
        m_NativeContext->SwapBuffers();
    }
}

void
OpenGlContext::glResetView()
{
    if (!m_NativeContext)
        return;

    auto rt = m_NativeContext->GetClientRect();
    if (!rt)
        return;

    if (rt->right != m_CachedRect.right || rt->bottom != m_CachedRect.bottom ||
        rt->left != m_CachedRect.left || rt->top != m_CachedRect.top) {

        glResetViewDim(static_cast<size_t>(rt->right), static_cast<size_t>(rt->bottom));
        m_CachedRect = *rt;
    }
}

void
OpenGlContext::glResetViewDim(size_t width, size_t height)
{
    if (!m_NativeContext || !m_NativeContext->IsValid())
        return;

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

bool
OpenGlContext::IsValid() const
{
    return m_Valid;
}

bool
OpenGlContext::IsSoftwareRenderer() const
{
    return m_IsSoftwareRenderer;
}

GLint
OpenGlContext::GetMaxTextureSize() const
{
    return m_MaxTextureSize;
}

void
OpenGlContext::DrawGlBox(bool swapBuffers)
{
    if (!MakeCurrent())
        return;

    glResetView();

    auto rt = m_NativeContext->GetClientRect();
    if (!rt)
        return;

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glClear(GL_COLOR_BUFFER_BIT);

    glBegin(GL_LINES);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2i(0, 0);
    glVertex2i(static_cast<int>(rt->right), static_cast<int>(rt->bottom));

    glVertex2i(static_cast<int>(rt->right), 0);
    glVertex2i(0, static_cast<int>(rt->bottom));
    glEnd();

    glFlush();
    if (swapBuffers) {
        SwapBuffers();
    }
}
