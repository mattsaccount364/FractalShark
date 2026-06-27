// OpenGLContext.cpp
#include "stdafx.h"

// clang-format off
#include "GlIncludes.h"
// clang-format on

#include "OpenGLContext.h"
#include "WPngImage/WPngImage.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

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

void
OpenGlContext::DrawFractalShark(void *nativeWindow)
{
    if (!nativeWindow || !MakeCurrent())
        return;

    glResetView();

    auto windowDimensions = m_NativeContext->GetClientRect();
    if (!windowDimensions)
        return;

    WPngImage image{};
    image.loadImage("FractalShark.png");

    std::vector<uint8_t> imageBytes;
    imageBytes.resize((size_t)image.width() * (size_t)image.height() * 4);

    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            auto pixel = image.get8(x, y);
            const size_t idx = ((size_t)y * (size_t)image.width() + (size_t)x) * 4;
            imageBytes[idx + 0] = pixel.r;
            imageBytes[idx + 1] = pixel.g;
            imageBytes[idx + 2] = pixel.b;
            imageBytes[idx + 3] = pixel.a;
        }
    }

    const GLint scrnHeight = static_cast<GLint>(windowDimensions->bottom);
    const GLint scrnWidth = static_cast<GLint>(windowDimensions->right);

    GLuint texid = 0;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA8,
                 (GLsizei)image.width(),
                 (GLsizei)image.height(),
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 imageBytes.data());

    glColor4f(1.f, 1.f, 1.f, 1.f);

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2i(0, scrnHeight);
    glTexCoord2i(0, 1);
    glVertex2i(0, 0);
    glTexCoord2i(1, 1);
    glVertex2i(scrnWidth, 0);
    glTexCoord2i(1, 0);
    glVertex2i(scrnWidth, scrnHeight);
    glEnd();

    glFlush();
    glFinish();
    SwapBuffers();

    glDeleteTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
