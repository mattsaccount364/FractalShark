#pragma once

// Convenience header for including OpenGL headers portably.
// On Windows, <GL/gl.h> requires <windows.h> to be included first.

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
