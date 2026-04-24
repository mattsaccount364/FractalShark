#pragma once

// Convenience header for including OpenGL headers portably.
// On Windows, <GL/gl.h> requires <windows.h> to be included first.
//
// Kept separate from Environment.h because it transitively pulls in the full
// <windows.h> (with WIN32_LEAN_AND_MEAN), which would otherwise pollute every
// translation unit that includes Environment.h and conflict with files that
// rely on the non-lean windows.h (e.g. ntdll wrappers using NTSTATUS).

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/gl.h>
#include <GL/glu.h>
