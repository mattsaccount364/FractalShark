#pragma once

#ifdef _DEBUG

//
// See MSDN for more information on debugging memory leaks.
// _CRTDBG_MAP_ALLOC is defined for use with the debug version of the C run-time libraries.
//

// #define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
// #include <crtdbg.h>

#define DEBUG_NEW new

constexpr bool FractalSharkDebug = true;

#else

#define DEBUG_NEW new
constexpr bool FractalSharkDebug = false;

#endif

// Define global operator delete to invoke free directly
void operator delete(void *ptr) noexcept;
void *operator new(size_t size);
