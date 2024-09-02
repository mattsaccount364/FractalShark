// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#pragma once

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

// Windows.h and STL workaround
#define NOMINMAX

// C4996:
// 1 > H:\Documents\Programming\FractalShark\FractalSharkLib\PerturbationResults.cpp(1669, 9) :
// warning C4996 : 'std::complex<Imagina::HRReal>::complex' : warning STL4037 : The effect of
// instantiating the template std::complex for any type other than float, double, or long
// double is unspecified.You can define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING to
// suppress this warning.
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING

// Windows Header Files:
#include <windows.h>
#include <Winuser.h>
#include <Windowsx.h>

#ifdef _DEBUG

//
// See MSDN for more information on debugging memory leaks.
// _CRTDBG_MAP_ALLOC is defined for use with the debug version of the C run-time libraries.
// 

//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
//#include <crtdbg.h>

// #define DEBUG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type

#define DEBUG_NEW new

constexpr bool FractalSharkDebug = true;

#else

#define DEBUG_NEW new
constexpr bool FractalSharkDebug = false;

#endif

// Define global operator delete to invoke free directly
void operator delete(void *ptr) noexcept;
void *operator new(size_t size);

// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#include <stdio.h>

#include <math.h>
#include <io.h>
#include <time.h>

#include <locale>
#include <codecvt>

#include <GL/gl.h>      /* OpenGL header file */
#include <GL/glu.h>     /* OpenGL utilities header file */

// Local Header Files
#include "OpenGLContext.h"
