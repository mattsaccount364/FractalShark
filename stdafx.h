// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#pragma once

// Windows.h and STL workaround
#define NOMINMAX

// C4996:
// 1 > H:\Documents\Programming\FractalShark\FractalSharkLib\PerturbationResults.cpp(1669, 9) :
// warning C4996 : 'std::complex<Imagina::HRReal>::complex' : warning STL4037 : The effect of
// instantiating the template std::complex for any type other than float, double, or long
// double is unspecified.You can define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING to
// suppress this warning.
#define _SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING

#include "DbgHeap.h"

// C RunTime Header Files
#include <malloc.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>

#include <codecvt>
#include <locale>

#include "Environment.h"
