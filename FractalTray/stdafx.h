// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently,
// but are changed infrequently

#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#ifndef WINVER
#define WINVER 0x0A00
#endif

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif

#include <commctrl.h>
#include <shellapi.h>
#include <windows.h>
#include <windowsx.h>

#pragma comment(lib, "comctl32.lib")

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "DbgHeap.h"
#include "OpenGLContext.h"
