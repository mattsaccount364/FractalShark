#include "EarlyCommandLine.h"
#include "Environment.h"
#include <stdint.h>

#ifndef _MSC_VER
#define __forceinline inline __attribute__((always_inline))
#endif

#ifdef _WIN32
extern FancyHeap EnableFractalSharkHeap;
#else
// The custom FractalShark heap (HpSharkFloatLib/heap_allocator/HeapCpp.cpp) is
// Windows-only. On Linux allocations go through glibc malloc directly, but
// HasSafeModeFlag_NoCRT() and EarlyInit_SafeMode_NoCRT() still need a target
// for the flag. Provide a local definition here so the TU links standalone.
FancyHeap EnableFractalSharkHeap = FancyHeap::Unknown;
#endif

static __forceinline bool
IsSpaceW(wchar_t c)
{
    return c == L' ' || c == L'\t' || c == L'\r' || c == L'\n';
}

static __forceinline wchar_t
ToLowerAsciiW(wchar_t c)
{
    // good enough for ASCII flags; avoids CharLowerBuffW / CRT towlower
    if (c >= L'A' && c <= L'Z')
        return (wchar_t)(c + (L'a' - L'A'));
    return c;
}

static bool
TokenEqualsI(const wchar_t *tok, int tokLen, const wchar_t *lit)
{
    // Compare tok[0..tokLen) case-insensitively to null-terminated lit.
    int i = 0;
    for (; i < tokLen; i++) {
        wchar_t a = ToLowerAsciiW(tok[i]);
        wchar_t b = ToLowerAsciiW(lit[i]);
        if (b == 0)
            return false; // token longer than literal
        if (a != b)
            return false;
    }
    return lit[i] == 0; // must end exactly
}

bool
HasSafeModeFlag_NoCRT()
{
    const wchar_t *cmd = Environment::GetCommandLineWide();
    if (!cmd)
        return false;

    const wchar_t *p = cmd;

    while (*p) {
        // Skip whitespace
        while (*p && IsSpaceW(*p))
            p++;
        if (!*p)
            break;

        // Parse one token [start, end)
        const wchar_t *start = p;
        bool inQuote = false;

        if (*p == L'"') {
            inQuote = true;
            start = ++p;
            while (*p && *p != L'"')
                p++;
            // token is [start, p)
        } else {
            while (*p && !IsSpaceW(*p))
                p++;
            // token is [start, p)
        }

        const wchar_t *end = p;
        if (inQuote && *p == L'"')
            p++; // consume closing quote

        int len = (int)(end - start);
        if (len <= 0)
            continue;

        // Normalize optional prefix -, --, /
        // We compare against "safemode" with optional leading '-'/'/'.
        // Accept: safemode, -safemode, --safemode, /safemode
        const wchar_t *t = start;
        int tl = len;

        if (tl >= 1 && (t[0] == L'-' || t[0] == L'/')) {
            t++;
            tl--;
            if (tl >= 1 && t[0] == L'-') {
                t++;
                tl--;
            } // allow --
        }

        // Also allow "--safe-mode"?? (optional). If you want that, add another compare.
        if (TokenEqualsI(t, tl, L"safemode"))
            return true;
    }

    return false;
}


#ifndef _MSC_VER
#ifndef __cdecl
#define __cdecl
#endif
#endif

extern "C" void __cdecl EarlyInit_SafeMode_NoCRT()
{
    // No CRT calls here. No malloc/new. Just WinAPI + simple logic.
    if (HasSafeModeFlag_NoCRT()) {
        EnableFractalSharkHeap = FancyHeap::Disable;
    } else {
        EnableFractalSharkHeap = FancyHeap::Enable;
    }
}
