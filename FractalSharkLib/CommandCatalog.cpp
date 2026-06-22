// CommandCatalog.cpp
// Formatting helpers for the portable command catalog.

#include "stdafx.h"

#include "CommandCatalog.h"

namespace FractalShark {

namespace {

char
NarrowHotKeyChar(wchar_t ch) noexcept
{
    return (ch >= 0 && ch <= 0x7f) ? static_cast<char>(ch) : '?';
}

wchar_t
ShiftedHotKeyGlyph(wchar_t key) noexcept
{
    if (key >= L'a' && key <= L'z') {
        return static_cast<wchar_t>(key - L'a' + L'A');
    }

    switch (key) {
        case L'=':
            return L'+';
        case L',':
            return L'<';
        case L'.':
            return L'>';
        default:
            return 0;
    }
}

} // namespace

std::wstring
FormatHotKey(HotKey hotkey)
{
    hotkey = NormalizeHotKey(hotkey);

    std::wstring text;
    if (hotkey.ctrl) {
        text += L"Ctrl+";
    }
    if (hotkey.alt) {
        text += L"Alt+";
    }

    if (hotkey.shift) {
        const wchar_t shiftedGlyph = ShiftedHotKeyGlyph(hotkey.key);
        if (!hotkey.ctrl && !hotkey.alt && shiftedGlyph != 0) {
            text.push_back(shiftedGlyph);
            return text;
        }
        text += L"Shift+";
    }

    text.push_back(hotkey.key);
    return text;
}

std::string
FormatHotKeyUtf8(HotKey hotkey)
{
    const std::wstring wide = FormatHotKey(hotkey);
    std::string text;
    text.reserve(wide.size());
    for (const wchar_t ch : wide) {
        text.push_back(NarrowHotKeyChar(ch));
    }
    return text;
}

} // namespace FractalShark
