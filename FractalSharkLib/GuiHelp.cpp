#include "stdafx.h"

#include "GuiHelp.h"

#include "CommandCatalog.h"

namespace FractalShark {
namespace {

constexpr GuiHelpContent kViewsHelp{
    "Views",
    "Views\n\nThe purpose of these is simply to make it easy to navigate to\n"
    "some interesting locations.\n"};

constexpr GuiHelpContent kAlgorithmsHelp{
    "Algorithms",
    "Algorithms\n\n"
    "- As a general recommendation, choose AUTO.  Auto will render the fractal using direct "
    "32-bit evaluation at the lowest zoom depths. From 1e4 to 1e9, it uses perturbation + "
    "32-bit floating point. From 1e9 to 1e34, it uses perturbation + 32-bit + linear "
    "approximation.  Past that, it uses perturbation a 32-bit \"high dynamic range\" "
    "implementation, which simply stores the exponent in a separate integer.\n\n"
    "- If you try rendering \"hard\" points, you may find that the 32-bit implementations are "
    "not accurate enough.  In this case, you can try the 64-bit implementations.  You may also "
    "find the 2x32 implementations to be faster than the 1x64. Generally, it's probably easiest "
    "to use the 32-bit implementations, and only switch to the 64-bit implementations when you "
    "need to.\n\n"
    "Note that professional/high-end chips offer superior 64-bit performance, so if you have one "
    "of those, you may find that the 64-bit implementations work well.  Most consumer GPUs offer "
    "poor 64-bit performance (even RTX 4090, 5090 etc).\n"};

constexpr std::string_view kDirectControls =
    "\nDirect controls\n"
    "Arrow keys - Pan viewport 25% of the view. Shift+Arrow: 10%, Ctrl+Arrow: 50%\n"
    "Numpad + - Zoom in at center\n"
    "Numpad - - Zoom out at center\n"
    "Left click/drag - Zoom in\n"
    "Right click - popup menu\n"
    "CTRL - Press and hold to abort autozoom\n"
    "ALT - Press, click/drag to move window when in windowed mode\n";

std::wstring
WidenWithCrLf(std::string_view text)
{
    std::wstring result;
    result.reserve(text.size());
    for (const char ch : text) {
        if (ch == '\n') {
            result += L"\r\n";
        } else {
            result.push_back(static_cast<unsigned char>(ch));
        }
    }
    return result;
}

} // namespace

GuiHelpContent
GetGuiHelpContent(GuiHelpTopic topic) noexcept
{
    return topic == GuiHelpTopic::Views ? kViewsHelp : kAlgorithmsHelp;
}

std::string
BuildHotkeysHelpUtf8()
{
    std::string body = "Hotkeys\n\nCommand shortcuts\n";
    for (const Command &command : kCommands) {
        body += FormatHotKeyUtf8(command.hotkey);
        body += " - ";
        for (const wchar_t ch : command.label) {
            body.push_back(ch >= 0 && ch <= 0x7f ? static_cast<char>(ch) : '?');
        }
        body.push_back('\n');
    }
    body += kDirectControls;
    return body;
}

std::wstring
BuildHotkeysHelpWide()
{
    return WidenWithCrLf(BuildHotkeysHelpUtf8());
}

std::wstring
GuiHelpTitleWide(GuiHelpTopic topic)
{
    return WidenWithCrLf(GetGuiHelpContent(topic).Title);
}

std::wstring
GuiHelpBodyWide(GuiHelpTopic topic)
{
    return WidenWithCrLf(GetGuiHelpContent(topic).Body);
}

} // namespace FractalShark
