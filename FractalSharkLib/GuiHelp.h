#pragma once

#include <string>
#include <string_view>

namespace FractalShark {

enum class GuiHelpTopic { Views, Algorithms };

struct GuiHelpContent {
    std::string_view Title;
    std::string_view Body;
};

GuiHelpContent GetGuiHelpContent(GuiHelpTopic topic) noexcept;
std::string BuildHotkeysHelpUtf8();
std::wstring BuildHotkeysHelpWide();
std::wstring GuiHelpTitleWide(GuiHelpTopic topic);
std::wstring GuiHelpBodyWide(GuiHelpTopic topic);

} // namespace FractalShark
