#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Environment::LocalIpcDetail {

inline constexpr std::intptr_t InvalidNativeHandle = -1;

inline constexpr bool
IsValidHandle(std::intptr_t handle) noexcept
{
    return handle != InvalidNativeHandle;
}

inline std::string
SanitizeName(std::string_view name)
{
    std::string result;
    result.reserve(name.size());
    for (const char ch : name) {
        const bool valid = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                           (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.';
        result.push_back(valid ? ch : '_');
    }
    return result;
}

} // namespace Environment::LocalIpcDetail
