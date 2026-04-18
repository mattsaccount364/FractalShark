#include "stdafx.h"
#include "Utilities.h"

#include <filesystem>

namespace Utilities {
bool
FileExists(const wchar_t *filename)
{
    std::error_code ec;
    auto st = std::filesystem::status(filename, ec);
    return !ec && std::filesystem::is_regular_file(st);
}
} // namespace Utilities