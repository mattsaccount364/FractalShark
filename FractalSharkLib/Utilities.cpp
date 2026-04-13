#include "stdafx.h"
#include "Utilities.h"

#include <filesystem>

namespace Utilities {
bool
FileExists(const wchar_t *filename)
{
    return std::filesystem::exists(filename);
}
} // namespace Utilities