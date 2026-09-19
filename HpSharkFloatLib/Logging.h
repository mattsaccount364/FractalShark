#pragma once

#include <cstdint>
#include <string_view>

namespace HpShark {

void LogCudaError(uint32_t errorCode, const char *file, int line);

void LogCudaError(std::string_view context, uint32_t errorCode, const char *file, int line);

} // namespace HpShark
