#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

class Fractal;

namespace RenderTests {
std::string Profile();
uint32_t MatrixDimension();
const std::filesystem::path &OutputDirectory();
void CheckChecksum(std::string_view caseId, std::span<const uint8_t> bytes);
bool CheckPng(std::string_view caseId,
              const std::filesystem::path &path,
              unsigned width,
              unsigned height);
bool SaveAndCheck(Fractal &fractal, std::string_view caseId);

class ScopedDirectory {
public:
    explicit ScopedDirectory(std::string_view caseId);
    ~ScopedDirectory();

private:
    std::filesystem::path m_Previous;
};
} // namespace RenderTests
