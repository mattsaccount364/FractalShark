#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace FractalSharkCli {

struct BatchImage {
    std::string Name;
    std::vector<std::string> Arguments;
    std::string Output;
    std::string Error;
};

struct BatchFile {
    std::vector<BatchImage> Images;
};

bool LoadBatchFile(const std::filesystem::path &path, BatchFile &batch, std::string &error);

} // namespace FractalSharkCli
