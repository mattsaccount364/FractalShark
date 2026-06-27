#include "stdafx.h"

#include "GuiFileOperations.h"

#include "Exceptions.h"
#include "Fractal.h"
#include "RecommendedSettings.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <filesystem>

namespace FractalShark {
namespace {

bool
HasPathExtension(std::string_view filename)
{
    const size_t slash = filename.find_last_of("/\\");
    const size_t dot = filename.find_last_of('.');
    return dot != std::string_view::npos && (slash == std::string_view::npos || dot > slash);
}

bool
HasPathExtension(std::wstring_view filename)
{
    const size_t slash = filename.find_last_of(L"/\\");
    const size_t dot = filename.find_last_of(L'.');
    return dot != std::wstring_view::npos && (slash == std::wstring_view::npos || dot > slash);
}

void
RemoveExistingRegularFile(const std::wstring &filename)
{
    const std::filesystem::path path(filename);
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error)) {
        if (!std::filesystem::remove(path, error)) {
            throw FractalSharkSeriousException("Could not replace the selected output file");
        }
    }
    if (error) {
        throw FractalSharkSeriousException("Could not inspect the selected output file");
    }
}

} // namespace

std::string
MakeTimestampedOutputStem()
{
    std::time_t now = std::time(nullptr);
    std::tm localTime{};
#ifdef _MSC_VER
    localtime_s(&localTime, &now);
#else
    localtime_r(&now, &localTime);
#endif

    char buffer[64];
    std::snprintf(buffer,
                  sizeof(buffer),
                  "output_%04d_%02d_%02d_%02d_%02d_%02d",
                  localTime.tm_year + 1900,
                  localTime.tm_mon + 1,
                  localTime.tm_mday,
                  localTime.tm_hour,
                  localTime.tm_min,
                  localTime.tm_sec);
    return std::string(buffer);
}

std::string
AppendExtensionIfMissing(std::string filename, std::string_view extension)
{
    if (filename.empty() || extension.empty() || HasPathExtension(filename)) {
        return filename;
    }
    filename.append(extension);
    return filename;
}

std::wstring
AppendExtensionIfMissing(std::wstring filename, std::wstring_view extension)
{
    if (filename.empty() || extension.empty() || HasPathExtension(filename)) {
        return filename;
    }
    filename.append(extension);
    return filename;
}

std::vector<std::filesystem::path>
FindReferenceOrbitFiles(const std::filesystem::path &directory, size_t maximumCount)
{
    std::vector<std::filesystem::path> files;
    for (const std::filesystem::directory_entry &entry :
         std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (extension == ".im") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end(), [](const auto &left, const auto &right) {
        std::string leftName = left.filename().string();
        std::string rightName = right.filename().string();
        const auto lower = [](std::string &value) {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
        };
        lower(leftName);
        lower(rightName);
        return leftName < rightName;
    });
    if (files.size() > maximumCount) {
        files.resize(maximumCount);
    }
    return files;
}

void
SaveFractalOutput(Fractal &fractal, FractalOutputFile type, std::wstring filename)
{
    if (auto *pool = fractal.GetRenderPool()) {
        pool->Drain();
    }
    RemoveExistingRegularFile(filename);
    switch (type) {
        case FractalOutputFile::CurrentImage:
            fractal.SaveCurrentFractal(std::move(filename), true);
            return;
        case FractalOutputFile::HighResolutionImage:
            fractal.SaveHiResFractal(std::move(filename));
            return;
        case FractalOutputFile::IterationsText:
            fractal.SaveItersAsText(std::move(filename));
            return;
    }
    throw FractalSharkSeriousException("Unknown fractal output type");
}

void
SaveReferenceOrbit(Fractal &fractal, CompressToDisk compression, std::wstring filename)
{
    fractal.EnqueueCommand([compression, filename = std::move(filename)](Fractal &f) {
        f.SaveRefOrbit(compression, filename);
    });
}

void
DiffReferenceOrbits(Fractal &fractal, std::wstring output, std::wstring first, std::wstring second)
{
    fractal.EnqueueCommand(
        [output = std::move(output), first = std::move(first), second = std::move(second)](Fractal &f) {
            f.DiffRefOrbits(CompressToDisk::MaxCompressionImagina, output, first, second);
        });
}

void
LoadReferenceOrbit(Fractal &fractal,
                   CompressToDisk compression,
                   ImaginaSettings settings,
                   std::wstring filename)
{
    fractal.EnqueueCommand([compression, settings, filename = std::move(filename)](Fractal &f) {
        RecommendedSettings recommended{};
        f.LoadRefOrbit(&recommended, compression, settings, filename);
        if (recommended.GetRenderAlgorithm() == RenderAlgorithmEnum::AUTO &&
            !f.SetRenderAlgorithm(recommended.GetRenderAlgorithm())) {
            throw FractalSharkSeriousException("Failed to restore the reference-orbit render algorithm");
        }
    });
}

} // namespace FractalShark
