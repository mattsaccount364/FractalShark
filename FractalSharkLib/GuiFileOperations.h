#pragma once

#include <filesystem>
#include <string>
#include <vector>

class Fractal;
enum class CompressToDisk;
enum class ImaginaSettings;

namespace FractalShark {

enum class FractalOutputFile { CurrentImage, HighResolutionImage, IterationsText };

void SaveFractalOutput(Fractal &fractal, FractalOutputFile type, std::wstring filename);
void SaveReferenceOrbit(Fractal &fractal, CompressToDisk compression, std::wstring filename);
void DiffReferenceOrbits(Fractal &fractal, std::wstring output, std::wstring first, std::wstring second);
void LoadReferenceOrbit(Fractal &fractal,
                        CompressToDisk compression,
                        ImaginaSettings settings,
                        std::wstring filename);
std::vector<std::filesystem::path> FindReferenceOrbitFiles(const std::filesystem::path &directory,
                                                           size_t maximumCount);

} // namespace FractalShark
