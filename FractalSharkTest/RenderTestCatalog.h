#pragma once

#include "Fractal.h"
#include "TestFramework.h"

#include <string>
#include <vector>

namespace RenderTests {
enum class Scenario { Basic, ReferenceSave, Compression, Reuse, Imagina, ReferenceBackend, HardView };

struct RenderCase {
    Scenario Kind = Scenario::Basic;
    RenderAlgorithm Algorithm;
    bool AutoGpu = false;
    size_t View = 0;
    IterTypeEnum Bits = IterTypeEnum::Bits32;
    AddPointOptions Storage = AddPointOptions::DontSave;
    RefOrbitCalc::PerturbationAlg Reference = RefOrbitCalc::PerturbationAlg::STPeriodicity;
    uint32_t Antialiasing = 1;
    uint32_t Step = 1;
    uint64_t IterationLimit = 0;
    int32_t Compression = 20;
    LAParameters::LADefaults LA = LAParameters::LADefaults::MaxAccuracy;
    LAParameters::LAThreadingAlgorithm Threading = LAParameters::LAThreadingAlgorithm::MultiThreaded;
    ImaginaSettings LoadSettings = ImaginaSettings::ConvertToCurrent;
    std::wstring Fixture;
    std::string Name;
    std::string DisabledReason;
};

const std::vector<RenderCase> &Cases();
void Execute(const RenderCase &test);
} // namespace RenderTests
