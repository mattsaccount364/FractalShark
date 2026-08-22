// Generates the checked-in CUDA explicit-instantiation units.

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Batch {
    std::string Name;
    std::string Header;
    std::string Instantiations;
    std::string Namespace;
};

struct ParameterGroup {
    std::string Name;
    std::string Tag;
    std::vector<std::string> Params;
};

static std::string
Trim(std::string value)
{
    const auto isSpace = [](unsigned char character) { return std::isspace(character) != 0; };
    while (!value.empty() && isSpace(static_cast<unsigned char>(value.front())))
        value.erase(value.begin());
    while (!value.empty() && isSpace(static_cast<unsigned char>(value.back())))
        value.pop_back();
    return value;
}

static std::string
AskLine(const std::string &prompt, const std::optional<std::string> &defaultValue)
{
    for (;;) {
        std::cout << prompt;
        if (defaultValue)
            std::cout << " [" << *defaultValue << ']';
        std::cout << ": ";

        std::string line;
        if (!std::getline(std::cin, line))
            return defaultValue.value_or("");
        line = Trim(line);
        if (line.empty() && defaultValue)
            return *defaultValue;
        if (!line.empty())
            return line;
    }
}

static bool
AskYesNo(const std::string &prompt, bool defaultValue)
{
    for (;;) {
        const std::string answer = AskLine(prompt + " (y/n)", defaultValue ? "Y" : "N");
        const char first = static_cast<char>(std::tolower(static_cast<unsigned char>(answer.front())));
        if (first == 'y')
            return true;
        if (first == 'n')
            return false;
    }
}

static void
WriteTextFile(const fs::path &path, const std::string &content, bool overwrite)
{
    if (fs::exists(path) && !overwrite) {
        std::cout << "SKIP (exists): " << path.string() << '\n';
        return;
    }
    fs::create_directories(path.parent_path());
    std::ofstream output{path, std::ios::binary};
    if (!output)
        throw std::runtime_error("Failed to open for write: " + path.string());
    output << content;
    std::cout << "WROTE: " << path.string() << '\n';
}

static std::vector<std::string>
MakeParams(const std::string &prefix)
{
    std::vector<std::string> params;
    params.reserve(12);
    for (int index = 1; index <= 12; ++index)
        params.push_back(prefix + std::to_string(index));
    return params;
}

static std::vector<ParameterGroup>
GetParameterGroups()
{
    return {
        {"P", "P", MakeParams("SharkParams")},
        {"NP", "NP", MakeParams("SharkParamsNP")},
        {"NR", "NR", MakeParams("SharkParamsNR")},
        {"Dbl", "Dbl", MakeParams("SharkParamsDbl")},
        {"Dbf", "Dbf", MakeParams("SharkParamsDbf")},
    };
}

static std::vector<Batch>
GetBatches()
{
    return {
        {"HpSharkFloat_Conversions",
         "../HpSharkFloat_cu.h",
         R"(template class HpSharkFloat<SharkFloatParams>;
template std::string Uint32ToMpf<SharkFloatParams>(
    const uint32_t *array, int32_t pow64Exponent, mpf_t &mpfValue);
template std::string MpfToString<SharkFloatParams>(const mpf_t mpfValue, size_t precInBits);)",
         ""},
        {"HpSharkReference",
         "../KernelInvokeReferencePerf_cu.h",
         R"(template std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                             const typename SharkFloatParams::Float hdrRadiusY,
                                             const mpf_t,
                                             const mpf_t,
                                             uint32_t);
template std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                             const typename SharkFloatParams::Float hdrRadiusY,
                                             const HpSharkFloat<SharkFloatParams> &xNum,
                                             const HpSharkFloat<SharkFloatParams> &yNum,
                                             uint32_t);
template std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams,
    const typename SharkFloatParams::Float hdrRadiusY,
    const HpSharkFloat<SharkFloatParams> &xNum,
    const HpSharkFloat<SharkFloatParams> &yNum,
    ReferencePreparedTables<SharkFloatParams> &preparedTables);
template void InvokeHpSharkReferenceKernel<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams,
    HpSharkReferenceResults<SharkFloatParams> &results,
    uint64_t numIters);
template void ShutdownHpSharkReferenceKernel<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams,
    HpSharkReferenceResults<SharkFloatParams> &results,
    DebugGpuCombo *debugResults);
template uint64_t EvaluateCriticalOrbitAndDerivs_GPU<SharkFloatParams>(
    const mpf_t, const mpf_t, uint64_t,
    mpf_t, mpf_t, mpf_t, mpf_t,
    HDRFloat<double> &, HDRFloat<double> &,
    const HpShark::LaunchParams &,
    ReferencePreparedTables<SharkFloatParams> *,
    uint64_t, bool (*)(), void (*)(uint64_t, void *), void *, uint64_t);
template uint64_t EvaluateCriticalOrbitAndDerivs_GPU<SharkFloatParams>(
    const mpf_t, const mpf_t, uint64_t,
    mpf_t, mpf_t, mpf_t, mpf_t,
    HDRFloat<double> &, HDRFloat<double> &,
    const HpShark::LaunchParams &,
    uint32_t, uint64_t, bool (*)(), void (*)(uint64_t, void *), void *, uint64_t);)",
         "HpShark"},
        {"ReferenceGpuLoop",
         "../KernelHpSharkReferenceOrbit_cu.h",
         R"(template void ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);
template void ComputeHpSharkReferenceSetup<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);)",
         ""},
        {"SharkNTT_Primitives",
         "../ReferenceNTT_cu.h",
         R"(template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(uint64_t a, uint64_t b);
template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t a, uint64_t b);
template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(uint64_t value);
template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value);
template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(uint64_t value);
template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value);
template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(uint64_t value, uint64_t exponent);
template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> &debugCombo, uint64_t value, uint64_t exponent);)",
         ""},
    };
}

static std::string
MakeBatchFile(const Batch &batch, const std::string &tag, const std::vector<std::string> &params)
{
    std::ostringstream output;
    output << "// Auto-generated explicit instantiation TU.\n"
              "// Batch: "
           << batch.Name << "\n// Tag: " << tag
           << "\n// This file is generated by HpSharkInstantiate.\n\n"
           << "#include \"" << batch.Header << "\"\n\n"
           << "#define ExplicitlyInstantiate(SharkFloatParams) \\\n";

    std::istringstream instantiations{batch.Instantiations};
    std::string line;
    while (std::getline(instantiations, line)) {
        line = Trim(line);
        if (!line.empty())
            output << "    " << line << " \\\n";
    }
    output << "    /* end */\n\n";
    if (!batch.Namespace.empty())
        output << "namespace " << batch.Namespace << " {\n";
    for (const std::string &param : params)
        output << "ExplicitlyInstantiate(" << param << ");\n";
    if (!batch.Namespace.empty())
        output << "} // namespace " << batch.Namespace << '\n';
    output << "\n#undef ExplicitlyInstantiate\n";
    return output.str();
}

} // namespace

int
main()
{
    try {
        const fs::path outputDirectory = AskLine("Output directory", "generated_inst");
        const bool overwrite = AskYesNo("Overwrite existing files?", true);
        const std::string baseName =
            AskLine("Base name for generated instantiation .cu files", "SharkExplicitInstantiate");

        const auto parameterGroups = GetParameterGroups();
        for (const Batch &batch : GetBatches()) {
            for (const ParameterGroup &group : parameterGroups) {
                WriteTextFile(outputDirectory / (baseName + '_' + batch.Name + '_' + group.Name + ".cu"),
                              MakeBatchFile(batch, group.Tag, group.Params),
                              overwrite);
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
