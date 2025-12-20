//
// Generates explicit-instantiation .cu files for a fixed parameter set.
//
// Fixed parameter set:
//   SharkParams1..SharkParams12
//   SharkParamsNP1..SharkParamsNP12
//
// This version hard-codes multiple "batches" of instantiations matching existing
// ExplicitlyInstantiate(...) patterns from the project.
//
// Output strategy:
//   For each batch, generate 2 .cu files:
//     - *_P.cu  contains SharkParams1..12 instantiations
//     - *_NP.cu contains SharkParamsNP1..12 instantiations
//   This is a good balance of parallelism vs object/link overhead.
//
// FIX: Some instantiations (e.g. InitHpSharkReferenceKernel) live in namespace HpShark.
// The generator now supports wrapping the generated instantiation calls in a namespace
// per-batch, so the explicit instantiations refer to the correct templates.

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

static std::string
Trim(std::string s)
{
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front())))
        s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back())))
        s.pop_back();
    return s;
}

static std::string
AskLine(const std::string &prompt, const std::optional<std::string> &def = std::nullopt)
{
    for (;;) {
        std::cout << prompt;
        if (def)
            std::cout << " [" << *def << "]";
        std::cout << ": ";

        std::string line;
        if (!std::getline(std::cin, line))
            return def.value_or("");
        line = Trim(line);

        if (line.empty() && def)
            return *def;
        if (!line.empty())
            return line;

        std::cout << "Please enter a value.\n";
    }
}

static bool
AskYesNo(const std::string &prompt, bool def)
{
    for (;;) {
        const std::string defStr = def ? "Y" : "N";
        std::string line = AskLine(prompt + " (y/n)", defStr);
        if (line.empty())
            return def;

        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(line[0])));
        if (c == 'y')
            return true;
        if (c == 'n')
            return false;

        std::cout << "Please answer y or n.\n";
    }
}

static void
WriteTextFile(const fs::path &p, const std::string &content, bool overwrite)
{
    if (fs::exists(p) && !overwrite) {
        std::cout << "SKIP (exists): " << p.string() << "\n";
        return;
    }
    fs::create_directories(p.parent_path());
    std::ofstream ofs(p, std::ios::binary);
    if (!ofs)
        throw std::runtime_error("Failed to open for write: " + p.string());
    ofs << content;
    ofs.close();
    std::cout << "WROTE: " << p.string() << "\n";
}

static std::vector<std::string>
MakeFixedParams_P()
{
    std::vector<std::string> v;
    v.reserve(12);
    for (int i = 1; i <= 12; ++i)
        v.push_back("SharkParams" + std::to_string(i));
    return v;
}

static std::vector<std::string>
MakeFixedParams_NP()
{
    std::vector<std::string> v;
    v.reserve(12);
    for (int i = 1; i <= 12; ++i)
        v.push_back("SharkParamsNP" + std::to_string(i));
    return v;
}

static std::vector<std::string>
MakeFixedParams_All()
{
    auto v = MakeFixedParams_P();
    auto np = MakeFixedParams_NP();
    v.insert(v.end(), np.begin(), np.end());
    return v;
}

static std::string
MakeInc(const std::vector<std::string> &params)
{
    std::ostringstream oss;
    oss << "// Auto-generated X-macro list.\n"
           "// No include guards on purpose.\n\n";
    for (const auto &p : params)
        oss << "X(" << p << ")\n";
    return oss.str();
}

// -----------------------------------------------------------------------------
// Hard-coded batches
// -----------------------------------------------------------------------------

struct Batch {
    std::string batchName;     // used in filename suffix
    std::string includeHeader; // header to include in generated instantiation TU
    std::string macroBody;     // lines that instantiate for SharkFloatParams
    std::string wrapNamespace; // "" = global, otherwise e.g. "HpShark"
};

static std::vector<Batch>
GetBatches()
{
    std::vector<Batch> b;

    // 1) HpSharkFloat + mpf conversions
    // (Assumed global; adjust wrapNamespace if these actually live in a namespace.)
    b.push_back(Batch{"HpSharkFloat_Conversions",
                      "..\\HpSharkFloat_cu.h",
                      R"(template class HpSharkFloat<SharkFloatParams>;
template std::string Uint32ToMpf<SharkFloatParams>(
    const uint32_t *array, int32_t pow64Exponent, mpf_t &mpf_val);
template std::string MpfToString<SharkFloatParams>(const mpf_t mpf_val, size_t precInBits);)",
                      ""});

    // 2) ComputeHpSharkReferenceGpuLoop
    // You call this unqualified inside namespace HpShark in your reference code; safest to wrap.
    b.push_back(Batch{"ReferenceGpuLoop",
                      "..\\KernelHpSharkReferenceOrbit_cu.h",
                      R"(template void ComputeHpSharkReferenceGpuLoop<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);)",
                      ""});

    // 3) HpSharkReference init/invoke/shutdown (definitely in namespace HpShark per your snippet)
    b.push_back(Batch{"HpSharkReference",
                      "..\\KernelInvokeReferencePerf_cu.h",
                      R"(template std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                             const typename SharkFloatParams::Float hdrRadiusY,
                                             const mpf_t,
                                             const mpf_t);
template void InvokeHpSharkReferenceKernel<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    uint64_t numIters);
template std::unique_ptr<HpSharkReferenceResults<SharkFloatParams>>
InitHpSharkReferenceKernel<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                             const typename SharkFloatParams::Float hdrRadiusY,
                                             const HpSharkFloat<SharkFloatParams> &xNum,
                                             const HpSharkFloat<SharkFloatParams> &yNum);
template void ShutdownHpSharkReferenceKernel<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams,
    HpSharkReferenceResults<SharkFloatParams> &combo,
    DebugGpuCombo *debugCombo);)",
                      "HpShark"});

    // 4) Add kernels (unknown namespace; leave global unless you know otherwise)
    b.push_back(Batch{
        "AddKernels",
        "..\\KernelTestAdd_cu.h",
        R"(template void ComputeAddGpu<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                                        void *kernelArgs[]);
template void ComputeAddGpuTestLoop<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                                      void *kernelArgs[]);)",
        ""});

    // 5) Multiply NTT kernels (unknown namespace; leave global unless you know otherwise)
    b.push_back(Batch{
        "MultiplyNTT",
        "..\\KernelTestMultiplyNTT_cu.h",
        R"(template void ComputeMultiplyNTTGpu<SharkFloatParams>(const HpShark::LaunchParams &launchParams,
                                                                void *kernelArgs[]);
template void ComputeMultiplyNTTGpuTestLoop<SharkFloatParams>(
    const HpShark::LaunchParams &launchParams, cudaStream_t &stream, void *kernelArgs[]);)",
        ""});

    // 6) SharkNTT primitives are already fully-qualified as SharkNTT::..., so no wrapper needed.
    b.push_back(Batch{
        "SharkNTT_Primitives",
        "..\\MultiplyNTTCudaSetup_cu.h",
        R"(template void SharkNTT::BuildRoots<SharkFloatParams>(uint32_t, uint32_t, SharkNTT::RootTables &);
template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(uint64_t a, uint64_t b);
template uint64_t SharkNTT::MontgomeryMul<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a, uint64_t b);
template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(uint64_t x);
template uint64_t SharkNTT::ToMontgomery<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x);
template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(uint64_t x);
template uint64_t SharkNTT::FromMontgomery<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t x);
template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(uint64_t a_mont, uint64_t e);
template uint64_t SharkNTT::MontgomeryPow<SharkFloatParams>(
    DebugHostCombo<SharkFloatParams> & debugCombo, uint64_t a_mont, uint64_t e);
template void SharkNTT::CopyRootsToCuda<SharkFloatParams>(SharkNTT::RootTables & outT,
                                                          const SharkNTT::RootTables &inT);
template void SharkNTT::DestroyRoots<SharkFloatParams>(bool cuda, SharkNTT::RootTables &T);)",
        ""});

    return b;
}

static std::string
MakeBatchCpp_ParamList(const Batch &batch,
                       const std::string &tag,
                       const std::vector<std::string> &params)
{
    std::ostringstream oss;
    oss << "// Auto-generated explicit instantiation TU.\n"
           "// Batch: "
        << batch.batchName
        << "\n"
           "// Tag: "
        << tag
        << "\n"
           "// This file is generated by gen_inst_files.cpp.\n\n";

    oss << "#include \"" << batch.includeHeader << "\"\n\n";

    // Define the instantiation macro for this batch
    oss << "#define ExplicitlyInstantiate(SharkFloatParams) \\\n";
    {
        std::istringstream in(batch.macroBody);
        std::string line;
        bool any = false;
        while (std::getline(in, line)) {
            line = Trim(line);
            if (line.empty())
                continue;
            oss << "    " << line << " \\\n";
            any = true;
        }
        if (!any) {
            oss << "    /* empty */\n";
        } else {
            // terminate macro cleanly
            oss << "    /* end */\n";
        }
    }
    oss << "\n";

    // Expand for the parameter list in the correct namespace, if needed
    if (!batch.wrapNamespace.empty()) {
        oss << "namespace " << batch.wrapNamespace << " {\n";
    }

    for (const auto &p : params) {
        oss << "ExplicitlyInstantiate(" << p << ");\n";
    }

    if (!batch.wrapNamespace.empty()) {
        oss << "} // namespace " << batch.wrapNamespace << "\n";
    }

    oss << "\n";
    oss << "#undef ExplicitlyInstantiate\n";

    return oss.str();
}

static std::string
MakeBatchCpp_CombinedParamLists(const Batch &batch,
                                const std::vector<std::string> &paramsP,
                                const std::vector<std::string> &paramsNP)
{
    std::ostringstream oss;
    oss << "// Auto-generated explicit instantiation TU.\n"
           "// Batch: "
        << batch.batchName
        << "\n"
           "// Tag: COMBINED (P + NP)\n"
           "// NOTE: This batch is generated in one TU to reduce file count.\n"
           "//       The corresponding *_NP.cu file is intentionally empty.\n"
           "// This file is generated by gen_inst_files.cpp.\n\n";

    oss << "#include \"" << batch.includeHeader << "\"\n\n";

    // Define the instantiation macro for this batch
    oss << "#define ExplicitlyInstantiate(SharkFloatParams) \\\n";
    {
        std::istringstream in(batch.macroBody);
        std::string line;
        bool any = false;
        while (std::getline(in, line)) {
            line = Trim(line);
            if (line.empty())
                continue;
            oss << "    " << line << " \\\n";
            any = true;
        }
        if (!any) {
            oss << "    /* empty */\n";
        } else {
            oss << "    /* end */\n";
        }
    }
    oss << "\n";

    // Expand for both parameter lists in the correct namespace, if needed
    if (!batch.wrapNamespace.empty()) {
        oss << "namespace " << batch.wrapNamespace << " {\n";
    }

    oss << "// ---- P params: SharkParams1..12 ----\n";
    for (const auto &p : paramsP) {
        oss << "ExplicitlyInstantiate(" << p << ");\n";
    }

    oss << "\n// ---- NP params: SharkParamsNP1..12 ----\n";
    for (const auto &p : paramsNP) {
        oss << "ExplicitlyInstantiate(" << p << ");\n";
    }

    if (!batch.wrapNamespace.empty()) {
        oss << "} // namespace " << batch.wrapNamespace << "\n";
    }

    oss << "\n";
    oss << "#undef ExplicitlyInstantiate\n";

    return oss.str();
}

static std::string
MakeEmptyNpStub(const Batch &batch)
{
    std::ostringstream oss;
    oss << "// Auto-generated stub TU.\n"
           "// Batch: "
        << batch.batchName
        << "\n"
           "// Tag: NP-STUB\n"
           "// Intentionally empty: NP instantiations for this batch are merged into the *_P.cu file.\n"
           "// This file exists to keep build scripts/project file lists unchanged.\n";
    return oss.str();
}


int
main()
{
    try {
        std::cout << "=== Explicit Instantiation File Generator (C++17) ===\n\n";
        std::cout << "Fixed parameter set:\n"
                     "  SharkParams1..SharkParams12\n"
                     "  SharkParamsNP1..SharkParamsNP12\n\n";
        std::cout << "Default: for each hard-coded batch, generates 2 .cu files (P and NP).\n"
                     "Optional: merge P+NP into *_P.cu and emit an empty *_NP.cu stub.\n\n";

        const std::string outDirStr = AskLine("Output directory", "generated_inst");
        const fs::path outDir = fs::path(outDirStr);

        const bool overwrite = AskYesNo("Overwrite existing files?", true);

        // Optional: still emit a single all-params .inc list for convenience/debugging
        const bool emitInc = AskYesNo("Also write a single *_All.inc param list file?", false);
        std::string incBase = "SharkParams";
        if (emitInc) {
            incBase = AskLine("Base name for .inc list (without extension)", "SharkParams");
        }

        const std::string cppBase =
            AskLine("Base name for generated instantiation .cu files", "SharkExplicitInstantiate");

        // Emit .inc list (optional)
        if (emitInc) {
            std::cout << "\n--- Generating .inc param list ---\n";
            const std::string allIncName = incBase + "_All.inc";
            WriteTextFile(outDir / allIncName, MakeInc(MakeFixedParams_All()), overwrite);
        }

        const bool mergeIntoP =
            AskYesNo("Merge P+NP instantiations into the _P file? (emit _NP as stub)", false);

        const auto paramsP = MakeFixedParams_P();
        const auto paramsNP = MakeFixedParams_NP();

        // Emit per-batch .cu files (P/NP)
        std::cout << "\n--- Generating batch instantiation .cu files ---\n";
        const auto batches = GetBatches();

        for (const auto &batch : batches) {
            const std::string cuNameP = cppBase + "_" + batch.batchName + "_P.cu";
            const std::string cuNameNP = cppBase + "_" + batch.batchName + "_NP.cu";

            if (mergeIntoP) {
                WriteTextFile(outDir / cuNameP,
                              MakeBatchCpp_CombinedParamLists(batch, paramsP, paramsNP),
                              overwrite);

                WriteTextFile(outDir / cuNameNP, MakeEmptyNpStub(batch), overwrite);
            } else {
                WriteTextFile(outDir / cuNameP,
                              MakeBatchCpp_ParamList(batch, "P (SharkParams1..12)", paramsP),
                              overwrite);

                WriteTextFile(outDir / cuNameNP,
                              MakeBatchCpp_ParamList(batch, "NP (SharkParamsNP1..12)", paramsNP),
                              overwrite);
            }
        }

        std::cout << "\nDone.\n";
        std::cout << "Add the generated .cu files to your build.\n";
        std::cout << "If you use unity/jumbo builds, exclude these instantiation TUs.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
