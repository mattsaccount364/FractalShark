// FractalSharkCli: headless PNG renderer.
//
// The normal single-shot mode is also the client/server front end. Server
// mode keeps the expensive Fractal and CUDA/reference-orbit state alive while
// client mode forwards the usual render arguments over local IPC.

#include "stdafx.h"

#include "BatchFile.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "Fractal.h"
#include "FractalPalette.h"
#include "LocalIpc.h"
#include "PointZoomBBConverter.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "RenderThreadPool.h"
#include "RenderToConsole.h"
#include "RenderToPng.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "heap_allocator/include/HeapCpp.h"

namespace {

enum class ViewSource { None, Builtin, LocationsFile, Direct };
enum class CliMode { SingleShot, Server, Client };

constexpr int ConsoleWidth = 80;
constexpr int ConsoleHeight = 40;

struct CliArgs {
    int Width = 1024;
    int Height = 768;
    bool WidthSet = false;
    bool HeightSet = false;

    CliMode Mode = CliMode::SingleShot;
    std::string Endpoint;
    bool Shutdown = false;
    std::string BatchPath;
    bool BatchSet = false;

    ViewSource Source = ViewSource::None;
    size_t BuiltinView = 0;
    std::string LocationsFile;
    size_t LocationIndex = SIZE_MAX; // SIZE_MAX => use last record
    std::string CenterX, CenterY, Zoom;

    uint64_t Iterations = 0;              // 0 => unspecified (parser rejects 0)
    uint32_t Antialiasing = 0;            // 0 => unspecified (parser rejects 0)
    uint64_t CommitCapBytes = UINT64_MAX; // UINT64_MAX => unlimited
    GpuMode GpuRuntimeMode = GpuMode::Auto;

    std::string RenderAlgorithm;
    std::string PerturbationAlg; // empty => unspecified
    std::string PaletteMapFile;
    std::optional<uint32_t> PaletteDepth;

    std::string OutFile;

    bool ListRenderAlgorithms = false;
    bool Console = false;
    bool Color = false;
    bool Quiet = false;
    bool Help = false;
};

void
PrintUsage()
{
    std::cout
        << "FractalSharkCli — headless Mandelbrot renderer\n"
           "\n"
           "Usage:\n"
           "  FractalSharkCli --render-algorithm NAME [--out FILE.png] [--console] [--color]\n"
           "                  [--width W --height H]\n"
           "                  {--builtin-view N |\n"
           "                   --locations FILE [--location-index N] |\n"
           "                   --center-x X --center-y Y --zoom Z}\n"
           "                  [--iterations N] [--antialiasing N]\n"
           "                  [--perturbation-alg NAME] [--palette-map FILE] [--palette-depth BITS]\n"
           "                  [--commit-cap-bytes N]\n"
           "                  [--no-gpu]\n"
           "                  [--quiet]\n"
           "\n"
           "  FractalSharkCli --server [--endpoint NAME] [--width W --height H]\n"
           "                   [--commit-cap-bytes N] [--no-gpu]\n"
           "  FractalSharkCli --connect [--endpoint NAME] <the render arguments above>\n"
           "  FractalSharkCli --connect --endpoint NAME --shutdown\n"
           "  FractalSharkCli [--connect [--endpoint NAME]] --batch FILE\n"
           "\n"
           "  FractalSharkCli --list-render-algorithms\n"
           "  FractalSharkCli --help\n"
           "\n"
           "The default endpoint is per-user: a Windows named pipe or a Unix-domain\n"
           "socket. Use --endpoint to select a different local endpoint. The server\n"
           "process stays in the foreground and handles requests in FIFO order.\n"
           "\n"
           "Output:\n"
           "  --out FILE.png    Write a PNG image (required unless --console is given)\n"
           "  --console         Print ASCII art to stdout (can combine with --out)\n"
           "  --color           Use ANSI 256-color for console output (implies --console)\n"
           "  --palette-map FILE\n"
           "                    Load Fractal Zoomer RGB8 triplets; blank, #, and // lines are ignored\n"
           "                    (the first three values per row are clamped to 0-255)\n"
           "  --palette-depth BITS\n"
           "                    Use a 5, 6, 8, 12, 16, or 20-bit palette resolution (default: 8)\n"
           "  --quiet           Suppress progress and rendering details\n"
           "  Successful renders print the GUI rendering-details report.\n"
           "  Batch files use [defaults] and named [image NAME] sections with key = value\n"
           "  render options. Batch summaries include render and CLI image times.\n"
           "\n"
           "Per-pixel render algorithm names match RenderAlgorithmEnum\n"
           "(e.g. Cpu64PerturbedBLAV2HDR, Gpu1x32PerturbedLAv2, CpuHigh).\n"
           "Run with --list-render-algorithms for the full list.\n";
}

void
PrintRenderAlgorithms()
{
    for (const auto &alg : RenderAlgorithms) {
        if (alg.AlgorithmStr && alg.AlgorithmStr[0] != '\0') {
            std::cout << alg.AlgorithmStr << "\n";
        }
    }
}

std::optional<RenderAlgorithm>
ParseRenderAlgorithm(const std::string &name)
{
    for (const auto &alg : RenderAlgorithms) {
        if (alg.AlgorithmStr && name == alg.AlgorithmStr) {
            return alg;
        }
    }
    return std::nullopt;
}

std::optional<RefOrbitCalc::PerturbationAlg>
ParsePerturbationAlg(const std::string &name)
{
    if (name == "ST")
        return RefOrbitCalc::PerturbationAlg::ST;
    if (name == "MT")
        return RefOrbitCalc::PerturbationAlg::MT;
    if (name == "STPeriodicity")
        return RefOrbitCalc::PerturbationAlg::STPeriodicity;
    if (name == "MTPeriodicity3")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3;
    if (name == "MTPeriodicity3PerturbMTHighSTMed")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed;
    if (name == "MTPeriodicity3PerturbMTHighMTMed1")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1;
    if (name == "MTPeriodicity3PerturbMTHighMTMed2")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed2;
    if (name == "MTPeriodicity3PerturbMTHighMTMed3")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3;
    if (name == "MTPeriodicity3PerturbMTHighMTMed4")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4;
    if (name == "MTPeriodicity5")
        return RefOrbitCalc::PerturbationAlg::MTPeriodicity5;
    if (name == "GPU")
        return RefOrbitCalc::PerturbationAlg::GPU;
    if (name == "Auto")
        return RefOrbitCalc::PerturbationAlg::Auto;
    return std::nullopt;
}

bool
ParseUint64(const char *s, uint64_t &out)
{
    if (!s || !*s || s[0] == '-')
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long v = std::strtoull(s, &end, 10);
    if (errno || !end || *end != '\0')
        return false;
    out = static_cast<uint64_t>(v);
    return true;
}

bool
ParseSizeT(const char *s, size_t &out)
{
    uint64_t v;
    if (!ParseUint64(s, v) || v > std::numeric_limits<size_t>::max())
        return false;
    out = static_cast<size_t>(v);
    return true;
}

bool
ParseInt(const char *s, int &out)
{
    uint64_t v;
    if (!ParseUint64(s, v))
        return false;
    if (v > static_cast<uint64_t>(INT32_MAX))
        return false;
    out = static_cast<int>(v);
    return true;
}

bool
ParsePaletteDepth(const char *s, uint32_t &out)
{
    uint64_t depth;
    if (!ParseUint64(s, depth) || depth > UINT32_MAX) {
        return false;
    }

    const auto it =
        std::find(FractalPalette::PaletteDepths.begin(), FractalPalette::PaletteDepths.end(), depth);
    if (it == FractalPalette::PaletteDepths.end()) {
        return false;
    }

    out = static_cast<uint32_t>(depth);
    return true;
}

// Returns true on success, false on error (message already printed).
bool
ParseArgs(int argc, char *argv[], CliArgs &a, std::ostream &errorOut)
{
    auto expectValue = [&](int &i, const char *flag) -> const char * {
        if (i + 1 >= argc) {
            errorOut << "error: " << flag << " requires an argument\n";
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            a.Help = true;
        } else if (arg == "--list-render-algorithms") {
            a.ListRenderAlgorithms = true;
        } else if (arg == "--server") {
            if (a.Mode == CliMode::Client) {
                errorOut << "error: --server and --connect cannot be combined\n";
                return false;
            }
            a.Mode = CliMode::Server;
        } else if (arg == "--connect") {
            if (a.Mode == CliMode::Server) {
                errorOut << "error: --server and --connect cannot be combined\n";
                return false;
            }
            a.Mode = CliMode::Client;
        } else if (arg == "--endpoint") {
            auto v = expectValue(i, "--endpoint");
            if (!v)
                return false;
            a.Endpoint = v;
        } else if (arg == "--shutdown") {
            a.Shutdown = true;
        } else if (arg == "--batch") {
            auto v = expectValue(i, "--batch");
            if (!v)
                return false;
            a.BatchPath = v;
            a.BatchSet = true;
        } else if (arg == "--quiet") {
            a.Quiet = true;
        } else if (arg == "--no-gpu") {
            a.GpuRuntimeMode = GpuMode::Disabled;
        } else if (arg == "--console") {
            a.Console = true;
        } else if (arg == "--color") {
            a.Color = true;
            a.Console = true; // --color implies --console
        } else if (arg == "--width") {
            auto v = expectValue(i, "--width");
            if (!v || !ParseInt(v, a.Width)) {
                errorOut << "error: --width must be a positive integer\n";
                return false;
            }
            a.WidthSet = true;
        } else if (arg == "--height") {
            auto v = expectValue(i, "--height");
            if (!v || !ParseInt(v, a.Height)) {
                errorOut << "error: --height must be a positive integer\n";
                return false;
            }
            a.HeightSet = true;
        } else if (arg == "--render-algorithm") {
            auto v = expectValue(i, "--render-algorithm");
            if (!v)
                return false;
            a.RenderAlgorithm = v;
        } else if (arg == "--out") {
            auto v = expectValue(i, "--out");
            if (!v)
                return false;
            a.OutFile = v;
        } else if (arg == "--builtin-view") {
            auto v = expectValue(i, "--builtin-view");
            if (!v || !ParseSizeT(v, a.BuiltinView)) {
                errorOut << "error: --builtin-view must be a non-negative integer\n";
                return false;
            }
            a.Source = ViewSource::Builtin;
        } else if (arg == "--locations") {
            auto v = expectValue(i, "--locations");
            if (!v)
                return false;
            a.LocationsFile = v;
            a.Source = ViewSource::LocationsFile;
        } else if (arg == "--location-index") {
            auto v = expectValue(i, "--location-index");
            size_t idx;
            if (!v || !ParseSizeT(v, idx)) {
                errorOut << "error: --location-index must be a non-negative integer\n";
                return false;
            }
            a.LocationIndex = idx;
        } else if (arg == "--center-x") {
            auto v = expectValue(i, "--center-x");
            if (!v)
                return false;
            a.CenterX = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--center-y") {
            auto v = expectValue(i, "--center-y");
            if (!v)
                return false;
            a.CenterY = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--zoom") {
            auto v = expectValue(i, "--zoom");
            if (!v)
                return false;
            a.Zoom = v;
            a.Source = ViewSource::Direct;
        } else if (arg == "--iterations") {
            auto v = expectValue(i, "--iterations");
            uint64_t n;
            if (!v || !ParseUint64(v, n)) {
                errorOut << "error: --iterations must be a positive integer\n";
                return false;
            }
            if (n == 0) {
                errorOut << "error: --iterations must be > 0\n";
                return false;
            }
            a.Iterations = n;
        } else if (arg == "--antialiasing") {
            auto v = expectValue(i, "--antialiasing");
            uint64_t n;
            if (!v || !ParseUint64(v, n) || n > UINT32_MAX) {
                errorOut << "error: --antialiasing must be an integer from 1 to 4294967295\n";
                return false;
            }
            if (n == 0) {
                errorOut << "error: --antialiasing must be >= 1\n";
                return false;
            }
            a.Antialiasing = static_cast<uint32_t>(n);
        } else if (arg == "--commit-cap-bytes") {
            auto v = expectValue(i, "--commit-cap-bytes");
            uint64_t n;
            if (!v || !ParseUint64(v, n)) {
                errorOut << "error: --commit-cap-bytes must be a non-negative integer\n";
                return false;
            }
            a.CommitCapBytes = n;
        } else if (arg == "--perturbation-alg") {
            auto v = expectValue(i, "--perturbation-alg");
            if (!v)
                return false;
            a.PerturbationAlg = v;
        } else if (arg == "--palette-map") {
            auto v = expectValue(i, "--palette-map");
            if (!v)
                return false;
            a.PaletteMapFile = v;
        } else if (arg == "--palette-depth") {
            auto v = expectValue(i, "--palette-depth");
            uint32_t depth;
            if (!v || !ParsePaletteDepth(v, depth)) {
                errorOut << "error: --palette-depth must be one of 5, 6, 8, 12, 16, or 20\n";
                return false;
            }
            a.PaletteDepth = depth;
        } else {
            errorOut << "error: unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

bool
ParseArgs(const std::vector<std::string> &arguments, CliArgs &a, std::ostream &errorOut)
{
    std::vector<std::string> argvStorage;
    argvStorage.reserve(arguments.size() + 1);
    argvStorage.push_back("FractalSharkCli");
    argvStorage.insert(argvStorage.end(), arguments.begin(), arguments.end());

    std::vector<char *> argv;
    argv.reserve(argvStorage.size());
    for (auto &argument : argvStorage) {
        argv.push_back(argument.data());
    }
    return ParseArgs(static_cast<int>(argv.size()), argv.data(), a, errorOut);
}

// Simple saved-location record. Mirrors the format implemented by
// FractalSharkLib/SavedLocation.h:
//   width height minX minY maxX maxY num_iterations antialiasing
//   <description line>
struct ParsedSavedLocation {
    size_t Width = 0;
    size_t Height = 0;
    uint64_t NumIterations = 0;
    uint32_t Antialiasing = 0;
    HighPrecision MinX, MinY, MaxX, MaxY;
    std::string Description;
};

bool
LoadLocations(const std::string &path, std::vector<ParsedSavedLocation> &out, std::string &error)
{
    std::ifstream in(path);
    if (!in) {
        error = "cannot open locations file: " + path;
        return false;
    }

    while (in.good()) {
        ParsedSavedLocation rec;
        in >> rec.Width >> rec.Height;
        in >> rec.MinX >> rec.MinY >> rec.MaxX >> rec.MaxY;
        in >> rec.NumIterations >> rec.Antialiasing;
        if (!in.good())
            break;
        in >> std::ws;
        std::getline(in, rec.Description);
        out.push_back(std::move(rec));
    }
    if (out.empty()) {
        error = "locations file contains no records: " + path;
        return false;
    }
    return true;
}

std::wstring
ToWStringUtf8(const std::string &s)
{
    // argv on Linux is UTF-8; on Windows we're limited to MultiByte (ACP)
    // since the vcxproj builds with CharacterSet=MultiByte. Plain widening
    // is adequate for ASCII filenames which is all the CLI smoke tests use.
    std::wstring w;
    w.reserve(s.size());
    for (unsigned char c : s)
        w.push_back(static_cast<wchar_t>(c));
    return w;
}

bool
ParseHighPrecision(const std::string &text,
                   HighPrecision &value,
                   const char *flag,
                   bool requirePositive,
                   std::string &error)
{
    const size_t exponentMarker = text.find_first_of("eE@");
    if (exponentMarker != std::string::npos) {
        std::string_view exponent{text.data() + exponentMarker + 1, text.size() - exponentMarker - 1};
        if (!exponent.empty() && (exponent.front() == '+' || exponent.front() == '-')) {
            exponent.remove_prefix(1);
        }
        if (exponent.empty() || !std::all_of(exponent.begin(), exponent.end(), [](char digit) {
                return digit >= '0' && digit <= '9';
            })) {
            error = std::string(flag) + " exponent must be an integer";
            return false;
        }

        errno = 0;
        std::strtol(text.c_str() + exponentMarker + 1, nullptr, 10);
        if (errno == ERANGE) {
            error = std::string(flag) + " exponent is out of range";
            return false;
        }
    }

    HighPrecision parsed;
    if (mpf_set_str(parsed.backend(), text.c_str(), 10) != 0) {
        error = std::string(flag) + " is not a valid high-precision decimal";
        return false;
    }
    MpfNormalize(parsed.backend());
    if (requirePositive && mpf_sgn(parsed.backend()) <= 0) {
        error = std::string(flag) + " must be greater than zero";
        return false;
    }
    value = std::move(parsed);
    return true;
}

bool
ValidateRenderArgs(const CliArgs &args, std::string &error, bool fromClient)
{
    if (fromClient) {
        if (args.Mode != CliMode::Client) {
            error = "--connect is required for a client render request";
            return false;
        }
    } else if (args.Mode != CliMode::SingleShot) {
        error = "render requests cannot contain --server or --connect";
        return false;
    }
    if (args.Help || args.ListRenderAlgorithms) {
        error = "--help and --list-render-algorithms are not render requests";
        return false;
    }
    if (args.RenderAlgorithm.empty()) {
        error = "--render-algorithm is required";
        return false;
    }
    if (args.OutFile.empty() && !args.Console) {
        error = "--out is required (unless --console is given)";
        return false;
    }
    if (args.Source == ViewSource::None) {
        error = "one of --builtin-view, --locations, or --center-x/--center-y/--zoom is required";
        return false;
    }
    if (args.Source == ViewSource::Direct &&
        (args.CenterX.empty() || args.CenterY.empty() || args.Zoom.empty())) {
        error = "--center-x, --center-y, and --zoom must be specified together";
        return false;
    }
    if (args.Width <= 0 || args.Height <= 0) {
        error = "--width and --height must be positive";
        return false;
    }
    if (fromClient && args.CommitCapBytes != UINT64_MAX) {
        error = "--commit-cap-bytes is a server startup option; put it on --server";
        return false;
    }
    if (fromClient && args.GpuRuntimeMode == GpuMode::Disabled) {
        error = "--no-gpu is a server startup option; put it on --server";
        return false;
    }
    if (!ParseRenderAlgorithm(args.RenderAlgorithm)) {
        error = "unknown render algorithm: " + args.RenderAlgorithm +
                "\n(run --list-render-algorithms for valid names)";
        return false;
    }
    if (!args.PerturbationAlg.empty() && !ParsePerturbationAlg(args.PerturbationAlg)) {
        error = "unknown perturbation algorithm: " + args.PerturbationAlg;
        return false;
    }
    return true;
}

bool
ValidateServerArgs(const CliArgs &args, std::string &error)
{
    if (args.Mode != CliMode::Server) {
        error = "--server is required";
        return false;
    }
    if (args.Shutdown) {
        error = "--shutdown is only valid with --connect";
        return false;
    }
    if (args.Width <= 0 || args.Height <= 0) {
        error = "server --width and --height must be positive";
        return false;
    }
    if (!args.RenderAlgorithm.empty() || args.Source != ViewSource::None || !args.OutFile.empty() ||
        args.Console || args.Color || args.Iterations != 0 || args.Antialiasing != 0 ||
        !args.PerturbationAlg.empty() || !args.PaletteMapFile.empty() || args.PaletteDepth.has_value() ||
        args.LocationIndex != SIZE_MAX) {
        error = "server accepts only --endpoint, --width, --height, --commit-cap-bytes, --no-gpu, and "
                "--quiet";
        return false;
    }
    return true;
}

bool
ValidateShutdownArgs(const CliArgs &args, std::string &error)
{
    if (args.Mode != CliMode::Client) {
        error = "--shutdown requires --connect";
        return false;
    }
    if (args.GpuRuntimeMode == GpuMode::Disabled) {
        error = "--no-gpu is a server startup option; put it on --server";
        return false;
    }
    if (!args.RenderAlgorithm.empty() || args.Source != ViewSource::None || !args.OutFile.empty() ||
        args.Console || args.Color || args.WidthSet || args.HeightSet || args.Iterations != 0 ||
        args.Antialiasing != 0 || args.CommitCapBytes != UINT64_MAX || !args.PerturbationAlg.empty() ||
        !args.PaletteMapFile.empty() || args.PaletteDepth.has_value() ||
        args.LocationIndex != SIZE_MAX) {
        error = "--shutdown cannot be combined with render arguments";
        return false;
    }
    return true;
}

bool
ValidateBatchArgs(const CliArgs &args, std::string &error)
{
    if (args.BatchPath.empty()) {
        error = "--batch requires a nonempty file path";
        return false;
    }
    if (args.Mode == CliMode::Server || args.Shutdown) {
        error = "--batch cannot be combined with --server or --shutdown";
        return false;
    }
    if (args.Mode == CliMode::SingleShot && !args.Endpoint.empty()) {
        error = "--endpoint requires --connect";
        return false;
    }
    if (!args.RenderAlgorithm.empty() || args.Source != ViewSource::None || !args.OutFile.empty() ||
        args.Console || args.Color || args.WidthSet || args.HeightSet || args.Iterations != 0 ||
        args.Antialiasing != 0 || args.CommitCapBytes != UINT64_MAX ||
        args.GpuRuntimeMode == GpuMode::Disabled || !args.PerturbationAlg.empty() ||
        !args.PaletteMapFile.empty() || args.PaletteDepth.has_value() ||
        args.LocationIndex != SIZE_MAX || args.Quiet) {
        error = "render options for --batch belong in the batch file";
        return false;
    }
    return true;
}

int
BuildRenderRequest(const CliArgs &args,
                   RenderRequest &req,
                   std::string &error,
                   int defaultWidth,
                   int defaultHeight,
                   uint64_t commitCapBytes)
{
    // RecenterViewCalc lowers the MPIR default to the current view's working
    // precision. Each new location must be parsed at full input precision.
    HighPrecision::defaultPrecisionInBits(FractalLimits::MaxPrecisionLame);

    auto parsedAlg = ParseRenderAlgorithm(args.RenderAlgorithm);
    if (!parsedAlg) {
        error = "unknown render algorithm: " + args.RenderAlgorithm;
        return 2;
    }

    int width = args.WidthSet ? args.Width : defaultWidth;
    int height = args.HeightSet ? args.Height : defaultHeight;
    req.Width = width;
    req.Height = height;
    req.CommitCapBytes = commitCapBytes;
    req.Algorithm = *parsedAlg;
    req.Iterations = args.Iterations;
    req.Antialiasing = args.Antialiasing;
    req.Quiet = args.Quiet;
    req.PreserveIterationBuffer = args.Console;

    std::vector<ParsedSavedLocation> locations;
    const ParsedSavedLocation *loc = nullptr;
    if (args.Source == ViewSource::LocationsFile) {
        if (!LoadLocations(args.LocationsFile, locations, error)) {
            return 1;
        }
        size_t idx = (args.LocationIndex == SIZE_MAX) ? locations.size() - 1 : args.LocationIndex;
        if (idx >= locations.size()) {
            error = "--location-index " + std::to_string(idx) + " out of range (file has " +
                    std::to_string(locations.size()) + " records)";
            return 2;
        }
        loc = &locations[idx];
        if (!args.WidthSet) {
            if (loc->Width > static_cast<size_t>(INT32_MAX)) {
                error = "saved location width is too large";
                return 2;
            }
            width = static_cast<int>(loc->Width);
        }
        if (!args.HeightSet) {
            if (loc->Height > static_cast<size_t>(INT32_MAX)) {
                error = "saved location height is too large";
                return 2;
            }
            height = static_cast<int>(loc->Height);
        }
    }

    // Console-only: use console dimensions for the fractal computation
    // instead of the default 1024x768. This avoids computing far more pixels
    // than are needed. 80x40 gives correct visual proportions because
    // terminal characters are roughly 2:1 (height:width).
    const bool consoleOnly = args.Console && args.OutFile.empty();
    if (consoleOnly && !args.WidthSet && !args.HeightSet) {
        width = ConsoleWidth;
        height = ConsoleHeight;
    }
    if (width <= 0 || height <= 0) {
        error = "render width and height must be positive";
        return 2;
    }
    req.Width = width;
    req.Height = height;

    switch (args.Source) {
        case ViewSource::Builtin:
            req.ViewSource = RenderRequest::ViewSourceKind::Builtin;
            req.BuiltinView = args.BuiltinView;
            break;
        case ViewSource::LocationsFile:
            req.ViewSource = RenderRequest::ViewSourceKind::BoundingBox;
            req.MinX = loc->MinX;
            req.MinY = loc->MinY;
            req.MaxX = loc->MaxX;
            req.MaxY = loc->MaxY;
            if (args.Iterations == 0)
                req.Iterations = loc->NumIterations;
            if (args.Antialiasing == 0)
                req.Antialiasing = loc->Antialiasing;
            break;
        case ViewSource::Direct:
            req.ViewSource = RenderRequest::ViewSourceKind::Direct;
            if (!ParseHighPrecision(args.CenterX, req.CenterX, "--center-x", false, error) ||
                !ParseHighPrecision(args.CenterY, req.CenterY, "--center-y", false, error) ||
                !ParseHighPrecision(args.Zoom, req.Zoom, "--zoom", true, error)) {
                return 2;
            }
            break;
        case ViewSource::None:
            error = "render view source is missing";
            return 2;
    }

    if (!args.PerturbationAlg.empty()) {
        auto perturbation = ParsePerturbationAlg(args.PerturbationAlg);
        if (!perturbation) {
            error = "unknown perturbation algorithm: " + args.PerturbationAlg;
            return 2;
        }
        req.Perturbation = *perturbation;
    }

    if (!args.OutFile.empty()) {
        req.OutPngBasename = ToWStringUtf8(args.OutFile);
        const std::wstring pngExtension = L".png";
        if (req.OutPngBasename.size() >= pngExtension.size() &&
            req.OutPngBasename.compare(req.OutPngBasename.size() - pngExtension.size(),
                                       pngExtension.size(),
                                       pngExtension) == 0) {
            req.OutPngBasename.resize(req.OutPngBasename.size() - pngExtension.size());
        }
    }

    return 0;
}

int
ExecuteRenderRequest(const CliArgs &args,
                     const RenderRequest &req,
                     Fractal &fractal,
                     std::ostream &out,
                     std::ostream &errorOut)
{
    // Single-shot construction of Fractal initializes its default view after
    // parsing the request, which also lowers the MPIR default precision.
    HighPrecision::defaultPrecisionInBits(FractalLimits::MaxPrecisionLame);

    if (!args.PaletteMapFile.empty()) {
        fractal.LoadCustomPalette(std::filesystem::path(args.PaletteMapFile));
    }
    if (args.PaletteDepth) {
        fractal.UsePalette(static_cast<int>(*args.PaletteDepth));
    }

    std::string error;
    int rc = RenderToPng(req, fractal, &error, out);
    if (rc != 0) {
        errorOut << "error: " << error << "\n";
        return rc;
    }

    if (args.Console) {
        ConsoleRenderOptions consoleOpts;
        consoleOpts.ConsoleWidth = ConsoleWidth;
        consoleOpts.ConsoleHeight = ConsoleHeight;
        consoleOpts.Color = args.Color;
        RenderToConsole(fractal, consoleOpts, out);
    }
    if (!args.Quiet && !args.OutFile.empty()) {
        out << "Wrote " << args.OutFile << "\n";
        out.flush();
    }
    if (!args.Quiet) {
        std::string renderDetailsShort, renderDetailsLong;
        fractal.GetRenderDetails(renderDetailsShort, renderDetailsLong, false);
        std::erase(renderDetailsShort, '\r');
        out << renderDetailsShort;
        out.flush();
    }
    return 0;
}

int
ExecuteServerRender(const CliArgs &args,
                    Fractal &fractal,
                    std::ostream &out,
                    std::ostream &errorOut,
                    int defaultWidth,
                    int defaultHeight,
                    uint64_t commitCapBytes)
{
    RenderRequest req;
    std::string error;
    int rc = BuildRenderRequest(args, req, error, defaultWidth, defaultHeight, commitCapBytes);
    if (rc != 0) {
        errorOut << "error: " << error << "\n";
        return rc;
    }
    req.PngCompletion = PngCompletionMode::Background;

    return ExecuteRenderRequest(args, req, fractal, out, errorOut);
}

void
ResetBatchRenderSettings(Fractal &fractal)
{
    fractal.GetRenderPool()->Drain();
    fractal.ResetNumIterations();
    fractal.ResetDimensions(SIZE_MAX, SIZE_MAX, 1);
    fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Intermediate);
    fractal.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxAccuracy);
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto);
    if (fractal.GetPaletteType() != FractalPaletteType::Default) {
        fractal.UsePaletteType(FractalPaletteType::Default);
    }
    if (fractal.GetPaletteDepth() != FractalPalette::DefaultPaletteDepth) {
        fractal.UsePalette(static_cast<int>(FractalPalette::DefaultPaletteDepth));
    }
}

FractalSharkCli::IpcBatchResult
ExecuteBatchImage(const std::vector<std::string> &arguments,
                  Fractal &fractal,
                  int defaultWidth,
                  int defaultHeight,
                  uint64_t commitCapBytes,
                  bool serverMode)
{
    const auto started = std::chrono::steady_clock::now();
    FractalSharkCli::IpcBatchResult result;
    std::ostringstream output;
    std::ostringstream errors;
    CliArgs args;
    try {
        if (!ParseArgs(arguments, args, errors)) {
            result.Response.Status = 2;
        } else {
            std::string validationError;
            if (!ValidateRenderArgs(args, validationError, false)) {
                errors << "error: " << validationError << "\n";
                result.Response.Status = 2;
            } else if (serverMode && args.GpuRuntimeMode == GpuMode::Disabled) {
                errors << "error: --no-gpu is a server startup option; put it on --server\n";
                result.Response.Status = 2;
            } else if (serverMode && args.CommitCapBytes != UINT64_MAX &&
                       args.CommitCapBytes != commitCapBytes) {
                errors << "error: request commit cap does not match server startup cap\n";
                result.Response.Status = 2;
            } else {
                ResetBatchRenderSettings(fractal);
                RenderRequest request;
                std::string requestError;
                result.Response.Status = BuildRenderRequest(
                    args, request, requestError, defaultWidth, defaultHeight, commitCapBytes);
                if (result.Response.Status != 0) {
                    errors << "error: " << requestError << "\n";
                } else {
                    request.PngCompletion = PngCompletionMode::Background;
                    result.Width = static_cast<uint32_t>(request.Width);
                    result.Height = static_cast<uint32_t>(request.Height);
                    result.Response.Status =
                        ExecuteRenderRequest(args, request, fractal, output, errors);
                    if (result.Response.Status == 0) {
                        result.Iterations = fractal.GetNumIterations<uint64_t>();
                        result.OverallMs = fractal.GetBenchmark().m_Overall.GetDeltaInMs();
                        result.PerPixelMs = fractal.GetBenchmark().m_PerPixel.GetDeltaInMs();
                        RefOrbitDetails details;
                        fractal.GetSomeDetails(details);
                        result.RefOrbitMs = details.OrbitMilliseconds;
                    }
                }
            }
        }
    } catch (const std::exception &exception) {
        errors << "error: " << exception.what() << "\n";
        result.Response.Status = 1;
    }
    result.CliImageMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)
            .count());
    result.Response.Stdout = output.str();
    result.Response.Stderr = errors.str();
    return result;
}

std::vector<std::string>
BuildForwardedArguments(int argc, char *argv[])
{
    std::vector<std::string> result;
    for (int i = 1; i < argc; i++) {
        const std::string_view argument = argv[i];
        if (argument == "--connect" || argument == "--server" || argument == "--shutdown") {
            continue;
        }
        if (argument == "--endpoint") {
            if (i + 1 < argc) {
                i++;
            }
            continue;
        }
        result.emplace_back(argv[i]);
    }
    return result;
}

void
InitializeCliProcess()
{
    Environment::RegisterHeapCleanup();
    HighPrecision::defaultPrecisionInBits(FractalLimits::MaxPrecisionLame);
    Environment::CrashHandler::Install();
}

int
RunClient(const CliArgs &args, std::vector<std::string> forwardedArguments)
{
    const auto startTime = std::chrono::steady_clock::now();

    FractalSharkCli::IpcRequest request;
    request.Operation =
        args.Shutdown ? FractalSharkCli::IpcOperation::Shutdown : FractalSharkCli::IpcOperation::Render;
    request.Arguments = std::move(forwardedArguments);

    FractalSharkCli::IpcResponse response;
    std::string error;
    if (!FractalSharkCli::SendRequest(args.Endpoint, request, response, error)) {
        std::cerr << "error: " << error << "\n";
        return 3;
    }

    const auto endTime = std::chrono::steady_clock::now();

    if (!response.Stdout.empty()) {
        std::cout << response.Stdout;
        std::cout.flush();
    }
    if (!response.Stderr.empty()) {
        std::cerr << response.Stderr;
        std::cerr.flush();
    }
    if (!args.Shutdown) {
        const std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
        std::cout << std::fixed << std::setprecision(1) << "Frame time: " << elapsed.count() << " ms\n";
        std::cout.flush();
    }

    if (response.Status < 0 || response.Status > 255) {
        return 1;
    }
    return static_cast<int>(response.Status);
}

void
PrintBatchImageError(size_t index, size_t count, const FractalSharkCli::BatchImage &image)
{
    std::cerr << "[" << index + 1 << "/" << count << "] " << image.Name
              << " status=failed: " << image.Error << "\n";
}

void
PrintBatchImageResult(size_t index,
                      size_t count,
                      const FractalSharkCli::BatchImage &image,
                      const FractalSharkCli::IpcBatchResult &result)
{
    const bool success = result.Response.Status == 0;
    std::cout << "[" << index + 1 << "/" << count << "] " << image.Name
              << " status=" << (success ? "ok" : "failed") << " cli_image_ms=" << result.CliImageMs;
    if (success) {
        std::cout << " overall_ms=" << result.OverallMs << " per_pixel_ms=" << result.PerPixelMs
                  << " ref_orbit_ms=" << result.RefOrbitMs << " size=" << result.Width << "x"
                  << result.Height << " iterations=" << result.Iterations;
    }
    if (!image.Output.empty()) {
        std::cout << " out=" << std::quoted(image.Output);
    }
    std::cout << "\n";
    if (!result.Response.Stdout.empty()) {
        std::cout << result.Response.Stdout;
    }
    if (!result.Response.Stderr.empty()) {
        std::cerr << result.Response.Stderr;
    }
    std::cout.flush();
    std::cerr.flush();
}

int
RunBatch(const CliArgs &args)
{
    const auto started = std::chrono::steady_clock::now();
    FractalSharkCli::BatchFile batch;
    std::string error;
    if (!FractalSharkCli::LoadBatchFile(args.BatchPath, batch, error)) {
        std::cerr << "error: " << error << "\n";
        return 2;
    }

    std::vector<size_t> validIndexes;
    std::set<std::string> outputPaths;
    size_t failedCount = 0;
    for (size_t index = 0; index < batch.Images.size(); ++index) {
        auto &image = batch.Images[index];
        if (image.Error.empty()) {
            CliArgs imageArgs;
            std::ostringstream parseError;
            if (!ParseArgs(image.Arguments, imageArgs, parseError)) {
                image.Error = parseError.str();
            } else if (!ValidateRenderArgs(imageArgs, image.Error, false)) {
                // The validation error is stored on the image for ordered reporting.
            }
        }
        if (image.Error.empty() && !image.Output.empty()) {
            std::string outputKey = image.Output;
#ifdef _WIN32
            std::transform(outputKey.begin(), outputKey.end(), outputKey.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
#endif
            if (!outputPaths.insert(outputKey).second) {
                image.Error = "duplicate output path: " + image.Output;
            }
        }
        if (image.Error.empty()) {
            validIndexes.push_back(index);
        } else {
            ++failedCount;
        }
    }

    if (args.Mode == CliMode::Client && !validIndexes.empty()) {
        FractalSharkCli::IpcRequest request;
        request.Operation = FractalSharkCli::IpcOperation::Batch;
        request.BatchArguments.reserve(validIndexes.size());
        for (const size_t index : validIndexes) {
            request.BatchArguments.push_back(batch.Images[index].Arguments);
        }
        size_t nextToReport = 0;
        size_t completedCount = 0;
        const auto onResult = [&](size_t validIndex, const FractalSharkCli::IpcBatchResult &result) {
            const size_t imageIndex = validIndexes[validIndex];
            while (nextToReport < imageIndex) {
                PrintBatchImageError(nextToReport, batch.Images.size(), batch.Images[nextToReport]);
                ++nextToReport;
            }
            PrintBatchImageResult(imageIndex, batch.Images.size(), batch.Images[imageIndex], result);
            if (result.Response.Status != 0) {
                ++failedCount;
            }
            nextToReport = imageIndex + 1;
        };
        if (!FractalSharkCli::SendBatch(args.Endpoint, request, onResult, completedCount, error)) {
            std::cerr << "error: batch IPC failed after " << completedCount << " responses: " << error
                      << "\n";
            size_t unknownCount = 0;
            while (nextToReport < batch.Images.size()) {
                const auto &image = batch.Images[nextToReport];
                if (image.Error.empty()) {
                    std::cerr << "[" << nextToReport + 1 << "/" << batch.Images.size() << "] "
                              << image.Name << " status=unknown: no server response\n";
                    ++unknownCount;
                } else {
                    PrintBatchImageError(nextToReport, batch.Images.size(), image);
                }
                ++nextToReport;
            }
            std::cout << "Batch: " << batch.Images.size() - failedCount - unknownCount << " succeeded, "
                      << failedCount << " failed, " << unknownCount << " unknown\n";
            return 3;
        }
        while (nextToReport < batch.Images.size()) {
            PrintBatchImageError(nextToReport, batch.Images.size(), batch.Images[nextToReport]);
            ++nextToReport;
        }
    } else {
        std::unique_ptr<Fractal> fractal;
        if (!validIndexes.empty()) {
            fractal = std::make_unique<Fractal>(1024,
                                                768,
                                                /*nativeWindow=*/nullptr,
                                                /*UseSensoCursor=*/false,
                                                UINT64_MAX,
                                                /*hostOwnedGlPresentation=*/true,
                                                GpuMode::Auto);
        }
        for (size_t index = 0; index < batch.Images.size(); ++index) {
            const auto &image = batch.Images[index];
            if (!image.Error.empty()) {
                PrintBatchImageError(index, batch.Images.size(), image);
                continue;
            }
            const auto result =
                ExecuteBatchImage(image.Arguments, *fractal, 1024, 768, UINT64_MAX, false);
            PrintBatchImageResult(index, batch.Images.size(), image, result);
            if (result.Response.Status != 0) {
                ++failedCount;
            }
        }
        if (fractal) {
            fractal->CleanupThreads(/*all=*/true);
        }
    }

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started)
            .count();
    std::cout << "Batch: " << batch.Images.size() - failedCount << " succeeded, " << failedCount
              << " failed; elapsed_ms=" << elapsed << "\n";
    return failedCount == 0 ? 0 : 1;
}

int
RunServer(const CliArgs &serverArgs)
{
    Fractal fractal(serverArgs.Width,
                    serverArgs.Height,
                    /*nativeWindow=*/nullptr,
                    /*UseSensoCursor=*/false,
                    serverArgs.CommitCapBytes,
                    /*hostOwnedGlPresentation=*/true,
                    serverArgs.GpuRuntimeMode);

    Environment::LocalIpcListener listener;
    std::string error;
    if (!listener.Open(FractalSharkCli::ServiceName, serverArgs.Endpoint, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    std::cout << "FractalSharkCli server listening on " << listener.Endpoint() << "\n";
    std::cout.flush();

    for (;;) {
        std::string acceptError;
        Environment::LocalIpcConnection connection = listener.Accept(acceptError);
        if (!connection.IsOpen()) {
            std::cerr << "error: " << acceptError << "\n";
            listener.Close();
            return 1;
        }

        FractalSharkCli::IpcRequest request;
        if (!FractalSharkCli::ReadRequest(connection, request, error)) {
            std::cerr << "warning: rejected malformed IPC request: " << error << "\n";
            continue;
        }

        if (request.Operation == FractalSharkCli::IpcOperation::Batch) {
            for (const auto &arguments : request.BatchArguments) {
                const auto result = ExecuteBatchImage(arguments,
                                                      fractal,
                                                      serverArgs.Width,
                                                      serverArgs.Height,
                                                      serverArgs.CommitCapBytes,
                                                      true);
                std::string writeError;
                if (!FractalSharkCli::WriteBatchResult(connection, result, writeError)) {
                    std::cerr << "warning: could not send batch result: " << writeError << "\n";
                    break;
                }
            }
            continue;
        }

        FractalSharkCli::IpcResponse response;
        if (request.Operation == FractalSharkCli::IpcOperation::Shutdown) {
            fractal.CleanupThreads(/*all=*/true);
            response.Status = 0;
            response.Stdout = "FractalSharkCli server stopped.\n";
        } else {
            std::ostringstream requestOut;
            std::ostringstream requestError;
            CliArgs requestArgs;
            try {
                if (!ParseArgs(request.Arguments, requestArgs, requestError)) {
                    response.Status = 2;
                } else {
                    std::string validationError;
                    if (!ValidateRenderArgs(requestArgs, validationError, false)) {
                        requestError << "error: " << validationError << "\n";
                        response.Status = 2;
                    } else if (requestArgs.GpuRuntimeMode == GpuMode::Disabled) {
                        requestError
                            << "error: --no-gpu is a server startup option; put it on --server\n";
                        response.Status = 2;
                    } else if (requestArgs.CommitCapBytes != UINT64_MAX &&
                               requestArgs.CommitCapBytes != serverArgs.CommitCapBytes) {
                        requestError << "error: request commit cap does not match server startup cap\n";
                        response.Status = 2;
                    } else {
                        response.Status = ExecuteServerRender(requestArgs,
                                                              fractal,
                                                              requestOut,
                                                              requestError,
                                                              serverArgs.Width,
                                                              serverArgs.Height,
                                                              serverArgs.CommitCapBytes);
                    }
                }
            } catch (const std::exception &exception) {
                requestError << "error: " << exception.what() << "\n";
                response.Status = 1;
            }

            response.Stdout = requestOut.str();
            response.Stderr = requestError.str();
        }

        std::string writeError;
        if (!FractalSharkCli::WriteResponse(connection, response, writeError)) {
            std::cerr << "warning: could not send response: " << writeError << "\n";
        }

        if (request.Operation == FractalSharkCli::IpcOperation::Shutdown) {
            break;
        }
    }

    listener.Close();
    return 0;
}

int
RunCli(int argc, char *argv[])
{
    InitializeCliProcess();

    CliArgs args;
    if (!ParseArgs(argc, argv, args, std::cerr)) {
        PrintUsage();
        return 2;
    }

    if (args.Help) {
        PrintUsage();
        return 0;
    }

    if (args.ListRenderAlgorithms) {
        PrintRenderAlgorithms();
        return 0;
    }

    std::string validationError;
    if (args.BatchSet) {
        if (!ValidateBatchArgs(args, validationError)) {
            std::cerr << "error: " << validationError << "\n";
            return 2;
        }
        return RunBatch(args);
    }
    if (args.Mode == CliMode::Client) {
        if (args.Shutdown) {
            if (!ValidateShutdownArgs(args, validationError)) {
                std::cerr << "error: " << validationError << "\n";
                return 2;
            }
        } else if (!ValidateRenderArgs(args, validationError, true)) {
            std::cerr << "error: " << validationError << "\n";
            return 2;
        }
        auto forwardedArguments = BuildForwardedArguments(argc, argv);
        return RunClient(args, std::move(forwardedArguments));
    }

    if (args.Shutdown) {
        std::cerr << "error: --shutdown requires --connect\n";
        return 2;
    }
    if (args.Mode == CliMode::SingleShot && !args.Endpoint.empty()) {
        std::cerr << "error: --endpoint requires --server or --connect\n";
        return 2;
    }

    if (args.Mode == CliMode::Server) {
        if (!ValidateServerArgs(args, validationError)) {
            std::cerr << "error: " << validationError << "\n";
            return 2;
        }
        return RunServer(args);
    }

    if (!ValidateRenderArgs(args, validationError, false)) {
        std::cerr << "error: " << validationError << "\n";
        PrintUsage();
        return 2;
    }

    RenderRequest request;
    std::string requestError;
    int requestStatus =
        BuildRenderRequest(args, request, requestError, args.Width, args.Height, args.CommitCapBytes);
    if (requestStatus != 0) {
        std::cerr << "error: " << requestError << "\n";
        return requestStatus;
    }

    Fractal fractal(request.Width,
                    request.Height,
                    /*nativeWindow=*/nullptr,
                    /*UseSensoCursor=*/false,
                    request.CommitCapBytes,
                    /*hostOwnedGlPresentation=*/true,
                    args.GpuRuntimeMode);
    return ExecuteRenderRequest(args, request, fractal, std::cout, std::cerr);
}

} // anonymous namespace

int
main(int argc, char *argv[])
{
    try {
        return RunCli(argc, argv);
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << "\n";
        return 1;
    }
}
