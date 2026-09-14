// FractalSharkCli: headless PNG renderer.
//
// The normal single-shot mode is also the client/server front end. Server
// mode keeps the expensive Fractal and CUDA/reference-orbit state alive while
// client mode forwards the usual render arguments over local IPC.

#include "stdafx.h"

#include "CrashHandler.h"
#include "Environment.h"
#include "Fractal.h"
#include "LocalIpc.h"
#include "PointZoomBBConverter.h"
#include "RefOrbitCalc.h"
#include "RenderAlgorithm.h"
#include "RenderThreadPool.h"
#include "RenderToConsole.h"
#include "RenderToPng.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
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

    ViewSource Source = ViewSource::None;
    size_t BuiltinView = 0;
    std::string LocationsFile;
    size_t LocationIndex = SIZE_MAX; // SIZE_MAX => use last record
    std::string CenterX, CenterY, Zoom;

    uint64_t Iterations = 0;              // 0 => unspecified (parser rejects 0)
    uint32_t Antialiasing = 0;            // 0 => unspecified (parser rejects 0)
    uint64_t CommitCapBytes = UINT64_MAX; // UINT64_MAX => unlimited

    std::string RenderAlgorithm;
    std::string PerturbationAlg; // empty => unspecified

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
    std::cout << "FractalSharkCli — headless Mandelbrot renderer\n"
                 "\n"
                 "Usage:\n"
                 "  FractalSharkCli --render-algorithm NAME [--out FILE.png] [--console] [--color]\n"
                 "                  [--width W --height H]\n"
                 "                  {--builtin-view N |\n"
                 "                   --locations FILE [--location-index N] |\n"
                 "                   --center-x X --center-y Y --zoom Z}\n"
                 "                  [--iterations N] [--antialiasing N]\n"
                 "                  [--perturbation-alg NAME] [--commit-cap-bytes N]\n"
                 "                  [--quiet]\n"
                 "\n"
                 "  FractalSharkCli --server [--endpoint NAME] [--width W --height H]\n"
                 "                   [--commit-cap-bytes N]\n"
                 "  FractalSharkCli --connect [--endpoint NAME] <the render arguments above>\n"
                 "  FractalSharkCli --connect --endpoint NAME --shutdown\n"
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
    using P = RefOrbitCalc::PerturbationAlg;
    if (name == "ST")
        return P::ST;
    if (name == "MT")
        return P::MT;
    if (name == "STPeriodicity")
        return P::STPeriodicity;
    if (name == "MTPeriodicity3")
        return P::MTPeriodicity3;
    if (name == "MTPeriodicity3PerturbMTHighSTMed")
        return P::MTPeriodicity3PerturbMTHighSTMed;
    if (name == "MTPeriodicity3PerturbMTHighMTMed1")
        return P::MTPeriodicity3PerturbMTHighMTMed1;
    if (name == "MTPeriodicity3PerturbMTHighMTMed2")
        return P::MTPeriodicity3PerturbMTHighMTMed2;
    if (name == "MTPeriodicity3PerturbMTHighMTMed3")
        return P::MTPeriodicity3PerturbMTHighMTMed3;
    if (name == "MTPeriodicity3PerturbMTHighMTMed4")
        return P::MTPeriodicity3PerturbMTHighMTMed4;
    if (name == "MTPeriodicity5")
        return P::MTPeriodicity5;
    if (name == "GPU")
        return P::GPU;
    if (name == "Auto")
        return P::Auto;
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
        } else if (arg == "--quiet") {
            a.Quiet = true;
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
    try {
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
    } catch (const std::exception &) {
        error = std::string(flag) + " is not a valid high-precision decimal";
        return false;
    }
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
        !args.PerturbationAlg.empty() || args.LocationIndex != SIZE_MAX) {
        error = "server accepts only --endpoint, --width, --height, --commit-cap-bytes, and --quiet";
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
    if (!args.RenderAlgorithm.empty() || args.Source != ViewSource::None || !args.OutFile.empty() ||
        args.Console || args.Color || args.WidthSet || args.HeightSet || args.Iterations != 0 ||
        args.Antialiasing != 0 || args.CommitCapBytes != UINT64_MAX || !args.PerturbationAlg.empty() ||
        args.LocationIndex != SIZE_MAX) {
        error = "--shutdown cannot be combined with render arguments";
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
                     std::ostream &errorOut,
                     bool useQueuedSetup,
                     PngCompletionMode completionMode)
{
    // Single-shot construction of Fractal initializes its default view after
    // parsing the request, which also lowers the MPIR default precision.
    HighPrecision::defaultPrecisionInBits(FractalLimits::MaxPrecisionLame);

    std::string error;
    int rc = useQueuedSetup ? RenderToPngQueued(req, fractal, &error, out, completionMode)
                            : RenderToPng(req, fractal, &error, out, completionMode);
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
    return 0;
}

int
ExecuteRender(const CliArgs &args,
              Fractal &fractal,
              std::ostream &out,
              std::ostream &errorOut,
              int defaultWidth,
              int defaultHeight,
              uint64_t commitCapBytes,
              bool useQueuedSetup,
              PngCompletionMode completionMode)
{
    RenderRequest req;
    std::string error;
    int rc = BuildRenderRequest(args, req, error, defaultWidth, defaultHeight, commitCapBytes);
    if (rc != 0) {
        errorOut << "error: " << error << "\n";
        return rc;
    }

    return ExecuteRenderRequest(args, req, fractal, out, errorOut, useQueuedSetup, completionMode);
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

int
RunServer(const CliArgs &serverArgs)
{
    Fractal fractal(serverArgs.Width,
                    serverArgs.Height,
                    /*nativeWindow=*/nullptr,
                    /*UseSensoCursor=*/false,
                    serverArgs.CommitCapBytes);

    FractalSharkCli::LocalListener listener;
    std::string error;
    if (!listener.Open(serverArgs.Endpoint, error)) {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    std::cout << "FractalSharkCli server listening on " << listener.Endpoint() << "\n";
    std::cout.flush();

    for (;;) {
        std::string acceptError;
        FractalSharkCli::LocalConnection connection = listener.Accept(acceptError);
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
                    } else if (requestArgs.CommitCapBytes != UINT64_MAX &&
                               requestArgs.CommitCapBytes != serverArgs.CommitCapBytes) {
                        requestError << "error: request commit cap does not match server startup cap\n";
                        response.Status = 2;
                    } else {
                        response.Status = ExecuteRender(requestArgs,
                                                        fractal,
                                                        requestOut,
                                                        requestError,
                                                        serverArgs.Width,
                                                        serverArgs.Height,
                                                        serverArgs.CommitCapBytes,
                                                        /*useQueuedSetup=*/false,
                                                        PngCompletionMode::Background);
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

} // anonymous namespace

int
main(int argc, char *argv[])
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
        try {
            return RunServer(args);
        } catch (const std::exception &exception) {
            std::cerr << "error: " << exception.what() << "\n";
            return 1;
        }
    }

    if (!ValidateRenderArgs(args, validationError, false)) {
        std::cerr << "error: " << validationError << "\n";
        PrintUsage();
        return 2;
    }

    try {
        RenderRequest request;
        std::string requestError;
        int requestStatus = BuildRenderRequest(
            args, request, requestError, args.Width, args.Height, args.CommitCapBytes);
        if (requestStatus != 0) {
            std::cerr << "error: " << requestError << "\n";
            return requestStatus;
        }

        Fractal fractal(request.Width,
                        request.Height,
                        /*nativeWindow=*/nullptr,
                        /*UseSensoCursor=*/false,
                        request.CommitCapBytes);
        return ExecuteRenderRequest(args,
                                    request,
                                    fractal,
                                    std::cout,
                                    std::cerr,
                                    /*useQueuedSetup=*/false,
                                    PngCompletionMode::Wait);
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << "\n";
        return 1;
    }
}
