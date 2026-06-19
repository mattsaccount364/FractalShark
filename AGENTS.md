# Repository Guidelines

## Project Structure & Module Organization

FractalShark is a C++23/CUDA Mandelbrot renderer. Core fractal math, perturbation, reference orbits, palettes, PNG output, and render orchestration live in `FractalSharkLib/`. CUDA kernels and GPU rendering helpers are in `FractalSharkGpuLib/`; high-precision GPU arithmetic is in `HpSharkFloatLib/`. Platform abstractions live in `FractalSharkPlatform/`. Windows GUI code is split across `FractalShark/` and `FractalSharkGUILib/`; Linux GUI work is in `FractalSharkGuiLinux/`; CLI entry points are in `FractalSharkCli/`. Tests live in `FractalSharkTest/`, `HpSharkFloatTest/`, and `HpSharkFloatTestLib/`. LaTeX docs are in `Notes/`; sample images and inputs are in `Pics/` and `FromImagina/`.

## Build, Test, and Development Commands

- `msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Debug /p:Platform=x64`: build the Windows Debug solution.
- `Debug\FractalSharkTest.exe`: run the main Windows CPU unit tests.
- `./build_linux.sh`: preferred Linux build command; configures and builds Debug and Release with `--parallel`.
- `cmake --build build-debug --parallel`: incrementally rebuild Linux Debug targets when a full
  Debug-and-Release build is unnecessary. Use `build-release` for Release.
- `./build-debug/FractalSharkTest/FractalSharkTest`: run portable Linux tests after a Debug build.

Linux build output belongs in `build-debug/` or `build-release/`, matching `build_linux.sh`. Do not
remove Linux build artifacts after validation; leave binaries available for manual testing and reuse
the configured directories for incremental builds.

Linux validation must build the persistent `~/FractalShark` WSL mirror after force-synchronizing its
tracked files from the authoritative Windows host working tree. WSL-local tracked source edits are
disposable mirror drift. Isolated WSL worktree builds do not replace rebuilding the persistent
runnable binaries. Transfer unpublished host commits directly into WSL without pushing them. See
`.github/copilot-instructions.md` for the required synchronization procedure.

Windows development targets Visual Studio 2026, CUDA Toolkit, MPIR, and bundled YASM tooling; `.github/workflows/build.yml` is the best reference for CI setup.

## Coding Style & Naming Conventions

Use `.clang-format`: 4-space indentation, no tabs, 105-column limit, right-aligned pointers/references, and `stdafx.h` first where used. Classes, methods, enums, and CMake targets use PascalCase. Member variables use `m_PascalCase`; locals and parameters use `camelCase`. Prefer existing file patterns such as `Test*.cpp`, `*Kernel*.cuh`, and descriptive module names.

Portable command and menu contracts belong directly in `FractalShark`. Platform-specific GUI
implementation belongs in `FractalShark::Win32` or `FractalShark::Linux`. Keep OS entry points and
third-party-mandated APIs global. Do not use `using namespace`; use unqualified names inside the
owning namespace and narrow aliases or using-declarations where needed.

## Testing Guidelines

`FractalSharkTest` uses the custom framework in `FractalSharkTest/TestFramework.h`; add cases with `TEST(Name)` and `ASSERT_*` macros. Register new test files in both CMake and Visual Studio project files when the project supports both build systems. `HpSharkFloatTest` requires real CUDA hardware and is not expected to run on hosted CI.

Allow at least 30 minutes before timing out a full `FractalSharkTest` execution, especially for Windows
Debug builds or when validation is running in parallel on both hosts. If a test process does time out,
terminate it explicitly before retrying so the previous run does not continue consuming resources.

## Commit & Pull Request Guidelines

Recent commits are short and informal, for example `More gui` or `Some notes`; keep summaries concise and outcome-focused. PRs should state the affected platform, list build/test commands run, note CUDA hardware or driver assumptions, link related issues, and include screenshots or rendered outputs for GUI, rendering, or documentation changes.

## Agent-Specific Instructions

AI/code assistants should also read `.github/copilot-instructions.md`. That file contains detailed architecture rules, build-system caveats, rendering-path hazards, and workflow constraints that are intentionally too detailed for this contributor overview.

## Security & Configuration Tips

Do not commit local build directories, generated checkpoints, profiling reports, or ad hoc render outputs unless they are intentional fixtures. Follow `SECURITY.md` for vulnerability reporting.
