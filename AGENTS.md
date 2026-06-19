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
the required WSL synchronization procedure below.

Windows development targets Visual Studio 2026, CUDA Toolkit, MPIR, and bundled YASM tooling; `.github/workflows/build.yml` is the best reference for CI setup.

FractalShark has parallel Visual Studio and CMake build systems. When adding or removing source files,
changing include paths, or changing compiler settings, update both systems when the project has a
`CMakeLists.txt`. A Windows change to any header, CUDA source, or shared build setting requires a full
Debug and Release rebuild; an incremental build is appropriate only for `.cpp`-only changes. Do not
pass `/nologo` to MSBuild because this solution can interpret it as a directory path.

### Required WSL Synchronization

The Windows working tree is authoritative, including uncommitted changes. Before every Linux build,
force the persistent `~/FractalShark` mirror to the Windows `HEAD`, then overlay all task-relevant
staged, unstaged, deleted, and untracked files. Preserve its untracked `build-debug/` and
`build-release/` directories. Do not develop, commit, push, or preserve tracked edits in the WSL
mirror, and do not use an isolated worktree as a substitute for rebuilding the persistent binaries.

Transfer unpublished commits directly instead of pushing them solely for validation:

```powershell
$sha = git rev-parse HEAD
$bundle = Join-Path $env:TEMP "FractalShark-$sha.bundle"
$remoteBundle = "/tmp/FractalShark-$sha.bundle"
git diff --name-status HEAD --
Remove-Item -LiteralPath $bundle -ErrorAction SilentlyContinue
git bundle create $bundle HEAD
scp $bundle "matthew@localhost:$remoteBundle"
ssh matthew@localhost "set -e; trap 'rm -f $remoteBundle' EXIT; cd ~/FractalShark; git reset --hard; git fetch $remoteBundle HEAD; git checkout --detach -f $sha; git reset --hard $sha"
Remove-Item -LiteralPath $bundle
scp <local-file> matthew@localhost:~/FractalShark/<repo-relative-path>
ssh matthew@localhost "cd ~/FractalShark && rm -- <deleted-repo-relative-path>"
```

When validation depends on Git LFS assets, copy their materialized Windows files; pointer files are
not valid runtime assets. Before reporting Linux success, confirm matching Windows and WSL `HEAD`
values, compare overlaid text while ignoring CRLF/LF differences, rebuild the persistent target, and
verify that its runnable binary is newer than its sources. Report both `HEAD` values, overlaid files,
the persistent binary path, and commands run. Windows and Linux builds may run in parallel.

## Coding Style & Naming Conventions

Use `.clang-format`: 4-space indentation, no tabs, 105-column limit, right-aligned pointers/references, and `stdafx.h` first where used. Classes, methods, enums, and CMake targets use PascalCase. Member variables use `m_PascalCase`; locals and parameters use `camelCase`. Prefer existing file patterns such as `Test*.cpp`, `*Kernel*.cuh`, and descriptive module names.

Portable command and menu contracts belong directly in `FractalShark`. Platform-specific GUI
implementation belongs in `FractalShark::Win32` or `FractalShark::Linux`. Keep OS entry points and
third-party-mandated APIs global. Do not use `using namespace`; use unqualified names inside the
owning namespace and narrow aliases or using-declarations where needed.

When modifying a function, rename legacy snake_case identifiers within that function, but do not run
whole-file naming sweeps. Use forward slashes in include paths. Use `std::unique_ptr` for ownership
and raw pointers for non-owning references; do not introduce `std::shared_ptr`. Define `NOMINMAX` and
`WIN32_LEAN_AND_MEAN` before including `<Windows.h>`. Wrap warning suppressions in
`#pragma warning(push)`/`pop`; never suppress warnings project-wide. Use `std::cout` for ordinary
diagnostics and reserve `OutputDebugStringA` for heap or panic paths.

After editing a text file, preserve its existing line-ending convention and verify that the file does
not contain mixed CRLF and LF endings. Patch tools may insert LF lines into CRLF files, so normalize
the complete modified file back to its pre-edit convention before finishing. Files explicitly marked
`eol=lf` in `.gitattributes`, including shell scripts, must remain LF-only.

All tracked C++ and CUDA source/header files ending in `.cpp`, `.h`, `.cu`, `.cuh`, `.cc`, `.hh`, or
`.hpp` must use CRLF in the Windows working tree. After editing any of these files, verify them with
`git ls-files --eol -- '*.cpp' '*.h' '*.cu' '*.cuh' '*.cc' '*.hh' '*.hpp'`; no matching file may
report `w/lf` or `w/mixed`.

## Testing Guidelines

`FractalSharkTest` uses the custom framework in `FractalSharkTest/TestFramework.h`; add cases with `TEST(Name)` and `ASSERT_*` macros. Register new test files in both CMake and Visual Studio project files when the project supports both build systems. `HpSharkFloatTest` requires real CUDA hardware and is not expected to run on hosted CI.

Allow at least 30 minutes before timing out a full `FractalSharkTest` execution, especially for Windows
Debug builds or when validation is running in parallel on both hosts. If a test process does time out,
terminate it explicitly before retrying so the previous run does not continue consuming resources.

`CrummyTest` is a functional suite invoked from the GUI menu. It must call `Drain()` and use the direct
rendering path (`CalcFractal(true)` followed by `SaveCurrentFractal`).

## Critical Rendering and Lifetime Rules

- Normal UI rendering uses the render-pool path, which writes `workerIters` and does not update
  `m_CurIters`. Anything that reads `m_CurIters`, including image save and AutoZoom analysis, must
  call `Drain()` and use the direct path. Do not substitute `EnqueueCommand`.
- `RenderThreadPool`/`Fractal` provide `EnqueueCommand` (mutate and render), `EnqueueMutation`
  (mutate without rendering), and `EnqueueRender` (render only). UI code must not mutate `Fractal`
  state directly. AutoZoomer work items must be non-supersedable.
- Destroy every `mpf_t` and `HighPrecision` object allocated under `MPIRBoundedAllocator` or
  `MPIRBumpAllocator` before `ShutdownTls()`. Use nested scopes to guarantee destruction order.
- `FeatureFinder` NR backends interpret `startIter > 0` as checkpoint resume and `startIter == 0` as
  fresh initialization; preserve that distinction.
- Reference-orbit changes must account for the authoritative single-threaded MPIR backend, the
  multithreaded CPU backend, the GPU `HpSharkFloat`/NTT backend, reuse modes, waypoints, and
  Imagina-compatible max compression.

## Commit & Pull Request Guidelines

Recent commits are short and informal, for example `More gui` or `Some notes`; keep summaries concise and outcome-focused. PRs should state the affected platform, list build/test commands run, note CUDA hardware or driver assumptions, link related issues, and include screenshots or rendered outputs for GUI, rendering, or documentation changes.

## Agent-Specific Instructions

Do not run `git add` or `git commit`; leave changes unstaged for review. Do not modify source files in
plan-only workflows. Run the configured clang-format executable on modified C++/CUDA files.

## Notes Document

The LaTeX master is `Notes/FractalShark.tex`. Run `pdflatex -interaction=nonstopmode -halt-on-error`
twice for cross-references; use another `-jobname` if the normal PDF is locked. Main and engineering
prose uses formal third-person American English; user documentation and development history may be
informal. Avoid sensationalized language and bare `This` followed by a verb. Use pure TikZ figures,
`\caption[short]{long}`, `\cref{}` for cross-references, `\eqref{}` where appropriate, `\code{}` for
inline code, and `lstlisting` for blocks. Keep canonical notation from the master notation table and
spell the prose name as `LA~v2` (`LAv2` only in identifiers).

## Security & Configuration Tips

Do not commit local build directories, generated checkpoints, profiling reports, or ad hoc render outputs unless they are intentional fixtures. Follow `SECURITY.md` for vulnerability reporting.
