# Repository Guidelines

## Repository Map

FractalShark is a C++23/CUDA Mandelbrot renderer. Keep changes in the owning subsystem:

- `FractalSharkLib/`: fractal math, perturbation, reference orbits, palettes, PNG output, render orchestration.
- `FractalSharkGpuLib/`: CUDA kernels and GPU rendering helpers.
- `HpSharkFloatLib/`: high-precision GPU arithmetic.
- `FractalSharkPlatform/`: platform abstractions.
- `FractalShark/`, `FractalSharkGuiWin32/`: Windows GUI.
- `FractalSharkGuiLinux/`: Linux GUI.
- `FractalSharkCli/`: CLI entry points.
- `FractalSharkTest/`, `HpSharkFloatTest/`, `HpSharkFloatTestLib/`: tests.
- `Notes/`: LaTeX docs. `Pics/` and `FromImagina/`: sample images and inputs.

## Build And Validation

Common commands:

- `msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Debug /p:Platform=x64`: Windows Debug build.
- `Debug\FractalSharkTest.exe`: main Windows CPU tests.
- `./build_linux.sh`: preferred Linux build; configures and builds Debug and Release with `--parallel`.
- `cmake --build build-debug --parallel`: incremental Linux Debug build. Use `build-release` for Release.
- `./build-debug/FractalSharkTest/FractalSharkTest`: portable Linux tests.

Windows development targets Visual Studio 2026, CUDA Toolkit, MPIR, and bundled YASM tooling. Use
`.github/workflows/build.yml` as the CI setup reference. Do not pass `/nologo` to MSBuild because this
solution can interpret it as a directory path.

Linux build output belongs in persistent `build-debug/` and `build-release/` directories. Do not remove
those artifacts after validation; leave binaries available for manual testing and incremental rebuilds.

FractalShark has parallel Visual Studio and CMake build systems. When adding/removing source files,
changing include paths, or changing compiler settings, update both systems when the project has a
`CMakeLists.txt`. Header, CUDA source, or shared build-setting changes require full Windows Debug and
Release rebuilds; `.cpp`-only changes may use incremental builds.

## Required WSL Synchronization

The Windows working tree is authoritative, including uncommitted changes. Before every Linux validation,
force the persistent `~/FractalShark` WSL mirror to Windows `HEAD`, then overlay task-relevant staged,
unstaged, deleted, and untracked files. Preserve the mirror's untracked `build-debug/` and
`build-release/` directories. Do not develop, commit, push, or preserve tracked edits in the WSL mirror.
Isolated WSL worktree builds do not replace rebuilding the persistent runnable binaries.

Transfer unpublished host commits directly instead of pushing only for validation:

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

When validation depends on Git LFS assets, copy the materialized Windows files; pointer files are not
valid runtime assets. Before reporting Linux success, confirm matching Windows and WSL `HEAD` values,
compare overlaid text while ignoring CRLF/LF differences, rebuild the persistent target, and verify the
runnable binary is newer than its sources. Report both `HEAD` values, overlaid files, the persistent
binary path, and commands run. Windows and Linux builds may run in parallel.

## Coding And File Hygiene

Use `.clang-format`: 4-space indentation, no tabs, 105-column limit, right-aligned pointers/references,
and `stdafx.h` first where used. Classes, methods, enums, and CMake targets use PascalCase. Member
variables use `m_PascalCase`; locals and parameters use `camelCase`.

Portable command and menu contracts belong directly in `FractalShark`. Platform-specific GUI
implementation belongs in `FractalShark::Win32` or `FractalShark::Linux`. Keep OS entry points and
third-party-mandated APIs global. Do not use `using namespace`; use unqualified names inside the owning
namespace and narrow aliases or using-declarations where needed.

When modifying a function, rename legacy `snake_case` identifiers within that function, but do not run
whole-file naming sweeps. Use forward slashes in include paths. Use `std::unique_ptr` for ownership and
raw pointers for non-owning references; do not introduce `std::shared_ptr`. Define `NOMINMAX` and
`WIN32_LEAN_AND_MEAN` before including `<Windows.h>`. Wrap warning suppressions in
`#pragma warning(push)`/`pop`; never suppress warnings project-wide. Use `std::cout` for ordinary
diagnostics and reserve `OutputDebugStringA` for heap or panic paths.

Preserve each edited text file's existing line-ending convention and avoid mixed endings. Files marked
`eol=lf` in `.gitattributes`, including shell scripts, must remain LF-only. All tracked C++/CUDA
source and header files ending in `.cpp`, `.h`, `.cu`, `.cuh`, `.cc`, `.hh`, or `.hpp` must be CRLF in
the Windows working tree. After editing those files, verify with:

```powershell
git ls-files --eol -- '*.cpp' '*.h' '*.cu' '*.cuh' '*.cc' '*.hh' '*.hpp'
```

No matching file may report `w/lf` or `w/mixed`.

## Testing

`FractalSharkTest` uses `FractalSharkTest/TestFramework.h`; add cases with `TEST(Name)` and `ASSERT_*`
macros. Register new test files in both CMake and Visual Studio project files when both build systems
cover the project. `HpSharkFloatTest` requires real CUDA hardware and is not expected to run on hosted
CI.

Allow at least 30 minutes before timing out a full `FractalSharkTest`, especially for Windows Debug or
parallel host validation. If a test times out, terminate it explicitly before retrying.

`CrummyTest` is a functional suite invoked from the GUI menu. It must call `Drain()` and use the direct
rendering path: `CalcFractal(true)` followed by `SaveCurrentFractal`.

## Rendering And Lifetime Invariants

- Normal UI rendering uses the render-pool path, which renders into `workerIters` and publishes a
  successful final frame into `m_CurIters` when dimensions match. Anything that needs a guaranteed
  current CPU iteration buffer must call `Drain()` first; direct `CalcFractal(true)` remains required
  for workflows that bypass the pool, such as `CrummyTest` and high-resolution saves.
- `RenderThreadPool`/`Fractal` provide `EnqueueCommand` (mutate and render), `EnqueueMutation` (mutate
  without rendering), and `EnqueueRender` (render only). UI code must not mutate `Fractal` state
  directly. AutoZoomer work items must be non-supersedable.
- Destroy every `mpf_t` and `HighPrecision` object allocated under `MPIRBoundedAllocator` or
  `MPIRBumpAllocator` before `ShutdownTls()`. Use nested scopes to guarantee destruction order.
- `FeatureFinder` NR backends interpret `startIter > 0` as checkpoint resume and `startIter == 0` as
  fresh initialization; preserve that distinction.
- Reference-orbit changes must account for the authoritative single-threaded MPIR backend, the
  multithreaded CPU backend, the GPU `HpSharkFloat`/NTT backend, reuse modes, waypoints, and
  Imagina-compatible max compression.

## Notes And Generated Files

The LaTeX master is `Notes/FractalShark.tex`. Run `pdflatex -interaction=nonstopmode -halt-on-error`
twice for cross-references; use another `-jobname` if the normal PDF is locked. Main and engineering
prose uses formal third-person American English; user documentation and development history may be
informal. Avoid sensationalized language and bare `This` followed by a verb. Use pure TikZ figures,
`\caption[short]{long}`, `\cref{}` for cross-references, `\eqref{}` where appropriate, `\code{}` for
inline code, and `lstlisting` for blocks. Keep canonical notation from the master notation table and
spell the prose name as `LA~v2` (`LAv2` only in identifiers).

Do not commit local build directories, generated checkpoints, profiling reports, or ad hoc render
outputs unless they are intentional fixtures. Follow `SECURITY.md` for vulnerability reporting.

## Agent Guardrails

Do not run `git add` or `git commit`; leave changes unstaged for review. Do not modify source files in
plan-only workflows. Run the configured clang-format executable on modified C++/CUDA files.
