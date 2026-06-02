# Copilot Instructions for FractalShark

## Project Overview

FractalShark is a high-performance Mandelbrot renderer focused on extreme zoom depths via CUDA GPU acceleration. C++23/CUDA, Windows (Visual Studio 2026, v145 toolset on the primary dev machine) + Linux (CMake + Clang). Targets NVIDIA GPUs (GTX 900+, RTX 2xxx+ for GPU reference orbits) and AVX-2 CPUs.

For a concise contributor overview, see `AGENTS.md`. This file is the detailed AI/code-assistant operating manual.

Deeper architectural background lives in `Notes/FractalShark-*.tex` (perturbation theory, BLA/LA, ref orbits, NTT GPU arithmetic, FeatureFinder, render pipeline, memory system) and in source-file headers. Read those when a task touches the relevant area; do not duplicate that prose here.

### Template Vocabulary

Multi-parameter templates appear throughout the codebase:

```cpp
template<typename IterType, class T, class SubType, PerturbExtras PExtras>
```

`IterType` = iter counter (`uint32_t`/`uint64_t`); `T` = ref-orbit precision; `SubType` = per-pixel delta precision; `PExtras` = compile-time enum for extra data. Explicit instantiation is via macros (see `RefOrbitCalcTemplates.h`, `HpSharkInstantiate`).

## Build Systems

Two parallel build systems:

- **Windows** (primary): `.vcxproj`/`.sln` + MSBuild.
- **Linux**: CMake + Clang, `CMakeLists.txt` alongside `.vcxproj`. Porting is incremental; not all projects have `CMakeLists.txt` yet.

**Dual build system rule:** When adding/removing source files, updating include paths, or changing compiler settings, update **both** systems if the project has a `CMakeLists.txt`. If only `.vcxproj`, update that alone.

### Solution Projects

| Project | Purpose | CMakeLists.txt |
|---|---|---|
| **FractalShark** | Main GUI app (Win32 window, render loop) | ❌ |
| **FractalSharkCli** | Command-line entry point | ✅ |
| **FractalSharkGUILib** | Shared Windows GUI support | ❌ |
| **FractalSharkGuiLinux** | Linux GUI shell and Xlib/ImGui integration | ✅ |
| **FractalSharkLib** | Core: fractal math, perturbation, ref orbits, LA | ✅ |
| **FractalSharkGpuLib** | CUDA rendering kernels (perturbation, BLA, LA, reduction) | ✅ |
| **FractalSharkPlatform** | Cross-platform abstraction (Environment, types, alloc) | ✅ |
| **HpSharkFloatLib** | High-precision GPU arithmetic (NTT, custom floats, LA/DLA) | ✅ |
| **HpSharkFloatTest** | GPU test harness for HP arithmetic | ✅ |
| **HpSharkFloatTestLib** | Shared GPU arithmetic test logic | ✅ |
| **HpSharkInstantiate** | Explicit template instantiation generator | ✅ |
| **FractalTray** | System tray utility (functional) | ❌ |
| **FractalSaver** | Screen saver (legacy) | ❌ |
| **FractalSharkTest** | CPU unit tests for utility types | ✅ |

## Build Instructions (Windows)

Authoritative CI process: `.github/workflows/build.yml`. Windows builds prefer Visual Studio 2026/v145 when available and fall back to v143.

**Prereqs:** Visual Studio 2026 with C++ + CUDA, CUDA Toolkit 13.3.0, YASM/vsyasm at `C:\Program Files\vsyasm\*` (bundled in `tools/yasm.zip`), MPIR cloned to repo root and built via `mpir\msvc\vs22\mpir.sln` (`lib_mpir_skylake_avx`, x64).

```powershell
# Full rebuild (required if any .h, .cu, .cuh, CUDA .props/.targets, or shared build settings changed):
msbuild FractalShark\FractalShark.sln /t:Rebuild /m /v:m /p:Configuration=Release /p:Platform=x64
msbuild FractalShark\FractalShark.sln /t:Rebuild /m /v:m /p:Configuration=Debug /p:Platform=x64

# Incremental (safe ONLY if .cpp-only changes — no headers, no CUDA):
msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Release /p:Platform=x64
```

Only x64 is supported. **Do not pass `/nologo`** — MSBuild misinterprets it as a directory path on this project.

## Build Instructions (Linux)

Linux build uses Clang (not GCC). Prereqs: CMake 3.20+, Clang (tested 18.x), `libgmp-dev`.

The custom heap allocator (`HpSharkFloatLib/heap_allocator/HeapCpp.cpp`) is Windows-only by design. On Linux, allocations flow through glibc `malloc` via `Environment::SystemHeap*` and `Environment::RegisterHeapCleanup()` is an empty stub (`FractalSharkPlatform/EnvironmentLinux.cpp`). Permanent decision.

### Linux Test Machine

Local WSL Ubuntu environment at `matthew@localhost` (SSH key auth). CUDA 13.2 at `/usr/local/cuda` — not on PATH; prefix with `export PATH=/usr/local/cuda/bin:$PATH`. **No GPU**, so only compile/link is validated, not runtime CUDA.

The Windows working tree is the only authoritative source tree, including uncommitted changes. The
Linux checkout at `~/FractalShark` is a disposable test mirror: never develop, commit, push, or
preserve tracked source edits there. Reuse this persistent mirror for normal validation so its
configured build directories and runnable output artifacts remain available for manual testing.

Before every Linux build, force the persistent mirror to the exact Windows `HEAD`, then overlay the
host working-tree changes required by the task. WSL-local tracked edits are mirror drift and should
be discarded without preserving them. Keep untracked `build-debug/` and `build-release/` artifacts.
The host `HEAD` may contain unpublished commits: transfer it directly to WSL with a temporary Git
bundle. Do not rely on `origin` being current and do not push Windows commits merely to run Linux
validation.

The overlay must include staged and unstaged modifications, deletions, and any newly created
task-relevant files. `git diff --name-status HEAD --` reports staged and unstaged tracked changes but
does not report untracked files, so maintain an explicit list of new files created during the task.

```bash
# Run from Windows PowerShell in the authoritative working tree.
$sha = git rev-parse HEAD
$bundle = Join-Path $env:TEMP "FractalShark-$sha.bundle"
$remoteBundle = "/tmp/FractalShark-$sha.bundle"
git diff --name-status HEAD --

# Transfer host HEAD without pushing, then reset tracked WSL mirror state while retaining untracked build artifacts.
Remove-Item -LiteralPath $bundle -ErrorAction SilentlyContinue
git bundle create $bundle HEAD
scp $bundle "matthew@localhost:$remoteBundle"
ssh matthew@localhost "set -e; trap 'rm -f $remoteBundle' EXIT; cd ~/FractalShark; git reset --hard; git fetch $remoteBundle HEAD; git checkout --detach -f $sha; git reset --hard $sha"
Remove-Item -LiteralPath $bundle

# Overlay each host file modified or added for the task.
scp <local-file> matthew@localhost:~/FractalShark/<repo-relative-path>

# Apply each host deletion in the mirror.
ssh matthew@localhost "cd ~/FractalShark && rm -- <repo-relative-path>"
```

```bash
cd ~/FractalShark
./build_linux.sh
```

`build_linux.sh` is the preferred Linux build entry point. It exports the CUDA path, configures Debug
and Release with Clang, and builds both with `cmake --build ... --parallel`. Its output directories are
`build-debug/` and `build-release/`. For example, the Linux GUI binaries are written to
`build-debug/FractalSharkGuiLinux/FractalSharkGuiLinux` and
`build-release/FractalSharkGuiLinux/FractalSharkGuiLinux`; portable test binaries follow the same
pattern, such as `build-debug/FractalSharkTest/FractalSharkTest`.

For a custom incremental build, use the existing configuration-specific directory and keep parallel
compilation enabled:

```bash
cd ~/FractalShark
export PATH=/usr/local/cuda/bin:$PATH
cmake --build build-debug --parallel
cmake --build build-debug --target FractalSharkGuiLinux --parallel
```

Use `build-release/` instead when validating Release. Do not create a generic `build/` directory for
routine validation. Do not remove `build-debug/`, `build-release/`, or their binaries after testing;
leave artifacts in place for manual execution and future incremental builds.

Build `~/FractalShark`, not an isolated WSL worktree, for routine validation. The Debug and Release
binaries under this persistent mirror are the user-facing Linux artifacts. If an isolated worktree
is exceptionally required for diagnosis, its validation is provisional only: Linux work is not
complete until `~/FractalShark` has been resynchronized, rebuilt, and verified.

Before reporting Linux success:

1. Confirm Windows and `~/FractalShark` report the same `git rev-parse HEAD`.
2. Confirm every task-relevant overlaid file matches the Windows host. Normalize CRLF/LF when hashing
   text files, for example with `tr -d '\r' < file | sha256sum` on WSL.
3. Rebuild the persistent mirror target and confirm its runnable binary is newer than the overlaid
   sources.
4. State the Windows `HEAD`, WSL mirror `HEAD`, overlaid files, persistent binary path, and tests run.

**Parallel cross-host builds:** When validating on both hosts, kick off Windows MSBuild and Linux `cmake --build` simultaneously in separate `mode="async"` shells. Don't serialize unless there's a real artifact dependency (there isn't, for Win↔Linux validation).

## Testing

- **`FractalSharkTest`** — standalone CPU executable. Validates `HDRFloat`, `HDRFloatComplex`, `HighPrecision` parsing, `PointZoomBBConverter`. Custom header-only framework (`TEST`/`ASSERT_*`). Build: `msbuild ... /t:FractalSharkTest`. Run: `Release\FractalSharkTest.exe`. Exit 0 = pass.
- **`HpSharkFloatTest`** — standalone CUDA executable. Three-level cross-validation (GPU vs MPIR ground truth, GPU vs CPU reference, CPU reference vs MPIR) with checksum-guided debugging. `msbuild ... /t:HpSharkFloatTest`, then `Release\HpSharkFloatTest.exe`.
- **`CrummyTest`** (in `FractalSharkLib/CrummyTest.cpp`) — functional suite invoked from the GUI right-click menu (IDM_BASICTEST). Calls `Drain()` then uses the **direct rendering path** (`CalcFractal(true)` → `SaveCurrentFractal`).

Use a timeout of at least 30 minutes for a full `FractalSharkTest` execution, especially Windows Debug
or simultaneous Windows/Linux validation. A timed-out test process may continue running after the
command wrapper returns; terminate that process explicitly before retrying.

## Critical Operational Rules

These prevent real bugs; do not strip them when editing:

- **Two rendering paths for `CalcFractal`.** Render-pool path (normal UI) writes to `workerIters` and does **not** update `m_CurIters`. Direct path (`CalcFractal(true)`, used by CrummyTest, AutoZoomer Default/Max, save ops) writes to `m_CurIters` and **requires `Drain()` first**. Anything that reads `m_CurIters` (image save, AutoZoom analysis) **must** use the direct path, not `EnqueueCommand`.
- **Three enqueue APIs** on `RenderThreadPool`/`Fractal`: `EnqueueCommand` (mutate + render), `EnqueueMutation` (mutate, no render), `EnqueueRender` (render only). UI never mutates `Fractal` state directly. New renders supersede earlier supersedable ones; AutoZoomer items set `Supersedable = false`.
- **MPIR allocator lifecycle.** All MPIR objects (`mpf_t`, `HighPrecision`) allocated under a custom allocator (`MPIRBoundedAllocator`, `MPIRBumpAllocator`) **must be destroyed before `ShutdownTls()`**. After shutdown, `mpf_clear` falls back to the default allocator and crashes on custom-allocator pointers. Use `{ }` scope blocks to force destruction order.
- **Reference orbit backends.** `RefOrbitCalc` has single-threaded CPU (MPIR, authoritative), multi-threaded CPU, and GPU (`HpSharkFloat` + NTT). `SaveForReuse` modes persist orbits for reuse at different `SubType` precisions; waypoint and Imagina-compatible "max" compression are supported.
- **NR checkpoint resume.** `FeatureFinder` NR inner-loop backends treat `startIter > 0` as resume mode; `startIter == 0` reinitializes z/dzdc/d2.

## Code Conventions

### Naming

- Classes / methods / enums: `PascalCase` (`RefOrbitCalc`, `GetPrecision()`, `PerturbExtras::Disable`).
- Class member variables: `m_PascalCase` (`m_RefOrbit`, `m_DrawThreads`).
- Plain-struct fields (no `m_`): `PascalCase` (`Width`, `CommitCapBytes`).
- Locals, parameters, lambda params: `camelCase`.
- When modifying a function, rename legacy snake_case identifiers inside it. No whole-file rename sweeps.

### Formatting

`.clang-format` is configured: C++20, 105 cols, 4-space indent, no tabs, right-aligned pointers/refs, Allman braces for functions, K&R for control flow, `stdafx.h` first. `clang-format` is at `C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\Llvm\bin\clang-format.exe` (not on PATH). Run on modified files after edits.

### Other rules

- **Error handling:** CUDA via return codes (`cudaError_t`). Both exceptions and status enums are acceptable. CUDA allocation failures should fall back gracefully (e.g., `cudaMalloc` → `cudaMallocHost`).
- **Warnings:** Always `#pragma warning(push)`/`pop` — never project-wide suppression. Common: 4702, 4324, 4100.
- **Ownership:** `std::unique_ptr` for ownership; raw pointers for non-owning. **Do not use `std::shared_ptr`** — refactor instead.
- **Logging:** `std::cout` for high-level diagnostics. `OutputDebugStringA` reserved for heap/panic paths. No logging frameworks or LOG macros.
- **Windows headers:** `NOMINMAX` and `WIN32_LEAN_AND_MEAN` must be defined before any `<Windows.h>` (already in `stdafx.h`; redefine in files that include Windows headers directly).
- **`static_assert`** for compile-time layout / platform / range checks.
- **Include order:** `"stdafx.h"` first, then project headers, then system/third-party. Use **forward slashes** in paths — clang requires it; MSVC accepts either.
- **Git LFS** tracks `*.png`, `*.jpg`, `*.zip` per `.gitattributes`. Don't add large binaries without LFS.

## CI

`.github/workflows/build.yml` runs three parallel jobs:

- **build-pdf** — LaTeX notes PDF on Windows.
- **build** (Debug + Release matrix) — Windows MSBuild + CUDA, runs `FractalSharkTest.exe`.
- **build-linux** — Ubuntu CMake/Clang. Configures and builds the top-level CMake graph, then runs `./build/FractalSharkTest/FractalSharkTest`. `TestMpirSerialization` runs on both platforms; the Linux MPIR wire-format implementation atop GMP `mpz_export`/`mpz_import` is validated byte-for-byte against golden bytes embedded in `FractalSharkTest/TestMpirSerialization.cpp`.

## Workflow

- **Do not run `git add` or `git commit`.** Leave changes unstaged for the user to review and commit manually.
- **Do not change code in plan mode.** Plan mode is analysis and planning only.

## Notes Document (`Notes/`)

LaTeX technical document covering FractalShark's algorithms in depth, followed by user documentation,
engineering-reference material, and development history in three appendices. Master file: `FractalShark.tex`.

```powershell
cd Notes
pdflatex -interaction=nonstopmode -halt-on-error FractalShark.tex
# Run twice for cross-refs. Use -jobname=FractalShark_test if the PDF is locked.
# bibtex FractalShark && pdflatex ... (twice more) for bibliography.
```

### Writing Conventions (Notes/)

- **Tone:** Formal third-person academic in the main narrative and engineering-reference appendix. The
  user-documentation and development-history appendices are intentionally informal.
- **No sensationalized language:** avoid "dramatically", "crucially", "key insight", "notoriously", "massively parallel", etc.
- **No bare "This"** + verb. Always follow `This` with a noun (`This approach…`, not `This ensures…`).
- **Figures:** pure TikZ; `\caption[short]{long}`; cross-ref via `\cref{fig:...}`.
- **Equations:** number ones referenced elsewhere; reference via `\eqref{}` or `\cref{eq:...}`. Unnumbered display math only when used at point of definition.
- **Cross-refs:** `\cref{}` (cleveref). Add forward-linking sentences at chapter ends.
- **Code:** `\code{}` inline; `lstlisting` (not `verbatim`) for blocks.
- **Notation:** canonical symbols defined in the notation table in `FractalShark.tex` (~lines 63–130).
- **American English** prose (`-ize`, `center`, `color`, `behavior`, `synchronize`, `analyze`, `optimize`). British spellings only inside `\code{}` identifiers, bibliography titles, and quoted material.
- **LA v2 formatting:** `LA~v2` (non-breaking space) consistently. `LAv2` only inside `\code{}` identifiers.
