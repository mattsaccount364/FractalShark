# Copilot Instructions for FractalShark

## Project Overview

FractalShark is a high-performance Mandelbrot renderer focused on extreme zoom depths via CUDA GPU acceleration. C++23/CUDA, Windows (Visual Studio 2026, v145 toolset) + Linux (CMake + Clang). Targets NVIDIA GPUs (GTX 900+, RTX 2xxx+ for GPU reference orbits) and AVX-2 CPUs.

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
| **FractalSharkLib** | Core: fractal math, perturbation, ref orbits, LA | ✅ |
| **FractalSharkGpuLib** | CUDA rendering kernels (perturbation, BLA, LA, reduction) | ✅ |
| **FractalSharkPlatform** | Cross-platform abstraction (Environment, types, alloc) | ✅ |
| **HpSharkFloatLib** | High-precision GPU arithmetic (NTT, custom floats, LA/DLA) | ⏳ |
| **HpSharkFloatTest** | GPU test harness for HP arithmetic | ❌ |
| **HpSharkInstantiate** | Explicit template instantiation generator | ❌ |
| **FractalTray** | System tray utility (functional) | ❌ |
| **FractalSaver** | Screen saver (legacy) | ❌ |
| **FractalSharkTest** | CPU unit tests for utility types | ✅ |

## Build Instructions (Windows)

Authoritative process: `.github/workflows/build.yml`.

**Prereqs:** Visual Studio 2026 with C++ + CUDA, CUDA Toolkit 13.0.2, YASM at `C:\Program Files\yasm\*` (bundled in `tools/yasm.zip`), MPIR cloned to repo root and built via `mpir\msvc\vs22\mpir.sln` (`lib_mpir_skylake_avx`, x64).

```powershell
# Full rebuild (required if any .h, .cu, or .cuh changed):
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

Ubuntu 24.04 box at `mrenz@ubuntu24` (SSH key auth). CUDA 13.2 at `/usr/local/cuda` — not on PATH; prefix with `export PATH=/usr/local/cuda/bin:$PATH`. **No GPU**, so only compile/link is validated, not runtime CUDA.

Windows working tree is authoritative. The Linux checkout at `~/FractalShark` is a disposable test mirror — never commit or push from it. Always verify the Linux checkout's commit matches the Windows working tree before testing; sync via `git fetch && git checkout <sha>`, overlay uncommitted Windows changes via `scp`.

```bash
ssh mrenz@ubuntu24
cd ~/FractalShark && git rev-parse HEAD && git status --short   # verify vs Windows host
cd ~/FractalShark && git fetch origin && git checkout <branch-or-sha>
scp <local-file> mrenz@ubuntu24:~/FractalShark/<repo-relative-path>
```

```bash
mkdir -p build && cd build
export PATH=/usr/local/cuda/bin:$PATH
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++ ..
cmake --build . --parallel
```

**Parallel cross-host builds:** When validating on both hosts, kick off Windows MSBuild and Linux `cmake --build` simultaneously in separate `mode="async"` shells. Don't serialize unless there's a real artifact dependency (there isn't, for Win↔Linux validation).

## Testing

- **`FractalSharkTest`** — standalone CPU executable. Validates `HDRFloat`, `HDRFloatComplex`, `HighPrecision` parsing, `PointZoomBBConverter`. Custom header-only framework (`TEST`/`ASSERT_*`). Build: `msbuild ... /t:FractalSharkTest`. Run: `Release\FractalSharkTest.exe`. Exit 0 = pass.
- **`HpSharkFloatTest`** — standalone CUDA executable. Three-level cross-validation (GPU vs MPIR ground truth, GPU vs CPU reference, CPU reference vs MPIR) with checksum-guided debugging. `msbuild ... /t:HpSharkFloatTest`, then `Release\HpSharkFloatTest.exe`.
- **`CrummyTest`** (in `FractalSharkLib/CrummyTest.cpp`) — functional suite invoked from the GUI right-click menu (IDM_BASICTEST). Calls `Drain()` then uses the **direct rendering path** (`CalcFractal(true)` → `SaveCurrentFractal`).

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
- **build-linux** — Ubuntu CMake/Clang. Builds `FractalSharkPlatform` + `FractalSharkLib` + `FractalSharkTest`, runs `./build/FractalSharkTest/FractalSharkTest`. `TestMpirSerialization` runs on both platforms; the Linux MPIR wire-format implementation atop GMP `mpz_export`/`mpz_import` is validated byte-for-byte against the committed `FractalSharkTest/golden_mpir.bin`.

## Workflow

- **Do not run `git add` or `git commit`.** Leave changes unstaged for the user to review and commit manually.
- **Do not change code in plan mode.** Plan mode is analysis and planning only.

## Notes Document (`Notes/`)

LaTeX technical document (~7,500 lines, 13 chapters + 2 appendices) covering FractalShark's algorithms in depth. Master file: `FractalShark.tex`.

```powershell
cd Notes
pdflatex -interaction=nonstopmode -halt-on-error FractalShark.tex
# Run twice for cross-refs. Use -jobname=FractalShark_test if the PDF is locked.
# bibtex FractalShark && pdflatex ... (twice more) for bibliography.
```

### Writing Conventions (Notes/)

- **Tone:** Formal third-person academic in chapters 1–12. Appendices A1/A2 are intentionally informal.
- **No sensationalized language:** avoid "dramatically", "crucially", "key insight", "notoriously", "massively parallel", etc.
- **No bare "This"** + verb. Always follow `This` with a noun (`This approach…`, not `This ensures…`).
- **Figures:** pure TikZ; `\caption[short]{long}`; cross-ref via `\cref{fig:...}`.
- **Equations:** number ones referenced elsewhere; reference via `\eqref{}` or `\cref{eq:...}`. Unnumbered display math only when used at point of definition.
- **Cross-refs:** `\cref{}` (cleveref). Add forward-linking sentences at chapter ends.
- **Code:** `\code{}` inline; `lstlisting` (not `verbatim`) for blocks.
- **Notation:** canonical symbols defined in the notation table in `FractalShark.tex` (~lines 63–130).
- **American English** prose (`-ize`, `center`, `color`, `behavior`, `synchronize`, `analyze`, `optimize`). British spellings only inside `\code{}` identifiers, bibliography titles, and quoted material.
- **LA v2 formatting:** `LA~v2` (non-breaking space) consistently. `LAv2` only inside `\code{}` identifiers.