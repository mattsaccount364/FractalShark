# Copilot Instructions for FractalShark

## Project Overview

FractalShark is a high-performance Mandelbrot set renderer focused on extreme zoom depths using CUDA GPU acceleration. It is a Windows-native C++20/CUDA application built with Visual Studio 2026 (v145 toolset). The project targets NVIDIA GPUs (GTX 900-series+, RTX 2xxx+ for GPU reference orbits) and requires AVX-2 CPUs.

## Build Instructions

The authoritative build process is defined in `.github/workflows/build.yml`.

### Prerequisites

- Visual Studio 2026 with C++ and CUDA support (this is the only supported version)
- NVIDIA CUDA Toolkit 13.0.2
- YASM assembler at `C:\Program Files\yasm\*` (bundled in `tools/yasm.zip` for CI)
- MPIR library: clone `https://github.com/BrianGladman/mpir.git` into the repo root, then build `mpir\msvc\vs22\mpir.sln` (lib_mpir_skylake_avx, x64)

### Building

```powershell
# From repo root, using Developer Command Prompt or MSBuild on PATH:
msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Release /p:Platform=x64
msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Debug /p:Platform=x64
```

Only x64 builds are supported. x86 configurations in the solution are mapped to x64.

> **Do not use `/nologo`** — MSBuild on this project misinterprets it as a directory path.

### Solution Projects

| Project | Purpose |
|---|---|
| **FractalShark** | Main GUI application (Win32 window, rendering loop) |
| **FractalSharkLib** | Core library: fractal math, perturbation, reference orbits, linear approximation |
| **FractalSharkGpuLib** | CUDA kernels for Mandelbrot rendering (perturbation, BLA, LA, reduction) |
| **HpSharkFloatLib** | High-precision GPU arithmetic: NTT multiplication, custom float types, LA/DLA |
| **HpSharkFloatTest** | GPU test harness for high-precision arithmetic |
| **HpSharkInstantiate** | Explicit template instantiation compilation unit |
| **FractalTray** | System tray utility (functional) |
| **FractalSaver** | Screen saver (legacy) |

### Testing

HpSharkFloatTest is a custom test harness (no standard framework). Build and run it:

```powershell
msbuild FractalShark\FractalShark.sln /m /v:m /p:Configuration=Release /p:Platform=x64 /t:HpSharkFloatTest
# Run the built executable directly
Release\HpSharkFloatTest.exe
```

## Architecture

### Template-Heavy Design

The codebase uses extensive multi-parameter templates to support different precision levels and iteration types at compile time:

```cpp
template<typename IterType, class T, class SubType, PerturbExtras PExtras>
```

- `IterType`: iteration counter type (`uint32_t` or `uint64_t`)
- `T`: high-precision type for reference orbit computation
- `SubType`: lower-precision type for per-pixel perturbation
- `PExtras`: compile-time enum controlling extra data collection

Explicit template instantiation is done via macro patterns (see `RefOrbitCalcTemplates.h`, `HpSharkInstantiate` project) to manage compile times across 36+ type combinations.

### Perturbation Theory Rendering

Rather than iterating every pixel at full precision, FractalShark computes one **reference orbit** at high precision (MPIR arbitrary precision on CPU, or custom high-precision floats on GPU), then renders each pixel by computing only its low-precision **delta** from that reference. This is the "perturbation" technique. Template parameter `T` controls the reference orbit precision; `SubType` controls the per-pixel delta precision (float, double, HDRFloat, 2×32 custom type).

**Linear approximation (LA)** further accelerates rendering by skipping iterations where the delta can be well-approximated by a linear function.

### GPU Code Organization

- `FractalSharkGpuLib/`: Rendering kernels (`.cu`/`.cuh` files) — perturbation, BLA, antialiasing, reduction
- `HpSharkFloatLib/`: High-precision arithmetic kernels — NTT multiply, GPU BLAS, LA/DLA computation
- CUDA device code uses `CUDA_CRAP` macro as a device/host qualifier shorthand

### Numeric Type Hierarchy

Per-pixel perturbation computation uses these types, from lowest to highest precision:

| Type | Components | Effective bits | Practical role |
|---|---|---|---|
| `float` | 1×fp32 | ~24 | fastest; sometimes insufficient |
| `dblflt` | 2×fp32 | ~48 | practical sweet spot on GPU |
| `double` | 1×fp64 | ~53 | standard; slow on consumer GPUs |
| `dbldbl` | 2×fp64 | ~104 | academic |
| `GQF::gqf_real` | 4×fp32 | ~112 | academic |
| `GQD::gqd_real` | 4×fp64 | ~212 | academic |
| MPIR / `HpSharkFloat` | N limbs | arbitrary | reference orbit only |

Any of these can be wrapped with `HDRFloat<>` for wide dynamic range (separate exponent + mantissa). In practice, the vast majority of rendering uses `float`, `double`, or `HDRFloat<CudaDblflt<dblflt>>` as the perturbation delta type (`SubType`).

- `dblflt`: A 2×fp32 double-float expansion type. `CudaDblflt<dblflt>` is its CUDA wrapper class with `HDRFloat` integration.
- `GQF`/`GQD`: Quad-float/quad-double types adapted from the QD library. Use namespaced helper functions (no wrapper class).
- `HDRFloat<T>`: Decouples dynamic range from precision — stores a base-2 exponent plus a normalized `T` mantissa. Used extensively in perturbation and reference orbit paths where values span many orders of magnitude.

### GPU Kernel Families

Base (non-perturbation) rendering kernels follow the naming pattern `mandel_{limbs}x_{base_type}`:

- `mandel_1x_float`, `mandel_1x_double` — single-precision and double-precision direct iteration
- `mandel_2x_float`, `mandel_2x_double` — double-float and double-double expansion types
- `mandel_4x_float`, `mandel_4x_double` — quad-float and quad-double expansion types
- `mandel_hdr_float` — HDR-normalized expansion type

Perturbation rendering has additional kernel families for BLA, LA v2, and perturbation-only paths, selected via the `LAv2Mode` enum. `iteration_precision` controls chunked iteration that can switch numeric types mid-render.

### Reference Orbit Pipeline

`RefOrbitCalc` orchestrates reference orbit computation with three backends:

1. **Single-threaded CPU** — MPIR arbitrary precision; authoritative orbit with periodicity detection via ∂z/∂c derivative tracking
2. **Multi-threaded CPU** — Splits orbit into segments; precision-dependent tradeoff (beneficial at moderate precision, overhead-dominated at extreme precision)
3. **GPU** — `HpSharkFloat`-based computation using NTT multiplication (see GPU Arithmetic below)

**Compression and reuse modes**: Reference orbits are expensive to compute but cheap to reuse. `SaveForReuse` modes store orbits for re-rendering at different `SubType` precisions. Waypoint-based compression stores only periodic checkpoints and replays intermediate values on demand, trading compute for memory. Imagina-compatible "max" compression format is also supported.

**Evaluation modes**: Expanded evaluation computes `z_n = z_{n-1}^2 + c` directly; factored evaluation uses `z_n = (z_{n-1} + c_root) * (z_{n-1} - c_root)` near periodic points for better numerical stability.

### Rendering Algorithm Selection

Three main acceleration layers, selected via `LAv2Mode` enum:

- **Perturbation-only** — Iterates each pixel's delta `Δz` from the reference orbit without approximation skipping. Simplest path; uses rebasing when `|Δz|` grows too large relative to `|z|`.
- **BLA v1 (Bilinear Approximation)** — `BLA<T>` stores linear coefficients (A, B) such that `Δz_{n+s} ≈ A·Δz_n + B·Δc`. Steps are composable (two short steps → one longer step). GPU lookup selects the longest valid aligned step at each iteration.
- **LA v2 (Linear Approximation)** — More aggressive: precomputes linear approximation coefficients from the reference orbit, allowing large iteration skips. HDR kernel variant. Can run in full-LA, LA-only, or perturbation-only sub-modes.

### Key Data Structures

- **`GrowableVector<EltT>`** — Growable array with stable virtual addresses (no reallocation/copy). Uses `VirtualAlloc` reserve-and-commit: reserves a large virtual region up front, then commits pages incrementally. Also supports file-backed (disk-mapped) mode for persistence and low commit overhead. Used for orbit storage, metadata streams, and the custom heap arena. Critical for multi-billion-element reference orbits.

- **`PointZoomBBConverter`** — Manages screen-pixel ↔ complex-plane coordinate transforms. All internal quantities (centre, bounding box, zoom factor) stored as `HighPrecision` (MPIR `mpf_t`) for meaningful precision at extreme zoom depths. Computes per-pixel deltas at full precision, then down-converts to the kernel's working type.

### Feature Finder

`FeatureFinder` (in `FeatureFinder.h`/`.cpp`) locates the center (nucleus) of nearby mini-Mandelbrot copies — the unique parameter `c*` where the critical orbit is exactly periodic with period `p`.

- **Phase A (T-space)**: Detect candidate period `p` and perform coarse Newton iteration using the renderer's native float type
- **Phase B (MPF refinement)**: Polish to arbitrary precision using MPIR `mpf_t` arithmetic with mixed Newton–Halley iteration

Three evaluation backends are available for Phase A: direct iteration, perturbation theory, and linear approximation — so the finder works at any zoom depth. Influenced by the Imagina fractal viewer's periodic-point finder.

### GPU High-Precision Arithmetic (HpSharkFloatLib)

The GPU reference orbit backend in `HpSharkFloatLib/` implements:

- **NTT-based multiplication**: Number Theoretic Transform over the "magic prime" `2^64 - 2^32 + 1` (Goldilocks form enabling efficient modular reduction). Pipeline: packing → twisting → forward NTT → pointwise multiply (Montgomery domain) → inverse NTT → untwisting → unpacking/normalization. Multiple products are batched simultaneously to amortize GPU synchronization.

- **High-precision addition**: Parallel prefix carry propagation on GPU — signed limbwise accumulation followed by single-pass parallel prefix scan for carry resolution and normalization.

- **Checksum-guided debugging**: Stage checksums (homomorphic summaries) are computed at each pipeline stage on both host and device, then cross-compared to isolate GPU kernel bugs. This pattern is used throughout HpSharkFloat development for correctness validation.

### Key Third-Party Dependencies

- **MPIR** (arbitrary precision integers, fork of GMP) — built from source, linked as static lib
- **WPngImage** — PNG image I/O

## Code Conventions

### Naming

- **Classes/Methods/Enums**: `PascalCase` (e.g., `RefOrbitCalc`, `GetPrecision()`, `PerturbExtras::Disable`)
- **Member variables**: `m_PascalCase` prefix (e.g., `m_RefOrbit`, `m_DrawThreads`)
- **Parameters**: `camelCase` or descriptive (e.g., `width`, `height`, `RequireReuse`)

### Formatting

A `.clang-format` is configured for the project:
- C++20 standard
- 105-column limit, 4-space indentation, no tabs
- Pointer/reference alignment: right (`int *ptr`, `int &ref`)
- Braces: Allman style for functions, K&R for control flow
- `stdafx.h` precompiled header must be included first

### Error Handling

- CUDA errors are checked via return codes (`cudaError_t`)
- Both C++ exceptions and return codes/status enums are acceptable depending on context
- CUDA allocation failures fall back gracefully (e.g., `cudaMalloc` → `cudaMallocHost`)

### Include Order

1. `"stdafx.h"` (precompiled header, always first)
2. Project headers
3. System/third-party headers

## Workflow

- **Do not create git commits.** Leave all changes uncommitted for the user to review and commit manually.
- **Do not make code changes in plan mode.** Plan mode is for analysis and planning only.
