# Copilot Instructions for FractalShark

## Project Overview

FractalShark is a high-performance Mandelbrot set renderer focused on extreme zoom depths using CUDA GPU acceleration. It is a Windows-native C++20/CUDA application built with Visual Studio 2022 (v143 toolset). The project targets NVIDIA GPUs (GTX 900-series+, RTX 2xxx+ for GPU reference orbits) and requires AVX-2 CPUs.

## Build Instructions

The authoritative build process is defined in `.github/workflows/build.yml`.

### Prerequisites

- Visual Studio 2022 with C++ and CUDA support
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
