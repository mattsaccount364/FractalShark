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
- `Notes/`: LaTeX docs and plain-Markdown analysis notes (Reference2 perf studies).
- `tools/NcuAnalysis/`: NCU profile analysis toolkit; see `tools/NcuAnalysis/README.md`.
- `Pics/` and `FromImagina/`: sample images and inputs.

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
compare overlaid text with its Windows counterpart, rebuild the persistent target, and verify the
runnable binary is newer than its sources. Report both `HEAD` values, overlaid files, the persistent
binary path, and commands run. Windows and Linux builds may run in parallel.

## Coding And File Hygiene

Use `.clang-format`: 4-space indentation, no tabs, 105-column limit, right-aligned pointers/references,
and `stdafx.h` first where used. Classes, methods, enums, and CMake targets use PascalCase. Member
variables use `m_PascalCase`; locals and parameters use `camelCase`. Avoid adding C++ default
function parameters to new or modified interfaces; pass required values explicitly through callers.

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

Never add local arrays to CUDA code. Always pass parameters one at a time so they can be allocated in
registers.

Do not introduce `#if 0` blocks. Remove obsolete code or preserve it through version control instead
of leaving disabled implementations in the source tree.

After modifying C++/CUDA files, run the formatting script from the repository root:

```powershell
.\format_cpp_sources.ps1
```

The repository formatter is a required validation step. If it fails, report the
exact command and error, correct the working-directory or process-environment
problem, and rerun the script. Do not silently substitute a manual formatter or
bypass Git, PowerShell, or other safety checks and then report formatting as
successful; any such exception requires explicit user direction.

## Testing

`FractalSharkTest` uses `FractalSharkTest/TestFramework.h`; add cases with `TEST(Name)` and `ASSERT_*`
macros. Register new test files in both CMake and Visual Studio project files when both build systems
cover the project. `HpSharkFloatTest` requires real CUDA hardware and is not expected to run on hosted
CI.

Allow at least 30 minutes before timing out a full `FractalSharkTest`, especially for Windows Debug or
parallel host validation. If a test times out, terminate it explicitly before retrying.

`CrummyTest` is a functional suite invoked from the GUI menu. It must call `Drain()` and use the direct
rendering path: `CalcFractal(true)` followed by `SaveCurrentFractal`.

NCU profile analysis scripts live under `tools/NcuAnalysis/`. Start from
`tools/NcuAnalysis/README.md`, which lists the full script set (source-CSV
export, stall composition, per-line attribution, pipe/issue/divergence,
memory/occupancy, metric discovery, and the library-line bar-sync resolvers),
the standard workflow, and the environment gotchas. Durable invariants: the
per-line join must reconcile to `unattributed: 0` before any line ranking is
trusted, and the scripts match NCU metric names by candidate regex — after an
NCU or GPU/architecture change, confirm the metric families exist with
`metric_probe.py` before trusting their output.

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

The LaTeX master is `Notes/FractalShark.tex`. To build `Notes/FractalShark.pdf`, run
`build.ps1` from the `Notes` directory; the script performs the required LaTeX and bibliography
passes and opens the resulting PDF. Use another `-jobname` only if the normal PDF is locked. Main and engineering
prose uses formal third-person American English; user documentation and development history may be
informal. Avoid sensationalized language and bare `This` followed by a verb. Use pure TikZ figures,
`\caption[short]{long}`, `\cref{}` for cross-references, `\eqref{}` where appropriate, `\code{}` for
inline code, and `lstlisting` for blocks. Keep canonical notation from the master notation table and
spell the prose name as `LA~v2` (`LAv2` only in identifiers).

Do not commit local build directories, generated checkpoints, profiling reports, or ad hoc render
outputs unless they are intentional fixtures. Follow `SECURITY.md` for vulnerability reporting.

## Agent Guardrails

Do not run `git add` or `git commit`; leave changes unstaged for review. Do not modify source files in
plan-only workflows.

## Copilot Secondary Review

Only invoke the local Qwen-backed GitHub Copilot CLI as a secondary reviewer when the user explicitly
requests Copilot or Qwen secondary review for the current task. Do not infer permission from the
task's difficulty, subject matter, or potential benefit from another opinion. When explicitly
requested, use the reusable workflow in `.agents/skills/copilot-helper/SKILL.md` and invoke it through:

```powershell
.\copilot.ps1 --codex --prompt-file <prompt-file>
```

On Windows, use `--prompt-file` for any review prompt containing spaces. The
wrapper passes the file contents as one process argument; do not invoke
`copilot.exe` directly or rely on nested shell quoting. Use one review launch
per request.

Before invoking any Qwen-backed review, verify that no other Qwen/Ollama job is
currently running, including one started outside this repository. This machine
has one GPU; concurrent Qwen jobs can conflict and fail. Check `ollama ps` and
the relevant `copilot`/`ollama` process state first, and wait for any active
Qwen job to finish before starting another.

The `--codex` profile is intentionally quiet and the local Ollama-backed model
may take several minutes to produce a final response. A lack of immediate
output is not evidence that the review failed. For a non-trivial investigation,
request streamed output and diagnostic logs:

```powershell
$logDirectory = Join-Path $env:TEMP "copilot-helper-logs"
New-Item -ItemType Directory -Force $logDirectory | Out-Null
.\copilot.ps1 --codex --stream on --log-level info --log-dir $logDirectory `
    --prompt-file <prompt-file>
```

Use a generous outer timeout, 30 minutes by default for a substantial review,
and poll rather than abandoning a live run. Thirty minutes is a default
starting budget, not a hard cutoff: if the active Copilot session/event log,
streamed log, or Copilot/Ollama model activity grows during checks every few
minutes, continue waiting and report periodic status. Stop only after at least
three consecutive checks show no event/log growth and no model/process activity,
or after an explicit cancellation decision. From a second PowerShell window,
check process liveness, the loaded model, and log growth:

```powershell
$logDirectory = Join-Path $env:TEMP "copilot-helper-logs"
Get-Process -Name copilot,ollama -ErrorAction SilentlyContinue |
    Select-Object ProcessName,Id,StartTime,CPU
ollama ps
Get-ChildItem -LiteralPath $logDirectory -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name,Length,LastWriteTime
Get-Content -LiteralPath "<active-log-file>" -Wait
```

These checks show liveness and recent activity, not an exact token-level
completion percentage. Do not ignore Copilot output solely because it is
delayed; wait for completion and reconcile its claims with the independent
investigation. Only use `.\copilot_stop.ps1` for an intentional cancellation
or after repeated checks show that the run is genuinely stuck, not merely
because a short default timeout expired.

If Copilot's noninteractive permission manager repeatedly denies repository or
process probes, distinguish that from model latency. The reusable `--codex`
profile now supplies the user-authorized `--allow-all-paths` and
`--allow-all-tools` flags so a requested review does not need a second Qwen
invocation; retain the explicit no-edit/no-commit review prompt.

Codex remains responsible for investigation, architecture, edits, decisions, and validation. Treat
Copilot output as advisory and verify important claims independently. Unless explicitly requested,
ask Copilot not to edit files or commit changes. When the user requests a full secondary-review
workflow, a useful sequence is independent investigation, Copilot review, reconciliation of
disagreements, implementation, Copilot diff review, and final validation.
