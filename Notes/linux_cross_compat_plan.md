# Linux Cross-Compatibility Roadmap

Status date: 2026-05-25.

Goal: FractalShark should be buildable, testable, and usable on Linux from the
top-level CMake graph while keeping the Windows and Linux GUIs separate. Shared
engine/platform behavior belongs in `FractalSharkLib` or `FractalSharkPlatform`;
windowing, widget, input, and native-dialog behavior stays in the platform GUI.

## Current Status

The original Linux plan is mostly implemented structurally. The remaining work
is less about creating targets and more about parity, validation, and cleanup.

| Area | Status | Notes |
|------|--------|-------|
| Top-level Linux CMake | Mostly done | Root `CMakeLists.txt` adds platform, math, GPU, tests, CLI, and Linux GUI targets. |
| CPU tests | Done | `FractalSharkTest` builds through CMake and includes `CommandCatalogTest`. |
| CLI | Done | `FractalSharkCli` exists with CMake and Visual Studio project files. |
| CUDA/high-precision libs | Structurally done | `HpSharkFloatLib`, `HpSharkFloatTestLib`, and `HpSharkFloatTest` have CMake targets. Runtime GPU validation still requires a CUDA host. |
| Platform abstraction | Mostly done | Linux environment and crash handler are in `FractalSharkPlatform`. |
| Shared command/menu data | Partial | `CommandCatalog` and `MenuTree` exist. `kCommands` hotkey metadata is still empty, and menu-state logic is duplicated per GUI. |
| Linux GUI | Partial but substantial | Xlib window, GLX presentation, ImGui overlay, right-click menu, resize, wheel zoom, drag-zoom, clipboard, fullscreen, minimize, and many commands exist. |
| Runtime validation | Incomplete | Need documented Linux build/run results, X11-forwarded GUI smoke test, and GPU-host runtime test. |

## Architecture Principles

- Keep `FractalSharkGUILib` and `FractalSharkGuiLinux` separate. Do not add a
  shared GUI framework or shared widget layer.
- Share behavior only when it is not GUI-specific: command IDs, command routing,
  menu tree data, platform services, render engine code, and test utilities.
- Linux remains Xlib + GLX + Dear ImGui for now. Wayland/EGL is a future port.
- Windows remains the primary native GUI. Linux should match behavior where
  practical, with intentional differences documented in
  `FractalSharkGuiLinux/PARITY.md`.
- Linux GUI runtime should work on a GPU-less host through the engine's GPU
  bypass path, but CUDA correctness must be validated separately on real NVIDIA
  hardware.

## Already Implemented

### Build and Platform

- Top-level CMake includes:
  - `FractalSharkPlatform`
  - `HpSharkFloatLib`
  - `FractalSharkLib`
  - `FractalSharkGpuLib`
  - `HpSharkInstantiate`
  - `HpSharkFloatTestLib`
  - `HpSharkFloatTest`
  - `FractalSharkTest`
  - `FractalSharkCli`
  - `FractalSharkGuiLinux` on non-Apple Unix
- `FractalSharkPlatform` has `EnvironmentLinux.cpp` and
  `CrashHandlerLinux.cpp`.
- Linux crash handling installs `sigaction` handlers and emits
  `std::stacktrace` before re-raising the signal.
- Linux stacktrace linkage is handled in `FractalSharkPlatform/CMakeLists.txt`.

### Shared Command and Menu Work

- `FractalSharkLib/CommandCatalog.{h,cpp}` defines `FractalCommand`,
  `ExecuteCommandHost`, and `ExecuteCommand`.
- `FractalSharkLib/MenuTree.h` and `MenuTreeDef.h` define the shared menu tree.
- `FractalSharkTest/CommandCatalogTest.cpp` checks command routing and verifies
  static menu leaves route through `ExecuteCommand`.
- The Linux ImGui context menu walks the shared menu tree.

### Linux GUI

- `FractalSharkGuiLinux` is a CMake target.
- Dear ImGui is fetched by CMake `FetchContent` at tag `v1.91.5`.
- Xlib window creation, GLX-compatible visual selection, and `OpenGlContext`
  integration are implemented.
- GUI-thread GL presentation is implemented through render-pool
  `TryPresentTick`, cached-frame representation, ImGui overlay rendering, and
  `SwapBuffers`.
- Resize events call `Fractal::ResetDimensions`.
- Mouse wheel zoom, right-click context menu anchoring, left-drag zoom capture,
  live drag rectangle rendering, and drag release recentering exist.
- ImGui support exists for context menus, info modals, simple save dialogs,
  list-pick dialogs, and enter-location dialogs.
- Linux clipboard uses the X11 `CLIPBOARD` selection with `UTF8_STRING`.
- Fullscreen, square fullscreen, minimize, and exit hooks exist.
- Many Linux command handlers mirror Win32 command bodies through
  `LinuxCommandHandlers`.

## Remaining Work

### 1. Linux GUI Command Parity Pass

Make every menu command either functional on Linux or explicitly documented as
unsupported.

- Replace the current TODO/stub behavior in:
  - `LinuxMainWindow::DispatchByIdm`
  - `OnMemoryLimit0`
  - `OnMemoryLimit1`
  - `OnPaletteRotate`
  - `OnPerturbResults`
- Audit every `FractalCommand` menu leaf against Linux behavior using
  `CommandCatalogTest` as the mechanical starting point and
  `FractalSharkGuiLinux/PARITY.md` as the manual checklist.
- Remove stale comments in `FractalSharkGuiLinux/main.cpp` that still describe
  earlier implementation phases.
- Keep Linux-only platform behavior in `FractalSharkGuiLinux`; do not push
  ImGui or Xlib logic into shared libraries.

### 2. Hotkey Strategy

Choose and finish one approach.

Recommended: complete the original catalog-driven hotkey plan.

- Populate `FractalShark::kCommands` with menu-command hotkeys and labels.
- Route Linux `KeyPress` events through `HotKey`/catalog lookup where the key
  corresponds to a discrete command.
- Keep latency-sensitive or gesture-like direct paths outside the catalog:
  continuous nudge keys, drag-zoom, resize, and any command that needs richer
  event payload than the catalog carries.
- Update the hotkey help modal to use catalog data instead of static text.
- If the current direct `switch (ch)` approach is kept instead, document that
  as intentional and remove plan text that says hotkeys are single-source.

### 3. File Dialog and Path Handling

Current dialogs are useful but not yet full parity with Win32 file operations.

- Add a real ImGui file browser or improve the existing modals enough for:
  - save current image
  - save high-resolution image
  - save iterations text
  - save/load reference orbit `.im`
  - load saved location data
- Preserve current-directory or last-directory state per dialog type.
- Add overwrite confirmation for save paths.
- Convert file paths as UTF-8 rather than the current ASCII-only widening helper.
- Keep external dependencies at zero; do not add `zenity`, `kdialog`, GTK, Qt,
  or portal requirements unless this decision is reopened.

### 4. Menu-State Sharing Decision

Current state: Linux has `LinuxMenuState`; Win32 has `MainWindowMenuState`.
This is acceptable short-term but duplicates enable/check/radio logic.

Recommended path:

- Leave GUI-specific menu rendering duplicated.
- Move only rule evaluation and radio/check selection logic into shared engine
  code if duplication starts causing drift.
- If duplication remains small and stable, document it as intentional in
  `PARITY.md` and stop treating `MenuTreeState.cpp` as planned work.

### 5. Build and CI Hygiene

- Decide whether `FetchContent` for ImGui is acceptable for CI and fresh Linux
  builds. If offline/reproducible configure matters, vendor ImGui or add a
  documented cache/bootstrap step.
- Keep Linux GUI compile-only in GitHub-hosted CI unless an X server and GPU
  runner are added.
- Add explicit CI or local instructions for:
  - configure
  - build all
  - run `FractalSharkTest`
  - run CLI smoke render
  - build `HpSharkFloatTest`
- Avoid making hosted CI run `HpSharkFloatTest`; that needs real CUDA hardware.

### 6. Runtime Validation

Runtime validation is the main missing proof.

Minimum CPU/X11 validation on `ubuntu24`:

```bash
cd ~/FractalShark
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++
cmake --build build -j 1
./build/FractalSharkTest/FractalSharkTest
./build/FractalSharkCli/FractalSharkCli --help
./build/FractalSharkGuiLinux/FractalSharkGuiLinux
```

Manual GUI smoke checklist:

- Window opens at 1600x1000 and renders the default view.
- Resizing triggers a correct redraw.
- Right-click menu opens at cursor.
- Menu items execute representative commands from each top-level group.
- Keyboard shortcuts match the current intended Linux strategy.
- Mouse wheel zoom works.
- Drag-zoom rectangle displays and commits correctly.
- Clipboard copy/paste for location data works.
- Save/load flows work for image, location, and reference-orbit operations.
- Fullscreen, square fullscreen, minimize, and close work.
- CPU fallback works on GPU-less Linux.

CUDA validation on a real NVIDIA Linux host:

- Build with the same CMake graph.
- Run `HpSharkFloatTest`.
- Launch the GUI and select representative GPU render algorithms.
- Validate GPU reference orbit mode separately from basic GPU perturbation and
  rendering kernels.

## Validation Matrix

| Scenario | Target | Required result |
|----------|--------|-----------------|
| Configure Linux CMake | `cmake -B build ...` | No missing packages or network surprises beyond documented ImGui fetch. |
| Build all Linux targets | `cmake --build build -j 1` | All library, test, CLI, and GUI targets link. |
| CPU unit tests | `FractalSharkTest` | Exit code 0. |
| Command catalog test | included in `FractalSharkTest` | All static menu leaves route to hooks. |
| CLI smoke | `FractalSharkCli` | Starts and can render or report help without crashing. |
| Linux GUI CPU smoke | `FractalSharkGuiLinux` over X11 | Window, render, menu, input, save/load basics work. |
| CUDA library runtime | `HpSharkFloatTest` on GPU host | Exit code 0. |
| GPU GUI smoke | Linux NVIDIA host | GPU algorithms render or gracefully fall back with clear diagnostics. |
| Windows regression | MSBuild + `FractalSharkTest.exe` | Windows remains unaffected by Linux-only GUI work. |

## Risks and Deferred Work

- `FetchContent` makes fresh configure depend on network access. This is fine
  for development but should be revisited before treating Linux builds as
  fully reproducible.
- X11 only. Wayland requires an EGL/windowing follow-up and is out of scope for
  this roadmap.
- GitHub-hosted CI cannot prove Linux GUI runtime or CUDA runtime behavior.
- Linux path handling currently has ASCII-oriented helpers in some save/load
  paths. Full UTF-8 cleanup is part of file-dialog parity.
- `FractalSharkGuiLinux/PARITY.md` has some stale line references and planned
  behaviors. Update it after the command/hotkey/file-dialog decisions settle.
- `FractalTray` and `FractalSaver` remain Windows-only by design.

## Recommended Next Milestone

Target: "Linux GUI parity pass 1".

Do this next:

1. Implement or intentionally disable the remaining Linux command stubs.
2. Audit the context menu by walking every top-level menu group manually.
3. Decide the hotkey strategy and update `kCommands`, `HandleKeyPress`, and the
   hotkey help modal accordingly.
4. Replace ASCII-only path widening in Linux save/load paths with UTF-8-aware
   conversion.
5. Run the Linux build/test/GUI smoke checklist on `ubuntu24`.
6. Record the result in this file or in a short dated validation note.
