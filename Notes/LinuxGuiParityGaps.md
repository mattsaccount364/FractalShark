# Linux GUI Parity Gaps

Last reviewed: 2026-06-07.

This file tracks Linux GUI parity work against the Win32 GUI baseline. It is a working checklist for
fixes and deliberate platform differences, not user-facing documentation.

## Core GUI Gaps

| Area | Linux state | Win32 baseline | Notes / likely fix |
| --- | --- | --- | --- |
| Memory limit controls | `OnMemoryLimit0` and `OnMemoryLimit1` are stubs in `FractalSharkGuiLinux/main.cpp`; `LinuxMenuState` always reports `IDM_MEMORY_LIMIT_0`. | `MainWindow` toggles a `JobObject`, and menu state reports whether the job object is active. | Decide whether Linux should use `setrlimit`, cgroups, or remain unlimited. Then wire real handlers and menu state. |
| Dynamic `.im` quick-pick menus | Linux has file dialogs for Imagina orbit load/save, but no current-directory quick menu. | Win32 has native dynamic command ranges in `FractalShark/resource.h`, routed by `FractalSharkGUILib/CommandDispatcher.cpp`, with `MenuLoadImagDyn`. | Add an X11/ImGui pick-list for current-directory `.im` files or consciously keep file-dialog-only UX. |
| Startup polish | Linux opens directly into the X11/GLX window and logs diagnostics to stderr. | Win32 starts a splash window and background console while the main window initializes. | Add a Linux splash/progress surface only if startup latency warrants it. Diagnostics may remain stderr-first. |
| Window mode behavior | Linux toggles EWMH fullscreen and supports Alt+drag window move. | Win32 has custom borderless/windowed behavior, non-client suppression, and resize hit-test handling. | Either implement Linux borderless resize/move behavior or document that Linux delegates this to the window manager. |
| Palette rotation | Linux command exists but appears visually broken. | Win32 has the same TODO and visible behavior problem. | Shared behavior likely needs a render/recolor invalidation path after `RotateFractalPalette()`. |
| Perturbation result display | `OnPerturbResults` only logs a TODO. | Win32 also only logs a TODO. | Implement a real overlay/view mode on both platforms if this command is kept in the menu. |

## Already Present On Linux

- Shared `MenuTreeDef.h` context menu structure and command catalog dispatch.
- Native X11 context menu windows with nested submenus, radio/check state, disabled items, scrolling, and keyboard navigation.
- Hotkey dispatch from X11 key events through `CommandCatalog`.
- Drag zoom, wheel zoom, arrow-key panning, numpad zoom, and Alt+drag window move.
- Render algorithm selection, antialiasing controls, iteration controls, iteration precision, LA settings, perturbation mode controls, palette controls, and built-in views.
- Feature finder commands, NR checkpoint resume enablement, and NR inner-loop backend selection.
- Help modals, current-position modal, and clipboard copy for render details.
- ImGui file/pick/location dialogs for saving and loading locations, PNG images, iteration text, and reference orbits.
- Host-owned GLX presentation path in `FractalSharkGuiLinux/main.cpp`.

## Windows-Only Custom Heap Allocator

Win32 has a process-wide custom heap path in `HpSharkFloatLib/heap_allocator/HeapCpp.cpp`. Linux does
not use this allocator.

### What Win32 Uses

- `HeapCpp.cpp` overrides `malloc`, `calloc`, `realloc`, `free`, `aligned_alloc`, and global
  `new`/`delete` when `EnableFractalSharkHeap == FancyHeap::Enable`.
- The heap is backed by `GrowableVector<uint8_t>` over `HeapFile.bin`.
- Allocation metadata uses bins and free-list coalescing.
- Debug/safety checks include tail canaries, header guards, metadata checksums, free poisoning, double
  free detection, buffer overrun/underflow detection, and use-after-free detection.
- `Environment::RegisterHeapCleanup()` in `HeapCpp.cpp` registers cleanup on the fancy-heap path and
  initializes vector support on the disabled/safemode path.
- The bypass/system-heap path calls `Environment::SystemHeap*`, which maps to Win32 process-heap calls
  in `FractalSharkPlatform/EnvironmentWin32.cpp`.

### What Linux Uses Instead

- `FractalSharkPlatform/EnvironmentLinux.cpp` routes `SystemHeapAlloc`, `SystemHeapRealloc`,
  `SystemHeapAllocZeroed`, `SystemHeapFree`, and `SystemHeapSize` through glibc
  `malloc`, `realloc`, `calloc`, `free`, and `malloc_usable_size`.
- `Environment::RegisterHeapCleanup()` is a Linux stub. The current comment says the custom heap is
  Windows-only by design and that there is nothing to clean up on Linux.
- This is distinct from the scoped MPIR allocation helpers such as `MPIRBoundedAllocator` and
  `MPIRBumpAllocator`, which are reference-orbit/MPIR allocation tools rather than the process-wide
  `malloc`/`new` override.

### Parity Decision

Treat the lack of `HeapCpp` on Linux as an explicit platform difference for now. Revisit only if Linux
needs the same heap diagnostics, persistent allocation behavior, or process-wide allocation override
semantics as the Win32 GUI.

## Source References

- Linux GUI shell: `FractalSharkGuiLinux/main.cpp`
- Linux menu state: `FractalSharkGuiLinux/LinuxMenuState.cpp`
- Linux platform-agnostic command handlers: `FractalSharkGuiLinux/LinuxCommandHandlers.cpp`
- Win32 GUI shell: `FractalSharkGUILib/MainWindow.cpp`
- Win32 native dynamic command dispatcher: `FractalSharkGUILib/CommandDispatcher.cpp`
- Win32 native dynamic command IDs: `FractalShark/resource.h`
- Windows-only custom heap: `HpSharkFloatLib/heap_allocator/HeapCpp.cpp`
- Win32 system heap wrappers: `FractalSharkPlatform/EnvironmentWin32.cpp`
- Linux system heap wrappers and cleanup stub: `FractalSharkPlatform/EnvironmentLinux.cpp`
