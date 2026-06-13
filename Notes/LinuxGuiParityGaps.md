# Linux GUI Parity Gaps

Last reviewed: 2026-06-12.

This file tracks Linux GUI parity work against the Win32 GUI baseline. It is a working checklist for
fixes and deliberate platform differences, not user-facing documentation.

## Core GUI Gaps

| Area | Linux state | Win32 baseline | Notes / likely fix |
| --- | --- | --- | --- |
| Window mode behavior | Linux toggles EWMH fullscreen and supports Alt+drag window move. | Win32 has custom borderless/windowed behavior, non-client suppression, and resize hit-test handling. | Either implement Linux borderless resize/move behavior or document that Linux delegates this to the window manager. |
| Palette rotation | Linux command exists but appears visually broken. | Win32 has the same TODO and visible behavior problem. | Shared behavior likely needs a render/recolor invalidation path after `RotateFractalPalette()`. |

## Deliberate Platform Differences

- Runtime memory-limit menu controls were removed because neither backend had reliable reversible
  behavior. Win32 keeps its startup `JobObject` cap, while the Linux GUI currently starts unlimited.
- Linux diagnostics remain stderr-first. Win32 also has a startup background console, but Linux does
  not add a separate console surface.

## Already Present On Linux

- Shared `MenuTreeDef.h` context menu structure and command catalog dispatch.
- Native X11 context menu windows with nested submenus, radio/check state, disabled items, scrolling, and keyboard navigation.
- Hotkey dispatch from X11 key events through `CommandCatalog`.
- Drag zoom, wheel zoom, arrow-key panning, numpad zoom, and Alt+drag window move.
- Render algorithm selection, antialiasing controls, iteration controls, iteration precision, LA settings, perturbation mode controls, palette controls, and built-in views.
- Feature finder commands, NR checkpoint resume enablement, and NR inner-loop backend selection.
- Help modals, current-position modal, and clipboard copy for render details.
- ImGui file/pick/location dialogs for saving and loading locations, PNG images, iteration text, and reference orbits.
- Randomized image splash window using compiled-in copies of the same four splash PNGs as Win32. It
  appears before the heavier X11/GLX main-window initialization and closes before the main event loop
  starts.
- Host-owned GLX presentation path in `FractalSharkGuiLinux/main.cpp`.

## Source References

- Linux GUI shell: `FractalSharkGuiLinux/main.cpp`
- Linux startup splash: `FractalSharkGuiLinux/LinuxSplashWindow.cpp`
- Linux menu state: `FractalSharkGuiLinux/LinuxMenuState.cpp`
- Linux platform-agnostic command handlers: `FractalSharkGuiLinux/LinuxCommandHandlers.cpp`
- Win32 GUI shell: `FractalSharkGUILib/MainWindow.cpp`
- Win32 native dynamic command dispatcher: `FractalSharkGUILib/CommandDispatcher.cpp`
- Win32 native dynamic command IDs: `FractalShark/resource.h`
