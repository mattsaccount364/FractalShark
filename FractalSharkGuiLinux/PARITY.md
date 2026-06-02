# PARITY.md — Win32 ↔ Linux GUI behavior parity

This file is the **manual cross-check list** that PR reviewers use to verify
behavior parity between `FractalSharkGUILib` (Win32) and `FractalSharkGuiLinux`
(Xlib + Dear ImGui).

The `FractalCommand` catalog (`FractalSharkLib/CommandCatalog.h`) handles
discrete enum-dispatched commands and is statically enforced by both compilers
(`-Werror=switch-enum` / MSVC `/we4062`). This file covers the things
**outside** the catalog: gestures, modal look-and-feel, file-dialog filter
strings, clipboard payload format, etc. — anything where the two GUIs could
silently diverge without a compile error.

When a PR changes any row, update **both** sides in the same PR (or document
why the platforms intentionally differ). When in doubt, Win32 is authoritative
— Linux mirrors Win32 semantics exactly unless the row says otherwise.

## Drag-to-zoom (rubber-band rectangle)

The flagship interaction. Mirror Win32 semantics exactly.

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Begin gesture                               | `WM_LBUTTONDOWN` → `MainWindow.cpp`. `lButtonDown=true`, store `(dragBoxX1, dragBoxY1)`.             | `ButtonPress` (button 1) → `main.cpp`. `dragging=true`, store `(dragAnchorX, dragAnchorY)`.             |
| Alt+LMB                                     | Drags the window via `WM_NCLBUTTONDOWN HTCAPTION` (`MainWindow.cpp:820`).                            | WM-driven move via `_NET_WM_MOVERESIZE` ClientMessage with `_NET_WM_MOVERESIZE_MOVE`.                  |
| Live update on move                         | `WM_MOUSEMOVE` `MainWindow.cpp` updates the GL drag rectangle.                                      | `MotionNotify` → `ImGui::GetForegroundDrawList()->AddRect`. `TODO(linux-parity)`: add white outer + black inner strokes. |
| Aspect lock                                 | Default ON. Shift held = OFF. Computed from client rect ratio (`MainWindow.cpp:854-869`).            | Same: Shift state from `XKeyEvent.state & ShiftMask`. Compute ratio from `XGetWindowAttributes`.       |
| Aspect formula                              | `right = left + ratio * (bottom - top)` (`MainWindow.cpp:862`).                                      | Identical formula, identical operand order.                                                            |
| Commit on release                           | `WM_LBUTTONUP` `MainWindow.cpp:837` → `EnqueueCommand([rect, aspect](Fractal &f){ f.RecenterViewScreen(rect); if (aspect) f.SquareCurrentView(); })`. | Same lambda body. Identical order: recenter then optionally square.            |
| Cancel on focus loss                        | `WM_CANCELMODE` / `WM_CAPTURECHANGED` `MainWindow.cpp` erase the outline and clear `lButtonDown`.     | `FocusOut` clears `dragging` and hides the overlay.                                                     |
| Outline rendering                           | GL drag rectangle rendered by the Win32 presentation path.                                           | `TODO(linux-parity)`: replace the single magenta stroke with thick white outer + thin black inner strokes. |
| Capture                                     | `SetCapture` / `ReleaseCapture` so motion events arrive when cursor leaves window.                   | `TODO(linux-parity)`: explicitly `XGrabPointer` until `ButtonRelease`; current code relies on implicit delivery. |

## Mouse interactions outside drag-to-zoom

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Wheel zoom                                  | `WM_MOUSEWHEEL` `MainWindow.cpp:955` — forward = zoom in, backward = zoom out.                       | `ButtonPress` button 4/5 with `XButtonEvent.state` decoded similarly.                                  |
| Right-click → context menu                  | `WM_RBUTTONDOWN` → `TrackPopupMenu` of the dynamic `HMENU` tree.                                     | `ButtonPress` button 3 opens native root-level X11 popup windows, allowing menus to extend outside the client area. |
| Cursor                                      | `IDC_ARROW` while idle, `IDC_CROSS` during drag-zoom.                                                | `TODO(linux-parity)`: add `XC_left_ptr` / `XC_crosshair` switching via `XCreateFontCursor` + `XDefineCursor`. |

## Keyboard

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Discrete commands (menu items, toggles)     | Context-menu items route through `ExecuteCommand`; character shortcuts remain in `HandleKeyDown`.    | Context-menu items route through `ExecuteCommand`; character shortcuts remain in `HandleKeyPress`.     |
| Continuous nudge keys (`+`/`−`/arrows held) | Direct `Fractal::EnqueueCommand` lambdas — bypass catalog (latency-sensitive).                       | Same direct path. Mirror lambda bodies exactly.                                                        |
| Auto-repeat                                 | OS-driven; only `WM_KEYDOWN` repeats, never `WM_KEYUP`.                                              | `XkbSetDetectableAutoRepeat(True)` matches Win32 semantics. Set in `LinuxMainWindow` ctor.             |
| Modifier extraction                         | `GetAsyncKeyState(VK_SHIFT)` / `(GetKeyState(VK_CONTROL) & 0x8000)`.                                 | `XKeyEvent.state & ShiftMask` / `& ControlMask`. NOT `XQueryKeymap` (slow path).                       |
| Ctrl+A "select all" interception            | Win32 edit controls special-case Ctrl+A.                                                             | Dear ImGui `InputText` handles text selection.                                                         |

## Clipboard

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Format                                      | `CF_TEXT` (8-bit ANSI). `MainWindow.cpp:566`, `:1156`.                                               | `UTF8_STRING` target, fallback `STRING`/`TEXT`. UTF-8 round-trips with ASCII-only payloads we send.    |
| Selection                                   | System clipboard.                                                                                    | `CLIPBOARD` atom (NOT `PRIMARY`). `LinuxClipboard.cpp`.                                                |
| Set                                         | `OpenClipboard` → `EmptyClipboard` → `SetClipboardData(CF_TEXT, GMEM_MOVEABLE handle)`.              | `XSetSelectionOwner(CLIPBOARD)`; respond to `SelectionRequest` events with stored payload.             |
| Get                                         | `OpenClipboard` → `GetClipboardData(CF_TEXT)` → copy.                                                | `XConvertSelection` + synchronous pump for `SelectionNotify` (250ms deadline).                         |
| INCR (huge payloads)                        | OS-handled.                                                                                          | Not supported. Coordinate strings are ≪ X11 max-request-size.                                          |

## File dialogs

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Open / save orbit                           | `GetOpenFileNameW` / `GetSaveFileNameW` with `OPENFILENAME`.                                        | `TODO(linux-parity)`: replace the filename-only save modal and cwd-only pick lists with a browsable open/save dialog. |
| Filter strings                              | `"All\0*.*\0Imagina\0*.im\0"` in `MainWindow.cpp`.                                                   | `TODO(linux-parity)`: mirror open/save filters in the browsable dialog.                                |
| Last-directory persistence                  | Not currently persisted: `lpstrInitialDir = NULL`.                                                   | `TODO(linux-parity, follow-up)`: decide whether to add persistence to both GUIs.                        |
| Overwrite confirm                           | Not currently requested: save-dialog flags are `0`.                                                  | `TODO(linux-parity, follow-up)`: decide whether to add overwrite confirmation to both GUIs.             |

## Saved locations

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Save location                               | Prompt for maximum/current dimensions, then append a `FractalTrayDestination` record to `locations.txt`. | Same record format and dimension choice via an ImGui pick-list modal.                                  |
| Load location                               | Read `locations.txt`, show saved descriptions, and activate the selected view.                        | Same parser shape and activation behavior via an ImGui pick-list modal.                                |

## Window lifecycle

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Close button                                | `WM_CLOSE` → `DestroyWindow` → `WM_DESTROY` → `PostQuitMessage`.                                     | `ClientMessage` w/ `WM_DELETE_WINDOW` atom → `running=false` → exit event loop.                        |
| Resize                                      | `WM_SIZE` → `Fractal::ResetDimensions(w, h)`.                                                        | `ConfigureNotify` (filter to size changes only) → same `ResetDimensions` call.                         |
| Initial size                                | 1600×1000 (legacy default).                                                                          | 1600×1000. `kInitialWidth/Height` in `main.cpp`.                                                       |
| Title                                       | `"FractalShark"`.                                                                                    | `"FractalShark (Linux)"` — distinguishable for screenshots/bug reports.                                |
| Crash handler                               | SEH + MiniDump (`CrashHandlerWin32.cpp`).                                                            | sigaction(SIGSEGV/SIGABRT/SIGFPE/SIGILL/SIGBUS) + `std::stacktrace` + re-raise (`CrashHandlerLinux.cpp`). |

## Out-of-scope (deliberately divergent)

- **Window decorations / WM theming** — let each OS do its thing.
- **Menu visual style** — Win32 native menus vs Xlib-drawn Linux menus look different.
- **OS file-drop onto window** (drag a `.png` from a file manager).
  `TODO(linux-parity, deferred)`: Linux needs `XdndSelection`.
- **Wayland**. `TODO(linux-parity, deferred)`: `OpenGLContext.cpp` is GLX-only;
  add an EGL backend as a follow-up.

## Searchable Linux backlog

Search source and this file with `rg "TODO\\(linux-parity"`.

- `TODO(linux-parity)`: implement the command fallback, memory-limit toggles,
  and palette rotation currently stubbed in `LinuxMainWindow`.
- `TODO(linux-parity)`: keep menu autozoom responsive while Xlib/ImGui events
  are pending.
- `TODO(linux-parity)`: implement global cursor coordinates, UTF-8 filesystem
  path conversion, explicit drag capture, cursor switching, and the two-stroke
  drag outline.
- `TODO(linux-parity, deferred)`: provide Linux GUI wait-cursor feedback.
- `TODO(linux-parity, permanent)`: keep heap cleanup as a no-op while Linux
  allocations use glibc.

## Updating this file

When you change any of the cited Win32 line numbers (e.g. via a refactor in
`MainWindow.cpp`), grep this file for the old line number and update. A failed
review hint is generally better than a stale citation.
