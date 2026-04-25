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
| Begin gesture                               | `WM_LBUTTONDOWN` → `MainWindow.cpp:819`. `lButtonDown=true`, store `(dragBoxX1, dragBoxY1)`.         | `ButtonPress` (button 1) → `LinuxInput.cpp` (TBD: input-mouse).                                        |
| Alt+LMB                                     | Drags the window via `WM_NCLBUTTONDOWN HTCAPTION` (`MainWindow.cpp:820`).                            | WM-driven move via `_NET_WM_MOVERESIZE` ClientMessage with `_NET_WM_MOVERESIZE_MOVE`.                  |
| Live update on move                         | `WM_MOUSEMOVE` `MainWindow.cpp:907` — `InvertRect` XOR for outline.                                  | `MotionNotify` → `ImGui::GetForegroundDrawList()->AddRect` with white outer + black inner outline.     |
| Aspect lock                                 | Default ON. Shift held = OFF. Computed from client rect ratio (`MainWindow.cpp:854-869`).            | Same: Shift state from `XKeyEvent.state & ShiftMask`. Compute ratio from `XGetWindowAttributes`.       |
| Aspect formula                              | `right = left + ratio * (bottom - top)` (`MainWindow.cpp:862`).                                      | Identical formula, identical operand order.                                                            |
| Commit on release                           | `WM_LBUTTONUP` `MainWindow.cpp:837` → `EnqueueCommand([rect, aspect](Fractal &f){ f.RecenterViewScreen(rect); if (aspect) f.SquareCurrentView(); })`. | Same lambda body. Identical order: recenter then optionally square.            |
| Cancel on focus loss                        | `WM_CANCELMODE` / `WM_CAPTURECHANGED` `MainWindow.cpp:885` — erase outline, `lButtonDown=false`.     | `FocusOut` and selection-clear from window manager (drag-cancellation events).                         |
| Outline rendering                           | `InvertRect` (XOR; legible on any background).                                                       | ImGui has no XOR; use `AddRect` with a thick white outer + thin black inner overlay (visibility hack). |
| Capture                                     | `SetCapture` / `ReleaseCapture` so motion events arrive when cursor leaves window.                   | `XGrabPointer(ButtonReleaseMask | PointerMotionMask, GrabModeAsync)` until `ButtonRelease`.            |

## Mouse interactions outside drag-to-zoom

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Wheel zoom                                  | `WM_MOUSEWHEEL` `MainWindow.cpp:955` — forward = zoom in, backward = zoom out.                       | `ButtonPress` button 4/5 with `XButtonEvent.state` decoded similarly.                                  |
| Right-click → context menu                  | `WM_RBUTTONDOWN` → `TrackPopupMenu` of the dynamic `HMENU` tree.                                     | `ButtonPress` button 3 sets `m_ShowContextMenu=true`; ImGui `BeginPopup` opens it next frame.          |
| Cursor                                      | `IDC_ARROW` while idle, `IDC_CROSS` during drag-zoom.                                                | `XC_left_ptr` / `XC_crosshair` via `XCreateFontCursor` + `XDefineCursor`.                              |

## Keyboard

| Behavior                                    | Win32                                                                                                | Linux                                                                                                  |
|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Discrete commands (menu items, toggles)     | `WM_KEYDOWN` → catalog lookup → `ExecuteCommand`.                                                    | `KeyPress` → catalog lookup → `ExecuteCommand`. Identical hotkey table.                                |
| Continuous nudge keys (`+`/`−`/arrows held) | Direct `Fractal::EnqueueCommand` lambdas — bypass catalog (latency-sensitive).                       | Same direct path. Mirror lambda bodies exactly.                                                        |
| Auto-repeat                                 | OS-driven; only `WM_KEYDOWN` repeats, never `WM_KEYUP`.                                              | `XkbSetDetectableAutoRepeat(True)` matches Win32 semantics. Set in `LinuxMainWindow` ctor.             |
| Modifier extraction                         | `GetAsyncKeyState(VK_SHIFT)` / `(GetKeyState(VK_CONTROL) & 0x8000)`.                                 | `XKeyEvent.state & ShiftMask` / `& ControlMask`. NOT `XQueryKeymap` (slow path).                       |
| Ctrl+A "select all" interception            | `MainWindow.cpp:1364` — special-cased.                                                               | Mirror in input-keys task.                                                                             |

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
| Open PNG / locations / orbit                | `GetOpenFileNameW` with `OPENFILENAME` filter strings.                                               | Hand-rolled ImGui modal (`LinuxFileDialog`, file-dialogs task). NO `zenity`/`kdialog`/portal dep.      |
| Filter strings (must match per dialog)      | `"PNG\0*.png\0All\0*.*\0"` etc. — see each call site in `MainWindow.cpp`.                            | Mirror filter list exactly. Validated manually until we add a parity unit test.                        |
| Last-directory persistence                  | Stored in `OPENFILENAME.lpstrInitialDir` between calls.                                              | Stored on `LinuxMainWindow` member; passed to `LinuxFileDialog::Show`.                                 |
| Overwrite confirm                           | `OFN_OVERWRITEPROMPT` for Save dialogs.                                                              | Manual ImGui modal: "File exists. Overwrite? [Yes/No]".                                                |

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
- **Menu bar visual style** — Win32 native menus vs ImGui menus look different.
- **OS file-drop onto window** (drag a `.png` from a file manager). Linux
  needs `XdndSelection`; deferred until a user asks.
- **Wayland** — `OpenGLContext.cpp` is GLX-only. EGL port follow-up.

## Updating this file

When you change any of the cited Win32 line numbers (e.g. via a refactor in
`MainWindow.cpp`), grep this file for the old line number and update. A failed
review hint is generally better than a stale citation.
