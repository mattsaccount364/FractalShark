// LinuxXlibBackend — custom Dear ImGui platform backend for Xlib.
//
// Drop-in replacement for upstream `imgui_impl_glfw.cpp`.  The renderer side
// stays as upstream `imgui_impl_opengl2.cpp` (or _opengl3) — that's pure GL,
// platform-agnostic.
//
// Wiring (called by FractalSharkGuiLinux main loop):
//
//     ImGui::CreateContext();
//     ImGui_ImplXlib_Init(display, window);
//     ImGui_ImplOpenGL2_Init();
//     while (running) {
//         while (XPending(display)) {
//             XNextEvent(display, &ev);
//             if (clipboard.ProcessEvent(ev)) continue;
//             if (ImGui_ImplXlib_ProcessEvent(ev)) continue;
//             // app-specific dispatch (drag-zoom, render, ...)
//         }
//         ImGui_ImplOpenGL2_NewFrame();
//         ImGui_ImplXlib_NewFrame();
//         ImGui::NewFrame();
//         // ... draw ImGui windows ...
//         ImGui::Render();
//         ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
//     }
//     ImGui_ImplOpenGL2_Shutdown();
//     ImGui_ImplXlib_Shutdown();
//     ImGui::DestroyContext();
//
// Clipboard:  ImGui's Set/Get clipboard callbacks delegate to the
// `LinuxClipboard` instance the host provides via SetClipboardHelper.

#pragma once

#include <X11/Xlib.h>

namespace FractalShark {
struct LinuxClipboard;
} // namespace FractalShark

bool ImGui_ImplXlib_Init(Display *display, Window window);
void ImGui_ImplXlib_Shutdown();
void ImGui_ImplXlib_NewFrame();

// Returns true if the event was consumed by ImGui (e.g. mouse over a popup,
// keystroke targeted at a text input).  Host should still call its own
// dispatch when this returns false.
bool ImGui_ImplXlib_ProcessEvent(const XEvent &ev);

// Optional: route ImGui clipboard Get/Set through the host's LinuxClipboard
// (so ImGui text widgets share the same CLIPBOARD selection as the rest of
// the app).  May be called once after Init.
void ImGui_ImplXlib_SetClipboardHelper(FractalShark::LinuxClipboard *clipboard);
