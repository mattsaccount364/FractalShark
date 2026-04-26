// LinuxHelpModals.h — UTF-8 help / hotkey strings for the Linux GUI's
// info modals.  The text is duplicated verbatim from
// FractalSharkGUILib/MainWindow.cpp's MenuShowHotkeys / MenuViewsHelp /
// MenuAlgHelp message-box bodies, with the Win32 wide-string + CRLF
// rewritten as plain UTF-8 + LF.  Keep this file in sync with those
// three Win32 helpers when their wording changes.
#pragma once

namespace FractalSharkLinux {

inline constexpr const char *kAlgorithmsModalTitle = "Algorithms";
inline constexpr const char *kAlgorithmsModalBody =
    "Algorithms\n"
    "\n"
    "- As a general recommendation, choose AUTO.  Auto will render the fractal using "
    "direct 32-bit evaluation at the lowest zoom depths. "
    "From 1e4 to 1e9, it uses perturbation + 32-bit floating point. "
    "From 1e9 to 1e34, it uses perturbation + 32-bit + linear approximation.  "
    "Past that, it uses perturbation a 32-bit \"high dynamic range\" implementation, "
    "which simply stores the exponent in a separate integer.\n"
    "\n"
    "- If you try rendering \"hard\" points, you may find that the 32-bit implementations "
    "are not accurate enough.  In this case, you can try the 64-bit implementations.  "
    "You may also find the 2x32 implementations to be faster than the 1x64."
    "Generally, it's probably easiest to use the 32-bit implementations, and only "
    "switch to the 64-bit implementations when you need to.\n"
    "\n"
    "Note that professional/high-end chips offer superior 64-bit performance, so if you have one "
    "of those, you may find that the 64-bit implementations work well.  Most consumer GPUs offer"
    "poor 64-bit performance (even RTX 4090, 5090 etc)\n";

inline constexpr const char *kViewsModalTitle = "Views";
inline constexpr const char *kViewsModalBody =
    "Views\n"
    "\n"
    "The purpose of these is simply to make it easy to navigate to\n"
    "some interesting locations.\n";

inline constexpr const char *kHotkeysModalTitle = "Hotkeys";
inline constexpr const char *kHotkeysModalBody =
    "Hotkeys\n"
    "\n"
    "Navigation\n"
    "a - Autozoom using feature heuristic (zooms toward perturbation reference point).  Hold CTRL "
    "to abort.\n"
    "A - Autozoom using default heuristic (weighted geometric mean of iteration counts).  Hold "
    "CTRL to abort.\n"
    "b - Go back to the previous view\n"
    "c - Center the view at the current mouse position\n"
    "C - Center the view at the current mouse position + recalculate reference orbit\n"
    "z - Zoom in predefined amount\n"
    "Z - Zoom out predefined amount\n"
    "Left click/drag - Zoom in\n"
    "\n"
    "Recaluating and Benchmarking\n"
    "I - Clear medium-res perturbation results, recalculate, and benchmark\n"
    "i - Recalculate and benchmark current display, reusing perturbation results\n"
    "O - Clear high-res perturbation results, recalculate, and benchmark\n"
    "o - Recalculate and benchmark current display, reusing perturbation results\n"
    "P - Clear all perturbation results and recalculate\n"
    "p - Recalculate current display, reusing perturbation results\n"
    "R - Clear all perturbation results and recalculate\n"
    "r - Recalculate current display, reusing perturbation results\n"
    "\n"
    "Reference Compression\n"
    "e - Clear all perturbation results, reset error exponent to 19 (default).  Recalculate.\n"
    "q - Decrease intermediate orbit compression: less error, more memory. Recalculate.\n"
    "Q - Increase intermediate orbit compression: more error, less memory. Recalculate.\n"
    "w - Decrease reference compression: less error, more memory. Recalculate.\n"
    "W - Increase reference compression: more error, less memory. Recalculate.\n"
    "\n"
    "Linear Approximation parameters, adjustments by powers of two\n"
    "H - Decrease LA Threshold Scale exponents.  More accurate/slower per-pixel\n"
    "h - Increase LA Threshold Scale exponents.  Less accurate/faster per-pixel\n"
    "J - Decrease LA period detection exponents.  Less memory/slower per-pixel\n"
    "j - Increase LA period detection exponents.  More memory/faster per-pixel\n"
    "\n"
    "Palettes\n"
    "T - Use prior auxiliary palette depth (mul/div iteration count by 2)\n"
    "t - Use next auxiliary palette depth (mul/div iteration count by 2)\n"
    "D - Create and use new random palette\n"
    "d - Use next palette lookup table depth\n"
    "\n"
    "Iterations\n"
    "Use these keys to increase/decrease the number of iterations used to calculate the fractal.\n"
    "= - Multiply max iterations by 24\n"
    "- - Multiply max iterations by 2/3\n"
    "\n"
    "Feature Finder\n"
    "n - Find periodic point at cursor (Direct mode)\n"
    "N - Find periodic point at cursor (DirectScan mode)\n"
    "m - Find periodic point at cursor (PT mode)\n"
    "M - Find periodic point at cursor (PTScan mode)\n"
    ", - Find periodic point at cursor (LA mode)\n"
    "< - Find periodic point at cursor (LAScan mode)\n"
    ". - Zoom to found feature\n"
    "> - Clear all found features\n"
    "\n"
    "Misc\n"
    "CTRL - Press and hold to abort autozoom\n"
    "ALT - Press, click/drag to move window when in windowed mode\n"
    "Right click - popup menu\n";

} // namespace FractalSharkLinux
