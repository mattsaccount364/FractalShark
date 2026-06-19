// LinuxHelpModals.h — UTF-8 help strings for the Linux GUI's info modals.
// The static text here mirrors FractalSharkGUILib/MainWindow.cpp's
// MenuViewsHelp / MenuAlgHelp message-box bodies, with the Win32 wide-string
// + CRLF rewritten as plain UTF-8 + LF. Hotkey help is generated from
// FractalShark::kCommands instead of duplicated here.
#pragma once

namespace FractalShark::Linux {

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

} // namespace FractalShark::Linux
