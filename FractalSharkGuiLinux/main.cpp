// FractalSharkGuiLinux - Linux GUI entry point.

#include "CrashHandler.h"
#include "Environment.h"
#include "Exceptions.h"
#include "LinuxMainWindow.h"
#include "LinuxSplashWindow.h"

#include <X11/Xlib.h>

#include <exception>
#include <iostream>

int
main(int /*argc*/, char ** /*argv*/)
{
    Environment::RegisterHeapCleanup();
    Environment::CrashHandler::Install();

    // Xlib functions are touched from both the GUI thread and the GL presentation path.
    // This must happen before any other Xlib call.
    try {
        if (XInitThreads() == 0) {
            throw FractalSharkSeriousException("XInitThreads failed");
        }

        FractalShark::Linux::SplashWindow splash;
        splash.Start();

        return FractalShark::Linux::RunMainWindow([&splash] { splash.Stop(); });
    } catch (const std::exception &exception) {
        std::cerr << "FractalSharkGuiLinux: " << exception.what() << '\n';
        return 1;
    }
}
