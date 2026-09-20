// FractalSharkGuiLinux - Linux GUI entry point.

#include "ConsoleLog.h"
#include "CrashHandler.h"
#include "Environment.h"
#include "Exceptions.h"
#include "LinuxMainWindow.h"
#include "LinuxSplashWindow.h"
#include "heap_allocator/include/HeapCpp.h"

#include <X11/Xlib.h>

#include <exception>
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
        FractalSharkLog::WriteException("FractalSharkGuiLinux", exception, __FILE__, __LINE__);
        return 1;
    }
}
