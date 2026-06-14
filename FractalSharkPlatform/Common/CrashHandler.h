#pragma once

// Standalone crash-handler module.  Call Environment::CrashHandler::Install() once
// early in process lifetime (before any work that might crash).

namespace Environment {

struct CrashHandler {
    // Registers all crash/termination handlers and reserves extra stack
    // space for the handler itself.  Safe to call exactly once.
    static void Install();
};

} // namespace Environment
