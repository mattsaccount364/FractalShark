#pragma once

// Standalone crash-handler module.  Call CrashHandler::Install() once
// early in process lifetime (before any work that might crash).

struct CrashHandler {
    // Registers all crash/termination handlers and reserves extra stack
    // space for the handler itself.  Safe to call exactly once.
    static void Install();
};
