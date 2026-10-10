#pragma once

#include <string>

enum class SaveOutcome { Written, SkippedExisting, Failed };
enum class SaveOverwritePolicy { SkipExisting, ReplaceExisting };

// Local result of synchronous encoding and file publication. Queue submission does
// not imply that a file exists. A skipped existing destination is successful.
struct SaveResult {
    SaveOutcome Outcome = SaveOutcome::Written;
    std::wstring OutputPath;
    int ErrorCode = 0;
    std::string Diagnostic;

    bool
    Succeeded() const
    {
        return Outcome != SaveOutcome::Failed;
    }
};
