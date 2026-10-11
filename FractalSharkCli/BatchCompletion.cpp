#include "stdafx.h"

#include "BatchCompletion.h"

#include "Environment.h"

#include <chrono>
#include <exception>
#include <utility>

namespace FractalSharkCli {
namespace {

void
FailSave(PendingBatchResult &pending, const std::string &path, const std::string &diagnostic)
{
    pending.Result.Response.Status = 1;
    pending.Result.Response.Stderr +=
        "error: could not write PNG file \"" + path + "\": " + diagnostic + "\n";
}

void
CompleteSave(PendingBatchResult &pending)
{
    if (!pending.PngCompletion.valid()) {
        return;
    }
    SaveResult save;
    // A packaged task transports exceptions across the save-worker boundary. Convert
    // them here to an image failure so subsequent batch images can still be reported.
    try {
        save = pending.PngCompletion.get();
    } catch (const std::exception &exception) {
        FailSave(pending, pending.Output, exception.what());
        return;
    } catch (...) {
        FailSave(pending, pending.Output, "unknown save-worker exception");
        return;
    }
    const auto path =
        save.OutputPath.empty() ? pending.Output : Environment::WideToUtf8(save.OutputPath);
    if (!save.Succeeded()) {
        FailSave(pending, path, save.Diagnostic.empty() ? "PNG save failed" : save.Diagnostic);
    } else if (save.Outcome == SaveOutcome::Written && pending.Result.Response.Status == 0 &&
               pending.AnnounceOutput) {
        pending.Result.Response.Stdout += "Wrote " + path + "\n";
    }
}

} // namespace

void
BatchCompletionQueue::Push(size_t index, PendingBatchResult pending)
{
    m_Pending.push_back({index, std::move(pending)});
}

bool
BatchCompletionQueue::ReportReady(const Reporter &reporter)
{
    return Report(false, reporter);
}

bool
BatchCompletionQueue::ReportAll(const Reporter &reporter)
{
    return Report(true, reporter);
}

bool
BatchCompletionQueue::Report(bool wait, const Reporter &reporter)
{
    while (!m_Pending.empty()) {
        const auto &completion = m_Pending.front().Pending.PngCompletion;
        if (!wait && completion.valid() &&
            completion.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
            return true;
        }
        auto entry = std::move(m_Pending.front());
        m_Pending.pop_front();
        CompleteSave(entry.Pending);
        if (!reporter(entry.Index, entry.Pending.Result)) {
            return false;
        }
    }
    return true;
}

} // namespace FractalSharkCli
