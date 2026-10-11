#pragma once

#include "LocalIpc.h"
#include "SaveResult.h"

#include <deque>
#include <functional>
#include <future>
#include <string>

namespace FractalSharkCli {

// Render statistics are captured before the Fractal starts another image. Only the
// save outcome is filled in later; no completion code reads mutable render state.
struct PendingBatchResult {
    IpcBatchResult Result;
    std::future<SaveResult> PngCompletion;
    std::string Output;
    bool AnnounceOutput = false;
};

class BatchCompletionQueue {
public:
    // False stops reporting (for example, after a client disconnects).
    using Reporter = std::function<bool(size_t, const IpcBatchResult &)>;

    void Push(size_t index, PendingBatchResult pending);
    // Never wait between renders. Later completed saves cannot overtake an earlier image.
    bool ReportReady(const Reporter &reporter);
    // At the end of a batch, wait for and report the remaining images in order.
    bool ReportAll(const Reporter &reporter);

private:
    struct Entry {
        size_t Index;
        PendingBatchResult Pending;
    };
    bool Report(bool wait, const Reporter &reporter);
    std::deque<Entry> m_Pending;
};

} // namespace FractalSharkCli
