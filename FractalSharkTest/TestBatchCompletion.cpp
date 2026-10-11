#include "FractalSharkCli/BatchCompletion.h"
#include "TestFramework.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

FractalSharkCli::PendingBatchResult
MakePending(std::promise<SaveResult> &completion)
{
    FractalSharkCli::PendingBatchResult pending;
    pending.Result.Response.Status = 0;
    pending.Result.CliImageMs = 37;
    pending.PngCompletion = completion.get_future();
    pending.Output = "requested.png";
    pending.AnnounceOutput = true;
    return pending;
}

} // namespace

TEST(BatchCompletion_PollingPreservesOverlapAndOrder)
{
    FractalSharkCli::BatchCompletionQueue queue;
    std::promise<SaveResult> first;
    std::promise<SaveResult> second;
    queue.Push(0, MakePending(first));
    queue.Push(1, MakePending(second));
    second.set_value({SaveOutcome::Written, L"second.png", 0, {}});

    FractalSharkCli::PendingBatchResult invalid;
    invalid.Result.Response.Status = 2;
    invalid.Result.Response.Stderr = "invalid view";
    queue.Push(2, std::move(invalid));

    std::vector<size_t> indexes;
    std::vector<FractalSharkCli::IpcBatchResult> results;
    const auto report = [&](size_t index, const FractalSharkCli::IpcBatchResult &result) {
        indexes.push_back(index);
        results.push_back(result);
        return true;
    };
    // A guarded thread keeps this regression test finite if polling accidentally blocks.
    // Release the first save and join before any assertion can unwind the test.
    std::promise<bool> polled;
    auto returned = polled.get_future();
    std::thread poller([&] { polled.set_value(queue.ReportReady(report)); });
    const bool returnedWithoutWaiting =
        returned.wait_for(std::chrono::seconds{5}) == std::future_status::ready;
    first.set_value({SaveOutcome::Written, L"first.png", 0, {}});
    poller.join();
    ASSERT_TRUE(returnedWithoutWaiting);
    ASSERT_TRUE(returned.get());
    ASSERT_TRUE(indexes.empty());

    ASSERT_TRUE(queue.ReportReady(report));
    ASSERT_TRUE(indexes == std::vector<size_t>({0, 1, 2}));
    ASSERT_EQ(results[0].Response.Status, 0);
    ASSERT_EQ(results[0].CliImageMs, uint64_t{37});
    ASSERT_EQ(results[0].Response.Stdout, "Wrote first.png\n");
    ASSERT_EQ(results[1].Response.Stdout, "Wrote second.png\n");
    ASSERT_EQ(results[2].Response.Status, 2);
    ASSERT_EQ(results[2].Response.Stderr, "invalid view");
    ASSERT_TRUE(queue.ReportAll(report));
    ASSERT_EQ(indexes.size(), size_t{3});
}

TEST(BatchCompletion_SaveFailuresAndExceptionsRemainPerImage)
{
    FractalSharkCli::BatchCompletionQueue queue;
    std::promise<SaveResult> failed;
    std::promise<SaveResult> succeeded;
    std::promise<SaveResult> exception;
    std::promise<SaveResult> unknownException;
    queue.Push(0, MakePending(failed));
    queue.Push(1, MakePending(succeeded));
    queue.Push(2, MakePending(exception));
    queue.Push(3, MakePending(unknownException));
    failed.set_value({SaveOutcome::Failed, L"resolved.png", 13, "permission denied"});
    succeeded.set_value({SaveOutcome::Written, L"valid.png", 0, {}});
    exception.set_exception(std::make_exception_ptr(std::runtime_error("encoder threw")));
    unknownException.set_exception(std::make_exception_ptr(7));

    std::vector<FractalSharkCli::IpcBatchResult> results;
    ASSERT_TRUE(queue.ReportAll([&](size_t, const FractalSharkCli::IpcBatchResult &result) {
        results.push_back(result);
        return true;
    }));
    ASSERT_EQ(results.size(), size_t{4});
    ASSERT_EQ(results[0].Response.Status, 1);
    ASSERT_TRUE(results[0].Response.Stderr.find("resolved.png") != std::string::npos);
    ASSERT_TRUE(results[0].Response.Stderr.find("permission denied") != std::string::npos);
    ASSERT_TRUE(results[0].Response.Stdout.empty());
    ASSERT_EQ(results[1].Response.Status, 0);
    ASSERT_TRUE(results[1].Response.Stderr.empty());
    ASSERT_EQ(results[1].Response.Stdout, "Wrote valid.png\n");
    ASSERT_EQ(results[2].Response.Status, 1);
    ASSERT_TRUE(results[2].Response.Stderr.find("requested.png") != std::string::npos);
    ASSERT_TRUE(results[2].Response.Stderr.find("encoder threw") != std::string::npos);
    ASSERT_TRUE(results[2].Response.Stdout.empty());
    ASSERT_EQ(results[3].Response.Status, 1);
    ASSERT_TRUE(results[3].Response.Stderr.find("unknown save-worker exception") != std::string::npos);
}

TEST(BatchCompletion_SkippedAndQuietOutputsDoNotAnnounceWrites)
{
    FractalSharkCli::BatchCompletionQueue queue;
    std::promise<SaveResult> skipped;
    std::promise<SaveResult> quiet;
    queue.Push(0, MakePending(skipped));
    auto pending = MakePending(quiet);
    pending.AnnounceOutput = false;
    queue.Push(1, std::move(pending));
    skipped.set_value({SaveOutcome::SkippedExisting, L"existing.png", 0, {}});
    quiet.set_value({SaveOutcome::Written, L"quiet.png", 0, {}});
    size_t reported = 0;
    ASSERT_TRUE(queue.ReportAll([&](size_t, const FractalSharkCli::IpcBatchResult &result) {
        ++reported;
        ASSERT_EQ(result.Response.Status, 0);
        ASSERT_TRUE(result.Response.Stdout.empty());
        ASSERT_TRUE(result.Response.Stderr.empty());
        return true;
    }));
    ASSERT_EQ(reported, size_t{2});
}

TEST(BatchCompletion_FinalDrainWaitsForOutstandingSaves)
{
    FractalSharkCli::BatchCompletionQueue queue;
    std::promise<SaveResult> completion;
    queue.Push(0, MakePending(completion));
    std::promise<bool> drained;
    auto returned = drained.get_future();
    std::atomic<size_t> reported{0};
    std::thread reporter([&] {
        drained.set_value(queue.ReportAll([&](size_t, const FractalSharkCli::IpcBatchResult &) {
            reported.fetch_add(1);
            return true;
        }));
    });
    const bool finishedEarly =
        returned.wait_for(std::chrono::milliseconds{50}) == std::future_status::ready;
    completion.set_value({SaveOutcome::Written, L"finished.png", 0, {}});
    reporter.join();
    ASSERT_FALSE(finishedEarly);
    ASSERT_TRUE(returned.get());
    ASSERT_EQ(reported.load(), size_t{1});
}

TEST(BatchCompletion_DisconnectionStopsReporting)
{
    FractalSharkCli::BatchCompletionQueue queue;
    std::promise<SaveResult> first;
    std::promise<SaveResult> second;
    queue.Push(0, MakePending(first));
    queue.Push(1, MakePending(second));
    first.set_value({SaveOutcome::Written, L"first.png", 0, {}});
    second.set_value({SaveOutcome::Written, L"second.png", 0, {}});
    size_t attempts = 0;
    ASSERT_FALSE(queue.ReportAll([&](size_t, const FractalSharkCli::IpcBatchResult &) {
        ++attempts;
        return false;
    }));
    ASSERT_EQ(attempts, size_t{1});
}
