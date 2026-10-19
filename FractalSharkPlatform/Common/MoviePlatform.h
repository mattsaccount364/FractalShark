#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Environment {

// Movie jobs alone use these durability operations. Native reference mappings and
// ordinary PNG publication deliberately retain their existing flush semantics.
void FlushMovieFile(const std::filesystem::path &path);
void PublishMovieFile(const std::filesystem::path &temporary,
                      const std::filesystem::path &target,
                      bool replace);

class MovieJobLock {
public:
    explicit MovieJobLock(const std::filesystem::path &path);
    ~MovieJobLock();
    MovieJobLock(const MovieJobLock &) = delete;
    MovieJobLock &operator=(const MovieJobLock &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};

struct MovieProcessResult {
    int ExitCode;
    std::string ErrorOutput;
};

// No shell is involved. Both output pipes are drained concurrently with the
// producer, with bounded stderr retention. Callback exceptions terminate/reap
// the child and propagate; native failures use std::system_error.
MovieProcessResult RunMovieProcess(
    const std::filesystem::path &executable,
    const std::vector<std::string> &arguments,
    const std::function<void(const std::function<void(std::span<const unsigned char>)> &)> &input,
    const std::function<void(std::span<const unsigned char>)> &output,
    const std::function<void(std::string_view)> &errorOutput);

} // namespace Environment
