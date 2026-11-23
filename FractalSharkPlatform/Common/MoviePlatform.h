#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace Environment {

enum class MovieDriver { Cuda, Nvenc };

// Resolve driver APIs lazily so CPU programs can start without a GPU driver.
class MovieDriverLibrary {
public:
    explicit MovieDriverLibrary(MovieDriver driver);
    ~MovieDriverLibrary();
    MovieDriverLibrary(const MovieDriverLibrary &) = delete;
    MovieDriverLibrary &operator=(const MovieDriverLibrary &) = delete;
    void *Symbol(const char *name) const;

private:
    void *m_Library;
};

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

// A standalone movie owns the process-wide interrupt subscription. Native
// handlers only set a lock-free flag; no renderer or callback runs in a handler.
class MovieInterruptScope {
public:
    MovieInterruptScope();
    ~MovieInterruptScope();
    MovieInterruptScope(const MovieInterruptScope &) = delete;
    MovieInterruptScope &operator=(const MovieInterruptScope &) = delete;
    bool StopRequested() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_Impl;
};

} // namespace Environment
