#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>

namespace FractalShark::Linux {

class SplashWindow {
public:
    SplashWindow() noexcept = default;
    ~SplashWindow() noexcept;

    SplashWindow(const SplashWindow &) = delete;
    SplashWindow &operator=(const SplashWindow &) = delete;

    bool Start();
    void Stop() noexcept;

    [[nodiscard]] bool
    IsRunning() const noexcept
    {
        return m_Running.load(std::memory_order_acquire);
    }

private:
    void ThreadMain(std::stop_token stopToken);
    void SignalStarted(bool succeeded) noexcept;

    std::jthread m_Worker;
    std::atomic_bool m_Running{false};

    std::mutex m_StartMutex;
    std::condition_variable m_StartCondition;
    bool m_StartCompleted = false;
    bool m_StartSucceeded = false;
};

} // namespace FractalShark::Linux
