#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
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

    void Start();
    void Stop();

    [[nodiscard]] bool
    IsRunning() const noexcept
    {
        return m_Running.load(std::memory_order_acquire);
    }

private:
    void ThreadMain(std::stop_token stopToken);
    void SignalStarted(std::exception_ptr exception) noexcept;

    std::jthread m_Worker;
    std::atomic_bool m_Running{false};

    std::mutex m_StartMutex;
    std::condition_variable m_StartCondition;
    bool m_StartCompleted = false;
    std::exception_ptr m_StartException;
    std::exception_ptr m_WorkerException;
};

} // namespace FractalShark::Linux
