#pragma once

#include <thread>
#include <atomic>
#include <chrono>

class AbortMonitor {
public:
    AbortMonitor(bool useSensoCursor);
    ~AbortMonitor();

    AbortMonitor(const AbortMonitor &) = delete;
    AbortMonitor &operator=(const AbortMonitor &) = delete;

    bool GetStopCalculating() const;
    void SetStopCalculating(bool value);
    void ResetStopCalculating();

    static bool GetStopCalculatingGlobal();
    static void ResetStopCalculatingGlobal();

    static constexpr uint64_t AbortCheckInterval = 16384;

private:
    void Run();
    static bool IsDownControl();

    static AbortMonitor *s_Instance;

    std::thread m_Thread;
    std::atomic<bool> m_QuitFlag;
    std::atomic<bool> m_StopCalculating;
    bool m_UseSensoCursor;
};
