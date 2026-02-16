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

private:
    void Run();
    static bool IsDownControl();

    std::thread m_Thread;
    std::atomic<bool> m_QuitFlag;
    std::atomic<bool> m_StopCalculating;
    bool m_UseSensoCursor;
};
