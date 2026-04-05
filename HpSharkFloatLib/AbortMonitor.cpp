#include "AbortMonitor.h"
#include "Environment.h"

#include <iostream>

AbortMonitor *AbortMonitor::s_Instance = nullptr;

AbortMonitor::AbortMonitor(bool useSensoCursor)
    : m_QuitFlag{false}, m_StopCalculating{false}, m_UseSensoCursor{useSensoCursor},
      m_Thread{&AbortMonitor::Run, this}
{
    s_Instance = this;
}

AbortMonitor::~AbortMonitor()
{
    m_QuitFlag = true;
    if (m_Thread.joinable()) {
        m_Thread.join();
    }
    if (s_Instance == this) {
        s_Instance = nullptr;
    }
}

bool
AbortMonitor::GetStopCalculating() const
{
    return m_StopCalculating.load(std::memory_order_relaxed);
}

void
AbortMonitor::SetStopCalculating(bool value)
{
    m_StopCalculating.store(value, std::memory_order_relaxed);
}

void
AbortMonitor::ResetStopCalculating()
{
    m_StopCalculating.store(false, std::memory_order_relaxed);
}

bool
AbortMonitor::GetStopCalculatingGlobal()
{
    if (s_Instance) {
        return s_Instance->GetStopCalculating();
    }
    return false;
}

void
AbortMonitor::ResetStopCalculatingGlobal()
{
    if (s_Instance) {
        s_Instance->ResetStopCalculating();
    }
}

bool
AbortMonitor::IsDownControlAlt()
{
    return Environment::IsKeyDown(Environment::Key::Control) &&
           Environment::IsKeyDown(Environment::Key::Alt);
}

void
AbortMonitor::Run()
{
    auto [OrgX, OrgY] = Environment::GetCursorPosition();

    static constexpr int CtrlHoldThreshold = 12; // 12 × 250ms = 3 seconds

    int ctrlHeldCount = 0;

    for (;;) {
        if (m_QuitFlag.load(std::memory_order_relaxed)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        // Escape: cancels/resets the abort flag
        if (Environment::IsKeyDown(Environment::Key::Escape)) {
            if (m_StopCalculating.load(std::memory_order_relaxed)) {
                std::wcerr << L"AbortMonitor: abort cancelled (Escape pressed)" << std::endl;
                m_StopCalculating.store(false, std::memory_order_relaxed);
            }
        }

        // Ctrl+Alt: debounced — must hold for ~3 seconds
        if (IsDownControlAlt()) {
            ctrlHeldCount++;
            if (ctrlHeldCount >= CtrlHoldThreshold) {
                std::wcerr << L"AbortMonitor: stop signal set (Ctrl+Alt held 3s)" << std::endl;
                m_StopCalculating.store(true, std::memory_order_relaxed);
            }
        } else {
            ctrlHeldCount = 0;
        }

        if (m_UseSensoCursor) {
            auto [curX, curY] = Environment::GetCursorPosition();

            if (abs(curX - OrgX) >= 5 || abs(curY - OrgY) >= 5) {
                OrgX = curX;
                OrgY = curY;
                m_StopCalculating.store(true, std::memory_order_relaxed);
            }
        }
    }
}
