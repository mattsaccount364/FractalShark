#include "stdafx.h"
#include "AbortMonitor.h"

#include <Windows.h>

AbortMonitor::AbortMonitor(bool useSensoCursor)
    : m_QuitFlag{ false },
      m_StopCalculating{ false },
      m_UseSensoCursor{ useSensoCursor },
      m_Thread{ &AbortMonitor::Run, this } {
}

AbortMonitor::~AbortMonitor() {
    m_QuitFlag = true;
    if (m_Thread.joinable()) {
        m_Thread.join();
    }
}

bool AbortMonitor::GetStopCalculating() const {
    return m_StopCalculating.load(std::memory_order_relaxed);
}

void AbortMonitor::SetStopCalculating(bool value) {
    m_StopCalculating.store(value, std::memory_order_relaxed);
}

void AbortMonitor::ResetStopCalculating() {
    m_StopCalculating.store(false, std::memory_order_relaxed);
}

bool
AbortMonitor::IsDownControl() {
    return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000);
}

void
AbortMonitor::Run() {
    POINT pt;
    GetCursorPos(&pt);
    int OrgX = pt.x;
    int OrgY = pt.y;

    for (;;) {
        if (m_QuitFlag.load(std::memory_order_relaxed)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        if (IsDownControl()) {
            m_StopCalculating.store(true, std::memory_order_relaxed);
        }

        if (m_UseSensoCursor) {
            GetCursorPos(&pt);

            if (abs(pt.x - OrgX) >= 5 || abs(pt.y - OrgY) >= 5) {
                OrgX = pt.x;
                OrgY = pt.y;
                m_StopCalculating.store(true, std::memory_order_relaxed);
            }
        }
    }
}
