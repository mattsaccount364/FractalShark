#pragma once

#include <cstdint>

namespace Environment {

// RAII Class that sets/resets mouse cursor to wait cursor
class WaitCursor {
public:
    WaitCursor();
    ~WaitCursor();

    WaitCursor &operator=(const WaitCursor &) = delete;
    WaitCursor &operator=(WaitCursor &&) = delete;
    WaitCursor(const WaitCursor &) = delete;
    WaitCursor(WaitCursor &&) = delete;

    void ResetCursor();

    static void RegisterLinuxCursorTarget(void *display,
                                          std::uintptr_t window,
                                          std::uintptr_t normalCursor);
    static void UnregisterLinuxCursorTarget();

private:
    void *m_PreviousCursor = nullptr;
    bool m_CursorSet = false;
};

} // namespace Environment
