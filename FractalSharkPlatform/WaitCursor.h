#pragma once

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// RAII Class that sets/resets mouse cursor to wait cursor
// Use SetCursor(LoadCursor(nullptr, IDC_WAIT));
class WaitCursor {
public:
    WaitCursor();
    ~WaitCursor();

    WaitCursor &
    operator=(const WaitCursor &other)
    {
        if (this == &other) {
            return *this;
        }

        m_hCursor = other.m_hCursor;
    }

    WaitCursor &
    operator=(WaitCursor &&other)
    {
        if (this == &other) {
            return *this;
        }

        m_hCursor = other.m_hCursor;
        return *this;
    }

    WaitCursor(const WaitCursor &) = delete;
    WaitCursor(WaitCursor &&) = delete;

    void ResetCursor();

private:
    HCURSOR m_hCursor;
};

#else

// TODO(linux-parity, deferred): Provide Linux GUI busy-cursor feedback for benchmarks.
// This remains an intentional no-op for non-GUI Linux callers until a display-aware hook exists.
class WaitCursor {
public:
    WaitCursor() = default;
    ~WaitCursor() = default;
    void
    ResetCursor()
    {
    }
};

#endif
