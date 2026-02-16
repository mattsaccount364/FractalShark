#pragma once

// RAII Class that sets/resets mouse cursor to wait cursor
// Use SetCursor(LoadCursor(nullptr, IDC_WAIT));
class WaitCursor {
public:
    WaitCursor();
    ~WaitCursor();

    WaitCursor &operator=(const WaitCursor &other) {
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
