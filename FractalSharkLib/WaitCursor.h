#pragma once

// RAII Class that sets/resets mouse cursor to wait cursor
// Use SetCursor(LoadCursor(nullptr, IDC_WAIT));
class WaitCursor {
public:
    WaitCursor();
    ~WaitCursor();

    WaitCursor &operator=(const WaitCursor &) = delete;
    WaitCursor &operator=(WaitCursor &&) = delete;
    WaitCursor(const WaitCursor &) = delete;
    WaitCursor(WaitCursor &&) = delete;

    void ResetCursor();

private:
    HCURSOR m_hCursor;
};
