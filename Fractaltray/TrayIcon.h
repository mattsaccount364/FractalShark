#pragma once

class TrayIcon {
public:
    TrayIcon() = default;
    ~TrayIcon();

    TrayIcon(const TrayIcon &) = delete;
    TrayIcon &operator=(const TrayIcon &) = delete;

    bool Create(HWND parentWnd, UINT callbackMessage,
                const wchar_t *tooltip, HICON icon, UINT menuId);
    void Remove();

    bool SetIcon(HICON icon);
    bool SetTooltipText(const wchar_t *tip);
    void Show();
    void Hide();

    LRESULT OnTrayNotification(WPARAM wParam, LPARAM lParam);

    bool IsEnabled() const { return m_Enabled; }
    bool IsVisible() const { return m_Enabled && !m_Hidden; }

private:
    NOTIFYICONDATA m_Nid{};
    UINT m_MenuId = 0;
    bool m_Enabled = false;
    bool m_Hidden = false;
};
