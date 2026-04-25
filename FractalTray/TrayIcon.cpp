#include "stdafx.h"
#include "TrayIcon.h"

TrayIcon::~TrayIcon() {
    Remove();
}

bool TrayIcon::Create(HWND parentWnd, UINT callbackMessage,
                      const wchar_t *tooltip, HICON icon, UINT menuId) {
    if (!parentWnd || !::IsWindow(parentWnd)) {
        m_Enabled = false;
        return false;
    }

    m_Nid.cbSize = sizeof(NOTIFYICONDATA);
    m_Nid.hWnd = parentWnd;
    m_Nid.uID = menuId;
    m_Nid.hIcon = icon;
    m_Nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    m_Nid.uCallbackMessage = callbackMessage;
    wcsncpy_s(m_Nid.szTip, tooltip, _TRUNCATE);

    m_MenuId = menuId;
    m_Enabled = Shell_NotifyIcon(NIM_ADD, &m_Nid) != FALSE;
    m_Hidden = false;
    return m_Enabled;
}

void TrayIcon::Remove() {
    if (!m_Enabled) return;
    m_Nid.uFlags = 0;
    Shell_NotifyIcon(NIM_DELETE, &m_Nid);
    m_Enabled = false;
}

bool TrayIcon::SetIcon(HICON icon) {
    if (!m_Enabled) return false;
    m_Nid.uFlags = NIF_ICON;
    m_Nid.hIcon = icon;
    return Shell_NotifyIcon(NIM_MODIFY, &m_Nid) != FALSE;
}

bool TrayIcon::SetTooltipText(const wchar_t *tip) {
    if (!m_Enabled) return false;
    m_Nid.uFlags = NIF_TIP;
    wcsncpy_s(m_Nid.szTip, tip, _TRUNCATE);
    return Shell_NotifyIcon(NIM_MODIFY, &m_Nid) != FALSE;
}

void TrayIcon::Show() {
    if (m_Enabled && m_Hidden) {
        m_Nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
        Shell_NotifyIcon(NIM_ADD, &m_Nid);
        m_Hidden = false;
    }
}

void TrayIcon::Hide() {
    if (m_Enabled && !m_Hidden) {
        m_Nid.uFlags = NIF_ICON;
        Shell_NotifyIcon(NIM_DELETE, &m_Nid);
        m_Hidden = true;
    }
}

LRESULT TrayIcon::OnTrayNotification(WPARAM wParam, LPARAM lParam) {
    if (wParam != m_Nid.uID) return 0;

    if (LOWORD(lParam) == WM_RBUTTONUP) {
        HMENU hMenu = LoadMenu(
            GetModuleHandle(nullptr), MAKEINTRESOURCE(m_MenuId));
        if (!hMenu) return 0;

        HMENU hSubMenu = GetSubMenu(hMenu, 0);
        if (!hSubMenu) {
            DestroyMenu(hMenu);
            return 0;
        }

        SetMenuDefaultItem(hSubMenu, 0, TRUE);

        POINT pos;
        GetCursorPos(&pos);
        SetForegroundWindow(m_Nid.hWnd);
        TrackPopupMenu(hSubMenu, 0, pos.x, pos.y, 0, m_Nid.hWnd, nullptr);

        // Required workaround for tray context menus.
        // See "PRB: Menus for Notification Icons Don't Work Correctly"
        PostMessage(m_Nid.hWnd, WM_NULL, 0, 0);

        DestroyMenu(hMenu);
    }

    return 1;
}
