/////////////////////////////////////////////////////////////////
// TrayIcon.cpp : implementation file
//
// This is a conglomeration of ideas from the MSJ "Webster"
// application, sniffing round the online docs, and from other
// implementations such as PJ Naughter's "CTrayIconifyIcon"
// (http://indigo.ie/~pjn / ntray.html) especially the
// "CTrayIcon::OnTrayNotification" member function.
//
// This class is a light wrapper around the windows system
// tray stuff. It adds an icon to the system tray with
// the specified ToolTip text and callback notification value,
// which is sent back to the Parent window.
//
// The tray icon can be instantiated using either the
// constructor or by declaring the object and
// creating (and displaying) it later on in the
// program. eg.
//
// CTrayIcon m_TrayIcon; // Member variable of some class
//
// ...
// // in some member function maybe...
// m_TrayIcon.Create (pParentWnd, WM_MY_NOTIFY,
//�����������������������"Click here", hIcon, nTrayIconID);
//
//��Written by Chris Maunder (Chris.Maunder@cbr.clw.csiro.au)
//
//
/////////////////////////////////////////////////////////////////
// To use:
//  1) Create a new "Resource Symbol" called WM_ICON_NOTIFY.
//     Give it a number of something higher, say 1200.
//  2) Put this command in the message map:
//     ON_MESSAGE (WM_ICON_NOTIFY, OnTrayNotification)
//  3) Define a function called OnTrayNotification that looks like:
//     LRESULT CYourDlg::OnTrayNotification (WPARAM wParam, LPARAM lParam)
//     { // Delegate all the work back to the default implementation in CTrayIcon.
//       return m_TrayIcon.OnTrayNotification (wParam, lParam);
//     }
//     with a prototype that looks like:
//     LRESULT OnTrayNotification (WPARAM, LPARAM);
//  4) Create a new menu resource.  There should be one actual
//     menu in it, probably called something like POPUP.
//     There will be on item "under" popup called Restore.  The
//     Restore ID should be ID_RESTORE.
//  5) Associate the menu restore with the main window of your app.
//  6) Associate the menu item "Restore" with the following function:
//     void CYourDlg::OnRestore ()
//     { ShowWindow (SW_SHOW); }
//  7) Create a member variable of the class called m_TrayIcon.
//  8) In the dialog's InitInstance class, put the following:
//     m_TrayIcon.Create (this, WM_ICON_NOTIFY, "Tooltip text",
//                        GetIcon (FALSE), IDR_MENU_POPUP);
//     m_TrayIcon.ShowIcon ();
//  9) When you want the window to minimize, tell add the command:
//     ShowWindow (SW_HIDE);
// 10) Test it out!
////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include "TrayIcon.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

IMPLEMENT_DYNAMIC (CTrayIcon, CObject)

/////////////////////////////////////////////////////////////////
// CTrayIcon construction / creation / destruction

CTrayIcon::CTrayIcon ()
{
  memset (&m_tnd, 0, sizeof (m_tnd));
  m_bEnabled = FALSE;
  m_bHidden  = FALSE;
}

CTrayIcon::CTrayIcon (CWnd* pWnd, UINT uCallbackMessage, LPCTSTR szToolTip, HICON icon, UINT uID)
{
  Create (pWnd, uCallbackMessage, szToolTip, icon, uID);
  m_bHidden = FALSE;
}

#pragma warning(disable : 4706)
BOOL CTrayIcon::Create (CWnd* pWnd, UINT uCallbackMessage, LPCTSTR szToolTip, HICON icon, UINT uID)
{
  //Make sure Notification window is valid
  VERIFY (m_bEnabled = (pWnd && ::IsWindow (pWnd->GetSafeHwnd ())));
  if (!m_bEnabled) return FALSE;

  //Make sure we avoid conflict with other messages
  ASSERT (uCallbackMessage >= WM_USER);

  //Tray only supports tooltip text up to 64 characters
  ASSERT (_tcslen (szToolTip) <= 64);

  // load up the NOTIFYICONDATA structure
  m_tnd.cbSize = sizeof (NOTIFYICONDATA);
  m_tnd.hWnd = pWnd->GetSafeHwnd ();
  m_tnd.uID = uID;
  m_tnd.hIcon = icon;
  m_tnd.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
  m_tnd.uCallbackMessage = uCallbackMessage;
  wcscpy (m_tnd.szTip, szToolTip);

  // Set the tray icon
  VERIFY (m_bEnabled = Shell_NotifyIcon (NIM_ADD, &m_tnd));
  return m_bEnabled;
}
#pragma warning(default : 4706)

CTrayIcon::~CTrayIcon ()
{ RemoveIcon (); }


/////////////////////////////////////////////////////////////////
// CTrayIcon icon manipulation

void CTrayIcon::MoveToRight ()
{ HideIcon ();
  ShowIcon ();
}

void CTrayIcon::RemoveIcon ()
{ if (!m_bEnabled)
  { return; }

  m_tnd.uFlags = 0;
  Shell_NotifyIcon (NIM_DELETE, &m_tnd);
  m_bEnabled = FALSE;
}

void CTrayIcon::HideIcon ()
{ if (m_bEnabled && !m_bHidden)
  { m_tnd.uFlags = NIF_ICON;
    Shell_NotifyIcon (NIM_DELETE, &m_tnd);
    m_bHidden = TRUE;
  }
}

void CTrayIcon::ShowIcon ()
{ if (m_bEnabled && m_bHidden)
  { m_tnd.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    Shell_NotifyIcon (NIM_ADD, &m_tnd);
    m_bHidden = FALSE;
  }
}

BOOL CTrayIcon::SetIcon (HICON hIcon)
{ if (!m_bEnabled)
  { return FALSE; }

  m_tnd.uFlags = NIF_ICON;
  m_tnd.hIcon = hIcon;

  return Shell_NotifyIcon (NIM_MODIFY, &m_tnd);
}

BOOL CTrayIcon::SetIcon (LPCTSTR lpszIconName)
{ HICON hIcon = AfxGetApp ()->LoadIcon (lpszIconName);
  return SetIcon (hIcon);
}

BOOL CTrayIcon::SetIcon (UINT nIDResource)
{ HICON hIcon = AfxGetApp ()->LoadIcon (nIDResource);
  return SetIcon (hIcon);
}

BOOL CTrayIcon::SetStandardIcon (LPCTSTR lpIconName)
{ HICON hIcon = LoadIcon (NULL, lpIconName);
  return SetIcon (hIcon);
}

BOOL CTrayIcon::SetStandardIcon (UINT nIDResource)
{ HICON hIcon = LoadIcon (NULL, MAKEINTRESOURCE (nIDResource));
  return SetIcon (hIcon);
}

HICON CTrayIcon::GetIcon () const
{ HICON hIcon = NULL;
  if (m_bEnabled)
  { hIcon = m_tnd.hIcon; }

  return hIcon;
}

/////////////////////////////////////////////////////////////////
// CTrayIcon tooltip text manipulation

BOOL CTrayIcon::SetTooltipText (LPCTSTR pszTip)
{ if (!m_bEnabled)
  { return FALSE; }

  m_tnd.uFlags = NIF_TIP;
  _tcscpy (m_tnd.szTip, pszTip);

  return Shell_NotifyIcon (NIM_MODIFY, &m_tnd);
}

BOOL CTrayIcon::SetTooltipText (UINT nID)
{ CString strText;
  VERIFY (strText.LoadString (nID));

  return SetTooltipText (strText);
}

CString CTrayIcon::GetTooltipText () const
{ CString strText;
  if (m_bEnabled)
  { strText = m_tnd.szTip; }

  return strText;
}

/////////////////////////////////////////////////////////////////
// CTrayIcon notification window stuff

BOOL CTrayIcon::SetNotificationWnd (CWnd* pWnd)
{ if (!m_bEnabled)
  { return FALSE; }

  //Make sure Notification window is valid
  ASSERT (pWnd && ::IsWindow (pWnd->GetSafeHwnd ()));

  m_tnd.hWnd = pWnd->GetSafeHwnd ();
  m_tnd.uFlags = 0;

  return Shell_NotifyIcon (NIM_MODIFY, &m_tnd);
}

CWnd* CTrayIcon::GetNotificationWnd () const
{ return CWnd::FromHandle (m_tnd.hWnd); }

/////////////////////////////////////////////////////////////////
// CTrayIcon implentation of OnTrayNotification

#pragma warning(disable : 4706)
LRESULT CTrayIcon::OnTrayNotification (WPARAM wParam, LPARAM lParam)
{ //Return quickly if its not for this tray icon
  if (wParam != m_tnd.uID)
  { return 0L; }

  CMenu menu, *pSubMenu;

  // Clicking with right button brings up a context menu
  if (LOWORD (lParam) == WM_RBUTTONUP)
  { if (!menu.LoadMenu (m_tnd.uID))
    { return 0; }
    if (!(pSubMenu = menu.GetSubMenu (0)))
    { return 0; }
    // Make first menu item the default (bold font)
    ::SetMenuDefaultItem (pSubMenu->m_hMenu, 0, TRUE);

    //Display and track the popup menu
    CPoint pos;
    GetCursorPos (&pos);

    ::SetForegroundWindow (m_tnd.hWnd);
    ::TrackPopupMenu (pSubMenu->m_hMenu, 0, pos.x, pos.y, 0, m_tnd.hWnd, NULL);

    // BUGFIX: See "PRB: Menus for Notification Icons Don't Work Correctly"
    ::PostMessage (m_tnd.hWnd, WM_USER, 0, 0);

    menu.DestroyMenu ();
  }
  /*else if (LOWORD (lParam) == WM_LBUTTONDBLCLK)
  { if (!menu.LoadMenu (m_tnd.uID))
    { return 0; }
    if (!(pSubMenu = menu.GetSubMenu (0)))
    { return 0; }

    // double click received, the default action is to execute first menu item
    ::SetForegroundWindow (m_tnd.hWnd);
    ::SendMessage (m_tnd.hWnd, WM_COMMAND, pSubMenu->GetMenuItemID (0), 0);

    menu.DestroyMenu ();
  }*/

  return 1;
}
#pragma warning(default : 4706)