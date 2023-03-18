// FractalTrayDlg.h : header file
//

#pragma once

#define WM_ICON_NOTIFY 1200
#define WM_FINISHED_CALCULATING 1201

#include "TrayIcon.h"
#include <string>

struct CThreadParam
{ volatile bool stop;
  const CString *LocationFilename;
  int Algorithm;
  HWND hWnd;
};

DWORD WINAPI CalcProc (LPVOID lpParameter);
bool FileExists (const std::wstring &filename);

// CFractalTrayDlg dialog
class CFractalTrayDlg : public CDialog
{
// Construction
public:
  CFractalTrayDlg(CWnd* pParent = NULL);  // standard constructor

// Dialog Data
  enum { IDD = IDD_FRACTALTRAY_DIALOG };

  protected:
  virtual void DoDataExchange(CDataExchange* pDX);  // DDX/DDV support


// Implementation
protected:
  HICON m_hIcon1;
  HICON m_hIcon2;

  // Generated message map functions
  virtual BOOL OnInitDialog();
  afx_msg void OnPaint();
  afx_msg HCURSOR OnQueryDragIcon();
  afx_msg LRESULT OnSysCommand(WPARAM nID, LPARAM lParam);
  afx_msg LRESULT OnDestroy(WPARAM nID, LPARAM lParam);
  DECLARE_MESSAGE_MAP()
public:
  afx_msg void OnBnClickedButtonGenerate();
  int HowManyFrames (void);
  LRESULT OnTrayNotification (WPARAM, LPARAM);
  LRESULT OnFinishedCalculating (WPARAM, LPARAM);
  void OnRestore ();
  void OnExit ();

  CString m_SourceCoords;
  CString m_DestCoords;
  double m_ScaleFactor;
  int m_ResX;
  int m_ResY;
  int m_GpuAntialiasing;

  CString m_Messages;
  CString m_LocationFilename;

  HANDLE m_Thread;
  CThreadParam m_ThreadParam;
  CTrayIcon m_TrayIcon;
  int m_Algorithm;
};
