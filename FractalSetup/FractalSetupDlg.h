// FractalSetupDlg.h : header file
//

#if !defined(AFX_FRACTALSETUPDLG_H__5C6EC51E_3255_407A_9CA1_F11038368B76__INCLUDED_)
#define AFX_FRACTALSETUPDLG_H__5C6EC51E_3255_407A_9CA1_F11038368B76__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "..\FractalSetupData.h"

/////////////////////////////////////////////////////////////////////////////
// CFractalSetupDlg dialog

class CFractalSetupDlg : public CDialog
{
// Construction
public:
	CFractalSetupDlg(CWnd* pParent = NULL);	// standard constructor
  FractalSetupData m_Data;

// Dialog Data
	//{{AFX_DATA(CFractalSetupDlg)
	enum { IDD = IDD_FRACTALSETUP_DIALOG };
	int		m_WorkClient;
	CString	m_Location1;
	int		m_L1NumFrames;
	BOOL	m_AZDrawProgress;
	BOOL	m_AZSaveImages;
	CString	m_SaveDir;
	BOOL	m_SSAutoZoom;
	BOOL	m_AZSaveReducedSize;
	CString	m_LocalIP;
	CString	m_ServerIP1;
	CString	m_ServerIP2;
	CString	m_ServerIP3;
	CString	m_ServerIP4;
	int		m_WorkServer1;
	int		m_WorkServer2;
	int		m_WorkServer3;
	int		m_WorkServer4;
	BOOL	m_BeNetworkClient;
	BOOL	m_BeNetworkServer;
	BOOL	m_UseThisServer1;
	BOOL	m_UseThisServer2;
	BOOL	m_UseThisServer3;
	BOOL	m_UseThisServer4;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CFractalSetupDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	//{{AFX_MSG(CFractalSetupDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	virtual void OnOK();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
public:
	int m_AlgHighRes;
	int m_AlgLowRes;
        BOOL m_AltDraw;
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_FRACTALSETUPDLG_H__5C6EC51E_3255_407A_9CA1_F11038368B76__INCLUDED_)
