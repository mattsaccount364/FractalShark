// FractalSetup.h : main header file for the FRACTALSETUP application
//

#if !defined(AFX_FRACTALSETUP_H__6E6DA5E8_A10B_49B8_82FF_940A451AE88B__INCLUDED_)
#define AFX_FRACTALSETUP_H__6E6DA5E8_A10B_49B8_82FF_940A451AE88B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef __AFXWIN_H__
	#error include 'stdafx.h' before including this file for PCH
#endif

#include "resource.h"		// main symbols

/////////////////////////////////////////////////////////////////////////////
// CFractalSetupApp:
// See FractalSetup.cpp for the implementation of this class
//

class CFractalSetupApp : public CWinApp
{
public:
	CFractalSetupApp();

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CFractalSetupApp)
	public:
	virtual BOOL InitInstance();
	//}}AFX_VIRTUAL

// Implementation

	//{{AFX_MSG(CFractalSetupApp)
		// NOTE - the ClassWizard will add and remove member functions here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};


/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_FRACTALSETUP_H__6E6DA5E8_A10B_49B8_82FF_940A451AE88B__INCLUDED_)
