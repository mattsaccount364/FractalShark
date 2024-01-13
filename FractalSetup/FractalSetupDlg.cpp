// FractalSetupDlg.cpp : implementation file
//

#include "stdafx.h"
#include "FractalSetup.h"
#include "FractalSetupDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CFractalSetupDlg dialog

CFractalSetupDlg::CFractalSetupDlg(CWnd* pParent /*=nullptr*/)
  : CDialog(CFractalSetupDlg::IDD, pParent)
{
  //{{AFX_DATA_INIT(CFractalSetupDlg)
	m_WorkClient = 20;
	m_Location1 = _T("");
	m_L1NumFrames = 1500;
	m_AZDrawProgress = FALSE;
	m_AZSaveImages = FALSE;
	m_SaveDir = _T("");
	m_SSAutoZoom = FALSE;
	m_AZSaveReducedSize = FALSE;
	m_LocalIP = _T("");
	m_ServerIP1 = _T("");
	m_ServerIP2 = _T("");
	m_ServerIP3 = _T("");
	m_ServerIP4 = _T("");
	m_WorkServer1 = 20;
	m_WorkServer2 = 20;
	m_WorkServer3 = 20;
	m_WorkServer4 = 20;
	m_BeNetworkClient = FALSE;
	m_BeNetworkServer = FALSE;
	m_UseThisServer1 = FALSE;
	m_UseThisServer2 = FALSE;
	m_UseThisServer3 = FALSE;
	m_UseThisServer4 = FALSE;
	//}}AFX_DATA_INIT
  // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
  m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CFractalSetupDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CFractalSetupDlg)
	DDX_Text(pDX, IDC_EDIT_WORKCLIENT, m_WorkClient);
	DDV_MinMaxInt(pDX, m_WorkClient, 1, 100);
	DDX_Text(pDX, IDC_EDIT_LOCATION1, m_Location1);
	DDX_Text(pDX, IDC_EDIT_NUMFRAMES, m_L1NumFrames);
	DDV_MinMaxInt(pDX, m_L1NumFrames, 1, 9999);
	DDX_Check(pDX, IDC_CHECK_SHOWPROGRESS, m_AZDrawProgress);
	DDX_Check(pDX, IDC_CHECK_SAVEIMAGES, m_AZSaveImages);
	DDX_Text(pDX, IDC_EDIT_SAVEDIR, m_SaveDir);
	DDX_Check(pDX, IDC_CHECK_SS_AUTOZOOM, m_SSAutoZoom);
	DDX_Check(pDX, IDC_CHECK_SAVEREDUCEDSIZE, m_AZSaveReducedSize);
	DDX_Text(pDX, IDC_EDIT_LOCALIP, m_LocalIP);
	DDX_Text(pDX, IDC_EDIT_SERVERIP1, m_ServerIP1);
	DDX_Text(pDX, IDC_EDIT_SERVERIP2, m_ServerIP2);
	DDX_Text(pDX, IDC_EDIT_SERVERIP3, m_ServerIP3);
	DDX_Text(pDX, IDC_EDIT_SERVERIP4, m_ServerIP4);
	DDX_Text(pDX, IDC_EDIT_WORKSERVER1, m_WorkServer1);
	DDV_MinMaxInt(pDX, m_WorkServer1, 1, 100);
	DDX_Text(pDX, IDC_EDIT_WORKSERVER2, m_WorkServer2);
	DDV_MinMaxInt(pDX, m_WorkServer2, 1, 100);
	DDX_Text(pDX, IDC_EDIT_WORKSERVER3, m_WorkServer3);
	DDV_MinMaxInt(pDX, m_WorkServer3, 1, 100);
	DDX_Text(pDX, IDC_EDIT_WORKSERVER4, m_WorkServer4);
	DDV_MinMaxInt(pDX, m_WorkServer4, 1, 100);
	DDX_Check(pDX, IDC_CHECK_BENETWORKCLIENT, m_BeNetworkClient);
	DDX_Check(pDX, IDC_CHECK_BENETWORKSERVER, m_BeNetworkServer);
	DDX_Check(pDX, IDC_CHECK_USE1, m_UseThisServer1);
	DDX_Check(pDX, IDC_CHECK_USE2, m_UseThisServer2);
	DDX_Check(pDX, IDC_CHECK_USE3, m_UseThisServer3);
	DDX_Check(pDX, IDC_CHECK_USE4, m_UseThisServer4);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CFractalSetupDlg, CDialog)
  //{{AFX_MSG_MAP(CFractalSetupDlg)
  ON_WM_PAINT()
  ON_WM_QUERYDRAGICON()
  //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CFractalSetupDlg message handlers

BOOL CFractalSetupDlg::OnInitDialog()
{
  CDialog::OnInitDialog();

  // Set the icon for this dialog.  The framework does this automatically
  //  when the application's main window is not a dialog
  SetIcon(m_hIcon, TRUE);      // Set big icon
  SetIcon(m_hIcon, FALSE);    // Set small icon

  // TODO: Add extra initialization here
  m_Data.Load ();

  ASSERT (MAXSERVERS == 4);
  m_LocalIP = m_Data.m_LocalIP;
  m_ServerIP1 = m_Data.m_ServerIPs[0];
  m_ServerIP2 = m_Data.m_ServerIPs[1];
  m_ServerIP3 = m_Data.m_ServerIPs[2];
  m_ServerIP4 = m_Data.m_ServerIPs[3];
  m_UseThisServer1 = (m_Data.m_UseThisServer[0] == 'y') ? TRUE : FALSE;
  m_UseThisServer2 = (m_Data.m_UseThisServer[1] == 'y') ? TRUE : FALSE;
  m_UseThisServer3 = (m_Data.m_UseThisServer[2] == 'y') ? TRUE : FALSE;
  m_UseThisServer4 = (m_Data.m_UseThisServer[3] == 'y') ? TRUE : FALSE;

  m_BeNetworkClient = (m_Data.m_BeNetworkClient == 'y') ? TRUE : FALSE;
  m_BeNetworkServer = (m_Data.m_BeNetworkServer == 'y') ? TRUE : FALSE;

  m_WorkClient = m_Data.m_WorkClient;
  m_WorkServer1 = m_Data.m_WorkServers[0];
  m_WorkServer2 = m_Data.m_WorkServers[1];
  m_WorkServer3 = m_Data.m_WorkServers[2];
  m_WorkServer4 = m_Data.m_WorkServers[3];

  m_Location1.Format (L"%.15f %.15f %.15f %.15f %d",
                      m_Data.m_L1MinX, m_Data.m_L1MinY,
                      m_Data.m_L1MaxX, m_Data.m_L1MaxY,
                      m_Data.m_L1Iterations);
  m_L1NumFrames = m_Data.m_L1NumFrames;

  m_AZSaveImages = (m_Data.m_AZSaveImages == 'y') ? TRUE : FALSE;
  m_AZDrawProgress = (m_Data.m_AZDrawProgress == 'y') ? TRUE : FALSE;
  m_SaveDir = m_Data.m_SaveDir;
  m_SSAutoZoom = (m_Data.m_SSAutoZoom == 'y') ? TRUE : FALSE;
  m_AZSaveReducedSize = (m_Data.m_AZSaveReducedSize == 'y') ? TRUE : FALSE;

  UpdateData (FALSE);
  return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CFractalSetupDlg::OnPaint()
{
  if (IsIconic())
  {
    CPaintDC dc(this); // device context for painting

    SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

    // Center icon in client rectangle
    int cxIcon = GetSystemMetrics(SM_CXICON);
    int cyIcon = GetSystemMetrics(SM_CYICON);
    CRect rect;
    GetClientRect(&rect);
    int x = (rect.Width() - cxIcon + 1) / 2;
    int y = (rect.Height() - cyIcon + 1) / 2;

    // Draw the icon
    dc.DrawIcon(x, y, m_hIcon);
  }
  else
  {
    CDialog::OnPaint();
  }
}

// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFractalSetupDlg::OnQueryDragIcon()
{
  return (HCURSOR) m_hIcon;
}

void CFractalSetupDlg::OnOK ()
{ UpdateData (TRUE);

  strcpy (m_Data.m_LocalIP, CStringA(m_LocalIP).operator LPCSTR ());
  strcpy(m_Data.m_ServerIPs[0], CStringA(m_ServerIP1).operator LPCSTR ());
  strcpy(m_Data.m_ServerIPs[1], CStringA(m_ServerIP2).operator LPCSTR ());
  strcpy(m_Data.m_ServerIPs[2], CStringA(m_ServerIP3).operator LPCSTR ());
  strcpy(m_Data.m_ServerIPs[3], CStringA(m_ServerIP4).operator LPCSTR ());
  m_Data.m_UseThisServer[0] = (m_UseThisServer1 == TRUE) ? 'y': 'n';
  m_Data.m_UseThisServer[1] = (m_UseThisServer2 == TRUE) ? 'y': 'n';
  m_Data.m_UseThisServer[2] = (m_UseThisServer3 == TRUE) ? 'y': 'n';
  m_Data.m_UseThisServer[3] = (m_UseThisServer4 == TRUE) ? 'y': 'n';

  m_Data.m_BeNetworkClient = (m_BeNetworkClient == TRUE) ? 'y': 'n';
  m_Data.m_BeNetworkServer = (m_BeNetworkServer == TRUE) ? 'y': 'n';

  m_Data.m_WorkClient = m_WorkClient;
  m_Data.m_WorkServers[0] = m_WorkServer1;
  m_Data.m_WorkServers[1] = m_WorkServer2;
  m_Data.m_WorkServers[2] = m_WorkServer3;
  m_Data.m_WorkServers[3] = m_WorkServer4;

  sscanf ((const char *) m_Location1.operator LPCTSTR (),
          "%lf %lf %lf %lf %d",
          &m_Data.m_L1MinX, &m_Data.m_L1MinY,
          &m_Data.m_L1MaxX, &m_Data.m_L1MaxY,
          &m_Data.m_L1Iterations);
  m_Data.m_L1NumFrames = m_L1NumFrames;

  m_Data.m_AZSaveImages = (m_AZSaveImages == TRUE) ? 'y' : 'n';
  m_Data.m_AZDrawProgress = (m_AZDrawProgress == TRUE) ? 'y' : 'n';
  wcscpy (m_Data.m_SaveDir, m_SaveDir);

  m_Data.m_SSAutoZoom = (m_SSAutoZoom == TRUE) ? 'y' : 'n';
  m_Data.m_AZSaveReducedSize = (m_AZSaveReducedSize == TRUE) ? 'y' : 'n';
  
  m_Data.Save ();

  CDialog::OnOK();
}