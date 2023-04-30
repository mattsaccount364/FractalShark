// FractalTrayDlg.cpp : implementation file
//

#include "stdafx.h"
#include "FractalTray.h"
#include "FractalTrayDlg.h"
#include ".\fractaltraydlg.h"

#include <math.h>
#include <io.h>
#include "..\Fractal.h"
//#include "..\cximage599a_full\CxImage\xImage.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

void ConvertCStringToDest(const CString& str,
    uint32_t& width,
    uint32_t& height,
    HighPrecision& minX,
    HighPrecision& minY,
    HighPrecision& maxX,
    HighPrecision& maxY,
    uint32_t *iters = nullptr,
    uint32_t *iterationAntialiasing = nullptr,
    uint32_t *gpuAntialiasing = nullptr,
    uint32_t *iterationPrecision = nullptr,
    std::wstring* filename = nullptr);

void OutputMessage (const wchar_t *szFormat, ...);
CFractalTrayDlg *theActiveOne;

CFractalTrayDlg::CFractalTrayDlg (CWnd* pParent /*=NULL*/)
  : CDialog (CFractalTrayDlg::IDD, pParent)
  , m_DestCoords (_T ("16384 6826 -0.91160860655737704918032786885245901639344262295081967213114754098360655737704918032786885245901639344262295081962950819609401041 0.234257812499999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999953905244375 -0.90897008984867591424968474148802017654476670870113493064312736443883984867591424968474148802017654476670870113498701134965572916 0.235359374999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999953905244375 8192 1 2 1"))
  , m_ScaleFactor (75)
  , m_ResX (3840)
  , m_ResY (1600)
  , m_SourceCoords (_T ("16384 6826 -4.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 -1.5 3.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 1.5 8192 1 2 1"))
  , m_LocationFilename ("locations.txt")
  , m_Messages (_T (""))
  , m_Algorithm (2)
{
  m_hIcon1 = AfxGetApp ()->LoadIcon (IDR_FRACTALTRAY1);
  m_hIcon2 = AfxGetApp ()->LoadIcon (IDR_FRACTALTRAY2);
}

void CFractalTrayDlg::DoDataExchange (CDataExchange* pDX)
{ DDX_Text (pDX, IDC_EDIT_SOURCECOORDS, m_SourceCoords);
  DDX_Text (pDX, IDC_EDIT_DESTCOORDS, m_DestCoords);
  DDX_Text (pDX, IDC_EDIT_SCALEFACTOR, m_ScaleFactor);
  DDX_Text (pDX, IDC_EDIT_RESX, m_ResX);
  DDX_Text (pDX, IDC_EDIT_RESY, m_ResY);
  DDX_Text (pDX, IDC_EDIT_MESSAGES, m_Messages);
  DDX_Radio (pDX, IDC_RADIO_ALG_C, m_Algorithm);
  CDialog::DoDataExchange (pDX);
}

BEGIN_MESSAGE_MAP (CFractalTrayDlg, CDialog)
  ON_WM_PAINT ()
  ON_WM_QUERYDRAGICON ()
  //}}AFX_MSG_MAP
  ON_MESSAGE (WM_SYSCOMMAND, OnSysCommand)
  ON_BN_CLICKED (IDC_BUTTON_GENERATE, OnBnClickedButtonGenerate)
  ON_MESSAGE (WM_ICON_NOTIFY, OnTrayNotification)
  ON_MESSAGE (WM_DESTROY, OnDestroy)
  ON_MESSAGE (WM_FINISHED_CALCULATING, OnFinishedCalculating)
  ON_COMMAND (ID_POPUP_RESTORE, OnRestore)
  ON_COMMAND (ID_POPUP_EXIT, OnExit)
END_MESSAGE_MAP ()


// CFractalTrayDlg message handlers

BOOL CFractalTrayDlg::OnInitDialog ()
{ CDialog::OnInitDialog ();

  // TODO: Add extra initialization here
  theActiveOne = this;

  m_TrayIcon.Create (this, WM_ICON_NOTIFY, L"FractalTray",
                     GetIcon (FALSE), IDR_MENU_TRAY);

  UpdateData (FALSE);
  if (FileExists (m_LocationFilename.operator LPCWSTR()) == true)
  { m_ThreadParam.LocationFilename = &m_LocationFilename;
    m_ThreadParam.stop = false;
    m_ThreadParam.Algorithm = m_Algorithm + 1;
    m_ThreadParam.hWnd = m_hWnd;

    m_Thread = CreateThread (NULL, 0, CalcProc, &m_ThreadParam, NULL, NULL);
    PostMessage (WM_SYSCOMMAND, SC_MINIMIZE, 0);

    SetIcon (m_hIcon1, TRUE);      // Set big icon
    SetIcon (m_hIcon1, FALSE);    // Set small icon
    m_TrayIcon.SetIcon (m_hIcon1);
  }
  else
  { m_Thread = NULL;
    SetIcon (m_hIcon2, TRUE);      // Set big icon
    SetIcon (m_hIcon2, FALSE);    // Set small icon
    m_TrayIcon.SetIcon (m_hIcon2);
  }

  m_TrayIcon.ShowIcon ();

  //SetWindowPos (NULL, 0, 0, 0, 0, SWP_HIDEWINDOW | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER);

  return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document \ view model,
//  this is automatically done for you by the framework.

LRESULT CFractalTrayDlg::OnSysCommand (WPARAM nID, LPARAM lParam)
{ if (nID == SC_MINIMIZE)
  { ShowWindow (SW_HIDE);
    return 0;
  }
  else
  { CDialog::OnSysCommand ((UINT) nID, lParam);
    return 0;
  }
}

LRESULT CFractalTrayDlg::OnDestroy (WPARAM, LPARAM)
{ if (m_Thread != NULL)
  { m_ThreadParam.stop = true;
    WaitForSingleObject (m_Thread, INFINITE);
  }

  //MessageBox ("!");

  CDialog::OnDestroy ();
  return 0;
}

void CFractalTrayDlg::OnPaint ()
{
  if (IsIconic ())
  {
    CPaintDC dc (this); // device context for painting

    SendMessage (WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc ()), 0);

    // Center icon in client rectangle
    int cxIcon = GetSystemMetrics (SM_CXICON);
    int cyIcon = GetSystemMetrics (SM_CYICON);
    CRect rect;
    GetClientRect (&rect);
    int x = (rect.Width () - cxIcon + 1) / 2;
    int y = (rect.Height () - cyIcon + 1) / 2;

    // Draw the icon
    dc.DrawIcon (x, y, m_hIcon1);
  }
  else
  {
    CDialog::OnPaint ();
  }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFractalTrayDlg::OnQueryDragIcon ()
{
  return static_cast<HCURSOR>(m_hIcon2);
}

void GetTokensFromString(const CString& str,
                         std::vector<std::wstring> &tokens,
                         std::vector<const wchar_t *> &tokens_wstr,
                         std::vector<std::string> &tokens_str) {
    std::wstring dest_coords = str.operator LPCTSTR ();

    std::wstringstream check1(dest_coords);
    std::wstring intermediate;

    // Tokenizing w.r.t. space ' '
    size_t i = 0;
    while (std::getline(check1, intermediate, L' '))
    {
        char temp[1024];
        tokens.push_back(intermediate);
        tokens_wstr.push_back(tokens[i].c_str());

        sprintf(temp, "%ws", tokens[i].c_str());
        tokens_str.push_back(temp);
        i++;
    }
}

void ConvertCStringToDest(const CString& str,
    uint32_t& width,
    uint32_t& height,
    HighPrecision& minX,
    HighPrecision& minY,
    HighPrecision& maxX,
    HighPrecision& maxY,
    uint32_t *iters,
    uint32_t *iterationAntialiasing,
    uint32_t *gpuAntialiasing,
    uint32_t* iterationPrecision,
    std::wstring *filename) {

    std::vector<std::wstring> tokens;
    std::vector<const wchar_t*> tokens_wstr;
    std::vector<std::string> tokens_str;
    GetTokensFromString(str, tokens, tokens_wstr, tokens_str);

    width = atoi(tokens_str[0].c_str());
    height = atoi(tokens_str[1].c_str());

    minX = HighPrecision(tokens_str[2].c_str());
    minY = HighPrecision(tokens_str[3].c_str());
    maxX = HighPrecision(tokens_str[4].c_str());
    maxY = HighPrecision(tokens_str[5].c_str());

    if (iters != nullptr) {
        *iters = atoi(tokens_str[6].c_str());
    }

    if (iterationAntialiasing != nullptr) {
        *iterationAntialiasing = atoi(tokens_str[7].c_str());
    }

    if (gpuAntialiasing != nullptr) {
        *gpuAntialiasing = atoi(tokens_str[8].c_str());
    }

    if (iterationPrecision != nullptr) {
        *iterationPrecision = atoi(tokens_str[9].c_str());
    }

    if (filename != nullptr) {
        *filename = tokens_wstr[10];
    }
}

void CFractalTrayDlg::OnBnClickedButtonGenerate ()
{ if (m_Thread != NULL)
  { MessageBox (L"Can't generate a new location file while the current one is being processed.  Exit the program, delete the location.txt file, and restart.");
    return;
  }

  UpdateData (TRUE);
  CStdioFile locationFile (m_LocationFilename, CFile::modeCreate | CFile::modeWrite | CFile::typeText);

  HighPrecision destMinX, destMinY, destMaxX, destMaxY;
  HighPrecision curMinX, curMinY, curMaxX, curMaxY;
  HighPrecision deltaXMin, deltaYMin, deltaXMax, deltaYMax;

  uint32_t targetIters;
  uint32_t sourceIters;
  double curIters;

  uint32_t destWidth;
  uint32_t destHeight;
  uint32_t srcWidth;
  uint32_t srcHeight;

  uint32_t iterationAntialiasing;
  uint32_t gpuAntialiasing;
  uint32_t iterationPrecision;

  ConvertCStringToDest(m_SourceCoords, srcWidth, srcHeight, curMinX, curMinY, curMaxX, curMaxY, &sourceIters, &iterationAntialiasing, &gpuAntialiasing, &iterationPrecision);
  ConvertCStringToDest(m_DestCoords, destWidth, destHeight, destMinX, destMinY, destMaxX, destMaxY, &targetIters, &iterationAntialiasing, &gpuAntialiasing, &iterationPrecision);

  curIters = sourceIters;

  int MaxFrames = HowManyFrames ();
  double incIters = (targetIters - curIters) / (double) MaxFrames;

  /*CString debug;
  debug.Format ("\"%s\" \"%s\" %d %lf %lf %d %d",
     m_DestCoords, m_SourceCoords, m_NumFrames, m_ScaleFactor, incIters, baseIters, targetIters);
  MessageBox (debug);*/

  char outputImageFilename[256];
  size_t i;
  for (i = 0; i < MaxFrames; i++)
  { deltaXMin = (destMinX - curMinX) / m_ScaleFactor;
    deltaYMin = (destMinY - curMinY) / m_ScaleFactor;
    deltaXMax = (destMaxX - curMaxX) / m_ScaleFactor;
    deltaYMax = (destMaxY - curMaxY) / m_ScaleFactor;

    curMinX += deltaXMin;
    curMinY += deltaYMin;
    curMaxX += deltaXMax;
    curMaxY += deltaYMax;
    curIters += (double) incIters;

    sprintf(outputImageFilename, "output%05zd", i);

    std::stringstream ss;
    ss << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
    ss << m_ResX << " ";
    ss << m_ResY << " ";
    ss << curMinX << " ";
    ss << curMinY << " ";
    ss << curMaxX << " ";
    ss << curMaxY << " ";
    ss << curIters << " ";
    ss << iterationAntialiasing << " ";
    ss << gpuAntialiasing << " ";
    ss << iterationPrecision << " ";
    ss << std::string(outputImageFilename) << std::endl;

    std::string s = ss.str();
    const std::wstring outputString(s.begin(), s.end());

    locationFile.WriteString (outputString.c_str());
  }

  locationFile.Close ();
}

int CFractalTrayDlg::HowManyFrames (void)
{ UpdateData (TRUE);

  HighPrecision destMinX, destMinY, destMaxX, destMaxY;
  HighPrecision deltaXMin, deltaYMin, deltaXMax, deltaYMax;
  HighPrecision curMinX, curMinY, curMaxX, curMaxY;

  uint32_t srcWidth, srcHeight;
  uint32_t destWidth, destHeight;

  ConvertCStringToDest(m_SourceCoords, srcWidth, srcHeight, curMinX, curMinY, curMaxX, curMaxY);
  ConvertCStringToDest(m_DestCoords, destWidth, destHeight, destMinX, destMinY, destMaxX, destMaxY);

  int i;
  for (i = 0;; i++)
  { deltaXMin = (destMinX - curMinX) / m_ScaleFactor;
    deltaYMin = (destMinY - curMinY) / m_ScaleFactor;
    deltaXMax = (destMaxX - curMaxX) / m_ScaleFactor;
    deltaYMax = (destMaxY - curMaxY) / m_ScaleFactor;

    curMinX += deltaXMin;
    curMinY += deltaYMin;
    curMaxX += deltaXMax;
    curMaxY += deltaYMax;

    if (((destMaxX - destMinX) * (destMaxY - destMinY)) / ((curMaxX - curMinX) * (curMaxY - curMinY)) > .9)
    { break; }
  }

  return i;
}

LRESULT CFractalTrayDlg::OnTrayNotification (WPARAM wParam, LPARAM lParam)
{ // Delegate all the work back to the default implementation in CTrayIcon.
  return m_TrayIcon.OnTrayNotification (wParam, lParam);
}

void CFractalTrayDlg::OnRestore ()
{ ShowWindow (SW_SHOW); }

void CFractalTrayDlg::OnExit ()
{ PostMessage (WM_CLOSE); }

LRESULT CFractalTrayDlg::OnFinishedCalculating (WPARAM, LPARAM)
{ m_TrayIcon.SetIcon (m_hIcon2);
  SetIcon (m_hIcon2, TRUE);      // Set big icon
  SetIcon (m_hIcon2, FALSE);    // Set small icon
  m_Thread = NULL;
  return 0;
}

DWORD WINAPI CalcProc (LPVOID lpParameter)
{ CThreadParam *param = (CThreadParam *) lpParameter;
  CStdioFile locationFile;

  // Open the location file.
  if (locationFile.Open (*param->LocationFilename, CFile::modeRead | CFile::typeText) == 0)
  { PostMessage (param->hWnd, WM_FINISHED_CALCULATING, 0, 0);
    return 1;
  }

  CString line;
  uint32_t resX, resY;
  HighPrecision minX, minY, maxX, maxY;
  uint32_t numIters;

  uint32_t iterationAntialiasing;
  uint32_t gpuAntialiasing;
  uint32_t iterationPrecision;

  std::wstring filename, filename_bmp, filename_png;

  FractalSetupData setup;
  setup.Load (true);
  setup.m_AlgHighRes = param->Algorithm;
  setup.m_AlgLowRes = 0;
  GetCurrentDirectory (128, setup.m_SaveDir);

  // default width/height:
  Fractal *fractal = new Fractal (&setup, 3840, 1600, OutputMessage, NULL, false);

  for (int i = 0;; i++)
  { if (locationFile.ReadString (line) == FALSE)
    { break; }

    ConvertCStringToDest(
        line,
        resX,
        resY,
        minX,
        minY,
        maxX,
        maxY,
        &numIters,
        &iterationAntialiasing,
        &gpuAntialiasing,
        &iterationPrecision,
        &filename);

    filename_bmp = filename + std::wstring(L".bmp");
    filename_png = filename + std::wstring(L".png");

    if (FileExists (filename) || FileExists (filename_png) || FileExists(filename_bmp))
    { continue; }

    fractal->SetNumIterations (numIters);
    fractal->SetIterationPrecision(iterationPrecision);
    fractal->ResetDimensions(resX, resY, iterationAntialiasing, gpuAntialiasing);
    fractal->RecenterViewCalc(minX, minY, maxX, maxY);
    fractal->CalcFractal (true);
    fractal->SaveCurrentFractal (filename);
    //fractal->CalcDiskFractal (filename_bmp);

    if (param->stop == true)
    { break; }
  }

  delete fractal;

  locationFile.Close ();
  PostMessage (param->hWnd, WM_FINISHED_CALCULATING, 0, 0);
  return 0;
}

bool FileExists (const std::wstring &filename)
{ _wfinddata_t fileinfo;
  intptr_t handle = _wfindfirst (filename.c_str(), &fileinfo);
  if (handle == -1)
  { return false; }

  _findclose (handle);

  return true;
}

void OutputMessage (const wchar_t *szFormat, ...)
{ if (theActiveOne->m_ThreadParam.stop == true)
  { return; }

  va_list argList;
  va_start (argList, szFormat);

  wchar_t newMessage[2048];
  vswprintf (newMessage, 2048, szFormat, argList);

  theActiveOne->m_Messages = newMessage;
  //theActiveOne->UpdateData (FALSE);

  va_end (argList);
}
