// FractalTrayDlg.cpp : implementation file
//

#include "stdafx.h"


// --- MPIR/GMP include fence: avoid ASSERT macro collision with MFC ---
#ifdef ASSERT
#pragma push_macro("ASSERT")
#undef ASSERT
#define RESTORE_MFC_ASSERT
#endif


#include "FractalTray.h"
#include "FractalTrayDlg.h"
#include ".\fractaltraydlg.h"
#include "PrecisionCalculator.h"

#include <math.h>
#include <io.h>
#include "Fractal.h"
//#include "..\cximage599a_full\CxImage\xImage.h"

constexpr size_t PrecisionLimit = 50000;
constexpr double DefaultScaleFactor = 75;
constexpr int DefaultWidth = 3840;
constexpr int DefaultHeight = 1600;
//constexpr auto *fileprefix = L"\\\\192.168.4.1\\Archive\\Fractal Saves\\2023_10e4000\\";
constexpr auto* fileprefix = L"\\\\192.168.4.1\\Archive\\Fractal Saves\\lav2\\";
//constexpr auto* fileprefix = L"";
constexpr int startAt = 0;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

void ConvertCStringToDest(const CString& str,
    uint32_t* width,
    uint32_t* height,
    HighPrecision* minX,
    HighPrecision* minY,
    HighPrecision* maxX,
    HighPrecision* maxY,
    uint32_t* iters = nullptr,
    uint32_t* gpuAntialiasing = nullptr,
    std::wstring* filename = nullptr);

CFractalTrayDlg* theActiveOne;

CFractalTrayDlg::CFractalTrayDlg(CWnd* pParent /*=nullptr*/)
    : CDialog(CFractalTrayDlg::IDD, pParent)
    , m_DestCoords(_T(""))
    , m_ScaleFactor(DefaultScaleFactor)
    , m_ResX(DefaultWidth)
    , m_ResY(DefaultHeight)
    , m_SourceCoords(_T("16384 6826 -4.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 -1.5 3.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 1.5 8192 2"))
    , m_LocationFilename("locations.txt")
    , m_Messages(_T(""))
{
    m_hIcon1 = AfxGetApp()->LoadIcon(IDR_FRACTALTRAY1);
    m_hIcon2 = AfxGetApp()->LoadIcon(IDR_FRACTALTRAY2);

    TryLoadDestCoords();
}

// The idea here is that if the user has a location.txt file, we'll load it
// and use it to populate the destination coordinates.  This way, the user
// can just edit the location.txt file to change the destination coordinates
// without having to use the GUI.
void CFractalTrayDlg::TryLoadDestCoords() {
    CStdioFile locationFile;

    // Open the location file.
    if (locationFile.Open(m_LocationFilename, CFile::modeRead | CFile::typeText) == 0) {
        return;
    }

    CString line;
    if (locationFile.ReadString(line) == FALSE) {
        return;
    }

    std::wstring filename;

    ConvertCStringToDest(
        line,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &filename);

    if (filename == L"FractalTrayDestination") {
        m_DestCoords = line;

        locationFile.Close();

        // Delete the location file so we don't keep loading it.
        auto result = MessageBox(L"Do you want to delete the location file?", L"Delete location file?", MB_YESNO);
        if (result == IDYES) {
            DeleteFile(m_LocationFilename);
        }
    }
}

void CFractalTrayDlg::DoDataExchange(CDataExchange* pDX)
{
    DDX_Text(pDX, IDC_EDIT_SOURCECOORDS, m_SourceCoords);
    DDX_Text(pDX, IDC_EDIT_DESTCOORDS, m_DestCoords);
    DDX_Text(pDX, IDC_EDIT_SCALEFACTOR, m_ScaleFactor);
    DDX_Text(pDX, IDC_EDIT_RESX, m_ResX);
    DDX_Text(pDX, IDC_EDIT_RESY, m_ResY);
    DDX_Text(pDX, IDC_EDIT_MESSAGES, m_Messages);
    CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CFractalTrayDlg, CDialog)
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    //}}AFX_MSG_MAP
    ON_MESSAGE(WM_SYSCOMMAND, OnSysCommand)
    ON_BN_CLICKED(IDC_BUTTON_GENERATE, OnBnClickedButtonGenerate)
    ON_MESSAGE(WM_ICON_NOTIFY, OnTrayNotification)
    ON_MESSAGE(WM_DESTROY, OnDestroy)
    ON_MESSAGE(WM_FINISHED_CALCULATING, OnFinishedCalculating)
    ON_COMMAND(ID_POPUP_RESTORE, OnRestore)
    ON_COMMAND(ID_POPUP_EXIT, OnExit)
END_MESSAGE_MAP()


// CFractalTrayDlg message handlers

BOOL CFractalTrayDlg::OnInitDialog()
{
    CDialog::OnInitDialog();

    // TODO: Add extra initialization here
    theActiveOne = this;

    m_TrayIcon.Create(this, WM_ICON_NOTIFY, L"FractalTray",
        GetIcon(FALSE), IDR_MENU_TRAY);

    UpdateData(FALSE);
    if (FileExists(m_LocationFilename.operator LPCWSTR()) == true)
    {
        m_ThreadParam.LocationFilename = &m_LocationFilename;
        m_ThreadParam.stop = false;
        m_ThreadParam.hWnd = m_hWnd;

        m_Thread = CreateThread(nullptr, 0, CalcProc, &m_ThreadParam, 0, nullptr);
        PostMessage(WM_SYSCOMMAND, SC_MINIMIZE, 0);

        SetIcon(m_hIcon1, TRUE);      // Set big icon
        SetIcon(m_hIcon1, FALSE);    // Set small icon
        m_TrayIcon.SetIcon(m_hIcon1);
    }
    else
    {
        m_Thread = nullptr;
        SetIcon(m_hIcon2, TRUE);      // Set big icon
        SetIcon(m_hIcon2, FALSE);    // Set small icon
        m_TrayIcon.SetIcon(m_hIcon2);
    }

    m_TrayIcon.ShowIcon();

    //SetWindowPos (nullptr, 0, 0, 0, 0, SWP_HIDEWINDOW | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER);

    return TRUE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document \ view model,
//  this is automatically done for you by the framework.

LRESULT CFractalTrayDlg::OnSysCommand(WPARAM nID, LPARAM lParam)
{
    if (nID == SC_MINIMIZE)
    {
        ShowWindow(SW_HIDE);
        return 0;
    }
    else
    {
        CDialog::OnSysCommand((UINT)nID, lParam);
        return 0;
    }
}

LRESULT CFractalTrayDlg::OnDestroy(WPARAM, LPARAM)
{
    if (m_Thread != nullptr)
    {
        m_ThreadParam.stop = true;
        WaitForSingleObject(m_Thread, INFINITE);
    }

    //MessageBox ("!");

    CDialog::OnDestroy();
    return 0;
}

void CFractalTrayDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon1);
    }
    else
    {
        CDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CFractalTrayDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon2);
}

void GetTokensFromString(const CString& str,
    std::vector<std::wstring>& tokens,
    std::vector<const wchar_t*>& tokens_wstr,
    std::vector<std::string>& tokens_str) {
    std::wstring dest_coords = str.operator LPCTSTR ();
    dest_coords.erase(std::remove(dest_coords.begin(), dest_coords.end(), L'\n'), dest_coords.cend());
    dest_coords.erase(std::remove(dest_coords.begin(), dest_coords.end(), L'\r'), dest_coords.cend());

    std::wstringstream check1(dest_coords);
    std::wstring intermediate;

    // Tokenizing w.r.t. space ' '
    size_t i = 0;
    while (std::getline(check1, intermediate, L' '))
    {
        char temp[65535];
        tokens.push_back(intermediate);
        tokens_wstr.push_back(tokens[i].c_str());

        sprintf(temp, "%ws", tokens[i].c_str());
        tokens_str.push_back(temp);
        i++;
    }
}

void ConvertCStringToDest(const CString& str,
    uint32_t* width,
    uint32_t* height,
    HighPrecision* minX,
    HighPrecision* minY,
    HighPrecision* maxX,
    HighPrecision* maxY,
    uint32_t* iters,
    uint32_t* gpuAntialiasing,
    std::wstring* filename) {

    std::vector<std::wstring> tokens;
    std::vector<const wchar_t*> tokens_wstr;
    std::vector<std::string> tokens_str;
    GetTokensFromString(str, tokens, tokens_wstr, tokens_str);

    if (width != nullptr) {
        *width = atoi(tokens_str[0].c_str());
    }

    if (height != nullptr) {
        *height = atoi(tokens_str[1].c_str());
    }

    if (minX != nullptr) {
        *minX = HighPrecision(tokens_str[2].c_str());
    }

    if (minY != nullptr) {
        *minY = HighPrecision(tokens_str[3].c_str());
    }

    if (maxX != nullptr) {
        *maxX = HighPrecision(tokens_str[4].c_str());
    }

    if (maxY != nullptr) {
        *maxY = HighPrecision(tokens_str[5].c_str());
    }

    if (iters != nullptr) {
        *iters = atoi(tokens_str[6].c_str());
    }

    if (gpuAntialiasing != nullptr) {
        *gpuAntialiasing = atoi(tokens_str[7].c_str());
    }

    if (filename != nullptr) {
        *filename = tokens_wstr[8];
    }
}

void CFractalTrayDlg::OnBnClickedButtonGenerate()
{
    if (m_Thread != nullptr)
    {
        MessageBox(L"Can't generate a new location file while the current one is being processed.  Exit the program, delete the location.txt file, and restart.");
        return;
    }

    UpdateData(TRUE);
    CStdioFile locationFile(m_LocationFilename, CFile::modeCreate | CFile::modeWrite | CFile::typeText | CFile::shareDenyNone);

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

    uint32_t gpuAntialiasing;

    // Presumably 50k is enough
    // TODO vary dynamically as zoom deepens
    // TODO or just use float exp
    HighPrecision::defaultPrecisionInBits(PrecisionLimit);

    ConvertCStringToDest(
        m_SourceCoords,
        &srcWidth,
        &srcHeight,
        &curMinX,
        &curMinY,
        &curMaxX,
        &curMaxY,
        &sourceIters,
        &gpuAntialiasing);
    ConvertCStringToDest(
        m_DestCoords,
        &destWidth,
        &destHeight,
        &destMinX,
        &destMinY,
        &destMaxX,
        &destMaxY,
        &targetIters,
        &gpuAntialiasing);

    // Reduce precision to something semi-sane
    const bool requiresReuse = false;
    PointZoomBBConverter ptz{ curMinX, curMinY, curMaxX, curMaxY };
    auto precInBits = PrecisionCalculator::GetPrecision(ptz, requiresReuse);
    ptz.SetPrecision(precInBits);

    auto precInDigits = static_cast<uint64_t>(static_cast<double>(precInBits) / std::log10(2));

    curIters = sourceIters;

    int MaxFrames = HowManyFrames();
    double incIters = (targetIters - curIters) / (double)MaxFrames;

    /*CString debug;
    debug.Format ("\"%s\" \"%s\" %d %lf %lf %d %d",
       m_DestCoords, m_SourceCoords, m_NumFrames, m_ScaleFactor, incIters, baseIters, targetIters);
    MessageBox (debug);*/

    char outputImageFilename[256];
    size_t i;
    std::vector<std::wstring> final_file;
    for (i = 0; i < MaxFrames; i++)
    {
        HighPrecision scaleFactor{ m_ScaleFactor };
        deltaXMin = (destMinX - curMinX) / scaleFactor;
        deltaYMin = (destMinY - curMinY) / scaleFactor;
        deltaXMax = (destMaxX - curMaxX) / scaleFactor;
        deltaYMax = (destMaxY - curMaxY) / scaleFactor;

        curMinX += deltaXMin;
        curMinY += deltaYMin;
        curMaxX += deltaXMax;
        curMaxY += deltaYMax;
        curIters += (double)incIters;

        ptz = PointZoomBBConverter{ curMinX, curMinY, curMaxX, curMaxY };

        sprintf(outputImageFilename, "output-%06zd", i);

        std::stringstream ss;
        ss << std::setprecision(precInDigits);
        ss << m_ResX << " ";
        ss << m_ResY << " ";
        ss << curMinX << " ";
        ss << curMinY << " ";
        ss << curMaxX << " ";
        ss << curMaxY << " ";
        ss << curIters << " ";
        ss << gpuAntialiasing << " ";
        ss << std::string(outputImageFilename) << std::endl;

        std::string s = ss.str();
        const std::wstring outputString(s.begin(), s.end());
        final_file.push_back(outputString);
    }
    std::reverse(final_file.begin(), final_file.end());

    for (const auto& l : final_file) {
        locationFile.WriteString(l.c_str());
    }
    locationFile.Close();
}

int CFractalTrayDlg::HowManyFrames(void)
{
    UpdateData(TRUE);

    HighPrecision destMinX, destMinY, destMaxX, destMaxY;
    HighPrecision deltaXMin, deltaYMin, deltaXMax, deltaYMax;
    HighPrecision curMinX, curMinY, curMaxX, curMaxY;

    uint32_t srcWidth, srcHeight;
    uint32_t destWidth, destHeight;

    ConvertCStringToDest(m_SourceCoords, &srcWidth, &srcHeight, &curMinX, &curMinY, &curMaxX, &curMaxY);
    ConvertCStringToDest(m_DestCoords, &destWidth, &destHeight, &destMinX, &destMinY, &destMaxX, &destMaxY);

    int i;
    HighPrecision scaleFactor{ m_ScaleFactor };
    HighPrecision point9{ 0.9 };
    for (i = 0;; i++)
    {
        deltaXMin = (destMinX - curMinX) / scaleFactor;
        deltaYMin = (destMinY - curMinY) / scaleFactor;
        deltaXMax = (destMaxX - curMaxX) / scaleFactor;
        deltaYMax = (destMaxY - curMaxY) / scaleFactor;

        curMinX += deltaXMin;
        curMinY += deltaYMin;
        curMaxX += deltaXMax;
        curMaxY += deltaYMax;

        if (((destMaxX - destMinX) * (destMaxY - destMinY)) / ((curMaxX - curMinX) * (curMaxY - curMinY)) > point9)
        {
            break;
        }
    }

    return i;
}

LRESULT CFractalTrayDlg::OnTrayNotification(WPARAM wParam, LPARAM lParam)
{ // Delegate all the work back to the default implementation in CTrayIcon.
    return m_TrayIcon.OnTrayNotification(wParam, lParam);
}

void CFractalTrayDlg::OnRestore()
{
    ShowWindow(SW_SHOW);
}

void CFractalTrayDlg::OnExit()
{
    PostMessage(WM_CLOSE);
}

LRESULT CFractalTrayDlg::OnFinishedCalculating(WPARAM, LPARAM)
{
    m_TrayIcon.SetIcon(m_hIcon2);
    SetIcon(m_hIcon2, TRUE);      // Set big icon
    SetIcon(m_hIcon2, FALSE);    // Set small icon
    m_Thread = nullptr;
    return 0;
}

DWORD WINAPI CalcProc(LPVOID lpParameter)
{
    CThreadParam* param = (CThreadParam*)lpParameter;
    CStdioFile locationFile;

    // Open the location file.
    if (locationFile.Open(*param->LocationFilename, CFile::modeRead | CFile::typeText) == 0) {
        PostMessage(param->hWnd, WM_FINISHED_CALCULATING, 0, 0);
        return 1;
    }

    CString line;
    uint32_t resX, resY;
    HighPrecision minX, minY, maxX, maxY;
    uint32_t numIters;
    const bool requiresReuse = false;

    uint32_t gpuAntialiasing;

    std::wstring filename, filename_bmp, filename_png;

    // default width/height:
    Fractal* fractal = DEBUG_NEW Fractal(DefaultWidth, DefaultHeight, nullptr, false, 0);

    for (int i = 0;; i++) {
        if (param->stop == true) {
            break;
        }

        if (locationFile.ReadString(line) == FALSE) {
            break;
        }

        if (startAt != 0 && i < startAt) {
            continue;
        }

        ConvertCStringToDest(
            line,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            &filename);

        filename_bmp = fileprefix + filename + std::wstring(L".bmp");
        filename_png = fileprefix + filename + std::wstring(L".png");

        if (FileExists(filename) || FileExists(filename_png) || FileExists(filename_bmp)) {
            continue;
        }

        HighPrecision::defaultPrecisionInBits(PrecisionLimit);

        ConvertCStringToDest(
            line,
            &resX,
            &resY,
            &minX,
            &minY,
            &maxX,
            &maxY,
            &numIters,
            &gpuAntialiasing,
            &filename);

        filename = fileprefix + filename;

        PointZoomBBConverter ptz{ minX, minY, maxX, maxY };
        auto prec = PrecisionCalculator::GetPrecision(ptz, requiresReuse);
        fractal->SetPrecision(prec);

        // lame hack
        //numIters /= 200;
        //if (numIters < 10000) {
        //    numIters = 10000;
        //}

        fractal->SetNumIterations<uint32_t>(numIters);
        fractal->ResetDimensions(resX, resY, gpuAntialiasing);
        fractal->RecenterViewCalc(ptz);
        fractal->UsePalette(8);
        fractal->CalcFractal(true);
        fractal->SaveCurrentFractal(filename, false);
        //fractal->CalcDiskFractal (filename_bmp);
    }

    delete fractal;

    locationFile.Close();
    PostMessage(param->hWnd, WM_FINISHED_CALCULATING, 0, 0);
    return 0;
}

bool FileExists(const std::wstring& filename)
{
    //_wfinddata_t fileinfo;
    //intptr_t handle = _wfindfirst (filename.c_str(), &fileinfo);
    //if (handle == -1)
    //{ return false; }

    //_findclose (handle);

    DWORD dwAttrib = GetFileAttributes(filename.c_str());

    return (dwAttrib != INVALID_FILE_ATTRIBUTES &&
        !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}
