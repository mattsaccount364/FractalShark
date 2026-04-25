#include "stdafx.h"
#include "FractalTrayDlg.h"
#include "resource.h"
#include "PrecisionCalculator.h"
#include "Fractal.h"

constexpr size_t PrecisionLimit = 50000;
constexpr auto *FilePrefix = L"\\\\192.168.4.1\\Archive\\Fractal Saves\\lav2\\";
constexpr int StartAt = 0;

// --- Utility functions ---

static std::string NarrowString(const std::wstring &wide) {
    std::string narrow;
    narrow.reserve(wide.size());
    for (wchar_t ch : wide) {
        narrow.push_back(static_cast<char>(ch));
    }
    return narrow;
}

static std::wstring WidenString(const std::string &narrow) {
    return std::wstring(narrow.begin(), narrow.end());
}

static std::vector<std::string> TokenizeLine(const std::string &line) {
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

struct FrameParams {
    uint32_t width = 0;
    uint32_t height = 0;
    HighPrecision minX, minY, maxX, maxY;
    uint32_t iters = 0;
    uint32_t gpuAntialiasing = 0;
    std::wstring filename;
};

static FrameParams ParseFrameParams(const std::string &line) {
    auto tokens = TokenizeLine(line);
    FrameParams params;

    if (tokens.size() >= 2) {
        params.width = static_cast<uint32_t>(std::stoi(tokens[0]));
        params.height = static_cast<uint32_t>(std::stoi(tokens[1]));
    }
    if (tokens.size() >= 6) {
        params.minX = HighPrecision{tokens[2].c_str()};
        params.minY = HighPrecision{tokens[3].c_str()};
        params.maxX = HighPrecision{tokens[4].c_str()};
        params.maxY = HighPrecision{tokens[5].c_str()};
    }
    if (tokens.size() >= 8) {
        params.iters = static_cast<uint32_t>(std::stoi(tokens[6]));
        params.gpuAntialiasing = static_cast<uint32_t>(std::stoi(tokens[7]));
    }
    if (tokens.size() >= 9) {
        params.filename = WidenString(tokens[8]);
    }
    return params;
}

static std::wstring ExtractFilename(const std::string &line) {
    auto tokens = TokenizeLine(line);
    if (tokens.size() >= 9) {
        return WidenString(tokens[8]);
    }
    return {};
}

// --- FractalTrayDialog ---

FractalTrayDialog::FractalTrayDialog()
    : m_SourceCoords(
          L"16384 6826 "
          L"-4.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 "
          L"-1.5 "
          L"3.118537200504413619167717528373266078184110970996216897856242118537200504413619167717528373266078184110970996216756329721 "
          L"1.5 8192 2") {
}

INT_PTR FractalTrayDialog::DoModal(HINSTANCE hInst) {
    m_hInst = hInst;
    return DialogBoxParam(
        hInst,
        MAKEINTRESOURCE(IDD_FRACTALTRAY_DIALOG),
        nullptr,
        StaticDlgProc,
        reinterpret_cast<LPARAM>(this));
}

INT_PTR CALLBACK FractalTrayDialog::StaticDlgProc(
    HWND hDlg, UINT msg, WPARAM wParam, LPARAM lParam) {

    FractalTrayDialog *self;
    if (msg == WM_INITDIALOG) {
        self = reinterpret_cast<FractalTrayDialog *>(lParam);
        self->m_hDlg = hDlg;
        SetWindowLongPtr(hDlg, DWLP_USER, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<FractalTrayDialog *>(
            GetWindowLongPtr(hDlg, DWLP_USER));
    }

    if (self) {
        return self->HandleMessage(msg, wParam, lParam);
    }
    return FALSE;
}

INT_PTR FractalTrayDialog::HandleMessage(
    UINT msg, WPARAM wParam, LPARAM lParam) {

    switch (msg) {
    case WM_INITDIALOG:
        return OnInitDialog();

    case WM_PAINT:
        if (IsIconic(m_hDlg)) {
            OnPaintIconic();
            return TRUE;
        }
        return FALSE;

    case WM_SYSCOMMAND:
        if (wParam == SC_MINIMIZE) {
            ShowWindow(m_hDlg, SW_HIDE);
            return TRUE;
        }
        return FALSE;

    case WM_CLOSE:
        OnClose();
        return TRUE;

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDC_BUTTON_GENERATE:
            OnGenerate();
            return TRUE;
        case ID_POPUP_RESTORE:
            OnRestore();
            return TRUE;
        case ID_POPUP_EXIT:
            OnExit();
            return TRUE;
        case IDCANCEL:
            OnClose();
            return TRUE;
        case IDOK:
            return TRUE;
        }
        break;

    case WM_ICON_NOTIFY:
        SetWindowLongPtr(m_hDlg, DWLP_MSGRESULT,
                         OnTrayNotification(wParam, lParam));
        return TRUE;

    case WM_FINISHED_CALCULATING:
        SetWindowLongPtr(m_hDlg, DWLP_MSGRESULT,
                         OnFinishedCalculating());
        return TRUE;

    case WM_QUERYDRAGICON:
        SetWindowLongPtr(m_hDlg, DWLP_MSGRESULT,
                         reinterpret_cast<LONG_PTR>(m_hIconIdle));
        return TRUE;
    }

    return FALSE;
}

BOOL FractalTrayDialog::OnInitDialog() {
    m_hIconActive = LoadIcon(m_hInst, MAKEINTRESOURCE(IDR_FRACTALTRAY1));
    m_hIconIdle = LoadIcon(m_hInst, MAKEINTRESOURCE(IDR_FRACTALTRAY2));

    TryLoadDestCoords();

    SetDlgText(IDC_EDIT_SOURCECOORDS, m_SourceCoords);
    SetDlgText(IDC_EDIT_DESTCOORDS, m_DestCoords);
    SetDlgText(IDC_EDIT_SCALEFACTOR, std::format(L"{}", m_ScaleFactor));
    SetDlgItemInt(m_hDlg, IDC_EDIT_RESX, m_ResX, FALSE);
    SetDlgItemInt(m_hDlg, IDC_EDIT_RESY, m_ResY, FALSE);
    SetDlgText(IDC_EDIT_MESSAGES, L"");

    m_TrayIcon.Create(m_hDlg, WM_ICON_NOTIFY, L"FractalTray",
                      m_hIconIdle, IDR_MENU_TRAY);

    if (std::filesystem::exists(m_LocationFilename)) {
        StartCalculation();
        PostMessage(m_hDlg, WM_SYSCOMMAND, SC_MINIMIZE, 0);

        SendMessage(m_hDlg, WM_SETICON, ICON_BIG,
                    reinterpret_cast<LPARAM>(m_hIconActive));
        SendMessage(m_hDlg, WM_SETICON, ICON_SMALL,
                    reinterpret_cast<LPARAM>(m_hIconActive));
        m_TrayIcon.SetIcon(m_hIconActive);
    } else {
        SendMessage(m_hDlg, WM_SETICON, ICON_BIG,
                    reinterpret_cast<LPARAM>(m_hIconIdle));
        SendMessage(m_hDlg, WM_SETICON, ICON_SMALL,
                    reinterpret_cast<LPARAM>(m_hIconIdle));
        m_TrayIcon.SetIcon(m_hIconIdle);
    }

    m_TrayIcon.Show();
    return TRUE;
}

void FractalTrayDialog::TryLoadDestCoords() {
    std::ifstream file(m_LocationFilename);
    if (!file.is_open()) return;

    std::string line;
    if (!std::getline(file, line)) return;

    auto filename = ExtractFilename(line);
    if (filename == L"FractalTrayDestination") {
        m_DestCoords = WidenString(line);
        file.close();

        auto result = ::MessageBox(
            m_hDlg,
            L"Do you want to delete the location file?",
            L"Delete location file?", MB_YESNO);
        if (result == IDYES) {
            std::filesystem::remove(m_LocationFilename);
        }
    }
}

void FractalTrayDialog::OnPaintIconic() {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(m_hDlg, &ps);

    SendMessage(m_hDlg, WM_ICONERASEBKGND,
                reinterpret_cast<WPARAM>(hdc), 0);

    int cxIcon = GetSystemMetrics(SM_CXICON);
    int cyIcon = GetSystemMetrics(SM_CYICON);
    RECT rect;
    GetClientRect(m_hDlg, &rect);
    int x = (rect.right - rect.left - cxIcon + 1) / 2;
    int y = (rect.bottom - rect.top - cyIcon + 1) / 2;

    DrawIcon(hdc, x, y, m_hIconActive);
    EndPaint(m_hDlg, &ps);
}

void FractalTrayDialog::OnClose() {
    if (m_CalcThread.joinable()) {
        m_CalcThread.request_stop();
        m_CalcThread.join();
    }
    m_TrayIcon.Remove();
    EndDialog(m_hDlg, IDCANCEL);
}

LRESULT FractalTrayDialog::OnTrayNotification(WPARAM wParam, LPARAM lParam) {
    return m_TrayIcon.OnTrayNotification(wParam, lParam);
}

void FractalTrayDialog::OnRestore() {
    ShowWindow(m_hDlg, SW_SHOW);
}

void FractalTrayDialog::OnExit() {
    PostMessage(m_hDlg, WM_CLOSE, 0, 0);
}

LRESULT FractalTrayDialog::OnFinishedCalculating() {
    m_TrayIcon.SetIcon(m_hIconIdle);
    SendMessage(m_hDlg, WM_SETICON, ICON_BIG,
                reinterpret_cast<LPARAM>(m_hIconIdle));
    SendMessage(m_hDlg, WM_SETICON, ICON_SMALL,
                reinterpret_cast<LPARAM>(m_hIconIdle));
    m_CalcThread = {};
    return 0;
}

// --- Frame generation ---

void FractalTrayDialog::OnGenerate() {
    if (m_CalcThread.joinable()) {
        ::MessageBox(m_hDlg,
            L"Can't generate a new location file while the current one is "
            L"being processed. Exit the program, delete the location.txt "
            L"file, and restart.",
            L"FractalTray", MB_OK);
        return;
    }

    ReadControlsToMembers();

    std::ofstream locationFile(m_LocationFilename);
    if (!locationFile.is_open()) return;

    HighPrecision::defaultPrecisionInBits(PrecisionLimit);

    auto srcParams = ParseFrameParams(NarrowString(m_SourceCoords));
    auto dstParams = ParseFrameParams(NarrowString(m_DestCoords));

    const bool requiresReuse = false;
    PointZoomBBConverter ptz{
        srcParams.minX, srcParams.minY, srcParams.maxX, srcParams.maxY,
        PointZoomBBConverter::TestMode::Enabled};
    auto precInBits = PrecisionCalculator::GetPrecision(ptz, requiresReuse);
    ptz.SetPrecision(precInBits);

    auto precInDigits = static_cast<int>(
        static_cast<double>(precInBits) * std::log10(2.0));

    double curIters = srcParams.iters;
    int maxFrames = CalculateFrameCount();
    double incIters = (dstParams.iters - curIters) / static_cast<double>(maxFrames);

    HighPrecision curMinX = srcParams.minX;
    HighPrecision curMinY = srcParams.minY;
    HighPrecision curMaxX = srcParams.maxX;
    HighPrecision curMaxY = srcParams.maxY;

    std::vector<std::string> lines;
    for (int i = 0; i < maxFrames; i++) {
        HighPrecision scaleFactor{m_ScaleFactor};
        HighPrecision deltaXMin = (dstParams.minX - curMinX) / scaleFactor;
        HighPrecision deltaYMin = (dstParams.minY - curMinY) / scaleFactor;
        HighPrecision deltaXMax = (dstParams.maxX - curMaxX) / scaleFactor;
        HighPrecision deltaYMax = (dstParams.maxY - curMaxY) / scaleFactor;

        curMinX += deltaXMin;
        curMinY += deltaYMin;
        curMaxX += deltaXMax;
        curMaxY += deltaYMax;
        curIters += incIters;

        auto outputFilename = std::format("output-{:06}", i);

        std::ostringstream ss;
        ss << std::setprecision(precInDigits);
        ss << m_ResX << " " << m_ResY << " ";
        ss << curMinX << " " << curMinY << " ";
        ss << curMaxX << " " << curMaxY << " ";
        ss << curIters << " " << dstParams.gpuAntialiasing << " ";
        ss << outputFilename << "\n";

        lines.push_back(ss.str());
    }

    std::ranges::reverse(lines);

    for (const auto &line : lines) {
        locationFile << line;
    }
}

int FractalTrayDialog::CalculateFrameCount() {
    ReadControlsToMembers();

    HighPrecision::defaultPrecisionInBits(PrecisionLimit);

    auto srcParams = ParseFrameParams(NarrowString(m_SourceCoords));
    auto dstParams = ParseFrameParams(NarrowString(m_DestCoords));

    HighPrecision curMinX = srcParams.minX;
    HighPrecision curMinY = srcParams.minY;
    HighPrecision curMaxX = srcParams.maxX;
    HighPrecision curMaxY = srcParams.maxY;

    HighPrecision scaleFactor{m_ScaleFactor};
    HighPrecision threshold{0.9};

    int frameCount;
    for (frameCount = 0;; frameCount++) {
        HighPrecision deltaXMin = (dstParams.minX - curMinX) / scaleFactor;
        HighPrecision deltaYMin = (dstParams.minY - curMinY) / scaleFactor;
        HighPrecision deltaXMax = (dstParams.maxX - curMaxX) / scaleFactor;
        HighPrecision deltaYMax = (dstParams.maxY - curMaxY) / scaleFactor;

        curMinX += deltaXMin;
        curMinY += deltaYMin;
        curMaxX += deltaXMax;
        curMaxY += deltaYMax;

        auto destArea =
            (dstParams.maxX - dstParams.minX) * (dstParams.maxY - dstParams.minY);
        auto curArea =
            (curMaxX - curMinX) * (curMaxY - curMinY);

        if (destArea / curArea > threshold) break;
    }

    return frameCount;
}

// --- Background calculation thread ---

void FractalTrayDialog::StartCalculation() {
    m_CalcThread = std::jthread([this](std::stop_token stopToken) {
        RunCalculation(std::move(stopToken));
    });
}

void FractalTrayDialog::RunCalculation(std::stop_token stopToken) {
    std::ifstream locationFile(m_LocationFilename);
    if (!locationFile.is_open()) {
        PostMessage(m_hDlg, WM_FINISHED_CALCULATING, 0, 0);
        return;
    }

    const bool requiresReuse = false;
    auto fractal = std::make_unique<Fractal>(
        DefaultWidth, DefaultHeight, nullptr, false, 0);

    std::string line;
    for (int i = 0; std::getline(locationFile, line); i++) {
        if (stopToken.stop_requested()) break;
        if constexpr (StartAt != 0) { if (i < StartAt) continue; }

        auto filename = ExtractFilename(line);
        if (filename.empty()) continue;

        auto filenameBmp = std::wstring(FilePrefix) + filename + L".bmp";
        auto filenamePng = std::wstring(FilePrefix) + filename + L".png";

        if (std::filesystem::exists(filenamePng) ||
            std::filesystem::exists(filenameBmp)) {
            continue;
        }

        HighPrecision::defaultPrecisionInBits(PrecisionLimit);

        auto params = ParseFrameParams(line);
        auto fullPath = std::wstring(FilePrefix) + params.filename;

        PointZoomBBConverter ptz{
            params.minX, params.minY, params.maxX, params.maxY,
            PointZoomBBConverter::TestMode::Enabled};
        auto prec = PrecisionCalculator::GetPrecision(ptz, requiresReuse);
        fractal->SetPrecision(prec);
        fractal->SetNumIterations<uint32_t>(params.iters);
        fractal->ResetDimensions(params.width, params.height, params.gpuAntialiasing);
        fractal->RecenterViewCalc(ptz);
        fractal->UsePalette(8);
        fractal->CalcFractal(true);
        fractal->SaveCurrentFractal(fullPath, false);
    }

    PostMessage(m_hDlg, WM_FINISHED_CALCULATING, 0, 0);
}

// --- UI helpers ---

std::wstring FractalTrayDialog::GetDlgText(int controlId) const {
    HWND hCtrl = GetDlgItem(m_hDlg, controlId);
    int len = GetWindowTextLength(hCtrl);
    if (len == 0) return {};
    std::wstring text(static_cast<size_t>(len) + 1, L'\0');
    GetWindowText(hCtrl, text.data(), len + 1);
    text.resize(static_cast<size_t>(len));
    return text;
}

void FractalTrayDialog::SetDlgText(int controlId, const std::wstring &text) {
    SetDlgItemText(m_hDlg, controlId, text.c_str());
}

void FractalTrayDialog::ReadControlsToMembers() {
    m_SourceCoords = GetDlgText(IDC_EDIT_SOURCECOORDS);
    m_DestCoords = GetDlgText(IDC_EDIT_DESTCOORDS);
    m_ScaleFactor = std::stod(NarrowString(GetDlgText(IDC_EDIT_SCALEFACTOR)));
    m_ResX = static_cast<int>(GetDlgItemInt(m_hDlg, IDC_EDIT_RESX, nullptr, FALSE));
    m_ResY = static_cast<int>(GetDlgItemInt(m_hDlg, IDC_EDIT_RESY, nullptr, FALSE));
}
