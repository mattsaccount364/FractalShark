#include "stdafx.h"
#include "FractalTrayDlg.h"

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int) {
    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_WIN95_CLASSES};
    InitCommonControlsEx(&icc);

    FractalTrayDialog dlg;
    dlg.DoModal(hInstance);

    return 0;
}
