#include "stdafx.h"
#include "WaitCursor.h"

WaitCursor::WaitCursor() : m_hCursor{} {
    m_hCursor = SetCursor(LoadCursor(nullptr, IDC_WAIT));
}

WaitCursor::~WaitCursor() {
    ResetCursor();
}

void WaitCursor::ResetCursor() {
    SetCursor(m_hCursor);
}