#include "WaitCursor.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace Environment {

WaitCursor::WaitCursor() : m_PreviousCursor{SetCursor(LoadCursor(nullptr, IDC_WAIT))}, m_CursorSet{true}
{
}

WaitCursor::~WaitCursor() { ResetCursor(); }

void
WaitCursor::ResetCursor()
{
    if (!m_CursorSet) {
        return;
    }

    SetCursor(static_cast<HCURSOR>(m_PreviousCursor));
    m_CursorSet = false;
}

void
WaitCursor::RegisterLinuxCursorTarget(void * /*display*/,
                                      std::uintptr_t /*window*/,
                                      std::uintptr_t /*normalCursor*/)
{
}

void
WaitCursor::UnregisterLinuxCursorTarget()
{
}

} // namespace Environment
