#include "StdAfx.h"
#include "DynamicPopupMenu.h"
#include "UniqueHMenu.h"

#include <cstddef> // size_t
#include <cwchar>  // wmemcpy, wcslen, wmemset
#include <utility> // std::exchange

#include "resource.h"

namespace FractalShark {

// -------------------- UniqueHMenu --------------------

UniqueHMenu::~UniqueHMenu() { reset(); }

UniqueHMenu &
UniqueHMenu::operator=(UniqueHMenu &&other) noexcept
{
    if (this != &other) {
        reset();
        h_ = std::exchange(other.h_, nullptr);
    }
    return *this;
}

void
UniqueHMenu::reset(HMENU h) noexcept
{
    if (h_) {
        ::DestroyMenu(h_);
    }
    h_ = h;
}

// -------------------- DynamicPopupMenu --------------------

HMENU
DynamicPopupMenu::GetPopup(HMENU rootMenu) noexcept
{
    return rootMenu ? ::GetSubMenu(rootMenu, 0) : nullptr;
}

UINT
DynamicPopupMenu::GetMenuItemCountSafe(HMENU menu) noexcept
{
    const int c = ::GetMenuItemCount(menu);
    return (c < 0) ? 0u : static_cast<UINT>(c);
}

UINT
DynamicPopupMenu::MapEnabledStateToMFS(UINT mfEnabled) noexcept
{
    // We accept a bool-like MF enabled decision and convert to MFS.
    // (Kept as UINT to match old style.)
    return (mfEnabled != 0) ? MFS_ENABLED : MFS_DISABLED;
}

UINT
DynamicPopupMenu::GetEnabledState(const Node &n, const IMenuState &state) noexcept
{
    // Convert Rule evaluation into MFS state.
    const bool enabled = state.IsEnabled(n.enableRule);
    return enabled ? MFS_ENABLED : MFS_DISABLED;
}

bool
DynamicPopupMenu::IsCheckedNow(const Node &n, const IMenuState &state) noexcept
{
    switch (n.checkKind) {
        case CheckKind::Toggle:
            return state.IsChecked(n.id);

        case CheckKind::Radio: {
            const UINT sel = state.GetRadioSelection(n.radioGroup);
            return (sel != 0 && sel == n.id);
        }

        case CheckKind::None:
        default:
            return false;
    }
}

void
DynamicPopupMenu::BuildPopupLabel(const Node &n,
                                  const IMenuState &state,
                                  /*out*/ wchar_t *buf,
                                  size_t bufCount) noexcept
{
    if (!buf || bufCount == 0) {
        return;
    }

    // Default: just the base label.
    buf[0] = L'\0';

    // Copy base label.
    const size_t baseLen = n.text.size();
    const size_t baseCopy = (baseLen < (bufCount - 1)) ? baseLen : (bufCount - 1);
    if (baseCopy > 0) {
        wmemcpy(buf, n.text.data(), baseCopy);
        buf[baseCopy] = L'\0';
    }

    // Optional adornment: "Label (Selection)"
    if (n.adornGroup == RadioGroup::None) {
        return;
    }

    const UINT adornId = state.GetPopupAdornmentCommandId(n.adornGroup);
    const UINT selId = (adornId != 0) ? adornId : state.GetRadioSelection(n.adornGroup);
    if (selId == 0) {
        return;
    }

    const std::wstring_view selLabel = state.GetCommandLabel(selId);
    if (selLabel.empty()) {
        return; // no label available => no adornment
    }

    // Append " (" + selLabel + ")"
    size_t cur = wcslen(buf);
    auto append = [&](const wchar_t *s) {
        if (!s)
            return;
        const size_t sl = wcslen(s);
        const size_t rem = (cur < bufCount) ? (bufCount - 1 - cur) : 0;
        const size_t cp = (sl < rem) ? sl : rem;
        if (cp > 0) {
            wmemcpy(buf + cur, s, cp);
            cur += cp;
            buf[cur] = L'\0';
        }
    };
    auto append_view = [&](std::wstring_view v) {
        const size_t rem = (cur < bufCount) ? (bufCount - 1 - cur) : 0;
        const size_t cp = (v.size() < rem) ? v.size() : rem;
        if (cp > 0) {
            wmemcpy(buf + cur, v.data(), cp);
            cur += cp;
            buf[cur] = L'\0';
        }
    };

    append(L" (");
    append_view(selLabel);
    append(L")");
}

bool
DynamicPopupMenu::InsertSeparatorAtEnd(HMENU menu)
{
    MENUITEMINFOW mii{};
    mii.cbSize = sizeof(mii);
    mii.fMask = MIIM_FTYPE;
    mii.fType = MFT_SEPARATOR;

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*by position*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertItemAtEnd(HMENU menu, const Node &n, const IMenuState &state)
{
    MENUITEMINFOW mii{};
    mii.cbSize = sizeof(mii);

    mii.fMask = MIIM_FTYPE | MIIM_STRING | MIIM_ID | MIIM_STATE;

    // Optional extras
    if (n.hbmpItem != nullptr) {
        mii.fMask |= MIIM_BITMAP;
        mii.hbmpItem = n.hbmpItem;
    }
    if (n.itemData != 0) {
        mii.fMask |= MIIM_DATA;
        mii.dwItemData = n.itemData;
    }

    mii.fType = MFT_STRING;

    // Radio-check styling for radio items.
    if (n.checkKind == CheckKind::Radio) {
        mii.fType |= MFT_RADIOCHECK;
    }

    if (n.ownerDraw) {
        mii.fType |= MFT_OWNERDRAW;
    }

    mii.wID = n.id;

    UINT stateFlags = GetEnabledState(n, state);

    if (IsCheckedNow(n, state)) {
        stateFlags |= MFS_CHECKED;
    }

    if (n.isDefault) {
        stateFlags |= MFS_DEFAULT;
    }

    mii.fState = stateFlags;

    // InsertMenuItemW uses non-const pointer type; Windows won't modify it.
    mii.dwTypeData = const_cast<wchar_t *>(n.text.data());
    mii.cch = static_cast<UINT>(n.text.size());

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*by position*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertPopupAtEnd(HMENU menu, const Node &n, HMENU popup, const IMenuState &state)
{
    MENUITEMINFOW mii{};
    mii.cbSize = sizeof(mii);

    mii.fMask = MIIM_FTYPE | MIIM_STRING | MIIM_SUBMENU | MIIM_STATE;

    // Optional extras
    if (n.hbmpItem != nullptr) {
        mii.fMask |= MIIM_BITMAP;
        mii.hbmpItem = n.hbmpItem;
    }
    if (n.itemData != 0) {
        mii.fMask |= MIIM_DATA;
        mii.dwItemData = n.itemData;
    }

    mii.fType = MFT_STRING;
    if (n.ownerDraw) {
        mii.fType |= MFT_OWNERDRAW;
    }

    UINT stateFlags = GetEnabledState(n, state);
    if (n.isDefault) {
        stateFlags |= MFS_DEFAULT;
    }
    mii.fState = stateFlags;

    mii.hSubMenu = popup;

    // If adornment is requested, build a temporary label.
    wchar_t labelBuf[256];
    if (n.adornGroup != RadioGroup::None) {
        BuildPopupLabel(n, state, labelBuf, sizeof(labelBuf) / sizeof(labelBuf[0]));
        mii.dwTypeData = labelBuf;
        mii.cch = static_cast<UINT>(wcslen(labelBuf));
    } else {
        mii.dwTypeData = const_cast<wchar_t *>(n.text.data());
        mii.cch = static_cast<UINT>(n.text.size());
    }

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*by position*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertNodeAtEnd(HMENU menu, const Node &n, const IMenuState &state)
{
    switch (n.kind) {
        case Kind::Separator:
            return InsertSeparatorAtEnd(menu);

        case Kind::Item:
            return InsertItemAtEnd(menu, n, state);

        case Kind::Popup: {
            HMENU sub = ::CreatePopupMenu();
            if (!sub) {
                return false;
            }

            if (!InsertPopupAtEnd(menu, n, sub, state)) {
                ::DestroyMenu(sub);
                return false;
            }

            if (!BuildMenuTree(sub, n.kids, state)) {
                // Parent menu owns the submenu only if insertion succeeded;
                // since insertion succeeded, destroying parent will destroy submenus.
                // But to be safe here, explicitly destroy.
                ::DestroyMenu(sub);
                return false;
            }

            return true;
        }

        default:
            return false;
    }
}

bool
DynamicPopupMenu::BuildMenuTree(HMENU parent, std::span<const Node> nodes, const IMenuState &state)
{
    for (const Node &n : nodes) {
        if (!InsertNodeAtEnd(parent, n, state)) {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Menu creation
// -----------------------------------------------------------------------------

UniqueHMenu
DynamicPopupMenu::Create(const IMenuState &state)
{
    UniqueHMenu root(::CreateMenu());
    if (!root) {
        return {};
    }

    HMENU popup = ::CreatePopupMenu();
    if (!popup) {
        return {};
    }

    // Top-level "POPUP" item.
    Node top = Popup(L"POPUP", {}, Rule::Always, RadioGroup::None);

    if (!InsertPopupAtEnd(root.get(), top, popup, state)) {
        ::DestroyMenu(popup);
        return {};
    }

    // Build popup contents from the tree.
    // This header defines: static const Node menu[] = {...};
    // It is included here to keep the structure in a data-only definition unit.
    if (!BuildPopupContents(popup, state)) {
        // Destroying root destroys attached submenus too.
        return {};
    }

    return root;
}

bool
DynamicPopupMenu::BuildPopupContents(HMENU popup, const IMenuState &state)
{
#include "DynamicPopupTreeDef.h"

    return BuildMenuTree(
        popup, std::initializer_list<Node>(menu, menu + (sizeof(menu) / sizeof(menu[0]))), state);
}

} // namespace FractalShark
