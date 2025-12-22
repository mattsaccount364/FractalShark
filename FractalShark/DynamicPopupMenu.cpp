#include "StdAfx.h"
#include "DynamicPopupMenu.h"
#include "UniqueHMenu.h"

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


// -------------------- Render algorithm radio support --------------------

static bool
IsRenderAlgorithmCommandId(UINT id) noexcept
{
    switch (id) {
        case IDM_ALG_AUTO:
        case IDM_ALG_CPU_1_32_HDR:
        case IDM_ALG_CPU_1_32_PERTURB_BLAV2_HDR:
        case IDM_ALG_CPU_1_32_PERTURB_BLA_HDR:
        case IDM_ALG_CPU_1_32_PERTURB_RC_BLAV2_HDR:
        case IDM_ALG_CPU_1_64:
        case IDM_ALG_CPU_1_64_HDR:
        case IDM_ALG_CPU_1_64_PERTURB_BLA:
        case IDM_ALG_CPU_1_64_PERTURB_BLAV2_HDR:
        case IDM_ALG_CPU_1_64_PERTURB_BLA_HDR:
        case IDM_ALG_CPU_1_64_PERTURB_RC_BLAV2_HDR:
        case IDM_ALG_CPU_HIGH:
        case IDM_ALG_GPU_1_32:
        case IDM_ALG_GPU_1_32_PERTURB_LAV2:
        case IDM_ALG_GPU_1_32_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_1_32_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_1_32_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_1_32_PERTURB_RC_LAV2_PO:
        case IDM_ALG_GPU_1_32_PERTURB_SCALED:
        case IDM_ALG_GPU_1_64:
        case IDM_ALG_GPU_1_64_PERTURB_BLA:
        case IDM_ALG_GPU_1_64_PERTURB_LAV2:
        case IDM_ALG_GPU_1_64_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_1_64_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_1_64_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_1_64_PERTURB_RC_LAV2_PO:
        case IDM_ALG_GPU_2X32_HDR:
        case IDM_ALG_GPU_2_32:
        case IDM_ALG_GPU_2_32_PERTURB_LAV2:
        case IDM_ALG_GPU_2_32_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_2_32_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_2_32_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_2_32_PERTURB_RC_LAV2_PO:
        case IDM_ALG_GPU_2_32_PERTURB_SCALED:
        case IDM_ALG_GPU_2_64:
        case IDM_ALG_GPU_4_32:
        case IDM_ALG_GPU_4_64:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_HDR_2X32_PERTURB_RC_LAV2_PO:
        case IDM_ALG_GPU_HDR_32_PERTURB_BLA:
        case IDM_ALG_GPU_HDR_32_PERTURB_LAV2:
        case IDM_ALG_GPU_HDR_32_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_HDR_32_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_HDR_32_PERTURB_RC_LAV2_PO:
        case IDM_ALG_GPU_HDR_32_PERTURB_SCALED:
        case IDM_ALG_GPU_HDR_64_PERTURB_BLA:
        case IDM_ALG_GPU_HDR_64_PERTURB_LAV2:
        case IDM_ALG_GPU_HDR_64_PERTURB_LAV2_LAO:
        case IDM_ALG_GPU_HDR_64_PERTURB_LAV2_PO:
        case IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2:
        case IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_LAO:
        case IDM_ALG_GPU_HDR_64_PERTURB_RC_LAV2_PO:
            return true;
        default:
            return false;
    }
}

static UINT g_currentRenderAlgorithmId = IDM_ALG_AUTO;

void
DynamicPopupMenu::SetCurrentRenderAlgorithmId(UINT id) noexcept
{
    if (IsRenderAlgorithmCommandId(id))
        g_currentRenderAlgorithmId = id;
}

UINT
DynamicPopupMenu::GetCurrentRenderAlgorithmId() noexcept
{
    if (g_currentRenderAlgorithmId == 0)
        g_currentRenderAlgorithmId = IDM_ALG_AUTO;
    return g_currentRenderAlgorithmId;
}

static void
ApplyRenderAlgorithmRadioChecksRecursive(HMENU menu, UINT checkedId) noexcept
{
    const int count = ::GetMenuItemCount(menu);
    if (count <= 0)
        return;

    for (int i = 0; i < count; ++i) {
        MENUITEMINFOW mii{};
        mii.cbSize = sizeof(mii);
        mii.fMask = MIIM_ID | MIIM_SUBMENU | MIIM_FTYPE;
        if (!::GetMenuItemInfoW(menu, static_cast<UINT>(i), TRUE /*by position*/, &mii))
            continue;

        const UINT id = mii.wID;
        if (IsRenderAlgorithmCommandId(id)) {
            // Ensure radio-check styling.
            if ((mii.fType & MFT_RADIOCHECK) == 0) {
                MENUITEMINFOW miitype{};
                miitype.cbSize = sizeof(miitype);
                miitype.fMask = MIIM_FTYPE;
                miitype.fType = mii.fType | MFT_RADIOCHECK;
                (void)::SetMenuItemInfoW(menu, static_cast<UINT>(i), TRUE /*by position*/, &miitype);
            }

            (void)::CheckMenuItem(menu,
                                  id,
                                  MF_BYCOMMAND |
                                      ((id == checkedId) ? MF_CHECKED : MF_UNCHECKED));
        }

        if (mii.hSubMenu)
            ApplyRenderAlgorithmRadioChecksRecursive(mii.hSubMenu, checkedId);
    }
}

void
DynamicPopupMenu::ApplyRenderAlgorithmRadioChecks(HMENU menuRoot, UINT checkedId) noexcept
{
    if (!menuRoot)
        return;

    if (!IsRenderAlgorithmCommandId(checkedId))
        checkedId = GetCurrentRenderAlgorithmId();

    ApplyRenderAlgorithmRadioChecksRecursive(menuRoot, checkedId);
}

static UINT
GetMenuItemCountSafe(HMENU menu) noexcept
{
    const int c = ::GetMenuItemCount(menu);
    return (c < 0) ? 0u : static_cast<UINT>(c);
}

static UINT
MapEnabledStateMFToMFS(UINT stateFlags) noexcept
{
    // Your tree uses MF_GRAYED / MF_ENABLED.
    // MENUITEMINFO expects MFS_DISABLED / MFS_ENABLED.
    if ((stateFlags & MF_GRAYED) || (stateFlags & MF_DISABLED)) {
        return MFS_DISABLED;
    }
    return MFS_ENABLED;
}

bool
DynamicPopupMenu::InsertSeparatorAtEnd(HMENU menu)
{
    MENUITEMINFOW mii{};
    mii.cbSize = sizeof(mii);
    mii.fMask = MIIM_FTYPE;
    mii.fType = MFT_SEPARATOR;

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*fByPosition*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertItemAtEnd(HMENU menu, const Node &n)
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
    const bool isAlg = IsRenderAlgorithmCommandId(n.id);
    if (n.radio || isAlg) {
        mii.fType |= MFT_RADIOCHECK;
    }
    if (n.ownerDraw) {
        mii.fType |= MFT_OWNERDRAW;
    }

    mii.wID = n.id;

    UINT state = MapEnabledStateMFToMFS(n.stateFlags);
    const bool checked = isAlg ? (n.id == DynamicPopupMenu::GetCurrentRenderAlgorithmId()) : n.checked;
    if (checked)
        state |= MFS_CHECKED;
    if (n.isDefault)
        state |= MFS_DEFAULT;
    mii.fState = state;

    // InsertMenuItemW uses a non-const pointer type; Windows won't modify it on insert.
    mii.dwTypeData = const_cast<wchar_t *>(n.text.data());
    mii.cch = static_cast<UINT>(n.text.size());

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*fByPosition*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertPopupAtEnd(HMENU menu, const Node &n, HMENU popup)
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

    UINT state = MapEnabledStateMFToMFS(n.stateFlags);
    if (n.isDefault)
        state |= MFS_DEFAULT;
    mii.fState = state;

    mii.hSubMenu = popup;

    mii.dwTypeData = const_cast<wchar_t *>(n.text.data());
    mii.cch = static_cast<UINT>(n.text.size());

    const UINT pos = GetMenuItemCountSafe(menu);
    return ::InsertMenuItemW(menu, pos, TRUE /*fByPosition*/, &mii) != FALSE;
}

bool
DynamicPopupMenu::InsertNodeAtEnd(HMENU menu, const Node &n)
{
    switch (n.kind) {
        case Kind::Separator:
            return InsertSeparatorAtEnd(menu);

        case Kind::Item:
            return InsertItemAtEnd(menu, n);

        case Kind::Popup: {
            HMENU sub = ::CreatePopupMenu();
            if (!sub)
                return false;

            if (!InsertPopupAtEnd(menu, n, sub)) {
                ::DestroyMenu(sub);
                return false;
            }

            if (!BuildMenuTree(sub, n.kids))
                return false;

            return true;
        }

        default:
            return false;
    }
}

bool
DynamicPopupMenu::BuildMenuTree(HMENU parent, std::initializer_list<Node> nodes)
{
    for (const Node &n : nodes) {
        if (!InsertNodeAtEnd(parent, n))
            return false;
    }
    return true;
}

UniqueHMenu
DynamicPopupMenu::Create()
{
    UniqueHMenu root(::CreateMenu());
    if (!root)
        return {};

    HMENU popup = ::CreatePopupMenu();
    if (!popup)
        return {};

    // Top-level "POPUP" item.
    // NOTE: We must provide values for ALL Node fields we care about;
    // extra fields can be value-initialized.
    Node top = {Kind::Popup,
                L"POPUP",
                0,
                Enabled(),
                {},

                // extended fields (default):
                false,
                false,
                false,
                false,
                nullptr,
                0};

    if (!InsertPopupAtEnd(root.get(), top, popup)) {
        ::DestroyMenu(popup);
        return {};
    }

    if (!BuildPopupContents(popup)) {
        // Destroying root destroys attached submenus too.
        return {};
    }

    // Apply mutually-exclusive radio checks for render algorithms.
    ApplyRenderAlgorithmRadioChecks(popup, GetCurrentRenderAlgorithmId());

    return root;
}

bool
DynamicPopupMenu::BuildPopupContents(HMENU popup)
{
#include "DynamicPopupTreeDef.h"

    return BuildMenuTree(popup,
                         std::initializer_list<Node>(menu, menu + (sizeof(menu) / sizeof(menu[0]))));
}

} // namespace FractalShark
