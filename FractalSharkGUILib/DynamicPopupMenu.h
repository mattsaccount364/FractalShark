//
// DynamicPopupMenu - Win32 builder over the portable FractalShark::Menu tree.
//
// The menu *data* (Node tree, Rule, RadioGroup, IMenuState) lives in
// FractalSharkLib/MenuTree.h so that both the Win32 GUI and the Linux GUI
// can walk the same definition. This header keeps the existing public API
// (FractalShark::DynamicPopupMenu::Create / GetPopup / BuildPopupContents)
// for the Win32 GUI; types are re-exported as nested aliases so call sites
// like `FractalShark::DynamicPopupMenu::IMenuState` and
// `DynamicPopupMenu::RadioGroup` keep compiling unchanged.
//
// What this builder does:
//   1. Create() creates a root HMENU, attaches a single popup submenu, then
//      recursively builds items/submenus from the Node tree (defined in
//      MenuTreeDef.h, included inside BuildPopupContents).
//   2. Each item's enabled/checked state is computed on the fly from the
//      caller-supplied IMenuState.
//
// Once a submenu HMENU is successfully inserted into its parent
// (MIIM_SUBMENU / hSubMenu), ownership transfers to the parent menu; the
// builder must not destroy that submenu handle unless it first detaches the
// parent item (or abandons and destroys the entire root menu).
//

#pragma once

#include <windows.h>

#include "AlgCmds.h"
#include "MenuTree.h"

namespace FractalShark {

class UniqueHMenu;

class DynamicPopupMenu final {
public:
    // Re-export portable types so existing callers can keep using
    // DynamicPopupMenu::RadioGroup / DynamicPopupMenu::Rule /
    // DynamicPopupMenu::IMenuState unchanged.
    using RadioGroup = Menu::RadioGroup;
    using Rule = Menu::Rule;
    using IMenuState = Menu::IMenuState;
    using Node = Menu::Node;

    // Build a new root menu containing the single "POPUP" submenu.
    // Call this each time the context menu is shown so dynamic state is fresh.
    static UniqueHMenu Create(const IMenuState &state);

    static HMENU GetPopup(HMENU rootMenu) noexcept;

    static bool BuildPopupContents(HMENU popup, const IMenuState &state);

private:
    using Kind = Menu::Kind;
    using CheckKind = Menu::CheckKind;

    static bool BuildMenuTree(HMENU parent, std::span<const Node> nodes, const IMenuState &state);
    static bool InsertNodeAtEnd(HMENU menu, const Node &n, const IMenuState &state);

    static bool InsertSeparatorAtEnd(HMENU menu);
    static bool InsertItemAtEnd(HMENU menu, const Node &n, const IMenuState &state);
    static bool InsertPopupAtEnd(HMENU menu, const Node &n, HMENU popup, const IMenuState &state);

    static UINT GetEnabledState(const Node &n, const IMenuState &state) noexcept;
    static bool IsCheckedNow(const Node &n, const IMenuState &state) noexcept;

    static void BuildPopupLabel(const Node &n,
                                const IMenuState &state,
                                /*out*/ wchar_t *buf,
                                size_t bufCount) noexcept;

    static UINT GetMenuItemCountSafe(HMENU menu) noexcept;
    static UINT MapEnabledStateToMFS(UINT mfEnabled) noexcept;
};

} // namespace FractalShark
