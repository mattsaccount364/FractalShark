//
// How DynamicPopupMenu is structured and built
//
// This module splits the menu into two parts:
//
//   1) A mostly-static, data-only tree of `Node` objects.
//      - `Node` is an aggregate so the menu can be expressed as a big constant
//        definition (see DynamicPopupTreeDef.h).
//      - Popups store their children as `std::span<const Node>`.
//        This is a *non-owning view*; it assumes the referenced child arrays have
//        stable lifetime (typically `static const Node[]`).
//
//   2) A small builder that turns that tree into a real Win32 HMENU each time the
//      context menu is shown.
//      - Create(state) creates a root HMENU, attaches a single popup submenu,
//        then recursively builds items/submenus from the Node tree.
//      - Each item’s enabled/checked state is computed on the fly from `IMenuState`:
//          * Rule -> enabled/disabled
//          * Toggle -> checked via IsChecked(commandId)
//          * Radio  -> checked if commandId == GetRadioSelection(group)
//      - Optional popup "adornment" appends the current selection label to a
//        submenu caption (e.g. "GPU Antialiasing (4x)").
//
// Key invariants:
//
//   - `Node::kids` is a span, so it never owns memory; it only points at existing
//     static storage. Avoid storing `std::initializer_list` inside `Node`, because
//     braced child lists create temporary arrays whose backing storage can go away
//     before the builder recurses.
//
//   - Once a submenu HMENU is successfully inserted into its parent
//     (MIIM_SUBMENU / hSubMenu), ownership transfers to the parent menu; the
//     builder must not destroy that submenu handle unless it first detaches the
//     parent item (or abandons and destroys the entire root menu).
//

#pragma once

#include <windows.h>

#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <span>

#include "AlgCmds.h"

namespace FractalShark {

class UniqueHMenu;

class DynamicPopupMenu final {
public:
    struct IMenuState;

    // Build a new root menu containing the single "POPUP" submenu.
    // Call this each time you want to show the context menu so dynamic state is fresh.
    static UniqueHMenu Create(const IMenuState &state);

    static HMENU GetPopup(HMENU rootMenu) noexcept;

    // -------------------------------------------------------------------------
    // Public extension points
    // -------------------------------------------------------------------------

    // Radio groups: mutually exclusive selection across a set of command ids.
    // Make these match your settings model (AA, perturb mode, palette bits, etc).
    enum class RadioGroup : uint16_t {
        None = 0,

        // Examples (extend as needed):
        RenderAlgorithm,
        GpuAntialiasing,
        IterationPrecision,
        LaThreading,
        PerturbationMode,
        PaletteType,
        PaletteBitDepth,
        MemoryAutosave,
        MemoryLimit,
        IterationsWidth,
    };

    // Simple enable/disable rules referenced by items.
    // Implemented by IMenuState.
    enum class Rule : uint16_t {
        Always = 0,

        // Examples (extend as needed):
        EnableIfGpuActive,
        EnableIfCpuActive,
        EnableIfPerturbationAvailable,
        EnableIfPaletteRotationSupported,
    };

    // Settings provider for menu state.
    // Your app owns the implementation (likely backed by your main Settings struct).
    struct IMenuState {
        virtual ~IMenuState() = default;

        // Enabled/disabled state for a rule.
        virtual bool IsEnabled(Rule rule) const noexcept = 0;

        // Independent checkbox state (toggles that are NOT mutually exclusive).
        // For non-toggle commands, return false.
        virtual bool IsChecked(UINT commandId) const noexcept = 0;

        // The selected command id for a given radio group.
        // Return 0 if "no selection" (menu will show none checked).
        virtual UINT GetRadioSelection(RadioGroup group) const noexcept = 0;

        // Optional: allow parent popup labels to reflect current selection.
        // Return 0 to mean "no adornment".
        virtual UINT
        GetPopupAdornmentCommandId(RadioGroup /*group*/) const noexcept
        {
            return 0;
        }

        // Optional: resolve a commandId to a short label for adornment.
        // If you don't override this, adornment is omitted.
        virtual std::wstring_view
        GetCommandLabel(UINT /*commandId*/) const noexcept
        {
            return {};
        }
    };
        
    static bool BuildPopupContents(HMENU popup, const IMenuState &state);

private:
    enum class Kind : uint8_t { Item, Separator, Popup };

    // Menu item "check behavior" for items of Kind::Item.
    enum class CheckKind : uint8_t {
        None = 0, // plain command (no check)
        Toggle,   // independent checkbox (MFS_CHECKED)
        Radio,    // radio selection within a group (MFT_RADIOCHECK + MFS_CHECKED)
    };

    // IMPORTANT:
    //  - This is still aggregate-initializable in a clean way.
    struct Node final {
        Kind kind;
        std::wstring_view text;
        UINT id;
        Rule enableRule;

        std::span<const Node> kids;
        CheckKind checkKind;
        RadioGroup radioGroup;
        bool isDefault;
        bool ownerDraw;
        HBITMAP hbmpItem;
        ULONG_PTR itemData;
        RadioGroup adornGroup;
    };


    // Convenience initializers so your tree reads nicely.
    static constexpr Node
    Sep() noexcept
    {
        return Node{Kind::Separator,
                    L"",
                    0u,
                    Rule::Always,
                    {},
                    CheckKind::None,
                    RadioGroup::None,
                    false,
                    false,
                    nullptr,
                    0u,
                    RadioGroup::None};
    }

    static constexpr Node
    Item(std::wstring_view text, UINT id, Rule enableRule = Rule::Always) noexcept
    {
        return Node{Kind::Item,
                    text,
                    id,
                    enableRule,
                    {},
                    CheckKind::None,
                    RadioGroup::None,
                    false,
                    false,
                    nullptr,
                    0u,
                    RadioGroup::None};
    }

    static constexpr Node
    Toggle(std::wstring_view text, UINT id, Rule enableRule = Rule::Always) noexcept
    {
        return Node{Kind::Item,
                    text,
                    id,
                    enableRule,
                    {},
                    CheckKind::Toggle,
                    RadioGroup::None,
                    false,
                    false,
                    nullptr,
                    0u,
                    RadioGroup::None};
    }

    static constexpr Node
    Radio(std::wstring_view text, UINT id, RadioGroup group, Rule enableRule = Rule::Always) noexcept
    {
        return Node{Kind::Item,
                    text,
                    id,
                    enableRule,
                    {},
                    CheckKind::Radio,
                    group,
                    false,
                    false,
                    nullptr,
                    0u,
                    RadioGroup::None};
    }

    static constexpr Node
    Popup(std::wstring_view text,
          std::span<const Node> kids,
          Rule enableRule = Rule::Always,
          RadioGroup adornGroup = RadioGroup::None) noexcept
    {
        return Node{Kind::Popup,
                    text,
                    0u,
                    enableRule,
                    kids,
                    CheckKind::None,
                    RadioGroup::None,
                    false,
                    false,
                    nullptr,
                    0u,
                    adornGroup};
    }


    // -------------------------------------------------------------------------
    // Build helpers
    // -------------------------------------------------------------------------
    static bool BuildMenuTree(HMENU parent, std::span<const Node> nodes, const IMenuState &state);
    static bool InsertNodeAtEnd(HMENU menu, const Node &n, const IMenuState &state);

    static bool InsertSeparatorAtEnd(HMENU menu);
    static bool InsertItemAtEnd(HMENU menu, const Node &n, const IMenuState &state);
    static bool InsertPopupAtEnd(HMENU menu, const Node &n, HMENU popup, const IMenuState &state);

    // Maps Rule -> MF/MFS enabled state. (We keep this internal; tree uses Rule.)
    static UINT GetEnabledState(const Node &n, const IMenuState &state) noexcept;

    // Derive checked state from IMenuState for Toggle/Radio.
    static bool IsCheckedNow(const Node &n, const IMenuState &state) noexcept;

    // Optional: decorate popup labels with current selection, if adornGroup != None.
    // Example: "GPU Antialiasing (4x)".
    static void BuildPopupLabel(const Node &n,
                                const IMenuState &state,
                                /*out*/ wchar_t *buf,
                                size_t bufCount) noexcept;

    static UINT GetMenuItemCountSafe(HMENU menu) noexcept;
    static UINT MapEnabledStateToMFS(UINT mfEnabled) noexcept;

    // -------------------------------------------------------------------------
    // Menu definition
    // -------------------------------------------------------------------------
    // Kept out-of-class in DynamicPopupMenu.cpp:
    //   extern const Node g_menu[];
    //   extern const size_t g_menuCount;
    //
    // Create() will include/compile that definition unit and build from it.
};

} // namespace FractalShark
