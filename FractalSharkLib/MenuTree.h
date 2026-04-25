//
// MenuTree.h - portable, backend-agnostic menu data tree.
//
// This header defines the data layer of the application's context menu so
// both the Win32 GUI and the Linux GUI can walk the same tree. It is
// intentionally free of any windowing-system dependency: command IDs are
// plain uint32_t, optional Win32-only fields (HBITMAP/ULONG_PTR) are stored
// as void*/uintptr_t and treated as opaque. The Linux GUI ignores them.
//
// Lifetime invariants (carried over from the original DynamicPopupMenu
// design):
//
//   - Node::Kids is std::span<const Node>. It is non-owning and assumes the
//     referenced child arrays have stable lifetime (typically static const
//     Node[]).
//
//   - Avoid storing std::initializer_list inside Node; braced child lists
//     create temporary arrays whose backing storage can go away before the
//     consumer recurses. The FS_POPUP macros in MenuTreeDef.h hoist children
//     into hidden static const Node arrays to keep storage stable for the
//     lifetime of the program.
//
// The Win32 builder (FractalSharkGUILib/DynamicPopupMenu.{h,cpp}) and the
// Linux builder (FractalSharkGuiLinux, future) both consume this same tree.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace FractalShark::Menu {

enum class RadioGroup : uint16_t {
    None = 0,

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
    NRInnerLoopBackend,
};

enum class Rule : uint16_t {
    Always = 0,

    EnableIfGpuActive,
    EnableIfCpuActive,
    EnableIfPerturbationAvailable,
    EnableIfPaletteRotationSupported,
    EnableIfNRCheckpointExists,
};

struct IMenuState {
    virtual ~IMenuState() = default;

    virtual bool IsEnabled(Rule rule) const noexcept = 0;
    virtual bool IsChecked(uint32_t commandId) const noexcept = 0;
    virtual uint32_t GetRadioSelection(RadioGroup group) const noexcept = 0;

    virtual uint32_t
    GetPopupAdornmentCommandId(RadioGroup /*group*/) const noexcept
    {
        return 0;
    }

    virtual std::wstring_view
    GetCommandLabel(uint32_t /*commandId*/) const noexcept
    {
        return {};
    }
};

enum class Kind : uint8_t { Item, Separator, Popup };

enum class CheckKind : uint8_t {
    None = 0,
    Toggle,
    Radio,
};

struct Node final {
    Kind kind;
    std::wstring_view text;
    uint32_t id;
    Rule enableRule;

    std::span<const Node> kids;
    CheckKind checkKind;
    RadioGroup radioGroup;
    bool isDefault;
    bool ownerDraw;
    void *hbmpItem;          // Win32 HBITMAP cast to void *; nullptr on Linux
    uintptr_t itemData;      // Win32 ULONG_PTR; 0 on Linux
    RadioGroup adornGroup;
};

constexpr Node
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

constexpr Node
Item(std::wstring_view text, uint32_t id, Rule enableRule = Rule::Always) noexcept
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

constexpr Node
Toggle(std::wstring_view text, uint32_t id, Rule enableRule = Rule::Always) noexcept
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

constexpr Node
Radio(std::wstring_view text,
      uint32_t id,
      RadioGroup group,
      Rule enableRule = Rule::Always) noexcept
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

constexpr Node
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

} // namespace FractalShark::Menu
