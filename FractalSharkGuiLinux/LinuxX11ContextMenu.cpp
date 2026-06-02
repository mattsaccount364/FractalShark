// LinuxX11ContextMenu.cpp

#include "LinuxX11ContextMenu.h"

#include "CommandCatalog.h"
#include "MenuTree.h"

#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace FractalShark::Linux {

namespace {

using namespace FractalShark::Menu;

constexpr int kScreenMargin = 4;
constexpr int kMenuPadding = 3;
constexpr int kItemHeight = 22;
constexpr int kSeparatorHeight = 9;
constexpr int kCheckAreaWidth = 20;
constexpr int kArrowAreaWidth = 18;
constexpr int kMinimumWidth = 120;
constexpr int kMaximumWidth = 700;
constexpr int kScrollStep = kItemHeight * 3;

std::span<const Node>
GetMenuNodes()
{
#include "MenuTreeDef.h"

    return {menu, sizeof(menu) / sizeof(menu[0])};
}

std::string
WideToUtf8(std::wstring_view in)
{
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        uint32_t cp = static_cast<uint32_t>(in[i]);
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < in.size()) {
            uint32_t low = static_cast<uint32_t>(in[i + 1]);
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                ++i;
            }
        }
        if (cp < 0x80) {
            out.push_back(char(cp));
        } else if (cp < 0x800) {
            out.push_back(char(0xC0 | (cp >> 6)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(char(0xE0 | (cp >> 12)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(char(0xF0 | (cp >> 18)));
            out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(char(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

std::string
StripMenuMnemonics(std::string label)
{
    std::string out;
    out.reserve(label.size());
    for (std::size_t i = 0; i < label.size(); ++i) {
        if (label[i] == '&' && i + 1 < label.size()) {
            if (label[i + 1] == '&') {
                out.push_back('&');
                ++i;
            }
            continue;
        }
        out.push_back(label[i]);
    }
    return out;
}

unsigned long
AllocateNamedColor(Display *display, int screen, const char *name, unsigned long fallback)
{
    XColor color{};
    XColor exact{};
    if (XAllocNamedColor(display, DefaultColormap(display, screen), name, &color, &exact)) {
        return color.pixel;
    }
    return fallback;
}

} // namespace

struct X11ContextMenu::Impl {
    struct Colors {
        unsigned long Background;
        unsigned long Text;
        unsigned long DisabledText;
        unsigned long Highlight;
        unsigned long HighlightText;
        unsigned long Border;
    };

    struct Row {
        const Node *MenuNode;
        std::string Label;
        int Y;
        int Height;
    };

    struct Level {
        Window MenuWindow = 0;
        std::span<const Node> Nodes;
        std::vector<Row> Rows;
        int X = 0;
        int Y = 0;
        int Width = 0;
        int Height = 0;
        int ContentHeight = 0;
        int ScrollOffset = 0;
        int SelectedRow = -1;
    };

    Display *DisplayHandle;
    int Screen;
    Window Owner;
    const Menu::IMenuState *State;
    ExecuteCommandHost *Host;
    Window Root;
    GC GraphicsContext = nullptr;
    XFontSet FontSet = nullptr;
    XFontStruct *FallbackFont = nullptr;
    Colors Palette;
    Atom WmWindowType;
    Atom WmWindowTypePopupMenu;
    std::vector<Level> Levels;

    Impl(Display *display,
         int screen,
         Window owner,
         const Menu::IMenuState *state,
         ExecuteCommandHost *host)
        : DisplayHandle(display), Screen(screen), Owner(owner), State(state), Host(host),
          Root(RootWindow(display, screen)),
          Palette{AllocateNamedColor(display, screen, "#f0f0f0", WhitePixel(display, screen)),
                  AllocateNamedColor(display, screen, "#101010", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#777777", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#3875d6", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#ffffff", WhitePixel(display, screen)),
                  AllocateNamedColor(display, screen, "#606060", BlackPixel(display, screen))},
          WmWindowType(XInternAtom(display, "_NET_WM_WINDOW_TYPE", False)),
          WmWindowTypePopupMenu(XInternAtom(display, "_NET_WM_WINDOW_TYPE_POPUP_MENU", False))
    {
        GraphicsContext = XCreateGC(DisplayHandle, Root, 0, nullptr);

        std::setlocale(LC_CTYPE, "");
        char **missingCharsets = nullptr;
        int missingCharsetCount = 0;
        char *defaultString = nullptr;
        FontSet = XCreateFontSet(DisplayHandle,
                                 "-misc-fixed-*-*-*-*-14-*-*-*-*-*-*-*",
                                 &missingCharsets,
                                 &missingCharsetCount,
                                 &defaultString);
        if (missingCharsets) {
            XFreeStringList(missingCharsets);
        }
        if (!FontSet) {
            FallbackFont = XLoadQueryFont(DisplayHandle, "fixed");
            if (FallbackFont) {
                XSetFont(DisplayHandle, GraphicsContext, FallbackFont->fid);
            }
        }
    }

    ~Impl()
    {
        Close();
        if (FontSet) {
            XFreeFontSet(DisplayHandle, FontSet);
        }
        if (FallbackFont) {
            XFreeFont(DisplayHandle, FallbackFont);
        }
        if (GraphicsContext) {
            XFreeGC(DisplayHandle, GraphicsContext);
        }
    }

    int
    TextWidth(const std::string &text) const
    {
        if (FontSet) {
            XRectangle ink{};
            XRectangle logical{};
            Xutf8TextExtents(FontSet, text.data(), static_cast<int>(text.size()), &ink, &logical);
            return logical.width;
        }
        if (FallbackFont) {
            return XTextWidth(FallbackFont, text.data(), static_cast<int>(text.size()));
        }
        return static_cast<int>(text.size()) * 8;
    }

    int
    TextBaseline(int rowY, int rowHeight) const
    {
        if (FontSet) {
            const XFontSetExtents *extents = XExtentsOfFontSet(FontSet);
            const XRectangle &logical = extents->max_logical_extent;
            return rowY + (rowHeight - logical.height) / 2 - logical.y;
        }
        if (FallbackFont) {
            const int textHeight = FallbackFont->ascent + FallbackFont->descent;
            return rowY + (rowHeight - textHeight) / 2 + FallbackFont->ascent;
        }
        return rowY + rowHeight - 5;
    }

    void
    DrawText(Window window, int x, int baseline, const std::string &text, unsigned long color)
    {
        XSetForeground(DisplayHandle, GraphicsContext, color);
        if (FontSet) {
            Xutf8DrawString(DisplayHandle,
                            window,
                            FontSet,
                            GraphicsContext,
                            x,
                            baseline,
                            text.data(),
                            static_cast<int>(text.size()));
        } else {
            XDrawString(DisplayHandle,
                        window,
                        GraphicsContext,
                        x,
                        baseline,
                        text.data(),
                        static_cast<int>(text.size()));
        }
    }

    bool
    IsEnabled(const Node &node) const
    {
        return !State || State->IsEnabled(node.enableRule);
    }

    bool
    IsChecked(const Node &node) const
    {
        if (!State) {
            return false;
        }
        if (node.checkKind == CheckKind::Toggle) {
            return State->IsChecked(node.id);
        }
        if (node.checkKind == CheckKind::Radio) {
            return State->GetRadioSelection(node.radioGroup) == node.id;
        }
        return false;
    }

    bool
    IsSelectable(const Row &row) const
    {
        return row.MenuNode->kind != Kind::Separator && IsEnabled(*row.MenuNode);
    }

    void
    DrawCheck(const Level &level, int rowY, int rowHeight, bool radio, unsigned long color)
    {
        const int centerX = kMenuPadding + kCheckAreaWidth / 2;
        const int centerY = rowY + rowHeight / 2;
        XSetForeground(DisplayHandle, GraphicsContext, color);
        if (radio) {
            XFillArc(DisplayHandle,
                     level.MenuWindow,
                     GraphicsContext,
                     centerX - 3,
                     centerY - 3,
                     7,
                     7,
                     0,
                     360 * 64);
            return;
        }
        XDrawLine(DisplayHandle,
                  level.MenuWindow,
                  GraphicsContext,
                  centerX - 5,
                  centerY,
                  centerX - 1,
                  centerY + 4);
        XDrawLine(DisplayHandle,
                  level.MenuWindow,
                  GraphicsContext,
                  centerX - 1,
                  centerY + 4,
                  centerX + 6,
                  centerY - 5);
    }

    void
    DrawSubmenuArrow(const Level &level, int rowY, int rowHeight, unsigned long color)
    {
        const int centerX = level.Width - kArrowAreaWidth / 2;
        const int centerY = rowY + rowHeight / 2;
        XSetForeground(DisplayHandle, GraphicsContext, color);
        XDrawLine(DisplayHandle,
                  level.MenuWindow,
                  GraphicsContext,
                  centerX - 3,
                  centerY - 4,
                  centerX + 2,
                  centerY);
        XDrawLine(DisplayHandle,
                  level.MenuWindow,
                  GraphicsContext,
                  centerX + 2,
                  centerY,
                  centerX - 3,
                  centerY + 4);
    }

    void
    DrawScrollIndicator(const Level &level, bool atTop)
    {
        const int centerX = level.Width / 2;
        const int centerY = atTop ? 4 : level.Height - 5;
        XSetForeground(DisplayHandle, GraphicsContext, Palette.Text);
        if (atTop) {
            XDrawLine(DisplayHandle,
                      level.MenuWindow,
                      GraphicsContext,
                      centerX - 4,
                      centerY + 2,
                      centerX,
                      centerY - 2);
            XDrawLine(DisplayHandle,
                      level.MenuWindow,
                      GraphicsContext,
                      centerX,
                      centerY - 2,
                      centerX + 4,
                      centerY + 2);
        } else {
            XDrawLine(DisplayHandle,
                      level.MenuWindow,
                      GraphicsContext,
                      centerX - 4,
                      centerY - 2,
                      centerX,
                      centerY + 2);
            XDrawLine(DisplayHandle,
                      level.MenuWindow,
                      GraphicsContext,
                      centerX,
                      centerY + 2,
                      centerX + 4,
                      centerY - 2);
        }
    }

    void
    DrawLevel(const Level &level)
    {
        XSetWindowBackground(DisplayHandle, level.MenuWindow, Palette.Background);
        XClearWindow(DisplayHandle, level.MenuWindow);

        for (std::size_t i = 0; i < level.Rows.size(); ++i) {
            const Row &row = level.Rows[i];
            const int rowY = row.Y - level.ScrollOffset;
            if (rowY + row.Height <= 0 || rowY >= level.Height) {
                continue;
            }

            const Node &node = *row.MenuNode;
            if (node.kind == Kind::Separator) {
                const int y = rowY + row.Height / 2;
                XSetForeground(DisplayHandle, GraphicsContext, Palette.DisabledText);
                XDrawLine(DisplayHandle,
                          level.MenuWindow,
                          GraphicsContext,
                          kMenuPadding,
                          y,
                          level.Width - kMenuPadding,
                          y);
                continue;
            }

            const bool enabled = IsEnabled(node);
            const bool selected = static_cast<int>(i) == level.SelectedRow;
            if (selected) {
                XSetForeground(DisplayHandle, GraphicsContext, Palette.Highlight);
                XFillRectangle(DisplayHandle,
                               level.MenuWindow,
                               GraphicsContext,
                               0,
                               rowY,
                               static_cast<unsigned int>(level.Width),
                               static_cast<unsigned int>(row.Height));
            }

            const unsigned long textColor =
                !enabled ? Palette.DisabledText : (selected ? Palette.HighlightText : Palette.Text);
            if (IsChecked(node)) {
                DrawCheck(level, rowY, row.Height, node.checkKind == CheckKind::Radio, textColor);
            }
            DrawText(level.MenuWindow,
                     kMenuPadding + kCheckAreaWidth,
                     TextBaseline(rowY, row.Height),
                     row.Label,
                     textColor);
            if (node.kind == Kind::Popup) {
                DrawSubmenuArrow(level, rowY, row.Height, textColor);
            }
        }

        if (level.ScrollOffset > 0) {
            DrawScrollIndicator(level, true);
        }
        if (level.ScrollOffset + level.Height < level.ContentHeight) {
            DrawScrollIndicator(level, false);
        }
    }

    void
    DrawAll()
    {
        for (const Level &level : Levels) {
            DrawLevel(level);
        }
        XFlush(DisplayHandle);
    }

    Level
    BuildLevel(std::span<const Node> nodes)
    {
        Level level;
        level.Nodes = nodes;
        level.Width = kMinimumWidth;

        int y = 0;
        for (const Node &node : nodes) {
            const int height = node.kind == Kind::Separator ? kSeparatorHeight : kItemHeight;
            std::string label = StripMenuMnemonics(WideToUtf8(node.text));
            level.Width = std::max(
                level.Width, kMenuPadding * 2 + kCheckAreaWidth + TextWidth(label) + kArrowAreaWidth);
            level.Rows.push_back(Row{&node, std::move(label), y, height});
            y += height;
        }
        level.Width = std::min(level.Width, kMaximumWidth);
        level.ContentHeight = y;
        level.Height =
            std::min(y, std::max(1, DisplayHeight(DisplayHandle, Screen) - 2 * kScreenMargin));
        return level;
    }

    void
    PositionLevel(Level &level, int desiredX, int desiredY, int leftAnchorX = -1)
    {
        const int displayWidth = DisplayWidth(DisplayHandle, Screen);
        const int displayHeight = DisplayHeight(DisplayHandle, Screen);
        if (desiredX + level.Width > displayWidth - kScreenMargin && leftAnchorX >= 0) {
            desiredX = leftAnchorX - level.Width + 1;
        }
        level.X = std::clamp(desiredX,
                             kScreenMargin,
                             std::max(kScreenMargin, displayWidth - level.Width - kScreenMargin));
        level.Y = std::clamp(desiredY,
                             kScreenMargin,
                             std::max(kScreenMargin, displayHeight - level.Height - kScreenMargin));
    }

    bool
    AddLevel(std::span<const Node> nodes, int desiredX, int desiredY, int leftAnchorX = -1)
    {
        Level level = BuildLevel(nodes);
        PositionLevel(level, desiredX, desiredY, leftAnchorX);

        XSetWindowAttributes attributes{};
        attributes.override_redirect = True;
        attributes.save_under = True;
        attributes.background_pixel = Palette.Background;
        attributes.border_pixel = Palette.Border;
        attributes.event_mask = ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                                EnterWindowMask | LeaveWindowMask;

        level.MenuWindow =
            XCreateWindow(DisplayHandle,
                          Root,
                          level.X,
                          level.Y,
                          static_cast<unsigned int>(level.Width),
                          static_cast<unsigned int>(level.Height),
                          1,
                          CopyFromParent,
                          InputOutput,
                          CopyFromParent,
                          CWOverrideRedirect | CWSaveUnder | CWBackPixel | CWBorderPixel | CWEventMask,
                          &attributes);
        if (!level.MenuWindow) {
            return false;
        }

        XChangeProperty(DisplayHandle,
                        level.MenuWindow,
                        WmWindowType,
                        XA_ATOM,
                        32,
                        PropModeReplace,
                        reinterpret_cast<unsigned char *>(&WmWindowTypePopupMenu),
                        1);
        XMapRaised(DisplayHandle, level.MenuWindow);
        Levels.push_back(std::move(level));
        DrawLevel(Levels.back());
        XFlush(DisplayHandle);
        return true;
    }

    void
    HideAndDestroyLevelsFrom(std::size_t firstIndex)
    {
        if (firstIndex >= Levels.size()) {
            return;
        }

        bool unmapped = false;
        for (std::size_t i = Levels.size(); i > firstIndex; --i) {
            XUnmapWindow(DisplayHandle, Levels[i - 1].MenuWindow);
            unmapped = true;
        }
        if (unmapped) {
            XSync(DisplayHandle, False);
        }
        while (Levels.size() > firstIndex) {
            XDestroyWindow(DisplayHandle, Levels.back().MenuWindow);
            Levels.pop_back();
        }
        XFlush(DisplayHandle);
    }

    void
    HideAndDestroyLevelsAfter(std::size_t index)
    {
        HideAndDestroyLevelsFrom(index + 1);
    }

    void
    ReplaceChildLevel(
        std::size_t levelIndex, std::span<const Node> nodes, int desiredX, int desiredY, int leftAnchorX)
    {
        HideAndDestroyLevelsAfter(levelIndex + 1);

        Level level = BuildLevel(nodes);
        PositionLevel(level, desiredX, desiredY, leftAnchorX);
        level.MenuWindow = Levels[levelIndex + 1].MenuWindow;

        XUnmapWindow(DisplayHandle, level.MenuWindow);
        XSync(DisplayHandle, False);
        XMoveResizeWindow(DisplayHandle,
                          level.MenuWindow,
                          level.X,
                          level.Y,
                          static_cast<unsigned int>(level.Width),
                          static_cast<unsigned int>(level.Height));
        Levels[levelIndex + 1] = std::move(level);
        XMapRaised(DisplayHandle, Levels[levelIndex + 1].MenuWindow);
        DrawLevel(Levels[levelIndex + 1]);
        XFlush(DisplayHandle);
    }

    int
    FindLevelAt(int rootX, int rootY) const
    {
        for (int i = static_cast<int>(Levels.size()) - 1; i >= 0; --i) {
            const Level &level = Levels[static_cast<std::size_t>(i)];
            if (rootX >= level.X && rootX < level.X + level.Width && rootY >= level.Y &&
                rootY < level.Y + level.Height) {
                return i;
            }
        }
        return -1;
    }

    int
    FindLevelByWindow(Window window) const
    {
        for (std::size_t i = 0; i < Levels.size(); ++i) {
            if (Levels[i].MenuWindow == window) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    int
    FindRowAt(const Level &level, int rootY) const
    {
        const int contentY = rootY - level.Y + level.ScrollOffset;
        for (std::size_t i = 0; i < level.Rows.size(); ++i) {
            const Row &row = level.Rows[i];
            if (contentY >= row.Y && contentY < row.Y + row.Height) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void
    EnsureSelectedVisible(Level &level)
    {
        if (level.SelectedRow < 0) {
            return;
        }
        const Row &row = level.Rows[static_cast<std::size_t>(level.SelectedRow)];
        if (row.Y < level.ScrollOffset) {
            level.ScrollOffset = row.Y;
        } else if (row.Y + row.Height > level.ScrollOffset + level.Height) {
            level.ScrollOffset = row.Y + row.Height - level.Height;
        }
        level.ScrollOffset =
            std::clamp(level.ScrollOffset, 0, std::max(0, level.ContentHeight - level.Height));
    }

    void
    SelectFirst(std::size_t levelIndex)
    {
        Level &level = Levels[levelIndex];
        for (std::size_t i = 0; i < level.Rows.size(); ++i) {
            if (IsSelectable(level.Rows[i])) {
                level.SelectedRow = static_cast<int>(i);
                EnsureSelectedVisible(level);
                DrawLevel(level);
                return;
            }
        }
    }

    void
    OpenSubmenu(std::size_t levelIndex, int rowIndex, bool selectFirst)
    {
        Level &parent = Levels[levelIndex];
        if (rowIndex < 0 || rowIndex >= static_cast<int>(parent.Rows.size())) {
            HideAndDestroyLevelsAfter(levelIndex);
            return;
        }
        const Row &row = parent.Rows[static_cast<std::size_t>(rowIndex)];
        const Node &node = *row.MenuNode;
        if (node.kind != Kind::Popup || !IsEnabled(node)) {
            HideAndDestroyLevelsAfter(levelIndex);
            return;
        }

        if (Levels.size() > levelIndex + 1 && Levels[levelIndex + 1].Nodes.data() == node.kids.data()) {
            return;
        }

        const int desiredX = parent.X + parent.Width + 1;
        const int desiredY = parent.Y + row.Y - parent.ScrollOffset;
        if (Levels.size() > levelIndex + 1) {
            ReplaceChildLevel(levelIndex, node.kids, desiredX, desiredY, parent.X);
        } else if (!AddLevel(node.kids, desiredX, desiredY, parent.X)) {
            return;
        }
        if (selectFirst) {
            SelectFirst(levelIndex + 1);
        }
    }

    void
    UpdateHover(int rootX, int rootY)
    {
        const int levelIndex = FindLevelAt(rootX, rootY);
        if (levelIndex < 0) {
            return;
        }
        Level &level = Levels[static_cast<std::size_t>(levelIndex)];
        const int rowIndex = FindRowAt(level, rootY);
        if (level.SelectedRow != rowIndex) {
            level.SelectedRow = rowIndex;
            DrawLevel(level);
        }
        OpenSubmenu(static_cast<std::size_t>(levelIndex), rowIndex, false);
    }

    void
    Scroll(int rootX, int rootY, int delta)
    {
        const int levelIndex = FindLevelAt(rootX, rootY);
        if (levelIndex < 0) {
            return;
        }
        Level &level = Levels[static_cast<std::size_t>(levelIndex)];
        level.ScrollOffset =
            std::clamp(level.ScrollOffset + delta, 0, std::max(0, level.ContentHeight - level.Height));
        HideAndDestroyLevelsAfter(static_cast<std::size_t>(levelIndex));
        DrawLevel(level);
        XFlush(DisplayHandle);
    }

    void
    MoveSelection(int direction)
    {
        if (Levels.empty()) {
            return;
        }
        Level &level = Levels.back();
        const int rowCount = static_cast<int>(level.Rows.size());
        if (rowCount == 0) {
            return;
        }
        int rowIndex = level.SelectedRow;
        if (rowIndex < 0 && direction < 0) {
            rowIndex = 0;
        }
        for (int i = 0; i < rowCount; ++i) {
            rowIndex = (rowIndex + direction + rowCount) % rowCount;
            if (IsSelectable(level.Rows[static_cast<std::size_t>(rowIndex)])) {
                level.SelectedRow = rowIndex;
                EnsureSelectedVisible(level);
                DrawLevel(level);
                OpenSubmenu(Levels.size() - 1, rowIndex, false);
                XFlush(DisplayHandle);
                return;
            }
        }
    }

    void
    Activate(std::size_t levelIndex, int rowIndex)
    {
        if (levelIndex >= Levels.size()) {
            return;
        }
        const Level &level = Levels[levelIndex];
        if (rowIndex < 0 || rowIndex >= static_cast<int>(level.Rows.size())) {
            return;
        }
        const Node &node = *level.Rows[static_cast<std::size_t>(rowIndex)].MenuNode;
        if (!IsEnabled(node)) {
            return;
        }
        if (node.kind == Kind::Popup) {
            OpenSubmenu(levelIndex, rowIndex, true);
            return;
        }
        if (node.kind == Kind::Separator) {
            return;
        }

        const uint32_t commandId = node.id;
        Close();
        if (Host) {
            ExecuteCommand(CommandFromIdm(commandId), *Host);
        }
    }

    void
    HandleKeyPress(const XKeyEvent &event)
    {
        const KeySym key = XLookupKeysym(const_cast<XKeyEvent *>(&event), 0);
        switch (key) {
            case XK_Escape:
                Close();
                break;
            case XK_Up:
                MoveSelection(-1);
                break;
            case XK_Down:
                MoveSelection(1);
                break;
            case XK_Right:
                if (!Levels.empty()) {
                    Activate(Levels.size() - 1, Levels.back().SelectedRow);
                }
                break;
            case XK_Left:
                if (Levels.size() > 1) {
                    HideAndDestroyLevelsAfter(Levels.size() - 2);
                    DrawLevel(Levels.back());
                    XFlush(DisplayHandle);
                }
                break;
            case XK_Return:
            case XK_KP_Enter:
                if (!Levels.empty()) {
                    Activate(Levels.size() - 1, Levels.back().SelectedRow);
                }
                break;
            case XK_Home:
                if (!Levels.empty()) {
                    Levels.back().SelectedRow = -1;
                    MoveSelection(1);
                }
                break;
            case XK_End:
                if (!Levels.empty()) {
                    Levels.back().SelectedRow = 0;
                    MoveSelection(-1);
                }
                break;
            default:
                break;
        }
    }

    void
    Open(int rootX, int rootY)
    {
        Close();
        if (!AddLevel(GetMenuNodes(), rootX, rootY)) {
            return;
        }

        const int pointerGrab = XGrabPointer(DisplayHandle,
                                             Levels.front().MenuWindow,
                                             True,
                                             ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
                                             GrabModeAsync,
                                             GrabModeAsync,
                                             0,
                                             0,
                                             CurrentTime);
        const int keyboardGrab =
            XGrabKeyboard(DisplayHandle, Owner, True, GrabModeAsync, GrabModeAsync, CurrentTime);
        if (pointerGrab != GrabSuccess || keyboardGrab != GrabSuccess) {
            std::fprintf(stderr,
                         "FractalSharkGuiLinux: native context-menu grab failed "
                         "(pointer=%d, keyboard=%d).\n",
                         pointerGrab,
                         keyboardGrab);
            Close();
            return;
        }
        XFlush(DisplayHandle);
    }

    void
    Close()
    {
        if (Levels.empty()) {
            return;
        }
        XUngrabPointer(DisplayHandle, CurrentTime);
        XUngrabKeyboard(DisplayHandle, CurrentTime);
        HideAndDestroyLevelsFrom(0);
    }

    bool
    ProcessEvent(const XEvent &event)
    {
        if (Levels.empty()) {
            return false;
        }

        switch (event.type) {
            case Expose: {
                const int levelIndex = FindLevelByWindow(event.xexpose.window);
                if (levelIndex >= 0) {
                    DrawLevel(Levels[static_cast<std::size_t>(levelIndex)]);
                    return true;
                }
                return false;
            }

            case MotionNotify:
                UpdateHover(event.xmotion.x_root, event.xmotion.y_root);
                return true;

            case ButtonPress:
                if (event.xbutton.button == Button4) {
                    Scroll(event.xbutton.x_root, event.xbutton.y_root, -kScrollStep);
                    return true;
                }
                if (event.xbutton.button == Button5) {
                    Scroll(event.xbutton.x_root, event.xbutton.y_root, kScrollStep);
                    return true;
                }
                if (FindLevelAt(event.xbutton.x_root, event.xbutton.y_root) < 0) {
                    Close();
                    return true;
                }
                UpdateHover(event.xbutton.x_root, event.xbutton.y_root);
                return true;

            case ButtonRelease: {
                // Let the opener's right-button release reach ImGui so its
                // input state remains balanced after the native popup takes over.
                if (event.xbutton.button == Button3) {
                    return false;
                }
                if (event.xbutton.button != Button1) {
                    return true;
                }
                const int levelIndex = FindLevelAt(event.xbutton.x_root, event.xbutton.y_root);
                if (levelIndex >= 0) {
                    Level &level = Levels[static_cast<std::size_t>(levelIndex)];
                    Activate(static_cast<std::size_t>(levelIndex),
                             FindRowAt(level, event.xbutton.y_root));
                } else {
                    Close();
                }
                return true;
            }

            case KeyPress:
                HandleKeyPress(event.xkey);
                return true;

            case FocusIn:
            case FocusOut:
                // Keyboard grabs synthesize NotifyGrab/NotifyUngrab focus
                // transitions.  They are bookkeeping, not a real loss of app
                // focus, and closing here would dismiss the menu immediately.
                if (event.xfocus.mode == NotifyGrab || event.xfocus.mode == NotifyUngrab) {
                    return true;
                }
                if (event.type == FocusOut) {
                    Close();
                }
                return false;

            case EnterNotify:
            case LeaveNotify:
                return true;

            default:
                return FindLevelByWindow(event.xany.window) >= 0;
        }
    }
};

X11ContextMenu::X11ContextMenu(
    Display *display, int screen, Window owner, const Menu::IMenuState *state, ExecuteCommandHost *host)
    : m_Impl(std::make_unique<Impl>(display, screen, owner, state, host))
{
}

X11ContextMenu::~X11ContextMenu() = default;

void
X11ContextMenu::Open(int rootX, int rootY)
{
    m_Impl->Open(rootX, rootY);
}

void
X11ContextMenu::Close()
{
    m_Impl->Close();
}

bool
X11ContextMenu::ProcessEvent(const XEvent &event)
{
    return m_Impl->ProcessEvent(event);
}

bool
X11ContextMenu::IsOpen() const noexcept
{
    return !m_Impl->Levels.empty();
}

} // namespace FractalShark::Linux
