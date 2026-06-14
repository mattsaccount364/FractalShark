// LinuxX11ContextMenu.cpp

#include "LinuxX11ContextMenu.h"

#include "CommandCatalog.h"
#include "Exceptions.h"
#include "MenuTree.h"

#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <functional>
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
constexpr int kShapeBounding = 0;
constexpr int kShapeSet = 0;
constexpr int kShapeUnsorted = 0;

extern "C" {
Bool XShapeQueryExtension(Display *display, int *eventBase, int *errorBase);
void XShapeCombineRectangles(Display *display,
                             Window dest,
                             int destKind,
                             int xOffset,
                             int yOffset,
                             XRectangle *rectangles,
                             int rectangleCount,
                             int operation,
                             int ordering);
}

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
    std::function<void()> RepaintOwner;
    Window Root;
    Window PopupWindow = 0;
    int PopupX = 0;
    int PopupY = 0;
    int PopupWidth = 0;
    int PopupHeight = 0;
    bool PopupMapped = false;
    Pixmap BackBuffer = 0;
    int BackBufferWidth = 0;
    int BackBufferHeight = 0;
    GC GraphicsContext = nullptr;
    XFontSet FontSet = nullptr;
    XFontStruct *FallbackFont = nullptr;
    Colors Palette;
    bool ShapeDirty = true;
    std::vector<Level> Levels;

    Impl(Display *display,
         int screen,
         Window owner,
         const Menu::IMenuState *state,
         ExecuteCommandHost *host,
         std::function<void()> repaintOwner)
        : DisplayHandle(display), Screen(screen), Owner(owner), State(state), Host(host),
          RepaintOwner(std::move(repaintOwner)), Root(RootWindow(display, screen)),
          Palette{AllocateNamedColor(display, screen, "#f0f0f0", WhitePixel(display, screen)),
                  AllocateNamedColor(display, screen, "#101010", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#777777", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#3875d6", BlackPixel(display, screen)),
                  AllocateNamedColor(display, screen, "#ffffff", WhitePixel(display, screen)),
                  AllocateNamedColor(display, screen, "#606060", BlackPixel(display, screen))}
    {
        int eventBase = 0;
        int errorBase = 0;
        if (XShapeQueryExtension(DisplayHandle, &eventBase, &errorBase) == 0) {
            throw FractalSharkSeriousException("FractalSharkGuiLinux: XShape extension unavailable");
        }

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
    DrawText(Drawable target, int x, int baseline, const std::string &text, unsigned long color)
    {
        XSetForeground(DisplayHandle, GraphicsContext, color);
        if (FontSet) {
            Xutf8DrawString(DisplayHandle,
                            target,
                            FontSet,
                            GraphicsContext,
                            x,
                            baseline,
                            text.data(),
                            static_cast<int>(text.size()));
        } else {
            XDrawString(DisplayHandle,
                        target,
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
    DrawCheck(Drawable target, int originX, int rowY, int rowHeight, bool radio, unsigned long color)
    {
        const int centerX = originX + kMenuPadding + kCheckAreaWidth / 2;
        const int centerY = rowY + rowHeight / 2;
        XSetForeground(DisplayHandle, GraphicsContext, color);
        if (radio) {
            XFillArc(
                DisplayHandle, target, GraphicsContext, centerX - 3, centerY - 3, 7, 7, 0, 360 * 64);
            return;
        }
        XDrawLine(
            DisplayHandle, target, GraphicsContext, centerX - 5, centerY, centerX - 1, centerY + 4);
        XDrawLine(
            DisplayHandle, target, GraphicsContext, centerX - 1, centerY + 4, centerX + 6, centerY - 5);
    }

    void
    DrawSubmenuArrow(
        Drawable target, const Level &level, int originX, int rowY, int rowHeight, unsigned long color)
    {
        const int centerX = originX + level.Width - kArrowAreaWidth / 2;
        const int centerY = rowY + rowHeight / 2;
        XSetForeground(DisplayHandle, GraphicsContext, color);
        XDrawLine(
            DisplayHandle, target, GraphicsContext, centerX - 3, centerY - 4, centerX + 2, centerY);
        XDrawLine(
            DisplayHandle, target, GraphicsContext, centerX + 2, centerY, centerX - 3, centerY + 4);
    }

    void
    DrawScrollIndicator(Drawable target, const Level &level, int originX, int originY, bool atTop)
    {
        const int centerX = originX + level.Width / 2;
        const int centerY = originY + (atTop ? 4 : level.Height - 5);
        XSetForeground(DisplayHandle, GraphicsContext, Palette.Text);
        if (atTop) {
            XDrawLine(
                DisplayHandle, target, GraphicsContext, centerX - 4, centerY + 2, centerX, centerY - 2);
            XDrawLine(
                DisplayHandle, target, GraphicsContext, centerX, centerY - 2, centerX + 4, centerY + 2);
        } else {
            XDrawLine(
                DisplayHandle, target, GraphicsContext, centerX - 4, centerY - 2, centerX, centerY + 2);
            XDrawLine(
                DisplayHandle, target, GraphicsContext, centerX, centerY + 2, centerX + 4, centerY - 2);
        }
    }

    void
    FreeBackBuffer()
    {
        if (BackBuffer) {
            XFreePixmap(DisplayHandle, BackBuffer);
            BackBuffer = 0;
        }
        BackBufferWidth = 0;
        BackBufferHeight = 0;
    }

    bool
    EnsureBackBuffer()
    {
        if (!PopupWindow || PopupWidth <= 0 || PopupHeight <= 0) {
            return false;
        }
        if (BackBuffer && BackBufferWidth == PopupWidth && BackBufferHeight == PopupHeight) {
            return true;
        }

        FreeBackBuffer();
        BackBuffer = XCreatePixmap(DisplayHandle,
                                   PopupWindow,
                                   static_cast<unsigned int>(PopupWidth),
                                   static_cast<unsigned int>(PopupHeight),
                                   static_cast<unsigned int>(DefaultDepth(DisplayHandle, Screen)));
        if (!BackBuffer) {
            return false;
        }
        BackBufferWidth = PopupWidth;
        BackBufferHeight = PopupHeight;
        return true;
    }

    void
    DrawLevelTo(Drawable target, const Level &level)
    {
        if (!PopupWindow) {
            return;
        }

        const int originX = level.X - PopupX;
        const int originY = level.Y - PopupY;
        XRectangle clip{static_cast<short>(originX),
                        static_cast<short>(originY),
                        static_cast<unsigned short>(level.Width),
                        static_cast<unsigned short>(level.Height)};
        XSetClipRectangles(DisplayHandle, GraphicsContext, 0, 0, &clip, 1, Unsorted);

        XSetForeground(DisplayHandle, GraphicsContext, Palette.Background);
        XFillRectangle(DisplayHandle,
                       target,
                       GraphicsContext,
                       originX,
                       originY,
                       static_cast<unsigned int>(level.Width),
                       static_cast<unsigned int>(level.Height));
        XSetForeground(DisplayHandle, GraphicsContext, Palette.Border);
        XDrawRectangle(DisplayHandle,
                       target,
                       GraphicsContext,
                       originX,
                       originY,
                       static_cast<unsigned int>(std::max(0, level.Width - 1)),
                       static_cast<unsigned int>(std::max(0, level.Height - 1)));

        for (std::size_t i = 0; i < level.Rows.size(); ++i) {
            const Row &row = level.Rows[i];
            const int rowY = originY + row.Y - level.ScrollOffset;
            if (rowY + row.Height <= originY || rowY >= originY + level.Height) {
                continue;
            }

            const Node &node = *row.MenuNode;
            if (node.kind == Kind::Separator) {
                const int y = rowY + row.Height / 2;
                XSetForeground(DisplayHandle, GraphicsContext, Palette.DisabledText);
                XDrawLine(DisplayHandle,
                          target,
                          GraphicsContext,
                          originX + kMenuPadding,
                          y,
                          originX + level.Width - kMenuPadding,
                          y);
                continue;
            }

            const bool enabled = IsEnabled(node);
            const bool selected = static_cast<int>(i) == level.SelectedRow;
            if (selected) {
                XSetForeground(DisplayHandle, GraphicsContext, Palette.Highlight);
                XFillRectangle(DisplayHandle,
                               target,
                               GraphicsContext,
                               originX,
                               rowY,
                               static_cast<unsigned int>(level.Width),
                               static_cast<unsigned int>(row.Height));
            }

            const unsigned long textColor =
                !enabled ? Palette.DisabledText : (selected ? Palette.HighlightText : Palette.Text);
            if (IsChecked(node)) {
                DrawCheck(
                    target, originX, rowY, row.Height, node.checkKind == CheckKind::Radio, textColor);
            }
            DrawText(target,
                     originX + kMenuPadding + kCheckAreaWidth,
                     TextBaseline(rowY, row.Height),
                     row.Label,
                     textColor);
            if (node.kind == Kind::Popup) {
                DrawSubmenuArrow(target, level, originX, rowY, row.Height, textColor);
            }
        }

        if (level.ScrollOffset > 0) {
            DrawScrollIndicator(target, level, originX, originY, true);
        }
        if (level.ScrollOffset + level.Height < level.ContentHeight) {
            DrawScrollIndicator(target, level, originX, originY, false);
        }
        XSetClipMask(DisplayHandle, GraphicsContext, 0);
    }

    void
    DrawLevel(const Level &level)
    {
        if (!EnsureBackBuffer()) {
            return;
        }

        DrawLevelTo(BackBuffer, level);
        const int originX = level.X - PopupX;
        const int originY = level.Y - PopupY;
        XCopyArea(DisplayHandle,
                  BackBuffer,
                  PopupWindow,
                  GraphicsContext,
                  originX,
                  originY,
                  static_cast<unsigned int>(level.Width),
                  static_cast<unsigned int>(level.Height),
                  originX,
                  originY);
        XFlush(DisplayHandle);
    }

    void
    DrawAll()
    {
        if (!EnsureBackBuffer()) {
            return;
        }

        XSetClipMask(DisplayHandle, GraphicsContext, 0);
        XSetForeground(DisplayHandle, GraphicsContext, Palette.Background);
        XFillRectangle(DisplayHandle,
                       BackBuffer,
                       GraphicsContext,
                       0,
                       0,
                       static_cast<unsigned int>(PopupWidth),
                       static_cast<unsigned int>(PopupHeight));
        for (const Level &level : Levels) {
            DrawLevelTo(BackBuffer, level);
        }
        XSetClipMask(DisplayHandle, GraphicsContext, 0);
        XCopyArea(DisplayHandle,
                  BackBuffer,
                  PopupWindow,
                  GraphicsContext,
                  0,
                  0,
                  static_cast<unsigned int>(PopupWidth),
                  static_cast<unsigned int>(PopupHeight),
                  0,
                  0);
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
    EnsurePopupWindow()
    {
        if (PopupWindow) {
            return true;
        }

        XSetWindowAttributes attributes{};
        attributes.override_redirect = True;
        attributes.background_pixel = Palette.Background;
        attributes.border_pixel = Palette.Border;
        attributes.event_mask = ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                                EnterWindowMask | LeaveWindowMask;

        PopupWindow = XCreateWindow(DisplayHandle,
                                    Root,
                                    0,
                                    0,
                                    1,
                                    1,
                                    0,
                                    CopyFromParent,
                                    InputOutput,
                                    CopyFromParent,
                                    CWOverrideRedirect | CWBackPixel | CWBorderPixel | CWEventMask,
                                    &attributes);
        if (!PopupWindow) {
            return false;
        }

        return true;
    }

    void
    DestroyPopupWindow()
    {
        if (!PopupWindow) {
            return;
        }

        FreeBackBuffer();
        XDestroyWindow(DisplayHandle, PopupWindow);
        PopupWindow = 0;
        PopupX = 0;
        PopupY = 0;
        PopupWidth = 0;
        PopupHeight = 0;
        PopupMapped = false;
        ShapeDirty = true;
        XFlush(DisplayHandle);
    }

    bool
    UpdatePopupWindow()
    {
        if (Levels.empty()) {
            DestroyPopupWindow();
            return false;
        }
        if (!EnsurePopupWindow()) {
            return false;
        }

        int minX = Levels.front().X;
        int minY = Levels.front().Y;
        int maxX = Levels.front().X + Levels.front().Width;
        int maxY = Levels.front().Y + Levels.front().Height;
        for (const Level &level : Levels) {
            minX = std::min(minX, level.X);
            minY = std::min(minY, level.Y);
            maxX = std::max(maxX, level.X + level.Width);
            maxY = std::max(maxY, level.Y + level.Height);
        }

        const int newX = minX;
        const int newY = minY;
        const int newWidth = std::max(1, maxX - minX);
        const int newHeight = std::max(1, maxY - minY);
        const bool geometryChanged =
            newX != PopupX || newY != PopupY || newWidth != PopupWidth || newHeight != PopupHeight;

        if (geometryChanged) {
            FreeBackBuffer();
        }

        PopupX = newX;
        PopupY = newY;
        PopupWidth = newWidth;
        PopupHeight = newHeight;

        if (geometryChanged) {
            XMoveResizeWindow(DisplayHandle,
                              PopupWindow,
                              PopupX,
                              PopupY,
                              static_cast<unsigned int>(PopupWidth),
                              static_cast<unsigned int>(PopupHeight));
        }

        if (geometryChanged || ShapeDirty) {
            std::vector<XRectangle> rectangles;
            rectangles.reserve(Levels.size());
            for (const Level &level : Levels) {
                rectangles.push_back(XRectangle{static_cast<short>(level.X - PopupX),
                                                static_cast<short>(level.Y - PopupY),
                                                static_cast<unsigned short>(level.Width),
                                                static_cast<unsigned short>(level.Height)});
            }
            XShapeCombineRectangles(DisplayHandle,
                                    PopupWindow,
                                    kShapeBounding,
                                    0,
                                    0,
                                    rectangles.data(),
                                    static_cast<int>(rectangles.size()),
                                    kShapeSet,
                                    kShapeUnsorted);
            ShapeDirty = false;
        }

        if (!PopupMapped) {
            XMapRaised(DisplayHandle, PopupWindow);
            PopupMapped = true;
        }

        return true;
    }

    bool
    AddLevel(std::span<const Node> nodes, int desiredX, int desiredY, int leftAnchorX = -1)
    {
        Level level = BuildLevel(nodes);
        PositionLevel(level, desiredX, desiredY, leftAnchorX);
        Levels.push_back(std::move(level));
        ShapeDirty = true;
        if (!UpdatePopupWindow()) {
            Levels.pop_back();
            ShapeDirty = true;
            return false;
        }
        DrawAll();
        XFlush(DisplayHandle);
        return true;
    }

    bool
    HideAndDestroyLevelsFrom(std::size_t firstIndex, bool present = true)
    {
        if (firstIndex >= Levels.size()) {
            return false;
        }

        Levels.resize(firstIndex);
        ShapeDirty = true;
        if (present) {
            if (Levels.empty()) {
                DestroyPopupWindow();
            } else {
                UpdatePopupWindow();
                DrawAll();
            }
            XFlush(DisplayHandle);
        }
        return true;
    }

    bool
    HideAndDestroyLevelsAfter(std::size_t index)
    {
        return HideAndDestroyLevelsFrom(index + 1);
    }

    bool
    ReplaceMappedChildLevel(
        std::size_t levelIndex, std::span<const Node> nodes, int desiredX, int desiredY, int leftAnchorX)
    {
        const std::size_t childIndex = levelIndex + 1;
        if (childIndex >= Levels.size()) {
            return false;
        }

        HideAndDestroyLevelsFrom(childIndex + 1, false);

        Level replacement = BuildLevel(nodes);
        PositionLevel(replacement, desiredX, desiredY, leftAnchorX);
        Levels[childIndex] = std::move(replacement);
        ShapeDirty = true;
        UpdatePopupWindow();
        DrawAll();
        XFlush(DisplayHandle);
        return true;
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

    bool
    IsPopupWindow(Window window) const
    {
        return PopupWindow != 0 && window == PopupWindow;
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

    bool
    OpenSubmenu(std::size_t levelIndex, int rowIndex, bool selectFirst)
    {
        Level &parent = Levels[levelIndex];
        if (rowIndex < 0 || rowIndex >= static_cast<int>(parent.Rows.size())) {
            return HideAndDestroyLevelsAfter(levelIndex);
        }
        const Row &row = parent.Rows[static_cast<std::size_t>(rowIndex)];
        const Node &node = *row.MenuNode;
        if (node.kind != Kind::Popup || !IsEnabled(node)) {
            return HideAndDestroyLevelsAfter(levelIndex);
        }

        if (Levels.size() > levelIndex + 1 && Levels[levelIndex + 1].Nodes.data() == node.kids.data()) {
            if (selectFirst) {
                SelectFirst(levelIndex + 1);
                return true;
            }
            return false;
        }

        const int desiredX = parent.X + parent.Width + 1;
        const int desiredY = parent.Y + row.Y - parent.ScrollOffset;
        const int leftAnchorX = parent.X;
        bool presented = false;
        if (Levels.size() > levelIndex + 1) {
            presented = ReplaceMappedChildLevel(levelIndex, node.kids, desiredX, desiredY, leftAnchorX);
        } else {
            presented = AddLevel(node.kids, desiredX, desiredY, leftAnchorX);
            if (!presented) {
                return false;
            }
        }
        if (selectFirst) {
            SelectFirst(levelIndex + 1);
            presented = true;
        }
        return presented;
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
        bool selectionChanged = false;
        if (level.SelectedRow != rowIndex) {
            level.SelectedRow = rowIndex;
            selectionChanged = true;
        }
        const bool presented = OpenSubmenu(static_cast<std::size_t>(levelIndex), rowIndex, false);
        if (selectionChanged && !presented && static_cast<std::size_t>(levelIndex) < Levels.size()) {
            DrawLevel(Levels[static_cast<std::size_t>(levelIndex)]);
        }
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
        if (!HideAndDestroyLevelsAfter(static_cast<std::size_t>(levelIndex))) {
            DrawLevel(level);
        }
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
                const std::size_t levelIndex = Levels.size() - 1;
                const bool presented = OpenSubmenu(levelIndex, rowIndex, false);
                if (!presented && levelIndex < Levels.size()) {
                    DrawLevel(Levels[levelIndex]);
                }
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
                    if (!HideAndDestroyLevelsAfter(Levels.size() - 2)) {
                        DrawLevel(Levels.back());
                        XFlush(DisplayHandle);
                    }
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
                                             PopupWindow,
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
        if (Levels.empty() && !PopupWindow) {
            return;
        }
        XUngrabPointer(DisplayHandle, CurrentTime);
        XUngrabKeyboard(DisplayHandle, CurrentTime);
        Levels.clear();
        DestroyPopupWindow();
    }

    bool
    ProcessEvent(const XEvent &event)
    {
        if (Levels.empty()) {
            return false;
        }

        switch (event.type) {
            case Expose: {
                if (IsPopupWindow(event.xexpose.window)) {
                    DrawAll();
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
                return IsPopupWindow(event.xany.window);
        }
    }
};

X11ContextMenu::X11ContextMenu(Display *display,
                               int screen,
                               Window owner,
                               const Menu::IMenuState *state,
                               ExecuteCommandHost *host,
                               std::function<void()> repaintOwner)
    : m_Impl(std::make_unique<Impl>(display, screen, owner, state, host, std::move(repaintOwner)))
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
