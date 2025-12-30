#pragma once

#include <functional>
#include <unordered_map>

// Forward declare to avoid dragging all of MainWindow.h into every TU that uses this.
class MainWindow;

// Routes WM_COMMAND ids to MainWindow actions (Menu* methods, fractal operations, etc.).
// NOTE: This class will typically need either:
//   - MainWindow to declare `friend class CommandDispatcher;`
//   - OR MainWindow to expose the used members via public/protected accessors/thin command methods.
class CommandDispatcher {
public:
    explicit CommandDispatcher(MainWindow &owner);

    // Returns true if wmId was handled.
    bool Dispatch(int wmId);

private:
    using Fn = std::function<void(MainWindow &)>;

    bool HandleCommandRange(int wmId);
    bool HandleAlgCommand(int wmId);
    bool HandleCommandTable(int wmId);

    void BuildTable();

private:
    MainWindow &w_;
    std::unordered_map<int, Fn> table_;
};
