# FractalSharkGuiLinux — Linux-only GUI executable.
#
# This subdirectory is only added on UNIX from the root CMakeLists.txt.  The
# Windows GUI lives in FractalSharkGUILib + FractalShark/Fractals.vcxproj and
# is intentionally separate.

include(FetchContent)

# Dear ImGui is the only third-party GUI dep.  Vendored at configure time at a
# pinned tag so reproducible builds don't drift with upstream master.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_executable(FractalSharkGuiLinux
    main.cpp
    LinuxClipboard.cpp
    LinuxXlibBackend.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)

target_include_directories(FractalSharkGuiLinux PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform
    ${imgui_SOURCE_DIR}
)

find_package(X11 REQUIRED)

target_link_libraries(FractalSharkGuiLinux PRIVATE
    FractalSharkPlatform
    X11::X11
)

# ImGui itself emits some warnings under -Wpedantic; keep the rest of the tree
# strict but relax for vendored sources.
set_source_files_properties(
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    PROPERTIES
    COMPILE_OPTIONS "-Wno-pedantic"
)

set_target_properties(FractalSharkGuiLinux PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
