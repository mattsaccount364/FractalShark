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
    LinuxCommandHandlers.cpp
    LinuxMenuState.cpp
    LinuxX11ContextMenu.cpp
    LinuxXlibBackend.cpp
    LinuxImGuiOverlay.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
)

# BLA<T> template definitions live in FractalSharkGpuLib/BLA.cuh; mirror the
# CLI workaround so the CPU-only Linux GUI link can resolve those symbols
# without pulling in CUDA.
target_sources(FractalSharkGuiLinux PRIVATE
    ${CMAKE_SOURCE_DIR}/FractalSharkTest/BlaHostInstantiate.cpp
)

target_include_directories(FractalSharkGuiLinux PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
    ${CMAKE_SOURCE_DIR}/FractalSharkGUILib
    ${CMAKE_SOURCE_DIR}/FractalSharkGpuLib
    ${imgui_SOURCE_DIR}
)

find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL GLX)

set(FRACTALSHARK_LINUX_GLX_LIBRARY)
if(TARGET OpenGL::GLX)
    set(FRACTALSHARK_LINUX_GLX_LIBRARY OpenGL::GLX)
elseif(OPENGL_glx_LIBRARY)
    set(FRACTALSHARK_LINUX_GLX_LIBRARY ${OPENGL_glx_LIBRARY})
else()
    message(FATAL_ERROR
        "OpenGL GLX component was found, but neither OpenGL::GLX nor "
        "OPENGL_glx_LIBRARY is available.")
endif()

target_link_libraries(FractalSharkGuiLinux PRIVATE
    FractalSharkLib
    FractalSharkGpuLib
    FractalSharkPlatform
    X11::X11
    OpenGL::GL
    ${FRACTALSHARK_LINUX_GLX_LIBRARY}
    ${CMAKE_DL_LIBS}
)

# ImGui itself emits some warnings under -Wpedantic; keep the rest of the tree
# strict but relax for vendored sources.
set_source_files_properties(
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp
    PROPERTIES
    COMPILE_OPTIONS "-Wno-pedantic"
)

set_target_properties(FractalSharkGuiLinux PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)
