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

set(FRACTALSHARK_LINUX_SPLASH_IMAGE_NAMES
    FractalShark.png
    "Chatgpt Fractal Shark.png"
    PrecisionShark.png
    FractalShark2.png
)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    message(FATAL_ERROR "Linux splash binary embedding currently supports x86_64 ELF targets.")
endif()

find_program(FRACTALSHARK_LINUX_OBJCOPY NAMES objcopy REQUIRED)

set(FRACTALSHARK_LINUX_SPLASH_EMBED_DIR "${CMAKE_CURRENT_BINARY_DIR}/embedded_splash")
set(FRACTALSHARK_LINUX_SPLASH_OBJECTS)
set(FRACTALSHARK_LINUX_EMBEDDED_SPLASH_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedLinuxSplashImages.cpp")
set(generated_splash_source
    "#include \"LinuxEmbeddedSplashImages.h\"\n\n#include <array>\n#include <cstdint>\n\nextern \"C\" {\n")

function(fractalshark_add_embedded_splash_image image_name image_symbol)
    set(image_path "${CMAKE_SOURCE_DIR}/Pics/${image_name}")
    set(copied_image "${FRACTALSHARK_LINUX_SPLASH_EMBED_DIR}/${image_symbol}.bin")
    set(image_object "${FRACTALSHARK_LINUX_SPLASH_EMBED_DIR}/${image_symbol}.o")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${image_path}")

    add_custom_command(
        OUTPUT "${image_object}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FRACTALSHARK_LINUX_SPLASH_EMBED_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy "${image_path}" "${copied_image}"
        COMMAND ${CMAKE_COMMAND} -E chdir "${FRACTALSHARK_LINUX_SPLASH_EMBED_DIR}"
            "${FRACTALSHARK_LINUX_OBJCOPY}"
            -I binary
            -O elf64-x86-64
            -B i386:x86-64
            --rename-section .data=.rodata,alloc,load,readonly,data,contents
            --redefine-sym "_binary_${image_symbol}_bin_start=fractalshark_splash_${image_symbol}_start"
            --redefine-sym "_binary_${image_symbol}_bin_end=fractalshark_splash_${image_symbol}_end"
            --redefine-sym "_binary_${image_symbol}_bin_size=fractalshark_splash_${image_symbol}_size"
            "${image_symbol}.bin"
            "${image_object}"
        COMMAND ${CMAKE_COMMAND} -E chdir "${FRACTALSHARK_LINUX_SPLASH_EMBED_DIR}"
            "${FRACTALSHARK_LINUX_OBJCOPY}"
            --add-section .note.GNU-stack=/dev/null
            --set-section-flags .note.GNU-stack=readonly
            "${image_object}"
        DEPENDS "${image_path}"
        VERBATIM
        COMMENT "Embedding Linux splash image ${image_name}"
    )

    set_source_files_properties("${image_object}" PROPERTIES
        EXTERNAL_OBJECT TRUE
        GENERATED TRUE
    )
    set(FRACTALSHARK_LINUX_SPLASH_OBJECTS
        ${FRACTALSHARK_LINUX_SPLASH_OBJECTS}
        "${image_object}"
        PARENT_SCOPE
    )
    string(APPEND generated_splash_source
        "extern const std::byte fractalshark_splash_${image_symbol}_start[];\n"
        "extern const std::byte fractalshark_splash_${image_symbol}_end[];\n")
    set(generated_splash_source "${generated_splash_source}" PARENT_SCOPE)
endfunction()

fractalshark_add_embedded_splash_image("FractalShark.png" "FractalShark_png")
fractalshark_add_embedded_splash_image("Chatgpt Fractal Shark.png" "Chatgpt_Fractal_Shark_png")
fractalshark_add_embedded_splash_image("PrecisionShark.png" "PrecisionShark_png")
fractalshark_add_embedded_splash_image("FractalShark2.png" "FractalShark2_png")

string(APPEND generated_splash_source
    "}\n\nnamespace FractalShark::Linux {\n\nnamespace {\n\n"
    "std::span<const std::byte>\n"
    "SpanFromSymbols(const std::byte *start, const std::byte *end)\n"
    "{\n"
    "    const auto startAddress = reinterpret_cast<std::uintptr_t>(start);\n"
    "    const auto endAddress = reinterpret_cast<std::uintptr_t>(end);\n"
    "    return std::span<const std::byte>{start, endAddress - startAddress};\n"
    "}\n\n")

list(LENGTH FRACTALSHARK_LINUX_SPLASH_IMAGE_NAMES image_count)
string(APPEND generated_splash_source
    "const std::array<EmbeddedSplashImage, ${image_count}> kEmbeddedSplashImages{{\n")

string(APPEND generated_splash_source
    "    {\"FractalShark.png\", SpanFromSymbols(fractalshark_splash_FractalShark_png_start, fractalshark_splash_FractalShark_png_end)},\n"
    "    {\"Chatgpt Fractal Shark.png\", SpanFromSymbols(fractalshark_splash_Chatgpt_Fractal_Shark_png_start, fractalshark_splash_Chatgpt_Fractal_Shark_png_end)},\n"
    "    {\"PrecisionShark.png\", SpanFromSymbols(fractalshark_splash_PrecisionShark_png_start, fractalshark_splash_PrecisionShark_png_end)},\n"
    "    {\"FractalShark2.png\", SpanFromSymbols(fractalshark_splash_FractalShark2_png_start, fractalshark_splash_FractalShark2_png_end)},\n")

string(APPEND generated_splash_source
    "}};\n\n"
    "} // namespace\n\n"
    "std::span<const EmbeddedSplashImage>\n"
    "GetEmbeddedSplashImages()\n"
    "{\n"
    "    return kEmbeddedSplashImages;\n"
    "}\n\n"
    "} // namespace FractalShark::Linux\n")

file(WRITE "${FRACTALSHARK_LINUX_EMBEDDED_SPLASH_SOURCE}" "${generated_splash_source}")

add_executable(FractalSharkGuiLinux
    main.cpp
    LinuxClipboard.cpp
    LinuxCommandHandlers.cpp
    LinuxMenuState.cpp
    LinuxSplashWindow.cpp
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
    "${FRACTALSHARK_LINUX_EMBEDDED_SPLASH_SOURCE}"
    ${FRACTALSHARK_LINUX_SPLASH_OBJECTS}
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
