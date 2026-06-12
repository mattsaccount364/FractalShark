add_library(FractalSharkPlatform STATIC)

if(WIN32)
    target_sources(FractalSharkPlatform PRIVATE
        WaitCursorWin32.cpp
        EnvironmentWin32.cpp
        CrashHandlerWin32.cpp
        JobObjectWin32.cpp
    )
else()
    set(FRACTALSHARK_IMAGINA_FIXTURE_NAMES
        View0.im
        View5.im
        View14.im
        View15.im
        View19.im
        ViewEasy1.im
    )
    set(FRACTALSHARK_IMAGINA_FIXTURE_PRESETS
        0
        5
        14
        15
        19
        none
    )

    function(fractalshark_bytes_for_file output_var file_path)
        file(READ "${file_path}" fixture_hex HEX)
        string(LENGTH "${fixture_hex}" fixture_hex_len)
        math(EXPR byte_count "${fixture_hex_len} / 2")

        set(bytes "")
        if(byte_count GREATER 0)
            math(EXPR last_byte "${byte_count} - 1")
            foreach(byte_index RANGE 0 ${last_byte})
                math(EXPR hex_index "${byte_index} * 2")
                math(EXPR column "${byte_index} % 16")
                string(SUBSTRING "${fixture_hex}" ${hex_index} 2 byte_hex)
                if(column EQUAL 0)
                    string(APPEND bytes "\n    ")
                endif()
                string(APPEND bytes "std::byte{0x${byte_hex}}, ")
            endforeach()
        endif()

        set(${output_var} "${bytes}" PARENT_SCOPE)
    endfunction()

    set(FRACTALSHARK_EMBEDDED_IMAGINA_SOURCE
        "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedImaginaFixturesLinux.cpp")
    set(generated_source "#include \"Environment.h\"\n\n#include <array>\n\nnamespace {\n\n")

    list(LENGTH FRACTALSHARK_IMAGINA_FIXTURE_NAMES fixture_count)
    math(EXPR last_fixture "${fixture_count} - 1")

    foreach(fixture_index RANGE 0 ${last_fixture})
        list(GET FRACTALSHARK_IMAGINA_FIXTURE_NAMES ${fixture_index} fixture_name)
        string(MAKE_C_IDENTIFIER "${fixture_name}" fixture_symbol)
        set(fixture_path "${CMAKE_SOURCE_DIR}/FromImagina/${fixture_name}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${fixture_path}")
        fractalshark_bytes_for_file(fixture_bytes "${fixture_path}")
        string(APPEND generated_source
            "alignas(16) const std::byte k${fixture_symbol}Bytes[] = {${fixture_bytes}\n};\n\n")
    endforeach()

    string(APPEND generated_source
        "struct ImaginaFixtureResource {\n"
        "    std::wstring_view name;\n"
        "    std::span<const std::byte> bytes;\n"
        "};\n\n"
        "const std::array<Environment::ImaginaFixtureInfo, ${fixture_count}> kImaginaFixtureInfos{{\n")

    foreach(fixture_index RANGE 0 ${last_fixture})
        list(GET FRACTALSHARK_IMAGINA_FIXTURE_NAMES ${fixture_index} fixture_name)
        list(GET FRACTALSHARK_IMAGINA_FIXTURE_PRESETS ${fixture_index} fixture_preset)
        if(fixture_preset STREQUAL "none")
            set(preset_expr "std::nullopt")
        else()
            set(preset_expr "size_t{${fixture_preset}}")
        endif()
        string(APPEND generated_source "    {L\"${fixture_name}\", ${preset_expr}},\n")
    endforeach()

    string(APPEND generated_source
        "}};\n\n"
        "const std::array<ImaginaFixtureResource, ${fixture_count}> kImaginaFixtureResources{{\n")

    foreach(fixture_index RANGE 0 ${last_fixture})
        list(GET FRACTALSHARK_IMAGINA_FIXTURE_NAMES ${fixture_index} fixture_name)
        string(MAKE_C_IDENTIFIER "${fixture_name}" fixture_symbol)
        string(APPEND generated_source
            "    {L\"${fixture_name}\", std::span<const std::byte>{k${fixture_symbol}Bytes}},\n")
    endforeach()

    string(APPEND generated_source
        "}};\n\n"
        "} // namespace\n\n"
        "std::span<const Environment::ImaginaFixtureInfo>\n"
        "Environment::GetEmbeddedImaginaFixtureInfos()\n"
        "{\n"
        "    return kImaginaFixtureInfos;\n"
        "}\n\n"
        "std::optional<Environment::EmbeddedResourceView>\n"
        "Environment::FindEmbeddedImaginaFixture(std::wstring_view name)\n"
        "{\n"
        "    for (const auto &resource : kImaginaFixtureResources) {\n"
        "        if (resource.name == name) {\n"
        "            return EmbeddedResourceView{resource.name, resource.bytes};\n"
        "        }\n"
        "    }\n"
        "    return std::nullopt;\n"
        "}\n")

    file(WRITE "${FRACTALSHARK_EMBEDDED_IMAGINA_SOURCE}" "${generated_source}")

    target_sources(FractalSharkPlatform PRIVATE
        WaitCursorLinux.cpp
        EnvironmentLinux.cpp
        CrashHandlerLinux.cpp
        JobObjectLinux.cpp
        "${FRACTALSHARK_EMBEDDED_IMAGINA_SOURCE}"
    )
    find_package(X11 REQUIRED)
    target_link_libraries(FractalSharkPlatform PUBLIC X11::X11)
endif()

target_include_directories(FractalSharkPlatform PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# GMP (MPIR on Windows, GMP on Linux — Environment.h pulls in the header)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(FractalSharkPlatform PUBLIC ${GMP_LIBRARY})
