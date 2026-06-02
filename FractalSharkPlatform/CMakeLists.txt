add_library(FractalSharkPlatform STATIC)

target_sources(FractalSharkPlatform PRIVATE
    WaitCursor.cpp
)

if(WIN32)
    target_sources(FractalSharkPlatform PRIVATE
        EnvironmentWin32.cpp
        CrashHandlerWin32.cpp
    )
else()
    target_sources(FractalSharkPlatform PRIVATE
        EnvironmentLinux.cpp
        CrashHandlerLinux.cpp
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

# std::stacktrace on libstdc++ (kept consistent with HpSharkFloatLib).
# libstdc++ 13 ships the impl in libstdc++_libbacktrace; libstdc++ 14
# moved it into libstdc++exp. Link whichever one the active toolchain
# provides. Mirrors the block in HpSharkFloatLib/CMakeLists.txt.
if(NOT WIN32)
    find_library(FSP_STDCXX_STACKTRACE_LIB
        NAMES stdc++exp stdc++_libbacktrace
        PATHS /usr/lib/gcc/x86_64-linux-gnu/14
              /usr/lib/gcc/x86_64-linux-gnu/13
        NO_DEFAULT_PATH
    )
    if(NOT FSP_STDCXX_STACKTRACE_LIB)
        message(FATAL_ERROR "Could not find libstdc++exp or libstdc++_libbacktrace "
                            "(needed by std::stacktrace). Install libstdc++-14-dev "
                            "or libstdc++-13-dev.")
    endif()
    target_link_libraries(FractalSharkPlatform PUBLIC ${FSP_STDCXX_STACKTRACE_LIB})
endif()
