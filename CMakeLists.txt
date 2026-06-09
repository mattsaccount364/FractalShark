cmake_minimum_required(VERSION 3.20)
project(FractalShark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

option(FRACTALSHARK_ENABLE_STACKTRACE "Enable std::stacktrace diagnostics when the platform supports them" ON)

set(FRACTALSHARK_HAS_STACKTRACE_DEFINE 0)
set(FRACTALSHARK_STACKTRACE_LINK_LIB "")

if(WIN32)
    set(FRACTALSHARK_HAS_STACKTRACE_DEFINE 1)
elseif(FRACTALSHARK_ENABLE_STACKTRACE)
    find_library(FRACTALSHARK_STACKTRACE_LIB
        NAMES stdc++exp stdc++_libbacktrace
        HINTS /usr/lib/gcc/x86_64-linux-gnu/15
              /usr/lib/gcc/x86_64-linux-gnu/14
              /usr/lib/gcc/x86_64-linux-gnu/13
    )
    if(FRACTALSHARK_STACKTRACE_LIB)
        set(FRACTALSHARK_HAS_STACKTRACE_DEFINE 1)
        set(FRACTALSHARK_STACKTRACE_LINK_LIB "${FRACTALSHARK_STACKTRACE_LIB}")
        message(STATUS "Using std::stacktrace backend: ${FRACTALSHARK_STACKTRACE_LIB}")
    else()
        message(WARNING "Could not find libstdc++exp or libstdc++_libbacktrace; "
                        "building without std::stacktrace diagnostics.")
    endif()
endif()

# Warnings (CXX only — nvcc doesn't support these flags)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

add_subdirectory(FractalSharkPlatform)
add_subdirectory(HpSharkFloatLib)
add_subdirectory(FractalSharkLib)
add_subdirectory(FractalSharkGpuLib)
add_subdirectory(HpSharkInstantiate)
add_subdirectory(HpSharkFloatTestLib)
add_subdirectory(HpSharkFloatTest)
add_subdirectory(FractalSharkTest)
add_subdirectory(FractalSharkCli)

if(UNIX AND NOT APPLE)
    add_subdirectory(FractalSharkGuiLinux)
endif()
