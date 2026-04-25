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
