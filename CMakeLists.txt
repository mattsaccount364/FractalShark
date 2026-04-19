cmake_minimum_required(VERSION 3.20)
project(FractalShark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic)

add_subdirectory(FractalSharkPlatform)
add_subdirectory(HpSharkFloatLib)
add_subdirectory(FractalSharkLib)
add_subdirectory(FractalSharkTest)
