add_executable(FractalSharkCli
    main.cpp
)

# BLA<T> template definitions live in FractalSharkGpuLib/BLA.cuh; mirror
# FractalSharkTest's BlaHostInstantiate.cpp workaround so the CPU-only
# Linux build of the CLI can link without pulling in CUDA.
target_sources(FractalSharkCli PRIVATE ${CMAKE_SOURCE_DIR}/FractalSharkTest/BlaHostInstantiate.cpp)

target_include_directories(FractalSharkCli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
    ${CMAKE_SOURCE_DIR}/FractalSharkGUILib
    ${CMAKE_SOURCE_DIR}/FractalSharkGpuLib
)

target_link_libraries(FractalSharkCli PRIVATE FractalSharkLib FractalSharkGpuLib)
