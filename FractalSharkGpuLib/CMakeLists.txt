add_library(FractalSharkGpuLib STATIC)

target_sources(FractalSharkGpuLib PRIVATE
    GPU_Render.cu
    Main.cpp
)

target_include_directories(FractalSharkGpuLib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform/Common
    ${CMAKE_SOURCE_DIR}
)

target_compile_options(FractalSharkGpuLib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>
)
