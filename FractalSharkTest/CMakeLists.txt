add_executable(FractalSharkTest)

target_sources(FractalSharkTest PRIVATE
    main.cpp
    GpuTestRuntime.cpp
    RenderTestSupport.cpp
    RenderTestCatalog.cpp
    TestRenderCases.cpp
    TestRenderUtilities.cpp
    TestGpuRenderModes.cpp
    TestGpuRecolor.cpp
    TestGpuReferenceBuckets.cpp
    TestTestFramework.cpp
    TestLAEvaluation.cpp
    TestLAStorage.cpp
    TestBLA.cpp
    TestCudaDblflt.cpp
    TestFloatComplex.cpp
    TestHeapAllocator.cpp
    TestHDRFloat.cpp
    TestHDRFloatComplex.cpp
    TestHighPrecisionStrings.cpp
    TestLAParameters.cpp
    MenuStateTest.cpp
    TestMpirOrbitEval.cpp
    TestPointZoomBBConverter.cpp
    TestPrecisionCalculator.cpp
    TestGrowableVector.cpp
    TestRenderGoldens.cpp
    TestGuiShared.cpp
    TestFractalSharkLib.cpp
)

# TestMpirSerialization exercises raw stream I/O that is byte-compatible
# between MPIR (Windows) and GMP (Linux via mpz_export/mpz_import).
target_sources(FractalSharkTest PRIVATE TestMpirSerialization.cpp)

# HpSharkFloatLib is now a proper CMake library linked via FractalSharkLib.
# Its PUBLIC include directory and portable sources (HighPrecision,
# LAParameters, MpirSerialization, Vectors/VectorsLinux, Exceptions) are
# pulled in transitively — no direct target_sources entries are needed here.

# BLA<T> template definitions live in FractalSharkGpuLib/BLA.cuh (a CUDA
# header whose body compiles fine as host C++ since CUDA_CRAP expands to
# nothing). On Linux we compile those instantiations via a small host-side
# wrapper so TestBLA.cpp links.
target_sources(FractalSharkTest PRIVATE BlaHostInstantiate.cpp)
target_include_directories(FractalSharkTest PRIVATE
    ${CMAKE_SOURCE_DIR}/FractalSharkGpuLib
)

# The custom heap allocator in HpSharkFloatLib/heap_allocator/HeapCpp.cpp is
# linked through HpSharkFloatLib on Linux and Windows.

target_include_directories(FractalSharkTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
)

find_package(CUDAToolkit REQUIRED)

if(WIN32)
    enable_language(RC)
    target_sources(FractalSharkTest PRIVATE ImaginaFixtures.rc)
endif()

target_link_libraries(FractalSharkTest PRIVATE CUDA::cudart_static FractalSharkLib FractalSharkGpuLib FractalSharkPlatform)

# Separate hardware suite; the portable CPU suite remains runnable without CUDA hardware.
add_executable(FractalSharkCudaTest main.cpp GpuTestRuntime.cpp TestLAUpload.cu TestLACudaRender.cpp TestPngCudaEncoder.cu TestPngSave.cpp TestGpuRecolor.cpp)
target_include_directories(FractalSharkCudaTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkGpuLib
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
)
target_compile_options(FractalSharkCudaTest PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>
)
target_link_libraries(FractalSharkCudaTest PRIVATE
    FractalSharkLib FractalSharkGpuLib FractalSharkPlatform CUDA::cudart_static)
