add_executable(FractalSharkTest)

target_sources(FractalSharkTest PRIVATE
    main.cpp
    TestATInfo.cpp
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
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform/Common
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
)

target_link_libraries(FractalSharkTest PRIVATE FractalSharkLib FractalSharkGpuLib)
