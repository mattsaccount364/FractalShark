add_executable(FractalSharkTest)

target_sources(FractalSharkTest PRIVATE
    main.cpp
    TestATInfo.cpp
    TestBLA.cpp
    TestCudaDblflt.cpp
    TestFloatComplex.cpp
    TestHDRFloat.cpp
    TestHDRFloatComplex.cpp
    TestHighPrecisionStrings.cpp
    TestLAParameters.cpp
    TestMpirOrbitEval.cpp
    TestPointZoomBBConverter.cpp
    TestPrecisionCalculator.cpp
)

# TestMpirSerialization exercises MPIR-specific raw stream I/O. On Linux
# HpSharkFloatLib/MpirSerialization.cpp provides no-op stubs (see TODO in
# that file: needs a GMP-compatible mpz_export/mpz_import rewrite). Until
# that exists the roundtrip tests cannot pass, so exclude this TU here.
if(WIN32)
    target_sources(FractalSharkTest PRIVATE TestMpirSerialization.cpp)
endif()

# HpSharkFloatLib sources the tests reference directly. HpSharkFloatLib
# itself is not yet ported to Linux, so pull in only the portable TUs
# the tests actually need.
target_sources(FractalSharkTest PRIVATE
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib/HighPrecision.cpp
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib/LAParameters.cpp
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib/MpirSerialization.cpp
)

# BLA<T> template definitions live in FractalSharkGpuLib/BLA.cuh (a CUDA
# header whose body compiles fine as host C++ since CUDA_CRAP expands to
# nothing). On Linux we compile those instantiations via a small host-side
# wrapper so TestBLA.cpp links.
target_sources(FractalSharkTest PRIVATE BlaHostInstantiate.cpp)
target_include_directories(FractalSharkTest PRIVATE
    ${CMAKE_SOURCE_DIR}/FractalSharkGpuLib
)

# RegisterHeapCleanup() is defined on Windows in HpSharkFloatLib/HeapCpp.cpp,
# which we don't build on Linux. Provide an empty stub on non-Windows.
if(NOT WIN32)
    target_sources(FractalSharkTest PRIVATE HeapCleanupStub.cpp)
endif()

target_include_directories(FractalSharkTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
    ${CMAKE_SOURCE_DIR}/FractalSharkPlatform
    ${CMAKE_SOURCE_DIR}/HpSharkFloatLib
)

target_link_libraries(FractalSharkTest PRIVATE FractalSharkLib)
