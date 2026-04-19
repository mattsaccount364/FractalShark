add_library(HpSharkFloatTestLib STATIC)

target_sources(HpSharkFloatTestLib PRIVATE
    Conversion.cpp
    HpSharkFloatTestLib.cpp
    KernelInvokeAddCorrectness.cpp
    KernelInvokeAddPerf.cpp
    KernelInvokeNTTCorrectness.cpp
    KernelInvokeNTTPerf.cpp
    KernelInvokeReferenceCorrectness.cpp
    ReferenceAdd.cpp
    ReferenceNTT2.cpp
    ReferenceReferenceOrbit.cpp
    ShowMostEfficientSizes.cpp
    TestNewtonRaphson.cpp
    Tests.cpp
    TestTracker.cpp
    # Shared across FractalSharkLib — reach across source trees as the
    # Windows project does (vcxproj has matching <ClCompile> entries).
    ${CMAKE_SOURCE_DIR}/FractalSharkLib/MpirComplex.cpp
    ${CMAKE_SOURCE_DIR}/FractalSharkLib/MpirOrbitEval.cpp
)

target_include_directories(HpSharkFloatTestLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# stdafx.h at the repo root, and FractalSharkLib headers for MpirComplex/OrbitEval.
target_include_directories(HpSharkFloatTestLib PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/FractalSharkLib
)

target_link_libraries(HpSharkFloatTestLib PUBLIC HpSharkFloatLib)
