add_library(HpSharkFloatTestLib STATIC)

target_sources(HpSharkFloatTestLib PRIVATE
    Conversion.cpp
    HpSharkFloatTestLib.cpp
    KernelInvokeReferenceCorrectness.cpp
    ReferenceReferenceOrbit2.cpp
    TestNewtonRaphson.cpp
    Tests.cpp
    TestTracker.cpp
)

target_include_directories(HpSharkFloatTestLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# stdafx.h at the repo root.
target_include_directories(HpSharkFloatTestLib PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(HpSharkFloatTestLib PUBLIC HpSharkFloatLib)
