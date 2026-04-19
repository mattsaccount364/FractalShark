add_library(HpSharkFloatLib STATIC)

target_sources(HpSharkFloatLib PRIVATE
    HighPrecision.cpp
    LAParameters.cpp
    MpirSerialization.cpp
)

if(WIN32)
    target_sources(HpSharkFloatLib PRIVATE
        Vectors.cpp
    )
else()
    target_sources(HpSharkFloatLib PRIVATE
        VectorsLinux.cpp
        Callstacks.cpp
        EarlyCommandLine.cpp
        Exceptions.cpp
    )
endif()

target_include_directories(HpSharkFloatLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# stdafx.h lives in the repo root
target_include_directories(HpSharkFloatLib PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(HpSharkFloatLib PUBLIC FractalSharkPlatform)

# GMP (MPIR on Windows, GMP on Linux — MpirGmp.h handles the include)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(HpSharkFloatLib PUBLIC ${GMP_LIBRARY})

# std::stacktrace on libstdc++ (used by Exceptions.cpp / Callstacks.cpp)
# requires linking against stdc++_libbacktrace. The library ships with GCC
# 13+/libstdc++ and is what backs std::stacktrace_entry::_M_get_info and
# std::stacktrace::current().
if(NOT WIN32)
    target_link_libraries(HpSharkFloatLib PUBLIC stdc++_libbacktrace)
endif()
