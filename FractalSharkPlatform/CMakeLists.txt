add_library(FractalSharkPlatform STATIC)

target_sources(FractalSharkPlatform PRIVATE
    WaitCursor.cpp
)

if(WIN32)
    target_sources(FractalSharkPlatform PRIVATE EnvironmentWin32.cpp)
else()
    target_sources(FractalSharkPlatform PRIVATE EnvironmentLinux.cpp)
endif()

target_include_directories(FractalSharkPlatform PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# GMP (MPIR on Windows, GMP on Linux — Environment.h pulls in the header)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(FractalSharkPlatform PUBLIC ${GMP_LIBRARY})
