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
