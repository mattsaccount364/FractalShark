add_library(FractalSharkLib STATIC)

target_sources(FractalSharkLib PRIVATE
    AutoZoomer.cpp
    BenchmarkData.cpp
    BLAS.cpp
    CrummyTest.cpp
    FeatureFinder.cpp
    FeatureFinderOrchestrator.cpp
    FeatureSummary.cpp
    Fractal.cpp
    FractalPalette.cpp
    FractalViewPresets.cpp
    ImaginaOrbit.cpp
    ItersMemoryContainer.cpp
    LAReference.cpp
    OpenGLContext.cpp
    OrbitEndpointEvaluator.cpp
    OrbitParameterPack.cpp
    PerturbationResults.cpp
    PngParallelSave.cpp
    PointZoomBBConverter.cpp
    PrecisionCalculator.cpp
    RecommendedSettings.cpp
    RefOrbitCalc.cpp
    RenderAlgorithm.cpp
    RenderThreadPool.cpp
    RenderToPng.cpp
    ScopedMpir.cpp
    Utilities.cpp

    WPngImage/lodepng.cpp
    WPngImage/WPngImage.cc
)
# stdafx.cpp is the MSVC PCH creator — excluded on CMake builds.
# GPU_Render.cu is compiled by the FractalSharkGpuLib target.

target_include_directories(FractalSharkLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# stdafx.h lives in the repo root
target_include_directories(FractalSharkLib PRIVATE
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(FractalSharkLib PUBLIC FractalSharkPlatform)
target_link_libraries(FractalSharkLib PUBLIC HpSharkFloatLib)

# GMP (MPIR on Windows, GMP on Linux — Environment.h handles the include)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(FractalSharkLib PUBLIC ${GMP_LIBRARY})

# OpenGL + GLU (gluOrtho2D used by OpenGLContext.cpp)
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
target_link_libraries(FractalSharkLib PUBLIC OpenGL::GL)
if(TARGET OpenGL::GLU)
    target_link_libraries(FractalSharkLib PUBLIC OpenGL::GLU)
else()
    find_library(GLU_LIBRARY GLU REQUIRED)
    target_link_libraries(FractalSharkLib PUBLIC ${GLU_LIBRARY})
endif()

# X11 (Linux only)
if(NOT WIN32)
    find_package(X11 REQUIRED)
    target_link_libraries(FractalSharkLib PUBLIC X11::X11)
endif()
