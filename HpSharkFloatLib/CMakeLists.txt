set(HPSHARK_GENERATED_INST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated_inst)

set(HPSHARK_GENERATED_CU_FILES
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_AddKernels_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_AddKernels_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_AddKernels_Dbl.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkFloat_Conversions_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkFloat_Conversions_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkFloat_Conversions_Dbl.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkReference_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkReference_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_HpSharkReference_Dbl.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_MultiplyNTT_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_MultiplyNTT_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_MultiplyNTT_Dbl.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_ReferenceGpuLoop_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_ReferenceGpuLoop_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_ReferenceGpuLoop_Dbl.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_SharkNTT_Primitives_P.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_SharkNTT_Primitives_NR.cu
    ${HPSHARK_GENERATED_INST_DIR}/SharkExplicitInstantiate_SharkNTT_Primitives_Dbl.cu
)

add_library(HpSharkFloatLib STATIC)

target_sources(HpSharkFloatLib PRIVATE
    HighPrecision.cpp
    LAParameters.cpp
    MpirSerialization.cpp
    AbortMonitor.cpp
    BenchmarkTimer.cpp
    LaunchParamsCalculator.cpp
    TestVerbose.cpp
    ${HPSHARK_GENERATED_CU_FILES}
)

if(WIN32)
    target_sources(HpSharkFloatLib PRIVATE
        Vectors.cpp
        WorkaroundVcxAnnoyingCudaBug.cpp
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

# Plain .cpp files that include <cuda_runtime.h> (e.g. LaunchParamsCalculator.cpp,
# AbortMonitor.cpp) need the CUDA toolkit include dir since they are compiled by
# the C++ compiler, not nvcc.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(HpSharkFloatLib PUBLIC CUDA::cudart_static)

# Mirror the CUDA flags used by FractalSharkGpuLib.
target_compile_options(HpSharkFloatLib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--threads=0>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>
)

target_link_libraries(HpSharkFloatLib PUBLIC FractalSharkPlatform)

# GMP (MPIR on Windows, GMP on Linux — Environment.h handles the include)
find_library(GMP_LIBRARY gmp REQUIRED)
target_link_libraries(HpSharkFloatLib PUBLIC ${GMP_LIBRARY})

# MpirSerialization.cpp (Linux branch) uses mpz_export / mpz_import to emit
# the MPIR-compatible raw stream format. Fail configure clearly if the GMP
# headers don't expose these symbols so a silent no-op rebuild is impossible.
if(NOT WIN32)
    include(CheckSymbolExists)
    check_symbol_exists(mpz_export gmp.h HAVE_MPZ_EXPORT)
    check_symbol_exists(mpz_import gmp.h HAVE_MPZ_IMPORT)
    check_symbol_exists(mpz_out_raw gmp.h HAVE_MPZ_OUT_RAW)
    if(NOT HAVE_MPZ_EXPORT OR NOT HAVE_MPZ_IMPORT OR NOT HAVE_MPZ_OUT_RAW)
        message(FATAL_ERROR
            "GMP headers at 'gmp.h' do not expose mpz_export / mpz_import / "
            "mpz_out_raw. MpirSerialization requires these on Linux. Install "
            "libgmp-dev (or equivalent) >= 4.1.")
    endif()
endif()

# std::stacktrace on libstdc++ (used by Exceptions.cpp / Callstacks.cpp).
# libstdc++ 13 ships the impl in libstdc++_libbacktrace; libstdc++ 14
# moved it into libstdc++exp. Link whichever one the active toolchain
# provides.
if(NOT WIN32)
    find_library(STDCXX_STACKTRACE_LIB
        NAMES stdc++exp stdc++_libbacktrace
        PATHS /usr/lib/gcc/x86_64-linux-gnu/14
              /usr/lib/gcc/x86_64-linux-gnu/13
        NO_DEFAULT_PATH
    )
    if(NOT STDCXX_STACKTRACE_LIB)
        message(FATAL_ERROR "Could not find libstdc++exp or libstdc++_libbacktrace "
                            "(needed by std::stacktrace). Install libstdc++-14-dev "
                            "or libstdc++-13-dev.")
    endif()
    target_link_libraries(HpSharkFloatLib PUBLIC ${STDCXX_STACKTRACE_LIB})
endif()
