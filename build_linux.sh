#!/usr/bin/bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
# CMake reads CUDACXX when it configures a build directory for the first time.
if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda-13.5/bin/nvcc ]]; then
    export CUDACXX=/usr/local/cuda-13.5/bin/nvcc
fi
rm -rf ./build
for cfg in Debug Release; do
    cmakeArgs=(
        -S .
        -B "build-${cfg,,}"
        -DCMAKE_BUILD_TYPE="$cfg"
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
        -DCMAKE_CUDA_HOST_COMPILER=g++
    )

    if [ "$cfg" = "Debug" ]; then
        cmakeArgs+=(
            -DCMAKE_CXX_FLAGS_DEBUG="-g3"
            -DCMAKE_CUDA_FLAGS_DEBUG="-g -G"
        )
    else
        cmakeArgs+=(
            -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g3 -DNDEBUG"
            -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -g -lineinfo -DNDEBUG"
        )
    fi

    cmake "${cmakeArgs[@]}"
    cmake --build "build-${cfg,,}" --parallel
done
