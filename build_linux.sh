#!/usr/bin/bash
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH
rm -rf ./build
for cfg in Debug Release; do
    cmake -S . -B "build-${cfg,,}" -DCMAKE_BUILD_TYPE="$cfg" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=g++
    cmake --build "build-${cfg,,}" --parallel
done
