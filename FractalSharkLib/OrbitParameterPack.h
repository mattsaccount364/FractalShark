#pragma once

#include "ImaginaOrbit.h"
#include "HighPrecision.h"

struct OrbitParameterPack {
    Imagina::IMFileHeader fileHeader;
    HighPrecision orbitX;
    HighPrecision orbitY;
    uint64_t iterationLimit;
    Imagina::HRReal halfH;
    bool extendedRange;
    std::unique_ptr<std::ifstream> file;

    template<typename IterType>
    IterType GetSaturatedIterationCount() const {
        if (iterationLimit > std::numeric_limits<IterType>::max() - 1) {
            return std::numeric_limits<IterType>::max() - 1;
        }

        return static_cast<IterType>(iterationLimit);
    }

    OrbitParameterPack();
    OrbitParameterPack(
        const Imagina::IMFileHeader &fileHeader,
        HighPrecision &&orbitX,
        HighPrecision &&orbitY,
        uint64_t iterationLimit,
        Imagina::HRReal halfH,
        bool extendedRange,
        std::unique_ptr<std::ifstream> &&file);
    OrbitParameterPack(const OrbitParameterPack &other) = delete;
    OrbitParameterPack &operator=(const OrbitParameterPack &other) = delete;
    OrbitParameterPack &operator=(OrbitParameterPack &&other);
    OrbitParameterPack(OrbitParameterPack &&other);
    ~OrbitParameterPack();
};