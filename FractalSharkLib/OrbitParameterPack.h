#pragma once

#include "ImaginaOrbit.h"
#include "HighPrecision.h"

struct OrbitParameterPack {
    enum class IncludedOrbit {
        NoOrbit,
        OrbitIncluded
    };

    Imagina::IMFileHeader fileHeader;
    HighPrecision orbitX;
    HighPrecision orbitY;
    uint64_t iterationLimit;
    Imagina::HRReal halfH;
    bool extendedRange;
    IncludedOrbit m_OrbitType;
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
        IncludedOrbit orbitType,
        std::unique_ptr<std::ifstream> &&file);
    OrbitParameterPack(const OrbitParameterPack &other) = delete;
    OrbitParameterPack &operator=(const OrbitParameterPack &other) = delete;
    OrbitParameterPack &operator=(OrbitParameterPack &&other) noexcept;
    OrbitParameterPack(OrbitParameterPack &&other) noexcept;
    ~OrbitParameterPack();
};