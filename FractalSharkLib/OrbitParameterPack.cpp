#include "stdafx.h"
#include "OrbitParameterPack.h"

OrbitParameterPack::OrbitParameterPack() :
    fileHeader{},
    orbitX{},
    orbitY{},
    iterationLimit{},
    halfH{},
    extendedRange{},
    m_OrbitType{ IncludedOrbit::NoOrbit },
    file{} {
}

OrbitParameterPack::OrbitParameterPack(
    const Imagina::IMFileHeader &fileHeader,
    HighPrecision &&orbitX,
    HighPrecision &&orbitY,
    uint64_t iterationLimit,
    Imagina::HRReal halfH,
    bool extendedRange,
    IncludedOrbit orbitType,
    std::unique_ptr<std::ifstream> &&file) :
    fileHeader(fileHeader),
    orbitX(std::move(orbitX)),
    orbitY(std::move(orbitY)),
    iterationLimit(iterationLimit),
    halfH(halfH),
    extendedRange(extendedRange),
    m_OrbitType(orbitType),
    file(std::move(file)) {
}

OrbitParameterPack &OrbitParameterPack::operator=(OrbitParameterPack &&other) noexcept {
    if (this != &other) {
        fileHeader = other.fileHeader;
        orbitX = std::move(other.orbitX);
        orbitY = std::move(other.orbitY);
        iterationLimit = other.iterationLimit;
        halfH = other.halfH;
        extendedRange = other.extendedRange;
        m_OrbitType = other.m_OrbitType;
        file = std::move(other.file);
    }

    return *this;
}

OrbitParameterPack::OrbitParameterPack(OrbitParameterPack &&other) noexcept :
    fileHeader(other.fileHeader),
    orbitX(std::move(other.orbitX)),
    orbitY(std::move(other.orbitY)),
    iterationLimit(other.iterationLimit),
    halfH(other.halfH),
    extendedRange(other.extendedRange),
    m_OrbitType(other.m_OrbitType),
    file(std::move(other.file)) {
}

OrbitParameterPack::~OrbitParameterPack() {
    if (file != nullptr) {
        file->close();
        file = nullptr;
    }
}