#include "stdafx.h"
#include "OrbitParameterPack.h"

OrbitParameterPack::OrbitParameterPack() :
    fileHeader{},
    orbitX{},
    orbitY{},
    iterationLimit{},
    halfH{},
    extendedRange{},
    file{} {
}

OrbitParameterPack::OrbitParameterPack(
    const Imagina::IMFileHeader &fileHeader,
    HighPrecision &&orbitX,
    HighPrecision &&orbitY,
    uint64_t iterationLimit,
    Imagina::HRReal halfH,
    bool extendedRange,
    std::unique_ptr<std::ifstream> &&file) :
    fileHeader(fileHeader),
    orbitX(std::move(orbitX)),
    orbitY(std::move(orbitY)),
    iterationLimit(iterationLimit),
    halfH(halfH),
    extendedRange(extendedRange),
    file(std::move(file)) {
}

OrbitParameterPack &OrbitParameterPack::operator=(OrbitParameterPack &&other) {
    if (this != &other) {
        fileHeader = other.fileHeader;
        orbitX = std::move(other.orbitX);
        orbitY = std::move(other.orbitY);
        iterationLimit = other.iterationLimit;
        halfH = other.halfH;
        extendedRange = other.extendedRange;
        file = std::move(other.file);
    }

    return *this;
}

OrbitParameterPack::OrbitParameterPack(OrbitParameterPack &&other) :
    fileHeader(other.fileHeader),
    orbitX(std::move(other.orbitX)),
    orbitY(std::move(other.orbitY)),
    iterationLimit(other.iterationLimit),
    halfH(other.halfH),
    extendedRange(other.extendedRange),
    file(std::move(other.file)) {
}

OrbitParameterPack::~OrbitParameterPack() {
    if (file != nullptr) {
        file->close();
        file = nullptr;
    }
}