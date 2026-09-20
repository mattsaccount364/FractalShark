#include "stdafx.h"
#include "OrbitParameterPack.h"

OrbitParameterPack::OrbitParameterPack()
    : fileHeader{}, orbitX{}, orbitY{}, iterationLimit{}, halfH{}, extendedRange{},
      m_OrbitType{IncludedOrbit::NoOrbit}, file{}
{
}

OrbitParameterPack::OrbitParameterPack(const Imagina::IMFileHeader &fileHeader,
                                       HighPrecision &&orbitX,
                                       HighPrecision &&orbitY,
                                       uint64_t iterationLimit,
                                       Imagina::HRReal halfH,
                                       bool extendedRange,
                                       IncludedOrbit orbitType,
                                       std::unique_ptr<std::ifstream> &&file)
    : fileHeader(fileHeader), orbitX(std::move(orbitX)), orbitY(std::move(orbitY)),
      iterationLimit(iterationLimit), halfH(halfH), extendedRange(extendedRange), m_OrbitType(orbitType),
      file(std::move(file))
{
}

OrbitParameterPack &OrbitParameterPack::operator=(OrbitParameterPack &&) noexcept = default;

OrbitParameterPack::OrbitParameterPack(OrbitParameterPack &&) noexcept = default;

OrbitParameterPack::~OrbitParameterPack() = default;
