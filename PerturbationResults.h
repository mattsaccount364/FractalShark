#pragma once

#include <vector>
#include <stdint.h>
#include <fstream>
#include <type_traits>

#include "HighPrecision.h"
#include "HDRFloatComplex.h"
#include "RefOrbitCalc.h"
#include "LAReference.h"

#include "GPU_Render.h"

template<typename IterType, class T, CalcBad Bad>
class PerturbationResults {
public:
    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<
        std::is_fundamental<T>::value,
        T>::type;

    HighPrecision hiX, hiY;
    T maxRadius;
    IterType MaxIterations;
    IterType PeriodMaybeZero;  // Zero if not worked out

    std::vector<MattReferenceSingleIter<T, Bad>> orb;

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;

    std::unique_ptr<LAReference<IterType, SubType>> LaReference;

    void Write(const std::wstring& filename) {
        const auto orbfilename = filename + L".orb";

        std::ofstream orbfile(orbfilename, std::ios::binary);
        if (!orbfile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return;
        }

        uint64_t size = orb.size();
        orbfile.write((char*)orb.data(), sizeof(orb[0]) * size);


        const auto metafilename = filename + L".met";

        std::ofstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return;
        }

        metafile << orb.size() << std::endl;
        metafile << hiX.precision() << std::endl;
        metafile << HdrToString(hiX) << std::endl;
        metafile << hiY.precision() << std::endl;
        metafile << HdrToString(hiY) << std::endl;
        metafile << HdrToString(maxRadius) << std::endl;
        metafile << MaxIterations << std::endl;
        metafile << PeriodMaybeZero << std::endl;
    }

    // This function uses CreateFileMapping and MapViewOfFile to map
    // the file into memory.  Then it loads the meta file using ifstream
    // to get the other data.
    void Load(const std::wstring& filename) {
        const auto metafilename = filename + L".met";
        std::ifstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return;
        }

        size_t sz;
        metafile >> sz;

        {
            uint32_t prec;
            metafile >> prec;

            scoped_mpfr_precision p{ prec };

            std::string shiX;
            metafile >> shiX;

            hiX = HighPrecision{ shiX };
        }

        {
            uint32_t prec;
            metafile >> prec;

            scoped_mpfr_precision p{prec};

            std::string shiY;
            metafile >> shiY;

            hiY.precision(prec);
            hiY = HighPrecision{ shiY };
        }

        {
            std::string maxRadiusMantissaStr;
            metafile >> maxRadiusMantissaStr;
            metafile >> maxRadiusMantissaStr;
            auto m = std::stod(maxRadiusMantissaStr);

            std::string maxRadiusExpStr;
            metafile >> maxRadiusExpStr;
            metafile >> maxRadiusExpStr;
            auto e = std::stoi(maxRadiusExpStr);

            if constexpr (
                std::is_same<T, HDRFloat<float>>::value ||
                std::is_same<T, HDRFloat<double>>::value) {
                maxRadius = T{ e, (SubType)m };
            }
            else if constexpr (
                std::is_same<T, float>::value ||
                std::is_same<T, double>::value) {
                maxRadius = m;
            }
            else {
                ::MessageBox(NULL, L"Unexpected type in Load", L"", MB_OK);
                return;
            }
        }

        {
            std::string maxIterationsStr;
            metafile >> maxIterationsStr;
            MaxIterations = (IterType)std::stoll(maxIterationsStr);
        }

        {
            std::string periodMaybeZeroStr;
            metafile >> periodMaybeZeroStr;
            PeriodMaybeZero = (IterType)std::stoll(periodMaybeZeroStr);
        }

        const auto orbfilename = filename + L".orb";

        HANDLE hFile = CreateFile(orbfilename.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            ::MessageBox(NULL, L"Failed to open file for reading", L"", MB_OK);
            return;
        }

        HANDLE hMapFile = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapFile == NULL) {
            ::MessageBox(NULL, L"Failed to create file mapping", L"", MB_OK);
            return;
        }

        orb.clear();
        orb.resize(sz);

        void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
        if (pBuf == NULL) {
            ::MessageBox(NULL, L"Failed to map view of file", L"", MB_OK);
            return;
        }

        memcpy(orb.data(), pBuf, sz * sizeof(MattReferenceSingleIter<T, Bad>));

        // TODO just use the mapped data directly, might be neat.
        UnmapViewOfFile(pBuf);
        CloseHandle(hMapFile);
        CloseHandle(hFile);
    }

    void clear() {
        hiX = {};
        hiY = {};
        maxRadius = {};
        MaxIterations = 0;
        PeriodMaybeZero = 0;

        orb.clear();

        AuthoritativePrecision = false;
        ReuseX.clear();
        ReuseY.clear();
    }

    // TODO WTF is this for again?
    template<class Other, CalcBad Bad = CalcBad::Disable>
    void Copy(const PerturbationResults<IterType, Other, Bad>& other) {
        clear();

        //hiX = other.hiX;
        //hiY = other.hiY;

        //maxRadius = other.maxRadius;

        orb.reserve(other.orb.size());

        ReuseX.reserve(other.ReuseX.size());
        ReuseY.reserve(other.ReuseY.size());

        for (size_t i = 0; i < other.orb.size(); i++) {
            if constexpr (Bad == CalcBad::Enable) {
                orb.push_back({ (T)other.orb[i].x, (T)other.orb[i].y, other.orb[i].bad != 0 });
            }
            else {
                orb.push_back({ (T)other.orb[i].x, (T)other.orb[i].y });
            }
        }

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX = other.ReuseX;
        ReuseY = other.ReuseY;
    }

    template<class U>
    HDRFloatComplex<U> GetComplex(size_t index) const {
        return { orb[index].x, orb[index].y };
    }

    void InitReused() {
        HighPrecision Zero = 0;

        //assert(RequiresReuse());
        Zero.precision(AuthoritativeReuseExtraPrecision);

        ReuseX.push_back(Zero);
        ReuseY.push_back(Zero);
    }

    template<class T, CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void InitResults(
        const HighPrecision& cx,
        const HighPrecision& cy,
        const HighPrecision& minX,
        const HighPrecision& minY,
        const HighPrecision& maxX,
        const HighPrecision& maxY,
        IterType NumIterations,
        size_t GuessReserveSize) {
        auto radiusX = maxX - minX;
        auto radiusY = maxY - minY;

        hiX = cx;
        hiY = cy;
        maxRadius = (T)(radiusX > radiusY ? radiusX : radiusY);
        HdrReduce(maxRadius);

        MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : NumIterations;

        orb.reserve(ReserveSize);

        orb.push_back({});

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            AuthoritativePrecision = cx.precision();
            ReuseX.reserve(ReserveSize);
            ReuseY.reserve(ReserveSize);

            InitReused();
        }
        else if constexpr (Reuse == RefOrbitCalc::ReuseMode::DontSaveForReuse) {
            AuthoritativePrecision = 0;
        }
    }

    template<CalcBad Bad, RefOrbitCalc::ReuseMode Reuse>
    void TrimResults() {
        orb.shrink_to_fit();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            ReuseX.shrink_to_fit();
            ReuseY.shrink_to_fit();
        }
    }
};