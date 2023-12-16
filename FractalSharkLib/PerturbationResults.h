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
    PerturbationResults(size_t Generation = 0, size_t LaGeneration = 0) :
        hiX{},
        hiY{},
        maxRadius{},
        MaxIterations{},
        PeriodMaybeZero{},
        orb{},
        GenerationNumber{ Generation },
        LaReference{},
        LaGenerationNumber{ LaGeneration },
        AuthoritativePrecision{},
        ReuseX{},
        ReuseY{} {

    }
        
    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<
        std::is_fundamental<T>::value,
        T>::type;

    static constexpr bool IsHDR =
        std::is_same<T, ::HDRFloat<float>>::value ||
        std::is_same<T, ::HDRFloat<double>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<T, ::HDRFloat<CudaDblflt<dblflt>>>::value;

    template<class LocalSubType>
    using HDRFloatComplex =
        std::conditional<IsHDR,
        ::HDRFloatComplex<LocalSubType>,
        ::FloatComplex<LocalSubType>>::type;

    HighPrecision hiX, hiY;
    T maxRadius;
    IterType MaxIterations;
    IterType PeriodMaybeZero;  // Zero if not worked out

    std::vector<MattReferenceSingleIter<T, Bad>> orb;
private:
    size_t GenerationNumber;

    std::unique_ptr<LAReference<IterType, T, SubType>> LaReference;
    size_t LaGenerationNumber;

public:
    size_t GetGenerationNumber() const {
        return GenerationNumber;
    }
    size_t GetLaGenerationNumber() const {
        return GenerationNumber;
    }

    void ClearLaReference() {
        LaReference = nullptr;
        LaGenerationNumber = 0;
    }

    void SetLaReference(
        std::unique_ptr<LAReference<IterType, T, SubType>> laReference,
        size_t NewLaGenerationNumber) {
        LaReference = std::move(laReference);
        LaGenerationNumber = NewLaGenerationNumber;
    }

    LAReference<IterType, T, SubType>* GetLaReference() const {
        return LaReference.get();
    }

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;
    
    template<bool IncludeLA, class Other, CalcBad Bad = CalcBad::Disable>
    void Copy(
        const PerturbationResults<IterType, Other, Bad>& other,
        size_t NewGenerationNumber,
        size_t NewLaGenerationNumber) {
        clear();

        hiX = other.hiX;
        hiY = other.hiY;
        maxRadius = (T)other.maxRadius;
        MaxIterations = other.MaxIterations;
        PeriodMaybeZero = other.PeriodMaybeZero;

        orb.reserve(other.orb.size());
        GenerationNumber = NewGenerationNumber;

        AuthoritativePrecision = other.AuthoritativePrecision;
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

        if constexpr (IncludeLA) {
            if (other.GetLaReference() != nullptr) {
                LaReference = std::make_unique<LAReference<IterType, T, SubType>>(*other.GetLaReference());
                LaGenerationNumber = NewLaGenerationNumber;
            }
        }
    }

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

        if constexpr (std::is_same<IterType, uint32_t>::value) {
            metafile << "uint32_t" << std::endl;
        } else if constexpr (std::is_same<IterType, uint64_t>::value) {
            metafile << "uint64_t" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid size.", L"", MB_OK);
            return;
        }

        if constexpr (std::is_same<T, float>::value) {
            metafile << "float" << std::endl;
        } else if constexpr (std::is_same<T, double>::value) {
            metafile << "double" << std::endl;
        } else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            metafile << "HDRFloat<float>" << std::endl;
        } else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            metafile << "HDRFloat<double>" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid type.", L"", MB_OK);
            return;
        }

        if constexpr (Bad == CalcBad::Enable) {
            metafile << "CalcBad::Enable" << std::endl;
        } else if constexpr (Bad == CalcBad::Disable) {
            metafile << "CalcBad::Disable" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid bad.", L"", MB_OK);
            return;
        }

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
    bool Load(const std::wstring& filename) {
        const auto metafilename = filename + L".met";
        std::ifstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return false;
        }

        size_t sz;
        metafile >> sz;

        bool typematch1 = false;
        bool typematch2 = false;
        bool typematch3 = false;
        {
            std::string typestr;
            metafile >> typestr;

            if (typestr == "uint32_t") {
                if constexpr (std::is_same<IterType, uint32_t>::value) {
                    typematch1 = true;
                }
            } else if (typestr == "uint64_t") {
                if constexpr (std::is_same<IterType, uint64_t>::value) {
                    typematch1 = true;
                }
            } else {
                ::MessageBox(NULL, L"Invalid size.", L"", MB_OK);
                return false;
            }
        }

        {
            std::string tstr;
            metafile >> tstr;

            if (tstr == "float") {
                if constexpr (std::is_same<T, float>::value) {
                    typematch2 = true;
                }
            }
            else if (tstr == "double") {
                if constexpr (std::is_same<T, double>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "HDRFloat<float>") {
                if constexpr (std::is_same<T, HDRFloat<float>>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "HDRFloat<double>") {
                if constexpr (std::is_same<T, HDRFloat<double>>::value) {
                    typematch2 = true;
                }
            } else {
                ::MessageBox(NULL, L"Invalid type.", L"", MB_OK);
                return false;
            }
        }

        {
            std::string badstr;
            metafile >> badstr;

            if (badstr == "CalcBad::Enable") {
                if constexpr (Bad == CalcBad::Enable) {
                    typematch3 = true;
                }
            }
            else if (badstr == "CalcBad::Disable") {
                if constexpr (Bad == CalcBad::Disable) {
                    typematch3 = true;
                }
            } else {
                ::MessageBox(NULL, L"Invalid bad.", L"", MB_OK);
                return false;
            }
        }

        if (!typematch1 || !typematch2 || !typematch3) {
            return false;
        }

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
                maxRadius = static_cast<T>(m);
            }
            else {
                ::MessageBox(NULL, L"Unexpected type in Load", L"", MB_OK);
                return false;
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
            return false;
        }

        HANDLE hMapFile = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMapFile == NULL) {
            ::MessageBox(NULL, L"Failed to create file mapping", L"", MB_OK);
            return false;
        }

        orb.clear();
        orb.resize(sz);

        void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
        if (pBuf == NULL) {
            ::MessageBox(NULL, L"Failed to map view of file", L"", MB_OK);
            return false;
        }

        memcpy(orb.data(), pBuf, sz * sizeof(MattReferenceSingleIter<T, Bad>));

        // TODO just use the mapped data directly, might be neat.
        UnmapViewOfFile(pBuf);
        CloseHandle(hMapFile);
        CloseHandle(hFile);

        return true;
    }

    void clear() {
        hiX = {};
        hiY = {};
        maxRadius = {};
        MaxIterations = 0;
        PeriodMaybeZero = 0;

        orb.clear();
        GenerationNumber = 0;

        AuthoritativePrecision = false;
        ReuseX.clear();
        ReuseY.clear();

        LaReference = nullptr;
        LaGenerationNumber = 0;
    }

    // TODO this is fucked up at the moment.  Look in lareference.cpp to uncomment the template instantiations.
    // It's not clear that this is right.
    template<class U>
    HDRFloatComplex<U> GetComplex(size_t index) const {
        return { orb[index].x, orb[index].y };
    }

    //template<class U,
    //    typename std::enable_if_t<!IsHDR, bool>::type = true>
    //FloatComplex<U> GetComplex(size_t index) const {
    //    return FloatComplex<U>{ static_cast<U>(orb[index].x), static_cast<U>(orb[index].y) };
    //}

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
        size_t GuessReserveSize,
        size_t Generation) {
        auto radiusX = maxX - minX;
        auto radiusY = maxY - minY;

        hiX = cx;
        hiY = cy;

        // TODO I don't get it.  Why is this here?
        // Periodicity checking results in different detected periods depending on the type,
        // e.g. HDRFloat<float> vs HDRFloat<double>.  This 2.0 here seems to compensat, but
        // makes no sense.  So what bug are we covering up here?
        if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            maxRadius = (T)((radiusX > radiusY ? radiusX : radiusY) * 2.0);
        }
        else {
            maxRadius = (T)(radiusX > radiusY ? radiusX : radiusY);
        }

        HdrReduce(maxRadius);

        MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : NumIterations;

        GenerationNumber = Generation;
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

        LaReference = nullptr;
        LaGenerationNumber = 0;
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