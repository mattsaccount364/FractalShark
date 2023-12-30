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

#include "PerturbationResultsHelpers.h"

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults : public TemplateHelpers<IterType, T, PExtras> {

public:
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class CompressionHelper<IterType, T, PExtras>;

    PerturbationResults(size_t Generation = 0, size_t LaGeneration = 0) :
        OrbitX{},
        OrbitY{},
        OrbitXLow{},
        OrbitYLow{},
        MaxRadius{},
        MaxIterations{},
        PeriodMaybeZero{},
        CompressionHelper{*this},
        FullOrbit{},
        UncompressedItersInOrbit{},
        GenerationNumber{ Generation },
        LaReference{},
        LaGenerationNumber{ LaGeneration },
        AuthoritativePrecision{},
        ReuseX{},
        ReuseY{} {

    }
        
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
        std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> laReference,
        size_t NewLaGenerationNumber) {
        LaReference = std::move(laReference);
        LaGenerationNumber = NewLaGenerationNumber;
    }

    LAReference<IterType, T, SubType, PExtras>* GetLaReference() const {
        return LaReference.get();
    }

    template<bool IncludeLA, class Other, PerturbExtras PExtras = PerturbExtras::Disable>
    void Copy(
        const PerturbationResults<IterType, Other, PExtras>& other,
        size_t NewGenerationNumber,
        size_t NewLaGenerationNumber) {
        clear();

        OrbitX = other.GetHiX();
        OrbitY = other.GetHiY();
        if constexpr (std::is_same<T, Other>::value) {
            OrbitXLow = other.GetOrbitXLow();
            OrbitYLow = other.GetOrbitYLow();
        } else {
            // TODO
            // We're copying from a different type, and we're not currently
            // going to support that.  We could just static_cast but that's
            // not really the right thing to do.
            OrbitXLow = {};
            OrbitYLow = {};
        }
        MaxRadius = (T)other.GetMaxRadius();
        MaxIterations = other.GetMaxIterations();
        PeriodMaybeZero = other.GetPeriodMaybeZero();
        // CompressionHelper = {*this}; // Nothing to do here

        FullOrbit.reserve(other.FullOrbit.size());
        UncompressedItersInOrbit = other.UncompressedItersInOrbit;
        GenerationNumber = NewGenerationNumber;

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX.reserve(other.ReuseX.size());
        ReuseY.reserve(other.ReuseY.size());

        for (size_t i = 0; i < other.FullOrbit.size(); i++) {
            if constexpr (PExtras == PerturbExtras::Bad) {
                FullOrbit.push_back({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y, other.FullOrbit[i].bad != 0 });
            }
            else if constexpr (PExtras == PerturbExtras::EnableCompression) {
                FullOrbit.push_back({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y, other.FullOrbit[i].CompressionIndex });
            }
            else {
                FullOrbit.push_back({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y });
            }
        }

        assert(UncompressedItersInOrbit == FullOrbit.size());

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX = other.ReuseX;
        ReuseY = other.ReuseY;

        if constexpr (IncludeLA) {
            if (other.GetLaReference() != nullptr) {
                LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(*other.GetLaReference());
                LaGenerationNumber = NewLaGenerationNumber;
            }
        }
    }

    template<PerturbExtras OtherBad>
    void CopySettingsWithoutOrbit(const PerturbationResults<IterType, T, OtherBad>& other) {
        OrbitX = other.GetHiX();
        OrbitY = other.GetHiY();
        OrbitXLow = other.GetOrbitXLow();
        OrbitYLow = other.GetOrbitYLow();

        MaxRadius = other.GetMaxRadius();
        MaxIterations = other.GetMaxIterations();
        PeriodMaybeZero = other.GetPeriodMaybeZero();
        // CompressionHelper = {}; // Nothing to do here

        FullOrbit = {};
        UncompressedItersInOrbit = other.UncompressedItersInOrbit;
        // GenerationNumber = other.GetGenerationNumber(); // Don't copy unless some other stuff changes

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX = other.ReuseX;
        ReuseY = other.ReuseY;

        // compression - don't copy.  Regenerate if needed.
        //if (other.GetLaReference() != nullptr) {
        //    LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(*other.GetLaReference());
        //    LaGenerationNumber = other.GetLaGenerationNumber();
        //}
    }

    void Write(const std::wstring& filename) {
        const auto orbfilename = filename + L".FullOrbit";

        std::ofstream orbfile(orbfilename, std::ios::binary);
        if (!orbfile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return;
        }

        uint64_t size = FullOrbit.size();
        orbfile.write((char*)FullOrbit.data(), sizeof(FullOrbit[0]) * size);


        const auto metafilename = filename + L".met";

        std::ofstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK);
            return;
        }

        metafile << FullOrbit.size() << std::endl;

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

        if constexpr (PExtras == PerturbExtras::Bad) {
            metafile << "PerturbExtras::Bad" << std::endl;
        } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            metafile << "PerturbExtras::EnableCompression" << std::endl;
        } else if constexpr (PExtras == PerturbExtras::Disable) {
            metafile << "PerturbExtras::Disable" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid bad.", L"", MB_OK);
            return;
        }

        metafile << OrbitX.precision() << std::endl;
        metafile << HdrToString(OrbitX) << std::endl;
        metafile << OrbitY.precision() << std::endl;
        metafile << HdrToString(OrbitY) << std::endl;
        metafile << HdrToString(MaxRadius) << std::endl;
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

            if (badstr == "PerturbExtras::Bad") {
                if constexpr (PExtras == PerturbExtras::Bad) {
                    typematch3 = true;
                }
            }
            else if (badstr == "PerturbExtras::EnableCompression") {
                if constexpr (PExtras == PerturbExtras::EnableCompression) {
                    typematch3 = true;
                }
            }
            else if (badstr == "PerturbExtras::Disable") {
                if constexpr (PExtras == PerturbExtras::Disable) {
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

            OrbitX = HighPrecision{ shiX };
        }

        {
            uint32_t prec;
            metafile >> prec;

            scoped_mpfr_precision p{prec};

            std::string shiY;
            metafile >> shiY;

            OrbitY.precision(prec);
            OrbitY = HighPrecision{ shiY };
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
                MaxRadius = T{ e, (SubType)m };
            }
            else if constexpr (
                std::is_same<T, float>::value ||
                std::is_same<T, double>::value) {
                MaxRadius = static_cast<T>(m);
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

        const auto orbfilename = filename + L".FullOrbit";

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

        FullOrbit.clear();
        FullOrbit.resize(sz);

        void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
        if (pBuf == NULL) {
            ::MessageBox(NULL, L"Failed to map view of file", L"", MB_OK);
            return false;
        }

        memcpy(FullOrbit.data(), pBuf, sz * sizeof(GPUReferenceIter<T, PExtras>));

        // TODO just use the mapped data directly, might be neat.
        UnmapViewOfFile(pBuf);
        CloseHandle(hMapFile);
        CloseHandle(hFile);

        return true;
    }

    void clear() {
        OrbitX = {};
        OrbitY = {};
        OrbitXLow = {};
        OrbitYLow = {};
        MaxRadius = {};
        MaxIterations = 0;
        PeriodMaybeZero = 0;
        // CompressionHelper = {}; // Nothing to do here

        FullOrbit.clear();
        UncompressedItersInOrbit = 0;
        GenerationNumber = 0;

        AuthoritativePrecision = false;
        ReuseX.clear();
        ReuseY.clear();

        LaReference = nullptr;
        LaGenerationNumber = 0;
    }

    template<class U = SubType>
    typename HDRFloatComplex<U> GetComplex(size_t uncompressed_index) const {
        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            return { FullOrbit[uncompressed_index].x, FullOrbit[uncompressed_index].y };
        } else {
            return CompressionHelper.GetCompressedComplex<U>(uncompressed_index);
        }
    }

    void InitReused() {
        HighPrecision Zero = 0;

        //assert(RequiresReuse());
        Zero.precision(AuthoritativeReuseExtraPrecision);

        ReuseX.push_back(Zero);
        ReuseY.push_back(Zero);
    }

    template<class T, PerturbExtras PExtras, RefOrbitCalc::ReuseMode Reuse>
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

        OrbitX = cx;
        OrbitY = cy;
        OrbitXLow = T{ cx };
        OrbitYLow = T{ cy };

        // TODO I don't get it.  Why is this here?
        // Periodicity checking results in different detected periods depending on the type,
        // e.g. HDRFloat<float> vs HDRFloat<double>.  This 2.0 here seems to compensat, but
        // makes no sense.  So what bug are we covering up here?
        if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            MaxRadius = (T)((radiusX > radiusY ? radiusX : radiusY) * 2.0);
        }
        else {
            MaxRadius = (T)(radiusX > radiusY ? radiusX : radiusY);
        }

        HdrReduce(MaxRadius);

        MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : NumIterations;

        GenerationNumber = Generation;
        FullOrbit.reserve(ReserveSize);

        // Add an empty entry at the start
        FullOrbit.push_back({});
        UncompressedItersInOrbit = 1;

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

    template<PerturbExtras PExtras, RefOrbitCalc::ReuseMode Reuse>
    void TrimResults() {
        FullOrbit.shrink_to_fit();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            ReuseX.shrink_to_fit();
            ReuseY.shrink_to_fit();
        }
    }

    template<typename = typename std::enable_if<PExtras == PerturbExtras::EnableCompression>>
    size_t GetCompressedOrbitSize() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(FullOrbit.size() <= UncompressedItersInOrbit);
        }

        return FullOrbit.size();
    }

    IterType GetCountOrbitEntries() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(FullOrbit.size() <= UncompressedItersInOrbit);
        } else {
            assert(FullOrbit.size() == UncompressedItersInOrbit);
        }

        return UncompressedItersInOrbit;
    }

    void AddUncompressedIteration(GPUReferenceIter<T, PExtras> result) {
        assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad);

        FullOrbit.push_back(result);
        UncompressedItersInOrbit++;
    }

    const GPUReferenceIter<T, PExtras>* GetOrbitData() const {
        return FullOrbit.data();
    }

    // Location of orbit
    const HighPrecision &GetHiX() const {
        return OrbitX;
    }

    // Location of orbit
    const HighPrecision &GetHiY() const {
        return OrbitY;
    }

    T GetOrbitXLow() const {
        return OrbitXLow;
    }

    T GetOrbitYLow() const {
        return OrbitYLow;
    }

    // Radius used for periodicity checking
    T GetMaxRadius() const {
        return MaxRadius;
    }

    // Used only with scaled kernels
    void SetBad(bool bad) {
        FullOrbit.back().bad = bad;
    }

    uint32_t GetAuthoritativePrecision() const {
        return AuthoritativePrecision;
    }

    IterType GetPeriodMaybeZero() const {
        return PeriodMaybeZero;
    }

    void SetPeriodMaybeZero(IterType period) {
        PeriodMaybeZero = period;
    }

    size_t GetReuseSize() const {
        assert(ReuseX.size() == ReuseY.size());
        return ReuseX.size();
    }

    void AddReusedEntry(HighPrecision x, HighPrecision y) {
        ReuseX.push_back(std::move(x));
        ReuseY.push_back(std::move(y));
    }

    const HighPrecision& GetReuseXEntry(size_t uncompressed_index) const {
        return ReuseX[uncompressed_index];
    }

    const HighPrecision& GetReuseYEntry(size_t uncompressed_index) const {
        return ReuseY[uncompressed_index];
    }

    IterType GetMaxIterations() const {
        return MaxIterations;
    }

    // For reference:
    // Used:
    //   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
    //   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
    // as a reference for the compression algorithm.
    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>
    Compress(size_t NewGenerationNumber, size_t NewLaGenerationNumber) {

        auto compressed =
            std::make_unique<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>(
                NewGenerationNumber, NewLaGenerationNumber);
        compressed->CopySettingsWithoutOrbit(*this);

        assert (FullOrbit.size() > 1);

        T zx{};
        T zy{};
        OrbitXLow = FullOrbit[1].x;
        OrbitYLow = FullOrbit[1].y;

        for (size_t i = 0; i < FullOrbit.size(); i++) {
            auto errX = zx - FullOrbit[i].x;
            auto errY = zy - FullOrbit[i].y;

            auto norm_z = FullOrbit[i].x * FullOrbit[i].x + FullOrbit[i].y * FullOrbit[i].y;
            HdrReduce(norm_z);

            auto err = (errX * errX + errY * errY) * T{ 1e16f };
            HdrReduce(err);

            if (HdrCompareToBothPositiveReducedGE(err, norm_z)) {
                zx = FullOrbit[i].x;
                zy = FullOrbit[i].y;

                compressed->FullOrbit.push_back({ FullOrbit[i].x, FullOrbit[i].y, i });
            }

            auto zx_old = zx;
            zx = zx * zx - zy * zy + OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2 } * zx_old * zy + OrbitYLow;
            HdrReduce(zy);
        }

        return compressed;
    }

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>> 
    Decompress() {

        auto compressed_orb = std::move(FullOrbit);
        auto decompressed = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
            GenerationNumber, LaGenerationNumber);
        decompressed->CopySettingsWithoutOrbit(*this);

        T zx{};
        T zy{};

        size_t compressed_index = 0;

        for (size_t i = 0; i < UncompressedItersInOrbit; i++) {
            if (compressed_index < compressed_orb.size() &&
                compressed_orb[compressed_index].CompressionIndex == i) {
                zx = compressed_orb[compressed_index].x;
                zy = compressed_orb[compressed_index].y;
                compressed_index++;
            }
            else {
                auto zx_old = zx;
                zx = zx * zx - zy * zy + OrbitXLow;
                HdrReduce(zx);
                zy = T{ 2 } *zx_old * zy + OrbitYLow;
                HdrReduce(zy);
            }

            decompressed->AddUncompressedIteration({ zx, zy });
        }

        return decompressed;
    }

private:
    HighPrecision OrbitX;
    HighPrecision OrbitY;
    T OrbitXLow;
    T OrbitYLow;
    T MaxRadius;
    IterType MaxIterations;
    IterType PeriodMaybeZero;  // Zero if not worked out

    CompressionHelper<IterType, T, PExtras> CompressionHelper;
    std::vector<GPUReferenceIter<T, PExtras>> FullOrbit;
    IterType UncompressedItersInOrbit;

    size_t GenerationNumber;

    std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> LaReference;
    size_t LaGenerationNumber;

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;
};