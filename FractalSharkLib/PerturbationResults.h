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
#include "Vectors.h"

std::wstring GetTimeAsString();

template<typename IterType, class T, PerturbExtras PExtras>
class RefOrbitCompressor;

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults : public TemplateHelpers<IterType, T, PExtras> {

public:
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class CompressionHelper<IterType, T, PExtras>;
    friend class RefOrbitCompressor<IterType, T, PExtras>;

    PerturbationResults(
        RefOrbitCalc::AddPointOptions MemoryOpt,
        size_t Generation = 0,
        size_t LaGeneration = 0) :
      
        OrbitX{},
        OrbitY{},
        OrbitXLow{},
        OrbitYLow{},
        MaxRadius{},
        MaxIterations{},
        PeriodMaybeZero{},
        CompressionErrorExp{},
        CompressionHelper{ *this },
        FullOrbit{ MemoryOpt == RefOrbitCalc::AddPointOptions::DontSave ? L"" : GetTimeAsString() + L".FullOrbit" },
        UncompressedItersInOrbit{},
        GenerationNumber{ Generation },
        LaReference{},
        LaGenerationNumber{ LaGeneration },
        AuthoritativePrecision{},
        ReuseX{},
        ReuseY{}{}

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
        OrbitXLow = static_cast<T>(Convert<HighPrecision, double>(OrbitX));
        OrbitYLow = static_cast<T>(Convert<HighPrecision, double>(OrbitY));
        MaxRadius = (T)other.GetMaxRadius();
        MaxIterations = other.GetMaxIterations();
        PeriodMaybeZero = other.GetPeriodMaybeZero();
        CompressionErrorExp = other.GetCompressionErrorExp();
        // CompressionHelper = {*this}; // Nothing to do here

        //FullOrbit.MutableCommit(other.FullOrbit.GetSize());
        UncompressedItersInOrbit = other.UncompressedItersInOrbit;
        GenerationNumber = NewGenerationNumber;

        AuthoritativePrecision = other.AuthoritativePrecision;
        ReuseX.reserve(other.ReuseX.size());
        ReuseY.reserve(other.ReuseY.size());

        for (size_t i = 0; i < other.FullOrbit.GetSize(); i++) {
            if constexpr (PExtras == PerturbExtras::Bad) {
                FullOrbit.PushBack({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y, other.FullOrbit[i].bad != 0 });
            }
            else if constexpr (PExtras == PerturbExtras::EnableCompression) {
                FullOrbit.PushBack({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y, other.FullOrbit[i].CompressionIndex });
            }
            else {
                FullOrbit.PushBack({ (T)other.FullOrbit[i].x, (T)other.FullOrbit[i].y });
            }
        }

        assert(UncompressedItersInOrbit == FullOrbit.GetSize());

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
        CompressionErrorExp = other.GetCompressionErrorExp();
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

    void Write(bool include_orbit) const {
        std::wstring filename;
        filename = FullOrbit.GetFilename();
        if (filename == L"") {
            filename = GetTimeAsString();
        }

        Write(include_orbit, filename);
    }

    void Write(bool include_orbit, std::wstring filename) const {

        if (include_orbit) {
            const auto orbfilename = filename + L".FullOrbit";

            std::ofstream orbfile(orbfilename, std::ios::binary);
            if (!orbfile.is_open()) {
                ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK | MB_APPLMODAL);
                return;
            }

            uint64_t size = FullOrbit.GetSize();
            orbfile.write((char*)FullOrbit.GetData(), sizeof(FullOrbit[0]) * size);
            orbfile.close();
        }

        const auto metafilename = filename + L".met";

        std::ofstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        metafile << FullOrbit.GetSize() << std::endl;

        if constexpr (std::is_same<IterType, uint32_t>::value) {
            metafile << "uint32_t" << std::endl;
        } else if constexpr (std::is_same<IterType, uint64_t>::value) {
            metafile << "uint64_t" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid size.", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        if constexpr (std::is_same<T, float>::value) {
            metafile << "float" << std::endl;
        } else if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
            metafile << "CudaDblflt<MattDblflt>" << std::endl;
        } else if constexpr (std::is_same<T, double>::value) {
            metafile << "double" << std::endl;
        } else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
            metafile << "HDRFloat<float>" << std::endl;
        } else if constexpr (std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
            metafile << "HDRFloat<CudaDblflt<MattDblflt>>" << std::endl;
        } else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            metafile << "HDRFloat<double>" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid type.", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        if constexpr (PExtras == PerturbExtras::Bad) {
            metafile << "PerturbExtras::Bad" << std::endl;
        } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
            metafile << "PerturbExtras::EnableCompression" << std::endl;
        } else if constexpr (PExtras == PerturbExtras::Disable) {
            metafile << "PerturbExtras::Disable" << std::endl;
        } else {
            ::MessageBox(NULL, L"Invalid bad.", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        metafile << "Precision: " << OrbitX.precision() << std::endl;
        metafile << "HighPrecisionReal: " << HdrToString(OrbitX) << std::endl;
        metafile << "Precision: " << OrbitY.precision() << std::endl;
        metafile << "HighPrecisionImaginary: " << HdrToString(OrbitY) << std::endl;
        metafile << "LowPrecisionReal: " << HdrToString(OrbitXLow) << std::endl;
        metafile << "LowPrecisionImaginary: " << HdrToString(OrbitYLow) << std::endl;
        metafile << "MaxRadius: " << HdrToString(MaxRadius) << std::endl;
        metafile << "MaxIterationsPerPixel: " << MaxIterations << std::endl;
        metafile << "Period: " << PeriodMaybeZero << std::endl;
        metafile << "CompressionErrorExponent: " << CompressionErrorExp << std::endl;
        metafile << "UncompressedIterationsInOrbit: " << UncompressedItersInOrbit << std::endl;
    }

    // This function uses CreateFileMapping and MapViewOfFile to map
    // the file into memory.  Then it loads the meta file using ifstream
    // to get the other data.
    //
    // TODO Loading a 2x32 (with or without HDR) does not work with LA enabled.
    // The code below works, but 
    // The LA table generation doesn't currently work with 2x32 and only supports 1x64
    // (which it then converts to 2x32).  This is a bug.
    bool Load(const std::wstring& filename) {
        const auto metafilename = filename + L".met";
        std::ifstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK | MB_APPLMODAL);
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
                ::MessageBox(NULL, L"Invalid size.", L"", MB_OK | MB_APPLMODAL);
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
            } else if (tstr == "CudaDblflt<MattDblflt>") {
                if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "double") {
                if constexpr (std::is_same<T, double>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "HDRFloat<float>") {
                if constexpr (std::is_same<T, HDRFloat<float>>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "HDRFloat<CudaDblflt<MattDblflt>>") {
                if constexpr (std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
                    typematch2 = true;
                }
            } else if (tstr == "HDRFloat<double>") {
                if constexpr (std::is_same<T, HDRFloat<double>>::value) {
                    typematch2 = true;
                }
            } else {
                ::MessageBox(NULL, L"Invalid type.", L"", MB_OK | MB_APPLMODAL);
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
                ::MessageBox(NULL, L"Invalid bad.", L"", MB_OK | MB_APPLMODAL);
                return false;
            }
        }

        if (!typematch1 || !typematch2 || !typematch3) {
            return false;
        }

        std::string descriptor_string_junk;

        {
            uint32_t prec;
            metafile >> descriptor_string_junk;
            metafile >> prec;

            scoped_mpfr_precision p{ prec };

            std::string shiX;
            metafile >> descriptor_string_junk;
            metafile >> shiX;

            OrbitX = HighPrecision{ shiX };
        }

        {
            uint32_t prec;
            metafile >> descriptor_string_junk;
            metafile >> prec;

            scoped_mpfr_precision p{prec};

            std::string shiY;
            metafile >> descriptor_string_junk;
            metafile >> shiY;

            OrbitY.precision(prec);
            OrbitY = HighPrecision{ shiY };
        }

        auto ss_to_hdr = [&](T &output) {
            double mantissa;
            int32_t exponent;

            metafile >> descriptor_string_junk;

            std::string maxRadiusMantissaStr;
            metafile >> maxRadiusMantissaStr;
            metafile >> maxRadiusMantissaStr;
            mantissa = std::stod(maxRadiusMantissaStr);

            std::string maxRadiusExpStr;
            metafile >> maxRadiusExpStr;
            metafile >> maxRadiusExpStr;
            exponent = std::stoi(maxRadiusExpStr);

            if constexpr (
                std::is_same<T, HDRFloat<float>>::value ||
                std::is_same<T, HDRFloat<double>>::value ||
                std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
                output = T{ exponent, (SubType)mantissa };
            }
            else if constexpr (
                std::is_same<T, float>::value ||
                std::is_same<T, double>::value ||
                std::is_same<T, CudaDblflt<MattDblflt>>::value) {
                output = static_cast<T>(mantissa);
            }
            else {
                ::MessageBox(NULL, L"Unexpected type in Load", L"", MB_OK | MB_APPLMODAL);
                return false;
            }
        };

        ss_to_hdr(OrbitXLow);
        ss_to_hdr(OrbitYLow);
        ss_to_hdr(MaxRadius);

        {
            std::string maxIterationsStr;
            metafile >> descriptor_string_junk;
            metafile >> maxIterationsStr;
            MaxIterations = (IterType)std::stoll(maxIterationsStr);
        }

        {
            std::string periodMaybeZeroStr;
            metafile >> descriptor_string_junk;
            metafile >> periodMaybeZeroStr;
            PeriodMaybeZero = (IterType)std::stoll(periodMaybeZeroStr);
        }

        {
            std::string compressionErrorStr;
            metafile >> descriptor_string_junk;
            metafile >> compressionErrorStr;
            CompressionErrorExp = static_cast<int32_t>(std::stoll(compressionErrorStr));
        }

        {
            std::string uncompressedItersInOrbitStr;
            metafile >> descriptor_string_junk;
            metafile >> uncompressedItersInOrbitStr;
            UncompressedItersInOrbit = (IterType)std::stoll(uncompressedItersInOrbitStr);
        }

        const auto orbfilename = filename + L".FullOrbit";
        FullOrbit = GrowableVector<GPUReferenceIter<T, PExtras>>(orbfilename, sz);
        if (FullOrbit.GetData() == nullptr) {
            return false;
        }

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
        CompressionErrorExp = 0;
        // CompressionHelper = {}; // Nothing to do here

        FullOrbit = {};
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
            return {
                FullOrbit[uncompressed_index].x,
                FullOrbit[uncompressed_index].y };
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

        const size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : 1'000'000;

        GenerationNumber = Generation;

        // Do not resize prematurely -- file-backed vectors only resize when needed.
        // FullOrbit.MutableCommit(ReserveSize);

        // Add an empty entry at the start
        FullOrbit.PushBack({});
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
        FullOrbit.Trim();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            ReuseX.shrink_to_fit();
            ReuseY.shrink_to_fit();
        }
    }

    template<typename = typename std::enable_if<PExtras == PerturbExtras::EnableCompression>>
    size_t GetCompressedOrbitSize() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(FullOrbit.GetSize() <= UncompressedItersInOrbit);
        }

        return FullOrbit.GetSize();
    }

    IterType GetCountOrbitEntries() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(FullOrbit.GetSize() <= UncompressedItersInOrbit);
        } else {
            assert(FullOrbit.GetSize() == UncompressedItersInOrbit);
        }

        return UncompressedItersInOrbit;
    }

    void AddUncompressedIteration(GPUReferenceIter<T, PExtras> result) {
        static_assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad, "!");

        FullOrbit.PushBack(result);
        UncompressedItersInOrbit++;
    }

    const GPUReferenceIter<T, PExtras>* GetOrbitData() const {
        return FullOrbit.GetData();
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
        FullOrbit.Back().bad = bad;
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

    int32_t GetCompressionErrorExp() const {
        return CompressionErrorExp;
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
    Compress(
        int32_t CompressionErrorExpParam,
        size_t NewGenerationNumber,
        size_t NewLaGenerationNumber) {

        constexpr bool disable_compression = false;
        const auto Two = T{ 2 };

        CompressionErrorExp = CompressionErrorExpParam;
        auto err = std::pow(10, CompressionErrorExp);
        const auto CompressionError = static_cast<T>(err);

        auto compressed =
            std::make_unique<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>(
                FullOrbit.FileBacked(), NewGenerationNumber, NewLaGenerationNumber);
        compressed->CopySettingsWithoutOrbit(*this);

        assert (FullOrbit.GetSize() > 1);

        if constexpr (disable_compression) {
            for (size_t i = 0; i < FullOrbit.GetSize(); i++) {
                compressed->FullOrbit.PushBack({ FullOrbit[i].x, FullOrbit[i].y, i });
            }

            return compressed;
        }
        else {
            T zx{};
            T zy{};

            for (size_t i = 0; i < FullOrbit.GetSize(); i++) {
                auto errX = zx - FullOrbit[i].x;
                auto errY = zy - FullOrbit[i].y;

                auto norm_z = FullOrbit[i].x * FullOrbit[i].x + FullOrbit[i].y * FullOrbit[i].y;
                HdrReduce(norm_z);

                auto err = (errX * errX + errY * errY) * CompressionError;
                HdrReduce(err);

                if (HdrCompareToBothPositiveReducedGE(err, norm_z)) {
                    zx = FullOrbit[i].x;
                    zy = FullOrbit[i].y;

                    compressed->FullOrbit.PushBack({ FullOrbit[i].x, FullOrbit[i].y, i });
                }

                auto zx_old = zx;
                zx = zx * zx - zy * zy + OrbitXLow;
                HdrReduce(zx);
                zy = Two *zx_old * zy + OrbitYLow;
                HdrReduce(zy);
            }
        }

        return compressed;
    }

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>> 
    Decompress() {

        auto compressed_orb = std::move(FullOrbit);
        auto decompressed = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
            FullOrbit.FileBacked(), GenerationNumber, LaGenerationNumber);
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
    int32_t CompressionErrorExp;

    CompressionHelper<IterType, T, PExtras> CompressionHelper;
    GrowableVector<GPUReferenceIter<T, PExtras>> FullOrbit;
    IterType UncompressedItersInOrbit;

    size_t GenerationNumber;

    std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> LaReference;
    size_t LaGenerationNumber;

    uint32_t AuthoritativePrecision;
    std::vector<HighPrecision> ReuseX;
    std::vector<HighPrecision> ReuseY;
};

template<typename IterType, class T, PerturbExtras PExtras>
class RefOrbitCompressor : public TemplateHelpers<IterType, T, PExtras> {
    template<typename IterType, class T, PerturbExtras PExtras> friend class PerturbationResults;

    using TemplateHelpers = TemplateHelpers<IterType, T, PExtras>;
    using SubType = TemplateHelpers::SubType;

    template<class LocalSubType>
    using HDRFloatComplex = TemplateHelpers::template HDRFloatComplex<LocalSubType>;

    friend class CompressionHelper<IterType, T, PExtras>;

    PerturbationResults<IterType, T, PExtras> &results;
    T zx;
    T zy;
    T Two;
    T CompressionError;
    int32_t CompressionErrorExp;
    IterTypeFull CurCompressedIndex;

public:
    RefOrbitCompressor(
        PerturbationResults<IterType, T, PExtras>& results,
        int32_t CompressionErrorExp) :
        results{ results },
        zx{results.OrbitXLow},
        zy{results.OrbitYLow},
        Two(2),
        CompressionError(static_cast<T>(std::pow(10, CompressionErrorExp))),
        CurCompressedIndex{} {
        // This code can run even if compression is disabled, but it doesn't matter.
        results.CompressionErrorExp = CompressionErrorExp;
    }

    void MaybeAddCompressedIteration(GPUReferenceIter<T, PExtras> iter) {
        // This should only run if compression is enabled.
        static_assert(PExtras == PerturbExtras::EnableCompression, "!");

        auto errX = zx - iter.x;
        auto errY = zy - iter.y;

        auto norm_z = iter.x * iter.x + iter.y * iter.y;
        HdrReduce(norm_z);

        auto err = (errX * errX + errY * errY) * CompressionError;
        HdrReduce(err);

        if (HdrCompareToBothPositiveReducedGE(err, norm_z)) {
            results.FullOrbit.PushBack(iter);

            zx = iter.x;
            zy = iter.y;

            // Corresponds to the entry just added
            CurCompressedIndex++;
            assert(CurCompressedIndex == results.FullOrbit.GetSize() - 1);
        }

        auto zx_old = zx;
        zx = zx * zx - zy * zy + results.OrbitXLow;
        HdrReduce(zx);
        zy = Two * zx_old * zy + results.OrbitYLow;
        HdrReduce(zy);

        results.UncompressedItersInOrbit++;
    }
};
