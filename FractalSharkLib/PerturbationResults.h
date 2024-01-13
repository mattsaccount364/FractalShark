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
      
        m_OrbitX{},
        m_OrbitY{},
        m_OrbitXLow{},
        m_OrbitYLow{},
        m_MaxRadius{},
        m_MaxIterations{},
        m_PeriodMaybeZero{},
        m_CompressionErrorExp{},
        m_CompressionHelper{ *this },
        m_RefOrbitOptions{ MemoryOpt },
        m_FullOrbit{ MemoryOpt == RefOrbitCalc::AddPointOptions::DontSave ? L"" : GetTimeAsString() + L".FullOrbit" },
        m_UncompressedItersInOrbit{},
        m_GenerationNumber{ Generation },
        m_LaReference{},
        m_LaGenerationNumber{ LaGeneration },
        m_AuthoritativePrecision{},
        m_ReuseX{},
        m_ReuseY{}{}

    size_t GetGenerationNumber() const {
        return m_GenerationNumber;
    }
    size_t GetLaGenerationNumber() const {
        return m_GenerationNumber;
    }

    void ClearLaReference() {
        m_LaReference = nullptr;
        m_LaGenerationNumber = 0;
    }

    void SetLaReference(
        std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> laReference,
        size_t NewLaGenerationNumber) {
        m_LaReference = std::move(laReference);
        m_LaGenerationNumber = NewLaGenerationNumber;
    }

    LAReference<IterType, T, SubType, PExtras>* GetLaReference() const {
        return m_LaReference.get();
    }

    template<bool IncludeLA, class Other, PerturbExtras PExtras = PerturbExtras::Disable>
    void Copy(
        const PerturbationResults<IterType, Other, PExtras>& other,
        size_t NewGenerationNumber,
        size_t NewLaGenerationNumber) {
        clear();

        m_OrbitX = other.GetHiX();
        m_OrbitY = other.GetHiY();
        m_OrbitXLow = static_cast<T>(Convert<HighPrecision, double>(m_OrbitX));
        m_OrbitYLow = static_cast<T>(Convert<HighPrecision, double>(m_OrbitY));
        m_MaxRadius = (T)other.GetMaxRadius();
        m_MaxIterations = other.GetMaxIterations();
        m_PeriodMaybeZero = other.GetPeriodMaybeZero();
        m_CompressionErrorExp = other.GetCompressionErrorExp();
        // m_CompressionHelper = {*this}; // Nothing to do here

        //m_FullOrbit.MutableCommit(other.m_FullOrbit.GetSize());
        m_UncompressedItersInOrbit = other.m_UncompressedItersInOrbit;
        m_GenerationNumber = NewGenerationNumber;

        m_AuthoritativePrecision = other.m_AuthoritativePrecision;
        m_ReuseX.reserve(other.m_ReuseX.size());
        m_ReuseY.reserve(other.m_ReuseY.size());

        for (size_t i = 0; i < other.m_FullOrbit.GetSize(); i++) {
            if constexpr (PExtras == PerturbExtras::Bad) {
                m_FullOrbit.PushBack({ (T)other.m_FullOrbit[i].x, (T)other.m_FullOrbit[i].y, other.m_FullOrbit[i].bad != 0 });
            }
            else if constexpr (PExtras == PerturbExtras::EnableCompression) {
                m_FullOrbit.PushBack({ (T)other.m_FullOrbit[i].x, (T)other.m_FullOrbit[i].y, other.m_FullOrbit[i].CompressionIndex });
            }
            else {
                m_FullOrbit.PushBack({ (T)other.m_FullOrbit[i].x, (T)other.m_FullOrbit[i].y });
            }
        }

        assert(m_UncompressedItersInOrbit == m_FullOrbit.GetSize());

        m_AuthoritativePrecision = other.m_AuthoritativePrecision;
        m_ReuseX = other.m_ReuseX;
        m_ReuseY = other.m_ReuseY;

        if constexpr (IncludeLA) {
            if (other.GetLaReference() != nullptr) {
                m_LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(*other.GetLaReference());
                m_LaGenerationNumber = NewLaGenerationNumber;
            }
        }
    }

    template<PerturbExtras OtherBad>
    void CopySettingsWithoutOrbit(const PerturbationResults<IterType, T, OtherBad>& other) {
        m_OrbitX = other.GetHiX();
        m_OrbitY = other.GetHiY();
        m_OrbitXLow = other.GetOrbitXLow();
        m_OrbitYLow = other.GetOrbitYLow();

        m_MaxRadius = other.GetMaxRadius();
        m_MaxIterations = other.GetMaxIterations();
        m_PeriodMaybeZero = other.GetPeriodMaybeZero();
        m_CompressionErrorExp = other.GetCompressionErrorExp();
        // m_CompressionHelper = {}; // Nothing to do here

        m_FullOrbit = {};
        m_UncompressedItersInOrbit = other.m_UncompressedItersInOrbit;
        // m_GenerationNumber = other.GetGenerationNumber(); // Don't copy unless some other stuff changes

        m_AuthoritativePrecision = other.m_AuthoritativePrecision;
        m_ReuseX = other.m_ReuseX;
        m_ReuseY = other.m_ReuseY;

        // compression - don't copy.  Regenerate if needed.
        //if (other.GetLaReference() != nullptr) {
        //    m_LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(*other.GetLaReference());
        //    m_LaGenerationNumber = other.GetLaGenerationNumber();
        //}
    }

    void Write(bool include_orbit) const {
        std::wstring filename;
        filename = m_FullOrbit.GetFilename();
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

            uint64_t size = m_FullOrbit.GetSize();
            orbfile.write((char*)m_FullOrbit.GetData(), sizeof(m_FullOrbit[0]) * size);
            orbfile.close();
        }

        const auto metafilename = filename + L".met";

        std::ofstream metafile(metafilename, std::ios::binary);
        if (!metafile.is_open()) {
            ::MessageBox(NULL, L"Failed to open file for writing", L"", MB_OK | MB_APPLMODAL);
            return;
        }

        metafile << m_FullOrbit.GetSize() << std::endl;

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

        metafile << "Precision: " << m_OrbitX.precision() << std::endl;
        metafile << "HighPrecisionReal: " << HdrToString(m_OrbitX) << std::endl;
        metafile << "Precision: " << m_OrbitY.precision() << std::endl;
        metafile << "HighPrecisionImaginary: " << HdrToString(m_OrbitY) << std::endl;
        metafile << "LowPrecisionReal: " << HdrToString(m_OrbitXLow) << std::endl;
        metafile << "LowPrecisionImaginary: " << HdrToString(m_OrbitYLow) << std::endl;
        metafile << "MaxRadius: " << HdrToString(m_MaxRadius) << std::endl;
        metafile << "MaxIterationsPerPixel: " << m_MaxIterations << std::endl;
        metafile << "Period: " << m_PeriodMaybeZero << std::endl;
        metafile << "CompressionErrorExponent: " << m_CompressionErrorExp << std::endl;
        metafile << "UncompressedIterationsInOrbit: " << m_UncompressedItersInOrbit << std::endl;
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

            m_OrbitX = HighPrecision{ shiX };
        }

        {
            uint32_t prec;
            metafile >> descriptor_string_junk;
            metafile >> prec;

            scoped_mpfr_precision p{prec};

            std::string shiY;
            metafile >> descriptor_string_junk;
            metafile >> shiY;

            m_OrbitY.precision(prec);
            m_OrbitY = HighPrecision{ shiY };
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

        ss_to_hdr(m_OrbitXLow);
        ss_to_hdr(m_OrbitYLow);
        ss_to_hdr(m_MaxRadius);

        {
            std::string maxIterationsStr;
            metafile >> descriptor_string_junk;
            metafile >> maxIterationsStr;
            m_MaxIterations = (IterType)std::stoll(maxIterationsStr);
        }

        {
            std::string periodMaybeZeroStr;
            metafile >> descriptor_string_junk;
            metafile >> periodMaybeZeroStr;
            m_PeriodMaybeZero = (IterType)std::stoll(periodMaybeZeroStr);
        }

        {
            std::string compressionErrorStr;
            metafile >> descriptor_string_junk;
            metafile >> compressionErrorStr;
            m_CompressionErrorExp = static_cast<int32_t>(std::stoll(compressionErrorStr));
        }

        {
            std::string uncompressedItersInOrbitStr;
            metafile >> descriptor_string_junk;
            metafile >> uncompressedItersInOrbitStr;
            m_UncompressedItersInOrbit = (IterType)std::stoll(uncompressedItersInOrbitStr);
        }

        const auto orbfilename = filename + L".FullOrbit";
        m_FullOrbit = GrowableVector<GPUReferenceIter<T, PExtras>>(orbfilename, sz);
        if (m_FullOrbit.GetData() == nullptr) {
            return false;
        }

        return true;
    }

    void clear() {
        m_OrbitX = {};
        m_OrbitY = {};
        m_OrbitXLow = {};
        m_OrbitYLow = {};
        m_MaxRadius = {};
        m_MaxIterations = 0;
        m_PeriodMaybeZero = 0;
        m_CompressionErrorExp = 0;
        // m_CompressionHelper = {}; // Nothing to do here

        m_FullOrbit = {};
        m_UncompressedItersInOrbit = 0;
        m_GenerationNumber = 0;

        m_AuthoritativePrecision = false;
        m_ReuseX.clear();
        m_ReuseY.clear();

        m_LaReference = nullptr;
        m_LaGenerationNumber = 0;
    }

    template<class U = SubType>
    typename HDRFloatComplex<U> GetComplex(size_t uncompressed_index) const {
        if constexpr (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {
            return {
                m_FullOrbit[uncompressed_index].x,
                m_FullOrbit[uncompressed_index].y };
        } else {
            return m_CompressionHelper.GetCompressedComplex<U>(uncompressed_index);
        }
    }

    void InitReused() {
        HighPrecision Zero = 0;

        //assert(RequiresReuse());
        Zero.precision(AuthoritativeReuseExtraPrecision);

        m_ReuseX.push_back(Zero);
        m_ReuseY.push_back(Zero);
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

        m_OrbitX = cx;
        m_OrbitY = cy;
        m_OrbitXLow = T{ cx };
        m_OrbitYLow = T{ cy };

        // TODO I don't get it.  Why is this here?
        // Periodicity checking results in different detected periods depending on the type,
        // e.g. HDRFloat<float> vs HDRFloat<double>.  This 2.0 here seems to compensat, but
        // makes no sense.  So what bug are we covering up here?
        if constexpr (std::is_same<T, HDRFloat<double>>::value) {
            m_MaxRadius = (T)((radiusX > radiusY ? radiusX : radiusY) * 2.0);
        }
        else {
            m_MaxRadius = (T)(radiusX > radiusY ? radiusX : radiusY);
        }

        HdrReduce(m_MaxRadius);

        m_MaxIterations = NumIterations + 1; // +1 for push_back(0) below

        const size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : 1'000'000;

        m_GenerationNumber = Generation;

        // Do not resize prematurely -- file-backed vectors only resize when needed.
        // m_FullOrbit.MutableCommit(ReserveSize);

        // Add an empty entry at the start
        m_FullOrbit.PushBack({});
        m_UncompressedItersInOrbit = 1;

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            m_AuthoritativePrecision = cx.precision();
            m_ReuseX.reserve(ReserveSize);
            m_ReuseY.reserve(ReserveSize);

            InitReused();
        }
        else if constexpr (Reuse == RefOrbitCalc::ReuseMode::DontSaveForReuse) {
            m_AuthoritativePrecision = 0;
        }

        m_LaReference = nullptr;
        m_LaGenerationNumber = 0;
    }

    template<PerturbExtras PExtras, RefOrbitCalc::ReuseMode Reuse>
    void TrimResults() {
        m_FullOrbit.Trim();

        if constexpr (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse) {
            m_ReuseX.shrink_to_fit();
            m_ReuseY.shrink_to_fit();
        }
    }

    template<typename = typename std::enable_if<PExtras == PerturbExtras::EnableCompression>>
    size_t GetCompressedOrbitSize() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
        }

        return m_FullOrbit.GetSize();
    }

    IterType GetCountOrbitEntries() const {
        if constexpr (PExtras == PerturbExtras::EnableCompression) {
            assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
        } else {
            assert(m_FullOrbit.GetSize() == m_UncompressedItersInOrbit);
        }

        return m_UncompressedItersInOrbit;
    }

    void AddUncompressedIteration(GPUReferenceIter<T, PExtras> result) {
        static_assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad, "!");

        m_FullOrbit.PushBack(result);
        m_UncompressedItersInOrbit++;
    }

    const GPUReferenceIter<T, PExtras>* GetOrbitData() const {
        return m_FullOrbit.GetData();
    }

    // Location of orbit
    const HighPrecision &GetHiX() const {
        return m_OrbitX;
    }

    // Location of orbit
    const HighPrecision &GetHiY() const {
        return m_OrbitY;
    }

    T GetOrbitXLow() const {
        return m_OrbitXLow;
    }

    T GetOrbitYLow() const {
        return m_OrbitYLow;
    }

    // Radius used for periodicity checking
    T GetMaxRadius() const {
        return m_MaxRadius;
    }

    // Used only with scaled kernels
    void SetBad(bool bad) {
        m_FullOrbit.Back().bad = bad;
    }

    uint32_t GetAuthoritativePrecision() const {
        return m_AuthoritativePrecision;
    }

    IterType GetPeriodMaybeZero() const {
        return m_PeriodMaybeZero;
    }

    void SetPeriodMaybeZero(IterType period) {
        m_PeriodMaybeZero = period;
    }

    int32_t GetCompressionErrorExp() const {
        return m_CompressionErrorExp;
    }

    size_t GetReuseSize() const {
        assert(m_ReuseX.size() == m_ReuseY.size());
        return m_ReuseX.size();
    }

    void AddReusedEntry(HighPrecision x, HighPrecision y) {
        m_ReuseX.push_back(std::move(x));
        m_ReuseY.push_back(std::move(y));
    }

    const HighPrecision& GetReuseXEntry(size_t uncompressed_index) const {
        return m_ReuseX[uncompressed_index];
    }

    const HighPrecision& GetReuseYEntry(size_t uncompressed_index) const {
        return m_ReuseY[uncompressed_index];
    }

    IterType GetMaxIterations() const {
        return m_MaxIterations;
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

        m_CompressionErrorExp = CompressionErrorExpParam;
        auto err = std::pow(10, m_CompressionErrorExp);
        const auto CompressionError = static_cast<T>(err);

        auto compressed =
            std::make_unique<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>(
                m_FullOrbit.FileBacked(), NewGenerationNumber, NewLaGenerationNumber);
        compressed->CopySettingsWithoutOrbit(*this);

        assert (m_FullOrbit.GetSize() > 1);

        if constexpr (disable_compression) {
            for (size_t i = 0; i < m_FullOrbit.GetSize(); i++) {
                compressed->FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
            }

            return compressed;
        }
        else {
            T zx{};
            T zy{};

            for (size_t i = 0; i < m_FullOrbit.GetSize(); i++) {
                auto errX = zx - m_FullOrbit[i].x;
                auto errY = zy - m_FullOrbit[i].y;

                auto norm_z = m_FullOrbit[i].x * m_FullOrbit[i].x + m_FullOrbit[i].y * m_FullOrbit[i].y;
                HdrReduce(norm_z);

                auto err = (errX * errX + errY * errY) * CompressionError;
                HdrReduce(err);

                if (HdrCompareToBothPositiveReducedGE(err, norm_z)) {
                    zx = m_FullOrbit[i].x;
                    zy = m_FullOrbit[i].y;

                    compressed->FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
                }

                auto zx_old = zx;
                zx = zx * zx - zy * zy + m_OrbitXLow;
                HdrReduce(zx);
                zy = Two *zx_old * zy + m_OrbitYLow;
                HdrReduce(zy);
            }
        }

        return compressed;
    }

    std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>> 
    Decompress() {

        auto compressed_orb = std::move(m_FullOrbit);
        auto decompressed = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
            m_FullOrbit.FileBacked(), m_GenerationNumber, m_LaGenerationNumber);
        decompressed->CopySettingsWithoutOrbit(*this);

        T zx{};
        T zy{};

        size_t compressed_index = 0;

        for (size_t i = 0; i < m_UncompressedItersInOrbit; i++) {
            if (compressed_index < compressed_orb.size() &&
                compressed_orb[compressed_index].CompressionIndex == i) {
                zx = compressed_orb[compressed_index].x;
                zy = compressed_orb[compressed_index].y;
                compressed_index++;
            }
            else {
                auto zx_old = zx;
                zx = zx * zx - zy * zy + m_OrbitXLow;
                HdrReduce(zx);
                zy = T{ 2 } *zx_old * zy + m_OrbitYLow;
                HdrReduce(zy);
            }

            decompressed->AddUncompressedIteration({ zx, zy });
        }

        return decompressed;
    }

private:
    HighPrecision m_OrbitX;
    HighPrecision m_OrbitY;
    T m_OrbitXLow;
    T m_OrbitYLow;
    T m_MaxRadius;
    IterType m_MaxIterations;
    IterType m_PeriodMaybeZero;  // Zero if not worked out
    int32_t m_CompressionErrorExp;
    RefOrbitCalc::AddPointOptions m_RefOrbitOptions;

    CompressionHelper<IterType, T, PExtras> m_CompressionHelper;
    GrowableVector<GPUReferenceIter<T, PExtras>> m_FullOrbit;
    IterType m_UncompressedItersInOrbit;

    size_t m_GenerationNumber;

    std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> m_LaReference;
    size_t m_LaGenerationNumber;

    uint32_t m_AuthoritativePrecision;
    std::vector<HighPrecision> m_ReuseX;
    std::vector<HighPrecision> m_ReuseY;
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
        zx{results.m_OrbitXLow},
        zy{results.m_OrbitYLow},
        Two(2),
        CompressionError(static_cast<T>(std::pow(10, CompressionErrorExp))),
        CurCompressedIndex{} {
        // This code can run even if compression is disabled, but it doesn't matter.
        results.m_CompressionErrorExp = CompressionErrorExp;
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
            results.m_FullOrbit.PushBack(iter);

            zx = iter.x;
            zy = iter.y;

            // Corresponds to the entry just added
            CurCompressedIndex++;
            assert(CurCompressedIndex == results.m_FullOrbit.GetSize() - 1);
        }

        auto zx_old = zx;
        zx = zx * zx - zy * zy + results.m_OrbitXLow;
        HdrReduce(zx);
        zy = Two * zx_old * zy + results.m_OrbitYLow;
        HdrReduce(zy);

        results.m_UncompressedItersInOrbit++;
    }
};
