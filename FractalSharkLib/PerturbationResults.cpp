#include "stdafx.h"

#include "Fractal.h"
#include "PerturbationResults.h"


// Returns the current time as a string
std::wstring GetTimeAsString(size_t generation_number) {
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y-%m-%d-%H-%M-%S"); // HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    auto res = oss.str();

    std::wstring wide;
    std::transform(res.begin(), res.end(), std::back_inserter(wide), [](char c) {
        return (wchar_t)c;
        });

    wide += L"-" + std::to_wstring(generation_number);

    return wide;
}

template<typename IterType, class T, PerturbExtras PExtras>
std::wstring PerturbationResults<IterType, T, PExtras>::GenBaseFilename(size_t generation_number) const {
    return GenBaseFilename(GetRefOrbitOptions(), generation_number);
}

template<typename IterType, class T, PerturbExtras PExtras>
std::wstring PerturbationResults<IterType, T, PExtras>::GenBaseFilename(
    AddPointOptions add_point_options,
    size_t generation_number) const {

    if (add_point_options == AddPointOptions::DontSave) {
        return L"";
    }

    return GetTimeAsString(generation_number);
}

template<typename IterType, class T, PerturbExtras PExtras>
std::wstring PerturbationResults<IterType, T, PExtras>::GenFilename(
    GrowableVectorTypes Type,
    std::wstring optional_suffix,
    bool add_additional_suffix) const {

    if (GetRefOrbitOptions() == AddPointOptions::DontSave) {
        return L"";
    }

    size_t counter = 1;
    std::wstring result;
    do {
        result = m_BaseFilename + optional_suffix + GetFileExtension(Type);
        if (!Utilities::FileExists(result.c_str())) {
            break;
        }

        if (add_additional_suffix) {
            optional_suffix += L" - " + std::to_wstring(counter);
        }

        counter++;
    } while (add_additional_suffix);

    return result;
}

template<typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras>::PerturbationResults(
    std::wstring base_filename,
    AddPointOptions add_point_options,
    size_t Generation) :

    m_OrbitX{},
    m_OrbitY{},
    m_OrbitXStr{},
    m_OrbitYStr{},
    m_OrbitXLow{},
    m_OrbitYLow{},
    m_MaxRadius{},
    m_MaxRadiusHigh{},
    m_MaxIterations{},
    m_PeriodMaybeZero{},
    m_CompressionErrorExp{},
    m_IntermediateCompressionErrorExp{},
    m_RefOrbitOptions{ add_point_options },
    m_BaseFilename{ base_filename },
    m_MetaFileHandle{ INVALID_HANDLE_VALUE },
    m_FullOrbit{ },
    m_UncompressedItersInOrbit{},
    m_GenerationNumber{ Generation },
    m_LaReference{},
    m_AuthoritativePrecisionInBits{},
    m_ReuseX{},
    m_ReuseY{},
    m_ReuseIndices{},
    m_ReuseAllocations{},
    m_BenchmarkOrbit{},
    m_DeltaPrecisionCached{},
    m_ExtraPrecisionCached{} {

    if (add_point_options == AddPointOptions::OpenExistingWithSave) {
        return;
    }

    MapExistingFiles();
}

template<typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras>::PerturbationResults(
    AddPointOptions add_point_options,
    size_t Generation)
    : PerturbationResults{
        GenBaseFilename(add_point_options, Generation),
        add_point_options,
        Generation } {
}

template<typename IterType, class T, PerturbExtras PExtras>
PerturbationResults<IterType, T, PExtras>::~PerturbationResults() {
    // Clear these first to ensure the memory is freed before the 
    // bump allocator is destroyed.
    m_ReuseX.clear();
    m_ReuseY.clear();
    m_ReuseIndices.clear();

    CloseMetaFileIfOpen();
}

//template<typename IterType, class T, PerturbExtras PExtras>
//PerturbationResults<IterType, T, PExtras>::PerturbationResults(PerturbationResults&& other)
//    : m_GenerationNumber{ other.m_GenerationNumber } {
//    *this = std::move(other);
//
//    other.m_MetaFileHandle = INVALID_HANDLE_VALUE;
//}

template<typename IterType, class T, PerturbExtras PExtras>
size_t PerturbationResults<IterType, T, PExtras>::GetGenerationNumber() const {
    return m_GenerationNumber;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::ClearLaReference() {
    m_LaReference = nullptr;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SetLaReference(
    std::unique_ptr<LAReference<IterType, T, SubType, PExtras>> laReference) {
    m_LaReference = std::move(laReference);
}

template<typename IterType, class T, PerturbExtras PExtras>
LAReference<IterType, T, typename PerturbationResults<IterType, T, PExtras>::SubType, PExtras> *
PerturbationResults<IterType, T, PExtras>::GetLaReference() const {
    return m_LaReference.get();
}

template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PExtras>>
PerturbationResults<IterType, T, PExtras>::CopyPerturbationResults(
    AddPointOptions add_point_options,
    size_t new_generation_number) {

    auto new_ptr = std::make_unique<PerturbationResults<IterType, T, PExtras>>(
        add_point_options,
        new_generation_number);
    new_ptr->CopyPerturbationResults<true, T, PExtras>(*this);
    return new_ptr;
}

template<typename IterType, class T, PerturbExtras PExtras>
template<bool IncludeLA, class Other, PerturbExtras PExtrasOther>
void PerturbationResults<IterType, T, PExtras>::CopyFullOrbitVector(
    const PerturbationResults<IterType, Other, PExtrasOther> &other) {

    m_FullOrbit.MutableResize(other.m_FullOrbit.GetSize());

    // Split other.m_FullOrbit across multiple threads.
    // Use std::hardware_concurrency() to determine the number of threads.
    // Each thread will get a range of indices to copy.
    // Each thread will copy the range of indices to m_FullOrbit.
    const auto workPerThread = 1'000'000;
    const auto altNumThreads = other.m_FullOrbit.GetSize() / workPerThread;
    const auto maxThreads = std::thread::hardware_concurrency();
    const auto numThreadsMaybeZero = altNumThreads > maxThreads ? maxThreads : altNumThreads;
    const auto numThreads = numThreadsMaybeZero == 0 ? 1 : numThreadsMaybeZero;
    auto numElementsPerThread = other.m_FullOrbit.GetSize() / numThreads;

    auto oneThread = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
            if constexpr (PExtras == PerturbExtras::Bad) {
                m_FullOrbit[i] = GPUReferenceIter<T, PExtras>{
                    (T)other.m_FullOrbit[i].x,
                    (T)other.m_FullOrbit[i].y,
                    other.m_FullOrbit[i].bad != 0
                };
            } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
                m_FullOrbit[i] = GPUReferenceIter<T, PExtras>{
                    (T)other.m_FullOrbit[i].x,
                    (T)other.m_FullOrbit[i].y,
                    other.m_FullOrbit[i].CompressionIndex
                };
            } else {
                m_FullOrbit[i] = GPUReferenceIter<T, PExtras>{
                    (T)other.m_FullOrbit[i].x,
                    (T)other.m_FullOrbit[i].y
                };
            }
        }
        };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < numThreads; i++) {
        size_t start = i * numElementsPerThread;
        size_t end = (i + 1) * numElementsPerThread;

        if (i == numThreads - 1) {
            end = other.m_FullOrbit.GetSize();
        }

        threads.push_back(std::thread(oneThread, start, end));
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
template<bool IncludeLA, class Other, PerturbExtras PExtrasOther>
void PerturbationResults<IterType, T, PExtras>::CopyPerturbationResults(
    const PerturbationResults<IterType, Other, PExtrasOther> &other) {

    m_OrbitX = other.GetHiX();
    m_OrbitY = other.GetHiY();
    m_OrbitXStr = other.GetHiXStr();
    m_OrbitYStr = other.GetHiYStr();
    m_OrbitXLow = static_cast<T>(Convert<HighPrecision, double>(m_OrbitX));
    m_OrbitYLow = static_cast<T>(Convert<HighPrecision, double>(m_OrbitY));
    m_MaxRadius = (T)other.GetMaxRadius();
    m_MaxRadiusHigh = other.GetMaxRadiusHigh();
    m_MaxIterations = other.GetMaxIterations();
    m_PeriodMaybeZero = other.GetPeriodMaybeZero();
    m_CompressionErrorExp = other.GetCompressionErrorExp();
    m_IntermediateCompressionErrorExp = other.GetIntermediateCompressionErrorExp();

    // m_RefOrbitOptions // Unchanged
    m_BaseFilename = GenBaseFilename(m_GenerationNumber);
    m_MetaFileHandle = INVALID_HANDLE_VALUE;

    m_FullOrbit = { GetRefOrbitOptions(), GenFilename(GrowableVectorTypes::GPUReferenceIter) };
    m_FullOrbit.MutableResize(other.m_FullOrbit.GetSize(), 0);
    m_UncompressedItersInOrbit = other.m_UncompressedItersInOrbit;

    m_AuthoritativePrecisionInBits = other.m_AuthoritativePrecisionInBits;

    // TODO: for reuse case, we don't copy the allocations.  This is a bug
    // but we shouldn't hit it in practice?  I think?
    m_ReuseX = {};
    m_ReuseY = {};
    m_ReuseIndices = {};
    m_ReuseAllocations = nullptr;

    CopyFullOrbitVector<IncludeLA, Other, PExtrasOther>(other);

    assert(m_UncompressedItersInOrbit == m_FullOrbit.GetSize());

    m_BenchmarkOrbit = other.m_BenchmarkOrbit;

    m_DeltaPrecisionCached = 0;
    m_ExtraPrecisionCached = 0;

    if constexpr (IncludeLA) {
        if (other.GetLaReference() != nullptr) {
            m_LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(
                GetRefOrbitOptions(),
                GenFilename(GrowableVectorTypes::LAInfoDeep),
                GenFilename(GrowableVectorTypes::LAStageInfo));
            m_LaReference->CopyLAReference(*other.GetLaReference());
        }
    }
}

// Explicitly instantiate CopyPerturbationResults for all types
#define InstantiateCopyPerturbationResultsLA(T, OtherT, PExtras, PExtrasOther) \
    template void PerturbationResults<uint32_t, T, PExtras>::CopyPerturbationResults<true, OtherT, PExtrasOther>( \
        const PerturbationResults<uint32_t, OtherT, PExtrasOther>& other); \
    template void PerturbationResults<uint64_t, T, PExtras>::CopyPerturbationResults<true, OtherT, PExtrasOther>( \
        const PerturbationResults<uint64_t, OtherT, PExtrasOther>& other);

#define InstantiateCopyPerturbationResultsNoLA(T, OtherT, PExtras, PExtrasOther) \
    template void PerturbationResults<uint32_t, T, PExtras>::CopyPerturbationResults<false, OtherT, PExtrasOther>(\
        const PerturbationResults<uint32_t, OtherT, PExtrasOther>& other); \
    template void PerturbationResults<uint64_t, T, PExtras>::CopyPerturbationResults<false, OtherT, PExtrasOther>(\
        const PerturbationResults<uint64_t, OtherT, PExtrasOther>& other)

#define InstantiateCopyPerturbationResults(T, OtherT, PExtras, PExtrasOther) \
    InstantiateCopyPerturbationResultsLA(T, OtherT, PExtras, PExtrasOther); \
    InstantiateCopyPerturbationResultsNoLA(T, OtherT, PExtras, PExtrasOther)

InstantiateCopyPerturbationResults(CudaDblflt<MattDblflt>, double, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResults(HDRFloat<CudaDblflt<MattDblflt>>, HDRFloat<double>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResults(CudaDblflt<MattDblflt>, double, PerturbExtras::EnableCompression, PerturbExtras::EnableCompression);
InstantiateCopyPerturbationResults(HDRFloat<CudaDblflt<MattDblflt>>, HDRFloat<double>, PerturbExtras::EnableCompression, PerturbExtras::EnableCompression);
InstantiateCopyPerturbationResultsLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResultsLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::EnableCompression, PerturbExtras::EnableCompression);
InstantiateCopyPerturbationResults(float, double, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResults(float, double, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResults(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResults(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResultsNoLA(float, HDRFloat<float>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResultsNoLA(float, HDRFloat<float>, PerturbExtras::Disable, PerturbExtras::Disable);

InstantiateCopyPerturbationResultsNoLA(float, double, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(float, double, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(float, HDRFloat<float>, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(float, HDRFloat<float>, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(double, HDRFloat<double>, PerturbExtras::Bad, PerturbExtras::Bad);
InstantiateCopyPerturbationResultsNoLA(double, HDRFloat<double>, PerturbExtras::Bad, PerturbExtras::Bad);


template<typename IterType, class T, PerturbExtras PExtras>
template<PerturbExtras OtherBad>
void PerturbationResults<IterType, T, PExtras>::CopySettingsWithoutOrbit(const PerturbationResults<IterType, T, OtherBad> &other) {
    m_OrbitX = other.GetHiX();
    m_OrbitY = other.GetHiY();
    m_OrbitXStr = other.GetHiXStr();
    m_OrbitYStr = other.GetHiYStr();
    m_OrbitXLow = other.GetOrbitXLow();
    m_OrbitYLow = other.GetOrbitYLow();
    m_MaxRadius = other.GetMaxRadius();
    m_MaxRadiusHigh = other.GetMaxRadiusHigh();
    m_MaxIterations = other.GetMaxIterations();
    m_PeriodMaybeZero = other.GetPeriodMaybeZero();
    m_CompressionErrorExp = other.GetCompressionErrorExp();
    m_IntermediateCompressionErrorExp = other.GetIntermediateCompressionErrorExp();
    m_RefOrbitOptions = other.GetRefOrbitOptions();
    m_BaseFilename = GenBaseFilename(m_GenerationNumber);
    m_MetaFileHandle = INVALID_HANDLE_VALUE;

    m_FullOrbit = {};
    m_UncompressedItersInOrbit = other.m_UncompressedItersInOrbit;

    m_AuthoritativePrecisionInBits = other.m_AuthoritativePrecisionInBits;

    // Not supported
    m_ReuseX = {};
    m_ReuseY = {};
    m_ReuseAllocations = nullptr;

    m_BenchmarkOrbit = other.m_BenchmarkOrbit;

    m_DeltaPrecisionCached = other.m_DeltaPrecisionCached;
    m_ExtraPrecisionCached = other.m_ExtraPrecisionCached;

    // compression - don't copy LA data.  Regenerate if needed.
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::WriteMetadata() const {
    // In this case, we want to close and delete the file if it's open.
    // The code after this function will open the new file for delete.
    CloseMetaFileIfOpen();

    std::ofstream metafile(GenFilename(GrowableVectorTypes::Metadata), std::ios::binary);
    if (!metafile.is_open()) {
        ::MessageBox(nullptr, L"Failed to open file for writing 2", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    // Write a version number
    metafile << Version << std::endl;

    metafile << m_FullOrbit.GetSize() << std::endl;

    if constexpr (std::is_same<IterType, uint32_t>::value) {
        metafile << "uint32_t" << std::endl;
    } else if constexpr (std::is_same<IterType, uint64_t>::value) {
        metafile << "uint64_t" << std::endl;
    } else {
        ::MessageBox(nullptr, L"Invalid size.", L"", MB_OK | MB_APPLMODAL);
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
        ::MessageBox(nullptr, L"Invalid type.", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    if constexpr (PExtras == PerturbExtras::Bad) {
        metafile << "PerturbExtras::Bad" << std::endl;
    } else if constexpr (PExtras == PerturbExtras::EnableCompression) {
        metafile << "PerturbExtras::EnableCompression" << std::endl;
    } else if constexpr (PExtras == PerturbExtras::Disable) {
        metafile << "PerturbExtras::Disable" << std::endl;
    } else {
        ::MessageBox(nullptr, L"Invalid bad.", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    metafile << "PrecisionInBits: " << m_OrbitX.precisionInBits() << std::endl;
    metafile << "HighPrecisionReal: " << HdrToString<true>(m_OrbitX) << std::endl;
    metafile << "PrecisionInBits: " << m_OrbitY.precisionInBits() << std::endl;
    metafile << "HighPrecisionImaginary: " << HdrToString<true>(m_OrbitY) << std::endl;
    metafile << "LowPrecisionReal: " << HdrToString<true>(m_OrbitXLow) << std::endl;
    metafile << "LowPrecisionImaginary: " << HdrToString<true>(m_OrbitYLow) << std::endl;
    metafile << "MaxRadius: " << HdrToString<true>(m_MaxRadius) << std::endl;
    // don't bother with m_MaxRadiusHigh
    metafile << "MaxIterationsPerPixel: " << m_MaxIterations << std::endl;
    metafile << "Period: " << m_PeriodMaybeZero << std::endl;
    metafile << "CompressionErrorExponent: " << m_CompressionErrorExp << std::endl;
    metafile << "IntermediateCompressionErrorExponent: " << m_IntermediateCompressionErrorExp << std::endl;
    metafile << "UncompressedIterationsInOrbit: " << m_UncompressedItersInOrbit << std::endl;

    if (m_LaReference != nullptr &&
        m_LaReference->IsValid()) {
        bool ret = m_LaReference->WriteMetadata(metafile);
        if (!ret) {
            ::MessageBox(nullptr, L"Failed to write LA metadata.", L"", MB_OK | MB_APPLMODAL);
            return;
        }
    }

    metafile.close();

    MaybeOpenMetaFileForDelete();
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::MaybeOpenMetaFileForDelete() const {
    // This call can fail but that should be OK
    if (GetRefOrbitOptions() == AddPointOptions::EnableWithoutSave &&
        m_MetaFileHandle == INVALID_HANDLE_VALUE) {

        const auto attributes = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE;
        m_MetaFileHandle = CreateFile(
            GenFilename(GrowableVectorTypes::Metadata).c_str(),
            0, // no read or write
            FILE_SHARE_DELETE,
            nullptr,
            OPEN_EXISTING,
            attributes,
            nullptr);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
bool PerturbationResults<IterType, T, PExtras>::ReadMetadata() {
    std::ifstream metafile(GenFilename(GrowableVectorTypes::Metadata), std::ios::binary);
    if (!metafile.is_open()) {
        ::MessageBox(nullptr, L"Failed to open file for reading 1", L"", MB_OK | MB_APPLMODAL);
        return false;
    }

    // Read version number
    std::string version;
    metafile >> version;
    if (version != Version) {
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
            ::MessageBox(nullptr, L"Invalid size.", L"", MB_OK | MB_APPLMODAL);
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
            ::MessageBox(nullptr, L"Invalid type.", L"", MB_OK | MB_APPLMODAL);
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
        } else if (badstr == "PerturbExtras::EnableCompression") {
            if constexpr (PExtras == PerturbExtras::EnableCompression) {
                typematch3 = true;
            }
        } else if (badstr == "PerturbExtras::Disable") {
            if constexpr (PExtras == PerturbExtras::Disable) {
                typematch3 = true;
            }
        } else {
            ::MessageBox(nullptr, L"Invalid bad.", L"", MB_OK | MB_APPLMODAL);
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

        MPIRPrecision p{ prec };

        std::string shiX;
        metafile >> descriptor_string_junk;
        metafile >> shiX;

        m_OrbitX.precisionInBits(prec);
        m_OrbitX = HighPrecision{ shiX };
        m_OrbitXStr = shiX;
    }

    {
        uint32_t prec;
        metafile >> descriptor_string_junk;
        metafile >> prec;

        MPIRPrecision p{ prec };

        std::string shiY;
        metafile >> descriptor_string_junk;
        metafile >> shiY;

        m_OrbitY.precisionInBits(prec);
        m_OrbitY = HighPrecision{ shiY };
        m_OrbitYStr = shiY;
    }

    HdrFromIfStream<true, T, SubType>(m_OrbitXLow, metafile);
    HdrFromIfStream<true, T, SubType>(m_OrbitYLow, metafile);
    HdrFromIfStream<true, T, SubType>(m_MaxRadius, metafile);
    // don't bother with m_MaxRadiusHigh

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
        std::string intermediateCompressionErrorStr;
        metafile >> descriptor_string_junk;
        metafile >> intermediateCompressionErrorStr;
        m_IntermediateCompressionErrorExp = static_cast<int32_t>(std::stoll(intermediateCompressionErrorStr));
    }

    {
        std::string uncompressedItersInOrbitStr;
        metafile >> descriptor_string_junk;
        metafile >> uncompressedItersInOrbitStr;
        m_UncompressedItersInOrbit = (IterType)std::stoll(uncompressedItersInOrbitStr);
    }

    MapExistingFiles();

    if (m_LaReference != nullptr) {
        m_LaReference->ReadMetadata(metafile);

        if (!m_LaReference->IsValid()) {
            m_LaReference = nullptr;
        }
    }

    // Sometimes the mapped file is a bit bigger.  Just set it to the right
    // number of elements.
    m_FullOrbit.MutableResize(sz, sz);

    return true;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::InitReused() {
    HighPrecision Zero = 0;

    //assert(RequiresReuse());
    Zero.precisionInBits(AuthoritativeReuseExtraPrecisionInBits);

    m_ReuseX.push_back(Zero);
    m_ReuseY.push_back(Zero);
    m_ReuseIndices.push_back(0);
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::InitResults(
    RefOrbitCalc::ReuseMode Reuse,
    const HighPrecision &cx,
    const HighPrecision &cy,
    const HighPrecision &minX,
    const HighPrecision &minY,
    const HighPrecision &maxX,
    const HighPrecision &maxY,
    IterType NumIterations,
    size_t GuessReserveSize) {

    m_BenchmarkOrbit.StartTimer();

    auto radiusX = maxX - minX;
    auto radiusY = maxY - minY;

    m_OrbitX = cx;
    m_OrbitY = cy;
    m_OrbitXStr = HdrToString<false>(cx);
    m_OrbitYStr = HdrToString<false>(cy);
    m_OrbitXLow = T{ cx };
    m_OrbitYLow = T{ cy };

    // TODO I don't get it.  Why is this here?
    // Periodicity checking results in different detected periods depending on the type,
    // e.g. HDRFloat<float> vs HDRFloat<double>.  This 2.0 here seems to compensat, but
    // makes no sense.  So what bug are we covering up here?
    if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        m_MaxRadiusHigh = (radiusX > radiusY ? radiusX : radiusY) * HighPrecision{ 2.0 };
        m_MaxRadius = T(m_MaxRadiusHigh);
    } else {
        m_MaxRadiusHigh = radiusX > radiusY ? radiusX : radiusY;
        m_MaxRadius = T(m_MaxRadiusHigh);
    }

    HdrReduce(m_MaxRadius);

    m_MaxIterations = NumIterations + 1; // +1 for push_back(0) below

    const size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : 1'000'000;

    // Do not resize prematurely -- file-backed vectors only resize when needed.
    m_FullOrbit.MutableResize(ReserveSize, 0);

    // Add an empty entry at the start
    m_FullOrbit.PushBack({});
    m_UncompressedItersInOrbit = 1;

    if (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
        m_AuthoritativePrecisionInBits =
            PrecisionCalculator::GetPrecision(minX, minY, maxX, maxY, true);
        m_ReuseX.reserve(ReserveSize);
        m_ReuseY.reserve(ReserveSize);
        m_ReuseAllocations = nullptr;
    } else if (Reuse == RefOrbitCalc::ReuseMode::DontSaveForReuse) {
        m_AuthoritativePrecisionInBits = 0;
    }

    m_LaReference = nullptr;
}

//#define InstantiatePerturbationResultFunction(T, PExtras, AdditionalText) \
//    template class PerturbationResults<uint32_t, T, PExtras> AdditionalText; \
//    template class PerturbationResults<uint64_t, T, PExtras> AdditionalText;
//
//#define InstantiateMemberFunction(AdditionalText) \
//InstantiatePerturbationResultFunction(float, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(float, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(float, PerturbExtras::EnableCompression, AdditionalText); \
// \
//InstantiatePerturbationResultFunction(double, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(double, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(double, PerturbExtras::EnableCompression, AdditionalText); \
// \
//InstantiatePerturbationResultFunction(CudaDblflt<MattDblflt>, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(CudaDblflt<MattDblflt>, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression, AdditionalText); \
// \
//InstantiatePerturbationResultFunction(HDRFloat<float>, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<float>, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<float>, PerturbExtras::EnableCompression, AdditionalText); \
// \
//InstantiatePerturbationResultFunction(HDRFloat<double>, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<double>, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<double>, PerturbExtras::EnableCompression, AdditionalText); \
// \
//InstantiatePerturbationResultFunction(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad, AdditionalText); \
//InstantiatePerturbationResultFunction(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression, AdditionalText); \
//
//InstantiateMemberFunction(::<RefOrbitCalc::ReuseMode::DontSaveForReuse>());
//InstantiateMemberFunction(::<RefOrbitCalc::ReuseMode::SaveForReuse1>);
//InstantiateMemberFunction(::<RefOrbitCalc::ReuseMode::SaveForReuse2>);
//InstantiateMemberFunction(::<RefOrbitCalc::ReuseMode::SaveForReuse3>);

template<typename IterType, class T, PerturbExtras PExtras>
uint64_t PerturbationResults<IterType, T, PExtras>::GetBenchmarkOrbit() const {
    return m_BenchmarkOrbit.GetDeltaInMs();
}

template<typename IterType, class T, PerturbExtras PExtras>
template<RefOrbitCalc::ReuseMode Reuse>
void PerturbationResults<IterType, T, PExtras>::CompleteResults(std::unique_ptr<ThreadMemory> allocatorIfAny) {
    m_FullOrbit.Trim();

    if constexpr (
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3) {
        //m_ReuseX.shrink_to_fit();
        //m_ReuseY.shrink_to_fit();
        m_ReuseAllocations = std::move(allocatorIfAny);
    }

    m_BenchmarkOrbit.StopTimer();
}

// Instantiate CompleteResults for all types
#define InstantiateCompleteResults(T, PExtras) \
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse1>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse2>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse3>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::DontSaveForReuse>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse1>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse2>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse3>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::DontSaveForReuse>(std::unique_ptr<ThreadMemory>)

InstantiateCompleteResults(float, PerturbExtras::Disable);
InstantiateCompleteResults(float, PerturbExtras::Bad);
InstantiateCompleteResults(float, PerturbExtras::EnableCompression);

InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);

InstantiateCompleteResults(double, PerturbExtras::Disable);
InstantiateCompleteResults(double, PerturbExtras::Bad);
InstantiateCompleteResults(double, PerturbExtras::EnableCompression);

InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::EnableCompression);

InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);

InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::EnableCompression);


template<typename IterType, class T, PerturbExtras PExtras>
size_t PerturbationResults<IterType, T, PExtras>::GetCompressedOrbitSize() const {
    if constexpr (PExtras == PerturbExtras::EnableCompression) {
        assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
    }

    return m_FullOrbit.GetSize();
}

template<typename IterType, class T, PerturbExtras PExtras>
IterType PerturbationResults<IterType, T, PExtras>::GetCountOrbitEntries() const {
    if constexpr (PExtras == PerturbExtras::EnableCompression) {
        assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
    } else {
        assert(m_FullOrbit.GetSize() == m_UncompressedItersInOrbit);
    }

    return m_UncompressedItersInOrbit;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::AddUncompressedIteration(GPUReferenceIter<T, PExtras> result)
    requires (PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad) {

    static_assert(PExtras == PerturbExtras::Disable || PExtras == PerturbExtras::Bad, "!");

    m_FullOrbit.PushBack(result);
    m_UncompressedItersInOrbit++;
}

template<typename IterType, class T, PerturbExtras PExtras>
const GPUReferenceIter<T, PExtras> *PerturbationResults<IterType, T, PExtras>::GetOrbitData() const {
    return m_FullOrbit.GetData();
}

template<typename IterType, class T, PerturbExtras PExtras>
const HighPrecision &PerturbationResults<IterType, T, PExtras>::GetHiX() const {
    return m_OrbitX;
}

template<typename IterType, class T, PerturbExtras PExtras>
const HighPrecision &PerturbationResults<IterType, T, PExtras>::GetHiY() const {
    return m_OrbitY;
}

template<typename IterType, class T, PerturbExtras PExtras>
const std::string &PerturbationResults<IterType, T, PExtras>::GetHiXStr() const {
    return m_OrbitXStr;
}

template<typename IterType, class T, PerturbExtras PExtras>
const std::string &PerturbationResults<IterType, T, PExtras>::GetHiYStr() const {
    return m_OrbitYStr;
}

template<typename IterType, class T, PerturbExtras PExtras>
T PerturbationResults<IterType, T, PExtras>::GetOrbitXLow() const {
    return m_OrbitXLow;
}

template<typename IterType, class T, PerturbExtras PExtras>
T PerturbationResults<IterType, T, PExtras>::GetOrbitYLow() const {
    return m_OrbitYLow;
}

template<typename IterType, class T, PerturbExtras PExtras>
T PerturbationResults<IterType, T, PExtras>::GetMaxRadius() const {
    return m_MaxRadius;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SetBad(bool bad)
    requires (PExtras == PerturbExtras::Bad) {

    m_FullOrbit.Back().bad = bad;
}

template<typename IterType, class T, PerturbExtras PExtras>
HighPrecision PerturbationResults<IterType, T, PExtras>::GetMaxRadiusHigh() const {
    return m_MaxRadiusHigh;
}

template<typename IterType, class T, PerturbExtras PExtras>
uint64_t PerturbationResults<IterType, T, PExtras>::GetAuthoritativePrecisionInBits() const {
    return m_AuthoritativePrecisionInBits;
}

template<typename IterType, class T, PerturbExtras PExtras>
IterType PerturbationResults<IterType, T, PExtras>::GetPeriodMaybeZero() const {
    return m_PeriodMaybeZero;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SetPeriodMaybeZero(IterType period) {
    m_PeriodMaybeZero = period;
}

template<typename IterType, class T, PerturbExtras PExtras>
int32_t PerturbationResults<IterType, T, PExtras>::GetCompressionErrorExp() const {
    return m_CompressionErrorExp;
}

template<typename IterType, class T, PerturbExtras PExtras>
int32_t PerturbationResults<IterType, T, PExtras>::GetIntermediateCompressionErrorExp() const {
    return m_IntermediateCompressionErrorExp;
}

template<typename IterType, class T, PerturbExtras PExtras>
AddPointOptions PerturbationResults<IterType, T, PExtras>::GetRefOrbitOptions() const {
    return m_RefOrbitOptions;
}

template<typename IterType, class T, PerturbExtras PExtras>
size_t PerturbationResults<IterType, T, PExtras>::GetReuseSize() const {
    assert(m_ReuseX.size() == m_ReuseY.size());
    return m_ReuseX.size();
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::AddUncompressedReusedEntry(
    HighPrecision x,
    HighPrecision y,
    IterTypeFull index) {

    m_ReuseX.push_back(std::move(x));
    m_ReuseY.push_back(std::move(y));
    m_ReuseIndices.push_back(index);
}

// Take references to pointers to avoid copying.
// Set the pointers to point at the specified index.
template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::GetCompressedReuseEntries(
    IntermediateRuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
    size_t uncompressed_index,
    const mpf_t *&x,
    const mpf_t *&y) const {

    PerThreadCompressionHelper.GetReuseEntries(
        uncompressed_index,
        x,
        y);
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::GetUncompressedReuseEntries(
    size_t uncompressed_index,
    const mpf_t *&x,
    const mpf_t *&y) const {

    x = m_ReuseX[uncompressed_index].backendRaw();
    y = m_ReuseY[uncompressed_index].backendRaw();
}

template<typename IterType, class T, PerturbExtras PExtras>
IterType PerturbationResults<IterType, T, PExtras>::GetMaxIterations() const {
    return m_MaxIterations;
}

// For reference:
// Used:
//   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
//   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
// as a reference for the compression algorithm.
template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>
PerturbationResults<IterType, T, PExtras>::Compress(
    int32_t compression_error_exp_param,
    size_t new_generation_number)
    requires (PExtras == PerturbExtras::Disable && !Introspection::IsTDblFlt<T>()) {

    constexpr bool disable_compression = false;
    const auto Two = T{ 2.0f };

    m_CompressionErrorExp = compression_error_exp_param;
    auto compErr = std::pow(10, m_CompressionErrorExp);
    const auto CompressionError = static_cast<T>(compErr);

    auto compressed =
        std::make_unique<PerturbationResults<IterType, T, PerturbExtras::EnableCompression>>(
            m_FullOrbit.GetAddPointOptions(), new_generation_number);
    compressed->CopySettingsWithoutOrbit(*this);

    assert(m_FullOrbit.GetSize() > 1);

    if constexpr (disable_compression) {
        for (size_t i = 0; i < m_FullOrbit.GetSize(); i++) {
            compressed->FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
        }

        return compressed;
    } else {
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

                compressed->m_FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
            }

            auto zx_old = zx;
            zx = zx * zx - zy * zy + m_OrbitXLow;
            HdrReduce(zx);
            zy = Two * zx_old * zy + m_OrbitYLow;
            HdrReduce(zy);
        }
    }

    return compressed;
}

template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>>
PerturbationResults<IterType, T, PExtras>::Decompress(size_t NewGenerationNumber)
    requires (PExtras == PerturbExtras::EnableCompression && !Introspection::IsTDblFlt<T>()) {

    auto compressed_orb = std::move(m_FullOrbit);
    auto decompressed = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        m_FullOrbit.GetAddPointOptions(), NewGenerationNumber);
    decompressed->CopySettingsWithoutOrbit(*this);

    T zx{};
    T zy{};

    size_t compressed_index = 0;

    for (size_t i = 0; i < m_UncompressedItersInOrbit; i++) {
        if (compressed_index < compressed_orb.GetSize() &&
            compressed_orb[compressed_index].CompressionIndex == i) {
            zx = compressed_orb[compressed_index].x;
            zy = compressed_orb[compressed_index].y;
            compressed_index++;
        } else {
            auto zx_old = zx;
            zx = zx * zx - zy * zy + m_OrbitXLow;
            HdrReduce(zx);
            zy = T{ 2.0f } *zx_old * zy + m_OrbitYLow;
            HdrReduce(zy);
        }

        decompressed->AddUncompressedIteration({ zx, zy });
    }

    return decompressed;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SaveOrbitAsText() const {
    auto outFile = GenFilename(GrowableVectorTypes::DebugOutput);

    std::ofstream out(outFile);
    if (!out.is_open()) {
        ::MessageBox(nullptr, L"Failed to open file for writing 3", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    out << "m_OrbitX: " << HdrToString<false>(m_OrbitX) << std::endl;
    out << "m_OrbitY: " << HdrToString<false>(m_OrbitY) << std::endl;
    out << "m_OrbitXStr: " << m_OrbitXStr << std::endl;
    out << "m_OrbitYStr: " << m_OrbitYStr << std::endl;
    out << "m_OrbitXLow: " << HdrToString<false>(m_OrbitXLow) << std::endl;
    out << "m_OrbitYLow: " << HdrToString<false>(m_OrbitYLow) << std::endl;
    out << "m_MaxRadius: " << HdrToString<false>(m_MaxRadius) << std::endl;
    // don't bother with m_MaxRadiusHigh
    out << "m_MaxIterations: " << m_MaxIterations << std::endl;
    out << "m_PeriodMaybeZero: " << m_PeriodMaybeZero << std::endl;
    out << "m_CompressionErrorExp: " << m_CompressionErrorExp << std::endl;
    out << "m_IntermediateCompressionErrorExp: " << m_IntermediateCompressionErrorExp << std::endl;
    out << "m_RefOrbitOptions: " << static_cast<int>(m_RefOrbitOptions) << std::endl;
    out << "m_FullOrbit: " << m_FullOrbit.GetSize() << std::endl;
    out << "m_UncompressedItersInOrbit: " << m_UncompressedItersInOrbit << std::endl;
    out << "m_AuthoritativePrecisionInBits: " << m_AuthoritativePrecisionInBits << std::endl;

    // Write out all values in m_FullOrbit:
    for (size_t i = 0; i < m_FullOrbit.GetSize(); i++) {
        out << "m_FullOrbit[" << i << "].x: " << HdrToString<false>(m_FullOrbit[i].x) << std::endl;
        out << "m_FullOrbit[" << i << "].y: " << HdrToString<false>(m_FullOrbit[i].y) << std::endl;
    }

    out << "m_ReuseX: " << m_ReuseX.size() << std::endl;
    out << "m_ReuseY: " << m_ReuseY.size() << std::endl;
    out << "m_ReuseIndices: " << m_ReuseIndices.size() << std::endl;
    if (m_ReuseX.size() != m_ReuseY.size() ||
        m_ReuseX.size() != m_ReuseIndices.size()) {
        ::MessageBox(nullptr, L"m_ReuseX and m_ReuseY are different sizes.", L"", MB_OK | MB_APPLMODAL);
        out.close();
        return;
    }

    for (size_t i = 0; i < m_ReuseX.size(); i++) {
        out << "m_ReuseX[" << i << "]: " << HdrToString<false>(m_ReuseX[i]) << std::endl;
        out << "m_ReuseY[" << i << "]: " << HdrToString<false>(m_ReuseY[i]) << std::endl;
        out << "m_ReuseIndices[" << i << "]: " << m_ReuseIndices[i] << std::endl;
    }

    out.close();
}

// For information purposes only, not used for anything
// other than reporting.
template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::GetIntermediatePrecision(
    int64_t &deltaPrecision,
    int64_t &extraPrecision) const {

    deltaPrecision = m_DeltaPrecisionCached;
    extraPrecision = m_ExtraPrecisionCached;
}

// For information purposes only, not used for anything
// other than reporting.
template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SetIntermediateCachedPrecision(
    int64_t deltaPrecision,
    int64_t extraPrecision) {

    m_DeltaPrecisionCached = deltaPrecision;
    m_ExtraPrecisionCached = extraPrecision;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::CloseMetaFileIfOpen() const {
    if (m_MetaFileHandle != INVALID_HANDLE_VALUE) {
        CloseHandle(m_MetaFileHandle);
        m_MetaFileHandle = INVALID_HANDLE_VALUE;
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::MapExistingFiles() {
    assert(m_LaReference == nullptr);
    assert(m_FullOrbit.GetCapacity() == 0);

    m_FullOrbit = { GetRefOrbitOptions(), GenFilename(GrowableVectorTypes::GPUReferenceIter) };

    try {
        m_LaReference = std::make_unique<LAReference<IterType, T, SubType, PExtras>>(
            GetRefOrbitOptions(),
            GenFilename(GrowableVectorTypes::LAInfoDeep),
            GenFilename(GrowableVectorTypes::LAStageInfo));
    }
    catch (const std::exception &) {
        m_LaReference = nullptr;
    }
}


template<typename IterType, class T, PerturbExtras PExtras>
RefOrbitCompressor<IterType, T, PExtras>::RefOrbitCompressor(
    PerturbationResults<IterType, T, PExtras> &results,
    int32_t CompressionErrorExp) :
    results{ results },
    zx{ results.m_OrbitXLow },
    zy{ results.m_OrbitYLow },
    Two{ 2.0f },
    CompressionError(static_cast<T>(std::pow(10, CompressionErrorExp))),
    CompressionErrorExp{ CompressionErrorExp },
    CurCompressedIndex{} {

    // This code can run even if compression is disabled, but it doesn't matter.
    results.m_CompressionErrorExp = CompressionErrorExp;
}

template<typename IterType, class T, PerturbExtras PExtras>
void RefOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration(GPUReferenceIter<T, PExtras> iter)
    requires (PExtras == PerturbExtras::EnableCompression && !Introspection::IsTDblFlt<T>()) {

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

template<typename IterType, class T, PerturbExtras PExtras>
IntermediateOrbitCompressor<IterType, T, PExtras>::IntermediateOrbitCompressor(
    PerturbationResults<IterType, T, PExtras> &results,
    int32_t CompressionErrorExp) :
    results{ results },
    zx{},
    zy{},
    cx{},
    cy{},
    Two{},
    CompressionError{},
    ReducedZx{},
    ReducedZy{},
    Temp{},
    IntermediateCompressionErrorExp{},
    CurCompressedIndex{} {

    // This code can run even if compression is disabled, but it doesn't matter.

    mpf_init2(zx, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set(zx, *results.GetHiX().backendRaw());

    mpf_init2(zy, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set(zy, *results.GetHiY().backendRaw());

    mpf_init2(cx, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set(cx, *results.GetHiX().backendRaw());

    mpf_init2(cy, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set(cy, *results.GetHiY().backendRaw());

    mpf_init2(Two, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set_d(Two, 2);

    results.m_IntermediateCompressionErrorExp = CompressionErrorExp;

    mpf_init2(CompressionError, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set_d(CompressionError, 10);
    mpf_pow_ui(CompressionError, CompressionError, CompressionErrorExp);

    mpf_init2(ReducedZx, AuthoritativeReuseExtraPrecisionInBits);
    mpf_init2(ReducedZy, AuthoritativeReuseExtraPrecisionInBits);

    for (size_t i = 0; i < 6; i++) {
        mpf_init2(Temp[i], AuthoritativeReuseExtraPrecisionInBits);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
IntermediateOrbitCompressor<IterType, T, PExtras>::~IntermediateOrbitCompressor() {
    mpf_clear(zx);
    mpf_clear(zy);
    mpf_clear(cx);
    mpf_clear(cy);
    mpf_clear(Two);
    mpf_clear(CompressionError);
    mpf_clear(ReducedZx);
    mpf_clear(ReducedZy);

    for (size_t i = 0; i < 6; i++) {
        mpf_clear(Temp[i]);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void IntermediateOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration(
    mpf_t incomingZx,
    mpf_t incomingZy,
    IterTypeFull index) {

    mpf_set(ReducedZx, incomingZx);
    mpf_set(ReducedZy, incomingZy);

    {
        // auto errX = zx - ReducedZx;
        mpf_sub(Temp[0], zx, ReducedZx);

        // auto errY = zy - ReducedZy;
        mpf_sub(Temp[1], zy, ReducedZy);
    }

    // Temp[0] = errX
    // Temp[1] = errY

    // auto err = (errX * errX + errY * errY) * CompressionError;
    {
        mpf_mul(Temp[2], Temp[0], Temp[0]);
        mpf_mul(Temp[3], Temp[1], Temp[1]);
        mpf_add(Temp[4], Temp[2], Temp[3]);
        mpf_mul(Temp[5], Temp[4], CompressionError);
    }

    // Temp[5] = err

    // auto norm_z = ReducedZx * ReducedZx + ReducedZy * ReducedZy;
    {
        mpf_mul(Temp[2], ReducedZx, ReducedZx);
        mpf_mul(Temp[3], ReducedZy, ReducedZy);
        mpf_add(Temp[4], Temp[2], Temp[3]);
    }

    // Temp[4] = norm_z

    if (mpf_cmp(Temp[5], Temp[4]) >= 0) {
        results.AddUncompressedReusedEntry(ReducedZx, ReducedZy, index);

        mpf_set(zx, ReducedZx);
        mpf_set(zy, ReducedZy);

        // Corresponds to the entry just added
        CurCompressedIndex++;
        //assert(CurCompressedIndex == results.m_ReuseX.GetSize() - 1);
        //assert(CurCompressedIndex == results.m_ReuseY.GetSize() - 1);
    }

    // auto zx_old = zx;
    // zx = zx * zx - zy * zy + cx;
    // zy = Two * zx_old * zy + cy;

    {
        mpf_set(Temp[0], zx); // zx_old

        mpf_mul(Temp[2], zx, zx);
        mpf_mul(Temp[3], zy, zy);
        mpf_sub(zx, Temp[2], Temp[3]);
        mpf_add(zx, zx, cx);

        mpf_mul(Temp[2], Two, Temp[0]);
        mpf_mul(zy, Temp[2], zy);
        mpf_add(zy, zy, cy);
    }
}

#define InstantiatePerturbationResult(T, PExtras) \
    template class PerturbationResults<uint32_t, T, PExtras>; \
    template class PerturbationResults<uint64_t, T, PExtras>;

InstantiatePerturbationResult(float, PerturbExtras::Disable);
InstantiatePerturbationResult(float, PerturbExtras::Bad);
InstantiatePerturbationResult(float, PerturbExtras::EnableCompression);

InstantiatePerturbationResult(double, PerturbExtras::Disable);
InstantiatePerturbationResult(double, PerturbExtras::Bad);
InstantiatePerturbationResult(double, PerturbExtras::EnableCompression);

InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);

InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::EnableCompression);

InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::EnableCompression);

InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);

#define InstantiateRefOrbitCompressor(T, PExtras) \
    template class RefOrbitCompressor<uint32_t, T, PExtras>; \
    template class RefOrbitCompressor<uint64_t, T, PExtras>;

InstantiateRefOrbitCompressor(float, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(float, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(float, PerturbExtras::EnableCompression);

InstantiateRefOrbitCompressor(double, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(double, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(double, PerturbExtras::EnableCompression);

InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);

InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::EnableCompression);

InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::EnableCompression);

InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);

#define InstantiateIntermediateOrbitCompressor(T, PExtras) \
    template class IntermediateOrbitCompressor<uint32_t, T, PExtras>; \
    template class IntermediateOrbitCompressor<uint64_t, T, PExtras>;

InstantiateIntermediateOrbitCompressor(float, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(float, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(float, PerturbExtras::EnableCompression);

InstantiateIntermediateOrbitCompressor(double, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(double, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(double, PerturbExtras::EnableCompression);

InstantiateIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::EnableCompression);

InstantiateIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::EnableCompression);

InstantiateIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::EnableCompression);

InstantiateIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::EnableCompression);
