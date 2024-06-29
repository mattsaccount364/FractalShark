#include "stdafx.h"

#include "Fractal.h"
#include "PerturbationResults.h"
#include "LAReference.h"
#include "MPIRSerialization.h"

template<typename IterType>
const HighPrecision &PerturbationResultsBase<IterType>::GetHiX() const {
    return this->m_OrbitX;
}

template<typename IterType>
const HighPrecision &PerturbationResultsBase<IterType>::GetHiY() const {
    return this->m_OrbitY;
}

template<typename IterType>
const HighPrecision &PerturbationResultsBase<IterType>::GetHiZoomFactor() const {
    return this->m_ZoomFactor;
}

template<typename IterType>
IterType PerturbationResultsBase<IterType>::GetMaxIterations() const {
    return this->m_MaxIterations;
}

// ============================================================================

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

    PerturbationResultsBase<IterType>{ },
    m_OrbitXStr{},
    m_OrbitYStr{},
    m_ZoomFactorStr{},
    m_OrbitXLow{},
    m_OrbitYLow{},
    m_ZoomFactorLow{},
    m_MaxRadius{},
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
            } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
                m_FullOrbit[i] = GPUReferenceIter<T, PExtras>{
                    (T)other.m_FullOrbit[i].x,
                    (T)other.m_FullOrbit[i].y,
                    other.m_FullOrbit[i].u.f.CompressionIndex
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

    this->m_OrbitX = other.GetHiX();
    this->m_OrbitY = other.GetHiY();
    this->m_ZoomFactor = other.GetHiZoomFactor();
    m_OrbitXStr = other.GetHiXStr();
    m_OrbitYStr = other.GetHiYStr();
    m_ZoomFactorStr = other.GetZoomFactorStr();
    m_OrbitXLow = static_cast<T>(Convert<HighPrecision, double>(this->m_OrbitX));
    m_OrbitYLow = static_cast<T>(Convert<HighPrecision, double>(this->m_OrbitY));
    m_ZoomFactorLow = static_cast<T>(Convert<HighPrecision, double>(this->m_ZoomFactor));
    m_MaxRadius = (T)other.GetMaxRadius();
    this->m_MaxIterations = other.GetMaxIterations();
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

    assert(GetCountOrbitEntries() == m_FullOrbit.GetSize());

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
InstantiateCopyPerturbationResults(CudaDblflt<MattDblflt>, double, PerturbExtras::SimpleCompression, PerturbExtras::SimpleCompression);
InstantiateCopyPerturbationResults(HDRFloat<CudaDblflt<MattDblflt>>, HDRFloat<double>, PerturbExtras::SimpleCompression, PerturbExtras::SimpleCompression);
InstantiateCopyPerturbationResultsLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::Disable, PerturbExtras::Disable);
InstantiateCopyPerturbationResultsLA(HDRFloat<float>, HDRFloat<double>, PerturbExtras::SimpleCompression, PerturbExtras::SimpleCompression);
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
    this->m_OrbitX = other.GetHiX();
    this->m_OrbitY = other.GetHiY();
    this->m_ZoomFactor = other.GetHiZoomFactor();
    m_OrbitXStr = other.GetHiXStr();
    m_OrbitYStr = other.GetHiYStr();
    m_ZoomFactorStr = other.GetZoomFactorStr();
    m_OrbitXLow = other.GetOrbitXLow();
    m_OrbitYLow = other.GetOrbitYLow();
    m_ZoomFactorLow = other.GetZoomFactorLow();
    m_MaxRadius = other.GetMaxRadius();
    this->m_MaxIterations = other.GetMaxIterations();
    m_PeriodMaybeZero = other.GetPeriodMaybeZero();
    m_CompressionErrorExp = other.GetCompressionErrorExp();
    m_IntermediateCompressionErrorExp = other.GetIntermediateCompressionErrorExp();
    m_RefOrbitOptions = other.GetRefOrbitOptions();
    m_BaseFilename = GenBaseFilename(m_GenerationNumber);
    m_MetaFileHandle = INVALID_HANDLE_VALUE;

    m_FullOrbit = {}; // Note: not copied
    m_UncompressedItersInOrbit = other.m_UncompressedItersInOrbit;

    // compression - don't copy LA data.  Regenerate if needed.
    m_LaReference = nullptr;

    m_AuthoritativePrecisionInBits = other.m_AuthoritativePrecisionInBits;

    // Not supported
    m_ReuseX = {};
    m_ReuseY = {};
    m_ReuseAllocations = nullptr;

    m_BenchmarkOrbit = other.m_BenchmarkOrbit;

    m_DeltaPrecisionCached = other.m_DeltaPrecisionCached;
    m_ExtraPrecisionCached = other.m_ExtraPrecisionCached;
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
    } else if constexpr (PExtras == PerturbExtras::SimpleCompression) {
        metafile << "PerturbExtras::SimpleCompression" << std::endl;
    } else if constexpr (PExtras == PerturbExtras::Disable) {
        metafile << "PerturbExtras::Disable" << std::endl;
    } else {
        ::MessageBox(nullptr, L"Invalid bad.", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    metafile << "PrecisionInBits: " << this->m_OrbitX.precisionInBits() << std::endl;
    metafile << "HighPrecisionReal: " << HdrToString<true>(this->m_OrbitX) << std::endl;
    metafile << "PrecisionInBits: " << this->m_OrbitY.precisionInBits() << std::endl;
    metafile << "HighPrecisionImaginary: " << HdrToString<true>(this->m_OrbitY) << std::endl;
    metafile << "PrecisionInBits: " << this->m_ZoomFactor.precisionInBits() << std::endl;
    metafile << "HighPrecisionZoomFactor: " << HdrToString<true>(this->m_ZoomFactor) << std::endl;
    metafile << "LowPrecisionReal: " << HdrToString<true>(m_OrbitXLow) << std::endl;
    metafile << "LowPrecisionImaginary: " << HdrToString<true>(m_OrbitYLow) << std::endl;
    metafile << "LowPrecisionZoomFactor: " << HdrToString<true>(m_ZoomFactorLow) << std::endl;
    metafile << "MaxRadius: " << HdrToString<true>(m_MaxRadius) << std::endl;
    // don't bother with m_MaxRadiusHigh
    metafile << "MaxIterationsPerPixel: " << this->m_MaxIterations << std::endl;
    metafile << "Period: " << m_PeriodMaybeZero << std::endl;
    metafile << "CompressionErrorExponent: " << m_CompressionErrorExp << std::endl;
    metafile << "IntermediateCompressionErrorExponent: " << m_IntermediateCompressionErrorExp << std::endl;
    metafile << "UncompressedIterationsInOrbit: " << GetCountOrbitEntries() << std::endl;

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
        } else if (badstr == "PerturbExtras::SimpleCompression") {
            if constexpr (PExtras == PerturbExtras::SimpleCompression) {
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

        this->m_OrbitX.precisionInBits(prec);
        this->m_OrbitX = HighPrecision{ shiX };
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

        this->m_OrbitY.precisionInBits(prec);
        this->m_OrbitY = HighPrecision{ shiY };
        m_OrbitYStr = shiY;
    }

    {
        uint32_t prec;
        metafile >> descriptor_string_junk;
        metafile >> prec;

        MPIRPrecision p{ prec };

        std::string shiZ;
        metafile >> descriptor_string_junk;
        metafile >> shiZ;

        this->m_ZoomFactor.precisionInBits(prec);
        this->m_ZoomFactor = HighPrecision{ shiZ };
        m_ZoomFactorStr = shiZ;
    }

    HdrFromIfStream<true, T, SubType>(m_OrbitXLow, metafile);
    HdrFromIfStream<true, T, SubType>(m_OrbitYLow, metafile);
    HdrFromIfStream<true, T, SubType>(m_ZoomFactorLow, metafile);

    HdrFromIfStream<true, T, SubType>(m_MaxRadius, metafile);

    {
        std::string maxIterationsStr;
        metafile >> descriptor_string_junk;
        metafile >> maxIterationsStr;
        this->m_MaxIterations = (IterType)std::stoll(maxIterationsStr);
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
    HighPrecision Zero{};

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
    const HighPrecision &/*minX*/, // TODO
    const HighPrecision &minY,
    const HighPrecision &/*maxX*/,
    const HighPrecision &maxY,
    IterType NumIterations,
    size_t GuessReserveSize) {

    const T radiusY{ T{ maxY - minY } / T{ 2.0f } };

    InitResults(Reuse, cx, cy, radiusY, NumIterations, GuessReserveSize);
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::InitResults(
    RefOrbitCalc::ReuseMode Reuse,
    const HighPrecision &cx,
    const HighPrecision &cy,
    const T &radiusY,
    IterType NumIterations,
    size_t GuessReserveSize) {
    m_BenchmarkOrbit.StartTimer();

    HighPrecision zoomFactor{ radiusY };
    zoomFactor = HighPrecision{ 2 } / zoomFactor;
    std::string zoomFactorStr = zoomFactor.str();

    this->m_OrbitX = cx;
    this->m_OrbitY = cy;
    this->m_ZoomFactor = zoomFactor;
    m_OrbitXStr = HdrToString<false>(cx);
    m_OrbitYStr = HdrToString<false>(cy);
    m_ZoomFactorStr = zoomFactorStr;
    m_OrbitXLow = T{ cx };
    m_OrbitYLow = T{ cy };
    m_ZoomFactorLow = T{ zoomFactor };
    m_MaxRadius = radiusY;

    HdrReduce(m_MaxRadius);

    this->m_MaxIterations = NumIterations + 1; // +1 for push_back(0) below

    const size_t ReserveSize = (GuessReserveSize != 0) ? GuessReserveSize : 1'000'000;

    // Do not resize prematurely -- file-backed vectors only resize when needed.
    m_FullOrbit.MutableResize(ReserveSize, 0);

    // Add an empty entry at the start
    m_FullOrbit.PushBack({});
    m_UncompressedItersInOrbit = 1;

    if (Reuse == RefOrbitCalc::ReuseMode::SaveForReuse1 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse2 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {

        m_AuthoritativePrecisionInBits =
            PrecisionCalculator::GetPrecision(radiusY, radiusY, true);
        m_ReuseX.reserve(ReserveSize);
        m_ReuseY.reserve(ReserveSize);
        m_ReuseAllocations = nullptr;
    } else if (Reuse == RefOrbitCalc::ReuseMode::DontSaveForReuse) {
        m_AuthoritativePrecisionInBits = 0;
    }

    m_LaReference = nullptr;
}

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
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse3 ||
        Reuse == RefOrbitCalc::ReuseMode::SaveForReuse4) {
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
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse4>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint32_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::DontSaveForReuse>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse1>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse2>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse3>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::SaveForReuse4>(std::unique_ptr<ThreadMemory>); \
    template void PerturbationResults<uint64_t, T, PExtras>::CompleteResults<RefOrbitCalc::ReuseMode::DontSaveForReuse>(std::unique_ptr<ThreadMemory>)

InstantiateCompleteResults(float, PerturbExtras::Disable);
InstantiateCompleteResults(float, PerturbExtras::Bad);
InstantiateCompleteResults(float, PerturbExtras::SimpleCompression);

InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateCompleteResults(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiateCompleteResults(double, PerturbExtras::Disable);
InstantiateCompleteResults(double, PerturbExtras::Bad);
InstantiateCompleteResults(double, PerturbExtras::SimpleCompression);

InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<float>, PerturbExtras::SimpleCompression);

InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::Disable);
InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::Bad);
InstantiateCompleteResults(HDRFloat<double>, PerturbExtras::SimpleCompression);


template<typename IterType, class T, PerturbExtras PExtras>
size_t PerturbationResults<IterType, T, PExtras>::GetCompressedOrUncompressedOrbitSize() const {

    if constexpr (PExtras == PerturbExtras::SimpleCompression) {
        assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
    } else {
        assert(m_FullOrbit.GetSize() == m_UncompressedItersInOrbit);
    }

    return m_FullOrbit.GetSize();
}

template<typename IterType, class T, PerturbExtras PExtras>
size_t PerturbationResults<IterType, T, PExtras>::GetCompressedOrbitSize() const
    requires (PExtras == PerturbExtras::SimpleCompression) {

    assert(m_FullOrbit.GetSize() <= m_UncompressedItersInOrbit);
    return m_FullOrbit.GetSize();
}

template<typename IterType, class T, PerturbExtras PExtras>
IterType PerturbationResults<IterType, T, PExtras>::GetCountOrbitEntries() const {
    if constexpr (PExtras == PerturbExtras::SimpleCompression) {
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
const std::string &PerturbationResults<IterType, T, PExtras>::GetHiXStr() const {
    return m_OrbitXStr;
}

template<typename IterType, class T, PerturbExtras PExtras>
const std::string &PerturbationResults<IterType, T, PExtras>::GetHiYStr() const {
    return m_OrbitYStr;
}

template<typename IterType, class T, PerturbExtras PExtras>
const std::string &PerturbationResults<IterType, T, PExtras>::GetZoomFactorStr() const {
    return m_ZoomFactorStr;
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
T PerturbationResults<IterType, T, PExtras>::GetZoomFactorLow() const {
    return m_ZoomFactorLow;
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

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::AddUncompressedReusedEntry(
    const mpf_t x,
    const mpf_t y,
    IterTypeFull index) {

    // This may be a perf problem if we have a lot of calls here.
    // This is a full copy.
    AddUncompressedReusedEntry(HighPrecision{ x }, HighPrecision{ y }, index);
}


template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::AddUncompressedRebase(
    IterTypeFull i,
    IterTypeFull index) {

    if (!m_Rebases.empty() &&
        m_Rebases[m_Rebases.size() - 1] > index) {
        m_Rebases[m_Rebases.size() - 1] = i;
    } else {
        m_Rebases.push_back(i);
    }
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
void PerturbationResults<IterType, T, PExtras>::GetMaxCompressedReuseEntries(
    IntermediateMaxRuntimeDecompressor<IterType, T, PExtras> &PerThreadCompressionHelper,
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

// For reference:
// Used:
//   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
//   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
// as a reference for the compression algorithm.
template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
PerturbationResults<IterType, T, PExtras>::Compress(
    int32_t compression_error_exp_param,
    size_t new_generation_number) const
    requires (PExtras == PerturbExtras::Disable && !Introspection::IsTDblFlt<T>()) {

    constexpr bool DisableCompression = false;
    const auto Two = T{ 2.0f };

    m_CompressionErrorExp = compression_error_exp_param;
    auto compErr = std::pow(10, m_CompressionErrorExp);
    const auto CompressionError = static_cast<T>(compErr);

    auto compressed =
        std::make_unique<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>(
            GetRefOrbitOptions(), new_generation_number);
    compressed->CopySettingsWithoutOrbit(*this);

    assert(GetCountOrbitEntries() > 1);

    if constexpr (DisableCompression) {
        for (size_t i = 0; i < GetCountOrbitEntries(); i++) {
            compressed->FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
        }

        return compressed;
    } else {
        T zx{};
        T zy{};

        for (size_t i = 0; i < GetCountOrbitEntries(); i++) {
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
    const
    requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>()) {

    const auto &compressed_orb = m_FullOrbit;
    auto decompressed = std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
        GetRefOrbitOptions(), NewGenerationNumber);
    decompressed->CopySettingsWithoutOrbit(*this);

    const auto targetUncompressedIters = decompressed->GetCountOrbitEntries();
    decompressed->m_UncompressedItersInOrbit = 0;

    T zx{};
    T zy{};

    size_t compressed_index = 0;

    for (size_t i = 0; i < targetUncompressedIters; i++) {
        if (compressed_index < compressed_orb.GetSize() &&
            compressed_orb[compressed_index].u.f.CompressionIndex == i) {
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

// For reference:
// Used:
//   https://code.mathr.co.uk/fractal-bits/tree/HEAD:/mandelbrot-reference-compression
//   https://fractalforums.org/fractal-mathematics-and-new-theories/28/reference-compression/5142
// as a reference for the compression algorithm.
template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>
PerturbationResults<IterType, T, PExtras>::CompressMax(
    int32_t compression_error_exp_param,
    size_t new_generation_number)
    const
    requires (!Introspection::IsTDblFlt<T>()) {

    static constexpr IterTypeFull MaxItersSinceLastWrite = 1000;

    // This is an optional adjustment.  Idea is to bound number of iterations
    // that need to reside in memory after decompression.  Set to false
    // for maximum compression, but more memory at decompression time.
    static constexpr bool UseMaxItersSinceLastWrite = false;

    // Testing
    static constexpr bool DisableCompression = false;

    // The idea behind MaxIntermediateOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration
    // is to use "max compression" on the intermediate orbit to limit memory usage.
    // Unfortunately, this is not working as expected.  Setting this value to true
    // produces a file that shows why.  During compression, we rely on most of the earlier
    // iterations to be present in memory.  If we keep it all in memory in order to
    // compress it, then the memory savings are lost.
    // Thus, MaxIntermediateOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration
    // is fundamentally broken - it does not appear workable.
    static constexpr bool TrackJIndexesUsed = false;


    const auto Two = T{ 2.0f };

    m_CompressionErrorExp = compression_error_exp_param;
    auto compErr = std::pow(10, m_CompressionErrorExp);
    const auto CompressionError = static_cast<T>(compErr);

    auto compressed =
        std::make_unique<PerturbationResults<IterType, T, PerturbExtras::SimpleCompression>>(
            GetRefOrbitOptions(), new_generation_number);
    compressed->CopySettingsWithoutOrbit(*this);

    assert(GetCountOrbitEntries() > 1);

    if constexpr (DisableCompression) {
        for (size_t i = 0; i < GetCompressedOrbitSize(); i++) {
            compressed->FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i });
        }

        return compressed;
    }

    auto normZ = [](T x, T y) -> T {
        auto norm_z = x * x + y * y;
        HdrReduce(norm_z);
        return norm_z;
        };

    auto normZTimesT = [](T x, T y, T t) -> T {
        auto norm_z = (x * x + y * y) * t;
        HdrReduce(norm_z);
        return norm_z;
        };

    const auto threshold2 = CompressionError;
    const T constant1{ 0x1.0p-4 };
    const T constant2{ (T)0x1.000001p0 };
    //const T constant2{ 20000 };
    T zx{};
    T zy{};

    RuntimeDecompressor<IterType, T, PExtras> PerThreadCompressionHelper{ *this };

    IterTypeFull i = 1;
    for (; i < GetCountOrbitEntries(); i++) {
        
        const auto norm_z = normZ(m_FullOrbit[i].x, m_FullOrbit[i].y);

        if (HdrCompareToBothPositiveReducedLT(norm_z, constant1)) {
            zx = m_FullOrbit[i].x;
            zy = m_FullOrbit[i].y;

            compressed->m_FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i, true });
            break;
        } else {
            auto err = normZTimesT(zx - m_FullOrbit[i].x, zy - m_FullOrbit[i].y, threshold2);

            if (HdrCompareToBothPositiveReducedGE(err, norm_z)) {
                // else if (std::norm(z - Z[i]) * Threshold2 > std::norm(Z[i]))
                zx = m_FullOrbit[i].x;
                zy = m_FullOrbit[i].y;

                compressed->m_FullOrbit.PushBack({ m_FullOrbit[i].x, m_FullOrbit[i].y, i, false });
            }
        }

        //z = z * z + c;
        const auto zx_old = zx;
        zx = zx * zx - zy * zy + m_OrbitXLow;
        HdrReduce(zx);
        zy = Two * zx_old * zy + m_OrbitYLow;
        HdrReduce(zy);
    }

    auto dzX = zx;
    auto dzY = zy;
    IterTypeFull PrevWayPointIteration = i;

    // dz = (Z[0] * 2 + dz) * dz;
    // dzX + dzY * i = (Z[0].x + Z[0].y * i) * 2
    // 2 * z0x + 2 * z0y * i + zx + zy * i
    // x: 2 * z0x + zx
    // y: 2 * z0y + zy

    //dzX = Two * m_FullOrbit[0].x * dzX + dzX * dzX;
    //dzY = MinusTwo * m_FullOrbit[0].y * dzY - dzY * dzY;

    const auto dzX_old = dzX;
    dzX = Two * m_FullOrbit[0].x * dzX - Two * m_FullOrbit[0].y * dzY + dzX * dzX - dzY * dzY;
    HdrReduce(dzX);
    dzY = Two * m_FullOrbit[0].x * dzY + Two * m_FullOrbit[0].y * dzX_old + Two * dzX_old * dzY;
    HdrReduce(dzY);

    i++;
    IterTypeFull j = 1;
    IterTypeFull itersSinceLastWrite = 0;

    std::vector<IterTypeFull> JIndexes;

    for (; i < GetCountOrbitEntries(); i++, j++) {
        // z = dz + Z[j] where m_FullOrbit[j] = Z[j]
        zx = dzX + m_FullOrbit[j].x;
        zy = dzY + m_FullOrbit[j].y;

        if constexpr (TrackJIndexesUsed) {
            JIndexes.push_back(j);
        }

        const auto norm_z_orig = normZ(zx, zy);
        const auto norm_dz_orig = normZTimesT(dzX, dzY, constant2);

        const auto err = normZTimesT(zx - m_FullOrbit[i].x, zy - m_FullOrbit[i].y, threshold2);

        const bool condition1 = j >= PrevWayPointIteration;
        const bool condition2 = HdrCompareToBothPositiveReducedGE(err, norm_z_orig);
        const bool condition3 = UseMaxItersSinceLastWrite ? (itersSinceLastWrite >= MaxItersSinceLastWrite) : false;

        if (condition1 || condition2) {
            PrevWayPointIteration = i;
            zx = m_FullOrbit[i].x;
            zy = m_FullOrbit[i].y;
            dzX = zx - m_FullOrbit[j].x;
            dzY = zy - m_FullOrbit[j].y;

            const auto norm_z = normZ(zx, zy);
            const auto norm_dz = normZ(dzX, dzY);
            if (HdrCompareToBothPositiveReducedLT(norm_z, norm_dz) || (i - j) * 4 < i) {
                dzX = zx;
                dzY = zy;
                j = 0;
                compressed->m_FullOrbit.PushBack({ dzX, dzY, i, true });
                itersSinceLastWrite = 0;
            } else {
                compressed->m_FullOrbit.PushBack({ dzX, dzY, i, false });
                itersSinceLastWrite = 0;
            }
        } else if (condition3) {
            if constexpr (UseMaxItersSinceLastWrite) {
                // TODO So can we use this to keep a bounded amount of
                // shit in memory during decompression e.g. for the intermediate
                // precision thing.
                //
                // Include extra UseMaxItersSinceLastWrite check to make
                // sure optimizer gets rid of this if needed.
                compressed->m_FullOrbit.PushBack({ dzX, dzY, i, false });
                itersSinceLastWrite = 0;
            }
        } else if (HdrCompareToBothPositiveReducedLT(norm_z_orig, norm_dz_orig)) {
            dzX = zx;
            dzY = zy;
            j = 0;

            const auto rebaseSize = compressed->m_Rebases.size();
            const auto fullOrbitSize = compressed->GetCountOrbitEntries();
            if (rebaseSize != 0 &&
                compressed->m_Rebases[rebaseSize - 1] > compressed->m_FullOrbit[fullOrbitSize - 1].u.f.CompressionIndex) {
                compressed->m_Rebases[rebaseSize - 1] = i;
            } else {
                compressed->m_Rebases.push_back(i);
            }
        }

        // dz = (Z[j] * (long double) 2 + dz) * dz;
        // Using zx/zy here to indicate m_FullOrbit[j].x/m_FullOrbit[j].y
        // (dzX + dzY * i) = ((zx + zy * i) * 2 + (dzX + dzY * i)) * (dzX + dzY * i)
        // (dzX + dzY * i) = ((2 * zx + 2 * zy * i) + (dzX + dzY * i)) * (dzX + dzY * i)
        // (dzX + dzY * i) = (2 * zx + dzX + 2 * zy * i + dzY * i) * (dzX + dzY * i)
        // 
        // dzX = (2 * zx + dzX) * dzX
        // dzX = 2 * zx * dzX + dzX * dzX
        // 
        // dzY = (2 * zy * i + dzY * i) * dzY * i
        // dzY = -2 * zy * dzY - dzY * dzY

        //dzX = Two * m_FullOrbit[j].x * dzX + dzX * dzX;
        //HdrReduce(dzX);
        //dzY = MinusTwo * m_FullOrbit[j].y * dzY - dzY * dzY;
        //HdrReduce(dzY);

        const auto dzX_old2 = dzX;
        dzX = Two * m_FullOrbit[j].x * dzX - Two * m_FullOrbit[j].y * dzY + dzX * dzX - dzY * dzY;
        HdrReduce(dzX);
        dzY = Two * m_FullOrbit[j].x * dzY + Two * m_FullOrbit[j].y * dzX_old2 + Two * dzX_old2 * dzY;
        HdrReduce(dzY);

        // auto zx_old = zx;
        // zx = zx * zx - zy * zy + m_OrbitXLow;
        // HdrReduce(zx);
        // zy = Two * zx_old * zy + m_OrbitYLow;
        // HdrReduce(zy);

        itersSinceLastWrite++;
    }

    compressed->m_FullOrbit.PushBack({ {}, {}, ~0ull, false });
    compressed->m_Rebases.push_back(~0ull);

    if constexpr (TrackJIndexesUsed) {
        // Write out the vector to a text file
        std::ofstream myfile;
        myfile.open("j_indexes.txt");
        for (const auto &j : JIndexes) {
            myfile << j << std::endl;
        }
    }

    return compressed;
}

template<typename IterType, class T, PerturbExtras PExtras>
std::unique_ptr<PerturbationResults<IterType, T, PerturbExtras::Disable>>
PerturbationResults<IterType, T, PExtras>::DecompressMax(size_t NewGenerationNumber)
    const
    requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>()) {

    // Show range of indices touched
    constexpr bool outputFile = false;

    // Just curious what this looked like - set to true if you want the answer.
    constexpr bool trackCountsPerIndex = false;

    assert(GetCompressedOrbitSize() > 0);
    T zx{};
    T zy{};
    IterTypeFull wayPointIndex = 0;
    IterTypeFull rebaseIndex = 0;
    GPUReferenceIter<T, PExtras> nextWayPoint = m_FullOrbit[0];
    IterTypeFull nextRebase = m_Rebases[0];

    auto decompressed =
        std::make_unique<PerturbationResults<IterType, T, PerturbExtras::Disable>>(
            GetRefOrbitOptions(), NewGenerationNumber);
    decompressed->CopySettingsWithoutOrbit(*this);

    const auto targetUncompressedIters = decompressed->GetCountOrbitEntries();
    decompressed->m_UncompressedItersInOrbit = 0;

    std::vector<IterTypeFull> countsPerIndex;
    std::vector<IterTypeFull> indexTouched;

    if constexpr (trackCountsPerIndex) {
        countsPerIndex.resize(targetUncompressedIters);
    }

    // Write  out the begin and end to a file
    std::ofstream myfile;
    if constexpr (outputFile) {
        myfile.open("begin_end.txt");
    }

    auto CorrectOrbit = [&](IterTypeFull begin, IterTypeFull end, T diffX, T diffY) {
        T dzdcX{ 1 }; // FIXME: scaling factor removed
        T dzdcY{ 0 };

        HdrReduce(diffX);
        HdrReduce(diffY);

        if constexpr (outputFile) {
            myfile << begin << " " << end << std::endl;
        }

        for (IterTypeFull i = end; i > begin; ) {
            i--;
            // dzdc *= Z[i] * 2;
            // dzdcX + dzdcY * i = (dzdcX + dzdcY * i) * (Z[i].x + Z[i].y * i) * 2
            // dzdcX + dzdcY * i = (2 * dzdcX + 2 * dzdcY * i) * (Z[i].x + Z[i].y * i)
            // dzdcX = dzdcX * Z[i].x * 2 - dzdcY * Z[i].y * 2
            // dzdcY = dzdcX * Z[i].y * 2 + dzdcY * Z[i].x * 2
            // 
            // TODO double check this:
            const auto old_dzdcX = dzdcX;
            dzdcX = dzdcX * decompressed->m_FullOrbit[i].x * 2 - dzdcY * decompressed->m_FullOrbit[i].y * 2;
            HdrReduce(dzdcX);
            dzdcY = old_dzdcX * decompressed->m_FullOrbit[i].y * 2 + dzdcY * decompressed->m_FullOrbit[i].x * 2;
            HdrReduce(dzdcY);

            // Z[i] += diff / dzdc;
            // (diffX + diffY*i) / (dzdcX + dzdcY * i) = (diffX + diffY * i) / (dzdcX + dzdcY * i)
            // resultReal = (diffX * dzdcX + diffY * dzdcY) / (dzdcX * dzdcX + dzdcY * dzdcY)
            // resultImag = (diffY * dzdcX - diffX * dzdcY) / (dzdcX * dzdcX + dzdcY * dzdcY)

            auto resultReal = (diffX * dzdcX + diffY * dzdcY) / (dzdcX * dzdcX + dzdcY * dzdcY);
            HdrReduce(resultReal);

            auto resultImag = (diffY * dzdcX - diffX * dzdcY) / (dzdcX * dzdcX + dzdcY * dzdcY);
            HdrReduce(resultImag);

            decompressed->m_FullOrbit[i].x += resultReal;
            HdrReduce(decompressed->m_FullOrbit[i].x);
            decompressed->m_FullOrbit[i].y += resultImag;
            HdrReduce(decompressed->m_FullOrbit[i].y);

            if constexpr (trackCountsPerIndex) {
                countsPerIndex[i]++;
                indexTouched.push_back(i);
            }

            //HdrReduce(decompressed->m_FullOrbit[i].x);
            //HdrReduce(decompressed->m_FullOrbit[i].y);
        }
        };

    IterTypeFull uncompressedCounter = 0;
    IterTypeFull i = 0;
    IterTypeFull UncorrectedOrbitBegin = 1;
    for (; i < targetUncompressedIters; i++) {
        if (i == nextWayPoint.u.f.CompressionIndex) {
            CorrectOrbit(UncorrectedOrbitBegin, i, nextWayPoint.x - zx, nextWayPoint.y - zy);
            UncorrectedOrbitBegin = i + 1;
            zx = nextWayPoint.x;
            zy = nextWayPoint.y;
            bool Rebase = nextWayPoint.u.f.Rebase;
            wayPointIndex++;
            nextWayPoint = m_FullOrbit[wayPointIndex];
            if (Rebase) {
                break;
            }
        }
        uncompressedCounter++;
        decompressed->AddUncompressedIteration({ zx, zy });

        if constexpr (trackCountsPerIndex) {
            countsPerIndex[decompressed->GetCountOrbitEntries() - 1]++;
            indexTouched.push_back(decompressed->GetCountOrbitEntries() - 1);
        }

        //z = z * z + c;
        auto zx_old = zx;
        zx = zx * zx - zy * zy + m_OrbitXLow;
        HdrReduce(zx);
        zy = T{ 2.0f } *zx_old * zy + m_OrbitYLow;
        HdrReduce(zy);
    }
    const auto Two = T{ 2.0f };
    IterTypeFull j = 0;
    auto dzX = zx;
    auto dzY = zy;
    for (; i < targetUncompressedIters; i++, j++) {
        zx = dzX + decompressed->m_FullOrbit[j].x;
        zy = dzY + decompressed->m_FullOrbit[j].y;

        if (i == nextWayPoint.u.f.CompressionIndex) {
            if (nextWayPoint.u.f.Rebase) {
                dzX = zx;
                dzY = zy;
                j = 0;
            }

            CorrectOrbit(UncorrectedOrbitBegin, i, nextWayPoint.x - dzX, nextWayPoint.y - dzY);
            UncorrectedOrbitBegin = i + 1;
            dzX = nextWayPoint.x;
            dzY = nextWayPoint.y;
            zx = dzX + decompressed->m_FullOrbit[j].x;
            zy = dzY + decompressed->m_FullOrbit[j].y;
            wayPointIndex++;
            nextWayPoint = m_FullOrbit[wayPointIndex];
        } else if (i == nextRebase) {
            rebaseIndex++;
            nextRebase = m_Rebases[rebaseIndex];
            dzX = zx;
            dzY = zy;
            j = 0;
        } else {
            // std::norm(z)
            auto norm_z = zx * zx + zy * zy;
            HdrReduce(norm_z);

            // std::norm(dz)
            auto norm_dz = dzX * dzX + dzY * dzY;
            HdrReduce(norm_dz);

            if (HdrCompareToBothPositiveReducedLT(norm_z, norm_dz)) {
                // dz = z;
                dzX = zx;
                dzY = zy;

                j = 0;
            }
        }

        uncompressedCounter++;
        decompressed->AddUncompressedIteration({ zx, zy });

        if constexpr (trackCountsPerIndex) {
            countsPerIndex[decompressed->GetCountOrbitEntries() - 1]++;
            indexTouched.push_back(decompressed->GetCountOrbitEntries() - 1);
        }

        // dz = (Z[j] * 2 + dz) * dz;

        //dzX = (decompressed->m_FullOrbit[j].x * Two + dzX) * dzX;
        //HdrReduce(dzX);
        //dzY = MinusTwo * decompressed->m_FullOrbit[j].y * dzY - dzY * dzY;
        //HdrReduce(dzY);

        const auto dzX_old = dzX;
        dzX = Two * decompressed->m_FullOrbit[j].x * dzX - Two * decompressed->m_FullOrbit[j].y * dzY + dzX * dzX - dzY * dzY;
        HdrReduce(dzX);
        dzY = Two * decompressed->m_FullOrbit[j].x * dzY + Two * decompressed->m_FullOrbit[j].y * dzX_old + Two * dzX_old * dzY;
        HdrReduce(dzY);
    }

    if constexpr (trackCountsPerIndex) {
        std::unordered_map<IterTypeFull, IterTypeFull> countsPerIndexMap;
        for (size_t i = 0; i < countsPerIndex.size(); i++) {
            countsPerIndexMap[countsPerIndex[i]]++;
        }

        std::vector<std::pair<IterTypeFull, IterTypeFull>> countsPerIndexVec;
        countsPerIndexVec.reserve(countsPerIndexMap.size());
        for (const auto &pair : countsPerIndexMap) {
            countsPerIndexVec.push_back(pair);
        }

        std::sort(countsPerIndexVec.begin(), countsPerIndexVec.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.first < rhs.first;
            });

        std::ofstream out("countsPerIndex.txt");
        for (const auto &pair : countsPerIndexVec) {
            out << pair.first << " " << pair.second << std::endl;
        }

        std::ofstream out2("indexTouched.txt");
        for (const auto &index : indexTouched) {
            out2 << index << std::endl;
        }
    }

    return decompressed;
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SaveOrbit(std::wstring filename) const {
    auto outFile = filename.empty() ? GenFilename(GrowableVectorTypes::DebugOutput) : filename;

    std::ofstream out(outFile);
    if (!out.is_open()) {
        ::MessageBox(nullptr, L"Failed to open file for writing 3", L"", MB_OK | MB_APPLMODAL);
        return;
    }

    out << "m_OrbitX: " << HdrToString<false>(this->m_OrbitX) << std::endl;
    out << "m_OrbitY: " << HdrToString<false>(this->m_OrbitY) << std::endl;
    out << "m_OrbitXStr: " << m_OrbitXStr << std::endl;
    out << "m_OrbitYStr: " << m_OrbitYStr << std::endl;
    out << "m_OrbitXLow: " << HdrToString<false>(m_OrbitXLow) << std::endl;
    out << "m_OrbitYLow: " << HdrToString<false>(m_OrbitYLow) << std::endl;
    out << "m_MaxRadius: " << HdrToString<false>(m_MaxRadius) << std::endl;
    out << "m_MaxIterations: " << this->m_MaxIterations << std::endl;
    out << "m_PeriodMaybeZero: " << m_PeriodMaybeZero << std::endl;
    out << "m_CompressionErrorExp: " << m_CompressionErrorExp << std::endl;
    out << "m_IntermediateCompressionErrorExp: " << m_IntermediateCompressionErrorExp << std::endl;
    out << "m_RefOrbitOptions: " << static_cast<int>(m_RefOrbitOptions) << std::endl;
    out << "m_FullOrbit: " << m_FullOrbit.GetSize() << std::endl;
    out << "m_UncompressedItersInOrbit: " << GetCountOrbitEntries() << std::endl;
    out << "m_AuthoritativePrecisionInBits: " << m_AuthoritativePrecisionInBits << std::endl;

    // Write out all values in m_FullOrbit:
    for (size_t i = 0; i < GetCompressedOrUncompressedOrbitSize(); i++) {
        out << "m_FullOrbit[" << i << "].x: " << HdrToString<false>(m_FullOrbit[i].x) << std::endl;
        out << "m_FullOrbit[" << i << "].y: " << HdrToString<false>(m_FullOrbit[i].y) << std::endl;

        if constexpr (PExtras == PerturbExtras::SimpleCompression) {
            out << "m_FullOrbit[" << i << "].CompressionIndex: " << m_FullOrbit[i].u.f.CompressionIndex << std::endl;
            out << "m_FullOrbit[" << i << "].Rebase: " << m_FullOrbit[i].u.f.Rebase << std::endl;
        }
    }

    if constexpr (PExtras == PerturbExtras::SimpleCompression) {
        out << "m_Rebases: " << m_Rebases.size() << std::endl;
        for (size_t i = 0; i < m_Rebases.size(); i++) {
            out << "m_Rebases[" << i << "]: " << m_Rebases[i] << std::endl;
        }
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

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SaveOrbitBin(std::ofstream &out) const
    requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>()) {

    // This is basically how Imagina stores it so we'll just match it.  TODO Why *16?
    // Are we just counting bits differently or something?  We don't have pow2 either
    // Imagina::HRReal prec{ pow2(-double(m_OrbitX.precisionInBits())) * 16.0_hr };
    Imagina::HRReal prec{ -static_cast<int64_t>(this->m_OrbitX.precisionInBits()), 2 };
    Imagina::HRReal maxRadius{ m_MaxRadius };
    Imagina::ReferenceTrivialContent content = {
        prec,
        {},
        maxRadius
    };

    out.write(reinterpret_cast<const char *>(&content), sizeof(content));

    // Note: m_FullOrbit.size() is the number of iterations in the compressed orbit.
    // Instead, we need to write out the number of iterations in the
    // decompressed orbit, which is m_UncompressedItersInOrbit.
    std::complex<double> refc{ static_cast<double>(this->m_OrbitX), static_cast<double>(this->m_OrbitY) };
    Imagina::LAReferenceTrivialContent laContent {
        refc,
        m_UncompressedItersInOrbit - 1, // TODO -1?
        this->m_MaxIterations,
        {}, // DoublePrecisionPT
        {}, // DirectEvaluate
        m_PeriodMaybeZero != 0,
        {}, // UseAT
        {}, // AT
        {} // LAStageCount
    };

    out.write(reinterpret_cast<const char *>(&laContent), sizeof(laContent));

    size_t orbitSize = GetCompressedOrbitSize();
    out.write(reinterpret_cast<const char *>(&orbitSize), sizeof(orbitSize));

    // Write out all values in m_FullOrbit:
    if constexpr (TemplateHelpers::IsHDR) {
        Imagina::HRReal x;
        Imagina::HRReal y;
        CompressionIndexField compressionIndex;

        for (size_t i = 0; i < GetCompressedOrbitSize(); i++) {
            x.setExp(m_FullOrbit[i].x.getExp());
            x.setMantissa(m_FullOrbit[i].x.getMantissa());
            out.write(reinterpret_cast<const char *>(&x), sizeof(x));

            y.setExp(m_FullOrbit[i].y.getExp());
            y.setMantissa(m_FullOrbit[i].y.getMantissa());
            out.write(reinterpret_cast<const char *>(&y), sizeof(y));

            // Compression index
            compressionIndex.u.f.CompressionIndex = m_FullOrbit[i].u.f.CompressionIndex;
            compressionIndex.u.f.Rebase = m_FullOrbit[i].u.f.Rebase;
            out.write(reinterpret_cast<const char *>(&compressionIndex), sizeof(compressionIndex));
        }
    } else {
        double x;
        double y;
        CompressionIndexField compressionIndex;

        for (size_t i = 0; i < GetCompressedOrbitSize(); i++) {
            x = static_cast<T>(m_FullOrbit[i].x);
            out.write(reinterpret_cast<const char *>(&x), sizeof(x));

            y = static_cast<T>(m_FullOrbit[i].y);
            out.write(reinterpret_cast<const char *>(&y), sizeof(y));

            // Compression index
            compressionIndex.u.f.CompressionIndex = m_FullOrbit[i].u.f.CompressionIndex;
            compressionIndex.u.f.Rebase = m_FullOrbit[i].u.f.Rebase;
            out.write(reinterpret_cast<const char *>(&compressionIndex), sizeof(compressionIndex));
        }
    }

    // Save rebases
    size_t rebaseSize = m_Rebases.size();
    out.write(reinterpret_cast<const char *>(&rebaseSize), sizeof(rebaseSize));
    out.write(reinterpret_cast<const char *>(m_Rebases.data()), rebaseSize * sizeof(IterTypeFull));
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::SaveOrbitLocation(std::ofstream &file) const {
    this->m_OrbitX.SaveToStream(file);
    file.flush();
    this->m_OrbitY.SaveToStream(file);
    file.flush();
}

template<typename IterType, class T, PerturbExtras PExtras>
void PerturbationResults<IterType, T, PExtras>::LoadOrbitBin(
    HighPrecision orbitX,
    HighPrecision orbitY,
    const Imagina::HRReal &halfH,
    std::ifstream &file)
    requires(PExtras == PerturbExtras::SimpleCompression) // std::is_same_v<T, HDRFloat<double>> && 
{
    constexpr bool singleStepHelper = false;

    // Notes
    // - RelativePrecision not used.  Imagina appears to ignore it.
    Imagina::ReferenceTrivialContent content;
    file.read(reinterpret_cast<char *>(&content), sizeof(content));

    // Notes
    // - Refc not used.  We convert it from the high precision orbitX and orbitY.
    //   Imagina uses it though.
    // We recalculate these next items, and it looks like Imagina does too.  All
    // are effectively not used during save/load.
    // - UseAT
    // - LAStageCount
    // - AT structure
    Imagina::LAReferenceTrivialContent laContent;
    file.read(reinterpret_cast<char *>(&laContent), sizeof(laContent));

    // TODO ValidRadius not used?
    // T radius{};
    // radius.setExp(static_cast<int32_t>(content.ValidRadius.getExp()));
    // radius.setMantissa(content.ValidRadius.getMantissa());

    T radius;
    if constexpr (TemplateHelpers::IsHDR) {
        radius.setExp(static_cast<int32_t>(halfH.getExp()));
        radius.setMantissa(static_cast<SubType>(halfH.getMantissa()));
    } else {
        radius = static_cast<T>(halfH.toDouble());
    }

    InitResults(
        RefOrbitCalc::ReuseMode::DontSaveForReuse,
        orbitX,
        orbitY,
        radius,
        static_cast<IterType>(laContent.MaxIt),
        0);

    if (laContent.IsPeriodic) {
        // TODO + 1?
        m_PeriodMaybeZero = static_cast<IterType>(laContent.RefIt) + 1;
    } else {
        m_PeriodMaybeZero = 0;
    }

    size_t compressedSize;
    file.read(reinterpret_cast<char *>(&compressedSize), sizeof(compressedSize));
    m_FullOrbit.MutableResize(compressedSize);

    // Set the target decompressed iteration count.
    // This is the number of iterations that will be in the decompressed orbit.
    // TODO + 1?
    m_UncompressedItersInOrbit = static_cast<IterType>(laContent.RefIt) + 1;

    for (size_t i = 0; i < compressedSize; i++) {
        if constexpr (TemplateHelpers::IsHDR) {
            Imagina::HRReal x;
            Imagina::HRReal y;
            file.read(reinterpret_cast<char *>(&x), sizeof(x));
            file.read(reinterpret_cast<char *>(&y), sizeof(y));

            m_FullOrbit[i].x.setMantissa(static_cast<SubType>(x.getMantissa()));

            const auto newExp = static_cast<int32_t>(x.getExp());
            m_FullOrbit[i].x.setExp(newExp);

            m_FullOrbit[i].y.setMantissa(static_cast<SubType>(y.getMantissa()));

            const auto newExp2 = static_cast<int32_t>(y.getExp());
            m_FullOrbit[i].y.setExp(newExp2);
        } else {
            double x;
            double y;
            file.read(reinterpret_cast<char *>(&x), sizeof(x));
            file.read(reinterpret_cast<char *>(&y), sizeof(y));

            m_FullOrbit[i].x = static_cast<T>(x);
            m_FullOrbit[i].y = static_cast<T>(y);
        }

        // Compression index
        CompressionIndexField compressionIndex;
        file.read(reinterpret_cast<char *>(&compressionIndex), sizeof(compressionIndex));

        // Low order 63 bits
        m_FullOrbit[i].u.f.CompressionIndex = compressionIndex.u.f.CompressionIndex;

        // High order bit
        m_FullOrbit[i].u.f.Rebase = compressionIndex.u.f.Rebase;
    }

    if constexpr (singleStepHelper) {
        uint64_t curPos3 = file.tellg();
    }

    // Rebases
    size_t rebaseSize;
    file.read((char *)&rebaseSize, sizeof(rebaseSize));
    m_Rebases.resize(rebaseSize);

    file.read((char *)m_Rebases.data(), rebaseSize * sizeof(IterTypeFull));

    // Append empty elements to m_Rebases and m_FullOrbit
    m_FullOrbit.PushBack({});
    m_Rebases.push_back(0);
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
    requires (PExtras == PerturbExtras::SimpleCompression && !Introspection::IsTDblFlt<T>()) {

    // This should only run if compression is enabled.
    static_assert(PExtras == PerturbExtras::SimpleCompression, "!");

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

#define InstantiatePerturbationResult(T, PExtras) \
    template class PerturbationResults<uint32_t, T, PExtras>; \
    template class PerturbationResults<uint64_t, T, PExtras>;

InstantiatePerturbationResult(float, PerturbExtras::Disable);
InstantiatePerturbationResult(float, PerturbExtras::Bad);
InstantiatePerturbationResult(float, PerturbExtras::SimpleCompression);

InstantiatePerturbationResult(double, PerturbExtras::Disable);
InstantiatePerturbationResult(double, PerturbExtras::Bad);
InstantiatePerturbationResult(double, PerturbExtras::SimpleCompression);

InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiatePerturbationResult(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<float>, PerturbExtras::SimpleCompression);

InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiatePerturbationResult(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

#define InstantiateRefOrbitCompressor(T, PExtras) \
    template class RefOrbitCompressor<uint32_t, T, PExtras>; \
    template class RefOrbitCompressor<uint64_t, T, PExtras>;

InstantiateRefOrbitCompressor(float, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(float, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(float, PerturbExtras::SimpleCompression);

InstantiateRefOrbitCompressor(double, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(double, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(double, PerturbExtras::SimpleCompression);

InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<float>, PerturbExtras::SimpleCompression);

InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateRefOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename IterType, class T, PerturbExtras PExtras>
SimpleIntermediateOrbitCompressor<IterType, T, PExtras>::SimpleIntermediateOrbitCompressor(
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
SimpleIntermediateOrbitCompressor<IterType, T, PExtras>::~SimpleIntermediateOrbitCompressor() {
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
void SimpleIntermediateOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration(
    const mpf_t incomingZx,
    const mpf_t incomingZy,
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

#define InstantiateSimpleIntermediateOrbitCompressor(T, PExtras) \
    template class SimpleIntermediateOrbitCompressor<uint32_t, T, PExtras>; \
    template class SimpleIntermediateOrbitCompressor<uint64_t, T, PExtras>;

InstantiateSimpleIntermediateOrbitCompressor(float, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(float, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(float, PerturbExtras::SimpleCompression);

InstantiateSimpleIntermediateOrbitCompressor(double, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(double, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(double, PerturbExtras::SimpleCompression);

InstantiateSimpleIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::SimpleCompression);

InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateSimpleIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

/////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename IterType, class T, PerturbExtras PExtras>
MaxIntermediateOrbitCompressor<IterType, T, PExtras>::MaxIntermediateOrbitCompressor(
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
    Err{},
    Constant1{},
    Constant2{},
    Threshold2{},
    I{},
    J{},
    PrevWayPointIteration{},
    ItersSinceLastWrite{},
    PhaseDone{},
    Zx{},
    Zy{},
    Dzx{},
    Dzy{},
    DzxOld{},
    Temp{},
    NormTemp{},
    NormInternalTemp{},
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

    mpf_init2(Err, AuthoritativeReuseExtraPrecisionInBits);

    mpf_init2(Constant1, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set_d(Constant1, 0x1.0p-4); // TODO

    mpf_init2(Constant2, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set_d(Constant2, 0x1.000001p0); // TODO

    mpf_init2(Threshold2, AuthoritativeReuseExtraPrecisionInBits);
    mpf_set(Threshold2, CompressionError);

    mpf_init2(Zx, AuthoritativeReuseExtraPrecisionInBits);
    mpf_init2(Zy, AuthoritativeReuseExtraPrecisionInBits);
    mpf_init2(Dzx, AuthoritativeReuseExtraPrecisionInBits);
    mpf_init2(Dzy, AuthoritativeReuseExtraPrecisionInBits);
    mpf_init2(DzxOld, AuthoritativeReuseExtraPrecisionInBits);

    for (auto &curTemp : Temp) {
        mpf_init2(curTemp, AuthoritativeReuseExtraPrecisionInBits);
    }

    for (auto &curTemp : NormTemp) {
        mpf_init2(curTemp, AuthoritativeReuseExtraPrecisionInBits);
    }

    for (auto &curTemp : NormInternalTemp) {
        mpf_init2(curTemp, AuthoritativeReuseExtraPrecisionInBits);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
MaxIntermediateOrbitCompressor<IterType, T, PExtras>::~MaxIntermediateOrbitCompressor() {
    mpf_clear(zx);
    mpf_clear(zy);
    mpf_clear(cx);
    mpf_clear(cy);
    mpf_clear(Two);
    mpf_clear(CompressionError);
    mpf_clear(ReducedZx);
    mpf_clear(ReducedZy);

    mpf_clear(Err);

    mpf_clear(Constant1);
    mpf_clear(Constant2);
    mpf_clear(Threshold2);

    mpf_clear(Zx);
    mpf_clear(Zy);
    mpf_clear(Dzx);
    mpf_clear(Dzy);
    mpf_clear(DzxOld);

    for (auto &curTemp : Temp) {
        mpf_clear(curTemp);
    }

    for (auto &curTemp : NormTemp) {
        mpf_clear(curTemp);
    }

    for (auto &curTemp : NormInternalTemp) {
        mpf_clear(curTemp);
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void MaxIntermediateOrbitCompressor<IterType, T, PExtras>::WriteResultsForTesting() {
    // Iterate over the intermediate results, convert to HdrFloat, and write to file.
    // Use GetUncompressedReuseEntries
    std::ofstream out("uncompressedReuseEntries.txt");
    for (size_t i = 0; i < results.m_ReuseX.size(); i++) {
        out << "m_ReuseX[" << i << "]: " << HdrToString<false>(results.m_ReuseX[i]) << std::endl;
        out << "m_ReuseY[" << i << "]: " << HdrToString<false>(results.m_ReuseY[i]) << std::endl;
        out << "m_ReuseIndices[" << i << "]: " << results.m_ReuseIndices[i] << std::endl;
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void MaxIntermediateOrbitCompressor<IterType, T, PExtras>::MaybeAddCompressedIteration(
    const mpf_t incomingZx,
    const mpf_t incomingZy,
    IterTypeFull index) {

    // This code is broken.  It is not clear how to fix it without compromising runtime memory
    // consumption.  The initial thought was we could extend "max compression" (see ::CompressMax)
    // to runtime intermediate orbit compression, but this is not feasible.  The intermediate orbit
    // compression objective is to minimize the number of reference orbit entries that are
    // stored in memory.  But this compression algorithm relies on most of them being in memory
    // in order to achieve a decent compression ratio.

    assert(false); // Not implemented

    auto normZ = [&](mpf_t &out, const mpf_t x, const mpf_t y) -> void {
        // auto norm_z = x * x + y * y;
        mpf_mul(NormInternalTemp[0], x, x);
        mpf_mul(NormInternalTemp[1], y, y);
        mpf_add(out, NormInternalTemp[0], NormInternalTemp[1]);
        };

    auto normZTimesT = [&](mpf_t &out, const mpf_t x, const mpf_t y, const mpf_t t) -> void {
        // auto norm_z = (x * x + y * y) * t;
        mpf_mul(NormInternalTemp[0], x, x);
        mpf_mul(NormInternalTemp[1], y, y);
        mpf_add(NormInternalTemp[0], NormInternalTemp[0], NormInternalTemp[1]);
        mpf_mul(out, NormInternalTemp[0], t);
        };

    // I == i from original code

    if (PhaseDone == 0) {
        // const auto norm_z = normZ(m_FullOrbit[i].x, m_FullOrbit[i].y);
        normZ(NormTemp[0], incomingZx, incomingZy);

        // if (HdrCompareToBothPositiveReducedLT(norm_z, constant1)) {
        if (mpf_cmp(NormTemp[0], Constant1) < 0) {
            mpf_set(Zx, incomingZx);
            mpf_set(Zy, incomingZy);

            results.AddUncompressedReusedEntry(Zx, Zy, index);

            PhaseDone++;
        } else {
            // auto err = normZTimesT(zx - m_FullOrbit[i].x, zy - m_FullOrbit[i].y, threshold2);
            mpf_sub(Temp[0], Zx, incomingZx);
            mpf_sub(Temp[1], Zy, incomingZy);

            normZTimesT(NormTemp[1], Temp[0], Temp[1], Threshold2);
            if (mpf_cmp(NormTemp[1], NormTemp[0]) >= 0) {
                mpf_set(Zx, incomingZx);
                mpf_set(Zy, incomingZy);

                results.AddUncompressedReusedEntry(Zx, Zy, index);
            }
        }

        if (PhaseDone == 0) {
            // z = z * z + c;
            const auto &ZxOld = Temp[2];
            mpf_set(Temp[2], Zx);

            mpf_mul(Temp[0], Zx, Zx);
            mpf_mul(Temp[1], Zy, Zy);
            mpf_sub(Zx, Temp[0], Temp[1]);
            mpf_add(Zx, Zx, cx);

            mpf_mul(Temp[0], Two, ZxOld);
            mpf_mul(Zy, Temp[0], Zy);
            mpf_add(Zy, Zy, cy);
            I++;
        }

        return;
    }

    // Initialize persistent state
    if (PhaseDone == 1) {
        mpf_set(Dzx, zx);
        mpf_set(Dzy, zy);

        PrevWayPointIteration = I;
        mpf_set(DzxOld, Dzx);

        // Use Temp array for intermediate calculations
        // dzX = Two * cx * dzX - Two * cy * dzY + dzX * dzX - dzY * dzY;
        // dzY = Two * cx * dzY + Two * cy * dzX_old + Two * dzX_old * dzY;

        // Temp[1] = Two * cx * dzX
        mpf_mul(Temp[0], Two, cx);
        mpf_mul(Temp[1], Temp[0], Dzx);

        // Temp[3] = Two * cy * dzY
        mpf_mul(Temp[2], Two, cy);
        mpf_mul(Temp[3], Temp[2], Dzy);

        // Temp[4] = Two * cx * dzX - Two * cy * dzY
        mpf_sub(Temp[4], Temp[1], Temp[3]);

        // Temp[5] = dzX * dzX
        mpf_mul(Temp[5], Dzx, Dzx);

        // Temp[6] = dzY * dzY
        mpf_mul(Temp[6], Dzy, Dzy);

        // dzX = Two * cx * dzX - Two * cy * dzY + dzX * dzX - dzY * dzY;
        mpf_sub(Dzx, Temp[4], Temp[6]);
        mpf_add(Dzx, Dzx, Temp[5]);

        //////////////////////////////////////////////

        // Temp[8] = Two * cx * dzY
        mpf_mul(Temp[7], Two, cx);
        mpf_mul(Temp[8], Temp[7], Dzy);

        // Temp[10] = Two * cy * dzX_old
        mpf_mul(Temp[9], Two, cy);
        mpf_mul(Temp[10], Temp[9], DzxOld);

        // Temp[12] = Two * dzX_old * dzY
        mpf_mul(Temp[11], Two, DzxOld);
        mpf_mul(Temp[12], Temp[11], Dzy);

        // dzY = Two * cx * dzY + Two * cy * dzX_old + Two * dzX_old * dzY;
        mpf_add(Dzy, Temp[8], Temp[10]);
        mpf_add(Dzy, Dzy, Temp[12]);

        I++;
        J = 1;
        ItersSinceLastWrite = 0;

        PhaseDone++;
    }

    // Main loop
    if (PhaseDone == 2) {
        // TODO: wrong, m_FullOrbit[J] isn't the same as m_FullOrbit[I]
        // We could fix this by keeping everything in memory but then that would defeat the purpose
        mpf_add(Zx, Dzx, incomingZx);
        mpf_add(Zy, Dzy, incomingZy);

        // const auto norm_z_orig = normZ(zx, zy);
        normZ(NormTemp[0], Zx, Zy);

        // const auto norm_dz_orig = normZTimesT(dzX, dzY, constant2);
        normZTimesT(NormTemp[1], Dzx, Dzy, Constant2);

        // const auto err = normZTimesT(zx - m_FullOrbit[i].x, zy - m_FullOrbit[i].y, threshold2);
        mpf_sub(Temp[0], Zx, incomingZx);
        mpf_sub(Temp[1], Zy, incomingZy);
        normZTimesT(Err, Temp[0], Temp[1], Threshold2);

        // const bool condition1 = j >= PrevWayPointIteration;
        // const bool condition2 = HdrCompareToBothPositiveReducedGE(err, norm_z_orig);
        // const bool condition3 = itersSinceLastWrite >= MaxItersSinceLastWrite;
        const bool Condition1 = J >= PrevWayPointIteration;
        const bool Condition2 = mpf_cmp(Err, NormTemp[0]) >= 0;
        const bool Condition3 = ItersSinceLastWrite >= MaxItersSinceLastWrite;

        if (Condition1 || Condition2) {
            PrevWayPointIteration = I;
            mpf_set(Zx, incomingZx);
            mpf_set(Zy, incomingZy);

            // dzX = zx - m_FullOrbit[j].x;
            // dzY = zy - m_FullOrbit[j].y;
            // 
            // TODO: mpf_sub(Dzx, Zx, ...
            // TODO: mpf_sub(Dzy, Zy, ...

            normZ(NormTemp[0], Zx, Zy);
            normZ(NormTemp[1], Dzx, Dzy);

            if (mpf_cmp(NormTemp[0], NormTemp[1]) < 0 || (I - J) * 4 < I) {
                mpf_set(Dzx, incomingZx);
                mpf_set(Dzy, incomingZy);
                J = 0;

                results.AddUncompressedReusedEntry(Dzx, Dzy, index);
                ItersSinceLastWrite = 0;
            } else {
                mpf_set(ReducedZx, incomingZx);
                mpf_set(ReducedZy, incomingZy);

                results.AddUncompressedReusedEntry(ReducedZx, ReducedZy, index);
                ItersSinceLastWrite = 0;
            }
        } else if (Condition3) {
            mpf_set(ReducedZx, incomingZx);
            mpf_set(ReducedZy, incomingZy);

            results.AddUncompressedReusedEntry(ReducedZx, ReducedZy, index);
            ItersSinceLastWrite = 0;
        } else if (mpf_cmp(NormTemp[0], NormTemp[1]) < 0) {
            mpf_set(Dzx, incomingZx);
            mpf_set(Dzy, incomingZy);
            J = 0;

            results.AddUncompressedRebase(I, index);
        }

        mpf_set(DzxOld, Dzx);

        // TODO: wrong, m_FullOrbit[J] isn't the same as m_FullOrbit[I]
        // dzX = Two * m_FullOrbit[j].x * dzX - Two * m_FullOrbit[j].y * dzY + dzX * dzX - dzY * dzY;
        // dzY = Two * m_FullOrbit[j].x * dzY + Two * m_FullOrbit[j].y * dzX_old2 + Two * dzX_old2 * dzY;
        mpf_mul(Temp[0], Two, cx);
        mpf_mul(Temp[1], Temp[0], Dzx);
        mpf_mul(Temp[2], Two, cy);
        mpf_mul(Temp[3], Temp[2], Dzy);
        mpf_sub(Temp[4], Temp[1], Temp[3]);
        mpf_mul(Temp[5], Two, DzxOld);
        mpf_mul(Temp[6], Two, cy);
        mpf_mul(Temp[7], Temp[6], DzxOld);
        mpf_mul(Temp[8], Two, DzxOld);
        mpf_mul(Temp[9], Temp[8], Dzy);
        mpf_add(Dzx, Temp[4], Temp[5]);
        mpf_add(Dzy, Temp[7], Temp[9]);

        ItersSinceLastWrite++;
    }
}

template<typename IterType, class T, PerturbExtras PExtras>
void MaxIntermediateOrbitCompressor<IterType, T, PExtras>::CompleteResults() {
    assert(PhaseDone == 2);
    results.AddUncompressedReusedEntry(Dzx, Dzy, ~0ull);
    results.m_Rebases.push_back(~0ull);

    constexpr bool EnableWriteResultsForTesting = true;
    if constexpr (EnableWriteResultsForTesting) {
        WriteResultsForTesting();
    }
}


#define InstantiateMaxIntermediateOrbitCompressor(T, PExtras) \
    template class MaxIntermediateOrbitCompressor<uint32_t, T, PExtras>; \
    template class MaxIntermediateOrbitCompressor<uint64_t, T, PExtras>;

InstantiateMaxIntermediateOrbitCompressor(float, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(float, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(float, PerturbExtras::SimpleCompression);

InstantiateMaxIntermediateOrbitCompressor(double, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(double, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(double, PerturbExtras::SimpleCompression);

InstantiateMaxIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiateMaxIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<float>, PerturbExtras::SimpleCompression);

InstantiateMaxIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiateMaxIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
InstantiateMaxIntermediateOrbitCompressor(HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::SimpleCompression);

