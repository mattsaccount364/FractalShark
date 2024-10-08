#pragma once
//package fractalzoomer.core.la;
//
//import fractalzoomer.core.*;
//import fractalzoomer.functions.Fractal;
//
//import java.util.Arrays;

#include <cmath>
#include "HDRFloat.h"
#include "LAstep.h"
#include "LAInfoDeep.h"
#include "LAInfoI.h"
#include "Vectors.h"
#include "BenchmarkData.h"
#include "Introspection.h"

#include <thread>
#include <vector>

template<typename IterType, class HDRFloat, class SubType>
class ATInfo;

template<typename IterType>
class LAStageInfo;

class RefOrbitCalc;

template<typename IterType, class T, PerturbExtras PExtras>
class PerturbationResults;

template<typename IterType, class T, class SubType>
class GPU_LAReference;

template<typename IterType, class Float, class SubType, PerturbExtras PExtras>
class LAReference {
private:
    static constexpr bool IsHDR =
        std::is_same<Float, ::HDRFloat<float>>::value ||
        std::is_same<Float, ::HDRFloat<double>>::value ||
        std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
        std::is_same<Float, ::HDRFloat<CudaDblflt<dblflt>>>::value;
    using FloatComplexT =
        std::conditional<
        IsHDR,
        ::HDRFloatComplex<SubType>,
        ::FloatComplex<SubType>>::type;

    friend class GPU_LAReference<IterType, Float, float>;
    friend class GPU_LAReference<IterType, Float, double>;
    friend class GPU_LAReference<IterType, Float, CudaDblflt<dblflt>>;
    friend class GPU_LAReference<IterType, Float, CudaDblflt<MattDblflt>>;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAReference<IterType, float, float, PExtras>;
    friend class LAReference<IterType, double, double, PExtras>;
    friend class LAReference<IterType, CudaDblflt<dblflt>, CudaDblflt<dblflt>, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<float>, float, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<double>, double, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>, PExtras>;

    static const int lowBound = 64;
    static const int periodDivisor;

public:
    LAReference() = delete;
    LAReference(const LAReference &other) = delete;
    LAReference &operator=(const LAReference &other) = delete;
    LAReference &operator=(LAReference &&other) = delete;
    LAReference(LAReference &&other) = delete;

    LAReference(
        LAParameters la_parameters,
        AddPointOptions addPointOptions,
        std::wstring las_filename,
        std::wstring la_stages_filename)
        requires (Introspection::TestPExtras<PExtras>::value)
        :
        m_AddPointOptions(addPointOptions),
        m_UseAT{},
        m_AT{},
        m_LAStageCount{},
        m_LAParameters{ la_parameters },
        m_IsValid{},
        m_LAs(addPointOptions, las_filename.c_str()),
        m_LAStages(addPointOptions, la_stages_filename.c_str()),
        m_BenchmarkDataLA{} {

        static_assert(PExtras != PerturbExtras::MaxCompression, "MaxCompression not supported in LAReference");
    }

    LAReference(
        AddPointOptions addPointOptions,
        std::wstring las_filename,
        std::wstring la_stages_filename)
        requires (Introspection::TestPExtras<PExtras>::value)
        :
        m_AddPointOptions(addPointOptions),
        m_UseAT{},
        m_AT{},
        m_LAStageCount{},
        m_LAParameters{ },
        m_IsValid{},
        m_LAs(addPointOptions, las_filename.c_str()),
        m_LAStages(addPointOptions, la_stages_filename.c_str()),
        m_BenchmarkDataLA{} {

        static_assert(PExtras != PerturbExtras::MaxCompression, "MaxCompression not supported in LAReference");
    }

    ~LAReference()
        requires (Introspection::TestPExtras<PExtras>::value) {
    }

    bool WriteMetadata(std::ofstream &metafile) const {
        metafile << "LAReference:" << std::endl;
        metafile << "AddPointOptions: " << static_cast<uint64_t>(m_AddPointOptions) << std::endl;
        metafile << "UseAT: " << static_cast<uint64_t>(m_UseAT) << std::endl;
        metafile << "LAStageCount: " << static_cast<uint64_t>(m_LAStageCount) << std::endl;
        metafile << "IsValid: " << static_cast<uint64_t>(m_IsValid) << std::endl;

        bool res = m_LAParameters.WriteMetadata(metafile);
        if (!res) {
            return false;
        }

        return m_AT.WriteMetadata(metafile);
    }

    bool ReadMetadata(std::ifstream &metafile) {
        std::string descriptor_string_junk;

        // "LAReference:"
        metafile >> descriptor_string_junk;

        auto convert = []<typename T>(const std::string & str) {
            return static_cast<T>(std::stoll(str));
        };

        {
            std::string addPointOptions;
            metafile >> descriptor_string_junk;
            metafile >> addPointOptions;
            m_AddPointOptions = convert.template operator() < AddPointOptions > (addPointOptions);
        }

        {
            std::string use_at;
            metafile >> descriptor_string_junk;
            metafile >> use_at;
            m_UseAT = convert.template operator() < bool > (use_at);
        }

        {
            std::string la_stage_count;
            metafile >> descriptor_string_junk;
            metafile >> la_stage_count;
            m_LAStageCount = convert.template operator() < IterType > (la_stage_count);
        }

        {
            std::string is_valid;
            metafile >> descriptor_string_junk;
            metafile >> is_valid;
            m_IsValid = convert.template operator() < bool > (is_valid);
        }

        bool res = m_LAParameters.ReadMetadata(metafile);
        if (!res) {
            return false;
        }

        return m_AT.ReadMetadata(metafile);
    }

    template<class OtherT, class Other>
    void CopyLAReference(const LAReference<IterType, OtherT, Other, PExtras> &other) {
        // m_AddPointOptions is defined at construction time and not changed here.
        m_UseAT = other.m_UseAT;
        m_AT = other.m_AT;
        m_LAStageCount = other.m_LAStageCount;
        m_IsValid = other.m_IsValid;

        m_LAs.MutableResize(other.m_LAs.GetSize());

        // Split other.m_LAs across multiple threads.
        // Use std::hardware_concurrency() to determine the number of threads.
        // Each thread will get a range of indices to copy.
        // Each thread will copy the range of indices to m_LAs.
        const auto workPerThread = 1'000'000;
        const auto altNumThreads = other.m_LAs.GetSize() / workPerThread;
        const auto maxThreads = std::thread::hardware_concurrency();
        const auto numThreadsMaybeZero = altNumThreads > maxThreads ? maxThreads : altNumThreads;
        const auto numThreads = numThreadsMaybeZero == 0 ? 1 : numThreadsMaybeZero;
        auto numElementsPerThread = other.m_LAs.GetSize() / numThreads;

        auto oneThread = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; i++) {
                m_LAs[i] = other.m_LAs[i];
            }
            };

        std::vector<std::thread> threads;
        for (size_t i = 0; i < numThreads; i++) {
            size_t start = i * numElementsPerThread;
            size_t end = (i + 1) * numElementsPerThread;
            if (i == numThreads - 1) {
                end = other.m_LAs.GetSize();
            }
            threads.push_back(std::thread(oneThread, start, end));
        }

        m_LAStages.MutableResize(other.m_LAStages.GetSize());

        for (auto &thread : threads) {
            thread.join();
        }

        for (int i = 0; i < other.m_LAStages.GetSize(); i++) {
            m_LAStages[i] = other.m_LAStages[i];
        }

        m_BenchmarkDataLA = other.m_BenchmarkDataLA;
    }

    bool IsValid() const {
        return m_IsValid;
    }

    bool UseAT() const {
        return m_UseAT;
    }

    const ATInfo<IterType, Float, SubType> &GetAT() const {
        return m_AT;
    }

    IterType GetLAStageCount() const {
        return m_LAStageCount;
    }

    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &GetLAs() {
        return m_LAs;
    }

    const GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &GetLAs() const {
        return m_LAs;
    }

    GrowableVector<LAStageInfo<IterType>> &GetLAStages() {
        return m_LAStages;
    }

    const GrowableVector<LAStageInfo<IterType>> &GetLAStages() const {
        return m_LAStages;
    }

private:
    AddPointOptions m_AddPointOptions;
    bool m_UseAT;
    ATInfo<IterType, Float, SubType> m_AT;
    IterType m_LAStageCount;
    LAParameters m_LAParameters;
    bool m_IsValid;

    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;
    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> m_LAs;
    GrowableVector<LAStageInfo<IterType>> m_LAStages;

    BenchmarkData m_BenchmarkDataLA;

    IterType LAsize();
    template<typename PerturbType>
    bool CreateLAFromOrbit(
        const LAParameters &la_parameters,
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        IterType maxRefIteration)
        requires (PExtras != PerturbExtras::MaxCompression);
    template<typename PerturbType>
    bool CreateLAFromOrbitMT(
        const LAParameters &la_parameters,
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        IterType maxRefIteration)
        requires (PExtras != PerturbExtras::MaxCompression);
    template<typename PerturbType>
    bool CreateNewLAStage(
        const LAParameters &la_parameters,
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        IterType maxRefIteration);

public:
    template<typename PerturbType>
    void GenerateApproximationData(
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        Float radius,
        bool UseSmallExponents)
        requires (PExtras != PerturbExtras::MaxCompression);

    const BenchmarkData &GetBenchmarkLA() const {
        return m_BenchmarkDataLA;
    }

private:
    void CreateATFromLA(Float radius, bool UseSmallExponents);

public:
    bool isLAStageInvalid(IterType LAIndex, FloatComplexT dc);
    IterType getLAIndex(IterType CurrentLAStage);
    IterType getMacroItCount(IterType CurrentLAStage);

    LAstep<IterType, Float, SubType, PExtras>
        getLA(IterType LAIndex,
            FloatComplexT dz,
            /*FloatComplexT dc, */IterType j,
            IterType iterations,
            IterType max_iterations);
};
