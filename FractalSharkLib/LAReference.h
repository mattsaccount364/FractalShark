#pragma once

#include "BenchmarkData.h"
#include "HDRFloat.h"
#include "Introspection.h"
#include "LAInfoDeep.h"
#include "LAInfoI.h"
#include "LAstep.h"
#include "Vectors.h"
#include <cmath>
#include <cstring>

#include <thread>
#include <vector>

template <typename IterType, class HDRFloat, class SubType> class ATInfo;

template <typename IterType, class Float> class LAStageInfo;

class RefOrbitCalc;

template <typename IterType, class T, PerturbExtras PExtras> class RuntimeDecompressor;

template <typename IterType, class T, PerturbExtras PExtras> class PerturbationResults;

template <typename IterType, class T, class SubType> class GPU_LAReference;

void SetCopyThreadDescription();

// Note: The helper functions in this header cannot be moved to the .cpp file
// unless we can properly instantiate the template with CudaDblflt, which we
// currently cannot do.
template <typename IterType, class Float, class SubType, PerturbExtras PExtras> class LAReference {
private:
    static constexpr bool IsHDR = std::is_same<Float, ::HDRFloat<float>>::value ||
                                  std::is_same<Float, ::HDRFloat<double>>::value ||
                                  std::is_same<Float, ::HDRFloat<CudaDblflt<MattDblflt>>>::value ||
                                  std::is_same<Float, ::HDRFloat<CudaDblflt<dblflt>>>::value;
    using FloatComplexT =
        std::conditional<IsHDR, ::HDRFloatComplex<SubType>, ::FloatComplex<SubType>>::type;

    template <typename OtherIterType, class OtherFloat, class OtherSubType> friend class GPU_LAReference;

    // TODO this is overly broad -- many types don't need these friends
    friend class LAReference<IterType, float, float, PExtras>;
    friend class LAReference<IterType, double, double, PExtras>;
    friend class LAReference<IterType, CudaDblflt<dblflt>, CudaDblflt<dblflt>, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<float>, float, PExtras>;
    friend class LAReference<IterType, ::HDRFloat<double>, double, PExtras>;
    friend class LAReference<IterType,
                             ::HDRFloat<CudaDblflt<MattDblflt>>,
                             CudaDblflt<MattDblflt>,
                             PExtras>;
    friend class LAReference<IterType, ::HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>, PExtras>;

    static const int lowBound = 64;
    static const int periodDivisor;

public:
    LAReference() = delete;
    LAReference(const LAReference &other) = delete;
    LAReference &operator=(const LAReference &other) = delete;
    LAReference &operator=(LAReference &&other) = delete;
    LAReference(LAReference &&other) = delete;

    LAReference(LAParameters parameters,
                AddPointOptions addPointOptions,
                std::wstring lasFilename,
                std::wstring laStagesFilename)
    requires(Introspection::TestPExtras<PExtras>::value)
        : m_AddPointOptions(addPointOptions), m_UseAT{}, m_AT{}, m_LAStageCount{},
          m_LAParameters{parameters}, m_IsValid{}, m_LAs(addPointOptions, lasFilename.c_str()),
          m_LAStages(addPointOptions, laStagesFilename.c_str()), m_BenchmarkDataLA{}
    {

        static_assert(PExtras != PerturbExtras::MaxCompression,
                      "MaxCompression not supported in LAReference");
    }

    LAReference(AddPointOptions addPointOptions, std::wstring lasFilename, std::wstring laStagesFilename)
    requires(Introspection::TestPExtras<PExtras>::value)
        : m_AddPointOptions(addPointOptions), m_UseAT{}, m_AT{}, m_LAStageCount{}, m_LAParameters{},
          m_IsValid{}, m_LAs(addPointOptions, lasFilename.c_str()),
          m_LAStages(addPointOptions, laStagesFilename.c_str()), m_BenchmarkDataLA{}
    {

        static_assert(PExtras != PerturbExtras::MaxCompression,
                      "MaxCompression not supported in LAReference");
    }

    ~LAReference()
    requires(Introspection::TestPExtras<PExtras>::value)
    = default;

    bool
    WriteMetadata(std::ofstream &metafile) const
    {
        metafile << "LAReference:" << std::endl;
        metafile << "AddPointOptions: " << static_cast<uint64_t>(m_AddPointOptions) << std::endl;
        metafile << "UseAT(bool): " << static_cast<uint64_t>(m_UseAT) << std::endl;
        metafile << "LAStageCount: " << static_cast<uint64_t>(m_LAStageCount) << std::endl;
        metafile << "IsValid(bool): " << static_cast<uint64_t>(m_IsValid) << std::endl;

        bool res = m_LAParameters.WriteMetadata(metafile);
        if (!res) {
            return false;
        }

        return m_AT.WriteMetadata(metafile);
    }

    bool
    ReadMetadata(std::ifstream &metafile)
    {
        std::string descriptorJunk;

        // "LAReference:"
        metafile >> descriptorJunk;

        auto convert = []<typename T>(const std::string &str) {
            return static_cast<T>(std::stoll(str));
        };

        {
            std::string addPointOptions;
            metafile >> descriptorJunk;
            metafile >> addPointOptions;
            m_AddPointOptions = convert.template operator()<AddPointOptions>(addPointOptions);
        }

        {
            std::string useAt;
            metafile >> descriptorJunk;
            metafile >> useAt;
            m_UseAT = convert.template operator()<bool>(useAt);
        }

        {
            std::string laStageCount;
            metafile >> descriptorJunk;
            metafile >> laStageCount;
            const uint64_t count = std::stoull(laStageCount);
            if (count > MaxLAStages) {
                return false;
            }
            m_LAStageCount = static_cast<IterType>(count);
        }

        {
            std::string isValid;
            metafile >> descriptorJunk;
            metafile >> isValid;
            m_IsValid = convert.template operator()<bool>(isValid);
        }

        bool res = m_LAParameters.ReadMetadata(metafile);
        if (!res) {
            return false;
        }

        return m_AT.ReadMetadata(metafile) && ValidateTables();
    }

    bool
    ValidateTables() const
    {
        if (m_LAStageCount > MaxLAStages || m_LAStageCount > m_LAStages.GetSize()) {
            return false;
        }
        size_t nextIndex = 0;
        for (IterType stage = 0; stage < m_LAStageCount; ++stage) {
            const auto &descriptor = m_LAStages[stage];
            const size_t index = descriptor.LAIndex;
            const size_t count = descriptor.MacroItCount;
            if (index != nextIndex || index >= m_LAs.GetSize() || count == 0 ||
                count >= m_LAs.GetSize() - index) {
                return false;
            }
            nextIndex = index + count + 1;
        }
        return !m_IsValid || m_LAStageCount != 0;
    }

    template <class OtherT, class Other>
    void
    CopyLAReference(const LAReference<IterType, OtherT, Other, PExtras> &other)
    {
        // m_AddPointOptions is defined at construction time and not changed here.
        m_UseAT = other.m_UseAT;
        m_AT = other.m_AT;
        m_LAStageCount = other.m_LAStageCount;
        m_IsValid = other.m_IsValid;
        m_LAParameters = other.m_LAParameters;

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
            SetCopyThreadDescription();
            for (size_t i = start; i < end; i++) {
                m_LAs[i] = other.m_LAs[i];
            }
        };

        std::vector<std::thread> threads;
        if (numThreads == 1) {
            for (size_t i = 0; i < other.m_LAs.GetSize(); i++) {
                m_LAs[i] = other.m_LAs[i];
            }
        } else {
            threads.reserve(numThreads);
            for (size_t i = 0; i < numThreads; i++) {
                size_t start = i * numElementsPerThread;
                size_t end = (i + 1) * numElementsPerThread;
                if (i == numThreads - 1) {
                    end = other.m_LAs.GetSize();
                }
                threads.emplace_back(oneThread, start, end);
            }
        }

        m_LAStages.MutableResize(other.m_LAStages.GetSize());

        for (auto &thread : threads) {
            thread.join();
        }

        for (size_t i = 0; i < other.m_LAStages.GetSize(); i++) {
            m_LAStages[i] = other.m_LAStages[i];
        }

        m_BenchmarkDataLA = other.m_BenchmarkDataLA;
    }

    bool
    IsValid() const
    {
        return m_IsValid;
    }

    void
    InitializePixel(IterType maxIterations,
                    const FloatComplexT &deltaC,
                    IterType &iterations,
                    FloatComplexT &deltaZ) const
    {
        m_AT.InitializePixel(m_IsValid && m_UseAT,
                             maxIterations,
                             deltaC,
                             FloatComplexT{SubType{0}, SubType{0}},
                             iterations,
                             deltaZ);
    }

    IterType
    GetLAStageCount() const
    {
        return m_LAStageCount;
    }

    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &
    GetLAs()
    {
        return m_LAs;
    }

    const GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &
    GetLAs() const
    {
        return m_LAs;
    }

    GrowableVector<LAStageInfo<IterType, Float>> &
    GetLAStages()
    {
        return m_LAStages;
    }

    const GrowableVector<LAStageInfo<IterType, Float>> &
    GetLAStages() const
    {
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
    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> m_LAs;
    GrowableVector<LAStageInfo<IterType, Float>> m_LAStages;

    BenchmarkData m_BenchmarkDataLA;

    using ConstructionEntry = LAConstructionEntry<IterType, Float, SubType, PExtras>;
    class ConstructionTable {
        GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &m_Rows;
        GrowableVector<LAConstructionInfo<Float>> m_Info;

    public:
        ConstructionTable(GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> &rows,
                          AddPointOptions options,
                          const std::wstring &filename)
            : m_Rows{rows},
              m_Info{options == AddPointOptions::DontSave ? options : AddPointOptions::EnableWithoutSave,
                     filename.c_str()}
        {
        }
        ConstructionTable(ConstructionTable &&) = default;
        ConstructionTable(const ConstructionTable &) = delete;
        size_t
        GetSize() const
        {
            return m_Rows.GetSize();
        }
        ConstructionEntry
        operator[](size_t index) const
        {
            return {m_Rows[index], m_Info[index]};
        }
        void
        PushBack(const ConstructionEntry &entry)
        {
            if (m_Rows.GetSize() == m_Rows.GetCapacity()) {
                constexpr size_t growth = 256 * 1024 * 1024 /
                                          (sizeof(LAInfoDeep<IterType, Float, SubType, PExtras>) +
                                           sizeof(LAConstructionInfo<Float>));
                m_Rows.MutableReserveKeepFileSize(m_Rows.GetSize() + growth);
            }
            if (m_Info.GetCapacity() < m_Rows.GetCapacity()) {
                m_Info.MutableReserveKeepFileSize(m_Rows.GetCapacity());
            }
            m_Rows.PushBack(entry);
            m_Info.PushBack(entry.m_Construction);
        }
        void
        PopBack()
        {
            m_Rows.PopBack();
            m_Info.PopBack();
        }
        void
        Append(const ConstructionTable &other)
        {
            if (other.GetSize() == 0) {
                return;
            }
            const size_t begin = GetSize();
            m_Rows.MutableResize(begin + other.GetSize());
            m_Info.MutableResize(begin + other.GetSize());
            std::memcpy(m_Rows.GetData() + begin,
                        other.m_Rows.GetData(),
                        other.GetSize() * sizeof(LAInfoDeep<IterType, Float, SubType, PExtras>));
            std::memcpy(m_Info.GetData() + begin,
                        other.m_Info.GetData(),
                        other.GetSize() * sizeof(LAConstructionInfo<Float>));
        }
        Float
        GetThresholdC(size_t index) const
        {
            return m_Info[index].LAThresholdC;
        }
    };

    struct OrbitStageState {
        ConstructionEntry m_LA;
        LAInfoI<IterType> m_LAI;
        IterType m_Index{};
        IterType m_Period{};
        IterType m_PeriodBegin{};
        IterType m_PeriodEnd{};
    };

    IterType LAsize();
    static IterType CalculatePeriod(IterType maxRefIteration, double ratio, IterType stepLength);
    template <typename PerturbType>
    bool InitializeOrbitStage(const LAParametersRuntime<Float> &parameters,
                              const PerturbationResults<IterType, PerturbType, PExtras> &results,
                              IterType maxRefIteration,
                              RuntimeDecompressor<IterType, Float, PExtras> &decompressor,
                              OrbitStageState &state,
                              ConstructionTable &constructionTable)
    requires(PExtras != PerturbExtras::MaxCompression);

    template <typename PerturbType>
    ConstructionEntry MakeOrbitEntry(const LAParametersRuntime<Float> &parameters,
                                     const PerturbationResults<IterType, PerturbType, PExtras> &results,
                                     RuntimeDecompressor<IterType, Float, PExtras> &decompressor,
                                     IterType index,
                                     bool appendNext);

    void
    AppendTerminalEntry(const LAParametersRuntime<Float> &parameters,
                        FloatComplexT ref,
                        ConstructionTable &constructionTable)
    {
        constructionTable.PushBack(ConstructionEntry{parameters, ref});
    }

    template <typename PerturbType>
    bool CreateLAFromOrbit(
        const LAParametersRuntime<Float> &parameters,
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        IterType maxRefIteration,
        ConstructionTable &constructionTable)
    requires(PExtras != PerturbExtras::MaxCompression);
    template <typename PerturbType>
    bool CreateLAFromOrbitMT(
        const LAParametersRuntime<Float> &parameters,
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        IterType maxRefIteration,
        ConstructionTable &constructionTable)
    requires(PExtras != PerturbExtras::MaxCompression);
    template <typename PerturbType>
    bool CreateNewLAStage(const LAParametersRuntime<Float> &parameters,
                          const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
                          IterType maxRefIteration,
                          ConstructionTable &constructionTable);

public:
    template <typename PerturbType>
    void GenerateApproximationData(
        const PerturbationResults<IterType, PerturbType, PExtras> &PerturbationResults,
        Float radius,
        bool UseSmallExponents)
    requires(PExtras != PerturbExtras::MaxCompression);

    const BenchmarkData &
    GetBenchmarkLA() const
    {
        return m_BenchmarkDataLA;
    }

private:
    void CreateATFromLA(Float radius, bool UseSmallExponents);

public:
    bool IsLAStageInvalid(IterType stageIndex, FloatComplexT dc) const;
    IterType getLAIndex(IterType CurrentLAStage);
    IterType getMacroItCount(IterType CurrentLAStage);

    // Make preparation visible to the shared traversal so the step stays in the pixel loop.
#if defined(_MSC_VER) && !defined(__CUDACC__)
    __forceinline
#endif
        LAstep<IterType, Float, SubType, PExtras>
        getLA(IterType laIndex,
              FloatComplexT deltaZ,
              IterType stageIteration,
              IterType iterations,
              IterType maxIterations)
    {
        const IterType entryIndex = laIndex + stageIteration;
        const LAInfoI<IterType> &metadata = m_LAs[entryIndex].GetLAi();
        LAstep<IterType, Float, SubType, PExtras> step;

        const IterType stepLength = metadata.StepLength;
        const bool usable = iterations + stepLength <= maxIterations;
        if (usable) {
            LAInfoDeep<IterType, Float, SubType, PExtras> &laEntry = m_LAs[entryIndex];
            step = laEntry.Prepare(deltaZ);
            if (!step.unusable) {
                step.LAjdeep = &laEntry;
                step.Refp1Deep = (FloatComplexT)m_LAs[entryIndex + 1].getRef();
                step.step = metadata.StepLength;
            }
        } else {
            step = LAstep<IterType, Float, SubType, PExtras>();
            step.unusable = true;
        }

        step.nextStageLAindex = metadata.NextStageLAIndex;
        return step;
    }
};
