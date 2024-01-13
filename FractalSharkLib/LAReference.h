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
    LAReference(const LAReference& other) = delete;
    LAReference& operator=(const LAReference& other) = delete;
    LAReference& operator=(LAReference&& other) = delete;
    LAReference(LAReference&& other) = delete;

    LAReference(
        AddPointOptions add_point_options,
        std::wstring las_filename,
        std::wstring la_stages_filename) :
        m_AddPointOptions(add_point_options),
        m_UseAT{},
        m_AT{},
        m_LAStageCount{}, 
        m_IsValid{},
        m_LAs(add_point_options, las_filename),
        m_LAStages(add_point_options, la_stages_filename) {
    }

    bool WriteMetadata(std::ofstream& metafile) const {
        metafile << "LAReference:" << std::endl;
        metafile << "AddPointOptions: " << static_cast<uint64_t>(m_AddPointOptions) << std::endl;
        metafile << "UseAT: " << static_cast<uint64_t>(m_UseAT) << std::endl;
        metafile << "LAStageCount: " << static_cast<uint64_t>(m_LAStageCount) << std::endl;
        metafile << "IsValid: " << static_cast<uint64_t>(m_IsValid) << std::endl;

        return m_AT.WriteMetadata(metafile);
    }

    bool ReadMetadata(std::ifstream& metafile) {
        std::string descriptor_string_junk;

        // "LAReference:"
        metafile >> descriptor_string_junk;

        auto convert = []<typename T>(const std::string &str) {
            return static_cast<T>(std::stoll(str));
        };

        {
            std::string add_point_options;
            metafile >> descriptor_string_junk;
            metafile >> add_point_options;
            m_AddPointOptions = convert.template operator()<AddPointOptions>(add_point_options);
        }

        {
            std::string use_at;
            metafile >> descriptor_string_junk;
            metafile >> use_at;
            m_UseAT = convert.template operator()<bool>(use_at);
        }

        {
            std::string la_stage_count;
            metafile >> descriptor_string_junk;
            metafile >> la_stage_count;
            m_LAStageCount = convert.template operator()<IterType>(la_stage_count);
        }

        {
            std::string is_valid;
            metafile >> descriptor_string_junk;
            metafile >> is_valid;
            m_IsValid = convert.template operator()<bool>(is_valid);
        }

        return m_AT.ReadMetadata(metafile);
    }

    template<class OtherT, class Other>
    void CopyLAReference(const LAReference<IterType, OtherT, Other, PExtras>& other) {
        assert(m_AddPointOptions == other.m_AddPointOptions);
        m_UseAT = other.m_UseAT;
        m_AT = other.m_AT;
        m_LAStageCount = other.m_LAStageCount;
        m_IsValid = other.m_IsValid;

        m_LAs.MutableResize(other.m_LAs.GetSize());
        for (int i = 0; i < other.m_LAs.GetSize(); i++) {
            m_LAs[i] = other.m_LAs[i];
        }
        m_LAStages.MutableResize(other.m_LAStages.GetSize());
        for (int i = 0; i < other.m_LAStages.GetSize(); i++) {
            m_LAStages[i] = other.m_LAStages[i];
        }
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

    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>>& GetLAs() {
        return m_LAs;
    }

    const GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>>& GetLAs() const {
        return m_LAs;
    }

    GrowableVector<LAStageInfo<IterType>>& GetLAStages() {
        return m_LAStages;
    }

    const GrowableVector<LAStageInfo<IterType>>& GetLAStages() const {
        return m_LAStages;
    }

private:
    AddPointOptions m_AddPointOptions;
    bool m_UseAT;
    ATInfo<IterType, Float, SubType> m_AT;
    IterType m_LAStageCount;
    bool m_IsValid;

    static constexpr int MaxLAStages = 1024;
    static constexpr int DEFAULT_SIZE = 10000;
    GrowableVector<LAInfoDeep<IterType, Float, SubType, PExtras>> m_LAs;
    GrowableVector<LAStageInfo<IterType>> m_LAStages;

    IterType LAsize();
    template<typename PerturbType>
    bool CreateLAFromOrbit(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateLAFromOrbitMT(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);
    template<typename PerturbType>
    bool CreateNewLAStage(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        IterType maxRefIteration);

public:
    template<typename PerturbType>
    void GenerateApproximationData(
        const PerturbationResults<IterType, PerturbType, PExtras>& PerturbationResults,
        Float radius,
        IterType maxRefIteration,
        bool UseSmallExponents);
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
