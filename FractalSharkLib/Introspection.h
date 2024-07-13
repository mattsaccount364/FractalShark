#pragma once

#include "HighPrecision.h"

namespace Introspection {
    //    requires (PExtras == PerturbExtras::SimpleCompression ||
    //PExtras == PerturbExtras::Bad ||
    //PExtras == PerturbExtras::Disable)
    template<PerturbExtras PExtras>
    struct TestPExtras {
        static constexpr bool value = (PExtras != PerturbExtras::MaxCompression);
    };

    template<typename IterType, typename Float, PerturbExtras PExtras>
    struct PerturbTypeParams {
        using IterType_ = IterType;
        using Float_ = Float;
        static constexpr auto PExtras_ = PExtras;
    };

    template<
        template <typename, typename, PerturbExtras> typename PerturbType,
        typename IterType,
        typename Float,
        PerturbExtras PExtras>
    constexpr auto Extract(const PerturbType<IterType, Float, PExtras> &) -> PerturbTypeParams<IterType, Float, PExtras>;

    template<typename PerturbType>
    constexpr auto Extract_PExtras = decltype(Extract(std::declval<PerturbType>()))::PExtras_;

    template<typename PerturbType>
    using Extract_Float = typename decltype(Extract(std::declval<PerturbType>()))::Float_;

    template<typename PerturbType, PerturbExtras PExtrasOther>
    static constexpr bool PerturbTypeHasPExtras() {
        return Extract_PExtras<PerturbType> == PExtrasOther;
    }

    template<typename PerturbType>
    static constexpr bool IsDblFlt() {
        return
            std::is_same<Extract_Float<PerturbType>, CudaDblflt<MattDblflt>>::value ||
            std::is_same<Extract_Float<PerturbType>, CudaDblflt<dblflt>>::value ||
            std::is_same<Extract_Float<PerturbType>, HDRFloat<CudaDblflt<MattDblflt>>>::value ||
            std::is_same<Extract_Float<PerturbType>, HDRFloat<CudaDblflt<dblflt>>>::value;
    }

    template<typename T>
    static constexpr bool IsTDblFlt() {
        return
            std::is_same<T, CudaDblflt<MattDblflt>>::value ||
            std::is_same<T, CudaDblflt<dblflt>>::value ||
            std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value ||
            std::is_same<T, HDRFloat<CudaDblflt<dblflt>>>::value;
    }
} // namespace Introspection
