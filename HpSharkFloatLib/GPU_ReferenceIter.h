#pragma once

#include "HDRFloat.h"
#include "HighPrecision.h"
#include "dblflt.h"
#include <stdint.h>

struct Empty {};

struct BadField {
    BadField() : bad{false}, padding{} {}

    BadField(bool bad) : bad{bad}, padding{} {}

    [[msvc::no_unique_address]] uint32_t bad;
    [[msvc::no_unique_address]] uint32_t padding;
};

// TODO if we template this on IterType, update m_CompressionHelper
// (GPU + CPU versions) to avoid static_cast all over
struct CompressionIndexField {
    CompressionIndexField() : u{} {}

    CompressionIndexField(IterTypeFull initIndex)
    {
        u.f.CompressionIndex = initIndex;
        u.f.Rebase = 0;
    }

    CompressionIndexField(IterTypeFull initIndex, IterTypeFull rebase)
    {
        u.f.CompressionIndex = initIndex;
        u.f.Rebase = rebase;
    }

    union U {
        struct F {
            IterTypeFull CompressionIndex : 63;
            IterTypeFull Rebase : 1;
        } f;

        IterTypeFull Raw;
    } u;
};

template <PerturbExtras PExtras> class PerturbExtrasHack {
public:
    static constexpr PerturbExtras Val = PExtras;
};

#pragma pack(push, 8)
template <typename Type, PerturbExtras PExtras>
class /*alignas(8)*/ GPUReferenceIter
    : public std::conditional_t<PExtras == PerturbExtras::Bad,
                                BadField,
                                std::conditional_t<PExtras == PerturbExtras::SimpleCompression ||
                                                       PExtras == PerturbExtras::MaxCompression,
                                                   CompressionIndexField,
                                                   Empty>> {

public:
    using BaseClass =
        std::conditional_t<PExtras == PerturbExtras::Bad,
                           BadField,
                           std::conditional_t<PExtras == PerturbExtras::SimpleCompression ||
                                                  PExtras == PerturbExtras::MaxCompression,
                                              CompressionIndexField,
                                              Empty>>;

    static constexpr IterTypeFull BadCompressionIndex = 0xFFFF'FFFF'FFFF'FFFFull;

    struct Enabler {};

    GPUReferenceIter() : BaseClass{}, x{}, y{} {}

    template <typename U = PerturbExtrasHack<PExtras>,
              std::enable_if_t<U::Val == PerturbExtras::Disable, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y) : x{init_x}, y{init_y}
    {
    }

    template <typename U = PerturbExtrasHack<PExtras>,
              std::enable_if_t<U::Val == PerturbExtras::SimpleCompression ||
                                   U::Val == PerturbExtras::MaxCompression,
                               int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, IterTypeFull init_compression_index, IterTypeFull rebase)
        : CompressionIndexField(init_compression_index, rebase), x{init_x}, y{init_y}
    {
    }

    template <typename U = PerturbExtrasHack<PExtras>,
              std::enable_if_t<U::Val == PerturbExtras::SimpleCompression ||
                                   U::Val == PerturbExtras::MaxCompression,
                               int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, IterTypeFull init_compression_index)
        : CompressionIndexField(init_compression_index), x{init_x}, y{init_y}
    {
    }

    template <typename U = PerturbExtrasHack<PExtras>,
              std::enable_if_t<U::Val == PerturbExtras::Bad, int> = 0>
    GPUReferenceIter(Type init_x, Type init_y, bool init_bad) : BadField(init_bad), x{init_x}, y{init_y}
    {
    }

    GPUReferenceIter(const GPUReferenceIter &other) = default;
    GPUReferenceIter &operator=(const GPUReferenceIter &other) = default;

    // Example of how to pull the SubType out for HdrFloat, or keep the primitive float/double
    using SubType = typename SubTypeChooser<std::is_fundamental<Type>::value ||
                                                std::is_same<Type, MattDblflt>::value,
                                            Type>::type;

    static constexpr bool TypeCond =
        std::is_same<Type, HDRFloat<float, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<double, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<CudaDblflt<MattDblflt>, HDROrder::Left, int32_t>>::value ||
        std::is_same<Type, HDRFloat<CudaDblflt<dblflt>, HDROrder::Left, int32_t>>::value;
    std::conditional<TypeCond, Type, Type>::type x;
    std::conditional<TypeCond, HDRFloat<SubType, HDROrder::Right, int32_t>, Type>::type y;
};
#pragma pack(pop)
