#pragma once

#include <string>
#include <array>
#include <optional>

#include "HighPrecision.h"
#include "HDRFloat.h"

enum class TestTypeEnum {
    ViewInit,
    View0 = ViewInit,
    View5,
    View10,
    View11,
    ViewMax,
    ReferenceSave0,
    ReferenceSave5,
    ReferenceSave10,
    ReferenceSave13,
    ReferenceSave14,
    PerturbedPerturb12,
    End
};

enum class TestViewEnum {
    Disabled = -1,
    View0 = 0,
    View5 = 5,
    View10 = 10,
    View11 = 11,
    View12 = 12,
    View13 = 13,
    View14 = 14,
};

template<typename Key, typename Value, size_t Size, Value MissingVal>
struct ConstexprMap {
    std::array<std::pair<Key, Value>, static_cast<size_t>(Size)> data;

    using ConstExprMap = ConstexprMap<Key, Value, Size, MissingVal>;

    constexpr ConstexprMap() = default;
    constexpr ConstexprMap(const ConstexprMap &other) :
        data(other.data) {
    }
    constexpr ConstexprMap(ConstexprMap &&other) :
        data(std::move(other.data)) {
    }
    constexpr ConstexprMap &operator=(const ConstexprMap &other) = default;
    constexpr ConstexprMap &operator=(ConstexprMap &&other) = default;

    constexpr ConstexprMap(std::array<std::pair<Key, Value>, Size> &&data) : data(std::move(data)) {}
    constexpr ConstexprMap(std::array<std::pair<Key, Value>, Size> &data) : data(data) {}

    constexpr TestViewEnum Lookup(const Key &key) const {
        for (const auto &[k, v] : data) {
            if (k == key) {
                return v;
            }
        }

        return MissingVal;
    }
};

using TestViewMap = ConstexprMap<
    TestTypeEnum,
    TestViewEnum,
    static_cast<size_t>(TestTypeEnum::End),
    TestViewEnum::Disabled>;

// These should match the UI menu for sanity's sake
enum class RenderAlgorithmEnum {
    // CPU algorithms
    CpuHigh,
    Cpu64,
    CpuHDR32,
    CpuHDR64,

    Cpu64PerturbedBLA,
    Cpu32PerturbedBLAHDR,
    Cpu64PerturbedBLAHDR,

    Cpu32PerturbedBLAV2HDR,
    Cpu64PerturbedBLAV2HDR,
    Cpu32PerturbedRCBLAV2HDR,
    Cpu64PerturbedRCBLAV2HDR,

    // GPU - low zoom depth:
    Gpu1x32,
    Gpu2x32,
    Gpu4x32,
    Gpu1x64,
    Gpu2x64,
    Gpu4x64,
    GpuHDRx32,

    // GPU
    Gpu1x32PerturbedScaled,
    Gpu2x32PerturbedScaled,
    GpuHDRx32PerturbedScaled,

    Gpu1x64PerturbedBLA,
    GpuHDRx32PerturbedBLA,
    GpuHDRx64PerturbedBLA,

    Gpu1x32PerturbedLAv2,
    Gpu1x32PerturbedLAv2PO,
    Gpu1x32PerturbedLAv2LAO,
    Gpu1x32PerturbedRCLAv2,
    Gpu1x32PerturbedRCLAv2PO,
    Gpu1x32PerturbedRCLAv2LAO,

    Gpu2x32PerturbedLAv2,
    Gpu2x32PerturbedLAv2PO,
    Gpu2x32PerturbedLAv2LAO,
    Gpu2x32PerturbedRCLAv2,
    Gpu2x32PerturbedRCLAv2PO,
    Gpu2x32PerturbedRCLAv2LAO,

    Gpu1x64PerturbedLAv2,
    Gpu1x64PerturbedLAv2PO,
    Gpu1x64PerturbedLAv2LAO,
    Gpu1x64PerturbedRCLAv2,
    Gpu1x64PerturbedRCLAv2PO,
    Gpu1x64PerturbedRCLAv2LAO,

    GpuHDRx32PerturbedLAv2,
    GpuHDRx32PerturbedLAv2PO,
    GpuHDRx32PerturbedLAv2LAO,
    GpuHDRx32PerturbedRCLAv2,
    GpuHDRx32PerturbedRCLAv2PO,
    GpuHDRx32PerturbedRCLAv2LAO,

    GpuHDRx2x32PerturbedLAv2,
    GpuHDRx2x32PerturbedLAv2PO,
    GpuHDRx2x32PerturbedLAv2LAO,
    GpuHDRx2x32PerturbedRCLAv2,
    GpuHDRx2x32PerturbedRCLAv2PO,
    GpuHDRx2x32PerturbedRCLAv2LAO,

    GpuHDRx64PerturbedLAv2,
    GpuHDRx64PerturbedLAv2PO,
    GpuHDRx64PerturbedLAv2LAO,
    GpuHDRx64PerturbedRCLAv2,
    GpuHDRx64PerturbedRCLAv2PO,
    GpuHDRx64PerturbedRCLAv2LAO,

    AUTO,
    MAX
};

template<RenderAlgorithmEnum RenderEnum>
class RenderAlgorithmCompileTime {
public:
    static const RenderAlgorithmEnum Algorithm = RenderEnum;
    static const char *AlgorithmStr;
    static bool UseLocalColor;
    static bool RequiresCompression;

    static bool TestIncludeInBasic;
    static bool TestIncludeInView5;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHigh> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::CpuHigh;
    static constexpr char AlgorithmStr[] = "CpuHigh";
    static constexpr wchar_t AlgorithmStrW[] = L"CpuHigh";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = HighPrecision;
    using OriginatingType = MainType;
    using SubType = HighPrecision;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu64;
    static constexpr char AlgorithmStr[] = "Cpu64";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu64";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::CpuHDR32;
    static constexpr char AlgorithmStr[] = "CpuHDR32";
    static constexpr wchar_t AlgorithmStrW[] = L"CpuHDR32";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::CpuHDR64;
    static constexpr char AlgorithmStr[] = "CpuHDR64";
    static constexpr wchar_t AlgorithmStrW[] = L"CpuHDR64";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLA> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu64PerturbedBLA;
    static constexpr char AlgorithmStr[] = "Cpu64PerturbedBLA";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu64PerturbedBLA";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAHDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu32PerturbedBLAHDR;
    static constexpr char AlgorithmStr[] = "Cpu32PerturbedBLAHDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu32PerturbedBLAHDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAHDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu64PerturbedBLAHDR;
    static constexpr char AlgorithmStr[] = "Cpu64PerturbedBLAHDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu64PerturbedBLAHDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;


    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = MainType;
    using SubType = double;
};


template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR;
    static constexpr char AlgorithmStr[] = "Cpu32PerturbedBLAV2HDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu32PerturbedBLAV2HDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR;
    static constexpr char AlgorithmStr[] = "Cpu64PerturbedBLAV2HDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu64PerturbedBLAV2HDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
         {
             std::pair{TestTypeEnum::View0, TestViewEnum::View0},
             }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR;
    static constexpr char AlgorithmStr[] = "Cpu32PerturbedRCBLAV2HDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu32PerturbedRCBLAV2HDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR;
    static constexpr char AlgorithmStr[] = "Cpu64PerturbedRCBLAV2HDR";
    static constexpr wchar_t AlgorithmStrW[] = L"Cpu64PerturbedRCBLAV2HDR";
    static constexpr bool UseLocalColor = true;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32;
    static constexpr char AlgorithmStr[] = "Gpu1x32";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
        }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32;
    static constexpr char AlgorithmStr[] = "Gpu2x32";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
        }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x32> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu4x32;
    static constexpr char AlgorithmStr[] = "Gpu4x32";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu4x32";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = void;
    using OriginatingType = MainType;
    using SubType = void;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64;
    static constexpr char AlgorithmStr[] = "Gpu1x64";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
        }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x64> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x64;
    static constexpr char AlgorithmStr[] = "Gpu2x64";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x64";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = void;
    using OriginatingType = MainType;
    using SubType = void;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x64> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu4x64;
    static constexpr char AlgorithmStr[] = "Gpu4x64";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu4x64";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = void;
    using OriginatingType = MainType;
    using SubType = void;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32;
    static constexpr char AlgorithmStr[] = "GpuHDRx32";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
        }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedScaled> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedScaled;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedScaled";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedScaled";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedScaled> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedScaled;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedScaled";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedScaled";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedScaled> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedScaled;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedScaled";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedScaled";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedBLA> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedBLA;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedBLA";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedBLA";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedBLA> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedBLA;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedBLA";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedBLA";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedBLA> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedBLA;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedBLA";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedBLA";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu1x32PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x32PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = float;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu2x32PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu2x32PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = CudaDblflt<dblflt>;
    using OriginatingType = double;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            // std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12}, // should not crash but does
        }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            // std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12}, // should not crash but does
        }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "Gpu1x64PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"Gpu1x64PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            }
    };

    using MainType = double;
    using OriginatingType = MainType;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::ReferenceSave13, TestViewEnum::View13},
            std::pair{TestTypeEnum::ReferenceSave14, TestViewEnum::View14},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::ReferenceSave13, TestViewEnum::View13},
            std::pair{TestTypeEnum::ReferenceSave14, TestViewEnum::View14},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx32PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx32PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<float>;
    using OriginatingType = MainType;
    using SubType = float;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx2x32PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx2x32PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<CudaDblflt<dblflt>>;
    using OriginatingType = HDRFloat<double>;
    using SubType = CudaDblflt<dblflt>;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::ReferenceSave13, TestViewEnum::View13},
            std::pair{TestTypeEnum::ReferenceSave14, TestViewEnum::View14},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedRCLAv2";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedRCLAv2";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            std::pair{TestTypeEnum::ReferenceSave0, TestViewEnum::View0},
            std::pair{TestTypeEnum::ReferenceSave5, TestViewEnum::View5},
            std::pair{TestTypeEnum::ReferenceSave10, TestViewEnum::View10},
            std::pair{TestTypeEnum::ReferenceSave13, TestViewEnum::View13},
            std::pair{TestTypeEnum::ReferenceSave14, TestViewEnum::View14},
            std::pair{TestTypeEnum::PerturbedPerturb12, TestViewEnum::View12},
        }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedRCLAv2PO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedRCLAv2PO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO;
    static constexpr char AlgorithmStr[] = "GpuHDRx64PerturbedRCLAv2LAO";
    static constexpr wchar_t AlgorithmStrW[] = L"GpuHDRx64PerturbedRCLAv2LAO";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = true;
    static constexpr bool RequiresReferencePoints = true;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            std::pair{TestTypeEnum::View11, TestViewEnum::View11},
            }
    };

    using MainType = HDRFloat<double>;
    using OriginatingType = HDRFloat<double>;
    using SubType = double;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::AUTO;
    static constexpr char AlgorithmStr[] = "AutoSelect";
    static constexpr wchar_t AlgorithmStrW[] = L"AutoSelect";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            std::pair{TestTypeEnum::View0, TestViewEnum::View0},
            std::pair{TestTypeEnum::View5, TestViewEnum::View5},
            std::pair{TestTypeEnum::View10, TestViewEnum::View10},
            }
    };

    using MainType = void;
    using OriginatingType = MainType;
    using SubType = void;
};

template<>
class RenderAlgorithmCompileTime<RenderAlgorithmEnum::MAX> {
public:
    static constexpr RenderAlgorithmEnum Algorithm = RenderAlgorithmEnum::MAX;
    static constexpr char AlgorithmStr[] = "MAX";
    static constexpr wchar_t AlgorithmStrW[] = L"MAX";
    static constexpr bool UseLocalColor = false;
    static constexpr bool RequiresCompression = false;
    static constexpr bool RequiresReferencePoints = false;

    static constexpr TestViewMap TestInclude{
        {
            }
    };

    using MainType = void;
    using OriginatingType = MainType;
    using SubType = void;
};

// operator== for RenderAlgorithmCompileTime.  Use templates.
template<typename RenderAlgCompileTimeRhs, typename RenderAlgCompileTimeLhs>
constexpr bool operator== (RenderAlgCompileTimeRhs, RenderAlgCompileTimeLhs)
    requires std::is_base_of_v<RenderAlgorithmCompileTime<RenderAlgCompileTimeRhs::Algorithm>, RenderAlgCompileTimeRhs>
{
    return RenderAlgCompileTimeRhs::Algorithm == RenderAlgCompileTimeLhs::Algorithm;
}

class RenderAlgorithm {
public:
    constexpr RenderAlgorithm() :
        Algorithm{ RenderAlgorithmEnum::AUTO },
        AlgorithmStr{ RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>::AlgorithmStr },
        UseLocalColor{ RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>::UseLocalColor },
        RequiresCompression{ RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>::RequiresCompression },
        RequiresReferencePoints{ RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>::RequiresReferencePoints },
        TestInclude{ RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>::TestInclude } {
    }

    // Requires RenderAlgorithmCompileTime
    template<typename RenderAlgCompileTime>
    constexpr RenderAlgorithm(RenderAlgCompileTime Alg)
        requires std::is_base_of_v<RenderAlgorithmCompileTime<RenderAlgCompileTime::Algorithm>, RenderAlgCompileTime>
        : Algorithm{ Alg.Algorithm },
        AlgorithmStr{ Alg.AlgorithmStr },
        UseLocalColor{ Alg.UseLocalColor },
        RequiresCompression{ Alg.RequiresCompression },
        RequiresReferencePoints{ Alg.RequiresReferencePoints },
        TestInclude{ Alg.TestInclude } {
    }

    constexpr RenderAlgorithm(const RenderAlgorithm &other) :
        Algorithm{ other.Algorithm },
        AlgorithmStr{ other.AlgorithmStr },
        UseLocalColor{ other.UseLocalColor },
        RequiresCompression{ other.RequiresCompression },
        RequiresReferencePoints{ other.RequiresReferencePoints },
        TestInclude{ other.TestInclude } {
    }

    constexpr RenderAlgorithm(RenderAlgorithm &&other) :
        Algorithm{ other.Algorithm },
        AlgorithmStr{ other.AlgorithmStr },
        UseLocalColor{ other.UseLocalColor },
        RequiresCompression{ other.RequiresCompression },
        RequiresReferencePoints{ other.RequiresReferencePoints },
        TestInclude{ other.TestInclude } {
    }

    constexpr bool operator== (const RenderAlgorithm &other) const {
        return Algorithm == other.Algorithm;
    }

    constexpr bool operator== (RenderAlgorithmEnum other) const {
        return Algorithm == other;
    }

    RenderAlgorithm &operator= (const RenderAlgorithm &other);
    RenderAlgorithm &operator= (RenderAlgorithm &&other);

    RenderAlgorithmEnum Algorithm;
    const char *AlgorithmStr;
    bool UseLocalColor;
    bool RequiresCompression;
    bool RequiresReferencePoints;

    TestViewMap TestInclude;
};

static constexpr
std::array<RenderAlgorithm, static_cast<size_t>(RenderAlgorithmEnum::MAX) + 1> RenderAlgorithms{
    // CPU algorithms
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHigh>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLA>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAHDR>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAHDR>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR>{}},

    // GPU - low zoom depth:
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x32>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x64>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x64>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32>{}},

    // GPU
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedScaled>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedScaled>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedScaled>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedBLA>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedBLA>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedBLA>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>{}},

    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>{}},
    RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::MAX>{}}
};


using RenderAlgorithmsTupleT = std::tuple <
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHigh>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLA>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAHDR>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAHDR>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR>,

    // GPU - low zoom depth:
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x32>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x64>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu4x64>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32>,

    // GPU
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedScaled>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedScaled>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedScaled>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedBLA>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedBLA>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedBLA>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>,

    RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>,
    RenderAlgorithmCompileTime<RenderAlgorithmEnum::MAX>
> ;

constexpr static RenderAlgorithmsTupleT RenderAlgorithmsTuple;

template<typename F>
constexpr void IterateRenderAlgs(F &&function) {
    auto unfold = [&]<size_t... Ints>(std::index_sequence<Ints...>) {
        (std::forward<F>(function)(std::integral_constant<size_t, Ints>{}), ...);
    };

    constexpr auto size = std::tuple_size_v<RenderAlgorithmsTupleT>;
    unfold(std::make_index_sequence<size>());
}

template<typename F>
constexpr auto IterateRenderAlgsRet(F &&function) {
    auto unfold = [&]<size_t... Ints>(std::index_sequence<Ints...>) {
        return (std::forward<F>(function)(std::integral_constant<size_t, Ints>{}), ...);
    };

    constexpr auto size = std::tuple_size_v<RenderAlgorithmsTupleT>;
    return unfold(std::make_index_sequence<size>());
}

// Given a RenderAlgorithmEnum at runtime, return the associated entry in the 
// tuple RenderAlgorithmsTupleT.  Template parameter doesn't work for this.
// Use the IterateRenderAlgs function above to iterate over the tuple.
constexpr RenderAlgorithm GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum algorithm) {
    return RenderAlgorithms[static_cast<size_t>(algorithm)];
}