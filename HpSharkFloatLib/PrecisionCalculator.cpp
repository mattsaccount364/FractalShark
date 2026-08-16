#include "CudaDblflt.h"
#include "HDRFloat.h"
#include "PrecisionCalculator.h"
#include "dblflt.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace PrecisionCalculator {

uint64_t
GetPrecision(const PointZoomBBConverter &converter, bool requiresReuse)
{
    return GetPrecision(converter.GetMinX(),
                        converter.GetMinY(),
                        converter.GetMaxX(),
                        converter.GetMaxY(),
                        requiresReuse);
}

uint64_t
GetPrecision(const HighPrecision &minX,
             const HighPrecision &minY,
             const HighPrecision &maxX,
             const HighPrecision &maxY,
             bool requiresReuse)
{
    const auto deltaX = abs(maxX - minX);
    const auto deltaY = abs(maxY - minY);

    return GetPrecision(deltaX, deltaY, requiresReuse);
}

uint64_t
GetPrecision(const HighPrecision &deltaX, const HighPrecision &deltaY, bool requiresReuse)
{
    HDRFloat<double> tempX{deltaX};
    HDRFloat<double> tempY{deltaY};

    return GetPrecision(tempX, tempY, requiresReuse);
}

template <typename T>
constexpr const char *
GetTypeName()
{
    return typeid(T).name();
}

template <typename T>
struct UnsupportedType {
    static void
    trigger()
    {
        static_assert(!std::is_same<T, T>::value, "Unsupported type for GetPrecision: ");
    }
};

template <typename T>
uint64_t
GetPrecision(const T &radiusX, const T &radiusY, bool requiresReuse)
{
    int tempExpX;
    int tempExpY;

    if constexpr (!std::is_same<T, HDRFloat<float>>::value &&
                  !std::is_same<T, HDRFloat<double>>::value && !std::is_same<T, float>::value &&
                  !std::is_same<T, double>::value &&
                  !std::is_same<T, CudaDblflt<MattDblflt>>::value &&
                  !std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
        UnsupportedType<T>::trigger();
    }

    if constexpr (std::is_same<T, HDRFloat<float>>::value ||
                  std::is_same<T, HDRFloat<double>>::value) {
        tempExpX = radiusX.getExp();
        tempExpY = radiusY.getExp();
    } else if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
        std::ignore = std::frexp(radiusX, &tempExpX);
        std::ignore = std::frexp(radiusY, &tempExpY);
    } else if constexpr (std::is_same<T, CudaDblflt<MattDblflt>>::value) {
        std::frexp(radiusX.d.head, &tempExpX);
        std::frexp(radiusY.d.head, &tempExpY);
    } else if constexpr (std::is_same<HDRFloat<CudaDblflt<MattDblflt>>, T>::value) {
        tempExpX = radiusX.getExp();
        tempExpY = radiusY.getExp();
    }

    uint64_t larger = static_cast<uint64_t>(std::max(std::abs(tempExpX), std::abs(tempExpY)));

    if (requiresReuse) {
        larger += AuthoritativeReuseExtraPrecisionInBits;
    } else {
        larger += AuthoritativeMinExtraPrecisionInBits;
    }
    return larger;
}

template uint64_t GetPrecision(const HDRFloat<double> &radiusX,
                               const HDRFloat<double> &radiusY,
                               bool requiresReuse);
template uint64_t GetPrecision(const HDRFloat<float> &radiusX,
                               const HDRFloat<float> &radiusY,
                               bool requiresReuse);
template uint64_t GetPrecision(const HDRFloat<CudaDblflt<MattDblflt>> &radiusX,
                               const HDRFloat<CudaDblflt<MattDblflt>> &radiusY,
                               bool requiresReuse);
template uint64_t GetPrecision(const double &radiusX, const double &radiusY, bool requiresReuse);
template uint64_t GetPrecision(const float &radiusX, const float &radiusY, bool requiresReuse);
template uint64_t GetPrecision(const CudaDblflt<MattDblflt> &radiusX,
                               const CudaDblflt<MattDblflt> &radiusY,
                               bool requiresReuse);
} // namespace PrecisionCalculator
