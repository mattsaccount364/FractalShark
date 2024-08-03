#include "stdafx.h"
#include "PrecisionCalculator.h"
#include "HDRFloat.h"
#include "GPU_Types.h"

namespace PrecisionCalculator {

    uint64_t GetPrecision(
        const PointZoomBBConverter &converter,
        bool requiresReuse) {

        return GetPrecision(
            converter.GetMinX(),
            converter.GetMinY(),
            converter.GetMaxX(),
            converter.GetMaxY(),
            requiresReuse);
    }

    uint64_t GetPrecision(
        const HighPrecision &minX,
        const HighPrecision &minY,
        const HighPrecision &maxX,
        const HighPrecision &maxY,
        bool RequiresReuse) {

        const auto deltaX = abs(maxX - minX);
        const auto deltaY = abs(maxY - minY);

        return GetPrecision(deltaX, deltaY, RequiresReuse);
    }

    uint64_t GetPrecision(
        const HighPrecision &deltaX,
        const HighPrecision &deltaY,
        bool RequiresReuse) {

        HDRFloat<double> tempX{deltaX};
        HDRFloat<double> tempY{deltaY};

        return GetPrecision(tempX, tempY, RequiresReuse);
    }

    template <typename T>
    constexpr const char *GetTypeName() {
        return typeid(T).name();
    }

    template <typename T>
    struct UnsupportedType {
        static void trigger() {
            static_assert(!std::is_same<T, T>::value, "Unsupported type for GetPrecision: ");
        }
    };

    template<typename T>
    uint64_t GetPrecision(
        const T &radiusX,
        const T &radiusY,
        bool RequiresReuse) {

        int temp_expX;
        int temp_expY;

        // static_assert if the type is not one of those following
        if constexpr (
            !std::is_same<T, HDRFloat<float>>::value &&
            !std::is_same<T, HDRFloat<double>>::value &&
            !std::is_same<T, float>::value &&
            !std::is_same<T, double>::value &&
            !std::is_same<T, CudaDblflt<MattDblflt>>::value &&
            !std::is_same<T, HDRFloat<CudaDblflt<MattDblflt>>>::value) {
            UnsupportedType<T>::trigger();
        }

        if constexpr (
            std::is_same<T, HDRFloat<float>>::value ||
            std::is_same<T, HDRFloat<double>>::value) {
            
            temp_expX = radiusX.getExp();
            temp_expY = radiusY.getExp();
        } else if constexpr (
            std::is_same<T, float>::value ||
            std::is_same<T, double>::value) {

            // Get the base-2 exponent of the floating point number
            // using frexp

            std::ignore = std::frexp(radiusX, &temp_expX);
            std::ignore = std::frexp(radiusY, &temp_expY);
        } else if constexpr (
            std::is_same<T, CudaDblflt<MattDblflt>>::value) {

            std::frexp(radiusX.d.head, &temp_expX);
            std::frexp(radiusY.d.head, &temp_expY);
        } else if constexpr(std::is_same<HDRFloat<CudaDblflt<MattDblflt>>, T>::value) {
            temp_expX = radiusX.getExp();
            temp_expY = radiusY.getExp();
        }

        uint64_t larger = (uint64_t)std::max(abs(temp_expX), abs(temp_expY));

        if (RequiresReuse) {
            larger += AuthoritativeReuseExtraPrecisionInBits;
        } else {
            larger += AuthoritativeMinExtraPrecisionInBits;
        }
        return larger;
    }

    template uint64_t GetPrecision(const HDRFloat<double> &radiusX, const HDRFloat<double> &radiusY, bool RequiresReuse);
    template uint64_t GetPrecision(const HDRFloat<float> &radiusX, const HDRFloat<float> &radiusY, bool RequiresReuse);
    template uint64_t GetPrecision(const HDRFloat<CudaDblflt<MattDblflt>> &radiusX, const HDRFloat<CudaDblflt<MattDblflt>> &radiusY, bool RequiresReuse);
    template uint64_t GetPrecision(const double &radiusX, const double &radiusY, bool RequiresReuse);
    template uint64_t GetPrecision(const float &radiusX, const float &radiusY, bool RequiresReuse);
    template uint64_t GetPrecision(const CudaDblflt<MattDblflt> &radiusX, const CudaDblflt<MattDblflt> &radiusY, bool RequiresReuse);
} // namespace PrecisionCalculator
