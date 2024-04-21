#include "PrecisionCalculator.h"

namespace PrecisionCalculator {

    uint64_t GetPrecision(
        const HighPrecision &minX,
        const HighPrecision &minY,
        const HighPrecision &maxX,
        const HighPrecision &maxY,
        bool RequiresReuse) {
        auto deltaX = abs(maxX - minX);
        auto deltaY = abs(maxY - minY);

        long temp_expX;
        double tempMantissaX;
        deltaX.frexp(tempMantissaX, temp_expX);
        long temp_expY;
        double tempMantissaY;
        deltaY.frexp(tempMantissaY, temp_expY);

        uint64_t larger = (uint64_t)std::max(abs(temp_expX), abs(temp_expY));

        if (RequiresReuse) {
            larger += AuthoritativeReuseExtraPrecisionInBits;
        } else {
            larger += AuthoritativeMinExtraPrecisionInBits;
        }
        return larger;
    }
} // namespace PrecisionCalculator
