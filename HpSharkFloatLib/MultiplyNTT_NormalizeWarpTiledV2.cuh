struct WarpNormalizeTriple {
    uint32_t carry_lo;
    uint32_t changedMask; // bit0=XX, bit1=YY, bit2=XY
};

template <class SharkFloatParams>
static __device__ SharkForceInlineReleaseOnly WarpNormalizeTriple
WarpNormalizeTile(unsigned fullMask,
                  const int32_t numActualDigitsPlusGuard, // total digits (N + guard), base-2^32
                  const int lane,                         // 0..31
                  const int tileIndex,                    // which 32-digit tile
                  const int iteration,
                  uint32_t &loXX,
                  uint32_t &loYY,
                  uint32_t &loXY,
                  const uint32_t tileCarryIn)
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    constexpr int warpSz = SharkFloatParams::GlobalThreadsPerBlock;
#else
    constexpr int warpSz = 32;
#endif

    //
    // If you want to fuss with this, also update WarpProcessTileCarry
    // Similar logic.  Here we have only 0/1 carries, no negatives.
    //

    const int base = tileIndex * warpSz;
    const auto basePlusLane = base + lane;
    const bool isLastLaneInTile = (lane == warpSz - 1);
    const bool isLastDigit = (basePlusLane == numActualDigitsPlusGuard - 1);

    uint32_t r1 = 0;

    uint32_t carryOut_xx;
    uint32_t carryOut_yy;
    uint32_t carryOut_xy;

    const uint32_t tileCarryIn_xx = tileCarryIn & 0x1;
    const uint32_t tileCarryIn_yy = (tileCarryIn >> 1) & 0x1;
    const uint32_t tileCarryIn_xy = (tileCarryIn >> 2) & 0x1;

    uint32_t changedMaskLocal = 0u;

#pragma unroll
    for (int step = 0; step < warpSz; ++step) {
        uint32_t carryIn = __shfl_up_sync(fullMask, r1, 1);
        uint32_t carryIn_xx = carryIn & 0x1;
        uint32_t carryIn_yy = (carryIn >> 1) & 0x1;
        uint32_t carryIn_xy = (carryIn >> 2) & 0x1;

        const bool laneIsZero = (lane == 0);
        const bool stepIsZero = (step == 0);

        if (laneIsZero) {
            // Lane 0: inject tileCarryIn at step 0, zero otherwise
            if (stepIsZero) {
                carryIn_xx = tileCarryIn_xx;
                carryIn_yy = tileCarryIn_yy;
                carryIn_xy = tileCarryIn_xy;
            } else {
                carryIn_xx = 0;
                carryIn_yy = 0;
                carryIn_xy = 0;
            }
        } else {
            // Other lanes: special only for iteration==0, step==0
            if (iteration == 0 && stepIsZero) {
                carryIn_xx = tileCarryIn_xx;
                carryIn_yy = tileCarryIn_yy;
                carryIn_xy = tileCarryIn_xy;
            }
            // else: keep the propagated carryIn_* from the shuffle
        }

        auto process_channel = [](uint32_t &lo,
                                  const uint32_t carryIn,
                                  uint32_t &carryOut,
                                  uint32_t &r1,
                                  const uint32_t shift) {
            const uint64_t s_lo = static_cast<uint64_t>(lo) + carryIn;
            carryOut = (s_lo >> 32);
            lo = static_cast<uint32_t>(s_lo & 0xffffffffu);

            // Note: we really only need this on the last lane
            // of the tile or last digit but we get slightly
            // better performance by doing it every time.

            r1 |= carryOut << shift;
        };

        if (!(isLastLaneInTile || isLastDigit)) {
            r1 = 0;
        }

        process_channel(loXX, carryIn_xx, carryOut_xx, r1, 0);
        process_channel(loYY, carryIn_yy, carryOut_yy, r1, 1);
        process_channel(loXY, carryIn_xy, carryOut_xy, r1, 2);

        // Track whether any non-zero outgoing carry needs further propagation.
        // We do NOT set bits on the very last produced digit (it will not propagate beyond).
        if (basePlusLane < numActualDigitsPlusGuard - 1) {
            changedMaskLocal |= carryOut_xx | carryOut_yy | carryOut_xy;
        }
    }

    return {r1, changedMaskLocal};
}
