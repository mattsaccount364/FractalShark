template <class SharkFloatParams>
static __device__ inline void
Normalize_GridStride_3WayV1(cooperative_groups::grid_group &grid,
                            cooperative_groups::thread_block &block,
                            DebugGlobalCount<SharkFloatParams> *debugGlobalState,
                            DebugState<SharkFloatParams> *debugStates,
                            // outputs
                            HpSharkFloat<SharkFloatParams> &outXX,
                            HpSharkFloat<SharkFloatParams> &outYY,
                            HpSharkFloat<SharkFloatParams> &outXY,
                            // input exponents
                            const HpSharkFloat<SharkFloatParams> &inA,
                            const HpSharkFloat<SharkFloatParams> &inB,
                            // convolution sums (lo,hi per 32-bit position), NORMAL domain
                            const uint64_t *SharkRestrict final128XX,
                            const uint64_t *SharkRestrict final128YY,
                            const uint64_t *SharkRestrict final128XY,
                            const uint32_t Ddigits,
                            const int32_t addTwoXX,
                            const int32_t addTwoYY,
                            const int32_t addTwoXY,
                            // global workspaces (NOT shared memory)
                            uint64_t *SharkRestrict CarryPropagationBuffer2, // >= 6 + 6*lanes u64
                            uint64_t *SharkRestrict CarryPropagationBuffer,
                            uint64_t *SharkRestrict globalCarryCheck, // 1 u64
                            uint64_t *SharkRestrict resultXX,         // len >= Ddigits
                            uint64_t *SharkRestrict resultYY,         // len >= Ddigits
                            uint64_t *SharkRestrict resultXY)         // len >= Ddigits
{
    // We only ever produce digits in [0, Ddigits).
    const int T_all = static_cast<int>(grid.size());
    const auto tid = block.thread_index().x + block.group_index().x * blockDim.x;

    // Active participants are min(Ddigits, T_all)
    const int lanes = (Ddigits < T_all) ? Ddigits : T_all;
    const bool active = (tid < lanes);

    // Linear partition of [0, Ddigits) across 'lanes'
    const int start = active ? ((Ddigits * tid) / lanes) : 0;
    const int end = active ? ((Ddigits * (tid + 1)) / lanes) : 0;

    // --- 1) Initial pass over our slice (no tail beyond Ddigits) ---
    uint64_t carry_xx_lo = 0ull, carry_xx_hi = 0ull;
    uint64_t carry_yy_lo = 0ull, carry_yy_hi = 0ull;
    uint64_t carry_xy_lo = 0ull, carry_xy_hi = 0ull;

    if (active) {
        for (int idx = start; idx < end; ++idx) {
            const bool in = (idx < Ddigits);
            const size_t s = static_cast<size_t>(idx) * 2u;

            // XX
            {
                const uint64_t lo = in ? final128XX[s + 0] : 0ull;
                const uint64_t hi = in ? final128XX[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_xx_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_xx_hi + c0;

                resultXX[idx] = static_cast<uint32_t>(s_lo);
                carry_xx_lo = (s_lo >> 32) | (s_hi << 32);
                carry_xx_hi = (s_hi >> 32);
            }
            // YY
            {
                const uint64_t lo = in ? final128YY[s + 0] : 0ull;
                const uint64_t hi = in ? final128YY[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_yy_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_yy_hi + c0;

                resultYY[idx] = static_cast<uint32_t>(s_lo);
                carry_yy_lo = (s_lo >> 32) | (s_hi << 32);
                carry_yy_hi = (s_hi >> 32);
            }
            // XY
            {
                const uint64_t lo = in ? final128XY[s + 0] : 0ull;
                const uint64_t hi = in ? final128XY[s + 1] : 0ull;

                const uint64_t s_lo = lo + carry_xy_lo;
                const uint64_t c0 = (s_lo < lo) ? 1ull : 0ull;
                const uint64_t s_hi = hi + carry_xy_hi + c0;

                resultXY[idx] = static_cast<uint32_t>(s_lo);
                carry_xy_lo = (s_lo >> 32) | (s_hi << 32);
                carry_xy_hi = (s_hi >> 32);
            }
        }

        // Publish residual to the next lane except for the **last lane**.
        // By design (matches simplified scalar), we DROP any residual that would flow beyond Ddigits.
        if (tid == lanes - 1) {
            carry_xx_lo = carry_xx_hi = 0ull;
            carry_xy_lo = carry_xy_hi = 0ull;
            carry_yy_lo = carry_yy_hi = 0ull;
        }

        const int base = 6 + tid * 6;
        CarryPropagationBuffer2[base + 0] = carry_xx_lo;
        CarryPropagationBuffer2[base + 1] = carry_xx_hi;
        CarryPropagationBuffer2[base + 2] = carry_xy_lo;
        CarryPropagationBuffer2[base + 3] = carry_xy_hi;
        CarryPropagationBuffer2[base + 4] = carry_yy_lo;
        CarryPropagationBuffer2[base + 5] = carry_yy_hi;
    }

    if constexpr (HpShark::DebugGlobalState) {
        DebugNormalizeIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
    }

    grid.sync();

    // --- 2) Iterative carry propagation within [0, Ddigits) (drop at right edge) ---
    while (true) {
        if (tid == 0)
            *globalCarryCheck = 0ull;

        uint64_t in_xx_lo = 0ull, in_xx_hi = 0ull;
        uint64_t in_yy_lo = 0ull, in_yy_hi = 0ull;
        uint64_t in_xy_lo = 0ull, in_xy_hi = 0ull;

        if (active) {
            if (tid > 0) {
                const int prev = 6 + (tid - 1) * 6;
                in_xx_lo = CarryPropagationBuffer2[prev + 0];
                in_xx_hi = CarryPropagationBuffer2[prev + 1];
                in_xy_lo = CarryPropagationBuffer2[prev + 2];
                in_xy_hi = CarryPropagationBuffer2[prev + 3];
                in_yy_lo = CarryPropagationBuffer2[prev + 4];
                in_yy_hi = CarryPropagationBuffer2[prev + 5];
            }
        }

        grid.sync();

        if (active) {
            for (int idx = start; idx < end; ++idx) {
                // XX
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_xx_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultXX[idx]) + add32;
                    resultXX[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_xx_lo >> 32) | (in_xx_hi << 32);
                    const uint64_t next_hi = (in_xx_hi >> 32);
                    in_xx_lo = next_lo + c32;
                    in_xx_hi = next_hi;
                }
                // YY
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_yy_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultYY[idx]) + add32;
                    resultYY[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_yy_lo >> 32) | (in_yy_hi << 32);
                    const uint64_t next_hi = (in_yy_hi >> 32);
                    in_yy_lo = next_lo + c32;
                    in_yy_hi = next_hi;
                }
                // XY
                {
                    const uint64_t add32 = static_cast<uint32_t>(in_xy_lo);
                    const uint64_t sum = static_cast<uint32_t>(resultXY[idx]) + add32;
                    resultXY[idx] = static_cast<uint32_t>(sum);
                    const uint64_t c32 = (sum >> 32);
                    const uint64_t next_lo = (in_xy_lo >> 32) | (in_xy_hi << 32);
                    const uint64_t next_hi = (in_xy_hi >> 32);
                    in_xy_lo = next_lo + c32;
                    in_xy_hi = next_hi;
                }
            }

            // Drop residual at the right boundary (last tid), publish otherwise.
            if (tid == lanes - 1) {
                in_xx_lo = in_xx_hi = 0ull;
                in_xy_lo = in_xy_hi = 0ull;
                in_yy_lo = in_yy_hi = 0ull;
            }

            const int base = 6 + tid * 6;
            CarryPropagationBuffer2[base + 0] = in_xx_lo;
            CarryPropagationBuffer2[base + 1] = in_xx_hi;
            CarryPropagationBuffer2[base + 2] = in_xy_lo;
            CarryPropagationBuffer2[base + 3] = in_xy_hi;
            CarryPropagationBuffer2[base + 4] = in_yy_lo;
            CarryPropagationBuffer2[base + 5] = in_yy_hi;

            // Only signal continuation if something remains to hand to the *next* tid.
            if (in_xx_lo | in_xx_hi | in_xy_lo | in_xy_hi | in_yy_lo | in_yy_hi)
                atomicAdd(globalCarryCheck, 1ull);
        }

        grid.sync();

        if constexpr (HpShark::DebugGlobalState) {
            DebugNormalizeIncrement<SharkFloatParams>(debugGlobalState, grid, block, 1);
        }

        // Atomic read to avoid any visibility doubt
        if (atomicAdd(globalCarryCheck, 0ull) == 0ull)
            break;
        grid.sync();
    }

    FinalizeNormalize<SharkFloatParams>(grid,
                                        block,
                                        debugStates,
                                        outXX,
                                        outYY,
                                        outXY,
                                        inA,
                                        inB,
                                        Ddigits,
                                        addTwoXX,
                                        addTwoYY,
                                        addTwoXY,
                                        CarryPropagationBuffer2,
                                        CarryPropagationBuffer,
                                        resultXX,
                                        resultYY,
                                        resultXY);
}
