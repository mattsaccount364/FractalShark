// --------------------------- helpers ---------------------------

template<int P, int Valid>
struct PosEnabled : std::bool_constant<(P < Valid)> {};

template<ConditionalAccess Cond, int Valid>
__device__ __forceinline__ void load_slot_scalar(
    int base_i, int k,
    const uint32_t *SharkRestrict aDigits_base,
    const uint32_t *SharkRestrict bDigits_base,
    int a_offset, int b_offset,
    const uint32_t *SharkRestrict x_diff_abs,
    const uint32_t *SharkRestrict y_diff_abs,
    uint32_t(&ax)[8], uint32_t(&bx)[8],
    uint32_t(&ay)[8], uint32_t(&by)[8],
    uint32_t(&cy)[8], uint32_t(&dy)[8]) {
#define DO_POS(P)                                                              \
    if constexpr (PosEnabled<P,Valid>::value) {                                \
        const int idx  = base_i + (P);                                         \
        const int idx2 = k - idx;                                              \
        if constexpr (Cond == ConditionalAccess::True) {                       \
            ax[P] = x_diff_abs[idx];   bx[P] = x_diff_abs[idx2];               \
            ay[P] = x_diff_abs[idx];   by[P] = y_diff_abs[idx2];               \
            cy[P] = y_diff_abs[idx];   dy[P] = y_diff_abs[idx2];               \
        } else {                                                                \
            ax[P] = aDigits_base[idx + a_offset];                              \
            bx[P] = aDigits_base[idx2 + a_offset];                             \
            ay[P] = aDigits_base[idx + a_offset];                              \
            by[P] = bDigits_base[idx2 + b_offset];                             \
            cy[P] = bDigits_base[idx + b_offset];                              \
            dy[P] = bDigits_base[idx2 + b_offset];                             \
        }                                                                       \
    }

    DO_POS(0) DO_POS(1) DO_POS(2) DO_POS(3)
        DO_POS(4) DO_POS(5) DO_POS(6) DO_POS(7)
#undef DO_POS
}

template<class SharkFloatParams, int Valid>
__device__ __forceinline__ void compute_slot_scalar(
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugGlobalCount<SharkFloatParams> *debugGlobalState,
    const uint32_t(&ax)[8], const uint32_t(&bx)[8],
    const uint32_t(&ay)[8], const uint32_t(&by)[8],
    const uint32_t(&cy)[8], const uint32_t(&dy)[8],
    uint64_t &xx_low, uint64_t &xx_high,
    uint64_t &xy_low, uint64_t &xy_high,
    uint64_t &yy_low, uint64_t &yy_high) {
#define LANE(P)                                                                \
    if constexpr (P < Valid) {                                                 \
        const uint64_t xx_a = (uint64_t)ax[P], xx_b = (uint64_t)bx[P];         \
        const uint64_t xy_a = (uint64_t)ay[P], xy_b = (uint64_t)by[P];         \
        const uint64_t yy_a = (uint64_t)cy[P], yy_b = (uint64_t)dy[P];         \
        uint64_t p;                                                            \
        p = xx_a * xx_b; xx_low += p; if (xx_low < p) xx_high += 1;            \
        p = xy_a * xy_b; xy_low += p; if (xy_low < p) xy_high += 1;            \
        p = yy_a * yy_b; yy_low += p; if (yy_low < p) yy_high += 1;            \
        DebugMultiplyIncrement<SharkFloatParams>(debugGlobalState, grid, block, 3); \
    }

    LANE(0) LANE(1) LANE(2) LANE(3)
        LANE(4) LANE(5) LANE(6) LANE(7)
#undef LANE
}

// Convenience macros to dispatch Valid at compile time (for tails)
#define LOAD_SLOT_VALID(VALID_, SLOT_AX,SLOT_BX,SLOT_AY,SLOT_BY,SLOT_CY,SLOT_DY, BASE_I) \
    load_slot_scalar<UseConditionalAccess, VALID_>(                                       \
        (BASE_I), k, aDigits_base, bDigits_base, a_offset, b_offset,                      \
        x_diff_abs, y_diff_abs,                                                           \
        SLOT_AX, SLOT_BX, SLOT_AY, SLOT_BY, SLOT_CY, SLOT_DY)

#define COMPUTE_SLOT_VALID(VALID_, SLOT_AX,SLOT_BX,SLOT_AY,SLOT_BY,SLOT_CY,SLOT_DY)       \
    compute_slot_scalar<SharkFloatParams, VALID_>(                                        \
        SLOT_AX, SLOT_BX, SLOT_AY, SLOT_BY, SLOT_CY, SLOT_DY,                             \
        xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high)

// ---------------- replacement: scalar 2-stage pipeline ----------------

template<
    class SharkFloatParams,
    int BatchSize,
    ConditionalAccess UseConditionalAccess,
    int RecursionDepth,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void ProcessConvolutionDirectLoad_Unaligned(
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugGlobalCount<SharkFloatParams> *debugGlobalState,
    int &i, const int i_end, const int k,
    const uint32_t *SharkRestrict aDigits_base,
    const uint32_t *SharkRestrict bDigits_base,
    const int a_offset, const int b_offset,
    uint64_t &xx_sum_low, uint64_t &xx_sum_high,
    uint64_t &xy_sum_low, uint64_t &xy_sum_high,
    uint64_t &yy_sum_low, uint64_t &yy_sum_high,
    const uint32_t *SharkRestrict x_diff_abs,
    const uint32_t *SharkRestrict y_diff_abs) {
    static_assert(BatchSize == 8, "This path expects BatchSize == 8");

    if (i > i_end) return;

    const int N_total = i_end - i + 1;
    int full_steps = N_total >> 3;   // number of 8-wide batches
    int tail = N_total & 7;    // remaining lanes (≤7)

    // Two slots of scalar registers (direct loads):
    uint32_t ax0[8], bx0[8], ay0[8], by0[8], cy0[8], dy0[8];
    uint32_t ax1[8], bx1[8], ay1[8], by1[8], cy1[8], dy1[8];

    // ================= warp-coherent steady (min) =================
    int full_steps_min = warpMinI32(full_steps);
    if (full_steps_min > 0) {
        // Seed: prefetch slot0 at base i
        LOAD_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0, i);

        int done = 0;
        while (done < full_steps_min) {
            // Prefetch next into other slot if we have another batch
            if (done + 1 < full_steps_min) {
                const int base_next = i + ((done + 1) << 3);
                LOAD_SLOT_VALID(8, ax1, bx1, ay1, by1, cy1, dy1, base_next);
            }
            // Compute current slot
            COMPUTE_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0);
            ++done; if (done >= full_steps_min) break;

            // Prefetch following into slot0 (reuse) if another batch remains
            if (done + 1 < full_steps_min) {
                const int base_next = i + ((done + 1) << 3);
                LOAD_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0, base_next);
            }
            // Compute other slot
            COMPUTE_SLOT_VALID(8, ax1, bx1, ay1, by1, cy1, dy1);
            ++done;
        }

        i += (full_steps_min << 3);
        full_steps -= full_steps_min;
    }

    // ================= remainder full steps (per-thread) =================
    if (full_steps > 0) {
        // Seed
        LOAD_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0, i);

        int done = 0;
        while (done < full_steps) {
            if (done + 1 < full_steps) {
                const int base_next = i + ((done + 1) << 3);
                LOAD_SLOT_VALID(8, ax1, bx1, ay1, by1, cy1, dy1, base_next);
            }
            COMPUTE_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0);
            ++done; if (done >= full_steps) break;

            if (done + 1 < full_steps) {
                const int base_next = i + ((done + 1) << 3);
                LOAD_SLOT_VALID(8, ax0, bx0, ay0, by0, cy0, dy0, base_next);
            }
            COMPUTE_SLOT_VALID(8, ax1, bx1, ay1, by1, cy1, dy1);
            ++done;
        }
        i += (full_steps << 3);
    }

    // ================= tail (≤7), single prefetch + compute =================
    if (tail > 0) {
        const int base_tail = i;
        switch (tail) {
        case 1:  LOAD_SLOT_VALID(1, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(1, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 2:  LOAD_SLOT_VALID(2, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(2, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 3:  LOAD_SLOT_VALID(3, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(3, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 4:  LOAD_SLOT_VALID(4, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(4, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 5:  LOAD_SLOT_VALID(5, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(5, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 6:  LOAD_SLOT_VALID(6, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(6, ax0, bx0, ay0, by0, cy0, dy0); break;
        case 7:  LOAD_SLOT_VALID(7, ax0, bx0, ay0, by0, cy0, dy0, base_tail);
            COMPUTE_SLOT_VALID(7, ax0, bx0, ay0, by0, cy0, dy0); break;
        default: break;
        }
        i += tail;
    }
}

#undef LOAD_SLOT_VALID
#undef COMPUTE_SLOT_VALID


template<
    class SharkFloatParams,
    int BatchSize,
    ConditionalAccess UseConditionalAccess,
    int RecursionDepth,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void ProcessConvolutionDirectLoad_Unaligned2 (
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugGlobalCount<SharkFloatParams> *debugGlobalState,
    int &i,
    const int i_start,
    const int i_end,
    const int k,
    const uint32_t *SharkRestrict aDigits_base,
    const uint32_t *SharkRestrict bDigits_base,
    const int a_offset,
    const int b_offset,
    uint64_t &xx_sum_low,
    uint64_t &xx_sum_high,
    uint64_t &xy_sum_low,
    uint64_t &xy_sum_high,
    uint64_t &yy_sum_low,
    uint64_t &yy_sum_high,
    const uint32_t *SharkRestrict x_diff_abs,
    const uint32_t *SharkRestrict y_diff_abs) {

    // Generic non-4 batching → just use your existing scalar-batched fallback
    for (int base_i = i_start; base_i <= i_end; ) {
        const int remaining = i_end - base_i + 1;
        const int bsz = remaining >= BatchSize ? BatchSize : remaining;

        if (bsz == BatchSize) {
            // scalar batched load + compute
            uint32_t ax[BatchSize], bx[BatchSize];
            uint32_t ay[BatchSize], by[BatchSize];
            uint32_t cy[BatchSize], dy[BatchSize];

#pragma unroll
            for (int j = 0; j < BatchSize; ++j) {
                const int idx = base_i + j;
                const int idx2 = k - idx;

                if constexpr (UseConditionalAccess == ConditionalAccess::True) {
                    ax[j] = x_diff_abs[idx];
                    bx[j] = x_diff_abs[idx2];
                    ay[j] = x_diff_abs[idx];
                    by[j] = y_diff_abs[idx2];
                    cy[j] = y_diff_abs[idx];
                    dy[j] = y_diff_abs[idx2];
                } else {
                    ax[j] = aDigits_base[idx + a_offset];
                    bx[j] = aDigits_base[idx2 + a_offset];
                    ay[j] = aDigits_base[idx + a_offset];
                    by[j] = bDigits_base[idx2 + b_offset];
                    cy[j] = bDigits_base[idx + b_offset];
                    dy[j] = bDigits_base[idx2 + b_offset];
                }
            }
#pragma unroll
            for (int j = 0; j < BatchSize; ++j) {
                const uint64_t xx_a = (uint64_t)ax[j], xx_b = (uint64_t)bx[j];
                const uint64_t xy_a = (uint64_t)ay[j], xy_b = (uint64_t)by[j];
                const uint64_t yy_a = (uint64_t)cy[j], yy_b = (uint64_t)dy[j];
                uint64_t p;
                
                p = xx_a * xx_b;
                xx_sum_low += p;
                if (xx_sum_low < p)
                    xx_sum_high += 1;
                
                p = xy_a * xy_b;
                xy_sum_low += p;
                if (xy_sum_low < p)
                    xy_sum_high += 1;

                p = yy_a * yy_b;
                yy_sum_low += p;
                if (yy_sum_low < p)
                    yy_sum_high += 1;

                DebugMultiplyIncrement<SharkFloatParams>(debugGlobalState, grid, block, 3);
            }

        } else {
            // scalar tail
            accumulate_scalar_span<SharkFloatParams, UseConditionalAccess>(
                grid,
                block,
                debugGlobalState,
                base_i,
                base_i + bsz - 1,
                k,
                aDigits_base,
                bDigits_base,
                a_offset,
                b_offset,
                x_diff_abs,
                y_diff_abs,
                xx_sum_low,
                xx_sum_high,
                xy_sum_low,
                xy_sum_high,
                yy_sum_low,
                yy_sum_high);
        }
        base_i += bsz;
    }
    return;
}