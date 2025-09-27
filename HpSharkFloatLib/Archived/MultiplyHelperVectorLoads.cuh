// ===== aligned v4 load + triple helper (unchanged) =====
__device__ __forceinline__ void ld4_aligned_ca(
    const uint32_t *__restrict__ p,
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];\n\t"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "l"(p));
}
__device__ __forceinline__ void load_raw_triple_from_aligned(
    const uint32_t *__restrict__ p_aligned,
    uint32_t(&L0)[4], uint32_t(&L1)[4], uint32_t(&L2)[4]) {
    ld4_aligned_ca(p_aligned + 0, L0[0], L0[1], L0[2], L0[3]);
    ld4_aligned_ca(p_aligned + 4, L1[0], L1[1], L1[2], L1[3]);
    ld4_aligned_ca(p_aligned + 8, L2[0], L2[1], L2[2], L2[3]);
}

// ===== mirror-only normalization (forward removed) =====
__device__ __forceinline__ void normalize_mirror_inplace(
    uint32_t(&L0)[4], uint32_t(&L1)[4], const uint32_t(&L2)[4], int M) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    switch (M & 3) {  // rotate forward to r[0..7]
    case 0: r0 = L0[0]; r1 = L0[1]; r2 = L0[2]; r3 = L0[3]; r4 = L1[0]; r5 = L1[1]; r6 = L1[2]; r7 = L1[3]; break;
    case 1: r0 = L0[1]; r1 = L0[2]; r2 = L0[3]; r3 = L1[0]; r4 = L1[1]; r5 = L1[2]; r6 = L1[3]; r7 = L2[0]; break;
    case 2: r0 = L0[2]; r1 = L0[3]; r2 = L1[0]; r3 = L1[1]; r4 = L1[2]; r5 = L1[3]; r6 = L2[0]; r7 = L2[1]; break;
    default: r0 = L0[3]; r1 = L1[0]; r2 = L1[1]; r3 = L1[2]; r4 = L1[3]; r5 = L2[0]; r6 = L2[1]; r7 = L2[2]; break;
    }
    // reverse to L0|L1
    L0[0] = r7; L0[1] = r6; L0[2] = r5; L0[3] = r4;
    L1[0] = r3; L1[1] = r2; L1[2] = r1; L1[3] = r0;
}

// ===== slot pick (unchanged) =====
template<int S> __device__ __forceinline__
auto pick(const uint32_t(&A0)[4], const uint32_t(&A1)[4]) -> const uint32_t(&)[4] {
    if constexpr (S == 0) return A0; else return A1;
}

// ===== compute per-slot with constant indexing (unchanged) =====
template<int S, int Valid>
__device__ __forceinline__ void compute_slot_valid_const(
    const uint32_t(&AF0_L0)[4], const uint32_t(&AF0_L1)[4], const uint32_t(&AF0_L2)[4],
    const uint32_t(&BF0_L0)[4], const uint32_t(&BF0_L1)[4], const uint32_t(&BF0_L2)[4],
    const uint32_t(&AM0_L0)[4], const uint32_t(&AM0_L1)[4], const uint32_t(&AM0_L2)[4],
    const uint32_t(&BM0_L0)[4], const uint32_t(&BM0_L1)[4], const uint32_t(&BM0_L2)[4],
    const uint32_t(&AF1_L0)[4], const uint32_t(&AF1_L1)[4], const uint32_t(&AF1_L2)[4],
    const uint32_t(&BF1_L0)[4], const uint32_t(&BF1_L1)[4], const uint32_t(&BF1_L2)[4],
    const uint32_t(&AM1_L0)[4], const uint32_t(&AM1_L1)[4], const uint32_t(&AM1_L2)[4],
    const uint32_t(&BM1_L0)[4], const uint32_t(&BM1_L1)[4], const uint32_t(&BM1_L2)[4],
    uint64_t &xx_low, uint64_t &xx_high,
    uint64_t &xy_low, uint64_t &xy_high,
    uint64_t &yy_low, uint64_t &yy_high) {
    const uint32_t(&AF_L0)[4] = pick<S>(AF0_L0, AF1_L0);
    const uint32_t(&AF_L1)[4] = pick<S>(AF0_L1, AF1_L1);
    const uint32_t(&BF_L0)[4] = pick<S>(BF0_L0, BF1_L0);
    const uint32_t(&BF_L1)[4] = pick<S>(BF0_L1, BF1_L1);
    const uint32_t(&AM_L0)[4] = pick<S>(AM0_L0, AM1_L0);
    const uint32_t(&AM_L1)[4] = pick<S>(AM0_L1, AM1_L1);
    const uint32_t(&BM_L0)[4] = pick<S>(BM0_L0, BM1_L0);
    const uint32_t(&BM_L1)[4] = pick<S>(BM0_L1, BM1_L1);

#define LANE(P) do {                                                        \
        if constexpr (P < Valid) {                                              \
            const uint64_t af = (P < 4 ? AF_L0[P] : AF_L1[P-4]);                \
            const uint64_t bf = (P < 4 ? BF_L0[P] : BF_L1[P-4]);                \
            const uint64_t am = (P < 4 ? AM_L0[P] : AM_L1[P-4]);                \
            const uint64_t bm = (P < 4 ? BM_L0[P] : BM_L1[P-4]);                \
            uint64_t p;                                                         \
            p = af * am; xx_low += p; if (xx_low < p) xx_high += 1;             \
            p = af * bm; xy_low += p; if (xy_low < p) xy_high += 1;             \
            p = bf * bm; yy_low += p; if (yy_low < p) yy_high += 1;             \
        }                                                                       \
    } while(0)

    LANE(0); LANE(1); LANE(2); LANE(3);
    LANE(4); LANE(5); LANE(6); LANE(7);
#undef LANE
}

// ===== mirror-only normalization for a slot =====
template<int S>
__device__ __forceinline__ void normalize_slot_mirror_only(
    // slot-0 RAW
    uint32_t(&AF0_L0)[4], uint32_t(&AF0_L1)[4], uint32_t(&AF0_L2)[4],
    uint32_t(&BF0_L0)[4], uint32_t(&BF0_L1)[4], uint32_t(&BF0_L2)[4],
    uint32_t(&AM0_L0)[4], uint32_t(&AM0_L1)[4], uint32_t(&AM0_L2)[4],
    uint32_t(&BM0_L0)[4], uint32_t(&BM0_L1)[4], uint32_t(&BM0_L2)[4],
    // slot-1 RAW
    uint32_t(&AF1_L0)[4], uint32_t(&AF1_L1)[4], uint32_t(&AF1_L2)[4],
    uint32_t(&BF1_L0)[4], uint32_t(&BF1_L1)[4], uint32_t(&BF1_L2)[4],
    uint32_t(&AM1_L0)[4], uint32_t(&AM1_L1)[4], uint32_t(&AM1_L2)[4],
    uint32_t(&BM1_L0)[4], uint32_t(&BM1_L1)[4], uint32_t(&BM1_L2)[4],
    // misalignments: forward are guaranteed 0; only mirrors used
    int /*M_AF*/, int /*M_BF*/, int M_AM, int M_BM) {
    if constexpr (S == 0) {
        // forward already aligned: no-op for AF0/BF0
        normalize_mirror_inplace(AM0_L0, AM0_L1, AM0_L2, M_AM);
        normalize_mirror_inplace(BM0_L0, BM0_L1, BM0_L2, M_BM);
    } else {
        // forward already aligned: no-op for AF1/BF1
        normalize_mirror_inplace(AM1_L0, AM1_L1, AM1_L2, M_AM);
        normalize_mirror_inplace(BM1_L0, BM1_L1, BM1_L2, M_BM);
    }
}

// ===== main: BatchSize==8, 2-stage, forward-aligned + mirror-only normalize =====
template<
    class SharkFloatParams,
    int BatchSize,
    ConditionalAccess UseConditionalAccess,
    int RecursionDepth,
    int ExecutionBlockBase,
    int ExecutionNumBlocks>
__device__ SharkForceInlineReleaseOnly static void ProcessConvolutionDirectLoad_BS8_FwdAligned(
    cg::grid_group &grid,
    cg::thread_block &block,
    DebugMultiplyCount<SharkFloatParams> *debugMultiplyCounts,
    int &i, const int i_end, const int k,
    const uint32_t *aDigits_base, const uint32_t *bDigits_base,
    const int a_offset, const int b_offset,
    uint64_t &xx_sum_low, uint64_t &xx_sum_high,
    uint64_t &xy_sum_low, uint64_t &xy_sum_high,
    uint64_t &yy_sum_low, uint64_t &yy_sum_high,
    const uint32_t *x_diff_abs, const uint32_t *y_diff_abs) {
    static_assert(BatchSize == 8, "BatchSize must be 8");

    // ---- base selection ----
    const uint32_t *Abase, *Bbase; int a_off_eff = 0, b_off_eff = 0;
    if constexpr (UseConditionalAccess == ConditionalAccess::True) {
        Abase = x_diff_abs;  Bbase = y_diff_abs;
    } else {
        Abase = aDigits_base; Bbase = bDigits_base;
        a_off_eff = a_offset; b_off_eff = b_offset;
    }

    // ================= scalar prologue to force M_AF==M_BF==0 =================
    // consume up to 3 elements until both forward streams 16B-aligned
    if (i <= i_end) {
        int shift = 0;
        while (i + shift <= i_end) {
            const int mA = (a_off_eff + i + shift) & 3;
            const int mB = (b_off_eff + i + shift) & 3;
            if (mA == 0 && mB == 0) break;
            ++shift; if (shift == 4) break;
        }
        if (shift > 0) {
            const int pro_end = min(i_end, i + shift - 1);
            accumulate_scalar_span<SharkFloatParams, UseConditionalAccess>(
                grid,
                block,
                debugMultiplyCounts,
                i, pro_end, k,
                aDigits_base, bDigits_base, a_offset, b_offset,
                x_diff_abs, y_diff_abs,
                xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high);
            i = pro_end + 1;
            if (i > i_end) return;
        }
    }

    const int i0 = i;
    const int N = i_end - i0 + 1;
    if (N <= 0) return;

    // steps & tail
    int full_steps = N >> 3;
    const int tail = N & 7;

    // ---- Forward misalignments are now zero by construction ----
    const int M_AF = 0;
    const int M_BF = 0;

    // ---- Mirror misalignments (invariant across 8-step batches) ----
    const int n0 = k - i0 - 7;
    const int M_AM = (a_off_eff + n0) & 3;
    const int M_BM = (b_off_eff + n0) & 3;

    // aligned bases for first batch
    const uint32_t *AF_base = Abase + ((a_off_eff + i0) & ~3);  // aligned, M_AF==0
    const uint32_t *BF_base = Bbase + ((b_off_eff + i0) & ~3);  // aligned, M_BF==0
    const uint32_t *AM_base = Abase + ((a_off_eff + n0) & ~3);  // aligned, needs mirror normalize
    const uint32_t *BM_base = Bbase + ((b_off_eff + n0) & ~3);  // aligned, needs mirror normalize

    // two slots of RAW vectors
    uint32_t AF0_L0[4], AF0_L1[4], AF0_L2[4];
    uint32_t BF0_L0[4], BF0_L1[4], BF0_L2[4];
    uint32_t AM0_L0[4], AM0_L1[4], AM0_L2[4];
    uint32_t BM0_L0[4], BM0_L1[4], BM0_L2[4];

    uint32_t AF1_L0[4], AF1_L1[4], AF1_L2[4];
    uint32_t BF1_L0[4], BF1_L1[4], BF1_L2[4];
    uint32_t AM1_L0[4], AM1_L1[4], AM1_L2[4];
    uint32_t BM1_L0[4], BM1_L1[4], BM1_L2[4];

    auto load_AF_into_slot = [&](auto S, const uint32_t *p) {
        if constexpr (decltype(S)::value == 0) load_raw_triple_from_aligned(p, AF0_L0, AF0_L1, AF0_L2);
        else                                   load_raw_triple_from_aligned(p, AF1_L0, AF1_L1, AF1_L2);
        };
    auto load_BF_into_slot = [&](auto S, const uint32_t *p) {
        if constexpr (decltype(S)::value == 0) load_raw_triple_from_aligned(p, BF0_L0, BF0_L1, BF0_L2);
        else                                   load_raw_triple_from_aligned(p, BF1_L0, BF1_L1, BF1_L2);
        };
    auto load_AM_into_slot = [&](auto S, const uint32_t *p) {
        if constexpr (decltype(S)::value == 0) load_raw_triple_from_aligned(p, AM0_L0, AM0_L1, AM0_L2);
        else                                   load_raw_triple_from_aligned(p, AM1_L0, AM1_L1, AM1_L2);
        };
    auto load_BM_into_slot = [&](auto S, const uint32_t *p) {
        if constexpr (decltype(S)::value == 0) load_raw_triple_from_aligned(p, BM0_L0, BM0_L1, BM0_L2);
        else                                   load_raw_triple_from_aligned(p, BM1_L0, BM1_L1, BM1_L2);
        };

    // rolling pointers
    const uint32_t *pAF = AF_base;
    const uint32_t *pBF = BF_base;
    const uint32_t *pAM = AM_base;
    const uint32_t *pBM = BM_base;

    // ================= steady: warp-min(full_steps), Valid=8, mirror-only normalize =================
    int full_steps_min = warpMinI32(full_steps);
    if (full_steps_min > 0) {
        // prefetch batch 0 into slot 0 + mirror normalize only
        load_AF_into_slot(std::integral_constant<int, 0>{}, pAF);
        load_BF_into_slot(std::integral_constant<int, 0>{}, pBF);
        load_AM_into_slot(std::integral_constant<int, 0>{}, pAM);
        load_BM_into_slot(std::integral_constant<int, 0>{}, pBM);
        normalize_slot_mirror_only<0>(
            AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2, AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
            AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2, AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
            M_AF, M_BF, M_AM, M_BM);

        pAF += 8; pBF += 8; pAM -= 8; pBM -= 8;

#define PREFETCH_NORM_MIRROR(SLOT) do {                                  \
            load_AF_into_slot(std::integral_constant<int,SLOT>{}, pAF);          \
            load_BF_into_slot(std::integral_constant<int,SLOT>{}, pBF);          \
            load_AM_into_slot(std::integral_constant<int,SLOT>{}, pAM);          \
            load_BM_into_slot(std::integral_constant<int,SLOT>{}, pBM);          \
            normalize_slot_mirror_only<SLOT>(                                     \
                AF0_L0,AF0_L1,AF0_L2, BF0_L0,BF0_L1,BF0_L2, AM0_L0,AM0_L1,AM0_L2, BM0_L0,BM0_L1,BM0_L2, \
                AF1_L0,AF1_L1,AF1_L2, BF1_L0,BF1_L1,BF1_L2, AM1_L0,AM1_L1,AM1_L2, BM1_L0,BM1_L1,BM1_L2, \
                M_AF,M_BF,M_AM,M_BM);                                            \
            pAF += 8; pBF += 8; pAM -= 8; pBM -= 8;                              \
        } while(0)

#define COMPUTE_SLOT8(SLOT)                                              \
            compute_slot_valid_const<SLOT, 8>(                                    \
                AF0_L0,AF0_L1,AF0_L2, BF0_L0,BF0_L1,BF0_L2,                       \
                AM0_L0,AM0_L1,AM0_L2, BM0_L0,BM0_L1,BM0_L2,                       \
                AF1_L0,AF1_L1,AF1_L2, BF1_L0,BF1_L1,BF1_L2,                       \
                AM1_L0,AM1_L1,AM1_L2, BM1_L0,BM1_L1,BM1_L2,                       \
                xx_sum_low,xx_sum_high, xy_sum_low,xy_sum_high, yy_sum_low,yy_sum_high)

        int done = 0;
        while (done < full_steps_min) {
            if (done + 1 < full_steps_min) PREFETCH_NORM_MIRROR(1);
            COMPUTE_SLOT8(0);
            ++done; if (done >= full_steps_min) break;

            if (done + 1 < full_steps_min) PREFETCH_NORM_MIRROR(0);
            COMPUTE_SLOT8(1);
            ++done;
        }
#undef COMPUTE_SLOT8
#undef PREFETCH_NORM_MIRROR

        i += (full_steps_min << 3);
        full_steps -= full_steps_min;
    }

    // ================= remainder full steps (per-thread), mirror-only normalize =================
    if (full_steps > 0) {
        load_AF_into_slot(std::integral_constant<int, 0>{}, pAF);
        load_BF_into_slot(std::integral_constant<int, 0>{}, pBF);
        load_AM_into_slot(std::integral_constant<int, 0>{}, pAM);
        load_BM_into_slot(std::integral_constant<int, 0>{}, pBM);
        normalize_slot_mirror_only<0>(
            AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2, AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
            AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2, AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
            M_AF, M_BF, M_AM, M_BM);
        pAF += 8; pBF += 8; pAM -= 8; pBM -= 8;

        int done = 0;
        while (done < full_steps) {
            if (done + 1 < full_steps) {
                load_AF_into_slot(std::integral_constant<int, 1>{}, pAF);
                load_BF_into_slot(std::integral_constant<int, 1>{}, pBF);
                load_AM_into_slot(std::integral_constant<int, 1>{}, pAM);
                load_BM_into_slot(std::integral_constant<int, 1>{}, pBM);
                normalize_slot_mirror_only<1>(
                    AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2, AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
                    AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2, AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
                    M_AF, M_BF, M_AM, M_BM);
                pAF += 8; pBF += 8; pAM -= 8; pBM -= 8;
            }
            compute_slot_valid_const<0, 8>(
                AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2,
                AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
                AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2,
                AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
                xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high);
            ++done; if (done >= full_steps) break;

            if (done + 1 < full_steps) {
                load_AF_into_slot(std::integral_constant<int, 0>{}, pAF);
                load_BF_into_slot(std::integral_constant<int, 0>{}, pBF);
                load_AM_into_slot(std::integral_constant<int, 0>{}, pAM);
                load_BM_into_slot(std::integral_constant<int, 0>{}, pBM);
                normalize_slot_mirror_only<0>(
                    AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2, AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
                    AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2, AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
                    M_AF, M_BF, M_AM, M_BM);
                pAF += 8; pBF += 8; pAM -= 8; pBM -= 8;
            }
            compute_slot_valid_const<1, 8>(
                AF0_L0, AF0_L1, AF0_L2, BF0_L0, BF0_L1, BF0_L2,
                AM0_L0, AM0_L1, AM0_L2, BM0_L0, BM0_L1, BM0_L2,
                AF1_L0, AF1_L1, AF1_L2, BF1_L0, BF1_L1, BF1_L2,
                AM1_L0, AM1_L1, AM1_L2, BM1_L0, BM1_L1, BM1_L2,
                xx_sum_low, xx_sum_high, xy_sum_low, xy_sum_high, yy_sum_low, yy_sum_high);
            ++done;
        }
        i += (full_steps << 3);
    }

    // ================= tail once (≤7), mirror-only normalize =================
    if (tail > 0) {
#define PREFETCH_TAIL()                                                  \
            do {                                                                 \
                load_AF_into_slot(std::integral_constant<int,0>{}, pAF);         \
                load_BF_into_slot(std::integral_constant<int,0>{}, pBF);         \
                load_AM_into_slot(std::integral_constant<int,0>{}, pAM);         \
                load_BM_into_slot(std::integral_constant<int,0>{}, pBM);         \
                normalize_slot_mirror_only<0>(                                   \
                    AF0_L0,AF0_L1,AF0_L2, BF0_L0,BF0_L1,BF0_L2, AM0_L0,AM0_L1,AM0_L2, BM0_L0,BM0_L1,BM0_L2, \
                    AF1_L0,AF1_L1,AF1_L2, BF1_L0,BF1_L1,BF1_L2, AM1_L0,AM1_L1,AM1_L2, BM1_L0,BM1_L1,BM1_L2, \
                    M_AF,M_BF,M_AM,M_BM);                                        \
            } while(0)
#define COMPUTE_TAIL(N_)                                                 \
            compute_slot_valid_const<0, N_>(                                      \
                AF0_L0,AF0_L1,AF0_L2, BF0_L0,BF0_L1,BF0_L2,                       \
                AM0_L0,AM0_L1,AM0_L2, BM0_L0,BM0_L1,BM0_L2,                       \
                AF1_L0,AF1_L1,AF1_L2, BF1_L0,BF1_L1,BF1_L2,                       \
                AM1_L0,AM1_L1,AM1_L2, BM1_L0,BM1_L1,BM1_L2,                       \
                xx_sum_low,xx_sum_high, xy_sum_low,xy_sum_high, yy_sum_low,yy_sum_high)

        PREFETCH_TAIL();
        switch (tail) {
        case 1: COMPUTE_TAIL(1); break; case 2: COMPUTE_TAIL(2); break;
        case 3: COMPUTE_TAIL(3); break; case 4: COMPUTE_TAIL(4); break;
        case 5: COMPUTE_TAIL(5); break; case 6: COMPUTE_TAIL(6); break;
        case 7: COMPUTE_TAIL(7); break; default: break;
        }
        i += tail;

#undef COMPUTE_TAIL
#undef PREFETCH_TAIL
    }
}
