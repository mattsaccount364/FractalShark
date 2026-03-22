// ---- Templated N-channel DigitTransfer ----

template <int NumChannels>
struct DigitTransfer {
    uint8_t g; // NumChannels bits
    uint8_t p; // NumChannels bits
};

template <int NumChannels>
__device__ constexpr DigitTransfer<NumChannels>
DigitTransfer_identity()
{
    return DigitTransfer<NumChannels>{
        static_cast<uint8_t>(0u),
        static_cast<uint8_t>((1u << NumChannels) - 1u)};
}

template <int NumChannels>
__device__ inline DigitTransfer<NumChannels>
make_digit_transfer(uint8_t gMask, uint8_t pMask)
{
    constexpr uint8_t mask = static_cast<uint8_t>((1u << NumChannels) - 1u);
    return DigitTransfer<NumChannels>{
        static_cast<uint8_t>(gMask & mask),
        static_cast<uint8_t>(pMask & mask)};
}

template <int NumChannels>
__device__ inline DigitTransfer<NumChannels>
compose(const DigitTransfer<NumChannels> &a, const DigitTransfer<NumChannels> &b)
{
    return DigitTransfer<NumChannels>{
        static_cast<uint8_t>(b.g | (b.p & a.g)),
        static_cast<uint8_t>(b.p & a.p)};
}

template <int NumChannels>
__device__ inline uint32_t
apply_transfer(const DigitTransfer<NumChannels> &t, uint32_t inMask)
{
    uint32_t out = 0u;
#pragma unroll
    for (int ch = 0; ch < NumChannels; ++ch) {
        const uint32_t inBit = (inMask >> ch) & 0x1u;
        const uint32_t gBit = (t.g >> ch) & 0x1u;
        const uint32_t pBit = (t.p >> ch) & 0x1u;
        const uint32_t outBit = gBit | (pBit & inBit);
        out |= (outBit << ch);
    }
    return out;
}

// ---- Backward-compatible DigitTransfer3 aliases ----

using DigitTransfer3 = DigitTransfer<3>;

__device__ constexpr DigitTransfer3
DigitTransfer3_identity()
{
    return DigitTransfer_identity<3>();
}

__device__ inline DigitTransfer3
make_digit_transfer3(uint8_t gMask, uint8_t pMask)
{
    return make_digit_transfer<3>(gMask, pMask);
}

__device__ inline DigitTransfer3
compose(const DigitTransfer3 &a, const DigitTransfer3 &b)
{
    return compose<3>(a, b);
}

__device__ inline uint32_t
apply_transfer(const DigitTransfer3 &t, uint32_t inMask)
{
    return apply_transfer<3>(t, inMask);
}

// ---- Templated N-channel ParallelPrefixNormalize ----

template <class SharkFloatParams, int NumChannels>
static __device__ inline void
ParallelPrefixNormalize(cooperative_groups::grid_group &grid,
                        cooperative_groups::thread_block &block,
                        uint64_t *SharkRestrict cur,
                        const uint32_t Ddigits,
                        uint64_t *SharkRestrict *results,
                        DigitTransfer<NumChannels> *SharkRestrict digitXfer,
                        DigitTransfer<NumChannels> *SharkRestrict scanTemp,
                        uint32_t *SharkRestrict carryInMask)
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    const int warpSz = block.dim_threads().x;
#else
    constexpr int warpSz = 32;
#endif
    (void)warpSz;

    const int totalThreads = static_cast<int>(grid.size());
    const int tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    if (Ddigits == 0u) {
        return;
    }

    // 1) Build per-digit transfer functions (G/P) for all NumChannels channels.
    for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
        uint8_t gMask = 0u;
        uint8_t pMask = 0u;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint32_t dig = static_cast<uint32_t>(results[ch][i]);
            const uint8_t B = static_cast<uint8_t>(cur[i * NumChannels + ch] & 0x1u);

            const uint64_t sum0 = static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + 0u;
            const uint64_t sum1 = static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + 1u;

            const uint8_t cout0 = static_cast<uint8_t>((sum0 >> 32) & 0x1u);
            const uint8_t cout1 = static_cast<uint8_t>((sum1 >> 32) & 0x1u);

            const uint8_t G = cout0;
            const uint8_t P = static_cast<uint8_t>(cout0 ^ cout1);

            gMask |= static_cast<uint8_t>(G << ch);
            pMask |= static_cast<uint8_t>(P << ch);
        }

        digitXfer[i] = make_digit_transfer<NumChannels>(gMask, pMask);
    }

    grid.sync();

    // 2) Hillis-Steele inclusive scan over digitXfer.
    DigitTransfer<NumChannels> *in = digitXfer;
    DigitTransfer<NumChannels> *out = scanTemp;

    for (uint32_t offset = 1u; offset < Ddigits; offset <<= 1u) {
        for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
            DigitTransfer<NumChannels> v = in[i];
            if (i >= offset) {
                v = compose<NumChannels>(in[i - offset], v);
            }
            out[i] = v;
        }
        grid.sync();
        DigitTransfer<NumChannels> *tmp = in;
        in = out;
        out = tmp;
    }

    // 3) Apply exclusive prefix to compute incoming carry mask per digit,
    //    then add local B + c_in to the digits.
    for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
        const DigitTransfer<NumChannels> &prefix =
            (i == 0u) ? DigitTransfer_identity<NumChannels>() : in[i - 1u];

        const uint32_t c_in = apply_transfer<NumChannels>(prefix, 0u);

        carryInMask[i] = c_in;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint32_t dig = static_cast<uint32_t>(results[ch][i]);
            const uint32_t B = static_cast<uint32_t>(cur[i * NumChannels + ch] & 0x1u);
            const uint32_t c = (c_in >> ch) & 0x1u;

            const uint64_t full =
                static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + static_cast<uint64_t>(c);

            results[ch][i] = static_cast<uint32_t>(full & 0xffffffffu);
            cur[i * NumChannels + ch] = 0;
        }
    }

    grid.sync();
}

// ---- Backward-compatible 3-way wrapper ----

template <class SharkFloatParams>
static __device__ inline void
ParallelPrefixNormalize3WayV3(uint64_t *SharkRestrict shared_data,
                              cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              uint64_t *SharkRestrict cur,
                              const uint32_t Ddigits,
                              uint64_t *SharkRestrict resultXX,
                              uint64_t *SharkRestrict resultYY,
                              uint64_t *SharkRestrict resultXY,
                              DigitTransfer3 *SharkRestrict digitXfer,
                              DigitTransfer3 *SharkRestrict scanTemp,
                              uint32_t *SharkRestrict carryInMask)
{
    constexpr bool DLB = true;
    uint64_t *results[3] = {resultXX, resultYY, resultXY};

    if constexpr (DLB) {
        auto *descBuf = reinterpret_cast<uint32_t *>(scanTemp);
        ParallelPrefixNormalize_DLB<SharkFloatParams, 3>(
            shared_data, grid, block, cur, Ddigits, results, digitXfer, descBuf, carryInMask);
    } else {
        ParallelPrefixNormalize<SharkFloatParams, 3>(
            grid, block, cur, Ddigits, results, digitXfer, scanTemp, carryInMask);
    }
}

// ============================================================================
// DLB (Decoupled Lookback) variant of ParallelPrefixNormalize.
// Replaces Hillis-Steele O(log N) grid.sync() scan with warp-cooperative
// lookback using only 2 grid.sync() total (build transfers + apply).
// ============================================================================

template <int NumChannels>
__device__ inline DigitTransfer<NumChannels>
shfl_up_dt(unsigned mask, DigitTransfer<NumChannels> val, int offset)
{
    // Pack g/p into a single uint32 for shuffle
    uint32_t packed = (static_cast<uint32_t>(val.g) << 8) | static_cast<uint32_t>(val.p);
    packed = __shfl_up_sync(mask, packed, offset);
    return DigitTransfer<NumChannels>{
        static_cast<uint8_t>(packed >> 8),
        static_cast<uint8_t>(packed & 0xFF)};
}

template <int NumChannels>
__device__ inline DigitTransfer<NumChannels>
shfl_down_dt(unsigned mask, DigitTransfer<NumChannels> val, int offset)
{
    uint32_t packed = (static_cast<uint32_t>(val.g) << 8) | static_cast<uint32_t>(val.p);
    packed = __shfl_down_sync(mask, packed, offset);
    return DigitTransfer<NumChannels>{
        static_cast<uint8_t>(packed >> 8),
        static_cast<uint8_t>(packed & 0xFF)};
}

template <class SharkFloatParams, int NumChannels>
static __device__ inline void
ParallelPrefixNormalize_DLB(uint64_t *SharkRestrict shared_data,
                             cooperative_groups::grid_group &grid,
                             cooperative_groups::thread_block &block,
                             uint64_t *SharkRestrict cur,
                             const uint32_t Ddigits,
                             uint64_t *SharkRestrict *results,
                             DigitTransfer<NumChannels> *SharkRestrict digitXfer,
                             uint32_t *SharkRestrict descBuf,
                             uint32_t *SharkRestrict carryInMask)
{
    if (Ddigits == 0u) return;

    constexpr int warpSz = 32;
    const int totalThreads = static_cast<int>(grid.size());
    const int tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;
    const int lane = static_cast<int>(threadIdx.x & 31);
    const int warpId = static_cast<int>(threadIdx.x >> 5);
    const int numWarps = static_cast<int>((blockDim.x + 31) >> 5);
    const uint32_t warpMask = 0xFFFF'FFFFu;

    // Descriptor packing: FLAG (2 bits) | g (NumChannels bits) | p (NumChannels bits)
    // For NumChannels=7: 2+7+7=16 bits → fits in uint32_t
    constexpr uint32_t DATA_BITS = 2 * NumChannels;
    constexpr uint32_t FLAG_SHIFT = DATA_BITS;
    enum : uint32_t { FLAG_X = 0u, FLAG_A = 1u, FLAG_P = 2u };

    auto pack_desc = [](uint32_t flag, DigitTransfer<NumChannels> dt) -> uint32_t {
        return (flag << FLAG_SHIFT) | (static_cast<uint32_t>(dt.g) << NumChannels) |
               static_cast<uint32_t>(dt.p);
    };
    auto unpack_flag = [](uint32_t d) -> uint32_t { return d >> FLAG_SHIFT; };
    auto unpack_dt = [](uint32_t d) -> DigitTransfer<NumChannels> {
        return DigitTransfer<NumChannels>{
            static_cast<uint8_t>((d >> NumChannels) & ((1u << NumChannels) - 1u)),
            static_cast<uint8_t>(d & ((1u << NumChannels) - 1u))};
    };

    auto store_desc = [](uint32_t *addr, uint32_t v) {
        asm volatile("membar.gl;\n\tst.global.u32 [%0], %1;\n\t" :: "l"(addr), "r"(v) : "memory");
    };
    auto load_desc = [](const uint32_t *addr) -> uint32_t {
        uint32_t v;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(addr));
        return v;
    };

    // Partition: 1 digit per thread, 1 partition per block
    const int32_t digitsPerPart = static_cast<int32_t>(blockDim.x);
    const int32_t numParts = (static_cast<int32_t>(Ddigits) + digitsPerPart - 1) / digitsPerPart;

    // 1) Build per-digit transfer functions
    for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
        uint8_t gMask = 0u;
        uint8_t pMask = 0u;

#pragma unroll
        for (int ch = 0; ch < NumChannels; ++ch) {
            const uint32_t dig = static_cast<uint32_t>(results[ch][i]);
            const uint8_t B = static_cast<uint8_t>(cur[i * NumChannels + ch] & 0x1u);

            const uint64_t sum0 = static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + 0u;
            const uint64_t sum1 = static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + 1u;

            const uint8_t cout0 = static_cast<uint8_t>((sum0 >> 32) & 0x1u);
            const uint8_t cout1 = static_cast<uint8_t>((sum1 >> 32) & 0x1u);

            gMask |= static_cast<uint8_t>(cout0 << ch);
            pMask |= static_cast<uint8_t>((cout0 ^ cout1) << ch);
        }

        digitXfer[i] = make_digit_transfer<NumChannels>(gMask, pMask);
    }

    // Init descriptor array + allocate procId (all before single sync)
    for (int32_t i = tid; i < numParts; i += totalThreads) {
        descBuf[i] = pack_desc(FLAG_X, DigitTransfer_identity<NumChannels>());
    }

    // Shared memory layout: warpAgg[numWarps] + warpPref[numWarps] + broadcast[2]
    uint32_t *smem = reinterpret_cast<uint32_t *>(shared_data);
    uint32_t *warpAgg = smem;
    uint32_t *warpPref = smem + numWarps;
    uint32_t *broadcast = smem + 2 * numWarps;

    // Cooperative kernel: all blocks active. Use blockIdx directly — no atomicAdd needed.
    const int32_t procId = static_cast<int32_t>(block.group_index().x);
    const int32_t P_active = static_cast<int32_t>(gridDim.x);

    grid.sync(); // transfers built + desc initialized

    // 2) DLB scan — warp-cooperative, no grid.sync()
    for (int32_t partIdx = procId; partIdx < numParts; partIdx += P_active) {
        const int base = partIdx * digitsPerPart;
        const int remaining = static_cast<int>(Ddigits) - base;
        const int partLen = (remaining > digitsPerPart) ? digitsPerPart : max(0, remaining);
        const int iDigit = base + static_cast<int>(threadIdx.x);
        const bool hasDigit = (static_cast<int>(threadIdx.x) < partLen);

        // Per-thread transfer
        DigitTransfer<NumChannels> myT =
            hasDigit ? digitXfer[iDigit] : DigitTransfer_identity<NumChannels>();

        // Intra-warp inclusive scan
        DigitTransfer<NumChannels> inclWarp = myT;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            DigitTransfer<NumChannels> y = shfl_up_dt<NumChannels>(warpMask, inclWarp, offset);
            if (lane >= offset) {
                inclWarp = compose<NumChannels>(y, inclWarp);
            }
        }

        // Warp aggregate → shared memory
        if (lane == 31) {
            warpAgg[warpId] = pack_desc(FLAG_X, inclWarp); // just data, flag unused here
        }
        __syncthreads();

        // Warp 0: scan warp aggregates
        if (warpId == 0) {
            DigitTransfer<NumChannels> x =
                (lane < numWarps) ? unpack_dt(warpAgg[lane]) : DigitTransfer_identity<NumChannels>();

#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                if (offset >= numWarps) break;
                DigitTransfer<NumChannels> y = shfl_up_dt<NumChannels>(warpMask, x, offset);
                if (lane >= offset) {
                    x = compose<NumChannels>(y, x);
                }
            }

            // Exclusive prefix for each warp
            DigitTransfer<NumChannels> prev = shfl_up_dt<NumChannels>(warpMask, x, 1);
            DigitTransfer<NumChannels> pref =
                (lane == 0) ? DigitTransfer_identity<NumChannels>() : prev;
            warpPref[lane] = pack_desc(FLAG_X, pref);

            // Full partition aggregate
            if (lane == numWarps - 1) {
                warpAgg[0] = pack_desc(FLAG_X, x);
            }
        }
        __syncthreads();

        DigitTransfer<NumChannels> aggPart = unpack_dt(warpAgg[0]);

        // Publish aggregate
        if (threadIdx.x == 0) {
            store_desc(&descBuf[partIdx], pack_desc(FLAG_A, aggPart));
        }

        // DLB lookback (warp 0 only)
        DigitTransfer<NumChannels> exclPart = DigitTransfer_identity<NumChannels>();

        if (warpId == 0) {
            int32_t j = partIdx - 1;

            while (j >= 0) {
                const int32_t k = j - lane;

                uint32_t w = 0;
                if (k >= 0) {
                    int spin = 0;
                    do {
                        w = load_desc(&descBuf[k]);
                        if (unpack_flag(w) != FLAG_X) break;
                        if (++spin > 64) {
                            __nanosleep(64);
                            spin = 0;
                        }
                    } while (true);
                }

                const uint32_t f = (k >= 0) ? unpack_flag(w) : 0u;
                const uint32_t pMask_local = __ballot_sync(warpMask, (k >= 0) && (f == FLAG_P));
                const uint32_t aMask_local = __ballot_sync(warpMask, (k >= 0) && (f == FLAG_A));

                if (pMask_local) {
                    const int pLane = __ffs(pMask_local) - 1;
                    const uint32_t wP = __shfl_sync(warpMask, w, pLane);
                    DigitTransfer<NumChannels> prefK = unpack_dt(wP);

                    DigitTransfer<NumChannels> my = DigitTransfer_identity<NumChannels>();
                    if (lane < pLane && ((aMask_local >> lane) & 1u)) {
                        my = unpack_dt(w);
                    }

#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        DigitTransfer<NumChannels> partner =
                            shfl_down_dt<NumChannels>(warpMask, my, offset);
                        if ((lane + offset) < 32) {
                            my = compose<NumChannels>(my, partner);
                        }
                    }

                    if (lane == 0) {
                        exclPart = compose<NumChannels>(compose<NumChannels>(prefK, my), exclPart);
                    }
                    break;
                }

                if (aMask_local) {
                    DigitTransfer<NumChannels> my = DigitTransfer_identity<NumChannels>();
                    if ((aMask_local >> lane) & 1u) {
                        my = unpack_dt(w);
                    }

#pragma unroll
                    for (int offset = 1; offset < 32; offset <<= 1) {
                        DigitTransfer<NumChannels> partner =
                            shfl_down_dt<NumChannels>(warpMask, my, offset);
                        if ((lane + offset) < 32) {
                            my = compose<NumChannels>(my, partner);
                        }
                    }

                    if (lane == 0) {
                        exclPart = compose<NumChannels>(my, exclPart);
                    }
                }

                j -= 32;
            }
        }

        // Broadcast exclPart to block
        if (threadIdx.x == 0) {
            broadcast[0] = pack_desc(FLAG_X, exclPart);
        }
        __syncthreads();
        exclPart = unpack_dt(broadcast[0]);

        // Publish inclusive prefix
        if (threadIdx.x == 0) {
            DigitTransfer<NumChannels> inclPart = compose<NumChannels>(exclPart, aggPart);
            store_desc(&descBuf[partIdx], pack_desc(FLAG_P, inclPart));
        }

        // Apply: compute per-digit carry and update digits
        DigitTransfer<NumChannels> warpPre = unpack_dt(warpPref[warpId]);
        DigitTransfer<NumChannels> prevDT = shfl_up_dt<NumChannels>(warpMask, inclWarp, 1);
        DigitTransfer<NumChannels> exclWarp =
            (lane == 0) ? DigitTransfer_identity<NumChannels>() : prevDT;
        DigitTransfer<NumChannels> prefixBeforeDigit = compose<NumChannels>(warpPre, exclWarp);

        // Full prefix = exclPart ∘ prefixBeforeDigit (evaluated at carry_in=0)
        DigitTransfer<NumChannels> fullPrefix = compose<NumChannels>(exclPart, prefixBeforeDigit);
        const uint32_t c_in = apply_transfer<NumChannels>(fullPrefix, 0u);

        if (hasDigit) {
            carryInMask[iDigit] = c_in;

#pragma unroll
            for (int ch = 0; ch < NumChannels; ++ch) {
                const uint32_t dig = static_cast<uint32_t>(results[ch][iDigit]);
                const uint32_t B = static_cast<uint32_t>(cur[iDigit * NumChannels + ch] & 0x1u);
                const uint32_t c = (c_in >> ch) & 0x1u;

                const uint64_t full =
                    static_cast<uint64_t>(dig) + static_cast<uint64_t>(B) + static_cast<uint64_t>(c);

                results[ch][iDigit] = static_cast<uint32_t>(full & 0xffffffffu);
                cur[iDigit * NumChannels + ch] = 0;
            }
        }
    }

    grid.sync(); // #2: all digits written
}
