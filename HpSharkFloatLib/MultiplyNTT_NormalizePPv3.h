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
ParallelPrefixNormalize3WayV3(cooperative_groups::grid_group &grid,
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
    uint64_t *results[3] = {resultXX, resultYY, resultXY};
    ParallelPrefixNormalize<SharkFloatParams, 3>(
        grid, block, cur, Ddigits, results, digitXfer, scanTemp, carryInMask);
}
