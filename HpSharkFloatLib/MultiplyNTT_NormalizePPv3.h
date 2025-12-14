struct DigitTransfer3 {
    // bit0=XX, bit1=YY, bit2=XY
    uint8_t g; // generate bits
    uint8_t p; // propagate bits
};

__device__ constexpr DigitTransfer3
DigitTransfer3_identity()
{
    // Identity for the carry-transfer algebra:
    // f(c_in) = c_in → g = 0, p = 1 (for each channel).
    return DigitTransfer3{static_cast<uint8_t>(0u),
                          static_cast<uint8_t>(0x7u)}; // three channels, all propagate
}

__device__ inline DigitTransfer3
make_digit_transfer3(uint8_t gMask, uint8_t pMask)
{
    return DigitTransfer3{static_cast<uint8_t>(gMask & 0x7u), static_cast<uint8_t>(pMask & 0x7u)};
}

// Compose two transfer functions: a then b (a = lower digits, b = higher).
// Per channel: g' = g_b | (p_b & g_a),  p' = p_b & p_a
__device__ inline DigitTransfer3
compose(const DigitTransfer3 &a, const DigitTransfer3 &b)
{
    DigitTransfer3 r{};
    const uint8_t g = static_cast<uint8_t>(b.g | (b.p & a.g));
    const uint8_t p = static_cast<uint8_t>(b.p & a.p);
    r.g = g;
    r.p = p;
    return r;
}

// Apply transfer to a packed 3-bit carry mask (bit0=XX, bit1=YY, bit2=XY).
__device__ inline uint32_t
apply_transfer(const DigitTransfer3 &t, uint32_t inMask)
{
    uint32_t out = 0u;

#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        const uint32_t inBit = (inMask >> ch) & 0x1u;
        const uint32_t gBit = (t.g >> ch) & 0x1u;
        const uint32_t pBit = (t.p >> ch) & 0x1u;

        const uint32_t outBit = gBit | (pBit & inBit);
        out |= (outBit << ch);
    }
    return out;
}

template <class SharkFloatParams>
static __device__ inline void
ParallelPrefixNormalize3WayV3(cooperative_groups::grid_group &grid,
                              cooperative_groups::thread_block &block,
                              uint64_t *SharkRestrict cur, // carries from previous stage
                              const uint32_t Ddigits,
                              uint64_t *SharkRestrict resultXX,
                              uint64_t *SharkRestrict resultYY,
                              uint64_t *SharkRestrict resultXY,
                              DigitTransfer3 *SharkRestrict digitXfer, // size >= Ddigits
                              DigitTransfer3 *SharkRestrict scanTemp,  // size >= Ddigits
                              uint32_t *SharkRestrict carryInMask)     // size >= Ddigits
{
#ifdef TEST_SMALL_NORMALIZE_WARP
    constexpr int warpSz = block.dim_threads().x;
#else
    constexpr int warpSz = 32;
#endif
    (void)warpSz; // avoid unused warning

    const int totalThreads = static_cast<int>(grid.size());
    const int tid = block.thread_index().x + block.group_index().x * block.dim_threads().x;

    if (Ddigits == 0u) {
        return;
    }

    // 1) Build per-digit transfer functions (G/P) for all three channels.
    //    For each channel, we model:
    //      sum = digit + B(local) + c_in,  c_out = sum >> 32
    //    and encode f(c_in) as G/P using:
    //      G = c_out(0),  P = c_out(0) ^ c_out(1).
    for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
        const uint32_t digXX = static_cast<uint32_t>(resultXX[i]);
        const uint32_t digYY = static_cast<uint32_t>(resultYY[i]);
        const uint32_t digXY = static_cast<uint32_t>(resultXY[i]);

        // IMPORTANT: these are at *3 + 0/1/2*, matching the original warp path.
        const uint8_t BXX = static_cast<uint8_t>(cur[i * 3 + 0] & 0x1u);
        const uint8_t BYY = static_cast<uint8_t>(cur[i * 3 + 1] & 0x1u);
        const uint8_t BXY = static_cast<uint8_t>(cur[i * 3 + 2] & 0x1u);

        uint8_t gMask = 0u;
        uint8_t pMask = 0u;

        auto computeGP = [](uint32_t d, uint8_t B) -> DigitTransfer3 {
            const uint64_t sum0 = static_cast<uint64_t>(d) + static_cast<uint64_t>(B) + 0u;
            const uint64_t sum1 = static_cast<uint64_t>(d) + static_cast<uint64_t>(B) + 1u;

            const uint8_t cout0 = static_cast<uint8_t>((sum0 >> 32) & 0x1u);
            const uint8_t cout1 = static_cast<uint8_t>((sum1 >> 32) & 0x1u);

            const uint8_t G = cout0;
            const uint8_t P = static_cast<uint8_t>(cout0 ^ cout1);

            DigitTransfer3 t{};
            t.g = G;
            t.p = P;
            return t;
        };

        {
            DigitTransfer3 t = computeGP(digXX, BXX);
            gMask |= static_cast<uint8_t>(t.g << 0);
            pMask |= static_cast<uint8_t>(t.p << 0);
        }
        {
            DigitTransfer3 t = computeGP(digYY, BYY);
            gMask |= static_cast<uint8_t>(t.g << 1);
            pMask |= static_cast<uint8_t>(t.p << 1);
        }
        {
            DigitTransfer3 t = computeGP(digXY, BXY);
            gMask |= static_cast<uint8_t>(t.g << 2);
            pMask |= static_cast<uint8_t>(t.p << 2);
        }

        digitXfer[i] = make_digit_transfer3(gMask, pMask);
        // NOTE: do NOT clear cur here; we still need B when we apply carries.
    }

    grid.sync();

    // 2) Hillis–Steele inclusive scan over digitXfer, for arbitrary Ddigits.
    //
    // After this, "in[i]" holds:
    //    in[i] = digitXfer[0] o digitXfer[1] o ... o digitXfer[i]
    // Exclusive prefix at i is:
    //    prefix(i) = (i == 0 ? identity : in[i - 1]).
    DigitTransfer3 *in = digitXfer;
    DigitTransfer3 *out = scanTemp;

    for (uint32_t offset = 1u; offset < Ddigits; offset <<= 1u) {
        for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
            DigitTransfer3 v = in[i];
            if (i >= offset) {
                v = compose(in[i - offset], v);
            }
            out[i] = v;
        }
        grid.sync();
        // swap in/out for next stage
        DigitTransfer3 *tmp = in;
        in = out;
        out = tmp;
    }

    // 3) Apply exclusive prefix to compute incoming carry mask per digit,
    //    then add local B + c_in to the digits.
    for (uint32_t i = tid; i < Ddigits; i += static_cast<uint32_t>(totalThreads)) {
        const DigitTransfer3 &prefix = (i == 0u) ? DigitTransfer3_identity() : in[i - 1u];

        const uint32_t c_in = apply_transfer(prefix, 0u); // no global carry-in

        carryInMask[i] = c_in; // optional, for debugging

        uint32_t digXX = static_cast<uint32_t>(resultXX[i]);
        uint32_t digYY = static_cast<uint32_t>(resultYY[i]);
        uint32_t digXY = static_cast<uint32_t>(resultXY[i]);

        const uint32_t BXX = static_cast<uint32_t>(cur[i * 3 + 0] & 0x1u);
        const uint32_t BYY = static_cast<uint32_t>(cur[i * 3 + 1] & 0x1u);
        const uint32_t BXY = static_cast<uint32_t>(cur[i * 3 + 2] & 0x1u);

        const uint32_t cXX = (c_in >> 0) & 0x1u;
        const uint32_t cYY = (c_in >> 1) & 0x1u;
        const uint32_t cXY = (c_in >> 2) & 0x1u;

        const uint64_t fullXX =
            static_cast<uint64_t>(digXX) + static_cast<uint64_t>(BXX) + static_cast<uint64_t>(cXX);
        const uint64_t fullYY =
            static_cast<uint64_t>(digYY) + static_cast<uint64_t>(BYY) + static_cast<uint64_t>(cYY);
        const uint64_t fullXY =
            static_cast<uint64_t>(digXY) + static_cast<uint64_t>(BXY) + static_cast<uint64_t>(cXY);

        resultXX[i] = static_cast<uint32_t>(fullXX & 0xffffffffu);
        resultYY[i] = static_cast<uint32_t>(fullYY & 0xffffffffu);
        resultXY[i] = static_cast<uint32_t>(fullXY & 0xffffffffu);

        // Done with the residual per-digit carries for this stage.
        cur[i * 3 + 0] = 0;
        cur[i * 3 + 1] = 0;
        cur[i * 3 + 2] = 0;
    }

    grid.sync();
}
