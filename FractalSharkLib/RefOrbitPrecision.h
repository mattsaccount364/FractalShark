#pragma once

#include "HDRFloat.h"
#include "RefOrbitKaratsuba.h"
#include "RefOrbitPrecisionStats.h"

#include <cstdint>
#include <limits>
#include <span>

namespace FractalShark {

// A nonnegative binary magnitude with an extended exponent. Upper operations round
// outward; FromMpfLower deliberately truncates and is used only for certification gaps.
// HDRFloat supplies the representation and arithmetic; this wrapper adds directed
// rounding and range validation required for an enclosure, rather than a point value.
class RefOrbitErrorBound {
public:
    static RefOrbitErrorBound PowerOfTwo(int64_t exponent);
    static RefOrbitErrorBound LimbPower(int64_t exponent);
    static RefOrbitErrorBound FromMpfUpper(mpf_srcptr value);
    static RefOrbitErrorBound FromMpfLower(mpf_srcptr value);
    static RefOrbitErrorBound Add(RefOrbitErrorBound left, RefOrbitErrorBound right);
    static RefOrbitErrorBound Multiply(RefOrbitErrorBound left, RefOrbitErrorBound right);
    static RefOrbitErrorBound ScaleByPowerOfTwo(RefOrbitErrorBound value, int64_t shift);
    static RefOrbitErrorBound Norm(RefOrbitErrorBound x, RefOrbitErrorBound y);
    static bool Less(RefOrbitErrorBound left, RefOrbitErrorBound right);

    bool IsZero() const;
    bool IsFinite() const;
    double Log2() const;

private:
    friend class RefOrbitPrecisionSchedule;

    static RefOrbitErrorBound Invalid();
    static RefOrbitErrorBound Checked(RefOrbitErrorBound value);
    static RefOrbitErrorBound Normalize(double mantissa, int64_t exponent);
    // Finite kernels retain zero/alignment handling but defer exponent validation to
    // their caller. The step recurrence has bounded depth and ample int64_t headroom.
    static RefOrbitErrorBound AddFinite(RefOrbitErrorBound left, RefOrbitErrorBound right);
    static RefOrbitErrorBound MultiplyFinite(RefOrbitErrorBound left, RefOrbitErrorBound right);
    static RefOrbitErrorBound NormFinite(RefOrbitErrorBound x, RefOrbitErrorBound y);
    // Sort zero below every nonzero finite-kernel intermediate. Its exponent is
    // ignored by validation, allowing addition to use ordinary alignment for zero.
    static constexpr int64_t ZeroExponent =
        HDRFloat<double, HDROrder::Left, int64_t>::MIN_BIG_EXPONENT();
    HDRFloat<double, HDROrder::Left, int64_t> m_Value;
};

// On-demand diagnostics for one complex state, not a minimum over the orbit.
// HeadroomBits = log2(gapLowerBound / comparisonErrorBound); the strict gap
// comparison passes only above zero. Infinite headroom with no component means
// an exact-zero error bound needs no gap test; unsupported conversions have NaNs.
struct RefOrbitCertificationDiagnostics {
    double AccumulatedErrorBoundLog2{-std::numeric_limits<double>::infinity()};
    double GapRoundingBoundLog2{-std::numeric_limits<double>::infinity()};
    double ComparisonErrorBoundLog2{-std::numeric_limits<double>::infinity()};
    double GapLowerBoundLog2{std::numeric_limits<double>::infinity()};
    double HeadroomBits{std::numeric_limits<double>::infinity()};
    const char *Component{"none"};
    const char *Boundary{"none"};
    RefOrbitPrecisionFailure Failure{RefOrbitPrecisionFailure::None};
};

// MT9 only: the derivative proposes a work precision, while an independent forward
// error enclosure certifies the actual conversion. Neither derivative growth nor
// raising precision is interpreted as recovering information that was discarded.
class RefOrbitPrecisionSchedule {
public:
    RefOrbitPrecisionSchedule(
        mp_bitcnt_t fullBits, mpf_srcptr cx, mpf_srcptr cy, mpf_srcptr copiedCx, mpf_srcptr copiedCy);
    ~RefOrbitPrecisionSchedule();
    RefOrbitPrecisionSchedule(const RefOrbitPrecisionSchedule &) = delete;
    RefOrbitPrecisionSchedule &operator=(const RefOrbitPrecisionSchedule &) = delete;

    mp_bitcnt_t BeginStep(mpf_srcptr x, mpf_srcptr y, bool allowReduction);
    void AddStateTruncation(RefOrbitErrorBound xError, RefOrbitErrorBound yError);
    void FinishStep(RefOrbitErrorBound localXError, RefOrbitErrorBound localYError);
    // Products have already completed. Consume their rounding metadata before each
    // MPIR operation overwrites it, then advance the state and its error disk together.
    void ComputeStep(mpf_ptr x,
                     mpf_ptr y,
                     mpf_ptr xSquared,
                     mpf_ptr ySquared,
                     mpf_ptr cross,
                     mpf_ptr temporary,
                     mpf_srcptr cx,
                     mpf_srcptr cy);
    bool Certify(mpf_srcptr x, mpf_srcptr y, bool nativeDouble);
    const RefOrbitDynamicPrecisionStats &GetStats() const;
    RefOrbitErrorBound GetError() const;
    // Reuses certification scratch only at termination; never changes attempt status.
    RefOrbitCertificationDiagnostics DescribeCertification(mpf_srcptr x,
                                                           mpf_srcptr y,
                                                           bool nativeDouble);

    static RefOrbitErrorBound CopyError(mpf_srcptr source, mpf_srcptr destination);
    static RefOrbitErrorBound ProductError(mpf_srcptr product);
    static RefOrbitErrorBound SumError(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination);
    static RefOrbitErrorBound DoublingError(mpf_srcptr source);

private:
    static RefOrbitErrorBound AccumulateRoundingErrors(std::span<const int64_t> quanta);
    bool CertifyComponent(mpf_srcptr value, bool nativeDouble);
    bool CheckGap(mpf_srcptr larger, mpf_srcptr smaller);

    mp_bitcnt_t m_FullBits;
    mp_bitcnt_t m_Quantum;
    // A shared binary exponent preserves complex rotations without overflowing a double.
    double m_DerivativeX{1.0};
    double m_DerivativeY{};
    int64_t m_DerivativeExponent{};
    RefOrbitErrorBound m_CScale;
    RefOrbitErrorBound m_CError;
    RefOrbitErrorBound m_Error;
    RefOrbitErrorBound m_StepRadius;
    bool m_RealAxis;
    RefOrbitDynamicPrecisionStats m_Stats;
    // The boundary is a short exact dyadic; the gap has enough preallocated limbs
    // for exact subtraction from any working state, including very deep near-returns.
    // No verification multiplication or per-sample allocation growth is needed.
    class Scratch {
    public:
        explicit Scratch(mp_bitcnt_t bits);
        ~Scratch();
        Scratch(const Scratch &) = delete;
        Scratch &operator=(const Scratch &) = delete;
        mpf_ptr Get();

    private:
        mpf_t m_Value;
    };
    Scratch m_Boundary;
    Scratch m_Gap;
};

// Raw precision changes never change the allocation. Restore its original size before
// HighPrecision calls mpf_clear, including on cancellation and an abandoned attempt.
class RefOrbitRawPrecisionGuard {
public:
    RefOrbitRawPrecisionGuard(
        mpf_ptr x, mpf_ptr y, mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_ptr temporary);
    ~RefOrbitRawPrecisionGuard();
    RefOrbitRawPrecisionGuard(const RefOrbitRawPrecisionGuard &) = delete;
    RefOrbitRawPrecisionGuard &operator=(const RefOrbitRawPrecisionGuard &) = delete;
    void Set(mp_bitcnt_t bits, RefOrbitPrecisionSchedule &schedule);

private:
    mpf_ptr m_X;
    mpf_ptr m_Y;
    mpf_ptr m_XSquared;
    mpf_ptr m_YSquared;
    mpf_ptr m_Cross;
    mpf_ptr m_Temporary;
    mp_bitcnt_t m_FullBits;
    mp_bitcnt_t m_CurrentBits;
};

} // namespace FractalShark
