#include "stdafx.h"

#include "RefOrbitPrecision.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif

namespace FractalShark {
namespace {

constexpr double ScheduleGuardBits = 112.0;
constexpr mp_bitcnt_t ScheduleTierCount = 64;
constexpr mp_bitcnt_t CertificationBits = 256;
constexpr mp_bitcnt_t GapGuardLimbs = 2;
// Reserve ample headroom for exponent additions and conversion to limb units. An
// out-of-range bound rejects the attempt rather than wrapping or underflowing to zero.
constexpr int64_t BoundExponentLimit = std::numeric_limits<int64_t>::max() / 8;
constexpr int MaximumAlignment =
    -std::numeric_limits<double>::min_exponent + std::numeric_limits<double>::digits;
// Normalized magnitudes, their products, and their roots are in [0.25, 2).
// Their three leading binary64 encoding bits are 001, versus 000 for exact zero.
// Extract a 0/1 arithmetic factor without a comparison or a Debug-build branch.
constexpr int MagnitudeFlagShift = std::numeric_limits<uint64_t>::digits - 3;
constexpr int WordSignShift = std::numeric_limits<uint64_t>::digits - 1;
static_assert(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::radix == 2 &&
              std::numeric_limits<double>::digits == 53);

mp_size_t
MagnitudeSize(mpf_srcptr value)
{
    return value->_mp_size < 0 ? -value->_mp_size : value->_mp_size;
}

bool
HasBoundableExponent(mpf_srcptr value)
{
    return value->_mp_exp >= -BoundExponentLimit / GMP_NUMB_BITS &&
           value->_mp_exp <= BoundExponentLimit / GMP_NUMB_BITS;
}

// Local MPIR errors are either exact zero or a single binary quantum. Reserve two
// tags outside the supported exponent range; no floating-point bound is needed
// until all terms for one component have been collected.
constexpr int64_t ZeroQuantum = -std::numeric_limits<int64_t>::max() / 2;
constexpr int64_t InvalidQuantum = std::numeric_limits<int64_t>::max();

int64_t
LimbQuantum(int64_t exponent, int64_t shift)
{
    if (exponent < -BoundExponentLimit / GMP_NUMB_BITS ||
        exponent > BoundExponentLimit / GMP_NUMB_BITS) {
        return InvalidQuantum;
    }
    const int64_t bits = exponent * GMP_NUMB_BITS + shift;
    return bits < -BoundExponentLimit || bits >= BoundExponentLimit ? InvalidQuantum : bits;
}

int64_t
CopyQuantum(mpf_srcptr source, mpf_srcptr destination)
{
    if (!HasBoundableExponent(source)) {
        return InvalidQuantum;
    }
    const auto retained = static_cast<mp_size_t>(destination->_mp_prec) + 1;
    return MagnitudeSize(source) <= retained
               ? ZeroQuantum
               : LimbQuantum(static_cast<int64_t>(source->_mp_exp) - retained, 0);
}

int64_t
ProductQuantum(mpf_srcptr product, int64_t shift)
{
    if (!HasBoundableExponent(product)) {
        return InvalidQuantum;
    }
    const auto retained = static_cast<mp_size_t>(product->_mp_prec) + 1;
    return MagnitudeSize(product) < retained
               ? ZeroQuantum
               : LimbQuantum(static_cast<int64_t>(product->_mp_exp) - retained, shift);
}

int64_t
SumQuantum(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination)
{
    if (!HasBoundableExponent(left) || !HasBoundableExponent(right)) {
        return InvalidQuantum;
    }
    const int64_t high = std::max(left->_mp_exp, right->_mp_exp);
    const int64_t low = std::min(static_cast<int64_t>(left->_mp_exp) - MagnitudeSize(left),
                                 static_cast<int64_t>(right->_mp_exp) - MagnitudeSize(right));
    // MPIR clips aligned operands to destination precision. Four quanta cover both
    // operand tails and final truncation, including subtraction near cancellation.
    return high - low <= destination->_mp_prec ? ZeroQuantum
                                               : LimbQuantum(high - destination->_mp_prec, 2);
}

int64_t
DoublingQuantum(mpf_srcptr source)
{
    if (!HasBoundableExponent(source)) {
        return InvalidQuantum;
    }
    return MagnitudeSize(source) <= source->_mp_prec
               ? ZeroQuantum
               : LimbQuantum(static_cast<int64_t>(source->_mp_exp) - source->_mp_prec, 2);
}

RefOrbitErrorBound
QuantumBound(int64_t quantum)
{
    return quantum == ZeroQuantum ? RefOrbitErrorBound{} : RefOrbitErrorBound::PowerOfTwo(quantum);
}

} // namespace

RefOrbitErrorBound
RefOrbitErrorBound::Invalid()
{
    RefOrbitErrorBound result;
    result.m_Mantissa = std::numeric_limits<double>::infinity();
    return result;
}

RefOrbitErrorBound
RefOrbitErrorBound::Checked(RefOrbitErrorBound value)
{
    return value.m_Mantissa != 0.0 &&
                   (value.m_Exponent < -BoundExponentLimit || value.m_Exponent > BoundExponentLimit)
               ? Invalid()
               : value;
}

RefOrbitErrorBound
RefOrbitErrorBound::Normalize(double mantissa, int64_t exponent)
{
    RefOrbitErrorBound result;
    if (!std::isfinite(mantissa) || exponent < -BoundExponentLimit || exponent > BoundExponentLimit) {
        return Invalid();
    }
    int adjustment;
    result.m_Mantissa = std::frexp(mantissa, &adjustment);
    const auto nonzero =
        static_cast<int64_t>(std::bit_cast<uint64_t>(result.m_Mantissa) >> MagnitudeFlagShift);
    result.m_Exponent = (exponent + adjustment) * nonzero + ZeroExponent * (1 - nonzero);
    return Checked(result);
}

RefOrbitErrorBound
RefOrbitErrorBound::PowerOfTwo(int64_t exponent)
{
    if (exponent < -BoundExponentLimit || exponent >= BoundExponentLimit) {
        return Invalid();
    }
    RefOrbitErrorBound result;
    result.m_Mantissa = 0.5;
    result.m_Exponent = exponent + 1;
    return result;
}

RefOrbitErrorBound
RefOrbitErrorBound::LimbPower(int64_t exponent)
{
    if (exponent < -BoundExponentLimit / GMP_NUMB_BITS ||
        exponent > BoundExponentLimit / GMP_NUMB_BITS) {
        return Invalid();
    }
    return PowerOfTwo(exponent * GMP_NUMB_BITS);
}

RefOrbitErrorBound
RefOrbitErrorBound::FromMpfLower(mpf_srcptr value)
{
    if (!HasBoundableExponent(value)) {
        return Normalize(std::numeric_limits<double>::infinity(), 0);
    }
    double mantissa;
    const int64_t exponent = mpf_get_2exp_d(&mantissa, value);
    return Normalize(std::abs(mantissa), exponent);
}

RefOrbitErrorBound
RefOrbitErrorBound::FromMpfUpper(mpf_srcptr value)
{
    if (!HasBoundableExponent(value)) {
        return Normalize(std::numeric_limits<double>::infinity(), 0);
    }
    double mantissa;
    const int64_t exponent = mpf_get_2exp_d(&mantissa, value);
    // MPIR truncates its conversion; the adjacent double encloses every omitted bit.
    return Normalize(std::nextafter(std::abs(mantissa), std::numeric_limits<double>::infinity()),
                     exponent);
}

RefOrbitErrorBound
RefOrbitErrorBound::Add(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    if (!left.IsFinite() || !right.IsFinite()) {
        return Invalid();
    }
    return Checked(AddFinite(left, right));
}

RefOrbitErrorBound
RefOrbitErrorBound::AddFinite(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    if (left.m_Exponent < right.m_Exponent) {
        std::swap(left, right);
    }
    const int64_t difference = left.m_Exponent - right.m_Exponent;
    // If alignment underflows, its positive contribution is still covered by the
    // outward ulp on the larger operand. All arithmetic significands stay normalized.
    const double aligned = difference > MaximumAlignment
                               ? 0.0
                               : std::ldexp(right.m_Mantissa, -static_cast<int>(difference));
    // Zero sorts last without a special-case branch. For nonnegative binary64,
    // incrementing the encoding advances one ulp. The normalized addend's encoding
    // supplies that increment directly, including when alignment underflows.
    const double sum = left.m_Mantissa + aligned;
    left.m_Mantissa =
        std::bit_cast<double>(std::bit_cast<uint64_t>(sum) +
                              (std::bit_cast<uint64_t>(right.m_Mantissa) >> MagnitudeFlagShift));
    // The outward sum is in [0.5, 2]. A carry to exactly 2 needs two shifts.
    if (left.m_Mantissa >= 1.0) {
        const int shift = left.m_Mantissa >= 2.0 ? 2 : 1;
        left.m_Mantissa *= shift == 2 ? 0.25 : 0.5;
        left.m_Exponent += shift;
    }
    return left;
}

RefOrbitErrorBound
RefOrbitErrorBound::Multiply(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    if (!left.IsFinite() || !right.IsFinite()) {
        return Invalid();
    }
    return Checked(MultiplyFinite(left, right));
}

RefOrbitErrorBound
RefOrbitErrorBound::MultiplyFinite(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    const uint64_t product = std::bit_cast<uint64_t>(left.m_Mantissa * right.m_Mantissa);
    const auto nonzero = static_cast<int64_t>(product >> MagnitudeFlagShift);
    left.m_Mantissa = std::bit_cast<double>(product + static_cast<uint64_t>(nonzero));
    left.m_Exponent += right.m_Exponent;
    // Normalized inputs give an outward product in [0.25, 1]. Scaling is exact.
    if (left.m_Mantissa < 0.5) {
        left.m_Mantissa *= 2.0;
        --left.m_Exponent;
    } else if (left.m_Mantissa >= 1.0) {
        left.m_Mantissa *= 0.5;
        ++left.m_Exponent;
    }
    // Preserve the canonical sorting exponent through ordinary zero arithmetic,
    // without an early return. Both-zero exponent addition also fits int64_t.
    left.m_Exponent = left.m_Exponent * nonzero + ZeroExponent * (1 - nonzero);
    return left;
}

RefOrbitErrorBound
RefOrbitErrorBound::ScaleByPowerOfTwo(RefOrbitErrorBound value, int64_t shift)
{
    if (!value.IsFinite() || shift < -BoundExponentLimit || shift > BoundExponentLimit) {
        return Invalid();
    }
    value.m_Exponent +=
        shift * static_cast<int64_t>(std::bit_cast<uint64_t>(value.m_Mantissa) >> MagnitudeFlagShift);
    return Checked(value);
}

RefOrbitErrorBound
RefOrbitErrorBound::Norm(RefOrbitErrorBound x, RefOrbitErrorBound y)
{
    if (!x.IsFinite() || !y.IsFinite()) {
        return Invalid();
    }
    return Checked(NormFinite(x, y));
}

RefOrbitErrorBound
RefOrbitErrorBound::NormFinite(RefOrbitErrorBound x, RefOrbitErrorBound y)
{
    const auto squared = AddFinite(MultiplyFinite(x, x), MultiplyFinite(y, y));
    double mantissa = squared.m_Mantissa;
    int64_t exponent = squared.m_Exponent;
    if (exponent % 2 != 0) {
        mantissa *= 2.0;
        --exponent;
    }
    // Binary64 sqrt is correctly rounded on the supported x64 Windows/Linux targets.
    // Squared inputs and addition were already rounded upward, so one outward ulp
    // encloses the square root without changing the process floating-point environment.
    RefOrbitErrorBound result;
    const uint64_t root = std::bit_cast<uint64_t>(std::sqrt(mantissa));
    const auto nonzero = static_cast<int64_t>(root >> MagnitudeFlagShift);
    result.m_Mantissa = std::bit_cast<double>(root + static_cast<uint64_t>(nonzero));
    result.m_Exponent = exponent / 2;
    // sqrt's significand is in [sqrt(0.5), sqrt(2)], so at most one shift is needed.
    if (result.m_Mantissa >= 1.0) {
        result.m_Mantissa *= 0.5;
        ++result.m_Exponent;
    }
    result.m_Exponent = result.m_Exponent * nonzero + ZeroExponent * (1 - nonzero);
    return result;
}

bool
RefOrbitErrorBound::Less(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    if (!left.IsFinite() || !right.IsFinite()) {
        return false;
    }
    return left.m_Exponent != right.m_Exponent ? left.m_Exponent < right.m_Exponent
                                               : left.m_Mantissa < right.m_Mantissa;
}

bool
RefOrbitErrorBound::IsZero() const
{
    return m_Mantissa == 0.0;
}

bool
RefOrbitErrorBound::IsFinite() const
{
    return std::isfinite(m_Mantissa);
}

double
RefOrbitErrorBound::Log2() const
{
    return std::log2(m_Mantissa) + static_cast<double>(m_Exponent);
}

RefOrbitPrecisionSchedule::RefOrbitPrecisionSchedule(
    mp_bitcnt_t fullBits, mpf_srcptr cx, mpf_srcptr cy, mpf_srcptr copiedCx, mpf_srcptr copiedCy)
    : m_FullBits(fullBits),
      m_Quantum(std::max(mp_bitcnt_t{GMP_NUMB_BITS},
                         fullBits / (ScheduleTierCount * GMP_NUMB_BITS) * GMP_NUMB_BITS)),
      m_CScale(RefOrbitErrorBound::Norm(RefOrbitErrorBound::FromMpfUpper(copiedCx),
                                        RefOrbitErrorBound::FromMpfUpper(copiedCy))),
      m_CError(RefOrbitErrorBound::Norm(CopyError(cx, copiedCx), CopyError(cy, copiedCy))),
      m_Error(m_CError), m_RealAxis(cy->_mp_size == 0), m_Boundary(CertificationBits),
      m_Gap(fullBits + GapGuardLimbs * GMP_NUMB_BITS)
{
    m_Stats.FullBits = fullBits;
    m_Stats.MinimumBits = fullBits;
    // A bin boundary differs from the state's limb exponent by at most one.
    // Two extra limbs cover that alignment and the state's stored guard limb,
    // allowing exact gap subtraction even when only bits 100,000 places down differ.
}

RefOrbitPrecisionSchedule::~RefOrbitPrecisionSchedule() = default;

// Independent RAII members also release the first allocation if initializing the
// second scratch value fails, before the caller shuts down bounded allocator TLS.
RefOrbitPrecisionSchedule::Scratch::Scratch(mp_bitcnt_t bits) { mpf_init2(m_Value, bits); }

RefOrbitPrecisionSchedule::Scratch::~Scratch() { mpf_clear(m_Value); }

mpf_ptr
RefOrbitPrecisionSchedule::Scratch::Get()
{
    return m_Value;
}

mp_bitcnt_t
RefOrbitPrecisionSchedule::BeginStep(mpf_srcptr x, mpf_srcptr y, bool allowReduction)
{
    // Evaluate 2*Z*D+1 with one shared exponent per complex number. This tracker
    // only proposes work; dropping a very small aligned term never certifies a sample.
    double zx;
    double zy;
    const int64_t xExponent = mpf_get_2exp_d(&zx, x);
    const int64_t yExponent = mpf_get_2exp_d(&zy, y);
    const int64_t zExponent = x->_mp_size == 0   ? yExponent
                              : y->_mp_size == 0 ? xExponent
                                                 : std::max(xExponent, yExponent);
    const auto align = [&](double mantissa, int64_t difference) {
        return difference > MaximumAlignment ? 0.0 : std::ldexp(mantissa, -static_cast<int>(difference));
    };
    zx = x->_mp_size == 0 ? 0.0 : align(zx, zExponent - xExponent);
    zy = y->_mp_size == 0 ? 0.0 : align(zy, zExponent - yExponent);
    double nextX = 2.0 * (zx * m_DerivativeX - zy * m_DerivativeY);
    double nextY = 2.0 * (zx * m_DerivativeY + zy * m_DerivativeX);
    int64_t nextExponent = zExponent + m_DerivativeExponent;
    const double productNorm = std::max(std::abs(nextX), std::abs(nextY));
    if (productNorm != 0.0 && std::isfinite(productNorm)) {
        int shift;
        std::frexp(productNorm, &shift);
        nextX = std::ldexp(nextX, -shift);
        nextY = std::ldexp(nextY, -shift);
        nextExponent += shift;
    }
    if (productNorm == 0.0 || nextExponent < -MaximumAlignment) {
        nextX = 1.0;
        nextY = 0.0;
        nextExponent = 0;
    } else if (nextExponent < 0) {
        nextX = std::ldexp(nextX, static_cast<int>(nextExponent)) + 1.0;
        nextY = std::ldexp(nextY, static_cast<int>(nextExponent));
        nextExponent = 0;
    } else {
        nextX +=
            nextExponent > MaximumAlignment ? 0.0 : std::ldexp(1.0, -static_cast<int>(nextExponent));
    }
    const double nextNorm = std::max(std::abs(nextX), std::abs(nextY));
    if (nextNorm == 0.0) {
        // D can cancel to zero, but its next update must still recover D=1.
        m_DerivativeX = 0.0;
        m_DerivativeY = 0.0;
        m_DerivativeExponent = 0;
    } else if (std::isfinite(nextNorm) && std::abs(nextExponent) < BoundExponentLimit) {
        int shift;
        std::frexp(nextNorm, &shift);
        m_DerivativeX = std::ldexp(nextX, -shift);
        m_DerivativeY = std::ldexp(nextY, -shift);
        m_DerivativeExponent = nextExponent + shift;
    } else {
        m_DerivativeX = std::numeric_limits<double>::quiet_NaN();
        m_DerivativeY = 0.0;
        m_DerivativeExponent = 0;
    }
    m_StepRadius = RefOrbitErrorBound::Norm(RefOrbitErrorBound::FromMpfUpper(x),
                                            RefOrbitErrorBound::FromMpfUpper(y));
    // The infinity norm underestimates |D| by at most sqrt(2), conservatively asking
    // for up to one extra bit. This derivative is only a proposal, never the certificate.
    const double derivativeLog = std::log2(std::max(std::abs(m_DerivativeX), std::abs(m_DerivativeY))) +
                                 static_cast<double>(m_DerivativeExponent);
    const double scaleLog = std::max(2.0 * m_StepRadius.Log2(), m_CScale.Log2());
    const double need = static_cast<double>(m_FullBits) + scaleLog - derivativeLog + ScheduleGuardBits;
    mp_bitcnt_t bits = m_FullBits;
    if (allowReduction && std::isfinite(need) && need < static_cast<double>(m_FullBits)) {
        const double tiers = std::ceil(std::max(static_cast<double>(GMP_NUMB_BITS), need) /
                                       static_cast<double>(m_Quantum));
        bits = std::min(m_FullBits, static_cast<mp_bitcnt_t>(tiers) * m_Quantum);
    }
    ++m_Stats.Steps;
    m_Stats.ReducedSteps += bits < m_FullBits ? 1 : 0;
    m_Stats.MinimumBits = std::min(m_Stats.MinimumBits, static_cast<uint64_t>(bits));
    m_Stats.MeanBits +=
        (static_cast<double>(bits) - m_Stats.MeanBits) / static_cast<double>(m_Stats.Steps);
    return bits;
}

void
RefOrbitPrecisionSchedule::AddStateTruncation(RefOrbitErrorBound xError, RefOrbitErrorBound yError)
{
    m_Error = RefOrbitErrorBound::Add(m_Error, RefOrbitErrorBound::Norm(xError, yError));
}

void
RefOrbitPrecisionSchedule::FinishStep(RefOrbitErrorBound localXError, RefOrbitErrorBound localYError)
{
    // Validate once at the finite kernel's boundary. Nonzero input exponents are
    // bounded by L = INT64_MAX/8; nonzero intermediates stay within +/- (2*L+8).
    // Zero's sorting exponent is about -4*L, so alignment differences are at most
    // 6*L+12 and still fit int64_t. Only the final exponent needs another check.
    // An invalid term must never disappear through a zero.
    if (!m_Error.IsFinite() || !m_StepRadius.IsFinite() || !m_CError.IsFinite() ||
        !localXError.IsFinite() || !localYError.IsFinite()) {
        m_Error = RefOrbitErrorBound::Invalid();
        return;
    }
    // If z = zHat - e, the next error is 2*zHat*e - e^2 + localError.
    // A Euclidean disk preserves complex multiplication's norm exactly; propagating
    // independent absolute component errors would spuriously amplify every rotation.
    auto twiceRadius = m_StepRadius;
    twiceRadius.m_Exponent +=
        static_cast<int64_t>(std::bit_cast<uint64_t>(twiceRadius.m_Mantissa) >> MagnitudeFlagShift);
    const auto linear = RefOrbitErrorBound::MultiplyFinite(twiceRadius, m_Error);
    const auto quadratic = RefOrbitErrorBound::MultiplyFinite(m_Error, m_Error);
    const auto local = RefOrbitErrorBound::AddFinite(
        RefOrbitErrorBound::NormFinite(localXError, localYError), m_CError);
    m_Error = RefOrbitErrorBound::Checked(
        RefOrbitErrorBound::AddFinite(RefOrbitErrorBound::AddFinite(linear, quadratic), local));
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::AccumulateRoundingErrors(std::span<const int64_t> quanta)
{
    static_assert(ZeroQuantum == RefOrbitErrorBound::ZeroExponent);
    const int64_t largest = *std::max_element(quanta.begin(), quanta.end());
    if (largest == InvalidQuantum) {
        return RefOrbitErrorBound::Invalid();
    }
    // Zero quanta fall through as zero addends. Their low sorting exponent keeps
    // alignment differences in range, including when the whole component is zero.
    double sum = 0.0;
    for (const int64_t quantum : quanta) {
        const int64_t difference = largest - quantum;
        // OR with unsigned negation moves any set bit into the sign position,
        // producing the zero-quantum factor without a conditional selection.
        const uint64_t tag = static_cast<uint64_t>(quantum) ^ static_cast<uint64_t>(ZeroQuantum);
        const uint64_t nonzero = (tag | (uint64_t{0} - tag)) >> WordSignShift;
        const double aligned = difference > MaximumAlignment ? 0.0
                                                             : std::ldexp(static_cast<double>(nonzero),
                                                                          -static_cast<int>(difference));
        sum = std::bit_cast<double>(std::bit_cast<uint64_t>(sum + aligned) + nonzero);
    }
    // There are at most four terms. The sum is zero or in [1, 4 + four outward ulps],
    // so one exact scaling normalizes the entire component without frexp.
    assert(quanta.size() <= 4);
    const int shift = sum >= 4.0 ? 3 : sum >= 2.0 ? 2 : 1;
    RefOrbitErrorBound result;
    result.m_Mantissa = sum * (shift == 3 ? 0.125 : shift == 2 ? 0.25 : 0.5);
    const uint64_t sumBits = std::bit_cast<uint64_t>(sum);
    const auto nonzero = static_cast<int64_t>((sumBits | (uint64_t{0} - sumBits)) >> WordSignShift);
    result.m_Exponent = (largest + shift) * nonzero + ZeroQuantum * (1 - nonzero);
    return RefOrbitErrorBound::Checked(result);
}

void
RefOrbitPrecisionSchedule::ComputeStep(mpf_ptr x,
                                       mpf_ptr y,
                                       mpf_ptr xSquared,
                                       mpf_ptr ySquared,
                                       mpf_ptr cross,
                                       mpf_ptr temporary,
                                       mpf_srcptr cx,
                                       mpf_srcptr cy)
{
    // Read each bound before its operands are overwritten. All seven estimates
    // are zero or powers of two; doubling the cross-product error shifts its
    // exponent exactly rather than performing rounded bound multiplication.
    std::array<int64_t, 4> real{
        ProductQuantum(xSquared, 0), ProductQuantum(ySquared, 0), ZeroQuantum, ZeroQuantum};
    std::array<int64_t, 3> imaginary{ProductQuantum(cross, 1), DoublingQuantum(cross), ZeroQuantum};
    mpf_mul_ui(cross, cross, 2);
    imaginary[2] = SumQuantum(cross, cy, y);
    mpf_add(y, cross, cy);
    real[2] = SumQuantum(xSquared, ySquared, temporary);
    mpf_sub(temporary, xSquared, ySquared);
    real[3] = SumQuantum(temporary, cx, x);
    mpf_add(x, temporary, cx);
    FinishStep(AccumulateRoundingErrors(real), AccumulateRoundingErrors(imaginary));
}

bool
RefOrbitPrecisionSchedule::CheckGap(mpf_srcptr larger, mpf_srcptr smaller)
{
    const auto rounding = SumError(larger, smaller, m_Gap.Get());
    mpf_sub(m_Gap.Get(), larger, smaller);
    return mpf_sgn(m_Gap.Get()) > 0 &&
           RefOrbitErrorBound::Less(RefOrbitErrorBound::Add(m_Error, rounding),
                                    RefOrbitErrorBound::FromMpfLower(m_Gap.Get()));
}

bool
RefOrbitPrecisionSchedule::CertifyComponent(mpf_srcptr value, bool nativeDouble)
{
    if (!HasBoundableExponent(value)) {
        m_Stats.Failure = RefOrbitPrecisionFailure::UnsupportedConversion;
        return false;
    }
    double mantissa;
    const int64_t exponent = mpf_get_2exp_d(&mantissa, value);
    // Reject native subnormals/overflow and unrepresentable HDR exponents: the
    // certificate does not cover their platform-specific conversion rules.
    if ((nativeDouble && value->_mp_size != 0 &&
         (exponent < std::numeric_limits<double>::min_exponent ||
          exponent > std::numeric_limits<double>::max_exponent)) ||
        exponent < std::numeric_limits<int32_t>::min() ||
        exponent > std::numeric_limits<int32_t>::max()) {
        m_Stats.Failure = RefOrbitPrecisionFailure::UnsupportedConversion;
        return false;
    }
    if (m_Error.IsZero()) {
        return true;
    }
    if (value->_mp_size == 0) {
        return false;
    }
    // A positive magnitude truncated to d lies in [d, nextafter(d,+inf)). Enclose
    // both distances by exact dyadic subtraction plus a defensive rounding bound.
    // Requiring strict positive gaps is conservative at inclusive bin endpoints.
    // This descriptor borrows the input limbs. It changes only the sign and must
    // never be cleared or used as an arithmetic destination.
    __mpf_struct magnitude = *value;
    magnitude._mp_size = static_cast<decltype(magnitude._mp_size)>(MagnitudeSize(value));
    const auto setBoundary = [&](double boundary) {
        mpf_set_d(m_Boundary.Get(), boundary);
        if (exponent >= 0) {
            mpf_mul_2exp(m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(exponent));
        } else {
            mpf_div_2exp(m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(-exponent));
        }
    };
    setBoundary(std::abs(mantissa));
    if (!CheckGap(&magnitude, m_Boundary.Get())) {
        return false;
    }
    setBoundary(std::nextafter(std::abs(mantissa), std::numeric_limits<double>::infinity()));
    return CheckGap(m_Boundary.Get(), &magnitude);
}

bool
RefOrbitPrecisionSchedule::Certify(mpf_srcptr x, mpf_srcptr y, bool nativeDouble)
{
    if (!m_Error.IsFinite()) {
        m_Stats.Failure = RefOrbitPrecisionFailure::InvalidBound;
        return false;
    }
    // For real c, the imaginary recurrence is exactly zero independently of the
    // real error disk. Preserve that structural fact instead of rejecting every axis.
    const bool valid = CertifyComponent(x, nativeDouble) &&
                       ((m_RealAxis && y->_mp_size == 0) || CertifyComponent(y, nativeDouble));
    if (!valid && m_Stats.Failure == RefOrbitPrecisionFailure::None) {
        m_Stats.Failure = RefOrbitPrecisionFailure::AmbiguousSample;
    }
    return valid;
}

const RefOrbitDynamicPrecisionStats &
RefOrbitPrecisionSchedule::GetStats() const
{
    return m_Stats;
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::GetError() const
{
    return m_Error;
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::CopyError(mpf_srcptr source, mpf_srcptr destination)
{
    return QuantumBound(CopyQuantum(source, destination));
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::ProductError(mpf_srcptr product)
{
    return QuantumBound(ProductQuantum(product, 0));
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::SumError(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination)
{
    return QuantumBound(SumQuantum(left, right, destination));
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::DoublingError(mpf_srcptr source)
{
    return QuantumBound(DoublingQuantum(source));
}

RefOrbitRawPrecisionGuard::RefOrbitRawPrecisionGuard(
    mpf_ptr x, mpf_ptr y, mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_ptr temporary)
    : m_X(x), m_Y(y), m_XSquared(xSquared), m_YSquared(ySquared), m_Cross(cross), m_Temporary(temporary),
      m_FullBits(mpf_get_prec(x)), m_CurrentBits(m_FullBits)
{
    assert(mpf_get_prec(y) == m_FullBits && mpf_get_prec(xSquared) == m_FullBits &&
           mpf_get_prec(ySquared) == m_FullBits && mpf_get_prec(cross) == m_FullBits &&
           mpf_get_prec(temporary) == m_FullBits);
}

RefOrbitRawPrecisionGuard::~RefOrbitRawPrecisionGuard()
{
    for (auto *value : {m_X, m_Y, m_XSquared, m_YSquared, m_Cross, m_Temporary}) {
        mpf_set_prec_raw(value, m_FullBits);
    }
}

void
RefOrbitRawPrecisionGuard::Set(mp_bitcnt_t bits, RefOrbitPrecisionSchedule &schedule)
{
    assert(bits <= m_FullBits);
    if (bits == m_CurrentBits) {
        return;
    }
    for (auto *value : {m_X, m_Y, m_XSquared, m_YSquared, m_Cross, m_Temporary}) {
        mpf_set_prec_raw(value, bits);
    }
    const auto xError = RefOrbitPrecisionSchedule::CopyError(m_X, m_X);
    const auto yError = RefOrbitPrecisionSchedule::CopyError(m_Y, m_Y);
    // set_prec_raw alone preserves occupied mantissa size. Self-assignment removes
    // low limbs before the narrower worker buffers snapshot the state.
    mpf_set(m_X, m_X);
    mpf_set(m_Y, m_Y);
    schedule.AddStateTruncation(xError, yError);
    m_CurrentBits = bits;
}

} // namespace FractalShark

#ifdef _MSC_VER
#pragma float_control(pop)
#endif
