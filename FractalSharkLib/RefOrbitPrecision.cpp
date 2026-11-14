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
constexpr int FastCertificateTailBits = 32;
constexpr int DoubleSignificandBits = std::numeric_limits<double>::digits;
// Reserve ample headroom for exponent additions and conversion to limb units. An
// out-of-range bound rejects the attempt rather than wrapping or underflowing to zero.
constexpr int64_t BoundExponentLimit = std::numeric_limits<int64_t>::max() / 8;
constexpr int MaximumAlignment =
    -std::numeric_limits<double>::min_exponent + std::numeric_limits<double>::digits;
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

bool
SupportsConversion(mpf_srcptr value, int64_t exponent, bool nativeDouble)
{
    // Native subnormals/overflow and HDR exponent overflow retain their existing
    // unsupported status, even when dynamic and fixed states are exactly equal.
    return exponent >= std::numeric_limits<int32_t>::min() &&
           exponent <= std::numeric_limits<int32_t>::max() &&
           (!nativeDouble || value->_mp_size == 0 ||
            (exponent >= std::numeric_limits<double>::min_exponent &&
             exponent <= std::numeric_limits<double>::max_exponent));
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

template <bool Subtract>
int64_t
SumQuantum(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination)
{
    if (!HasBoundableExponent(left) || !HasBoundableExponent(right)) {
        return InvalidQuantum;
    }
    // Zero addition/subtraction is a copy (possibly negated), retaining pi+1 limbs.
    // Aliasing can make the operation exact even sooner; the copy bound remains valid.
    if (left->_mp_size == 0 || right->_mp_size == 0) {
        return CopyQuantum(left->_mp_size == 0 ? right : left, destination);
    }
    const bool sameSign = (left->_mp_size < 0) == (right->_mp_size < 0);
    const bool subtractMagnitudes = Subtract ? sameSign : !sameSign;
    // MPIR/GMP magnitude addition clips aligned inputs to pi limbs. Magnitude
    // subtraction uses pi+1. Its special borrow/cancellation path can retain only
    // pi limbs, but first lowers the common exponent by at least one limb, so the
    // original-exponent pi+1 bound still covers that path. Equal-prefix scans can
    // improve exactness further; no limb scan is needed for this conservative bound.
    const int64_t retained = static_cast<int64_t>(destination->_mp_prec) + subtractMagnitudes;
    const int64_t high = std::max(left->_mp_exp, right->_mp_exp);
    const int64_t low = std::min(static_cast<int64_t>(left->_mp_exp) - MagnitudeSize(left),
                                 static_cast<int64_t>(right->_mp_exp) - MagnitudeSize(right));
    // Four quanta enclose both clipped operand tails and result rounding. Matching
    // the effective operation avoids charging an unnecessary full limb of error.
    return high - low <= retained ? ZeroQuantum : LimbQuantum(high - retained, 2);
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

RefOrbitErrorBound
MpfMagnitudeUpper(mpf_srcptr value)
{
    // Exact zero must remain zero in differential accounting. The scheduling
    // estimate deliberately retains its existing conversion/rounding behavior.
    return value->_mp_size == 0 ? RefOrbitErrorBound{} : RefOrbitErrorBound::FromMpfUpper(value);
}

} // namespace

RefOrbitErrorBound
RefOrbitErrorBound::Invalid()
{
    RefOrbitErrorBound result;
    result.m_Value.setMantissa(std::numeric_limits<double>::infinity());
    return result;
}

RefOrbitErrorBound
RefOrbitErrorBound::Checked(RefOrbitErrorBound value)
{
    // Legacy bounds used frexp's [0.5,1) convention. HDR's exponent is one lower;
    // preserve the existing permitted values and overflow behavior at that boundary.
    const int64_t exponent = value.m_Value.getExp();
    return !value.IsZero() && (exponent < -BoundExponentLimit - 1 || exponent >= BoundExponentLimit)
               ? Invalid()
               : value;
}

RefOrbitErrorBound
RefOrbitErrorBound::Normalize(double mantissa, int64_t exponent)
{
    if (!std::isfinite(mantissa) || exponent < -BoundExponentLimit || exponent > BoundExponentLimit) {
        return Invalid();
    }
    RefOrbitErrorBound result;
    result.m_Value = HDRFloat<double, HDROrder::Left, int64_t>{exponent, mantissa};
    result.m_Value.Reduce();
    return Checked(result);
}

RefOrbitErrorBound
RefOrbitErrorBound::PowerOfTwo(int64_t exponent)
{
    if (exponent < -BoundExponentLimit || exponent >= BoundExponentLimit) {
        return Invalid();
    }
    RefOrbitErrorBound result;
    result.m_Value = HDRFloat<double, HDROrder::Left, int64_t>{exponent, 1.0};
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
        return Invalid();
    }
    double mantissa;
    const int64_t exponent = mpf_get_2exp_d(&mantissa, value);
    return Normalize(std::abs(mantissa), exponent);
}

RefOrbitErrorBound
RefOrbitErrorBound::FromMpfUpper(mpf_srcptr value)
{
    if (!HasBoundableExponent(value)) {
        return Invalid();
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
    const uint64_t rounding =
        static_cast<uint64_t>(!left.IsZero()) & static_cast<uint64_t>(!right.IsZero());
    left.m_Value += right.m_Value;
    // For positive binary64 values the next encoding is the next outward ulp.
    // HDR may discard a tiny aligned addend; that ulp still covers it. Adding an
    // exact zero needs no rounding increment and follows the same arithmetic path.
    // Finite-kernel operands/results are zero or normal, so Reduce's nonfinite and
    // subnormal handling compiles out here and in the other finite kernels.
    left.m_Value.setMantissa(
        std::bit_cast<double>(std::bit_cast<uint64_t>(left.m_Value.getMantissa()) + rounding));
    left.m_Value.Reduce<false, false>();
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
    left.m_Value *= right.m_Value;
    const uint64_t rounding = static_cast<uint64_t>(!left.IsZero());
    left.m_Value.setMantissa(
        std::bit_cast<double>(std::bit_cast<uint64_t>(left.m_Value.getMantissa()) + rounding));
    left.m_Value.Reduce<false, false>();
    return left;
}

RefOrbitErrorBound
RefOrbitErrorBound::ScaleByPowerOfTwo(RefOrbitErrorBound value, int64_t shift)
{
    if (!value.IsFinite() || shift < -BoundExponentLimit || shift > BoundExponentLimit) {
        return Invalid();
    }
    value.m_Value.setExp(value.m_Value.getExp() + shift * static_cast<int64_t>(!value.IsZero()));
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
    RefOrbitErrorBound result;
    result.m_Value = HdrSqrt(squared.m_Value);
    // The squared norm is already rounded upward. Enclose the correctly rounded
    // square root by one additional ulp, retaining an exact zero without a branch.
    const uint64_t rounding = static_cast<uint64_t>(!result.IsZero());
    result.m_Value.setMantissa(
        std::bit_cast<double>(std::bit_cast<uint64_t>(result.m_Value.getMantissa()) + rounding));
    result.m_Value.Reduce<false, false>();
    return result;
}

bool
RefOrbitErrorBound::Less(RefOrbitErrorBound left, RefOrbitErrorBound right)
{
    return left.IsFinite() && right.IsFinite() &&
           left.m_Value.compareToBothPositiveReduced(right.m_Value) < 0;
}

bool
RefOrbitErrorBound::IsZero() const
{
    return m_Value.getMantissa() == 0.0;
}

bool
RefOrbitErrorBound::IsFinite() const
{
    return std::isfinite(m_Value.getMantissa());
}

double
RefOrbitErrorBound::Log2() const
{
    // Keep the previous evaluation order so a normalization convention change
    // cannot shift a precision tier at a rounding boundary.
    return std::log2(m_Value.getMantissa() * 0.5) + static_cast<double>(m_Value.getExp() + 1);
}

RefOrbitPrecisionSchedule::RefOrbitPrecisionSchedule(mp_bitcnt_t fullBits,
                                                     mpf_srcptr copiedCx,
                                                     mpf_srcptr copiedCy)
    : m_FullBits(fullBits),
      m_Quantum(std::max(mp_bitcnt_t{GMP_NUMB_BITS},
                         fullBits / (ScheduleTierCount * GMP_NUMB_BITS) * GMP_NUMB_BITS)),
      m_CScale(RefOrbitErrorBound::Norm(RefOrbitErrorBound::FromMpfUpper(copiedCx),
                                        RefOrbitErrorBound::FromMpfUpper(copiedCy))),
      m_FixedCoordinateScale(
          RefOrbitErrorBound::Norm(MpfMagnitudeUpper(copiedCx), MpfMagnitudeUpper(copiedCy))),
      m_RealAxis(copiedCy->_mp_size == 0), m_Boundary(CertificationBits),
      m_Gap(fullBits + GapGuardLimbs * GMP_NUMB_BITS)
{
    assert(fullBits >= GMP_NUMB_BITS && fullBits % GMP_NUMB_BITS == 0);
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
    if (m_DiagnosticCapture) {
        m_StepDiagnostics = {};
        m_StepDiagnostics.InputError = m_Error;
    }
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
    // Reuse the scheduling radius for ordinary complex states. A structural zero
    // needs no second norm: the remaining component's magnitude is its exact norm.
    m_DifferenceRadius = x->_mp_size == 0   ? MpfMagnitudeUpper(y)
                         : y->_mp_size == 0 ? MpfMagnitudeUpper(x)
                                            : m_StepRadius;
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
    m_DivergenceActive = m_DivergenceActive || bits < m_FullBits;
    m_Stats.ReducedSteps += bits < m_FullBits ? 1 : 0;
    m_Stats.MinimumBits = std::min(m_Stats.MinimumBits, static_cast<uint64_t>(bits));
    m_Stats.MeanBits +=
        (static_cast<double>(bits) - m_Stats.MeanBits) / static_cast<double>(m_Stats.Steps);
    return bits;
}

void
RefOrbitPrecisionSchedule::AddStateTruncation(RefOrbitErrorBound xError, RefOrbitErrorBound yError)
{
    if (m_DiagnosticCapture) {
        m_StepDiagnostics.ClipX = xError;
        m_StepDiagnostics.ClipY = yError;
    }
    m_Error = RefOrbitErrorBound::Add(m_Error, RefOrbitErrorBound::Norm(xError, yError));
    // Also permits direct accounting of a known truncation in bound tests. A zero
    // allowance during a precision increase does not recover discarded information.
    m_DivergenceActive = m_DivergenceActive || !xError.IsZero() || !yError.IsZero();
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::FixedRoundingAllowance(RefOrbitErrorBound fixedRadius) const
{
    const auto scale = RefOrbitErrorBound::Add(RefOrbitErrorBound::Multiply(fixedRadius, fixedRadius),
                                               m_FixedCoordinateScale);
    if (!scale.IsFinite() || scale.IsZero()) {
        return scale;
    }
    // S=U^2+|c| covers products and coordinates of every hypothetical fixed
    // state in the current disk. All intermediate magnitudes fit below 8*S.
    // For pi>=2, q<=8*S/beta^(pi-1) makes the primitive <=4*q errors too small
    // to exhaust that slack. This uses MPIR's full limb precision pi=P/w+1.
    constexpr int IntermediateMagnitudeShift = 3;
    const auto intermediate = RefOrbitErrorBound::ScaleByPowerOfTwo(scale, IntermediateMagnitudeShift);
    if (!intermediate.IsFinite()) {
        return intermediate;
    }
    // HDR mantissas are in [1,2). K=floor(e/w)+1 therefore gives the smallest
    // strict limb-power upper bound, including exact powers and negative e.
    // No logarithm, search, or MPIR arithmetic is required in this hot path.
    const int64_t exponent = intermediate.m_Value.getExp();
    const int64_t limbExponent = exponent / GMP_NUMB_BITS - (exponent % GMP_NUMB_BITS < 0) + 1;
    const int64_t fullLimbs = static_cast<int64_t>(m_FullBits / GMP_NUMB_BITS) + 1;
    const auto quantum = RefOrbitErrorBound::LimbPower(limbExponent - fullLimbs);
    // Real: (1+1+4+4)*q. Imaginary: (2+4+4)*q. The Euclidean local
    // residual is <=sqrt(10^2+10^2)*q<16*q. Copied coordinates are identical
    // in both trajectories, so their common copy error is not divergence.
    constexpr int LocalNormAllowanceShift = 4;
    return RefOrbitErrorBound::ScaleByPowerOfTwo(quantum, LocalNormAllowanceShift);
}

void
RefOrbitPrecisionSchedule::FinishStep(RefOrbitErrorBound localXError, RefOrbitErrorBound localYError)
{
    if (m_DiagnosticCapture) {
        m_StepDiagnostics.LocalX = localXError;
        m_StepDiagnostics.LocalY = localYError;
    }
    if (!m_DivergenceActive) {
        // Both specializations have the same inputs and operation order here.
        // Common rounding does not introduce any dynamic-minus-fixed difference.
        return;
    }
    // Validate once at the finite kernel's boundary. Nonzero input exponents are
    // bounded by L = INT64_MAX/8; nonzero intermediates stay within +/- (2*L+8).
    // Zero's sorting exponent is about -4*L, so alignment differences are at most
    // 6*L+12 and still fit int64_t. Only the final exponent needs another check.
    // An invalid term must never disappear through a zero.
    if (!m_Error.IsFinite() || !m_DifferenceRadius.IsFinite() || !localXError.IsFinite() ||
        !localYError.IsFinite()) {
        m_Error = RefOrbitErrorBound::Invalid();
        if (m_DiagnosticCapture) {
            m_StepDiagnostics.OutputError = m_Error;
        }
        return;
    }
    const auto fixedAllowance =
        FixedRoundingAllowance(RefOrbitErrorBound::Add(m_DifferenceRadius, m_Error));
    if (!fixedAllowance.IsFinite()) {
        m_Error = RefOrbitErrorBound::Invalid();
        if (m_DiagnosticCapture) {
            m_StepDiagnostics.OutputError = m_Error;
        }
        return;
    }
    // e=d-f gives eNext=2*d*e-e^2+rhoDynamic-rhoFixed. Bound both
    // residuals independently after divergence; shared-prefix rounding cancels.
    // The Euclidean disk does not amplify complex rotations spuriously.
    auto twiceRadius = m_DifferenceRadius;
    twiceRadius.m_Value.multiply2_mutable();
    const auto linear = RefOrbitErrorBound::MultiplyFinite(twiceRadius, m_Error);
    const auto quadratic = RefOrbitErrorBound::MultiplyFinite(m_Error, m_Error);
    const auto local = RefOrbitErrorBound::AddFinite(
        RefOrbitErrorBound::NormFinite(localXError, localYError), fixedAllowance);
    m_Error = RefOrbitErrorBound::Checked(
        RefOrbitErrorBound::AddFinite(RefOrbitErrorBound::AddFinite(linear, quadratic), local));
    if (m_DiagnosticCapture) {
        m_StepDiagnostics.Linear = linear;
        m_StepDiagnostics.Quadratic = quadratic;
        m_StepDiagnostics.FixedRoundingAllowance = fixedAllowance;
        m_StepDiagnostics.OutputError = m_Error;
    }
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
    // There are at most four terms. Let HDR normalize their outward-rounded sum,
    // including the canonical sorting exponent when every quantum is exact zero.
    assert(quanta.size() <= 4);
    RefOrbitErrorBound result;
    result.m_Value = HDRFloat<double, HDROrder::Left, int64_t>{largest, sum};
    result.m_Value.Reduce<false, false>();
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
    imaginary[2] = SumQuantum<false>(cross, cy, y);
    mpf_add(y, cross, cy);
    real[2] = SumQuantum<true>(xSquared, ySquared, temporary);
    mpf_sub(temporary, xSquared, ySquared);
    real[3] = SumQuantum<false>(temporary, cx, x);
    mpf_add(x, temporary, cx);
    if (m_DiagnosticCapture) {
        for (size_t index = 0; index < real.size(); ++index) {
            m_StepDiagnostics.PrimitiveErrors[index] = QuantumBound(real[index]);
        }
        for (size_t index = 0; index < imaginary.size(); ++index) {
            m_StepDiagnostics.PrimitiveErrors[real.size() + index] = QuantumBound(imaginary[index]);
        }
    }
    FinishStep(AccumulateRoundingErrors(real), AccumulateRoundingErrors(imaginary));
}

void
RefOrbitPrecisionSchedule::EnableDiagnosticCapture(bool enabled)
{
    m_DiagnosticCapture = enabled;
}

const RefOrbitStepDiagnostics &
RefOrbitPrecisionSchedule::GetStepDiagnostics() const
{
    return m_StepDiagnostics;
}

void
RefOrbitPrecisionSchedule::RecordCertificationStatus(CertificationRecord &record,
                                                     RefOrbitErrorBudgetStatus status)
{
    // A finite check supersedes exact checks; any failure supersedes a finite
    // record. Once a failure is observed, retain its earliest state.
    if (record.Failed || (status == RefOrbitErrorBudgetStatus::Exact &&
                          record.Status != RefOrbitErrorBudgetStatus::NotChecked)) {
        return;
    }
    record.Status = status;
    record.StateIteration = m_Stats.Steps + 1;
    record.Failed = status != RefOrbitErrorBudgetStatus::Exact;
}

void
RefOrbitPrecisionSchedule::RecordCertification(CertificationRecord &record,
                                               RefOrbitErrorBound comparison,
                                               RefOrbitErrorBound gap,
                                               int64_t componentExponent,
                                               bool passes)
{
    const auto status = !comparison.IsFinite() || !gap.IsFinite()
                            ? RefOrbitErrorBudgetStatus::InvalidBound
                        : gap.IsZero()        ? RefOrbitErrorBudgetStatus::ZeroGap
                        : comparison.IsZero() ? RefOrbitErrorBudgetStatus::Exact
                                              : RefOrbitErrorBudgetStatus::Bounded;
    if (status != RefOrbitErrorBudgetStatus::Bounded) {
        RecordCertificationStatus(record, status);
        return;
    }
    if (record.Failed) {
        return;
    }

    // Larger comparison/gap means less certification headroom. Separate exponents
    // avoid underflow at deep zooms; the quotient of normalized significands is
    // within [1/2, 2). This ranking is diagnostic only: acceptance still uses the
    // original directed bounds. Equal binary64 pressures retain the earlier state.
    int64_t pressureExponent = comparison.m_Value.getExp() - gap.m_Value.getExp();
    double pressureMantissa = comparison.m_Value.getMantissa() / gap.m_Value.getMantissa();
    if (pressureMantissa < 1.0) {
        pressureMantissa *= 2.0;
        --pressureExponent;
    }
    // A failed strict comparison always replaces an accepted record, even when
    // their diagnostic quotients round to the same binary64 pressure.
    if (passes && record.Status == RefOrbitErrorBudgetStatus::Bounded &&
        (pressureExponent < record.PressureExponent ||
         (pressureExponent == record.PressureExponent && pressureMantissa <= record.PressureMantissa))) {
        return;
    }
    record.Comparison = comparison;
    record.Gap = gap;
    record.ComponentExponent = componentExponent;
    record.PressureExponent = pressureExponent;
    record.PressureMantissa = pressureMantissa;
    record.StateIteration = m_Stats.Steps + 1;
    record.Status = RefOrbitErrorBudgetStatus::Bounded;
    record.Failed = !passes;
}

RefOrbitPrecisionFailure
RefOrbitPrecisionSchedule::CertifyComponent(mpf_srcptr value,
                                            bool nativeDouble,
                                            CertificationRecord &record)
{
    if (TryCertifyComponentFast(value, nativeDouble, record)) {
        ++m_Stats.FastCertifiedComponents;
        return RefOrbitPrecisionFailure::None;
    }
    ++m_Stats.RegularCertificationComponents;
    const auto certificate = EvaluateComponent(value, nativeDouble, false);
    if (certificate.Status != RefOrbitErrorBudgetStatus::Bounded) {
        RecordCertificationStatus(record, certificate.Status);
        return certificate.Status == RefOrbitErrorBudgetStatus::Exact ? RefOrbitPrecisionFailure::None
               : certificate.Status == RefOrbitErrorBudgetStatus::UnsupportedConversion
                   ? RefOrbitPrecisionFailure::UnsupportedConversion
                   : RefOrbitPrecisionFailure::AmbiguousSample;
    }
    for (size_t index = 0; index < certificate.GapCount; ++index) {
        const auto &boundary = certificate.Gaps[index];
        const bool passes = RefOrbitErrorBound::Less(boundary.Comparison, boundary.Gap);
        if (boundary.Gap.IsZero()) {
            RecordCertificationStatus(record, RefOrbitErrorBudgetStatus::ZeroGap);
        } else {
            RecordCertification(record, boundary.Comparison, boundary.Gap, certificate.Exponent, passes);
        }
        if (!passes) {
            return RefOrbitPrecisionFailure::AmbiguousSample;
        }
    }
    return RefOrbitPrecisionFailure::None;
}

bool
RefOrbitPrecisionSchedule::TryCertifyComponentFast(mpf_srcptr value,
                                                   bool nativeDouble,
                                                   CertificationRecord &record)
{
    // The bit layout below is deliberately restricted to the supported x64 limb
    // representation. Other layouts retain the original MPIR conversion certificate.
    if constexpr (GMP_NUMB_BITS != 64 || GMP_NAIL_BITS != 0) {
        return false;
    } else {
        if (!HasBoundableExponent(value)) {
            return false;
        }
        constexpr int WordBits = std::numeric_limits<uint64_t>::digits;
        const mp_size_t size = MagnitudeSize(value);
        const uint64_t high = size == 0 ? 0 : static_cast<uint64_t>(value->_mp_d[size - 1]);
        const int shift = size == 0 ? 0 : std::countl_zero(high);
        // HasBoundableExponent leaves enough int64_t room for normalization and
        // the additional 85-bit shift, even at the supported exponent endpoints.
        const int64_t exponent = size == 0 ? 0 : static_cast<int64_t>(value->_mp_exp) * WordBits - shift;
        if (!SupportsConversion(value, exponent, nativeDouble)) {
            return false;
        }
        // Preserve conversion restrictions even when the state has exact zero
        // error. Such supported checks need neither gap extraction nor MPIR calls.
        if (m_Error.IsZero()) {
            RecordCertificationStatus(record, RefOrbitErrorBudgetStatus::Exact);
            return true;
        }
        if (size == 0) {
            return false;
        }
        const uint64_t middle = size > 1 ? static_cast<uint64_t>(value->_mp_d[size - 2]) : 0;
        const uint64_t low = size > 2 ? static_cast<uint64_t>(value->_mp_d[size - 3]) : 0;
        uint64_t normalizedHigh = high;
        uint64_t normalizedLow = middle;
        if (shift != 0) {
            normalizedHigh = (high << shift) | (middle >> (WordBits - shift));
            normalizedLow = (middle << shift) | (low >> (WordBits - shift));
        }
        // After normalization, 11 discarded bits remain in the first word and
        // 21 more come from the second. The cast discards the retained 53 bits.
        constexpr int TailBitsInHigh = WordBits - DoubleSignificandBits;
        constexpr int TailBitsInLow = FastCertificateTailBits - TailBitsInHigh;
        const auto tail = static_cast<uint32_t>((normalizedHigh << TailBitsInLow) |
                                                (normalizedLow >> (WordBits - TailBitsInLow)));
        const uint32_t coefficient = std::min(tail, std::numeric_limits<uint32_t>::max() - tail);
        if (coefficient == 0) {
            return false;
        }
        // If U=2^(exponent-53), q=U/2^32, and t is the extracted prefix, the
        // unread tail gives t*q <= |v|-B0 < (t+1)*q. Both boundary distances
        // therefore exceed or equal min(t,2^32-1-t)*q. Its 32-bit coefficient
        // converts to double exactly; HDR normalization preserves the exact dyadic
        // lower bound. Upward-rounded multiplication would be unsound here.
        const auto gap =
            RefOrbitErrorBound::Normalize(static_cast<double>(coefficient),
                                          exponent - DoubleSignificandBits - FastCertificateTailBits);
        if (!RefOrbitErrorBound::Less(m_Error, gap)) {
            return false;
        }
        RecordCertification(record, m_Error, gap, exponent, true);
        return true;
    }
}

RefOrbitPrecisionSchedule::ComponentCertificate
RefOrbitPrecisionSchedule::EvaluateComponent(mpf_srcptr value, bool nativeDouble, bool detailed)
{
    ComponentCertificate result;
    double mantissa = 0.0;
    const bool boundable = HasBoundableExponent(value);
    if (boundable) {
        result.Exponent = mpf_get_2exp_d(&mantissa, value);
    }
    if (!boundable || !SupportsConversion(value, result.Exponent, nativeDouble)) {
        result.Status = RefOrbitErrorBudgetStatus::UnsupportedConversion;
        return result;
    }
    if (m_Error.IsZero()) {
        return result;
    }
    if (value->_mp_size == 0) {
        result.Status = RefOrbitErrorBudgetStatus::ZeroGap;
        return result;
    }
    // A positive magnitude truncated to d lies in [d, nextafter(d,+inf)). Enclose
    // both distances by exact dyadic subtraction plus a defensive rounding bound.
    // Requiring strict positive gaps is conservative at inclusive bin endpoints.
    // This descriptor borrows the input limbs. It changes only the sign and must
    // never be cleared or used as an arithmetic destination.
    __mpf_struct magnitude = *value;
    magnitude._mp_size = static_cast<decltype(magnitude._mp_size)>(MagnitudeSize(value));
    result.Status = RefOrbitErrorBudgetStatus::Bounded;
    for (auto &boundary : result.Gaps) {
        const bool upper = result.GapCount != 0;
        const double endpoint =
            upper ? std::nextafter(std::abs(mantissa), std::numeric_limits<double>::infinity())
                  : std::abs(mantissa);
        mpf_set_d(m_Boundary.Get(), endpoint);
        if (result.Exponent >= 0) {
            mpf_mul_2exp(m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(result.Exponent));
        } else {
            mpf_div_2exp(m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(-result.Exponent));
        }
        mpf_srcptr larger = upper ? m_Boundary.Get() : &magnitude;
        mpf_srcptr smaller = upper ? &magnitude : m_Boundary.Get();
        boundary.Rounding = SubtractError(larger, smaller, m_Gap.Get());
        mpf_sub(m_Gap.Get(), larger, smaller);
        boundary.Gap = mpf_sgn(m_Gap.Get()) > 0 ? RefOrbitErrorBound::FromMpfLower(m_Gap.Get())
                                                : RefOrbitErrorBound{};
        boundary.Comparison = RefOrbitErrorBound::Add(m_Error, boundary.Rounding);
        ++result.GapCount;
        // Production stops at the first rejected boundary. Diagnostic replay
        // evaluates both, but never changes records, counters, or failure history.
        if (!detailed && !RefOrbitErrorBound::Less(boundary.Comparison, boundary.Gap)) {
            break;
        }
    }
    return result;
}

bool
RefOrbitPrecisionSchedule::Certify(mpf_srcptr x, mpf_srcptr y, bool nativeDouble)
{
    // Keep checking after a warning, but latch its original cause and state. Later
    // checks and error propagation are diagnostics and never steer the schedule.
    auto failure = RefOrbitPrecisionFailure::None;
    if (!m_Error.IsFinite()) {
        failure = RefOrbitPrecisionFailure::InvalidBound;
        RecordCertificationStatus(m_WorstReal, RefOrbitErrorBudgetStatus::InvalidBound);
        RecordCertificationStatus(m_WorstImaginary, RefOrbitErrorBudgetStatus::InvalidBound);
    } else {
        // A real coordinate has an exactly zero imaginary recurrence independently
        // of the real error disk.
        failure = CertifyComponent(x, nativeDouble, m_WorstReal);
        if (failure == RefOrbitPrecisionFailure::None) {
            if (m_RealAxis && y->_mp_size == 0) {
                RecordCertificationStatus(m_WorstImaginary, RefOrbitErrorBudgetStatus::Exact);
            } else {
                failure = CertifyComponent(y, nativeDouble, m_WorstImaginary);
            }
        }
    }
    if (m_Stats.Failure == RefOrbitPrecisionFailure::None && failure != RefOrbitPrecisionFailure::None) {
        m_Stats.Failure = failure;
        m_Stats.FailureIteration = m_Stats.Steps + 1;
        m_Stats.FailureWorkingBits = mpf_get_prec(x);
    }
    return failure == RefOrbitPrecisionFailure::None;
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

RefOrbitComponentErrorBudget
RefOrbitPrecisionSchedule::DescribeBudget(RefOrbitErrorBudgetStatus status,
                                          uint64_t stateIteration,
                                          RefOrbitErrorBound comparison,
                                          RefOrbitErrorBound gap,
                                          int64_t componentExponent) const
{
    RefOrbitComponentErrorBudget budget;
    budget.Status = status;
    budget.StateIteration = stateIteration;
    if (status == RefOrbitErrorBudgetStatus::Bounded) {
        // Cancel integer exponents before converting to double so small budgets
        // survive extreme binary scaling. Logarithms remain outside production checks.
        const int64_t baseline = componentExponent - static_cast<int64_t>(m_FullBits);
        budget.UsedBits = static_cast<double>(comparison.m_Value.getExp() - baseline) +
                          std::log2(comparison.m_Value.getMantissa());
        budget.MaxBits =
            static_cast<double>(gap.m_Value.getExp() - baseline) + std::log2(gap.m_Value.getMantissa());
    }
    return budget;
}

RefOrbitWorstCertificationDiagnostics
RefOrbitPrecisionSchedule::DescribeWorstCertification() const
{
    const auto describe = [&](const CertificationRecord &record) {
        return DescribeBudget(record.Status,
                              record.StateIteration,
                              record.Comparison,
                              record.Gap,
                              record.ComponentExponent);
    };
    return {describe(m_WorstReal), describe(m_WorstImaginary)};
}

RefOrbitCertificationDiagnostics
RefOrbitPrecisionSchedule::DescribeCertification(mpf_srcptr x, mpf_srcptr y, bool nativeDouble)
{
    RefOrbitCertificationDiagnostics result;
    result.Real.StateIteration = result.Imaginary.StateIteration = m_Stats.Steps + 1;
    result.AccumulatedErrorBoundLog2 = m_Error.Log2();
    if (!m_Error.IsFinite()) {
        result.Failure = RefOrbitPrecisionFailure::InvalidBound;
        result.HeadroomBits = std::numeric_limits<double>::quiet_NaN();
        result.Real.Status = result.Imaginary.Status = RefOrbitErrorBudgetStatus::InvalidBound;
        return result;
    }

    // Reuse the regular certificate without changing attempt state. Unlike the
    // production traversal, describe both components and both boundary distances.
    // Unsupported conversions and zero components fix the aggregate result, while
    // the other component's budget is still evaluated for diagnostic completeness.
    bool aggregateComplete = false;
    for (int component = 0; component < 2; ++component) {
        const auto certificate = component == 1 && m_RealAxis && y->_mp_size == 0
                                     ? ComponentCertificate{}
                                     : EvaluateComponent(component == 0 ? x : y, nativeDouble, true);
        const char *componentName = component == 0 ? "real" : "imaginary";
        auto &budget = component == 0 ? result.Real : result.Imaginary;
        budget.Status = certificate.Status;
        if (certificate.Status == RefOrbitErrorBudgetStatus::Exact) {
            continue;
        }
        if (certificate.Status != RefOrbitErrorBudgetStatus::Bounded) {
            if (!aggregateComplete) {
                const bool unsupported =
                    certificate.Status == RefOrbitErrorBudgetStatus::UnsupportedConversion;
                result.Component = componentName;
                result.Boundary = unsupported ? "none" : "zero";
                result.Failure = unsupported ? RefOrbitPrecisionFailure::UnsupportedConversion
                                             : RefOrbitPrecisionFailure::AmbiguousSample;
                result.GapRoundingBoundLog2 = -std::numeric_limits<double>::infinity();
                result.ComparisonErrorBoundLog2 =
                    unsupported ? std::numeric_limits<double>::quiet_NaN() : m_Error.Log2();
                result.GapLowerBoundLog2 = result.HeadroomBits =
                    unsupported ? std::numeric_limits<double>::quiet_NaN()
                                : -std::numeric_limits<double>::infinity();
                aggregateComplete = true;
            }
            continue;
        }
        double componentHeadroom = std::numeric_limits<double>::infinity();
        for (size_t index = 0; index < certificate.GapCount; ++index) {
            const auto &boundary = certificate.Gaps[index];
            if (!aggregateComplete && !RefOrbitErrorBound::Less(boundary.Comparison, boundary.Gap)) {
                result.Failure = RefOrbitPrecisionFailure::AmbiguousSample;
            }
            const auto status = !boundary.Comparison.IsFinite() ? RefOrbitErrorBudgetStatus::InvalidBound
                                : boundary.Gap.IsZero()         ? RefOrbitErrorBudgetStatus::ZeroGap
                                                                : RefOrbitErrorBudgetStatus::Bounded;
            const double headroom = status != RefOrbitErrorBudgetStatus::Bounded
                                        ? -std::numeric_limits<double>::infinity()
                                        : static_cast<double>(boundary.Gap.m_Value.getExp() -
                                                              boundary.Comparison.m_Value.getExp()) +
                                              std::log2(boundary.Gap.m_Value.getMantissa() /
                                                        boundary.Comparison.m_Value.getMantissa());
            if (headroom < componentHeadroom) {
                componentHeadroom = headroom;
                budget = DescribeBudget(
                    status, m_Stats.Steps + 1, boundary.Comparison, boundary.Gap, certificate.Exponent);
            }
            if (!aggregateComplete && headroom < result.HeadroomBits) {
                result.Component = componentName;
                result.Boundary = index == 0 ? "lower_magnitude" : "upper_magnitude";
                result.GapRoundingBoundLog2 = boundary.Rounding.Log2();
                result.ComparisonErrorBoundLog2 = boundary.Comparison.Log2();
                result.GapLowerBoundLog2 = boundary.Gap.Log2();
                result.HeadroomBits = headroom;
            }
        }
    }
    return result;
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
RefOrbitPrecisionSchedule::AddError(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination)
{
    return QuantumBound(SumQuantum<false>(left, right, destination));
}

RefOrbitErrorBound
RefOrbitPrecisionSchedule::SubtractError(mpf_srcptr left, mpf_srcptr right, mpf_srcptr destination)
{
    return QuantumBound(SumQuantum<true>(left, right, destination));
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
