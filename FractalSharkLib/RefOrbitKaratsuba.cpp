#include "stdafx.h"

#include "Environment.h"
#include "RefOrbitKaratsuba.h"
#include "ScopedMpir.h"

#include <algorithm>
#include <cassert>
#include <immintrin.h>

namespace FractalShark {

namespace {

// The caller owns installation of MPIR's allocator hooks. Workers only initialize their
// private arena; every task's MPIR temporaries disappear before this guard shuts TLS down.
class WorkerMpirTls {
public:
    WorkerMpirTls() { MPIRBoundedAllocator::InitTls(); }
    ~WorkerMpirTls() { MPIRBoundedAllocator::ShutdownTls(); }
};

class MpfScratch {
public:
    explicit MpfScratch(mp_bitcnt_t precisionBits) { mpf_init2(m_Value, precisionBits); }
    ~MpfScratch() { mpf_clear(m_Value); }
    mpf_ptr
    Get()
    {
        return m_Value;
    }

private:
    mpf_t m_Value;
};

mp_size_t
MagnitudeSize(mpf_srcptr value)
{
    const mp_size_t size = value->_mp_size;
    return size < 0 ? -size : size;
}

} // namespace

// During constructor unwinding, publish Stop before the worker array's destructors
// join any threads that started successfully. No batch has been dispatched yet.
template <bool DynamicPrecision> struct RefOrbitKaratsubaProductsT<DynamicPrecision>::StartupGuard {
    RefOrbitKaratsubaProductsT<DynamicPrecision> &Owner;
    bool Active{true};
    ~StartupGuard()
    {
        if (Active) {
            Owner.StopWorkers();
        }
    }
};

template <bool DynamicPrecision>
RefOrbitKaratsubaProductsT<DynamicPrecision>::RefOrbitKaratsubaProductsT(mp_bitcnt_t precisionBits)
{
    static_assert(GMP_NAIL_BITS == 0, "Reference Karatsuba requires nail-free limbs");
    // Ask MPIR for the actual mantissa capacity, including its guard limb. Both
    // split parts must be nonempty even at minimum precision. Fixed padding also
    // accepts zero and one-limb values without changing their MPIR representation.
    MpfScratch layout(precisionBits);
    const mp_size_t width =
        std::max(MinimumSplitWidth, static_cast<mp_size_t>(layout.Get()->_mp_prec) + 1);
    m_LowWidth = width / 2;
    m_HighWidth = width - m_LowWidth;
    for (auto *operand : {&m_X, &m_Y}) {
        operand->High.resize(static_cast<size_t>(m_HighWidth));
        operand->Low.resize(static_cast<size_t>(m_LowWidth));
        operand->Sum.resize(static_cast<size_t>(m_HighWidth + 1));
    }
    for (auto &product : m_Products) {
        product.High.resize(static_cast<size_t>(2 * m_HighWidth));
        product.Low.resize(static_cast<size_t>(2 * m_LowWidth));
        product.Sum.resize(static_cast<size_t>(2 * (m_HighWidth + 1)));
        // Two extra limbs absorb padded sum carries during assembly. The exact
        // product fits in 2*width limbs; these extras are subsequently zero.
        product.Full.resize(static_cast<size_t>(2 * width + 2));
    }
    m_CallerJob = BindJob<ProductKind::Cross, PartKind::High>();

    // If thread creation fails, stop already-created workers before member teardown
    // joins them. The states and arithmetic buffers outlive the thread array.
    StartupGuard startup{*this};
    StartWorker<0, ProductKind::XSquare, PartKind::High>();
    StartWorker<1, ProductKind::XSquare, PartKind::Low>();
    StartWorker<2, ProductKind::XSquare, PartKind::Sum>();
    StartWorker<3, ProductKind::YSquare, PartKind::High>();
    StartWorker<4, ProductKind::YSquare, PartKind::Low>();
    StartWorker<5, ProductKind::YSquare, PartKind::Sum>();
    StartWorker<6, ProductKind::Cross, PartKind::Low>();
    StartWorker<7, ProductKind::Cross, PartKind::Sum>();
    startup.Active = false;
}

template <bool DynamicPrecision>
RefOrbitKaratsubaProductsT<DynamicPrecision>::~RefOrbitKaratsubaProductsT()
{
    StopWorkers();
    for (auto &worker : m_Workers) {
        worker.join();
    }
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Compute(
    mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_srcptr x, mpf_srcptr y)
{
    if constexpr (DynamicPrecision) {
        // The previous Compute joined every batch. Publishing Run below releases these
        // active lengths along with the snapshots, while buffer addresses stay fixed.
        assert(xSquared->_mp_prec == ySquared->_mp_prec && xSquared->_mp_prec == cross->_mp_prec);
        const mp_size_t width =
            std::max(MinimumSplitWidth, static_cast<mp_size_t>(xSquared->_mp_prec) + 1);
        m_LowWidth = width / 2;
        m_HighWidth = width - m_LowWidth;
        assert(m_HighWidth <= static_cast<mp_size_t>(m_X.High.size()));
        m_CallerJob.Width = m_HighWidth;
    }
    Prepare(m_X, x);
    Prepare(m_Y, y);

    // Publish followers first. A lead acquiring Run therefore cannot mistake a
    // follower's previous Idle state for completion of the new multiplication.
    // Each worker has exactly one outstanding job; no generation counter is needed.
    Dispatch<1>();
    Dispatch<2>();
    Dispatch<4>();
    Dispatch<5>();
    Dispatch<6>();
    Dispatch<7>();
    Dispatch<0>();
    Dispatch<3>();

    Multiply<ProductKind::Cross>(m_CallerJob);
    WaitIdle<6>();
    WaitIdle<7>();
    Assemble(m_Products[2]);
    WaitIdle<0>();
    WaitIdle<3>();
    // All input snapshots remain intact until every product is assembled, so outputs
    // may alias inputs. A lead's release of Idle publishes its complete integer product.
    Store(xSquared, m_Products[0], m_X, m_X);
    Store(ySquared, m_Products[1], m_Y, m_Y);
    Store(cross, m_Products[2], m_X, m_Y);
}

template <bool DynamicPrecision>
template <RefOrbitKaratsubaProductsT<DynamicPrecision>::PartKind part, class Parts>
auto &
RefOrbitKaratsubaProductsT<DynamicPrecision>::SelectPart(Parts &parts)
{
    if constexpr (part == PartKind::High) {
        return parts.High;
    } else if constexpr (part == PartKind::Low) {
        return parts.Low;
    } else {
        return parts.Sum;
    }
}

template <bool DynamicPrecision>
template <RefOrbitKaratsubaProductsT<DynamicPrecision>::ProductKind kind, bool right>
typename RefOrbitKaratsubaProductsT<DynamicPrecision>::Operand &
RefOrbitKaratsubaProductsT<DynamicPrecision>::SelectOperand()
{
    if constexpr (kind == ProductKind::YSquare || (kind == ProductKind::Cross && right)) {
        return m_Y;
    } else {
        return m_X;
    }
}

template <bool DynamicPrecision>
template <RefOrbitKaratsubaProductsT<DynamicPrecision>::ProductKind kind,
          RefOrbitKaratsubaProductsT<DynamicPrecision>::PartKind part>
typename RefOrbitKaratsubaProductsT<DynamicPrecision>::ArithmeticJob
RefOrbitKaratsubaProductsT<DynamicPrecision>::BindJob()
{
    auto &left = SelectPart<part>(SelectOperand<kind, false>());
    auto &right = SelectPart<part>(SelectOperand<kind, true>());
    auto &result = SelectPart<part>(m_Products[static_cast<size_t>(kind)]);
    return {result.data(), left.data(), right.data(), static_cast<mp_size_t>(left.size())};
}

template <bool DynamicPrecision>
template <RefOrbitKaratsubaProductsT<DynamicPrecision>::ProductKind kind>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Multiply(const ArithmeticJob &job) noexcept
{
    // Fixed square/multiply roles remove all task decoding and operand selection
    // from the hot loop. Cross-product operands always have the same padded width.
    if constexpr (kind == ProductKind::Cross) {
        mpn_mul_n(job.Result, job.Left, job.Right, job.Width);
    } else {
        mpn_sqr(job.Result, job.Left, job.Width);
    }
}

template <bool DynamicPrecision>
template <std::size_t index>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Dispatch() noexcept
{
    m_Slots[index].State.store(WorkState::Run, std::memory_order_release);
}

template <bool DynamicPrecision>
template <std::size_t index>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::WaitIdle() noexcept
{
    while (m_Slots[index].State.load(std::memory_order_acquire) != WorkState::Idle) {
        _mm_pause();
    }
}

template <bool DynamicPrecision>
template <std::size_t index,
          RefOrbitKaratsubaProductsT<DynamicPrecision>::ProductKind kind,
          RefOrbitKaratsubaProductsT<DynamicPrecision>::PartKind part>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::WorkerLoop() noexcept
{
    WorkerMpirTls tls;
    // Roles and pointers are bound once because backing vectors never resize.
    // Fixed workers also bind lengths once; dynamic workers acquire active lengths
    // from the caller's release of Run for each completed batch.
    constexpr std::array<const wchar_t *, WorkerCount> names{L"MT9 X high",
                                                             L"MT9 X low",
                                                             L"MT9 X sum",
                                                             L"MT9 Y high",
                                                             L"MT9 Y low",
                                                             L"MT9 Y sum",
                                                             L"MT9 XY low",
                                                             L"MT9 XY sum"};
    Environment::SetCurrentThreadName(names[index]);
    ArithmeticJob job = BindJob<kind, part>();
    auto &state = m_Slots[index].State;
    for (;;) {
        WorkState command = state.load(std::memory_order_acquire);
        while (command == WorkState::Idle) {
            _mm_pause();
            command = state.load(std::memory_order_acquire);
        }
        if (command == WorkState::Stop) {
            return;
        }
        if constexpr (DynamicPrecision) {
            if constexpr (part == PartKind::Low) {
                job.Width = m_LowWidth;
            } else if constexpr (part == PartKind::High) {
                job.Width = m_HighWidth;
            } else {
                job.Width = m_HighWidth + 1;
            }
        }
        Multiply<kind>(job);
        if constexpr (part == PartKind::High) {
            // Only square high-part workers are leads; the caller leads XY. Waiting
            // on each product separately lets assembly overlap the other products.
            WaitIdle<index + 1>();
            WaitIdle<index + 2>();
            Assemble(m_Products[static_cast<size_t>(kind)]);
        }
        state.store(WorkState::Idle, std::memory_order_release);
    }
}

template <bool DynamicPrecision>
template <std::size_t index,
          RefOrbitKaratsubaProductsT<DynamicPrecision>::ProductKind kind,
          RefOrbitKaratsubaProductsT<DynamicPrecision>::PartKind part>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::StartWorker()
{
    // A worker failure is unrecoverable, matching MT3. Arithmetic has no per-job
    // exception bookkeeping; allocations of owned buffers happen before startup.
    // MPIR may still use its own per-thread allocator for internal arithmetic scratch.
    m_Workers[index] = std::jthread([this]() noexcept { WorkerLoop<index, kind, part>(); });
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::StopWorkers() noexcept
{
    // Called only between complete batches, or before any batch during constructor
    // unwinding. No worker can still be waiting for a stopped follower in either case.
    for (auto &slot : m_Slots) {
        slot.State.store(WorkState::Stop, std::memory_order_release);
    }
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Prepare(Operand &output, mpf_srcptr input)
{
    output.Size = MagnitudeSize(input);
    assert(output.Size <= m_LowWidth + m_HighWidth);
    output.Exponent = input->_mp_exp;
    output.Negative = input->_mp_size < 0;
    // Copy the stored magnitude in bulk and zero only unused padding. Exponent
    // conversion uses the actual source lengths rather than the padded split width.
    const mp_size_t lowSize = std::min(output.Size, m_LowWidth);
    const mp_size_t highSize = output.Size - lowSize;
    std::copy_n(input->_mp_d, lowSize, output.Low.data());
    std::fill(output.Low.begin() + lowSize, output.Low.begin() + m_LowWidth, 0);
    if (highSize != 0) {
        std::copy_n(input->_mp_d + lowSize, highSize, output.High.data());
    }
    std::fill(output.High.begin() + highSize, output.High.begin() + m_HighWidth, 0);
    const mp_limb_t carry =
        mpn_add(output.Sum.data(), output.High.data(), m_HighWidth, output.Low.data(), m_LowWidth);
    output.Sum[static_cast<size_t>(m_HighWidth)] = carry;
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::AddShifted(std::vector<mp_limb_t> &destination,
                                                         const std::vector<mp_limb_t> &source,
                                                         mp_size_t shift)
{
    const mp_size_t fullSize = DynamicPrecision ? 2 * (m_LowWidth + m_HighWidth) + 2
                                                : static_cast<mp_size_t>(destination.size());
    const mp_size_t sourceSize =
        DynamicPrecision ? 2 * (m_HighWidth + 1) : static_cast<mp_size_t>(source.size());
    const auto remaining = fullSize - shift;
    const mp_limb_t carry = mpn_add(
        destination.data() + shift, destination.data() + shift, remaining, source.data(), sourceSize);
    assert(carry == 0);
    (void)carry;
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Assemble(Product &product)
{
    // With S=beta^m, C-A-B = XH*YL + XL*YH is nonnegative. The subtraction and
    // shifted additions are exact integers, with no truncated intermediate limbs.
    const mp_size_t sumSize =
        DynamicPrecision ? 2 * (m_HighWidth + 1) : static_cast<mp_size_t>(product.Sum.size());
    const mp_size_t highSize =
        DynamicPrecision ? 2 * m_HighWidth : static_cast<mp_size_t>(product.High.size());
    const mp_size_t lowSize =
        DynamicPrecision ? 2 * m_LowWidth : static_cast<mp_size_t>(product.Low.size());
    const mp_limb_t firstBorrow =
        mpn_sub(product.Sum.data(), product.Sum.data(), sumSize, product.High.data(), highSize);
    const mp_limb_t secondBorrow =
        mpn_sub(product.Sum.data(), product.Sum.data(), sumSize, product.Low.data(), lowSize);
    assert(firstBorrow == 0 && secondBorrow == 0);
    (void)firstBorrow;
    (void)secondBorrow;
    // Low and shifted High occupy adjacent, nonoverlapping ranges. Initialize them
    // by copying rather than clearing the full product and performing two additions.
    // Only the two guard limbs need clearing before the single middle-term addition.
    std::copy_n(product.Low.begin(), lowSize, product.Full.begin());
    std::copy_n(product.High.begin(), highSize, product.Full.begin() + 2 * m_LowWidth);
    const auto fullSize = static_cast<size_t>(2 * (m_LowWidth + m_HighWidth) + 2);
    product.Full[fullSize - 2] = 0;
    product.Full[fullSize - 1] = 0;
    AddShifted(product.Full, product.Sum, m_LowWidth);
}

template <bool DynamicPrecision>
void
RefOrbitKaratsubaProductsT<DynamicPrecision>::Store(mpf_ptr destination,
                                                    const Product &product,
                                                    const Operand &left,
                                                    const Operand &right)
{
    mp_size_t size = DynamicPrecision ? 2 * (m_LowWidth + m_HighWidth) + 2
                                      : static_cast<mp_size_t>(product.Full.size());
    while (size > 0 && product.Full[static_cast<size_t>(size - 1)] == 0) {
        --size;
    }
    const mp_size_t retained = std::min(size, static_cast<mp_size_t>(destination->_mp_prec) + 1);
    const mp_size_t offset = size - retained;
    const auto adjustment = static_cast<mp_exp_t>(left.Size + right.Size - size);
    std::copy_n(product.Full.data() + offset, retained, destination->_mp_d);
    // The existing normalization scan also handles an all-zero product. An empty
    // copy leaves unused limbs alone; zero's canonical exponent is independent of
    // the nonzero operand's exponent and occupied width.
    destination->_mp_exp = retained == 0 ? 0 : left.Exponent + right.Exponent - adjustment;
    destination->_mp_size = static_cast<decltype(destination->_mp_size)>(
        left.Negative != right.Negative ? -retained : retained);
    // For exact nonzero XY with normalized limb exponent e, discarding low limbs
    // toward zero gives |stored-XY| < beta^(e-retained). Karatsuba adds no error.
    // This is a local multiplication bound. The fixed orbit keeps full precision;
    // the dynamic orbit separately propagates every arithmetic/truncation error.
    // Neither specialization asserts equality with MPIR's clipped-input mpf_mul.
}

template class RefOrbitKaratsubaProductsT<false>;
template class RefOrbitKaratsubaProductsT<true>;

} // namespace FractalShark
