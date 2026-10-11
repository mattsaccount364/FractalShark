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
struct RefOrbitKaratsubaProducts::StartupGuard {
    RefOrbitKaratsubaProducts &Owner;
    bool Active{true};
    ~StartupGuard()
    {
        if (Active) {
            Owner.StopWorkers();
        }
    }
};

RefOrbitKaratsubaProducts::RefOrbitKaratsubaProducts(mp_bitcnt_t precisionBits)
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

RefOrbitKaratsubaProducts::~RefOrbitKaratsubaProducts()
{
    StopWorkers();
    for (auto &worker : m_Workers) {
        worker.join();
    }
}

void
RefOrbitKaratsubaProducts::Compute(
    mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_srcptr x, mpf_srcptr y)
{
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

template <RefOrbitKaratsubaProducts::PartKind part, class Parts>
auto &
RefOrbitKaratsubaProducts::SelectPart(Parts &parts)
{
    if constexpr (part == PartKind::High) {
        return parts.High;
    } else if constexpr (part == PartKind::Low) {
        return parts.Low;
    } else {
        return parts.Sum;
    }
}

template <RefOrbitKaratsubaProducts::ProductKind kind, bool right>
RefOrbitKaratsubaProducts::Operand &
RefOrbitKaratsubaProducts::SelectOperand()
{
    if constexpr (kind == ProductKind::YSquare || (kind == ProductKind::Cross && right)) {
        return m_Y;
    } else {
        return m_X;
    }
}

template <RefOrbitKaratsubaProducts::ProductKind kind, RefOrbitKaratsubaProducts::PartKind part>
RefOrbitKaratsubaProducts::ArithmeticJob
RefOrbitKaratsubaProducts::BindJob()
{
    auto &left = SelectPart<part>(SelectOperand<kind, false>());
    auto &right = SelectPart<part>(SelectOperand<kind, true>());
    auto &result = SelectPart<part>(m_Products[static_cast<size_t>(kind)]);
    return {result.data(), left.data(), right.data(), static_cast<mp_size_t>(left.size())};
}

template <RefOrbitKaratsubaProducts::ProductKind kind>
void
RefOrbitKaratsubaProducts::Multiply(const ArithmeticJob &job) noexcept
{
    // Fixed square/multiply roles remove all task decoding and operand selection
    // from the hot loop. Cross-product operands always have the same padded width.
    if constexpr (kind == ProductKind::Cross) {
        mpn_mul_n(job.Result, job.Left, job.Right, job.Width);
    } else {
        mpn_sqr(job.Result, job.Left, job.Width);
    }
}

template <std::size_t index>
void
RefOrbitKaratsubaProducts::Dispatch() noexcept
{
    m_Slots[index].State.store(WorkState::Run, std::memory_order_release);
}

template <std::size_t index>
void
RefOrbitKaratsubaProducts::WaitIdle() noexcept
{
    while (m_Slots[index].State.load(std::memory_order_acquire) != WorkState::Idle) {
        _mm_pause();
    }
}

template <std::size_t index,
          RefOrbitKaratsubaProducts::ProductKind kind,
          RefOrbitKaratsubaProducts::PartKind part>
void
RefOrbitKaratsubaProducts::WorkerLoop() noexcept
{
    WorkerMpirTls tls;
    // Roles are named once at startup for CPU traces. All pointers and lengths are
    // bound once because the backing vectors never resize after thread creation.
    constexpr std::array<const wchar_t *, WorkerCount> names{L"MT9 X high",
                                                             L"MT9 X low",
                                                             L"MT9 X sum",
                                                             L"MT9 Y high",
                                                             L"MT9 Y low",
                                                             L"MT9 Y sum",
                                                             L"MT9 XY low",
                                                             L"MT9 XY sum"};
    Environment::SetCurrentThreadName(names[index]);
    const ArithmeticJob job = BindJob<kind, part>();
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

template <std::size_t index,
          RefOrbitKaratsubaProducts::ProductKind kind,
          RefOrbitKaratsubaProducts::PartKind part>
void
RefOrbitKaratsubaProducts::StartWorker()
{
    // A worker failure is unrecoverable, matching MT3. Arithmetic has no per-job
    // exception bookkeeping; allocations of owned buffers happen before startup.
    // MPIR may still use its own per-thread allocator for internal arithmetic scratch.
    m_Workers[index] = std::jthread([this]() noexcept { WorkerLoop<index, kind, part>(); });
}

void
RefOrbitKaratsubaProducts::StopWorkers() noexcept
{
    // Called only between complete batches, or before any batch during constructor
    // unwinding. No worker can still be waiting for a stopped follower in either case.
    for (auto &slot : m_Slots) {
        slot.State.store(WorkState::Stop, std::memory_order_release);
    }
}

void
RefOrbitKaratsubaProducts::Prepare(Operand &output, mpf_srcptr input)
{
    output.Size = MagnitudeSize(input);
    output.Exponent = input->_mp_exp;
    output.Negative = input->_mp_size < 0;
    // Copy the stored magnitude in bulk and zero only unused padding. Exponent
    // conversion uses the actual source lengths rather than the padded split width.
    const mp_size_t lowSize = std::min(output.Size, m_LowWidth);
    const mp_size_t highSize = output.Size - lowSize;
    std::copy_n(input->_mp_d, lowSize, output.Low.data());
    std::fill(output.Low.begin() + lowSize, output.Low.end(), 0);
    if (highSize != 0) {
        std::copy_n(input->_mp_d + lowSize, highSize, output.High.data());
    }
    std::fill(output.High.begin() + highSize, output.High.end(), 0);
    const mp_limb_t carry =
        mpn_add(output.Sum.data(), output.High.data(), m_HighWidth, output.Low.data(), m_LowWidth);
    output.Sum[static_cast<size_t>(m_HighWidth)] = carry;
}

void
RefOrbitKaratsubaProducts::AddShifted(std::vector<mp_limb_t> &destination,
                                      const std::vector<mp_limb_t> &source,
                                      mp_size_t shift)
{
    const auto remaining = static_cast<mp_size_t>(destination.size()) - shift;
    const mp_limb_t carry = mpn_add(destination.data() + shift,
                                    destination.data() + shift,
                                    remaining,
                                    source.data(),
                                    static_cast<mp_size_t>(source.size()));
    assert(carry == 0);
    (void)carry;
}

void
RefOrbitKaratsubaProducts::Assemble(Product &product)
{
    // With S=beta^m, C-A-B = XH*YL + XL*YH is nonnegative. The subtraction and
    // shifted additions are exact integers, with no truncated intermediate limbs.
    const auto sumSize = static_cast<mp_size_t>(product.Sum.size());
    const mp_limb_t firstBorrow = mpn_sub(product.Sum.data(),
                                          product.Sum.data(),
                                          sumSize,
                                          product.High.data(),
                                          static_cast<mp_size_t>(product.High.size()));
    const mp_limb_t secondBorrow = mpn_sub(product.Sum.data(),
                                           product.Sum.data(),
                                           sumSize,
                                           product.Low.data(),
                                           static_cast<mp_size_t>(product.Low.size()));
    assert(firstBorrow == 0 && secondBorrow == 0);
    (void)firstBorrow;
    (void)secondBorrow;
    // Low and shifted High occupy adjacent, nonoverlapping ranges. Initialize them
    // by copying rather than clearing the full product and performing two additions.
    // Only the two guard limbs need clearing before the single middle-term addition.
    std::copy(product.Low.begin(), product.Low.end(), product.Full.begin());
    std::copy(product.High.begin(), product.High.end(), product.Full.begin() + 2 * m_LowWidth);
    product.Full[product.Full.size() - 2] = 0;
    product.Full.back() = 0;
    AddShifted(product.Full, product.Sum, m_LowWidth);
}

void
RefOrbitKaratsubaProducts::Store(mpf_ptr destination,
                                 const Product &product,
                                 const Operand &left,
                                 const Operand &right)
{
    auto size = static_cast<mp_size_t>(product.Full.size());
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
    // This bound is local to multiplication; orbit precision and MPIR add/sub remain
    // full width. It does not assert equality with MPIR's clipped-input mpf_mul.
}

} // namespace FractalShark
