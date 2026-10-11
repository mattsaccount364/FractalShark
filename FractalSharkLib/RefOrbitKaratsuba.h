#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

// Only the raw arithmetic types are needed here; HighPrecision and platform services
// would introduce unrelated dependencies into every user of this class.
#ifdef _MSC_VER
#include <mpir.h>
#else
#include <gmp.h>
#endif

namespace FractalShark {

// This threshold selects the backend in Auto mode; explicit selection is unrestricted.
inline constexpr uint64_t RefOrbitKaratsubaAutoPrecisionBits = 16'384;

// Computes three products at full destination precision. Outputs must be distinct, but may
// alias either input. Full stored input mantissas, including guard limbs, are snapshotted.
// The fixed precision applies to all outputs and bounds input sizes. Zero and one-limb
// inputs use the same padded split buffers. All scratch is sized once at construction and reused.
class RefOrbitKaratsubaProducts {
public:
    explicit RefOrbitKaratsubaProducts(mp_bitcnt_t precisionBits);
    ~RefOrbitKaratsubaProducts();

    // Workers capture this object and bind pointers into its fixed scratch buffers.
    // Its address and ownership must remain stable until every worker has joined.
    RefOrbitKaratsubaProducts(const RefOrbitKaratsubaProducts &) = delete;
    RefOrbitKaratsubaProducts &operator=(const RefOrbitKaratsubaProducts &) = delete;
    RefOrbitKaratsubaProducts(RefOrbitKaratsubaProducts &&) = delete;
    RefOrbitKaratsubaProducts &operator=(RefOrbitKaratsubaProducts &&) = delete;

    void Compute(mpf_ptr xSquared, mpf_ptr ySquared, mpf_ptr cross, mpf_srcptr x, mpf_srcptr y);

private:
    enum class ProductKind { XSquare, YSquare, Cross };
    enum class PartKind { High, Low, Sum };
    enum class WorkState : uint8_t { Idle, Run, Stop };
    static constexpr std::size_t WorkerCount = 8;
    static constexpr std::size_t CacheLineBytes = 64;
    static constexpr mp_size_t MinimumSplitWidth = 2;

    // Keep independent handoffs on separate cache lines. Only the caller and the owner
    // write a state; there is no shared completion counter or read-modify-write operation.
    struct alignas(CacheLineBytes) Slot {
        std::atomic<WorkState> State{WorkState::Idle};
        std::array<uint8_t, CacheLineBytes - sizeof(std::atomic<WorkState>)> Padding{};
    };
    struct ArithmeticJob {
        mp_ptr Result{};
        mp_srcptr Left{};
        mp_srcptr Right{};
        mp_size_t Width{};
    };
    struct StartupGuard;

    struct Operand {
        std::vector<mp_limb_t> High;
        std::vector<mp_limb_t> Low;
        std::vector<mp_limb_t> Sum;
        mp_size_t Size{};
        mp_exp_t Exponent{};
        bool Negative{};
    };
    struct Product {
        std::vector<mp_limb_t> High;
        std::vector<mp_limb_t> Low;
        std::vector<mp_limb_t> Sum;
        std::vector<mp_limb_t> Full;
    };

    // These templates are used only by this class's implementation. Keep their bodies
    // in the .cpp while retaining compile-time worker roles in the arithmetic hot path.
    template <PartKind part, class Parts> static auto &SelectPart(Parts &parts);
    template <ProductKind kind, bool right> Operand &SelectOperand();
    template <ProductKind kind, PartKind part> ArithmeticJob BindJob();
    template <ProductKind kind> static void Multiply(const ArithmeticJob &job) noexcept;
    template <std::size_t index> void Dispatch() noexcept;
    template <std::size_t index> void WaitIdle() noexcept;
    template <std::size_t index, ProductKind kind, PartKind part> void WorkerLoop() noexcept;
    template <std::size_t index, ProductKind kind, PartKind part> void StartWorker();

    void StopWorkers() noexcept;
    void Prepare(Operand &output, mpf_srcptr input);
    static void AddShifted(std::vector<mp_limb_t> &destination,
                           const std::vector<mp_limb_t> &source,
                           mp_size_t shift);
    void Assemble(Product &product);
    static void Store(mpf_ptr destination,
                      const Product &product,
                      const Operand &left,
                      const Operand &right);

    mp_size_t m_LowWidth{};
    mp_size_t m_HighWidth{};
    Operand m_X;
    Operand m_Y;
    std::array<Product, 3> m_Products;
    ArithmeticJob m_CallerJob;
    std::array<Slot, WorkerCount> m_Slots;
    // Declared last: joins during constructor unwinding finish before buffers and states
    // are destroyed. StartupGuard publishes Stop before that member teardown begins.
    std::array<std::jthread, WorkerCount> m_Workers;
};

} // namespace FractalShark
