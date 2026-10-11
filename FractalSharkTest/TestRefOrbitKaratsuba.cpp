#include "TestFramework.h"

#include "AbortMonitor.h"
#include "AlgCmds.h"
#include "Fractal.h"
#include "MenuState.h"
#include "MenuTree.h"
#include "PerturbationResults.h"
#include "PortableCommandHandlers.h"
#include "RefOrbitKaratsuba.h"
#include "RenderTestCatalog.h"
#include "RenderTestSupport.h"
#include "RenderThreadPool.h"
#include "ScopedMpir.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

class MpfValue {
public:
    explicit MpfValue(mp_bitcnt_t precisionBits) { mpf_init2(m_Value, precisionBits); }
    ~MpfValue() { mpf_clear(m_Value); }
    mpf_ptr
    Get()
    {
        return m_Value;
    }

private:
    mpf_t m_Value;
};

class MpzValue {
public:
    MpzValue() { mpz_init(m_Value); }
    ~MpzValue() { mpz_clear(m_Value); }
    mpz_ptr
    Get()
    {
        return m_Value;
    }

private:
    mpz_t m_Value;
};

mp_size_t
MagnitudeSize(mpf_srcptr value)
{
    const mp_size_t size = value->_mp_size;
    return size < 0 ? -size : size;
}

void
CheckSame(mpf_srcptr actual, mpf_srcptr expected)
{
    ASSERT_EQ(actual->_mp_prec, expected->_mp_prec);
    ASSERT_EQ(actual->_mp_size, expected->_mp_size);
    ASSERT_EQ(actual->_mp_exp, expected->_mp_exp);
    ASSERT_TRUE(std::equal(actual->_mp_d, actual->_mp_d + MagnitudeSize(actual), expected->_mp_d));
}

// Independent oracle: mpz_mul consumes every stored limb and computes the complete
// integer product. It uses neither the split-product helper nor clipped mpf_mul inputs.
void
Oracle(mpf_ptr destination, mpf_srcptr x, mpf_srcptr y)
{
    const mp_size_t xSize = MagnitudeSize(x);
    const mp_size_t ySize = MagnitudeSize(y);
    if (xSize == 0 || ySize == 0) {
        mpf_set_ui(destination, 0);
        return;
    }
    MpzValue left, right, full, remainder, bound;
    mpz_import(
        left.Get(), static_cast<size_t>(xSize), -1, sizeof(mp_limb_t), 0, GMP_NAIL_BITS, x->_mp_d);
    mpz_import(
        right.Get(), static_cast<size_t>(ySize), -1, sizeof(mp_limb_t), 0, GMP_NAIL_BITS, y->_mp_d);
    mpz_mul(full.Get(), left.Get(), right.Get());
    std::vector<mp_limb_t> limbs(static_cast<size_t>(xSize + ySize));
    size_t count = 0;
    mpz_export(limbs.data(), &count, -1, sizeof(mp_limb_t), 0, GMP_NAIL_BITS, full.Get());
    const auto size = static_cast<mp_size_t>(count);
    const mp_size_t retained = std::min(size, static_cast<mp_size_t>(destination->_mp_prec) + 1);
    const mp_size_t discarded = size - retained;
    const mp_bitcnt_t discardedBits = static_cast<mp_bitcnt_t>(discarded) * GMP_NUMB_BITS;
    mpz_fdiv_r_2exp(remainder.Get(), full.Get(), discardedBits);
    mpz_set_ui(bound.Get(), 1);
    mpz_mul_2exp(bound.Get(), bound.Get(), discardedBits);
    ASSERT_TRUE(mpz_cmp(remainder.Get(), bound.Get()) < 0);
    const mp_exp_t exponent = x->_mp_exp + y->_mp_exp - static_cast<mp_exp_t>(xSize + ySize - size);
    const bool negative = (x->_mp_size < 0) != (y->_mp_size < 0);
    std::copy_n(limbs.data() + discarded, retained, destination->_mp_d);
    destination->_mp_size =
        static_cast<decltype(destination->_mp_size)>(negative ? -retained : retained);
    destination->_mp_exp = exponent;
}

void
Randomize(mpf_ptr value, mp_size_t size, bool negative, std::mt19937_64 &random)
{
    ASSERT_TRUE(size <= value->_mp_prec + 1);
    for (mp_size_t limb = 0; limb < size; ++limb) {
        value->_mp_d[limb] = static_cast<mp_limb_t>(random()) & GMP_NUMB_MAX;
    }
    // A nonzero most significant limb is the MPIR normalized-magnitude invariant.
    value->_mp_d[size - 1] |= 1;
    value->_mp_size = static_cast<decltype(value->_mp_size)>(negative ? -size : size);
    value->_mp_exp = static_cast<mp_exp_t>(random() % 35) - 17;
}

void
CheckBatch(FractalShark::RefOrbitKaratsubaProducts &products,
           mpf_ptr x,
           mpf_ptr y,
           mpf_ptr xSquared,
           mpf_ptr ySquared,
           mpf_ptr cross)
{
    const mp_bitcnt_t precisionBits = mpf_get_prec(xSquared);
    MpfValue expectedX(precisionBits), expectedY(precisionBits), expectedCross(precisionBits);
    Oracle(expectedX.Get(), x, x);
    Oracle(expectedY.Get(), y, y);
    Oracle(expectedCross.Get(), x, y);
    products.Compute(xSquared, ySquared, cross, x, y);
    CheckSame(xSquared, expectedX.Get());
    CheckSame(ySquared, expectedY.Get());
    CheckSame(cross, expectedCross.Get());
}

PointZoomBBConverter
ViewAtPrecision(uint64_t requestedBits)
{
    MPIRPrecision precision(requestedBits + 256);
    HighPrecision radius{1};
    mpf_div_2exp(
        radius.backend(), radius.backend(), requestedBits - AuthoritativeMinExtraPrecisionInBits);
    PointZoomBBConverter view(-radius, -radius, radius, radius, PointZoomBBConverter::TestMode::Enabled);
    const auto measured = Fractal::GetPrecision(view, false);
    // Correct for the calculator's exponent convention using exact power-of-two scaling.
    if (measured > requestedBits) {
        mpf_mul_2exp(radius.backend(), radius.backend(), measured - requestedBits);
    } else if (measured < requestedBits) {
        mpf_div_2exp(radius.backend(), radius.backend(), requestedBits - measured);
    }
    PointZoomBBConverter result(
        -radius, -radius, radius, radius, PointZoomBBConverter::TestMode::Enabled);
    ASSERT_EQ(Fractal::GetPrecision(result, false), requestedBits);
    return result;
}

class CommandFixture : public FractalShark::PortableCommandHandlers {
public:
    explicit CommandFixture(Fractal &fractal) : m_Fractal(fractal) {}

protected:
    Fractal &
    GetFractal() noexcept override
    {
        return m_Fractal;
    }
    FractalShark::MenuPoint
    GetMenuMousePos() const override
    {
        return {0, 0};
    }
    void
    OnAutoZoomFeatureWithCheckpointSave() override
    {
    }
    void
    OnShowHotkeys() override
    {
    }
    void
    OnViewsHelp() override
    {
    }
    void
    OnHelpAlg() override
    {
    }
    void
    OnWindowed() override
    {
    }
    void
    OnWindowedSq() override
    {
    }
    void
    OnMinimize() override
    {
    }
    void
    OnCurPos() override
    {
    }
    void
    OnExit() override
    {
    }
    void
    OnSaveLocation() override
    {
    }
    void
    OnSaveHiResBmp() override
    {
    }
    void
    OnSaveItersText() override
    {
    }
    void
    OnSaveBmp() override
    {
    }
    void
    OnSaveRefOrbitText() override
    {
    }
    void
    OnSaveRefOrbitTextSimple() override
    {
    }
    void
    OnSaveRefOrbitTextMax() override
    {
    }
    void
    OnSaveRefOrbitImagMax() override
    {
    }
    void
    OnDiffRefOrbitImagMax() override
    {
    }
    void
    OnLoadLocation() override
    {
    }
    void
    OnLoadEnterLocation() override
    {
    }
    void
    OnLoadRefOrbitImagMax() override
    {
    }
    void
    OnLoadRefOrbitImagMaxSaved() override
    {
    }
    void
    OnLoadCustomPalette() override
    {
    }

private:
    Fractal &m_Fractal;
};

const FractalShark::Node *
FindNode(std::span<const FractalShark::Node> nodes, uint32_t id)
{
    for (const auto &node : nodes) {
        if (node.kind == FractalShark::Kind::Popup) {
            if (const auto *found = FindNode(node.kids, id)) {
                return found;
            }
        } else if (node.kind == FractalShark::Kind::Item && node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

std::vector<unsigned char>
SavePixels(Fractal &fractal, const std::filesystem::path &basename)
{
    fractal.GetRenderPool()->Drain();
    ASSERT_EQ(fractal.SaveCurrentFractal(basename.wstring(), false, SaveOverwritePolicy::SkipExisting),
              0);
    fractal.CleanupThreads(true);
    std::vector<unsigned char> pixels;
    unsigned width = 0, height = 0;
    const auto filename = basename.string() + ".png";
    ASSERT_EQ(lodepng::decode(pixels, width, height, filename, LCT_RGBA, 16), 0u);
    ASSERT_EQ(width, static_cast<unsigned>(fractal.GetRenderWidth()));
    ASSERT_EQ(height, static_cast<unsigned>(fractal.GetRenderHeight()));
    return pixels;
}

constexpr uint64_t ProfileDefaultIterations = 16'384;
constexpr uint64_t ProfileWarmupIterations = 32;

uint64_t
ProfileIterations()
{
    const char *value = std::getenv("FRACTALSHARK_REF_ORBIT_PROFILE_ITERATIONS");
    if (value == nullptr) {
        return ProfileDefaultIterations;
    }
    const auto *end = value + std::char_traits<char>::length(value);
    uint64_t iterations = 0;
    const auto parsed = std::from_chars(value, end, iterations);
    ASSERT_TRUE(parsed.ec == std::errc{} && parsed.ptr == end && iterations >= ProfileDefaultIterations);
    return iterations;
}

void
PrintProfile(std::string_view workload,
             mp_bitcnt_t precisionBits,
             uint64_t iterations,
             std::chrono::steady_clock::duration elapsed)
{
    const double nanoseconds = std::chrono::duration<double, std::nano>(elapsed).count();
    // One machine-readable row per fixture; no clock reads or diagnostics occur in
    // the multiplication loop. External CPU tracing supplies per-thread attribution.
    std::ostringstream row;
    row << std::fixed << std::setprecision(3) << "REF_ORBIT_PROFILE workload=" << workload
        << " bits=" << precisionBits << " iterations=" << iterations << " total_ms=" << nanoseconds / 1e6
        << " ns_per_iteration=" << nanoseconds / static_cast<double>(iterations);
    std::cout << row.str() << std::endl;
}

void
ProfileProducts(mp_bitcnt_t precisionBits)
{
    const uint64_t iterations = ProfileIterations();
    MpfValue x(precisionBits), y(precisionBits), xSquared(precisionBits), ySquared(precisionBits),
        cross(precisionBits);
    std::mt19937_64 random{0x9ca4a75};
    Randomize(x.Get(), x.Get()->_mp_prec + 1, false, random);
    Randomize(y.Get(), y.Get()->_mp_prec + 1, true, random);
    std::chrono::steady_clock::duration elapsed;
    {
        // Match production's MPIR allocator/TLS configuration. The public inputs and
        // outputs were allocated before installing these hooks; NewFree recognizes
        // their foreign ownership. Destroy the helper before shutting TLS down.
        MPIRBoundedAllocator allocator;
        allocator.InitScopedAllocators();
        struct CallerTls {
            CallerTls() { MPIRBoundedAllocator::InitTls(); }
            ~CallerTls() { MPIRBoundedAllocator::ShutdownTls(); }
        } callerTls;
        FractalShark::RefOrbitKaratsubaProducts products(precisionBits);
        for (uint64_t iteration = 0; iteration < ProfileWarmupIterations; ++iteration) {
            products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
        }
        const auto start = std::chrono::steady_clock::now();
        for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
            products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
        }
        elapsed = std::chrono::steady_clock::now() - start;
    }
    // The mpz oracle can resize its own allocations, so run it after restoring the
    // ordinary allocator, outside timing and outside the bounded allocator's TLS.
    MpfValue expectedX(precisionBits), expectedY(precisionBits), expectedCross(precisionBits);
    Oracle(expectedX.Get(), x.Get(), x.Get());
    Oracle(expectedY.Get(), y.Get(), y.Get());
    Oracle(expectedCross.Get(), x.Get(), y.Get());
    CheckSame(xSquared.Get(), expectedX.Get());
    CheckSame(ySquared.Get(), expectedY.Get());
    CheckSame(cross.Get(), expectedCross.Get());
    PrintProfile("ProductsMT9", precisionBits, iterations, elapsed);
}

void
ProfileOrbit(mp_bitcnt_t precisionBits, RefOrbitCalc::PerturbationAlg algorithm)
{
    const uint64_t iterations = ProfileIterations();
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(precisionBits);
    const HighPrecision cx{"-0.1"};
    const HighPrecision cy{"0.1"};
    const auto view = ViewAtPrecision(precisionBits).Recentered(cx, cy);
    ASSERT_TRUE(fractal.RecenterViewCalc(view));
    HighPrecision::defaultPrecisionInBits(precisionBits);
    fractal.SetNumIterations<uint64_t>(iterations);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(algorithm);
    calculator.ResetGuess(cx, cy);
    // This measures the real sequential recurrence, sample publication, derivative,
    // and worker startup. Benchmark mode prevents the bounded orbit terminating
    // early through periodicity; it does not batch independent products or render.
    const auto start = std::chrono::steady_clock::now();
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             HDRFloat<double>,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto *results = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
    ASSERT_EQ(results->GetCountOrbitEntries(), iterations);
    const auto *orbit = results->GetOrbitData();
    ASSERT_NEAR(static_cast<double>(orbit[1].x), -0.1, 1e-15);
    ASSERT_NEAR(static_cast<double>(orbit[1].y), 0.1, 1e-15);
    double expectedX = -0.1, expectedY = 0.1;
    for (size_t iteration = 0; iteration < 128; ++iteration) {
        const double nextX = expectedX * expectedX - expectedY * expectedY - 0.1;
        expectedY = 2 * expectedX * expectedY + 0.1;
        expectedX = nextX;
    }
    ASSERT_NEAR(static_cast<double>(orbit[iterations - 1].x), expectedX, 1e-15);
    ASSERT_NEAR(static_cast<double>(orbit[iterations - 1].y), expectedY, 1e-15);
    PrintProfile(algorithm == RefOrbitCalc::PerturbationAlg::MTPeriodicity3 ? "OrbitMT3" : "OrbitMT9",
                 precisionBits,
                 iterations,
                 elapsed);
}

} // namespace

TEST(RefOrbitKaratsuba_ProfileProducts_16384) { ProfileProducts(16'384); }
TEST(RefOrbitKaratsuba_ProfileProducts_33344) { ProfileProducts(33'344); }
TEST(RefOrbitKaratsuba_ProfileProducts_100096) { ProfileProducts(100'096); }
TEST(RefOrbitKaratsuba_ProfileOrbit_MT3_16384)
{
    ProfileOrbit(16'384, RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
}
TEST(RefOrbitKaratsuba_ProfileOrbit_MT3_33344)
{
    ProfileOrbit(33'344, RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
}
TEST(RefOrbitKaratsuba_ProfileOrbit_MT3_100096)
{
    ProfileOrbit(100'096, RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
}
TEST(RefOrbitKaratsuba_ProfileOrbit_MT9_16384)
{
    ProfileOrbit(16'384, RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
}
TEST(RefOrbitKaratsuba_ProfileOrbit_MT9_33344)
{
    ProfileOrbit(33'344, RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
}
TEST(RefOrbitKaratsuba_ProfileOrbit_MT9_100096)
{
    ProfileOrbit(100'096, RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
}

TEST(RefOrbitKaratsuba_RandomProductsAndSquares)
{
    std::mt19937_64 random{0x9ca4a75};
    // Includes both sides of Auto's threshold and odd/even stored limb counts.
    for (const mp_bitcnt_t bits : {mp_bitcnt_t{128},
                                   mp_bitcnt_t{16'320},
                                   mp_bitcnt_t{16'384},
                                   mp_bitcnt_t{16'448},
                                   mp_bitcnt_t{32'768},
                                   mp_bitcnt_t{100'000}}) {
        MpfValue x(bits), y(bits), xSquared(bits), ySquared(bits), cross(bits);
        FractalShark::RefOrbitKaratsubaProducts products(bits);
        for (size_t trial = 0; trial < 24; ++trial) {
            Randomize(x.Get(), x.Get()->_mp_prec + 1, (trial & 1) != 0, random);
            Randomize(y.Get(), y.Get()->_mp_prec + (trial % 2), (trial & 2) != 0, random);
            CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
        }
    }
}

TEST(RefOrbitKaratsuba_CarriesNormalizationAndUnequalSources)
{
    constexpr mp_bitcnt_t bits = 16'384;
    MpfValue x(bits), y(bits), xSquared(bits), ySquared(bits), cross(bits);
    FractalShark::RefOrbitKaratsubaProducts products(bits);
    std::mt19937_64 random{42};
    for (const bool allMax : {false, true}) {
        Randomize(x.Get(), x.Get()->_mp_prec + 1, true, random);
        Randomize(y.Get(), 2, false, random);
        std::fill_n(x.Get()->_mp_d, MagnitudeSize(x.Get()), allMax ? GMP_NUMB_MAX : mp_limb_t{0});
        std::fill_n(y.Get()->_mp_d, MagnitudeSize(y.Get()), allMax ? GMP_NUMB_MAX : mp_limb_t{0});
        if (!allMax) {
            x.Get()->_mp_d[MagnitudeSize(x.Get()) - 1] = 1;
            y.Get()->_mp_d[MagnitudeSize(y.Get()) - 1] = 1;
        }
        CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
    }
}

TEST(RefOrbitKaratsuba_CompleteProductAndInputGuardLimbs)
{
    constexpr mp_bitcnt_t sourceBits = 16'384;
    constexpr mp_bitcnt_t outputBits = 2 * sourceBits + 4 * GMP_NUMB_BITS;
    MpfValue x(sourceBits), y(sourceBits), xSquared(outputBits), ySquared(outputBits), cross(outputBits);
    FractalShark::RefOrbitKaratsubaProducts products(outputBits);
    std::mt19937_64 random{57};
    Randomize(x.Get(), x.Get()->_mp_prec + 1, false, random);
    Randomize(y.Get(), y.Get()->_mp_prec + 1, true, random);
    x.Get()->_mp_d[MagnitudeSize(x.Get()) - 1] |= mp_limb_t{1} << (GMP_NUMB_BITS - 1);
    y.Get()->_mp_d[MagnitudeSize(y.Get()) - 1] |= mp_limb_t{1} << (GMP_NUMB_BITS - 1);
    CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
    ASSERT_EQ(MagnitudeSize(cross.Get()), MagnitudeSize(x.Get()) + MagnitudeSize(y.Get()));
    // A product with a nonzero input guard limb must use that limb even when the
    // destination has nominal source precision: compare against the full oracle.
    MpfValue narrowX(sourceBits), narrowY(sourceBits), narrowCross(sourceBits), baseline(sourceBits);
    FractalShark::RefOrbitKaratsubaProducts narrowProducts(sourceBits);
    CheckBatch(narrowProducts, x.Get(), y.Get(), narrowX.Get(), narrowY.Get(), narrowCross.Get());
    mpf_mul(baseline.Get(), x.Get(), y.Get());
    ASSERT_TRUE(mpf_cmp(narrowCross.Get(), baseline.Get()) != 0);
}

TEST(RefOrbitKaratsuba_Aliasing)
{
    constexpr mp_bitcnt_t bits = 16'384;
    MpfValue x(bits), y(bits), cross(bits);
    FractalShark::RefOrbitKaratsubaProducts products(bits);
    std::mt19937_64 random{61};
    Randomize(x.Get(), x.Get()->_mp_prec + 1, true, random);
    Randomize(y.Get(), y.Get()->_mp_prec + 1, false, random);
    CheckBatch(products, x.Get(), y.Get(), x.Get(), y.Get(), cross.Get());
}

TEST(RefOrbitKaratsuba_FullPrecisionRecurrence)
{
    constexpr mp_bitcnt_t bits = 16'384;
    for (const auto &coordinates :
         {std::array{"-0.751", "0.1"}, std::array{"-0.1", "0.1"}, std::array{"-0.125", "0.125"}}) {
        MpfValue x(bits), y(bits), cx(bits), cy(bits), xSquared(bits), ySquared(bits), cross(bits),
            temporary(bits);
        MpfValue expectedX(bits), expectedY(bits), oracleX(bits), oracleY(bits), oracleCross(bits),
            oracleTemp(bits);
        mpf_set_str(cx.Get(), coordinates[0], 10);
        mpf_set_str(cy.Get(), coordinates[1], 10);
        mpf_set(x.Get(), cx.Get());
        mpf_set(y.Get(), cy.Get());
        mpf_set(expectedX.Get(), x.Get());
        mpf_set(expectedY.Get(), y.Get());
        FractalShark::RefOrbitKaratsubaProducts products(bits);
        for (size_t iteration = 0; iteration < 32; ++iteration) {
            if (MagnitudeSize(x.Get()) >= 2 && MagnitudeSize(y.Get()) >= 2) {
                Oracle(oracleX.Get(), expectedX.Get(), expectedX.Get());
                Oracle(oracleY.Get(), expectedY.Get(), expectedY.Get());
                Oracle(oracleCross.Get(), expectedX.Get(), expectedY.Get());
                products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
            } else {
                // Mirror the caller's short-operand path; subsequent full-width
                // iterations continue to use the independent exact integer oracle.
                mpf_mul(oracleX.Get(), expectedX.Get(), expectedX.Get());
                mpf_mul(oracleY.Get(), expectedY.Get(), expectedY.Get());
                mpf_mul(oracleCross.Get(), expectedX.Get(), expectedY.Get());
                mpf_mul(xSquared.Get(), x.Get(), x.Get());
                mpf_mul(ySquared.Get(), y.Get(), y.Get());
                mpf_mul(cross.Get(), x.Get(), y.Get());
            }
            mpf_mul_ui(cross.Get(), cross.Get(), 2);
            mpf_add(y.Get(), cross.Get(), cy.Get());
            mpf_sub(temporary.Get(), xSquared.Get(), ySquared.Get());
            mpf_add(x.Get(), temporary.Get(), cx.Get());
            mpf_mul_ui(oracleCross.Get(), oracleCross.Get(), 2);
            mpf_add(expectedY.Get(), oracleCross.Get(), cy.Get());
            mpf_sub(oracleTemp.Get(), oracleX.Get(), oracleY.Get());
            mpf_add(expectedX.Get(), oracleTemp.Get(), cx.Get());
            CheckSame(x.Get(), expectedX.Get());
            CheckSame(y.Get(), expectedY.Get());
        }
    }
}

TEST(RefOrbitKaratsuba_WorkerHandoffStress)
{
    // Change every batch's inputs, signs and occupied widths. Independent integer oracles
    // expose stale worker completions as well as arithmetic errors across repeated handoffs.
    for (const mp_bitcnt_t bits : {mp_bitcnt_t{128}, mp_bitcnt_t{16'384}}) {
        MpfValue x(bits), y(bits), xSquared(bits), ySquared(bits), cross(bits);
        FractalShark::RefOrbitKaratsubaProducts products(bits);
        std::mt19937_64 random(0x9A11D0FF);
        for (size_t batch = 0; batch < 1024; ++batch) {
            const mp_size_t capacity = static_cast<mp_size_t>(x.Get()->_mp_prec) + 1;
            const mp_size_t xSize = batch % 2 == 0 ? capacity : 2;
            const mp_size_t ySize = batch % 3 == 0 ? 2 : capacity;
            Randomize(x.Get(), xSize, batch % 2 != 0, random);
            Randomize(y.Get(), ySize, batch % 4 != 0, random);
            CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
        }
    }
}

TEST(RefOrbitKaratsuba_IdleWorkerShutdown)
{
    // Exercise shutdown before a worker receives its first job and after a completed job.
    for (size_t repeat = 0; repeat < 8; ++repeat) {
        FractalShark::RefOrbitKaratsubaProducts idleProducts(128);
    }
    MpfValue x(128), y(128), xSquared(128), ySquared(128), cross(128);
    std::mt19937_64 random(0x5A07D0FF);
    Randomize(x.Get(), 2, false, random);
    Randomize(y.Get(), 2, true, random);
    for (size_t repeat = 0; repeat < 8; ++repeat) {
        FractalShark::RefOrbitKaratsubaProducts products(128);
        CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
    }
}

TEST(RefOrbitKaratsuba_AutoSelectionAndExplicitOverride)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    for (const auto bits : {uint64_t{400},
                            uint64_t{3400},
                            uint64_t{10'000},
                            uint64_t{16'383},
                            uint64_t{16'384},
                            uint64_t{16'385}}) {
        fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto);
        ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits)));
        if (bits >= FractalShark::RefOrbitKaratsubaAutoPrecisionBits) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                        RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
        } else if (bits == 3400) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
        } else if (bits == 400) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::STPeriodicity);
        } else {
            ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                        RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
        }
        fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
        ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
        fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
        ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    }
}

TEST(RefOrbitKaratsuba_MenuCommandAndCatalog)
{
#include "MenuTreeDef.h"
    const auto command = FractalShark::FractalCommand::PerturbationMultithread9KaratsubaPeriodicity;
    const auto *node = FindNode(menu, static_cast<uint32_t>(command));
    ASSERT_TRUE(node != nullptr);
    ASSERT_TRUE(node->text == L"MT9 + Periodicity + Karatsuba");
    ASSERT_TRUE(node->radioGroup == FractalShark::RadioGroup::PerturbationMode);
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    CommandFixture commands(fractal);
    commands.ExecuteCommand(command);
    fractal.GetRenderPool()->Drain();
    ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    FractalShark::MenuState state(fractal);
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::PerturbationMode),
              static_cast<uint32_t>(IDM_PERTURBATION_MULTITHREAD9_KARATSUBA_PERIODICITY));
    ASSERT_EQ(std::count_if(RenderTests::Cases().begin(),
                            RenderTests::Cases().end(),
                            [](const auto &test) {
                                return test.Reference ==
                                       RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba;
                            }),
              2);
}

TEST(RefOrbitKaratsuba_CancellationAndPeriodicity)
{
    MPIRPrecision restore(1024);
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.SetNumIterations<uint64_t>(64);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    AbortMonitor abort(false);
    abort.SetStopCalculating(true);
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             double,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Disable>(fractal.GetPtz());
    ASSERT_TRUE(
        (calculator.GetLast<uint64_t, double, PerturbExtras::Disable>()->GetCountOrbitEntries() <= 2));
    abort.ResetStopCalculating();
    calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             double,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Disable>(fractal.GetPtz());
    const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    ASSERT_TRUE(results->GetPeriodMaybeZero() > 0);
    ASSERT_TRUE(results->GetCountOrbitEntries() < 64);
}

TEST(RefOrbitKaratsuba_CompressionExportsAndImaginaRoundtrip)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    RenderTests::ScopedDirectory directory("RefOrbitKaratsuba_CompressionExportsAndImaginaRoundtrip");
    Fractal fractal(32, 32, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    fractal.View(1, false);
    fractal.SetNumIterations<uint64_t>(256);
    const auto algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(algorithm));
    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    fractal.CalcFractal(true);
    const auto expectedPixels = SavePixels(fractal, "original");
    for (const auto compression : {CompressToDisk::Disable,
                                   CompressToDisk::SimpleCompression,
                                   CompressToDisk::MaxCompression,
                                   CompressToDisk::MaxCompressionImagina}) {
        const auto suffix = std::to_string(static_cast<int>(compression));
        const std::filesystem::path orbit = "orbit-" + suffix;
        fractal.SaveRefOrbit(compression, orbit.wstring());
        ASSERT_TRUE(std::filesystem::file_size(orbit) > 0);
        // The three text formats support export; only Imagina/max supports loading.
        if (compression == CompressToDisk::MaxCompressionImagina) {
            fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            fractal.LoadRefOrbit(nullptr, compression, ImaginaSettings::UseSaved, orbit.wstring());
            ASSERT_TRUE(fractal.SetRenderAlgorithm(algorithm));
            fractal.SetIterType(IterTypeEnum::Bits64);
            fractal.SetNumIterations<uint64_t>(256);
            fractal.ForceRecalc();
            fractal.GetRenderPool()->Drain();
            fractal.CalcFractal(true);
            ASSERT_TRUE(SavePixels(fractal, "loaded-" + suffix) == expectedPixels);
        }
    }
}
