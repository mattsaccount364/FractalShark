#include "TestFramework.h"

#include "AbortMonitor.h"
#include "AlgCmds.h"
#include "ConsoleLog.h"
#include "Environment.h"
#include "Exceptions.h"
#include "Fractal.h"
#include "GpuPrecisionDispatch.h"
#include "MenuState.h"
#include "MenuTree.h"
#include "PerturbationResults.h"
#include "PortableCommandHandlers.h"
#include "RefOrbitKaratsuba.h"
#include "RefOrbitPrecision.h"
#include "RenderTestCatalog.h"
#include "RenderTestSupport.h"
#include "RenderThreadPool.h"
#include "ScopedMpir.h"
#include "WPngImage/lodepng.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

// Each caller drains its fixture before taking a baseline. Compare deltas rather
// than resetting shared totals, so earlier tests and other sample types retain theirs.
void
CheckDynamicPrecisionCounterDelta(
    const std::optional<FractalShark::RefOrbitDynamicPrecisionCounters> &before,
    uint64_t successes,
    uint64_t failures)
{
    const auto after = RefOrbitCalc::GetDynamicPrecisionCounters();
    ASSERT_EQ(after.has_value(), before.has_value());
    if (before.has_value()) {
        ASSERT_EQ(after->Successes - before->Successes, successes);
        ASSERT_EQ(after->Failures - before->Failures, failures);
    }
}

// Capture the existing console sink only while the drained fixture performs the
// synchronous call. Always restore stderr, including if an assertion or call throws.
class ScopedConsoleLogCapture {
public:
    ScopedConsoleLogCapture() : m_Previous(std::cerr.rdbuf(m_Message.rdbuf())) {}
    // A supplied sink can request cancellation synchronously when a warning is emitted.
    explicit ScopedConsoleLogCapture(std::streambuf &capture) : m_Previous(std::cerr.rdbuf(&capture)) {}
    ~ScopedConsoleLogCapture() { std::cerr.rdbuf(m_Previous); }
    ScopedConsoleLogCapture(const ScopedConsoleLogCapture &) = delete;
    ScopedConsoleLogCapture &operator=(const ScopedConsoleLogCapture &) = delete;
    std::string
    GetMessage() const
    {
        return m_Message.str();
    }

private:
    std::ostringstream m_Message;
    std::streambuf *m_Previous;
};

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

// Reconstruct the logged dyadic with independent MPIR integer parsing so that
// diagnostics must preserve the original coordinate, including bits lost on copy.
void
CheckLoggedCoordinate(const std::string &message, const char *name, mpf_srcptr expected)
{
    const auto marker = std::string{", "} + name + '=';
    const auto start = message.find(marker);
    ASSERT_TRUE(start != std::string::npos);
    const auto coefficientStart = start + marker.size();
    const auto power = message.find(" * 2^", coefficientStart);
    ASSERT_TRUE(power != std::string::npos);
    std::string coefficient = message.substr(coefficientStart, power - coefficientStart);
    const auto prefix = coefficient.find("0x");
    ASSERT_TRUE(prefix != std::string::npos);
    coefficient.erase(prefix, 2);
    MpzValue integer;
    ASSERT_EQ(mpz_set_str(integer.Get(), coefficient.c_str(), 16), 0);
    const auto exponentStart = power + std::string_view{" * 2^"}.size();
    const auto exponentEnd = message.find_first_of(",\n", exponentStart);
    ASSERT_TRUE(exponentEnd != std::string::npos);
    int64_t exponent{};
    const auto parsed =
        std::from_chars(message.data() + exponentStart, message.data() + exponentEnd, exponent);
    ASSERT_TRUE(parsed.ec == std::errc{});
    ASSERT_TRUE(parsed.ptr == message.data() + exponentEnd);
    MpfValue reconstructed(mpf_get_prec(expected));
    mpf_set_z(reconstructed.Get(), integer.Get());
    if (exponent < 0) {
        mpf_div_2exp(reconstructed.Get(), reconstructed.Get(), static_cast<mp_bitcnt_t>(-exponent));
    } else {
        mpf_mul_2exp(reconstructed.Get(), reconstructed.Get(), static_cast<mp_bitcnt_t>(exponent));
    }
    ASSERT_EQ(mpf_cmp(reconstructed.Get(), expected), 0);
}

void
CheckCertificationReport(const std::string &message,
                         const FractalShark::RefOrbitDynamicPrecisionStats &stats)
{
    const size_t start = message.find("MT9 dynamic precision ");
    ASSERT_TRUE(start != std::string::npos);
    const size_t end = message.find('\n', start);
    const std::string_view report{message.data() + start,
                                  (end == std::string::npos ? message.size() : end) - start};
    const auto read = [&](std::string_view field) {
        const size_t position = report.find(field);
        ASSERT_TRUE(position != std::string_view::npos);
        double value{};
        const auto parsed = std::from_chars(
            report.data() + position + field.size(), report.data() + report.size(), value);
        ASSERT_TRUE(parsed.ec == std::errc{});
        return value;
    };
    const double fast = static_cast<double>(stats.FastCertifiedComponents);
    const double regular = static_cast<double>(stats.RegularCertificationComponents);
    ASSERT_EQ(read("certification_fast_components="), fast);
    ASSERT_EQ(read("certification_regular_components="), regular);
    ASSERT_NEAR(read("certification_fast_percent="),
                fast + regular == 0.0 ? 0.0 : 100.0 * fast / (fast + regular),
                1e-12);
    // Both component budgets must be on the route-statistics line. Special states
    // have readable names; finite budgets use two decimal places on each side.
    for (const std::string_view field : {"worst_real_error_bits=", "worst_imaginary_error_bits="}) {
        const size_t position = report.find(field);
        ASSERT_TRUE(position != std::string_view::npos);
        const size_t begin2 = position + field.size();
        const size_t end2 = report.find_first_of(", \r", begin2);
        const auto budget = report.substr(begin2, end2 == std::string_view::npos ? end2 : end2 - begin2);
        if (budget == "not_checked" || budget == "exact" || budget == "zero_gap" ||
            budget == "unsupported_conversion" || budget == "invalid_bound") {
            continue;
        }
        const size_t slash = budget.find('/');
        ASSERT_TRUE(slash != std::string_view::npos);
        for (const auto number : {budget.substr(0, slash), budget.substr(slash + 1)}) {
            double value{};
            const auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
            ASSERT_TRUE(parsed.ec == std::errc{});
            ASSERT_TRUE(parsed.ptr == number.data() + number.size());
            ASSERT_TRUE(std::isfinite(value));
            const size_t point = number.find('.');
            ASSERT_TRUE(point != std::string_view::npos);
            ASSERT_EQ(number.size() - point - 1, size_t{2});
        }
    }
    for (const std::string_view field :
         {"worst_real_error_state_Z=", "worst_imaginary_error_state_Z="}) {
        const double state = read(field);
        ASSERT_TRUE(state >= 0.0 && state <= static_cast<double>(stats.Steps) + 1.0);
        ASSERT_EQ(state, std::floor(state));
        if (fast + regular == 0.0) {
            ASSERT_EQ(state, 0.0);
        }
    }
    for (const std::string_view field : {"accumulated_error_bound_log2=",
                                         "comparison_error_bound_log2=",
                                         "gap_lower_bound_log2=",
                                         "gap_rounding_bound_log2=",
                                         "certification_headroom_bits=",
                                         "final_computed_state_Z=",
                                         "final_state_certifies="}) {
        ASSERT_TRUE(report.find(field) == std::string_view::npos);
    }
}

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

template <bool DynamicPrecision>
void
CheckBatch(FractalShark::RefOrbitKaratsubaProductsT<DynamicPrecision> &products,
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

TEST(RefOrbitBoundedAllocator_LargeValuesAndReallocation)
{
    MPIRBoundedAllocator allocator;
    allocator.InitScopedAllocators();
    struct CallerTls {
        CallerTls() { MPIRBoundedAllocator::InitTls(); }
        ~CallerTls() { MPIRBoundedAllocator::ShutdownTls(); }
    } callerTls;
    // Each of these values needs more than half an arena block. Keeping twelve
    // live values exercises exhaustion without depending on allocator internals.
    // Their destructors, including assertion unwinding, run before ShutdownTls.
    std::vector<std::unique_ptr<MpfValue>> values;
    for (unsigned i = 0; i < 12; ++i) {
        values.push_back(std::make_unique<MpfValue>(5'000'000));
        mpf_set_ui(values.back()->Get(), 1);
        mpf_div_ui(values.back()->Get(), values.back()->Get(), i + 2);
    }
    MpfValue large(14'000'000);
    for (unsigned i = 0; i < values.size(); ++i) {
        mpf_mul_ui(large.Get(), values[i]->Get(), i + 2);
        ASSERT_NEAR(mpf_get_d(large.Get()), 1.0, 1e-14);
    }
    // Integer growth invokes MPIR's realloc callback: first inside the arena,
    // then across its size limit, and finally on a heap-owned allocation.
    MpzValue integer;
    mpz_set_ui(integer.Get(), 17);
    for (const mp_bitcnt_t shift : {mp_bitcnt_t{128}, mp_bitcnt_t{14'000'000}, mp_bitcnt_t{2'000'000}}) {
        mpz_mul_2exp(integer.Get(), integer.Get(), shift);
    }
    mpz_tdiv_q_2exp(integer.Get(), integer.Get(), 16'000'128);
    ASSERT_EQ(mpz_cmp_ui(integer.Get(), 17), 0);
}

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

TEST(RefOrbitKaratsuba_ZeroOneLimbAndMixedInputs)
{
    std::mt19937_64 random{0x5A07};
    // Minimum requested precision exercises construction rounding. Reusing each backend
    // across zero, one-limb, and full-width batches exposes stale scratch and metadata.
    for (const mp_bitcnt_t bits : {mp_bitcnt_t{1}, mp_bitcnt_t{128}, mp_bitcnt_t{16'384}}) {
        MpfValue x(bits), y(bits), xSquared(bits), ySquared(bits), cross(bits);
        FractalShark::RefOrbitKaratsubaProducts products(bits);
        for (const int left : {-3, -1, 0, 1, 5}) {
            for (const int right : {-3, -1, 0, 1, 5}) {
                mpf_set_si(x.Get(), left);
                mpf_set_si(y.Get(), right);
                ASSERT_EQ(MagnitudeSize(x.Get()), left == 0 ? mp_size_t{0} : mp_size_t{1});
                ASSERT_EQ(MagnitudeSize(y.Get()), right == 0 ? mp_size_t{0} : mp_size_t{1});
                CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
                CheckBatch(products, x.Get(), y.Get(), x.Get(), y.Get(), cross.Get());
            }
        }
        for (const mp_size_t shortSize : {mp_size_t{0}, mp_size_t{1}}) {
            for (const bool negative : {false, true}) {
                if (shortSize == 0) {
                    mpf_set_ui(x.Get(), 0);
                } else {
                    Randomize(x.Get(), shortSize, negative, random);
                }
                // Include the guard limb of the wide input even when its partner is short.
                Randomize(y.Get(), y.Get()->_mp_prec + 1, !negative, random);
                CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
                mpf_swap(x.Get(), y.Get());
                CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
                CheckBatch(products, x.Get(), y.Get(), x.Get(), y.Get(), cross.Get());
            }
        }
    }
}

TEST(RefOrbitKaratsuba_FullPrecisionRecurrence)
{
    constexpr mp_bitcnt_t bits = 16'384;
    // Axis and cancellation cases change occupied widths after initialization;
    // the same split backend must handle every iteration without a caller fallback.
    for (const auto &coordinates : {std::array{"-0.751", "0.1"},
                                    std::array{"-0.1", "0.1"},
                                    std::array{"-0.125", "0.125"},
                                    std::array{"0", "0"},
                                    std::array{"-1", "0"},
                                    std::array{"0", "1"}}) {
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
            Oracle(oracleX.Get(), expectedX.Get(), expectedX.Get());
            Oracle(oracleY.Get(), expectedY.Get(), expectedY.Get());
            Oracle(oracleCross.Get(), expectedX.Get(), expectedY.Get());
            products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
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

TEST(RefOrbitKaratsuba_StartupStorageModes)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(16'384);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    const HighPrecision cx{"-0.1"}, cy{"0.1"};
    // Exercise the production specializations for every storage mode. A deep view
    // keeps the periodicity radius far below this orbit's magnitude during the fixture.
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(16'384).Recentered(cx, cy)));
    const std::array expectedSamples{
        std::array{0.0, 0.0}, std::array{-0.1, 0.1}, std::array{-0.1, 0.08}};
    AbortMonitor abort(false);
    auto checkStartup = [&]<PerturbExtras extras>(uint64_t limit, bool cancelBeforeStartup) {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(cx, cy);
        fractal.SetNumIterations<uint64_t>(limit);
        abort.SetStopCalculating(cancelBeforeStartup);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 HDRFloat<double>,
                                                 double,
                                                 extras,
                                                 RefOrbitCalc::BenchmarkMode::Disable>(fractal.GetPtz());
        abort.ResetStopCalculating();
        const auto *results = calculator.GetLast<uint64_t, HDRFloat<double>, extras>();
        // InitResults always stores Z0, even if iteration zero is skipped or unpublished.
        const uint64_t expectedCount = cancelBeforeStartup ? 1 : std::max(uint64_t{1}, limit);
        ASSERT_EQ(results->GetCountOrbitEntries(), expectedCount);
        ASSERT_EQ(results->GetPeriodMaybeZero(), uint64_t{0});
        ASSERT_EQ(results->GetCompressionErrorExp(),
                  fractal.GetCompressionErrorExp(Fractal::CompressionError::Low));
        RuntimeDecompressor<uint64_t, HDRFloat<double>, extras> decompressor{*results};
        for (size_t index = 0; index < expectedCount; ++index) {
            HDRFloat<double> x, y;
            results->GetComplex(decompressor, index, x, y);
            ASSERT_NEAR(static_cast<double>(x), expectedSamples[index][0], 1e-15);
            ASSERT_NEAR(static_cast<double>(y), expectedSamples[index][1], 1e-15);
        }
        if constexpr (extras == PerturbExtras::SimpleCompression) {
            ASSERT_EQ(results->GetOrbitData()[0].u.f.CompressionIndex, uint64_t{0});
        } else if constexpr (extras == PerturbExtras::Bad) {
            ASSERT_FALSE(results->GetOrbitData()[expectedCount - 1].bad);
        }
    };
    for (const uint64_t limit : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{3}}) {
        for (const bool cancelBeforeStartup : {false, true}) {
            checkStartup.template operator()<PerturbExtras::Disable>(limit, cancelBeforeStartup);
            checkStartup.template operator()<PerturbExtras::SimpleCompression>(limit,
                                                                               cancelBeforeStartup);
            checkStartup.template operator()<PerturbExtras::Bad>(limit, cancelBeforeStartup);
        }
    }

    // Force stored corrections in a short orbit so this checks actual waypoint indices,
    // rather than only reconstructing all samples from the initial compressed entry.
    constexpr uint64_t waypointIterations = 8;
    constexpr int32_t waypointCompressionErrorExp = 100;
    fractal.SetNumIterations<uint64_t>(waypointIterations);
    fractal.SetCompressionErrorExp(Fractal::CompressionError::Low, waypointCompressionErrorExp);
    auto generateWaypoints = [&]<PerturbExtras extras>() {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(cx, cy);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 HDRFloat<double>,
                                                 double,
                                                 extras,
                                                 RefOrbitCalc::BenchmarkMode::Disable>(fractal.GetPtz());
        return calculator.GetLast<uint64_t, HDRFloat<double>, extras>();
    };
    const auto *plain = generateWaypoints.template operator()<PerturbExtras::Disable>();
    ASSERT_EQ(plain->GetCountOrbitEntries(), waypointIterations);
    std::vector<std::array<double, 2>> plainSamples;
    for (uint64_t index = 0; index < waypointIterations; ++index) {
        plainSamples.push_back({static_cast<double>(plain->GetOrbitData()[index].x),
                                static_cast<double>(plain->GetOrbitData()[index].y)});
    }
    const auto *compressed = generateWaypoints.template operator()<PerturbExtras::SimpleCompression>();
    ASSERT_EQ(compressed->GetCountOrbitEntries(), waypointIterations);
    ASSERT_TRUE(compressed->GetCompressedOrbitSize() > 1);
    RuntimeDecompressor<uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression> decompressor{
        *compressed};
    for (size_t index = 0; index < waypointIterations; ++index) {
        HDRFloat<double> x, y;
        compressed->GetComplex(decompressor, index, x, y);
        ASSERT_NEAR(static_cast<double>(x), plainSamples[index][0], 1e-15);
        ASSERT_NEAR(static_cast<double>(y), plainSamples[index][1], 1e-15);
    }
    uint64_t previousIndex = 0;
    for (size_t waypoint = 1; waypoint < compressed->GetCompressedOrbitSize(); ++waypoint) {
        const auto &entry = compressed->GetOrbitData()[waypoint];
        const uint64_t index = entry.u.f.CompressionIndex;
        ASSERT_TRUE(index > previousIndex && index < waypointIterations);
        ASSERT_NEAR(static_cast<double>(entry.x), plainSamples[index][0], 1e-15);
        ASSERT_NEAR(static_cast<double>(entry.y), plainSamples[index][1], 1e-15);
        previousIndex = index;
    }
}

TEST(RefOrbitKaratsuba_FirstFinalAndActiveCancellation)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(16'384);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    const HighPrecision cx{"-0.1"}, cy{"0.1"};
    for (const uint64_t limit : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{3}}) {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(cx, cy);
        fractal.SetNumIterations<uint64_t>(limit);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
        ASSERT_EQ(results->GetCountOrbitEntries(), std::max(uint64_t{1}, limit));
        ASSERT_EQ(results->GetOrbitData()[0].x, 0.0);
        ASSERT_EQ(results->GetOrbitData()[0].y, 0.0);
        if (limit > 1) {
            ASSERT_NEAR(results->GetOrbitData()[1].x, -0.1, 1e-15);
            ASSERT_NEAR(results->GetOrbitData()[1].y, 0.1, 1e-15);
        }
        if (limit > 2) {
            ASSERT_NEAR(results->GetOrbitData()[2].x, -0.1, 1e-15);
            ASSERT_NEAR(results->GetOrbitData()[2].y, 0.08, 1e-15);
        }
    }
    calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    calculator.ResetGuess(cx, cy);
    constexpr uint64_t limit = 1'048'576;
    fractal.SetNumIterations<uint64_t>(limit);
    AbortMonitor abort(false);
    {
        // Cancel a deliberately long bounded orbit after arithmetic has begun. No
        // reference data is read concurrently; join the requester before assertions.
        std::jthread requester([&abort] {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            abort.SetStopCalculating(true);
        });
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    }
    abort.ResetStopCalculating();
    const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    ASSERT_TRUE(results->GetCountOrbitEntries() > 1);
    ASSERT_TRUE(results->GetCountOrbitEntries() < limit);
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

TEST(RefOrbitAuto_PrecisionBoundaries)
{
    const auto renderer = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2);
    struct SelectionCase {
        uint64_t PrecisionBits;
        RefOrbitCalc::PerturbationAlg Expected;
    };
    // Expectations are explicit so moving a threshold cannot make the test follow the implementation.
    const std::array cases{
        SelectionCase{639, RefOrbitCalc::PerturbationAlg::STPeriodicity},
        SelectionCase{640, RefOrbitCalc::PerturbationAlg::MTPeriodicity3},
        SelectionCase{641, RefOrbitCalc::PerturbationAlg::MTPeriodicity3},
        SelectionCase{4'095, RefOrbitCalc::PerturbationAlg::MTPeriodicity3},
        SelectionCase{4'096, RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3},
        SelectionCase{4'097, RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3},
        SelectionCase{8'191, RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3},
        SelectionCase{8'192, RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision},
        SelectionCase{8'193, RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision},
        SelectionCase{32'767, RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision},
        SelectionCase{32'768, RefOrbitCalc::PerturbationAlg::GPU},
        SelectionCase{32'769, RefOrbitCalc::PerturbationAlg::GPU}};
    for (const auto &test : cases) {
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(test.PrecisionBits, true, renderer, 9) ==
                    test.Expected);
        const auto withoutGpu =
            test.Expected == RefOrbitCalc::PerturbationAlg::GPU
                ? RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision
                : test.Expected;
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(test.PrecisionBits, false, renderer, 9) ==
                    withoutGpu);
    }
}

TEST(RefOrbitAuto_LogicalProcessorEligibility)
{
    const auto renderer = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2);
    const auto incompatibleRenderer =
        GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedScaled);
    constexpr uint64_t maximumGpuBits = static_cast<uint64_t>(MaxSupportedLimbCount) * 32;
    struct SelectionCase {
        uint64_t PrecisionBits;
        RefOrbitCalc::PerturbationAlg Expected;
    };
    const std::array lowerPrecisionCases{
        SelectionCase{639, RefOrbitCalc::PerturbationAlg::STPeriodicity},
        SelectionCase{640, RefOrbitCalc::PerturbationAlg::MTPeriodicity3},
        SelectionCase{4'095, RefOrbitCalc::PerturbationAlg::MTPeriodicity3},
        SelectionCase{4'096, RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3},
        SelectionCase{8'191, RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3}};

    // Inject capacities instead of restricting the test process or relying on the host CPU.
    // In particular, eight workers still need a ninth logical CPU for the calling thread.
    for (const size_t logicalProcessorCount : {0, 1, 2, 4, 8, 9, 10, 32}) {
        const auto expectedCpu =
            logicalProcessorCount >= 9
                ? RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision
                : RefOrbitCalc::PerturbationAlg::MTPeriodicity3;
        for (const uint64_t bits : {uint64_t{8'192},
                                    uint64_t{8'193},
                                    uint64_t{32'767},
                                    uint64_t{32'768},
                                    maximumGpuBits,
                                    maximumGpuBits + 1,
                                    std::numeric_limits<uint64_t>::max()}) {
            ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(
                            bits, false, renderer, logicalProcessorCount) == expectedCpu);
            ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(
                            bits, true, incompatibleRenderer, logicalProcessorCount) == expectedCpu);
            const auto expectedWithGpu = bits >= 32'768 && bits <= maximumGpuBits
                                             ? RefOrbitCalc::PerturbationAlg::GPU
                                             : expectedCpu;
            ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(
                            bits, true, renderer, logicalProcessorCount) == expectedWithGpu);
        }
        for (const auto &test : lowerPrecisionCases) {
            ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(
                            test.PrecisionBits, false, renderer, logicalProcessorCount) ==
                        test.Expected);
        }
    }
}

TEST(RefOrbitAuto_GpuTypeAndFormatCompatibility)
{
    // Exercise the whole catalog against a concrete supported set, including CPU HDR consumers.
    // Plain types, scaled bookkeeping, and compressed double-float conversion use CPU dynamic MT9.
    const std::array supported{RenderAlgorithmEnum::Cpu32PerturbedBLAHDR,
                               RenderAlgorithmEnum::Cpu64PerturbedBLAHDR,
                               RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR,
                               RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR,
                               RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR,
                               RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedBLA,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedBLA,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO,
                               RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO,
                               RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2,
                               RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO,
                               RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO,
                               RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO};
    for (const auto &renderer : RenderAlgorithms) {
        const bool compatible =
            std::find(supported.begin(), supported.end(), renderer.Algorithm) != supported.end();
        const auto expected =
            compatible ? RefOrbitCalc::PerturbationAlg::GPU
                       : RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision;
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'768, true, renderer, 9) == expected);
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'767, true, renderer, 9) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'768, false, renderer, 9) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    }
}

TEST(RefOrbitAuto_GpuPrecisionCapacity)
{
    const auto renderer = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2);
    // GPU buckets store 32-bit limbs. Test the capacity without allocating a viewport at that depth.
    constexpr uint64_t maximumGpuBits = static_cast<uint64_t>(MaxSupportedLimbCount) * 32;
    for (const auto bits : {maximumGpuBits - 1, maximumGpuBits}) {
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(bits, true, renderer, 9) ==
                    RefOrbitCalc::PerturbationAlg::GPU);
    }
    for (const auto bits : {maximumGpuBits + 1, std::numeric_limits<uint64_t>::max()}) {
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(bits, true, renderer, 9) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    }
}

TEST(RefOrbitKaratsuba_AutoSelectionAndExplicitOverride)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    const auto expectedHighPrecision =
        Environment::LogicalProcessorCount() >= 9
            ? RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision
            : RefOrbitCalc::PerturbationAlg::MTPeriodicity3;
    for (const auto bits : {uint64_t{400},
                            uint64_t{639},
                            uint64_t{640},
                            uint64_t{641},
                            uint64_t{3400},
                            uint64_t{4'095},
                            uint64_t{4'096},
                            uint64_t{4'097},
                            uint64_t{8'191},
                            uint64_t{8'192},
                            uint64_t{8'193},
                            uint64_t{10'000},
                            uint64_t{32'767},
                            uint64_t{32'768},
                            uint64_t{32'769}}) {
        fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto);
        ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits)));
        if (bits >= 8'192) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() == expectedHighPrecision);
        } else if (bits >= 4'096) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                        RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
        } else if (bits >= 640) {
            ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
        } else {
            ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::STPeriodicity);
        }
        const auto selected = fractal.GetPerturbationAlg();
        const uint64_t reusePadding =
            bits >= 4'096 && bits < 8'192
                ? AuthoritativeReuseExtraPrecisionInBits - AuthoritativeMinExtraPrecisionInBits
                : 0;
        ASSERT_EQ(fractal.GetPrecision(), bits + reusePadding);
        ASSERT_TRUE(fractal.GetPerturbationAlg() == selected);

        // Explicit GPU selection is also preserved even when runtime CUDA is disabled;
        // generation retains its existing error contract, while this test checks the picker only.
        for (const auto explicitAlg :
             {RefOrbitCalc::PerturbationAlg::MTPeriodicity3,
              RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba,
              RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision,
              RefOrbitCalc::PerturbationAlg::GPU}) {
            fractal.SetPerturbationAlg(explicitAlg);
            ASSERT_TRUE(fractal.GetPerturbationAlg() == explicitAlg);
        }
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

TEST(RefOrbitKaratsuba_BenchmarkPeriodicityGuard)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(16'384);
    constexpr uint64_t iterationLimit = 32;
    fractal.SetNumIterations<uint64_t>(iterationLimit);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    auto generate = [&]<RefOrbitCalc::BenchmarkMode benchmarkState>(const HighPrecision &cx,
                                                                    const HighPrecision &cy) {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(cx, cy);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 benchmarkState>(fractal.GetPtz());
        return calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    };
    const HighPrecision zero{0};
    const auto *ordinary =
        generate.template operator()<RefOrbitCalc::BenchmarkMode::Disable>(zero, zero);
    ASSERT_EQ(ordinary->GetPeriodMaybeZero(), uint64_t{2});
    ASSERT_EQ(ordinary->GetCountOrbitEntries(), uint64_t{2});

    // The same zero orbit must run to its limit when periodicity exits are compiled out.
    const auto *benchmark =
        generate.template operator()<RefOrbitCalc::BenchmarkMode::Enable>(zero, zero);
    ASSERT_EQ(benchmark->GetPeriodMaybeZero(), uint64_t{0});
    ASSERT_EQ(benchmark->GetCountOrbitEntries(), iterationLimit);
    for (uint64_t iteration = 0; iteration < iterationLimit; ++iteration) {
        ASSERT_EQ(benchmark->GetOrbitData()[iteration].x, 0.0);
        ASSERT_EQ(benchmark->GetOrbitData()[iteration].y, 0.0);
    }

    // Escape remains active in benchmark mode, independent of the periodicity guard.
    const HighPrecision escapingX{20};
    const auto *escaped =
        generate.template operator()<RefOrbitCalc::BenchmarkMode::Enable>(escapingX, zero);
    ASSERT_EQ(escaped->GetPeriodMaybeZero(), uint64_t{0});
    ASSERT_EQ(escaped->GetCountOrbitEntries(), uint64_t{2});
    const auto *ordinaryEscape =
        generate.template operator()<RefOrbitCalc::BenchmarkMode::Disable>(escapingX, zero);
    ASSERT_EQ(ordinaryEscape->GetPeriodMaybeZero(), uint64_t{0});
    ASSERT_EQ(ordinaryEscape->GetCountOrbitEntries(), uint64_t{2});
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

TEST(RefOrbitDynamicPrecision_VariableWidthProducts)
{
    constexpr mp_bitcnt_t fullBits = 4096;
    MpfValue x(fullBits), y(fullBits), xSquared(fullBits), ySquared(fullBits), cross(fullBits);
    // Restoration must also happen when an assertion throws, before MpfValue frees
    // the allocation using its original size. Only active precision changes here.
    struct RestorePrecision {
        std::array<mpf_ptr, 5> Values;
        ~RestorePrecision()
        {
            for (auto *value : Values) {
                mpf_set_prec_raw(value, fullBits);
            }
        }
    } restore{{x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get()}};
    FractalShark::RefOrbitKaratsubaProductsT<true> products(fullBits);
    std::mt19937_64 random{0xD19A61C};
    for (size_t batch = 0; batch < 96; ++batch) {
        const std::array<mp_bitcnt_t, 6> widths{4096, 192, 64, 1280, 128, 4096};
        const auto bits = widths[batch % widths.size()];
        for (auto *value : restore.Values) {
            mpf_set_prec_raw(value, bits);
        }
        const mp_size_t capacity = x.Get()->_mp_prec + 1;
        Randomize(x.Get(), batch % 7 == 0 ? 1 : capacity, batch % 2 != 0, random);
        Randomize(y.Get(), batch % 5 == 0 ? 1 : capacity, batch % 3 != 0, random);
        if (batch % 11 == 0) {
            mpf_set_ui(y.Get(), 0);
        }
        CheckBatch(products, x.Get(), y.Get(), xSquared.Get(), ySquared.Get(), cross.Get());
        CheckBatch(products, x.Get(), y.Get(), x.Get(), y.Get(), cross.Get());
    }
}

TEST(RefOrbitDynamicPrecision_LocalRoundingBounds)
{
    // The oracle destinations cover every source bit and aligned exponent: these
    // additions, subtractions and products are exact dyadic arithmetic, independently
    // of the low-precision MPIR result whose error is being bounded.
    MpfValue x(512), y(512), actual(128), exact(8192), difference(8192);
    FractalShark::RefOrbitKaratsubaProductsT<true> products(512);
    MpfValue xSquared(128), ySquared(128), cross(128);
    std::mt19937_64 random{0xB0A6D5};
    const auto checkError = [&](FractalShark::RefOrbitErrorBound bound, mpf_srcptr value) {
        mpf_sub(difference.Get(), exact.Get(), value);
        if (mpf_sgn(difference.Get()) != 0) {
            ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less(
                FractalShark::RefOrbitErrorBound::FromMpfUpper(difference.Get()), bound));
        }
    };
    for (size_t trial = 0; trial < 128; ++trial) {
        Randomize(x.Get(), x.Get()->_mp_prec + 1, trial % 2 != 0, random);
        Randomize(y.Get(), y.Get()->_mp_prec + 1, trial % 3 != 0, random);
        const auto sumBound =
            FractalShark::RefOrbitPrecisionSchedule::AddError(x.Get(), y.Get(), actual.Get());
        mpf_add(exact.Get(), x.Get(), y.Get());
        mpf_add(actual.Get(), x.Get(), y.Get());
        checkError(sumBound, actual.Get());
        const auto subtractBound =
            FractalShark::RefOrbitPrecisionSchedule::SubtractError(x.Get(), y.Get(), actual.Get());
        mpf_sub(exact.Get(), x.Get(), y.Get());
        mpf_sub(actual.Get(), x.Get(), y.Get());
        checkError(subtractBound, actual.Get());
        // Keep inputs within the dynamic product's active capacity while retaining
        // their full guard limbs. The independent wide mpf_mul is exact at 8192 bits.
        const auto retained = actual.Get()->_mp_prec + 1;
        Randomize(x.Get(), retained, trial % 2 != 0, random);
        Randomize(y.Get(), retained, trial % 3 != 0, random);
        products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
        mpf_mul(exact.Get(), x.Get(), y.Get());
        checkError(FractalShark::RefOrbitPrecisionSchedule::ProductError(cross.Get()), cross.Get());
    }
    // A borrowed descriptor exercises extreme exponents without asking MPIR to
    // perform arithmetic there or changing the owner. Windows MPIR uses a 32-bit
    // limb exponent, while GMP can use 64 bits; bound arithmetic must widen first.
    __mpf_struct extreme = *x.Get();
    extreme._mp_exp = std::numeric_limits<decltype(extreme._mp_exp)>::min();
    extreme._mp_prec = 2;
    if constexpr (sizeof(extreme._mp_exp) < sizeof(int64_t)) {
        ASSERT_TRUE(
            FractalShark::RefOrbitPrecisionSchedule::CopyError(&extreme, actual.Get()).IsFinite());
        ASSERT_TRUE(FractalShark::RefOrbitPrecisionSchedule::ProductError(&extreme).IsFinite());
        const auto doubled = FractalShark::RefOrbitPrecisionSchedule::DoublingError(&extreme);
        ASSERT_TRUE(doubled.IsFinite());
        const double expectedLog = static_cast<double>(
            (static_cast<int64_t>(extreme._mp_exp) - extreme._mp_prec) * GMP_NUMB_BITS + 2);
        ASSERT_NEAR(doubled.Log2(), expectedLog, 0.001);
    } else {
        ASSERT_FALSE(
            FractalShark::RefOrbitPrecisionSchedule::CopyError(&extreme, actual.Get()).IsFinite());
        ASSERT_FALSE(FractalShark::RefOrbitPrecisionSchedule::ProductError(&extreme).IsFinite());
        ASSERT_FALSE(FractalShark::RefOrbitPrecisionSchedule::DoublingError(&extreme).IsFinite());
    }
    const auto tiny = FractalShark::RefOrbitErrorBound::PowerOfTwo(-100'000);
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less(
        tiny,
        FractalShark::RefOrbitErrorBound::Add(tiny,
                                              FractalShark::RefOrbitErrorBound::PowerOfTwo(-102'000))));
    ASSERT_TRUE(
        FractalShark::RefOrbitErrorBound::Less(FractalShark::RefOrbitErrorBound::PowerOfTwo(-200'000),
                                               FractalShark::RefOrbitErrorBound::Multiply(tiny, tiny)));
}

TEST(RefOrbitDynamicPrecision_AddSubtractBounds)
{
    constexpr mp_bitcnt_t DestinationBits = 128;
    constexpr mp_bitcnt_t InputBits = 512;
    constexpr mp_bitcnt_t ExactBits = 8192;
    MpfValue left(InputBits), right(InputBits), tail(InputBits), actual(DestinationBits),
        exact(ExactBits), difference(ExactBits);
    const auto precision = static_cast<mp_bitcnt_t>(actual.Get()->_mp_prec);
    const mp_bitcnt_t extraLimbBit = precision * GMP_NUMB_BITS;
    std::mt19937_64 random{0xADD5B0A6D};
    // Exercise actual MPIR operations, including aliasing and its borrow/cancellation
    // branches, against an exact dyadic destination covering every input bit.
    for (unsigned shape = 0; shape < 12; ++shape) {
        for (const bool negativeLeft : {false, true}) {
            for (const bool negativeRight : {false, true}) {
                for (const bool subtract : {false, true}) {
                    for (unsigned alias = 0; alias < 3; ++alias) {
                        mpf_set_ui(left.Get(), 1);
                        mpf_set_ui(right.Get(), 1);
                        mpf_set_ui(tail.Get(), 1);
                        mpf_div_2exp(tail.Get(), tail.Get(), extraLimbBit);
                        switch (shape) {
                            case 0:
                                mpf_set_ui(left.Get(), 0);
                                mpf_set_ui(right.Get(), 0);
                                break;
                            case 1:
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                mpf_set_ui(right.Get(), 0);
                                break;
                            case 2:
                                mpf_add(right.Get(), right.Get(), tail.Get());
                                mpf_set_ui(left.Get(), 0);
                                break;
                            case 3:
                                mpf_mul_2exp(tail.Get(), tail.Get(), GMP_NUMB_BITS);
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                break;
                            case 4:
                                // Fits pi+1 limbs, but not pi: subtraction is exact while
                                // same-sign addition really can discard this final bit.
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                break;
                            case 5:
                                mpf_div_2exp(tail.Get(), tail.Get(), GMP_NUMB_BITS);
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                break;
                            case 6:
                                mpf_div_2exp(tail.Get(), tail.Get(), 2 * GMP_NUMB_BITS);
                                mpf_mul_2exp(left.Get(), left.Get(), GMP_NUMB_BITS);
                                mpf_sub(left.Get(), left.Get(), tail.Get());
                                mpf_set(right.Get(), tail.Get());
                                break;
                            case 7:
                                // Opposite sides of a limb boundary exercise long borrow chains.
                                mpf_div_2exp(tail.Get(), tail.Get(), 2 * GMP_NUMB_BITS);
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                mpf_sub(right.Get(), right.Get(), tail.Get());
                                break;
                            case 8:
                                mpf_div_2exp(right.Get(), right.Get(), 2000);
                                break;
                            case 9:
                                Randomize(left.Get(), left.Get()->_mp_prec + 1, false, random);
                                Randomize(right.Get(), right.Get()->_mp_prec + 1, false, random);
                                break;
                            case 10:
                                Randomize(left.Get(), left.Get()->_mp_prec + 1, false, random);
                                mpf_set(right.Get(), left.Get());
                                break;
                            case 11:
                                mpf_add(left.Get(), left.Get(), tail.Get());
                                mpf_div_2exp(left.Get(), left.Get(), 2000);
                                break;
                        }
                        if (negativeLeft) {
                            mpf_neg(left.Get(), left.Get());
                        }
                        if (negativeRight) {
                            mpf_neg(right.Get(), right.Get());
                        }
                        mpf_srcptr leftInput = left.Get();
                        mpf_srcptr rightInput = right.Get();
                        if (alias == 1) {
                            mpf_set(actual.Get(), leftInput);
                            leftInput = actual.Get();
                        } else if (alias == 2) {
                            mpf_set(actual.Get(), rightInput);
                            rightInput = actual.Get();
                        }
                        const auto bound = subtract
                                               ? FractalShark::RefOrbitPrecisionSchedule::SubtractError(
                                                     leftInput, rightInput, actual.Get())
                                               : FractalShark::RefOrbitPrecisionSchedule::AddError(
                                                     leftInput, rightInput, actual.Get());
                        if (subtract) {
                            mpf_sub(exact.Get(), leftInput, rightInput);
                            mpf_sub(actual.Get(), leftInput, rightInput);
                        } else {
                            mpf_add(exact.Get(), leftInput, rightInput);
                            mpf_add(actual.Get(), leftInput, rightInput);
                        }
                        mpf_sub(difference.Get(), exact.Get(), actual.Get());
                        ASSERT_TRUE(bound.IsFinite());
                        if (bound.IsZero()) {
                            ASSERT_EQ(mpf_sgn(difference.Get()), 0);
                        } else if (mpf_sgn(difference.Get()) != 0) {
                            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
                                bound,
                                FractalShark::RefOrbitErrorBound::FromMpfUpper(difference.Get())));
                        }
                        if (shape == 4) {
                            const bool subtractMagnitudes = subtract == (negativeLeft == negativeRight);
                            ASSERT_EQ(bound.IsZero(), subtractMagnitudes);
                            ASSERT_EQ(mpf_sgn(difference.Get()) == 0, subtractMagnitudes);
                        }
                    }
                }
            }
        }
    }
}

TEST(RefOrbitDynamicPrecision_BoundArithmetic)
{
    // These operands are exact binary64 dyadics. Wide MPIR destinations preserve
    // the complete sums/products even across a 10,000-bit alignment gap.
    MpfValue x(128), y(128), exact(32768), scratch(32768);
    std::mt19937_64 random{0xF1A17E};
    constexpr std::array<int, 8> Gaps{0, 1, 52, 53, 54, 1074, 1075, 10000};
    for (size_t trial = 0; trial < 64; ++trial) {
        const double left = trial == 0 ? std::nextafter(1.0, 0.0)
                                       : std::ldexp(static_cast<double>(random() >> 12), -52) + 1.0;
        const double right =
            trial == 0 ? left : std::ldexp(static_cast<double>(random() >> 12), -52) + 1.0;
        mpf_set_d(x.Get(), left);
        mpf_set_d(y.Get(), right);
        mpf_div_2exp(y.Get(), y.Get(), Gaps[trial % Gaps.size()]);
        if (trial % 2 != 0) {
            mpf_div_2exp(x.Get(), x.Get(), 301);
            mpf_div_2exp(y.Get(), y.Get(), 301);
        }
        const auto xBound = FractalShark::RefOrbitErrorBound::FromMpfLower(x.Get());
        const auto yBound = FractalShark::RefOrbitErrorBound::FromMpfLower(y.Get());
        const auto checkUpper = [&](FractalShark::RefOrbitErrorBound bound) {
            ASSERT_TRUE(bound.IsFinite());
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
                bound, FractalShark::RefOrbitErrorBound::FromMpfUpper(exact.Get())));
        };
        mpf_add(exact.Get(), x.Get(), y.Get());
        checkUpper(FractalShark::RefOrbitErrorBound::Add(xBound, yBound));
        checkUpper(FractalShark::RefOrbitErrorBound::Add(yBound, xBound));
        mpf_mul(exact.Get(), x.Get(), y.Get());
        checkUpper(FractalShark::RefOrbitErrorBound::Multiply(xBound, yBound));
        mpf_mul(exact.Get(), x.Get(), x.Get());
        mpf_mul(scratch.Get(), y.Get(), y.Get());
        mpf_add(exact.Get(), exact.Get(), scratch.Get());
        mpf_sqrt(exact.Get(), exact.Get());
        checkUpper(FractalShark::RefOrbitErrorBound::Norm(xBound, yBound));
        for (const int shift : {-10001, -1, 0, 1, 10001}) {
            mpf_set(exact.Get(), x.Get());
            if (shift < 0) {
                mpf_div_2exp(exact.Get(), exact.Get(), -shift);
            } else {
                mpf_mul_2exp(exact.Get(), exact.Get(), shift);
            }
            const auto scaled = FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo(xBound, shift);
            const auto expected = FractalShark::RefOrbitErrorBound::FromMpfLower(exact.Get());
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(scaled, expected));
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(expected, scaled));
        }
        ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
            xBound, FractalShark::RefOrbitErrorBound::Add({}, xBound)));
        ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
            xBound, FractalShark::RefOrbitErrorBound::Add(xBound, {})));
    }
    const auto invalid =
        FractalShark::RefOrbitErrorBound::PowerOfTwo(std::numeric_limits<int64_t>::max());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Add({}, {}).IsZero());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Multiply({}, {}).IsZero());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Norm({}, {}).IsZero());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo({}, 1).IsZero());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Add(invalid, {}).IsFinite());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Add({}, invalid).IsFinite());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Multiply({}, invalid).IsFinite());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Norm(invalid, {}).IsFinite());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo(invalid, -1).IsFinite());
    constexpr int64_t ExponentLimit = std::numeric_limits<int64_t>::max() / 8;
    // Zero must sort below even the smallest supported positive bound. Exercise
    // both operand orders and the shared zero path without adding an outward ulp.
    for (const auto exponent : {-ExponentLimit, int64_t{-100000}, int64_t{0}, ExponentLimit - 1}) {
        const auto value = FractalShark::RefOrbitErrorBound::PowerOfTwo(exponent);
        for (const auto sum : {FractalShark::RefOrbitErrorBound::Add({}, value),
                               FractalShark::RefOrbitErrorBound::Add(value, {})}) {
            ASSERT_TRUE(sum.IsFinite());
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(sum, value));
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(value, sum));
        }
        ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less({}, value));
        ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(value, {}));
    }
    const auto largest = FractalShark::RefOrbitErrorBound::PowerOfTwo(ExponentLimit - 1);
    ASSERT_TRUE(largest.IsFinite());
    ASSERT_FALSE(FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo(largest, 1).IsFinite());
    ASSERT_FALSE(
        FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo(largest, std::numeric_limits<int64_t>::max())
            .IsFinite());
}

TEST(RefOrbitDynamicPrecision_CombinedStepBounds)
{
    // Compare the complete step with exact dyadic arithmetic rather than duplicating
    // the seven estimates. This checks that metadata is captured before overwrites,
    // including near cancellation, zero components and very different operand scales.
    MpfValue x(128), y(128), cx(512), cy(512), xSquared(128), ySquared(128), cross(128), temporary(128);
    MpfValue exactX(8192), exactY(8192), scratch(8192), differenceX(8192), differenceY(8192);
    FractalShark::RefOrbitKaratsubaProductsT<true> products(128);
    std::mt19937_64 random{0x57E9B0A6D};
    for (size_t trial = 0; trial < 96; ++trial) {
        Randomize(x.Get(), x.Get()->_mp_prec + 1, trial % 2 != 0, random);
        Randomize(y.Get(), y.Get()->_mp_prec + 1, trial % 3 != 0, random);
        Randomize(cx.Get(), cx.Get()->_mp_prec + 1, trial % 2 == 0, random);
        Randomize(cy.Get(), cy.Get()->_mp_prec + 1, trial % 3 == 0, random);
        if (trial % 8 == 0) {
            mpf_set(y.Get(), x.Get());
            mpf_div_2exp(cx.Get(), cx.Get(), 2000);
        } else if (trial % 8 == 1) {
            mpf_set_ui(y.Get(), 0);
        } else if (trial % 8 == 2) {
            mpf_set_ui(x.Get(), 0);
        } else if (trial % 8 == 3) {
            for (auto *value : {x.Get(), y.Get(), cx.Get(), cy.Get()}) {
                mpf_set_ui(value, 0);
            }
        }
        mpf_mul(exactX.Get(), x.Get(), x.Get());
        mpf_mul(scratch.Get(), y.Get(), y.Get());
        mpf_sub(exactX.Get(), exactX.Get(), scratch.Get());
        mpf_add(exactX.Get(), exactX.Get(), cx.Get());
        mpf_mul(exactY.Get(), x.Get(), y.Get());
        mpf_mul_ui(exactY.Get(), exactY.Get(), 2);
        mpf_add(exactY.Get(), exactY.Get(), cy.Get());
        FractalShark::RefOrbitPrecisionSchedule schedule(128, cx.Get(), cy.Get());
        schedule.EnableDiagnosticCapture(true);
        schedule.BeginStep(x.Get(), y.Get(), false);
        products.Compute(xSquared.Get(), ySquared.Get(), cross.Get(), x.Get(), y.Get());
        schedule.ComputeStep(x.Get(),
                             y.Get(),
                             xSquared.Get(),
                             ySquared.Get(),
                             cross.Get(),
                             temporary.Get(),
                             cx.Get(),
                             cy.Get());
        mpf_sub(differenceX.Get(), exactX.Get(), x.Get());
        mpf_sub(differenceY.Get(), exactY.Get(), y.Get());
        mpf_mul(differenceX.Get(), differenceX.Get(), differenceX.Get());
        mpf_mul(differenceY.Get(), differenceY.Get(), differenceY.Get());
        mpf_add(scratch.Get(), differenceX.Get(), differenceY.Get());
        mpf_sqrt(scratch.Get(), scratch.Get());
        ASSERT_TRUE(schedule.GetError().IsFinite());
        const auto &step = schedule.GetStepDiagnostics();
        const auto localBound = FractalShark::RefOrbitErrorBound::Norm(step.LocalX, step.LocalY);
        if (mpf_sgn(scratch.Get()) != 0) {
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
                localBound, FractalShark::RefOrbitErrorBound::FromMpfUpper(scratch.Get())));
        }
        // The first full-width step is shared with fixed MT9 even when its
        // mathematical local residual is nonzero.
        ASSERT_TRUE(schedule.GetError().IsZero());
        if (trial % 8 == 3) {
            ASSERT_TRUE(schedule.GetError().IsZero());
        }
    }
}

TEST(RefOrbitDynamicPrecision_FiniteRecurrence)
{
    MpfValue x(128), y(128), zero(128), radius(8192), scratch(8192), expected(8192), error(8192);
    MpfValue localX(128), localY(128);
    for (size_t trial = 0; trial < 32; ++trial) {
        mpf_set_ui(x.Get(), trial % 5);
        mpf_set_ui(y.Get(), trial % 7);
        const auto errorBits = static_cast<mp_bitcnt_t>(50 + trial * 3);
        mpf_set_ui(error.Get(), 1);
        mpf_div_2exp(error.Get(), error.Get(), errorBits);
        mpf_set_ui(localX.Get(), 1);
        mpf_div_2exp(localX.Get(), localX.Get(), 200 + trial);
        mpf_set_ui(localY.Get(), trial % 2);
        mpf_div_2exp(localY.Get(), localY.Get(), 300 + trial);
        FractalShark::RefOrbitPrecisionSchedule schedule(128, zero.Get(), zero.Get());
        schedule.EnableDiagnosticCapture(true);
        schedule.AddStateTruncation(FractalShark::RefOrbitErrorBound::FromMpfLower(error.Get()), {});
        schedule.BeginStep(x.Get(), y.Get(), false);
        schedule.FinishStep(FractalShark::RefOrbitErrorBound::FromMpfLower(localX.Get()),
                            FractalShark::RefOrbitErrorBound::FromMpfLower(localY.Get()));
        // A separate wide calculation evaluates 2*|z|*E + E^2 + |local|.
        mpf_mul(radius.Get(), x.Get(), x.Get());
        mpf_mul(scratch.Get(), y.Get(), y.Get());
        mpf_add(radius.Get(), radius.Get(), scratch.Get());
        mpf_sqrt(radius.Get(), radius.Get());
        mpf_mul(expected.Get(), radius.Get(), error.Get());
        mpf_mul_ui(expected.Get(), expected.Get(), 2);
        mpf_mul(scratch.Get(), error.Get(), error.Get());
        mpf_add(expected.Get(), expected.Get(), scratch.Get());
        mpf_mul(radius.Get(), localX.Get(), localX.Get());
        mpf_mul(scratch.Get(), localY.Get(), localY.Get());
        mpf_add(radius.Get(), radius.Get(), scratch.Get());
        mpf_sqrt(radius.Get(), radius.Get());
        mpf_add(expected.Get(), expected.Get(), radius.Get());
        const auto fixedAllowance = schedule.GetStepDiagnostics().FixedRoundingAllowance;
        if (!fixedAllowance.IsZero()) {
            const auto allowanceExponent = static_cast<int64_t>(fixedAllowance.Log2());
            mpf_set_ui(scratch.Get(), 1);
            if (allowanceExponent < 0) {
                mpf_div_2exp(scratch.Get(), scratch.Get(), static_cast<mp_bitcnt_t>(-allowanceExponent));
            } else {
                mpf_mul_2exp(scratch.Get(), scratch.Get(), static_cast<mp_bitcnt_t>(allowanceExponent));
            }
            mpf_add(expected.Get(), expected.Get(), scratch.Get());
        }
        ASSERT_TRUE(schedule.GetError().IsFinite());
        ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
            schedule.GetError(), FractalShark::RefOrbitErrorBound::FromMpfUpper(expected.Get())));
    }
    FractalShark::RefOrbitPrecisionSchedule invalid(128, zero.Get(), zero.Get());
    invalid.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-64), {});
    invalid.BeginStep(zero.Get(), zero.Get(), false);
    invalid.FinishStep(
        {}, FractalShark::RefOrbitErrorBound::PowerOfTwo(std::numeric_limits<int64_t>::max()));
    invalid.FinishStep({}, {});
    ASSERT_FALSE(invalid.GetError().IsFinite());
    ASSERT_FALSE(invalid.Certify(zero.Get(), zero.Get(), true));
    ASSERT_TRUE(invalid.GetStats().Failure == FractalShark::RefOrbitPrecisionFailure::InvalidBound);
    const auto invalidWorst = invalid.DescribeWorstCertification();
    ASSERT_TRUE(invalidWorst.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::InvalidBound);
    ASSERT_TRUE(invalidWorst.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::InvalidBound);
    ASSERT_EQ(invalidWorst.Real.StateIteration, uint64_t{2});
    ASSERT_EQ(invalidWorst.Imaginary.StateIteration, uint64_t{2});
    FractalShark::RefOrbitPrecisionSchedule overflow(128, zero.Get(), zero.Get());
    constexpr int64_t ExponentLimit = std::numeric_limits<int64_t>::max() / 8;
    overflow.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(ExponentLimit - 1), {});
    ASSERT_TRUE(overflow.GetError().IsFinite());
    overflow.BeginStep(zero.Get(), zero.Get(), false);
    overflow.FinishStep({}, {});
    ASSERT_FALSE(overflow.GetError().IsFinite());
    // A nonzero intermediate below the supported range must survive addition to
    // zero and reach the final range check, rather than being aligned out of sight.
    FractalShark::RefOrbitPrecisionSchedule underflow(128, zero.Get(), zero.Get());
    underflow.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-ExponentLimit), {});
    ASSERT_TRUE(underflow.GetError().IsFinite());
    underflow.BeginStep(zero.Get(), zero.Get(), false);
    underflow.FinishStep({}, {});
    ASSERT_FALSE(underflow.GetError().IsFinite());
}

TEST(RefOrbitDynamicPrecision_ConversionCertificates)
{
    MpfValue x(1024), y(1024), offset(1024);
    for (const int sign : {-1, 1}) {
        mpf_set_ui(offset.Get(), 1);
        mpf_div_2exp(offset.Get(), offset.Get(), 900);
        mpf_set_ui(x.Get(), 1);
        mpf_add(x.Get(), x.Get(), offset.Get());
        if (sign < 0) {
            mpf_neg(x.Get(), x.Get());
        }
        FractalShark::RefOrbitPrecisionSchedule good(1024, x.Get(), y.Get());
        good.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-960), {});
        ASSERT_TRUE(good.Certify(x.Get(), y.Get(), true));
        FractalShark::RefOrbitPrecisionSchedule ambiguous(1024, x.Get(), y.Get());
        ambiguous.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-899), {});
        ASSERT_FALSE(ambiguous.Certify(x.Get(), y.Get(), true));
        ASSERT_TRUE(ambiguous.GetStats().Failure ==
                    FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
    }
    mpf_set_ui(x.Get(), 1);
    mpf_div_2exp(x.Get(), x.Get(), 2000);
    FractalShark::RefOrbitPrecisionSchedule nativeUnderflow(1024, x.Get(), y.Get());
    ASSERT_FALSE(nativeUnderflow.Certify(x.Get(), y.Get(), true));
    ASSERT_TRUE(nativeUnderflow.GetStats().Failure ==
                FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion);
    FractalShark::RefOrbitPrecisionSchedule hdr(1024, x.Get(), y.Get());
    ASSERT_TRUE(hdr.Certify(x.Get(), y.Get(), false));
}

TEST(RefOrbitDynamicPrecision_FastCertificateMpirOracle)
{
    constexpr mp_bitcnt_t Bits = 1024;
    constexpr mp_bitcnt_t OracleBits = Bits + 256;
    constexpr int ErrorOffsetBits = 100;
    MpfValue value(Bits), zero(Bits), magnitude(OracleBits), boundary(OracleBits), lowerGap(OracleBits),
        upperGap(OracleBits);
    std::mt19937_64 random{0x8bd03461};
    uint64_t fastChecks = 0;
    uint64_t regularChecks = 0;
    // Vary every leading-zero alignment and occupied width, including one-limb
    // states whose missing trailing limbs must not be read from spare capacity.
    for (int leadingZeros = 0; leadingZeros < GMP_NUMB_BITS; ++leadingZeros) {
        for (mp_size_t occupied = 1; occupied <= 4; ++occupied) {
            for (const bool negative : {false, true}) {
                Randomize(value.Get(), occupied, negative, random);
                const mp_limb_t leadingBit = mp_limb_t{1} << (GMP_NUMB_BITS - 1 - leadingZeros);
                value.Get()->_mp_d[occupied - 1] =
                    (value.Get()->_mp_d[occupied - 1] >> leadingZeros) | leadingBit;
                double mantissa;
                const int64_t exponent = mpf_get_2exp_d(&mantissa, value.Get());
                FractalShark::RefOrbitPrecisionSchedule schedule(Bits, value.Get(), zero.Get());
                schedule.AddStateTruncation(
                    FractalShark::RefOrbitErrorBound::PowerOfTwo(exponent - ErrorOffsetBits), {});
                mpf_abs(magnitude.Get(), value.Get());
                const auto setBoundary = [&](double endpoint) {
                    mpf_set_d(boundary.Get(), endpoint);
                    if (exponent < 0) {
                        mpf_div_2exp(
                            boundary.Get(), boundary.Get(), static_cast<mp_bitcnt_t>(-exponent));
                    } else {
                        mpf_mul_2exp(boundary.Get(), boundary.Get(), static_cast<mp_bitcnt_t>(exponent));
                    }
                };
                // Independently subtract the complete state from both exact
                // dyadic boundaries; this oracle never extracts the fast prefix.
                setBoundary(std::abs(mantissa));
                mpf_sub(lowerGap.Get(), magnitude.Get(), boundary.Get());
                setBoundary(std::nextafter(std::abs(mantissa), std::numeric_limits<double>::infinity()));
                mpf_sub(upperGap.Get(), boundary.Get(), magnitude.Get());
                const bool expected =
                    mpf_sgn(lowerGap.Get()) > 0 && mpf_sgn(upperGap.Get()) > 0 &&
                    FractalShark::RefOrbitErrorBound::Less(
                        schedule.GetError(),
                        FractalShark::RefOrbitErrorBound::FromMpfLower(lowerGap.Get())) &&
                    FractalShark::RefOrbitErrorBound::Less(
                        schedule.GetError(),
                        FractalShark::RefOrbitErrorBound::FromMpfLower(upperGap.Get()));
                const auto diagnostic = schedule.DescribeCertification(value.Get(), zero.Get(), false);
                ASSERT_EQ(diagnostic.Failure == FractalShark::RefOrbitPrecisionFailure::None, expected);
                ASSERT_EQ(schedule.GetStats().FastCertifiedComponents, uint64_t{0});
                ASSERT_EQ(schedule.GetStats().RegularCertificationComponents, uint64_t{0});
                ASSERT_EQ(schedule.Certify(value.Get(), zero.Get(), false), expected);
                const auto &stats = schedule.GetStats();
                ASSERT_EQ(stats.FastCertifiedComponents + stats.RegularCertificationComponents,
                          uint64_t{1});
                fastChecks += stats.FastCertifiedComponents;
                regularChecks += stats.RegularCertificationComponents;
            }
        }
    }
    if constexpr (GMP_NUMB_BITS == 64 && GMP_NAIL_BITS == 0) {
        ASSERT_TRUE(fastChecks > 0);
    } else {
        ASSERT_EQ(fastChecks, uint64_t{0});
    }
    ASSERT_TRUE(regularChecks > 0);
}

TEST(RefOrbitDynamicPrecision_FastCertificateRoutesAndThresholds)
{
    constexpr mp_bitcnt_t Bits = 1024;
    MpfValue x(Bits), y(Bits), zero(Bits), offset(Bits), ulp(Bits);
    mpf_set_ui(offset.Get(), 1);
    mpf_div_2exp(offset.Get(), offset.Get(), 60);
    mpf_add_ui(x.Get(), offset.Get(), 1);
    mpf_set_ui(offset.Get(), 1);
    mpf_div_2exp(offset.Get(), offset.Get(), 900);
    mpf_add_ui(y.Get(), offset.Get(), 1);
    FractalShark::RefOrbitPrecisionSchedule mixed(Bits, x.Get(), y.Get());
    mixed.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-960), {});
    ASSERT_TRUE(mixed.Certify(x.Get(), y.Get(), false));
    const uint64_t expectedFast = GMP_NUMB_BITS == 64 && GMP_NAIL_BITS == 0 ? 1 : 0;
    ASSERT_EQ(mixed.GetStats().FastCertifiedComponents, expectedFast);
    ASSERT_EQ(mixed.GetStats().RegularCertificationComponents, uint64_t{2} - expectedFast);
    const auto before = mixed.GetStats();
    const auto margin = mixed.DescribeCertification(x.Get(), y.Get(), false);
    const auto worst = mixed.DescribeWorstCertification();
    ASSERT_NEAR(margin.HeadroomBits, 60.0, 1e-9);
    ASSERT_TRUE(worst.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::Bounded);
    ASSERT_TRUE(worst.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::Bounded);
    ASSERT_NEAR(worst.Imaginary.MaxBits - worst.Imaginary.UsedBits, 60.0, 1e-9);
    ASSERT_EQ(worst.Real.StateIteration, uint64_t{1});
    ASSERT_EQ(worst.Imaginary.StateIteration, uint64_t{1});
    ASSERT_EQ(mixed.GetStats().FastCertifiedComponents, before.FastCertifiedComponents);
    ASSERT_EQ(mixed.GetStats().RegularCertificationComponents, before.RegularCertificationComponents);
    ASSERT_TRUE(mixed.Certify(x.Get(), y.Get(), false));
    ASSERT_EQ(mixed.GetStats().FastCertifiedComponents, 2 * expectedFast);
    ASSERT_EQ(mixed.GetStats().RegularCertificationComponents, 2 * (uint64_t{2} - expectedFast));
    FractalShark::RefOrbitPrecisionSchedule fresh(Bits, x.Get(), y.Get());
    ASSERT_EQ(fresh.GetStats().FastCertifiedComponents, uint64_t{0});
    ASSERT_EQ(fresh.GetStats().RegularCertificationComponents, uint64_t{0});

    mpf_set_ui(ulp.Get(), 1);
    mpf_div_2exp(ulp.Get(), ulp.Get(), std::numeric_limits<double>::digits - 1);
    mpf_set_ui(offset.Get(), 1);
    mpf_div_2exp(offset.Get(), offset.Get(), 80);
    for (const bool nearUpper : {false, true}) {
        for (const bool negative : {false, true}) {
            mpf_set_ui(x.Get(), 1);
            if (nearUpper) {
                mpf_add(x.Get(), x.Get(), ulp.Get());
                mpf_sub(x.Get(), x.Get(), offset.Get());
            } else {
                mpf_add(x.Get(), x.Get(), offset.Get());
            }
            if (negative) {
                mpf_neg(x.Get(), x.Get());
            }
            for (const int64_t errorExponent : {-81, -80, -79}) {
                FractalShark::RefOrbitPrecisionSchedule threshold(Bits, x.Get(), zero.Get());
                threshold.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(errorExponent),
                                             {});
                const bool expected = errorExponent < -80;
                ASSERT_EQ(threshold.Certify(x.Get(), zero.Get(), false), expected);
                ASSERT_EQ(threshold.GetStats().FastCertifiedComponents, expected ? expectedFast : 0);
                ASSERT_EQ(threshold.GetStats().RegularCertificationComponents,
                          uint64_t{1} - (expected ? expectedFast : 0));
            }
        }
    }
    // Supported zero-error states are fast even on a conversion boundary; a
    // nonzero enclosure at that same boundary must still reach regular rejection.
    mpf_set_ui(x.Get(), 1);
    FractalShark::RefOrbitPrecisionSchedule exact(Bits, x.Get(), zero.Get());
    ASSERT_TRUE(exact.Certify(x.Get(), zero.Get(), false));
    ASSERT_EQ(exact.GetStats().FastCertifiedComponents, expectedFast);
    FractalShark::RefOrbitPrecisionSchedule rejected(Bits, x.Get(), y.Get());
    rejected.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-960), {});
    ASSERT_FALSE(rejected.Certify(x.Get(), y.Get(), false));
    ASSERT_EQ(rejected.GetStats().FastCertifiedComponents, uint64_t{0});
    ASSERT_EQ(rejected.GetStats().RegularCertificationComponents, uint64_t{1});
    ASSERT_TRUE(rejected.GetStats().Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
}

TEST(RefOrbitDynamicPrecision_FastCertificateExponentLimits)
{
    constexpr mp_bitcnt_t Bits = 1024;
    MpfValue base(Bits), value(Bits), zero(Bits), offset(Bits);
    mpf_set_ui(offset.Get(), 1);
    mpf_div_2exp(offset.Get(), offset.Get(), 60);
    mpf_add_ui(base.Get(), offset.Get(), 1);
    const std::array<int64_t, 8> exponents{
        -1022,
        -1021,
        1024,
        1025,
        static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - 1,
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(),
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1};
    for (const int64_t exponent : exponents) {
        mpf_set(value.Get(), base.Get());
        if (exponent < 1) {
            mpf_div_2exp(value.Get(), value.Get(), static_cast<mp_bitcnt_t>(1 - exponent));
        } else {
            mpf_mul_2exp(value.Get(), value.Get(), static_cast<mp_bitcnt_t>(exponent - 1));
        }
        for (const bool nativeDouble : {false, true}) {
            FractalShark::RefOrbitPrecisionSchedule schedule(Bits, value.Get(), zero.Get());
            schedule.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(exponent - 100),
                                        {});
            const bool supported =
                exponent >= std::numeric_limits<int32_t>::min() &&
                exponent <= std::numeric_limits<int32_t>::max() &&
                (!nativeDouble || (exponent >= std::numeric_limits<double>::min_exponent &&
                                   exponent <= std::numeric_limits<double>::max_exponent));
            ASSERT_EQ(schedule.Certify(value.Get(), zero.Get(), nativeDouble), supported);
            ASSERT_EQ(schedule.DescribeCertification(value.Get(), zero.Get(), nativeDouble).Failure ==
                          FractalShark::RefOrbitPrecisionFailure::None,
                      supported);
            ASSERT_TRUE(schedule.GetStats().Failure ==
                        (supported ? FractalShark::RefOrbitPrecisionFailure::None
                                   : FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion));
        }
    }
}

TEST(RefOrbitDynamicPrecision_CertificationHeadroom)
{
    constexpr mp_bitcnt_t bits = 1024;
    constexpr mp_bitcnt_t BoundaryOffsetBits = 900;
    constexpr int64_t SafeErrorBits = 960;
    MpfValue x(bits), y(bits), offset(bits), ulp(bits);
    mpf_set_ui(offset.Get(), 1);
    mpf_div_2exp(offset.Get(), offset.Get(), BoundaryOffsetBits);
    mpf_set_ui(ulp.Get(), 1);
    mpf_div_2exp(ulp.Get(), ulp.Get(), std::numeric_limits<double>::digits - 1);
    for (const int sign : {-1, 1}) {
        for (const bool nearUpper : {false, true}) {
            mpf_set_ui(x.Get(), 1);
            if (nearUpper) {
                mpf_add(x.Get(), x.Get(), ulp.Get());
                mpf_sub(x.Get(), x.Get(), offset.Get());
            } else {
                mpf_add(x.Get(), x.Get(), offset.Get());
            }
            if (sign < 0) {
                mpf_neg(x.Get(), x.Get());
            }
            FractalShark::RefOrbitPrecisionSchedule safe(bits, x.Get(), y.Get());
            safe.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-SafeErrorBits), {});
            const auto margin = safe.DescribeCertification(x.Get(), y.Get(), true);
            ASSERT_TRUE(margin.Failure == FractalShark::RefOrbitPrecisionFailure::None);
            ASSERT_TRUE(std::string_view{margin.Component} == "real");
            ASSERT_TRUE(std::string_view{margin.Boundary} ==
                        (nearUpper ? "upper_magnitude" : "lower_magnitude"));
            ASSERT_EQ(margin.GapLowerBoundLog2, -static_cast<double>(BoundaryOffsetBits));
            // Gap = 2^-900 and error approximately 2^-960 give sixty bits of headroom.
            // Tiny outward allowances remain included in the reported comparison.
            ASSERT_NEAR(margin.HeadroomBits, 60.0, 1e-9);
            ASSERT_NEAR(margin.ComparisonErrorBoundLog2, -static_cast<double>(SafeErrorBits), 1e-9);
            // The component exponent is one: its full precision unit is 2^-1023.
            ASSERT_TRUE(margin.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::Bounded);
            ASSERT_NEAR(margin.Real.UsedBits, 63.0, 1e-9);
            ASSERT_NEAR(margin.Real.MaxBits, 123.0, 1e-9);
            ASSERT_TRUE(margin.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::Exact);
            ASSERT_TRUE(safe.Certify(x.Get(), y.Get(), true));
            ASSERT_EQ(safe.GetStats().FastCertifiedComponents, uint64_t{0});
            ASSERT_EQ(safe.GetStats().RegularCertificationComponents, uint64_t{1});

            FractalShark::RefOrbitPrecisionSchedule rejected(bits, x.Get(), y.Get());
            rejected.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(
                                            -static_cast<int64_t>(BoundaryOffsetBits) + 1),
                                        {});
            const auto failedMargin = rejected.DescribeCertification(x.Get(), y.Get(), true);
            ASSERT_TRUE(failedMargin.Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
            ASSERT_NEAR(failedMargin.HeadroomBits, -1.0, 1e-9);
            ASSERT_TRUE(failedMargin.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::Bounded);
            ASSERT_NEAR(failedMargin.Real.UsedBits, 124.0, 1e-9);
            ASSERT_NEAR(failedMargin.Real.MaxBits, 123.0, 1e-9);
            // Describing an unused final pipeline state must not change orbit validity.
            ASSERT_TRUE(rejected.GetStats().Failure == FractalShark::RefOrbitPrecisionFailure::None);
            ASSERT_FALSE(rejected.Certify(x.Get(), y.Get(), true));
            const auto rejectedWorst = rejected.DescribeWorstCertification();
            ASSERT_NEAR(rejectedWorst.Real.UsedBits, 124.0, 1e-9);
            ASSERT_NEAR(rejectedWorst.Real.MaxBits, 123.0, 1e-9);
            ASSERT_EQ(rejectedWorst.Real.StateIteration, uint64_t{1});
            ASSERT_TRUE(rejectedWorst.Imaginary.Status ==
                        FractalShark::RefOrbitErrorBudgetStatus::NotChecked);
        }
    }

    // Zero error passes without a boundary test, even exactly on a conversion boundary.
    mpf_set_ui(x.Get(), 1);
    FractalShark::RefOrbitPrecisionSchedule exact(bits, x.Get(), y.Get());
    const auto exactMargin = exact.DescribeCertification(x.Get(), y.Get(), true);
    ASSERT_TRUE(exactMargin.Failure == FractalShark::RefOrbitPrecisionFailure::None);
    ASSERT_EQ(exactMargin.HeadroomBits, std::numeric_limits<double>::infinity());
    ASSERT_TRUE(exactMargin.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::Exact);
    ASSERT_TRUE(exactMargin.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::Exact);
    exact.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-SafeErrorBits), {});
    const auto boundaryMargin = exact.DescribeCertification(x.Get(), y.Get(), true);
    ASSERT_TRUE(boundaryMargin.Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
    ASSERT_EQ(boundaryMargin.HeadroomBits, -std::numeric_limits<double>::infinity());
    ASSERT_EQ(boundaryMargin.GapLowerBoundLog2, -std::numeric_limits<double>::infinity());
    ASSERT_TRUE(boundaryMargin.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::ZeroGap);
    ASSERT_TRUE(boundaryMargin.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::Exact);

    // A conversion outside the supported native range has no numeric gap threshold.
    mpf_div_2exp(x.Get(), x.Get(), 2000);
    FractalShark::RefOrbitPrecisionSchedule unsupported(bits, x.Get(), y.Get());
    const auto unsupportedMargin = unsupported.DescribeCertification(x.Get(), y.Get(), true);
    ASSERT_TRUE(unsupportedMargin.Failure ==
                FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion);
    ASSERT_TRUE(std::isnan(unsupportedMargin.HeadroomBits));
    ASSERT_TRUE(unsupportedMargin.Real.Status ==
                FractalShark::RefOrbitErrorBudgetStatus::UnsupportedConversion);
    ASSERT_TRUE(unsupported.GetStats().Failure == FractalShark::RefOrbitPrecisionFailure::None);
}

TEST(RefOrbitDynamicPrecision_WorstCertificationMargins)
{
    constexpr mp_bitcnt_t bits = 1024;
    constexpr int64_t SafeErrorBits = 960;
    constexpr mp_bitcnt_t RealGapBits = 900;
    constexpr mp_bitcnt_t LooseImaginaryGapBits = 850;
    constexpr mp_bitcnt_t TightImaginaryGapBits = 950;
    constexpr mp_bitcnt_t ImaginaryScaleBits = 40;
    constexpr int64_t ContractedErrorBits = 1000;
    constexpr int64_t SmallerErrorBits = 1100;
    // Exercise cancellation of large absolute exponents without leaving HDR's range.
    constexpr int64_t ExponentScale = 1'000'000'000;
    MpfValue x(bits), y(bits), zero(bits), offset(bits), error(bits);
    const auto setNearBoundary = [&](mpf_ptr value, mp_bitcnt_t offsetBits, int64_t scale) {
        mpf_set_ui(offset.Get(), 1);
        mpf_div_2exp(offset.Get(), offset.Get(), offsetBits);
        mpf_add_ui(value, offset.Get(), 1);
        if (scale < 0) {
            mpf_div_2exp(value, value, static_cast<mp_bitcnt_t>(-scale));
        } else {
            mpf_mul_2exp(value, value, static_cast<mp_bitcnt_t>(scale));
        }
    };

    // Known dyadic gaps give independent numeric expectations. Scaling both
    // components and the shared error must leave the normalized budgets unchanged.
    for (const int64_t scale : {-ExponentScale, int64_t{0}, ExponentScale}) {
        setNearBoundary(x.Get(), RealGapBits, scale);
        setNearBoundary(
            y.Get(), LooseImaginaryGapBits, scale - static_cast<int64_t>(ImaginaryScaleBits));
        FractalShark::RefOrbitPrecisionSchedule schedule(bits, x.Get(), y.Get());
        const auto empty = schedule.DescribeWorstCertification();
        ASSERT_TRUE(empty.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::NotChecked);
        ASSERT_TRUE(empty.Imaginary.Status == FractalShark::RefOrbitErrorBudgetStatus::NotChecked);
        ASSERT_EQ(empty.Real.StateIteration, uint64_t{0});
        ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
        ASSERT_TRUE(schedule.DescribeWorstCertification().Real.Status ==
                    FractalShark::RefOrbitErrorBudgetStatus::Exact);
        schedule.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(scale - SafeErrorBits),
                                    {});
        ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
        const auto worst = schedule.DescribeWorstCertification();
        ASSERT_NEAR(worst.Real.UsedBits, 63.0, 1e-9);
        ASSERT_NEAR(worst.Real.MaxBits, 123.0, 1e-9);
        ASSERT_NEAR(worst.Imaginary.UsedBits, 103.0, 1e-9);
        ASSERT_NEAR(worst.Imaginary.MaxBits, 173.0, 1e-9);
        ASSERT_EQ(worst.Real.StateIteration, uint64_t{1});
        ASSERT_EQ(worst.Imaginary.StateIteration, uint64_t{1});
    }

    // Drive synthetic certificate states to isolate record selection from orbit
    // dynamics. Zero-radius recurrence updates supply controlled contractions.
    setNearBoundary(x.Get(), RealGapBits, 0);
    setNearBoundary(y.Get(), LooseImaginaryGapBits, -static_cast<int64_t>(ImaginaryScaleBits));
    // Keep the synthetic coordinate scale below the prescribed residuals while
    // retaining a nonreal coordinate so both component budgets are checked.
    FractalShark::RefOrbitPrecisionSchedule schedule(bits, zero.Get(), y.Get());
    schedule.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-SafeErrorBits), {});
    ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
    schedule.BeginStep(zero.Get(), zero.Get(), false);
    schedule.FinishStep(FractalShark::RefOrbitErrorBound::PowerOfTwo(-ContractedErrorBits), {});
    setNearBoundary(y.Get(), TightImaginaryGapBits, -static_cast<int64_t>(ImaginaryScaleBits));
    ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
    const auto tight = schedule.DescribeWorstCertification();
    ASSERT_EQ(tight.Real.StateIteration, uint64_t{1});
    ASSERT_EQ(tight.Imaginary.StateIteration, uint64_t{2});
    ASSERT_NEAR(tight.Real.MaxBits - tight.Real.UsedBits, 60.0, 1e-9);
    ASSERT_NEAR(tight.Imaginary.MaxBits - tight.Imaginary.UsedBits, 10.0, 1e-9);
    const double tightErrorLog2 = schedule.GetError().Log2();

    schedule.BeginStep(zero.Get(), zero.Get(), false);
    schedule.FinishStep(FractalShark::RefOrbitErrorBound::PowerOfTwo(-SmallerErrorBits), {});
    setNearBoundary(y.Get(), LooseImaginaryGapBits, -static_cast<int64_t>(ImaginaryScaleBits));
    ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
    ASSERT_EQ(schedule.DescribeWorstCertification().Real.StateIteration, uint64_t{1});
    ASSERT_EQ(schedule.DescribeWorstCertification().Imaginary.StateIteration, uint64_t{2});
    const auto final = schedule.DescribeCertification(x.Get(), y.Get(), false);
    ASSERT_TRUE(final.Real.UsedBits < 0.0);
    ASSERT_TRUE(final.Real.UsedBits < tight.Real.UsedBits);

    // Recreate the same outward-rounded pressure at Z4: equal records retain Z2.
    schedule.BeginStep(zero.Get(), zero.Get(), false);
    schedule.FinishStep(FractalShark::RefOrbitErrorBound::PowerOfTwo(-ContractedErrorBits), {});
    setNearBoundary(y.Get(), TightImaginaryGapBits, -static_cast<int64_t>(ImaginaryScaleBits));
    ASSERT_EQ(schedule.GetError().Log2(), tightErrorLog2);
    ASSERT_TRUE(schedule.Certify(x.Get(), y.Get(), false));
    ASSERT_EQ(schedule.DescribeWorstCertification().Imaginary.StateIteration, uint64_t{2});

    // A real-component failure replaces its earlier finite record without
    // claiming that the imaginary component was checked again after rejection.
    mpf_set_ui(x.Get(), 1);
    ASSERT_FALSE(schedule.Certify(x.Get(), y.Get(), false));
    const auto failed = schedule.DescribeWorstCertification();
    ASSERT_TRUE(failed.Real.Status == FractalShark::RefOrbitErrorBudgetStatus::ZeroGap);
    ASSERT_EQ(failed.Real.StateIteration, uint64_t{4});
    ASSERT_EQ(failed.Imaginary.StateIteration, uint64_t{2});

    // Gap = 1.5 fast-tail quanta and error = 1.25 quanta. The fast bound is
    // inconclusive, but regular certification passes with log2(1.2) headroom.
    constexpr mp_bitcnt_t FastTailQuantumBits = std::numeric_limits<double>::digits - 1 + 32;
    mpf_set_d(offset.Get(), 1.5);
    mpf_div_2exp(offset.Get(), offset.Get(), FastTailQuantumBits);
    mpf_add_ui(x.Get(), offset.Get(), 1);
    mpf_set_d(error.Get(), 1.25);
    mpf_div_2exp(error.Get(), error.Get(), FastTailQuantumBits);
    FractalShark::RefOrbitPrecisionSchedule regular(bits, x.Get(), zero.Get());
    regular.AddStateTruncation(FractalShark::RefOrbitErrorBound::FromMpfLower(error.Get()), {});
    ASSERT_TRUE(regular.Certify(x.Get(), zero.Get(), false));
    const auto regularWorst = regular.DescribeWorstCertification();
    ASSERT_EQ(regular.GetStats().FastCertifiedComponents, uint64_t{0});
    ASSERT_EQ(regular.GetStats().RegularCertificationComponents, uint64_t{1});
    ASSERT_TRUE(regularWorst.Real.UsedBits < regularWorst.Real.MaxBits);
    ASSERT_NEAR(regularWorst.Real.MaxBits - regularWorst.Real.UsedBits, std::log2(1.2), 1e-9);
}

TEST(RefOrbitDynamicPrecision_ReducedOrbitMatchesOracle)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    constexpr mp_bitcnt_t bits = 1024;
    constexpr uint64_t iterations = 640;
    HighPrecision::defaultPrecisionInBits(bits);
    // Non-dyadic offsets avoid placing every tiny component on a conversion bin.
    // The orbit follows c=i long enough for a genuinely reduced work precision.
    HighPrecision cx{"7.07e-272"};
    HighPrecision cy{std::string{"1."} + std::string(271, '0') + "95"};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(bits);
    fractal.SetNumIterations<uint64_t>(iterations);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    calculator.ResetGuess(cx, cy);
    const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             HDRFloat<double>,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    CheckDynamicPrecisionCounterDelta(countersBefore, 1, 0);
    const auto statistics = calculator.GetLastDynamicPrecisionStats();
    ASSERT_TRUE(statistics.has_value());
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Dynamic attempt: steps=" << statistics->Steps << " min_bits=" << statistics->MinimumBits
        << " failure=" << static_cast<int>(statistics->Failure);
    ASSERT_TRUE(statistics->Failure == FractalShark::RefOrbitPrecisionFailure::None);
    // A representation-only refactor must retain this fixture's precision tiers,
    // as well as the independently checked samples below.
    ASSERT_EQ(statistics->ReducedSteps, uint64_t{499});
    ASSERT_EQ(statistics->MinimumBits, uint64_t{384});
    const auto *dynamic = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
    ASSERT_EQ(dynamic->GetCountOrbitEntries(), iterations);
    std::vector<std::array<HDRFloat<double>, 2>> samples;
    for (size_t index = 0; index < iterations; ++index) {
        samples.push_back({dynamic->GetOrbitData()[index].x, dynamic->GetOrbitData()[index].y});
    }
    // The oracle has an additional 1024 bits and uses the authoritative sequential
    // MPIR recurrence. Compare normalized conversion bits, including tiny components.
    MpfValue x(2048), y(2048), cxOracle(2048), cyOracle(2048), xx(2048), yy(2048), xy(2048);
    mpf_set(cxOracle.Get(), cx.backend());
    mpf_set(cyOracle.Get(), cy.backend());
    mpf_set(x.Get(), cx.backend());
    mpf_set(y.Get(), cy.backend());
    const auto checkComponent = [](const HDRFloat<double> &actual, mpf_srcptr exact) {
        double mantissa;
        const auto exponent = mpf_get_2exp_d(&mantissa, exact);
        HDRFloat<double> reduced = actual;
        HdrReduce(reduced);
        HDRFloat<double> expected{static_cast<int32_t>(exponent), mantissa};
        HdrReduce(expected);
        ASSERT_EQ(reduced.getExp(), expected.getExp());
        ASSERT_EQ(reduced.getMantissa(), expected.getMantissa());
    };
    for (size_t index = 1; index < iterations; ++index) {
        checkComponent(samples[index][0], x.Get());
        checkComponent(samples[index][1], y.Get());
        mpf_mul(xx.Get(), x.Get(), x.Get());
        mpf_mul(yy.Get(), y.Get(), y.Get());
        mpf_mul(xy.Get(), x.Get(), y.Get());
        mpf_mul_ui(xy.Get(), xy.Get(), 2);
        mpf_add(y.Get(), xy.Get(), cyOracle.Get());
        mpf_sub(x.Get(), xx.Get(), yy.Get());
        mpf_add(x.Get(), x.Get(), cxOracle.Get());
    }
    calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    calculator.ResetGuess(cx, cy);
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             HDRFloat<double>,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    CheckDynamicPrecisionCounterDelta(countersBefore, 1, 0);
    const auto *fixed = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
    ASSERT_EQ(fixed->GetCountOrbitEntries(), iterations);
    for (size_t index = 1; index < iterations; ++index) {
        ASSERT_EQ(fixed->GetOrbitData()[index].x.getMantissa(), samples[index][0].getMantissa());
        ASSERT_EQ(fixed->GetOrbitData()[index].y.getMantissa(), samples[index][1].getMantissa());
        ASSERT_EQ(fixed->GetOrbitData()[index].x.getExp(), samples[index][0].getExp());
        ASSERT_EQ(fixed->GetOrbitData()[index].y.getExp(), samples[index][1].getExp());
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "Dynamic MT9: min_bits=" << statistics->MinimumBits << " mean_bits=" << statistics->MeanBits
        << " reduced_steps=" << statistics->ReducedSteps
        << " elapsed_ms=" << statistics->ElapsedMilliseconds;
}

TEST(RefOrbitDynamicPrecision_WarningLogsAndContinues)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetNumIterations<uint64_t>(16);
    for (const bool unsupportedNative : {false, true}) {
        for (const bool cachedOrbit : {false, true}) {
            HighPrecision::defaultPrecisionInBits(4096);
            HighPrecision cx{1}, cy{0};
            mpf_div_2exp(cx.backend(), cx.backend(), 2000);
            if (!unsupportedNative) {
                // Gaussian-integer states exercise post-reduction boundary uncertainty.
                mpf_set_ui(cx.backend(), 0);
                mpf_set_ui(cy.backend(), 1);
            }
            if (cachedOrbit) {
                mpf_neg(cx.backend(), cx.backend());
                mpf_neg(cy.backend(), cy.backend());
            }
            HighPrecision::defaultPrecisionInBits(unsupportedNative ? 64 : 1024);
            fractal.SetNumIterations<uint64_t>(unsupportedNative ? 16 : 256);
            RefOrbitCalc calculator(fractal, UINT64_MAX);
            calculator.SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
            uint64_t previousGeneration = 0;
            if (cachedOrbit) {
                calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
                calculator.AddPerturbationReferencePoint<uint64_t,
                                                         double,
                                                         double,
                                                         PerturbExtras::Disable,
                                                         RefOrbitCalc::BenchmarkMode::Enable>(
                    fractal.GetPtz());
                previousGeneration = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>()
                                         ->GetGenerationNumber();
            }
            calculator.ResetGuess(cx, cy);
            const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
            std::string diagnostic;
            {
                ScopedConsoleLogCapture capture;
                calculator.AddPerturbationReferencePoint<uint64_t,
                                                         double,
                                                         double,
                                                         PerturbExtras::Disable,
                                                         RefOrbitCalc::BenchmarkMode::Enable>(
                    fractal.GetPtz());
                diagnostic = capture.GetMessage();
            }
            const auto warning = diagnostic.find("MT9 dynamic precision warning:");
            ASSERT_TRUE(warning != std::string::npos);
            ASSERT_TRUE(diagnostic.find("MT9 dynamic precision warning:", warning + 1) ==
                        std::string::npos);
            ASSERT_TRUE(diagnostic.find("continuing dynamic orbit with uncertified samples") !=
                        std::string::npos);
            if (unsupportedNative) {
                ASSERT_TRUE(diagnostic.find(", Z1, steps=0/16, bits=64/64") != std::string::npos);
            }
            ASSERT_TRUE(diagnostic.find(unsupportedNative ? "UnsupportedConversion"
                                                          : "AmbiguousSample") != std::string::npos);
            CheckLoggedCoordinate(diagnostic, "cx", cx.backend());
            CheckLoggedCoordinate(diagnostic, "cy", cy.backend());
            CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
            const auto stats = calculator.GetLastDynamicPrecisionStats();
            ASSERT_TRUE(stats.has_value());
            if (unsupportedNative) {
                ASSERT_EQ(stats->FailureIteration, uint64_t{1});
            } else {
                ASSERT_TRUE(stats->FailureIteration > 1);
                ASSERT_TRUE(stats->FailureWorkingBits < stats->FullBits);
            }
            ASSERT_TRUE(stats->Steps > 0);
            ASSERT_TRUE(stats->Failure ==
                        (unsupportedNative
                             ? FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion
                             : FractalShark::RefOrbitPrecisionFailure::AmbiguousSample));
            const auto *published = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
            ASSERT_TRUE(published->GetCountOrbitEntries() > 1);
            ASSERT_EQ(stats->GenerationNumber, published->GetGenerationNumber());
            ASSERT_EQ(stats->PublishedGenerationNumber, published->GetGenerationNumber());
            ASSERT_TRUE(published->GetGenerationNumber() != previousGeneration);
            ASSERT_EQ(published->GetOrbitData()[1].x, 0.0);
            ASSERT_EQ(HighPrecision::defaultPrecisionInBits(),
                      uint64_t{unsupportedNative ? 64u : 1024u});
            // A later clean attempt must not inherit the warning or count it again.
            calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     double,
                                                     double,
                                                     PerturbExtras::Disable,
                                                     RefOrbitCalc::BenchmarkMode::Enable>(
                fractal.GetPtz());
            ASSERT_TRUE(calculator.GetLastDynamicPrecisionStats()->Failure ==
                        FractalShark::RefOrbitPrecisionFailure::None);
            ASSERT_EQ(calculator.GetLastDynamicPrecisionStats()->FailureIteration, uint64_t{0});
            CheckDynamicPrecisionCounterDelta(countersBefore, 1, 1);
            // Explicit fixed selection remains available and performs no certification.
            calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
            calculator.ResetGuess(cx, cy);
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     double,
                                                     double,
                                                     PerturbExtras::Disable,
                                                     RefOrbitCalc::BenchmarkMode::Enable>(
                fractal.GetPtz());
            ASSERT_FALSE(calculator.GetLastDynamicPrecisionStats().has_value());
            CheckDynamicPrecisionCounterDelta(countersBefore, 1, 1);
        }
    }
}

TEST(RefOrbitDynamicPrecision_StartupStorageAndPeriodicity)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(1024);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    const HighPrecision cx{"-0.1"}, cy{"0.1"};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(1024).Recentered(cx, cy)));
    AbortMonitor abort(false);
    const auto checkStartup = [&]<PerturbExtras extras>(uint64_t limit, bool cancelled) {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(cx, cy);
        fractal.SetNumIterations<uint64_t>(limit);
        abort.SetStopCalculating(cancelled);
        const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
        std::string diagnostic;
        {
            ScopedConsoleLogCapture capture;
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     HDRFloat<double>,
                                                     double,
                                                     extras,
                                                     RefOrbitCalc::BenchmarkMode::Disable>(
                fractal.GetPtz());
            diagnostic = capture.GetMessage();
        }
        abort.ResetStopCalculating();
        const auto stats = calculator.GetLastDynamicPrecisionStats();
        ASSERT_TRUE(stats.has_value());
        CheckCertificationReport(diagnostic, *stats);
        ASSERT_EQ(stats->FastCertifiedComponents + stats->RegularCertificationComponents,
                  2 * stats->Steps);
        if (cancelled || limit == 0) {
            CheckDynamicPrecisionCounterDelta(countersBefore, 0, 0);
            ASSERT_EQ(stats->FastCertifiedComponents, uint64_t{0});
            ASSERT_EQ(stats->RegularCertificationComponents, uint64_t{0});
            ASSERT_TRUE(diagnostic.find("worst_real_error_bits=not_checked") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("worst_imaginary_error_bits=not_checked") != std::string::npos);
        }
        const auto *results = calculator.GetLast<uint64_t, HDRFloat<double>, extras>();
        ASSERT_EQ(results->GetCountOrbitEntries(),
                  cancelled ? uint64_t{1} : std::max(uint64_t{1}, limit));
        ASSERT_EQ(results->GetCompressionErrorExp(),
                  fractal.GetCompressionErrorExp(Fractal::CompressionError::Low));
        ASSERT_EQ(results->GetPeriodMaybeZero(), uint64_t{0});
        if (limit > 1 && !cancelled) {
            RuntimeDecompressor<uint64_t, HDRFloat<double>, extras> decompressor{*results};
            HDRFloat<double> x, y;
            results->GetComplex(decompressor, 1, x, y);
            ASSERT_NEAR(static_cast<double>(x), -0.1, 1e-15);
            ASSERT_NEAR(static_cast<double>(y), 0.1, 1e-15);
        }
    };
    for (uint64_t limit = 0; limit <= 3; ++limit) {
        for (const bool cancelled : {false, true}) {
            checkStartup.template operator()<PerturbExtras::Disable>(limit, cancelled);
            checkStartup.template operator()<PerturbExtras::SimpleCompression>(limit, cancelled);
            checkStartup.template operator()<PerturbExtras::Bad>(limit, cancelled);
        }
    }
    const auto checkZero = [&]<RefOrbitCalc::BenchmarkMode mode>() {
        calculator.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
        fractal.SetNumIterations<uint64_t>(64);
        calculator.AddPerturbationReferencePoint<uint64_t, double, double, PerturbExtras::Disable, mode>(
            fractal.GetPtz());
        const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
        if constexpr (mode == RefOrbitCalc::BenchmarkMode::Enable) {
            ASSERT_EQ(results->GetCountOrbitEntries(), uint64_t{64});
            ASSERT_EQ(results->GetPeriodMaybeZero(), uint64_t{0});
        } else {
            ASSERT_EQ(results->GetCountOrbitEntries(), uint64_t{2});
            ASSERT_EQ(results->GetPeriodMaybeZero(), uint64_t{2});
        }
    };
    checkZero.template operator()<RefOrbitCalc::BenchmarkMode::Enable>();
    checkZero.template operator()<RefOrbitCalc::BenchmarkMode::Disable>();
}

TEST(RefOrbitDynamicPrecision_ActiveCancellation)
{
    MPIRPrecision restore(1024);
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(1024);
    constexpr uint64_t iterationLimit = 1'048'576;
    fractal.SetNumIterations<uint64_t>(iterationLimit);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
    AbortMonitor abort(false);
    const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
    {
        std::jthread requester([&abort] {
            std::this_thread::sleep_for(std::chrono::milliseconds{250});
            abort.SetStopCalculating(true);
        });
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    }
    abort.ResetStopCalculating();
    CheckDynamicPrecisionCounterDelta(countersBefore, 0, 0);
    const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    ASSERT_TRUE(results->GetCountOrbitEntries() > 1);
    ASSERT_TRUE(results->GetCountOrbitEntries() < iterationLimit);
    ASSERT_TRUE(calculator.GetLastDynamicPrecisionStats()->Failure ==
                FractalShark::RefOrbitPrecisionFailure::None);
    const auto stats = calculator.GetLastDynamicPrecisionStats();
    ASSERT_EQ(stats->FastCertifiedComponents + stats->RegularCertificationComponents, stats->Steps);
}

TEST(RefOrbitDynamicPrecision_ManualMenuAndAutoSelection)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
#include "MenuTreeDef.h"
    const auto command = FractalShark::FractalCommand::PerturbationMultithread9KaratsubaDynamicPrecision;
    const auto *node = FindNode(menu, static_cast<uint32_t>(command));
    ASSERT_TRUE(node != nullptr);
    ASSERT_TRUE(node->text == L"MT9 + Periodicity + Karatsuba + Dynamic Precision");
    ASSERT_TRUE(node->radioGroup == FractalShark::RadioGroup::PerturbationMode);
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetNumIterations<uint64_t>(32);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64)));
    CommandFixture commands(fractal);
    commands.ExecuteCommand(command);
    fractal.GetRenderPool()->Drain();
    ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    FractalShark::MenuState state(fractal);
    ASSERT_EQ(state.GetRadioSelection(FractalShark::RadioGroup::PerturbationMode), uint32_t{40523});
    commands.ExecuteCommand(FractalShark::FractalCommand::PerturbationMultithread9KaratsubaPeriodicity);
    fractal.GetRenderPool()->Drain();
    ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    // A deeper view preserves the explicit full-precision override. Returning to Auto
    // through the menu applies the logical CPU requirement when CUDA is disabled.
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(8'192)));
    ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    commands.ExecuteCommand(FractalShark::FractalCommand::PerturbationAuto);
    fractal.GetRenderPool()->Drain();
    const auto expectedAuto =
        Environment::LogicalProcessorCount() >= 9
            ? RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision
            : RefOrbitCalc::PerturbationAlg::MTPeriodicity3;
    ASSERT_TRUE(fractal.GetPerturbationAlg() == expectedAuto);
}

TEST(RefOrbitDynamicPrecision_RenderAndCompressionRoundtrip)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    RenderTests::ScopedDirectory directory("RefOrbitDynamicPrecision_RenderAndCompressionRoundtrip");
    Fractal fractal(32, 32, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    const auto algorithm = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR);
    for (const auto bits : {IterTypeEnum::Bits32, IterTypeEnum::Bits64}) {
        const auto suffix = bits == IterTypeEnum::Bits32 ? "32" : "64";
        fractal.View(1, false);
        fractal.SetIterType(bits);
        fractal.SetNumIterations<uint64_t>(256);
        ASSERT_TRUE(fractal.SetRenderAlgorithm(algorithm));
        fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
        fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        fractal.CalcFractal(true);
        const auto baseline = SavePixels(fractal, std::string{"fixed-"} + suffix);
        // SetPerturbationAlg must invalidate cached references. The diagnostic suffix
        // is produced only by a newly generated dynamic attempt, not by selecting a name.
        fractal.SetPerturbationAlg(
            RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        fractal.ForceRecalc();
        fractal.GetRenderPool()->Drain();
        fractal.CalcFractal(true);
        ASSERT_TRUE(SavePixels(fractal, std::string{"dynamic-"} + suffix) == baseline);
        RefOrbitDetails details;
        fractal.GetSomeDetails(details);
        ASSERT_TRUE(details.PerturbationAlg.find("reduced steps") != std::string::npos);
        for (const auto compression : {CompressToDisk::Disable,
                                       CompressToDisk::SimpleCompression,
                                       CompressToDisk::MaxCompression,
                                       CompressToDisk::MaxCompressionImagina}) {
            const auto path =
                "orbit-" + std::string{suffix} + "-" + std::to_string(static_cast<int>(compression));
            fractal.SaveRefOrbit(compression, std::filesystem::path{path}.wstring());
            ASSERT_TRUE(std::filesystem::file_size(path) > 0);
            if (compression == CompressToDisk::MaxCompressionImagina) {
                fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                fractal.LoadRefOrbit(nullptr,
                                     compression,
                                     ImaginaSettings::UseSaved,
                                     std::filesystem::path{path}.wstring());
                ASSERT_TRUE(fractal.SetRenderAlgorithm(algorithm));
                fractal.SetIterType(bits);
                fractal.SetNumIterations<uint64_t>(256);
                fractal.ForceRecalc();
                fractal.GetRenderPool()->Drain();
                fractal.CalcFractal(true);
                ASSERT_TRUE(SavePixels(fractal, std::string{"loaded-"} + suffix) == baseline);
            }
        }
    }
}

TEST(RefOrbitDynamicPrecision_NearReturnOperandScale)
{
    // Fractadyne's 1.2e148 regression: the result-based budget destroys a near-return
    // around 4383 and escapes around 6780. Keep the coordinates local to this fixture.
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    constexpr mp_bitcnt_t bits = 576;
    constexpr uint64_t iterations = 8000;
    HighPrecision::defaultPrecisionInBits(bits);
    HighPrecision cx{"-0."
                     "3158354656090698908113251908145989842764104941136552011217533774266655202463327904"
                     "910559501703762081531934176786217990113494418705307973163264218287292234362119"};
    HighPrecision cy{"0."
                     "6533553743954627788289923830392687875350977003260517837408108019649970888461393846"
                     "103786781501651324966145060684808980380361143296058258024081840162818693511972"};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(bits);
    fractal.SetNumIterations<uint64_t>(iterations);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    std::vector<std::array<double, 2>> baseline;
    for (const auto algorithm :
         {RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba,
          RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision}) {
        calculator.SetPerturbationAlg(algorithm);
        calculator.ResetGuess(cx, cy);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        const auto *results = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
        ASSERT_EQ(results->GetCountOrbitEntries(), iterations);
        for (size_t index = 0; index < iterations; ++index) {
            const auto &sample = results->GetOrbitData()[index];
            if (algorithm == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba) {
                baseline.push_back({sample.x, sample.y});
            } else {
                ASSERT_EQ(sample.x, baseline[index][0]);
                ASSERT_EQ(sample.y, baseline[index][1]);
            }
        }
    }
    double minimum = 1.0;
    for (size_t index = 4300; index < 4450; ++index) {
        minimum = std::min(minimum, std::hypot(baseline[index][0], baseline[index][1]));
    }
    ASSERT_TRUE(minimum < 1e-60);
}

namespace {

void
ProfileConversionCertificates(mp_bitcnt_t bits)
{
    // Prebuild a small working set so the timed loop measures certification alone,
    // with neither MPIR allocation nor random-number generation in its hot path.
    constexpr size_t SampleCount = 16;
    constexpr uint64_t Iterations = ProfileDefaultIterations * 4;
    constexpr mp_bitcnt_t DeepTailGuardBits = 128;
    constexpr int64_t ErrorGuardBits = 96;
    std::mt19937_64 random{0x71b46e09};
    for (const bool deepBoundary : {false, true}) {
        std::vector<std::unique_ptr<MpfValue>> values;
        MpfValue offset(bits), ulp(bits);
        mpf_set_ui(offset.Get(), 1);
        mpf_div_2exp(offset.Get(), offset.Get(), bits - DeepTailGuardBits);
        mpf_set_ui(ulp.Get(), 1);
        mpf_div_2exp(ulp.Get(), ulp.Get(), std::numeric_limits<double>::digits - 1);
        for (size_t index = 0; index < 2 * SampleCount; ++index) {
            auto value = std::make_unique<MpfValue>(bits);
            if (deepBoundary) {
                mpf_set_ui(value->Get(), 1);
                if ((index & 1) != 0) {
                    mpf_add(value->Get(), value->Get(), ulp.Get());
                    mpf_sub(value->Get(), value->Get(), offset.Get());
                } else {
                    mpf_add(value->Get(), value->Get(), offset.Get());
                }
                if ((index & 2) != 0) {
                    mpf_neg(value->Get(), value->Get());
                }
            } else {
                Randomize(value->Get(), value->Get()->_mp_prec, (index & 1) != 0, random);
            }
            values.push_back(std::move(value));
        }
        FractalShark::RefOrbitPrecisionSchedule schedule(bits, values[0]->Get(), values[1]->Get());
        schedule.AddStateTruncation(
            FractalShark::RefOrbitErrorBound::PowerOfTwo(-static_cast<int64_t>(bits) + ErrorGuardBits),
            {});
        for (uint64_t iteration = 0; iteration < ProfileWarmupIterations; ++iteration) {
            const size_t sample = 2 * (iteration % SampleCount);
            ASSERT_TRUE(schedule.Certify(values[sample]->Get(), values[sample + 1]->Get(), false));
        }
        const auto start = std::chrono::steady_clock::now();
        for (uint64_t iteration = 0; iteration < Iterations; ++iteration) {
            const size_t sample = 2 * (iteration % SampleCount);
            ASSERT_TRUE(schedule.Certify(values[sample]->Get(), values[sample + 1]->Get(), false));
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        PrintProfile(deepBoundary ? "CertificatesDeepBoundary" : "CertificatesOrdinary",
                     bits,
                     Iterations,
                     elapsed);
        ASSERT_TRUE(schedule.DescribeCertification(values[0]->Get(), values[1]->Get(), false).Failure ==
                    FractalShark::RefOrbitPrecisionFailure::None);
        const auto &stats = schedule.GetStats();
        const uint64_t checks = 2 * (Iterations + ProfileWarmupIterations);
        ASSERT_EQ(stats.FastCertifiedComponents + stats.RegularCertificationComponents, checks);
        if constexpr (GMP_NUMB_BITS == 64 && GMP_NAIL_BITS == 0) {
            ASSERT_EQ(stats.FastCertifiedComponents, deepBoundary ? uint64_t{0} : checks);
        } else {
            ASSERT_EQ(stats.FastCertifiedComponents, uint64_t{0});
        }
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "CERTIFICATION_PROFILE bits=" << bits << " deep_boundary=" << deepBoundary
            << " certification_fast_components=" << stats.FastCertifiedComponents
            << " certification_regular_components=" << stats.RegularCertificationComponents
            << " certification_fast_percent="
            << 100.0 * static_cast<double>(stats.FastCertifiedComponents) / static_cast<double>(checks);
    }
}

void
ProfileDynamicPrecision(mp_bitcnt_t bits)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(bits);
    // Generic offsets from c=i give steady growth without forcing tiny components
    // onto exact conversion boundaries. Stop well before either reference can escape.
    constexpr mp_bitcnt_t CoordinateGuardBits = 124;
    HighPrecision cx{
        "0.70710678118654752440084436210484903928483593768847403658833986899536623923105351942519"};
    HighPrecision cy{"0.95000312341103189456476531465587653123865263475126451237651234565213"};
    mpf_div_2exp(cx.backend(), cx.backend(), bits - CoordinateGuardBits);
    mpf_div_2exp(cy.backend(), cy.backend(), bits - CoordinateGuardBits);
    mpf_add_ui(cy.backend(), cy.backend(), 1);
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(bits);
    const uint64_t iterations = bits / 2;
    fractal.SetNumIterations<uint64_t>(iterations);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    std::vector<std::array<HDRFloat<double>, 2>> baseline;
    for (const auto algorithm :
         {RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba,
          RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision}) {
        const bool dynamic =
            algorithm == RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision;
        calculator.SetPerturbationAlg(algorithm);
        calculator.ResetGuess(cx, cy);
        const auto start = std::chrono::steady_clock::now();
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 HDRFloat<double>,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const auto *results = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
        ASSERT_EQ(results->GetCountOrbitEntries(), iterations);
        PrintProfile(
            dynamic ? "OrbitMT9DynamicGrowing" : "OrbitMT9FixedGrowing", bits, iterations, elapsed);
        for (size_t index = 0; index < iterations; ++index) {
            const auto &sample = results->GetOrbitData()[index];
            if (!dynamic) {
                baseline.push_back({sample.x, sample.y});
            } else {
                ASSERT_EQ(sample.x.getExp(), baseline[index][0].getExp());
                ASSERT_EQ(sample.y.getExp(), baseline[index][1].getExp());
                ASSERT_EQ(sample.x.getMantissa(), baseline[index][0].getMantissa());
                ASSERT_EQ(sample.y.getMantissa(), baseline[index][1].getMantissa());
            }
        }
        if (dynamic) {
            const auto stats = calculator.GetLastDynamicPrecisionStats();
            ASSERT_TRUE(stats.has_value());
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "DYNAMIC_PRECISION_PROFILE steps=" << stats->Steps
                << " min_bits=" << stats->MinimumBits << " mean_bits=" << stats->MeanBits
                << " reduced_steps=" << stats->ReducedSteps
                << " certification_fast_components=" << stats->FastCertifiedComponents
                << " certification_regular_components=" << stats->RegularCertificationComponents
                << " certification_fast_percent="
                << 100.0 * static_cast<double>(stats->FastCertifiedComponents) /
                       (static_cast<double>(stats->FastCertifiedComponents) +
                        static_cast<double>(stats->RegularCertificationComponents))
                << " failure=" << static_cast<int>(stats->Failure);
            ASSERT_EQ(stats->FastCertifiedComponents + stats->RegularCertificationComponents,
                      2 * stats->Steps);
            ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::None);
            ASSERT_TRUE(stats->MinimumBits < bits / 2);
            ASSERT_TRUE(stats->ReducedSteps > iterations / 2);
        }
    }
}

} // namespace

TEST(RefOrbitDynamicPrecision_ProfileCertificates_16384) { ProfileConversionCertificates(16'384); }
TEST(RefOrbitDynamicPrecision_ProfileCertificates_33344) { ProfileConversionCertificates(33'344); }
TEST(RefOrbitDynamicPrecision_ProfileCertificates_100096) { ProfileConversionCertificates(100'096); }
TEST(RefOrbitDynamicPrecision_ProfileGrowing_16384) { ProfileDynamicPrecision(16'384); }
TEST(RefOrbitDynamicPrecision_ProfileGrowing_33344) { ProfileDynamicPrecision(33'344); }
TEST(RefOrbitDynamicPrecision_ProfileGrowing_100096) { ProfileDynamicPrecision(100'096); }

TEST(RefOrbitDynamicPrecision_PrecisionRestorationAndDerivativeReset)
{
    constexpr mp_bitcnt_t fullBits = 1024;
    MpfValue x(fullBits), y(fullBits), xx(fullBits), yy(fullBits), xy(fullBits), temporary(fullBits),
        clipped(fullBits);
    std::mt19937_64 random{0x6A7D};
    Randomize(x.Get(), x.Get()->_mp_prec + 1, false, random);
    Randomize(y.Get(), y.Get()->_mp_prec + 1, true, random);
    {
        FractalShark::RefOrbitPrecisionSchedule schedule(fullBits, x.Get(), y.Get());
        FractalShark::RefOrbitRawPrecisionGuard guard(
            x.Get(), y.Get(), xx.Get(), yy.Get(), xy.Get(), temporary.Get());
        guard.Set(64, schedule);
        ASSERT_EQ(mpf_get_prec(x.Get()), mp_bitcnt_t{64});
        ASSERT_TRUE(MagnitudeSize(x.Get()) <= x.Get()->_mp_prec + 1);
        ASSERT_FALSE(schedule.GetError().IsZero());
        const auto errorAfterClipping = schedule.GetError();
        mpf_set(clipped.Get(), x.Get());
        guard.Set(512, schedule);
        // Raising precision restores capacity, not the discarded state or its accuracy.
        ASSERT_EQ(mpf_cmp(x.Get(), clipped.Get()), 0);
        ASSERT_EQ(schedule.GetError().Log2(), errorAfterClipping.Log2());
    }
    for (auto *value : {x.Get(), y.Get(), xx.Get(), yy.Get(), xy.Get(), temporary.Get()}) {
        ASSERT_EQ(mpf_get_prec(value), fullBits);
    }
    const auto rejectAfterReduction = [&] {
        FractalShark::RefOrbitPrecisionSchedule schedule(fullBits, x.Get(), y.Get());
        FractalShark::RefOrbitRawPrecisionGuard guard(
            x.Get(), y.Get(), xx.Get(), yy.Get(), xy.Get(), temporary.Get());
        guard.Set(64, schedule);
        ASSERT_EQ(mpf_get_prec(x.Get()), mp_bitcnt_t{64});
        throw FractalSharkSeriousException("Test precision restoration during rejection");
    };
    ASSERT_THROWS(rejectAfterReduction(), FractalSharkSeriousException);
    for (auto *value : {x.Get(), y.Get(), xx.Get(), yy.Get(), xy.Get(), temporary.Get()}) {
        ASSERT_EQ(mpf_get_prec(value), fullBits);
    }
    mpf_set_d(x.Get(), -0.5);
    mpf_set_ui(y.Get(), 0);
    FractalShark::RefOrbitPrecisionSchedule reset(fullBits, x.Get(), y.Get());
    ASSERT_EQ(reset.BeginStep(x.Get(), y.Get(), true), fullBits); // D'=2*(-.5)*1+1=0
    mpf_set_ui(x.Get(), 1);
    for (size_t index = 0; index < 300; ++index) {
        reset.BeginStep(x.Get(), y.Get(), true);
    }
    ASSERT_TRUE(reset.GetStats().ReducedSteps > 0);
    ASSERT_FALSE(
        FractalShark::RefOrbitErrorBound::LimbPower(std::numeric_limits<int64_t>::max()).IsFinite());
}

TEST(RefOrbitDynamicPrecision_StructuredDeepFailure)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    constexpr mp_bitcnt_t bits = 1'024;
    constexpr uint64_t iterations = 256;
    HighPrecision::defaultPrecisionInBits(bits);
    HighPrecision cx{0}, cy{1};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(bits);
    fractal.SetNumIterations<uint64_t>(iterations);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             HDRFloat<double>,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    const auto *previous = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
    const auto previousGeneration = previous->GetGenerationNumber();
    calculator.ResetGuess(cx, cy);
    const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
    std::string diagnostic;
    {
        ScopedConsoleLogCapture capture;
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 HDRFloat<double>,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        diagnostic = capture.GetMessage();
    }
    CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
    const auto stats = calculator.GetLastDynamicPrecisionStats();
    ASSERT_TRUE(stats.has_value());
    ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
    ASSERT_EQ(stats->Steps, iterations);
    ASSERT_TRUE(stats->FailureIteration > 1);
    ASSERT_TRUE(stats->FailureWorkingBits < stats->FullBits);
    // The first uncertainty remains latched while the orbit finishes all updates.
    ASSERT_TRUE(diagnostic.find("Cannot guarantee equivalence to fixed MT9") != std::string::npos);
    CheckLoggedCoordinate(diagnostic, "cx", cx.backend());
    CheckLoggedCoordinate(diagnostic, "cy", cy.backend());
    const auto *recovered = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
    ASSERT_EQ(recovered->GetGenerationNumber(), stats->PublishedGenerationNumber);
    ASSERT_TRUE(recovered->GetGenerationNumber() != previousGeneration);
    ASSERT_EQ(previous->GetGenerationNumber(), previousGeneration);
    ASSERT_TRUE(stats->GenerationNumber != previousGeneration);
    ASSERT_EQ(previous->GetCountOrbitEntries(), iterations);
    // Exercise a clean attempt after warning-backed completion.
    calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             HDRFloat<double>,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    CheckDynamicPrecisionCounterDelta(countersBefore, 1, 1);
    ASSERT_TRUE(calculator.GetLastDynamicPrecisionStats()->Failure ==
                FractalShark::RefOrbitPrecisionFailure::None);
}

// Probe the unchanged heuristic against stable wider oracles and actual fixed MT9
// products. A counterexample needs a wrong reduced-precision sample, a correct fixed
// prefix, and a rejection after reduction. A zero count is a negative search result;
// it does not establish whether certification is necessary.
TEST(RefOrbitDynamicPrecision_HeuristicCounterexampleProbe)
{
    struct ProbeOrbitState {
        MpfValue m_X, m_Y, m_XSquared, m_YSquared, m_Cross, m_Temporary;
        ProbeOrbitState(mp_bitcnt_t fullBits, mpf_srcptr cx, mpf_srcptr cy)
            : m_X(fullBits), m_Y(fullBits), m_XSquared(fullBits), m_YSquared(fullBits),
              m_Cross(fullBits), m_Temporary(fullBits)
        {
            mpf_set(m_X.Get(), cx);
            mpf_set(m_Y.Get(), cy);
        }
        void
        Combine(mpf_srcptr cx, mpf_srcptr cy)
        {
            mpf_mul_ui(m_Cross.Get(), m_Cross.Get(), 2);
            mpf_add(m_Y.Get(), m_Cross.Get(), cy);
            mpf_sub(m_Temporary.Get(), m_XSquared.Get(), m_YSquared.Get());
            mpf_add(m_X.Get(), m_Temporary.Get(), cx);
        }
        void
        Advance(mpf_srcptr cx, mpf_srcptr cy)
        {
            mpf_mul(m_XSquared.Get(), m_X.Get(), m_X.Get());
            mpf_mul(m_YSquared.Get(), m_Y.Get(), m_Y.Get());
            mpf_mul(m_Cross.Get(), m_X.Get(), m_Y.Get());
            Combine(cx, cy);
        }
    };
    const auto sameSample = [](mpf_srcptr x, mpf_srcptr y) {
        double xMantissa, yMantissa;
        const auto xExponent = mpf_get_2exp_d(&xMantissa, x);
        const auto yExponent = mpf_get_2exp_d(&yMantissa, y);
        return xMantissa == yMantissa && xExponent == yExponent;
    };
    const auto printDyadic = [](const char *name, mpf_srcptr x) {
        MpzValue integer;
        const auto size = MagnitudeSize(x);
        mpz_import(
            integer.Get(), static_cast<size_t>(size), -1, sizeof(mp_limb_t), 0, GMP_NAIL_BITS, x->_mp_d);
        if (x->_mp_size < 0) {
            mpz_neg(integer.Get(), integer.Get());
        }
        std::vector<char> text(mpz_sizeinbase(integer.Get(), 16) + 3);
        mpz_get_str(text.data(), 16, integer.Get());
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << name << "_hex_mantissa=" << text.data()
            << " binary_exponent=" << (x->_mp_exp - size) * GMP_NUMB_BITS;
    };
    constexpr mp_bitcnt_t fullBits = 1024;
    constexpr uint64_t firstTargetIteration = 144;
    constexpr uint64_t lastTargetIteration = 200;
    constexpr mp_bitcnt_t targetQuadraticBits = fullBits - GMP_NUMB_BITS;
    constexpr mp_bitcnt_t shiftIncrement = GMP_NUMB_BITS / 4;
    // Reuse the workers throughout the bounded grid; each batch joins before the
    // next trial changes operand storage or active widths.
    FractalShark::RefOrbitKaratsubaProductsT<true> products(fullBits);
    FractalShark::RefOrbitKaratsubaProductsT<false> fixedProducts(fullBits);
    uint64_t cases = 0;
    uint64_t reducedMismatches = 0;
    uint64_t fixedAlsoMismatches = 0;
    uint64_t rejectionsWithoutMismatch = 0;
    uint64_t counterexamples = 0;
    // At c=i the orbit and derivative have exact Gaussian-integer values. The
    // dyadic displacement conjugate(D_target)*2^-shift cancels the first-order
    // imaginary term at the target iteration, exposing small higher-order corrections.
    // All trials are 144--200 iterations; no expensive deep orbit is needed.
    for (uint64_t targetIteration = firstTargetIteration; targetIteration <= lastTargetIteration;
         ++targetIteration) {
        MpfValue zero(fullBits), one(fullBits), derivativeX(fullBits), derivativeY(fullBits),
            nextDerivativeX(fullBits), nextDerivativeY(fullBits), productA(fullBits), productB(fullBits),
            cx(fullBits), cy(fullBits);
        mpf_set_ui(one.Get(), 1);
        mpf_set_ui(derivativeX.Get(), 1);
        ProbeOrbitState base(fullBits, zero.Get(), one.Get());
        for (uint64_t n = 1; n < targetIteration; ++n) {
            mpf_mul(productA.Get(), base.m_X.Get(), derivativeX.Get());
            mpf_mul(productB.Get(), base.m_Y.Get(), derivativeY.Get());
            mpf_sub(nextDerivativeX.Get(), productA.Get(), productB.Get());
            mpf_mul_ui(nextDerivativeX.Get(), nextDerivativeX.Get(), 2);
            mpf_add_ui(nextDerivativeX.Get(), nextDerivativeX.Get(), 1);
            mpf_mul(productA.Get(), base.m_X.Get(), derivativeY.Get());
            mpf_mul(productB.Get(), base.m_Y.Get(), derivativeX.Get());
            mpf_add(nextDerivativeY.Get(), productA.Get(), productB.Get());
            mpf_mul_ui(nextDerivativeY.Get(), nextDerivativeY.Get(), 2);
            mpf_set(derivativeX.Get(), nextDerivativeX.Get());
            mpf_set(derivativeY.Get(), nextDerivativeY.Get());
            base.Advance(zero.Get(), one.Get());
        }
        // The c=i multiplier grows |D_n| roughly as 2^(5*n/4). With the chosen
        // displacement the quadratic term scales as |D_n|^4*2^(-2*shift), or
        // 2^(5*n-2*shift). Target roughly 2^-960, then decrease the term by
        // another half limb and full limb to probe conversion-boundary ambiguity.
        constexpr uint64_t quadraticGrowthBitsPerIteration = 5;
        const mp_bitcnt_t baseShift = static_cast<mp_bitcnt_t>(
            (targetIteration * quadraticGrowthBitsPerIteration + targetQuadraticBits) / 2);
        for (const mp_bitcnt_t shift :
             {baseShift, baseShift + shiftIncrement, baseShift + 2 * shiftIncrement}) {
            mpf_set(cx.Get(), derivativeX.Get());
            mpf_div_2exp(cx.Get(), cx.Get(), shift);
            mpf_neg(cy.Get(), derivativeY.Get());
            mpf_div_2exp(cy.Get(), cy.Get(), shift);
            mpf_add_ui(cy.Get(), cy.Get(), 1);
            ProbeOrbitState dynamic(fullBits, cx.Get(), cy.Get()), fixed(fullBits, cx.Get(), cy.Get()),
                oracle(fullBits + 512, cx.Get(), cy.Get()), wider(fullBits + 1024, cx.Get(), cy.Get());
            FractalShark::RefOrbitPrecisionSchedule schedule(
                fullBits, dynamic.m_X.Get(), dynamic.m_Y.Get());
            ASSERT_TRUE(schedule.GetError().IsZero());
            FractalShark::RefOrbitRawPrecisionGuard guard(dynamic.m_X.Get(),
                                                          dynamic.m_Y.Get(),
                                                          dynamic.m_XSquared.Get(),
                                                          dynamic.m_YSquared.Get(),
                                                          dynamic.m_Cross.Get(),
                                                          dynamic.m_Temporary.Get());
            uint64_t firstRejectedIteration = 0, reducedStepsBeforeRejection = 0,
                     firstWrongIteration = 0;
            bool fixedCorrect = true;
            bool foundCounterexample = false;
            for (uint64_t n = 1; n <= targetIteration; ++n) {
                const bool oraclesAgree = sameSample(oracle.m_X.Get(), wider.m_X.Get()) &&
                                          sameSample(oracle.m_Y.Get(), wider.m_Y.Get());
                ASSERT_TRUE(oraclesAgree);
                fixedCorrect &= sameSample(fixed.m_X.Get(), wider.m_X.Get()) &&
                                sameSample(fixed.m_Y.Get(), wider.m_Y.Get());
                const bool dynamicSampleCorrect = sameSample(dynamic.m_X.Get(), wider.m_X.Get()) &&
                                                  sameSample(dynamic.m_Y.Get(), wider.m_Y.Get());
                ASSERT_TRUE(schedule.GetError().IsFinite());
                const bool certified = schedule.Certify(dynamic.m_X.Get(), dynamic.m_Y.Get(), false);
                // A rejected trial keeps running only in this probe. Every later
                // acceptance must still match the fixed MT9 baseline.
                ASSERT_TRUE(!certified || (sameSample(dynamic.m_X.Get(), fixed.m_X.Get()) &&
                                           sameSample(dynamic.m_Y.Get(), fixed.m_Y.Get())));
                if (!certified && firstRejectedIteration == 0) {
                    firstRejectedIteration = n;
                    reducedStepsBeforeRejection = schedule.GetStats().ReducedSteps;
                }
                if (firstWrongIteration == 0 && !dynamicSampleCorrect) {
                    firstWrongIteration = n;
                    // Judge the fixed prefix at this first wrong sample. A later
                    // unrelated fixed mismatch must not hide a counterexample.
                    if (fixedCorrect && !certified && reducedStepsBeforeRejection != 0) {
                        foundCounterexample = true;
                        FractalSharkLog::LogLine(__FILE__, __LINE__)
                            << "MT9 reduction counterexample: Z=" << n
                            << " first_rejected_Z=" << firstRejectedIteration
                            << " active_bits=" << mpf_get_prec(dynamic.m_X.Get())
                            << " reduced_steps=" << schedule.GetStats().ReducedSteps;
                        printDyadic("cx", cx.Get());
                        printDyadic("cy", cy.Get());
                    }
                }
                if (n == targetIteration) {
                    break;
                }
                const auto workBits = schedule.BeginStep(dynamic.m_X.Get(), dynamic.m_Y.Get(), n != 1);
                if (n != 1) {
                    guard.Set(workBits, schedule);
                }
                products.Compute(dynamic.m_XSquared.Get(),
                                 dynamic.m_YSquared.Get(),
                                 dynamic.m_Cross.Get(),
                                 dynamic.m_X.Get(),
                                 dynamic.m_Y.Get());
                schedule.ComputeStep(dynamic.m_X.Get(),
                                     dynamic.m_Y.Get(),
                                     dynamic.m_XSquared.Get(),
                                     dynamic.m_YSquared.Get(),
                                     dynamic.m_Cross.Get(),
                                     dynamic.m_Temporary.Get(),
                                     cx.Get(),
                                     cy.Get());
                fixedProducts.Compute(fixed.m_XSquared.Get(),
                                      fixed.m_YSquared.Get(),
                                      fixed.m_Cross.Get(),
                                      fixed.m_X.Get(),
                                      fixed.m_Y.Get());
                fixed.Combine(cx.Get(), cy.Get());
                oracle.Advance(cx.Get(), cy.Get());
                wider.Advance(cx.Get(), cy.Get());
            }
            ++cases;
            reducedMismatches += firstWrongIteration != 0 && schedule.GetStats().ReducedSteps != 0;
            fixedAlsoMismatches += firstWrongIteration != 0 && !fixedCorrect;
            rejectionsWithoutMismatch += firstRejectedIteration != 0 && firstWrongIteration == 0;
            counterexamples += foundCounterexample;
        }
    }
    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "MT9 heuristic probe: cases=" << cases
        << " reduced_sample_mismatch_cases=" << reducedMismatches
        << " fixed_also_mismatch_cases=" << fixedAlsoMismatches
        << " rejections_without_sample_mismatch=" << rejectionsWithoutMismatch
        << " verified_reduction_counterexamples=" << counterexamples;
    if (counterexamples == 0) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "No qualifying point found; certificate rejections alone do not "
               "demonstrate that dynamic precision caused a wrong sample.";
    }
}

namespace {

// Snapshots own ordinary limb vectors, never allocations from the orbit's bounded
// MPIR TLS. A descriptor borrows the vector only for replay after generation ends.
struct MT9DiagnosticValue {
    std::vector<mp_limb_t> Limbs;
    int Size{};
    mp_exp_t Exponent{};
    int Precision{};

    void
    Capture(mpf_srcptr value)
    {
        Size = value->_mp_size;
        Exponent = value->_mp_exp;
        Precision = value->_mp_prec;
        Limbs.assign(value->_mp_d, value->_mp_d + MagnitudeSize(value));
    }
    __mpf_struct
    View() const
    {
        __mpf_struct value{};
        value._mp_size = Size;
        value._mp_exp = Exponent;
        value._mp_prec = Precision;
        value._mp_d = const_cast<mp_limb_t *>(Limbs.data());
        return value;
    }
};

struct MT9DiagnosticSnapshot {
    uint64_t StateIteration{};
    mp_bitcnt_t WorkingBits{};
    double InflationIncrease{};
    double PreviousActualErrorLog2{};
    FractalShark::RefOrbitStepDiagnostics Step;
    MT9DiagnosticValue BeforeX, BeforeY, InputX, InputY, OutputX, OutputY;
};

class ScopedMT9DiagnosticObserver {
public:
    ScopedMT9DiagnosticObserver(RefOrbitCalc &calculator,
                                FractalShark::RefOrbitMT9DiagnosticObserver &observer)
        : m_Calculator(calculator)
    {
        m_Calculator.SetMT9DiagnosticObserver(&observer);
    }
    ~ScopedMT9DiagnosticObserver() { m_Calculator.SetMT9DiagnosticObserver(nullptr); }
    ScopedMT9DiagnosticObserver(const ScopedMT9DiagnosticObserver &) = delete;
    ScopedMT9DiagnosticObserver &operator=(const ScopedMT9DiagnosticObserver &) = delete;

private:
    RefOrbitCalc &m_Calculator;
};

// Ground truth for the test only. The fixed product engine and MPIR combination
// order match MT9's fixed specialization; its states never enter the candidate.
class MT9FixedDiagnosticOrbit {
public:
    MT9FixedDiagnosticOrbit(mp_bitcnt_t fullBits, mpf_srcptr cx, mpf_srcptr cy)
        : m_Cx(fullBits), m_Cy(fullBits), m_X(fullBits), m_Y(fullBits), m_XX(fullBits), m_YY(fullBits),
          m_XY(fullBits), m_Temporary(fullBits), m_Products(fullBits)
    {
        mpf_set(m_Cx.Get(), cx);
        mpf_set(m_Cy.Get(), cy);
        mpf_set(m_X.Get(), m_Cx.Get());
        mpf_set(m_Y.Get(), m_Cy.Get());
    }

    mpf_srcptr
    GetX()
    {
        return m_X.Get();
    }
    mpf_srcptr
    GetY()
    {
        return m_Y.Get();
    }
    const FractalShark::RefOrbitErrorBound &
    GetLocalAllowance() const
    {
        return m_LocalAllowance;
    }

    void
    Advance()
    {
        m_Products.Compute(m_XX.Get(), m_YY.Get(), m_XY.Get(), m_X.Get(), m_Y.Get());
        const auto realProducts = FractalShark::RefOrbitErrorBound::Add(
            FractalShark::RefOrbitPrecisionSchedule::ProductError(m_XX.Get()),
            FractalShark::RefOrbitPrecisionSchedule::ProductError(m_YY.Get()));
        const auto imaginaryProducts = FractalShark::RefOrbitErrorBound::Add(
            FractalShark::RefOrbitErrorBound::ScaleByPowerOfTwo(
                FractalShark::RefOrbitPrecisionSchedule::ProductError(m_XY.Get()), 1),
            FractalShark::RefOrbitPrecisionSchedule::DoublingError(m_XY.Get()));
        mpf_mul_ui(m_XY.Get(), m_XY.Get(), 2);
        const auto imaginary = FractalShark::RefOrbitErrorBound::Add(
            imaginaryProducts,
            FractalShark::RefOrbitPrecisionSchedule::AddError(m_XY.Get(), m_Cy.Get(), m_Y.Get()));
        mpf_add(m_Y.Get(), m_XY.Get(), m_Cy.Get());
        const auto realSubtract = FractalShark::RefOrbitPrecisionSchedule::SubtractError(
            m_XX.Get(), m_YY.Get(), m_Temporary.Get());
        mpf_sub(m_Temporary.Get(), m_XX.Get(), m_YY.Get());
        const auto real = FractalShark::RefOrbitErrorBound::Add(
            realProducts,
            FractalShark::RefOrbitErrorBound::Add(realSubtract,
                                                  FractalShark::RefOrbitPrecisionSchedule::AddError(
                                                      m_Temporary.Get(), m_Cx.Get(), m_X.Get())));
        mpf_add(m_X.Get(), m_Temporary.Get(), m_Cx.Get());
        m_LocalAllowance = FractalShark::RefOrbitErrorBound::Norm(real, imaginary);
    }

private:
    MpfValue m_Cx, m_Cy, m_X, m_Y, m_XX, m_YY, m_XY, m_Temporary;
    FractalShark::RefOrbitErrorBound m_LocalAllowance;
    // Join the workers before destroying any arithmetic storage they borrow.
    FractalShark::RefOrbitKaratsubaProductsT<false> m_Products;
};

void
CheckMT9FixedEquivalenceEdges()
{
    constexpr mp_bitcnt_t FullBits = 256;
    constexpr int64_t DisplacementExponent = -128;
    MpfValue zero(FullBits), one(FullBits), coordinate(FullBits), input(FullBits);
    mpf_set_ui(one.Get(), 1);
    FractalShark::RefOrbitPrecisionSchedule schedule(FullBits, zero.Get(), zero.Get());
    schedule.EnableDiagnosticCapture(true);
    // Common nonzero rounding is not divergence during the identical prefix.
    schedule.BeginStep(one.Get(), zero.Get(), false);
    schedule.FinishStep(FractalShark::RefOrbitErrorBound::PowerOfTwo(-static_cast<int64_t>(FullBits)),
                        {});
    ASSERT_TRUE(schedule.GetError().IsZero());

    schedule.BeginStep(one.Get(), zero.Get(), false);
    schedule.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(DisplacementExponent), {});
    schedule.FinishStep({}, {});
    // Squaring 1 versus 1+d differs by 2*d+d^2, not just the clipped d.
    const auto linearDifference = FractalShark::RefOrbitErrorBound::PowerOfTwo(DisplacementExponent + 1);
    ASSERT_TRUE(schedule.GetError().IsFinite());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less(linearDifference, schedule.GetError()));
    const auto reducedBound = schedule.GetError();
    schedule.BeginStep(one.Get(), zero.Get(), false);
    schedule.FinishStep({}, {});
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less(reducedBound, schedule.GetError()));
    // A zero dynamic input contracts a pre-existing difference quadratically.
    const auto beforeCancellation = schedule.GetError();
    schedule.BeginStep(zero.Get(), zero.Get(), false);
    schedule.FinishStep({}, {});
    ASSERT_FALSE(schedule.GetError().IsZero());
    ASSERT_TRUE(FractalShark::RefOrbitErrorBound::Less(schedule.GetError(), beforeCancellation));
    ASSERT_FALSE(schedule.Certify(zero.Get(), zero.Get(), false));
    ASSERT_FALSE(schedule.Certify(one.Get(), zero.Get(), false));
    FractalShark::RefOrbitPrecisionSchedule exactZero(FullBits, zero.Get(), zero.Get());
    exactZero.BeginStep(zero.Get(), zero.Get(), true);
    exactZero.FinishStep({}, {});
    ASSERT_TRUE(exactZero.GetError().IsZero());
    ASSERT_TRUE(exactZero.Certify(zero.Get(), zero.Get(), false));
    // An invalid metadata bound must never become a successful zero certificate.
    exactZero.AddStateTruncation(
        FractalShark::RefOrbitErrorBound::PowerOfTwo(std::numeric_limits<int64_t>::max()), {});
    exactZero.FinishStep({}, {});
    ASSERT_FALSE(exactZero.GetError().IsFinite());
    ASSERT_FALSE(exactZero.Certify(zero.Get(), zero.Get(), false));

    // Independently tabulated K values straddle positive and negative limb
    // boundaries of 8*U^2; these catch truncating integer division for e<0.
    constexpr std::array<std::array<int64_t, 2>, 6> RadiusCases{
        {{-34, -1}, {-33, 0}, {-2, 0}, {-1, 1}, {30, 1}, {31, 2}}};
    const auto setPower = [](mpf_ptr value, int64_t exponent) {
        mpf_set_ui(value, 1);
        if (exponent < 0) {
            mpf_div_2exp(value, value, static_cast<mp_bitcnt_t>(-exponent));
        } else {
            mpf_mul_2exp(value, value, static_cast<mp_bitcnt_t>(exponent));
        }
    };
    constexpr int64_t FullLimbs = FullBits / GMP_NUMB_BITS + 1;
    for (const auto &testCase : RadiusCases) {
        setPower(input.Get(), testCase[0]);
        FractalShark::RefOrbitPrecisionSchedule boundaryCase(FullBits, zero.Get(), zero.Get());
        boundaryCase.EnableDiagnosticCapture(true);
        boundaryCase.BeginStep(input.Get(), zero.Get(), false);
        boundaryCase.AddStateTruncation(
            FractalShark::RefOrbitErrorBound::PowerOfTwo(testCase[0] - static_cast<int64_t>(FullBits)),
            {});
        boundaryCase.FinishStep({}, {});
        ASSERT_NEAR(boundaryCase.GetStepDiagnostics().FixedRoundingAllowance.Log2(),
                    static_cast<double>((testCase[1] - FullLimbs) * GMP_NUMB_BITS + 4),
                    1e-9);
    }
    // At exact 8*|c| limb powers the strict upper exponent advances one limb.
    for (const int64_t boundaryBit : {-64, 0, 64}) {
        setPower(coordinate.Get(), boundaryBit - 3);
        FractalShark::RefOrbitPrecisionSchedule boundaryCase(FullBits, coordinate.Get(), zero.Get());
        boundaryCase.EnableDiagnosticCapture(true);
        boundaryCase.BeginStep(zero.Get(), zero.Get(), false);
        boundaryCase.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(
                                            boundaryBit - 2 * static_cast<int64_t>(FullBits)),
                                        {});
        boundaryCase.FinishStep({}, {});
        ASSERT_NEAR(
            boundaryCase.GetStepDiagnostics().FixedRoundingAllowance.Log2(),
            static_cast<double>((boundaryBit / GMP_NUMB_BITS + 1 - FullLimbs) * GMP_NUMB_BITS + 4),
            1e-9);
    }
}

void
CheckMT9ProductionUncertainty()
{
    // The exactly representable c=i orbit deliberately exercises a conservative
    // post-reduction uncertainty at a conversion boundary. Benchmark mode keeps
    // the reference running past its short period; no wrong-sample claim is made.
    class WarningSink final : public std::stringbuf {
    public:
        WarningSink(AbortMonitor &abort, bool cancel) : m_Abort(abort), m_Cancel(cancel) {}
        std::streamsize
        xsputn(const char *text, std::streamsize length) override
        {
            const auto written = std::stringbuf::xsputn(text, length);
            if (m_Cancel && std::string_view{text, static_cast<size_t>(length)}.find(
                                "MT9 dynamic precision warning:") != std::string_view::npos) {
                m_Abort.SetStopCalculating(true);
            }
            return written;
        }

    private:
        AbortMonitor &m_Abort;
        bool m_Cancel;
    };
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    constexpr mp_bitcnt_t Bits = 1'024;
    constexpr uint64_t Iterations = 256;
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    HighPrecision::defaultPrecisionInBits(Bits);
    HighPrecision cx{0}, cy{1};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(Bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(Bits);
    AbortMonitor abort(false);
    for (const bool cancel : {false, true}) {
        // Cancellation retains the production polling interval; leave enough
        // iterations for the warning-triggered request to reach that boundary.
        const uint64_t limit = cancel ? 2 * AbortMonitor::AbortCheckInterval : Iterations;
        fractal.SetNumIterations<uint64_t>(limit);
        RefOrbitCalc calculator(fractal, UINT64_MAX);
        calculator.SetPerturbationAlg(
            RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        calculator.ResetGuess(cx, cy);
        WarningSink sink(abort, cancel);
        const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
        {
            ScopedConsoleLogCapture capture(sink);
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     HDRFloat<double>,
                                                     double,
                                                     PerturbExtras::Disable,
                                                     RefOrbitCalc::BenchmarkMode::Enable>(
                fractal.GetPtz());
        }
        abort.ResetStopCalculating();
        const auto message = sink.str();
        const auto first = message.find("MT9 dynamic precision warning:");
        ASSERT_TRUE(first != std::string::npos);
        ASSERT_TRUE(message.find("MT9 dynamic precision warning:", first + 1) == std::string::npos);
        ASSERT_TRUE(message.find("Cannot guarantee equivalence to fixed MT9") != std::string::npos);
        ASSERT_TRUE(message.find("certification_target=fixed_mt9") != std::string::npos);
        ASSERT_TRUE(message.find("headroom_bits=") != std::string::npos);
        ASSERT_TRUE(message.find("continuing dynamic orbit") != std::string::npos);
        CheckLoggedCoordinate(message, "cx", cx.backend());
        CheckLoggedCoordinate(message, "cy", cy.backend());
        const auto stats = calculator.GetLastDynamicPrecisionStats();
        ASSERT_TRUE(stats.has_value());
        ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
        ASSERT_TRUE(stats->FailureIteration > 1);
        ASSERT_TRUE(stats->FailureWorkingBits < stats->FullBits);
        ASSERT_TRUE(stats->ReducedSteps > 0);
        const auto *published = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
        ASSERT_TRUE(published != nullptr);
        ASSERT_EQ(published->GetGenerationNumber(), stats->PublishedGenerationNumber);
        if (cancel) {
            ASSERT_TRUE(published->GetCountOrbitEntries() < limit);
        } else {
            ASSERT_EQ(published->GetCountOrbitEntries(), Iterations);
            ASSERT_EQ(stats->Steps, Iterations);
        }
        CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
    }
}

class MT9NucleusDiagnostics final : public FractalShark::RefOrbitMT9DiagnosticObserver {
public:
    static constexpr uint64_t ReturnIteration = 410'594;
    static constexpr uint64_t IterationLimit = ReturnIteration + 2;
    // Bound diagnostic memory independently of orbit length. Eight candidates retain
    // the strongest jumps; a separate small list always retains the nucleus return.
    static constexpr size_t RetainedIncreases = 8;
    static constexpr mp_bitcnt_t OracleExtraBits = 2'048;
    static constexpr uint64_t ProgressInterval = 65'536;

    MT9NucleusDiagnostics(mpf_srcptr cx, mpf_srcptr cy, mp_bitcnt_t fullBits)
        : m_FullBits(fullBits), m_FixedOrbit(fullBits, cx, cy), m_Cx(fullBits + OracleExtraBits),
          m_Cy(fullBits + OracleExtraBits), m_X(fullBits + OracleExtraBits),
          m_Y(fullBits + OracleExtraBits), m_XX(fullBits + OracleExtraBits),
          m_YY(fullBits + OracleExtraBits), m_XY(fullBits + OracleExtraBits),
          m_Temporary(fullBits + OracleExtraBits), m_DifferenceX(fullBits + OracleExtraBits),
          m_DifferenceY(fullBits + OracleExtraBits), m_Boundary(fullBits + OracleExtraBits),
          m_Gap(fullBits + OracleExtraBits)
    {
        mpf_set(m_Cx.Get(), cx);
        mpf_set(m_Cy.Get(), cy);
        mpf_set(m_X.Get(), cx);
        mpf_set(m_Y.Get(), cy);
        m_Increases.reserve(RetainedIncreases);
    }

    static FractalShark::RefOrbitErrorBound
    Magnitude(mpf_srcptr value)
    {
        // FromMpfUpper takes an outward neighbour, including for zero. Measured
        // exact-zero residuals instead need the exact zero tag.
        return value->_mp_size == 0 ? FractalShark::RefOrbitErrorBound{}
                                    : FractalShark::RefOrbitErrorBound::FromMpfUpper(value);
    }

    static std::array<HDRFloat<double>, 2>
    Sample(mpf_srcptr x, mpf_srcptr y)
    {
        std::array<HDRFloat<double>, 2> result;
        for (size_t component = 0; component < result.size(); ++component) {
            double mantissa;
            const auto exponent = mpf_get_2exp_d(&mantissa, component == 0 ? x : y);
            result[component] = HDRFloat<double>{static_cast<int32_t>(exponent), mantissa};
            HdrReduce(result[component]);
        }
        return result;
    }

    static bool
    Same(const std::array<HDRFloat<double>, 2> &left, const std::array<HDRFloat<double>, 2> &right)
    {
        for (size_t component = 0; component < left.size(); ++component) {
            if (left[component].getMantissa() != right[component].getMantissa() ||
                (left[component].getMantissa() != 0.0 &&
                 left[component].getExp() != right[component].getExp())) {
                return false;
            }
        }
        return true;
    }

    void
    OnState(const FractalShark::RefOrbitMT9StateView &state) override
    {
        if ((state.StateIteration % ProgressInterval) == 0) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "MT9 nucleus progress: phase=equivalence+oracle Z=" << state.StateIteration
                << " bits=" << state.WorkingBits;
        }
        ++m_DynamicStates;
        const auto fixedSample = Sample(m_FixedOrbit.GetX(), m_FixedOrbit.GetY());
        const auto dynamicSample = Sample(state.X, state.Y);
        const bool sameFixedSample = Same(dynamicSample, fixedSample);
        m_FixedDynamicMismatches += !sameFixedSample;

        // This actual difference and the fixed operation metadata below are
        // ground truth only. Neither value is an input to the production schedule.
        mpf_sub(m_DifferenceX.Get(), state.X, m_FixedOrbit.GetX());
        mpf_sub(m_DifferenceY.Get(), state.Y, m_FixedOrbit.GetY());
        const auto measuredDifference = FractalShark::RefOrbitErrorBound::Norm(
            Magnitude(m_DifferenceX.Get()), Magnitude(m_DifferenceY.Get()));
        const auto &equivalenceBound = state.Step->OutputError;
        const bool validBound = equivalenceBound.IsFinite();
        const bool active = m_FirstReduction != 0 && state.StateIteration > m_FirstReduction;
        m_EquivalenceBoundViolations +=
            !validBound || FractalShark::RefOrbitErrorBound::Less(equivalenceBound, measuredDifference);
        if (!active) {
            m_PrefixDifferences += !measuredDifference.IsZero();
        } else {
            m_FixedModelViolations += FractalShark::RefOrbitErrorBound::Less(
                state.Step->FixedRoundingAllowance, m_FixedOrbit.GetLocalAllowance());
        }
        const auto headroom = [](const FractalShark::RefOrbitComponentErrorBudget &budget) {
            return budget.Status == FractalShark::RefOrbitErrorBudgetStatus::Exact
                       ? std::numeric_limits<double>::infinity()
                       : budget.MaxBits - budget.UsedBits;
        };
        const std::array<double, 2> equivalenceHeadroom{headroom(state.Certification->Real),
                                                        headroom(state.Certification->Imaginary)};
        const bool equivalent = state.Certified;
        m_EquivalenceAccepted += equivalent;
        m_EquivalenceCertifiedMismatches += equivalent && !sameFixedSample;
        if (!equivalent && m_FirstInconclusive == 0) {
            m_FirstInconclusive = state.StateIteration;
        }
        if (state.StateIteration == ReturnIteration) {
            m_ReturnEquivalent = equivalent;
        }
        const auto expected = Sample(m_X.Get(), m_Y.Get());
        const bool dynamicCorrect = Same(dynamicSample, expected);
        const bool fixedCorrect = Same(fixedSample, expected);
        m_DynamicMismatches += !dynamicCorrect;
        m_FixedMismatches += !fixedCorrect;
        const bool firstMismatch = !dynamicCorrect && m_FirstMismatch == 0;
        if (firstMismatch) {
            m_FirstMismatch = state.StateIteration;
        }
        const bool firstRejection = !state.Certified && m_FirstRejection == 0;
        if (firstRejection) {
            m_FirstRejection = state.StateIteration;
        }

        mpf_sub(m_DifferenceX.Get(), state.X, m_X.Get());
        mpf_sub(m_DifferenceY.Get(), state.Y, m_Y.Get());
        const double realError = Magnitude(m_DifferenceX.Get()).Log2();
        const double imaginaryError = Magnitude(m_DifferenceY.Get()).Log2();
        const double actualError = FractalShark::RefOrbitErrorBound::Norm(Magnitude(m_DifferenceX.Get()),
                                                                          Magnitude(m_DifferenceY.Get()))
                                       .Log2();
        const double radius = state.Certification->AccumulatedErrorBoundLog2;
        // Mathematical-orbit error is contextual information, not the certificate's
        // target. Rank retained snapshots by excess over the measured fixed difference.
        const double inflation = measuredDifference.IsZero() && equivalenceBound.IsZero()
                                     ? 0.0
                                     : radius - measuredDifference.Log2();
        const double increase = inflation - m_PreviousInflation;
        const bool focus = state.StateIteration == 2 ||
                           (m_FirstReduction != 0 && state.StateIteration == m_FirstReduction + 1) ||
                           (state.StateIteration >= ReturnIteration - 2 &&
                            state.StateIteration <= ReturnIteration + 1) ||
                           firstRejection || firstMismatch;
        bool retainIncrease = std::isfinite(increase) && increase > 0.0;
        auto weakest = m_Increases.end();
        if (retainIncrease && m_Increases.size() == RetainedIncreases) {
            weakest = std::min_element(
                m_Increases.begin(), m_Increases.end(), [](const auto &a, const auto &b) {
                    return a.InflationIncrease < b.InflationIncrease;
                });
            retainIncrease = increase > weakest->InflationIncrease;
        }
        if (state.StateIteration > 1 && (focus || retainIncrease)) {
            MT9DiagnosticSnapshot snapshot{state.StateIteration,
                                           state.WorkingBits,
                                           increase,
                                           m_PreviousActualError,
                                           *state.Step,
                                           m_InputBeforeX,
                                           m_InputBeforeY,
                                           m_InputX,
                                           m_InputY,
                                           {},
                                           {}};
            snapshot.OutputX.Capture(state.X);
            snapshot.OutputY.Capture(state.Y);
            if (focus) {
                m_Focus.push_back(snapshot);
            }
            if (retainIncrease) {
                if (weakest != m_Increases.end()) {
                    *weakest = std::move(snapshot);
                } else {
                    m_Increases.push_back(std::move(snapshot));
                }
            }
        }
        if (focus) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << std::setprecision(17)
                << "MT9 fixed-equivalence production: Z=" << state.StateIteration << " active=" << active
                << " valid=" << validBound << " accepted=" << equivalent
                << " same_fixed_sample=" << sameFixedSample
                << " difference_log2=" << measuredDifference.Log2()
                << " bound_log2=" << equivalenceBound.Log2() << " excess_bits="
                << (measuredDifference.IsZero() && equivalenceBound.IsZero()
                        ? 0.0
                        : equivalenceBound.Log2() - measuredDifference.Log2())
                << " full_rounding_allowance_log2=" << state.Step->FixedRoundingAllowance.Log2()
                << " observed_fixed_local_bound_log2=" << m_FixedOrbit.GetLocalAllowance().Log2()
                << " real_headroom_bits=" << equivalenceHeadroom[0]
                << " imaginary_headroom_bits=" << equivalenceHeadroom[1];
            const auto &certificate = *state.Certification;
            const auto &step = *state.Step;
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << std::setprecision(17) << "MT9 nucleus state: Z=" << state.StateIteration
                << " bits=" << state.WorkingBits << '/' << m_FullBits
                << " fixed_equivalent_certified=" << state.Certified
                << " dynamic_matches_wide=" << dynamicCorrect << " fixed_matches_wide=" << fixedCorrect
                << " real_error_sign=" << mpf_sgn(m_DifferenceX.Get())
                << " real_error_log2=" << realError
                << " imaginary_error_sign=" << mpf_sgn(m_DifferenceY.Get())
                << " imaginary_error_log2=" << imaginaryError << " norm_error_log2=" << actualError
                << " difference_bound_log2=" << radius << " component=" << certificate.Component
                << " boundary=" << certificate.Boundary << " gap_log2=" << certificate.GapLowerBoundLog2
                << " gap_rounding_log2=" << certificate.GapRoundingBoundLog2
                << " headroom_bits=" << certificate.HeadroomBits
                << " subtraction_overhead_bits=" << certificate.ComparisonErrorBoundLog2 - radius
                << " fixed_difference_enclosure_excess_bits=" << inflation
                << " input_radius_log2=" << step.InputError.Log2()
                << " clip_x_log2=" << step.ClipX.Log2() << " clip_y_log2=" << step.ClipY.Log2()
                << " linear_log2=" << step.Linear.Log2() << " quadratic_log2=" << step.Quadratic.Log2()
                << " fixed_rounding_allowance_log2=" << step.FixedRoundingAllowance.Log2()
                << " local_x_log2=" << step.LocalX.Log2() << " local_y_log2=" << step.LocalY.Log2();
            for (size_t component = 0; component < expected.size(); ++component) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << std::setprecision(17) << "MT9 nucleus sample: Z=" << state.StateIteration
                    << " component=" << component
                    << " dynamic=" << dynamicSample[component].getMantissa() << "*2^"
                    << dynamicSample[component].getExp()
                    << " fixed=" << fixedSample[component].getMantissa() << "*2^"
                    << fixedSample[component].getExp() << " oracle=" << expected[component].getMantissa()
                    << "*2^" << expected[component].getExp();
            }
            ReportGaps(state);
            if (state.StateIteration == ReturnIteration) {
                mpf_sub(m_DifferenceX.Get(), m_FixedOrbit.GetX(), m_X.Get());
                mpf_sub(m_DifferenceY.Get(), m_FixedOrbit.GetY(), m_Y.Get());
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << std::setprecision(17)
                    << "MT9 nucleus fixed errors: real_log2=" << Magnitude(m_DifferenceX.Get()).Log2()
                    << " imaginary_log2=" << Magnitude(m_DifferenceY.Get()).Log2();
                mpf_sub(m_DifferenceX.Get(), state.X, m_FixedOrbit.GetX());
                mpf_sub(m_DifferenceY.Get(), state.Y, m_FixedOrbit.GetY());
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << std::setprecision(17) << "MT9 nucleus dynamic-minus-fixed: real_log2="
                    << Magnitude(m_DifferenceX.Get()).Log2()
                    << " imaginary_log2=" << Magnitude(m_DifferenceY.Get()).Log2();
            }
        }
        m_PreviousInflation = inflation;
        m_PreviousActualError = actualError;
        m_PreviousX.Capture(state.X);
        m_PreviousY.Capture(state.Y);
        // The independent oracle uses ordinary MPIR operations, no MT9 product
        // workers or dynamic scheduler. All persistent destinations were preallocated.
        if (state.StateIteration < IterationLimit) {
            m_FixedOrbit.Advance();
            mpf_mul(m_XX.Get(), m_X.Get(), m_X.Get());
            mpf_mul(m_YY.Get(), m_Y.Get(), m_Y.Get());
            mpf_mul(m_XY.Get(), m_X.Get(), m_Y.Get());
            mpf_mul_ui(m_XY.Get(), m_XY.Get(), 2);
            mpf_add(m_Y.Get(), m_XY.Get(), m_Cy.Get());
            mpf_sub(m_Temporary.Get(), m_XX.Get(), m_YY.Get());
            mpf_add(m_X.Get(), m_Temporary.Get(), m_Cx.Get());
        }
    }

    void
    OnStepInput(uint64_t stateIteration, mpf_srcptr x, mpf_srcptr y) override
    {
        if (m_FirstReduction == 0 && mpf_get_prec(x) < m_FullBits) {
            m_FirstReduction = stateIteration;
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << std::setprecision(17) << "MT9 nucleus first reduction: input_Z=" << stateIteration
                << " bits=" << mpf_get_prec(x)
                << " incoming_actual_error_log2=" << m_PreviousActualError;
        }
        m_InputBeforeX = m_PreviousX;
        m_InputBeforeY = m_PreviousY;
        m_InputX.Capture(x);
        m_InputY.Capture(y);
    }

    void
    ReplayAndCheck()
    {
        FractalShark::RefOrbitKaratsubaProductsT<true> products(m_FullBits);
        for (const auto &snapshot : m_Focus) {
            Replay(snapshot, products);
        }
        for (const auto &snapshot : m_Increases) {
            Replay(snapshot, products);
        }
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "MT9 nucleus summary: states=" << m_DynamicStates
            << " first_rejected_Z=" << m_FirstRejection << " first_wrong_dynamic_Z=" << m_FirstMismatch
            << " dynamic_mismatches=" << m_DynamicMismatches << " fixed_mismatches=" << m_FixedMismatches
            << " certification_target=fixed_mt9";
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "MT9 fixed-equivalence summary: states=" << m_DynamicStates
            << " accepted=" << m_EquivalenceAccepted << " first_inconclusive_Z=" << m_FirstInconclusive
            << " sample_mismatches=" << m_FixedDynamicMismatches
            << " certified_mismatches=" << m_EquivalenceCertifiedMismatches
            << " bound_violations=" << m_EquivalenceBoundViolations
            << " fixed_rounding_model_violations=" << m_FixedModelViolations
            << " identical_prefix_differences=" << m_PrefixDifferences
            << " return_certified=" << m_ReturnEquivalent;
        ASSERT_EQ(m_DynamicStates, IterationLimit);
        // Mathematical-orbit disagreement remains visible above, but is not a
        // fixed-equivalence failure. The production contract is tested here.
        ASSERT_EQ(m_EquivalenceBoundViolations, uint64_t{0});
        ASSERT_EQ(m_FixedModelViolations, uint64_t{0});
        ASSERT_EQ(m_PrefixDifferences, uint64_t{0});
        ASSERT_EQ(m_EquivalenceCertifiedMismatches, uint64_t{0});
        ASSERT_EQ(m_FixedDynamicMismatches, uint64_t{0});
        ASSERT_TRUE(m_ReturnEquivalent);
    }

private:
    void
    ReportGaps(const FractalShark::RefOrbitMT9StateView &state)
    {
        for (size_t component = 0; component < 2; ++component) {
            mpf_srcptr value = component == 0 ? state.X : state.Y;
            __mpf_struct magnitude = *value;
            magnitude._mp_size = static_cast<decltype(magnitude._mp_size)>(MagnitudeSize(value));
            double mantissa;
            const auto exponent = mpf_get_2exp_d(&mantissa, value);
            const auto setBoundary = [&](double endpoint) {
                mpf_set_d(m_Boundary.Get(), endpoint);
                if (exponent < 0) {
                    mpf_div_2exp(
                        m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(-exponent));
                } else {
                    mpf_mul_2exp(m_Boundary.Get(), m_Boundary.Get(), static_cast<mp_bitcnt_t>(exponent));
                }
            };
            setBoundary(std::abs(mantissa));
            mpf_sub(m_Gap.Get(), &magnitude, m_Boundary.Get());
            const double lower = Magnitude(m_Gap.Get()).Log2();
            setBoundary(std::nextafter(std::abs(mantissa), std::numeric_limits<double>::infinity()));
            mpf_sub(m_Gap.Get(), m_Boundary.Get(), &magnitude);
            const double upper = Magnitude(m_Gap.Get()).Log2();
            const auto &budget =
                component == 0 ? state.Certification->Real : state.Certification->Imaginary;
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << std::setprecision(17) << "MT9 nucleus gaps: Z=" << state.StateIteration
                << " component=" << (component == 0 ? "real" : "imaginary") << " lower_log2=" << lower
                << " upper_log2=" << upper << " headroom_bits=" << budget.MaxBits - budget.UsedBits
                << " difference_bound_log2=" << state.Certification->AccumulatedErrorBoundLog2;
        }
    }

    void
    Replay(const MT9DiagnosticSnapshot &snapshot,
           FractalShark::RefOrbitKaratsubaProductsT<true> &products)
    {
        const auto inputX = snapshot.InputX.View();
        const auto inputY = snapshot.InputY.View();
        const auto beforeX = snapshot.BeforeX.View();
        const auto beforeY = snapshot.BeforeY.View();
        const auto outputX = snapshot.OutputX.View();
        const auto outputY = snapshot.OutputY.View();
        // Exact products occupy at most twice the source limb span. Include the
        // coordinate/output spans and four carry/alignment limbs for exact sums
        // and residual subtraction, even across the deep nucleus cancellation.
        int64_t high = 0, low = 0;
        const std::array<mpf_srcptr, 8> values{
            &inputX, &inputY, &beforeX, &beforeY, &outputX, &outputY, m_Cx.Get(), m_Cy.Get()};
        for (const auto *value : values) {
            const int64_t exponent = value->_mp_exp;
            const int64_t bottom = exponent - MagnitudeSize(value);
            high = std::max(high, std::max(exponent, 2 * exponent));
            low = std::min(low, std::min(bottom, 2 * bottom));
        }
        constexpr int64_t CarryAndAlignmentLimbs = 4;
        const auto exactBits =
            static_cast<mp_bitcnt_t>((high - low + CarryAndAlignmentLimbs) * GMP_NUMB_BITS);
        MpfValue exact(exactBits), difference(exactBits), exactX(exactBits), exactY(exactBits),
            scratch(exactBits), x(snapshot.WorkingBits), y(snapshot.WorkingBits),
            xx(snapshot.WorkingBits), yy(snapshot.WorkingBits), xy(snapshot.WorkingBits),
            temporary(snapshot.WorkingBits), cx(m_FullBits), cy(m_FullBits);
        mpf_set(x.Get(), &inputX);
        mpf_set(y.Get(), &inputY);
        mpf_set(cx.Get(), m_Cx.Get());
        mpf_set(cy.Get(), m_Cy.Get());
        const auto report = [&](const char *operation,
                                FractalShark::RefOrbitErrorBound bound,
                                mpf_srcptr actual) {
            mpf_sub(difference.Get(), exact.Get(), actual);
            const double residual = Magnitude(difference.Get()).Log2();
            const double excess = difference.Get()->_mp_size == 0
                                      ? (bound.IsZero() ? 0.0 : std::numeric_limits<double>::infinity())
                                      : bound.Log2() - residual;
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << std::setprecision(17) << "MT9 nucleus replay: Z=" << snapshot.StateIteration
                << " operation=" << operation << " bits=" << snapshot.WorkingBits
                << " charged_log2=" << bound.Log2() << " actual_residual_log2=" << residual
                << " actual_residual=" << (difference.Get()->_mp_size == 0 ? "exact_zero" : "nonzero")
                << " excess_bits=" << excess << " inflation_increase_bits=" << snapshot.InflationIncrease
                << " previous_actual_error_log2=" << snapshot.PreviousActualErrorLog2;
        };
        mpf_set(exact.Get(), &beforeX);
        report("clip_x", snapshot.Step.ClipX, &inputX);
        mpf_set(exact.Get(), &beforeY);
        report("clip_y", snapshot.Step.ClipY, &inputY);
        products.Compute(xx.Get(), yy.Get(), xy.Get(), x.Get(), y.Get());
        Oracle(exact.Get(), x.Get(), x.Get());
        report("x_square", snapshot.Step.PrimitiveErrors[0], xx.Get());
        Oracle(exact.Get(), y.Get(), y.Get());
        report("y_square", snapshot.Step.PrimitiveErrors[1], yy.Get());
        // The cross-product estimate includes a factor of two in the real code.
        Oracle(exact.Get(), x.Get(), y.Get());
        mpf_mul_2exp(exact.Get(), exact.Get(), 1);
        mpf_mul_2exp(scratch.Get(), xy.Get(), 1);
        report("twice_cross_product", snapshot.Step.PrimitiveErrors[4], scratch.Get());
        mpf_mul_2exp(exact.Get(), xy.Get(), 1);
        mpf_mul_ui(xy.Get(), xy.Get(), 2);
        report("doubling", snapshot.Step.PrimitiveErrors[5], xy.Get());
        mpf_add(exact.Get(), xy.Get(), cy.Get());
        mpf_add(y.Get(), xy.Get(), cy.Get());
        report("imaginary_coordinate_add", snapshot.Step.PrimitiveErrors[6], y.Get());
        mpf_sub(exact.Get(), xx.Get(), yy.Get());
        mpf_sub(temporary.Get(), xx.Get(), yy.Get());
        report("square_subtract", snapshot.Step.PrimitiveErrors[2], temporary.Get());
        mpf_add(exact.Get(), temporary.Get(), cx.Get());
        mpf_add(x.Get(), temporary.Get(), cx.Get());
        report("real_coordinate_add", snapshot.Step.PrimitiveErrors[3], x.Get());
        ASSERT_EQ(mpf_cmp(x.Get(), &outputX), 0);
        ASSERT_EQ(mpf_cmp(y.Get(), &outputY), 0);
        Oracle(exactX.Get(), &inputX, &inputX);
        Oracle(scratch.Get(), &inputY, &inputY);
        mpf_sub(exactX.Get(), exactX.Get(), scratch.Get());
        mpf_add(exactX.Get(), exactX.Get(), cx.Get());
        Oracle(exactY.Get(), &inputX, &inputY);
        mpf_mul_2exp(exactY.Get(), exactY.Get(), 1);
        mpf_add(exactY.Get(), exactY.Get(), cy.Get());
        mpf_set(exact.Get(), exactX.Get());
        report("complete_local_real", snapshot.Step.LocalX, &outputX);
        mpf_set(exact.Get(), exactY.Get());
        report("complete_local_imaginary", snapshot.Step.LocalY, &outputY);
    }

    mp_bitcnt_t m_FullBits;
    MT9FixedDiagnosticOrbit m_FixedOrbit;
    MpfValue m_Cx, m_Cy, m_X, m_Y, m_XX, m_YY, m_XY, m_Temporary, m_DifferenceX, m_DifferenceY;
    MpfValue m_Boundary, m_Gap;
    MT9DiagnosticValue m_PreviousX, m_PreviousY;
    MT9DiagnosticValue m_InputBeforeX, m_InputBeforeY, m_InputX, m_InputY;
    std::vector<MT9DiagnosticSnapshot> m_Focus, m_Increases;
    double m_PreviousInflation{};
    double m_PreviousActualError{-std::numeric_limits<double>::infinity()};
    uint64_t m_DynamicStates{}, m_FirstRejection{}, m_FirstMismatch{}, m_DynamicMismatches{};
    uint64_t m_FixedMismatches{};
    uint64_t m_FirstReduction{};
    uint64_t m_FixedDynamicMismatches{}, m_EquivalenceBoundViolations{}, m_FixedModelViolations{};
    uint64_t m_PrefixDifferences{}, m_EquivalenceAccepted{}, m_EquivalenceCertifiedMismatches{};
    uint64_t m_FirstInconclusive{};
    bool m_ReturnEquivalent{};
};

} // namespace

TEST(RefOrbitDynamicPrecision_FractadynePeriod410594Diagnostics)
{
    // MIT/Apache-2.0 Fractadyne bench-kit scene 60-mini-p410594-1e30000.
    // Preserve every decimal digit independently of the external benchmark checkout.
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    CheckMT9FixedEquivalenceEdges();
    CheckMT9ProductionUncertainty();
    Fractal fractal(1280, 720, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    constexpr mp_bitcnt_t InputBits = 100'096;
    HighPrecision::defaultPrecisionInBits(InputBits);
    const HighPrecision cx{
        "-2.8041054305504546698407770028983979273643258419006230007410381499044388400475119315630"
        "2939402835890872695541844511381853254064367395154734299739738212953023198883985608313668"
        "0750669080499973297978167962228901888145316384153397660013636627589100453562250014991711"
        "0824017207959673925623508173304630619668883563618713037818321302532691505739478167201179"
        "7448226319454382989911082965893494940744741852871198166747075985102104172927711491648659"
        "1916042269947061362608053059787816946740130750978854758372275640244789332857493833804021"
        "0405147738162520148953110435401334906308454160623216196242101020484780494111493638383057"
        "2886238647971817719507402555617790376587844612140455901397860020062089397481330777710296"
        "2505971296846855406323001963985329640578776476051291570733281216621575999548484469305147"
        "7126029705413991799204876111822867979205534626491677029722938635041972553576693097549011"
        "3325597454522711078528097366789176248580372629289271566669238786989136870313220812583898"
        "5869722663307117339446346482967206629048969762559369153975996403417069183894292829441337"
        "5977910222738035446457122389348999587261514494293460996125742722918124650832424834787744"
        "0280933612465938306007486721724415482974924321881538466480189122518420570235408377978290"
        "9414057426034939270564980678370092252697860648484520015152986691572885766701692664800446"
        "4364546414359257040337386235881985551773827967953607750765158828324378434621882530452802"
        "0338199751734561067008089571663465421384308911619909790168115822302011148017773087136462"
        "1309119680251041486889158303427857230808274744992229382880755170983859397352346465265142"
        "2981278257695793080497523281249240668801063009488733294646625814243401900820225117063274"
        "1335835159522545381436183480178864029044916919204383687674800363219539389307860711387212"
        "5805741790955682030734470543999185189534819387450148700521655336618329650720957863578256"
        "8325324181938691646706303291704420582884789848798929557995487602444506815286611694420200"
        "4543227939731333181370392645211439159069234626176503678228818411693468362045512217514928"
        "4631385423356464642373734567230509919610157771152484971424079373779334422933814184956754"
        "8416684331561197164796763741754701201148513327521336859646170322801478602829101845520432"
        "7403056651776885509914749061757463679254823280828585441448082256351758104263049090325039"
        "3330315388854838361152201999177724603092723775252717927257876731432924051157651368893912"
        "5155771903468084951819611722948281302565679086658550254238312056218812892468057508887771"
        "8784459164475661241180104067105204801650739608192279623003961280225054846211560485802658"
        "2645394554813154478916286485354302516637060385511200648215189918316242245856340534888830"
        "9641175552465906721676592723250959494670152930265325268670095226278210515719273912205424"
        "6514148975704544219068041651599142787912732016845066254437654259809780532873958260561127"
        "4083092910623746053640854351444869766227689650838359473670726471285804620930714582456770"
        "3925897319417519412462504058977247831674579214481642810007195738411022286363270845145581"
        "6617056738727378937913418171866248465703666542199233158421080151006533201535739112442105"
        "9549318739010619842226448741843715590288234416713958046077224401745210046082245443990630"
        "5380689872774147935183064823473495793022887480251918833046199696705696686124296595354882"
        "4807550863720233260450467374897261642385996971722493277392154559261045361505157013497166"
        "1124773248190423107213699647694863763389034224500530920564159124451552059604137147502149"
        "4767198837365926770541676973709146366994725079614198543141609892907795231661944979178560"
        "7684892118297613742602191047329422332737006258470057367585366811586465291119664492124351"
        "9339822924971200052172682899910386005399123119095884087161967791455571857962921362181572"
        "5982700048820020963043145185466952022103471640516033569615528065223533588064096714755797"
        "4987495547921634932476573588798423404258168950009282440382481774351037320807142211069565"
        "7731423051427619573619608434955150660794585883576097741640548873186373098197006425542387"
        "2005865765913483214035741154389218832315649541188132913095607221010427187383732409233509"
        "2722486269659114818494704030191562944405714241548372122892091115963400293694733560845061"
        "9757759313117658640494009238338997845123530708888694592118360758952886868028402018369171"
        "3733069516718325453926993389118030943293908056979654442280700118348821823613803147183175"
        "0131814072262925973776410218176039332863531317466622138416779698986316686954947324003439"
        "0972560145653758467838542184257494406914449902756208392447433322872552592072300472250983"
        "2592724407077025807622052710647209245048097206227914781990477584765733623895044545742678"
        "0638768702259916982285853206778098188456398112251302881360106012399287157338589177911505"
        "2660802700630660167248145301431325855698331162449983006730810317276480246428995394575097"
        "5670974727443613442848137326784132836090952214482571770683699771076448311487759746410075"
        "0092423589569325416658441788726173880129057449429600986112397467539929959095610910682046"
        "0777827218509753798790273361154162256212492147778573738736796373636293733976655420482711"
        "9214147773067298962986105859770472502022706668813236295071627900617851817202820099464948"
        "9162888927817906954843702061696638721168657947298133886496736881033932139262845805152178"
        "0855360195133269423686688019827588688373189705883923147597913702113415892353529887647297"
        "8050917185232071143805627552444514354543186637528610878282708533998006686050519342804353"
        "8519133659615895657453565167889166495373774223757605225784103410558261728399811089573188"
        "0007383379371266545790313839682072285145644633136054982668737525940916615746970882909344"
        "1354008663656179945341328992301614681142765137728813051813262046914913969466933486958276"
        "9033642727179257119759821951420763622163674873389487411268754285917423513713579568839911"
        "3276992420115253587325720143137335025970014949691194602397193551062337168035640741134108"
        "6313803579855475523725737720606471217233833773433298792377651438874355610188947749655672"
        "0521048768476334458291000895547858541731134504008106170091282297938128457311253854002879"
        "4011989449324850339067648041737688783461616004275099687650803131427121287048645993708338"
        "0639762619865893803032702000477253096296878129410344454570331494816757075855004168478700"
        "6975671634324514131069562312324837520943686542990432892853227658335974741334480870155675"
        "5553491465146361110744061348598107647946438596767318673165679477358242978627911804665703"
        "4181273967665558372896668152571143414083223523613702020665951901773248916390307742046195"
        "0112583689296066557359988565462321043910882955110583118856101025188732936751535628225101"
        "5288191442651258099862732947899367813405143459247969366838210751034562622901187770851831"
        "0583521357121984884119995051121932542879748542356695026250631850551354019503170758710370"
        "0873314583049083656511797448717256751251928883896456925312300956704292076545653338351458"
        "3214342014911246531170822504727604325345441501891775519091200715421307742438973568431069"
        "0740471013550027593352295863733545735495547372771662353123285491342538966475458929059466"
        "5411722166060724759365675732438654003581467242651982342678736772874741885586015078606325"
        "5402831261905500242526538402629474797335827330304710120193028888306848963412322358297448"
        "5015705298894738942980050888107620403148062502313220559729942678720423816845030304904174"
        "8286738541688833744979297187587974845311509054377476092761817914417735911192718696738027"
        "0837853356504285638864994805874280048067345970219528636672989674622223254474353562906828"
        "1641392781889302588010539127716593287054088118704677781178913940953742184370351623877748"
        "1745555237424233653456970990361625261314446082375833777902094968745092658177587959121028"
        "4158462381757667776907748629412425671899500545439135527680857304811470419676087131476047"
        "2695577815162820462805495944038555663981606094565940754717265786443783726640299736591551"
        "2513461834459739705534542255644659055029026225091088284611985452713839599790889134763126"
        "5411612412928610882752757829099863662244387516829236680181287724856023804016340142721231"
        "3793121718008683470713133650467647532281830197783178549473577827422729398263314933368888"
        "6186698117839855392821830513964901360385675408226501684139577693385920611724143790322761"
        "3345054774857873107908943518459943808751698849137007275932696116311640209646952306458680"
        "2944169357363948105585401135589772966265876209198027033390995145239765564593645255208851"
        "6380630043605008692715604154523449666925577414924905074912596077609200819091722058122978"
        "0065078786824736161802787869638505865805939924910533842033659562727018344461803344710587"
        "7345618993575405765654905651407871417629809203399100455394052230860437688917730771077702"
        "9729077592967503623512314463967679542940739606614506849555554445079519642946297453518664"
        "8105997206016274811276277929791543215135061274243244954539987083100868937594556968971336"
        "6057043305263776629632556521217425463195715683711497089955433368336930596008830509268773"
        "4741751814505965014145818074694043598665221516295265914066347385062213378319274158739682"
        "4283311682171608321412795525020424927514474107838914728278904745908933616357161909018664"
        "2224970833017250363609515731811378197353050263103868669214650757986596971783180087702204"
        "9335831306409594310310962559772734801489444725726349002586001466851870017457900820209727"
        "2961212092831291541057077580103494823559138720188010462140226699753869564263631834187358"
        "9758805517293956435536762153955121535447376132300590269707058531454012331930641558022407"
        "0787642241686713424879197532125857706552711212108244076168528486738101883725317603348828"
        "7069724221751415803265750209160039361218228057421886107458521599253261629114146805328930"
        "8784287448779326208970397820908957387418678296085940309787940217745720807013107309508064"
        "4614947892909628757184404611825240637155680233821737746054738823073258829286918794665478"
        "2685647508359585415803319541121102803471660755527667102045604327407593023356931522851716"
        "6701186423423183625198055560181457824038550327510106670447814322813729396733491849645921"
        "9128118437093794803503922322392544972198861093937724205484250464939997369678371113617524"
        "3509790472738604634322705574897803777325496162476649970354490502252490839351240581346530"
        "6909125496521402421545756848270165183039622101401832402942173203204852029271983451882453"
        "0557597100456237381518787362476123791777470263561405322084441618063722775420659159398171"
        "6061773943725885456763208810499234062411044517650376890623578400504574070561633965647461"
        "5898198359591667624155743404672217788737549066626036216707181604387383402461699500691542"
        "7877568029768814490217138631779900940984981854824785363138858070426023162470010959670757"
        "1915371419660928745467781931764915382378075261252131546055644323782809379406031198159489"
        "4500845860071356319507991836209981052316772190050922347808767604146897241216703030543159"
        "7923743551528799912139473232656833110760557113859235036556922076986242065210682331881297"
        "8156652739960790473791494429914064304714480255715059695015230960747648081753147811382813"
        "4995749395554741168804564894322095303564992185651825022681731722020776914081244705478157"
        "9145775829824637175523591211973683806664806622065461657904204093628568119322403786257294"
        "0230526371732541642786508639344256661728730378440454317705902165587177609089270966193179"
        "3722697820855043745453666894844033163551305050825869705846579025241097216879766747823158"
        "5062209787250773808152388572339978577572548234773000562689851569499899690010892783618477"
        "1446589989176539516549293651347133949425962083674387154523711600058225753319508534893445"
        "3504142703256126475963292294547198036164337465672019100763341021964166926631574196032696"
        "2874703984377302428385985795058859044386211099573669039720779041630828330496008249422045"
        "8122002260477633629113513905845163789244826735250895736278982555457002654078458642524146"
        "5225006248121478062596798609480581454649346066267322182665316765679647960022117949093461"
        "7601051123130844787372161892311852728839612381489783383524044838396080705434481980045581"
        "9179470761002781284512639320230680972952706185374335631639856393789226354981680105231723"
        "6138100760627756847595501602283807208919843707614977230754361991739179391463802370516561"
        "0333318440061936491527317598208923499528697629526329523717271257062042936224122813445348"
        "6661076446120503360605141061723999388821443647492321923442997396597735133010986685511377"
        "7919442631901924310921851698022797535947823267544377223213020303320737727779463238973826"
        "3106415761271693394112755452586313147264359105313716201707217910613953656495360772576962"
        "8597566892154304117415343970279033677422827979171919888402146565584241055982948295212808"
        "1997503756099395669389099342096872861115771834086972983304814586468789518024920268699124"
        "4784704996770554179768836385317473423264725818933496821209393947513419778166517105296516"
        "3038053470278302477577306110648899646427014065196588166426158388817157924299615707931976"
        "2469358309692318206671147120207233044414214585580207617297260192930014051197136295600538"
        "1826771215626058269231952551796818381716868318558417248101340819270092690378673234933661"
        "4237111493472289750191006684877715949073120387650043640125961490158107681962505716079358"
        "6260151672604127867859138198839773419703317794114367873591472618550480837022028013433273"
        "8814701631427061611868051423658273508241730795517977627496995371020608486290331674919605"
        "7397461323289456991496680068298634044632900654300879657694207464051018472177185825405518"
        "2294938158020782350446978076503132144127928680735698321716124375983598383406009370067667"
        "0828819566921992946284198531128614164392259484043960338264904410021030834149651328304283"
        "1742952450322675713411731507569340811568551094010014680959750703855510654899881446886303"
        "4586725678753383192586742082131984847137372123459353043284881270294191266866755671901665"
        "3757995962030966073052900683060155070968764482344765115502799252044433555562062397207162"
        "3557857505876388094487402251088111358082549431637374623784736515184240495268824150220131"
        "6622901809043940550752088142160964587616176891614048389690642920288765204003685136863716"
        "0005532265957153824931429954126573943108787764461285171839856837331973149083359754776745"
        "8775835481293605438886540717430482975792655326266097245860267342105517943369685851209432"
        "4999045789467468221165253801538815470974857890720156906019308079391059705710982872204408"
        "3675319171091071589530758656982460368321102017624123523260373404874152341577809827829062"
        "7307390578916218541132787361266688135759370107099118355760856008396990205820506005918655"
        "0101237793577297508370253716000890953627118703772922118056666461943930433442133085062748"
        "2376577214613370115307725442278099226356325911879856821356567998230742853026076574001859"
        "5638425548503650114157185037695636592148810107150177882201885188856010981942750138429140"
        "5434299749563678952575299889448401551711046946947780549646178606985915320148905373440599"
        "5951740617488336414723908122408295646373844349132303930780061734560344647305444305290720"
        "8905320703303524068211970676514741387422734378644444332379528442931675188159397512342426"
        "1754251695244921343740141321092752378404555106248279542934119175641261572413939809256818"
        "5500959312372134384908415935184886578323034590217579758674363764730170272471200585682859"
        "6295233194718684750317392351878854888725636537139908269346116329677470863594055889919853"
        "0173735916079948100632536479765317856696822598953117584246241883991728267086967772595832"
        "3775406074535122412907140547423372512501481315614292435438954236687914259707122832851899"
        "3171156737002061446662310077426056702731575289859545203552786204618437333604910259073310"
        "3375552503179859534631483266637126360841638748066265143043416128271252653531246595510556"
        "5730528783628144971425942766453913712748312706093031441573907619861194104136425767971397"
        "5837718581819852061553664949826271265404084732003496216831795059695571323174298028281412"
        "6006787961965737493358896485475925435593170843095686388934082756852861650031296673406426"
        "6102196829582235148190052325727024436427253610808352964313205389033012358923619924530439"
        "4614725700541038321058811422168640970676125539613226727030567107643988407764960952163217"
        "1531833276183275595657219313703704771294580767390748147460937527975143069189604790915236"
        "9964810006368354581188756410951922196563450901391584848329463098408668367533326918213009"
        "8004885137125517826728171104326867674733135451053270312825698059636679238129563826792714"
        "4094321320770319392187263195145030275687905882242935260574512254109343164620452344999160"
        "2485947442525804452871973007677041479285348690684568134080655530115555447619520477298306"
        "4144486106585458599000213211812948495067003976488486334669662968195547468758087659109248"
        "5038378100275862015431943223349884788986060644820718529997963058818213352644073181570756"
        "6940640402105685669014460098644315582466986801014805715234933621305075731112932951143343"
        "2460199934503072682762439381621933138856942016420303354461716909396330695564495123975334"
        "7317123291033204176962044029110026898060715921425221570079400268142301852101942124279710"
        "9427826834542912339412031248695030763660981869353458971165393200561063333651987827940211"
        "5975596112150074905047825271586505781043393089355725370761087026422809576029673336702206"
        "0557056647382707014032149661291950608363512533981997431646211036103380423008759907820677"
        "3492158667239612852658861823789277676979710772482147540119817485483300453502496609798197"
        "5484656978593961724032931491918006196439019810485632043336240745892698170811325048988079"
        "2825636636977404141704116969829920557328763926457068859444783107677957859574766521342389"
        "6770134111963678004133771898333321305830316810177993764368117020935898479670608008724903"
        "0322559320994743751758392724586011575518185115661053125080805021537154683445928854898087"
        "7929498811360914317195596973335872964841719516402876207581152564752384483941716525060685"
        "4927641051448686787550489654579462637241675216021145059779999155536382550116132329149012"
        "1112225799955386498145977687571879270166719578852271116062768057514835508517761526036788"
        "5204755122098605235708936529738129117936355326112659914833903885465384247986006355847692"
        "9798376568956816698193955898523262368261033883359375988997999309094967869260783317039081"
        "8854807231579908756362419768600434176187106808960295104591384599233867505372503279601270"
        "5248698129430420635038413964550248162292355245439130993258722516243054231584293527896561"
        "1620768259802000377672966180824853260841496909837953890855503587902566358102349938923708"
        "0099786748639886879189512280245263246800715915072798516791757917862545012030710125690148"
        "7804365145781713523269793430941221433711328361942614291968430657930700084291264395021566"
        "8587421311764387752079642014451486250095923379767760268912710422220687753503526747605846"
        "5842134947628303569009800178372541953450803028408461906072690578749507371046989672046835"
        "4077604753449436752413647493280151741553798779879457161295057386631999227679874158618684"
        "1587866568276500341762083237680103734961471065404351053433187335547124760129149484217508"
        "3372349063221847385991316358021202449205142484981069533881845515822692807097837371915241"
        "8882335887644042740295670171912938013480164406784742827593502723002932294089736138508071"
        "9574130874552950542814847300836028824766080530606795280467595856420203489196771203607588"
        "6342859665802762480495473298991890188938044585323480074450898663319174108164745089438342"
        "6026565797872534679634711109825383498226803077621051871916622255801461733201081949844471"
        "0879566088050331737076069714144555032936326221371159575362979245740829249055189521891861"
        "3750462370230782587464777538084404523355488035548542755106131671959383502046727823525228"
        "6415731013495391030767870116320948136708097099310804645926098620742414926136270447284904"
        "2045368698934568655291155218490808328325479172756853798459965623266826404227121065547439"
        "7984984976594926030082364897228463317226870355010298926421763249596725583560736253809780"
        "5963049434492474070135809726174434235083884368573208158636077377176853131496372551725327"
        "5248106317108502069826512738215100689247666491033673227117508297571456998476476910272060"
        "1923007912631856532923254498958297818565773401718540535537930188106474908987959554827618"
        "1560533457545596341375870994883138808777862157239802723838142529901496399317632199514329"
        "8767080686957434003740206653220108707194984821860737334921042038913866044738522350289302"
        "8655211496648239603344512790074566684338766607911195836581895689252839862104476498486210"
        "1011328425709592550272195920141267608765418346971655750892516181266862216142185627013548"
        "9580333964933688201232174362882315462352789322607059612149476690061785826103833223293288"
        "1247380186867652197192059308809460528904398059245250488424984882423866090390502804539471"
        "2381981192702224922126486251098273132222919196748383812821989757667747628143086386383366"
        "5282242951702751788504337228423118002326237314493184105019676929547942491158632368007907"
        "3958751900829730923075450059640397968052097279697981370305427836441374433645497882325041"
        "6800959090066199925989567377715295208047462509502747350266026199484887537026791112611646"
        "7053781275887779078866143985299741767159864751822100098120561046594152572586132250219995"
        "2687625728169750834942863671230858757404217948712774334316682150069136591267689534956280"
        "3528789785172652064549319390710971345385710026127420986699316631833996905309948189764731"
        "8712079024784272622227682432627601469197306630794628671704360996147052570824412689266488"
        "6159088068125412834323860295577504879358355861693056411389441752294211014154637593449351"
        "0611066202657775243836842512673895359077729408662772635238702700084472797860621506267930"
        "8872707729522794797123065039298766878890767162638940544751607884932421606660589347169070"
        "6253850789615251152712958489958452306041698594783030004053430925791629639932537437701921"
        "5506192319380480336026572399706690092298242548036304688034361708626069469367743236078361"
        "1916077335466301372483372324096764609350065077984466906974532023164690761988198332027552"
        "9055473670134653011783991081902581162284329676940654157655509842238885021479282375464309"
        "7136720186415371945371760962685299829312857906397270949526172841978053663998798688421129"
        "8512316391948323974648025822344332182946484887393945758379572232526620739915866034024122"
        "0803346103764392892971535700542053709393483603746451185442788122994896684376785069211184"
        "9476156899787264165868356905319395453849464784574928892038140549593893023458597583718945"
        "5316080151744381993267147680384592597371067146901550395731770866123500899302974213878970"
        "3604351421977535456355183762905090261091932008104303245730212709230582500790765314402469"
        "7715943730761471935538347359186659111881368323475800322072274503583574338766273274973269"
        "3713152008977438526135289151174261442016033279610831236794200517254966697986251320880795"
        "3233837804619651993147642695102387188398481359971400921735187896152049751401574277116628"
        "6699691774751279491164193705496490749581777383336967321832004507501720817891281083992094"
        "5698009598119894049973701983308650819960326850046332103441942557373180234743214676869072"
        "8287714386989523832215321104878891683727701212008749585502618411286385967605001780103270"
        "0084852752966267444077823887267460951970996318011145121324386870662609416388075801897561"
        "6448360246860849349707652873886975803963457408663441124415869451141471881417213341379208"
        "1399528262972183611898949661765262234135129252154664692363890493100864655748017631005583"
        "6074851836095341749048176487939501556425100741119534035080569699115276267055455179038243"
        "8000144941778531703269762915689341660152489052628228040956829171111982258196064209210857"
        "6594337392756517504102954968071176082885969740146212806635892346500171544889837056742593"
        "0346557814957772249019179919257377231792342713419955761505815969667935999172614204422276"
        "9079159333472215289945519428132876424810075592206600496747169711315858518828778757131792"
        "1491732256897044375278419343121357606091664258883619899199522046046042084744042211574593"
        "8550976007288528621705999766571828191082995481207820992456733842424691259989172180015214"
        "7026683374216528476998393967712109909452635087979338584777028188669493946574771650818012"
        "2588421892898289666578494179106637444527272393747948122214866732444505668447175372333157"
        "7842469125962182103718891867428738067240115250832359715389271784051233908650686598450254"
        "6737626833185042843137754474833083235090398412507843760441658716129031679429786839714080"
        "1919457564286264822460244079534167618717607371562123463842090462508437512679301649321855"
        "2756820395735453967072138541831872074893888987407435552747523809907583278111947869980257"
        "3303365164228675683592846557619327185433812142553399392805919076416727575540245400421580"
        "8841276992181849421192411164020690873818215684116301780135538403467296180246663717843893"
        "3637188757361407036605685679847169301659435539842423752412418793019611808841527241245189"
        "5253794612963365019789732756213940318508815005728981476361470151056507735842064043941198"
        "6197495570698396936065209512738304930840299951425103420872211319387767421617547385911734"
        "0353776526837020462709468918886633988384347628601450699884287499962054023635412182288969"
        "7824187087823690676232759774249895880612449600568525167003080401597155273770992131109630"
        "6984129023553876025552756211109544790005071408521205521144299061961637346370813636420014"
        "3561842168303326999472309842696554609761999251011532456981278295902397051862705392336504"
        "9075650814707194594400810649034753721855933968840212861534852120413489377135160376253235"
        "0959016196991741771431846058203357033051953291424844071608561388335057968327858695163140"
        "4496907042922912825395786590693519980941219551945710869921896636139768571656641515957131"
        "1736336142008856200249138234184196048549337713624524692815781997047627004632887182438913"
        "7161621939770730557738486769066194968627125988109111887402223269438724606461372546518897"
        "4585112312106098966981416301432485030435375048701425820629816031464735282023422980981980"
        "8625008412773362670226428267905761878984389958034213777477134048210735398366306660257356"
        "6695346057571503178915902772761721968108310936918678711976606763301015869058295547036453"
        "6756052836973466952041458410253390633531365072894852286156296456079762217572209633463840"
        "9823153284662177396489188789170811915006911020142360057967949628481749082219599064597715"
        "8594043516868482919982641999292086563522924460618184750238003961813564656366559306721645"
        "3301090473801918221149955053188152963441142863313818979062949949581806683075303734618377"
        "5462696357792454218695868280203488427350696766152620008183210849647783411296876446512466"
        "8794870763929660181079582443223705699150588308006671075389388280765866729266024146995799"
        "2761948194630912189674322659700950545899621848544529293493306334559568113173015433199177"
        "3682879024397277641723338347987144756385641574217079827615440954120670923759937279904285"
        "5186646676691277677944557604076422163422632378160450996992253623922319720112757708114947"
        "3537875658292863762381338573906245841187321504211446824909839716859616659693691099830806"
        "6594812005871647589773656763965023138116776455850008019349530916904743090532070742608147"
        "3849982325041448636145842968062814818629558562502581116420930842112503717507298925511864"
        "8723111824104528883078400507840176648355975214254224305323515422362031571475789989283303"
        "7530141419221743268526709593610894075431922193517699941790209793484415014678061261461985"
        "6353876510206476524356817434245602286170788726385183384332598893888274108788257976113290"
        "2132627972610444976153025091014159980245996618571595194289665013010237424584703106291943"
        "1858215498036460626564578096459134669422330956368559567230833969358366890527277811567800"
        "3759887667917409927357913701968549739490670680052080006892227381042748444932679886496009"
        "9909704490480454097884939664318974409127284552328742579702265960001179523331272916815628"
        "9456663350042593536274026229643683519221409519857975865747889388674698211759244129502661"
        "0002718113299372576059657754145391901478308425112693575835598469402652153044669486561519"
        "4541544930805796694138078012082440133687001344259805976621422678437896691678112845867970"
        "0403806744480480291012861167059482065767372875112769218902010183275586281506146886809741"
        "9296065885989005538349328444148496059472354404730407640630104411255868162226144708856442"
        "0728852361683750592767642936472450911954566070914238636719012402908766187810643465680650"
        "9094484693763543305361076134986862195005819066955192949387592306472940558692671296177179"
        "0996252819582934709830445469994332075444233609913176553154059067481750072338355486724360"
        "3771309184800378482946652606130935651294769284103106539874424978300710018661010204881964"
        "2664803467543516142720853360963481695852764877501867623695925320175289204065880054499967"
        "7491000654860887405261159125381247542416673670067829092240063626371047953934599011074100"
        "6060964850925105640763950223306496656400445781154186887619362189782098794385151526252823"
        "1627727594630785798623063864565258635098846174455575008697843933763691879406875606155063"
        "4123334179378248787743397537830117572506007939963327199766954266386865318209601353364800"
        "7169838717015703206029035754361137400540029565350972651424755153264861899544433936144182"
        "2328965347402790838918640189213341201207729094086163188418077173156475095098764315368294"
        "5256219872099461112009674410453734139402356132642969653877758748034073719682518694887348"
        "7749367944689566288279161742140463543864488821792483235205594475422735052181150581495599"
        "1859289424982274208015057810735538026112474689973115972265720867017273040179393064309829"
        "4541982233098432327963678459088227547549828591653610206819775243161722547684233489087153"
        "3302716396006553369296677389124172502001961134145090941729489805445869338786597553526893"
        "4393912957734376709125920945151009801308272832034209831503848156951585204066807328899969"
        "3309143400659437407077812917684915578516016783948425916618217034288398898344921109755528"
        "8783425914760010606215380349574271287157255302975071340081867787777595156499247882033150"
        "7315446369320825588681490005431927268676973410136494907620019477003690185672082671119660"
        "5627829782533102178794580889180032728807782347232271419218861185595599474961345962372204"
        "8250013327018030188322958468093681787725692935922405865943697742169743977802532032113151"
        "0301476105532700870486610789225085911940150141604893791447421573496472743290823669789542"
        "3071337306839122736957864401164086175610456471211754649202048164440540696086322119316051"
        "6877035564932074554851235010866541167552396211153831786143810257952949344770796894000665"
        "8958585026694125059732842467829949346492300546501519519412480901485057502915316321498199"
        "8863688016745972538516297807223497505541299228985e-2"};
    const HighPrecision cy{
        "6.94892753899652385892994339498967288039114990163797857613653087250435024223067409755982"
        "2837590245062961104774648014599213663284206541721402510345907311242690910726688375576574"
        "7507582511175491384428056496649045278238277643027193351680807001411307991749934224067841"
        "2617656778396051020622908492530402208749311492290271909198418074230262238237652971978506"
        "6512150067211751401360101005740031158014306245986093443087944742041208300249538672715427"
        "3245068460360681669668243290095372926320020473026395559961823385692285621188296608303835"
        "8419971676841145471655669371600197784433510481790831972731505901305399231841765019985162"
        "1270356528164338604148310735700078188214368710905950586495851837624546181555366225869883"
        "8547507994966471966462477270360185758719921690211434167496418084543776927628026367245767"
        "9016472363080824108222582424196757139233622983639018077411671928073620371872591924566420"
        "5030912942696617246187166232214088853079183294898833364946558913903279766606274003112206"
        "9132652365992399719813186532115428991149399519988513083440698686875971837538251442035086"
        "6679702438600441689051326829129363665281912460436597723056816975573345288087144379340770"
        "6632660052483673386054348219102763569527694302963714578675198297415894514951742888478205"
        "3618153319395694668255580329720651137205902746207312293940873275953047610295130270365886"
        "0510355732665370873248832952748805220924585676021946094601600776346728831250463569641203"
        "9762605139990985769134722941258838835820828327695416919400582562172516200084402685006425"
        "4269244815174118295345091444419454432937382961265377464994716745736986226750119177835956"
        "6745198020159263494524839069261640322828125130238139986430274010926710597489907042740969"
        "7975720844451926269007030973058299147337169749763615203323145472675527452655999831951830"
        "3677008689451451173277542808062288264446720965258475988863662245075452283090298367937636"
        "0815586729203138080438930447049788250524661984430089417640926836337094163356466742835674"
        "3885975884600800533959398280436795554703986983806068766915379970744304519563768285675180"
        "8038036380993868774603270736506044119545437528911282872546701073789625002234505735611777"
        "8327181338664656114459493867557888302318510876898986922859748066895752471298555869424836"
        "4940681997439753279703889145261710390965352965782800840941075235102329482711356185292917"
        "8636288946687576789366942627653491949809723025168207202819743200042960117645161729937458"
        "2036516375991743313195329281158820239397561170399018145984608120113767588455074918869438"
        "8157916439591422599831784725808520180492760708569480624408344094875143221544103322858019"
        "7831120809743038041910577868005764722057777911399027510741955040879541857582025211053657"
        "7900648406065266552287289548156583390147714035962788179334477750676411091544250902220427"
        "2739182369027446548635450032739310758385072651636993869748583637253556698087323074395782"
        "3518514179655923908547661562026209643392010761343400044794016231315046286390859118548394"
        "5591642142046506280008867643052763183371131709460004373932259168878004788092162224064891"
        "0533249829956975179181568400023118378029216070018896307963090397807260629866092725530708"
        "6849793938533155846721460852902761277873916771762055508409081494150419332201947944861132"
        "0777438009480303974273329546221342813647708313464242922615497605578298994566192622713055"
        "3501998981159754049661354030453809247121916717588410480436452202811218193160318990076334"
        "4457655188439239263921321327041771162484813038109013039005672060903759340936713671396046"
        "6442476410883497066044265813091938185952729993887849235094808062078518392018259889747795"
        "1244188478978353520377684392380687911936890751774904871095436699558731962432273716301189"
        "0937267183263534520310244600264882416045720524978895322536798991601856093859891333943441"
        "2842455609370271280436814228670358687772250952252287420311938197489452002750203933570221"
        "6365862645776219465310241371577798040456090925641415396130636523787157848529203343483575"
        "4721190426941290456966368040488211679237811060134210374496997823158177450902794826967516"
        "0817525427896344766562135706430678973611589735388754498120102574985753160350746627716904"
        "7809644707510328112870664575743543286774250678422807608801364180300229992936819960574028"
        "0767431419543736002827685107926584684226480473343858667249026205410635820776519252796784"
        "5518207601743329298709617684048481509488492469739813176848453165453045325893919032912918"
        "1932469992270688279233044448466629584208692513751482330360265631524561504358681490028069"
        "4183380109386595619476995277225668533213285359219301665190500411307206896731915121373666"
        "2530858233961962007049253221108328427145173444776456596028470922099311696030755065756656"
        "1443831678773640782177921957477065016250635072672810821693194445515981511417134449261750"
        "8930365361435664480362254716879824787171420631352944368578720765792864155561667985447125"
        "9495025114407561854390857896916894204443732866178499725330150422727397218601255500062592"
        "0316966018428931261697395822278919329024272940518331292201026675994448941601709515424503"
        "8973262678279893300243012179130226392730489338530922216052230398254224714576275190308269"
        "6506145267822571633163467954125016035781020840211124139102288521475249338409686518912882"
        "6190273437559557865991494107327042292443488351409501829607868381003652365479333343763028"
        "6805776986207453209523444623881580978936972155923048944443812323521671467623418992018570"
        "8831177173691764806000183968490589680133468633652877817329897463688719250178966190998598"
        "4935468879182975582249452520596146829714143682126593638857832482588264259595475200226389"
        "3947551502619211636828742682115863421519125303904553578164492316751293403171197433442518"
        "4767546438208225972679422359569718838820485659612837067380575493394563274259845069852765"
        "3520389262490191921531724776971729394142344522610476500568291719498645171316281069814460"
        "9080478124486564649717861764583021657291640581359810609831084025550953928092662762714947"
        "7951423589237252673086076938313071694716803837494761555996034409962807160977800106477567"
        "9347947158214383685368771126331359630917634988919280288200740822002482872230109077121994"
        "5999037195199538220913320162989470740982913027166732230716205253898105476491621619301399"
        "5217350355983907047742918372093578454309393515324953439002788191574855362727023287877826"
        "8880001583226219415195143924192887151832191260262631092847578560373126836915382542801466"
        "7881009240395696322616918909965826728690054969156997731850036205206815587449163168811221"
        "5369580702154355226759344140801090327592899056505680892955127969200900219862890460656059"
        "9632302493949251996706165586109628835729846845638879076337480648314757197850065716914746"
        "1373867539173897304308341096124821202837299677996863029346345270581181308103461778324978"
        "6892498784158985899792945133530106780279249683086913197772866107993332148328231604520444"
        "7064170680802956306187862482126901256560545251107219702625826243047989836003694796976574"
        "4964678230480627855857152277519978653001129395023850202571211249308068647407124173066299"
        "2101757033405591448934770040979445368294153777571416088800783167543285770759265815215700"
        "7231453014186662127744727242043558584926789827116057678339331431880661053240000743906100"
        "1469437273715863197385339382521303548724946214191934709787262235142004151965449706046480"
        "1545430700454133591101311014758472714425972764438549815255713627943662958468874494662264"
        "6122726069898428633409190748677430346537301550710715185488365094406588845177774445550307"
        "8286598107789775479527971831415290452506022916994567625560060511487600194431288347885823"
        "8653611371463207066156449362171060578123603646547031521596018515238287520721455781976411"
        "6424256978792792295739169804939870766901976660952193504454689292544847156103557844665104"
        "8757454758076943330800105470912826896520350913988776456900293729947058591508622422897318"
        "5241761673730128300259096633545788040328024551156596844393215120127982561966175211772661"
        "9710592090326583041501536427678545618059209057852336974610351954244982267851855028340391"
        "2299325532701721615209200998600456718760302111084011908882089465119398864737532221910947"
        "7025020016643667188982657148070715047170141282453689217756604168744734849244187843144456"
        "5191410045324041168765675718630309066364523534977932082235443425967269157882277269768358"
        "2086970468935925201413750817238889675081393110248957986196473501478021723313078559074499"
        "9565208913886548011550145951271365352491880641030215658711494061590253959862182093196859"
        "4454023552543512600715038004392940833122375926197483877014267147952811721997275776979137"
        "1904830422744730794077856981471676806285724010773031646724117520978238755801236110009688"
        "2536612711086934396972636174768374697624867652652752564734254370142488440628047799443903"
        "9766767755665505933208996861001001444191133297228270693881987533063810880978533010047000"
        "5289857335597754111623143834609213823415156315779839877978538512853112026487522352410271"
        "6354446355926848679430461606903784333783023770222618642161163103945935011304179737614378"
        "1758478621566402258050403457876524356304922473048017394858970772440742648312545498868670"
        "7648595496380619885127962994786276250286294068974460066931255036120373548290959559089433"
        "6559252290029235341499702265003176274270090109981007836717442281786270613415797555888221"
        "5774548432273720863313871160842236587770631407859610540937371094743788786042553303084320"
        "2749411070816967372023858614421295261679708367098900506994410882702495991763923791935762"
        "5251211094516172557889672450239253893241440058163102203102863542292515618606368674924458"
        "0465061415860567210429786091917449758591024858390244295528865780590477884293453375535084"
        "6104505981137898125812257710978301508822297151614783997813623157541551174511613335694810"
        "7939887168717167170591425393584797357701247771738436667854681355960603260352615458339351"
        "6177275428186034715639477710091062998276034177565528919710808128114177887033230967888400"
        "1942674877641360010222963234587996206492274751793488881552273834599022910097386017029698"
        "3313755341869394187836122930986461941896630235671057828471926608756663030372836679634803"
        "1458336230208881651566573787391768739009607470946093371747881718704421994756201514959472"
        "2515540505759173872714513089672801727615824319309462341478324304312674194576425185474224"
        "7433639087398166011623947436324192035345973472740153903197006012857750613800663825282123"
        "0659890704550813638568729527154355649005163768398761661436648582219635507242173082017380"
        "8018634867723818076729246467598206375409956597585492723359363615448541548670581806083183"
        "5535901144396775482109702413395510369278600055717787362287545827286228848211097401400269"
        "6324264399668776516195515509238289890305932747552323218627128688033132064395524314098041"
        "5031048671868948666720330804278093663661237720673389886490285315541029082682734697900650"
        "9521242340241409866097685444103821191329993168506424299716576060857866648096914048160922"
        "8438563043850788381526323326946813044959942599185261075590963654436517606095502818816759"
        "2661694178502787159547346161143189219164523268697770353160586553078921809828195045365927"
        "8862819211316773730874759876047577170233236494273525626574104939855282908645989424839387"
        "1758935574746088915317561720715983711034020515224462984166797179002289532325399515281394"
        "2658714156281043930332704675107087257472575156220510235661447339441461646226176122284642"
        "9711747926836048328037252773054064187672104405537675464306406905656551238741892444986638"
        "4535854908886353743704270242344766464902274509321245065324116718085984704519319199016258"
        "9689351151092444075181515263524072766869398783119559390095420321880094955402161186380926"
        "6604706749996157410618921990768154715756121956670700527193559620462394524164023520143371"
        "9820683331297846779144519856707240325211452879123984342009938794119425366779770014395951"
        "4034368660283028326256668652479468590344528441617870478482163083183174218901770178342106"
        "8445007117113202456188625313852886925678222554909583854357885207147987255908706706959856"
        "8380716364345831467229644933848246989329669202973079820278919129233063318416948491913319"
        "0169064890111668231110039341133772251757854450668359993833043179427596061964765932101539"
        "2381274497490674519752286921227620928124741809350164352069713838482348105072798056208288"
        "3162658614307804730244276588264662431199180960150483176146643208550353869174472773078605"
        "2275609168553635443540351093850605654751351422391462867978108845315775943830351405609828"
        "2495397066884836417233392881281306911141851283918518059700280811529547520711189339076710"
        "1497240521308813802449942234268683503816347126280175048087717169954656360767493761818592"
        "0025251495807455250731415180960462957904704579067146905065806922756833314266582030260636"
        "8507902426946873605392990260139008092540422945151843451113960590757623114621649324419930"
        "0493556044162903583458111161027178469538330345053296820306641827205963548923281666898366"
        "6663185908752620082005163049059413594815902372914102404191544820320727478333012712781258"
        "6620536836343067935787854050807143771191880765697189542158553961832754396339514609898885"
        "1818908669607613765753949165335379990428919427134475974487624971916237322936160077938968"
        "9602799274219722190498136398702269657787931899635273648167402735316409034686582674736305"
        "2426434576412429472664105018304977939117174912989616271427792453455718127802441807172999"
        "2736724623953308519506357034143657948298381820935269216475242807224786346219136379403242"
        "8084319112321893001149229223338949249112490161244055019858066630144177909293843299290386"
        "7422183932429050831523216667340864212378280969089204686738794749312940326566073369825315"
        "3167752491347567345675036271325271570050691650716774762491100622341186642405390061273623"
        "2428899330053060830815712782161902948153294723583141713635194521101601800788848864932704"
        "4069164093132255481518251934011602950874359243790609235393796549109139539775658494483795"
        "9935368299132291703494208096234980742715994339441266745129388442319037987969179702070571"
        "0318705715598763704438391402924411408002806315620675503472794486633782829461543003578972"
        "8999285879615197014603406380836942278153299670212429784741123098641972245811501581789413"
        "3349834597894299027647937115833858445201941587436297425198024707086407175676523877524496"
        "7907186944610171423007954073153527270774891835557244067282558951719623461410010869117423"
        "9429010659922868896084413081085840268103151508747391177611237171693812124402860086172284"
        "2124539196494333562940053631687033021362519877162990357363101075241050723099694774073457"
        "4132994263571335310592079022559370941656153564637762422775274169361307819326637249959553"
        "6004713176047204703721983756111870518711210346875380480166575217197212167554241476061423"
        "2696275272133688203342527523427622623286524521283350346316348354215140948702523466961767"
        "7554450923475086973080645776277743553645737196063779205151597987370911421533703812646874"
        "7218975057280006675034771580808172435477998803855301125771082646271731976294327113547027"
        "8885628630880203660552243300850343045007302405765573691364124099895975349363134218837606"
        "4525165164787075683332009753947984436935416698781050370907794201935826809424101111087481"
        "9842306451306487313809562587221088542821007054436929105239511195155338641609962499289410"
        "0159370981972596285217437859240736856516073618084366863053236802075615305132635676866833"
        "6000484030325385415448594814944145156914442649499335313048256239663662076516827061074127"
        "2773837792824358844177801228960271895408312721313429110546536391187105800946059797187499"
        "0409989071719530833152878771518542457831065396212773343883079443956293986533564210884124"
        "6005528776266543223174179071684836817889582159582907376315886564793597884735991806824439"
        "9225073413064569772990100755159877473352259716091625349497678911523886333173112561428854"
        "3797259381043413929383889059516667753889103099726629951112256906531393482938167082425340"
        "1911382393587224394143252060229453338535453545450396460358574282790294538767592805432917"
        "2354805359092667965699899527246100156471074645728546608416058360668165562919696310153195"
        "8974129818864169824441463354647265811237657657023082287626379305069956315043678098763136"
        "7201669145468937046388455625951899769158524097657890691997292041875175679898144807263289"
        "3559755344978291477855061113393139435475927897149087767306368635421673307521477622755794"
        "9269333912614836894807920919654379515540345215430252037418074755726216205058329855589093"
        "8665009089308197164246828001221979504033234176985301909353671249461873052572959799294189"
        "3234020989059247769702071144533657622291061732635287644708426049492543420887107085965765"
        "7375052654754832479217106811438058066464196527567310141727590793075709802705725027990495"
        "8014068369253965991807790428028684507430237147410548889805812480830605411967437642806872"
        "0070685999056984429772047039071240453604984425056822498910448818250263416180829623763637"
        "5762583698899777238946797280677875693790366025590069909244609810529644156902746556092182"
        "7838372558540475846022837223216274692429847598391973862436080760729031983283690492143761"
        "2302168388558843808015553875118015615870339796354122279919980578477937994485718754300326"
        "5858506816872190173672792426751334777130089706139294294135700556239072800423795556194750"
        "0811091925497924001321020141019872010529616082219737107998682457401464151664471130209574"
        "2349664812855366622516497188851269732100769617025973474463317920065940245600296609628452"
        "2688580931427421285576856077765102526487047578798651918875357775482010225435162937773884"
        "7073638475981311632540186543668499668422334724846775668259609446332452587212922328940324"
        "4825118143331154955895268597041230071583732792116746402604742400733442690230337311112805"
        "6639531113528298280808716717233658448485063514763415137704230481603829738352933524351006"
        "1459689639648264247184258600802576781001486230603551677968691788936575429267990640251366"
        "0146239920039060865656634005296989757727050281005457646054988511345678182207082486239264"
        "6111749338103032134969419144273833364548806307975752469405510637753224637474271866176269"
        "3537419681578319889408589192007647001724250933523320983023168523014928765883930690219960"
        "2296928839715197360651601626003945442902411339532429836222549679222138997111718041682540"
        "8525304755430528090127564279711176587711297301607886416259233836716961688250794432462740"
        "4369642836395094810485586506447470128220682245369510880376160102455431185761883355045299"
        "6698994550320149475603247596666298249795106663093215373769578017106279000269835262410732"
        "7742115933119784498425974898930662295320406467993893787951761063953083072405730043871455"
        "4985569496783689174111509147745564159937309863749853101931845506377283479602233571940649"
        "0990294648444785851440973838238178171783849423214774308688649453509054060511557952102087"
        "1612632992279254785530746964514369022842790260025142296691456617406831396158950231423254"
        "2087630172840873744959780456041135473866654815578766759147906427502140945299570479329576"
        "7478104415114253661807083355836378514961270110008054027981853537977648210192779479054969"
        "3232720355494359977399673861155094067151403582947122892249191867975301630945432173239118"
        "8881413883571952151867629691235690082314128958808653020773608694651389472346817520899528"
        "9130501443995825051054919156035627758252901615503808423461294190395210805786531736640592"
        "9061236076606239206753066126969947488158179604081571880582663479867717447457306298272486"
        "7124571128737482316148588627635666581391371493988555068224717070450159133369415004605337"
        "6328901016217888535408192782294211254948789800996583440056457173527754533661636767889082"
        "4591813110091137887622162596994167629879342710750826873726339225772782500670301511917886"
        "0915606705796022254419283345770918542289797363066218008954920478381275035606951119773799"
        "6620139458486707160435391969809939482014977769564052357808745351779679932524274775511356"
        "4274339754389631083162018421803399133923308262019907335588218058854984143604812611906411"
        "8119780485388286999906429043339065603815561649073913955505154891429596296156891433518776"
        "8308713876214876085196987571346306515394090273153157543816861300672811062742692310164771"
        "6640795508231603983155193001753663238282106276144444408801408563883404454877871289192180"
        "2021665661924090737708038280394418566685645222007555061056300009427893711661544923764735"
        "9574337095559077232721525890655321617699218450541741773770556915307669789990523786314497"
        "2013243261231432814180528935575941039291412004419673182440634993817115140786826445424844"
        "3554535879299680200403902693626798264094972108736060542997292488726270245112564201558227"
        "6094935330553119802065253650255547561851314392745637115456845297658769236462921357545867"
        "3836868105176013245523129383108413348102370236602433555509107181959696199220689694597383"
        "7569011516607968012386392007699913956095294366657619135912432349922899847916198797218103"
        "6511024475475373295733084173186425769366636468970864779431827380311315074119560340417589"
        "1611679329561390315288996926967874653851591580223524762664292218700466628441826933838972"
        "8302559957770182106516453258634269489062874603806982834089248596569196193197581868226987"
        "1795506205772506007837099901794991417969169134991534024733059768540072962865574625128444"
        "9790420543212292368331850809118505173835095586891421737455714332750432153197939278811790"
        "9017037317414756607020160029736824739560271616953256512547202791453254135432587175839631"
        "3238840101920910189136376506048822429290701558582138380020659171192221396883718867626649"
        "8991590685248827296000862632036780366709784307351321338478792557095486794679564176819274"
        "2977105065890570368658666372816437351628053411802429178709133256302630805960382755149864"
        "2042324425396009204722565569876680249293243640043013264898118926590344573174585891378238"
        "6512152766252792401709081950348591431498540195194362021417770293757782663243955561126560"
        "3677425808254698182843435606877916783310261271502695100854150481588261415325755088217646"
        "7265465640280220188129252937243487207604556416566045715498015085784347560107383336835950"
        "3613657642094058775616658519398653275609353437188159710296167959843498594977867130919898"
        "0344850869914634385358288520358600875862681776089619333334575955586156286305205283967333"
        "6390340561657426147774568896633131250950836906460385879633864649757290258273047406077793"
        "2095801070650166769650655421423331178558578488014366682968345257629310141897028289832587"
        "5407410842208846510285299278299644269640104254948530098452202017310864252848163888895698"
        "6419606438770987508133786548278168909459663557274468680167107382666005852978732007635567"
        "8587130198730752958820632565876549906605464869317983270553970650050189059595884938326013"
        "9148104453738777237984491733601665151667398756944935829180142945036913824763122077863100"
        "3640510085224306247440752783847068428412173755354003911545787657216863949372696239577649"
        "0433903148464634929289066366884507770723496049948036854999257398204611523594822977547817"
        "4912066605376990536072327935595755712740976454611456446831187704390584344428748711865012"
        "5293909357091137505699762818177388728009908064539297280006660499582654995253575358263510"
        "9226453747414077124344372066324123572475563944093581554260353051848859593996029960050575"
        "4945997341435388574457278171461821839439883104260480415598989772606816191702552310281269"
        "2989188318961950153263715430724113208883555629859853459891642073001118214864999831369302"
        "7429329010333032609258470676431661216563996309618384497275226523206308107245134061369838"
        "0874915283924037899942887500530656589965084221231391418831467444058479457972505893379002"
        "2200258882294519580200323601304522096740336979275058636412275595495633717450710495290978"
        "4917652267370983563368100739395709724692497472685431424673225266129342937315892875261658"
        "7177156575109448823975286840226070238293503744794922769510543518861952728128065366611450"
        "7640577117599180281749839724314523277560182589568632858056294707542423419951947839337734"
        "9892061366785698946619926212873369317632315020909620809460176039160229176943411757259419"
        "0129038355756630500247300858346928703390278062235756662903637283869515417942055234321033"
        "8533658176806835037016488164386902773085610161421463108403960129860689480545508684482702"
        "5953950281183612903748114491417794092684786770345112889761719244105657380188667809049771"
        "9742993671430645819009823300611747077503029016376473385943173877419735260155813042858175"
        "7596623863010726774487007182107686191999500906556073610775369005482803777449406787903396"
        "8927814139481420600096401793059122396307209609465820768629728835782333220947803045806583"
        "1500548967494045369423838158420434746424002134102938375660533737553761485334434651630641"
        "8562990006654993988832302574176360758742471194133720791656728394008698857479607987427993"
        "8971690044428120442505360398979661553941256914117189256705643351499758582237946162127921"
        "3282900644545119748860817216171743295616814548792800834157744500516107536822278317641003"
        "2581597035076114771173338178764696165505857480237017699884471934576960854987167093870110"
        "6173805765822496743014451019171898108064192623017395737167481445523098940362171337234267"
        "8679882968045765915504699467942122013591408927354663023864771340958938973466693532456610"
        "8025599249786059044043880437244420275120871411230112995530086394789066428195380462451949"
        "6913131694395133167163753242174460358376054728371483282644469247949052916577825670231382"
        "5424654341961830399276643511329775364191125333377900707961691314961811984454888050304503"
        "6967244644283855179266189012527869188038107070958912060253915989063486220788170621605859"
        "2138395172966037364899122336523530856641904104705210690789488365568218228687341101295617"
        "0311043125505016359741853149930119365609986551125424368960051204329752070720561023392291"
        "3570841265671296715801544623589961334944266731072437058399485983786247331032694625789400"
        "1252337610789299148402566441156089591954219220294733093581632806140510531258297815235266"
        "9271007700541210933420508444562170687274403402708184648862942078171328486507893694248319"
        "7837752778103450482302998212216725227096488257114327684053921103250594428130362245646918"
        "8240429502292616566613181847476802090552564583043334352444994520151238233442738516944569"
        "3425139958200519383551533333982678167911067065542595976713824523912796301541738965878660"
        "0821445315644262771248202631715946877999086645051507329126514221517229697413363906027599"
        "9012784862869809454744300799293448710394755076229359639555306352667134985308754420445945"
        "8580169213481837070530008134793597516080627880692065111934756285384992275390147042977463"
        "6054170181079128855290018221611343513764773228986971656777793434603074591008776370345176"
        "0252382339926550385667000967781163317720799729048009350429149343048172741434641659055310"
        "7453407115875620394337453674143286142909269302435074820847218624738770533386199931887633"
        "5061667366033335191525275750860207615191194819048281085758950822483864108516910504175946"
        "9010356021302827786033467747204440435751159598544260117956904505410926964040846671798351"
        "1388485181670221183698196432358658641192683188665466236664534312502738158268860515688950"
        "6118091253729537316129164705565326740592912632987524251715031561643312655799135026262192"
        "0751040206501610646702320391720366821355606203119178647016449558142443057175473706663187"
        "5293548005580573398659736041842961403118294840780196913175535267893649590476397739596246"
        "4892810767781458559929185272647961045877916281128363940187034886896213029447241346371717"
        "3633608562169991617517543172477165427233232951853665635212892949163149806345619322996259"
        "6501245050869369268289917357167773237175043561076039385975734944491531973733977625125603"
        "3944174252688852650490496273236581754205103140824644794602382695015551214567943410571976"
        "2072052386766757579644707940811637384279965572404893322627370295972435471035161365876340"
        "5314264467655719070349378692721233040795392551376719847228208432024928260008123943482175"
        "8787386741182938669317278419210041597262186928592972713940999054668810354653848847538943"
        "3271197173078759793333578223593875139786945960423008294187582598395235185978294628026637"
        "6017129177905347601207023989132436459226716293773994189872276535733863695504985034109381"
        "7641680335910780405023782805892258205276486455292271520206884704358917047073058498682145"
        "2059135752123131219244896669552746339093064006227062058955677209378604518465003578140463"
        "7255991479254509231866897789482334453764267481835765659424423330796989930320106532321506"
        "6400440983150232061649903477526480829661472687435634424012930667113520059460032924515251"
        "6846484939564870903199390911407760892265573248757406616228704444373607848110741573572810"
        "8688593342012505289307849098019852923693731459453652636463009834503601057451040021701244"
        "0834693946932847608507797029480025960700978435903995687545956801219065947797401499515387"
        "8467022529162606536348345405602813899319058664548030620850665268789814479967493758587903"
        "6629284064359293266529940098825646306655575725004942006759348021395996739388452124401958"
        "2153202417667531786221025975243423721881111630882828750845484790286630728555380319268096"
        "9718970623664730217052104272572497196241884910698373545118432341451605765345150324350791"
        "5175051047047368203718432826843748121368326807178006955972502571280700050252992344671888"
        "1728199854409982202021237525899871644986055711918607847121899239369627756081630381428653"
        "1582152171476942479737927122843001681318016865877354503980967955209211436835426905597109"
        "5474022236808086101120069255750372591021609377350811431767022293116679534893121941298080"
        "7930337759648468897651127118545957810902644260898121945629778531226684889839972969569949"
        "9592993243099888216714773808197138614842921066843898001568878089629122451792844355733216"
        "4360947614946889970173257737125881254112915396242759571490390298168379284111834117568349"
        "3987034520550245199496521888294169163599490032849359313494415393444282356023250219310096"
        "4301554152552958486080456315504759703951096777889639409837989639234706001206037309169119"
        "9434481460405805597058343089090255310112771344406505243381780656003416415298717661034232"
        "3167399495196253651391162184635640207404094238038891778522860002555968982950027946221295"
        "4108315533019310085196595874518804322750882124689675887570300935788259411590338999535393"
        "5916851610670061574616681801419141509834362612458157882251266116682678770853581769806341"
        "4135222940626330998681293683007220007908881084296231659896472627276017236546816056234614"
        "7906455549501636427147758544774010652556009374096176555758057273711935612747711495711518"
        "0619366729744459301052518736959663481796819431773774458402842161412480266352280471906109"
        "2838393391852467509494680194547682325151978879692926515950401254760292383500098727049588"
        "4341706536976823748206157366242023730155563869935363651818825455779616378482570841447593"
        "648032991162150574527072160465962283783975078831e-1"};
    const PointZoomBBConverter view(
        cx, cy, HighPrecision{"1.485936e30000"}, PointZoomBBConverter::TestMode::Enabled);
    ASSERT_TRUE(fractal.RecenterViewCalc(view));
    fractal.SetNumIterations<uint64_t>(MT9NucleusDiagnostics::IterationLimit);
    const HighPrecision productionCx =
        (fractal.GetPtz().GetMaxX() + fractal.GetPtz().GetMinX()) / HighPrecision{2};
    const HighPrecision productionCy =
        (fractal.GetPtz().GetMaxY() + fractal.GetPtz().GetMinY()) / HighPrecision{2};
    MT9NucleusDiagnostics observer(
        productionCx.backend(), productionCy.backend(), HighPrecision::defaultPrecisionInBits());
    {
        RefOrbitCalc calculator(fractal, UINT64_MAX);
        calculator.SetPerturbationAlg(
            RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        calculator.ResetGuess(productionCx, productionCy);
        ScopedMT9DiagnosticObserver capture(calculator, observer);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 HDRFloat<double>,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        const auto *results = calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>();
        ASSERT_TRUE(results != nullptr);
        ASSERT_EQ(results->GetCountOrbitEntries(), MT9NucleusDiagnostics::IterationLimit);
        const auto stats = calculator.GetLastDynamicPrecisionStats();
        ASSERT_TRUE(stats.has_value());
        ASSERT_TRUE(stats->ReducedSteps > 0);
        ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::None);
        ASSERT_EQ(stats->FailureIteration, uint64_t{0});
    }
    observer.ReplayAndCheck();
}

TEST(RefOrbitDynamicPrecision_FractadynePeriod41722Certified)
{
    // Fractadyne bench-kit 45-mini-p41722-1e3000 (MIT/Apache-2.0), reported against
    // FractalShark 0.546. Keep the complete decimal coordinates here so this regression
    // does not depend on an external checkout or on coordinate truncation in a script.
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(1280, 720, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    constexpr uint64_t IterationLimit = 1'251'660;
    constexpr uint64_t ReturnIteration = 41'722;
    // Input precision exceeds every supplied decimal digit. View setup then chooses
    // the same working precision as the CLI; the test exercises its midpoint guess.
    HighPrecision::defaultPrecisionInBits(16'384);
    const HighPrecision cx{
        "-2.8041054305504546698407770028983979273643258419006230007410381499044388400475119315630"
        "2939402835890872695541844511381853254064367395154734299739738212953023198883985608313668"
        "0750669080499973297978167962228901888145316384153397660013636627589100453562250014991711"
        "0824017207959673925623508173304630619668883563618713037818321302532691505739478167201179"
        "7448226319454382989911082965893494940744741852871198166747075985102104172927711491648659"
        "1916042269947061362608053059787816946740130750978854758372275640244789332857493833804021"
        "0405147738162520148953110435401334906308454160623216196242101020484780494111493638383057"
        "2886238647971817719507402555617790376587844612140455901397860020062089397481330777710296"
        "2505971296846855406323001963985329640578776476051291570733281216621575999548484469305147"
        "7126029705413991799204876111822867979205534626491677029722938635041972553576693097549011"
        "3325597454522711078528097366789176248580372629289271566669238786989136870313220812583898"
        "5869722663307117339446346482967206629048969762559369153975996403417069183894292829441337"
        "5977910222738035446457122389348999587261514494293460996125742722918124650832424834787744"
        "0280933612465938306007486721724415482974924321881538466480189122518420570235408377978290"
        "9414057426034939270564980678370092252697860648484520015152986691572885766701692664800446"
        "4364546414359257040337386235881985551773827967953607750765158828324378434621882530452802"
        "0338199751734561067008089571663465421384308911619909790168115822302011148017773087136462"
        "1309118250825786358943566216760634821034120509875450099509776276225262830610474288403884"
        "8450659589557147588860192157140906787729813842168330962972947885556667146728476526099424"
        "7433625382524695087770315376923812168024616019230774099552496187501802208838024062514035"
        "7610497338098733912449386571401203289022810583477231233669676418409884811443237106676128"
        "8573217469329362563702024586230251426817820971300695471183030734900522409574123592860084"
        "8171892859567778975971904374177180154511049992218032930436417158524746137393980947814162"
        "0487979090570939605559181973526211965966336854464446310822477688278348263982719395433311"
        "7653368472097133946445449059854019646122542921965235330029875015925207044611006204412235"
        "7938500930183044648297768517916566047023998292986644597492800933923833747134890300994731"
        "9885999444224050202096112998110801659670530232841854542903227659412036093468417164455200"
        "2670419860859250517374162529723913199087995159942044051799944844874991670905487891111911"
        "6714016463474843486380376154013851955998464317062726952835157770226589479763040310739330"
        "3285392984111687784624502065046262498865947666438584077294140113281806642357220994506716"
        "1398955904488918990985751555740568711002258861137568583113407378968443037561349587899005"
        "4001597881212763825977791999183639623096714867187274111232620651514399957839260626369909"
        "9744648518007286363188553806428932842658389077582515115929766202819247151493225553615329"
        "0145809326715112164385964019296548623018389074431247417282075906543971117008157924279622"
        "359851504553088767589180547411954741526347887939633207e-2"};
    const HighPrecision cy{
        "6.94892753899652385892994339498967288039114990163797857613653087250435024223067409755982"
        "2837590245062961104774648014599213663284206541721402510345907311242690910726688375576574"
        "7507582511175491384428056496649045278238277643027193351680807001411307991749934224067841"
        "2617656778396051020622908492530402208749311492290271909198418074230262238237652971978506"
        "6512150067211751401360101005740031158014306245986093443087944742041208300249538672715427"
        "3245068460360681669668243290095372926320020473026395559961823385692285621188296608303835"
        "8419971676841145471655669371600197784433510481790831972731505901305399231841765019985162"
        "1270356528164338604148310735700078188214368710905950586495851837624546181555366225869883"
        "8547507994966471966462477270360185758719921690211434167496418084543776927628026367245767"
        "9016472363080824108222582424196757139233622983639018077411671928073620371872591924566420"
        "5030912942696617246187166232214088853079183294898833364946558913903279766606274003112206"
        "9132652365992399719813186532115428991149399519988513083440698686875971837538251442035086"
        "6679702438600441689051326829129363665281912460436597723056816975573345288087144379340770"
        "6632660052483673386054348219102763569527694302963714578675198297415894514951742888478205"
        "3618153319395694668255580329720651137205902746207312293940873275953047610295130270365886"
        "0510355732665370873248832952748805220924585676021946094601600776346728831250463569641203"
        "9762605139990985769134722941258838835820828327695416919400582562172516200084402685006425"
        "4269246307815833006058235691268067389492429911452225462534640328269260165087121359468368"
        "1021888231521683793275495456261408044365237968674568379937351306864822476388022681591737"
        "1003659302613851357452673795057907973741231608888685745299365987902505809912922190577241"
        "1006010937685701581110620099854264549206368604155319125130177782476489385536655767883545"
        "4189198395096834538531256282047630280862614209577627581445158658066834501627939863260855"
        "7594926974870131843368804917754032182868003196662582403343874535622428984910926552792684"
        "3975448497934819888782709298126766086765883218124411451934560405104050663625081996297414"
        "0133205961292883629027758818682889190818215017468201574552783979913061144942370825494753"
        "3907326848872987102268351498325148935722933984946607735451159955403905043105635353310275"
        "9827830066036684708656509701477344930361844514055568288136084402371190865294510191057713"
        "7253938297763433711042615684014633981418559256957835398478449831047183800088294642633320"
        "4654709490983223495138904971941364621482590420094604145098020960850698650260030523967057"
        "6240321334604361503681182864322131283311083993210663611670982732932501158796364383902755"
        "0424645906154779019128994486070547638010104759833830403620846509332930181772262204072899"
        "8596458930844227543702510736142954678838724980237279016498435188794821740884308465083704"
        "3053267505131955516452962651309661003921280607749702834812653764634985062376255679305212"
        "0154030043620242886736617983224935161632107970936968121319410495289365270183658096337981"
        "316269411224268276478115180500962154730163674629593165e-1"};
    const HighPrecision zoom{"1.030386e3000"};
    const PointZoomBBConverter view(cx, cy, zoom, PointZoomBBConverter::TestMode::Enabled);
    ASSERT_TRUE(fractal.RecenterViewCalc(view));
    fractal.SetNumIterations<uint64_t>(IterationLimit);
    // Capture the same finite midpoint at production precision, including occupied
    // guard limbs. Re-evaluating this expression at oracle precision would change c.
    const HighPrecision productionCx =
        (fractal.GetPtz().GetMaxX() + fractal.GetPtz().GetMinX()) / HighPrecision(2);
    const HighPrecision productionCy =
        (fractal.GetPtz().GetMaxY() + fractal.GetPtz().GetMinY()) / HighPrecision(2);
    std::vector<std::array<HDRFloat<double>, 2>> oracleSamples;
    oracleSamples.push_back({HDRFloat<double>{0.0}, HDRFloat<double>{0.0}});
    for (const mp_bitcnt_t oracleBits : {mp_bitcnt_t{11'264}, mp_bitcnt_t{12'288}}) {
        MpfValue oracleCx(oracleBits), oracleCy(oracleBits), x(oracleBits), y(oracleBits),
            xx(oracleBits), yy(oracleBits), xy(oracleBits);
        mpf_set(oracleCx.Get(), productionCx.backend());
        mpf_set(oracleCy.Get(), productionCy.backend());
        mpf_set(x.Get(), oracleCx.Get());
        mpf_set(y.Get(), oracleCy.Get());
        const auto convert = [](mpf_srcptr value) {
            double mantissa;
            const auto exponent = mpf_get_2exp_d(&mantissa, value);
            HDRFloat<double> sample{static_cast<int32_t>(exponent), mantissa};
            HdrReduce(sample);
            return sample;
        };
        for (uint64_t index = 1; index <= ReturnIteration + 1; ++index) {
            const std::array<HDRFloat<double>, 2> sample{convert(x.Get()), convert(y.Get())};
            if (oracleBits == 11'264) {
                oracleSamples.push_back(sample);
            } else {
                for (size_t component = 0; component < sample.size(); ++component) {
                    ASSERT_EQ(sample[component].getExp(), oracleSamples[index][component].getExp());
                    ASSERT_EQ(sample[component].getMantissa(),
                              oracleSamples[index][component].getMantissa());
                }
            }
            // Independent wide MPIR arithmetic uses no MT9 products or precision schedule.
            mpf_mul(xx.Get(), x.Get(), x.Get());
            mpf_mul(yy.Get(), y.Get(), y.Get());
            mpf_mul(xy.Get(), x.Get(), y.Get());
            mpf_mul_ui(xy.Get(), xy.Get(), 2);
            mpf_add(y.Get(), xy.Get(), oracleCy.Get());
            mpf_sub(x.Get(), xx.Get(), yy.Get());
            mpf_add(x.Get(), x.Get(), oracleCx.Get());
        }
    }
    const auto check = [&]<class SubType>() {
        RefOrbitCalc calculator(fractal, UINT64_MAX);
        calculator.SetPerturbationAlg(
            RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
        std::string diagnostic;
        {
            ScopedConsoleLogCapture capture;
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     HDRFloat<SubType>,
                                                     SubType,
                                                     PerturbExtras::Disable,
                                                     RefOrbitCalc::BenchmarkMode::Disable>(
                fractal.GetPtz());
            diagnostic = capture.GetMessage();
        }
        CheckDynamicPrecisionCounterDelta(countersBefore, 1, 0);
        const auto stats = calculator.GetLastDynamicPrecisionStats();
        ASSERT_TRUE(stats.has_value());
        ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::None);
        ASSERT_EQ(stats->FailureIteration, uint64_t{0});
        ASSERT_EQ(stats->FullBits, uint64_t{10'112});
        ASSERT_EQ(stats->MinimumBits, uint64_t{5'248});
        ASSERT_TRUE(stats->Steps >= ReturnIteration);
        ASSERT_TRUE(stats->ReducedSteps > 0);
        ASSERT_TRUE(diagnostic.find("MT9 dynamic precision warning:") == std::string::npos);
        ASSERT_TRUE(diagnostic.find("worst_real_error_state_Z=41722") != std::string::npos);
        CheckCertificationReport(diagnostic, *stats);
        // Check the reported headroom without locking the test to rounded display values.
        const std::string_view report{diagnostic};
        const std::string_view field{"worst_real_error_bits="};
        const auto begin = report.find(field);
        ASSERT_TRUE(begin != std::string_view::npos);
        double used{}, maximum{};
        const auto usedEnd =
            std::from_chars(report.data() + begin + field.size(), report.data() + report.size(), used);
        ASSERT_TRUE(usedEnd.ec == std::errc{});
        ASSERT_EQ(*usedEnd.ptr, '/');
        const auto maximumEnd = std::from_chars(usedEnd.ptr + 1, report.data() + report.size(), maximum);
        ASSERT_TRUE(maximumEnd.ec == std::errc{});
        ASSERT_TRUE(maximum > used);
        const auto *published =
            calculator.GetLast<uint64_t, HDRFloat<SubType>, PerturbExtras::Disable>();
        ASSERT_TRUE(published != nullptr);
        ASSERT_EQ(published->GetPeriodMaybeZero(), ReturnIteration + 1);
        ASSERT_TRUE(published->GetCountOrbitEntries() > ReturnIteration);
        ASSERT_TRUE(published->GetCountOrbitEntries() <= oracleSamples.size());
        ASSERT_EQ(stats->GenerationNumber, published->GetGenerationNumber());
        ASSERT_EQ(stats->PublishedGenerationNumber, published->GetGenerationNumber());
        for (size_t index = 0; index < published->GetCountOrbitEntries(); ++index) {
            const auto &sample = published->GetOrbitData()[index];
            const std::array<HDRFloat<SubType>, 2> actual{sample.x, sample.y};
            for (size_t component = 0; component < actual.size(); ++component) {
                HDRFloat<SubType> expected{oracleSamples[index][component]};
                HdrReduce(expected);
                auto reduced = actual[component];
                HdrReduce(reduced);
                ASSERT_EQ(reduced.getMantissa(), expected.getMantissa());
                // Zero has multiple valid exponent encodings across HDR sample types.
                if (expected.getMantissa() != SubType{0}) {
                    ASSERT_EQ(reduced.getExp(), expected.getExp());
                }
            }
        }
        calculator.GetUsefulPerturbationResults<uint64_t,
                                                HDRFloat<SubType>,
                                                SubType,
                                                PerturbExtras::Disable,
                                                RefOrbitCalc::Extras::None,
                                                HDRFloat<SubType>>();
        RefOrbitDetails details;
        calculator.GetSomeDetails(details);
        ASSERT_TRUE(details.PerturbationAlg.find("fixed-MT9 equivalence uncertain") ==
                    std::string::npos);
        ASSERT_EQ(details.OrbitMilliseconds, stats->ElapsedMilliseconds);
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

TEST(RefOrbitDynamicPrecision_WarningThenCancellation)
{
    // Cancel from the warning sink to exercise ordinary cancellation after a failed
    // certificate. The synchronous, drained fixture keeps the sink local to this call.
    class CancelOnWarning final : public std::stringbuf {
    public:
        explicit CancelOnWarning(AbortMonitor &abort) : m_Abort(abort) {}

    protected:
        std::streamsize
        xsputn(const char *text, std::streamsize size) override
        {
            const auto written = std::stringbuf::xsputn(text, size);
            if (std::string_view{text, static_cast<size_t>(size)}.find(
                    "MT9 dynamic precision warning:") != std::string_view::npos) {
                m_Abort.SetStopCalculating(true);
            }
            return written;
        }

    private:
        AbortMonitor &m_Abort;
    };
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    constexpr uint64_t Iterations = 2 * AbortMonitor::AbortCheckInterval;
    fractal.SetNumIterations<uint64_t>(Iterations);
    HighPrecision::defaultPrecisionInBits(1024);
    HighPrecision cx{0}, cy{1};
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    // A previous result remains cached while cancellation publishes the normal prefix.
    calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
    calculator.AddPerturbationReferencePoint<uint64_t,
                                             double,
                                             double,
                                             PerturbExtras::Disable,
                                             RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    const auto *previous = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    calculator.ResetGuess(cx, cy);
    const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
    AbortMonitor abort(false);
    CancelOnWarning capture(abort);
    {
        ScopedConsoleLogCapture redirect(capture);
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
    }
    ASSERT_TRUE(AbortMonitor::GetStopCalculatingGlobal());
    abort.ResetStopCalculating();
    const auto stats = calculator.GetLastDynamicPrecisionStats();
    ASSERT_TRUE(stats.has_value());
    ASSERT_TRUE(stats->Failure != FractalShark::RefOrbitPrecisionFailure::None);
    ASSERT_TRUE(stats->FailureIteration > 1);
    const auto *prefix = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
    ASSERT_TRUE(prefix != previous);
    ASSERT_EQ(stats->PublishedGenerationNumber, prefix->GetGenerationNumber());
    ASSERT_TRUE(prefix->GetCountOrbitEntries() > 1);
    ASSERT_TRUE(prefix->GetCountOrbitEntries() < Iterations);
    CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
}

TEST(RefOrbitDynamicPrecision_ConvertedWarningReport)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    constexpr mp_bitcnt_t Bits = 1'024;
    HighPrecision::defaultPrecisionInBits(Bits);
    HighPrecision cx{0}, cy{1};
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(Bits).Recentered(cx, cy)));
    HighPrecision::defaultPrecisionInBits(Bits);
    fractal.SetNumIterations<uint64_t>(256);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    calculator.SetPerturbationAlg(
        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    calculator.ResetGuess(cx, cy);
    const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
    ScopedConsoleLogCapture capture;
    const auto *converted =
        calculator.GetAndCreateUsefulPerturbationResults<uint64_t,
                                                         HDRFloat<double>,
                                                         double,
                                                         PerturbExtras::Disable,
                                                         RefOrbitCalc::Extras::None,
                                                         HDRFloat<CudaDblflt<MattDblflt>>>(
            fractal.GetPtz());
    ASSERT_TRUE(converted != nullptr);
    const auto stats = calculator.GetLastDynamicPrecisionStats();
    ASSERT_TRUE(stats.has_value());
    ASSERT_EQ(stats->PublishedGenerationNumber, converted->GetGenerationNumber());
    RefOrbitDetails details;
    calculator.GetSomeDetails(details);
    ASSERT_TRUE(details.PerturbationAlg.find("fixed-MT9 equivalence uncertain at Z") !=
                std::string::npos);
    ASSERT_EQ(details.OrbitMilliseconds, stats->ElapsedMilliseconds);
    const auto *cached =
        calculator.GetAndCreateUsefulPerturbationResults<uint64_t,
                                                         HDRFloat<double>,
                                                         double,
                                                         PerturbExtras::Disable,
                                                         RefOrbitCalc::Extras::None,
                                                         HDRFloat<CudaDblflt<MattDblflt>>>(
            fractal.GetPtz());
    ASSERT_TRUE(cached == converted);
    CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
}

TEST(RefOrbitDynamicPrecision_ManualSelectionsInvalidateCachedReference)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
    fractal.SetNumIterations<uint64_t>(4);
    RefOrbitCalc calculator(fractal, UINT64_MAX);
    // An explicit fixed choice after Auto must discard the automatically generated
    // orbit, just as switching between fixed and dynamic choices discards it.
    for (const auto algorithm : {RefOrbitCalc::PerturbationAlg::Auto,
                                 RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba,
                                 RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision,
                                 RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba,
                                 RefOrbitCalc::PerturbationAlg::Auto}) {
        calculator.SetPerturbationAlg(algorithm);
        ASSERT_TRUE((calculator.GetUsefulPerturbationResults<uint64_t,
                                                             double,
                                                             double,
                                                             PerturbExtras::Disable,
                                                             RefOrbitCalc::Extras::None,
                                                             double>() == nullptr));
        ASSERT_FALSE(calculator.GetLastDynamicPrecisionStats().has_value());
        calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
        calculator.AddPerturbationReferencePoint<uint64_t,
                                                 double,
                                                 double,
                                                 PerturbExtras::Disable,
                                                 RefOrbitCalc::BenchmarkMode::Enable>(fractal.GetPtz());
        const auto *published = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
        ASSERT_TRUE(published->GetCountOrbitEntries() > 1);
        ASSERT_EQ(calculator.GetLastDynamicPrecisionStats().has_value(),
                  calculator.GetPerturbationAlg() ==
                      RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    }
}
