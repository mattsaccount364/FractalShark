#include "TestFramework.h"

#include "AbortMonitor.h"
#include "AlgCmds.h"
#include "ConsoleLog.h"
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
                                         "final_state_certifies=",
                                         "limiting_component=",
                                         "limiting_boundary="}) {
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
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(test.PrecisionBits, true, renderer) ==
                    test.Expected);
        const auto withoutGpu =
            test.Expected == RefOrbitCalc::PerturbationAlg::GPU
                ? RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision
                : test.Expected;
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(test.PrecisionBits, false, renderer) ==
                    withoutGpu);
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
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'768, true, renderer) == expected);
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'767, true, renderer) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(32'768, false, renderer) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    }
}

TEST(RefOrbitAuto_GpuPrecisionCapacity)
{
    const auto renderer = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2);
    // GPU buckets store 32-bit limbs. Test the capacity without allocating a viewport at that depth.
    constexpr uint64_t maximumGpuBits = static_cast<uint64_t>(MaxSupportedLimbCount) * 32;
    for (const auto bits : {maximumGpuBits - 1, maximumGpuBits}) {
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(bits, true, renderer) ==
                    RefOrbitCalc::PerturbationAlg::GPU);
    }
    for (const auto bits : {maximumGpuBits + 1, std::numeric_limits<uint64_t>::max()}) {
        ASSERT_TRUE(RefOrbitCalc::ResolveAutoPerturbationAlg(bits, true, renderer) ==
                    RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
    }
}

TEST(RefOrbitKaratsuba_AutoSelectionAndExplicitOverride)
{
    MPIRPrecision restore(HighPrecision::defaultPrecisionInBits());
    Fractal fractal(8, 8, nullptr, false, UINT64_MAX, true, GpuMode::Disabled);
    fractal.GetRenderPool()->Drain();
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
            ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                        RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
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
            FractalShark::RefOrbitPrecisionSchedule::SumError(x.Get(), y.Get(), actual.Get());
        mpf_add(exact.Get(), x.Get(), y.Get());
        mpf_add(actual.Get(), x.Get(), y.Get());
        checkError(sumBound, actual.Get());
        mpf_sub(exact.Get(), x.Get(), y.Get());
        mpf_sub(actual.Get(), x.Get(), y.Get());
        checkError(sumBound, actual.Get());
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
        FractalShark::RefOrbitPrecisionSchedule schedule(128, cx.Get(), cy.Get(), cx.Get(), cy.Get());
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
        if (mpf_sgn(scratch.Get()) != 0) {
            ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
                schedule.GetError(), FractalShark::RefOrbitErrorBound::FromMpfUpper(scratch.Get())));
        }
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
        FractalShark::RefOrbitPrecisionSchedule schedule(
            128, zero.Get(), zero.Get(), zero.Get(), zero.Get());
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
        ASSERT_TRUE(schedule.GetError().IsFinite());
        ASSERT_FALSE(FractalShark::RefOrbitErrorBound::Less(
            schedule.GetError(), FractalShark::RefOrbitErrorBound::FromMpfUpper(expected.Get())));
    }
    FractalShark::RefOrbitPrecisionSchedule invalid(128, zero.Get(), zero.Get(), zero.Get(), zero.Get());
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
    FractalShark::RefOrbitPrecisionSchedule overflow(
        128, zero.Get(), zero.Get(), zero.Get(), zero.Get());
    constexpr int64_t ExponentLimit = std::numeric_limits<int64_t>::max() / 8;
    overflow.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(ExponentLimit - 1), {});
    ASSERT_TRUE(overflow.GetError().IsFinite());
    overflow.BeginStep(zero.Get(), zero.Get(), false);
    overflow.FinishStep({}, {});
    ASSERT_FALSE(overflow.GetError().IsFinite());
    // A nonzero intermediate below the supported range must survive addition to
    // zero and reach the final range check, rather than being aligned out of sight.
    FractalShark::RefOrbitPrecisionSchedule underflow(
        128, zero.Get(), zero.Get(), zero.Get(), zero.Get());
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
        FractalShark::RefOrbitPrecisionSchedule good(1024, x.Get(), y.Get(), x.Get(), y.Get());
        good.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-960), {});
        ASSERT_TRUE(good.Certify(x.Get(), y.Get(), true));
        FractalShark::RefOrbitPrecisionSchedule ambiguous(1024, x.Get(), y.Get(), x.Get(), y.Get());
        ambiguous.AddStateTruncation(FractalShark::RefOrbitErrorBound::PowerOfTwo(-899), {});
        ASSERT_FALSE(ambiguous.Certify(x.Get(), y.Get(), true));
        ASSERT_TRUE(ambiguous.GetStats().Failure ==
                    FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
    }
    mpf_set_ui(x.Get(), 1);
    mpf_div_2exp(x.Get(), x.Get(), 2000);
    FractalShark::RefOrbitPrecisionSchedule nativeUnderflow(1024, x.Get(), y.Get(), x.Get(), y.Get());
    ASSERT_FALSE(nativeUnderflow.Certify(x.Get(), y.Get(), true));
    ASSERT_TRUE(nativeUnderflow.GetStats().Failure ==
                FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion);
    FractalShark::RefOrbitPrecisionSchedule hdr(1024, x.Get(), y.Get(), x.Get(), y.Get());
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
                FractalShark::RefOrbitPrecisionSchedule schedule(
                    Bits, value.Get(), zero.Get(), value.Get(), zero.Get());
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
    FractalShark::RefOrbitPrecisionSchedule mixed(Bits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule fresh(Bits, x.Get(), y.Get(), x.Get(), y.Get());
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
                FractalShark::RefOrbitPrecisionSchedule threshold(
                    Bits, x.Get(), zero.Get(), x.Get(), zero.Get());
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
    FractalShark::RefOrbitPrecisionSchedule exact(Bits, x.Get(), zero.Get(), x.Get(), zero.Get());
    ASSERT_TRUE(exact.Certify(x.Get(), zero.Get(), false));
    ASSERT_EQ(exact.GetStats().FastCertifiedComponents, expectedFast);
    FractalShark::RefOrbitPrecisionSchedule rejected(Bits, x.Get(), y.Get(), x.Get(), y.Get());
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
            FractalShark::RefOrbitPrecisionSchedule schedule(
                Bits, value.Get(), zero.Get(), value.Get(), zero.Get());
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
            FractalShark::RefOrbitPrecisionSchedule safe(bits, x.Get(), y.Get(), x.Get(), y.Get());
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

            FractalShark::RefOrbitPrecisionSchedule rejected(bits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule exact(bits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule unsupported(bits, x.Get(), y.Get(), x.Get(), y.Get());
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
        FractalShark::RefOrbitPrecisionSchedule schedule(bits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule schedule(bits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule regular(bits, x.Get(), zero.Get(), x.Get(), zero.Get());
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

TEST(RefOrbitDynamicPrecision_RejectionLogsAndThrows)
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
                // Copying at 64 bits loses a nonzero tail onto the exact boundary 1,
                // so certification must reject Z1 before publishing any samples.
                mpf_add_ui(cx.backend(), cx.backend(), 1);
            }
            if (cachedOrbit) {
                // Exercise exact signed-coordinate diagnostics as well as positive inputs.
                mpf_neg(cx.backend(), cx.backend());
            }
            HighPrecision::defaultPrecisionInBits(64);
            RefOrbitCalc calculator(fractal, UINT64_MAX);
            calculator.SetPerturbationAlg(
                RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
            const PerturbationResults<uint64_t, double, PerturbExtras::Disable> *previous = nullptr;
            if (cachedOrbit) {
                // Switching strategies clears the cache by design. Seed a valid
                // dynamic orbit so rejection alone is responsible for preserving it.
                calculator.ResetGuess(HighPrecision{0}, HighPrecision{0});
                calculator.AddPerturbationReferencePoint<uint64_t,
                                                         double,
                                                         double,
                                                         PerturbExtras::Disable,
                                                         RefOrbitCalc::BenchmarkMode::Enable>(
                    fractal.GetPtz());
                previous = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
            }
            calculator.ResetGuess(cx, cy);
            const auto countersBefore = RefOrbitCalc::GetDynamicPrecisionCounters();
            std::string exceptionMessage;
            std::string diagnostic;
            {
                ScopedConsoleLogCapture capture;
                try {
                    calculator.AddPerturbationReferencePoint<uint64_t,
                                                             double,
                                                             double,
                                                             PerturbExtras::Disable,
                                                             RefOrbitCalc::BenchmarkMode::Enable>(
                        fractal.GetPtz());
                } catch (const FractalSharkSeriousException &exception) {
                    exceptionMessage = exception.what();
                }
                diagnostic = capture.GetMessage();
            }
            ASSERT_FALSE(exceptionMessage.empty());
            ASSERT_TRUE(exceptionMessage.size() < FractalSharkSeriousException::MsgBufSize);
            ASSERT_TRUE(exceptionMessage.find(", Z1, steps=0/16, bits=64/64") != std::string::npos);
            const char *reason = unsupportedNative ? "UnsupportedConversion" : "AmbiguousSample";
            ASSERT_TRUE(exceptionMessage.find(reason) != std::string::npos);
            ASSERT_TRUE(diagnostic.find(exceptionMessage) != std::string::npos);
            ASSERT_TRUE(diagnostic.find("RefOrbitCalc.cpp:") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("min_bits=64") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("reduced_steps=0") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("elapsed_ms=") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("sample_type=") != std::string::npos);
            ASSERT_TRUE(diagnostic.find(unsupportedNative
                                            ? "worst_real_error_bits=unsupported_conversion"
                                            : "worst_real_error_bits=zero_gap") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("worst_real_error_state_Z=1") != std::string::npos);
            ASSERT_TRUE(diagnostic.find("worst_imaginary_error_bits=not_checked") != std::string::npos);
            CheckLoggedCoordinate(diagnostic, "cx", cx.backend());
            CheckLoggedCoordinate(diagnostic, "cy", cy.backend());
            CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
            const auto stats = calculator.GetLastDynamicPrecisionStats();
            ASSERT_TRUE(stats.has_value());
            CheckCertificationReport(diagnostic, *stats);
            ASSERT_EQ(stats->FastCertifiedComponents, uint64_t{0});
            ASSERT_EQ(stats->RegularCertificationComponents, uint64_t{1});
            ASSERT_TRUE(stats->Failure ==
                        (unsupportedNative
                             ? FractalShark::RefOrbitPrecisionFailure::UnsupportedConversion
                             : FractalShark::RefOrbitPrecisionFailure::AmbiguousSample));
            ASSERT_EQ(stats->Steps, uint64_t{0});
            ASSERT_EQ(HighPrecision::defaultPrecisionInBits(), uint64_t{64});
            if (cachedOrbit) {
                ASSERT_TRUE(
                    (calculator.GetLast<uint64_t, double, PerturbExtras::Disable>() == previous));
                ASSERT_TRUE(stats->GenerationNumber != previous->GetGenerationNumber());
                ASSERT_EQ(previous->GetCountOrbitEntries(), uint64_t{16});
                ASSERT_EQ(previous->GetOrbitData()[1].x, 0.0);
            } else {
                // GetLast requires a nonempty cache. The normal lookup safely handles
                // an empty one and must not find an unpublished rejected attempt.
                ASSERT_TRUE((calculator.GetUsefulPerturbationResults<uint64_t,
                                                                     double,
                                                                     double,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None,
                                                                     double>() == nullptr));
            }

            // A caller may explicitly select fixed precision after catching the error.
            // This proves the lock, worker resources, and TLS allocator were unwound.
            calculator.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
            calculator.ResetGuess(cx, cy);
            calculator.AddPerturbationReferencePoint<uint64_t,
                                                     double,
                                                     double,
                                                     PerturbExtras::Disable,
                                                     RefOrbitCalc::BenchmarkMode::Enable>(
                fractal.GetPtz());
            CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
            const auto *fixed = calculator.GetLast<uint64_t, double, PerturbExtras::Disable>();
            ASSERT_TRUE(fixed->GetCountOrbitEntries() > 1);
            ASSERT_EQ(fixed->GetOrbitData()[1].x, unsupportedNative ? 0.0 : (cachedOrbit ? -1.0 : 1.0));
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
    // through the menu selects dynamic MT9 at the CPU threshold when CUDA is disabled.
    ASSERT_TRUE(fractal.RecenterViewCalc(ViewAtPrecision(8'192)));
    ASSERT_TRUE(fractal.GetPerturbationAlg() == RefOrbitCalc::PerturbationAlg::MTPeriodicity9Karatsuba);
    commands.ExecuteCommand(FractalShark::FractalCommand::PerturbationAuto);
    fractal.GetRenderPool()->Drain();
    ASSERT_TRUE(fractal.GetPerturbationAlg() ==
                RefOrbitCalc::PerturbationAlg::MTPeriodicity9KaratsubaDynamicPrecision);
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
        FractalShark::RefOrbitPrecisionSchedule schedule(
            bits, values[0]->Get(), values[1]->Get(), values[0]->Get(), values[1]->Get());
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
        FractalShark::RefOrbitPrecisionSchedule schedule(fullBits, x.Get(), y.Get(), x.Get(), y.Get());
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
        FractalShark::RefOrbitPrecisionSchedule schedule(fullBits, x.Get(), y.Get(), x.Get(), y.Get());
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
    FractalShark::RefOrbitPrecisionSchedule reset(fullBits, x.Get(), y.Get(), x.Get(), y.Get());
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
    constexpr mp_bitcnt_t bits = 16'384;
    constexpr mp_bitcnt_t CoordinateGuardBits = 124;
    constexpr uint64_t iterations = 64;
    HighPrecision::defaultPrecisionInBits(bits);
    HighPrecision cx{"0.707"}, cy{"0.95"};
    mpf_div_2exp(cx.backend(), cx.backend(), bits - CoordinateGuardBits);
    mpf_div_2exp(cy.backend(), cy.backend(), bits - CoordinateGuardBits);
    mpf_add_ui(cy.backend(), cy.backend(), 1);
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
        ASSERT_THROWS((calculator.AddPerturbationReferencePoint<uint64_t,
                                                                HDRFloat<double>,
                                                                double,
                                                                PerturbExtras::Disable,
                                                                RefOrbitCalc::BenchmarkMode::Enable>(
                          fractal.GetPtz())),
                      FractalSharkSeriousException);
        diagnostic = capture.GetMessage();
    }
    CheckDynamicPrecisionCounterDelta(countersBefore, 0, 1);
    const auto stats = calculator.GetLastDynamicPrecisionStats();
    ASSERT_TRUE(stats.has_value());
    ASSERT_TRUE(stats->Failure == FractalShark::RefOrbitPrecisionFailure::AmbiguousSample);
    ASSERT_EQ(stats->Steps, uint64_t{34});
    // Steps count completed updates from Z1; rejection after 34 updates is Z35.
    ASSERT_TRUE(diagnostic.find(", Z35, steps=34/64, bits=") != std::string::npos);
    CheckLoggedCoordinate(diagnostic, "cx", cx.backend());
    CheckLoggedCoordinate(diagnostic, "cy", cy.backend());
    ASSERT_TRUE((calculator.GetLast<uint64_t, HDRFloat<double>, PerturbExtras::Disable>() == previous));
    ASSERT_EQ(previous->GetGenerationNumber(), previousGeneration);
    ASSERT_TRUE(stats->GenerationNumber != previousGeneration);
    ASSERT_EQ(previous->GetCountOrbitEntries(), iterations);
    // Exercise the dynamic specialization again after a worker-backed rejection.
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
                fullBits, cx.Get(), cy.Get(), dynamic.m_X.Get(), dynamic.m_Y.Get());
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
                // acceptance must still match both independent-width oracles.
                ASSERT_TRUE(!certified || dynamicSampleCorrect);
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
