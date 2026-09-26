//
// This feature finder logic is heavily based on the implementation in Imagina
// but likely screws up some of the details.
//

#include "stdafx.h"

#include "AbortMonitor.h"
#include "ConsoleLog.h"
#include "Exceptions.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "FloatComplex.h"
#include "HighPrecision.h"
#include "LAInfoDeep.h"
#include "LAReference.h"
#include "MpirOrbitEval.h"
#include "OrbitEndpointEvaluator.h"
#include "PerturbationResults.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>

// ------------------------------------------------------------
// Complex step computation in COORD precision:
//
// step = z / dzdc  computed as  (z * conj(dzdc)) / (|dzdc|^2)
//
// dzdc is provided in deriv precision, so we promote to coord once.
// ------------------------------------------------------------
template <typename IterType, typename T>
static inline bool
ComputeNewtonStep_mpf_coord_from_deriv(mpf_complex &step_coord,       // coord_prec (out)
                                       const mpf_complex &z_coord,    // coord_prec
                                       const mpf_complex &dzdc_deriv, // deriv_prec
                                       mpf_complex &dzdc_coord, // coord_prec scratch (promoted dzdc)
                                       mpf_t denom_c,
                                       mpf_t tr_c,
                                       mpf_t ti_c,
                                       mpf_t t1_c,
                                       mpf_t t2_c)
{
    // promote dzdc to coord precision
    mpf_set(dzdc_coord.re, dzdc_deriv.re);
    mpf_set(dzdc_coord.im, dzdc_deriv.im);

    // denom = br^2 + bi^2
    mpf_mul(t1_c, dzdc_coord.re, dzdc_coord.re);
    mpf_mul(t2_c, dzdc_coord.im, dzdc_coord.im);
    mpf_add(denom_c, t1_c, t2_c);
    if (mpf_cmp_ui(denom_c, 0) == 0)
        return false;

    // tr = ar*br + ai*bi
    mpf_mul(t1_c, z_coord.re, dzdc_coord.re);
    mpf_mul(t2_c, z_coord.im, dzdc_coord.im);
    mpf_add(tr_c, t1_c, t2_c);

    // ti = ai*br - ar*bi
    mpf_mul(t1_c, z_coord.im, dzdc_coord.re);
    mpf_mul(t2_c, z_coord.re, dzdc_coord.im);
    mpf_sub(ti_c, t1_c, t2_c);

    // step = (tr + i*ti) / denom
    mpf_div(step_coord.re, tr_c, denom_c);
    mpf_div(step_coord.im, ti_c, denom_c);
    return true;
}

// ------------------------------------------------------------
// Halley step computation in COORD precision (mpf) with LOW-PRECISION
// second derivative carried in HDRFloat.
//
// For F(c)=z_p(c), with F' = dzdc, F'' = d2zdc2:
//
//   Δ_H = (2 * F * F') / (2*(F')^2 - F*F'')
//
// Here:
//   - F is mpf (coord_prec): z_coord
//   - F' is mpf (deriv_prec): dzdc_deriv, promoted to coord_prec mpf
//   - F'' is HDRFloat (low-prec, huge exponent): d2r_hdr, d2i_hdr
//       promoted to coord_prec mpf via HighPrecision->mpf_t
//
// Returns false if denominator is (near-)singular.
//
// NOTE: This is a "drop-in" step analogous to Newton's step.
// You still update: c <- c - step
// ------------------------------------------------------------
template <typename IterType, typename T>
static inline bool
ComputeHalleyStep_mpf_coord_from_deriv(mpf_complex &step_coord,         // coord_prec (out)
                                       const mpf_complex &z_coord,      // coord_prec  F
                                       const mpf_complex &dzdc_deriv,   // deriv_prec  F'
                                       const HDRFloat<double> &d2r_hdr, // low-prec F'' real
                                       const HDRFloat<double> &d2i_hdr, // low-prec F'' imag
                                       mpf_complex &dzdc_coord, // coord_prec scratch (promoted F')
                                       mpf_complex &tmp1,       // coord_prec scratch complex
                                       mpf_complex &tmp2,       // coord_prec scratch complex
                                       mpf_t denom_c,           // coord_prec scalar
                                       mpf_t tr_c,
                                       mpf_t ti_c, // coord_prec scalars
                                       mpf_t t1_c,
                                       mpf_t t2_c) // coord_prec scalars
{
    // -----------------------------
    // Promote F' to coord precision
    // -----------------------------
    mpf_set(dzdc_coord.re, dzdc_deriv.re);
    mpf_set(dzdc_coord.im, dzdc_deriv.im);

    // -----------------------------
    // Promote HDRFloat d2 -> mpf (coord_prec) into tmp2 (reuse as d2_coord)
    //   tmp2 = d2_coord
    // -----------------------------
    {
        HighPrecision d2r_hp, d2i_hp;
        d2r_hdr.GetHighPrecision(d2r_hp);
        d2i_hdr.GetHighPrecision(d2i_hp);
        mpf_set(tmp2.re, *d2r_hp.backendRaw());
        mpf_set(tmp2.im, *d2i_hp.backendRaw());
    }

    // tmp1 = dzdc_coord^2  (F')^2
    // (a+bi)^2 = (a^2-b^2) + (2ab)i
    {
        mpf_mul(t1_c, dzdc_coord.re, dzdc_coord.re); // a^2
        mpf_mul(t2_c, dzdc_coord.im, dzdc_coord.im); // b^2
        mpf_sub(tr_c, t1_c, t2_c);                   // re

        mpf_mul(t1_c, dzdc_coord.re, dzdc_coord.im); // ab
        mpf_mul_ui(ti_c, t1_c, 2);                   // im

        mpf_set(tmp1.re, tr_c);
        mpf_set(tmp1.im, ti_c);
    }

    // tmp1 = 2*(F')^2
    mpf_mul_ui(tmp1.re, tmp1.re, 2);
    mpf_mul_ui(tmp1.im, tmp1.im, 2);

    // tmp2 = F * F''  (z_coord * d2_coord)
    // (ar+ai i)*(br+bi i) = (ar*br - ai*bi) + (ar*bi + ai*br)i
    // NOTE: tmp2 currently holds d2_coord, so we must compute into (tr_c,ti_c) and then overwrite tmp2.
    {
        mpf_mul(t1_c, z_coord.re, tmp2.re);
        mpf_mul(t2_c, z_coord.im, tmp2.im);
        mpf_sub(tr_c, t1_c, t2_c);

        mpf_mul(t1_c, z_coord.re, tmp2.im);
        mpf_mul(t2_c, z_coord.im, tmp2.re);
        mpf_add(ti_c, t1_c, t2_c);

        mpf_set(tmp2.re, tr_c);
        mpf_set(tmp2.im, ti_c);
    }

    // Den = 2*(F')^2 - F*F''  (complex)
    // Store Den in tmp2: tmp2 = tmp1 - tmp2
    mpf_sub(tmp2.re, tmp1.re, tmp2.re);
    mpf_sub(tmp2.im, tmp1.im, tmp2.im);

    // Numerator = 2 * F * F'  (complex)
    // Compute tmp1 = F*F' first, then scale by 2
    {
        mpf_mul(t1_c, z_coord.re, dzdc_coord.re);
        mpf_mul(t2_c, z_coord.im, dzdc_coord.im);
        mpf_sub(tr_c, t1_c, t2_c); // re = ar*br - ai*bi

        mpf_mul(t1_c, z_coord.re, dzdc_coord.im);
        mpf_mul(t2_c, z_coord.im, dzdc_coord.re);
        mpf_add(ti_c, t1_c, t2_c); // im = ar*bi + ai*br

        mpf_set(tmp1.re, tr_c);
        mpf_set(tmp1.im, ti_c);
    }
    mpf_mul_ui(tmp1.re, tmp1.re, 2);
    mpf_mul_ui(tmp1.im, tmp1.im, 2);

    // step = Numer / Den  computed as (Numer * conj(Den)) / |Den|^2
    // denom_c = |Den|^2
    mpf_mul(t1_c, tmp2.re, tmp2.re);
    mpf_mul(t2_c, tmp2.im, tmp2.im);
    mpf_add(denom_c, t1_c, t2_c);
    if (mpf_cmp_ui(denom_c, 0) == 0)
        return false;

    // tr = num_re*den_re + num_im*den_im    (since multiplying by conj)
    mpf_mul(t1_c, tmp1.re, tmp2.re);
    mpf_mul(t2_c, tmp1.im, tmp2.im);
    mpf_add(tr_c, t1_c, t2_c);

    // ti = num_im*den_re - num_re*den_im
    mpf_mul(t1_c, tmp1.im, tmp2.re);
    mpf_mul(t2_c, tmp1.re, tmp2.im);
    mpf_sub(ti_c, t1_c, t2_c);

    mpf_div(step_coord.re, tr_c, denom_c);
    mpf_div(step_coord.im, ti_c, denom_c);
    return true;
}

// ------------------------------------------------------------
// NR Checkpoint helpers
// ------------------------------------------------------------
static const char *NRCheckpointFilename = "nr_checkpoint.txt";

const char *
NRCheckpointPhaseName(NRCheckpointPhase phase)
{
    switch (phase) {
        case NRCheckpointPhase::Main:
            return "main";
        case NRCheckpointPhase::Final:
            return "final";
        case NRCheckpointPhase::Complete:
            return "complete";
    }

    return "main";
}

static std::string
ReconstructMpfString(const std::string &digits, mp_exp_t exp)
{
    if (digits.empty() || digits == "0")
        return "0";
    if (digits[0] == '-') {
        return "-0." + digits.substr(1) + "@" + std::to_string(exp);
    }
    return "0." + digits + "@" + std::to_string(exp);
}

static void
WriteMpfField(std::ostream &f, const char *label, mpf_srcptr val, size_t ndigits)
{
    mp_exp_t exp;
    char *str = mpf_get_str(nullptr, &exp, 10, ndigits, val);
    f << label << ": " << exp << " " << (str ? str : "0") << "\n";
    if (str)
        free(str);
}

struct NRCheckpointParams {
    mpf_srcptr c_re, c_im;
    mpf_srcptr cand_re, cand_im;
    mpf_srcptr sqrRadius, intrinsicRadius;
    uint64_t period;
    mp_bitcnt_t coord_prec;
    uint32_t iteration;
    NRCheckpointPhase phase;
    int scaleExp2;
    uint64_t numIterationsAtFind;
    uint64_t innerIteration;
    mpf_srcptr z_re, z_im;
    mpf_srcptr dzdc_re, dzdc_im;
    mp_bitcnt_t deriv_prec;
    HDRFloat<double> d2r, d2i;
    DiagnosticState diag;
};

std::string
ComputeNRCheckpointPreviewZoom(const DiagnosticState &diag, const HighPrecision &intrinsicRadius)
{
    if (!diag.valid || diag.step_norm.getMantissa() <= 0.0 || intrinsicRadius <= HighPrecision{0})
        return "unavailable";

    HighPrecision estimatedRemainingDistance;
    // step_norm is a legacy checkpoint label; it stores |step|^2 from mpf_complex_norm().
    HdrSqrt(diag.step_norm).GetHighPrecision(estimatedRemainingDistance);

    const HighPrecision &previewRadius =
        estimatedRemainingDistance > intrinsicRadius ? estimatedRemainingDistance : intrinsicRadius;

    return FeatureSummary::ComputeZoomFactorForRadius(previewRadius).str();
}

static std::string
ComputeCheckpointPreviewZoom(const NRCheckpointParams &p)
{
    const HighPrecision intrinsicRadius{p.intrinsicRadius};
    return ComputeNRCheckpointPreviewZoom(p.diag, intrinsicRadius);
}

static bool
WriteNRCheckpoint(const NRCheckpointParams &p)
{
    // Write to temp file, then rename for crash-safe atomic update.
    static const char *NRCheckpointTmpFilename = "nr_checkpoint.tmp";

    std::ofstream f(NRCheckpointTmpFilename, std::ios::trunc);
    if (!f)
        return false;

    const size_t ndigits = static_cast<size_t>(p.coord_prec * 0.302) + 10;

    f << "# Integer fields are decimal unless prefixed with 0x.\n"
      << "# Fields named exp or exp2 are decimal exponents for powers of 2.\n"
      << "# MPIR fields are written as decimal exponent plus decimal digits.\n"
      << "period: " << p.period << "\n"
      << "coord_prec: " << p.coord_prec << "\n"
      << "scaleExp2: " << p.scaleExp2 << "\n"
      << "iteration: " << p.iteration << "\n"
      << "phase: " << NRCheckpointPhaseName(p.phase) << "\n";

    WriteMpfField(f, "c_re", p.c_re, ndigits);
    WriteMpfField(f, "c_im", p.c_im, ndigits);
    WriteMpfField(f, "cand_re", p.cand_re, ndigits);
    WriteMpfField(f, "cand_im", p.cand_im, ndigits);
    WriteMpfField(f, "sqrRadius", p.sqrRadius, ndigits);
    WriteMpfField(f, "intrinsicRadius", p.intrinsicRadius, ndigits);

    f << "numIterationsAtFind: " << p.numIterationsAtFind << "\n"
      << "innerIteration: " << p.innerIteration << "\n"
      << "deriv_prec: " << p.deriv_prec << "\n";

    WriteMpfField(f, "z_re", p.z_re, ndigits);
    WriteMpfField(f, "z_im", p.z_im, ndigits);
    WriteMpfField(f, "dzdc_re", p.dzdc_re, ndigits);
    WriteMpfField(f, "dzdc_im", p.dzdc_im, ndigits);

    f << "d2r: " << p.d2r.getExp() << " " << std::setprecision(std::numeric_limits<double>::max_digits10)
      << p.d2r.getMantissa() << "\n"
      << "d2i: " << p.d2i.getExp() << " " << std::setprecision(std::numeric_limits<double>::max_digits10)
      << p.d2i.getMantissa() << "\n";

    f << "\n"
      << "# -----------------------------------------------------------------------------\n"
      << "# OPTIONAL INFORMATIONAL FIELDS BELOW - NOT REQUIRED FOR CHECKPOINT RESUME\n"
      << "# -----------------------------------------------------------------------------\n";

    f << "z_mag2: " << std::setprecision(std::numeric_limits<double>::max_digits10) << p.diag.z_mag2
      << "\n"
      << "c_cand_dist2: " << p.diag.c_cand_dist2.getExp() << " "
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << p.diag.c_cand_dist2.getMantissa() << "\n"
      << "inner_pct: " << std::setprecision(6) << p.diag.inner_pct << "\n"
      << "targetExp: " << p.diag.targetExp << "\n"
      << "diag_valid: " << (p.diag.valid ? 1 : 0) << "\n";

    if (p.diag.valid) {
        f << "rho2: " << p.diag.rho2.getExp() << " "
          << std::setprecision(std::numeric_limits<double>::max_digits10) << p.diag.rho2.getMantissa()
          << "\n"
          << "err: " << p.diag.err.getExp() << " "
          << std::setprecision(std::numeric_limits<double>::max_digits10) << p.diag.err.getMantissa()
          << "\n"
          << "step_norm: " << p.diag.step_norm.getExp() << " "
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << p.diag.step_norm.getMantissa() << "\n"
          << "wantHalley: " << (p.diag.wantHalley ? 1 : 0) << "\n"
          << "normalized_bits: " << p.diag.normalized_bits << "\n"
          << "est_remaining: " << p.diag.est_remaining << "\n";
    }

    f << "previewZoom: " << ComputeCheckpointPreviewZoom(p) << "\n";

    f.close();

    // Atomic replace: rename tmp over existing file in one step.
    std::filesystem::rename(NRCheckpointTmpFilename, NRCheckpointFilename);
    return true;
}

// ------------------------------------------------------------
// Async checkpoint writer — owns independent mpf copies so the
// slow mpf_get_str + file I/O happens off the critical path.
// ------------------------------------------------------------

struct CheckpointSnapshot {
    mpf_complex c, c0, z, dzdc;
    mpf_t sqrRadius, intrinsicRadius;
    uint64_t period, innerIteration, numIterationsAtFind;
    mp_bitcnt_t coord_prec, deriv_prec;
    uint32_t iteration;
    NRCheckpointPhase phase;
    int scaleExp2;
    HDRFloat<double> d2r, d2i;
    DiagnosticState diag;

    CheckpointSnapshot(const NRCheckpointParams &src, uint64_t iters)
    {
        coord_prec = src.coord_prec;
        deriv_prec = src.deriv_prec;

        mpf_complex_init(c, coord_prec);
        mpf_complex_init(c0, coord_prec);
        mpf_complex_init(z, coord_prec);
        mpf_complex_init(dzdc, coord_prec);
        mpf_init2(sqrRadius, coord_prec);
        mpf_init2(intrinsicRadius, coord_prec);

        mpf_set(c.re, src.c_re);
        mpf_set(c.im, src.c_im);
        mpf_set(c0.re, src.cand_re);
        mpf_set(c0.im, src.cand_im);
        mpf_set(sqrRadius, src.sqrRadius);
        mpf_set(intrinsicRadius, src.intrinsicRadius);
        mpf_set(z.re, src.z_re);
        mpf_set(z.im, src.z_im);
        mpf_set(dzdc.re, src.dzdc_re);
        mpf_set(dzdc.im, src.dzdc_im);

        period = src.period;
        innerIteration = iters;
        numIterationsAtFind = src.numIterationsAtFind;
        iteration = src.iteration;
        phase = src.phase;
        scaleExp2 = src.scaleExp2;
        d2r = src.d2r;
        d2i = src.d2i;
        diag = src.diag;
    }

    ~CheckpointSnapshot()
    {
        mpf_complex_clear(c);
        mpf_complex_clear(c0);
        mpf_complex_clear(z);
        mpf_complex_clear(dzdc);
        mpf_clear(sqrRadius);
        mpf_clear(intrinsicRadius);
    }

    CheckpointSnapshot(const CheckpointSnapshot &) = delete;
    CheckpointSnapshot &operator=(const CheckpointSnapshot &) = delete;

    NRCheckpointParams
    ToParams() const
    {
        return {c.re,
                c.im,
                c0.re,
                c0.im,
                sqrRadius,
                intrinsicRadius,
                period,
                coord_prec,
                iteration,
                phase,
                scaleExp2,
                numIterationsAtFind,
                innerIteration,
                z.re,
                z.im,
                dzdc.re,
                dzdc.im,
                deriv_prec,
                d2r,
                d2i,
                diag};
    }
};

class CheckpointWriter {
public:
    explicit CheckpointWriter(NRCheckpointSavePolicy savePolicy) : m_SavePolicy(savePolicy)
    {
        if (ShouldWrite()) {
            m_Thread = std::thread(&CheckpointWriter::Run, this);
        }
    }

    ~CheckpointWriter()
    {
        if (!ShouldWrite()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_Exit = true;
        }
        m_CV.notify_one();
        if (m_Thread.joinable()) {
            m_Thread.join();
        }
    }

    CheckpointWriter(const CheckpointWriter &) = delete;
    CheckpointWriter &operator=(const CheckpointWriter &) = delete;

    // Fire-and-forget: for inner-loop progress callbacks.
    // Supersedes any pending fire-and-forget write, but never
    // supersedes a synchronous WriteAndWait (whose caller would hang).
    void
    TriggerWrite(std::unique_ptr<CheckpointSnapshot> snapshot)
    {
        if (!ShouldWrite()) {
            if (m_SavePolicy == NRCheckpointSavePolicy::PreserveExisting && snapshot) {
                PrintCheckpointMessage(snapshot->ToParams(), false);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            if (m_DonePromise)
                return;
            m_Pending = std::move(snapshot);
        }
        m_CV.notify_one();
    }

    // Synchronous: enqueue a write and block until the writer thread
    // finishes the I/O.  Used by the outer NR loop for critical
    // checkpoints (abort, post-step, final).
    void
    WriteAndWait(const NRCheckpointParams &params)
    {
        if (!ShouldWrite()) {
            if (m_SavePolicy == NRCheckpointSavePolicy::PreserveExisting) {
                PrintCheckpointMessage(params, false);
            }
            return;
        }
        auto snapshot = std::make_unique<CheckpointSnapshot>(params, params.innerIteration);
        std::promise<void> done;
        auto future = done.get_future();
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_Pending = std::move(snapshot);
            m_DonePromise = &done;
        }
        m_CV.notify_one();
        future.wait();
    }

private:
    void
    Run()
    {
        for (;;) {
            std::unique_ptr<CheckpointSnapshot> snap;
            std::promise<void> *done = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_Mutex);
                m_CV.wait(lock, [&] { return m_Pending != nullptr || m_Exit; });
                snap = std::move(m_Pending);
                done = m_DonePromise;
                m_DonePromise = nullptr;
                if (m_Exit && !snap)
                    break;
            }

            if (snap) {
                auto params = snap->ToParams();
                if (WriteNRCheckpoint(params)) {
                    PrintCheckpointMessage(params, true);
                }
                if (done)
                    done->set_value();
            }
        }
    }

    bool
    ShouldWrite() const
    {
        return m_SavePolicy == NRCheckpointSavePolicy::Save;
    }

    void
    PrintCheckpointMessage(const NRCheckpointParams &params, bool saved) const
    {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "RefinePeriodicPoint: NR checkpoint "
            << (saved ? "saved" : "not saved because policy is PreserveExisting") << " at NR iter "
            << params.iteration << " phase " << NRCheckpointPhaseName(params.phase) << " innerIter "
            << params.innerIteration << " period " << params.period;
    }

    std::mutex m_Mutex;
    std::condition_variable m_CV;
    std::unique_ptr<CheckpointSnapshot> m_Pending;
    std::promise<void> *m_DonePromise = nullptr;
    NRCheckpointSavePolicy m_SavePolicy = NRCheckpointSavePolicy::Save;
    bool m_Exit = false;
    // Must be last: the worker can run immediately after construction starts it.
    std::thread m_Thread;
};

struct InnerLoopCheckpointData {
    uint64_t innerIteration{0};
    mp_bitcnt_t deriv_prec{0};
    NRCheckpointPhase phase{NRCheckpointPhase::Main};
};

static void
SkipCheckpointComments(std::ifstream &f)
{
    for (;;) {
        f >> std::ws;
        if (f.peek() != '#')
            return;

        std::string comment;
        std::getline(f, comment);
    }
}

[[noreturn]] static void
ThrowMissingOrMislabeledCheckpointField(const char *label)
{
    throw std::runtime_error("NR checkpoint missing or mislabeled required field '" +
                             std::string(label) + "'; refusing to resume or overwrite it.");
}

[[noreturn]] static void
ThrowInvalidCheckpointField(const char *label)
{
    throw std::runtime_error("NR checkpoint has invalid required value for field '" +
                             std::string(label) + "'; refusing to resume or overwrite it.");
}

static void
ReadRequiredLabel(std::ifstream &f, const char *expected)
{
    SkipCheckpointComments(f);

    std::string token;
    if (!(f >> token) || token != std::string(expected) + ":")
        ThrowMissingOrMislabeledCheckpointField(expected);
}

template <typename Value>
static void
ReadRequiredField(std::ifstream &f, const char *label, Value &value)
{
    ReadRequiredLabel(f, label);
    if (!(f >> value))
        ThrowInvalidCheckpointField(label);
}

static void
ReadRequiredMpfField(std::ifstream &f, const char *label, mp_exp_t &exp, std::string &digits)
{
    ReadRequiredLabel(f, label);
    if (!(f >> exp >> digits))
        ThrowInvalidCheckpointField(label);

    size_t firstDigit = (digits.starts_with('-')) ? 1 : 0;
    if (firstDigit == digits.size() ||
        !std::all_of(
            digits.begin() + firstDigit, digits.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) {
        ThrowInvalidCheckpointField(label);
    }
}

static void
ReadRequiredHdrField(std::ifstream &f, const char *label, HDRFloat<double> &value)
{
    int64_t exponent;
    double mantissa;
    ReadRequiredLabel(f, label);
    if (!(f >> exponent >> mantissa) || exponent < std::numeric_limits<int32_t>::min() ||
        exponent > std::numeric_limits<int32_t>::max() || !std::isfinite(mantissa)) {
        ThrowInvalidCheckpointField(label);
    }

    value = HDRFloat<double>(static_cast<int32_t>(exponent), mantissa);
}

static void
ReadRequiredDiagnosticFlag(std::ifstream &f, const char *label, bool &value)
{
    int intValue;
    ReadRequiredField(f, label, intValue);
    if (intValue != 0 && intValue != 1)
        ThrowInvalidCheckpointField(label);

    value = (intValue != 0);
}

static NRCheckpointPhase
ReadCheckpointPhase(std::ifstream &f)
{
    SkipCheckpointComments(f);

    std::string token;
    if (!(f >> token)) {
        throw std::runtime_error(
            "NR checkpoint missing required phase field; refusing to resume or overwrite it.");
    }

    if (token != "phase:") {
        throw std::runtime_error(
            "NR checkpoint missing required phase field; refusing to resume or overwrite it.");
    }

    std::string value;
    if (!(f >> value)) {
        throw std::runtime_error(
            "NR checkpoint missing required phase value; refusing to resume or overwrite it.");
    }

    if (value == "main") {
        return NRCheckpointPhase::Main;
    } else if (value == "final") {
        return NRCheckpointPhase::Final;
    } else if (value == "complete") {
        return NRCheckpointPhase::Complete;
    }

    throw std::runtime_error("NR checkpoint has invalid phase value: " + value +
                             "; refusing to resume or overwrite it.");
}

struct NRCheckpointHeader {
    uint64_t period;
    mp_bitcnt_t coordPrec;
    int scaleExp2;
    uint32_t iteration;
    NRCheckpointPhase phase;
};

static NRCheckpointHeader
ReadNRCheckpointHeader(std::ifstream &f)
{
    NRCheckpointHeader header{};
    ReadRequiredField(f, "period", header.period);
    ReadRequiredField(f, "coord_prec", header.coordPrec);
    ReadRequiredField(f, "scaleExp2", header.scaleExp2);
    ReadRequiredField(f, "iteration", header.iteration);
    header.phase = ReadCheckpointPhase(f);
    return header;
}

static bool
OpenExistingNRCheckpoint(std::ifstream &f)
{
    f.open(NRCheckpointFilename);
    if (f)
        return true;

    if (!std::filesystem::exists(NRCheckpointFilename))
        return false;

    throw std::runtime_error(
        "Unable to open existing NR checkpoint; refusing to resume or overwrite it.");
}

static void
ReadOptionalCheckpointDiagnostics(std::ifstream &f, DiagnosticState &diag)
{
    diag = {};
    SkipCheckpointComments(f);
    if (f.peek() == std::ifstream::traits_type::eof())
        return;

    ReadRequiredField(f, "z_mag2", diag.z_mag2);
    if (!std::isfinite(diag.z_mag2))
        ThrowInvalidCheckpointField("z_mag2");
    ReadRequiredHdrField(f, "c_cand_dist2", diag.c_cand_dist2);
    ReadRequiredField(f, "inner_pct", diag.inner_pct);
    if (!std::isfinite(diag.inner_pct))
        ThrowInvalidCheckpointField("inner_pct");
    ReadRequiredField(f, "targetExp", diag.targetExp);
    ReadRequiredDiagnosticFlag(f, "diag_valid", diag.valid);

    if (diag.valid) {
        ReadRequiredHdrField(f, "rho2", diag.rho2);
        ReadRequiredHdrField(f, "err", diag.err);
        ReadRequiredHdrField(f, "step_norm", diag.step_norm);
        ReadRequiredDiagnosticFlag(f, "wantHalley", diag.wantHalley);
        ReadRequiredField(f, "normalized_bits", diag.normalized_bits);
        ReadRequiredField(f, "est_remaining", diag.est_remaining);
    }

    std::string previewZoom;
    ReadRequiredField(f, "previewZoom", previewZoom);
}

static void
ValidateExistingNRCheckpointPhase()
{
    std::ifstream f;
    if (!OpenExistingNRCheckpoint(f))
        return;

    ReadNRCheckpointHeader(f);
}

// Reads checkpoint with label verification on every field.
// Returns true if the checkpoint matches expected_period/expected_prec.
static bool
TryReadNRCheckpointWithInner(mpf_complex &c,
                             const mpf_complex &expectedCandidate,
                             const mpf_t expectedRadius,
                             uint64_t expected_period,
                             mp_bitcnt_t expected_prec,
                             uint32_t &out_iteration,
                             InnerLoopCheckpointData &inner,
                             mpf_complex &out_z,
                             mpf_complex &out_dzdc,
                             HDRFloat<double> &out_d2r,
                             HDRFloat<double> &out_d2i,
                             DiagnosticState &out_diag)
{
    std::ifstream f;
    if (!OpenExistingNRCheckpoint(f))
        return false;

    const NRCheckpointHeader header = ReadNRCheckpointHeader(f);

    mp_exp_t exp_re, exp_im;
    std::string digits_re, digits_im;
    ReadRequiredMpfField(f, "c_re", exp_re, digits_re);
    ReadRequiredMpfField(f, "c_im", exp_im, digits_im);

    mp_exp_t candidateRealExp, candidateImagExp, radiusExp, skip_exp;
    std::string candidateRealDigits, candidateImagDigits, radiusDigits, skip_str;
    ReadRequiredMpfField(f, "cand_re", candidateRealExp, candidateRealDigits);
    ReadRequiredMpfField(f, "cand_im", candidateImagExp, candidateImagDigits);
    ReadRequiredMpfField(f, "sqrRadius", radiusExp, radiusDigits);
    ReadRequiredMpfField(f, "intrinsicRadius", skip_exp, skip_str);

    uint64_t skip_numIters;
    ReadRequiredField(f, "numIterationsAtFind", skip_numIters);

    uint64_t innerIter = 0;
    mp_bitcnt_t derivPrec;
    ReadRequiredField(f, "innerIteration", innerIter);
    ReadRequiredField(f, "deriv_prec", derivPrec);

    mp_exp_t z_exp_re, z_exp_im, dz_exp_re, dz_exp_im;
    std::string z_d_re, z_d_im, dz_d_re, dz_d_im;
    ReadRequiredMpfField(f, "z_re", z_exp_re, z_d_re);
    ReadRequiredMpfField(f, "z_im", z_exp_im, z_d_im);
    ReadRequiredMpfField(f, "dzdc_re", dz_exp_re, dz_d_re);
    ReadRequiredMpfField(f, "dzdc_im", dz_exp_im, dz_d_im);

    HDRFloat<double> d2r, d2i;
    ReadRequiredHdrField(f, "d2r", d2r);
    ReadRequiredHdrField(f, "d2i", d2i);

    DiagnosticState diag;
    ReadOptionalCheckpointDiagnostics(f, diag);

    if (header.period != expected_period || header.coordPrec != expected_prec)
        return false;

    const auto matchesMpf =
        [expected_prec](
            const std::string &digits, mp_exp_t exponent, mpf_srcptr expected, const char *label) {
            HighPrecision parsed{HighPrecision::SetPrecision::True, expected_prec};
            const std::string value = ReconstructMpfString(digits, exponent);
            if (mpf_set_str(parsed.backend(), value.c_str(), 10) != 0)
                ThrowInvalidCheckpointField(label);
            return mpf_cmp(parsed.backend(), expected) == 0;
        };
    if (!matchesMpf(candidateRealDigits, candidateRealExp, expectedCandidate.re, "cand_re") ||
        !matchesMpf(candidateImagDigits, candidateImagExp, expectedCandidate.im, "cand_im") ||
        !matchesMpf(radiusDigits, radiusExp, expectedRadius, "sqrRadius")) {
        return false;
    }

    auto setMpf =
        [](mpf_ptr destination, const std::string &digits, mp_exp_t exponent, const char *label) {
            const std::string value = ReconstructMpfString(digits, exponent);
            if (mpf_set_str(destination, value.c_str(), 10) != 0)
                ThrowInvalidCheckpointField(label);
        };

    setMpf(c.re, digits_re, exp_re, "c_re");
    setMpf(c.im, digits_im, exp_im, "c_im");
    setMpf(out_z.re, z_d_re, z_exp_re, "z_re");
    setMpf(out_z.im, z_d_im, z_exp_im, "z_im");
    setMpf(out_dzdc.re, dz_d_re, dz_exp_re, "dzdc_re");
    setMpf(out_dzdc.im, dz_d_im, dz_exp_im, "dzdc_im");

    out_iteration = header.iteration;
    inner = {innerIter, derivPrec, header.phase};
    out_d2r = d2r;
    out_d2i = d2i;
    out_diag = diag;

    return true;
}

// Public: read full checkpoint for orchestrator resume
bool
ReadFullNRCheckpoint(NRCheckpointData &out)
{
    std::ifstream f;
    if (!OpenExistingNRCheckpoint(f))
        return false;

    const NRCheckpointHeader header = ReadNRCheckpointHeader(f);

    out.period = header.period;
    out.coord_prec = header.coordPrec;
    out.scaleExp2 = header.scaleExp2;
    out.iteration = header.iteration;
    out.phase = header.phase;
    out.innerIteration = 0;
    out.diag = {};

    mp_exp_t exp_cre, exp_cim, exp_candre, exp_candim, exp_rad, exp_ir;
    std::string d_cre, d_cim, d_candre, d_candim, d_rad, d_ir;

    ReadRequiredMpfField(f, "c_re", exp_cre, d_cre);
    ReadRequiredMpfField(f, "c_im", exp_cim, d_cim);
    ReadRequiredMpfField(f, "cand_re", exp_candre, d_candre);
    ReadRequiredMpfField(f, "cand_im", exp_candim, d_candim);
    ReadRequiredMpfField(f, "sqrRadius", exp_rad, d_rad);
    ReadRequiredMpfField(f, "intrinsicRadius", exp_ir, d_ir);

    ReadRequiredField(f, "numIterationsAtFind", out.numIterationsAtFind);

    auto toHP = [&out](const std::string &digits, mp_exp_t exp, const char *label) -> HighPrecision {
        std::string s = ReconstructMpfString(digits, exp);
        HighPrecision hp{HighPrecision::SetPrecision::True, out.coord_prec};
        if (mpf_set_str(hp.backend(), s.c_str(), 10) != 0)
            ThrowInvalidCheckpointField(label);
        MpfNormalize(hp.backend());
        return hp;
    };

    out.c_re = toHP(d_cre, exp_cre, "c_re");
    out.c_im = toHP(d_cim, exp_cim, "c_im");
    out.cand_re = toHP(d_candre, exp_candre, "cand_re");
    out.cand_im = toHP(d_candim, exp_candim, "cand_im");
    out.sqrRadius = toHP(d_rad, exp_rad, "sqrRadius");
    out.intrinsicRadius = toHP(d_ir, exp_ir, "intrinsicRadius");

    mp_bitcnt_t derivPrec;
    mp_exp_t skipExp;
    std::string skipStr;
    HDRFloat<double> ignoredHdr;

    ReadRequiredField(f, "innerIteration", out.innerIteration);
    ReadRequiredField(f, "deriv_prec", derivPrec);
    ReadRequiredMpfField(f, "z_re", skipExp, skipStr);
    ReadRequiredMpfField(f, "z_im", skipExp, skipStr);
    ReadRequiredMpfField(f, "dzdc_re", skipExp, skipStr);
    ReadRequiredMpfField(f, "dzdc_im", skipExp, skipStr);
    ReadRequiredHdrField(f, "d2r", ignoredHdr);
    ReadRequiredHdrField(f, "d2i", ignoredHdr);

    ReadOptionalCheckpointDiagnostics(f, out.diag);

    return true;
}

void
DeleteNRCheckpoint()
{
    std::remove(NRCheckpointFilename);
}

// ------------------------------------------------------------
// Imagina-style Newton/Halley polish for periodic point
//
// Goal:
//   Solve F(c) = z_p(c) = 0
//
// Pipeline:
//   z        → mpf (coord precision)
//   dzdc     → mpf (deriv precision)
//   d2zdc2   → HDRFloat (low precision, large exponent)
//   err      → HDRFloat
//
// Iteration step:
//   Newton:  step = z / dzdc
//   Halley:  step = (2 F F') / (2(F')^2 − F F'')
//
// Halley is used only when the dimensionless ratio
//
//   rho^2 = |z|^2 |d2|^2 / |dzdc|^4
//
// is sufficiently small, ensuring the Halley denominator
// is dominated by 2(F')^2.
//
// Stop condition (Imagina-style):
//
//   err = |step|^4 * |d2|^2 / |dzdc|^2
//   stop when −ilogb(err) ≥ 2 * coord_prec
// ------------------------------------------------------------
template <typename IterType, typename T>
static inline NRPolishResult
RefinePeriodicPoint(mpf_complex &c_coord,        // coord_prec in/out
                    const mpf_complex &c0_coord, // coord_prec (initial seed)
                    mpf_t sqrRadius_coord,       // coord_prec (R^2) for final accept/reject
                    uint64_t period,
                    mp_bitcnt_t coord_prec,
                    int scaleExp2_for_deriv_choice, // exponent of Scale ≈ 1/|zcoeff*dzdc|
                    uint32_t max_nr_iters,
                    NRInnerLoopBackend backend,
                    mpf_t intrinsicRadius_mpf,
                    uint64_t numIterationsAtFind,
                    NRCheckpointSavePolicy checkpointSavePolicy)
{
    // Compile-time enable, runtime gating below.
    constexpr bool UseHalley = true;
    constexpr bool UseFullPrecDerivatives = false;

    // Gate: require rho^2 < 2^-k (k bigger => more conservative)
    // With rho^2 = |z|^2*|d2|^2 / |dzdc|^4.
    // For low-precision d2, be conservative.
    constexpr int HalleyRho2ExpThreshold = -12; // rho^2 < 2^-12

    // ---------------- coord temporaries ----------------
    mpf_t denom_c, tr_c, ti_c, t1_c, t2_c, abs2_c;
    mpf_init2(denom_c, coord_prec);
    mpf_init2(tr_c, coord_prec);
    mpf_init2(ti_c, coord_prec);
    mpf_init2(t1_c, coord_prec);
    mpf_init2(t2_c, coord_prec);
    mpf_init2(abs2_c, coord_prec);

    // We keep normStep in mpf (computed from mpf step), then promote to HDRFloat.
    mpf_t normStep;
    mpf_init2(normStep, coord_prec);

    // Also keep |z|^2 in mpf for the Halley gate (cheap; avoids mpf sqrt)
    mpf_t zNormSq_mpf;
    mpf_init2(zNormSq_mpf, coord_prec);

    // c-delta for final accept/reject
    mpf_complex dc;
    mpf_complex_init(dc, coord_prec);

    // ---------------- choose deriv precision (Imagina-like) ----------------
    // Estimate coordinate exponent from |c|
    const int coordExp2_re = approx_ilogb_mpf(c_coord.re);
    const int coordExp2_im = approx_ilogb_mpf(c_coord.im);
    const int coordExp2_max_abs = std::max(coordExp2_re, coordExp2_im);

    mp_bitcnt_t deriv_prec;

    if constexpr (UseFullPrecDerivatives) {
        deriv_prec = coord_prec;
    } else {
        deriv_prec = ChooseDerivPrec_ImaginaStyle(
            coord_prec, scaleExp2_for_deriv_choice, coordExp2_max_abs, /*minPrec*/ 256);
    }

    FractalSharkLog::LogLine(__FILE__, __LINE__)
        << "RefinePeriodicPoint: coord_prec(bits)=" << coord_prec << ", deriv_prec(bits)=" << deriv_prec
        << " (scaleExp2=" << scaleExp2_for_deriv_choice << ", coordExp2_max_abs=" << coordExp2_max_abs
        << ")";

    // ---------------- deriv temporaries ----------------
    mpf_t tr_d, ti_d, t1_d, t2_d;
    mpf_init2(tr_d, deriv_prec);
    mpf_init2(ti_d, deriv_prec);
    mpf_init2(t1_d, deriv_prec);
    mpf_init2(t2_d, deriv_prec);

    // ---------------- complex state (mpf) ----------------
    mpf_complex z_coord, step_coord, dzdc_coord, tmpZ_coord;
    mpf_complex_init(z_coord, coord_prec);
    mpf_complex_init(step_coord, coord_prec);
    mpf_complex_init(dzdc_coord, coord_prec);
    mpf_complex_init(tmpZ_coord, coord_prec);

    mpf_complex dzdc_deriv, z_deriv, tmpB_d;
    mpf_complex_init(dzdc_deriv, coord_prec);
    mpf_complex_init(z_deriv, coord_prec);
    mpf_complex_init(tmpB_d, coord_prec);

    // Extra coord scratch needed for Halley (mpf-only)
    mpf_complex d2_coord_scratch, htmp1, htmp2;
    mpf_complex_init(d2_coord_scratch, coord_prec);
    mpf_complex_init(htmp1, coord_prec);
    mpf_complex_init(htmp2, coord_prec);

    // ---------------- d2 output (HDRFloat) ----------------
    HDRFloat<double> d2r_hdr{}, d2i_hdr{};

    // ---------------- HDRFloat err pipeline scalars ----------------
    HDRFloat<double> normStep_hdr{};
    HDRFloat<double> normStep2_hdr{};
    HDRFloat<double> d2Norm_hdr{};
    HDRFloat<double> dzdcNorm_hdr{};
    HDRFloat<double> err_hdr{};

    // Halley gate scalars
    HDRFloat<double> zNorm_hdr{};
    HDRFloat<double> rho2_hdr{};
    HDRFloat<double> dzdcNormSq_hdr{}; // (|dzdc|^2)^2 i.e. |dzdc|^4

    // Imagina stop threshold: Precision*2 in exponent space
    const int targetExp = int(coord_prec) * 2;

    // Convergence diagnostics — carried forward across checkpoints.
    DiagnosticState diagState;

    // Async checkpoint writer — mpf_get_str + file I/O happens on a background thread.
    CheckpointWriter checkpointWriter(checkpointSavePolicy);

    // Context struct passed to onProgress callback via void* context pointer.
    struct ProgressContext {
        NRCheckpointParams *params;
        CheckpointWriter *writer;
        DiagnosticState *diagState; // outer diagState for live updates
    };

    auto onProgress = [](uint64_t itersCompleted, void *ctx) {
        auto *pctx = static_cast<ProgressContext *>(ctx);

        // Compute inner-loop diagnostics from live z values.
        const double zr = mpf_get_d(pctx->params->z_re);
        const double zi = mpf_get_d(pctx->params->z_im);
        const double z_mag2 = zr * zr + zi * zi;
        const double inner_pct =
            (pctx->params->period > 0)
                ? static_cast<double>(itersCompleted) / static_cast<double>(pctx->params->period) * 100.0
                : 0.0;

        pctx->diagState->z_mag2 = z_mag2;
        pctx->diagState->inner_pct = inner_pct;
        pctx->params->diag.z_mag2 = z_mag2;
        pctx->params->diag.inner_pct = inner_pct;

        pctx->writer->TriggerWrite(std::make_unique<CheckpointSnapshot>(*pctx->params, itersCompleted));
    };

    auto updateCandidateDistance = [&]() {
        mpf_complex_sub(dc, c_coord, c0_coord);
        mpf_complex_norm(abs2_c, dc, t1_c, t2_c);
        long distExp2;
        double distMant = mpf_get_d_2exp(&distExp2, abs2_c);
        diagState.c_cand_dist2 = HDRFloat<double>(static_cast<int32_t>(distExp2), distMant);
        HdrReduce(diagState.c_cand_dist2);
    };

    uint32_t startIter = 0;
    uint64_t innerStartIter = 0;
    uint64_t finalInnerStartIter = 0;
    bool resumeFinalPass = false;
    bool checkpointComplete = false;

    // Check for NR checkpoint file — resume automatically if found.
    {
        uint32_t savedIter = 0;
        InnerLoopCheckpointData inner{};
        if (TryReadNRCheckpointWithInner(c_coord,
                                         c0_coord,
                                         sqrRadius_coord,
                                         period,
                                         coord_prec,
                                         savedIter,
                                         inner,
                                         z_coord,
                                         dzdc_deriv,
                                         d2r_hdr,
                                         d2i_hdr,
                                         diagState)) {
            if (inner.phase == NRCheckpointPhase::Complete) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: checkpoint is already complete at iter " << savedIter
                    << "; skipping main and final NR phases.";
                startIter = savedIter;
                checkpointComplete = true;
            } else if (inner.phase == NRCheckpointPhase::Final) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: resuming final correction at innerIter "
                    << inner.innerIteration << "; skipping main NR loop.";
                startIter = savedIter;
                finalInnerStartIter = inner.innerIteration;
                resumeFinalPass = true;
            } else if (inner.innerIteration > 0) {
                // Resuming mid-inner-loop: outer loop at savedIter, inner at innerIteration
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: resuming from checkpoint iter " << savedIter
                    << " innerIter " << inner.innerIteration;
                startIter = savedIter;
                innerStartIter = inner.innerIteration;
            } else {
                // Resuming at completed NR step: outer loop at savedIter + 1
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: resuming from checkpoint iter " << savedIter;
                startIter = savedIter + 1;
            }
        }
    }

    // targetExp is authoritative — always restore after checkpoint read
    // (the reader zeros diagState, which would lose a pre-set targetExp).
    diagState.targetExp = targetExp;
    updateCandidateDistance();

    uint32_t it = startIter;
    NRPolishStatus polishStatus = NRPolishStatus::Accepted;
    for (; !resumeFinalPass && !checkpointComplete && it < max_nr_iters; ++it) {

        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "  Refinement iter " << it << " of " << max_nr_iters;

        // Full forward eval at current c
        uint64_t innerStart = (it == startIter) ? innerStartIter : 0;
        uint64_t completed = 0;

        // Checkpoint context for progress callbacks.
        NRCheckpointParams periodicParams{c_coord.re,
                                          c_coord.im,
                                          c0_coord.re,
                                          c0_coord.im,
                                          sqrRadius_coord,
                                          intrinsicRadius_mpf,
                                          period,
                                          coord_prec,
                                          it,
                                          NRCheckpointPhase::Main,
                                          scaleExp2_for_deriv_choice,
                                          numIterationsAtFind,
                                          0,
                                          z_coord.re,
                                          z_coord.im,
                                          dzdc_deriv.re,
                                          dzdc_deriv.im,
                                          deriv_prec,
                                          d2r_hdr,
                                          d2i_hdr,
                                          diagState};

        ProgressContext progressCtx{&periodicParams, &checkpointWriter, &diagState};

        completed = EvaluateCriticalOrbitAndDerivs(backend,
                                                   c_coord,
                                                   period,
                                                   z_coord,
                                                   dzdc_deriv,
                                                   d2r_hdr,
                                                   d2i_hdr,
                                                   coord_prec,
                                                   deriv_prec,
                                                   innerStart,
                                                   onProgress,
                                                   &progressCtx);

        if (completed < period) {
            // Compute inner-loop diagnostics for the abort checkpoint.
            const double zr = mpf_get_d(z_coord.re);
            const double zi = mpf_get_d(z_coord.im);
            diagState.z_mag2 = zr * zr + zi * zi;
            diagState.inner_pct =
                (period > 0) ? static_cast<double>(completed) / static_cast<double>(period) * 100.0
                             : 0.0;
            updateCandidateDistance();

            checkpointWriter.WriteAndWait({c_coord.re,
                                           c_coord.im,
                                           c0_coord.re,
                                           c0_coord.im,
                                           sqrRadius_coord,
                                           intrinsicRadius_mpf,
                                           period,
                                           coord_prec,
                                           it,
                                           NRCheckpointPhase::Main,
                                           scaleExp2_for_deriv_choice,
                                           numIterationsAtFind,
                                           completed,
                                           z_coord.re,
                                           z_coord.im,
                                           dzdc_deriv.re,
                                           dzdc_deriv.im,
                                           deriv_prec,
                                           d2r_hdr,
                                           d2i_hdr,
                                           diagState});
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "RefinePeriodicPoint: aborted at NR iter " << it << " innerIter " << completed;
            polishStatus = AbortMonitor::GetStopCalculatingGlobal() ? NRPolishStatus::Cancelled
                                                                    : NRPolishStatus::NumericalFailure;
            break;
        }

        // ------------------------------------------------------------
        // Build norms needed for: (a) Halley gate, (b) err estimate.
        // ------------------------------------------------------------

        // |z|^2 (mpf -> HDRFloat)
        mpf_complex_norm(zNormSq_mpf, z_coord, t1_c, t2_c);
        zNorm_hdr = HDRFloat<double>{zNormSq_mpf};
        HdrReduce(zNorm_hdr);

        // |d2|^2 (HDRFloat)
        d2Norm_hdr = d2r_hdr.square() + d2i_hdr.square();
        HdrReduce(d2Norm_hdr);

        // |dzdc|^2 (mpf -> HDRFloat)
        {
            HDRFloat<double> dzr{dzdc_deriv.re};
            HDRFloat<double> dzi{dzdc_deriv.im};
            HdrReduce(dzr);
            HdrReduce(dzi);

            dzdcNorm_hdr = dzr.square() + dzi.square();
            HdrReduce(dzdcNorm_hdr);
        }

        if (dzdcNorm_hdr.getMantissa() == 0.0) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "RefinePeriodicPoint: break after dzdcNorm==0";
            polishStatus = NRPolishStatus::NumericalFailure;
            break;
        }

        // ------------------------------------------------------------
        // Choose Halley vs Newton (gate on rho^2)
        //   rho^2 = |z|^2 * |d2|^2 / |dzdc|^4
        // ------------------------------------------------------------
        bool wantHalley = false;

        dzdcNormSq_hdr = dzdcNorm_hdr.square(); // |dzdc|^4
        HdrReduce(dzdcNormSq_hdr);

        rho2_hdr = (zNorm_hdr * d2Norm_hdr) / dzdcNormSq_hdr;
        HdrReduce(rho2_hdr);

        if constexpr (UseHalley) {
            // rho2_hdr.exp is ~ floor(log2(rho^2)) for reduced HDRFloat want
            // Halley if rho^2 is tiny (exp very negative) which should almost
            // every time.
            wantHalley = ((int)rho2_hdr.getExp() <= HalleyRho2ExpThreshold);
        }

        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "    rho2=" << rho2_hdr.ToString<false>() << " wantHalley=" << wantHalley;

        // ------------------------------------------------------------
        // Compute step
        // ------------------------------------------------------------
        if (wantHalley) {
            // Try Halley; fall back to Newton on failure.
            bool ok = ComputeHalleyStep_mpf_coord_from_deriv<IterType, T>(step_coord,
                                                                          z_coord,
                                                                          dzdc_deriv,
                                                                          d2r_hdr,
                                                                          d2i_hdr,
                                                                          dzdc_coord,
                                                                          htmp1,
                                                                          htmp2,
                                                                          denom_c,
                                                                          tr_c,
                                                                          ti_c,
                                                                          t1_c,
                                                                          t2_c);

            if (!ok) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: Halley denom singular, fallback to Newton";
                ok = ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c);
            }
            if (!ok) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: break after Halley/Newton failure";
                polishStatus = NRPolishStatus::NumericalFailure;
                break;
            }
        } else {
            // Newton step
            if (!ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "RefinePeriodicPoint: break after Newton";
                polishStatus = NRPolishStatus::NumericalFailure;
                break;
            }
        }

        // c <- c - step
        mpf_sub(c_coord.re, c_coord.re, step_coord.re);
        mpf_sub(c_coord.im, c_coord.im, step_coord.im);
        updateCandidateDistance();

        // ------------------------------------------------------------
        // Imagina error estimate (HDRFloat)
        //   err = |step|^4 * |d2|^2 / |dzdc|^2
        // Compute BEFORE checkpoint so diagnostics are populated.
        // ------------------------------------------------------------
        mpf_complex_norm(normStep, step_coord, t1_c, t2_c);

        normStep_hdr = HDRFloat<double>(normStep);
        HdrReduce(normStep_hdr);

        normStep2_hdr = normStep_hdr.square(); // |step|^4
        HdrReduce(normStep2_hdr);

        err_hdr = (normStep2_hdr * d2Norm_hdr) / dzdcNorm_hdr;
        HdrReduce(err_hdr);

        // Update convergence diagnostics.
        diagState.rho2 = rho2_hdr;
        diagState.err = err_hdr;
        diagState.step_norm = normStep_hdr;
        diagState.wantHalley = wantHalley;
        diagState.valid = true;

        // Inner loop completed: z_mag2 from final z, inner_pct = 100%.
        {
            const double zr = mpf_get_d(z_coord.re);
            const double zi = mpf_get_d(z_coord.im);
            diagState.z_mag2 = zr * zr + zi * zi;
        }
        diagState.inner_pct = 100.0;

        const int rho2Exp = (int)rho2_hdr.getExp();
        diagState.normalized_bits = (rho2Exp < 0) ? (-rho2Exp / 2) : 0;

        if (diagState.normalized_bits > 0 && targetExp > 0) {
            // Estimate remaining iterations via doubling model (quadratic convergence).
            const int targetBits = targetExp / 4;
            int bits = diagState.normalized_bits;
            int remaining = 0;
            while (bits < targetBits && remaining < 100) {
                bits *= 2;
                ++remaining;
            }
            diagState.est_remaining = remaining;
        } else {
            diagState.est_remaining = -1;
        }

        // Write checkpoint after each step (with diagnostics).
        checkpointWriter.TriggerWrite(
            std::make_unique<CheckpointSnapshot>(NRCheckpointParams{c_coord.re,
                                                                    c_coord.im,
                                                                    c0_coord.re,
                                                                    c0_coord.im,
                                                                    sqrRadius_coord,
                                                                    intrinsicRadius_mpf,
                                                                    period,
                                                                    coord_prec,
                                                                    it,
                                                                    NRCheckpointPhase::Main,
                                                                    scaleExp2_for_deriv_choice,
                                                                    numIterationsAtFind,
                                                                    0,
                                                                    z_coord.re,
                                                                    z_coord.im,
                                                                    dzdc_deriv.re,
                                                                    dzdc_deriv.im,
                                                                    deriv_prec,
                                                                    d2r_hdr,
                                                                    d2i_hdr,
                                                                    diagState},
                                                 0));

        // Print diagnostics between outer iterations.
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "  NR step " << it << " diag: rho2_exp2=" << rho2_hdr.getExp()
            << " rho2_mantissa=" << rho2_hdr.getMantissa() << ", err_exp2=" << err_hdr.getExp()
            << " err_mantissa=" << err_hdr.getMantissa() << ", step_norm_exp2=" << normStep_hdr.getExp()
            << " step_norm_mantissa=" << normStep_hdr.getMantissa() << ", z_mag2=" << diagState.z_mag2
            << ", |c-cand|2_exp2=" << diagState.c_cand_dist2.getExp()
            << " |c-cand|2_mantissa=" << diagState.c_cand_dist2.getMantissa()
            << ", bits=" << diagState.normalized_bits << ", est_remaining=" << diagState.est_remaining
            << (wantHalley ? " (Halley)" : " (Newton)");

        const int e = (int)err_hdr.getExp();
        if (-e >= targetExp) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "RefinePeriodicPoint: stop with err_hdr=" << err_hdr.ToString<false>()
                << " (err_exp2=" << e << " >= targetExp2=" << targetExp << ")";
            break;
        }

        // Check for user abort (Ctrl held 3s sets flag, Escape cancels it).
        // The latest checkpoint write has already been requested when enabled.
        if (AbortMonitor::GetStopCalculatingGlobal()) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "RefinePeriodicPoint: aborted at iter " << it;
            polishStatus = NRPolishStatus::Cancelled;
            break;
        }
    }

    // Skip final correction + accept/reject if aborted or already completed.
    if (AbortMonitor::GetStopCalculatingGlobal()) {
        polishStatus = NRPolishStatus::Cancelled;
    }
    if (!checkpointComplete && polishStatus == NRPolishStatus::Accepted) {
        // ---------------- Imagina final correction pass ----------------
        // Keep this Newton-only (matches Imagina + avoids Halley denom corner cases).
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "RefinePeriodicPoint: starting final correction phase at innerIter "
            << finalInnerStartIter;

        NRCheckpointParams finalParams{c_coord.re,
                                       c_coord.im,
                                       c0_coord.re,
                                       c0_coord.im,
                                       sqrRadius_coord,
                                       intrinsicRadius_mpf,
                                       period,
                                       coord_prec,
                                       it,
                                       NRCheckpointPhase::Final,
                                       scaleExp2_for_deriv_choice,
                                       numIterationsAtFind,
                                       finalInnerStartIter,
                                       z_coord.re,
                                       z_coord.im,
                                       dzdc_deriv.re,
                                       dzdc_deriv.im,
                                       deriv_prec,
                                       d2r_hdr,
                                       d2i_hdr,
                                       diagState};

        if (!resumeFinalPass) {
            checkpointWriter.WriteAndWait(finalParams);
        }

        ProgressContext finalProgressCtx{&finalParams, &checkpointWriter, &diagState};

        const uint64_t finalCompleted = EvaluateCriticalOrbitAndDerivs(backend,
                                                                       c_coord,
                                                                       period,
                                                                       z_coord,
                                                                       dzdc_deriv,
                                                                       d2r_hdr,
                                                                       d2i_hdr,
                                                                       coord_prec,
                                                                       deriv_prec,
                                                                       finalInnerStartIter,
                                                                       onProgress,
                                                                       &finalProgressCtx);

        if (finalCompleted < period || AbortMonitor::GetStopCalculatingGlobal()) {
            const double zr = mpf_get_d(z_coord.re);
            const double zi = mpf_get_d(z_coord.im);
            diagState.z_mag2 = zr * zr + zi * zi;
            diagState.inner_pct =
                (period > 0) ? static_cast<double>(finalCompleted) / static_cast<double>(period) * 100.0
                             : 0.0;
            updateCandidateDistance();

            checkpointWriter.WriteAndWait({c_coord.re,
                                           c_coord.im,
                                           c0_coord.re,
                                           c0_coord.im,
                                           sqrRadius_coord,
                                           intrinsicRadius_mpf,
                                           period,
                                           coord_prec,
                                           it,
                                           NRCheckpointPhase::Final,
                                           scaleExp2_for_deriv_choice,
                                           numIterationsAtFind,
                                           finalCompleted,
                                           z_coord.re,
                                           z_coord.im,
                                           dzdc_deriv.re,
                                           dzdc_deriv.im,
                                           deriv_prec,
                                           d2r_hdr,
                                           d2i_hdr,
                                           diagState});
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "RefinePeriodicPoint: aborted during final correction at innerIter "
                << finalCompleted;
            polishStatus = AbortMonitor::GetStopCalculatingGlobal() ? NRPolishStatus::Cancelled
                                                                    : NRPolishStatus::NumericalFailure;
        } else {
            if (!ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
                polishStatus = NRPolishStatus::NumericalFailure;
            } else {
                mpf_sub(c_coord.re, c_coord.re, step_coord.re);
                mpf_sub(c_coord.im, c_coord.im, step_coord.im);
            }

            // ---------------- Imagina accept/reject: stay within radius ----------------
            mpf_complex_sub(dc, c_coord, c0_coord);
            mpf_complex_norm(abs2_c, dc, t1_c, t2_c);

            if (polishStatus == NRPolishStatus::Accepted && mpf_cmp(abs2_c, sqrRadius_coord) > 0) {
                mpf_set(c_coord.re, c0_coord.re);
                mpf_set(c_coord.im, c0_coord.im);
                polishStatus = NRPolishStatus::Rejected;
            }
            updateCandidateDistance();

            if (polishStatus == NRPolishStatus::Accepted) {
                checkpointWriter.TriggerWrite(
                    std::make_unique<CheckpointSnapshot>(NRCheckpointParams{c_coord.re,
                                                                            c_coord.im,
                                                                            c0_coord.re,
                                                                            c0_coord.im,
                                                                            sqrRadius_coord,
                                                                            intrinsicRadius_mpf,
                                                                            period,
                                                                            coord_prec,
                                                                            it,
                                                                            NRCheckpointPhase::Complete,
                                                                            scaleExp2_for_deriv_choice,
                                                                            numIterationsAtFind,
                                                                            0,
                                                                            z_coord.re,
                                                                            z_coord.im,
                                                                            dzdc_deriv.re,
                                                                            dzdc_deriv.im,
                                                                            deriv_prec,
                                                                            d2r_hdr,
                                                                            d2i_hdr,
                                                                            diagState},
                                                         0));
            }
        }
    }

    // ---------------- cleanup ----------------
    mpf_clear(denom_c);
    mpf_clear(tr_c);
    mpf_clear(ti_c);
    mpf_clear(t1_c);
    mpf_clear(t2_c);
    mpf_clear(abs2_c);
    mpf_clear(normStep);
    mpf_clear(zNormSq_mpf);

    mpf_clear(tr_d);
    mpf_clear(ti_d);
    mpf_clear(t1_d);
    mpf_clear(t2_d);

    mpf_complex_clear(z_coord);
    mpf_complex_clear(step_coord);
    mpf_complex_clear(dzdc_coord);
    mpf_complex_clear(tmpZ_coord);

    mpf_complex_clear(dzdc_deriv);
    mpf_complex_clear(z_deriv);
    mpf_complex_clear(tmpB_d);

    mpf_complex_clear(d2_coord_scratch);
    mpf_complex_clear(htmp1);
    mpf_complex_clear(htmp2);

    mpf_complex_clear(dc);

    return {polishStatus, it};
}

// ------------------------------------------------------------
// Imagina-style MPF polish wrapper.
//
// - Builds MPF c from HighPrecision (mpf backend)
// - Preserves initial seed c0 for the final "stay within radius" reject
// - Chooses derivative precision in the Imagina spirit (scale-driven)
// - Runs Imagina-style mixed-precision NR polish
// - Writes back to HighPrecision
//
// Returns the outcome and number of NR iterations performed.
// ------------------------------------------------------------
template <class IterType, class T, PerturbExtras PExtras>
NRPolishResult
FeatureFinder<IterType, T, PExtras>::RefinePeriodicPoint_WithMPF(
    HighPrecision &cXHp,
    HighPrecision &cYHp,
    IterType period,
    mp_bitcnt_t coordPrec,
    const HighPrecision &sqrRadiusHp,
    int scaleExp2ForDeriv,
    NRInnerLoopBackend backend,
    const HighPrecision &intrinsicRadius,
    uint64_t numIterationsAtFind,
    NRCheckpointSavePolicy checkpointSavePolicy) const
{
    ValidateExistingNRCheckpointPhase();

    // ---- Convert inputs to MPF at coordPrec ----
    mpf_complex c;
    mpf_complex_init(c, coordPrec);

    mpf_complex c0;
    mpf_complex_init(c0, coordPrec);

    // Seed from HighPrecision backends
    // NOTE: HighPrecision::backend() is expected to be an mpf_t-compatible pointer.
    mpf_set(c.re, (mpf_srcptr)cXHp.backend());
    mpf_set(c.im, (mpf_srcptr)cYHp.backend());

    // Keep initial seed (Imagina reject check compares final c vs initial)
    mpf_set(c0.re, c.re);
    mpf_set(c0.im, c.im);

    mpf_t sqrRadiusMpf;
    mpf_init2(sqrRadiusMpf, coordPrec);
    mpf_set(sqrRadiusMpf, sqrRadiusHp.backend());

    // Convert intrinsicRadius to mpf for checkpoint persistence
    mpf_t intrinsicRadiusMpf;
    mpf_init2(intrinsicRadiusMpf, coordPrec);
    mpf_set(intrinsicRadiusMpf, (mpf_srcptr)intrinsicRadius.backend());

    // ---- Run Imagina-style polish ----
    const uint32_t maxPolish = 32;

    const NRPolishResult result = RefinePeriodicPoint<IterType, T>(c,
                                                                   c0,
                                                                   sqrRadiusMpf,
                                                                   (uint64_t)period,
                                                                   coordPrec,
                                                                   scaleExp2ForDeriv,
                                                                   maxPolish,
                                                                   backend,
                                                                   intrinsicRadiusMpf,
                                                                   numIterationsAtFind,
                                                                   checkpointSavePolicy);

    if (result.status == NRPolishStatus::Accepted) {
        cXHp = HighPrecision{c.re};
        cYHp = HighPrecision{c.im};
    }

    // ---- Cleanup ----
    mpf_clear(intrinsicRadiusMpf);
    mpf_clear(sqrRadiusMpf);
    mpf_complex_clear(c0);
    mpf_complex_clear(c);

    return result;
}

// Periodicity state for periodic-point detection
// All quantities are *squared* magnitudes.
//
// Imagina meanings:
//   Magnitude = norm(z)
//   norm(dzdc) = norm(dzdc)
//   SqrNearLinearRadius starts at R^2 and tightens over time.
//   SqrNearLinearRadiusScale = (0.25)^2
template <class IterType, class T, class C> struct PeriodicityPP {
    T SqrNearLinearRadius{};      // dynamic tightened R^2
    T SqrNearLinearRadiusScale{}; // (0.25)^2

    CUDA_CRAP void
    Init(T R)
    {
        const T zero = T{};
        const T near1 = HdrReduce(T{0.25});

        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero)) {
            // caller handles failure
            SqrNearLinearRadius = T{};
            SqrNearLinearRadiusScale = T{};
            return;
        }

        SqrNearLinearRadius = R * R;
        HdrReduce(SqrNearLinearRadius);

        SqrNearLinearRadiusScale = near1 * near1;
        HdrReduce(SqrNearLinearRadiusScale);
    }

    // Returns true if a period is detected at "Iteration".
    // Updates tightening every call.
    CUDA_CRAP bool
    CheckPeriodicity(const T &Magnitude,
                     const T &DzdcNormSq,
                     IterType Iteration,
                     IterType &outPrePeriod,
                     IterType &outPeriod)
    {
        // Trigger: Magnitude < SqrNearLinearRadius * norm(dzdc)
        T rhs = SqrNearLinearRadius * DzdcNormSq;
        HdrReduce(rhs);

        if (HdrCompareToBothPositiveReducedLT(Magnitude, rhs)) {
            outPrePeriod = 0;
            outPeriod = Iteration;
            return true;
        }

        // Tighten: if Magnitude * scale < SqrNearLinearRadius * norm(dzdc)
        // then SqrNearLinearRadius = Magnitude * scale / norm(dzdc)
        const T zero = T{};
        if (HdrCompareToBothPositiveReducedGT(DzdcNormSq, zero)) {
            T lhsTight = Magnitude * SqrNearLinearRadiusScale;
            HdrReduce(lhsTight);

            if (HdrCompareToBothPositiveReducedLT(lhsTight, rhs)) {
                T newSqr = lhsTight / DzdcNormSq;
                HdrReduce(newSqr);

                if (HdrCompareToBothPositiveReducedGT(newSqr, zero)) {
                    SqrNearLinearRadius = newSqr;
                }
            }
        }

        return false;
    }
};

template <class IterType, class T, PerturbExtras PExtras>
struct FeatureFinder<IterType, T, PExtras>::PeriodSearchState {
    PeriodSearchState()
    {
        dz.Reduce();
        z.Reduce();
        dzdc.Reduce();
        zcoeff.Reduce();
    }

    IterTypeFull iteration{};
    IterTypeFull refIteration{};
    C dz{};
    C z{};
    C dzdc{};
    C zcoeff{};
    PeriodicityPP<IterType, T, C> periodicity{};
    bool zcoeffValid{true};
};

template <class IterType, class T, PerturbExtras PExtras>
T
FeatureFinder<IterType, T, PExtras>::ChebAbs(const C &a) const
{
    return HdrMaxReduced(HdrAbs(a.getRe()), HdrAbs(a.getIm()));
}

template <class IterType, class T, PerturbExtras PExtras>
T
FeatureFinder<IterType, T, PExtras>::ToScalar(const HighPrecision &v)
{
    return T{v};
}

template <class IterType, class T, PerturbExtras PExtras>
typename FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::Div(const C &a, const C &b)
{
    // a / b = a * conj(b) / |b|^2
    // Always promote to HDRFloat<double> to avoid |b|^2 overflow.
    using H = HDRFloat<double>;
    const H br = PromoteToHdrD(b.getRe());
    const H bi = PromoteToHdrD(b.getIm());
    H denom = br * br + bi * bi;
    HdrReduce(denom);

    if (HdrCompareToBothPositiveReducedLE(denom, H{})) {
        return C{};
    }

    const H ar = PromoteToHdrD(a.getRe());
    const H ai = PromoteToHdrD(a.getIm());
    H rr = (ar * br + ai * bi) / denom;
    HdrReduce(rr);
    H ii = (ai * br - ar * bi) / denom;
    HdrReduce(ii);

    return C(DemoteToT(rr), DemoteToT(ii));
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_FindPeriod_Direct(const C &c,
                                                                IterTypeFull maxIters,
                                                                T R,
                                                                IterType &outPeriod,
                                                                C &outDiff,
                                                                C &outDzdc,
                                                                C &outZcoeff,
                                                                T &outResidual2) const
{
    // Require positive radius
    HdrReduce(R);
    const T zero = T{};
    if (HdrCompareToBothPositiveReducedLE(R, zero)) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "FeatureFinder::Evaluate_FindPeriod_Direct: R must be positive.";
        return false;
    }

    // Pre-reduce constants used in Reduced compares
    const T two = HdrReduce(T{2.0});
    const T one = HdrReduce(T{1.0});
    const T escape2 = HdrReduce(T{4096.0});

    // R2 (reduced)
    T R2 = R * R;
    HdrReduce(R2);

    C z{};      // z_0 = 0
    C dzdc{};   // dz/dc at z0 is 0
    C zcoeff{}; // matches your reference ordering (product-ish)

    for (IterTypeFull n = 0; n < maxIters; ++n) {
        // zcoeff ordering matches your double direct reference:
        // if (n==0) zcoeff = 1; else zcoeff *= (2*z)
        if (n == 0) {
            zcoeff = C(one, T{});
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        // dzdc <- 2*z*dzdc + 1
        dzdc = dzdc * (z * two) + C(one, T{});
        dzdc.Reduce();

        // Advance orbit: z <- z^2 + c
        z = (z * z) + c;
        z.Reduce();

        // Escape check on |z|^2
        T z2 = z.norm_squared();
        HdrReduce(z2);
        if (HdrCompareToBothPositiveReducedGT(z2, escape2)) {
            // Non-periodic / escaped before finding a candidate period.
            break;
        }

        // Period trigger:
        // if |z|^2 < R^2 * |dzdc|^2  => candidate period = n+1
        T d2 = dzdc.norm_squared();
        HdrReduce(d2);

        T rhs = R2 * d2;
        HdrReduce(rhs);

        if (HdrCompareToBothPositiveReducedLT(z2, rhs)) {
            const IterTypeFull cand = n + 1;
            if (cand <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                outPeriod = static_cast<IterType>(cand);
                outDiff = z;
                outDzdc = dzdc;
                outZcoeff = zcoeff;
                outResidual2 = z2;
                return true;
            }

            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "FeatureFinder::Evaluate_FindPeriod_Direct: candidate period exceeds IterType.";
            return false;
        }
    }

    return false;
}
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PeriodResidualAndDzdc_Direct(
    const C &c, IterType period, C &outDiff, C &outDzdc, C &outZcoeff, T &outResidual2) const
{
    C z{};
    C dzdc{};

    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    C oneC{one, T{}};
    C zcoeff{}; // will be set properly below

    oneC.Reduce();

    for (IterType i = 0; i < period; ++i) {
        // IMPORTANT: match Evaluate_FindPeriod_Direct ordering:
        // zcoeff = 1 at i==0, else zcoeff *= (2*z) using CURRENT z before update
        if (i == 0) {
            zcoeff = C(one, T{});
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        dzdc = dzdc * (z * two) + oneC;
        dzdc.Reduce();

        z = (z * z) + c;
        z.Reduce();

        T normSq = z.norm_squared();
        HdrReduce(normSq);

        if (HdrCompareToBothPositiveReducedGT(normSq, escape2)) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: orbit escaped.";
            return false;
        }
    }

    outDiff = z;
    outResidual2 = outDiff.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc;
    outZcoeff = zcoeff;

    return true;
}

template <class IterType, class T, PerturbExtras PExtras>
HighPrecision
FeatureFinder<IterType, T, PExtras>::ComputeIntrinsicRadius_HP(const C &zcoeff, const C &dzdc) const
{
    // Imagina: Scale = 1 / |zcoeff * dzdc|; radius = Scale * 4
    // Always promote to HDRFloat<double> to avoid overflow in the product.
    using H = HDRFloat<double>;
    const H zr = PromoteToHdrD(zcoeff.getRe());
    const H zi = PromoteToHdrD(zcoeff.getIm());
    const H dr = PromoteToHdrD(dzdc.getRe());
    const H di = PromoteToHdrD(dzdc.getIm());

    H wr = zr * dr - zi * di;
    HdrReduce(wr);
    H wi = zr * di + zi * dr;
    HdrReduce(wi);

    H w2 = wr * wr + wi * wi;
    HdrReduce(w2);

    if (HdrCompareToBothPositiveReducedLE(w2, H{})) {
        return HighPrecision{0};
    }

    H absW = HdrSqrt(w2);
    HdrReduce(absW);

    H radius = H{4.0} / absW;
    HdrReduce(radius);

    return HighPrecision{radius};
}

// Build a complex scalar (s + 0i) reduced.
template <class IterType, class T, PerturbExtras PExtras>
static inline typename FeatureFinder<IterType, T, PExtras>::C
MakeRealC(const T &s)
{
    using C = typename FeatureFinder<IterType, T, PExtras>::C;
    C out(s, T{});
    out.Reduce();
    return out;
}

template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_PT(
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const HighPrecision &cXHp,
    const HighPrecision &cYHp,
    T R,
    IterTypeFull maxIters,
    PeriodSearchState *resumeState,
    IterType &ioPeriod,
    C &outDiff,
    C &outDzdc,
    C &outZcoeff,
    T &outResidual2) const
{
    const T zero = T{};
    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    // dc = current c - reference center (HP -> T)
    const HighPrecision dcXHp = cXHp - results.GetHiX();
    const HighPrecision dcYHp = cYHp - results.GetHiY();
    C dc(ToScalar(dcXHp), ToScalar(dcYHp));
    dc.Reduce();

    const size_t refOrbitLength = (size_t)results.GetCountOrbitEntries();
    if (refOrbitLength < 2)
        return false;

    const IterTypeFull cap = FindPeriod ? maxIters : (IterTypeFull)ioPeriod;
    if (cap < 1)
        return false;

    int scaleExp = 0;
    const T ScalingFactor = HdrLdexp(one, -scaleExp);   // 2^-scaleExp
    const T InvScalingFactor = HdrLdexp(one, scaleExp); // 2^scaleExp

    const C ScalingFactorC(ScalingFactor, T{});
    const C InvScalingFactorC(InvScalingFactor, T{});

    // Precompute InvScalingFactor^2 for periodicity math (true dzdc norm)
    T InvScale2 = InvScalingFactor * InvScalingFactor;
    HdrReduce(InvScale2);

    PeriodSearchState freshState{};
    PeriodSearchState &state = resumeState != nullptr ? *resumeState : freshState;
    IterType prePeriod = 0;
    IterType period = 0;

    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero))
            return false;
        if (resumeState == nullptr) {
            state.periodicity.Init(R);
        }
    }

    if (state.iteration > cap || state.refIteration >= refOrbitLength) {
        return false;
    }

    for (IterTypeFull n = state.iteration; n < cap; ++n) {
        // A completed LA step may end at the final reference entry. Rebase
        // before reading the next pair of reference values.
        if (state.refIteration >= refOrbitLength - 1) {
            state.dz = state.z;
            state.dz.Reduce();
            state.refIteration = 0;
        }

        // if Iteration==0 zcoeff = ScalingFactor else zcoeff *= 2*z
        if (state.zcoeffValid) {
            if (n == 0) {
                state.zcoeff = ScalingFactorC;
            } else {
                state.zcoeff = state.zcoeff * (state.z * two);
            }
            state.zcoeff.Reduce();
        }

        // scaled dzdc: dzdc = dzdc*(2z) + ScalingFactor
        state.dzdc = state.dzdc * (state.z * two) + ScalingFactorC;
        state.dzdc.Reduce();

        // PT delta recurrence
        const C zref = results.GetComplex(dec, static_cast<size_t>(state.refIteration));
        state.dz = state.dz * (zref + state.z) + dc;
        state.dz.Reduce();

        state.refIteration++;
        const C zrefNext = results.GetComplex(dec, static_cast<size_t>(state.refIteration));
        state.z = zrefNext + state.dz;
        state.z.Reduce();
        state.iteration = n + 1;

        // Rebasing check
        T dzNorm = state.dz.norm_squared();
        HdrReduce(dzNorm);
        T zNorm = state.z.norm_squared();
        HdrReduce(zNorm);

        if (state.refIteration >= refOrbitLength - 1 ||
            HdrCompareToBothPositiveReducedLT(zNorm, dzNorm)) {
            state.dz = state.z;
            state.dz.Reduce();
            state.refIteration = 0;
        }

        // Escape check
        if (HdrCompareToBothPositiveReducedGT(zNorm, escape2)) {
            return false;
        }

        if constexpr (FindPeriod) {
            // Imagina uses:
            //   Magnitude = norm(z)
            //   norm(dzdc) in trigger is true dzdc norm
            //
            // dzdc is stored scaled, so convert norm to true:
            T dzdcNormStored = state.dzdc.norm_squared();
            HdrReduce(dzdcNormStored);

            T dzdcNormTrue = dzdcNormStored * InvScale2;
            HdrReduce(dzdcNormTrue);

            // Iteration increments then checks; your n is 0-based
            // At end of loop, we've computed z_{n+1}, dzdc at same time.
            if (state.iteration > static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                return false;
            }
            const IterType iteration = static_cast<IterType>(state.iteration);

            if (state.periodicity.CheckPeriodicity(zNorm, dzdcNormTrue, iteration, prePeriod, period)) {
                ioPeriod = period; // prePeriod is always 0 here

                // Unscale outputs (to match your direct conventions)
                outDiff = state.z;
                outDzdc = state.dzdc * InvScalingFactorC;
                outZcoeff = state.zcoeffValid ? state.zcoeff * InvScalingFactorC : ScalingFactorC;
                outDiff.Reduce();
                outDzdc.Reduce();
                outZcoeff.Reduce();

                outResidual2 = zNorm;
                return true;
            }
        }
    }

    if constexpr (FindPeriod) {
        return false;
    } else {
        // Fixed-period path: unscale outputs.
        outDiff = state.z;
        outResidual2 = state.z.norm_squared();
        HdrReduce(outResidual2);

        outDzdc = state.dzdc * InvScalingFactorC;
        outZcoeff = state.zcoeff * InvScalingFactorC;
        outDiff.Reduce();
        outDzdc.Reduce();
        outZcoeff.Reduce();
        return true;
    }
}

// =====================================================================================
// LA Evaluation (rewritten):
//
// Goals:
//  - Use LA strictly as an *accelerator* to advance z and dzdc safely (with ScalingFactor).
//  - For FindPeriod, use the SAME period trigger as PT/direct:
//        |z|^2 < R^2 * |dzdc_true|^2   (via PeriodicityPP tightening)
//    (NOT LAParameters::DetectPeriod — that is a different detector).
//  - NEVER attempt to synthesize DIRECT/PT "zcoeff" from LA internals.
//    On success, we output:
//      outDiff   = z  (same as PT/direct convention)
//      outDzdc   = dzdc_true (unscaled)
//      outZcoeff = 1 (dummy; caller should *not* use LA zcoeff for intrinsic radius)
//    and callers that need canonical (diff,dzdc,zcoeff) for precision/radius should
//    re-evaluate with PT or DIRECT at the final c (recommended).
//
// Notes / assumptions (based on your types):
//  - LAstep::Evaluate(dc) advances the prepared delta to the next macro step.
//  - LAstep::getZ(dz) returns absolute z = Refp1Deep + dz for that macro step.
//  - LAstep::EvaluateDzdcDeep(dz, dzdc, ScalingFactor) updates *stored scaled* dzdc:
//        dzdc_stored = dzdc_true * ScalingFactor
//  - ScalingFactor is a scalar Float (HDRFloat or float/double) and is applied exactly
//    like PT path uses ScalingFactor to keep derivatives stable.
//
//  - We keep dzdc and (optionally) zcoeff in the SAME scaling contract as PT:
//        dzdc_stored = dzdc_true * ScalingFactor
//    and we unscale for periodicity by multiplying norm^2 by InvScale2.
//
//  - Rebase rule matches PT:
//        if refIteration hits MacroItCount OR |dz| > |z| then
//            dz = z; refIteration = 0;
//
//  - We do NOT attempt to renormalize scaleExp adaptively here. In your current setup,
//    scaleExp==0 is fine because HDRFloatComplex already reduces and you are not pushing
//    mpf-level magnitudes through LA. If you later need renorm, it can be added exactly
//    like PT's stored-derivative renorm scheme.
// =====================================================================================
template <class IterType, class T, PerturbExtras PExtras>
typename FeatureFinder<IterType, T, PExtras>::LASearchResult
FeatureFinder<IterType, T, PExtras>::Evaluate_LA(
    const PerturbationResults<IterType, T, PExtras> &results,
    LAReference<IterType, T, SubType, PExtras> &laRef,
    const HighPrecision &cXHp,
    const HighPrecision &cYHp,
    T R,
    IterTypeFull maxIters,
    PeriodSearchState &state,
    IterType &ioPeriod,
    C &outDiff,
    C &outDzdc,
    C &outZcoeff,
    T &outResidual2) const
{
    if (!laRef.IsValid()) {
        return LASearchResult::RetryPT;
    }

    const T zero = T{};
    const T one = HdrReduce(T{1.0});
    const T escape2 = HdrReduce(T{4096.0});

    // --------------------------
    // Periodicity state (PT/direct semantics)
    // --------------------------
    IterType prePeriod = 0;
    IterType period = 0;

    HdrReduce(R);
    if (HdrCompareToBothPositiveReducedLE(R, zero)) {
        return LASearchResult::RetryPT;
    }
    state.periodicity.Init(R);
    state.zcoeffValid = false;

    // --------------------------
    // dc = c - referenceCenter
    // --------------------------
    const HighPrecision dcXHp = cXHp - results.GetHiX();
    const HighPrecision dcYHp = cYHp - results.GetHiY();
    C dc(ToScalar(dcXHp), ToScalar(dcYHp));
    dc.Reduce();

    // --------------------------
    // Cap
    // --------------------------
    const IterTypeFull cap = maxIters;
    if (cap < 1) {
        return LASearchResult::RetryPT;
    }

    // =========================================================================
    // Scaling contract (same as PT path)
    //
    // stored = true * ScalingFactor
    // Here we keep scaleExp = 0; ScalingFactor = 1.
    // =========================================================================
    int scaleExp = 0;
    const T ScalingFactor = HdrReduce(HdrLdexp(one, -scaleExp));   // 2^-scaleExp
    const T InvScalingFactor = HdrReduce(HdrLdexp(one, scaleExp)); // 2^scaleExp
    const C InvScalingFactorC(InvScalingFactor, T{});
    T InvScale2 = InvScalingFactor * InvScalingFactor;
    HdrReduce(InvScale2);

    // =========================================================================
    // State
    // =========================================================================
    // Period discovery does not use zcoeff. Fixed-period PT evaluates it later.
    const C oneC(one, T{});

    const IterType laStageCount = laRef.GetLAStageCount();

    // The next-stage index is relative to the finer stage. Preserve it when
    // descending; the orbit and derivative state already represent "iteration".
    for (IterType currentLAStage = laStageCount; currentLAStage > 0 && state.iteration < cap;) {
        --currentLAStage;

        const IterType laIndex = laRef.getLAIndex(currentLAStage);
        const IterType macroItCount = laRef.getMacroItCount(currentLAStage);

        if (state.refIteration >= macroItCount) {
            return LASearchResult::RetryPT;
        }

        const IterType stageRefIteration = static_cast<IterType>(state.refIteration);
        if (laRef.isLAStageInvalid(laIndex + stageRefIteration, dc)) {
            const auto stageEntry = laRef.getLA(laIndex,
                                                state.dz,
                                                stageRefIteration,
                                                static_cast<IterType>(state.iteration),
                                                static_cast<IterType>(cap));
            state.refIteration = stageEntry.nextStageLAindex;
            continue;
        }

        while (state.iteration < cap) {

            // Get LA step descriptor for this block
            auto las = laRef.getLA(laIndex,
                                   state.dz,
                                   static_cast<IterType>(state.refIteration),
                                   static_cast<IterType>(state.iteration),
                                   static_cast<IterType>(cap));

            if (las.unusable) {
                state.refIteration = las.nextStageLAindex;
                break;
            }

            // ------------------------------------------------------------
            // Update dzdc (stored scaled) for this macro-step
            // MUST use the scaled API: EvaluateDzdcDeep(dz, dzdc, ScalingFactor)
            // ------------------------------------------------------------
            las.EvaluateDzdcDeep(state.dz, state.dzdc, ScalingFactor);
            state.dzdc.Reduce();

            // ------------------------------------------------------------
            // Advance dz for this macro-step
            // ------------------------------------------------------------
            state.dz = las.Evaluate(dc);
            state.dz.Reduce();

            // Advance iteration/refIteration
            state.iteration += las.step;
            state.refIteration++;

            // Absolute z at the end of the macro-step
            state.z = las.getZ(state.dz);
            state.z.Reduce();

            // Escape check (like PT/direct)
            T zNorm = state.z.norm_squared();
            HdrReduce(zNorm);
            if (HdrCompareToBothPositiveReducedGT(zNorm, escape2)) {
                return LASearchResult::RetryPT;
            }

            // Rebase rule (match PT)
            if (state.refIteration >= macroItCount ||
                HdrCompareToBothPositiveReducedGT(ChebAbs(state.dz), ChebAbs(state.z))) {
                state.dz = state.z;
                state.dz.Reduce();
                state.refIteration = 0;
            }

            // ------------------------------------------------------------
            // Period detection (PT/direct semantics)
            // ------------------------------------------------------------
            T dzdcNormStored = state.dzdc.norm_squared();
            HdrReduce(dzdcNormStored);

            T dzdcNormTrue = dzdcNormStored * InvScale2;
            HdrReduce(dzdcNormTrue);

            if (state.iteration > static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                return LASearchResult::RetryPT;
            }
            const IterType iteration = static_cast<IterType>(state.iteration);
            if (state.periodicity.CheckPeriodicity(zNorm, dzdcNormTrue, iteration, prePeriod, period)) {
                ioPeriod = period;
                outDiff = state.z;
                outDzdc = state.dzdc * InvScalingFactorC;
                outDzdc.Reduce();
                outZcoeff = oneC;
                outZcoeff.Reduce();
                outResidual2 = zNorm;
                return LASearchResult::Found;
            }
        }
    }

    if (state.iteration > 0 && state.iteration < cap &&
        state.refIteration < results.GetCountOrbitEntries()) {
        return LASearchResult::ContinuePT;
    }
    return LASearchResult::RetryPT;
}

// DirectEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::DirectEvaluator::Eval(const C &c,
                                                           [[maybe_unused]] const HighPrecision &cX_hp,
                                                           [[maybe_unused]] const HighPrecision &cY_hp,
                                                           T SqrRadius,
                                                           IterTypeFull maxIters,
                                                           IterType &ioPeriod,
                                                           C &outDiff,
                                                           C &outDzdc,
                                                           C &outZcoeff,
                                                           T &outResidual2) const
{
    T R = HdrSqrt(SqrRadius);
    if constexpr (FindPeriod) {
        return self->Evaluate_FindPeriod_Direct(
            c, maxIters, R, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    } else {
        return self->Evaluate_PeriodResidualAndDzdc_Direct(
            c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
    }
}

// Add after the PTEvaluator::Eval implementation (around line 542)

// LAEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::LAEvaluator::Eval([[maybe_unused]] const C &c,
                                                       const HighPrecision &cXHp,
                                                       const HighPrecision &cYHp,
                                                       T sqrRadius,
                                                       IterTypeFull maxIters,
                                                       IterType &ioPeriod,
                                                       C &outDiff,
                                                       C &outDzdc,
                                                       C &outZcoeff,
                                                       T &outResidual2) const
{
    T radius = HdrSqrt(sqrRadius);
    if constexpr (FindPeriod) {
        PeriodSearchState state{};
        const LASearchResult laResult = self->Evaluate_LA(*results,
                                                          *laRef,
                                                          cXHp,
                                                          cYHp,
                                                          radius,
                                                          maxIters,
                                                          state,
                                                          ioPeriod,
                                                          outDiff,
                                                          outDzdc,
                                                          outZcoeff,
                                                          outResidual2);
        if (laResult == LASearchResult::Found) {
            return true;
        }
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Temporary LA status: result=" << static_cast<int>(laResult)
            << " iteration=" << state.iteration << " reference=" << state.refIteration;
        // Keep the fresh PT retry for invalid or uncertain LA state.
        PeriodSearchState *resumeState = laResult == LASearchResult::ContinuePT ? &state : nullptr;
        if (resumeState != nullptr) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Temporary LA-to-PT probe: iteration=" << state.iteration
                << " reference=" << state.refIteration;
        }
        return self->template Evaluate_PT<true>(*results,
                                                *dec,
                                                cXHp,
                                                cYHp,
                                                radius,
                                                maxIters,
                                                resumeState,
                                                ioPeriod,
                                                outDiff,
                                                outDzdc,
                                                outZcoeff,
                                                outResidual2);
    } else {
        return self->template Evaluate_PT<false>(*results,
                                                 *dec,
                                                 cXHp,
                                                 cYHp,
                                                 radius,
                                                 static_cast<IterTypeFull>(ioPeriod),
                                                 nullptr,
                                                 ioPeriod,
                                                 outDiff,
                                                 outDzdc,
                                                 outZcoeff,
                                                 outResidual2);
    }
}

// PTEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::PTEvaluator::Eval([[maybe_unused]] const C &c,
                                                       const HighPrecision &cXHp,
                                                       const HighPrecision &cYHp,
                                                       T SqrRadius,
                                                       IterTypeFull maxIters,
                                                       IterType &ioPeriod,
                                                       C &outDiff,
                                                       C &outDzdc,
                                                       C &outZcoeff,
                                                       T &outResidual2) const
{
    T R = HdrSqrt(SqrRadius);
    if (self->template Evaluate_PT<FindPeriod>(*results,
                                               *dec,
                                               cXHp,
                                               cYHp,
                                               R,
                                               maxIters,
                                               nullptr,
                                               ioPeriod,
                                               outDiff,
                                               outDzdc,
                                               outZcoeff,
                                               outResidual2)) {
        return true;
    }

    return false;
}

// Simplified FindPeriodicPoint (Direct)
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(IterType maxIters, FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, DirectEvaluator{this});
}

// Simplified FindPeriodicPoint (PT)
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(
    IterType maxIters,
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, PTEvaluator{this, &results, &dec});
}

// FindPeriodicPoint with LA support
template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint(
    IterType maxIters,
    const PerturbationResults<IterType, T, PExtras> &results,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    LAReference<IterType, T, SubType, PExtras> &laRef,
    FeatureSummary &feature) const
{
    return FindPeriodicPoint_Common(maxIters, feature, LAEvaluator{this, &results, &dec, &laRef});
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::RefinePeriodicPoint_HighPrecision(
    FeatureSummary &feature,
    NRInnerLoopBackend backend,
    NRCheckpointSavePolicy checkpointSavePolicy) const
{
    auto *cand = feature.GetCandidate();
    if (!cand)
        return false;

    // Already refined (e.g. from a previous pass); no need to redo
    if (feature.IsRefined())
        return true;

    HighPrecision cXHp = cand->cX_hp;
    HighPrecision cYHp = cand->cY_hp;

    // period conversion
    IterType period{};
    if (cand->period > (IterType)std::numeric_limits<IterType>::max())
        return false;
    period = (IterType)cand->period;

    // Use the candidate's stored radius^2 so Phase B is independent of the FeatureSummary radius.
    const HighPrecision &sqrRadiusHp = cand->sqrRadius_hp;

    // MPF polish only
    const NRPolishResult polish = RefinePeriodicPoint_WithMPF(cXHp,
                                                              cYHp,
                                                              period,
                                                              cand->mpfPrecBits,
                                                              sqrRadiusHp,
                                                              cand->scaleExp2_for_mpf,
                                                              backend,
                                                              feature.GetIntrinsicRadius(),
                                                              feature.GetNumIterationsAtFind(),
                                                              checkpointSavePolicy);
    if (polish.status != NRPolishStatus::Accepted) {
        return false;
    }

    // Commit only the refined coordinates + period.
    // Preserve existing intrinsicRadius if present — the refinement
    // improves coordinates but doesn't recompute intrinsic radius.
    feature.SetFound(cXHp, cYHp, (IterType)period, feature.GetResidual2(), feature.GetIntrinsicRadius());
    feature.SetRefined();

    // Optionally keep candidate (or clear it)
    // feature.ClearCandidate();

    return true;
}

template <class IterType, class T, PerturbExtras PExtras>
template <class EvalPolicy>
bool
FeatureFinder<IterType, T, PExtras>::FindPeriodicPoint_Common(IterType refIters,
                                                              FeatureSummary &feature,
                                                              EvalPolicy &&evaluator) const
{
    const HighPrecision &origCXHp = feature.GetOrigX();
    const HighPrecision &origCYHp = feature.GetOrigY();

    // Search radius (T-space), but we keep c updates in HP and regenerate T c each time.
    T radius{feature.GetRadius()};
    radius = HdrAbs(radius);
    T sqrRadius = radius * radius;
    HdrReduce(sqrRadius);

    // Canonical parameter in HP (ONLY updated in HP to avoid drift)
    HighPrecision cXHp = origCXHp;
    HighPrecision cYHp = origCYHp;
    const HighPrecision sqrRadiusHp = feature.GetRadius() * feature.GetRadius();

    const auto makeOffset = [&]() -> C {
        if constexpr (std::is_same_v<std::decay_t<EvalPolicy>, DirectEvaluator>) {
            return C(ToScalar(cXHp), ToScalar(cYHp));
        } else {
            C offset(ToScalar(cXHp - evaluator.results->GetHiX()),
                     ToScalar(cYHp - evaluator.results->GetHiY()));
            offset.Reduce();
            return offset;
        }
    };

    const auto applyStep = [&](const C &step) {
        const auto applyComponent = [](HighPrecision &coordinate, const T &component) {
            const HighPrecision correction{component};
            if (correction == HighPrecision{}) {
                return;
            }
            double mantissa;
            long coordinateExp;
            long correctionExp;
            coordinate.frexp(mantissa, coordinateExp);
            correction.frexp(mantissa, correctionExp);
            const auto neededBits =
                static_cast<uint64_t>(std::max<long>(512, coordinateExp - correctionExp + 256));
            if (neededBits > coordinate.precisionInBits()) {
                HighPrecision widened{HighPrecision::SetPrecision::True, neededBits};
                mpf_set(widened.backend(), coordinate.backend());
                coordinate = std::move(widened);
            }
            coordinate -= correction;
        };
        applyComponent(cXHp, step.getRe());
        applyComponent(cYHp, step.getIm());
    };

    // T-space parameter used by evaluators (ALWAYS derived from HP)
    auto makeCFromHp = [&]() -> C {
        C out(ToScalar(cXHp), ToScalar(cYHp));
        out.Reduce();
        return out;
    };

    const C origC(ToScalar(origCXHp), ToScalar(origCYHp));
    C c = makeCFromHp();

    IterType period = 0;
    C diff{}, dzdc{}, zcoeff{};
    T residual2{};

    // -----------------------------------------------------------------------------
    // 1) Find candidate period
    // -----------------------------------------------------------------------------
    if (!evaluator.template Eval<true>(
            c, cXHp, cYHp, sqrRadius, refIters, period, diff, dzdc, zcoeff, residual2)) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: findPeriod failed";
        return false;
    }

    // -----------------------------------------------------------------------------
    // 2) Initial Newton correction: c <- c - diff/dzdc  (HP only)
    // -----------------------------------------------------------------------------
    {
        T dzdc2Initial = dzdc.norm_squared();
        HdrReduce(dzdc2Initial);
        if (HdrCompareToBothPositiveReducedLE(dzdc2Initial, T{})) {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: initial dzdc too small";
            return false;
        }

        C step0 = Div(diff, dzdc);
        step0.Reduce();

        applyStep(step0);

        c = makeCFromHp();
    }

    // -----------------------------------------------------------------------------
    // Tolerances
    // -----------------------------------------------------------------------------
    const T relTol = m_params.RelStepTol;
    T relTol2 = relTol * relTol;
    HdrReduce(relTol2);

    // -----------------------------------------------------------------------------
    // 3) Newton loop
    // -----------------------------------------------------------------------------
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        // Always evaluate at c derived from HP
        c = makeCFromHp();

        if (!evaluator.template Eval<false>(
                c, cXHp, cYHp, sqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: evalAtPeriod failed in loop";
            return false;
        }

        {
            T diff2 = diff.norm_squared();
            HdrReduce(diff2);

            T offset2 = makeOffset().norm_squared();
            HdrReduce(offset2);

            T rhs = offset2 * relTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(diff2, rhs)) {
                // Residual is small enough relative to current c => converged
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Iter1 " << it << ": diff^2=" << HdrToString<false>(diff2)
                    << ", |dc|^2*tol^2=" << HdrToString<false>(rhs);
                break;
            }
        }

        // dzdc must be non-degenerate
        T dzdc2 = dzdc.norm_squared();
        HdrReduce(dzdc2);
        if (HdrCompareToBothPositiveReducedLE(dzdc2, T{})) {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: dzdc too small";
            return false;
        }

        // Newton step in T-space
        C step = Div(diff, dzdc);
        step.Reduce();

        const C previousOffset = makeOffset();
        applyStep(step);

        // Rebuild T-space c from HP
        c = makeCFromHp();

        // Optional: keep your original step-based stop as a secondary criterion
        {
            T step2 = step.norm_squared();
            HdrReduce(step2);

            T offset2 = makeOffset().norm_squared();
            HdrReduce(offset2);

            T rhs = offset2 * relTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(step2, rhs)) {
                FractalSharkLog::LogLine(__FILE__, __LINE__)
                    << "Iter2 " << it << ": step^2=" << HdrToString<false>(step2)
                    << ", |dc|^2*relTol^2=" << HdrToString<false>(rhs);
                break;
            }
        }

        const C currentOffset = makeOffset();
        if (currentOffset.getRe() == previousOffset.getRe() &&
            currentOffset.getIm() == previousOffset.getIm()) {
            FractalSharkLog::LogLine(__FILE__, __LINE__)
                << "Search offset reached its mantissa limit; deferring to MPF refinement.";
            break;
        }
    }

    // -----------------------------------------------------------------------------
    // Final correction pass (same idea)
    // -----------------------------------------------------------------------------
    c = makeCFromHp();

    if (!evaluator.template Eval<false>(
            c, cXHp, cYHp, sqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: final eval failed";
        return false;
    }

    {
        T dzdc2Final = dzdc.norm_squared();
        HdrReduce(dzdc2Final);
        if (HdrCompareToBothPositiveReducedLE(dzdc2Final, T{})) {
            FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: final dzdc too small";
            return false;
        }

        C step = Div(diff, dzdc);
        step.Reduce();

        applyStep(step);

        c = makeCFromHp();
    }

    // Final eval for residual and scale
    if (!evaluator.template Eval<false>(
            c, cXHp, cYHp, sqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: final eval failed";
        return false;
    }

    const HighPrecision acceptedDx = cXHp - origCXHp;
    const HighPrecision acceptedDy = cYHp - origCYHp;
    if (acceptedDx * acceptedDx + acceptedDy * acceptedDy > sqrRadiusHp) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Rejected: Newton candidate exceeds search radius";
        return false;
    }

    {
        int scaleExp2 = 0;
        mp_bitcnt_t precisionBits = cXHp.precisionInBits();

        {
            // Always promote to HDRFloat<double> to avoid overflow in the product.
            using H = HDRFloat<double>;
            const H zr = PromoteToHdrD(zcoeff.getRe());
            const H zi = PromoteToHdrD(zcoeff.getIm());
            const H dr = PromoteToHdrD(dzdc.getRe());
            const H di = PromoteToHdrD(dzdc.getIm());

            H wr = zr * dr - zi * di;
            HdrReduce(wr);
            H wi = zr * di + zi * dr;
            HdrReduce(wi);

            H w2 = wr * wr + wi * wi;
            HdrReduce(w2);

            if (!HdrCompareToBothPositiveReducedGT(w2, H{})) {
                FractalSharkLog::LogLine(__FILE__, __LINE__) << "Rejected: zero w in candidate store";
                feature.ClearCandidate();
                return false;
            }

            H absW = HdrSqrt(w2);
            HdrReduce(absW);

            H scaleH = H{1.0} / absW;
            HdrReduce(scaleH);

            HighPrecision scaleHP{scaleH};
            long exponentLong;
            double mant;
            scaleHP.frexp(mant, exponentLong);
            scaleExp2 = (int)exponentLong;
        }

        {
            const int bitsFromScale = std::max(0, -scaleExp2);
            const int marginBits = 256;

            mp_bitcnt_t want = (mp_bitcnt_t)(bitsFromScale + marginBits);
            want = std::max(want, (mp_bitcnt_t)cXHp.precisionInBits());
            want = std::max<mp_bitcnt_t>(want, 512);
            precisionBits = want;
        }

        // Store candidate for Phase B refinement
        feature.SetCandidate(cXHp,
                             cYHp,
                             (IterTypeFull)period,
                             HDRFloat<double>{residual2},
                             sqrRadiusHp,
                             scaleExp2,
                             precisionBits);
    }

    const HighPrecision intrinsicRadius = ComputeIntrinsicRadius_HP(zcoeff, dzdc);

    feature.SetFound(cXHp, cYHp, period, HDRFloat<double>{residual2}, intrinsicRadius);

    if (m_params.PrintResult) {
        FractalSharkLog::LogLine(__FILE__, __LINE__)
            << "Periodic point:  orig cx=" << HdrToString<false, T>(origC.getRe())
            << " orig cy=" << HdrToString<false, T>(origC.getIm())
            << " new cx=" << HdrToString<false, T>(c.getRe())
            << " new cy=" << HdrToString<false, T>(c.getIm())
            << " period=" << static_cast<uint64_t>(period)
            << " residual2=" << HdrToString<false, T>(residual2)
            << " intrinsicRadius=" << intrinsicRadius.str();
    }

    return true;
}

// ------------------------------
// Explicit instantiations (minimal set you enabled)
// ------------------------------
#define InstantiatePeriodicPointFinder(IterTypeT, TT, PExtrasT)                                         \
    template class FeatureFinder<IterTypeT, TT, PExtrasT>;

//// ---- Disable ----
InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Disable);

InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Disable);

// Bare CudaDblflt needs CudaDblflt arithmetic fixes — disabled for now.
// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Disable);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Disable);

// HDRFloat<CudaDblflt> needs more HDRFloat utility functions (HdrAbs, HdrMaxReduced, etc.)
// to accept CudaDblflt — disabled for now.
// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
//
// ---- Bad ----
// InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Bad);

// InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Bad);

// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);

// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Bad);

// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Bad);

// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);

// ---- SimpleCompression ----
InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::SimpleCompression);

// Bare CudaDblflt needs CudaDblflt arithmetic fixes — disabled for now.
// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression);

// HDRFloat<CudaDblflt> needs more HDRFloat utility functions — disabled for now.
// InstantiatePeriodicPointFinder(uint32_t,
//                                HDRFloat<CudaDblflt<MattDblflt>>,
//                                PerturbExtras::SimpleCompression);
// InstantiatePeriodicPointFinder(uint64_t,
//                                HDRFloat<CudaDblflt<MattDblflt>>,
//                                PerturbExtras::SimpleCompression);

#undef InstantiatePeriodicPointFinder
