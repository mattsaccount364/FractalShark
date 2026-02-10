//
// This feature finder logic is heavily based on the implementation in Imagina
// but likely screws up some of the details.
//

#include "stdafx.h"

#include "Exceptions.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "FloatComplex.h"
#include "HighPrecision.h"
#include "LAInfoDeep.h"
#include "LAReference.h"
#include "PerturbationResults.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>

struct mpf_complex {
    mpf_t re, im;
};

static inline void
mpf_complex_init(mpf_complex &z, mp_bitcnt_t prec)
{
    mpf_init2(z.re, prec);
    mpf_init2(z.im, prec);
}

static inline void
mpf_complex_clear(mpf_complex &z)
{
    mpf_clear(z.re);
    mpf_clear(z.im);
}

static inline void
mpf_complex_set(mpf_complex &dst, const mpf_complex &src)
{
    mpf_set(dst.re, src.re);
    mpf_set(dst.im, src.im);
}

static inline void
mpf_complex_set_ui(mpf_complex &z, unsigned long re, unsigned long im)
{
    mpf_set_ui(z.re, re);
    mpf_set_ui(z.im, im);
}

static inline void
mpf_complex_sub(mpf_complex &out, const mpf_complex &a, const mpf_complex &b)
{
    mpf_sub(out.re, a.re, b.re);
    mpf_sub(out.im, a.im, b.im);
}

static inline void
mpf_complex_add(mpf_complex &out, const mpf_complex &a, const mpf_complex &b)
{
    mpf_add(out.re, a.re, b.re);
    mpf_add(out.im, a.im, b.im);
}

static inline mp_bitcnt_t
ChooseDerivPrec_ImaginaStyle(mp_bitcnt_t coord_prec,
                             int scaleExp2,         // exponent of Scale ≈ 1/|zcoeff*dzdc|
                             int coordExp2_max_abs, // approx max exponent of |c.re|,|c.im|
                             mp_bitcnt_t minPrec = 256)
{
    // Imagina-ish: DerivativePrecision ≈ (-ScaleExp + 32)/4   (in "bits")
    // Here scaleExp2 is exponent of Scale (base2). If Scale is tiny => negative => need more.
    // Note: If your scaleExp2 is computed via frexp, it already matches this spirit.
    long dp = (long)((-scaleExp2 + 32) / 4);
    if (dp < (long)minPrec)
        dp = (long)minPrec;

    // Cap like Imagina: coord_prec + coordExponent + 32
    long cap = (long)coord_prec + (long)coordExp2_max_abs + 32;
    if (dp > cap)
        dp = cap;

    if (dp > (long)coord_prec)
        dp = (long)coord_prec;
    return (mp_bitcnt_t)dp;
}

static inline void
mpf_complex_norm(mpf_t out, const mpf_complex &z, mpf_t t1, mpf_t t2)
{
    mpf_mul(t1, z.re, z.re);
    mpf_mul(t2, z.im, z.im);
    mpf_add(out, t1, t2);
}

// Return floor(log2(x)) approx using mpf_get_d_2exp (same as your helper)
static inline int
approx_ilogb_mpf(const mpf_t x)
{
    if (mpf_cmp_ui(x, 0) == 0)
        return INT32_MIN;
    long exp2;
    (void)mpf_get_d_2exp(&exp2, x); // mantissa unused
    // mpf_get_d_2exp gives x = mant * 2^exp2, mant in [0.5,1)
    return int(exp2 - 1);
}

static inline int
approx_ilogb_mpf_abs2(const mpf_t re, const mpf_t im, mpf_t t1, mpf_t t2, mpf_t outAbs2)
{
    // outAbs2 = re^2 + im^2, return floor(log2(outAbs2)) approx
    mpf_mul(t1, re, re);
    mpf_mul(t2, im, im);
    mpf_add(outAbs2, t1, t2);
    return approx_ilogb_mpf(outAbs2);
}

// ============================================================
// Spin-based multiply worker helpers
// ============================================================

// Cache-line align each job so different workers don't false-share
struct alignas(64) MulJob {
    mpf_ptr out{};
    mpf_srcptr a{};
    mpf_srcptr b{};

    // Pad to a full cache line if needed (optional; alignas often enough,
    // but padding makes intent explicit across compilers/ABIs).
    // If this triggers warnings on some compilers, you can remove it.
    char _pad[64 - (sizeof(mpf_ptr) + sizeof(mpf_srcptr) + sizeof(mpf_srcptr) + sizeof(unsigned long)) >
                      0
                  ? 64 - (sizeof(mpf_ptr) + sizeof(mpf_srcptr) + sizeof(mpf_srcptr) +
                          sizeof(unsigned long))
                  : 1]{};
};

struct MulWorkerParams {
    int idx{};
    MulJob *jobs{};
    std::atomic<uint64_t> *job_gen{};
    std::atomic<uint64_t> *done_gen{};
};

static inline void
MulWorkerMain(MulWorkerParams p)
{
    SetThreadDescription(GetCurrentThread(), std::format(L"FeatureFinder MulWorker {}", p.idx).c_str()); 
    uint64_t seen = 0;

    for (;;) {
        // Wait for a new generation.
        uint64_t g;
        do {
            g = p.job_gen[p.idx].load(std::memory_order_acquire);
            if (g == std::numeric_limits<uint64_t>::max())
                return; // main thread signals stop by setting gen to max
        } while (g == seen);

        // Run job for generation g.
        const MulJob &jb = p.jobs[p.idx];
        mpf_mul(jb.out, jb.a, jb.b);

        // Publish completion for generation g.
        p.done_gen[p.idx].store(g, std::memory_order_release);
        seen = g;
    }
}

// ------------------------------------------------------------
template <typename IterType, typename T>
static inline void
EvaluateCriticalOrbitAndDerivs(const mpf_complex &c_coord, // coord_prec
                               uint64_t period,
                               mpf_complex &z_coord,      // coord_prec (output z_p)
                               mpf_complex &dzdc_deriv,   // deriv_prec (output dzdc_p)
                               HDRFloat<double> &d2r_hdr, // out
                               HDRFloat<double> &d2i_hdr, // out
                               mpf_complex &z_deriv,      // deriv_prec (scratch: z rounded)
                               mpf_complex &tmpB_deriv,   // deriv_prec (scratch)
                               mpf_complex &tmpZ_coord,   // coord_prec (scratch)
                               mpf_t tr_d,
                               mpf_t ti_d,
                               mpf_t t1_d,
                               mpf_t t2_d, // deriv_prec scalars
                               mpf_t tr_c,
                               mpf_t ti_c,
                               mpf_t t1_c,
                               mpf_t t2_c // coord_prec scalars
)
{
    // -------------------------
    // Initialize state
    // -------------------------
    mpf_set_ui(z_coord.re, 0);
    mpf_set_ui(z_coord.im, 0);

    mpf_set_ui(dzdc_deriv.re, 0);
    mpf_set_ui(dzdc_deriv.im, 0);

    T local_d2r = T{};
    T local_d2i = T{};

    // HDR scratch (kept local; no helpers)
    T zr, zi, dzr, dzi;

    T dz2r, dz2i; // dzdc^2
    T zd2r, zd2i; // z*d2
    T sumr, sumi; // dzdc^2 + z*d2

    // ============================================================
    // Spin-based multiply workers (NO mutex/condvar; atomics + spinning)
    // Generation counters (no reset-to-idle state machine).
    //
    // We use 7 workers so we can overlap:
    //   - 4 deriv multiplies (complex mul) + 3 orbit multiplies (sqr) concurrently.
    // ============================================================

    static constexpr int NWORK = 7;

    // Per-worker slots (aligned to reduce false sharing)
    alignas(64) MulJob w_job[NWORK];

    // main->worker: increments to signal a new job (must change each dispatch)
    alignas(64) std::atomic<uint64_t> job_gen[NWORK];
    // worker->main: set to gen that has completed
    alignas(64) std::atomic<uint64_t> done_gen[NWORK];

    for (int k = 0; k < NWORK; ++k) {
        job_gen[k].store(0, std::memory_order_relaxed);
        done_gen[k].store(0, std::memory_order_relaxed);
    }

    // Dedicated outputs:
    // - Workers 0..3 produce deriv-prec products
    // - Workers 4..6 produce coord-prec products
    const mp_bitcnt_t deriv_prec_bits = mpf_get_prec(t1_d);
    const mp_bitcnt_t coord_prec_bits = mpf_get_prec(t1_c);

    mpf_t w_out_deriv[4];
    mpf_t w_out_coord[3];
    for (int k = 0; k < 4; ++k)
        mpf_init2(w_out_deriv[k], deriv_prec_bits);
    for (int k = 0; k < 3; ++k)
        mpf_init2(w_out_coord[k], coord_prec_bits);

    MulWorkerParams w_params[NWORK];
    for (int k = 0; k < NWORK; ++k) {
        w_params[k].idx = k;
        w_params[k].jobs = w_job;
        w_params[k].job_gen = job_gen;
        w_params[k].done_gen = done_gen;
    }

    std::thread w_threads[NWORK] = {
        std::thread(MulWorkerMain, w_params[0]),
        std::thread(MulWorkerMain, w_params[1]),
        std::thread(MulWorkerMain, w_params[2]),
        std::thread(MulWorkerMain, w_params[3]),
        std::thread(MulWorkerMain, w_params[4]),
        std::thread(MulWorkerMain, w_params[5]),
        std::thread(MulWorkerMain, w_params[6]),
    };

    uint64_t main_gen[NWORK]{};

    auto dispatch_mul = [&](int idx, mpf_ptr out, mpf_srcptr a, mpf_srcptr b) {
        w_job[idx].out = out;
        w_job[idx].a = a;
        w_job[idx].b = b;

        const uint64_t g = ++main_gen[idx];               // purely local
        job_gen[idx].store(g, std::memory_order_release); // publish job
    };


    auto wait_done = [&](int idx) {
        const uint64_t g = main_gen[idx];
        while (done_gen[idx].load(std::memory_order_acquire) != g) {
            // spin
        }
    };


    for (uint64_t i = 0; i < period; ++i) {

        // ============================================================
        // Round/copy z from coord_prec -> deriv_prec (for derivative math)
        // ============================================================

        // Orbit multiplies:
        dispatch_mul(4, w_out_coord[0], z_coord.re, z_coord.re); // z.re * z.re
        dispatch_mul(5, w_out_coord[1], z_coord.im, z_coord.im); // z.im * z.im
        dispatch_mul(6, w_out_coord[2], z_coord.re, z_coord.im); // z.re * z.im

        // Deriv multiply:
        mpf_set(z_deriv.re, z_coord.re);
        mpf_mul_ui(tmpB_deriv.re, z_deriv.re, 2);
        dispatch_mul(0, w_out_deriv[0], dzdc_deriv.re, tmpB_deriv.re); // dzdc.re * tmpB.re

        mpf_set(z_deriv.im, z_coord.im);
        mpf_mul_ui(tmpB_deriv.im, z_deriv.im, 2);

        // Remaining deriv multiplies:
        dispatch_mul(1, w_out_deriv[1], dzdc_deriv.im, tmpB_deriv.im); // dzdc.im * tmpB.im
        dispatch_mul(2, w_out_deriv[2], dzdc_deriv.re, tmpB_deriv.im); // dzdc.re * tmpB.im
        dispatch_mul(3, w_out_deriv[3], dzdc_deriv.im, tmpB_deriv.re); // dzdc.im * tmpB.re

        // ============================================================
        // d2 SECTION (HDRFloat):  d2 <- 2*(dzdc^2 + z*d2)
        // ============================================================
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            zr = static_cast<T>(mpf_get_d(z_deriv.re));
            zi = static_cast<T>(mpf_get_d(z_deriv.im));
            dzr = static_cast<T>(mpf_get_d(dzdc_deriv.re));
            dzi = static_cast<T>(mpf_get_d(dzdc_deriv.im));
        } else {
            zr = T(z_deriv.re);
            zi = T(z_deriv.im);
            dzr = T(dzdc_deriv.re);
            dzi = T(dzdc_deriv.im);
        }

        // dzdc^2:
        //   (dzr + i*dzi)^2 = (dzr^2 - dzi^2) + i*(2*dzr*dzi)
        {
            const T dzr2 = dzr * dzr;
            const T dzi2 = dzi * dzi;

            dz2r = dzr2 - dzi2;
            HdrReduce(dz2r);

            dz2i = T{2.0} * (dzr * dzi); // 2*dzr*dzi
            HdrReduce(dz2i);
        }

        // z*d2:
        //   (zr + i*zi)*(d2r + i*d2i)
        // = (zr*d2r - zi*d2i) + i*(zr*d2i + zi*d2r)
        {
            const T zr_d2r = zr * local_d2r;
            const T zi_d2i = zi * local_d2i;
            zd2r = zr_d2r - zi_d2i;
            HdrReduce(zd2r);

            const T zr_d2i = zr * local_d2i;
            const T zi_d2r = zi * local_d2r;
            zd2i = zr_d2i + zi_d2r;
            HdrReduce(zd2i);
        }

        // sum = dzdc^2 + z*d2
        sumr = dz2r + zd2r;
        sumi = dz2i + zd2i;
        HdrReduce(sumr);
        HdrReduce(sumi);

        // d2 = 2*sum
        local_d2r = T{2.0} * sumr;
        local_d2i = T{2.0} * sumi;




        // Wait for all 7
        wait_done(0);
        wait_done(1);

        // ============================================================
        // Combine deriv results (main thread): dzdc *= tmpB; then +1
        // ============================================================
        mpf_sub(tr_d, w_out_deriv[0], w_out_deriv[1]); // tr_d = w0 - w1

        wait_done(2);
        wait_done(3);

        mpf_add(ti_d, w_out_deriv[2], w_out_deriv[3]); // ti_d = w2 + w3

        mpf_set(dzdc_deriv.re, tr_d);
        mpf_set(dzdc_deriv.im, ti_d);

        mpf_add_ui(dzdc_deriv.re, dzdc_deriv.re, 1);

        // ============================================================
        // Combine orbit results (main thread): z = z^2 + c
        // ============================================================

        wait_done(4);
        wait_done(5);

        mpf_sub(tr_c, w_out_coord[0], w_out_coord[1]); // tr_c = zr^2 - zi^2

        wait_done(6);

        mpf_mul_ui(ti_c, w_out_coord[2], 2); // ti_c = 2*zr*zi

        mpf_set(tmpZ_coord.re, tr_c);
        mpf_set(tmpZ_coord.im, ti_c);

        mpf_add(z_coord.re, tmpZ_coord.re, c_coord.re);
        mpf_add(z_coord.im, tmpZ_coord.im, c_coord.im);
    }

    d2r_hdr = HDRFloat<double>(local_d2r);
    d2i_hdr = HDRFloat<double>(local_d2i);

    // ============================================================
    // Tear down workers
    // ============================================================
    for (int k = 0; k < NWORK; ++k) {
        const uint64_t g = std::numeric_limits<uint64_t>::max();
        job_gen[k].store(g, std::memory_order_release);
    }

    for (int k = 0; k < NWORK; ++k) {
        if (w_threads[k].joinable())
            w_threads[k].join();
    }

    for (int k = 0; k < 4; ++k)
        mpf_clear(w_out_deriv[k]);
    for (int k = 0; k < 3; ++k)
        mpf_clear(w_out_coord[k]);
}


// ------------------------------------------------------------
// EvaluateCriticalOrbitAndDerivsST
//
// z_coord:      mpf (coord_prec) orbit
// dzdc_deriv:   mpf (deriv_prec) first derivative
// d2r_hdr/d2i_hdr: HDRFloat (low precision, huge exponent) second derivative
//
// No mpf d2 anywhere.
// ------------------------------------------------------------
template <typename IterType, typename T>
static inline void
EvaluateCriticalOrbitAndDerivsST(const mpf_complex &c_coord, // coord_prec
                               uint64_t period,
                               mpf_complex &z_coord,      // coord_prec (output z_p)
                               mpf_complex &dzdc_deriv,   // deriv_prec (output dzdc_p)
                               HDRFloat<double> &d2r_hdr, // out
                               HDRFloat<double> &d2i_hdr, // out
                               mpf_complex &z_deriv,      // deriv_prec (scratch: z rounded)
                               mpf_complex &tmpB_deriv,   // deriv_prec (scratch)
                               mpf_complex &tmpZ_coord,   // coord_prec (scratch)
                               mpf_t tr_d,
                               mpf_t ti_d,
                               mpf_t t1_d,
                               mpf_t t2_d, // deriv_prec scalars
                               mpf_t tr_c,
                               mpf_t ti_c,
                               mpf_t t1_c,
                               mpf_t t2_c // coord_prec scalars
)
{
    // -------------------------
    // Initialize state
    // -------------------------
    mpf_set_ui(z_coord.re, 0);
    mpf_set_ui(z_coord.im, 0);

    mpf_set_ui(dzdc_deriv.re, 0);
    mpf_set_ui(dzdc_deriv.im, 0);

    T local_d2r = T{};
    T local_d2i = T{};

    // HDR scratch (kept local; no helpers)
    T zr, zi, dzr, dzi;

    T dz2r, dz2i; // dzdc^2
    T zd2r, zd2i; // z*d2
    T sumr, sumi; // dzdc^2 + z*d2

    for (uint64_t i = 0; i < period; ++i) {

        // ============================================================
        // Round/copy z from coord_prec -> deriv_prec (for derivative math)
        // ============================================================
        mpf_set(z_deriv.re, z_coord.re);
        mpf_set(z_deriv.im, z_coord.im);

        // ============================================================
        // d2 SECTION (HDRFloat):  d2 <- 2*(dzdc^2 + z*d2)
        //
        // Promote mpf -> HDRFloat inline using HDRFloat(mpf_t) ctor.
        // ============================================================

        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
             // Fast path for common floating types: convert directly to T (potentially lossy, but we just want an approximation for the HDR math)
             zr =  static_cast<T>(mpf_get_d(z_deriv.re));
             zi =  static_cast<T>(mpf_get_d(z_deriv.im));
             dzr = static_cast<T>(mpf_get_d(dzdc_deriv.re));
             dzi = static_cast<T>(mpf_get_d(dzdc_deriv.im));
        } else {
             // Slow path for other HDRFloat types: convert to double first, then to T
             zr = T(z_deriv.re);
             zi = T(z_deriv.im);
             dzr = T(dzdc_deriv.re);
             dzi = T(dzdc_deriv.im);
        }

        // dzdc^2:
        //   (dzr + i*dzi)^2 = (dzr^2 - dzi^2) + i*(2*dzr*dzi)
        {
            const T dzr2 = dzr * dzr;
            const T dzi2 = dzi * dzi;

            dz2r = dzr2 - dzi2;
            HdrReduce(dz2r);

            dz2i = T{2.0} * (dzr * dzi); // 2*dzr*dzi
            HdrReduce(dz2i);
        }

        // z*d2:
        //   (zr + i*zi)*(d2r + i*d2i)
        // = (zr*d2r - zi*d2i) + i*(zr*d2i + zi*d2r)
        {
            const T zr_d2r = zr * local_d2r;
            const T zi_d2i = zi * local_d2i;
            zd2r = zr_d2r - zi_d2i;
            HdrReduce(zd2r);

            const T zr_d2i = zr * local_d2i;
            const T zi_d2r = zi * local_d2r;
            zd2i = zr_d2i + zi_d2r;
            HdrReduce(zd2i);
        }

        // sum = dzdc^2 + z*d2
        sumr = dz2r + zd2r;
        sumi = dz2i + zd2i;
        HdrReduce(sumr);
        HdrReduce(sumi);

        // d2 = 2*sum
        local_d2r = T{2.0} * sumr;
        local_d2i = T{2.0} * sumi;
        // multiply2 is exponent-only; reduce optional, but safe:
        // HdrReduce(d2r_hdr); HdrReduce(d2i_hdr);

        // ============================================================
        // dzdc SECTION (mpf, deriv_prec):  dzdc <- 2*z*dzdc + 1
        // ============================================================

        // tmpB_deriv = 2*z_deriv
        mpf_mul_ui(tmpB_deriv.re, z_deriv.re, 2);
        mpf_mul_ui(tmpB_deriv.im, z_deriv.im, 2);

        // dzdc_deriv = dzdc_deriv * tmpB_deriv
        {
            mpf_mul(t1_d, dzdc_deriv.re, tmpB_deriv.re);
            mpf_mul(t2_d, dzdc_deriv.im, tmpB_deriv.im);
            mpf_sub(tr_d, t1_d, t2_d);

            mpf_mul(t1_d, dzdc_deriv.re, tmpB_deriv.im);
            mpf_mul(t2_d, dzdc_deriv.im, tmpB_deriv.re);
            mpf_add(ti_d, t1_d, t2_d);

            mpf_set(dzdc_deriv.re, tr_d);
            mpf_set(dzdc_deriv.im, ti_d);
        }

        // +1 (real)
        mpf_add_ui(dzdc_deriv.re, dzdc_deriv.re, 1);

        // ============================================================
        // ORBIT SECTION (mpf, coord_prec):  z <- z^2 + c
        // ============================================================

        // tmpZ_coord = z_coord^2
        {
            mpf_mul(t1_c, z_coord.re, z_coord.re);
            mpf_mul(t2_c, z_coord.im, z_coord.im);
            mpf_sub(tr_c, t1_c, t2_c);

            mpf_mul(t1_c, z_coord.re, z_coord.im);
            mpf_mul_ui(ti_c, t1_c, 2);

            mpf_set(tmpZ_coord.re, tr_c);
            mpf_set(tmpZ_coord.im, ti_c);
        }

        // z_coord = tmpZ_coord + c_coord
        mpf_add(z_coord.re, tmpZ_coord.re, c_coord.re);
        mpf_add(z_coord.im, tmpZ_coord.im, c_coord.im);
    }

    d2r_hdr = HDRFloat<double>(local_d2r);
    d2i_hdr = HDRFloat<double>(local_d2i);
}

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
ComputeHalleyStep_mpf_coord_from_deriv(
    mpf_complex &step_coord,       // coord_prec (out)
    const mpf_complex &z_coord,    // coord_prec  F
    const mpf_complex &dzdc_deriv, // deriv_prec  F'
    const HDRFloat<double> &d2r_hdr,              // low-prec F'' real
    const HDRFloat<double> &d2i_hdr,              // low-prec F'' imag
    mpf_complex &dzdc_coord,       // coord_prec scratch (promoted F')
    mpf_complex &tmp1,             // coord_prec scratch complex
    mpf_complex &tmp2,             // coord_prec scratch complex
    mpf_t denom_c,                 // coord_prec scalar
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
static inline uint32_t
RefinePeriodicPoint(mpf_complex &c_coord,        // coord_prec in/out
                    const mpf_complex &c0_coord, // coord_prec (initial seed)
                    mpf_t sqrRadius_coord,       // coord_prec (R^2) for final accept/reject
                    uint64_t period,
                    mp_bitcnt_t coord_prec,
                    int scaleExp2_for_deriv_choice, // exponent of Scale ≈ 1/|zcoeff*dzdc|
                    uint32_t max_nr_iters)
{
    // Compile-time enable, runtime gating below.
    constexpr bool UseHalley = true;
    constexpr bool UseFullPrecDerivatives = true;

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

    std::cout << "RefinePeriodicPoint: coord_prec=" << coord_prec << " bits, deriv_prec=" << deriv_prec
              << " bits (scaleExp2=" << scaleExp2_for_deriv_choice
              << ", coordExp2_max_abs=" << coordExp2_max_abs << ")\n";

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
    mpf_complex_init(dzdc_deriv, deriv_prec);
    mpf_complex_init(z_deriv, deriv_prec);
    mpf_complex_init(tmpB_d, deriv_prec);

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

    uint32_t it = 0;
    for (; it < max_nr_iters; ++it) {

        std::cout << "  Refinement iter " << it << std::endl;

        // Full forward eval at current c (option 2: d2 in HDRFloat)
        EvaluateCriticalOrbitAndDerivs<IterType, T>(c_coord,
                                                    period,
                                                    z_coord,
                                                    dzdc_deriv,
                                                    d2r_hdr,
                                                    d2i_hdr,
                                                    z_deriv,
                                                    tmpB_d,
                                                    tmpZ_coord,
                                                    tr_d,
                                                    ti_d,
                                                    t1_d,
                                                    t2_d,
                                                    tr_c,
                                                    ti_c,
                                                    t1_c,
                                                    t2_c);

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
            std::cout << "RefinePeriodicPoint: break after dzdcNorm==0\n";
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

        std::cout << "    rho2=" << rho2_hdr.ToString<false>() << " wantHalley=" << wantHalley << "\n";

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
                std::cout << "RefinePeriodicPoint: Halley denom singular, fallback to Newton\n";
                ok = ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c);
            }
            if (!ok) {
                std::cout << "RefinePeriodicPoint: break after Halley/Newton failure\n";
                break;
            }
        } else {
            // Newton step
            if (!ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                    step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
                std::cout << "RefinePeriodicPoint: break after Newton\n";
                break;
            }
        }

        // c <- c - step
        mpf_sub(c_coord.re, c_coord.re, step_coord.re);
        mpf_sub(c_coord.im, c_coord.im, step_coord.im);

        // ------------------------------------------------------------
        // Imagina error estimate (HDRFloat)
        //   err = |step|^4 * |d2|^2 / |dzdc|^2
        // ------------------------------------------------------------
        mpf_complex_norm(normStep, step_coord, t1_c, t2_c);

        normStep_hdr = HDRFloat<double>(normStep);
        HdrReduce(normStep_hdr);

        normStep2_hdr = normStep_hdr.square(); // |step|^4
        HdrReduce(normStep2_hdr);

        err_hdr = (normStep2_hdr * d2Norm_hdr) / dzdcNorm_hdr;
        HdrReduce(err_hdr);

        const int e = (int)err_hdr.getExp();
        if (-e >= targetExp) {
            std::cout << "RefinePeriodicPoint: stop with err_hdr=" << err_hdr.ToString<false>()
                      << " (e=" << e << " >= targetExp=" << targetExp << ")\n";
            break;
        }
    }

    // ---------------- Imagina final correction pass ----------------
    // Keep this Newton-only (matches Imagina + avoids Halley denom corner cases).
    {
        EvaluateCriticalOrbitAndDerivs<IterType, T>(c_coord,
                                                    period,
                                                    z_coord,
                                                    dzdc_deriv,
                                                    d2r_hdr,
                                                    d2i_hdr,
                                                    z_deriv,
                                                    tmpB_d,
                                                    tmpZ_coord,
                                                    tr_d,
                                                    ti_d,
                                                    t1_d,
                                                    t2_d,
                                                    tr_c,
                                                    ti_c,
                                                    t1_c,
                                                    t2_c);

        if (ComputeNewtonStep_mpf_coord_from_deriv<IterType, T>(
                step_coord, z_coord, dzdc_deriv, dzdc_coord, denom_c, tr_c, ti_c, t1_c, t2_c)) {
            mpf_sub(c_coord.re, c_coord.re, step_coord.re);
            mpf_sub(c_coord.im, c_coord.im, step_coord.im);
        }
    }

    // ---------------- Imagina accept/reject: stay within radius ----------------
    mpf_complex_sub(dc, c_coord, c0_coord);
    mpf_complex_norm(abs2_c, dc, t1_c, t2_c);

    if (mpf_cmp(abs2_c, sqrRadius_coord) > 0) {
        mpf_set(c_coord.re, c0_coord.re);
        mpf_set(c_coord.im, c0_coord.im);
        it = 0;
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

    return it;
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
// Returns: number of NR iterations performed (0 can mean "rejected").
// ------------------------------------------------------------
template <class IterType, class T, PerturbExtras PExtras>
IterType
FeatureFinder<IterType, T, PExtras>::RefinePeriodicPoint_WithMPF(
    HighPrecision &cX_hp,
    HighPrecision &cY_hp,
    IterType period,
    mp_bitcnt_t coord_prec,
    const T &sqrRadius_T,          // NEW: radius^2 in T-space
    int scaleExp2_for_deriv) const // NEW: exponent of Scale≈1/|zcoeff*dzdc|
{
    // ---- Convert inputs to MPF at coord_prec ----
    mpf_complex c;
    mpf_complex_init(c, coord_prec);

    mpf_complex c0;
    mpf_complex_init(c0, coord_prec);

    // Seed from HighPrecision backends
    // NOTE: HighPrecision::backend() is expected to be an mpf_t-compatible pointer.
    mpf_set(c.re, (mpf_srcptr)cX_hp.backend());
    mpf_set(c.im, (mpf_srcptr)cY_hp.backend());

    // Keep initial seed (Imagina reject check compares final c vs initial)
    mpf_set(c0.re, c.re);
    mpf_set(c0.im, c.im);

    // Convert sqrRadius_T (T-space) -> mpf_t at coord_prec
    // We treat radius units consistently with your HP/T plane coordinates.
    mpf_t sqrRadius_mpf;
    mpf_init2(sqrRadius_mpf, coord_prec);
    {
        // sqrRadius_T is a T (HDRFloat<double> etc).
        // Convert to HighPrecision then to mpf.
        HighPrecision r2_hp{sqrRadius_T};
        mpf_set(sqrRadius_mpf, (mpf_srcptr)r2_hp.backend());
    }

    // ---- Run Imagina-style polish ----
    const uint32_t max_polish = 32;

    const uint32_t iters = RefinePeriodicPoint<IterType, T>(
        c, c0, sqrRadius_mpf, (uint64_t)period, coord_prec, scaleExp2_for_deriv, max_polish);

    // ---- Write back ----
    cX_hp = HighPrecision{c.re};
    cY_hp = HighPrecision{c.im};

    // ---- Cleanup ----
    mpf_clear(sqrRadius_mpf);
    mpf_complex_clear(c0);
    mpf_complex_clear(c);

    return (IterTypeFull)iters;
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
T
FeatureFinder<IterType, T, PExtras>::ChebAbs(const C &a) const
{
    return HdrMaxReduced(HdrAbs(a.getRe()), HdrAbs(a.getIm()));
}

template <class IterType, class T, PerturbExtras PExtras>
T
FeatureFinder<IterType, T, PExtras>::ToDouble(const HighPrecision &v)
{
    return T{v};
}

template <class IterType, class T, PerturbExtras PExtras>
typename FeatureFinder<IterType, T, PExtras>::C
FeatureFinder<IterType, T, PExtras>::Div(const C &a, const C &b)
{
    // a / b = a * conj(b) / |b|^2
    const T br = b.getRe();
    const T bi = b.getIm();
    T denom = br * br + bi * bi;
    HdrReduce(denom);

    // If denom is 0, caller is hosed; keep it explicit.
    if (HdrCompareToBothPositiveReducedLE(denom, T{})) {
        return C{};
    }

    // a * conj(b)
    const T ar = a.getRe();
    const T ai = a.getIm();
    const T rr = (ar * br + ai * bi) / denom;
    const T ii = (ai * br - ar * bi) / denom;
    return C(rr, ii);
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
        std::cout << "FeatureFinder::Evaluate_FindPeriod_Direct: R must be positive.\n";
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

            std::cout
                << "FeatureFinder::Evaluate_FindPeriod_Direct: candidate period exceeds IterType.\n";
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
            std::cout << "FeatureFinder::Evaluate_PeriodResidualAndDzdc_Direct: orbit escaped.\n";
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
    // We compute: |w| = sqrt(norm_squared(w)) with w = zcoeff*dzdc
    const T one = HdrReduce(T{1.0});
    const T four = HdrReduce(T{4.0});

    C w = zcoeff * dzdc;
    w.Reduce();

    T w2 = w.norm_squared(); // |w|^2
    HdrReduce(w2);

    if (HdrCompareToBothPositiveReducedLE(w2, T{})) {
        // Degenerate; return 0 so caller can fall back to something else if desired
        return HighPrecision{0};
    }

    T absW = HdrSqrt(w2); // |w|
    HdrReduce(absW);

    T scale = one / absW; // 1/|w|
    HdrReduce(scale);

    T radT = scale * four; // 4/|w|
    HdrReduce(radT);

    // Stay in HighPrecision for storage/zoom
    return HighPrecision{radT};
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
    const HighPrecision &cX_hp,
    const HighPrecision &cY_hp,
    T R,
    IterTypeFull maxIters,
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
    const HighPrecision dcX_hp = cX_hp - results.GetHiX();
    const HighPrecision dcY_hp = cY_hp - results.GetHiY();
    C dc(ToDouble(dcX_hp), ToDouble(dcY_hp));
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

    PeriodicityPP<IterType, T, C> pp;
    IterType prePeriod = 0;
    IterType period = 0;

    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero))
            return false;
        pp.Init(R);
        // pp.Init already builds R^2, scale = (0.25)^2
    }

    size_t refIteration = 0;
    C dz{}, z{};
    dz.Reduce();
    z.Reduce();

    // Stored scaled dzdc/zcoeff:
    C dzdc{};   // stored = true * ScalingFactor
    C zcoeff{}; // stored = true * ScalingFactor
    dzdc.Reduce();
    zcoeff.Reduce();

    for (IterTypeFull n = 0; n < cap; ++n) {
        // if Iteration==0 zcoeff = ScalingFactor else zcoeff *= 2*z
        if (n == 0) {
            zcoeff = ScalingFactorC;
        } else {
            zcoeff = zcoeff * (z * two);
        }
        zcoeff.Reduce();

        // scaled dzdc: dzdc = dzdc*(2z) + ScalingFactor
        dzdc = dzdc * (z * two) + ScalingFactorC;
        dzdc.Reduce();

        // PT delta recurrence
        const C zref = results.GetComplex(dec, refIteration);
        dz = dz * (zref + z) + dc;
        dz.Reduce();

        refIteration++;
        const C zrefNext = results.GetComplex(dec, refIteration);
        z = zrefNext + dz;
        z.Reduce();

        // Rebasing check
        T dzNorm = dz.norm_squared();
        HdrReduce(dzNorm);
        T zNorm = z.norm_squared();
        HdrReduce(zNorm);

        if (refIteration >= refOrbitLength - 1 || HdrCompareToBothPositiveReducedLT(zNorm, dzNorm)) {
            dz = z;
            dz.Reduce();
            refIteration = 0;
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
            T dzdcNormStored = dzdc.norm_squared();
            HdrReduce(dzdcNormStored);

            T dzdcNormTrue = dzdcNormStored * InvScale2;
            HdrReduce(dzdcNormTrue);

            // Iteration increments then checks; your n is 0-based
            // At end of loop, we've computed z_{n+1}, dzdc at same time.
            const IterType Iteration = (IterType)(n + 1);

            if (pp.CheckPeriodicity(zNorm, dzdcNormTrue, Iteration, prePeriod, period)) {
                ioPeriod = period; // prePeriod is always 0 here

                // Unscale outputs (to match your direct conventions)
                outDiff = z;
                outDzdc = dzdc * InvScalingFactorC;
                outZcoeff = zcoeff * InvScalingFactorC;
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
    }

    // Fixed-period path: unscale outputs
    outDiff = z;
    outResidual2 = z.norm_squared();
    HdrReduce(outResidual2);

    outDzdc = dzdc * InvScalingFactorC;
    outZcoeff = zcoeff * InvScalingFactorC;
    outDiff.Reduce();
    outDzdc.Reduce();
    outZcoeff.Reduce();
    return true;
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
//  - LAstep::Evaluate(dz, dc) advances dz to the next macro step (step length = las.step).
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
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_LA(
    const PerturbationResults<IterType, T, PExtras> &results,
    LAReference<IterType, T, SubType, PExtras> &laRef,
    const HighPrecision &cX_hp,
    const HighPrecision &cY_hp,
    T R,
    IterTypeFull maxIters,
    IterType &ioPeriod,
    C &outDiff,
    C &outDzdc,
    C &outZcoeff,
    T &outResidual2) const
{
    if (!laRef.IsValid()) {
        return false;
    }

    const T zero = T{};
    const T one = HdrReduce(T{1.0});
    const T two = HdrReduce(T{2.0});
    const T escape2 = HdrReduce(T{4096.0});

    // --------------------------
    // Periodicity state (PT/direct semantics)
    // --------------------------
    PeriodicityPP<IterType, T, C> pp;
    IterType prePeriod = 0;
    IterType period = 0;

    if constexpr (FindPeriod) {
        HdrReduce(R);
        if (HdrCompareToBothPositiveReducedLE(R, zero)) {
            return false;
        }
        pp.Init(R);
    }

    // --------------------------
    // dc = c - referenceCenter
    // --------------------------
    const HighPrecision dcX_hp = cX_hp - results.GetHiX();
    const HighPrecision dcY_hp = cY_hp - results.GetHiY();
    C dc(ToDouble(dcX_hp), ToDouble(dcY_hp));
    dc.Reduce();

    // --------------------------
    // Cap
    // --------------------------
    IterTypeFull cap;
    if constexpr (FindPeriod) {
        cap = maxIters;
    } else {
        cap = static_cast<IterTypeFull>(ioPeriod);
    }
    if (cap < 1) {
        return false;
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
    const C ScalingFactorC(ScalingFactor, T{});
    const C InvScalingFactorC(InvScalingFactor, T{});
    T InvScale2 = InvScalingFactor * InvScalingFactor;
    HdrReduce(InvScale2);

    // =========================================================================
    // State
    // =========================================================================
    C dz{};
    dz.Reduce(); // delta state
    C z{};
    z.Reduce(); // absolute z

    // Stored (scaled) derivative dzdc
    C dzdc{};
    dzdc.Reduce(); // stored = true * ScalingFactor

    // We do NOT use LA "zcoeff" for downstream intrinsic radius / MPF.
    // Set it to 1 at output as a sentinel.
    const C oneC(one, T{});

    IterTypeFull iteration = 0;
    const IterType laStageCount = laRef.GetLAStageCount();

    // Process stages coarse->fine (same as your previous loop)
    for (IterType currentLAStage = laStageCount; currentLAStage > 0 && iteration < cap;) {
        --currentLAStage;

        const IterType laIndex = laRef.getLAIndex(currentLAStage);
        const IterType macroItCount = laRef.getMacroItCount(currentLAStage);

        // Stage invalidity check uses LAThresholdC idea
        if (laRef.isLAStageInvalid(laIndex, dc)) {
            continue;
        }

        IterType refIteration = 0;

        while (iteration < cap) {

            // Get LA step descriptor for this block
            auto las = laRef.getLA(
                laIndex, dz, refIteration, static_cast<IterType>(iteration), static_cast<IterType>(cap));

            if (las.unusable) {
                // Jump within this stage
                const IterType nextRef = las.nextStageLAindex;

                if (nextRef == refIteration || nextRef >= macroItCount) {
                    // can't progress at this stage -> go finer stage
                    break;
                }

                refIteration = nextRef;
                continue;
            }

            // ------------------------------------------------------------
            // Update dzdc (stored scaled) for this macro-step
            // MUST use the scaled API: EvaluateDzdcDeep(dz, dzdc, ScalingFactor)
            // ------------------------------------------------------------
            las.EvaluateDzdcDeep(dz, dzdc, ScalingFactor);
            dzdc.Reduce();

            // ------------------------------------------------------------
            // Advance dz for this macro-step
            // ------------------------------------------------------------
            dz = las.Evaluate(dz, dc);
            dz.Reduce();

            // Advance iteration/refIteration
            iteration += las.step;
            refIteration++;

            // Absolute z at the end of the macro-step
            z = las.getZ(dz);
            z.Reduce();

            // Escape check (like PT/direct)
            T zNorm = z.norm_squared();
            HdrReduce(zNorm);
            if (HdrCompareToBothPositiveReducedGT(zNorm, escape2)) {
                return false;
            }

            // Rebase rule (match PT)
            if (refIteration >= macroItCount ||
                HdrCompareToBothPositiveReducedGT(ChebAbs(dz), ChebAbs(z))) {
                dz = z;
                dz.Reduce();
                refIteration = 0;
            }

            // ------------------------------------------------------------
            // Period detection (PT/direct semantics)
            // ------------------------------------------------------------
            if constexpr (FindPeriod) {
                // TRUE ||dzdc||^2 = STORED ||dzdc||^2 * InvScale2
                T dzdcNormStored = dzdc.norm_squared();
                HdrReduce(dzdcNormStored);

                T dzdcNormTrue = dzdcNormStored * InvScale2;
                HdrReduce(dzdcNormTrue);

                // iteration is IterTypeFull; PP expects IterType
                if (iteration <= static_cast<IterTypeFull>(std::numeric_limits<IterType>::max())) {
                    const IterType IterIt = static_cast<IterType>(iteration);

                    if (pp.CheckPeriodicity(zNorm, dzdcNormTrue, IterIt, prePeriod, period)) {
                        ioPeriod = period;

                        // Outputs:
                        outDiff = z; // same as PT/direct convention

                        // Unscale dzdc to match PT/direct
                        outDzdc = dzdc * InvScalingFactorC;
                        outDzdc.Reduce();

                        // DO NOT use LA zcoeff. Provide a safe sentinel value.
                        outZcoeff = oneC;
                        outZcoeff.Reduce();

                        outResidual2 = zNorm;
                        return true;
                    }
                } else {
                    // period won't fit IterType
                    return false;
                }
            }
        } // while iteration < cap
    } // for stages

    // If we didn't reach cap, caller should fall back (PT/direct)
    if (iteration < cap) {
        return false;
    }

    // Fixed-period evaluation path: return final state at period==cap
    if constexpr (!FindPeriod) {
        // We made it to iteration==cap (or beyond).
        // Output z, dzdc_true, dummy zcoeff.
        outDiff = z;
        outResidual2 = z.norm_squared();
        HdrReduce(outResidual2);

        outDzdc = dzdc * InvScalingFactorC;
        outDzdc.Reduce();

        outZcoeff = oneC;
        outZcoeff.Reduce();

        return true;
    }

    // FindPeriod: reached cap without trigger
    return false;
}

// DirectEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::DirectEvaluator::Eval(const C &c,
                                                           const HighPrecision &cX_hp,
                                                           const HighPrecision &cY_hp,
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
FeatureFinder<IterType, T, PExtras>::LAEvaluator::Eval(const C &c,
                                                       const HighPrecision &cX_hp,
                                                       const HighPrecision &cY_hp,
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
        // Try LA first if valid
        if (laRef->IsValid()) {
            if (self->template Evaluate_LA<FindPeriod>(*results,
                                                       *laRef,
                                                       cX_hp,
                                                       cY_hp,
                                                       R,
                                                       maxIters,
                                                       ioPeriod,
                                                       outDiff,
                                                       outDzdc,
                                                       outZcoeff,
                                                       outResidual2)) {
                return true;
            }

            std::cout
                << "[LA->PT fallback] INFO: LA eval incomplete; trying PT at same period cap="
                << (uint64_t)ioPeriod << "\n";
        } else {
            std::cout << "[LA->PT fallback] INFO: LA reference invalid; trying PT at same period cap="
                      << (uint64_t)ioPeriod << "\n";
        }

        // Fall back to PT
        if (self->template Evaluate_PT<FindPeriod>(*results,
                                                   *dec,
                                                   cX_hp,
                                                   cY_hp,
                                                   R,
                                                   maxIters,
                                                   ioPeriod,
                                                   outDiff,
                                                   outDzdc,
                                                   outZcoeff,
                                                   outResidual2)) {
            return true;
        }

        std::cout << "[LA->PT fallback] INFO: PT eval failed" << std::endl;

        return false;
    }

    // Final fallback to direct (for non-FindPeriod case)
    // Use the requested period as cap (your Evaluate_PT ignores maxIters for !FindPeriod)
    // Note: Here maxIters is actually passed as "period" by your caller, but we ignore it.
    if (self->template Evaluate_PT<false>(*results,
                                          *dec,
                                          cX_hp,
                                          cY_hp,
                                          R,
                                          /*maxIters*/ (IterTypeFull)ioPeriod,
                                          ioPeriod,
                                          outDiff,
                                          outDzdc,
                                          outZcoeff,
                                          outResidual2)) {
        return true;
    }

    std::cout << "[LA->Direct fallback] INFO: LA/PT eval failed; trying DIRECT at period="
              << (uint64_t)ioPeriod << "\n";
    return self->Evaluate_PeriodResidualAndDzdc_Direct(
        c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
}

// PTEvaluator::Eval implementation
// PTEvaluator::Eval implementation
template <class IterType, class T, PerturbExtras PExtras>
template <bool FindPeriod>
bool
FeatureFinder<IterType, T, PExtras>::PTEvaluator::Eval(const C &c,
                                                       const HighPrecision &cX_hp, // ADD
                                                       const HighPrecision &cY_hp, // ADD
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
                                               cX_hp,
                                               cY_hp,
                                               R,
                                               maxIters,
                                               ioPeriod,
                                               outDiff,
                                               outDzdc,
                                               outZcoeff,
                                               outResidual2)) {
        return true;
    }

    if constexpr (!FindPeriod) {
        std::cout << "[PT->Direct fallback] INFO: PT eval failed; trying DIRECT at same period="
                  << (uint64_t)ioPeriod << "\n";
        return self->Evaluate_PeriodResidualAndDzdc_Direct(
            c, ioPeriod, outDiff, outDzdc, outZcoeff, outResidual2);
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
FeatureFinder<IterType, T, PExtras>::RefinePeriodicPoint_HighPrecision(FeatureSummary &feature) const
{
    auto *cand = feature.GetCandidate();
    if (!cand)
        return false;

    // Already refined (e.g. from a previous pass); no need to redo
    if (feature.IsRefined())
        return true;

    HighPrecision cX_hp = cand->cX_hp;
    HighPrecision cY_hp = cand->cY_hp;

    // period conversion
    IterType period{};
    if (cand->period > (IterType)std::numeric_limits<IterType>::max())
        return false;
    period = (IterType)cand->period;

    // Use candidate’s stored radius^2 so Phase B is independent of current FeatureSummary radius
    // (Add this field to PeriodicPointCandidate)
    const HighPrecision &sqrRadius_hp = cand->sqrRadius_hp;

    // MPF polish only
    const IterType refineIters = RefinePeriodicPoint_WithMPF(cX_hp,
                                                             cY_hp,
                                                             period,
                                                             cand->mpfPrecBits,
                                                             /*sqrRadius_T*/ T{sqrRadius_hp},
                                                             cand->scaleExp2_for_mpf);

    // Commit only the refined coordinates + period.
    // residual2/intrinsicRadius can be filled later by a measurement pass.
    feature.SetFound(cX_hp,
                     cY_hp,
                     (IterType)period,
                     HDRFloat<double>{},
                     /*intrinsicRadius*/ HighPrecision{0});
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
    const HighPrecision &origCX_hp = feature.GetOrigX();
    const HighPrecision &origCY_hp = feature.GetOrigY();

    // Search radius (T-space), but we keep c updates in HP and regenerate T c each time.
    T R{feature.GetRadius()};
    R = HdrAbs(R);
    T SqrRadius = R * R;
    HdrReduce(SqrRadius);

    // Canonical parameter in HP (ONLY updated in HP to avoid drift)
    HighPrecision cX_hp = origCX_hp;
    HighPrecision cY_hp = origCY_hp;

    // T-space parameter used by evaluators (ALWAYS derived from HP)
    auto MakeCTFromHP = [&]() -> C {
        C out(ToDouble(cX_hp), ToDouble(cY_hp));
        out.Reduce();
        return out;
    };

    const C origC(ToDouble(origCX_hp), ToDouble(origCY_hp));
    C c = MakeCTFromHP();

    IterType period = 0;
    C diff{}, dzdc{}, zcoeff{};
    T residual2{};

    // -----------------------------------------------------------------------------
    // 1) Find candidate period
    // -----------------------------------------------------------------------------
    if (!evaluator.template Eval<true>(
            c, cX_hp, cY_hp, SqrRadius, refIters, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: findPeriod failed\n";
        return false;
    }

    // -----------------------------------------------------------------------------
    // 2) Initial Newton correction: c <- c - diff/dzdc  (HP only)
    // -----------------------------------------------------------------------------
    {
        T dzdc2_init = dzdc.norm_squared();
        HdrReduce(dzdc2_init);
        if (HdrCompareToBothPositiveReducedLE(dzdc2_init, T{})) {
            std::cout << "Rejected: initial dzdc too small\n";
            return false;
        }

        C step0 = Div(diff, dzdc);
        step0.Reduce();

        cX_hp = cX_hp - HighPrecision{step0.getRe()};
        cY_hp = cY_hp - HighPrecision{step0.getIm()};

        c = MakeCTFromHP();
    }

    // -----------------------------------------------------------------------------
    // Tolerances
    // -----------------------------------------------------------------------------
    const T relTol = m_params.RelStepTol;
    T relTol2 = relTol * relTol;
    HdrReduce(relTol2);

    // Periodicity convergence tolerance:
    //   stop if |diff|^2 <= |c|^2 * tol^2
    const T diffTol = HdrReduce(T{0x1p-40});
    T diffTol2 = diffTol * diffTol;
    HdrReduce(diffTol2);

    // -----------------------------------------------------------------------------
    // 3) Newton loop
    // -----------------------------------------------------------------------------
    for (uint32_t it = 0; it < m_params.MaxNewtonIters; ++it) {
        // Always evaluate at c derived from HP
        c = MakeCTFromHP();

        if (!evaluator.template Eval<false>(
                c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
            std::cout << "Rejected: evalAtPeriod failed in loop\n";
            return false;
        }

        {
            T diff2 = diff.norm_squared();
            HdrReduce(diff2);

            T c2 = c.norm_squared();
            HdrReduce(c2);

            T rhs = c2 * diffTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(diff2, rhs)) {
                // Residual is small enough relative to current c => converged
                std::cout << "Iter1 " << it << ": diff^2=" << HdrToString<false>(diff2)
                          << ", |c|^2*tol^2=" << HdrToString<false>(rhs) << "\n";
                break;
            }
        }

        // dzdc must be non-degenerate
        T dzdc2 = dzdc.norm_squared();
        HdrReduce(dzdc2);
        if (HdrCompareToBothPositiveReducedLE(dzdc2, T{})) {
            std::cout << "Rejected: dzdc too small\n";
            return false;
        }

        // Newton step in T-space
        C step = Div(diff, dzdc);
        step.Reduce();

        // Update ONLY HP
        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};

        // Rebuild T-space c from HP
        c = MakeCTFromHP();

        // Optional: keep your original step-based stop as a secondary criterion
        {
            T step2 = step.norm_squared();
            HdrReduce(step2);

            T c2 = c.norm_squared();
            HdrReduce(c2);

            T rhs = c2 * relTol2;
            HdrReduce(rhs);

            if (HdrCompareToBothPositiveReducedLE(step2, rhs)) {
                std::cout << "Iter2 " << it << ": step^2=" << HdrToString<false>(step2)
                          << ", |c|^2*relTol^2=" << HdrToString<false>(rhs) << "\n";
                break;
            }
        }

        // Optional absolute residual accept (if you already use this)
        if (HdrCompareToBothPositiveReducedGT(m_params.Eps2Accept, T{}) &&
            HdrCompareToBothPositiveReducedLE(residual2, m_params.Eps2Accept)) {
            std::cout << "Iter3 " << it << ": residual^2=" << HdrToString<false>(residual2) << "\n";
            break;
        }
    }

    // -----------------------------------------------------------------------------
    // Final correction pass (same idea)
    // -----------------------------------------------------------------------------
    c = MakeCTFromHP();

    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

    {
        T dzdc2_final = dzdc.norm_squared();
        HdrReduce(dzdc2_final);
        if (HdrCompareToBothPositiveReducedLE(dzdc2_final, T{})) {
            std::cout << "Rejected: final dzdc too small\n";
            return false;
        }

        C step = Div(diff, dzdc);
        step.Reduce();

        cX_hp = cX_hp - HighPrecision{step.getRe()};
        cY_hp = cY_hp - HighPrecision{step.getIm()};

        c = MakeCTFromHP();
    }

    // Final eval for residual and scale
    if (!evaluator.template Eval<false>(
            c, cX_hp, cY_hp, SqrRadius, period, period, diff, dzdc, zcoeff, residual2)) {
        std::cout << "Rejected: final eval failed\n";
        return false;
    }

    {
        C w = zcoeff * dzdc;
        w.Reduce();

        T w2 = w.norm_squared();
        HdrReduce(w2);

        if (!HdrCompareToBothPositiveReducedGT(w2, T{})) {
            std::cout << "Rejected: zero w in candidate store\n";
            feature.ClearCandidate();
            return false;
        }

        int scaleExp2 = 0;
        mp_bitcnt_t prec_bits = cX_hp.precisionInBits();

        {
            T absW = HdrSqrt(w2);
            HdrReduce(absW);

            T scaleT = HdrReduce(T{1.0}) / absW;
            HdrReduce(scaleT);

            HighPrecision scaleHP{scaleT};
            long exp_long;
            double mant;
            scaleHP.frexp(mant, exp_long);
            scaleExp2 = (int)exp_long;

            const int bitsFromScale = std::max(0, -scaleExp2);
            const int marginBits = 256;

            mp_bitcnt_t want = (mp_bitcnt_t)(bitsFromScale + marginBits);
            want = std::max(want, (mp_bitcnt_t)cX_hp.precisionInBits());
            want = std::max<mp_bitcnt_t>(want, 512);
            prec_bits = want;
        }

        // Convert T SqrRadius → HP once and store it
        HighPrecision sqrRadius_hp{SqrRadius};

        // Store candidate for Phase B refinement
        feature.SetCandidate(cX_hp,
                             cY_hp,
                             (IterTypeFull)period,
                             HDRFloat<double>{residual2},
                             sqrRadius_hp,
                             scaleExp2,
                             prec_bits);
    }

    const HighPrecision intrinsicRadius = ComputeIntrinsicRadius_HP(zcoeff, dzdc);

    feature.SetFound(cX_hp, cY_hp, period, HDRFloat<double>{residual2}, intrinsicRadius);

    if (m_params.PrintResult) {
        std::cout << "Periodic point: "
                  << " orig cx=" << HdrToString<false, T>(origC.getRe())
                  << " orig cy=" << HdrToString<false, T>(origC.getIm())
                  << " new cx=" << HdrToString<false, T>(c.getRe())
                  << " new cy=" << HdrToString<false, T>(c.getIm())
                  << " period=" << static_cast<uint64_t>(period)
                  << " residual2=" << HdrToString<false, T>(residual2)
                  << " intrinsicRadius=" << intrinsicRadius.str() << "\n";
    }

    return true;
}

template <class IterType, class T, PerturbExtras PExtras>
bool
FeatureFinder<IterType, T, PExtras>::Evaluate_AtPeriod(
    const PerturbationResults<IterType, T, PExtras> & /*results*/,
    RuntimeDecompressor<IterType, T, PExtras> &dec,
    const C &c,
    IterType period,
    EvalState &st,
    T &outResidual2) const
{
    // Re-evaluate orbit up to "period" and measure how close we are to z0.
    C z{};
    const C z0 = z;

    for (IterType i = 0; i < period; ++i) {
        z = (z * z) + c;
        (void)dec;
    }

    const C delta = z - z0;
    outResidual2 = delta.norm_squared();
    st.z = z;
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

// InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Disable);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Disable);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Disable);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Disable);

// InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
// InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Disable);
//
// ---- Bad ----
//InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::Bad);

//InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::Bad);

//InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::Bad);

//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::Bad);

//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::Bad);

//InstantiatePeriodicPointFinder(uint32_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);
//InstantiatePeriodicPointFinder(uint64_t, HDRFloat<CudaDblflt<MattDblflt>>, PerturbExtras::Bad);

// ---- SimpleCompression ----
InstantiatePeriodicPointFinder(uint32_t, double, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, double, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, float, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, float, PerturbExtras::SimpleCompression);

//InstantiatePeriodicPointFinder(uint32_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t, CudaDblflt<MattDblflt>, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<double>, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<double>, PerturbExtras::SimpleCompression);

InstantiatePeriodicPointFinder(uint32_t, HDRFloat<float>, PerturbExtras::SimpleCompression);
InstantiatePeriodicPointFinder(uint64_t, HDRFloat<float>, PerturbExtras::SimpleCompression);

//InstantiatePeriodicPointFinder(uint32_t,
//                               HDRFloat<CudaDblflt<MattDblflt>>,
//                               PerturbExtras::SimpleCompression);
//InstantiatePeriodicPointFinder(uint64_t,
//                               HDRFloat<CudaDblflt<MattDblflt>>,
//                               PerturbExtras::SimpleCompression);

#undef InstantiatePeriodicPointFinder
