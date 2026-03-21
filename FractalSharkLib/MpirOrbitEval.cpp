#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "MpirOrbitEval.h"

#include <atomic>
#include <format>
#include <limits>
#include <thread>

// ============================================================
// mpf_complex helpers
// ============================================================

void mpf_complex_init(mpf_complex &z, mp_bitcnt_t prec) {
    mpf_init2(z.re, prec);
    mpf_init2(z.im, prec);
}

void mpf_complex_clear(mpf_complex &z) {
    mpf_clear(z.re);
    mpf_clear(z.im);
}

void mpf_complex_set(mpf_complex &dst, const mpf_complex &src) {
    mpf_set(dst.re, src.re);
    mpf_set(dst.im, src.im);
}

void mpf_complex_set_ui(mpf_complex &z, unsigned long re, unsigned long im) {
    mpf_set_ui(z.re, re);
    mpf_set_ui(z.im, im);
}

void mpf_complex_sub(mpf_complex &out, const mpf_complex &a, const mpf_complex &b) {
    mpf_sub(out.re, a.re, b.re);
    mpf_sub(out.im, a.im, b.im);
}

void mpf_complex_add(mpf_complex &out, const mpf_complex &a, const mpf_complex &b) {
    mpf_add(out.re, a.re, b.re);
    mpf_add(out.im, a.im, b.im);
}

void mpf_complex_norm(mpf_t out, const mpf_complex &z, mpf_t t1, mpf_t t2) {
    mpf_mul(t1, z.re, z.re);
    mpf_mul(t2, z.im, z.im);
    mpf_add(out, t1, t2);
}

int approx_ilogb_mpf(const mpf_t x) {
    if (mpf_cmp_ui(x, 0) == 0)
        return INT32_MIN;
    long exp2;
    (void)mpf_get_d_2exp(&exp2, x);
    return int(exp2 - 1);
}

int approx_ilogb_mpf_abs2(const mpf_t re, const mpf_t im, mpf_t t1, mpf_t t2, mpf_t outAbs2) {
    mpf_mul(t1, re, re);
    mpf_mul(t2, im, im);
    mpf_add(outAbs2, t1, t2);
    return approx_ilogb_mpf(outAbs2);
}

mp_bitcnt_t ChooseDerivPrec_ImaginaStyle(mp_bitcnt_t coord_prec,
                                          int scaleExp2,
                                          int coordExp2_max_abs,
                                          mp_bitcnt_t minPrec) {
    long dp = (long)((-scaleExp2 + 32) / 4);
    if (dp < (long)minPrec)
        dp = (long)minPrec;
    long cap = (long)coord_prec + (long)coordExp2_max_abs + 32;
    if (dp > cap)
        dp = cap;
    if (dp > (long)coord_prec)
        dp = (long)coord_prec;
    return (mp_bitcnt_t)dp;
}

// ============================================================
// Spin-based multiply worker
// ============================================================

struct alignas(64) MulJob {
    mpf_ptr out{};
    mpf_srcptr a{};
    mpf_srcptr b{};
};

struct MulWorkerParams {
    int idx{};
    MulJob *jobs{};
    std::atomic<uint64_t> *job_gen{};
    std::atomic<uint64_t> *done_gen{};
};

static void
MulWorkerMain(MulWorkerParams p) {
    SetThreadDescription(GetCurrentThread(), std::format(L"MpirOrbit MulWorker {}", p.idx).c_str());
    uint64_t seen = 0;

    for (;;) {
        uint64_t g;
        do {
            g = p.job_gen[p.idx].load(std::memory_order_acquire);
            if (g == (std::numeric_limits<uint64_t>::max)())
                return;
        } while (g == seen);

        const MulJob &jb = p.jobs[p.idx];
        mpf_mul(jb.out, jb.a, jb.b);

        p.done_gen[p.idx].store(g, std::memory_order_release);
        seen = g;
    }
}

// ============================================================
// Multi-threaded MPIR orbit evaluation
// ============================================================

void EvaluateCriticalOrbitAndDerivsMT(
    const mpf_complex &c_coord,
    uint64_t period,
    mpf_complex &z_coord,
    mpf_complex &dzdc_deriv,
    HDRFloat<double> &d2r_hdr,
    HDRFloat<double> &d2i_hdr,
    mp_bitcnt_t deriv_prec,
    mp_bitcnt_t coord_prec)
{
    // Scratch temporaries
    mpf_complex z_deriv, tmpB_deriv, tmpZ_coord;
    mpf_complex_init(z_deriv, deriv_prec);
    mpf_complex_init(tmpB_deriv, deriv_prec);
    mpf_complex_init(tmpZ_coord, coord_prec);

    mpf_t tr_d, ti_d, t1_d, t2_d, tr_c, ti_c, t1_c, t2_c;
    mpf_init2(tr_d, deriv_prec); mpf_init2(ti_d, deriv_prec);
    mpf_init2(t1_d, deriv_prec); mpf_init2(t2_d, deriv_prec);
    mpf_init2(tr_c, coord_prec); mpf_init2(ti_c, coord_prec);
    mpf_init2(t1_c, coord_prec); mpf_init2(t2_c, coord_prec);

    // Initialize state
    mpf_set_ui(z_coord.re, 0);
    mpf_set_ui(z_coord.im, 0);
    mpf_set_ui(dzdc_deriv.re, 0);
    mpf_set_ui(dzdc_deriv.im, 0);

    using T = HDRFloat<double>;
    T local_d2r{}, local_d2i{};
    T zr, zi, dzr, dzi;
    T dz2r, dz2i, zd2r, zd2i, sumr, sumi;

    static constexpr int NWORK = 7;

    alignas(64) MulJob w_job[NWORK];
    alignas(64) std::atomic<uint64_t> job_gen[NWORK];
    alignas(64) std::atomic<uint64_t> done_gen[NWORK];

    for (int k = 0; k < NWORK; ++k) {
        job_gen[k].store(0, std::memory_order_relaxed);
        done_gen[k].store(0, std::memory_order_relaxed);
    }

    mpf_t w_out_deriv[4];
    mpf_t w_out_coord[3];
    for (int k = 0; k < 4; ++k)
        mpf_init2(w_out_deriv[k], deriv_prec);
    for (int k = 0; k < 3; ++k)
        mpf_init2(w_out_coord[k], coord_prec);

    MulWorkerParams w_params[NWORK];
    for (int k = 0; k < NWORK; ++k) {
        w_params[k].idx = k;
        w_params[k].jobs = w_job;
        w_params[k].job_gen = job_gen;
        w_params[k].done_gen = done_gen;
    }

    std::thread w_threads[NWORK];
    for (int k = 0; k < NWORK; ++k)
        w_threads[k] = std::thread(MulWorkerMain, w_params[k]);

    uint64_t main_gen[NWORK]{};

    auto dispatch_mul = [&](int idx, mpf_ptr out, mpf_srcptr a, mpf_srcptr b) {
        w_job[idx].out = out;
        w_job[idx].a = a;
        w_job[idx].b = b;
        const uint64_t g = ++main_gen[idx];
        job_gen[idx].store(g, std::memory_order_release);
    };

    auto wait_done = [&](int idx) {
        const uint64_t g = main_gen[idx];
        while (done_gen[idx].load(std::memory_order_acquire) != g) {
            // spin
        }
    };

    for (uint64_t i = 0; i < period; ++i) {
        dispatch_mul(4, w_out_coord[0], z_coord.re, z_coord.re);
        dispatch_mul(5, w_out_coord[1], z_coord.im, z_coord.im);
        dispatch_mul(6, w_out_coord[2], z_coord.re, z_coord.im);

        mpf_set(z_deriv.re, z_coord.re);
        mpf_mul_ui(tmpB_deriv.re, z_deriv.re, 2);
        dispatch_mul(0, w_out_deriv[0], dzdc_deriv.re, tmpB_deriv.re);

        mpf_set(z_deriv.im, z_coord.im);
        mpf_mul_ui(tmpB_deriv.im, z_deriv.im, 2);

        dispatch_mul(1, w_out_deriv[1], dzdc_deriv.im, tmpB_deriv.im);
        dispatch_mul(2, w_out_deriv[2], dzdc_deriv.re, tmpB_deriv.im);
        dispatch_mul(3, w_out_deriv[3], dzdc_deriv.im, tmpB_deriv.re);

        // d2 update (HDRFloat, overlaps with multiply workers)
        zr = T(mpf_get_d(z_deriv.re));
        zi = T(mpf_get_d(z_deriv.im));
        dzr = T(mpf_get_d(dzdc_deriv.re));
        dzi = T(mpf_get_d(dzdc_deriv.im));

        {
            const T dzr2 = dzr * dzr;
            const T dzi2 = dzi * dzi;
            dz2r = dzr2 - dzi2;
            HdrReduce(dz2r);
            dz2i = T{2.0} * (dzr * dzi);
            HdrReduce(dz2i);
        }

        {
            zd2r = zr * local_d2r - zi * local_d2i;
            HdrReduce(zd2r);
            zd2i = zr * local_d2i + zi * local_d2r;
            HdrReduce(zd2i);
        }

        sumr = dz2r + zd2r;
        sumi = dz2i + zd2i;
        HdrReduce(sumr);
        HdrReduce(sumi);

        local_d2r = T{2.0} * sumr;
        local_d2i = T{2.0} * sumi;

        wait_done(0);
        wait_done(1);
        mpf_sub(tr_d, w_out_deriv[0], w_out_deriv[1]);

        wait_done(2);
        wait_done(3);
        mpf_add(ti_d, w_out_deriv[2], w_out_deriv[3]);

        mpf_set(dzdc_deriv.re, tr_d);
        mpf_set(dzdc_deriv.im, ti_d);
        mpf_add_ui(dzdc_deriv.re, dzdc_deriv.re, 1);

        wait_done(4);
        wait_done(5);
        mpf_sub(tr_c, w_out_coord[0], w_out_coord[1]);

        wait_done(6);
        mpf_mul_ui(ti_c, w_out_coord[2], 2);

        mpf_set(tmpZ_coord.re, tr_c);
        mpf_set(tmpZ_coord.im, ti_c);

        mpf_add(z_coord.re, tmpZ_coord.re, c_coord.re);
        mpf_add(z_coord.im, tmpZ_coord.im, c_coord.im);
    }

    d2r_hdr = HDRFloat<double>(local_d2r);
    d2i_hdr = HDRFloat<double>(local_d2i);

    // Tear down workers
    for (int k = 0; k < NWORK; ++k)
        job_gen[k].store((std::numeric_limits<uint64_t>::max)(), std::memory_order_release);
    for (int k = 0; k < NWORK; ++k)
        if (w_threads[k].joinable()) w_threads[k].join();

    for (int k = 0; k < 4; ++k)
        mpf_clear(w_out_deriv[k]);
    for (int k = 0; k < 3; ++k)
        mpf_clear(w_out_coord[k]);

    mpf_clear(tr_d); mpf_clear(ti_d);
    mpf_clear(t1_d); mpf_clear(t2_d);
    mpf_clear(tr_c); mpf_clear(ti_c);
    mpf_clear(t1_c); mpf_clear(t2_c);
    mpf_complex_clear(z_deriv);
    mpf_complex_clear(tmpB_deriv);
    mpf_complex_clear(tmpZ_coord);
}
