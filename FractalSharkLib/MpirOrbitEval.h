#pragma once

// Shared multi-threaded MPIR orbit evaluation with derivatives.
// Used by both FeatureFinder (production) and TestNewtonRaphson (test benchmark).

#include "HDRFloat.h"
#include <gmp.h>
#include <cstdint>

struct mpf_complex {
    mpf_t re, im;
};

void mpf_complex_init(mpf_complex &z, mp_bitcnt_t prec);
void mpf_complex_clear(mpf_complex &z);
void mpf_complex_set(mpf_complex &dst, const mpf_complex &src);
void mpf_complex_set_ui(mpf_complex &z, unsigned long re, unsigned long im);
void mpf_complex_sub(mpf_complex &out, const mpf_complex &a, const mpf_complex &b);
void mpf_complex_add(mpf_complex &out, const mpf_complex &a, const mpf_complex &b);
void mpf_complex_norm(mpf_t out, const mpf_complex &z, mpf_t t1, mpf_t t2);
int approx_ilogb_mpf(const mpf_t x);
int approx_ilogb_mpf_abs2(const mpf_t re, const mpf_t im, mpf_t t1, mpf_t t2, mpf_t outAbs2);

mp_bitcnt_t ChooseDerivPrec_ImaginaStyle(mp_bitcnt_t coord_prec,
                                          int scaleExp2,
                                          int coordExp2_max_abs,
                                          mp_bitcnt_t minPrec = 256);

// Multi-threaded MPIR orbit: z=z^2+c for `period` iterations with dzdc and d2.
// Uses 7 spin-locked worker threads for parallel MPIR multiplies.
void EvaluateCriticalOrbitAndDerivsMT(
    const mpf_complex &c_coord,
    uint64_t period,
    mpf_complex &z_coord,
    mpf_complex &dzdc_deriv,
    HDRFloat<double> &d2r_hdr,
    HDRFloat<double> &d2i_hdr,
    mp_bitcnt_t deriv_prec,
    mp_bitcnt_t coord_prec);

// Single-threaded MPIR orbit: same math as MT but all multiplies sequential.
void EvaluateCriticalOrbitAndDerivsST(
    const mpf_complex &c_coord,
    uint64_t period,
    mpf_complex &z_coord,
    mpf_complex &dzdc_deriv,
    HDRFloat<double> &d2r_hdr,
    HDRFloat<double> &d2i_hdr,
    mp_bitcnt_t deriv_prec,
    mp_bitcnt_t coord_prec);

