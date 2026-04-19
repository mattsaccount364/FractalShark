#pragma once

// MPIR complex number helpers and precision utilities.
// Extracted from MpirOrbitEval for shared use by FeatureFinder,
// RefOrbitCalc, and test code.

#include "MpirGmp.h"
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

// Approximate binary exponent of x: floor(log2(|x|)).
// Returns INT32_MIN for zero.
int approx_ilogb_mpf(const mpf_t x);

// Approximate binary exponent of |re + i*im|^2.
// Writes the squared norm into outAbs2.
int approx_ilogb_mpf_abs2(const mpf_t re, const mpf_t im, mpf_t t1, mpf_t t2, mpf_t outAbs2);

// Imagina-style heuristic for choosing derivative precision.
// Returns a precision (in bits) for dzdc, balancing speed against convergence quality.
mp_bitcnt_t ChooseDerivPrec_ImaginaStyle(mp_bitcnt_t coord_prec,
                                         int scaleExp2,
                                         int coordExp2_max_abs,
                                         mp_bitcnt_t minPrec = 256);
