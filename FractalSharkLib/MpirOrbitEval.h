#pragma once

// Shared multi-threaded MPIR orbit evaluation with derivatives.
// Used by both FeatureFinder (production) and TestNewtonRaphson (test benchmark).

#include "HDRFloat.h"
#include "MpirComplex.h"

#include <cstdint>

// Multi-threaded MPIR orbit: z=z^2+c for `period` iterations with dzdc and d2.
// Uses 7 spin-locked worker threads for parallel MPIR multiplies.
// When startIter > 0, z_coord/dzdc_deriv/d2r_hdr/d2i_hdr must contain the
// state at iteration startIter (i.e., restored from a checkpoint).
// onProgress is called every progressInterval iterations with the current count.
// Returns the number of iterations completed (== period if finished,
// < period if aborted via AbortMonitor).
uint64_t EvaluateCriticalOrbitAndDerivsMT(const mpf_complex &c_coord,
                                          uint64_t period,
                                          mpf_complex &z_coord,
                                          mpf_complex &dzdc_deriv,
                                          HDRFloat<double> &d2r_hdr,
                                          HDRFloat<double> &d2i_hdr,
                                          mp_bitcnt_t deriv_prec,
                                          mp_bitcnt_t coord_prec,
                                          uint64_t startIter = 0,
                                          void (*onProgress)(uint64_t, void *) = nullptr,
                                          void *progressContext = nullptr,
                                          uint64_t progressInterval = 131072);

// Single-threaded MPIR orbit: same math as MT but all multiplies sequential.
uint64_t EvaluateCriticalOrbitAndDerivsST(const mpf_complex &c_coord,
                                          uint64_t period,
                                          mpf_complex &z_coord,
                                          mpf_complex &dzdc_deriv,
                                          HDRFloat<double> &d2r_hdr,
                                          HDRFloat<double> &d2i_hdr,
                                          mp_bitcnt_t deriv_prec,
                                          mp_bitcnt_t coord_prec,
                                          uint64_t startIter = 0,
                                          void (*onProgress)(uint64_t, void *) = nullptr,
                                          void *progressContext = nullptr,
                                          uint64_t progressInterval = 131072);
