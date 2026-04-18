#pragma once

// Thin dispatcher that evaluates the critical orbit z=z²+c for `period`
// iterations, computing z + dzdc (MPIR) + d2 (HDRFloat).  Dispatches to
// GPU, CPU multi-threaded, or CPU single-threaded backend.
//
// This consolidates the backend-selection logic that was duplicated in
// FeatureFinder's main NR loop and final correction pass.

#include "FeatureFinderMode.h"
#include "HDRFloat.h"
#include "MpirComplex.h"

#include <cstdint>

// Evaluate the critical orbit z=z²+c (with dzdc and d2 derivatives) for
// `period` iterations using the specified backend.
//
// coord_prec / deriv_prec: MPIR bit precision for coordinate and derivative
//                          computations.  coord_prec is also used to select
//                          the GPU SharkParamsNR limb-count.
//
// startIter:  When > 0, z/dzdc/d2 must contain state at that iteration
//             (checkpoint resume).
//
// onProgress / progressCtx: Optional callback invoked periodically with the
//                           current iteration count.
//
// Returns the number of iterations completed (== period if finished,
// < period if aborted).
uint64_t EvaluateCriticalOrbitAndDerivs(NRInnerLoopBackend backend,
                                        const mpf_complex &c,
                                        uint64_t period,
                                        mpf_complex &z,
                                        mpf_complex &dzdc,
                                        HDRFloat<double> &d2r,
                                        HDRFloat<double> &d2i,
                                        mp_bitcnt_t coord_prec,
                                        mp_bitcnt_t deriv_prec,
                                        uint64_t startIter = 0,
                                        void (*onProgress)(uint64_t, void *) = nullptr,
                                        void *progressCtx = nullptr);
