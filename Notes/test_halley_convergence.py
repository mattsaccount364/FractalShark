#!/usr/bin/env python3
"""
Empirical convergence test: Newton vs Halley vs 53-bit-truncated Halley
for Mandelbrot nucleus finding.

Tests the claim that Halley with 53-bit F'' gives cubic convergence
until ~53 correct bits, then degenerates to quadratic.

The script mirrors the production CPU Phase B path closely in the aspects
that matter for this claim:
  - d2 is accumulated from mpf_get_d-style 53-bit inputs
  - the Halley step promotes low-precision d2 back to high precision
  - the runtime Halley gate rho^2 < 2^-12 is applied

It does not explicitly round dzdc down to the production deriv_prec policy.
That omission is acceptable for the crossover question because deriv_prec has
at least a 256-bit floor in the code, well above the 53-bit F'' limit under test.

Reports BOTH:
  - correct_bits: -log2|c_k - c*|  (ground truth, needs known root)
  - normalized_bits: -log2(rho2)/2  (the runtime rho2 diagnostic)
"""

import mpmath
import math
import sys

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
WORK_DPS = 300          # decimal digits of working precision
ROOT_DPS = 400          # extra precision for computing "true" root
MAX_PERIOD_SEARCH = 50000  # max period to search for
HALLEY_RHO2_EXP_THRESHOLD = -12  # match FeatureFinder.cpp

START_BITS_SWEEP = [10, 14, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150]
TARGET_BITS_SWEEP = [24, 53, 80, 100, 150, 200, 300, 400, 600, 900]


def iterate_orbit(c, period, dps=None):
    """Iterate z = z^2 + c for `period` steps.
    Returns (z_p, dzdc_p, d2_p) all at full working precision.
    d2 recurrence: d2_{n+1} = 2*(dzdc_n^2 + z_n * d2_n)
    """
    if dps is not None:
        old = mpmath.mp.dps
        mpmath.mp.dps = dps

    z = mpmath.mpc(0, 0)
    dzdc = mpmath.mpc(0, 0)
    d2 = mpmath.mpc(0, 0)

    for _ in range(period):
        # d2 update first (uses current z, dzdc)
        d2 = 2 * (dzdc * dzdc + z * d2)
        # dzdc update
        dzdc = 2 * z * dzdc + 1
        # z update
        z = z * z + c

    if dps is not None:
        mpmath.mp.dps = old

    return z, dzdc, d2


def quantize_sigbits(value, bits):
    """Quantize an mpmath scalar to a fixed mantissa width while keeping a wide exponent.

    This approximates HDRFloat-style arithmetic more closely than IEEE float32/64
    because it reduces mantissa precision without forcing the exponent into a
    narrow hardware range.
    """
    x = mpmath.mpf(value)
    if x == 0:
        return mpmath.mpf('0')

    sign = -1 if x < 0 else 1
    mant, exp2 = mpmath.frexp(abs(x))  # x = mant * 2**exp2, mant in [0.5, 1)
    scale = mpmath.ldexp(1, bits)
    mant_q = mpmath.nint(mant * scale) / scale

    # Handle rounding overflow, e.g. 0.111111... -> 1.0
    if mant_q >= 1:
        mant_q = mpmath.mpf('0.5')
        exp2 += 1

    return sign * mpmath.ldexp(mant_q, exp2)


def iterate_orbit_quantized_d2(c, period, d2_mode, dps=None):
    """Orbit iteration with high-precision z/dzdc but quantized d2 update.

    d2_mode:
      - 'full': high-precision d2
      - 'fp64': z/dzdc truncated through Python float and d2 kept in float64
      - 'fp32': z/dzdc truncated to float32 and d2 kept in float32
    """
    if d2_mode == 'full':
        return iterate_orbit(c, period, dps)

    if dps is not None:
        old = mpmath.mp.dps
        mpmath.mp.dps = dps

    z = mpmath.mpc(0, 0)
    dzdc = mpmath.mpc(0, 0)

    if d2_mode == 'fp64':
        bits = 53
    elif d2_mode == 'fp32':
        bits = 24
    else:
        raise ValueError(f"Unknown d2_mode: {d2_mode}")

    q = lambda x: quantize_sigbits(x, bits)
    two = mpmath.mpf('2')

    d2_r = mpmath.mpf('0')
    d2_i = mpmath.mpf('0')

    for _ in range(period):
        # Truncate z and dzdc before the d2 update, matching the production
        # path where d2 is accumulated from reduced-precision copies.
        zr = q(mpmath.re(z))
        zi = q(mpmath.im(z))
        dr = q(mpmath.re(dzdc))
        di = q(mpmath.im(dzdc))

        # d2 = 2*(dzdc^2 + z*d2) in the chosen reduced precision
        dz2r = q(q(dr * dr) - q(di * di))
        dz2i = q(q(two) * q(dr * di))

        zd2r = q(q(zr * d2_r) - q(zi * d2_i))
        zd2i = q(q(zr * d2_i) + q(zi * d2_r))

        d2_r = q(q(two) * q(dz2r + zd2r))
        d2_i = q(q(two) * q(dz2i + zd2i))

        # dzdc and z at full precision
        dzdc = 2 * z * dzdc + 1
        z = z * z + c

    if dps is not None:
        mpmath.mp.dps = old

    # Return d2 as mpmath complex with quantized mantissa but wide exponent
    d2_mp = mpmath.mpc(d2_r, d2_i)
    return z, dzdc, d2_mp


def newton_step(z, dzdc):
    """Newton step: Delta = F/F' = z/dzdc"""
    if dzdc == 0:
        return None
    return z / dzdc


def halley_step(z, dzdc, d2):
    """Halley step: Delta = 2*F*F' / (2*F'^2 - F*F'')"""
    num = 2 * z * dzdc
    den = 2 * dzdc * dzdc - z * d2
    if den == 0:
        return None
    return num / den


def log2_abs(x):
    """Return log2(|x|) at mpmath precision."""
    ax = abs(x)
    if ax == 0:
        return float('-inf')
    return float(mpmath.log(ax, 2))


def compute_rho2_exp(z, dzdc, d2):
    """log2(rho^2) = 2 log2|F| + 2 log2|F''| - 4 log2|F'|."""
    dzdc_abs = abs(dzdc)
    if dzdc_abs == 0:
        return float('inf')
    return 2.0 * log2_abs(z) + 2.0 * log2_abs(d2) - 4.0 * log2_abs(dzdc)


def find_period(c, max_period=MAX_PERIOD_SEARCH):
    """Find the most likely period by iterating and tracking |z| minima."""
    mpmath.mp.dps = 50  # moderate precision for detection
    z = mpmath.mpc(0, 0)
    best_period = 0
    best_abs = float('inf')

    for n in range(1, max_period + 1):
        z = z * z + c
        az = float(abs(z))
        if az < best_abs:
            best_abs = az
            best_period = n

        if az > 4:
            print(f"  Point escapes at iteration {n}, not near a nucleus")
            return None

    mpmath.mp.dps = WORK_DPS
    print(f"  Best period candidate: {best_period} (|z_p| ~ {best_abs:.6e})")
    return best_period


def find_true_root(c_init, period, max_iters=80):
    """Find the true nucleus c* to very high precision using Newton at ROOT_DPS."""
    mpmath.mp.dps = ROOT_DPS
    c = mpmath.mpc(mpmath.re(c_init), mpmath.im(c_init))

    for it in range(max_iters):
        z, dzdc, _ = iterate_orbit(c, period)
        step = newton_step(z, dzdc)
        if step is None:
            break
        c = c - step
        step_mag = float(abs(step))
        if step_mag < mpmath.mpf(10) ** (-(ROOT_DPS - 20)):
            break

    mpmath.mp.dps = WORK_DPS
    return c


def run_convergence_test(c_init, c_true, period, method, max_iters=30):
    """Run NR iterations and measure convergence.

    method: 'newton', 'halley_full', 'halley_53'
    Returns list of dicts with per-step diagnostics.
    """
    mpmath.mp.dps = WORK_DPS
    c = mpmath.mpc(mpmath.re(c_init), mpmath.im(c_init))
    results = []

    for it in range(max_iters):
        # Evaluate orbit
        orbit_mode = {
            'newton': 'full',
            'halley_full': 'full',
            'halley_fp64': 'fp64',
            'halley_fp32': 'fp32',
            'halley_53': 'fp64',
        }[method]
        z, dzdc, d2 = iterate_orbit_quantized_d2(c, period, orbit_mode)

        # Metrics BEFORE taking the step
        err_c = float(abs(c - c_true))
        correct_bits = -math.log2(err_c) if err_c > 0 else float('inf')
        rho2_exp = compute_rho2_exp(z, dzdc, d2)
        norm_bits = (-rho2_exp / 2) if rho2_exp < 0 else 0
        want_halley = (method != 'newton') and (rho2_exp <= HALLEY_RHO2_EXP_THRESHOLD)

        # Compute step
        if method == 'newton':
            step = newton_step(z, dzdc)
        else:
            if want_halley:
                step = halley_step(z, dzdc, d2)
            else:
                step = newton_step(z, dzdc)
            if step is None:
                step = newton_step(z, dzdc)  # fallback

        step_norm = float(abs(step)) if step is not None else 0
        step_bits = -math.log2(step_norm) if step_norm > 0 else float('inf')

        results.append({
            'step': it,
            'correct_bits': correct_bits,
            'norm_bits': norm_bits,
            'rho2_exp': rho2_exp,
            'step_bits': step_bits,
            'want_halley': want_halley,
            'method': method,
        })

        if step is None:
            break
        c = c - step

        # Stop if converged
        if correct_bits > WORK_DPS * 3:
            break

    return results


def make_start_from_bits(c_true, start_bits):
    perturbation = mpmath.mpf(2) ** (-start_bits)
    return c_true + mpmath.mpc(perturbation, perturbation * mpmath.mpf('0.7'))


def iterations_to_target(results, target_bits):
    for r in results:
        if r['correct_bits'] >= target_bits:
            return r['step']
    return None


def compute_method_runs(c_true, period, start_bits_list, methods, max_iters=12):
    runs = {}
    for start_bits in start_bits_list:
        c_start = make_start_from_bits(c_true, start_bits)
        method_runs = {}
        for method in methods:
            method_runs[method] = run_convergence_test(c_start, c_true, period, method, max_iters=max_iters)
        runs[start_bits] = method_runs
    return runs


def print_savings_matrix(title, runs, halley_method, target_bits_list):
    print("\n" + "=" * 120)
    print(title)
    print("Cell value = Newton iterations - Halley iterations (positive means Halley saved iterations)")
    print("NA = target not reached within the configured iteration budget")
    print("=" * 120)

    header = "start\\target"
    for target_bits in target_bits_list:
        header += f" {target_bits:>6}"
    print(header)
    print("-" * len(header))

    csv_rows = ["start_bits,target_bits,newton_iters,halley_iters,savings"]

    for start_bits, method_runs in runs.items():
        row = f"{start_bits:>12}"
        newton_results = method_runs['newton']
        halley_results = method_runs[halley_method]
        for target_bits in target_bits_list:
            newton_iters = iterations_to_target(newton_results, target_bits)
            halley_iters = iterations_to_target(halley_results, target_bits)
            if newton_iters is None or halley_iters is None:
                cell = "NA"
                savings = "NA"
            else:
                savings = newton_iters - halley_iters
                cell = str(savings)
            row += f" {cell:>6}"
            csv_rows.append(f"{start_bits},{target_bits},{newton_iters},{halley_iters},{savings}")
        print(row)

    print("\nCSV")
    for line in csv_rows:
        print(line)


def print_gate_summary(title, runs, methods):
    print("\n" + "=" * 120)
    print(title)
    print("first_halley_step = first outer iteration with want_halley=true; gate_hits = count of Halley-enabled steps")
    print("=" * 120)
    print(f"{'start_bits':>10} | {'method':>12} | {'first_halley_step':>17} | {'gate_hits':>8}")
    print("-" * 60)

    for start_bits, method_runs in runs.items():
        for method in methods:
            results = method_runs[method]
            gate_steps = [r['step'] for r in results if r.get('want_halley')]
            first_halley_step = gate_steps[0] if gate_steps else "never"
            print(f"{start_bits:>10} | {method:>12} | {str(first_halley_step):>17} | {len(gate_steps):>8}")


def print_table(results_list, labels):
    """Print side-by-side convergence table."""
    max_len = max(len(r) for r in results_list)

    # Header
    hdr = f"{'Step':>4}"
    for label in labels:
        hdr += f" | {label + ' cbits':>18} {'ratio':>6} {'nbits':>8} {'ratio':>6} {'step_bits':>10}"
    print(hdr)
    print("-" * len(hdr))

    for i in range(max_len):
        row = f"{i:>4}"
        for results in results_list:
            if i < len(results):
                r = results[i]
                cb = r['correct_bits']
                nb = r['norm_bits']
                sb = r['step_bits']
                # Compute ratios
                if i > 0 and i - 1 < len(results):
                    prev = results[i - 1]
                    cb_ratio = cb / prev['correct_bits'] if prev['correct_bits'] > 0 else 0
                    nb_ratio = nb / prev['norm_bits'] if prev['norm_bits'] > 0 else 0
                else:
                    cb_ratio = 0
                    nb_ratio = 0
                row += f" | {cb:>18.1f} {cb_ratio:>6.2f} {nb:>8.1f} {nb_ratio:>6.2f} {sb:>10.1f}"
            else:
                row += f" | {'':>18} {'':>6} {'':>8} {'':>6} {'':>10}"
        print(row)


def main():
    mpmath.mp.dps = WORK_DPS

    # Bounding box center
    cx_str = "-0.5482057480704757084582125675467330293766992746228824538244448345949959996808952912997250594737971848370675761401078475"
    cy_str = "-0.577570838903603842805108982201850558675551728458255317158378952895736909832155423619018056768780831661920819731983179"

    c_init = mpmath.mpc(mpmath.mpf(cx_str), mpmath.mpf(cy_str))
    print(f"Initial c: ({cx_str[:60]}..., {cy_str[:60]}...)")

    # Step 1: Find period
    print("\n--- Period Detection ---")
    period = find_period(c_init)
    if period is None:
        print("Could not find period. Exiting.")
        sys.exit(1)
    print(f"Using period: {period}")

    # Step 2: Find true root at high precision
    print("\n--- Computing True Root ---")
    c_true = find_true_root(c_init, period)
    err0 = float(abs(c_init - c_true))
    print(f"Initial error: {err0:.6e} ({-math.log2(err0):.1f} bits)")

    # Step 3: Run convergence tests
    print("\n--- Newton Convergence ---")
    r_newton = run_convergence_test(c_init, c_true, period, 'newton')

    print("\n--- Halley (full-precision F'') Convergence ---")
    r_halley_full = run_convergence_test(c_init, c_true, period, 'halley_full')

    print("\n--- Halley (53-bit truncated F'') Convergence ---")
    r_halley_53 = run_convergence_test(c_init, c_true, period, 'halley_53')

    # Step 4: Print results
    print("\n" + "=" * 120)
    print("CONVERGENCE COMPARISON")
    print(f"Period: {period}")
    print(f"cbits = correct bits (-log2|c-c*|), ratio = cbits[k]/cbits[k-1]")
    print(f"nbits = normalized bits (-rho2_exp/2), ratio = nbits[k]/nbits[k-1]")
    print("=" * 120)

    print_table(
        [r_newton, r_halley_full, r_halley_53],
        ['Newton', 'Halley-Full', 'Halley-53']
    )

    # Step 5: Summary
    print("\n--- Summary ---")
    for label, results in [('Newton', r_newton),
                           ('Halley-Full', r_halley_full),
                           ('Halley-53', r_halley_53)]:
        if len(results) >= 3:
            ratios = []
            for i in range(1, len(results)):
                prev_cb = results[i - 1]['correct_bits']
                curr_cb = results[i]['correct_bits']
                if prev_cb > 0 and curr_cb < float('inf'):
                    ratios.append(curr_cb / prev_cb)
            if ratios:
                early = ratios[:3]
                late = ratios[-3:] if len(ratios) > 3 else ratios
                print(f"{label}:")
                print(f"  Early ratios (steps 1-3): {['%.2f' % r for r in early]}")
                print(f"  Late ratios  (last 3):    {['%.2f' % r for r in late]}")

    # Step 6: Test at several starting precisions to find the cubic burst
    print("\n" + "=" * 120)
    print("VARIABLE-PRECISION START TEST")
    print("Try progressively closer starting points to find the cubic burst.")
    print("=" * 120)

    for start_bits in [120, 100, 80, 60, 50, 40]:
        c_test = make_start_from_bits(c_true, start_bits)
        err_test = float(abs(c_test - c_true))
        actual_bits = -math.log2(err_test)
        print(f"\n--- Start at ~{start_bits} bits (actual: {actual_bits:.1f} bits) ---")

        r_n = run_convergence_test(c_test, c_true, period, 'newton', max_iters=10)
        r_hf = run_convergence_test(c_test, c_true, period, 'halley_full', max_iters=10)
        r_h53 = run_convergence_test(c_test, c_true, period, 'halley_fp64', max_iters=10)

        # Quick check: did it converge?
        if len(r_n) > 1 and r_n[1]['correct_bits'] > r_n[0]['correct_bits']:
            print_table([r_n, r_hf, r_h53], ['Newton', 'Halley-Full', 'Halley-53'])
        else:
            print(f"  Did not converge (outside basin of convergence)")

    # Step 7: Period-3 validation (the documented case)
    print("\n" + "=" * 120)
    print("PERIOD-3 VALIDATION (should match documented table)")
    print("=" * 120)
    c3_init = mpmath.mpc('-0.1226', '0.7449')  # near period-3 nucleus
    c3_true = find_true_root(c3_init, 3)
    err3 = float(abs(c3_init - c3_true))
    print(f"Period-3 nucleus: ({mpmath.nstr(mpmath.re(c3_true), 20)}, {mpmath.nstr(mpmath.im(c3_true), 20)})")
    print(f"Initial error: {err3:.6e} ({-math.log2(err3):.1f} bits)")

    r3_n = run_convergence_test(c3_init, c3_true, 3, 'newton')
    r3_hf = run_convergence_test(c3_init, c3_true, 3, 'halley_full')
    r3_h53 = run_convergence_test(c3_init, c3_true, 3, 'halley_fp64')

    print_table([r3_n, r3_hf, r3_h53], ['Newton', 'Halley-Full', 'Halley-53'])

    # Show |F''/F'| for period-3
    z3r, dzdc3r, d23r = iterate_orbit(c3_true, 3)
    f_prime3 = abs(dzdc3r)
    f_dprime3 = abs(d23r)
    ratio3 = float(f_dprime3 / f_prime3)
    print(f"\nPeriod-3: |F''/F'| = {ratio3:.6e} = 2^{math.log2(ratio3):.1f}")

    # Show |F''/F'| for the main test
    print(f"\n--- F''/F' ratio at the root ---")
    z_root, dzdc_root, d2_root = iterate_orbit(c_true, period)
    f_prime = abs(dzdc_root)
    f_dprime = abs(d2_root)
    ratio = float(f_dprime / f_prime)
    print(f"Period-{period}: |F''/F'| = {ratio:.6e} = 2^{math.log2(ratio):.1f}")
    print(f"This offset explains why normalized_bits = correct_bits - {math.log2(ratio):.1f}")

    # Step 8: regime maps for CPU-like and GPU-like low-precision Halley
    sweep_methods = ['newton', 'halley_fp64', 'halley_fp32']
    print("\n" + "=" * 120)
    print("HALLEY VALUE-REGION SWEEPS")
    print("=" * 120)

    runs_p3 = compute_method_runs(c3_true, 3, START_BITS_SWEEP, sweep_methods, max_iters=12)
    runs_p16045 = compute_method_runs(c_true, period, START_BITS_SWEEP, sweep_methods, max_iters=12)

    print_savings_matrix("Period-3 CPU-like sweep (Halley fp64 vs Newton)", runs_p3, 'halley_fp64', TARGET_BITS_SWEEP)
    print_savings_matrix("Period-3 GPU-like sweep (Halley fp32 vs Newton)", runs_p3, 'halley_fp32', TARGET_BITS_SWEEP)
    print_savings_matrix(f"Period-{period} CPU-like sweep (Halley fp64 vs Newton)", runs_p16045, 'halley_fp64', TARGET_BITS_SWEEP)
    print_savings_matrix(f"Period-{period} GPU-like sweep (Halley fp32 vs Newton)", runs_p16045, 'halley_fp32', TARGET_BITS_SWEEP)

    print_gate_summary("Gate activation summary: period-3", runs_p3, ['halley_fp64', 'halley_fp32'])
    print_gate_summary(f"Gate activation summary: period-{period}", runs_p16045, ['halley_fp64', 'halley_fp32'])


if __name__ == '__main__':
    main()
