module

import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Schedule
import FractalSharkProofs.ReferenceOrbit.MT3.Precision
import FractalSharkProofs.ReferenceOrbit.MT3.Mailbox
import FractalSharkProofs.ReferenceOrbit.ViewportPrecision
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-! Source timing regressions for authoritative generation. -/

namespace FractalSharkProofs.ReferenceOrbit

private noncomputable def signals : ShadowChecks where
  abortBefore := fun _ => false
  nearReturn := fun n _ => n == 2
  advanceDerivative := fun _ d => d + 1
  bailout := fun _ => false

example : (run .singleThreaded .disabled signals 0).samples .singleThreaded = [0] := rfl
example : (run .mt3 .disabled signals 0).samples .mt3 = [0] := rfl
example : (run .singleThreaded .disabled signals 1).samples .singleThreaded = [0, 1] := rfl
example : (run .mt3 .disabled signals 1).samples .mt3 = [0] := rfl

example : (run .singleThreaded .disabled signals 4).samples .singleThreaded = [0, 1, 2, 3, 4] := rfl
example : (run .mt3 .disabled signals 4).samples .mt3 = [0, 1, 2, 3] := rfl
example : (run .singleThreaded .enabled signals 4).samples .singleThreaded = [0, 1, 2] := rfl
example : (run .mt3 .enabled signals 4).samples .mt3 = [0, 1, 2] := rfl
example : (run .singleThreaded .enabled signals 4).maybePeriod .singleThreaded = some 3 := rfl
example : (run .mt3 .enabled signals 4).maybePeriod .mt3 = some 3 := rfl
example : (run .singleThreaded .enabled signals 4).coherentThrough .singleThreaded = 2 := rfl
example : (run .mt3 .enabled signals 4).coherentThrough .mt3 = 3 := rfl

-- Benchmark mode passes the near return without advancing that derivative.
example : (run .singleThreaded .benchmark signals 4).reason = .limit := rfl
example : (run .mt3 .benchmark signals 4).reason = .limit := rfl
example : (run .singleThreaded .benchmark signals 4).derivative = ((1 + 1) + 1) + 1 := rfl
example : (run .mt3 .benchmark signals 4).derivative = (1 + 1) + 1 := rfl

example : (run .singleThreaded .disabled {signals with bailout := fun i => i == 1} 4).coherentThrough
    .singleThreaded = 2 := rfl
example : (run .mt3 .disabled {signals with bailout := fun i => i == 1} 4).coherentThrough
    .mt3 = 2 := rfl
example : (run .mt3 .disabled {signals with abortBefore := fun i => i == 1} 4).samples
    .mt3 = [0] := rfl

example : MT3.exactStep (⟨1, 1⟩ : ℂ) (⟨1, 3⟩ : ℂ) = ⟨-7, 7⟩ := by
  apply Complex.ext <;> norm_num [MT3.exactStep]

-- The synthetic zero stays exact, and the first state inherits parameter error.
example : affineBound 2 (1 / 8) (1 / 4) 0 = 0 := rfl
example : affineBound 2 (1 / 8) (1 / 4) 1 = 1 / 8 := rfl

-- At amplification one the geometric sum counts the steps after initialization.
example : affineBound 1 (1 / 8) (1 / 4) (3 + 1) = 7 / 8 := by
  rw [affineBound_geometric]
  norm_num

-- The manual's matched 40-index comparison uses exact rational thresholds.
example : toleranceBound 1 (1 / 2 ^ 20) 0 (11 / 2 ^ 64) 40 ≤ (1 : ℝ) / 2 ^ 20 := by
  norm_num [toleranceBound, affineBound]

example : (1 : ℝ) / 2 ^ 20 < toleranceBound 1 (1 / 2 ^ 20) 0 (34 / 2 ^ 64) 40 := by
  norm_num [toleranceBound, affineBound]

-- The standard policy crosses a limb boundary between these two span exponents.
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-72) (-72) false)).limbs = 4 := rfl
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-73) (-73) false)).limbs = 5 := rfl

example : mpirViewportBound (MPIR.Precision.ofRequestedBits 64) 32 1 1 1 1 (1 / 8) 0 = 0 := rfl

example : mpirViewportBound (MPIR.Precision.ofRequestedBits 64) 32 1 1 1 1 (1 / 8) 1 =
    (1 : ℝ) / 8 + 1 / 2 ^ 64 := by
  norm_num [mpirViewportBound, toleranceBound, affineBound, MPIR.Precision.errorUnit_eq,
    MPIR.Precision.bits, MPIR.Precision.ofRequestedBits]

-- Once the input error uses the whole allowance, an additional positive
-- rounding contribution prevents the sufficient check from passing.
example (p : MPIR.Precision) :
    ¬ ((1 : ℝ) / 8 + p.errorUnit * 1 ≤ 1 / 8) := by
  rw [roundoffBudget_le_iff]
  norm_num

private noncomputable def unequalPitchView : Viewport where
  xMin := -1
  xMax := 3
  yMin := 0
  yMax := 1
  width := 4
  height := 2
  samplesPerAxis := 2
  x_pos := by norm_num
  y_pos := by norm_num
  width_pos := by decide
  height_pos := by decide
  samples_pos := by decide

example : unequalPitchView.sampleBudget (1 / 4) = 1 / 16 := by
  norm_num [Viewport.sampleBudget, Viewport.pitchX, Viewport.pitchY, Viewport.spanX, Viewport.spanY,
    Viewport.columns, Viewport.rows, unequalPitchView]

example : unequalPitchView.coverage ⟨4, -1⟩ = 7 := by
  norm_num [Viewport.coverage, unequalPitchView]

private noncomputable def radiusExample : ℝ := 1 / 2 ^ 73

private noncomputable def viewportExample : Viewport where
  xMin := -1 - 2 * radiusExample
  xMax := -1 + 2 * radiusExample
  yMin := -radiusExample
  yMax := radiusExample
  width := 800
  height := 400
  samplesPerAxis := 2
  x_pos := by norm_num [radiusExample]
  y_pos := by norm_num [radiusExample]
  width_pos := by decide
  height_pos := by decide
  samples_pos := by decide

example : viewportExample.Contains (viewportExample.sample 1599 799) :=
  viewportExample.sample_contains (by decide) (by decide)

example : viewportExample.pitchX = viewportExample.pitchY := by
  norm_num [Viewport.pitchX, Viewport.pitchY, Viewport.spanX, Viewport.spanY, Viewport.columns,
    Viewport.rows, viewportExample, radiusExample]

example : viewportExample.sampleBudget (1 / 4) = radiusExample / 1600 := by
  norm_num [Viewport.sampleBudget, Viewport.pitchX, Viewport.pitchY, Viewport.spanX, Viewport.spanY,
    Viewport.columns, Viewport.rows, viewportExample, radiusExample]

example : cpuRequestedBits (-70) (-71) false = 191 := rfl
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) false)).bits = 192 := rfl
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) false)).limbs = 4 := rfl
example : cpuRequestedBits (-70) (-71) true = 871 := rfl
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) true)).bits = 896 := rfl
example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) true)).limbs = 15 := rfl

example : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) false)).errorUnit * 1 ≤
    viewportExample.sampleBudget (1 / 4) := by
  apply viewportExample.cpu_roundoff_le_budget (-70) (-71) false (1 / 4) 1
  · norm_num
  · norm_num [Viewport.spanX, viewportExample, radiusExample]
  · norm_num [Viewport.spanY, viewportExample, radiusExample]
  · norm_num [cpuRequestedBits, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits,
      Viewport.columns, Viewport.rows, viewportExample]

example : displacementBound 1 (1 / 2 ^ 8) (viewportExample.coverage (-1)) 40 ≤ 1 / 2 ^ 8 := by
  norm_num [displacementBound, affineBound, Viewport.coverage, viewportExample, radiusExample,
    Complex.neg_re, Complex.neg_im, Complex.one_re, Complex.one_im]

private noncomputable def exampleBound (bits : ℕ) (coefficient displacement : ℝ) (last : ℕ) : ℝ :=
  mpirViewportBound (MPIR.Precision.ofRequestedBits bits) coefficient 1 (1 + displacement)
    (1 / 2 ^ 8) 1 0 last

-- Three internal limbs suffice for both reference budgets and MT3 reconstruction.
example : exampleBound 128 32 0 40 ≤ min (1 / 2 ^ 8) (radiusExample / 1600) := by
  norm_num [exampleBound, mpirViewportBound, toleranceBound, affineBound,
    MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits, radiusExample]

example : exampleBound 128 9 0 40 ≤ min (1 / 2 ^ 8) (radiusExample / 1600) := by
  norm_num [exampleBound, mpirViewportBound, toleranceBound, affineBound,
    MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits, radiusExample]

example : exampleBound 128 9 (1 / 2 ^ 8) 40 ≤ min (1 / 2 ^ 8) (radiusExample / 1600) := by
  norm_num [exampleBound, mpirViewportBound, toleranceBound, affineBound,
    MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits, radiusExample]

-- The standard CPU selection gives the fourth limb needed by the ST check.
example : radiusExample / 1600 < exampleBound 128 32 (1 / 2 ^ 8) 40 := by
  norm_num [exampleBound, mpirViewportBound, toleranceBound, affineBound,
    MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits, radiusExample]

example : exampleBound (cpuRequestedBits (-70) (-71) false) 32 (1 / 2 ^ 8) 40 ≤
    min (1 / 2 ^ 8) (radiusExample / 1600) := by
  norm_num [exampleBound, mpirViewportBound, toleranceBound, affineBound, cpuRequestedBits,
    MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits, radiusExample]

-- At the same view scale and limb count, the input-copy term alone makes
-- the longer reference bound inconclusive.
example : radiusExample / 1600 < exampleBound (cpuRequestedBits (-70) (-71) false) 32 0 112 := by
  unfold exampleBound
  rw [mpirViewportBound_split]
  let a : ℝ := 2 * (1 + 0) + 1 / 2 ^ 8
  have hsum : 0 ≤ stepWeight a 112 := by
    unfold stepWeight
    apply Finset.sum_nonneg
    intro i _
    exact pow_nonneg (by norm_num [a]) i
  have hpow := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2)
    (show (2 : ℝ) ≤ a by norm_num [a]) 111
  have hweight : (2 : ℝ) ^ 111 ≤ viewportRoundingFactor 32 1 a 1 112 := by
    simp only [viewportRoundingFactor, inputWeight, show (112 : ℕ) ≠ 0 by decide,
      ↓reduceIte, show (112 : ℕ) - 1 = 111 by decide, mul_one]
    linarith only [hpow, hsum]
  have hu : (MPIR.Precision.ofRequestedBits (cpuRequestedBits (-70) (-71) false)).errorUnit =
      (1 : ℝ) / 2 ^ 192 := by
    norm_num [MPIR.Precision.errorUnit_eq, MPIR.Precision.bits, MPIR.Precision.ofRequestedBits,
      cpuRequestedBits]
  rw [hu, mul_zero, zero_add]
  exact (show radiusExample / 1600 < (1 / 2 ^ 192 : ℝ) * 2 ^ 111 by
    norm_num [radiusExample]).trans_le (mul_le_mul_of_nonneg_left hweight (by positivity))

#print axioms Viewport.cpu_roundoff_le_budget
#print axioms Viewport.error_le_budget_iff_radius
#print axioms Viewport.displacement_le_iff_radius
#print axioms errorUnit_more_limbs
#print axioms mpirViewportBound_split
#print axioms roundoffBudget_le_iff
#print axioms SingleThreaded.viewportPrefix
#print axioms MT3.viewportPrefix
#print axioms MPIRViewportPrefix.reference_error_in_samples
#print axioms MPIRViewportPrefix.perturbed_error_in_samples
#print axioms affineBound_geometric
#print axioms prefix_error_le_of_tolerance
#print axioms SingleThreaded.error_at_observed_of_tolerance
#print axioms MT3.error_at_observed_of_tolerance
#print axioms derivativeStep_eq
#print axioms runFrom_entered_bounds
#print axioms disabled_reason
#print axioms serial_samples
#print axioms delayed_samples
#print axioms SingleThreaded.exact_at_observed
#print axioms SingleThreaded.error_at_observed
#print axioms MT3.exactStep_eq
#print axioms MT3.exact_at_observed
#print axioms MT3.error_at_observed
#print axioms MT3.active_stable
#print axioms MT3.either_completion_order
#print axioms MT3.consume_then_lookahead
#print axioms MT3.drain_queued_lookahead
#print axioms MT3.Mailbox.original_unordered
#print axioms MT3.Mailbox.corrected_payload_visibility
#print axioms MT3.Mailbox.corrected_buffer_reuse

end FractalSharkProofs.ReferenceOrbit
