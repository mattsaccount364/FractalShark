module

import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Schedule
import FractalSharkProofs.ReferenceOrbit.MT3.Precision
import FractalSharkProofs.ReferenceOrbit.MT3.Mailbox
import Mathlib.Tactic.NormNum

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
