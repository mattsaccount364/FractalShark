module

import FractalSharkProofs.ReferenceOrbit.MT9.FixedAllowance
import FractalSharkProofs.ReferenceOrbit.MT9.Certificate
import FractalSharkProofs.ReferenceOrbit.MT9.RadiusWitness
import Mathlib.Tactic.NormNum

namespace FractalSharkProofs.ReferenceOrbit.MT9

private def full : MPIR.Precision := .ofRequestedBits 320

example : lowWidth full = 3 ∧ highWidth full = 4 := by decide
example : lowWidth (workingPrecision full true (some 1)) = 1 ∧
    highWidth (workingPrecision full true (some 1)) = 2 := by decide
example : (workingPrecision full true (some 200)).bits = 320 := by decide
example : (workingPrecision full false (some 1)).bits = 320 := by decide

-- The first high slice is occupied and both low slices cover all their limbs.
example : assemble 2 (split 2 (MPIR.radix ^ 2 + MPIR.radix ^ 2 - 1))
    (split 2 (MPIR.radix ^ 2 - 1)) =
    (MPIR.radix ^ 2 + MPIR.radix ^ 2 - 1) * (MPIR.radix ^ 2 - 1) :=
  assembled_product _ _ _

example : HappensBefore (.multiply 4 .xSum) (.prepare 5) :=
  workers_before_reuse (lead := .xHigh) (by decide)
example : HappensBefore (.multiply 5 .xyLow) (.shutdown 5) :=
  workers_before_shutdown (lead := .xyHigh) (by decide)
example : ∀ observed, 2 * 5 + 1 ≤ 2 * (observed + 1) → observed ≤ 5 → observed = 5 :=
  fun _ => current_idle

private noncomputable def signals : ShadowChecks :=
  ⟨fun _ => false, fun n _ => n == 1, fun _ d => d, fun _ => false⟩

example : (run .mt9 .enabled signals 0).samples .mt9 = [0] := rfl
example : (run .mt9 .enabled signals 0).coherentThrough .mt9 = 0 := rfl
example : checkedStates (run .mt9 .enabled signals 1) = [1] := rfl
example : (run .mt9 .enabled signals 1).samples .mt9 = [0] := rfl
example : (run .mt9 .enabled signals 4).samples .mt9 = [0, 1] := rfl
example : (run .mt9 .enabled signals 4).coherentThrough .mt9 = 3 := rfl
example : checkedStates (run .mt9 .enabled signals 4) = [1, 2] := rfl
example : (run .mt9 .benchmark signals 4).samples .mt9 = [0, 1, 2, 3] := rfl
example : (run .mt9 .disabled signals 4).samples .mt9 = [0, 1, 2, 3] := rfl
example : (run .mt9 .benchmark { signals with bailout := fun n => n == 1 } 4).reason =
    .bailout := rfl
example : (run .mt9 .enabled { signals with abortBefore := fun _ => true } 4).samples .mt9 =
    [0] := rfl
example : (run .mt9 .benchmark { signals with abortBefore := fun n => n == 2 } 4).samples .mt9 =
    [0, 1] := rfl

private def stoppedSamples (tail : ℕ) (n : ℕ) : ℕ := if n ≤ 1 then n else tail

private def stoppedChecks (tail : ℕ) (nearReturn bailout : ℕ → Bool) : ShadowChecks :=
  shadowChecksFromSamples (stoppedSamples tail) (fun _ => false) (fun z _ => nearReturn z)
    (fun _ d => d) bailout

-- Both runs can differ in their lookahead without changing the exit that
-- publishes only Z0 and Z1.
example : run .mt9 .enabled (stoppedChecks 2 (· == 1) (fun _ => false)) 4 =
    run .mt9 .enabled (stoppedChecks 3 (· == 1) (fun _ => false)) 4 := by
  apply same_exit_of_sample_equivalence
  intro n hn
  change n ∈ ([0, 1] : List ℕ) at hn
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with rfl | rfl <;> rfl

example : run .mt9 .benchmark (stoppedChecks 2 (fun _ => false) (· == 1)) 4 =
    run .mt9 .benchmark (stoppedChecks 3 (fun _ => false) (· == 1)) 4 := by
  apply same_exit_of_sample_equivalence
  intro n hn
  change n ∈ ([0, 1] : List ℕ) at hn
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hn
  rcases hn with rfl | rfl <;> rfl

example : fastTail (2 ^ (63 : ℕ) + 2047) (2 ^ (64 : ℕ) - 1) = 2 ^ (32 : ℕ) - 1 := by decide
example : fastTail (2 ^ (63 : ℕ)) 0 = 0 := by decide
example : CertificateStatus.unsupported.accepted = false := rfl
example : CertificateStatus.invalid.accepted = false := rfl
example : (publish ((⟨none⟩ : Diagnostics).record 3 .unsupported) 9 .ambiguous (42 : ℕ)) =
    (⟨some (3, .unsupported)⟩, 42) := rfl

#print axioms assembled_product
#print axioms product_of_store
#print axioms batch_fits
#print axioms normalize_product
#print axioms follower_run_before_wait
#print axioms current_idle
#print axioms workers_before_reuse
#print axioms exact_at_observed
#print axioms error_at_observed
#print axioms viewportPrefix
#print axioms MPIR.add_span_bounds
#print axioms clip_error
#print axioms dynamic_local_error
#print axioms fixed_rounding_allowance
#print axioms disk_prefix
#print axioms dynamic_viewport_bound
#print axioms fastTail_two_words
#print axioms fast_gaps
#print axioms regular_gaps
#print axioms sample_equivalence
#print axioms accepted_prefix_equivalence
#print axioms same_exit_of_sample_equivalence
#print axioms publish_sample
#print axioms real_step
#print axioms real_clip
#print axioms RadiusWitness.fixed448_check
#print axioms RadiusWitness.dynamic448_check
#print axioms RadiusWitness.check704
#print axioms RadiusWitness.counterexample
#print axioms RadiusWitness.same_radius_failure
#print axioms RadiusWitness.meets_at_recommendation

end FractalSharkProofs.ReferenceOrbit.MT9
