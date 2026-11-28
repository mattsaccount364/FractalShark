module

import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Select
import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.PeriodicityWitness
import FractalSharkProofs.ReferenceOrbit.MT9.ViewportTransfer
meta import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Select
meta import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Budget
meta import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.PeriodicityWitness
meta import FractalSharkProofs.ReferenceOrbit.SingleThreaded.RealAxis
meta import FractalSharkProofs.MPIR
import Mathlib.Tactic.NormNum

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

private def workedRequest (x : ℚ) (cpu : CPU) (last : ℕ) : Request :=
  let center : Parameter := ⟨x, 0⟩
  let sampling := Sampling.ofZoom center (2 ^ (74 : ℕ)) 5120 2160 2
    (by norm_num) (by decide) (by decide) (by decide)
  ⟨center, sampling, 1 / 4, by norm_num, cpu, last, 16, by decide⟩

private def exampleRequest := workedRequest (-2 + 1 / 2 ^ (73 : ℕ))

private def zoomBounds (width height : ℕ) (hw : 0 < width) (hh : 0 < height) :
    ℚ × ℚ × ℚ × ℚ :=
  let sampling := Sampling.ofZoom ⟨3, -1⟩ 2 width height 2 (by decide) hw hh (by decide)
  (sampling.xMin, sampling.xMax, sampling.yMin, sampling.yMax)

#guard zoomBounds 4 4 (by decide) (by decide) == (2, 4, -2, 0)
#guard zoomBounds 6 4 (by decide) (by decide) == (3 / 2, 9 / 2, -2, 0)
#guard zoomBounds 4 6 (by decide) (by decide) == (2, 4, -5 / 2, 1 / 2)

private def referenceBits (r : Request) (maxBits : ℕ) : Except Failure Nat := do
  let answer ← estimate r maxBits
  return (← answer.reference).setting.requestedBits

private def allSettings (r : Request) : Except Failure (Nat × Nat × Nat) := do
  let answer ← estimate r 4096
  return ((← answer.reference).setting.requestedBits, (← answer.ideal).setting.requestedBits,
    (← answer.combined).setting.requestedBits)

#guard (referenceBits (exampleRequest .st 80) 4096).toOption == some 256
#guard (referenceBits (exampleRequest .mt3 80) 4096).toOption == some 256
#guard match allSettings (exampleRequest .st 80) with
  | .ok (reference, ideal, both) => reference == 256 && ideal == 448 && both == 448
  | .error _ => false

#guard match allSettings (workedRequest (-2 - 5 / 2 ^ (193 : ℕ)) .st 99) with
  | .ok (reference, ideal, both) => reference == 320 && ideal == 576 && both == 576
  | .error _ => false

#guard match referenceBits (exampleRequest .st 80) 192 with
  | .error .precisionLimit => true
  | _ => false

#guard (referenceBits (exampleRequest .st 0) 4096).toOption == some 64
#guard (referenceBits (exampleRequest .st 1) 4096).toOption == some 128

private def unequalSampling : Sampling :=
  ⟨-1, 3, 0, 1, 4, 2, 2, by decide, by decide, by decide, by decide, by decide⟩

#guard unequalSampling.tolerance (1 / 4) == 1 / 16
#guard (seed (MPIR.Precision.ofRequestedBits 64) (1 / 10)).isSome
#guard (seed (MPIR.Precision.ofRequestedBits 128) (-1 / 10)).isSome
#guard (seed (MPIR.Precision.ofRequestedBits 64) 0).isSome

#guard match estimate { exampleRequest .st 16 with center := ⟨100, 0⟩ } 64 with
  | .error .enclosureLimit => true
  | _ => false

#guard PeriodicityWitness.search [200, 198, 196] 80 == some (200, 77)
#guard PeriodicityWitness.requestedBits 200 == 317
#guard (PeriodicityWitness.precision 200).bits == 320

private def mt9ViewportRequest : Request :=
  let center : Parameter := ⟨-2 + 1 / (2 : ℚ) ^ (200 : ℕ), 0⟩
  let sampling := Sampling.ofZoom center (2 ^ (200 : ℕ)) 5120 2160 2
    (by norm_num) (by decide) (by decide) (by decide)
  ⟨center, sampling, 1 / 4, by norm_num, .mt9, 99, 16, by decide⟩

/-- The native FixedViewportBudget test bounds each drift by the same
dyadic cap. Comparing with the peak covers every published index by
`Profile.referenceErrorAt_le_peak`. -/
private def mt9ViewportFits (drift : ℚ) : Bool := Id.run do
  let request := mt9ViewportRequest
  let some p := profile request 4096 | return false
  let .ok fixed := recommend request p.referenceFactor 4096 | return false
  let bits := fixed.setting.precision.bits
  return decide (bits = 448 ∧ fixed.x.limbs.rational = request.center.x ∧
    fixed.y.limbs.rational = request.center.y ∧
    p.referenceFactor / (2 : ℚ) ^ bits + 2 * drift ≤ request.tolerance)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem mt9ViewportFixture : mt9ViewportFits (1 / (2 : ℚ) ^ (216 : ℕ)) = true := by decide +kernel

#guard !mt9ViewportFits mt9ViewportRequest.tolerance

#print axioms enclose_sound
#print axioms Selection.smallest
#print axioms Recommendation.copy_exact
#print axioms Recommendation.reference_correct
#print axioms Recommendation.reference_error_at
#print axioms Recommendation.ideal_correct
#print axioms SingleThreaded.realAxis_step
#print axioms PeriodicityWitness.checked_failure
#print axioms PeriodicityWitness.entered_check
#print axioms PeriodicityWitness.fitted_check
#print axioms mt9ViewportFixture
#print axioms Profile.referenceErrorAt_le_peak
#print axioms MT9.viewport_of_recommendation
#print axioms MT9.fixed_sufficiency_alone

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate
