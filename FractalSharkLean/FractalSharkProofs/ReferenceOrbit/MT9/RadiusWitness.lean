module

public import FractalSharkProofs.ReferenceOrbit.MT9.RealAxis
public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Budget
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-! A finite real-axis witness for actual viewport error at a given starting precision. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9.RadiusWitness

open PrecisionEstimate

def center : Parameter := ⟨-2 + grid 400, 0⟩
def witnessIndex : ℕ := 91

def sampling : Sampling :=
  Sampling.ofZoom center ((2 : ℚ) ^ 450) 1024 1024 1
    (by positivity) (by decide) (by decide) (by decide)

def tolerance : ℚ := sampling.tolerance (1 / 4)

theorem tolerance_eq : tolerance = grid 460 := by decide +kernel

private theorem square_pitches :
    (sampling.xMax - sampling.xMin) / (sampling.antialiasing * sampling.width : ℕ) =
      (sampling.yMax - sampling.yMin) / (sampling.antialiasing * sampling.height : ℕ) :=
  by decide +kernel

theorem sample_error_iff (error : ℂ) :
    sampling.viewport.sampleError error ≤ (1 / 4 : ℝ) ↔ norm1 error ≤ (tolerance : ℝ) := by
  have hp : sampling.viewport.pitchX = sampling.viewport.pitchY := by
    simpa [Sampling.viewport, Viewport.pitchX, Viewport.pitchY, Viewport.spanX, Viewport.spanY,
      Viewport.columns, Viewport.rows] using congrArg (fun x : ℚ => (x : ℝ)) square_pitches
  have ht : (tolerance : ℝ) = sampling.viewport.sampleBudget (1 / 4) := by
    have ht := sampling.tolerance_eq (1 / 4)
    simpa only [tolerance, Rat.cast_div, Rat.cast_one, Rat.cast_ofNat] using ht
  rw [ht, Viewport.sampleError, Viewport.sampleBudget, hp, min_self, ← add_div,
    div_le_iff₀ sampling.viewport.pitchY_pos, norm1]

/-- The native test checks the working widths, starting at full precision
and narrowing by one limb on the step producing the witness. -/
def precisionAt (full : ℕ) (dynamic : Bool) (n : ℕ) : MPIR.Precision :=
  .ofRequestedBits (if dynamic && decide (witnessIndex ≤ n) then full - 64 else full)

def budget (full : ℕ) : ℚ := if full = 448 then grid 478 else tolerance

structure Replay where
  real : ℚ
  box : Box
  valid : Bool
  within : Bool

def fitsBox (x : ℚ) (box : Box) (ceiling : ℚ) : Bool :=
  decide (|x - box.x.lo| < ceiling ∧ |x - box.x.hi| < ceiling)

/-- Check the limb range and the exact error as the replay runs. Endpoints
of the mathematical orbit's rectangle round outward on a dyadic grid. -/
def replay (full : ℕ) (dynamic : Bool) : ℕ → Replay
  | 0 => ⟨0, .zero, true, fitsBox 0 .zero (budget full)⟩
  | 1 =>
    let box := Box.zero.next center 768
    ⟨center.x, box, true, fitsBox center.x box (budget full)⟩
  | n + 2 =>
    let prev := replay full dynamic (n + 1)
    let p := precisionAt full dynamic (n + 2)
    let x := realNext p center.x prev.real
    let box := prev.box.next center 768
    ⟨x, box, prev.valid && decide (nearTwo prev.real ∧ RealWindow p center.x (realClip p prev.real)),
      prev.within && fitsBox x box (budget full)⟩

private theorem valid_step (full : ℕ) (dynamic : Bool) :
    ∀ last i, 1 ≤ i → i < last → (replay full dynamic last).valid = true →
      nearTwo (replay full dynamic i).real ∧
      RealWindow (precisionAt full dynamic (i + 1)) center.x
        (realClip (precisionAt full dynamic (i + 1)) (replay full dynamic i).real) := by
  intro last
  induction last with
  | zero => intro i hi hl; omega
  | succ last ih =>
    intro i hi hl hvalid
    cases last with
    | zero => omega
    | succ last =>
      have h : (replay full dynamic (last + 1)).valid = true ∧
          nearTwo (replay full dynamic (last + 1)).real ∧
          RealWindow (precisionAt full dynamic (last + 2)) center.x
            (realClip (precisionAt full dynamic (last + 2)) (replay full dynamic (last + 1)).real) := by
        simpa only [replay, Bool.and_eq_true, decide_eq_true_eq] using hvalid
      by_cases he : i = last + 1
      · simpa only [he] using h.2
      · exact ih i hi (by omega) h.1

private theorem box_contains (full : ℕ) (dynamic : Bool) (n : ℕ) :
    (replay full dynamic n).box.Contains (orbit center.value n) := by
  have hz : Box.zero.Contains (orbit center.value 0) := by
    simp [Box.zero, Box.Contains, Interval.Contains, Interval.point, orbit]
  induction n with
  | zero => exact hz
  | succ n ih =>
    cases n with
    | zero => exact Box.next_contains hz center 768
    | succ n => exact Box.next_contains ih center 768

private theorem exact_imaginary_zero (n : ℕ) : (orbit center.value n).im = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [orbit, Complex.add_im, pow_two, Complex.mul_im, ih]
    simp [center, Parameter.value]

private theorem norm1_error_eq (x : ℚ) (n : ℕ) :
    norm1 (realValue x - orbit center.value n) = |(x : ℝ) - (orbit center.value n).re| := by
  simp [norm1, realValue, exact_imaginary_zero]

private theorem fits_at (full : ℕ) (dynamic : Bool) :
    ∀ last n, n ≤ last → (replay full dynamic last).within = true →
      fitsBox (replay full dynamic n).real (replay full dynamic n).box (budget full) = true := by
  intro last
  induction last with
  | zero =>
    intro n hn h
    have : n = 0 := by omega
    subst n
    exact h
  | succ last ih =>
    intro n hn h
    cases last with
    | zero =>
      cases n with
      | zero =>
        have hp : 0 < budget full := by
          unfold budget
          split
          · exact grid_pos _
          · rw [tolerance_eq]
            exact grid_pos _
        simpa [replay, fitsBox, Box.zero, Interval.point] using hp
      | succ n =>
        have : n = 0 := by omega
        subst n
        exact h
    | succ last =>
      have hboth : (replay full dynamic (last + 1)).within = true ∧
          fitsBox (replay full dynamic (last + 2)).real (replay full dynamic (last + 2)).box
            (budget full) = true := by
        change ((replay full dynamic (last + 1)).within &&
          fitsBox (replay full dynamic (last + 2)).real (replay full dynamic (last + 2)).box
            (budget full)) = true at h
        simpa only [Bool.and_eq_true] using h
      by_cases he : n = last + 2
      · simpa only [he] using hboth.2
      · exact ih n (by omega) hboth.1

private theorem error_lt_budget (full : ℕ) (dynamic : Bool) (n : ℕ) (hn : n ≤ witnessIndex)
    (h : (replay full dynamic witnessIndex).within = true) :
    norm1 (realValue (replay full dynamic n).real - orbit center.value n) < (budget full : ℝ) := by
  have hfit := fits_at full dynamic witnessIndex n hn h
  simp only [fitsBox, decide_eq_true_eq] at hfit
  have hlo : |((replay full dynamic n).real : ℝ) - (replay full dynamic n).box.x.lo| <
      (budget full : ℝ) := by exact_mod_cast hfit.1
  have hhi : |((replay full dynamic n).real : ℝ) - (replay full dynamic n).box.x.hi| <
      (budget full : ℝ) := by exact_mod_cast hfit.2
  have hbox := (box_contains full dynamic n).1
  rw [Interval.Contains] at hbox
  rw [norm1_error_eq, abs_lt]
  rw [abs_lt] at hlo hhi
  exact ⟨by linarith [hbox.2, hhi.1], by linarith [hbox.1, hlo.2]⟩

structure Trace (full : ℕ) (dynamic : Bool) where
  working : State (.ofRequestedBits full)
  states : (n : ℕ) → State (precisionAt full dynamic n)
  parameter : working.value = realValue center.x
  zero : (states 0).value = 0
  start : (states 1).value = working.value
  step : ∀ i, 1 ≤ i → i < witnessIndex →
    ∃ clipped : State (precisionAt full dynamic (i + 1)),
      Clip (states i) clipped ∧ Step working clipped (states (i + 1))

theorem Trace.matches_replay {full : ℕ} {dynamic : Bool} (trace : Trace full dynamic)
    (valid : (replay full dynamic witnessIndex).valid = true) :
    ∀ n, n ≤ witnessIndex → (trace.states n).value = realValue (replay full dynamic n).real := by
  intro n
  induction n with
  | zero =>
    intro _
    rw [trace.zero]
    simp [replay, realValue, Complex.zero_re, Complex.zero_im, Complex.ext_iff]
  | succ n ih =>
    intro hn
    cases n with
    | zero => exact trace.start.trans trace.parameter
    | succ n =>
      obtain ⟨clipped, hclip, hstep⟩ := trace.step (n + 1) (by omega) (by omega)
      obtain ⟨hnear, hwin⟩ := valid_step full dynamic witnessIndex (n + 1)
        (by omega) (by omega) valid
      exact real_step trace.parameter (real_clip (ih (by omega)) hnear hclip) hwin hstep

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem fixed448_check :
    (replay 448 false witnessIndex).valid = true ∧ (replay 448 false witnessIndex).within = true :=
  by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem dynamic448_check :
    (replay 448 true witnessIndex).valid = true ∧
      (replay 448 true witnessIndex).real + grid 447 < (replay 448 true witnessIndex).box.x.lo :=
  by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 8000000 in
theorem check704 (dynamic : Bool) :
    (replay 704 dynamic witnessIndex).valid = true ∧ (replay 704 dynamic witnessIndex).within = true := by
  cases dynamic <;> decide +kernel

/-- These are errors against the exact mathematical orbit. Publication
through index 91 and the tier trace are checked in the native regression. -/
theorem counterexample (fixed : Trace 448 false) (dynamic : Trace 448 true) :
    (∀ n, n ≤ witnessIndex →
      norm1 ((fixed.states n).value - orbit center.value n) < (grid 478 : ℝ)) ∧
    (grid 447 : ℝ) < norm1 ((dynamic.states witnessIndex).value - orbit center.value witnessIndex) := by
  refine ⟨?_, ?_⟩
  · intro n hn
    rw [fixed.matches_replay fixed448_check.1 n hn]
    rw [← show budget 448 = grid 478 by rfl]
    exact error_lt_budget 448 false n hn fixed448_check.2
  · rw [dynamic.matches_replay dynamic448_check.1 _ le_rfl, norm1_error_eq]
    have h := (box_contains 448 true witnessIndex).1.1
    have hgap : ((replay 448 true witnessIndex).real : ℝ) + grid 447 <
        (replay 448 true witnessIndex).box.x.lo := by exact_mod_cast dynamic448_check.2
    have habs := neg_le_abs (((replay 448 true witnessIndex).real : ℝ) -
      (orbit center.value witnessIndex).re)
    linarith

theorem same_radius_failure (fixed : Trace 448 false) (dynamic : Trace 448 true) :
    (∀ n, n ≤ witnessIndex →
      sampling.viewport.sampleError ((fixed.states n).value - orbit center.value n) ≤ 1 / 4) ∧
    1 / 4 < sampling.viewport.sampleError
      ((dynamic.states witnessIndex).value - orbit center.value witnessIndex) := by
  have hsmall : (grid 478 : ℝ) ≤ tolerance := by
    exact_mod_cast (show grid 478 ≤ tolerance by decide +kernel)
  have hlarge : (tolerance : ℝ) ≤ grid 447 := by
    exact_mod_cast (show tolerance ≤ grid 447 by decide +kernel)
  obtain ⟨hfixed, hdynamic⟩ := counterexample fixed dynamic
  refine ⟨fun n hn => (sample_error_iff _).2 ((hfixed n hn).le.trans hsmall), ?_⟩
  exact lt_of_not_ge fun h => (not_le_of_gt (hlarge.trans_lt hdynamic)) ((sample_error_iff _).1 h)

theorem meets_at_recommendation {dynamic : Bool} (trace : Trace 704 dynamic) (n : ℕ)
    (hn : n ≤ witnessIndex) :
    norm1 ((trace.states n).value - orbit center.value n) < (tolerance : ℝ) := by
  rw [trace.matches_replay (check704 dynamic).1 n hn]
  rw [← show budget 704 = tolerance by rfl]
  exact error_lt_budget 704 dynamic n hn (check704 dynamic).2

end FractalSharkProofs.ReferenceOrbit.MT9.RadiusWitness
