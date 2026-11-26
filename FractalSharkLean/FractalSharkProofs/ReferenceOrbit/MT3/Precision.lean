module

public import FractalSharkProofs.ReferenceOrbit.MT3.Exact
public import FractalSharkProofs.ReferenceOrbit.MT3.Workers
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Local MPIR errors for the expanded schedule and their propagation to visible states. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT3

/-- Squares arrive with their immutable jobs. Both copies into jobs and the
saved Y square use the same precision, so those assignments preserve values. -/
structure Step {p : MPIR.Precision} (epoch : ℕ) (c old next : State p) : Prop where
  run : ∃ (x : Square (xJob epoch old)) (y : Square (yJob epoch old))
    (cross doubled difference : Slot p),
    MPIR.Mul p old.x.val old.y.val cross.val ∧
    MPIR.MulUiTwo p cross.val doubled.val ∧
    MPIR.Add p doubled.val c.y.val next.y.val ∧
    MPIR.Sub p x.value.val y.value.val difference.val ∧
    MPIR.Add p difference.val c.x.val next.x.val

private theorem square_bounds {p : MPIR.Precision} {a square : MPIR.Limbs}
    (h : MPIR.Mul p a a square) :
    |square.value| ≤ |a.value| ^ 2 ∧
      |square.value - a.value ^ 2| ≤ 3 * p.errorUnit * |a.value| ^ 2 := by
  simpa only [pow_two, abs_mul] using MPIR.mul_bounds h

/-- The two products each contribute three units and the difference contributes
two. Since the error unit is at most one half, the sum of square magnitudes
bounds the error of the subtraction, and twice that sum bounds its output. -/
private theorem real_error {p : MPIR.Precision} {x y c xs ys difference result : MPIR.Limbs}
    (hx : MPIR.Mul p x x xs) (hy : MPIR.Mul p y y ys)
    (hsub : MPIR.Sub p xs ys difference) (hadd : MPIR.Add p difference c result) :
    |result.value - (x.value ^ 2 - y.value ^ 2 + c.value)| ≤
      p.errorUnit * (9 * (|x.value| ^ 2 + |y.value| ^ 2) + 2 * |c.value|) := by
  obtain ⟨hxs, hxe⟩ := square_bounds hx
  obtain ⟨hys, hye⟩ := square_bounds hy
  let size := |x.value| ^ 2 + |y.value| ^ 2
  let u := p.errorUnit
  have hsize : 0 ≤ size := add_nonneg (sq_nonneg _) (sq_nonneg _)
  have hu : 0 ≤ u := p.errorUnit_pos.le
  have hsum : |xs.value| + |ys.value| ≤ size := add_le_add hxs hys
  have hsubError := (sub_error_sum hsub).trans
    (mul_le_mul_of_nonneg_left hsum (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hu))
  have hdiff : |difference.value| ≤ 2 * size := by
    have h := abs_sub_le difference.value (xs.value - ys.value) 0
    simp only [sub_zero] at h
    have hhalf := mul_le_mul_of_nonneg_right p.errorUnit_le_half hsize
    nlinarith only [h, abs_sub xs.value ys.value, hsum, hsubError, hhalf]
  have haddError := add_error_sum hadd
  have haddSize := mul_le_mul_of_nonneg_left hdiff hu
  have hproducts : |(xs.value - ys.value) - (x.value ^ 2 - y.value ^ 2)| ≤
      3 * u * size := by
    rw [show (xs.value - ys.value) - (x.value ^ 2 - y.value ^ 2) =
      (xs.value - x.value ^ 2) - (ys.value - y.value ^ 2) by ring]
    have h := abs_sub (xs.value - x.value ^ 2) (ys.value - y.value ^ 2)
    change _ ≤ 3 * u * (|x.value| ^ 2 + |y.value| ^ 2)
    nlinarith only [h, hxe, hye]
  calc
    _ ≤ |result.value - (difference.value + c.value)| +
        |(difference.value + c.value) - (x.value ^ 2 - y.value ^ 2 + c.value)| :=
      abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub]
      have h := abs_sub_le difference.value (xs.value - ys.value) (x.value ^ 2 - y.value ^ 2)
      change _ ≤ u * (9 * (|x.value| ^ 2 + |y.value| ^ 2) + 2 * |c.value|)
      nlinarith only [haddError, haddSize, h, hsubError, hproducts]

/-- The cross product is rounded before it is doubled with the unsigned
integer operation. Its two roundoff sources precede the addition of c_y. -/
private theorem imag_error {p : MPIR.Precision} {x y c cross doubled result : MPIR.Limbs}
    (hmul : MPIR.Mul p x y cross) (hdouble : MPIR.MulUiTwo p cross doubled)
    (hadd : MPIR.Add p doubled c result) :
    |result.value - (2 * (x.value * y.value) + c.value)| ≤
      p.errorUnit * (12 * |x.value * y.value| + 2 * |c.value|) := by
  obtain ⟨hcross, hcrossErr⟩ := MPIR.mul_bounds hmul
  obtain ⟨hdoubleSize, hdoubleErr⟩ := MPIR.mulUiTwo_bounds hdouble
  have hu := p.errorUnit_pos.le
  have hcrossSize := mul_le_mul_of_nonneg_left hcross hu
  have hsize := mul_le_mul_of_nonneg_left hdoubleSize hu
  have haddErr := add_error_sum hadd
  calc
    _ ≤ |result.value - (doubled.value + c.value)| +
        |(doubled.value + c.value) - (2 * (x.value * y.value) + c.value)| := abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub]
      have h := abs_sub_le doubled.value (2 * cross.value) (2 * (x.value * y.value))
      rw [← mul_sub, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at h
      nlinarith only [h, hcrossErr, hcrossSize, hsize, hdoubleErr, haddErr]

theorem step_error {p : MPIR.Precision} {epoch : ℕ} {c old next : State p}
    (h : Step epoch c old next) :
    norm1 (next.value - (old.value ^ 2 + c.value)) ≤
      p.errorUnit * (9 * norm1 old.value ^ 2 + 2 * norm1 c.value) := by
  obtain ⟨x, y, cross, doubled, difference, hmul, hdouble, hy, hsub, hx⟩ := h.run
  have hre := real_error x.squared y.squared hsub hx
  have him := imag_error hmul hdouble hy
  change |next.x.val.value - (old.x.val.value ^ 2 - old.y.val.value ^ 2 + c.x.val.value)| ≤
    p.errorUnit * (9 * (|old.x.val.value| ^ 2 + |old.y.val.value| ^ 2) + 2 * |c.x.val.value|) at hre
  rw [abs_mul] at him
  have hnonneg := mul_nonneg p.errorUnit_pos.le
    (mul_nonneg (abs_nonneg old.x.val.value) (abs_nonneg old.y.val.value))
  rw [← exactStep_eq]
  simp only [norm1, Complex.sub_re, Complex.sub_im, exactStep, State.value]
  nlinarith only [hre, him, hnonneg]

noncomputable def errorBound (p : MPIR.Precision) (target working : ℂ) (w : ℕ → ℂ) : ℕ → ℝ :=
  prefixBound target working w (fun n => p.errorUnit * (9 * norm1 (w n) ^ 2 + 2 * norm1 working))

theorem error_at_observed {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit : ℕ)
    (target : ℂ) (working : State p) (states : ℕ → State p)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt3 mode checks limit).coherentThrough .mt3 →
      Step i working (states i) (states (i + 1)))
    {n : ℕ} (hn : n ∈ (run .mt3 mode checks limit).samples .mt3) :
    norm1 ((states n).value - orbit target n) ≤
      errorBound p target working.value (fun i => (states i).value) n :=
  prefixBound_correct target working.value (fun i => (states i).value) _
    ((run .mt3 mode checks limit).coherentThrough .mt3) hzero (firstState_eq hx hy)
    (fun i hi hl => step_error (hstep i hi hl)) n (sample_le_coherent _ _ hn)

/-- The MT3 coefficient applies to each coherent step before the final observed index. -/
theorem error_at_observed_of_tolerance {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit : ℕ)
    (target : ℂ) (working : State p) (states : ℕ → State p) (last : ℕ) (cap tolerance : ℝ)
    (hlast : last ∈ (run .mt3 mode checks limit).samples .mt3)
    (hcap : 0 ≤ cap) (htolerance : 0 ≤ tolerance)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (states n).value ≤ cap)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt3 mode checks limit).coherentThrough .mt3 →
      Step i working (states i) (states (i + 1)))
    (hfinal : mpirToleranceBound p 9 target working.value cap tolerance last ≤ tolerance) :
    ∀ n, n ≤ last → norm1 ((states n).value - orbit target n) ≤ tolerance := by
  have hcoherent := sample_le_coherent .mt3 (run .mt3 mode checks limit) hlast
  exact mpir_prefix_error_le_of_tolerance 9 target working states last cap tolerance
    (by norm_num) hcap htolerance hzero hx hy hsize
    (fun i hi hl => step_error (hstep i hi (hl.trans_le hcoherent))) hfinal

end FractalSharkProofs.ReferenceOrbit.MT3
