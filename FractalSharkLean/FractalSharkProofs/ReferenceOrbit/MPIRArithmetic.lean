module

public import FractalSharkProofs.MPIR
public import FractalSharkProofs.ReferenceOrbit.Error
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Shared CPU state and bounds for common MPIR operation sequences. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

/-- The parameter, live state and scratch buffers use the same default MPIR precision. -/
abbrev Slot (p : MPIR.Precision) := {a : MPIR.Limbs // a.used ≤ p.limbs + 1}

structure State (p : MPIR.Precision) where
  x : Slot p
  y : Slot p

noncomputable def State.value {p : MPIR.Precision} (state : State p) : ℂ :=
  ⟨state.x.val.value, state.y.val.value⟩

/-- The error introduced by copying a represented input into the working parameter. -/
theorem parameterCopy_error {p : MPIR.Precision} {c : State p} {x y : MPIR.Limbs}
    (hx : MPIR.Set p x c.x.val) (hy : MPIR.Set p y c.y.val) :
    norm1 (c.value - ⟨x.value, y.value⟩) ≤ p.errorUnit * (|x.value| + |y.value|) := by
  simp only [norm1, State.value, Complex.sub_re, Complex.sub_im, mul_add]
  exact add_le_add (MPIR.set_error hx) (MPIR.set_error hy)

/-- The second pair of assignments copies the working parameter into the live state. -/
theorem firstState_eq {p : MPIR.Precision} {c first : State p}
    (hx : MPIR.Set p c.x.val first.x.val) (hy : MPIR.Set p c.y.val first.y.val) :
    first.value = c.value := by
  apply Complex.ext
  · exact MPIR.set_exact_of_fits c.x.property hx
  · exact MPIR.set_exact_of_fits c.y.property hy

theorem add_error_sum {p : MPIR.Precision} {a b r : MPIR.Limbs}
    (h : MPIR.Add p a b r) :
    |r.value - (a.value + b.value)| ≤ 2 * p.errorUnit * (|a.value| + |b.value|) := by
  have hmax : max |a.value| |b.value| ≤ |a.value| + |b.value| := by
    exact max_le (by linarith only [abs_nonneg b.value]) (by linarith only [abs_nonneg a.value])
  exact (MPIR.add_error h).trans
    (mul_le_mul_of_nonneg_left hmax (mul_nonneg (by norm_num) p.errorUnit_pos.le))

theorem sub_error_sum {p : MPIR.Precision} {a b r : MPIR.Limbs}
    (h : MPIR.Sub p a b r) :
    |r.value - (a.value - b.value)| ≤ 2 * p.errorUnit * (|a.value| + |b.value|) := by
  simpa only [sub_eq_add_neg, MPIR.Limbs.value_neg, abs_neg] using add_error_sum h

/-- The product contributes three truncations. The addition contributes two more,
and uses a product whose magnitude can only have decreased. -/
theorem mulThenAdd_error {p : MPIR.Precision} {a b c product r : MPIR.Limbs}
    (hmul : MPIR.Mul p a b product) (hadd : MPIR.Add p product c r) :
    |r.value - (a.value * b.value + c.value)| ≤
      p.errorUnit * (5 * |a.value * b.value| + 2 * |c.value|) := by
  obtain ⟨hsize, hmulErr⟩ := MPIR.mul_bounds hmul
  have haddErr := add_error_sum hadd
  have hsize' := mul_le_mul_of_nonneg_left hsize p.errorUnit_pos.le
  calc
    _ ≤ |r.value - (product.value + c.value)| +
        |(product.value + c.value) - (a.value * b.value + c.value)| := abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub]
      nlinarith only [haddErr, hmulErr, hsize']

/-- Real component of the expanded update, with a product-specific error factor. -/
theorem expanded_real_error {p : MPIR.Precision} {x y c xs ys difference result : MPIR.Limbs}
    (coefficient : ℝ)
    (hxs : |xs.value| ≤ |x.value| ^ 2) (hys : |ys.value| ≤ |y.value| ^ 2)
    (hxe : |xs.value - x.value ^ 2| ≤ coefficient * p.errorUnit * |x.value| ^ 2)
    (hye : |ys.value - y.value ^ 2| ≤ coefficient * p.errorUnit * |y.value| ^ 2)
    (hsub : MPIR.Sub p xs ys difference) (hadd : MPIR.Add p difference c result) :
    |result.value - (x.value ^ 2 - y.value ^ 2 + c.value)| ≤
      p.errorUnit * ((coefficient + 6) * (|x.value| ^ 2 + |y.value| ^ 2) + 2 * |c.value|) := by
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
      coefficient * u * size := by
    rw [show (xs.value - ys.value) - (x.value ^ 2 - y.value ^ 2) =
      (xs.value - x.value ^ 2) - (ys.value - y.value ^ 2) by ring]
    have h := abs_sub (xs.value - x.value ^ 2) (ys.value - y.value ^ 2)
    change _ ≤ coefficient * u * (|x.value| ^ 2 + |y.value| ^ 2)
    nlinarith only [h, hxe, hye]
  calc
    _ ≤ |result.value - (difference.value + c.value)| +
        |(difference.value + c.value) - (x.value ^ 2 - y.value ^ 2 + c.value)| :=
      abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub]
      have h := abs_sub_le difference.value (xs.value - ys.value) (x.value ^ 2 - y.value ^ 2)
      change _ ≤ u * ((coefficient + 6) * (|x.value| ^ 2 + |y.value| ^ 2) + 2 * |c.value|)
      nlinarith only [haddError, haddSize, h, hsubError, hproducts]

/-- The cross product is rounded before it is doubled with the unsigned
integer operation. Its two roundoff sources precede the addition of c_y. -/
theorem expanded_imag_error {p : MPIR.Precision} {x y c cross doubled result : MPIR.Limbs}
    (coefficient : ℝ)
    (hcross : |cross.value| ≤ |x.value * y.value|)
    (hcrossErr : |cross.value - x.value * y.value| ≤ coefficient * p.errorUnit * |x.value * y.value|)
    (hdouble : MPIR.MulUiTwo p cross doubled)
    (hadd : MPIR.Add p doubled c result) :
    |result.value - (2 * (x.value * y.value) + c.value)| ≤
      p.errorUnit * ((2 * coefficient + 6) * |x.value * y.value| + 2 * |c.value|) := by
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

/-- One complete expanded CPU step, parameterized by the product's relative
error factor. MT3 clips inputs as well as the product. MT9 trims the product. -/
theorem expanded_step_error {p q : MPIR.Precision} {old next : State p} {c : State q}
    {xs ys cross doubled difference : MPIR.Limbs} (coefficient : ℝ)
    (hx : |xs.value| ≤ |old.x.val.value * old.x.val.value| ∧
      |xs.value - old.x.val.value * old.x.val.value| ≤
        coefficient * p.errorUnit * |old.x.val.value * old.x.val.value|)
    (hy : |ys.value| ≤ |old.y.val.value * old.y.val.value| ∧
      |ys.value - old.y.val.value * old.y.val.value| ≤
        coefficient * p.errorUnit * |old.y.val.value * old.y.val.value|)
    (hcross : |cross.value| ≤ |old.x.val.value * old.y.val.value| ∧
      |cross.value - old.x.val.value * old.y.val.value| ≤
        coefficient * p.errorUnit * |old.x.val.value * old.y.val.value|)
    (hdouble : MPIR.MulUiTwo p cross doubled) (hyAdd : MPIR.Add p doubled c.y.val next.y.val)
    (hsub : MPIR.Sub p xs ys difference) (hxAdd : MPIR.Add p difference c.x.val next.x.val) :
    norm1 (next.value - (old.value ^ 2 + c.value)) ≤
      p.errorUnit * ((coefficient + 6) * norm1 old.value ^ 2 + 2 * norm1 c.value) := by
  simp only [← pow_two, abs_pow] at hx hy
  have hre := expanded_real_error coefficient hx.1 hy.1 hx.2 hy.2 hsub hxAdd
  have him := expanded_imag_error coefficient hcross.1 hcross.2 hdouble hyAdd
  rw [abs_mul] at him
  have hnonneg := mul_nonneg p.errorUnit_pos.le
    (mul_nonneg (abs_nonneg old.x.val.value) (abs_nonneg old.y.val.value))
  simp only [norm1, Complex.sub_re, Complex.sub_im, Complex.add_re, Complex.add_im, State.value,
    pow_two, Complex.mul_re, Complex.mul_im]
  rw [← pow_two old.x.val.value, ← pow_two old.y.val.value]
  rw [show old.x.val.value * old.y.val.value + old.y.val.value * old.x.val.value =
    2 * (old.x.val.value * old.y.val.value) by ring]
  nlinarith only [hre, him, hnonneg]

/-- Substitute the size cap into the backend's per-step rounding allowance. -/
noncomputable def mpirToleranceBound (p : MPIR.Precision) (coefficient : ℝ) (target working : ℂ)
    (cap tolerance : ℝ) : ℕ → ℝ :=
  toleranceBound cap tolerance (norm1 (working - target))
    (p.errorUnit * (coefficient * cap ^ 2 + 2 * norm1 working))

/-- Specialize the shared tolerance theorem to the MPIR initialization and a
backend whose proved local rounding allowance is quadratic in the state size. -/
theorem mpir_prefix_error_le_of_tolerance {p : MPIR.Precision} (coefficient : ℝ)
    (target : ℂ) (working : State p) (states : ℕ → State p) (last : ℕ) (cap tolerance : ℝ)
    (hcoefficient : 0 ≤ coefficient) (hcap : 0 ≤ cap) (htolerance : 0 ≤ tolerance)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (states n).value ≤ cap)
    (hstep : ∀ n, 1 ≤ n → n < last →
      norm1 ((states (n + 1)).value - ((states n).value ^ 2 + working.value)) ≤
        p.errorUnit * (coefficient * norm1 (states n).value ^ 2 + 2 * norm1 working.value))
    (hfinal : mpirToleranceBound p coefficient target working.value cap tolerance last ≤ tolerance) :
    ∀ n, n ≤ last → norm1 ((states n).value - orbit target n) ≤ tolerance := by
  refine prefix_error_le_of_tolerance target working.value (fun n => (states n).value)
    last cap tolerance _ hcap htolerance ?_ hzero (firstState_eq hx hy) hsize ?_ hfinal
  · exact mul_nonneg p.errorUnit_pos.le
      (add_nonneg (mul_nonneg hcoefficient (sq_nonneg _))
        (mul_nonneg (by norm_num) (norm1_nonneg _)))
  · intro n hn hl
    have hsq := pow_le_pow_left₀ (norm1_nonneg (states n).value) (hsize n hn hl) 2
    exact (hstep n hn hl).trans (mul_le_mul_of_nonneg_left
      (add_le_add (mul_le_mul_of_nonneg_left hsq hcoefficient) le_rfl) p.errorUnit_pos.le)

end FractalSharkProofs.ReferenceOrbit
