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
