module

public import FractalSharkProofs.ReferenceOrbit.Schedule
import Mathlib.Tactic.Ring

/-! The expanded reference update evaluated by the two square workers and caller. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT3

def exactStep (c old : ℂ) : ℂ :=
  ⟨old.re ^ 2 - old.im ^ 2 + c.re, 2 * (old.re * old.im) + c.im⟩

theorem exactStep_eq (c old : ℂ) : exactStep c old = old ^ 2 + c := by
  apply Complex.ext
  · simp only [exactStep, Complex.add_re, pow_two, Complex.mul_re]
  · simp only [exactStep, Complex.add_im, pow_two, Complex.mul_im]
    ring

theorem exact_at_observed (mode : PeriodicityMode) (checks : ShadowChecks) (limit : ℕ)
    (c : ℂ) (states : ℕ → ℂ) (hzero : states 0 = 0) (hfirst : states 1 = c)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt3 mode checks limit).coherentThrough .mt3 →
      states (i + 1) = exactStep c (states i))
    {n : ℕ} (hn : n ∈ (run .mt3 mode checks limit).samples .mt3) :
    states n = orbit c n := by
  exact exact_at_sample .mt3 _ c states hzero hfirst
    (fun i hi hl => (hstep i hi hl).trans (exactStep_eq _ _)) hn

end FractalSharkProofs.ReferenceOrbit.MT3
