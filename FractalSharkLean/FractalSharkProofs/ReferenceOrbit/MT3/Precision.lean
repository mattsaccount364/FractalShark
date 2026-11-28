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

theorem step_error {p : MPIR.Precision} {epoch : ℕ} {c old next : State p}
    (h : Step epoch c old next) :
    norm1 (next.value - (old.value ^ 2 + c.value)) ≤
      p.errorUnit * (9 * norm1 old.value ^ 2 + 2 * norm1 c.value) := by
  obtain ⟨x, y, cross, doubled, difference, hmul, hdouble, hy, hsub, hx⟩ := h.run
  simpa only [show (3 : ℝ) + 6 = 9 by norm_num] using
    expanded_step_error 3 (MPIR.mul_bounds x.squared) (MPIR.mul_bounds y.squared)
    (MPIR.mul_bounds hmul) hdouble hy hsub hx

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
