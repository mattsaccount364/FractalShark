module

public import FractalSharkProofs.ReferenceOrbit.Schedule
public import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Precision
import Mathlib.Tactic.NormNum

/-! The serial exact and rounded claims restricted to the loop's observations. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.SingleThreaded

theorem exact_at_observed (mode : PeriodicityMode) (checks : ShadowChecks) (limit : ℕ)
    (c : ℂ) (states : ℕ → ℂ) (hzero : states 0 = 0) (hfirst : states 1 = c)
    (hstep : ∀ i, 1 ≤ i →
      i < (run .singleThreaded mode checks limit).coherentThrough .singleThreaded →
      states (i + 1) = exactStep c (states i))
    {n : ℕ} (hn : n ∈ (run .singleThreaded mode checks limit).samples .singleThreaded) :
    states n = orbit c n := by
  exact exact_at_sample .singleThreaded _ c states hzero hfirst
    (fun i hi hl => (hstep i hi hl).trans (exactStep_eq _ _)) hn

theorem error_at_observed {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit : ℕ)
    (target : ℂ) (working : State p) (states : ℕ → State p)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i →
      i < (run .singleThreaded mode checks limit).coherentThrough .singleThreaded →
      Step working (states i) (states (i + 1)))
    {n : ℕ} (hn : n ∈ (run .singleThreaded mode checks limit).samples .singleThreaded) :
    norm1 ((states n).value - orbit target n) ≤
      errorBound p target working.value (fun i => (states i).value) n :=
  prefix_error_le target working states _ hzero hx hy hstep n (sample_le_coherent _ _ hn)

/-- A terminal tolerance check at an observed index certifies the whole prefix through it. -/
theorem error_at_observed_of_tolerance {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit : ℕ)
    (target : ℂ) (working : State p) (states : ℕ → State p) (last : ℕ) (cap tolerance : ℝ)
    (hlast : last ∈ (run .singleThreaded mode checks limit).samples .singleThreaded)
    (hcap : 0 ≤ cap) (htolerance : 0 ≤ tolerance)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (states n).value ≤ cap)
    (hstep : ∀ i, 1 ≤ i →
      i < (run .singleThreaded mode checks limit).coherentThrough .singleThreaded →
      Step working (states i) (states (i + 1)))
    (hfinal : mpirToleranceBound p 32 target working.value cap tolerance last ≤ tolerance) :
    ∀ n, n ≤ last → norm1 ((states n).value - orbit target n) ≤ tolerance := by
  have hcoherent := sample_le_coherent .singleThreaded (run .singleThreaded mode checks limit) hlast
  exact mpir_prefix_error_le_of_tolerance 32 target working states last cap tolerance
    (by norm_num) hcap htolerance hzero hx hy hsize
    (fun i hi hl => step_error (hstep i hi (hl.trans_le hcoherent))) hfinal

end FractalSharkProofs.ReferenceOrbit.SingleThreaded
