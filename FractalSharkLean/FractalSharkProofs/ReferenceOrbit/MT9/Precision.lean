module

public import FractalSharkProofs.ReferenceOrbit.MT9.Workers
public import FractalSharkProofs.ReferenceOrbit.ViewportPrecision
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Fixed-precision accuracy and the checked and published MT9 indices. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

def fixedCoefficient : ℕ := 7

def checkedStates (exit : LoopExit) : List ℕ :=
  (List.range exit.entered).map (· + 1)

/-- The first body checks and advances the initial state without publishing
it. In steady state a positive sample was checked in the preceding body. -/
theorem checked_before_publication (exit : LoopExit) {n : ℕ} (hn : 0 < n)
    (h : n ∈ exit.samples .mt9) : n ∈ checkedStates exit := by
  rw [mt9_samples, List.mem_range] at h
  apply List.mem_map.mpr
  refine ⟨n - 1, List.mem_range.mpr (by omega), by omega⟩

theorem terminal_advance (exit : LoopExit) (h : 0 < exit.entered) :
    exit.coherentThrough .mt9 = exit.entered + 1 := by
  simp [Nat.ne_of_gt h]

def exactStep (c old : ℂ) : ℂ :=
  ⟨old.re ^ 2 - old.im ^ 2 + c.re, 2 * (old.re * old.im) + c.im⟩

theorem exactStep_eq (c old : ℂ) : exactStep c old = old ^ 2 + c := by
  apply Complex.ext <;>
    simp only [exactStep, pow_two, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im];
    ring

theorem exact_at_observed (mode : PeriodicityMode) (checks : ShadowChecks) (limit : ℕ)
    (c : ℂ) (states : ℕ → ℂ) (hzero : states 0 = 0) (hfirst : states 1 = c)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt9 mode checks limit).coherentThrough .mt9 →
      states (i + 1) = exactStep c (states i))
    {n : ℕ} (hn : n ∈ (run .mt9 mode checks limit).samples .mt9) :
    states n = orbit c n :=
  exact_at_sample .mt9 _ c states hzero hfirst
    (fun i hi hl => (hstep i hi hl).trans (exactStep_eq _ _)) hn

/-- The products read the same snapshot. The copied coordinates stay at full
precision even when the dynamic specialization narrows every work slot. -/
structure Step {p full : MPIR.Precision} (c : State full) (old next : State p) : Prop where
  run : ∃ (xs ys cross doubled difference : Slot p),
    Product p old.x.val old.x.val xs.val ∧
    Product p old.y.val old.y.val ys.val ∧
    Product p old.x.val old.y.val cross.val ∧
    MPIR.MulUiTwo p cross.val doubled.val ∧
    MPIR.Add p doubled.val c.y.val next.y.val ∧
    MPIR.Sub p xs.val ys.val difference.val ∧
    MPIR.Add p difference.val c.x.val next.x.val

theorem step_error {p full : MPIR.Precision} {c : State full} {old next : State p}
    (h : Step c old next) :
    norm1 (next.value - (old.value ^ 2 + c.value)) ≤
      p.errorUnit * (fixedCoefficient * norm1 old.value ^ 2 + 2 * norm1 c.value) := by
  obtain ⟨_, _, _, _, _, hx, hy, hxy, hd, ha, hs, hb⟩ := h.run
  simpa only [one_mul, show (1 : ℝ) + 6 = fixedCoefficient by norm_num [fixedCoefficient]] using
    expanded_step_error 1 (by simpa only [one_mul] using product_bounds hx)
      (by simpa only [one_mul] using product_bounds hy)
      (by simpa only [one_mul] using product_bounds hxy) hd ha hs hb

noncomputable def errorBound (p : MPIR.Precision) (target working : ℂ) (states : ℕ → ℂ) : ℕ → ℝ :=
  prefixBound target working states
    (fun n => p.errorUnit * (fixedCoefficient * norm1 (states n) ^ 2 + 2 * norm1 working))

theorem error_at_observed {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit : ℕ)
    (target : ℂ) (working : State p) (states : ℕ → State p)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt9 mode checks limit).coherentThrough .mt9 →
      Step working (states i) (states (i + 1)))
    {n : ℕ} (hn : n ∈ (run .mt9 mode checks limit).samples .mt9) :
    norm1 ((states n).value - orbit target n) ≤
      errorBound p target working.value (fun i => (states i).value) n :=
  prefixBound_correct target working.value (fun i => (states i).value) _ _
    hzero (firstState_eq hx hy) (fun i hi hl => step_error (hstep i hi hl)) n
    (sample_le_coherent .mt9 _ hn)

noncomputable def viewportPrefix {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit last : ℕ)
    (hlast : last ∈ (run .mt9 mode checks limit).samples .mt9)
    (inputX inputY : MPIR.Limbs) (working : State p) (states : ℕ → State p)
    (hcopyX : MPIR.Set p inputX working.x.val) (hcopyY : MPIR.Set p inputY working.y.val)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt9 mode checks limit).coherentThrough .mt9 →
      Step working (states i) (states (i + 1))) :
    MPIRViewportPrefix p fixedCoefficient last :=
  MPIRViewportPrefix.fromStates inputX inputY working states (by positivity) hcopyX hcopyY hzero hx hy
    (fun i hi hl => step_error (hstep i hi (hl.trans_le (sample_le_coherent .mt9 _ hlast))))

end FractalSharkProofs.ReferenceOrbit.MT9
