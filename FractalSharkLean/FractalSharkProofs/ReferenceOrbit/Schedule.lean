module

public import FractalSharkProofs.ReferenceOrbit.Basic
import Mathlib.Tactic.Ring

/-! Sample timing and optional derivative checks for the authoritative CPU loops. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

inductive Backend where
  | singleThreaded
  | mt3
  | mt9
  deriving DecidableEq

inductive PeriodicityMode where
  | disabled
  | enabled
  | benchmark
  deriving DecidableEq

inductive StopReason where
  | limit
  | cancelled
  | nearReturn
  | bailout
  deriving DecidableEq

/-- The inputs are indexed by the body for abort, and by the observed iterate for
shadow arithmetic. They include the renderer's cancellation polling and its
reduced-precision conversions and comparisons. -/
structure ShadowChecks where
  abortBefore : ℕ → Bool
  nearReturn : ℕ → ℂ → Bool
  advanceDerivative : ℕ → ℂ → ℂ
  bailout : ℕ → Bool

/-- Real and imaginary derivative updates both use the preceding derivative. -/
def derivativeStep (z d : ℂ) : ℂ :=
  ⟨2 * (z.re * d.re - z.im * d.im) + 1, 2 * (z.re * d.im + z.im * d.re)⟩

theorem derivativeStep_eq (z d : ℂ) : derivativeStep z d = 2 * z * d + 1 := by
  have hre : (2 : ℂ).re = 2 := rfl
  have him : (2 : ℂ).im = 0 := rfl
  apply Complex.ext <;>
    simp only [derivativeStep, Complex.mul_re, Complex.mul_im, Complex.add_re, Complex.add_im,
      hre, him, Complex.one_re, Complex.one_im] <;> ring

/-- The unrounded derivative recurrence starts at zero, so its first live value is one. -/
def exactDerivative (c : ℂ) : ℕ → ℂ
  | 0 => 0
  | n + 1 => 2 * orbit c n * exactDerivative c n + 1

theorem exactDerivative_first (c : ℂ) : exactDerivative c 1 = 1 := by
  simp [exactDerivative]

theorem derivativeStep_on_orbit (c : ℂ) (n : ℕ) :
    derivativeStep (orbit c n) (exactDerivative c n) = exactDerivative c (n + 1) := by
  rw [derivativeStep_eq]
  rfl

def checkPeriodicity (mode : PeriodicityMode) (checks : ShadowChecks) (sample : ℕ)
    (d : ℂ) : Bool × ℂ :=
  if mode = .disabled then (false, d)
  else if checks.nearReturn sample d then (decide (mode = .enabled), d)
  else (false, checks.advanceDerivative sample d)

theorem disabled_keepsDerivative (checks : ShadowChecks) (n : ℕ) (d : ℂ) :
    checkPeriodicity .disabled checks n d = (false, d) := by simp [checkPeriodicity]

theorem benchmark_keepsDerivative_on_return (checks : ShadowChecks) (n : ℕ) (d : ℂ)
    (h : checks.nearReturn n d = true) :
    checkPeriodicity .benchmark checks n d = (false, d) := by simp [checkPeriodicity, h]

theorem enabled_stops_on_return (checks : ShadowChecks) (n : ℕ) (d : ℂ)
    (h : checks.nearReturn n d = true) :
    checkPeriodicity .enabled checks n d = (true, d) := by simp [checkPeriodicity, h]

/-- The caller emits the current ST state or the delayed MT3 shadow. MT3 has
no delayed shadow in its first body. -/
def Backend.observation (b : Backend) (i : ℕ) : Option ℕ :=
  match b, i with
  | .singleThreaded, i => some (i + 1)
  | .mt3, 0 => none
  | .mt3, i + 1 => some (i + 1)
  | .mt9, 0 => none
  | .mt9, i + 1 => some (i + 1)

structure LoopExit where
  entered : ℕ
  derivative : ℂ
  reason : StopReason

/-- A terminal near return leaves no new coherent pair. An MT3 bailout also
discards the final candidate pair because the caller skips saving the Y-square.
ST bailout follows the completed serial update. MT9 completes both terminal
updates, although the newly advanced state will not be sampled. -/
def LoopExit.coherentThrough (e : LoopExit) (b : Backend) : ℕ :=
  if b = .mt9 ∧ e.entered = 0 ∧ e.reason = .limit then 0
  else if b ≠ .mt9 ∧ (e.reason = .nearReturn ∨ (b = .mt3 ∧ e.reason = .bailout)) then e.entered
  else e.entered + 1

/-- Model which samples are emitted, without imposing arithmetic on their shadows. -/
def LoopExit.samples (e : LoopExit) (b : Backend) : List ℕ :=
  0 :: (List.range e.entered).filterMap b.observation

def LoopExit.maybePeriod (e : LoopExit) (b : Backend) : Option ℕ :=
  if e.reason = .nearReturn then some (e.samples b).length else none

private theorem serialIndices (n : ℕ) :
    0 :: (List.range n).filterMap Backend.singleThreaded.observation = List.range (n + 1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [List.range_succ (n := n), List.filterMap_append, ← List.cons_append, ih]
    simp [Backend.observation, List.range_succ (n := n + 1)]

private theorem delayedIndices (n : ℕ) :
    0 :: (List.range (n + 1)).filterMap Backend.mt3.observation = List.range (n + 1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    conv => lhs; rw [List.range_succ (n := n + 1)]
    rw [List.filterMap_append, ← List.cons_append, ih]
    simp [Backend.observation, List.range_succ (n := n + 1)]

theorem serial_samples (e : LoopExit) : e.samples .singleThreaded = List.range (e.entered + 1) :=
  serialIndices e.entered

theorem delayed_samples (e : LoopExit) : e.samples .mt3 = List.range (max 1 e.entered) := by
  change 0 :: (List.range e.entered).filterMap Backend.mt3.observation = _
  cases e.entered with
  | zero => rfl
  | succ n => simpa [max_eq_right (Nat.succ_le_succ (Nat.zero_le n))] using delayedIndices n

theorem mt9_samples (e : LoopExit) : e.samples .mt9 = List.range (max 1 e.entered) := by
  have h : Backend.mt9.observation = Backend.mt3.observation := by
    funext i
    cases i <;> rfl
  simpa only [LoopExit.samples, h] using delayed_samples e

@[simp] theorem mt9_coherent (e : LoopExit) :
    e.coherentThrough .mt9 = if e.entered = 0 ∧ e.reason = .limit then 0 else e.entered + 1 := by
  simp [LoopExit.coherentThrough]

def runFrom (b : Backend) (mode : PeriodicityMode) (checks : ShadowChecks) :
    ℕ → ℕ → ℂ → LoopExit
  | 0, i, d => ⟨i, d, .limit⟩
  | remaining + 1, i, d =>
    if checks.abortBefore i then ⟨i, d, .cancelled⟩
    else match b.observation i with
    | none => runFrom b mode checks remaining (i + 1) d
    | some n =>
      let (stop, nextD) := checkPeriodicity mode checks n d
      if stop then ⟨i + 1, nextD, .nearReturn⟩
      else if checks.bailout n then ⟨i + 1, nextD, .bailout⟩
      else runFrom b mode checks remaining (i + 1) nextD

def run (b : Backend) (mode : PeriodicityMode) (checks : ShadowChecks) (limit : ℕ) :
    LoopExit := runFrom b mode checks limit 0 1

theorem runFrom_entered_bounds (b : Backend) (mode : PeriodicityMode) (checks : ShadowChecks)
    (remaining i : ℕ) (d : ℂ) :
    i ≤ (runFrom b mode checks remaining i d).entered ∧
      (runFrom b mode checks remaining i d).entered ≤ i + remaining := by
  induction remaining generalizing i d with
  | zero => simp [runFrom]
  | succ remaining ih =>
    simp only [runFrom]
    split
    · simp
    · split
      · have h := ih (i + 1) d
        omega
      · rename_i n _
        split
        · simp
        · split
          · simp
          · have h := ih (i + 1) (checkPeriodicity mode checks n d).2
            omega

/-- Identical control inputs on the observed prefix give the same exit. There
is no requirement on terminal scratch states or observations beyond that exit. -/
private theorem runFrom_eq_on_samples (b : Backend) (mode : PeriodicityMode)
    (left right : ShadowChecks) (remaining i : ℕ) (d : ℂ)
    (habort : ∀ j, left.abortBefore j = right.abortBefore j)
    (hagree : ∀ n ∈ (runFrom b mode left remaining i d).samples b, ∀ deriv,
      checkPeriodicity mode left n deriv = checkPeriodicity mode right n deriv ∧
      left.bailout n = right.bailout n) :
    runFrom b mode left remaining i d = runFrom b mode right remaining i d := by
  induction remaining generalizing i d with
  | zero => rfl
  | succ remaining ih =>
    by_cases ha : left.abortBefore i = true
    · simp [runFrom, ha, ← habort i]
    simp only [runFrom, ← habort i, ha, Bool.false_eq_true, ite_false]
    cases hobs : b.observation i with
    | none =>
      exact ih (i + 1) d (by simpa only [runFrom, ha, Bool.false_eq_true, ite_false, hobs] using hagree)
    | some n =>
      have hentered : i < (runFrom b mode left (remaining + 1) i d).entered := by
        simp only [runFrom, ha, Bool.false_eq_true, ite_false, hobs]
        split
        · exact Nat.lt_succ_self _
        · split
          · exact Nat.lt_succ_self _
          · exact Nat.lt_of_succ_le (runFrom_entered_bounds b mode left remaining _ _).1
      have hsample : n ∈ (runFrom b mode left (remaining + 1) i d).samples b := by
        apply List.mem_cons_of_mem
        exact List.mem_filterMap.mpr ⟨i, List.mem_range.mpr hentered, hobs⟩
      obtain ⟨hperiod, hbail⟩ := hagree n hsample d
      dsimp only
      rw [← hperiod, ← hbail]
      split
      · rfl
      · rename_i hstop
        split
        · rfl
        · rename_i hescape
          exact ih _ _ (by simpa [runFrom, ha, hobs, hstop, hescape] using hagree)

theorem run_eq_on_samples (b : Backend) (mode : PeriodicityMode) (left right : ShadowChecks)
    (limit : ℕ) (habort : ∀ j, left.abortBefore j = right.abortBefore j)
    (hagree : ∀ n ∈ (run b mode left limit).samples b, ∀ deriv,
      checkPeriodicity mode left n deriv = checkPeriodicity mode right n deriv ∧
      left.bailout n = right.bailout n) :
    run b mode left limit = run b mode right limit :=
  runFrom_eq_on_samples b mode left right limit 0 1 habort hagree

theorem disabled_reason (b : Backend) (checks : ShadowChecks) (remaining i : ℕ) (d : ℂ) :
    (runFrom b .disabled checks remaining i d).reason ≠ .nearReturn := by
  induction remaining generalizing i d with
  | zero => simp [runFrom]
  | succ remaining ih =>
    by_cases ha : checks.abortBefore i = true
    · simp [runFrom, ha]
    cases ho : b.observation i with
    | none => simpa [runFrom, ha, ho] using ih (i + 1) d
    | some n =>
      by_cases hb : checks.bailout n = true
      · simp [runFrom, ha, ho, hb, checkPeriodicity]
      · simpa [runFrom, ha, ho, hb, checkPeriodicity] using ih (i + 1) d

theorem sample_le_coherent (b : Backend) (e : LoopExit) {n : ℕ} (h : n ∈ e.samples b) :
    n ≤ e.coherentThrough b := by
  have hEntered : n ≤ e.entered := by
    rw [LoopExit.samples, List.mem_cons, List.mem_filterMap] at h
    rcases h with rfl | ⟨i, hi, hobs⟩
    · exact Nat.zero_le _
    rw [List.mem_range] at hi
    cases b with
    | singleThreaded =>
      simp [Backend.observation] at hobs
      omega
    | mt3 | mt9 =>
      cases i with
      | zero => simp [Backend.observation] at hobs
      | succ i =>
        simp [Backend.observation] at hobs
        omega
  unfold LoopExit.coherentThrough
  split
  · omega
  · split <;> omega

theorem exact_at_sample (b : Backend) (e : LoopExit) (c : ℂ) (states : ℕ → ℂ)
    (hzero : states 0 = 0) (hfirst : states 1 = c)
    (hstep : ∀ i, 1 ≤ i → i < e.coherentThrough b → states (i + 1) = states i ^ 2 + c)
    {n : ℕ} (hn : n ∈ e.samples b) : states n = orbit c n := by
  have bound := sample_le_coherent b e hn
  clear hn
  induction n with
  | zero => simpa [orbit] using hzero
  | succ n ih =>
    cases n with
    | zero => simpa [orbit] using hfirst
    | succ n => rw [hstep (n + 1) (by omega) (by omega), ih (by omega)]; rfl

end FractalSharkProofs.ReferenceOrbit
