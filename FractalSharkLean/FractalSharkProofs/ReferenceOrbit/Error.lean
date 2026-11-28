module

public import FractalSharkProofs.ReferenceOrbit.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Backend-independent complex error propagation. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

/-- Componentwise absolute error, summed across the real and imaginary parts. -/
noncomputable def norm1 (z : ℂ) : ℝ := |z.re| + |z.im|

theorem norm1_nonneg (z : ℂ) : 0 ≤ norm1 z := add_nonneg (abs_nonneg _) (abs_nonneg _)

@[simp] theorem norm1_zero : norm1 0 = 0 := by simp [norm1]

theorem norm1_add_le (a b : ℂ) : norm1 (a + b) ≤ norm1 a + norm1 b := by
  simp only [norm1, Complex.add_re, Complex.add_im]
  linarith only [abs_add_le a.re b.re, abs_add_le a.im b.im]

theorem norm1_mul_le (a b : ℂ) : norm1 (a * b) ≤ norm1 a * norm1 b := by
  simp only [norm1, Complex.mul_re, Complex.mul_im]
  have h₁ : |a.re * b.re - a.im * b.im| ≤ |a.re * b.re| + |a.im * b.im| := abs_sub _ _
  have h₂ := abs_add_le (a.re * b.im) (a.im * b.re)
  simp only [abs_mul] at h₁ h₂
  nlinarith only [h₁, h₂]

private theorem sq_error (a b : ℂ) :
    norm1 (a ^ 2 - b ^ 2) ≤ (2 * norm1 a + norm1 (a - b)) * norm1 (a - b) := by
  have hb : norm1 b ≤ norm1 a + norm1 (a - b) := by
    have h := norm1_add_le a (-(a - b))
    rw [show a + -(a - b) = b by ring] at h
    simpa only [norm1, Complex.neg_re, Complex.neg_im, abs_neg] using h
  have hab : norm1 (a + b) ≤ 2 * norm1 a + norm1 (a - b) := by
    linarith only [hb, norm1_add_le a b]
  rw [show a ^ 2 - b ^ 2 = (a + b) * (a - b) by ring]
  exact (norm1_mul_le _ _).trans (mul_le_mul_of_nonneg_right hab (norm1_nonneg _))

/-- Propagate per-step local arithmetic residuals over the observed internal orbit.
The stored parameter discrepancy enters each recurrence separately. -/
noncomputable def prefixBound (target working : ℂ) (w : ℕ → ℂ) (localError : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => norm1 (working - target)
  | n + 2 =>
    let previous := prefixBound target working w localError (n + 1)
    (2 * norm1 (w (n + 1)) + previous) * previous + norm1 (working - target) +
      localError (n + 1)

private theorem propagate_error (target working next w exact : ℂ)
    (bound localError : ℝ) (he : norm1 (w - exact) ≤ bound)
    (hstep : norm1 (next - (w ^ 2 + working)) ≤
      localError) :
    norm1 (next - (exact ^ 2 + target)) ≤ (2 * norm1 w + bound) * bound +
      norm1 (working - target) + localError := by
  have hsum : 2 * norm1 w + norm1 (w - exact) ≤ 2 * norm1 w + bound := by
    linarith only [he]
  have hnonneg : 0 ≤ 2 * norm1 w + bound := by
    linarith only [he, norm1_nonneg w, norm1_nonneg (w - exact)]
  have hs := (sq_error w exact).trans (mul_le_mul hsum he (norm1_nonneg _) hnonneg)
  have hid : next - (exact ^ 2 + target) =
      (next - (w ^ 2 + working)) + ((w ^ 2 - exact ^ 2) + (working - target)) := by ring
  rw [hid]
  have h₁ := norm1_add_le (next - (w ^ 2 + working)) ((w ^ 2 - exact ^ 2) + (working - target))
  have h₂ := norm1_add_le (w ^ 2 - exact ^ 2) (working - target)
  linarith only [h₁, h₂, hs, hstep]

/-- A finite prefix bound for any backend with verified local residuals. -/
theorem prefixBound_correct (target working : ℂ) (w : ℕ → ℂ) (localError : ℕ → ℝ)
    (last : ℕ) (hzero : w 0 = 0) (hfirst : w 1 = working)
    (hstep : ∀ n, 1 ≤ n → n < last →
      norm1 (w (n + 1) - (w n ^ 2 + working)) ≤ localError n) :
    ∀ n, n ≤ last → norm1 (w n - orbit target n) ≤ prefixBound target working w localError n := by
  intro n
  induction n with
  | zero => simp [orbit, hzero, prefixBound]
  | succ n ih =>
    intro hn
    cases n with
    | zero => simp [orbit, hfirst, prefixBound]
    | succ n =>
      have hprev := ih (by omega)
      have hlocal := hstep (n + 1) (by omega) (by omega)
      simp only [orbit, prefixBound]
      exact propagate_error target working _ _ _ _ _ hprev hlocal

/-- Use one amplification and one error budget for every step of the prefix.
The zero entry is exact and the first iterate inherits the parameter error. -/
noncomputable def affineBound (amplification parameterError stepBudget : ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => parameterError
  | n + 2 => amplification * affineBound amplification parameterError stepBudget (n + 1) +
      stepBudget

/-- The last bound can be computed from a geometric sum, including at amplification one. -/
theorem affineBound_geometric (amplification parameterError stepBudget : ℝ) (n : ℕ) :
    affineBound amplification parameterError stepBudget (n + 1) =
      amplification ^ n * parameterError +
        stepBudget * ∑ k ∈ Finset.range n, amplification ^ k := by
  induction n with
  | zero => simp [affineBound]
  | succ n ih =>
    rw [affineBound, ih, Finset.sum_range_succ']
    simp only [pow_succ', pow_zero, ← Finset.mul_sum]
    ring

theorem affineBound_monotone {amplification parameterError stepBudget : ℝ}
    (ha : 0 ≤ amplification) (hparameter : 0 ≤ parameterError)
    (hbudget : parameterError ≤ stepBudget) :
    Monotone (affineBound amplification parameterError stepBudget) := by
  apply monotone_nat_of_le_succ
  intro n
  induction n with
  | zero => simpa [affineBound] using hparameter
  | succ n ih =>
    cases n with
    | zero =>
      simp only [affineBound]
      exact hbudget.trans (le_add_of_nonneg_left (mul_nonneg ha hparameter))
    | succ n =>
      change amplification * affineBound amplification parameterError stepBudget (n + 1) +
          stepBudget ≤ amplification * affineBound amplification parameterError stepBudget (n + 2) +
          stepBudget
      exact add_le_add (mul_le_mul_of_nonneg_left ih ha) le_rfl

/-- Increasing any coefficient of a nonnegative envelope can only enlarge it. -/
theorem affineBound_le {a a' δ δ' q q' : ℝ}
    (ha : 0 ≤ a) (hδ : 0 ≤ δ) (hq : 0 ≤ q)
    (ha' : a ≤ a') (hδ' : δ ≤ δ') (hq' : q ≤ q') (n : ℕ) :
    affineBound a δ q n ≤ affineBound a' δ' q' n := by
  induction n with
  | zero => exact le_rfl
  | succ n ih =>
    cases n with
    | zero => exact hδ'
    | succ n =>
      have hn : 0 ≤ affineBound a δ q (n + 1) := by
        clear ih
        induction n with
        | zero => exact hδ
        | succ n ih => exact add_nonneg (mul_nonneg ha ih) hq
      exact add_le_add (mul_le_mul ha' ih hn (ha.trans ha')) hq'

/-- The candidate tolerance bounds the quadratic part of the amplification
once the affine bound is shown to stay within that tolerance. -/
noncomputable def toleranceBound (cap tolerance parameterError localBudget : ℝ) : ℕ → ℝ :=
  affineBound (2 * cap + tolerance) parameterError (parameterError + localBudget)

/-- A single terminal check suffices. The affine bound is nondecreasing, so
the tolerance also bounds every earlier radius used in the induction. -/
theorem prefix_error_le_of_tolerance (target working : ℂ) (w : ℕ → ℂ)
    (last : ℕ) (cap tolerance localBudget : ℝ)
    (hcap : 0 ≤ cap) (htolerance : 0 ≤ tolerance) (hbudget : 0 ≤ localBudget)
    (hzero : w 0 = 0) (hfirst : w 1 = working)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (w n) ≤ cap)
    (hstep : ∀ n, 1 ≤ n → n < last → norm1 (w (n + 1) - (w n ^ 2 + working)) ≤ localBudget)
    (hfinal : toleranceBound cap tolerance (norm1 (working - target)) localBudget last ≤ tolerance) :
    ∀ n, n ≤ last → norm1 (w n - orbit target n) ≤ tolerance := by
  let δ := norm1 (working - target)
  let bound := toleranceBound cap tolerance δ localBudget
  have hmono : Monotone bound :=
    affineBound_monotone (by linarith only [hcap, htolerance]) (norm1_nonneg _) (by
      linarith only [hbudget])
  suffices hprefix : ∀ n, n ≤ last → norm1 (w n - orbit target n) ≤ bound n by
    intro n hn
    exact (hprefix n hn).trans ((hmono hn).trans hfinal)
  intro n
  induction n with
  | zero => simp [orbit, hzero, bound, toleranceBound, affineBound]
  | succ n ih =>
    intro hn
    cases n with
    | zero => simp [orbit, hfirst, bound, δ, toleranceBound, affineBound]
    | succ n =>
      have hprev := ih (by omega)
      have hwithin := (hmono (show n + 1 ≤ last by omega)).trans hfinal
      have hfactor : 2 * norm1 (w (n + 1)) + bound (n + 1) ≤ 2 * cap + tolerance := by
        linarith only [hwithin, hsize (n + 1) (by omega) (by omega)]
      have hproduct := mul_le_mul_of_nonneg_right hfactor ((norm1_nonneg _).trans hprev)
      have herr := propagate_error target working _ _ _ _ _ hprev
        (hstep (n + 1) (by omega) (by omega))
      change norm1 (w (n + 2) - (orbit target (n + 1) ^ 2 + target)) ≤
        (2 * cap + tolerance) * bound (n + 1) + (δ + localBudget)
      linarith only [herr, hproduct]

/-- Choose the amplification using an error ceiling before computing the
envelope. Once the envelope fits that ceiling, it is itself an error bound. -/
theorem prefix_error_le_of_affine_cap (target working : ℂ) (w : ℕ → ℂ)
    (last : ℕ) (cap ceiling parameterBudget localBudget : ℝ)
    (hcap : 0 ≤ cap) (hceiling : 0 ≤ ceiling)
    (hδ : norm1 (working - target) ≤ parameterBudget)
    (hbudget : 0 ≤ localBudget) (hzero : w 0 = 0) (hfirst : w 1 = working)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (w n) ≤ cap)
    (hstep : ∀ n, 1 ≤ n → n < last → norm1 (w (n + 1) - (w n ^ 2 + working)) ≤ localBudget)
    (hfinal : toleranceBound cap ceiling parameterBudget localBudget last ≤ ceiling) :
    ∀ n, n ≤ last → norm1 (w n - orbit target n) ≤
      toleranceBound cap ceiling parameterBudget localBudget last := by
  let bound := toleranceBound cap ceiling parameterBudget localBudget last
  have hparam := (norm1_nonneg _).trans hδ
  have ha : 0 ≤ 2 * cap + ceiling := by linarith only [hcap, hceiling]
  have hnonneg : 0 ≤ bound :=
    affineBound_monotone ha hparam (le_add_of_nonneg_right hbudget) (Nat.zero_le last)
  apply prefix_error_le_of_tolerance target working w last cap bound localBudget
    hcap hnonneg hbudget hzero hfirst hsize hstep
  exact affineBound_le (by linarith only [hcap, hnonneg]) (norm1_nonneg _)
    (add_nonneg (norm1_nonneg _) hbudget) (by linarith only [show bound ≤ ceiling from hfinal])
    hδ (add_le_add hδ le_rfl) last

end FractalSharkProofs.ReferenceOrbit
