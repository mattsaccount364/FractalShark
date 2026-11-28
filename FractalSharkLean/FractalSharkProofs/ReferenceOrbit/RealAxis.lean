module

public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Enclosure
public import FractalSharkProofs.MPIRLimbBounds
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-! Shared limb-grid facts for real-axis replay near the Mandelbrot tip. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.RealAxis

open PrecisionEstimate
open MPIR (abs_lt_top keep_eq_truncate)

theorem top_eq_one (a : MPIR.Limbs) (hlo : 1 ≤ |a.value|)
    (hhi : |a.value| < MPIR.grid 1) : a.top = 1 := by
  have hz : a.value ≠ 0 := by intro h; norm_num [h] at hlo
  have hl := a.grid_top_le hz
  have hh := abs_lt_top a
  have h₁ : ¬ a.top ≤ 0 := by
    intro h
    have he : MPIR.grid a.top ≤ 1 := by simpa [MPIR.grid] using MPIR.grid_mono h
    linarith
  have h₂ : ¬ 2 ≤ a.top := by
    intro h
    have he := MPIR.grid_mono (show 1 ≤ a.top - 1 by omega)
    linarith
  omega

theorem add_opposite_exact {p : MPIR.Precision} {a b r : MPIR.Limbs}
    (ha : a.used ≤ p.limbs + 1) (hb : b.used ≤ p.limbs + 1)
    (haTop : a.top = 1) (hbTop : b.top = 1) (haPos : 0 < a.value) (hbNeg : b.value < 0)
    (h : MPIR.Add p a b r) : r.value = a.value + b.value := by
  apply (MPIR.add_span_bounds h (ne_of_gt haPos) (ne_of_lt hbNeg)).1
  have hs := mul_neg_of_pos_of_neg haPos hbNeg
  simp [MPIR.addCut, MPIR.maximumTop, haTop, hbTop, hs]
  omega

theorem grid_eq_dyadic (n : ℕ) (hn : 0 < n) :
    MPIR.grid (1 - (n : ℤ)) = (grid (64 * (n - 1)) : ℝ) := by
  have he : (1 : ℤ) - n = -((n - 1 : ℕ) : ℤ) := by omega
  rw [he, MPIR.grid, MPIR.radix, zpow_neg, zpow_natCast, Nat.cast_pow, Nat.cast_ofNat, ← pow_mul]
  simp [grid, one_div]

theorem guard_grid (p : MPIR.Precision) :
    MPIR.grid (-(p.limbs : ℤ)) = (grid (p.bits + 64) : ℝ) := by
  have hbits : 64 * p.limbs = p.bits + 64 := by
    unfold MPIR.Precision.bits
    have := p.min_limbs
    omega
  have h := grid_eq_dyadic (p.limbs + 1) (by omega)
  simpa [hbits] using h

private theorem abs_truncate_eq_down (bits : ℕ) (x : ℚ) :
    |MPIR.truncate (x : ℝ) (grid bits : ℝ)| = (down bits |x| : ℝ) := by
  have hp : (0 : ℝ) < (grid bits : ℝ) := by exact_mod_cast grid_pos bits
  have hf : ⌊|(x : ℝ)| / (grid bits : ℝ)⌋ = ⌊|x| / grid bits⌋ := by
    rw [← Rat.cast_abs, ← Rat.cast_div, Rat.floor_cast]
  have hfpos : 0 ≤ ⌊|x| / grid bits⌋ :=
    Int.floor_nonneg.mpr (div_nonneg (abs_nonneg _) (grid_pos _).le)
  have hpos : 0 ≤ (⌊|x| / grid bits⌋ : ℝ) * (grid bits : ℝ) :=
    mul_nonneg (by exact_mod_cast hfpos) hp.le
  rw [MPIR.truncate, down, Rat.cast_mul, Rat.cast_intCast, hf]
  split <;> simp [abs_of_nonneg hpos]

theorem truncate_eq_down {bits : ℕ} {x : ℚ} (hx : 0 ≤ x) :
    MPIR.truncate (x : ℝ) (grid bits : ℝ) = (down bits x : ℝ) := by
  have hx' : 0 ≤ (x : ℝ) := Rat.cast_nonneg.mpr hx
  have hu : (0 : ℝ) < (grid bits : ℝ) := by exact_mod_cast grid_pos bits
  have hnonneg : 0 ≤ MPIR.truncate (x : ℝ) (grid bits : ℝ) := by
    rw [MPIR.truncate, ite_eq_right (not_lt_of_ge hx'), abs_of_nonneg hx']
    have hf : 0 ≤ ⌊(x : ℝ) / (grid bits : ℝ)⌋ :=
      Int.floor_nonneg.mpr (div_nonneg hx' hu.le)
    exact mul_nonneg (by exact_mod_cast hf) hu.le
  simpa only [abs_of_nonneg hx, abs_of_nonneg hnonneg] using abs_truncate_eq_down bits x

theorem abs_keep_eq_down (a : MPIR.Limbs) (x : ℚ) (n : ℕ) (hn : 0 < n)
    (hTop : a.top = 1) (hx : a.value = x) :
    |MPIR.keep a n| = (down (64 * (n - 1)) |x| : ℝ) := by
  rw [keep_eq_truncate, hTop, hx, grid_eq_dyadic n hn]
  exact abs_truncate_eq_down _ _

end FractalSharkProofs.ReferenceOrbit.RealAxis
