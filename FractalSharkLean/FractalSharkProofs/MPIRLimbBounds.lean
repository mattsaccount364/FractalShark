module

public import FractalSharkProofs.MPIR
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Exactness and error bounds for MPIR operations on known limb grids. -/

@[expose] public section

namespace FractalSharkProofs.MPIR

def Aligned (x : ℝ) (e : ℤ) : Prop := ∃ k : ℤ, x = k * grid e

theorem aligned_at_finer_grid {x : ℝ} {e f : ℤ}
    (h : Aligned x e) (hf : f ≤ e) : Aligned x f := by
  obtain ⟨k, rfl⟩ := h
  let n := (e - f).toNat
  have he : e = (n : ℤ) + f := by dsimp [n]; omega
  refine ⟨k * (radix : ℤ) ^ n, ?_⟩
  rw [he, grid_add, grid, zpow_natCast]
  push_cast
  ring

theorem aligned_sub {x y : ℝ} {e : ℤ}
    (hx : Aligned x e) (hy : Aligned y e) : Aligned (x - y) e := by
  obtain ⟨k, rfl⟩ := hx
  obtain ⟨l, rfl⟩ := hy
  exact ⟨k - l, by push_cast; ring⟩

theorem aligned_limbs (a : Limbs) {e : ℤ}
    (he : e ≤ a.top - (a.used : ℤ)) : Aligned a.value e :=
  aligned_at_finer_grid ⟨a.mantissa, rfl⟩ he

theorem truncate_aligned {x : ℝ} {e : ℤ} (h : Aligned x e) :
    truncate x (grid e) = x := by
  obtain ⟨k, rfl⟩ := h
  have hp := grid_pos e
  simp only [truncate, abs_mul, abs_of_pos hp, mul_div_cancel_right₀ _ hp.ne',
    ← Int.cast_abs, Int.floor_intCast]
  split <;> rename_i hs
  · have hk : (k : ℝ) < 0 := lt_of_mul_lt_mul_right (by simpa only [zero_mul] using hs) hp.le
    rw [Int.cast_abs, abs_of_neg hk, neg_mul, neg_neg]
  · have hk : (0 : ℝ) ≤ k := nonneg_of_mul_nonneg_left (le_of_not_gt hs) hp
    rw [Int.cast_abs, abs_of_nonneg hk]

theorem truncate_on_grid (x : ℝ) (e : ℤ) :
    Aligned (truncate x (grid e)) e := by
  unfold truncate
  split
  · exact ⟨-⌊|x| / grid e⌋, by push_cast; ring⟩
  · exact ⟨⌊|x| / grid e⌋, rfl⟩

theorem retainedSum_aligned {a b : Limbs} {top cut : ℤ}
    (hcut : cut ≤ top) (ha : Aligned a.value cut) (hb : Aligned b.value cut) :
    retainedSum a b top cut = a.value + b.value := by
  have htail (x : ℝ) (hx : Aligned x cut) : Aligned (tail x top) cut :=
    aligned_sub hx (aligned_at_finer_grid (truncate_on_grid x top) hcut)
  rw [retainedSum, truncate_aligned (htail a.value ha), truncate_aligned (htail b.value hb)]
  unfold tail prefixSum
  ring

theorem abs_lt_top (a : Limbs) : |a.value| < grid a.top := by
  rcases a.normalized with ⟨ha, _⟩ | ⟨_, _, ha⟩
  · simpa [Limbs.value, ha] using grid_pos a.top
  · have hu : |(a.mantissa : ℝ)| < (radix : ℝ) ^ a.used := by exact_mod_cast ha
    calc
      |a.value| = |(a.mantissa : ℝ)| * grid (a.top - a.used) := by
        simp [Limbs.value, abs_of_pos (grid_pos _)]
      _ < (radix : ℝ) ^ a.used * grid (a.top - a.used) :=
        mul_lt_mul_of_pos_right hu (grid_pos _)
      _ = grid a.top := by
        rw [← show grid a.used = (radix : ℝ) ^ a.used by simp [grid],
          ← grid_add]
        congr 1
        omega

theorem truncate_eq_zero {x : ℝ} {e : ℤ} (h : |x| < grid e) :
    truncate x (grid e) = 0 := by
  have hp := grid_pos e
  have hfloor : ⌊|x| / grid e⌋ = (0 : ℤ) :=
    Int.floor_eq_zero_iff.mpr ⟨div_nonneg (abs_nonneg _) hp.le, (div_lt_one hp).mpr h⟩
  simp [truncate, hfloor]

theorem keep_eq_truncate (a : Limbs) (n : ℕ) :
    keep a n = truncate a.value (grid (a.top - n)) := by
  unfold keep
  split
  · exact (truncate_aligned (aligned_limbs a (by omega))).symm
  · rfl

theorem add_zero_rhs {p : Precision} {a b r : Limbs}
    (ha : a.used ≤ p.limbs + 1) (hb : b.value = 0) (h : Add p a b r) :
    r.value = a.value := by
  by_cases hz : a.value = 0
  · have he := add_error h
    have he' : |r.value| ≤ 0 := by simpa [hz, hb] using he
    rw [hz, ← abs_eq_zero]
    exact le_antisymm he' (abs_nonneg _)
  cases h with
  | zeroRight _ hr => exact hr.elim id (set_exact_of_fits ha)
  | zeroLeft hz' _ => exact (hz hz').elim
  | sameSign hs _ => simp [hb] at hs
  | difference hs _ _ _ _ => simp [hb] at hs
  | borrow hs _ _ _ _ => simp [hb] at hs
  | remainder hs _ _ _ _ _ _ _ => simp [hb] at hs
/-- The least retained limb position for the nonzero magnitude operation. -/
noncomputable def addCut (p : Precision) (a b : Limbs) : ℤ :=
  maximumTop a b - p.limbs - if a.value * b.value < 0 then 1 else 0

private theorem prefix_zero_at_maximum (a b : Limbs) :
    prefixSum a b (maximumTop a b) = 0 := by
  have ha := (abs_lt_top a).trans_le (grid_mono (le_max_left a.top b.top))
  have hb := (abs_lt_top b).trans_le (grid_mono (le_max_right a.top b.top))
  simp only [prefixSum, leadingPart, maximumTop, truncate_eq_zero ha, truncate_eq_zero hb,
    add_zero]

/-- Alignment concerns represented values. A wider slot may include trailing
zero limbs beyond the grid retained by an operation. -/
theorem add_aligned_bounds {p : Precision} {a b r : Limbs} (h : Add p a b r)
    (ha : a.value ≠ 0) (hb : b.value ≠ 0) :
    (Aligned a.value (addCut p a b) → Aligned b.value (addCut p a b) →
      r.value = a.value + b.value) ∧
    |r.value - (a.value + b.value)| ≤ 2 * grid (addCut p a b) := by
  have tailCase {top cut : ℤ} (hc : cut ≤ top) (hle : cut ≤ addCut p a b)
      (hr : r.value = retainedSum a b top cut) :
      (Aligned a.value (addCut p a b) → Aligned b.value (addCut p a b) →
        r.value = a.value + b.value) ∧
      |r.value - (a.value + b.value)| ≤ 2 * grid (addCut p a b) := by
    constructor
    · intro hleft hright
      rw [hr]
      exact retainedSum_aligned hc
        (aligned_at_finer_grid hleft hle) (aligned_at_finer_grid hright hle)
    · rw [hr]
      exact (retainedSum_error a b top cut).trans
        (mul_le_mul_of_nonneg_left (grid_mono hle) (by norm_num))
  cases h with
  | zeroLeft hz _ => exact (ha hz).elim
  | zeroRight hz _ => exact (hb hz).elim
  | sameSign hs hr =>
    have hcut : addCut p a b = maximumTop a b - p.limbs := by
      simp [addCut, not_lt_of_ge hs.le]
    constructor
    · intro hleft hright
      rw [hcut] at hleft hright
      rw [hr, truncate_aligned hleft, truncate_aligned hright]
    · rw [hr, hcut]
      exact truncate_sum_error a.value b.value _
  | difference hs top ht _ hr =>
    apply tailCase (by omega) _ hr
    simp [addCut, hs]
    omega
  | remainder hs top cut ht hc _ _ hr =>
    apply tailCase (by omega) _ hr
    simp [addCut, hs]
    omega
  | borrow hs top ht hprefix hr =>
    have hlt : top < maximumTop a b := by
      by_contra hge
      have he : top = maximumTop a b := by omega
      rw [he, prefix_zero_at_maximum, abs_zero] at hprefix
      exact (grid_pos _).ne' hprefix.symm
    apply tailCase (by omega) _ hr
    simp [addCut, hs]
    omega

/-- A nonzero addition has zero error when the operands occupy the retained
aligned span. Otherwise two grid quanta already bound the modeled error. -/
theorem add_span_bounds {p : Precision} {a b r : Limbs} (h : Add p a b r)
    (ha : a.value ≠ 0) (hb : b.value ≠ 0) :
    (addCut p a b ≤ min (a.top - a.used) (b.top - b.used) →
      r.value = a.value + b.value) ∧
    |r.value - (a.value + b.value)| ≤ 2 * grid (addCut p a b) := by
  obtain ⟨haligned, herror⟩ := add_aligned_bounds h ha hb
  exact ⟨fun hcut => haligned (aligned_limbs a (hcut.trans (min_le_left _ _)))
    (aligned_limbs b (hcut.trans (min_le_right _ _))), herror⟩

theorem sub_span_bounds {p : Precision} {a b r : Limbs} (h : Sub p a b r)
    (ha : a.value ≠ 0) (hb : b.value ≠ 0) :
    (addCut p a b.neg ≤ min (a.top - a.used) (b.top - b.used) →
      r.value = a.value - b.value) ∧
      |r.value - (a.value - b.value)| ≤ 2 * grid (addCut p a b.neg) := by
  have ht : b.neg.top = b.top := rfl
  have hu : b.neg.used = b.used := rfl
  simpa only [Limbs.value_neg, sub_eq_add_neg, ht, hu] using
    add_span_bounds h ha (by simpa using hb)

noncomputable def copyAllowance (p : Precision) (a : Limbs) : ℝ :=
  if a.used ≤ p.limbs + 1 then 0 else grid (a.top - (p.limbs + 1 : ℕ))

theorem set_limb_error {p : Precision} {a r : Limbs} (h : Set p a r) :
    |r.value - a.value| ≤ copyAllowance p a := by
  rw [h, keep, copyAllowance]
  split
  · simp
  · exact (truncate_bounds _ (grid_pos _)).2

noncomputable def sumAllowance (p : Precision) (a b : Limbs) : ℝ :=
  if a.value = 0 then copyAllowance p b
  else if b.value = 0 then copyAllowance p a
  else if addCut p a b ≤ min (a.top - a.used) (b.top - b.used) then 0
  else 4 * grid (addCut p a b)

theorem used_eq_zero_of_value_eq_zero (v : Limbs) (hv : v.value = 0) : v.used = 0 := by
  rcases v.normalized with h | ⟨_, hlo, _⟩
  · exact h.2
  · have hm : (v.mantissa : ℝ) = 0 := (mul_eq_zero.mp hv).resolve_right (grid_pos _).ne'
    have hm' : v.mantissa = 0 := by exact_mod_cast hm
    have hp : (0 : ℤ) < (radix : ℤ) ^ (v.used - 1) := pow_pos (by norm_num [radix]) _
    rw [hm', abs_zero] at hlo
    omega

theorem add_limb_error {p : Precision} {a b r : Limbs} (h : Add p a b r) :
    |r.value - (a.value + b.value)| ≤ sumAllowance p a b := by
  have hcopy (v : Limbs) (hr : r.value = v.value ∨ Set p v r) :
      |r.value - v.value| ≤ copyAllowance p v := by
    rcases hr with hr | hr
    · rw [hr, sub_self, abs_zero, copyAllowance]
      split
      · exact le_rfl
      · exact (grid_pos _).le
    · exact set_limb_error hr
  by_cases ha : a.value = 0
  · cases h with
    | zeroLeft hz hr => simpa [sumAllowance, hz] using hcopy b hr
    | zeroRight hz hr =>
      simpa [sumAllowance, ha, hz, copyAllowance, used_eq_zero_of_value_eq_zero a ha,
        used_eq_zero_of_value_eq_zero b hz] using hcopy a hr
    | sameSign hs _ => simp [ha] at hs
    | difference hs _ _ _ _ => simp [ha] at hs
    | borrow hs _ _ _ _ => simp [ha] at hs
    | remainder hs _ _ _ _ _ _ _ => simp [ha] at hs
  by_cases hb : b.value = 0
  · cases h with
    | zeroLeft hz _ => exact (ha hz).elim
    | zeroRight hz hr => simpa [sumAllowance, ha, hz] using hcopy a hr
    | sameSign hs _ => simp [hb] at hs
    | difference hs _ _ _ _ => simp [hb] at hs
    | borrow hs _ _ _ _ => simp [hb] at hs
    | remainder hs _ _ _ _ _ _ _ => simp [hb] at hs
  have h' := add_span_bounds h ha hb
  simp only [sumAllowance, ha, hb, ite_false]
  split
  · rename_i hf
    simp [h'.1 hf]
  · exact h'.2.trans (by linarith [grid_pos (addCut p a b)])

theorem sub_limb_error {p : Precision} {a b r : Limbs} (h : Sub p a b r) :
    |r.value - (a.value - b.value)| ≤ sumAllowance p a b.neg := by
  simpa only [Limbs.value_neg, sub_eq_add_neg] using add_limb_error h

theorem top_le_of_abs_lt (a : Limbs) (hzero : a.value ≠ 0) {k : ℤ}
    (h : |a.value| < grid k) : a.top ≤ k := by
  by_contra hn
  have le := (grid_mono (show k ≤ a.top - 1 by omega)).trans (a.grid_top_le hzero)
  exact (not_lt_of_ge le) h

theorem doubling_limb_error {p : Precision} {a r : Limbs} (h : MulUiTwo p a r) :
    |r.value - 2 * a.value| ≤
      if a.used ≤ p.limbs then 0 else grid (a.top - p.limbs) := by
  rw [h]
  split
  · simp
  · exact (truncate_bounds _ (grid_pos _)).2

end FractalSharkProofs.MPIR
