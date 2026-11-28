module

public import FractalSharkProofs.ReferenceOrbit.RealAxis
public import FractalSharkProofs.ReferenceOrbit.MT9.Certificate
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

/-! Real-axis MT9 updates around the repelling state at two. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

open PrecisionEstimate RealAxis

def realValue (x : ℚ) : ℂ := ⟨x, 0⟩

def nearTwo (x : ℚ) : Prop := 7 / 4 ≤ |x| ∧ |x| ≤ 2

instance (x : ℚ) : Decidable (nearTwo x) :=
  inferInstanceAs (Decidable (_ ∧ _))

/-- A precision change trims the occupied low limbs. During multiplication
MT9 uses all the limbs left in the input, including the guard limb. -/
def realClip (p : MPIR.Precision) (x : ℚ) : ℚ :=
  if x < 0 then -down (p.bits + 64) (-x) else down (p.bits + 64) x

def realSquare (p : MPIR.Precision) (x : ℚ) : ℚ :=
  down (p.bits + 64) (x ^ (2 : ℕ))

def realNext (p : MPIR.Precision) (c x : ℚ) : ℚ := realSquare p (realClip p x) + c

def RealWindow (p : MPIR.Precision) (c x : ℚ) : Prop :=
  -2 ≤ c ∧ c < -1 ∧ nearTwo x ∧ 3 ≤ realSquare p x ∧ down (p.bits + 64) (-c) = -c

instance (p : MPIR.Precision) (c x : ℚ) : Decidable (RealWindow p c x) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

private theorem nearTwo_top (a : MPIR.Limbs) {x : ℚ}
    (hx : a.value = (x : ℝ)) (h : nearTwo x) : a.top = 1 := by
  have hlo : (7 / 4 : ℝ) ≤ |(x : ℝ)| := by simpa using (Rat.cast_le (K := ℝ)).mpr h.1
  have hhi : |(x : ℝ)| ≤ 2 := by exact_mod_cast h.2
  have hg : (2 : ℝ) < MPIR.grid 1 := by norm_num [MPIR.grid, MPIR.radix]
  apply top_eq_one a <;> rw [hx] <;> linarith

theorem real_clip {previous active : MPIR.Precision} {before : State previous}
    {clipped : State active} {x : ℚ} (hx : before.value = realValue x)
    (hwin : nearTwo x) (hclip : Clip before clipped) :
    clipped.value = realValue (realClip active x) := by
  have hr : before.x.val.value = (x : ℝ) := congrArg Complex.re hx
  have hi : before.y.val.value = 0 := congrArg Complex.im hx
  have htop := nearTwo_top before.x.val hr hwin
  apply Complex.ext
  · change clipped.x.val.value = (realClip active x : ℝ)
    rw [hclip.x, MPIR.keep_eq_truncate, htop, hr,
      show (1 : ℤ) - (↑(active.limbs + 1) : ℤ) = -↑active.limbs by omega, guard_grid]
    by_cases hn : x < 0
    · have hnr : (x : ℝ) < 0 := by exact_mod_cast hn
      have hneg : MPIR.truncate (x : ℝ) (grid (active.bits + 64)) =
          -MPIR.truncate (-(x : ℝ)) (grid (active.bits + 64)) := by
        simp [MPIR.truncate, hnr, not_lt.mpr (neg_nonneg.mpr hnr.le), abs_of_neg hnr]
      rw [realClip, ite_eq_left hn, Rat.cast_neg, hneg]
      congr 1
      simpa only [Rat.cast_neg] using
        (truncate_eq_down (bits := active.bits + 64) (x := -x) (by linarith))
    · have hp : 0 ≤ x := le_of_not_gt hn
      rw [realClip, ite_eq_right hn]
      exact truncate_eq_down hp
  · change clipped.y.val.value = 0
    rw [hclip.y, MPIR.keep_eq_truncate, hi]
    simp [MPIR.truncate]

set_option maxHeartbeats 500000 in
/-- The square rounds once at the destination grid. The subtraction of a
zero square and the addition of the aligned negative center are exact. -/
theorem real_step {p full : MPIR.Precision} {working : State full} {old next : State p}
    {c x : ℚ} (hc : working.value = realValue c) (hx : old.value = realValue x)
    (hwin : RealWindow p c x) (hstep : Step working old next) :
    next.value = realValue (realSquare p x + c) := by
  obtain ⟨hcLo, hcHi, hnear, hsqLo, hcenterGrid⟩ := hwin
  obtain ⟨xs, ys, _, _, diff, hxProd, hyProd, _, _, _, hsub, hadd⟩ := hstep.run
  have hcx : working.x.val.value = (c : ℝ) := congrArg Complex.re hc
  have hcy : working.y.val.value = 0 := congrArg Complex.im hc
  have hxx : old.x.val.value = (x : ℝ) := congrArg Complex.re hx
  have hxy : old.y.val.value = 0 := congrArg Complex.im hx
  have hy : next.y.val.value = 0 := imaginary_zero hcy hxy hstep
  have hys : ys.val.value = 0 := (product_zero hyProd (.inl hxy)).1
  have hdiff := MPIR.add_zero_rhs xs.property (by rw [MPIR.Limbs.value_neg, hys, neg_zero]) hsub
  have xlo : (7 / 4 : ℝ) ≤ |(x : ℝ)| := by simpa using (Rat.cast_le (K := ℝ)).mpr hnear.1
  have xhi : |(x : ℝ)| ≤ 2 := by exact_mod_cast hnear.2
  have hradix : (5 : ℝ) < MPIR.grid 1 := by norm_num [MPIR.grid, MPIR.radix]
  obtain ⟨square, hprod, hkeep, _, _⟩ := hxProd.stored
  rw [hxx, ← pow_two] at hprod
  have hprodTop : square.top = 1 := by
    apply top_eq_one
    all_goals rw [hprod, abs_of_nonneg (sq_nonneg _), ← sq_abs]
    · nlinarith only [xlo, sq_nonneg (|(x : ℝ)| - 7 / 4)]
    · have h4 := pow_le_pow_left₀ (abs_nonneg (x : ℝ)) xhi 2
      linarith
  have htrim : xs.val.value = (realSquare p x : ℝ) := by
    rw [hkeep, MPIR.keep_eq_truncate, hprodTop, hprod,
      show (1 : ℤ) - (↑(p.limbs + 1) : ℤ) = -↑p.limbs by omega, guard_grid]
    simpa only [realSquare, Rat.cast_pow] using
      truncate_eq_down (bits := p.bits + 64) (x := x ^ (2 : ℕ)) (sq_nonneg x)
  have hdst : diff.val.value = (realSquare p x : ℝ) := hdiff.trans htrim
  have h3 : (3 : ℝ) ≤ (realSquare p x : ℝ) := by exact_mod_cast hsqLo
  have h4 : (realSquare p x : ℝ) ≤ 4 := by
    have hdown : (realSquare p x : ℝ) ≤ (x : ℝ) ^ 2 := by
      simpa only [realSquare, Rat.cast_pow] using
        (Rat.cast_le (K := ℝ)).mpr (down_le (p.bits + 64) (x ^ (2 : ℕ)))
    have hb := pow_le_pow_left₀ (abs_nonneg _) xhi 2
    norm_num at hb
    exact hdown.trans hb
  have hcLo' : (-2 : ℝ) ≤ c := by exact_mod_cast hcLo
  have hcHi' : (c : ℝ) < -1 := by exact_mod_cast hcHi
  have hpos : 0 < diff.val.value := by rw [hdst]; linarith
  have hneg : working.x.val.value < 0 := by rw [hcx]; linarith
  have hdiffTop := top_eq_one diff.val (by rw [abs_of_pos hpos, hdst]; linarith)
    (by rw [abs_of_pos hpos, hdst]; linarith)
  have hcTop := top_eq_one working.x.val (by rw [abs_of_neg hneg, hcx]; linarith)
    (by rw [abs_of_neg hneg, hcx]; linarith)
  have hcut : MPIR.addCut p diff.val working.x.val = -(p.limbs : ℤ) := by
    simp [MPIR.addCut, MPIR.maximumTop, hdiffTop, hcTop, mul_neg_of_pos_of_neg hpos hneg]
  have hleft : MPIR.Aligned diff.val.value (-(p.limbs : ℤ)) := by
    apply MPIR.aligned_limbs
    rw [hdiffTop]
    have := diff.property
    omega
  have hright : MPIR.Aligned working.x.val.value (-(p.limbs : ℤ)) := by
    have hal := MPIR.truncate_on_grid (-(c : ℝ)) (-(p.limbs : ℤ))
    rw [guard_grid, ← Rat.cast_neg, truncate_eq_down (by linarith : 0 ≤ -c),
      hcenterGrid, Rat.cast_neg] at hal
    rw [hcx]
    have hz : MPIR.Aligned (0 : ℝ) (-(p.limbs : ℤ)) := ⟨0, by simp⟩
    simpa only [zero_sub, neg_neg] using MPIR.aligned_sub hz hal
  have hsum := (MPIR.add_aligned_bounds hadd hpos.ne' hneg.ne).1
    (by rw [hcut]; exact hleft) (by rw [hcut]; exact hright)
  apply Complex.ext
  · change next.x.val.value = ((realSquare p x + c : ℚ) : ℝ)
    rw [hsum, hdst, hcx, Rat.cast_add]
  · exact hy

end FractalSharkProofs.ReferenceOrbit.MT9
