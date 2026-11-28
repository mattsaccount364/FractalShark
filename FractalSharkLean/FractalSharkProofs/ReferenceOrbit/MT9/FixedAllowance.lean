module

public import FractalSharkProofs.ReferenceOrbit.MT9.Dynamic
import Mathlib.Tactic.Positivity

/-! The full-width residual bound used by the dynamic scheduler without a second orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

private theorem copy_below (p : MPIR.Precision) (a : MPIR.Limbs) (k : ℤ)
    (h : |a.value| < MPIR.grid k) :
    MPIR.copyAllowance p a ≤ MPIR.grid (k - p.limbs) := by
  unfold MPIR.copyAllowance
  split
  · exact (MPIR.grid_pos _).le
  · rename_i hn
    have ha : a.value ≠ 0 := by
      intro hz
      have := MPIR.used_eq_zero_of_value_eq_zero a hz
      omega
    have ht := MPIR.top_le_of_abs_lt a ha h
    exact MPIR.grid_mono (by omega)

private theorem sum_below (p : MPIR.Precision) (a b : MPIR.Limbs) (k : ℤ)
    (ha : |a.value| < MPIR.grid k) (hb : |b.value| < MPIR.grid k) :
    MPIR.sumAllowance p a b ≤ 4 * MPIR.grid (k - p.limbs) := by
  have hg := MPIR.grid_pos (k - p.limbs)
  unfold MPIR.sumAllowance
  split
  · exact (copy_below p b k hb).trans (by linarith)
  split
  · exact (copy_below p a k ha).trans (by linarith)
  rename_i hza hzb
  split
  · linarith
  have htop : MPIR.maximumTop a b ≤ k :=
    max_le (MPIR.top_le_of_abs_lt a hza ha) (MPIR.top_le_of_abs_lt b hzb hb)
  have hcut : MPIR.addCut p a b ≤ k - p.limbs := by
    unfold MPIR.addCut
    split <;> omega
  exact mul_le_mul_of_nonneg_left (MPIR.grid_mono hcut) (by norm_num)

private theorem product_below (p : MPIR.Precision) (r : MPIR.Limbs) (k : ℤ)
    (h : |r.value| < MPIR.grid k) : productAllowance p r ≤ MPIR.grid (k - p.limbs) := by
  unfold productAllowance
  split
  · exact (MPIR.grid_pos _).le
  · rename_i hn
    have hr : r.value ≠ 0 := by
      intro hz
      have := MPIR.used_eq_zero_of_value_eq_zero r hz
      omega
    have ht := MPIR.top_le_of_abs_lt r hr h
    exact MPIR.grid_mono (by omega)

private theorem doubling_below (p : MPIR.Precision) (a : MPIR.Limbs) (k : ℤ)
    (h : |a.value| < MPIR.grid k) : doublingAllowance p a ≤ 4 * MPIR.grid (k - p.limbs) := by
  unfold doublingAllowance
  split
  · exact mul_nonneg (by norm_num) (MPIR.grid_pos _).le
  · rename_i hn
    have ha : a.value ≠ 0 := by
      intro hz
      have := MPIR.used_eq_zero_of_value_eq_zero a hz
      omega
    have ht := MPIR.top_le_of_abs_lt a ha h
    exact mul_le_mul_of_nonneg_left (MPIR.grid_mono (by omega)) (by norm_num)

/-- A normalized binary64 mantissa is strictly below two. Integer floor
division therefore yields a strict limb cap even for negative exponents. -/
theorem normalized_limb_ceiling (mantissa : ℝ) (exponent : ℤ) (h : mantissa < 2) :
    mantissa * (2 : ℝ) ^ exponent < MPIR.grid (exponent / 64 + 1) := by
  have he : exponent + 1 ≤ 64 * (exponent / 64 + 1) := by omega
  calc
    _ < 2 * (2 : ℝ) ^ exponent := mul_lt_mul_of_pos_right h (zpow_pos (by norm_num) _)
    _ = (2 : ℝ) ^ (exponent + 1) := by
      rw [zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), zpow_one, mul_comm]
    _ ≤ (2 : ℝ) ^ (64 * (exponent / 64 + 1)) := zpow_le_zpow_right₀ (by norm_num) he
    _ = _ := by
      rw [zpow_mul, MPIR.grid]
      norm_num [MPIR.radix]

set_option maxHeartbeats 500000 in
/-- K is a strict exponent cap for eight times the fixed-step scale. The
implemented floor division picks the least such K. Each component has at
most ten quanta of error, and their Euclidean norm fits sixteen quanta. -/
theorem fixed_rounding_allowance {p : MPIR.Precision} {c old next : State p}
    (h : Step c old next) (U C : ℝ) (K : ℤ)
    (hU : ‖old.value‖ ≤ U) (hC : ‖c.value‖ ≤ C)
    (hK : 8 * (U ^ 2 + C) < MPIR.grid K) :
    ‖next.value - (old.value ^ 2 + c.value)‖ ≤ 16 * MPIR.grid (K - p.limbs) := by
  obtain ⟨xs, ys, cross, doubled, tmp, hx, hy, hxy, hd, ha, hs, hb⟩ := h.run
  let S := U ^ 2 + C
  let q := MPIR.grid (K - p.limbs)
  have hC0 : 0 ≤ C := (norm_nonneg _).trans hC
  have hS0 : 0 ≤ S := add_nonneg (sq_nonneg _) hC0
  have hX : |old.x.val.value| ≤ U := (Complex.abs_re_le_norm old.value).trans hU
  have hY : |old.y.val.value| ≤ U := (Complex.abs_im_le_norm old.value).trans hU
  have bounded (a b : MPIR.Limbs) (ha : |a.value| ≤ U) (hb : |b.value| ≤ U)
      {r : MPIR.Limbs} (hprod : Product p a b r) : |r.value| ≤ S := by
    have hm := mul_le_mul ha hb (abs_nonneg _) ((abs_nonneg _).trans ha)
    have hp := (product_bounds hprod).1
    rw [abs_mul] at hp
    dsimp only [S]
    nlinarith only [hm, hp, hC0]
  have hxs := bounded _ _ hX hX hx
  have hys := bounded _ _ hY hY hy
  have hcross := bounded _ _ hX hY hxy
  have hdouble : |doubled.val.value| ≤ 2 * S := by
    exact (MPIR.mulUiTwo_bounds hd).1.trans (mul_le_mul_of_nonneg_left hcross (by norm_num))
  have htmp : |tmp.val.value| ≤ 3 * S := by
    have ht := abs_sub_le tmp.val.value (xs.val.value - ys.val.value) 0
    simp only [sub_zero] at ht
    have herr := MPIR.sub_error hs
    have hmax := max_le hxs hys
    have hu := mul_le_mul_of_nonneg_right p.errorUnit_le_half hS0
    have hm := mul_le_mul_of_nonneg_left hmax
      (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) p.errorUnit_pos.le)
    nlinarith only [ht, herr, hm, hu, abs_sub xs.val.value ys.val.value, hxs, hys]
  have small (v : ℝ) (hv : v ≤ 3 * S) : v < MPIR.grid K := by
    change 8 * S < MPIR.grid K at hK
    linarith only [hS0, hK, hv]
  have hxsK := small |xs.val.value| (by linarith only [hxs, hS0])
  have hysK := small |ys.val.value| (by linarith only [hys, hS0])
  have hcrossK := small |cross.val.value| (by linarith only [hcross, hS0])
  have hdK := small |doubled.val.value| (by linarith only [hdouble, hS0])
  have htK := small |tmp.val.value| htmp
  have hcxK : |c.x.val.value| < MPIR.grid K := by
    apply small
    have h := (Complex.abs_re_le_norm c.value).trans hC
    change |c.x.val.value| ≤ C at h
    dsimp only [S]
    nlinarith only [sq_nonneg U, h, hC0]
  have hcyK : |c.y.val.value| < MPIR.grid K := by
    apply small
    have h := (Complex.abs_im_le_norm c.value).trans hC
    change |c.y.val.value| ≤ C at h
    dsimp only [S]
    nlinarith only [sq_nonneg U, h, hC0]
  have hrX := (product_limb_error hx).trans (product_below p xs.val K hxsK)
  have hrY := (product_limb_error hy).trans (product_below p ys.val K hysK)
  have hrXY := (product_limb_error hxy).trans (product_below p cross.val K hcrossK)
  have hrDouble := (doubling_bound hd).trans (doubling_below p cross.val K hcrossK)
  have hrAddY := (MPIR.add_limb_error ha).trans (sum_below p doubled.val c.y.val K hdK hcyK)
  have hrSub := (MPIR.sub_limb_error hs).trans
    (sum_below p xs.val ys.val.neg K hxsK (by simpa using hysK))
  have hrAddX := (MPIR.add_limb_error hb).trans (sum_below p tmp.val c.x.val K htK hcxK)
  have hcomponents := component_residuals (c := c) (old := old) (next := next)
    (xs := xs) (ys := ys) (by simpa only [pow_two] using hrX)
      (by simpa only [pow_two] using hrY) hrXY hrDouble hrAddY hrSub hrAddX
  let residual := next.value - (old.value ^ 2 + c.value)
  have hReal : |residual.re| ≤ q + q + 4 * q + 4 * q := hcomponents.1
  have hImag : |residual.im| ≤ 2 * q + 4 * q + 4 * q := hcomponents.2
  have bound : ‖residual‖ ≤ Real.sqrt ((10 * q) ^ 2 + (10 * q) ^ 2) :=
    norm_components residual (by linarith only [hReal]) (by linarith only [hImag])
  apply bound.trans
  have hq : 0 < q := MPIR.grid_pos _
  change _ ≤ 16 * q
  rw [Real.sqrt_le_left (by positivity : 0 ≤ 16 * q)]
  nlinarith only [sq_nonneg q]

theorem fixed_zero_allowance {p : MPIR.Precision} {c old next : State p} (h : Step c old next)
    (hz : old.value = 0) (hc : c.value = 0) : ‖next.value - (old.value ^ 2 + c.value)‖ ≤ 0 := by
  have he := step_error h
  simp [hz, hc] at he ⊢
  exact norm_eq_zero.mp (le_antisymm
    ((Complex.norm_le_abs_re_add_abs_im next.value).trans he) (norm_nonneg _))

/-- The runtime can use the working radius and current difference disk to
bound the fixed step, even though it evaluates only the dynamic state. -/
theorem fixed_rounding_of_disk {p : MPIR.Precision} {c fixed next : State p}
    (h : Step c fixed next) (dynamic : ℂ) (R E C : ℝ) (K : ℤ)
    (hR : ‖dynamic‖ ≤ R) (hE : ‖dynamic - fixed.value‖ ≤ E) (hC : ‖c.value‖ ≤ C)
    (hK : 8 * ((R + E) ^ 2 + C) < MPIR.grid K) :
    ‖next.value - (fixed.value ^ 2 + c.value)‖ ≤ 16 * MPIR.grid (K - p.limbs) := by
  have hnorm : ‖fixed.value‖ ≤ R + E := by
    rw [show fixed.value = dynamic + -(dynamic - fixed.value) by ring]
    exact (norm_add_le _ _).trans (add_le_add hR (by simpa only [norm_neg] using hE))
  exact fixed_rounding_allowance h (R + E) C K hnorm hC hK

end FractalSharkProofs.ReferenceOrbit.MT9
