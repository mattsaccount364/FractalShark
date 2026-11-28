module

public import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Precision
public import FractalSharkProofs.ReferenceOrbit.RealAxis
import Mathlib.Tactic.Positivity

/-! A computable ST step for real orbits whose nonzero operands have limb exponent one. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.SingleThreaded

open PrecisionEstimate RealAxis
open MPIR (keep_eq_truncate add_zero_rhs)

def realAxisSquare (p : MPIR.Precision) (x : ℚ) : ℚ :=
  down (p.bits + 64) ((down p.bits |x|) ^ 2)

def realAxisNext (p : MPIR.Precision) (c x : ℚ) : ℚ := realAxisSquare p x + c

def realAxisValue (x : ℚ) : ℂ := ⟨x, 0⟩

/-- The search checks these ranges before using the single-limb-exponent formulas.
Numbers near two use the same exponent for the input, square, and parameter. -/
def RealAxisWindow (p : MPIR.Precision) (c x : ℚ) : Prop :=
  -2 ≤ c ∧ c < -1 ∧ 7 / 4 ≤ |x| ∧ |x| ≤ 2 ∧ 3 ≤ realAxisSquare p x

instance (p : MPIR.Precision) (c x : ℚ) : Decidable (RealAxisWindow p c x) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _))

set_option maxHeartbeats 500000 in
/-- For a real step in the search window, each sum with zero copies exactly.
The square trims its inputs and product on different limb boundaries. The
opposite-sign addition uses the full product and parameter without further loss. -/
theorem realAxis_step {p : MPIR.Precision} {working old next : State p} {c x : ℚ}
    (hc : working.value = realAxisValue c) (hx : old.value = realAxisValue x)
    (hwin : RealAxisWindow p c x) (hstep : Step working old next) :
    next.value = realAxisValue (realAxisNext p c x) := by
  obtain ⟨cLo, cHi, xLo, xHi, sqLo⟩ := hwin
  obtain ⟨doubled, imagProduct, sum, diff, realProduct,
    _, hmulY, haddY, hsum, hdiff, hmulX, haddX⟩ := hstep.run
  have hcx : working.x.val.value = (c : ℝ) := congrArg Complex.re hc
  have hcy : working.y.val.value = 0 := congrArg Complex.im hc
  have hxx : old.x.val.value = (x : ℝ) := congrArg Complex.re hx
  have hxy : old.y.val.value = 0 := congrArg Complex.im hx
  have hsy : imagProduct.val.value = 0 := by
    have h := (MPIR.mul_bounds hmulY).1
    simpa [hxy] using h
  have hy : next.y.val.value = 0 := (add_zero_rhs imagProduct.property hcy haddY).trans hsy
  have hs : sum.val.value = (x : ℝ) := (add_zero_rhs old.x.property hxy hsum).trans hxx
  have hd : diff.val.value = (x : ℝ) :=
    (add_zero_rhs old.x.property (by rw [MPIR.Limbs.value_neg, hxy, neg_zero]) hdiff).trans hxx
  have hradix : (5 : ℝ) < MPIR.grid 1 := by norm_num [MPIR.grid, MPIR.radix]
  have hlo : (7 / 4 : ℝ) ≤ |(x : ℝ)| := by
    simpa using (Rat.cast_le (K := ℝ)).mpr xLo
  have hhi : |(x : ℝ)| ≤ 2 := by exact_mod_cast xHi
  have htop (a : MPIR.Limbs) (h : a.value = (x : ℝ)) : a.top = 1 :=
    top_eq_one a (by rw [h]; linarith) (by rw [h]; linarith)
  obtain ⟨product, hprod, hrp⟩ := hmulX
  have hp : 0 < p.limbs := by have := p.min_limbs; omega
  let u := down p.bits |x|
  have hsame : MPIR.keep sum.val p.limbs = MPIR.keep diff.val p.limbs := by
    rw [keep_eq_truncate sum.val, keep_eq_truncate diff.val, htop _ hs, htop _ hd, hs, hd]
  have hprod_eq : product.value = (u ^ 2 : ℚ) := by
    rw [hprod, ← hsame, ← pow_two, ← sq_abs, abs_keep_eq_down _ x p.limbs hp (htop _ hs) hs,
      Rat.cast_pow]
    rfl
  have hsqLo : (3 : ℝ) ≤ (realAxisSquare p x : ℝ) := by exact_mod_cast sqLo
  have hsq_le : (realAxisSquare p x : ℝ) ≤ (u : ℝ) ^ 2 := by
    rw [← Rat.cast_pow]
    exact (Rat.cast_le (K := ℝ)).mpr (down_le (p.bits + 64) (u ^ (2 : ℕ)))
  have huv : (u : ℝ) ≤ |(x : ℝ)| := by exact_mod_cast (down_le p.bits |x|)
  have hun : 0 ≤ (u : ℝ) := by
    dsimp [u, down]
    push_cast
    apply mul_nonneg _ (by exact_mod_cast (grid_pos p.bits).le)
    exact_mod_cast Int.floor_nonneg.mpr (div_nonneg (abs_nonneg _) (grid_pos _).le)
  have hu4 : (u : ℝ) ^ 2 ≤ 4 := by nlinarith
  have hprod_abs : |product.value| = (u : ℝ) ^ 2 := by simp [hprod_eq, Rat.cast_pow]
  have hprod_top : product.top = 1 := top_eq_one product (by rw [hprod_abs]; linarith)
    (by rw [hprod_abs]; linarith)
  have hbits : 64 * p.limbs = p.bits + 64 := by
    unfold MPIR.Precision.bits
    have := p.min_limbs
    omega
  have hprod_trim : realProduct.val.value = (realAxisSquare p x : ℝ) := by
    rw [hrp, keep_eq_truncate, hprod_top, hprod_eq, grid_eq_dyadic _ (by omega),
      Nat.add_sub_cancel, hbits]
    exact truncate_eq_down (sq_nonneg u)
  have hrpTop : realProduct.val.top = 1 := by
    apply top_eq_one _ <;> rw [hprod_trim, abs_of_nonneg (by linarith : 0 ≤ (realAxisSquare p x : ℝ))]
    · linarith
    · linarith
  have hcLo : (-2 : ℝ) ≤ c := by exact_mod_cast cLo
  have hcHi : (c : ℝ) < -1 := by exact_mod_cast cHi
  have hcTop : working.x.val.top = 1 := by
    apply top_eq_one _ <;> rw [hcx, abs_of_neg (by linarith : (c : ℝ) < 0)] <;> linarith
  have hreal := add_opposite_exact realProduct.property working.x.property hrpTop hcTop
    (by rw [hprod_trim]; linarith) (by rw [hcx]; linarith) haddX
  apply Complex.ext
  · simpa [State.value, realAxisValue, realAxisNext, hprod_trim, hcx, Rat.cast_add] using hreal
  · exact hy

end FractalSharkProofs.ReferenceOrbit.SingleThreaded
