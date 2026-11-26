module

public import FractalSharkProofs.ReferenceOrbit.MPIRArithmetic
public import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Exact
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Finite-precision accuracy of the serial reference orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.SingleThreaded

/-- The imaginary update is built in scratch while both old coordinates are still live.
The final swap publishes that scratch value exactly. -/
structure Step {p : MPIR.Precision} (c old next : State p) : Prop where
  run : ∃ (doubled imagProduct sum difference realProduct : Slot p),
    MPIR.Scale p 1 old.x.val doubled.val ∧
    MPIR.Mul p doubled.val old.y.val imagProduct.val ∧
    MPIR.Add p imagProduct.val c.y.val next.y.val ∧
    MPIR.Add p old.x.val old.y.val sum.val ∧
    MPIR.Sub p old.x.val old.y.val difference.val ∧
    MPIR.Mul p sum.val difference.val realProduct.val ∧
    MPIR.Add p realProduct.val c.x.val next.x.val

private theorem product_error (a b x y : ℝ) :
    |a * b - x * y| ≤ |a - x| * |b| + |x| * |b - y| := by
  rw [show a * b - x * y = (a - x) * b + x * (b - y) by ring]
  simpa only [abs_mul] using abs_add_le ((a - x) * b) (x * (b - y))

private theorem realStep_error {p : MPIR.Precision} {x y c sum diff prod r : MPIR.Limbs}
    (hsum : MPIR.Add p x y sum) (hdiff : MPIR.Sub p x y diff)
    (hmul : MPIR.Mul p sum diff prod) (hadd : MPIR.Add p prod c r) :
    |r.value - ((x.value + y.value) * (x.value - y.value) + c.value)| ≤
      p.errorUnit * (26 * (|x.value| + |y.value|) ^ 2 + 2 * |c.value|) := by
  let size := |x.value| + |y.value|
  let u := p.errorUnit
  have hsize : 0 ≤ size := add_nonneg (abs_nonneg _) (abs_nonneg _)
  have hsumErr : |sum.value - (x.value + y.value)| ≤ 2 * u * size := add_error_sum hsum
  have hdiffErr : |diff.value - (x.value - y.value)| ≤ 2 * u * size := sub_error_sum hdiff
  have ha : |x.value + y.value| ≤ size := abs_add_le _ _
  have hb : |x.value - y.value| ≤ size := abs_sub _ _
  have hunit := mul_le_mul_of_nonneg_right p.errorUnit_le_half hsize
  have hsumSize : |sum.value| ≤ 2 * size := by
    have h := abs_sub_le sum.value (x.value + y.value) 0
    simp only [sub_zero] at h
    nlinarith only [h, ha, hsumErr, hunit]
  have hdiffSize : |diff.value| ≤ 2 * size := by
    have h := abs_sub_le diff.value (x.value - y.value) 0
    simp only [sub_zero] at h
    nlinarith only [h, hb, hdiffErr, hunit]
  have hproductSize : |sum.value * diff.value| ≤ (2 * size) * (2 * size) := by
    rw [abs_mul]
    exact mul_le_mul hsumSize hdiffSize (abs_nonneg _) (by linarith)
  have hproductSize' := mul_le_mul_of_nonneg_left hproductSize p.errorUnit_pos.le
  have hpropagate₁ := mul_le_mul hsumErr hdiffSize (abs_nonneg diff.value)
    (mul_nonneg (mul_nonneg (by norm_num) p.errorUnit_pos.le) hsize)
  have hpropagate₂ := mul_le_mul ha hdiffErr (abs_nonneg _) hsize
  have hphysical := mulThenAdd_error hmul hadd
  have hfactors := product_error sum.value diff.value (x.value + y.value) (x.value - y.value)
  calc
    _ ≤ |r.value - (sum.value * diff.value + c.value)| +
        |(sum.value * diff.value + c.value) -
          ((x.value + y.value) * (x.value - y.value) + c.value)| := abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub]
      change _ ≤ u * (26 * size ^ 2 + 2 * |c.value|)
      nlinarith only [hphysical, hproductSize', hfactors, hpropagate₁, hpropagate₂]

private theorem imagStep_error {p : MPIR.Precision} {x y c doubled prod r : MPIR.Limbs}
    (hscale : MPIR.Scale p 1 x doubled) (hmul : MPIR.Mul p doubled y prod)
    (hadd : MPIR.Add p prod c r) :
    |r.value - ((2 * x.value) * y.value + c.value)| ≤
      p.errorUnit * (3 * (|x.value| + |y.value|) ^ 2 + 2 * |c.value|) := by
  obtain ⟨hsize, hscaleErr⟩ := MPIR.scale_bounds hscale
  simp only [pow_one] at hsize hscaleErr
  have hproduct : |doubled.value * y.value| ≤ (2 * |x.value|) * |y.value| := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_right hsize (abs_nonneg _)
  have hproduct' := mul_le_mul_of_nonneg_left hproduct p.errorUnit_pos.le
  have hpropagate := mul_le_mul_of_nonneg_right hscaleErr (abs_nonneg y.value)
  have hphysical := mulThenAdd_error hmul hadd
  have hxy : 4 * |x.value| * |y.value| ≤ (|x.value| + |y.value|) ^ 2 := by
    nlinarith only [sq_nonneg (|x.value| - |y.value|)]
  have hxy' := mul_le_mul_of_nonneg_left hxy p.errorUnit_pos.le
  calc
    _ ≤ |r.value - (doubled.value * y.value + c.value)| +
        |(doubled.value * y.value + c.value) - ((2 * x.value) * y.value + c.value)| :=
      abs_sub_le _ _ _
    _ ≤ _ := by
      rw [add_sub_add_right_eq_sub, ← sub_mul, abs_mul]
      nlinarith only [hphysical, hproduct', hpropagate, hxy']

/-- One body incurs two factor errors in the real coordinate and one scaling error
in the imaginary coordinate. The two coordinate bounds contribute 26 and 3 units
of roundoff times the squared state size, rounded up to 32 here. -/
theorem step_error {p : MPIR.Precision} {c old next : State p} (h : Step c old next) :
    norm1 (next.value - (old.value ^ 2 + c.value)) ≤
      p.errorUnit * (32 * norm1 old.value ^ 2 + 2 * norm1 c.value) := by
  obtain ⟨doubled, imagProduct, sum, difference, realProduct,
    hscale, hmulY, haddY, hsum, hdiff, hmulX, haddX⟩ := h.run
  rw [← exactStep_eq]
  have hx := realStep_error hsum hdiff hmulX haddX
  have hy := imagStep_error hscale hmulY haddY
  simp only [norm1, Complex.sub_re, Complex.sub_im, exactStep, State.value]
  have hnonneg : 0 ≤ p.errorUnit *
      (|old.x.val.value| + |old.y.val.value|) ^ 2 :=
    mul_nonneg p.errorUnit_pos.le (sq_nonneg _)
  nlinarith only [hx, hy, hnonneg]

/-- The serial schedule has its own local residual inside the common orbit bound. -/
noncomputable def errorBound (p : MPIR.Precision) (target working : ℂ) (w : ℕ → ℂ) : ℕ → ℝ :=
  prefixBound target working w (fun n => p.errorUnit * (32 * norm1 (w n) ^ 2 + 2 * norm1 working))

/-- Every state in a finite executed prefix satisfies the bound, including the synthetic zero entry. -/
theorem prefix_error_le {p : MPIR.Precision} (target : ℂ) (working : State p)
    (states : ℕ → State p) (last : ℕ)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ n, 1 ≤ n → n < last → Step working (states n) (states (n + 1))) :
    ∀ n, n ≤ last → norm1 ((states n).value - orbit target n) ≤
      errorBound p target working.value (fun i => (states i).value) n := by
  exact prefixBound_correct target working.value (fun i => (states i).value) _ last
    hzero (firstState_eq hx hy) (fun n hn hlast => step_error (hstep n hn hlast))

end FractalSharkProofs.ReferenceOrbit.SingleThreaded
