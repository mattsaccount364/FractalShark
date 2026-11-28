module

public import FractalSharkProofs.ReferenceOrbit.MT9.Precision
public import FractalSharkProofs.MPIRLimbBounds
public import Mathlib.Analysis.Complex.Norm
import Mathlib.Tactic.Positivity

/-! Dynamic working precision and its disk around the fixed MPIR trajectory. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

/-- The quantum is in 64-bit blocks. A proposal is the rounded-up tier
number, with none representing an unusable or full-precision proposal. -/
def tierQuantum (full : MPIR.Precision) : ℕ := max 1 ((full.limbs - 1) / 64)

def workingPrecision (full : MPIR.Precision) (allowReduction : Bool)
    (proposal : Option ℕ) : MPIR.Precision :=
  match allowReduction, proposal with
  | true, some tier =>
    ⟨min full.limbs (max 1 tier * tierQuantum full + 1), by
      have hq : 1 ≤ tierQuantum full := le_max_left _ _
      have ht : 1 ≤ max 1 tier * tierQuantum full := by
        simpa using Nat.mul_le_mul (le_max_left 1 tier) hq
      exact le_min full.min_limbs (by omega)⟩
  | _, _ => full

theorem workingPrecision_first (full : MPIR.Precision) (proposal : Option ℕ) :
    workingPrecision full false proposal = full := by cases proposal <;> rfl

theorem workingPrecision_fits (full : MPIR.Precision) (reduce : Bool) (proposal : Option ℕ) :
    (workingPrecision full reduce proposal).limbs ≤ full.limbs := by
  cases reduce <;> cases proposal <;> simp [workingPrecision]

theorem workingPrecision_aligned (full : MPIR.Precision) (reduce : Bool) (proposal : Option ℕ) :
    64 ∣ (workingPrecision full reduce proposal).bits ∧
      64 ≤ (workingPrecision full reduce proposal).bits := by
  refine ⟨dvd_mul_right _ _, ?_⟩
  have := (workingPrecision full reduce proposal).min_limbs
  unfold MPIR.Precision.bits
  omega

theorem resize_fits_full_allocation (full : MPIR.Precision) (reduce : Bool) (proposal : Option ℕ) :
    lowWidth (workingPrecision full reduce proposal) ≤ lowWidth full ∧
      highWidth (workingPrecision full reduce proposal) ≤ highWidth full :=
  active_width_fits (workingPrecision_fits full reduce proposal)

structure Clip {previous active : MPIR.Precision} (input : State previous) (output : State active) :
    Prop where
  x : MPIR.Set active input.x.val output.x.val
  y : MPIR.Set active input.y.val output.y.val

theorem clip_on_widening {previous active : MPIR.Precision}
    {input : State previous} {output : State active} (h : Clip input output)
    (hwidth : previous.limbs ≤ active.limbs) : output.value = input.value := by
  apply Complex.ext
  · exact MPIR.set_exact_of_fits (by have := input.x.property; omega) h.x
  · exact MPIR.set_exact_of_fits (by have := input.y.property; omega) h.y

/-- Restoring the allocation metadata does not copy or extend the occupied
mantissa. The original full-width storage still backs both slots. -/
def restoreFull {active full : MPIR.Precision} (h : active.limbs ≤ full.limbs)
    (state : State active) : State full :=
  ⟨⟨state.x.val, by have := state.x.property; omega⟩,
    ⟨state.y.val, by have := state.y.property; omega⟩⟩

@[simp] theorem restoreFull_value {active full : MPIR.Precision}
    (h : active.limbs ≤ full.limbs) (state : State active) :
    (restoreFull h state).value = state.value := rfl

theorem norm_components (z : ℂ) {x y : ℝ}
    (hx : |z.re| ≤ x) (hy : |z.im| ≤ y) :
    ‖z‖ ≤ Real.sqrt (x ^ 2 + y ^ 2) := by
  have hx2 := pow_le_pow_left₀ (abs_nonneg _) hx 2
  have hy2 := pow_le_pow_left₀ (abs_nonneg _) hy 2
  simp only [sq_abs] at hx2 hy2
  have hnorm : ‖z‖ ^ 2 = z.re ^ 2 + z.im ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    ring
  have hsum : 0 ≤ x ^ 2 + y ^ 2 := add_nonneg (sq_nonneg _) (sq_nonneg _)
  nlinarith only [hx2, hy2, hnorm, Real.sq_sqrt hsum, norm_nonneg z,
    Real.sqrt_nonneg (x ^ 2 + y ^ 2)]

/-- State clipping is a copy at the newly installed precision. Raising the
metadata again restores capacity, and the previous enclosure still applies. -/
theorem clip_error {previous active : MPIR.Precision} {input : State previous}
    {output : State active} (h : Clip input output) :
    ‖output.value - input.value‖ ≤ Real.sqrt
      (MPIR.copyAllowance active input.x.val ^ 2 + MPIR.copyAllowance active input.y.val ^ 2) :=
  norm_components _ (MPIR.set_limb_error h.x) (MPIR.set_limb_error h.y)

theorem clip_size {previous active : MPIR.Precision} {input : State previous}
    {output : State active} (h : Clip input output) : ‖output.value‖ ≤ ‖input.value‖ := by
  have hx := h.x ▸ MPIR.keep_abs_le input.x.val (active.limbs + 1)
  have hy := h.y ▸ MPIR.keep_abs_le input.y.val (active.limbs + 1)
  have hn := norm_components output.value hx hy
  apply hn.trans_eq
  rw [sq_abs, sq_abs]
  have he : input.x.val.value ^ 2 + input.y.val.value ^ 2 = ‖input.value‖ ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    simp only [State.value, pow_two]
  rw [he, Real.sqrt_sq (norm_nonneg _)]

noncomputable def productAllowance (p : MPIR.Precision) (r : MPIR.Limbs) : ℝ :=
  if r.used < p.limbs + 1 then 0 else MPIR.grid (r.top - (p.limbs + 1 : ℕ))

noncomputable def doublingAllowance (p : MPIR.Precision) (x : MPIR.Limbs) : ℝ :=
  if x.used ≤ p.limbs then 0 else 4 * MPIR.grid (x.top - p.limbs)

theorem doubling_bound {p : MPIR.Precision} {x r : MPIR.Limbs} (h : MPIR.MulUiTwo p x r) :
    |r.value - 2 * x.value| ≤ doublingAllowance p x := by
  refine (MPIR.doubling_limb_error h).trans ?_
  unfold doublingAllowance
  split
  · simp
  · linarith [MPIR.grid_pos (x.top - p.limbs)]

/-- The four real and three imaginary primitive errors in ComputeStep. The
cross-product error enters twice because the stored result is doubled. -/
theorem component_residuals {p full : MPIR.Precision} {c : State full} {old next : State p}
    {xs ys cross doubled temporary : Slot p}
    {a b g d e f j : ℝ}
    (hx : |xs.val.value - old.x.val.value ^ 2| ≤ a)
    (hy : |ys.val.value - old.y.val.value ^ 2| ≤ b)
    (hxy : |cross.val.value - old.x.val.value * old.y.val.value| ≤ g)
    (hdouble : |doubled.val.value - 2 * cross.val.value| ≤ d)
    (haddY : |next.y.val.value - (doubled.val.value + c.y.val.value)| ≤ e)
    (hsub : |temporary.val.value - (xs.val.value - ys.val.value)| ≤ f)
    (haddX : |next.x.val.value - (temporary.val.value + c.x.val.value)| ≤ j) :
    let residual := next.value - (old.value ^ 2 + c.value)
    |residual.re| ≤ a + b + f + j ∧ |residual.im| ≤ 2 * g + d + e := by
  rw [← exactStep_eq]
  simp only [exactStep, Complex.sub_re, Complex.sub_im, State.value]
  have hrSquare := abs_sub (xs.val.value - old.x.val.value ^ 2)
    (ys.val.value - old.y.val.value ^ 2)
  have hrSub := abs_sub_le temporary.val.value (xs.val.value - ys.val.value)
    (old.x.val.value ^ 2 - old.y.val.value ^ 2)
  have hrAdd := abs_sub_le next.x.val.value (temporary.val.value + c.x.val.value)
    (old.x.val.value ^ 2 - old.y.val.value ^ 2 + c.x.val.value)
  rw [add_sub_add_right_eq_sub] at hrAdd
  have hid : (xs.val.value - ys.val.value) - (old.x.val.value ^ 2 - old.y.val.value ^ 2) =
      (xs.val.value - old.x.val.value ^ 2) - (ys.val.value - old.y.val.value ^ 2) := by ring
  rw [hid] at hrSub
  have hiDouble := abs_sub_le doubled.val.value (2 * cross.val.value)
    (2 * (old.x.val.value * old.y.val.value))
  rw [← mul_sub, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at hiDouble
  have hiAdd := abs_sub_le next.y.val.value (doubled.val.value + c.y.val.value)
    (2 * (old.x.val.value * old.y.val.value) + c.y.val.value)
  rw [add_sub_add_right_eq_sub] at hiAdd
  exact ⟨by linarith only [hx, hy, hsub, haddX, hrSquare, hrSub, hrAdd],
    by linarith only [hxy, hdouble, haddY, hiDouble, hiAdd]⟩

/-- The rounded local disk can be formed from the very same limb data that
ComputeStep reads before its destinations are overwritten. -/
theorem dynamic_local_error {p full : MPIR.Precision} {c : State full} {old next : State p}
    (h : Step c old next) :
    ∃ (xs ys cross doubled temporary : Slot p),
      let a := productAllowance p xs.val + productAllowance p ys.val +
        MPIR.sumAllowance p xs.val ys.val.neg + MPIR.sumAllowance p temporary.val c.x.val
      let b := 2 * productAllowance p cross.val + doublingAllowance p cross.val +
        MPIR.sumAllowance p doubled.val c.y.val
      ‖next.value - (old.value ^ 2 + c.value)‖ ≤ Real.sqrt (a ^ 2 + b ^ 2) := by
  obtain ⟨xs, ys, cross, doubled, temp, hx, hy, hxy, hd, ha, hs, hb⟩ := h.run
  refine ⟨xs, ys, cross, doubled, temp, ?_⟩
  have hc := component_residuals (by simpa only [pow_two] using product_limb_error hx)
    (by simpa only [pow_two] using product_limb_error hy) (product_limb_error hxy)
    (doubling_bound hd) (MPIR.add_limb_error ha) (MPIR.sub_limb_error hs) (MPIR.add_limb_error hb)
  exact norm_components _ hc.1 hc.2

theorem enlarge_for_clip {before clipped fixed : ℂ} {E clip : ℝ}
    (hE : ‖before - fixed‖ ≤ E) (hclip : ‖clipped - before‖ ≤ clip) :
    ‖clipped - fixed‖ ≤ E + clip := by
  rw [show clipped - fixed = (before - fixed) + (clipped - before) by ring]
  exact (norm_add_le _ _).trans (add_le_add hE hclip)

/-- The quadratic disk works for arbitrary allowed precisions. The clipped
dynamic state is the center used in both the linear and quadratic terms. -/
theorem propagate_disk {d f dNext fNext c : ℂ} {R E τd τf out : ℝ}
    (hR : ‖d‖ ≤ R) (hE : ‖d - f‖ ≤ E)
    (hd : ‖dNext - (d ^ 2 + c)‖ ≤ τd) (hf : ‖fNext - (f ^ 2 + c)‖ ≤ τf)
    (hout : 2 * R * E + E ^ 2 + τd + τf ≤ out) :
    ‖dNext - fNext‖ ≤ out := by
  let e := d - f
  have he : ‖e‖ ≤ E := hE
  have hn : 0 ≤ E := (norm_nonneg _).trans he
  have hR0 : 0 ≤ R := (norm_nonneg _).trans hR
  have hsq : ‖2 * d * e - e ^ 2‖ ≤ 2 * R * E + E ^ 2 := by
    refine (norm_sub_le _ _).trans (add_le_add ?_ ?_)
    · rw [Complex.norm_mul, Complex.norm_mul, Complex.norm_two]
      exact mul_le_mul (mul_le_mul_of_nonneg_left hR (by norm_num)) he (norm_nonneg _)
        (mul_nonneg (by norm_num) hR0)
    · rw [Complex.norm_pow]
      exact pow_le_pow_left₀ (norm_nonneg _) he _
  rw [show dNext - fNext = (2 * d * e - e ^ 2) +
    ((dNext - (d ^ 2 + c)) - (fNext - (f ^ 2 + c))) by dsimp [e]; ring]
  have h := (norm_add_le _ _).trans (add_le_add hsq ((norm_sub_le _ _).trans (add_le_add hd hf)))
  linarith only [h, hout]

/-- Successful finite operations satisfy outward inequalities. The sum
of the two transport terms and the two local allowances can then be used
as a disk radius. Invalid arithmetic has no finite upper-bound witness. -/
theorem outward_disk_update {R E τd τf linear quadratic localError transport result : ℝ}
    (hl : 2 * R * E ≤ linear) (hq : E ^ 2 ≤ quadratic)
    (hc : τd + τf ≤ localError) (ht : linear + quadratic ≤ transport)
    (hout : transport + localError ≤ result) :
    2 * R * E + E ^ 2 + τd + τf ≤ result := by linarith

theorem component_gap_from_disk (dynamic fixed : ℂ) (E : ℝ)
    (h : ‖dynamic - fixed‖ ≤ E) :
    |dynamic.re - fixed.re| ≤ E ∧ |dynamic.im - fixed.im| ≤ E :=
  ⟨(Complex.abs_re_le_norm _).trans h, (Complex.abs_im_le_norm _).trans h⟩

/-- Induction over the disk checked by the dynamic scheduler. A step whose
operands, slots and operations share the fixed computation keeps a zero disk.
The other case includes clipping and uses the two independent residuals. -/
inductive DiskStep (d f dNext fNext c : ℂ) (E out : ℝ) : Prop where
  | common (h : dNext = fNext) (hout : 0 ≤ out)
  | rounded (clipped : ℂ) (clip R τd τf : ℝ)
      (hclip : ‖clipped - d‖ ≤ clip) (hR : ‖clipped‖ ≤ R)
      (hd : ‖dNext - (clipped ^ 2 + c)‖ ≤ τd) (hf : ‖fNext - (f ^ 2 + c)‖ ≤ τf)
      (hout : 2 * R * (E + clip) + (E + clip) ^ 2 + τd + τf ≤ out)

theorem disk_prefix (dynamic fixed : ℕ → ℂ) (c : ℂ) (bound : ℕ → ℝ) (last : ℕ)
    (hfirst : dynamic 1 = fixed 1) (hbase : 0 ≤ bound 1)
    (hstep : ∀ i, 1 ≤ i → i < last →
      DiskStep (dynamic i) (fixed i) (dynamic (i + 1)) (fixed (i + 1)) c (bound i) (bound (i + 1))) :
    ∀ n, 1 ≤ n → n ≤ last → ‖dynamic n - fixed n‖ ≤ bound n := by
  intro n
  induction n with
  | zero => omega
  | succ n ih =>
    intro hn hl
    cases n with
    | zero => simpa [hfirst] using hbase
    | succ n =>
      cases hstep (n + 1) (by omega) (by omega) with
      | common h hz => simpa [h] using hz
      | rounded _ _ _ _ _ hclip hR hd hf hout =>
        exact propagate_disk hR (enlarge_for_clip (ih (by omega) (by omega)) hclip) hd hf hout

theorem dynamic_viewport_bound (v : Viewport) (dynamic fixed target : ℂ)
    (E fixedError fraction : ℝ) (hd : ‖dynamic - fixed‖ ≤ E)
    (hf : norm1 (fixed - target) ≤ fixedError)
    (hbudget : fixedError + 2 * E ≤ v.sampleBudget fraction) :
    v.sampleError (dynamic - target) ≤ fraction := by
  apply v.sampleError_le
  rw [show dynamic - target = (dynamic - fixed) + (fixed - target) by ring]
  have hre := (Complex.abs_re_le_norm (dynamic - fixed)).trans hd
  have him := (Complex.abs_im_le_norm (dynamic - fixed)).trans hd
  have h := norm1_add_le (dynamic - fixed) (fixed - target)
  unfold norm1 at h
  unfold norm1
  unfold norm1 at hf
  linarith only [h, hre, him, hf, hbudget]

end FractalSharkProofs.ReferenceOrbit.MT9
