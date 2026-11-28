module

public import FractalSharkProofs.ReferenceOrbit.Error
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Exact affine sample geometry and unre-based perturbation on an internal orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

/-- Output dimensions and the number of calculation samples per pixel axis. -/
structure Viewport where
  xMin : ℝ
  xMax : ℝ
  yMin : ℝ
  yMax : ℝ
  width : ℕ
  height : ℕ
  samplesPerAxis : ℕ
  x_pos : xMin < xMax
  y_pos : yMin < yMax
  width_pos : 0 < width
  height_pos : 0 < height
  samples_pos : 0 < samplesPerAxis

namespace Viewport

noncomputable def spanX (v : Viewport) : ℝ := v.xMax - v.xMin
noncomputable def spanY (v : Viewport) : ℝ := v.yMax - v.yMin
def columns (v : Viewport) : ℕ := v.samplesPerAxis * v.width
def rows (v : Viewport) : ℕ := v.samplesPerAxis * v.height

theorem columns_pos (v : Viewport) : 0 < v.columns := Nat.mul_pos v.samples_pos v.width_pos
theorem rows_pos (v : Viewport) : 0 < v.rows := Nat.mul_pos v.samples_pos v.height_pos

noncomputable def pitchX (v : Viewport) : ℝ := v.spanX / v.columns
noncomputable def pitchY (v : Viewport) : ℝ := v.spanY / v.rows

theorem pitchX_pos (v : Viewport) : 0 < v.pitchX :=
  div_pos (sub_pos.mpr v.x_pos) (by exact_mod_cast v.columns_pos)

theorem pitchY_pos (v : Viewport) : 0 < v.pitchY :=
  div_pos (sub_pos.mpr v.y_pos) (by exact_mod_cast v.rows_pos)

noncomputable def verticalRadius (v : Viewport) : ℝ := v.spanY / 2

/-- Exact parameter for a calculation-grid sample in lower-left coordinates. -/
noncomputable def sample (v : Viewport) (x y : ℕ) : ℂ :=
  ⟨v.xMin + x * v.pitchX, v.yMin + y * v.pitchY⟩

def Contains (v : Viewport) (c : ℂ) : Prop :=
  v.xMin ≤ c.re ∧ c.re ≤ v.xMax ∧ v.yMin ≤ c.im ∧ c.im ≤ v.yMax

theorem sample_contains (v : Viewport) {x y : ℕ} (hx : x < v.columns) (hy : y < v.rows) :
    v.Contains (v.sample x y) := by
  have hcx : (0 : ℝ) < v.columns := by exact_mod_cast v.columns_pos
  have hcy : (0 : ℝ) < v.rows := by exact_mod_cast v.rows_pos
  have hspanX : (v.columns : ℝ) * v.pitchX = v.spanX := by
    simp [pitchX, mul_div_cancel₀, hcx.ne']
  have hspanY : (v.rows : ℝ) * v.pitchY = v.spanY := by
    simp [pitchY, mul_div_cancel₀, hcy.ne']
  have hdx := mul_le_mul_of_nonneg_right
    (show (x : ℝ) ≤ v.columns by exact_mod_cast hx.le) v.pitchX_pos.le
  have hdy := mul_le_mul_of_nonneg_right
    (show (y : ℝ) ≤ v.rows by exact_mod_cast hy.le) v.pitchY_pos.le
  change v.xMin ≤ v.xMin + x * v.pitchX ∧ v.xMin + x * v.pitchX ≤ v.xMax ∧
    v.yMin ≤ v.yMin + y * v.pitchY ∧ v.yMin + y * v.pitchY ≤ v.yMax
  refine ⟨le_add_of_nonneg_right (mul_nonneg (by positivity) v.pitchX_pos.le), ?_,
    le_add_of_nonneg_right (mul_nonneg (by positivity) v.pitchY_pos.le), ?_⟩
  · rw [hspanX] at hdx
    linarith only [hdx, show v.spanX = v.xMax - v.xMin from rfl]
  · rw [hspanY] at hdy
    linarith only [hdy, show v.spanY = v.yMax - v.yMin from rfl]

/-- Componentwise parameter-distance bound from the actual reference anchor. -/
noncomputable def coverage (v : Viewport) (anchor : ℂ) : ℝ :=
  max |v.xMin - anchor.re| |v.xMax - anchor.re| +
    max |v.yMin - anchor.im| |v.yMax - anchor.im|

private theorem abs_sub_le_ends {lo hi x a : ℝ} (hlo : lo ≤ x) (hhi : x ≤ hi) :
    |x - a| ≤ max |lo - a| |hi - a| := by
  have hl := (neg_abs_le (lo - a))
  have hh := le_abs_self (hi - a)
  have hmaxl : |lo - a| ≤ max |lo - a| |hi - a| := le_max_left _ _
  have hmaxr : |hi - a| ≤ max |lo - a| |hi - a| := le_max_right _ _
  exact abs_le.mpr ⟨by linarith only [hl, hmaxl, hlo], by linarith only [hh, hmaxr, hhi]⟩

theorem offset_le_coverage (v : Viewport) (anchor : ℂ) {c : ℂ} (hc : v.Contains c) :
    norm1 (c - anchor) ≤ v.coverage anchor :=
  add_le_add (abs_sub_le_ends hc.1 hc.2.1) (abs_sub_le_ends hc.2.2.1 hc.2.2.2)

theorem coverage_nonneg (v : Viewport) (anchor : ℂ) : 0 ≤ v.coverage anchor :=
  add_nonneg ((abs_nonneg _).trans (le_max_left _ _))
    ((abs_nonneg _).trans (le_max_left _ _))

/-- The budget uses the smaller pitch, so it also controls the sum of the
errors measured in the two potentially different sample spacings. -/
noncomputable def sampleBudget (v : Viewport) (fraction : ℝ) : ℝ :=
  fraction * min v.pitchX v.pitchY

noncomputable def sampleError (v : Viewport) (error : ℂ) : ℝ :=
  |error.re| / v.pitchX + |error.im| / v.pitchY

theorem sampleError_le (v : Viewport) {fraction : ℝ} {error : ℂ}
    (h : norm1 error ≤ v.sampleBudget fraction) : v.sampleError error ≤ fraction := by
  let pitch := min v.pitchX v.pitchY
  have hp : 0 < pitch := lt_min v.pitchX_pos v.pitchY_pos
  unfold sampleError
  calc
    _ ≤ |error.re| / pitch + |error.im| / pitch :=
      add_le_add (div_le_div_of_nonneg_left (abs_nonneg _) hp (min_le_left _ _))
        (div_le_div_of_nonneg_left (abs_nonneg _) hp (min_le_right _ _))
    _ ≤ fraction := by
      rw [← add_div]
      exact (div_le_iff₀ hp).2 h

theorem error_le_budget_iff_radius (v : Viewport) {fraction error : ℝ}
    (hf : 0 < fraction) (hsquare : v.pitchX = v.pitchY) :
    error ≤ v.sampleBudget fraction ↔ (v.rows : ℝ) * error / (2 * fraction) ≤ v.verticalRadius := by
  have hrows : (0 : ℝ) < v.rows := by exact_mod_cast v.rows_pos
  have hdenom : 0 < 2 * fraction := by positivity
  rw [sampleBudget, hsquare, min_self, pitchY, ← mul_div_assoc, le_div_iff₀ hrows,
    verticalRadius, div_le_iff₀ hdenom]
  have hcancel : v.spanY / 2 * (2 * fraction) = fraction * v.spanY := by ring
  rw [hcancel, mul_comm error]

theorem centered_coverage (v : Viewport) (anchor : ℂ)
    (hx : anchor.re = (v.xMin + v.xMax) / 2)
    (hy : anchor.im = (v.yMin + v.yMax) / 2) :
    v.coverage anchor = (v.spanX + v.spanY) / 2 := by
  have hxl : v.xMin - anchor.re = -(v.spanX / 2) := by
    rw [hx, spanX]
    ring
  have hxr : v.xMax - anchor.re = v.spanX / 2 := by
    rw [hx, spanX]
    ring
  have hyl : v.yMin - anchor.im = -(v.spanY / 2) := by
    rw [hy, spanY]
    ring
  have hyr : v.yMax - anchor.im = v.spanY / 2 := by
    rw [hy, spanY]
    ring
  have hxp : 0 ≤ v.spanX / 2 := div_nonneg (sub_nonneg.mpr v.x_pos.le) (by norm_num)
  have hyp : 0 ≤ v.spanY / 2 := div_nonneg (sub_nonneg.mpr v.y_pos.le) (by norm_num)
  rw [coverage, hxl, hxr, hyl, hyr, abs_neg, abs_neg, abs_of_nonneg hxp,
    abs_of_nonneg hyp, max_self, max_self]
  ring

theorem centered_square_pixel_coverage (v : Viewport) (anchor : ℂ)
    (hx : anchor.re = (v.xMin + v.xMax) / 2)
    (hy : anchor.im = (v.yMin + v.yMax) / 2) (hsquare : v.pitchX = v.pitchY) :
    v.coverage anchor = (1 + (v.columns : ℝ) / v.rows) * v.verticalRadius := by
  have hcx : (v.columns : ℝ) ≠ 0 := by exact_mod_cast v.columns_pos.ne'
  have hcy : (v.rows : ℝ) ≠ 0 := by exact_mod_cast v.rows_pos.ne'
  rw [centered_coverage v anchor hx hy, verticalRadius]
  have hspan := (div_eq_div_iff hcx hcy).mp hsquare
  field_simp [hcy]
  linarith only [hspan]

end Viewport

/-- Perturbation uses the internal reference at each mathematical index and
the exact parameter offset from the supplied anchor. -/
def idealDelta (reference : ℕ → ℂ) (offset : ℂ) : ℕ → ℂ
  | 0 => 0
  | n + 1 => (2 * reference n + idealDelta reference offset n) *
      idealDelta reference offset n + offset

def idealReconstruction (reference : ℕ → ℂ) (anchor c : ℂ) (n : ℕ) : ℂ :=
  reference n + idealDelta reference (c - anchor) n

/-- The reconstructed orbit inherits the reference arithmetic residual.
Its effective parameter includes the discrepancy of the MPIR anchor copy. -/
theorem idealReconstruction_residual (reference : ℕ → ℂ) (anchor working c : ℂ) (n : ℕ) :
    idealReconstruction reference anchor c (n + 1) -
        ((idealReconstruction reference anchor c n) ^ 2 + (working + (c - anchor))) =
      reference (n + 1) - ((reference n) ^ 2 + working) := by
  simp only [idealReconstruction, idealDelta]
  ring

/-- Only the inputs of transitions require a displacement cap. The zero
entry has zero displacement, and the first displacement is the offset. -/
noncomputable def displacementBound (referenceCap cap offsetCap : ℝ) (last : ℕ) : ℝ :=
  affineBound (2 * referenceCap + cap) offsetCap offsetCap (last - 1)

theorem displacementBound_eq (referenceCap cap offsetCap : ℝ) (last : ℕ) :
    displacementBound referenceCap cap offsetCap last =
      offsetCap * ∑ k ∈ Finset.range (last - 1), (2 * referenceCap + cap) ^ k := by
  unfold displacementBound
  generalize last - 1 = n
  cases n with
  | zero => simp [affineBound]
  | succ n =>
    rw [affineBound_geometric, Finset.sum_range_succ]
    ring

/-- The geometry check places an upper limit on the centered view's radius
when there is at least one nonzero predecessor displacement to bound. -/
theorem Viewport.displacement_le_iff_radius (v : Viewport) (anchor : ℂ)
    (referenceCap cap : ℝ) (last : ℕ)
    (hx : anchor.re = (v.xMin + v.xMax) / 2)
    (hy : anchor.im = (v.yMin + v.yMax) / 2) (hsquare : v.pitchX = v.pitchY)
    (hsum : 0 < ∑ k ∈ Finset.range (last - 1), (2 * referenceCap + cap) ^ k) :
    displacementBound referenceCap cap (v.coverage anchor) last ≤ cap ↔
      v.verticalRadius ≤ cap /
        ((1 + (v.columns : ℝ) / v.rows) *
          ∑ k ∈ Finset.range (last - 1), (2 * referenceCap + cap) ^ k) := by
  have hrows : (0 : ℝ) < v.rows := by exact_mod_cast v.rows_pos
  have hfactor : 0 < 1 + (v.columns : ℝ) / v.rows := by positivity
  rw [displacementBound_eq, v.centered_square_pixel_coverage anchor hx hy hsquare,
    le_div_iff₀ (mul_pos hfactor hsum)]
  simp only [mul_comm, mul_assoc]

theorem idealDelta_le_of_cap (reference : ℕ → ℂ) (offset : ℂ) (last : ℕ)
    (referenceCap cap offsetCap : ℝ) (hr : 0 ≤ referenceCap) (hc : 0 ≤ cap)
    (hoffset : norm1 offset ≤ offsetCap)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (reference n) ≤ referenceCap)
    (hfinal : displacementBound referenceCap cap offsetCap last ≤ cap) :
    ∀ n, n < last → norm1 (idealDelta reference offset n) ≤ cap := by
  let factor := 2 * referenceCap + cap
  let bound := affineBound factor offsetCap offsetCap
  have hm : Monotone bound :=
    affineBound_monotone (by linarith only [hr, hc]) ((norm1_nonneg _).trans hoffset) le_rfl
  suffices hprefix : ∀ n, n < last → norm1 (idealDelta reference offset n) ≤ bound n by
    intro n hn
    exact (hprefix n hn).trans ((hm (show n ≤ last - 1 by omega)).trans hfinal)
  intro n
  induction n with
  | zero => simp [idealDelta, bound, affineBound]
  | succ n ih =>
    intro hn
    cases n with
    | zero => simpa [idealDelta, bound, affineBound] using hoffset
    | succ n =>
      let d := idealDelta reference offset (n + 1)
      have hp : norm1 d ≤ bound (n + 1) := ih (by omega)
      have hwithin := (hm (show n + 1 ≤ last - 1 by omega)).trans hfinal
      have hdouble : norm1 (2 * reference (n + 1)) ≤ 2 * norm1 (reference (n + 1)) := by
        simpa [two_mul] using norm1_add_le (reference (n + 1)) (reference (n + 1))
      have hamp : norm1 (2 * reference (n + 1) + d) ≤ factor := by
        have hsum := norm1_add_le (2 * reference (n + 1)) d
        have hs := hsize (n + 1) (by omega) (by omega)
        dsimp only [factor]
        linarith only [hsum, hdouble, hs, hp, hwithin]
      have hfactor : 0 ≤ factor := by linarith only [hr, hc]
      change norm1 ((2 * reference (n + 1) + d) * d + offset) ≤
        factor * bound (n + 1) + offsetCap
      exact (norm1_add_le _ _).trans (add_le_add
        ((norm1_mul_le _ _).trans (mul_le_mul hamp hp (norm1_nonneg _) hfactor)) hoffset)

end FractalSharkProofs.ReferenceOrbit
