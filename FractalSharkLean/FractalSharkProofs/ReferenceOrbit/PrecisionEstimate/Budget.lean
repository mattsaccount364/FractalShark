module

public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Enclosure
public import FractalSharkProofs.ReferenceOrbit.ViewportPrecision
public import FractalSharkProofs.ReferenceOrbit.MT9.Precision
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Per-iterate error weights for an MPIR precision chosen before generation. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

structure Sampling where
  xMin : ℚ
  xMax : ℚ
  yMin : ℚ
  yMax : ℚ
  width : ℕ
  height : ℕ
  antialiasing : ℕ
  x_pos : xMin < xMax
  y_pos : yMin < yMax
  width_pos : 0 < width
  height_pos : 0 < height
  aa_pos : 0 < antialiasing

/-- Enter Location starts with a square of half-height `2 / zoom` and expands
the longer screen axis. Use the output dimensions before supersampling. -/
def Sampling.ofZoom (center : Parameter) (zoom : ℚ) (width height aa : ℕ)
    (hZoom : 0 < zoom) (hWidth : 0 < width) (hHeight : 0 < height) (hAA : 0 < aa) : Sampling :=
  let scale : ℚ := 2 / (zoom * (min width height : ℕ))
  let halfWidth : ℚ := scale * width
  let halfHeight : ℚ := scale * height
  have hMin : (0 : ℚ) < (min width height : ℕ) := by
    exact_mod_cast (lt_min hWidth hHeight)
  have hScale : 0 < scale := div_pos (by decide) (mul_pos hZoom hMin)
  have hx : 0 < halfWidth := mul_pos hScale (by exact_mod_cast hWidth)
  have hy : 0 < halfHeight := mul_pos hScale (by exact_mod_cast hHeight)
  ⟨center.x - halfWidth, center.x + halfWidth, center.y - halfHeight, center.y + halfHeight,
    width, height, aa, (sub_lt_self _ hx).trans (lt_add_of_pos_right _ hx),
    (sub_lt_self _ hy).trans (lt_add_of_pos_right _ hy), hWidth, hHeight, hAA⟩

noncomputable def Sampling.viewport (s : Sampling) : Viewport :=
  ⟨s.xMin, s.xMax, s.yMin, s.yMax, s.width, s.height, s.antialiasing,
    by exact_mod_cast s.x_pos, by exact_mod_cast s.y_pos, s.width_pos, s.height_pos, s.aa_pos⟩

def Sampling.tolerance (s : Sampling) (fraction : ℚ) : ℚ :=
  fraction * min ((s.xMax - s.xMin) / (s.antialiasing * s.width : ℕ))
    ((s.yMax - s.yMin) / (s.antialiasing * s.height : ℕ))

theorem Sampling.tolerance_eq (s : Sampling) (fraction : ℚ) :
    (s.tolerance fraction : ℝ) = s.viewport.sampleBudget fraction := by
  simp [Sampling.tolerance, Sampling.viewport, Viewport.sampleBudget, Viewport.pitchX,
    Viewport.pitchY, Viewport.spanX, Viewport.spanY, Viewport.columns, Viewport.rows]

theorem Sampling.tolerance_pos (s : Sampling) {fraction : ℚ} (h : 0 < fraction) :
    0 < s.tolerance fraction := by
  have hp : (0 : ℝ) < s.viewport.sampleBudget fraction :=
    mul_pos (show (0 : ℝ) < fraction by exact_mod_cast h)
      (lt_min s.viewport.pitchX_pos s.viewport.pitchY_pos)
  rw [← s.tolerance_eq] at hp
  exact_mod_cast hp

def Sampling.offset (s : Sampling) (c : Parameter) : ℚ :=
  max |s.xMin - c.x| |s.xMax - c.x| + max |s.yMin - c.y| |s.yMax - c.y|

theorem Sampling.offset_eq (s : Sampling) (c : Parameter) :
    (s.offset c : ℝ) = s.viewport.coverage c.value := by
  simp [Sampling.offset, Sampling.viewport, Viewport.coverage, Parameter.value]

inductive CPU where
  | st
  | mt3
  | mt9
  deriving DecidableEq

def CPU.coefficient : CPU → ℚ
  | .st => 32
  | .mt3 => 9
  | .mt9 => MT9.fixedCoefficient

theorem CPU.coefficient_nonneg (cpu : CPU) : 0 ≤ cpu.coefficient := by
  cases cpu <;> decide

structure Request where
  center : Parameter
  sampling : Sampling
  fraction : ℚ
  fraction_pos : 0 < fraction
  backend : CPU
  last : ℕ
  bailoutRadius : ℚ
  bailout_pos : 0 < bailoutRadius

def Request.tolerance (r : Request) : ℚ := r.sampling.tolerance r.fraction

theorem Request.tolerance_pos (r : Request) : 0 < r.tolerance :=
  r.sampling.tolerance_pos r.fraction_pos

def capAt (caps : Array ℚ) (n : ℕ) : ℚ := caps[n]?.getD 0

structure Weight where
  current : ℚ
  peak : ℚ

/-- Round each coefficient up to an integer so rational denominators do not
accumulate across a long prefix. The peak covers all earlier indices. -/
def weights (first : ℚ) (factor cost : ℕ → ℚ) : ℕ → Weight
  | 0 => ⟨0, 0⟩
  | 1 => ⟨first, first⟩
  | n + 2 =>
    let prev := weights first factor cost (n + 1)
    let current := up 0 (factor (n + 1) * prev.current + cost (n + 1))
    ⟨current, max prev.peak current⟩

private theorem current_le_peak (first : ℚ) (factor cost : ℕ → ℚ) (n : ℕ) :
    (weights first factor cost n).current ≤ (weights first factor cost n).peak := by
  cases n with
  | zero => exact le_rfl
  | succ n => cases n with
    | zero => exact le_rfl
    | succ n => exact le_max_right _ _

private theorem peak_mono {first : ℚ} (h : 0 ≤ first) (factor cost : ℕ → ℚ) :
    Monotone (fun n => (weights first factor cost n).peak) := by
  apply monotone_nat_of_le_succ
  intro n
  cases n with
  | zero => exact h
  | succ n => exact le_max_left _ _

def stepCost (r : Request) (cap ceiling : ℚ) : ℚ :=
  r.backend.coefficient * (cap + ceiling) ^ 2 + 2 * r.center.magnitude

def referenceWeights (r : Request) (caps : Array ℚ) (ceiling : ℚ) (n : ℕ) : Weight :=
  weights r.center.magnitude (fun i => 2 * capAt caps i + ceiling)
    (fun i => r.center.magnitude + stepCost r (capAt caps i) ceiling) n

theorem referenceWeights_current_le_peak (r : Request) (caps : Array ℚ) (ceiling : ℚ)
    {i last : ℕ} (hi : i ≤ last) :
    (referenceWeights r caps ceiling i).current ≤ (referenceWeights r caps ceiling last).peak := by
  unfold referenceWeights
  have hm : 0 ≤ r.center.magnitude := by unfold Parameter.magnitude; positivity
  exact (current_le_peak _ _ _ i).trans (peak_mono hm _ _ hi)

def pixelStep (r : Request) (caps : Array ℚ) (bits n : ℕ) (previous : ℚ × ℚ) : ℚ × ℚ :=
  let combined := capAt caps n + previous.2
  let displacement := up bits (combined * min previous.1 combined + r.sampling.offset r.center)
  (displacement, min (capAt caps (n + 1) + displacement) (2 * r.bailoutRadius))

/-- A pixel that has not crossed the exact bailout radius has componentwise
norm at most twice that radius. It also has the reference-relative cap. -/
def pixelCaps (r : Request) (caps : Array ℚ) (bits : ℕ) : ℕ → ℚ × ℚ
  | 0 => (0, 0)
  | n + 1 => pixelStep r caps bits n (pixelCaps r caps bits n)

/-- Advance the pixel cap and error weight together so each index's cap is
computed once when producing the full report. -/
def idealChain (r : Request) (caps : Array ℚ) (bits : ℕ) : ℕ → (ℚ × ℚ) × Weight
  | 0 => ((0, 0), ⟨0, 0⟩)
  | 1 => (pixelStep r caps bits 0 (0, 0), ⟨0, 0⟩)
  | n + 2 =>
    let prev := idealChain r caps bits (n + 1)
    let current := up 0 ((2 * prev.1.2 + r.tolerance) * prev.2.current +
      stepCost r (capAt caps (n + 1)) 1)
    (pixelStep r caps bits (n + 1) prev.1, ⟨current, max prev.2.peak current⟩)

def idealWeights (r : Request) (caps : Array ℚ) (bits n : ℕ) : Weight :=
  (idealChain r caps bits n).2

private theorem idealChain_eq (r : Request) (caps : Array ℚ) (bits n : ℕ) :
    idealChain r caps bits n = (pixelCaps r caps bits n,
      weights 0 (fun i => 2 * (pixelCaps r caps bits i).2 + r.tolerance)
        (fun i => stepCost r (capAt caps i) 1) n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    cases n with
    | zero => rfl
    | succ n => simp only [idealChain, ih, weights, pixelCaps]

private theorem idealWeights_eq (r : Request) (caps : Array ℚ) (bits n : ℕ) :
    idealWeights r caps bits n =
      weights 0 (fun i => 2 * (pixelCaps r caps bits i).2 + r.tolerance)
        (fun i => stepCost r (capAt caps i) 1) n := by
  rw [idealWeights, idealChain_eq]

private theorem norm1_double_le (z : ℂ) : norm1 (2 * z) ≤ 2 * norm1 z := by
  simpa only [two_mul] using norm1_add_le z z

private theorem quadratic_error (next w z c : ℂ) (cap tolerance : ℝ)
    (hz : norm1 z ≤ cap) (herr : norm1 (w - z) ≤ tolerance) :
    norm1 (next - (z ^ 2 + c)) ≤
      (2 * cap + tolerance) * norm1 (w - z) + norm1 (next - (w ^ 2 + c)) := by
  have hsum : norm1 (w + z) ≤ 2 * cap + tolerance := by
    rw [show w + z = 2 * z + (w - z) by ring]
    linarith only [norm1_add_le (2 * z) (w - z), norm1_double_le z, hz, herr]
  have hproduct := (norm1_mul_le (w + z) (w - z)).trans
    (mul_le_mul_of_nonneg_right hsum (norm1_nonneg _))
  rw [show next - (z ^ 2 + c) = (w + z) * (w - z) + (next - (w ^ 2 + c)) by ring]
  exact (norm1_add_le _ _).trans (add_le_add hproduct le_rfl)

private theorem error_of_peak (target : ℂ) (w : ℕ → ℂ) (last : ℕ) (u : ℝ)
    (ceiling first : ℚ) (size cost : ℕ → ℚ) (hu : 0 ≤ u) (ht : 0 ≤ ceiling)
    (hfirst : 0 ≤ first) (hzero : w 0 = 0)
    (hinit : norm1 (w 1 - orbit target 1) ≤ u * first)
    (hsize : ∀ i, 1 ≤ i → i < last → norm1 (orbit target i) ≤ size i)
    (hstep : ∀ i, 1 ≤ i → i < last → norm1 (w i - orbit target i) ≤ ceiling →
      norm1 (w (i + 1) - ((w i) ^ 2 + target)) ≤ u * cost i)
    (hpeak : u * (weights first (fun i => 2 * size i + ceiling) cost last).peak ≤ ceiling) :
    ∀ i, i ≤ last →
      norm1 (w i - orbit target i) ≤
        u * (weights first (fun j => 2 * size j + ceiling) cost i).current ∧
      norm1 (w i - orbit target i) ≤ ceiling := by
  let b := weights first (fun i => 2 * size i + ceiling) cost
  have hbound : ∀ i, i ≤ last → u * (b i).current ≤ ceiling := by
    intro i hi
    have h : (b i).current ≤ (b last).peak :=
      (current_le_peak first (fun j => 2 * size j + ceiling) cost i).trans
        (peak_mono hfirst _ cost hi)
    exact (mul_le_mul_of_nonneg_left (by exact_mod_cast h) hu).trans hpeak
  suffices h : ∀ i, i ≤ last → norm1 (w i - orbit target i) ≤ u * (b i).current by
    exact fun i hi => ⟨h i hi, (h i hi).trans (hbound i hi)⟩
  intro i
  induction i with
  | zero => simp [hzero, orbit, b, weights]
  | succ i ih =>
    intro hi
    cases i with
    | zero => exact hinit
    | succ i =>
      have hp := ih (by omega)
      have he := hp.trans (hbound (i + 1) (by omega))
      have hs := hsize (i + 1) (by omega) (by omega)
      have hcap := (norm1_nonneg _).trans hs
      have ha : 0 ≤ 2 * (size (i + 1) : ℝ) + ceiling :=
        add_nonneg (mul_nonneg (by norm_num) hcap) (by exact_mod_cast ht)
      have hc := hstep (i + 1) (by omega) (by omega) he
      have hround := le_up 0 ((2 * size (i + 1) + ceiling) * (b (i + 1)).current + cost (i + 1))
      simp only [orbit]
      calc
        _ ≤ (2 * (size (i + 1) : ℝ) + ceiling) * norm1 (w (i + 1) - orbit target (i + 1)) +
            norm1 (w (i + 2) - (w (i + 1) ^ 2 + target)) :=
          quadratic_error _ _ _ _ _ _ hs he
        _ ≤ u * ((2 * (size (i + 1) : ℝ) + ceiling) * (b (i + 1)).current + cost (i + 1)) := by
          have hh := add_le_add (mul_le_mul_of_nonneg_left hp ha) hc
          convert hh using 1
          ring
        _ ≤ u * (b (i + 2)).current := by
          apply mul_le_mul_of_nonneg_left _ hu
          exact_mod_cast hround

/-- Exact complex bailout implies the simpler componentwise cap. -/
theorem norm_le_of_inside_radius (z : ℂ) {radius : ℝ} (hr : 0 ≤ radius)
    (hz : z.re ^ 2 + z.im ^ 2 ≤ radius ^ 2) : norm1 z ≤ 2 * radius := by
  have component {x : ℝ} (h : x ^ 2 ≤ radius ^ 2) : |x| ≤ radius := by
    by_contra! h'
    have hsq := pow_lt_pow_left₀ h' hr (by decide : 2 ≠ 0)
    rw [sq_abs] at hsq
    linarith
  unfold norm1
  linarith only [component (show z.re ^ 2 ≤ radius ^ 2 by linarith [sq_nonneg z.im]),
    component (show z.im ^ 2 ≤ radius ^ 2 by linarith [sq_nonneg z.re])]

private theorem pixelCaps_sound (r : Request) (caps : Array ℚ) (bits last : ℕ) (c : ℂ)
    (hcap : ∀ i, i < last → norm1 (orbit r.center.value i) ≤ capAt caps i)
    (hc : r.sampling.viewport.Contains c)
    (hinside : ∀ i, i < last → (orbit c i).re ^ 2 + (orbit c i).im ^ 2 ≤
      (r.bailoutRadius : ℝ) ^ 2) :
    ∀ i, i < last →
      norm1 (orbit c i - orbit r.center.value i) ≤ (pixelCaps r caps bits i).1 ∧
      norm1 (orbit c i) ≤ (pixelCaps r caps bits i).2 := by
  intro i
  induction i with
  | zero => simp [orbit, pixelCaps]
  | succ i ih =>
    intro hi
    let x := orbit c i
    let z := orbit r.center.value i
    let d := (pixelCaps r caps bits i).1
    let v := (pixelCaps r caps bits i).2
    let D := (pixelCaps r caps bits (i + 1)).1
    have ⟨hd, hv⟩ := ih (by omega)
    have hR := hcap i (by omega)
    have hρ : norm1 (c - r.center.value) ≤ r.sampling.offset r.center := by
      rw [r.sampling.offset_eq]
      exact r.sampling.viewport.offset_le_coverage r.center.value hc
    have hsum : norm1 (x + z) ≤ (capAt caps i : ℝ) + v := by
      linarith only [norm1_add_le x z, show norm1 x ≤ v from hv, show norm1 z ≤ capAt caps i from hR]
    have hdiff : norm1 (x - z) ≤ (min d (capAt caps i + v) : ℚ) := by
      push_cast
      apply le_min hd
      have h := norm1_add_le x (-z)
      simp only [norm1, Complex.neg_re, Complex.neg_im, abs_neg] at h
      rw [← sub_eq_add_neg] at h
      change norm1 (x - z) ≤ norm1 x + norm1 z at h
      linarith only [h, show norm1 x ≤ v from hv, show norm1 z ≤ capAt caps i from hR]
    have hD : norm1 (orbit c (i + 1) - orbit r.center.value (i + 1)) ≤ D := by
      change norm1 ((x ^ 2 + c) - (z ^ 2 + r.center.value)) ≤ D
      rw [show (x ^ 2 + c) - (z ^ 2 + r.center.value) =
        (x + z) * (x - z) + (c - r.center.value) by ring]
      have hpos := (norm1_nonneg _).trans hsum
      calc
        _ ≤ ((capAt caps i : ℝ) + v) * (min d (capAt caps i + v) : ℚ) +
            r.sampling.offset r.center :=
          (norm1_add_le _ _).trans (add_le_add
            ((norm1_mul_le _ _).trans (mul_le_mul hsum hdiff (norm1_nonneg _) hpos)) hρ)
        _ ≤ D := by
          exact_mod_cast le_up bits ((capAt caps i + v) * min d (capAt caps i + v) +
            r.sampling.offset r.center)
    refine ⟨hD, ?_⟩
    change norm1 (orbit c (i + 1)) ≤ ((min (capAt caps (i + 1) + D) (2 * r.bailoutRadius) : ℚ) : ℝ)
    push_cast
    apply le_min
    · have h := (norm1_add_le (orbit r.center.value (i + 1))
        (orbit c (i + 1) - orbit r.center.value (i + 1))).trans (add_le_add (hcap _ hi) hD)
      rwa [add_sub_cancel] at h
    · exact norm_le_of_inside_radius _ (by exact_mod_cast r.bailout_pos.le) (hinside _ hi)

private theorem cost_bounds_step (r : Request) (caps : Array ℚ)
    {p : MPIR.Precision} {last : ℕ} (run : MPIRViewportPrefix p r.backend.coefficient last)
    (i : ℕ) (hi : 1 ≤ i) (hl : i < last) (ceiling : ℚ)
    (hm : norm1 run.working ≤ r.center.magnitude)
    (hcap : norm1 (orbit r.center.value i) ≤ capAt caps i)
    (herror : norm1 (run.state i - orbit r.center.value i) ≤ ceiling) :
    norm1 (run.state (i + 1) - (run.state i ^ 2 + run.working)) ≤
      p.errorUnit * stepCost r (capAt caps i) ceiling := by
  have hstate : norm1 (run.state i) ≤ (capAt caps i : ℝ) + ceiling := by
    have h := norm1_add_le (run.state i - orbit r.center.value i) (orbit r.center.value i)
    rw [sub_add_cancel] at h
    linarith only [h, hcap, herror]
  have hsq := pow_le_pow_left₀ (norm1_nonneg _) hstate 2
  have hk : (0 : ℝ) ≤ r.backend.coefficient := by exact_mod_cast r.backend.coefficient_nonneg
  have hmul := mul_le_mul_of_nonneg_left hsq hk
  have hmag := mul_le_mul_of_nonneg_left hm (by norm_num : (0 : ℝ) ≤ 2)
  refine (run.step_error i hi hl).trans (mul_le_mul_of_nonneg_left ?_ p.errorUnit_pos.le)
  simp only [stepCost]
  push_cast
  exact add_le_add hmul hmag

/-- The peak validates the induction ceiling, while the current weight gives
the error allowance at each individual index. -/
theorem reference_error_bounds (r : Request) (caps : Array ℚ) (ceiling : ℚ)
    {p : MPIR.Precision} {last : ℕ} (run : MPIRViewportPrefix p r.backend.coefficient last)
    (hlast : last ≤ r.last) (ht : 0 ≤ ceiling)
    (hcap : ∀ i, i < r.last → norm1 (orbit r.center.value i) ≤ capAt caps i)
    (hm : norm1 run.working ≤ r.center.magnitude)
    (hinput : norm1 (run.working - r.center.value) ≤ p.errorUnit * r.center.magnitude)
    (hpeak : p.errorUnit * (referenceWeights r caps ceiling r.last).peak ≤ ceiling) :
    ∀ i, i ≤ last →
      norm1 (run.state i - orbit r.center.value i) ≤
        p.errorUnit * (referenceWeights r caps ceiling i).current ∧
      norm1 (run.state i - orbit r.center.value i) ≤ ceiling := by
  have hm0 : 0 ≤ r.center.magnitude := by unfold Parameter.magnitude; positivity
  have hinit : norm1 (run.state 1 - orbit r.center.value 1) ≤ p.errorUnit * r.center.magnitude := by
    simpa [run.first, orbit] using hinput
  have hpeaks : (referenceWeights r caps ceiling last).peak ≤
      (referenceWeights r caps ceiling r.last).peak := peak_mono hm0 _ _ hlast
  apply error_of_peak r.center.value run.state last p.errorUnit ceiling r.center.magnitude
    (capAt caps) (fun i => r.center.magnitude + stepCost r (capAt caps i) ceiling)
    p.errorUnit_pos.le ht hm0 run.zero hinit (fun i _ hi => hcap i (hi.trans_le hlast)) ?_
    ((mul_le_mul_of_nonneg_left (by exact_mod_cast hpeaks) p.errorUnit_pos.le).trans hpeak)
  intro i hi hl herror
  have hlocal := cost_bounds_step r caps run i hi hl ceiling hm (hcap i (hl.trans_le hlast)) herror
  rw [show run.state (i + 1) - (run.state i ^ 2 + r.center.value) =
    (run.working - r.center.value) + (run.state (i + 1) - (run.state i ^ 2 + run.working)) by ring]
  have h := (norm1_add_le _ _).trans (add_le_add hinput hlocal)
  simpa only [Rat.cast_add, mul_add] using h

theorem reference_in_samples (r : Request) (caps : Array ℚ)
    {p : MPIR.Precision} {last : ℕ} (run : MPIRViewportPrefix p r.backend.coefficient last)
    (hlast : last ≤ r.last)
    (hcap : ∀ i, i < r.last → norm1 (orbit r.center.value i) ≤ capAt caps i)
    (hm : norm1 run.working ≤ r.center.magnitude)
    (hinput : norm1 (run.working - r.center.value) ≤ p.errorUnit * r.center.magnitude)
    (hpeak : p.errorUnit * (referenceWeights r caps r.tolerance r.last).peak ≤ r.tolerance) :
    ∀ i, i ≤ last → r.sampling.viewport.sampleError (run.state i - orbit r.center.value i) ≤
      r.fraction := by
  intro i hi
  apply r.sampling.viewport.sampleError_le
  rw [← r.sampling.tolerance_eq]
  exact (reference_error_bounds r caps _ run hlast r.tolerance_pos.le hcap hm hinput hpeak i hi).2

/-- Certify the ideal prefix through the first exact bailout. Its parameter
offset is measured from the same dyadic anchor that the reference uses. -/
theorem ideal_in_samples (r : Request) (caps : Array ℚ) (bits : ℕ)
    {p : MPIR.Precision} {last : ℕ} (run : MPIRViewportPrefix p r.backend.coefficient last)
    (hlast : last ≤ r.last)
    (hcap : ∀ i, i < r.last → norm1 (orbit r.center.value i) ≤ capAt caps i)
    (hm : norm1 run.working ≤ r.center.magnitude)
    (hinput : norm1 (run.working - r.center.value) ≤ p.errorUnit * r.center.magnitude)
    (hcopy : run.working = run.anchor)
    (hguard : p.errorUnit * (referenceWeights r caps 1 r.last).peak ≤ 1)
    (hpeak : p.errorUnit * (idealWeights r caps bits r.last).peak ≤ r.tolerance)
    (c : ℂ) (hc : r.sampling.viewport.Contains c) (n : ℕ) (hn : n ≤ last)
    (hinside : ∀ i, i < n → (orbit c i).re ^ 2 + (orbit c i).im ^ 2 ≤
      (r.bailoutRadius : ℝ) ^ 2) :
    r.sampling.viewport.sampleError (idealReconstruction run.state run.anchor c n - orbit c n) ≤
      r.fraction := by
  let z := idealReconstruction run.state run.anchor c
  have href := reference_error_bounds r caps 1 run hlast (by norm_num) hcap hm hinput
    (by exact_mod_cast hguard)
  have hsize := pixelCaps_sound r caps bits n c (fun i hi => hcap i (hi.trans_le (hn.trans hlast)))
    hc hinside
  have hzero : z 0 = 0 := by simp [z, idealReconstruction, idealDelta, run.zero]
  have hfirst : norm1 (z 1 - orbit c 1) ≤ p.errorUnit * ((0 : ℚ) : ℝ) := by
    simp [z, idealReconstruction, idealDelta, orbit, run.first, hcopy]
  have hpeaks : (idealWeights r caps bits n).peak ≤ (idealWeights r caps bits r.last).peak :=
    by rw [idealWeights_eq, idealWeights_eq]; exact peak_mono (by norm_num) _ _ (hn.trans hlast)
  have hb := error_of_peak c z n p.errorUnit r.tolerance 0 (fun i => (pixelCaps r caps bits i).2)
    (fun i => stepCost r (capAt caps i) 1) p.errorUnit_pos.le r.tolerance_pos.le
    (by norm_num) hzero hfirst (fun i _ hi => (hsize i hi).2) (by
      intro i hi hl _
      have hr := idealReconstruction_residual run.state run.anchor run.working c i
      rw [hcopy, add_sub_cancel] at hr
      change norm1 (z (i + 1) - (z i ^ 2 + c)) ≤ _
      rw [show z (i + 1) - (z i ^ 2 + c) =
        run.state (i + 1) - (run.state i ^ 2 + run.working) by simpa only [hcopy] using hr]
      exact cost_bounds_step r caps run i hi (hl.trans_le hn) 1 hm
        (hcap i (hl.trans_le (hn.trans hlast))) (href i (hl.le.trans hn)).2)
    (by
      rw [← idealWeights_eq]
      exact (mul_le_mul_of_nonneg_left (by exact_mod_cast hpeaks) p.errorUnit_pos.le).trans hpeak)
  exact r.sampling.viewport.sampleError_le ((hb n le_rfl).2.trans (r.sampling.tolerance_eq _).le)

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate
