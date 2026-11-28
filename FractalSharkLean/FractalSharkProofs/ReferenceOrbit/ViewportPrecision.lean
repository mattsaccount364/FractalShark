module

public import FractalSharkProofs.ReferenceOrbit.Viewport
public import FractalSharkProofs.ReferenceOrbit.SingleThreaded.Schedule
public import FractalSharkProofs.ReferenceOrbit.MT3.Precision
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! A limb and sample-spacing certificate for the authoritative CPU orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

/-- Request made by PrecisionCalculator after extracting the full-span
exponents. The precision of initialized MPIR buffers may also be overridden. -/
def cpuRequestedBits (eX eY : ℤ) (requiresReuse : Bool) : ℕ :=
  max eX.natAbs eY.natAbs + if requiresReuse then 800 else 120

theorem requestedBits_le (requested : ℕ) :
    requested ≤ (MPIR.Precision.ofRequestedBits requested).bits := by
  change requested ≤ 64 * ((max 53 requested + 127) / 64 - 1)
  omega

/-- MPIR's padding supplies a sufficient sample budget when the extracted
exponents bracket the actual spans and the input error contribution is zero. -/
theorem Viewport.cpu_roundoff_le_budget (v : Viewport) (eX eY : ℤ) (requiresReuse : Bool)
    (fraction roundingFactor : ℝ) (hf : 0 ≤ fraction)
    (hx : (2 : ℝ) ^ (eX - 1) ≤ v.spanX) (hy : (2 : ℝ) ^ (eY - 1) ≤ v.spanY)
    (hguard : let d := max eX.natAbs eY.natAbs
      let p := MPIR.Precision.ofRequestedBits (cpuRequestedBits eX eY requiresReuse)
      2 * (max v.columns v.rows : ℝ) * roundingFactor ≤ fraction * 2 ^ (p.bits - d)) :
    (MPIR.Precision.ofRequestedBits (cpuRequestedBits eX eY requiresReuse)).errorUnit *
      roundingFactor ≤ v.sampleBudget fraction := by
  let d := max eX.natAbs eY.natAbs
  let p := MPIR.Precision.ofRequestedBits (cpuRequestedBits eX eY requiresReuse)
  let samples := max v.columns v.rows
  have hd : d ≤ p.bits := le_trans (Nat.le_add_right _ _) (requestedBits_le _)
  have hsplit : p.bits = d + (p.bits - d) := by omega
  have hxp : (0 : ℝ) < v.columns := by exact_mod_cast v.columns_pos
  have hyp : (0 : ℝ) < v.rows := by exact_mod_cast v.rows_pos
  have hs : (0 : ℝ) < samples :=
    hxp.trans_le (by exact_mod_cast le_max_left v.columns v.rows)
  let base : ℝ := 1 / 2 ^ (d + 1)
  have hbase : 0 ≤ base := by
    dsimp only [base]
    positivity
  have hexpX : -(↑(d + 1) : ℤ) ≤ eX - 1 := by
    have h := le_max_left eX.natAbs eY.natAbs
    dsimp only [d]
    omega
  have hexpY : -(↑(d + 1) : ℤ) ≤ eY - 1 := by
    have h := le_max_right eX.natAbs eY.natAbs
    dsimp only [d]
    omega
  have hbase_eq : base = (2 : ℝ) ^ (-(↑(d + 1) : ℤ)) := by
    simp only [base, zpow_neg, zpow_natCast, one_div]
  have hx' : base ≤ v.spanX := hbase_eq.le.trans
    ((zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hexpX).trans hx)
  have hy' : base ≤ v.spanY := hbase_eq.le.trans
    ((zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2) hexpY).trans hy)
  have hpitch : base / samples ≤ min v.pitchX v.pitchY := by
    apply le_min
    · exact (div_le_div_of_nonneg_left hbase hxp (by exact_mod_cast le_max_left v.columns v.rows)
        ).trans (div_le_div_of_nonneg_right hx' hxp.le)
    · exact (div_le_div_of_nonneg_left hbase hyp (by exact_mod_cast le_max_right v.columns v.rows)
        ).trans (div_le_div_of_nonneg_right hy' hyp.le)
  have hguard' : roundingFactor ≤ fraction * 2 ^ (p.bits - d) / (2 * samples) :=
    (le_div_iff₀ (mul_pos (by norm_num) hs)).2 (by
      simpa only [samples, Nat.cast_max, p, d, mul_comm, mul_left_comm, mul_assoc] using hguard)
  have hu : p.errorUnit = 1 / (2 ^ d * 2 ^ (p.bits - d)) := by
    rw [p.errorUnit_eq, zpow_neg, zpow_natCast, one_div, ← pow_add, ← hsplit]
  change p.errorUnit * roundingFactor ≤ v.sampleBudget fraction
  calc
    _ ≤ p.errorUnit * (fraction * 2 ^ (p.bits - d) / (2 * samples)) :=
      mul_le_mul_of_nonneg_left hguard' p.errorUnit_pos.le
    _ = fraction * (base / samples) := by
      rw [hu]
      dsimp only [base]
      rw [pow_succ]
      field_simp [hs.ne']
    _ ≤ v.sampleBudget fraction := mul_le_mul_of_nonneg_left hpitch hf

/-- The precision unit changes by one factor of the limb radix per limb. -/
theorem errorUnit_more_limbs (p q : MPIR.Precision) (extra : ℕ)
    (h : q.limbs = p.limbs + extra) :
    q.errorUnit = p.errorUnit / (2 : ℝ) ^ (64 * extra) := by
  have hbits : q.bits = p.bits + 64 * extra := by
    have hp := p.min_limbs
    unfold MPIR.Precision.bits
    omega
  rw [q.errorUnit_eq, p.errorUnit_eq, hbits, Nat.cast_add, neg_add, zpow_add₀ (by norm_num)]
  simp only [zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- Choose a fixed error ceiling for the amplification factor. The propagation
cap includes any displacement, while only the reference incurs MPIR roundoff. -/
noncomputable def mpirViewportBound (p : MPIR.Precision)
    (coefficient referenceCap propagationCap ceiling parameterMagnitude inputError : ℝ)
    (last : ℕ) : ℝ :=
  toleranceBound propagationCap ceiling (inputError + p.errorUnit * parameterMagnitude)
    (p.errorUnit * (coefficient * referenceCap ^ 2 + 2 * parameterMagnitude)) last

noncomputable def stepWeight (amplification : ℝ) (last : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (last - 1), amplification ^ i

noncomputable def inputWeight (amplification : ℝ) (last : ℕ) : ℝ :=
  if last = 0 then 0 else amplification ^ (last - 1) + stepWeight amplification last

noncomputable def viewportRoundingFactor
    (coefficient referenceCap amplification parameterMagnitude : ℝ) (last : ℕ) : ℝ :=
  inputWeight amplification last * parameterMagnitude +
    stepWeight amplification last * (coefficient * referenceCap ^ 2 + 2 * parameterMagnitude)

/-- Separate the original input discrepancy from the error controlled by the
selected limb count. The zero prefix has neither contribution. -/
theorem mpirViewportBound_split (p : MPIR.Precision)
    (coefficient referenceCap propagationCap ceiling parameterMagnitude inputError : ℝ)
    (last : ℕ) :
    mpirViewportBound p coefficient referenceCap propagationCap ceiling parameterMagnitude
        inputError last =
      inputWeight (2 * propagationCap + ceiling) last * inputError +
        p.errorUnit * viewportRoundingFactor coefficient referenceCap (2 * propagationCap + ceiling)
          parameterMagnitude last := by
  cases last with
  | zero =>
    simp [mpirViewportBound, toleranceBound, affineBound, viewportRoundingFactor, inputWeight,
      stepWeight]
  | succ n =>
    simp only [mpirViewportBound, toleranceBound, affineBound_geometric, viewportRoundingFactor,
      inputWeight, stepWeight, Nat.succ_ne_zero, ↓reduceIte, Nat.succ_sub_one]
    ring

/-- The power inequality is an exact limb-count test without using logarithms. -/
theorem roundoffBudget_le_iff (p : MPIR.Precision) (inputFloor roundingFactor budget : ℝ) :
    inputFloor + p.errorUnit * roundingFactor ≤ budget ↔
      roundingFactor ≤ (2 : ℝ) ^ p.bits * (budget - inputFloor) := by
  rw [p.errorUnit_eq, zpow_neg, zpow_natCast, inv_mul_eq_div, add_comm inputFloor,
    ← le_sub_iff_add_le, div_le_iff₀ (by positivity), mul_comm (budget - inputFloor)]

/-- Verified initialization and local arithmetic from an observed CPU runPrefix. -/
structure MPIRViewportPrefix (p : MPIR.Precision) (coefficient : ℝ) (last : ℕ) where
  anchor : ℂ
  working : ℂ
  state : ℕ → ℂ
  coefficient_nonneg : 0 ≤ coefficient
  copy_error : norm1 (working - anchor) ≤ p.errorUnit * norm1 anchor
  working_size : norm1 working ≤ norm1 anchor
  zero : state 0 = 0
  first : state 1 = working
  step_error : ∀ n, 1 ≤ n → n < last →
    norm1 (state (n + 1) - (state n ^ 2 + working)) ≤
      p.errorUnit * (coefficient * norm1 (state n) ^ 2 + 2 * norm1 working)

namespace MPIRViewportPrefix

/-- Build the shared certificate inputs from MPIR copies and verified steps. -/
noncomputable def fromStates {p : MPIR.Precision} {coefficient : ℝ} {last : ℕ}
    (inputX inputY : MPIR.Limbs) (working : State p) (states : ℕ → State p)
    (hcoefficient : 0 ≤ coefficient)
    (hcopyX : MPIR.Set p inputX working.x.val) (hcopyY : MPIR.Set p inputY working.y.val)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ n, 1 ≤ n → n < last →
      norm1 ((states (n + 1)).value - ((states n).value ^ 2 + working.value)) ≤
        p.errorUnit * (coefficient * norm1 (states n).value ^ 2 + 2 * norm1 working.value)) :
    MPIRViewportPrefix p coefficient last where
  anchor := ⟨inputX.value, inputY.value⟩
  working := working.value
  state := fun n => (states n).value
  coefficient_nonneg := hcoefficient
  copy_error := parameterCopy_error hcopyX hcopyY
  working_size := by
    simp only [norm1, State.value]
    exact add_le_add (hcopyX ▸ MPIR.keep_abs_le inputX _) (hcopyY ▸ MPIR.keep_abs_le inputY _)
  zero := hzero
  first := firstState_eq hx hy
  step_error := hstep

private theorem step_le {p : MPIR.Precision} {coefficient : ℝ} {last : ℕ}
    (runPrefix : MPIRViewportPrefix p coefficient last) (cap magnitude : ℝ)
    (hmag : norm1 runPrefix.anchor ≤ magnitude)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (runPrefix.state n) ≤ cap) :
    ∀ n, 1 ≤ n → n < last →
      norm1 (runPrefix.state (n + 1) - (runPrefix.state n ^ 2 + runPrefix.working)) ≤
        p.errorUnit * (coefficient * cap ^ 2 + 2 * magnitude) := by
  intro n hn hl
  have hsq := pow_le_pow_left₀ (norm1_nonneg _) (hsize n hn hl) 2
  have hworking := mul_le_mul_of_nonneg_left (runPrefix.working_size.trans hmag)
    (by norm_num : (0 : ℝ) ≤ 2)
  exact (runPrefix.step_error n hn hl).trans (mul_le_mul_of_nonneg_left
    (add_le_add (mul_le_mul_of_nonneg_left hsq runPrefix.coefficient_nonneg) hworking)
    p.errorUnit_pos.le)

private theorem budget_nonneg {p : MPIR.Precision} {coefficient magnitude cap : ℝ}
    (hc : 0 ≤ coefficient) (hm : 0 ≤ magnitude) :
    0 ≤ p.errorUnit * (coefficient * cap ^ 2 + 2 * magnitude) :=
  mul_nonneg p.errorUnit_pos.le (add_nonneg (mul_nonneg hc (sq_nonneg _))
    (mul_nonneg (by norm_num) hm))

theorem reference_error_in_samples {p : MPIR.Precision} {coefficient : ℝ} {last : ℕ}
    (runPrefix : MPIRViewportPrefix p coefficient last) (v : Viewport) (target : ℂ)
    (cap ceiling magnitude inputError fraction : ℝ)
    (hcap : 0 ≤ cap) (hceiling : 0 ≤ ceiling)
    (hmag : norm1 runPrefix.anchor ≤ magnitude) (hinput : norm1 (runPrefix.anchor - target) ≤ inputError)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (runPrefix.state n) ≤ cap)
    (hfinal : mpirViewportBound p coefficient cap cap ceiling magnitude inputError last ≤
      min ceiling (v.sampleBudget fraction)) :
    ∀ n, n ≤ last → v.sampleError (runPrefix.state n - orbit target n) ≤ fraction := by
  have hd : norm1 (runPrefix.working - target) ≤ inputError + p.errorUnit * magnitude := by
    rw [show runPrefix.working - target = (runPrefix.anchor - target) + (runPrefix.working - runPrefix.anchor)
      by ring]
    exact (norm1_add_le _ _).trans (add_le_add hinput
      (runPrefix.copy_error.trans (mul_le_mul_of_nonneg_left hmag p.errorUnit_pos.le)))
  have hb := prefix_error_le_of_affine_cap target runPrefix.working runPrefix.state last cap ceiling _ _
    hcap hceiling hd (budget_nonneg (p := p) (cap := cap) runPrefix.coefficient_nonneg
      ((norm1_nonneg _).trans hmag)) runPrefix.zero runPrefix.first hsize
    (runPrefix.step_le cap magnitude hmag hsize) (hfinal.trans (min_le_left _ _))
  exact fun n hn => v.sampleError_le ((hb n hn).trans (hfinal.trans (min_le_right _ _)))

theorem perturbed_error_in_samples {p : MPIR.Precision} {coefficient : ℝ} {last : ℕ}
    (runPrefix : MPIRViewportPrefix p coefficient last) (v : Viewport)
    (cap displacement ceiling magnitude fraction : ℝ)
    (hcap : 0 ≤ cap) (hd : 0 ≤ displacement) (hceiling : 0 ≤ ceiling)
    (hmag : norm1 runPrefix.anchor ≤ magnitude)
    (hsize : ∀ n, 1 ≤ n → n < last → norm1 (runPrefix.state n) ≤ cap)
    (hgeometry : displacementBound cap displacement (v.coverage runPrefix.anchor) last ≤ displacement)
    (hfinal : mpirViewportBound p coefficient cap (cap + displacement) ceiling magnitude 0 last ≤
      min ceiling (v.sampleBudget fraction)) :
    ∀ c, v.Contains c → ∀ n, n ≤ last →
      v.sampleError (idealReconstruction runPrefix.state runPrefix.anchor c n - orbit c n) ≤ fraction := by
  intro c hc
  let z := idealReconstruction runPrefix.state runPrefix.anchor c
  let working := runPrefix.working + (c - runPrefix.anchor)
  have hdelta := idealDelta_le_of_cap runPrefix.state (c - runPrefix.anchor) last cap displacement _
    hcap hd (v.offset_le_coverage runPrefix.anchor hc) hsize hgeometry
  have hz : ∀ n, 1 ≤ n → n < last → norm1 (z n) ≤ cap + displacement := by
    intro n hn hl
    exact (norm1_add_le _ _).trans (add_le_add (hsize n hn hl) (hdelta n hl))
  have hcopy : norm1 (working - c) ≤ 0 + p.errorUnit * magnitude := by
    rw [show working - c = runPrefix.working - runPrefix.anchor by dsimp only [working]; ring, zero_add]
    exact runPrefix.copy_error.trans (mul_le_mul_of_nonneg_left hmag p.errorUnit_pos.le)
  have hzero : z 0 = 0 := by simp [z, idealReconstruction, idealDelta, runPrefix.zero]
  have hfirst : z 1 = working := by
    simp [z, working, idealReconstruction, idealDelta, runPrefix.first]
  have hb := prefix_error_le_of_affine_cap c working z last (cap + displacement) ceiling _ _
    (add_nonneg hcap hd) hceiling hcopy (budget_nonneg (p := p) (cap := cap) runPrefix.coefficient_nonneg
      ((norm1_nonneg _).trans hmag)) hzero hfirst hz (by
        intro n hn hl
        rw [show z (n + 1) - (z n ^ 2 + working) = _ from
          idealReconstruction_residual runPrefix.state runPrefix.anchor runPrefix.working c n]
        exact runPrefix.step_le cap magnitude hmag hsize n hn hl)
    (hfinal.trans (min_le_left _ _))
  exact fun n hn => v.sampleError_le ((hb n hn).trans (hfinal.trans (min_le_right _ _)))

end MPIRViewportPrefix

/-- A serial prefix with an observed terminal index, in any periodicity mode. -/
noncomputable def SingleThreaded.viewportPrefix {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit last : ℕ)
    (hlast : last ∈ (run .singleThreaded mode checks limit).samples .singleThreaded)
    (inputX inputY : MPIR.Limbs) (working : State p) (states : ℕ → State p)
    (hcopyX : MPIR.Set p inputX working.x.val) (hcopyY : MPIR.Set p inputY working.y.val)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i →
      i < (run .singleThreaded mode checks limit).coherentThrough .singleThreaded →
      SingleThreaded.Step working (states i) (states (i + 1))) :
    MPIRViewportPrefix p 32 last :=
  MPIRViewportPrefix.fromStates inputX inputY working states (by norm_num) hcopyX hcopyY hzero hx hy
    (fun i hi hl => SingleThreaded.step_error (hstep i hi
      (hl.trans_le (sample_le_coherent .singleThreaded _ hlast))))

/-- The MT3 terminal observation limits the prefix to coherent worker results. -/
noncomputable def MT3.viewportPrefix {p : MPIR.Precision} (mode : PeriodicityMode)
    (checks : ShadowChecks) (limit last : ℕ)
    (hlast : last ∈ (run .mt3 mode checks limit).samples .mt3)
    (inputX inputY : MPIR.Limbs) (working : State p) (states : ℕ → State p)
    (hcopyX : MPIR.Set p inputX working.x.val) (hcopyY : MPIR.Set p inputY working.y.val)
    (hzero : (states 0).value = 0)
    (hx : MPIR.Set p working.x.val (states 1).x.val)
    (hy : MPIR.Set p working.y.val (states 1).y.val)
    (hstep : ∀ i, 1 ≤ i → i < (run .mt3 mode checks limit).coherentThrough .mt3 →
      MT3.Step i working (states i) (states (i + 1))) :
    MPIRViewportPrefix p 9 last :=
  MPIRViewportPrefix.fromStates inputX inputY working states (by norm_num) hcopyX hcopyY hzero hx hy
    (fun i hi hl => MT3.step_error (hstep i hi (hl.trans_le (sample_le_coherent .mt3 _ hlast))))

end FractalSharkProofs.ReferenceOrbit
