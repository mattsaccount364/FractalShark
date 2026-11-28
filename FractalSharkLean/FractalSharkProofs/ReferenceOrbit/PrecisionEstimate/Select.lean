module

public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Budget
import Mathlib.Data.Rat.Cast.Lemmas
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Executable limb selection and its connection to the CPU prefix theorems. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

def maxMPIRBits : ℕ := 64_000_000
def defaultMaxBits : ℕ := 65_536

structure Profile (r : Request) where
  enclosureBits : ℕ
  caps : Array ℚ
  sound : ∀ i, i < r.last → norm1 (orbit r.center.value i) ≤ capAt caps i

def refineProfile (r : Request) (maxBits bits : ℕ) : Option (Profile r) :=
  match h : enclose r.center bits ((2 : ℚ) ^ maxBits) r.last with
  | some (box, caps) =>
    some ⟨bits, caps, (enclose_sound r.center bits _ r.last box caps h).2.2⟩
  | none =>
    if hlt : bits < maxBits then
      refineProfile r maxBits (min maxBits (2 * bits + 1))
    else none
termination_by maxBits - bits
decreasing_by omega

def profile (r : Request) (maxBits : ℕ) : Option (Profile r) :=
  refineProfile r maxBits (min 128 maxBits)

def Profile.referenceFactor {r : Request} (p : Profile r) : ℚ :=
  (referenceWeights r p.caps r.tolerance r.last).peak

def Profile.supportFactor {r : Request} (p : Profile r) : ℚ :=
  (referenceWeights r p.caps 1 r.last).peak

def Profile.reconstructionFactor {r : Request} (p : Profile r) : ℚ :=
  (idealWeights r p.caps p.enclosureBits r.last).peak

/-- The ideal estimate needs a reference within one unit to bound the local
MPIR arithmetic. Express both inequalities in terms of the sample tolerance. -/
def Profile.idealFactor {r : Request} (p : Profile r) : ℚ :=
  max (p.supportFactor * r.tolerance) p.reconstructionFactor

def Profile.combinedFactor {r : Request} (p : Profile r) : ℚ :=
  max p.referenceFactor p.idealFactor

def fits (factor tolerance : ℚ) (blocks : ℕ) : Prop :=
  factor ≤ tolerance * (2 : ℚ) ^ (64 * blocks)

instance (factor tolerance : ℚ) (blocks : ℕ) : Decidable (fits factor tolerance blocks) :=
  inferInstanceAs (Decidable (factor ≤ tolerance * (2 : ℚ) ^ (64 * blocks)))

private theorem fits_mono {factor tolerance : ℚ} {k l : ℕ}
    (ht : 0 ≤ tolerance) (hkl : k ≤ l) (hk : fits factor tolerance k) :
    fits factor tolerance l :=
  hk.trans (mul_le_mul_of_nonneg_left
    (pow_le_pow_right₀ (by norm_num) (Nat.mul_le_mul_left 64 hkl)) ht)

def lowerBlock (test : ℕ → Bool) (lo hi : ℕ) : ℕ :=
  if h : lo < hi then
    let mid := lo + (hi - lo) / 2
    if test mid then lowerBlock test lo mid else lowerBlock test (mid + 1) hi
  else lo
termination_by hi - lo
decreasing_by all_goals omega

structure Selection (factor tolerance : ℚ) (maxBits : ℕ) where
  blocks : ℕ
  blocks_pos : 0 < blocks
  bits_le : 64 * blocks ≤ maxBits
  sufficient : fits factor tolerance blocks
  previous_insufficient : 1 < blocks → ¬fits factor tolerance (blocks - 1)

/-- Check the result and its immediately preceding limb boundary. Monotonicity
extends the latter comparison to every smaller supported precision. -/
def select (factor tolerance : ℚ) (maxBits : ℕ) :
    Option (Selection factor tolerance maxBits) :=
  let last := maxBits / 64
  if last = 0 ∨ ¬fits factor tolerance last then none
  else
    let k := lowerBlock (fun i => decide (fits factor tolerance i)) 1 last
    if h : 0 < k ∧ 64 * k ≤ maxBits ∧ fits factor tolerance k then
      if hp : k = 1 ∨ ¬fits factor tolerance (k - 1) then
        some ⟨k, h.1, h.2.1, h.2.2, fun hlt => hp.resolve_left (by omega)⟩
      else none
    else none

theorem Selection.smallest {factor tolerance : ℚ} {maxBits : ℕ}
    (s : Selection factor tolerance maxBits) (ht : 0 ≤ tolerance) {k : ℕ}
    (hk : 0 < k) (hf : fits factor tolerance k) : s.blocks ≤ k := by
  by_contra! hlt
  exact s.previous_insufficient (by omega) (fits_mono ht (by omega) hf)

def Selection.requestedBits {factor tolerance : ℚ} {maxBits : ℕ}
    (s : Selection factor tolerance maxBits) : ℕ := 64 * s.blocks

def Selection.precision {factor tolerance : ℚ} {maxBits : ℕ}
    (s : Selection factor tolerance maxBits) : MPIR.Precision :=
  MPIR.Precision.ofRequestedBits s.requestedBits

theorem Selection.bits_eq {factor tolerance : ℚ} {maxBits : ℕ}
    (s : Selection factor tolerance maxBits) : s.precision.bits = s.requestedBits := by
  have h := s.blocks_pos
  change 64 * ((max 53 (64 * s.blocks) + 127) / 64 - 1) = 64 * s.blocks
  omega

theorem Selection.error_le {factor tolerance : ℚ} {maxBits : ℕ}
    (s : Selection factor tolerance maxBits) :
    s.precision.errorUnit * factor ≤ (tolerance : ℝ) := by
  have h : (factor : ℝ) ≤ (2 : ℝ) ^ s.precision.bits * (tolerance : ℝ) := by
    have hf : (factor : ℝ) ≤ (tolerance * (2 : ℚ) ^ (64 * s.blocks) : ℚ) := by
      exact_mod_cast (show factor ≤ tolerance * (2 : ℚ) ^ (64 * s.blocks) from s.sufficient)
    simpa only [s.bits_eq, requestedBits, Rat.cast_mul, Rat.cast_pow, Rat.cast_ofNat, mul_comm] using hf
  have h' := (roundoffBudget_le_iff s.precision 0 factor tolerance).2 (by simpa using h)
  simpa using h'

structure Seed (p : MPIR.Precision) (x : ℚ) where
  limbs : MPIR.Limbs
  fits : limbs.used ≤ p.limbs + 1
  bounded : |limbs.rational| ≤ |x|
  error : |limbs.rational - x| ≤ grid p.bits * |x|

/-- Round toward zero on a grid with the input's radix exponent. The checked
mantissa and exponent can be loaded exactly into the destination slot. -/
def seed (p : MPIR.Precision) (x : ℚ) : Option (Seed p x) :=
  if hzero : x = 0 then
    some ⟨MPIR.Limbs.zero, Nat.zero_le _, by simp [MPIR.Limbs.rational, MPIR.Limbs.zero, hzero],
      by simp [MPIR.Limbs.rational, MPIR.Limbs.zero, hzero]⟩
  else
    let approx : ℤ := (Int.ofNat x.num.natAbs.log2 - Int.ofNat x.den.log2) / 64 + 1
    let top := if |x| < (MPIR.radix : ℚ) ^ (approx - 1) then approx - 1
      else if (MPIR.radix : ℚ) ^ approx ≤ |x| then approx + 1 else approx
    let unit := (MPIR.radix : ℚ) ^ (top - (p.limbs : ℤ))
    let mag := ⌊|x| / unit⌋
    let mantissa := if x < 0 then -mag else mag
    if h : 0 < p.limbs ∧ (MPIR.radix : ℤ) ^ (p.limbs - 1) ≤ |mantissa| ∧
        |mantissa| < (MPIR.radix : ℤ) ^ p.limbs then
      let limbs : MPIR.Limbs := ⟨mantissa, p.limbs, top, Or.inr h⟩
      if hb : |limbs.rational| ≤ |x| ∧ |limbs.rational - x| ≤ grid p.bits * |x| then
        some ⟨limbs, Nat.le_succ _, hb.1, hb.2⟩
      else none
    else none

private theorem unit_eq (p : MPIR.Precision) : (grid p.bits : ℝ) = p.errorUnit := by
  simp [grid, p.errorUnit_eq, zpow_neg, zpow_natCast, one_div]

theorem Seed.real_bounded {p : MPIR.Precision} {x : ℚ} (s : Seed p x) :
    |s.limbs.value| ≤ |(x : ℝ)| := by
  rw [← s.limbs.cast_rational]
  exact_mod_cast s.bounded

theorem Seed.real_error {p : MPIR.Precision} {x : ℚ} (s : Seed p x) :
    |s.limbs.value - (x : ℝ)| ≤ p.errorUnit * |(x : ℝ)| := by
  rw [← s.limbs.cast_rational, ← unit_eq]
  exact_mod_cast s.error

structure Recommendation (r : Request) (factor : ℚ) (maxBits : ℕ) where
  setting : Selection factor r.tolerance maxBits
  x : Seed setting.precision r.center.x
  y : Seed setting.precision r.center.y

noncomputable def Recommendation.anchor {r : Request} {f : ℚ} {maxBits : ℕ}
    (s : Recommendation r f maxBits) : ℂ := ⟨s.x.limbs.value, s.y.limbs.value⟩

theorem Recommendation.copy_exact {r : Request} {f : ℚ} {maxBits : ℕ}
    (s : Recommendation r f maxBits) (work : State s.setting.precision)
    (hx : MPIR.Set s.setting.precision s.x.limbs work.x.val)
    (hy : MPIR.Set s.setting.precision s.y.limbs work.y.val) : work.value = s.anchor := by
  apply Complex.ext
  · exact MPIR.set_exact_of_fits s.x.fits hx
  · exact MPIR.set_exact_of_fits s.y.fits hy

private theorem Recommendation.magnitude_le {r : Request} {f : ℚ} {maxBits : ℕ}
    (s : Recommendation r f maxBits) : norm1 s.anchor ≤ r.center.magnitude := by
  simpa [Recommendation.anchor, Parameter.magnitude, norm1] using
    add_le_add s.x.real_bounded s.y.real_bounded

private theorem Recommendation.parameter_error {r : Request} {f : ℚ} {maxBits : ℕ}
    (s : Recommendation r f maxBits) :
    norm1 (s.anchor - r.center.value) ≤ s.setting.precision.errorUnit * r.center.magnitude := by
  simpa [Recommendation.anchor, Parameter.value, Parameter.magnitude, norm1, mul_add] using
    add_le_add s.x.real_error s.y.real_error

inductive Failure where
  | enclosureLimit
  | precisionLimit
  | coordinateEncoding
  deriving DecidableEq, Repr

def recommend (r : Request) (factor : ℚ) (maxBits : ℕ) :
    Except Failure (Recommendation r factor maxBits) := do
  let some setting := select factor r.tolerance maxBits | throw .precisionLimit
  let some x := seed setting.precision r.center.x | throw .coordinateEncoding
  let some y := seed setting.precision r.center.y | throw .coordinateEncoding
  pure ⟨setting, x, y⟩

structure Estimate (r : Request) (maxBits : ℕ) where
  profile : Profile r
  reference : Except Failure (Recommendation r profile.referenceFactor maxBits)
  ideal : Except Failure (Recommendation r profile.idealFactor maxBits)
  combined : Except Failure (Recommendation r profile.combinedFactor maxBits)

def estimate (r : Request) (maxBits : ℕ) : Except Failure (Estimate r maxBits) := do
  let some p := profile r maxBits | throw .enclosureLimit
  pure ⟨p, recommend r p.referenceFactor maxBits, recommend r p.idealFactor maxBits,
    recommend r p.combinedFactor maxBits⟩

/-- Evaluate the same fixed-orbit weights used by the precision selection,
but retain the allowance at a particular index for a later error budget. -/
def Profile.referenceErrorAt {r : Request} (profile : Profile r) (bits n : ℕ) : ℚ :=
  (referenceWeights r profile.caps r.tolerance n).current / (2 : ℚ) ^ bits

theorem Profile.referenceErrorAt_le_peak {r : Request} (profile : Profile r) {bits n : ℕ}
    (hn : n ≤ r.last) :
    profile.referenceErrorAt bits n ≤ profile.referenceFactor / (2 : ℚ) ^ bits := by
  unfold Profile.referenceErrorAt Profile.referenceFactor
  exact div_le_div_of_nonneg_right (referenceWeights_current_le_peak r profile.caps r.tolerance hn)
    (by positivity)

theorem Recommendation.reference_error_at {r : Request} (profile : Profile r) {f : ℚ}
    {maxBits last : ℕ} (s : Recommendation r f maxBits) (hf : profile.referenceFactor ≤ f)
    (run : MPIRViewportPrefix s.setting.precision r.backend.coefficient last)
    (hwork : run.working = s.anchor) (hlast : last ≤ r.last) :
    ∀ i, i ≤ last →
      norm1 (run.state i - orbit r.center.value i) ≤
        profile.referenceErrorAt s.setting.precision.bits i ∧
      norm1 (run.state i - orbit r.center.value i) ≤ r.tolerance := by
  have hmag : norm1 run.working ≤ r.center.magnitude := by
    rw [hwork]
    exact s.magnitude_le
  have hinput : norm1 (run.working - r.center.value) ≤
      s.setting.precision.errorUnit * r.center.magnitude := by rw [hwork]; exact s.parameter_error
  have hf' : (profile.referenceFactor : ℝ) ≤ f := by exact_mod_cast hf
  have hpeak := mul_le_mul_of_nonneg_left hf' s.setting.precision.errorUnit_pos.le
  intro i hi
  have bounds := reference_error_bounds r profile.caps r.tolerance run hlast r.tolerance_pos.le
    profile.sound hmag hinput (hpeak.trans s.setting.error_le) i hi
  refine ⟨?_, bounds.2⟩
  simpa [Profile.referenceErrorAt, MPIR.Precision.errorUnit_eq, div_eq_mul_inv, mul_comm] using bounds.1

theorem Recommendation.reference_correct {r : Request} (profile : Profile r) {f : ℚ}
    {maxBits last : ℕ} (s : Recommendation r f maxBits) (hf : profile.referenceFactor ≤ f)
    (run : MPIRViewportPrefix s.setting.precision r.backend.coefficient last)
    (hwork : run.working = s.anchor) (hlast : last ≤ r.last) :
    ∀ i, i ≤ last → r.sampling.viewport.sampleError (run.state i - orbit r.center.value i) ≤
      r.fraction := by
  intro i hi
  apply r.sampling.viewport.sampleError_le
  rw [← r.sampling.tolerance_eq]
  exact (s.reference_error_at profile hf run hwork hlast i hi).2

theorem Recommendation.ideal_correct {r : Request} (profile : Profile r) {f : ℚ}
    {maxBits last : ℕ} (s : Recommendation r f maxBits) (hf : profile.idealFactor ≤ f)
    (run : MPIRViewportPrefix s.setting.precision r.backend.coefficient last)
    (hwork : run.working = s.anchor) (hcopy : run.working = run.anchor) (hlast : last ≤ r.last)
    (c : ℂ) (hc : r.sampling.viewport.Contains c) (n : ℕ) (hn : n ≤ last)
    (hinside : ∀ i, i < n → (orbit c i).re ^ 2 + (orbit c i).im ^ 2 ≤
      (r.bailoutRadius : ℝ) ^ 2) :
    r.sampling.viewport.sampleError (idealReconstruction run.state run.anchor c n - orbit c n) ≤
      r.fraction := by
  have hmag : norm1 run.working ≤ r.center.magnitude := by
    rw [hwork]
    exact s.magnitude_le
  have hinput : norm1 (run.working - r.center.value) ≤
      s.setting.precision.errorUnit * r.center.magnitude := by rw [hwork]; exact s.parameter_error
  have hu := s.setting.precision.errorUnit_pos.le
  have hs : profile.supportFactor * r.tolerance ≤ f := (le_max_left _ _).trans hf
  have hi : profile.reconstructionFactor ≤ f := (le_max_right _ _).trans hf
  have ht : (0 : ℝ) < r.tolerance := by exact_mod_cast r.tolerance_pos
  have hguard : s.setting.precision.errorUnit * (profile.supportFactor : ℝ) ≤ 1 := by
    apply (mul_le_mul_iff_right₀ ht).mp
    rw [show (r.tolerance : ℝ) * (s.setting.precision.errorUnit * profile.supportFactor) =
      s.setting.precision.errorUnit * (profile.supportFactor * (r.tolerance : ℝ)) by ring, mul_one]
    exact (mul_le_mul_of_nonneg_left (by exact_mod_cast hs) hu).trans s.setting.error_le
  have hi' : (profile.reconstructionFactor : ℝ) ≤ f := by exact_mod_cast hi
  have hpeak := (mul_le_mul_of_nonneg_left hi' hu).trans s.setting.error_le
  exact ideal_in_samples r profile.caps profile.enclosureBits run hlast profile.sound hmag hinput
    hcopy (by exact_mod_cast hguard) hpeak c hc n hn hinside

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate
