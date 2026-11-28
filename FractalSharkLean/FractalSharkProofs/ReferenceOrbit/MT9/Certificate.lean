module

public import FractalSharkProofs.ReferenceOrbit.MT9.Dynamic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-! Conversion certificates for the implemented diagnostic paths. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

/-- Adjacent normalized significands and the common power-of-two scale give
one bin for MPIR's magnitude conversion toward zero. -/
structure ConversionBin where
  lower : ℝ
  upper : ℝ
  lower_pos : 0 < lower
  ordered : lower < upper

def ConversionBin.contains (bin : ConversionBin) (value : ℝ) : Prop :=
  bin.lower ≤ |value| ∧ |value| < bin.upper

def ConversionBin.normalized (bin : ConversionBin) : Prop :=
  ∃ (significand : ℕ) (exponent : ℤ),
    2 ^ (52 : ℕ) ≤ significand ∧ significand < 2 ^ (53 : ℕ) ∧
      bin.lower = significand * (2 : ℝ) ^ (exponent - 53) ∧
      bin.upper = (significand + 1) * (2 : ℝ) ^ (exponent - 53)

/-- A strict gap on both sides also preserves the component's sign. -/
theorem conversion_bin_stable {bin : ConversionBin} {dynamic fixed E : ℝ}
    (h : |dynamic - fixed| ≤ E) (hlo : E < |dynamic| - bin.lower)
    (hhi : E < bin.upper - |dynamic|) :
    bin.contains dynamic ∧ bin.contains fixed ∧ (dynamic < 0 ↔ fixed < 0) := by
  have hE := (abs_nonneg _).trans h
  have hlo' := (abs_sub_abs_le_abs_sub dynamic fixed).trans h
  have hhi' := abs_sub_abs_le_abs_sub fixed dynamic
  rw [abs_sub_comm fixed dynamic] at hhi'
  have hhi' := hhi'.trans h
  constructor
  · exact ⟨by linarith, by linarith⟩
  constructor
  · exact ⟨by linarith, by linarith⟩
  have hh := abs_le.mp h
  have hpos := bin.lower_pos
  constructor
  · intro hneg
    rw [abs_of_neg hneg] at hlo
    linarith
  · intro hneg
    by_contra hnonneg
    rw [abs_of_nonneg (le_of_not_gt hnonneg)] at hlo
    linarith

theorem regular_gaps {bin : ConversionBin} {dynamic E measuredLow measuredHigh sLow sHigh
    lowerBound upperBound : ℝ}
    (hl : |measuredLow - (|dynamic| - bin.lower)| ≤ sLow)
    (hh : |measuredHigh - (bin.upper - |dynamic|)| ≤ sHigh)
    (hlb : lowerBound ≤ measuredLow) (hub : upperBound ≤ measuredHigh)
    (hpassLow : E + sLow < lowerBound) (hpassHigh : E + sHigh < upperBound) :
    E < |dynamic| - bin.lower ∧ E < bin.upper - |dynamic| := by
  exact ⟨by linarith only [hpassLow, hlb, abs_le.mp hl],
    by linarith only [hpassHigh, hub, abs_le.mp hh]⟩

def tailWidth : ℕ := 32

/-- Bits 53 through 84 of the normalized fraction. The shifts do not overlap:
21 low bits from the second word and 11 high bits from the first. -/
def fastTail (normalizedHigh normalizedLow : ℕ) : ℕ :=
  (normalizedHigh % 2 ^ (11 : ℕ)) * 2 ^ (21 : ℕ) + normalizedLow / 2 ^ (43 : ℕ)

theorem fastTail_two_words (normalizedHigh normalizedLow : ℕ) :
    fastTail normalizedHigh normalizedLow =
      ((normalizedHigh * 2 ^ (64 : ℕ) + normalizedLow) / 2 ^ (43 : ℕ)) %
        2 ^ tailWidth
      ∨ 2 ^ (64 : ℕ) ≤ normalizedLow := by
  by_cases hlo : normalizedLow < 2 ^ (64 : ℕ)
  · left
    unfold fastTail tailWidth
    omega
  · exact Or.inr (by omega)

/-- Unread bits lie within one tail quantum. The smaller number of complete
quanta to either boundary is a lower bound for both true gaps. -/
theorem fast_gaps {bin : ConversionBin} {dynamic E q : ℝ} {tail : ℕ}
    (ht : tail < 2 ^ tailWidth) (hq : 0 < q)
    (hwidth : bin.upper = bin.lower + (2 : ℝ) ^ tailWidth * q)
    (hlo : bin.lower + tail * q ≤ |dynamic|)
    (hhi : |dynamic| < bin.lower + (tail + 1) * q)
    (hpass : E < (min tail (2 ^ tailWidth - 1 - tail) : ℕ) * q) :
    E < |dynamic| - bin.lower ∧ E < bin.upper - |dynamic| := by
  have h₁ : ((min tail (2 ^ tailWidth - 1 - tail) : ℕ) : ℝ) ≤ tail := by
    exact_mod_cast min_le_left tail (2 ^ tailWidth - 1 - tail)
  have h₂ : ((min tail (2 ^ tailWidth - 1 - tail) : ℕ) : ℝ) ≤
      (2 : ℝ) ^ tailWidth - 1 - tail := by
    have hsub : tail ≤ 2 ^ tailWidth - 1 := by omega
    have hpow : (1 : ℕ) ≤ 2 ^ tailWidth := by omega
    have hm : ((min tail (2 ^ tailWidth - 1 - tail) : ℕ) : ℝ) ≤
        ((2 ^ tailWidth - 1 - tail : ℕ) : ℝ) := by exact_mod_cast min_le_right tail _
    rw [Nat.cast_sub hsub, Nat.cast_sub hpow, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one] at hm
    exact hm
  exact ⟨by nlinarith only [hlo, hpass, mul_le_mul_of_nonneg_right h₁ hq.le],
    by nlinarith only [hwidth, hhi, hpass, mul_le_mul_of_nonneg_right h₂ hq.le]⟩

inductive CertificateStatus where
  | exact | realAxisZero | regular | fast | ambiguous | unsupported | invalid
  deriving DecidableEq

def CertificateStatus.accepted : CertificateStatus → Bool
  | .exact | .realAxisZero | .regular | .fast => true
  | _ => false

/-- The fast route and regular route both deliver true lower bounds for the
bin gaps. Structural imaginary zero has its own proof by the recurrence. -/
inductive AcceptedComponent (dynamic fixed E : ℝ) : Prop where
  | exact (hz : E = 0) (hbound : |dynamic - fixed| ≤ E)
  | realAxisZero (hd : dynamic = 0) (hf : fixed = 0)
  | bracket (bin : ConversionBin) (hbin : bin.normalized) (hbound : |dynamic - fixed| ≤ E)
      (hlo : E < |dynamic| - bin.lower) (hhi : E < bin.upper - |dynamic|)

theorem accepted_conversion {α : Type} (convert : ℝ → α)
    (hbin : ∀ (bin : ConversionBin) {x y : ℝ}, bin.normalized → bin.contains x → bin.contains y →
      (x < 0 ↔ y < 0) → convert x = convert y)
    {dynamic fixed E : ℝ} (h : AcceptedComponent dynamic fixed E) :
    convert dynamic = convert fixed := by
  cases h with
  | exact hz hbound =>
    have he := abs_le.mp hbound
    exact congrArg convert (by linarith only [hz, he.1, he.2])
  | realAxisZero hd hf => rw [hd, hf]
  | bracket bin hn hbound hlo hhi =>
    obtain ⟨hd, hf, hs⟩ := conversion_bin_stable hbound hlo hhi
    exact hbin bin hn hd hf hs

theorem sample_equivalence {α : Type} (convert : ℝ → α)
    (hbin : ∀ (bin : ConversionBin) {x y : ℝ}, bin.normalized → bin.contains x → bin.contains y →
      (x < 0 ↔ y < 0) → convert x = convert y)
    {dynamic fixed : ℂ} {E : ℝ} (hx : AcceptedComponent dynamic.re fixed.re E)
    (hy : AcceptedComponent dynamic.im fixed.im E) :
    (convert dynamic.re, convert dynamic.im) = (convert fixed.re, convert fixed.im) :=
  Prod.ext (accepted_conversion convert hbin hx) (accepted_conversion convert hbin hy)

/-- A published nonzero-index sample used the check from the preceding loop
body. Both result streams start with the same synthetic zero. -/
theorem accepted_prefix_equivalence {α : Type} (convert : ℝ → α)
    (hbin : ∀ (bin : ConversionBin) {x y : ℝ}, bin.normalized → bin.contains x → bin.contains y →
      (x < 0 ↔ y < 0) → convert x = convert y)
    (exit : LoopExit) (dynamic fixed : ℕ → ℂ) (bound : ℕ → ℝ)
    (hzero : dynamic 0 = fixed 0)
    (haccepted : ∀ n ∈ exit.samples .mt9, n ∈ checkedStates exit →
      AcceptedComponent (dynamic n).re (fixed n).re (bound n) ∧
      AcceptedComponent (dynamic n).im (fixed n).im (bound n)) :
    ∀ n ∈ exit.samples .mt9,
      (convert (dynamic n).re, convert (dynamic n).im) =
        (convert (fixed n).re, convert (fixed n).im) := by
  intro n hn
  cases n with
  | zero => rw [hzero]
  | succ n =>
    obtain ⟨hx, hy⟩ := haccepted _ hn (checked_before_publication exit (Nat.succ_pos n) hn)
    exact sample_equivalence convert hbin hx hy

/-- The shadow checks use the already converted sample and the previous
derivative. Identical callbacks capture a common center, radius, and numeric
format; the supplied abort trace accounts for external cancellation. -/
def shadowChecksFromSamples {α : Type} (samples : ℕ → α) (abortBefore : ℕ → Bool)
    (nearReturn : α → ℂ → Bool) (advanceDerivative : α → ℂ → ℂ)
    (bailout : α → Bool) : ShadowChecks :=
  ⟨abortBefore, fun n => nearReturn (samples n), fun n => advanceDerivative (samples n),
    fun n => bailout (samples n)⟩

theorem same_exit_of_sample_equivalence {α : Type} (mode : PeriodicityMode) (limit : ℕ)
    (dynamic fixed : ℕ → α) (abortBefore : ℕ → Bool) (nearReturn : α → ℂ → Bool)
    (advanceDerivative : α → ℂ → ℂ) (bailout : α → Bool)
    (heq : ∀ n ∈ (run .mt9 mode
      (shadowChecksFromSamples dynamic abortBefore nearReturn advanceDerivative bailout) limit).samples
        .mt9, dynamic n = fixed n) :
    run .mt9 mode (shadowChecksFromSamples dynamic abortBefore nearReturn advanceDerivative bailout)
        limit =
      run .mt9 mode (shadowChecksFromSamples fixed abortBefore nearReturn advanceDerivative bailout)
        limit := by
  refine run_eq_on_samples .mt9 mode _ _ limit ?_ ?_
  · intro j
    rfl
  · intro n hn d
    simp [checkPeriodicity, shadowChecksFromSamples, heq n hn]
    rfl

/-- Zero in the imaginary recurrence follows from zero imaginary input and
coordinate, independently of the size of the real component's disk. -/
theorem imaginary_zero {p full : MPIR.Precision} {c : State full} {old next : State p}
    (hc : c.y.val.value = 0) (hy : old.y.val.value = 0) (h : Step c old next) :
    next.y.val.value = 0 := by
  obtain ⟨_, _, _, doubled, _, _, _, hxy, hd, ha, _, _⟩ := h.run
  have hcross := (product_zero hxy (.inr hy)).1
  have hdoubled := (MPIR.mulUiTwo_bounds hd).1
  have hzero : doubled.val.value = 0 := by
    apply abs_eq_zero.mp
    have hb : |doubled.val.value| ≤ 0 := by simpa [hcross] using hdoubled
    exact le_antisymm hb (abs_nonneg _)
  exact (MPIR.add_zero_rhs doubled.property hc ha).trans hzero

structure Diagnostics where
  firstFailure : Option (ℕ × CertificateStatus)
  deriving DecidableEq

def Diagnostics.record (s : Diagnostics) (index : ℕ) (status : CertificateStatus) : Diagnostics :=
  if status.accepted then s
  else ⟨s.firstFailure.or (some (index, status))⟩

theorem record_latched (s : Diagnostics) (n : ℕ) (status : CertificateStatus)
    {first : ℕ × CertificateStatus} (h : s.firstFailure = some first) :
    (s.record n status).firstFailure = some first := by
  simp [Diagnostics.record, h]
  split <;> simp [h]

/-- A rejected check latches diagnostics alongside the normal publication. -/
def publish {α : Type} (s : Diagnostics) (index : ℕ) (status : CertificateStatus)
    (sample : α) : Diagnostics × α := (s.record index status, sample)

@[simp] theorem publish_sample {α : Type} (s : Diagnostics) (n : ℕ)
    (status : CertificateStatus) (sample : α) : (publish s n status sample).2 = sample := rfl

end FractalSharkProofs.ReferenceOrbit.MT9
