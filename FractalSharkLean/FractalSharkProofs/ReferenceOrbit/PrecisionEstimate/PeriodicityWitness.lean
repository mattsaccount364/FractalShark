module

public import FractalSharkProofs.ReferenceOrbit.SingleThreaded.RealAxis
public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Budget
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-! Search for, and certify, a retained ST sample whose error exceeds the viewport budget. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.PeriodicityWitness

open SingleThreaded

def center (exponent : ℕ) : ℚ := -2 + grid exponent

def zoom (exponent : ℕ) : ℚ := 2 ^ exponent

-- Before aspect fitting, both spans are 4 / zoom. Their extracted exponent
-- is 3 - exponent because MPIR returns a mantissa of one half for powers of two.
def requestedBits (exponent : ℕ) : ℕ :=
  cpuRequestedBits (3 - (exponent : ℤ)) (3 - (exponent : ℤ)) false

def precision (exponent : ℕ) : MPIR.Precision := .ofRequestedBits (requestedBits exponent)

def radius (exponent : ℕ) : ℚ := 2 * grid exponent

theorem radius_pos (exponent : ℕ) : 0 < radius exponent :=
  mul_pos (by norm_num) (grid_pos exponent)

def tolerance (exponent : ℕ) (c : ℚ) : ℚ :=
  (Sampling.ofZoom ⟨c, 0⟩ (zoom exponent) 5120 2160 2
    (by unfold zoom; positivity) (by decide) (by decide) (by decide)).tolerance (1 / 4)

/-- Carry the range check along with the replay so each prefix check takes one pass. -/
def replay (p : MPIR.Precision) (c : ℚ) : ℕ → ℚ × Bool
  | 0 => (0, true)
  | 1 => (c, decide (RealAxisWindow p c c))
  | n + 2 =>
    let previous := replay p c (n + 1)
    let x := realAxisNext p c previous.1
    (x, previous.2 && decide (RealAxisWindow p c x))

private theorem replay_window (p : MPIR.Precision) (c : ℚ) :
    ∀ last i, 1 ≤ i → i ≤ last → (replay p c last).2 = true →
      RealAxisWindow p c (replay p c i).1 := by
  intro last
  induction last with
  | zero => intro i hi hle; omega
  | succ last ih =>
    intro i hi hle h
    cases last with
    | zero =>
      have he : i = 1 := by omega
      subst i
      simpa only [replay, decide_eq_true_eq] using h
    | succ last =>
      have hw : (replay p c (last + 1)).2 = true ∧
          RealAxisWindow p c (replay p c (last + 2)).1 := by
        simpa only [replay, Bool.and_eq_true, decide_eq_true_eq] using h
      by_cases he : i = last + 2
      · simpa only [he] using hw.2
      · exact ih i hi (by omega) hw.1

-- A loose envelope for finite binary64 multiplication and aligned addition.
-- The derivative growth factor also accounts for the additive one.
def shadowInflation : ℚ := 65 / 64

def derivativeGrowth : ℚ := 11 / 2

-- At these depths 768 dyadic bits leave ample room for the interval width
-- accumulated over 80 squarings. Eight covers the states in the search window.
def enclosureBits : ℕ := 768
def enclosureLimit : ℚ := 8

/-- A successful check requires strict separation between the rounded state
and the outward enclosure, together with a margin against periodicity. -/
def check (exponent : ℕ) (c : ℚ) (n : ℕ) : Bool :=
  let trace := replay (precision exponent) c n
  trace.2 &&
    decide (1 ≤ n ∧ 2 * radius exponent * derivativeGrowth ^ (n - 1) < 1) &&
    match enclose ⟨c, 0⟩ enclosureBits enclosureLimit n with
    | some (box, _) => decide (trace.1 + tolerance exponent c < box.x.lo)
    | none => false

/-- Search in the supplied zoom order, then in increasing sample order at each zoom. -/
def search (exponents : List ℕ) (last : ℕ) : Option (ℕ × ℕ) :=
  exponents.findSome? fun exponent =>
    (List.range last).findSome? fun i =>
      if check exponent (center exponent) (i + 1) then some (exponent, i + 1) else none

/-- Bounds for the finite HDRx64 real operations used by the witness. The
imaginary parts stay zero. The radius and factor of two are powers of two,
so those scalings preserve the mantissa. Only the finite operations in the
checked prefix are represented by this envelope. -/
structure RealShadow where
  convert : ℚ → ℝ
  mul : ℝ → ℝ → ℝ
  add : ℝ → ℝ → ℝ
  convert_bound : ∀ q, 7 / 4 ≤ |q| → |q| ≤ 2 →
    1 ≤ |convert q| ∧ |convert q| ≤ 2
  mul_bound : ∀ a b, |mul a b| ≤ (shadowInflation : ℝ) * |a| * |b|
  add_bound : ∀ a b, |add a b| ≤ (shadowInflation : ℝ) * (|a| + |b|)

noncomputable def RealShadow.checks (shadow : RealShadow) (states : ℕ → ℚ)
    (c : ℚ) (r : ℝ) : ShadowChecks where
  abortBefore := fun _ => false
  nearReturn := fun n d => decide (|shadow.convert (states n)| < 2 * r * |d.re|)
  advanceDerivative := fun n d => ⟨shadow.add (2 * shadow.mul (shadow.convert (states n)) d.re) 1, 0⟩
  bailout := fun n =>
    let sum := shadow.add (shadow.convert (states n)) (shadow.convert c)
    decide (256 < shadow.mul sum sum)

private theorem derivative_bound (shadow : RealShadow) {z d D : ℝ}
    (hz : |z| ≤ 2) (hd : |d| ≤ D) (hD : 1 ≤ D) :
    |shadow.add (2 * shadow.mul z d) 1| ≤ (derivativeGrowth : ℝ) * D := by
  have hm := shadow.mul_bound z d
  have ha := shadow.add_bound (2 * shadow.mul z d) 1
  norm_num [shadowInflation, abs_mul] at hm ha
  norm_num [derivativeGrowth]
  have hmul := mul_le_mul hz hd (abs_nonneg _) (by norm_num : (0 : ℝ) ≤ 2)
  linarith

private theorem retained_of_shadow_bound (shadow : RealShadow) (values : ℕ → ℚ) (c : ℚ)
    (r : ℝ) (n limit : ℕ) (hr : 0 ≤ r) (hlimit : n ≤ limit)
    (hguard : 2 * r * (derivativeGrowth : ℝ) ^ (n - 1) < 1)
    (hc : |shadow.convert c| ≤ 2)
    (hx : ∀ i, 1 ≤ i → i ≤ n → 1 ≤ |shadow.convert (values i)| ∧
      |shadow.convert (values i)| ≤ 2) :
    n ∈ (run .singleThreaded .enabled (shadow.checks values c r) limit).samples .singleThreaded := by
  let checks := shadow.checks values c r
  have hg : (1 : ℝ) ≤ (derivativeGrowth : ℝ) := by norm_num [derivativeGrowth]
  have reach : ∀ remaining i (d : ℝ), n ≤ i + remaining →
      |d| ≤ (derivativeGrowth : ℝ) ^ i →
      n ≤ (runFrom .singleThreaded .enabled checks remaining i ⟨d, 0⟩).entered := by
    intro remaining
    induction remaining with
    | zero =>
      intro i d h _
      simpa [runFrom] using h
    | succ remaining ih =>
      intro i d hremaining hd
      by_cases hi : n ≤ i
      · exact hi.trans (runFrom_entered_bounds _ _ _ _ _ _).1
      have hin : i + 1 ≤ n := by omega
      obtain ⟨hlo, hhi⟩ := hx (i + 1) (by omega) hin
      have hpow : |d| ≤ (derivativeGrowth : ℝ) ^ (n - 1) :=
        hd.trans (pow_le_pow_right₀ hg (by omega))
      have hthreshold : 2 * r * |d| < 1 :=
        (mul_le_mul_of_nonneg_left hpow (by positivity)).trans_lt hguard
      have hnear : ¬ |shadow.convert (values (i + 1))| < 2 * r * |d| := by linarith
      let sum := shadow.add (shadow.convert (values (i + 1))) (shadow.convert c)
      have hsum : |sum| ≤ 5 := by
        have h := shadow.add_bound (shadow.convert (values (i + 1))) (shadow.convert c)
        change |sum| ≤ (shadowInflation : ℝ) * _ at h
        norm_num [shadowInflation] at h
        linarith
      have hbail : ¬ (256 : ℝ) < shadow.mul sum sum := by
        have h := shadow.mul_bound sum sum
        have hprod := mul_le_mul hsum hsum (abs_nonneg sum) (by norm_num : (0 : ℝ) ≤ 5)
        have hm := le_abs_self (shadow.mul sum sum)
        norm_num [shadowInflation] at h
        linarith
      have hbail' : ¬ (256 : ℝ) < shadow.mul
          (shadow.add (shadow.convert (values (i + 1))) (shadow.convert c))
          (shadow.add (shadow.convert (values (i + 1))) (shadow.convert c)) := hbail
      have hnext :
          |shadow.add (2 * shadow.mul (shadow.convert (values (i + 1))) d) 1| ≤
            (derivativeGrowth : ℝ) ^ (i + 1) := by
        rw [pow_succ, mul_comm ((derivativeGrowth : ℝ) ^ i)]
        exact derivative_bound shadow hhi hd (one_le_pow₀ hg)
      simpa [runFrom, Backend.observation, checkPeriodicity, checks, RealShadow.checks,
        hnear, hbail'] using
        ih (i + 1) _ (by omega) hnext
  have hent := reach limit 0 1 (by simpa using hlimit) (by simp)
  rw [serial_samples, List.mem_range]
  change n < (runFrom .singleThreaded .enabled checks limit 0 (1 : ℂ)).entered + 1
  exact Nat.lt_succ_of_le hent

/-- The certificate compares a prefix of the MPIR trajectory with the exact
orbit and proves that the enabled schedule reaches the witness under the
real HDRx64 envelope. The trajectory starts with the exactly loaded working
parameter, and the schedule runs without external cancellation. -/
theorem checked_failure {exponent n limit : ℕ} {c : ℚ} (hcheck : check exponent c n = true)
    (shadow : RealShadow) (working : State (precision exponent))
    (states : ℕ → State (precision exponent))
    (hc : working.value = realAxisValue c) (hfirst : (states 1).value = working.value)
    (hstep : ∀ i, 1 ≤ i → i < n → Step working (states i) (states (i + 1)))
    (hlimit : n ≤ limit) :
    let checks := shadow.checks (fun i => (states i).x.val.rational) c (radius exponent)
    n ∈ (run .singleThreaded .enabled checks limit).samples .singleThreaded ∧
      (tolerance exponent c : ℝ) < norm1 ((states n).value - orbit (realAxisValue c) n) := by
  let p := precision exponent
  have hdata : ((replay p c n).2 = true ∧
      (1 ≤ n ∧ 2 * radius exponent * derivativeGrowth ^ (n - 1) < 1)) ∧
      (match enclose ⟨c, 0⟩ enclosureBits enclosureLimit n with
        | some (box, _) => decide ((replay p c n).1 + tolerance exponent c < box.x.lo)
        | none => false) = true := by
    simpa only [check, p, Bool.and_eq_true, decide_eq_true_eq] using hcheck
  obtain ⟨⟨hw, hn, hguard⟩, herror⟩ := hdata
  have eqAt : ∀ i, 1 ≤ i → i ≤ n → (states i).value = realAxisValue (replay p c i).1 := by
    intro i
    induction i with
    | zero => omega
    | succ i ih =>
      intro hi hle
      cases i with
      | zero => simpa only [replay] using hfirst.trans hc
      | succ i =>
        exact realAxis_step hc (ih (by omega) (by omega))
          (replay_window p c n (i + 1) (by omega) (by omega) hw)
          (hstep _ (by omega) (by omega))
  have hx (i : ℕ) (hi : 1 ≤ i) (hl : i ≤ n) :
      (states i).x.val.rational = (replay p c i).1 := by
    apply (Rat.cast_injective (α := ℝ))
    rw [MPIR.Limbs.cast_rational]
    exact congrArg Complex.re (eqAt i hi hl)
  have hwin i (hi : 1 ≤ i) (hl : i ≤ n) :=
    replay_window p c n i hi hl hw
  have hcBound : |shadow.convert c| ≤ 2 := by
    obtain ⟨_, _, hlow, hhigh, _⟩ := hwin 1 le_rfl hn
    exact (shadow.convert_bound c hlow hhigh).2
  have hguardReal : 2 * (radius exponent : ℝ) * (derivativeGrowth : ℝ) ^ (n - 1) < 1 := by
    exact_mod_cast hguard
  refine ⟨retained_of_shadow_bound shadow _ c (radius exponent) n limit
    (by exact_mod_cast (radius_pos exponent).le) hlimit hguardReal hcBound ?_, ?_⟩
  · intro i hi hl
    rw [hx i hi hl]
    obtain ⟨_, _, hlow, hhigh, _⟩ := hwin i hi hl
    exact shadow.convert_bound _ hlow hhigh
  · cases hbox : enclose ⟨c, 0⟩ enclosureBits enclosureLimit n with
    | none => simp [hbox] at herror
    | some answer =>
      obtain ⟨box, caps⟩ := answer
      have hlt : (replay p c n).1 + tolerance exponent c < box.x.lo := by simpa [hbox] using herror
      have hlo := (enclose_sound ⟨c, 0⟩ enclosureBits enclosureLimit n box caps hbox).1.1.1
      have hr : ((replay p c n).1 : ℝ) + (tolerance exponent c : ℝ) < (box.x.lo : ℝ) := by
        exact_mod_cast hlt
      rw [eqAt n hn le_rfl]
      unfold norm1
      have hdiff : (tolerance exponent c : ℝ) <
          (orbit (realAxisValue c) n).re - ((replay p c n).1 : ℝ) := by
        have horbit : (box.x.lo : ℝ) ≤ (orbit (realAxisValue c) n).re := by
          simpa [Parameter.value, realAxisValue] using hlo
        linarith only [horbit, hr]
      have habs : (orbit (realAxisValue c) n).re - ((replay p c n).1 : ℝ) ≤
          |((replay p c n).1 : ℝ) - (orbit (realAxisValue c) n).re| := by
        simpa only [neg_sub] using neg_le_abs (((replay p c n).1 : ℝ) - (orbit (realAxisValue c) n).re)
      exact hdiff.trans_le (habs.trans (le_add_of_nonneg_right (abs_nonneg _)))

/-- Enter Location's landscape fit at 320 effective bits shifts the real
center of this example by half a unit of the multiplication input grid. -/
def fittedCenter : ℚ := center 200 + grid 321

set_option maxRecDepth 100000 in
set_option maxHeartbeats 4000000 in
theorem entered_check : check 200 (center 200) 80 = true := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 4000000 in
theorem fitted_check : check 200 fittedCenter 80 = true := by decide +kernel

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.PeriodicityWitness
