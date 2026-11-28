module

public import Mathlib.Algebra.Order.Archimedean.Real.Basic
import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
The value semantics of the Windows x64 MPIR primitives used by the reference orbit.
Exponents count 64-bit limbs. The integer mantissa includes any retained low zero limbs.
The operation relations describe the C branches at the level of limb values.
-/

@[expose] public section

namespace FractalSharkProofs.MPIR

def radix : ℕ := 2 ^ 64

noncomputable def grid (e : ℤ) : ℝ := (radix : ℝ) ^ e

theorem grid_pos (e : ℤ) : 0 < grid e := by
  unfold grid
  apply zpow_pos
  norm_num [radix]

theorem grid_mono {a b : ℤ} (h : a ≤ b) : grid a ≤ grid b := by
  unfold grid
  exact zpow_le_zpow_right₀ (by norm_num [radix]) h

theorem grid_add (a b : ℤ) : grid (a + b) = grid a * grid b := by
  exact zpow_add₀ (by norm_num [radix]) a b

/-- The logical contents of an `mpf_t` mantissa. A nonzero top limb is required. -/
structure Limbs where
  mantissa : ℤ
  used : ℕ
  top : ℤ
  normalized : mantissa = 0 ∧ used = 0 ∨
    0 < used ∧ (radix : ℤ) ^ (used - 1) ≤ |mantissa| ∧ |mantissa| < (radix : ℤ) ^ used

def Limbs.zero : Limbs :=
  ⟨0, 0, 0, Or.inl ⟨rfl, rfl⟩⟩

def Limbs.rational (a : Limbs) : ℚ :=
  (a.mantissa : ℚ) * (radix : ℚ) ^ (a.top - a.used)

noncomputable def Limbs.value (a : Limbs) : ℝ :=
  (a.mantissa : ℝ) * grid (a.top - a.used)

theorem Limbs.cast_rational (a : Limbs) : (a.rational : ℝ) = a.value := by
  simp [Limbs.rational, Limbs.value, grid]

@[simp] theorem Limbs.value_zero : Limbs.zero.value = 0 := by
  simp [Limbs.zero, Limbs.value]

def Limbs.neg (a : Limbs) : Limbs where
  mantissa := -a.mantissa
  used := a.used
  top := a.top
  normalized := by simpa only [neg_eq_zero, abs_neg] using a.normalized

@[simp] theorem Limbs.value_neg (a : Limbs) : a.neg.value = -a.value := by
  simp [Limbs.neg, Limbs.value]

theorem Limbs.grid_top_le (a : Limbs) (ha : a.value ≠ 0) :
    grid (a.top - 1) ≤ |a.value| := by
  rcases a.normalized with ⟨hz, _⟩ | ⟨hn, hlo, _⟩
  · exact (ha (by simp [Limbs.value, hz])).elim
  have hlo' : (radix : ℝ) ^ (a.used - 1 : ℕ) ≤ |(a.mantissa : ℝ)| := by
    exact_mod_cast hlo
  have he : (a.top - 1) = (↑(a.used - 1 : ℕ) : ℤ) + (a.top - ↑a.used) := by
    omega
  rw [he, grid_add, Limbs.value, abs_mul, abs_of_pos (grid_pos _), grid, zpow_natCast]
  exact mul_le_mul_of_nonneg_right hlo' (le_of_lt (grid_pos _))

/-- MPIR's internal `_mp_prec` includes one limb beyond the reported precision. -/
structure Precision where
  limbs : ℕ
  min_limbs : 2 ≤ limbs

def Precision.bits (p : Precision) : ℕ := 64 * (p.limbs - 1)

def Precision.ofRequestedBits (bits : ℕ) : Precision where
  limbs := (max 53 bits + 127) / 64
  min_limbs := by have := le_max_left 53 bits; omega

noncomputable def Precision.errorUnit (p : Precision) : ℝ :=
  grid (1 - p.limbs)

theorem Precision.errorUnit_eq (p : Precision) :
    p.errorUnit = (2 : ℝ) ^ (-(p.bits : ℤ)) := by
  change ((radix : ℝ) ^ ((1 : ℤ) - ↑p.limbs)) = _
  have hb : (radix : ℝ) = (2 : ℝ) ^ (64 : ℤ) := by norm_num [radix]
  rw [hb, ← zpow_mul]
  congr 1
  have := p.min_limbs
  simp only [Precision.bits, Nat.cast_mul, Nat.cast_ofNat]
  omega

theorem Precision.errorUnit_pos (p : Precision) : 0 < p.errorUnit := grid_pos _

theorem Precision.errorUnit_le_half (p : Precision) : p.errorUnit ≤ 1 / 2 := by
  have ht := p.min_limbs
  calc
    p.errorUnit ≤ grid (-1) := grid_mono (by omega)
    _ ≤ 1 / 2 := by norm_num [grid, radix]

/-- An initialized MPIR object has room for `_mp_prec + 1` limbs. -/
structure Float where
  precision : Precision
  limbs : Limbs
  fits : limbs.used ≤ precision.limbs + 1

def init (p : Precision) : Float := ⟨p, Limbs.zero, Nat.zero_le _⟩

def swap (a b : Float) : Float × Float := (b, a)

@[simp] theorem init_value (p : Precision) : (init p).limbs.value = 0 := Limbs.value_zero

@[simp] theorem swap_eq (a b : Float) : swap a b = (b, a) := rfl

/-- Discard every digit below the specified unit, toward zero for either sign. -/
noncomputable def truncate (x unit : ℝ) : ℝ :=
  let magnitude := (⌊|x| / unit⌋ : ℝ) * unit
  if x < 0 then -magnitude else magnitude

theorem truncate_bounds (x : ℝ) {unit : ℝ} (hu : 0 < unit) :
    |truncate x unit| ≤ |x| ∧ |truncate x unit - x| ≤ unit := by
  let magnitude := (⌊|x| / unit⌋ : ℝ) * unit
  have hq : 0 ≤ |x| / unit := div_nonneg (abs_nonneg _) hu.le
  have hf : (0 : ℝ) ≤ (⌊|x| / unit⌋ : ℝ) := by
    exact_mod_cast Int.floor_nonneg.mpr hq
  have hmag : 0 ≤ magnitude := mul_nonneg hf hu.le
  have hlo : magnitude ≤ |x| := (le_div_iff₀ hu).mp (Int.floor_le (|x| / unit))
  have hhi : |x| < magnitude + unit := by
    simpa only [magnitude, add_mul, one_mul] using
      (div_lt_iff₀ hu).mp (Int.lt_floor_add_one (|x| / unit))
  change |(if x < 0 then -magnitude else magnitude)| ≤ |x| ∧
    |(if x < 0 then -magnitude else magnitude) - x| ≤ unit
  by_cases hx : x < 0
  · rw [abs_of_neg hx] at hlo hhi
    simp [hx, abs_of_nonneg hmag, abs_of_neg hx]
    exact ⟨hlo, abs_le.mpr ⟨by linarith, by linarith⟩⟩
  · have hx' := le_of_not_gt hx
    rw [abs_of_nonneg hx'] at hlo hhi
    simp [hx, abs_of_nonneg hmag, abs_of_nonneg hx']
    exact ⟨hlo, abs_le.mpr ⟨by linarith, by linarith⟩⟩

/-- Keeping leading limbs is exact when the used size already fits. -/
noncomputable def keep (a : Limbs) (n : ℕ) : ℝ :=
  if a.used ≤ n then a.value else truncate a.value (grid (a.top - n))

theorem keep_abs_le (a : Limbs) (n : ℕ) : |keep a n| ≤ |a.value| := by
  unfold keep
  split
  · exact le_rfl
  · exact (truncate_bounds _ (grid_pos _)).1

private theorem grid_cut_le (p : Precision) (a : Limbs) (ha : a.value ≠ 0)
    {cut : ℤ} (hc : cut ≤ a.top - p.limbs) : grid cut ≤ p.errorUnit * |a.value| := by
  have he : a.top - (p.limbs : ℤ) = (1 - (p.limbs : ℤ)) + (a.top - 1) := by ring
  calc
    grid cut ≤ grid (a.top - p.limbs) := grid_mono hc
    _ = p.errorUnit * grid (a.top - 1) := by rw [he, grid_add, Precision.errorUnit]
    _ ≤ p.errorUnit * |a.value| :=
      mul_le_mul_of_nonneg_left (a.grid_top_le ha) p.errorUnit_pos.le

theorem keep_error (p : Precision) (a : Limbs) {n : ℕ} (hn : p.limbs ≤ n) :
    |keep a n - a.value| ≤ p.errorUnit * |a.value| := by
  unfold keep
  split
  · simp only [sub_self, abs_zero]
    exact mul_nonneg p.errorUnit_pos.le (abs_nonneg _)
  · rename_i hs
    have ha : a.value ≠ 0 := by
      intro hv
      rcases a.normalized with ⟨_, hz⟩ | ⟨_, hlo, _⟩
      · exact hs (hz ▸ Nat.zero_le n)
      · have hm : a.mantissa ≠ 0 := by
          intro hz
          rw [hz, abs_zero] at hlo
          have hp : 0 < (radix : ℤ) ^ (a.used - 1) := by
            apply pow_pos
            norm_num [radix]
          omega
        exact (mul_ne_zero (by exact_mod_cast hm) (grid_pos _).ne') hv
    exact (truncate_bounds _ (grid_pos _)).2.trans
      (grid_cut_le p a ha (by omega))

/-- `mpf_set` preserves the source exponent and can retain one extra limb. -/
def Set (p : Precision) (a r : Limbs) : Prop :=
  r.value = keep a (p.limbs + 1)

theorem set_error {p : Precision} {a r : Limbs} (h : Set p a r) :
    |r.value - a.value| ≤ p.errorUnit * |a.value| := by
  rw [h]
  exact keep_error p a (Nat.le_succ _)

theorem set_exact_of_fits {p : Precision} {a r : Limbs} (hf : a.used ≤ p.limbs + 1)
    (h : Set p a r) : r.value = a.value := by
  rw [h]
  simp [keep, hf]

theorem set_same_precision {a r : Float} (hp : a.precision = r.precision)
    (h : Set r.precision a.limbs r.limbs) : r.limbs.value = a.limbs.value := by
  exact set_exact_of_fits (hp ▸ a.fits) h

/-- `mpf_mul_2exp` reserves space for a carry on non-whole-limb shifts. -/
def Scale (p : Precision) (k : ℕ) (a r : Limbs) : Prop :=
  r.value = (2 : ℝ) ^ k * keep a (if k % 64 = 0 then p.limbs + 1 else p.limbs)

theorem scale_bounds {p : Precision} {k : ℕ} {a r : Limbs} (h : Scale p k a r) :
    |r.value| ≤ (2 : ℝ) ^ k * |a.value| ∧
      |r.value - (2 : ℝ) ^ k * a.value| ≤ p.errorUnit * ((2 : ℝ) ^ k * |a.value|) := by
  let n := if k % 64 = 0 then p.limbs + 1 else p.limbs
  have hn : p.limbs ≤ n := by dsimp [n]; split <;> omega
  have hp : 0 ≤ (2 : ℝ) ^ k := pow_nonneg (by norm_num) _
  change r.value = (2 : ℝ) ^ k * keep a n at h
  rw [h, ← mul_sub, abs_mul, abs_mul, abs_of_nonneg hp]
  exact ⟨mul_le_mul_of_nonneg_left (keep_abs_le a n) hp,
    by simpa only [mul_left_comm] using mul_le_mul_of_nonneg_left (keep_error p a hn) hp⟩

/-- In the unsigned multiply-by-two path, the leading discarded limb contributes
its high bit as carry-in. Any lower doubled limb contributes an even low word,
so adding the next carry bit cannot propagate across that word. The retained
value is the doubled input truncated at the original input's limb boundary. -/
def MulUiTwo (p : Precision) (a r : Limbs) : Prop :=
  r.value = if a.used ≤ p.limbs then 2 * a.value
    else truncate (2 * a.value) (grid (a.top - p.limbs))

theorem mulUiTwo_bounds {p : Precision} {a r : Limbs} (h : MulUiTwo p a r) :
    |r.value| ≤ 2 * |a.value| ∧
      |r.value - 2 * a.value| ≤ p.errorUnit * (2 * |a.value|) := by
  rw [h]
  have hnonneg := mul_nonneg p.errorUnit_pos.le (abs_nonneg a.value)
  split
  · rw [sub_self, abs_zero, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
    exact ⟨le_rfl, by linarith⟩
  · have hsize := truncate_bounds (2 * a.value) (grid_pos (a.top - p.limbs))
    rw [abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)] at hsize
    refine ⟨hsize.1, ?_⟩
    by_cases ha : a.value = 0
    · simp [ha, truncate]
    · have hgrid := grid_cut_le p a ha (cut := a.top - p.limbs) le_rfl
      exact hsize.2.trans (hgrid.trans (by linarith))

/-- `mpf_mul` trims both inputs, multiplies exactly, then trims the normalized product. -/
def Mul (p : Precision) (a b r : Limbs) : Prop :=
  ∃ product : Limbs, product.value = keep a p.limbs * keep b p.limbs ∧
    r.value = keep product (p.limbs + 1)

theorem mul_bounds {p : Precision} {a b r : Limbs} (h : Mul p a b r) :
    |r.value| ≤ |a.value * b.value| ∧
      |r.value - a.value * b.value| ≤ 3 * p.errorUnit * |a.value * b.value| := by
  obtain ⟨product, hp, hr⟩ := h
  have ha := keep_abs_le a p.limbs
  have hb := keep_abs_le b p.limbs
  have hpAbs : |product.value| ≤ |a.value| * |b.value| := by
    rw [hp, abs_mul]
    exact mul_le_mul ha hb (abs_nonneg _) (abs_nonneg _)
  have hrErr := keep_error p product (Nat.le_succ _)
  have haErr := keep_error p a (le_refl p.limbs)
  have hbErr := keep_error p b (le_refl p.limbs)
  rw [abs_mul]
  refine ⟨(hr ▸ keep_abs_le product (p.limbs + 1)).trans hpAbs, ?_⟩
  have h₁ : |r.value - product.value| ≤ p.errorUnit * (|a.value| * |b.value|) := by
    rw [hr]
    exact hrErr.trans (mul_le_mul_of_nonneg_left hpAbs p.errorUnit_pos.le)
  have h₂ : |(keep a p.limbs - a.value) * keep b p.limbs| ≤
      p.errorUnit * (|a.value| * |b.value|) := by
    rw [abs_mul, ← mul_assoc]
    exact mul_le_mul haErr hb (abs_nonneg _) (mul_nonneg p.errorUnit_pos.le (abs_nonneg _))
  have h₃ : |a.value * (keep b p.limbs - b.value)| ≤ p.errorUnit * (|a.value| * |b.value|) := by
    rw [abs_mul, mul_left_comm p.errorUnit]
    exact mul_le_mul_of_nonneg_left hbErr (abs_nonneg _)
  have he : r.value - a.value * b.value =
      (r.value - product.value) + ((keep a p.limbs - a.value) * keep b p.limbs +
        a.value * (keep b p.limbs - b.value)) := by rw [hp]; ring
  rw [he]
  refine (abs_add_le _ _).trans (add_le_add h₁ ((abs_add_le _ _).trans (add_le_add h₂ h₃)) |>.trans ?_)
  ring_nf
  exact le_rfl

def maximumTop (a b : Limbs) : ℤ := max a.top b.top

/-- For subtraction, a prefix and its following tail are split on the same limb boundary. -/
noncomputable def leadingPart (x : ℝ) (top : ℤ) : ℝ := truncate x (grid top)

noncomputable def tail (x : ℝ) (top : ℤ) : ℝ := x - leadingPart x top

noncomputable def prefixSum (a b : Limbs) (top : ℤ) : ℝ :=
  leadingPart a.value top + leadingPart b.value top

noncomputable def retainedSum (a b : Limbs) (top cutoff : ℤ) : ℝ :=
  prefixSum a b top + (truncate (tail a.value top) (grid cutoff) +
    truncate (tail b.value top) (grid cutoff))

/-- The two entry points share their magnitude branches. Subtraction flips the second sign.

The subtraction branches are expressed using their selected limb boundary. The C scans
select that boundary by matching leading limbs and by passing through borrow chains.
-/
inductive Add (p : Precision) (a b r : Limbs) : Prop where
  | zeroLeft (ha : a.value = 0) (hr : r.value = b.value ∨ Set p b r)
  | zeroRight (hb : b.value = 0) (hr : r.value = a.value ∨ Set p a r)
  | sameSign (hs : 0 < a.value * b.value)
      (hr : r.value = truncate a.value (grid (maximumTop a b - p.limbs)) +
        truncate b.value (grid (maximumTop a b - p.limbs)))
  | difference (hs : a.value * b.value < 0) (top : ℤ) (ht : top ≤ maximumTop a b)
      (hh : prefixSum a b top = 0)
      (hr : r.value = retainedSum a b top (top - (p.limbs + 1 : ℕ)))
  | borrow (hs : a.value * b.value < 0) (top : ℤ) (ht : top ≤ maximumTop a b)
      (hh : |prefixSum a b top| = grid top)
      (hr : r.value = retainedSum a b top (top - p.limbs))
  | remainder (hs : a.value * b.value < 0) (top cutoff : ℤ)
      (ht : top ≤ maximumTop a b) (hc : cutoff ≤ top - (p.limbs + 1 : ℕ))
      (hh : prefixSum a b top = 0)
      (he : tail a.value top = 0 ∨ tail b.value top = 0)
      (hr : r.value = retainedSum a b top cutoff)

def Sub (p : Precision) (a b r : Limbs) : Prop := Add p a b.neg r

theorem truncate_sum_error (a b : ℝ) (e : ℤ) :
    |(truncate a (grid e) + truncate b (grid e)) - (a + b)| ≤ 2 * grid e := by
  rw [show truncate a (grid e) + truncate b (grid e) - (a + b) =
      (truncate a (grid e) - a) + (truncate b (grid e) - b) by ring]
  calc
    _ ≤ |truncate a (grid e) - a| + |truncate b (grid e) - b| := abs_add_le _ _
    _ ≤ grid e + grid e :=
      add_le_add (truncate_bounds _ (grid_pos _)).2 (truncate_bounds _ (grid_pos _)).2
    _ = 2 * grid e := by ring

theorem retainedSum_error (a b : Limbs) (top cutoff : ℤ) :
    |retainedSum a b top cutoff - (a.value + b.value)| ≤ 2 * grid cutoff := by
  have he : a.value + b.value = prefixSum a b top + (tail a.value top + tail b.value top) := by
    unfold prefixSum tail
    ring
  change |prefixSum a b top + _ - (a.value + b.value)| ≤ _
  rw [he, add_sub_add_left_eq_sub]
  exact truncate_sum_error _ _ _

private theorem aligned_error (p : Precision) (a b : Limbs) {cut : ℤ}
    (ha : a.value ≠ 0) (hb : b.value ≠ 0) (hc : cut ≤ maximumTop a b - p.limbs) :
    2 * grid cut ≤ 2 * p.errorUnit * max |a.value| |b.value| := by
  have h : grid cut ≤ p.errorUnit * max |a.value| |b.value| := by
    unfold maximumTop at hc
    by_cases hab : a.top ≤ b.top
    · rw [max_eq_right hab] at hc
      exact (grid_cut_le p b hb hc).trans
        (mul_le_mul_of_nonneg_left (le_max_right _ _) p.errorUnit_pos.le)
    · rw [max_eq_left (le_of_not_ge hab)] at hc
      exact (grid_cut_le p a ha hc).trans
        (mul_le_mul_of_nonneg_left (le_max_left _ _) p.errorUnit_pos.le)
  linarith

private theorem zeroOperand_error (p : Precision) (a r : Limbs)
    (hr : r.value = a.value ∨ Set p a r) :
    |r.value - a.value| ≤ p.errorUnit * |a.value| := by
  rcases hr with he | he
  · rw [he, sub_self, abs_zero]
    exact mul_nonneg p.errorUnit_pos.le (abs_nonneg _)
  · exact set_error he

theorem add_error {p : Precision} {a b r : Limbs} (h : Add p a b r) :
    |r.value - (a.value + b.value)| ≤ 2 * p.errorUnit * max |a.value| |b.value| := by
  have aligned {cut : ℤ} (hs : a.value * b.value ≠ 0)
      (hc : cut ≤ maximumTop a b - p.limbs) :=
    aligned_error p a b (left_ne_zero_of_mul hs) (right_ne_zero_of_mul hs) hc
  cases h with
  | zeroLeft ha hr =>
    rw [ha, zero_add, abs_zero, max_eq_right (abs_nonneg _)]
    have he := zeroOperand_error p b r hr
    nlinarith [mul_nonneg p.errorUnit_pos.le (abs_nonneg b.value)]
  | zeroRight hb hr =>
    rw [hb, add_zero, abs_zero, max_eq_left (abs_nonneg _)]
    have he := zeroOperand_error p a r hr
    nlinarith [mul_nonneg p.errorUnit_pos.le (abs_nonneg a.value)]
  | sameSign hs hr =>
    rw [hr]
    exact (truncate_sum_error _ _ _).trans (aligned hs.ne' le_rfl)
  | difference hs top ht _ hr =>
    rw [hr]
    exact (retainedSum_error _ _ _ _).trans (aligned hs.ne (by omega))
  | borrow hs top ht _ hr =>
    rw [hr]
    exact (retainedSum_error _ _ _ _).trans (aligned hs.ne (by omega))
  | remainder hs top cutoff ht hc _ _ hr =>
    rw [hr]
    exact (retainedSum_error _ _ _ _).trans (aligned hs.ne (by omega))

theorem sub_error {p : Precision} {a b r : Limbs} (h : Sub p a b r) :
    |r.value - (a.value - b.value)| ≤ 2 * p.errorUnit * max |a.value| |b.value| := by
  simpa only [Limbs.value_neg, abs_neg, sub_eq_add_neg] using add_error h

end FractalSharkProofs.MPIR
