module

public import FractalSharkProofs.MPIR
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-! Integer split products, buffer lengths, and the final signed MPIR store. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

def activeWidth (p : MPIR.Precision) : ℕ := max 2 (p.limbs + 1)
def lowWidth (p : MPIR.Precision) : ℕ := activeWidth p / 2
def highWidth (p : MPIR.Precision) : ℕ := activeWidth p - lowWidth p

theorem split_widths (p : MPIR.Precision) :
    0 < lowWidth p ∧ lowWidth p ≤ highWidth p ∧
      lowWidth p + highWidth p = activeWidth p := by
  have := p.min_limbs
  simp only [lowWidth, highWidth, activeWidth]
  omega

theorem active_width_fits {active full : MPIR.Precision} (h : active.limbs ≤ full.limbs) :
    lowWidth active ≤ lowWidth full ∧ highWidth active ≤ highWidth full := by
  simp only [lowWidth, highWidth, activeWidth]
  omega

structure Split where
  high : ℕ
  low : ℕ
  deriving DecidableEq

def split (low x : ℕ) : Split :=
  ⟨x / MPIR.radix ^ low, x % MPIR.radix ^ low⟩

def Split.value (low : ℕ) (a : Split) : ℕ := a.high * MPIR.radix ^ low + a.low

/-- Zero padding is implicit: both slices denote their entire base-radix
integers, including all occupied MPIR guard limbs. -/
theorem split_value (low x : ℕ) : (split low x).value low = x := by
  simpa [split, Split.value, add_comm, mul_comm] using Nat.mod_add_div x (MPIR.radix ^ low)

theorem split_fits {lo hi x : ℕ} (h : x < MPIR.radix ^ (lo + hi)) :
    (split lo x).low < MPIR.radix ^ lo ∧ (split lo x).high < MPIR.radix ^ hi := by
  have hp : 0 < MPIR.radix ^ lo := pow_pos (by norm_num [MPIR.radix]) _
  refine ⟨Nat.mod_lt _ hp, (Nat.div_lt_iff_lt_mul hp).mpr ?_⟩
  rwa [← pow_add, add_comm hi lo]

def highProduct (a b : Split) : ℕ := a.high * b.high
def lowProduct (a b : Split) : ℕ := a.low * b.low
def sumProduct (a b : Split) : ℕ := (a.high + a.low) * (b.high + b.low)
def middleProduct (a b : Split) : ℕ := sumProduct a b - highProduct a b - lowProduct a b

theorem middle_eq (a b : Split) :
    middleProduct a b = a.high * b.low + a.low * b.high := by
  have he : sumProduct a b =
      highProduct a b + lowProduct a b + (a.high * b.low + a.low * b.high) := by
    unfold sumProduct highProduct lowProduct
    ring
  unfold middleProduct
  omega

theorem middle_no_borrow (a b : Split) :
    highProduct a b ≤ sumProduct a b ∧ lowProduct a b ≤ sumProduct a b - highProduct a b := by
  have he : sumProduct a b =
      highProduct a b + lowProduct a b + (a.high * b.low + a.low * b.high) := by
    unfold sumProduct highProduct lowProduct
    ring
  omega

def assemble (lo : ℕ) (a b : Split) : ℕ :=
  (lowProduct a b + highProduct a b * MPIR.radix ^ (2 * lo)) +
    middleProduct a b * MPIR.radix ^ lo

theorem assemble_eq (lo : ℕ) (a b : Split) :
    assemble lo a b = a.value lo * b.value lo := by
  rw [assemble, middle_eq, two_mul, pow_add]
  unfold highProduct lowProduct Split.value
  ring

theorem assembled_product (lo x y : ℕ) :
    assemble lo (split lo x) (split lo y) = x * y := by
  rw [assemble_eq, split_value, split_value]

/-- The sign is restored after unsigned assembly. The condition models the
exclusive-or of the original MPIR signs, including a zero operand. -/
def signedAssembly (lo : ℕ) (x y : ℤ) : ℤ :=
  let magnitude := assemble lo (split lo x.natAbs) (split lo y.natAbs)
  if ((x < 0) ∧ 0 ≤ y) ∨ ((y < 0) ∧ 0 ≤ x) then -(magnitude : ℤ) else magnitude

theorem signedAssembly_eq (lo : ℕ) (x y : ℤ) : signedAssembly lo x y = x * y := by
  simp only [signedAssembly, assembled_product, Nat.cast_mul, Int.natCast_natAbs]
  by_cases hx : x < 0
  · by_cases hy : y < 0
    · rw [ite_eq_right (by omega), abs_of_neg hx, abs_of_neg hy]
      ring
    · rw [ite_eq_left (by omega), abs_of_neg hx, abs_of_nonneg (by omega : 0 ≤ y)]
      ring
  · by_cases hy : y < 0
    · rw [ite_eq_left (by omega), abs_of_nonneg (by omega : 0 ≤ x), abs_of_neg hy]
      ring
    · rw [ite_eq_right (by omega), abs_of_nonneg (by omega : 0 ≤ x),
        abs_of_nonneg (by omega : 0 ≤ y)]

private theorem product_width {a b i j : ℕ} (ha : a < MPIR.radix ^ i)
    (hb : b < MPIR.radix ^ j) : a * b < MPIR.radix ^ (i + j) := by
  rw [pow_add]
  exact Nat.mul_lt_mul_of_lt_of_lt ha hb

/-- The native vectors hold the high and low products, the extra-limb sum
product, and the assembly. The complete product leaves both guards empty.
The shifted sum buffer fits the remaining destination range. -/
theorem batch_fits {lo hi x y : ℕ} (hlo : lo ≤ hi)
    (hx : x < MPIR.radix ^ (lo + hi)) (hy : y < MPIR.radix ^ (lo + hi)) :
    let a := split lo x
    let b := split lo y
    highProduct a b < MPIR.radix ^ (2 * hi) ∧
    lowProduct a b < MPIR.radix ^ (2 * lo) ∧
    sumProduct a b < MPIR.radix ^ (2 * (hi + 1)) ∧
    assemble lo a b < MPIR.radix ^ (2 * (lo + hi)) ∧
    2 * (hi + 1) ≤ 2 * (lo + hi) + 2 - lo := by
  obtain ⟨hxl, hxh⟩ := split_fits hx
  obtain ⟨hyl, hyh⟩ := split_fits hy
  have hwidth : MPIR.radix ^ lo ≤ MPIR.radix ^ hi :=
    Nat.pow_le_pow_right (by norm_num [MPIR.radix]) hlo
  have hsum {h l : ℕ} (hh : h < MPIR.radix ^ hi) (hl : l < MPIR.radix ^ lo) :
      h + l < MPIR.radix ^ (hi + 1) := by
    rw [pow_succ]
    have hb : 2 ≤ MPIR.radix := by norm_num [MPIR.radix]
    have ht := Nat.mul_le_mul_left (MPIR.radix ^ hi) hb
    omega
  dsimp only
  rw [assembled_product]
  simpa only [two_mul, highProduct, lowProduct, sumProduct] using And.intro (product_width hxh hyh)
    (And.intro (product_width hxl hyl)
      (And.intro (product_width (hsum hxh hxl) (hsum hyh hyl))
        (And.intro (product_width hx hy) (show (hi + 1) + (hi + 1) ≤
          (lo + hi) + (lo + hi) + 2 - lo by omega))))

/-- Signed normalization uses the occupied size of each input and the
normalized size of the full integer product. Padded widths never enter it. -/
theorem normalize_product (a b normalized : MPIR.Limbs)
    (hm : normalized.mantissa = a.mantissa * b.mantissa)
    (he : normalized.top = a.top + b.top - ((a.used : ℤ) + b.used - normalized.used)) :
    normalized.value = a.value * b.value := by
  have hp : normalized.top - (normalized.used : ℤ) =
      (a.top - a.used) + (b.top - b.used) := by omega
  simp only [MPIR.Limbs.value, hm, Int.cast_mul, hp, MPIR.grid_add]
  ring

/-- Contract of the three MT9 products. MPIR's own multiplication also clips
inputs. MT9 assembles their full product and trims only the output. -/
structure Product (p : MPIR.Precision) (a b r : MPIR.Limbs) : Prop where
  stored : ∃ product : MPIR.Limbs, product.value = a.value * b.value ∧
    r.value = MPIR.keep product (p.limbs + 1) ∧
    r.used = min product.used (p.limbs + 1) ∧
    r.top = if r.used = 0 then 0 else product.top

theorem product_of_store {p : MPIR.Precision} {a b r product : MPIR.Limbs} {lo : ℕ}
    (hm : product.mantissa = signedAssembly lo a.mantissa b.mantissa)
    (he : product.used ≠ 0 →
      product.top = a.top + b.top - ((a.used : ℤ) + b.used - product.used))
    (hr : r.value = MPIR.keep product (p.limbs + 1))
    (hu : r.used = min product.used (p.limbs + 1))
    (ht : r.top = if r.used = 0 then 0 else product.top) : Product p a b r := by
  rw [signedAssembly_eq] at hm
  refine ⟨⟨product, ?_, hr, hu, ht⟩⟩
  by_cases hz : product.used = 0
  · have hm0 : product.mantissa = 0 := by
      rcases product.normalized with hn | ⟨hn, _, _⟩
      · exact hn.1
      · omega
    have hzero := mul_eq_zero.mp (hm.symm.trans hm0)
    rcases hzero with hzInput | hzInput <;> simp [MPIR.Limbs.value, hm0, hzInput]
  · exact normalize_product a b product hm (he hz)

theorem product_bounds {p : MPIR.Precision} {a b r : MPIR.Limbs} (h : Product p a b r) :
    |r.value| ≤ |a.value * b.value| ∧
      |r.value - a.value * b.value| ≤ p.errorUnit * |a.value * b.value| := by
  obtain ⟨product, hp, hr, _, _⟩ := h.stored
  rw [hr, ← hp]
  exact ⟨MPIR.keep_abs_le _ _, MPIR.keep_error p product (Nat.le_succ _)⟩

theorem product_limb_error {p : MPIR.Precision} {a b r : MPIR.Limbs} (h : Product p a b r) :
    |r.value - a.value * b.value| ≤
      if r.used < p.limbs + 1 then 0 else MPIR.grid (r.top - (p.limbs + 1 : ℕ)) := by
  obtain ⟨product, hp, hr, hu, he⟩ := h.stored
  rw [← hp, hr]
  by_cases hs : product.used ≤ p.limbs + 1
  · simp only [MPIR.keep, hs, ite_true, sub_self, abs_zero]
    split
    · exact le_rfl
    · exact (MPIR.grid_pos _).le
  · have hrUsed : r.used = p.limbs + 1 := hu.trans (min_eq_right (by omega))
    have hrTop : r.top = product.top := by
      rw [he, hrUsed]
      simp
    simp only [MPIR.keep, hs, ite_false, hrUsed, lt_irrefl, hrTop]
    exact (MPIR.truncate_bounds _ (MPIR.grid_pos _)).2

theorem product_zero {p : MPIR.Precision} {a b r : MPIR.Limbs}
    (h : Product p a b r) (hz : a.value = 0 ∨ b.value = 0) :
    r.value = 0 ∧ r.used = 0 ∧ r.top = 0 := by
  obtain ⟨product, hp, hr, hu, he⟩ := h.stored
  have hzero : product.value = 0 := by rw [hp]; rcases hz with hz | hz <;> simp [hz]
  have hzUsed : product.used = 0 := by
    rcases product.normalized with hz | ⟨hused, hnorm, _⟩
    · exact hz.2
    · have hm : (product.mantissa : ℝ) = 0 := by
        exact (mul_eq_zero.mp hzero).resolve_right (MPIR.grid_pos _).ne'
      have hz : product.mantissa = 0 := by exact_mod_cast hm
      have hpos : (0 : ℤ) < (MPIR.radix : ℤ) ^ (product.used - 1) := by
        exact pow_pos (by norm_num [MPIR.radix]) _
      rw [hz, abs_zero] at hnorm
      omega
  simp [hr, MPIR.keep, hzero, hu, hzUsed, he]

end FractalSharkProofs.ReferenceOrbit.MT9
