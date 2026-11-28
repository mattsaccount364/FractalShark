module

public import FractalSharkProofs.ReferenceOrbit.Error
public import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Rational interval arithmetic with outward rounding to a fixed dyadic grid. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

structure Parameter where
  x : ℚ
  y : ℚ

def Parameter.value (c : Parameter) : ℂ := ⟨c.x, c.y⟩
def Parameter.magnitude (c : Parameter) : ℚ := |c.x| + |c.y|

theorem Parameter.magnitude_eq (c : Parameter) :
    (c.magnitude : ℝ) = norm1 c.value := by
  simp [Parameter.magnitude, Parameter.value, norm1]

def grid (bits : ℕ) : ℚ := 1 / 2 ^ bits

theorem grid_pos (bits : ℕ) : 0 < grid bits := by
  unfold grid
  positivity

def down (bits : ℕ) (x : ℚ) : ℚ := ⌊x / grid bits⌋ * grid bits
def up (bits : ℕ) (x : ℚ) : ℚ := ⌈x / grid bits⌉ * grid bits

theorem down_le (bits : ℕ) (x : ℚ) : down bits x ≤ x :=
  (le_div_iff₀ (grid_pos bits)).mp (Int.floor_le _)

theorem le_up (bits : ℕ) (x : ℚ) : x ≤ up bits x :=
  (div_le_iff₀ (grid_pos bits)).mp (Int.le_ceil _)

structure Interval where
  lo : ℚ
  hi : ℚ

namespace Interval

def Contains (a : Interval) (x : ℝ) : Prop := (a.lo : ℝ) ≤ x ∧ x ≤ (a.hi : ℝ)

def add (a b : Interval) : Interval := ⟨a.lo + b.lo, a.hi + b.hi⟩
def sub (a b : Interval) : Interval := ⟨a.lo - b.hi, a.hi - b.lo⟩
def mul (a b : Interval) : Interval :=
  ⟨min (min (a.lo * b.lo) (a.lo * b.hi)) (min (a.hi * b.lo) (a.hi * b.hi)),
    max (max (a.lo * b.lo) (a.lo * b.hi)) (max (a.hi * b.lo) (a.hi * b.hi))⟩
def square (a : Interval) : Interval :=
  ⟨if 0 ≤ a.lo then a.lo ^ 2 else if a.hi ≤ 0 then a.hi ^ 2 else 0,
    max (a.lo ^ 2) (a.hi ^ 2)⟩
def rounded (a : Interval) (bits : ℕ) : Interval := ⟨down bits a.lo, up bits a.hi⟩
def point (x : ℚ) : Interval := ⟨x, x⟩
def magnitude (a : Interval) : ℚ := max |a.lo| |a.hi|

theorem point_contains (x : ℚ) : (point x).Contains x := ⟨le_rfl, le_rfl⟩

theorem rounded_contains {a : Interval} {x : ℝ} (hx : a.Contains x) (bits : ℕ) :
    (a.rounded bits).Contains x :=
  ⟨(show (down bits a.lo : ℝ) ≤ a.lo by exact_mod_cast down_le bits a.lo).trans hx.1,
    hx.2.trans (show (a.hi : ℝ) ≤ up bits a.hi by exact_mod_cast le_up bits a.hi)⟩

theorem add_contains {a b : Interval} {x y : ℝ} (hx : a.Contains x) (hy : b.Contains y) :
    (a.add b).Contains (x + y) := by
  simpa [Contains, add] using And.intro (add_le_add hx.1 hy.1) (add_le_add hx.2 hy.2)

theorem sub_contains {a b : Interval} {x y : ℝ} (hx : a.Contains x) (hy : b.Contains y) :
    (a.sub b).Contains (x - y) := by
  simpa [Contains, sub] using And.intro (sub_le_sub hx.1 hy.2) (sub_le_sub hx.2 hy.1)

private theorem endpoints_mul {lo hi x : ℝ} (hlo : lo ≤ x) (hhi : x ≤ hi) (y : ℝ) :
    min (lo * y) (hi * y) ≤ x * y ∧ x * y ≤ max (lo * y) (hi * y) := by
  rcases le_total 0 y with hy | hy
  · exact ⟨(min_le_left _ _).trans (mul_le_mul_of_nonneg_right hlo hy),
      (mul_le_mul_of_nonneg_right hhi hy).trans (le_max_right _ _)⟩
  · exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_right hhi hy),
      (mul_le_mul_of_nonpos_right hlo hy).trans (le_max_left _ _)⟩

theorem mul_contains {a b : Interval} {x y : ℝ} (hx : a.Contains x) (hy : b.Contains y) :
    (a.mul b).Contains (x * y) := by
  have hx' := endpoints_mul hx.1 hx.2 y
  have hlo : min ((a.lo : ℝ) * b.lo) ((a.lo : ℝ) * b.hi) ≤ (a.lo : ℝ) * y ∧
      (a.lo : ℝ) * y ≤ max ((a.lo : ℝ) * b.lo) ((a.lo : ℝ) * b.hi) := by
    simpa only [mul_comm (a.lo : ℝ)] using endpoints_mul hy.1 hy.2 (a.lo : ℝ)
  have hhi : min ((a.hi : ℝ) * b.lo) ((a.hi : ℝ) * b.hi) ≤ (a.hi : ℝ) * y ∧
      (a.hi : ℝ) * y ≤ max ((a.hi : ℝ) * b.lo) ((a.hi : ℝ) * b.hi) := by
    simpa only [mul_comm (a.hi : ℝ)] using endpoints_mul hy.1 hy.2 (a.hi : ℝ)
  simpa [Contains, mul] using
    And.intro ((min_le_min hlo.1 hhi.1).trans hx'.1) (hx'.2.trans (max_le_max hlo.2 hhi.2))

theorem abs_le_magnitude {a : Interval} {x : ℝ} (hx : a.Contains x) :
    |x| ≤ (a.magnitude : ℝ) := by
  rw [magnitude, Rat.cast_max, Rat.cast_abs, Rat.cast_abs, abs_le]
  exact ⟨(neg_le_neg (le_max_left _ _)).trans ((neg_abs_le _).trans hx.1),
    hx.2.trans ((le_abs_self _).trans (le_max_right _ _))⟩

theorem square_contains {a : Interval} {x : ℝ} (hx : a.Contains x) :
    a.square.Contains (x ^ 2) := by
  have habs := abs_le_magnitude hx
  refine ⟨?_, ?_⟩
  · dsimp only [square]
    split
    · rename_i h
      have h' : 0 ≤ (a.lo : ℝ) := by exact_mod_cast h
      simpa only [pow_two, Rat.cast_mul] using pow_le_pow_left₀ h' hx.1 2
    · split
      · rename_i h
        have h' : (a.hi : ℝ) ≤ 0 := by exact_mod_cast h
        have hs := pow_le_pow_left₀ (neg_nonneg.mpr h') (neg_le_neg hx.2) 2
        rw [neg_sq, neg_sq] at hs
        simpa only [pow_two, Rat.cast_mul] using hs
      · simpa only [Rat.cast_zero] using sq_nonneg x
  · rcases le_total |(a.lo : ℝ)| |(a.hi : ℝ)| with h | h
    · have h' : |x| ≤ |(a.hi : ℝ)| := by simpa [magnitude, max_eq_right h] using habs
      have hs := pow_le_pow_left₀ (abs_nonneg x) h' 2
      simp only [sq_abs] at hs
      exact hs.trans (by
        have hh : ((a.hi ^ (2 : ℕ) : ℚ) : ℝ) ≤
            (max (a.lo ^ (2 : ℕ)) (a.hi ^ (2 : ℕ)) : ℚ) := by
          exact_mod_cast le_max_right (a.lo ^ (2 : ℕ)) (a.hi ^ (2 : ℕ))
        simpa only [square, pow_two, Rat.cast_mul] using hh)
    · have h' : |x| ≤ |(a.lo : ℝ)| := by simpa [magnitude, max_eq_left h] using habs
      have hs := pow_le_pow_left₀ (abs_nonneg x) h' 2
      simp only [sq_abs] at hs
      exact hs.trans (by
        have hh : ((a.lo ^ (2 : ℕ) : ℚ) : ℝ) ≤
            (max (a.lo ^ (2 : ℕ)) (a.hi ^ (2 : ℕ)) : ℚ) := by
          exact_mod_cast le_max_left (a.lo ^ (2 : ℕ)) (a.hi ^ (2 : ℕ))
        simpa only [square, pow_two, Rat.cast_mul] using hh)

end Interval

structure Box where
  x : Interval
  y : Interval

namespace Box

def zero : Box := ⟨.point 0, .point 0⟩
def Contains (box : Box) (z : ℂ) : Prop := box.x.Contains z.re ∧ box.y.Contains z.im
def magnitude (box : Box) : ℚ := box.x.magnitude + box.y.magnitude

def next (box : Box) (c : Parameter) (bits : ℕ) : Box :=
  ⟨((box.x.square.sub box.y.square).add (.point c.x)).rounded bits,
    (((Interval.point 2).mul (box.x.mul box.y)).add (.point c.y)).rounded bits⟩

theorem next_contains {box : Box} {z : ℂ} (hz : box.Contains z) (c : Parameter) (bits : ℕ) :
    (box.next c bits).Contains (z ^ 2 + c.value) := by
  have hreal := Interval.rounded_contains (Interval.add_contains
    (Interval.sub_contains (Interval.square_contains hz.1) (Interval.square_contains hz.2))
    (Interval.point_contains c.x)) bits
  have himag := Interval.rounded_contains (Interval.add_contains
    (Interval.mul_contains (Interval.point_contains 2) (Interval.mul_contains hz.1 hz.2))
    (Interval.point_contains c.y)) bits
  simp only [Contains, next, Parameter.value, Complex.add_re, Complex.add_im, pow_two, Complex.mul_re,
    Complex.mul_im]
  refine ⟨by simpa only [pow_two] using hreal, ?_⟩
  rw [show z.re * z.im + z.im * z.re + (c.y : ℝ) =
    ((2 : ℚ) : ℝ) * (z.re * z.im) + (c.y : ℝ) by push_cast; ring]
  exact himag

theorem norm_le_magnitude {box : Box} {z : ℂ} (hz : box.Contains z) :
    norm1 z ≤ (box.magnitude : ℝ) := by
  simpa [magnitude, norm1] using
    add_le_add (Interval.abs_le_magnitude hz.1) (Interval.abs_le_magnitude hz.2)

end Box

/-- Retain only the magnitude caps and the latest box. Width guards trigger
refinement before interval wrapping makes the bounds unhelpful. -/
def enclose (c : Parameter) (bits : ℕ) (magnitudeLimit : ℚ) :
    ℕ → Option (Box × Array ℚ)
  | 0 => some (.zero, #[])
  | n + 1 =>
    match enclose c bits magnitudeLimit n with
    | none => none
    | some (box, caps) =>
      if box.x.hi - box.x.lo ≤ 1 ∧ box.y.hi - box.y.lo ≤ 1 ∧ box.magnitude ≤ magnitudeLimit then
        some (box.next c bits, caps.push box.magnitude)
      else none

theorem enclose_sound (c : Parameter) (bits : ℕ) (limit : ℚ) (n : ℕ)
    (box : Box) (caps : Array ℚ) (h : enclose c bits limit n = some (box, caps)) :
    box.Contains (orbit c.value n) ∧ caps.size = n ∧
      ∀ i, i < n → norm1 (orbit c.value i) ≤ ((caps[i]?.getD 0 : ℚ) : ℝ) := by
  induction n generalizing box caps with
  | zero =>
    simp only [enclose, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨?_, rfl, fun i hi => by omega⟩
    simp [Box.zero, Box.Contains, Interval.Contains, Interval.point, orbit]
  | succ n ih =>
    cases hrun : enclose c bits limit n with
    | none => simp [enclose, hrun] at h
    | some pair =>
      obtain ⟨old, prev⟩ := pair
      have ⟨hcontains, hsize, hcap⟩ := ih old prev hrun
      simp only [enclose, hrun] at h
      split at h
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨old.next_contains hcontains c bits, by simp [hsize], ?_⟩
        intro i hi
        rw [Array.getElem?_push, hsize]
        by_cases he : i = n
        · subst i
          simpa using old.norm_le_magnitude hcontains
        · simpa [he] using hcap i (by omega)
      · contradiction

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate
