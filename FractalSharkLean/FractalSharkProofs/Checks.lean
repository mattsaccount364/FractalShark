module

import FractalSharkProofs
import FractalSharkProofs.ReferenceOrbit.Checks
import FractalSharkProofs.ReferenceOrbit.MT9.Checks
import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Checks
import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.NormNum

/-! Check the sample indexing, limb boundary cases, and public theorem dependencies. -/

namespace FractalSharkProofs.ReferenceOrbit.SingleThreaded

example : exactPrefix (1 : ℂ) 0 = [0] := by simp [exactPrefix]

example : exactPrefix (-1 : ℂ) 4 = [0, -1, 0, -1, 0] := by
  rw [exactPrefix_eq]
  simp [List.ofFn_succ, Fin.succ, orbit]

example : exactLoop (⟨1, 1⟩ : ℂ) 2 = ⟨-7, 7⟩ := by
  apply Complex.ext <;> norm_num [exactLoop, exactStep]

#print axioms exactStep_eq
#print axioms exactPrefix_eq
#print axioms parameterCopy_error
#print axioms step_error
#print axioms prefix_error_le

end FractalSharkProofs.ReferenceOrbit.SingleThreaded

namespace FractalSharkProofs.MPIR

private def p : Precision := Precision.ofRequestedBits 64

private def positive (m : ℤ) (used : ℕ) (top : ℤ) (hused : 0 < used)
    (lo : (radix : ℤ) ^ (used - 1) ≤ m) (hi : m < (radix : ℤ) ^ used) : Limbs where
  mantissa := m
  used := used
  top := top
  normalized := by
    have hm := (pow_nonneg (show (0 : ℤ) ≤ radix by norm_num [radix]) (used - 1)).trans lo
    exact Or.inr ⟨hused, by simpa [abs_of_nonneg hm] using lo,
      by simpa [abs_of_nonneg hm] using hi⟩

private def one : Limbs := positive 1 1 1 (by decide) (by decide) (by decide)

private def almostRadix : Limbs :=
  positive (radix - 1) 1 1 (by decide) (by decide) (by decide)

private def carry : Limbs := positive radix 2 2 (by decide) (by decide) (by decide)

private def long : Limbs :=
  positive ((radix : ℤ) ^ 2 + 3) 3 3 (by decide) (by decide) (by decide)

private def low : Limbs := positive 3 1 1 (by decide) (by decide) (by decide)

private def radixSquared : Limbs := positive 1 1 3 (by decide) (by decide) (by decide)

private def radixFourth : Limbs := positive 1 1 5 (by decide) (by decide) (by decide)

private def twiceRadixSquared : Limbs := positive 2 1 3 (by decide) (by decide) (by decide)

private def highDiscardedBit : Limbs :=
  positive ((radix : ℤ) ^ 2 + (radix : ℤ) / 2) 3 3
    (by decide) (by decide) (by decide)

private def doubledWithCarry : Limbs :=
  positive (2 * (radix : ℤ) + 1) 2 3 (by decide) (by decide) (by decide)

example : p.bits = 64 := by decide

-- A carry may occupy the extra limb of the destination.
example : Add p almostRadix one carry := by
  apply Add.sameSign
  · norm_num [almostRadix, one, positive, Limbs.value, grid, radix]
  · norm_num [almostRadix, one, carry, positive, Limbs.value, maximumTop, p,
      Precision.ofRequestedBits, truncate, grid, radix]

-- The matched high limbs cancel, leaving the unrounded low tail.
example : Sub p long radixSquared low := by
  apply Add.remainder (top := 2) (cutoff := -1)
  · norm_num [long, radixSquared, positive, Limbs.value, Limbs.neg, grid, radix]
  · decide
  · decide
  · norm_num [prefixSum, leadingPart, truncate, long, radixSquared, positive,
      Limbs.value, Limbs.neg, grid, radix]
  · right
    norm_num [tail, leadingPart, truncate, long, radixSquared, positive, Limbs.value, Limbs.neg,
      grid, radix]
  · norm_num [retainedSum, prefixSum, tail, leadingPart, truncate, long, radixSquared, low,
      positive, Limbs.value, Limbs.neg, grid, radix]

private def highPlusOne : Limbs :=
  positive ((radix : ℤ) ^ 2 + 1) 3 3 (by decide) (by decide) (by decide)

private def highMinusOne : Limbs :=
  positive ((radix : ℤ) ^ 2 - 1) 2 2 (by decide) (by decide) (by decide)

private def two : Limbs := positive 2 1 1 (by decide) (by decide) (by decide)

-- Subtraction borrows across the zero and all-ones limb at the exponent boundary.
example : Sub p highPlusOne highMinusOne two := by
  apply Add.borrow (top := 1)
  · norm_num [highPlusOne, highMinusOne, positive, Limbs.value, Limbs.neg, grid, radix]
  · decide
  · norm_num [prefixSum, leadingPart, truncate, highPlusOne, highMinusOne, positive,
      Limbs.value, Limbs.neg, grid, radix]
  · norm_num [retainedSum, prefixSum, tail, leadingPart, truncate, highPlusOne, highMinusOne,
      two, positive, p, Precision.ofRequestedBits, Limbs.value, Limbs.neg, grid, radix]

-- The low limb is discarded from both multiplication operands and from the shift input.
example : Mul p long long radixFourth := by
  refine ⟨radixFourth, ?_, ?_⟩
  · norm_num [keep, p, Precision.ofRequestedBits, long, radixFourth, positive,
      truncate, Limbs.value, grid, radix]
  · simp [keep, p, Precision.ofRequestedBits, radixFourth, positive]

example : Scale p 1 long twiceRadixSquared := by
  norm_num [Scale, keep, p, Precision.ofRequestedBits, long, twiceRadixSquared,
    positive, truncate, Limbs.value, grid, radix]

-- mpf_mul_ui preserves a carry from the discarded limb, unlike the shift above.
example : MulUiTwo p highDiscardedBit doubledWithCarry := by
  norm_num [MulUiTwo, p, Precision.ofRequestedBits, highDiscardedBit, doubledWithCarry,
    positive, truncate, Limbs.value, grid, radix]

-- Starting at c = 1 + i, the first rounded body publishes 1 + 3i.
example :
    let oneSlot : ReferenceOrbit.Slot p := ⟨one, by decide⟩
    let initial : ReferenceOrbit.State p := ⟨oneSlot, oneSlot⟩
    ReferenceOrbit.SingleThreaded.Step initial initial ⟨oneSlot, ⟨low, by decide⟩⟩ := by
  dsimp only
  let twoSlot : ReferenceOrbit.Slot p := ⟨two, by decide⟩
  let zeroSlot : ReferenceOrbit.Slot p := ⟨Limbs.zero, by decide⟩
  refine ⟨⟨twoSlot, twoSlot, twoSlot, zeroSlot, zeroSlot, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩
  · norm_num [Scale, keep, p, Precision.ofRequestedBits, one, two, twoSlot, positive,
      Limbs.value, grid, radix]
  · refine ⟨two, ?_, ?_⟩ <;>
      norm_num [keep, p, Precision.ofRequestedBits, one, two, twoSlot, positive,
        Limbs.value, grid, radix]
  · apply Add.sameSign <;>
      norm_num [maximumTop, p, Precision.ofRequestedBits, one, two, twoSlot, low, positive,
        truncate, Limbs.value, grid, radix]
  · apply Add.sameSign <;>
      norm_num [maximumTop, p, Precision.ofRequestedBits, one, two, twoSlot, positive,
        truncate, Limbs.value, grid, radix]
  · apply Add.remainder (top := 0) (cutoff := -3)
    · norm_num [one, positive, Limbs.value, Limbs.neg, grid, radix]
    · decide
    · decide
    · norm_num [prefixSum, leadingPart, truncate, one, positive, Limbs.value, Limbs.neg, grid, radix]
    · left
      norm_num [tail, leadingPart, truncate, one, positive, Limbs.value, grid, radix]
    · norm_num [retainedSum, prefixSum, tail, leadingPart, truncate, one, zeroSlot, positive,
        Limbs.value, Limbs.zero, Limbs.neg, grid, radix]
  · refine ⟨Limbs.zero, ?_, ?_⟩ <;>
      simp [keep, zeroSlot, twoSlot, two, positive, p, Precision.ofRequestedBits,
        Limbs.zero, Limbs.value]
  · exact Add.zeroLeft (Limbs.value_zero) (Or.inl rfl)

#print axioms Precision.errorUnit_eq
#print axioms set_error
#print axioms set_same_precision
#print axioms scale_bounds
#print axioms mulUiTwo_bounds
#print axioms mul_bounds
#print axioms add_error
#print axioms sub_error

end FractalSharkProofs.MPIR
