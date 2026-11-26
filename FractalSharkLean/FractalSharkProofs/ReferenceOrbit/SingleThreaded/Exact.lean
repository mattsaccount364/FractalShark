module

public import FractalSharkProofs.ReferenceOrbit.Basic
import Mathlib.Tactic.Ring

/-! Exact arithmetic and sample indexing of the serial reference orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.SingleThreaded

/-- Both coordinates in the factored CPU update read the incoming state. -/
def exactStep (c z : ℂ) : ℂ :=
  let x := z.re
  let y := z.im
  ⟨(x + y) * (x - y) + c.re, (2 * x) * y + c.im⟩

theorem exactStep_eq (c z : ℂ) : exactStep c z = z ^ 2 + c := by
  apply Complex.ext <;>
    simp only [exactStep, pow_two, Complex.add_re, Complex.add_im,
      Complex.mul_re, Complex.mul_im] <;> ring

/-- The first state observed by the loop already incorporates the first update. -/
def exactLoop (c : ℂ) : ℕ → ℂ
  | 0 => c
  | i + 1 => exactStep c (exactLoop c i)

theorem exactLoop_eq (c : ℂ) (i : ℕ) : exactLoop c i = orbit c (i + 1) := by
  induction i with
  | zero => simp [exactLoop, orbit]
  | succ i ih => simp only [exactLoop, exactStep_eq, ih, orbit]

/-- A zero entry followed by the old-state observations of the executed bodies. -/
def exactPrefix (c : ℂ) (n : ℕ) : List ℂ :=
  0 :: List.ofFn (fun i : Fin n => exactLoop c i.val)

theorem exactPrefix_eq (c : ℂ) (n : ℕ) :
    exactPrefix c n = List.ofFn (fun i : Fin (n + 1) => orbit c i.val) := by
  rw [List.ofFn_succ]
  simp [exactPrefix, orbit, exactLoop_eq, Fin.succ]

end FractalSharkProofs.ReferenceOrbit.SingleThreaded
