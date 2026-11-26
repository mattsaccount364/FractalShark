module

public import Mathlib.Basic.Complex.Basic

/-! The common exact reference orbit. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit

/-- The critical orbit of the quadratic map, indexed from the critical point. -/
def orbit (c : ℂ) : ℕ → ℂ
  | 0 => 0
  | n + 1 => orbit c n ^ 2 + c

end FractalSharkProofs.ReferenceOrbit
