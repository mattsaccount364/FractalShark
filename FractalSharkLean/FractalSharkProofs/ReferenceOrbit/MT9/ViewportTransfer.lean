module

public import FractalSharkProofs.ReferenceOrbit.MT9.Dynamic
public import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Select

/-! Transfer of the calculator's fixed-MT9 error budget to an observed dynamic prefix. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

open PrecisionEstimate in
/-- The fixed trajectory serves as a mathematical comparison. The numerical
check needs the calculator's per-index bounds and the dynamic drift enclosure. -/
theorem viewport_of_recommendation {r : Request} (profile : Profile r) {f : ℚ}
    {maxBits last : ℕ} (setting : Recommendation r f maxBits) (hf : profile.referenceFactor ≤ f)
    (_hbackend : r.backend = .mt9)
    (fixed : MPIRViewportPrefix setting.setting.precision r.backend.coefficient last)
    (hwork : fixed.working = setting.anchor) (hlast : last ≤ r.last)
    (exit : LoopExit) (dynamic : ℕ → ℂ) (drift : ℕ → ℝ)
    (hzero : dynamic 0 = 0) (hfirst : dynamic 1 = fixed.state 1) (hbase : 0 ≤ drift 1)
    (hstep : ∀ i, 1 ≤ i → i < last →
      DiskStep (dynamic i) (fixed.state i) (dynamic (i + 1)) (fixed.state (i + 1))
        fixed.working (drift i) (drift (i + 1)))
    (hcovered : ∀ n ∈ exit.samples .mt9, n ≤ last)
    (hbudget : ∀ n ∈ exit.samples .mt9, 1 ≤ n →
      (profile.referenceErrorAt setting.setting.precision.bits n : ℝ) + 2 * drift n ≤ r.tolerance) :
    ∀ n ∈ exit.samples .mt9,
      r.sampling.viewport.sampleError (dynamic n - orbit r.center.value n) ≤ r.fraction := by
  intro n hn
  cases n with
  | zero => simpa [hzero, orbit, Viewport.sampleError] using r.fraction_pos.le
  | succ n =>
    have hi : 1 ≤ n + 1 := Nat.succ_le_succ (Nat.zero_le n)
    have hcovered' := hcovered _ hn
    have he := disk_prefix dynamic fixed.state fixed.working drift last hfirst hbase hstep
      _ hi hcovered'
    have hfixed := (setting.reference_error_at profile hf fixed hwork hlast _ hcovered').1
    have hfit := hbudget _ hn hi
    change _ ≤ (r.sampling.tolerance r.fraction : ℝ) at hfit
    rw [r.sampling.tolerance_eq] at hfit
    exact dynamic_viewport_bound _ _ _ _ _ _ _ he hfixed hfit

/-- Even an arbitrarily small positive drift can point out of a saturated
viewport budget. These points illustrate the error-bound hypotheses. -/
theorem fixed_sufficiency_alone (v : Viewport) {fraction E : ℝ}
    (hf : 0 ≤ fraction) (hE : 0 < E) :
    ∃ fixed dynamic : ℂ, v.sampleError fixed ≤ fraction ∧ ‖dynamic - fixed‖ ≤ E ∧
      fraction < v.sampleError dynamic := by
  let x := fraction * v.pitchX
  have hx : 0 ≤ x := mul_nonneg hf v.pitchX_pos.le
  refine ⟨(x : ℂ), ((x + E : ℝ) : ℂ), ?_, ?_, ?_⟩
  · simp only [Viewport.sampleError, Complex.ofReal_re, Complex.ofReal_im, abs_zero, zero_div,
      add_zero, abs_of_nonneg hx, x]
    rw [mul_div_cancel_right₀ _ v.pitchX_pos.ne']
  · rw [← Complex.ofReal_sub, add_sub_cancel_left, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hE]
  · simp only [Viewport.sampleError, Complex.ofReal_re, Complex.ofReal_im, abs_zero, zero_div,
      add_zero, abs_of_pos (add_pos_of_nonneg_of_pos hx hE)]
    exact (lt_div_iff₀ v.pitchX_pos).mpr (lt_add_of_pos_right x hE)

end FractalSharkProofs.ReferenceOrbit.MT9
