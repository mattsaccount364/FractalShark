module

import FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.Select
import Lean.Data.Json

/-! JSON interface for the standalone CPU orbit precision calculator. -/

open Lean

namespace FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

private def decimal (value : Json) : Except String Rat := do
  let number ← match value with
    | .str s => (← Json.parse s).getNum?
    | _ => value.getNum?
  return (number.mantissa : Rat) / 10 ^ number.exponent

private def decimalField (json : Json) (key : String) : Except String Rat :=
  ((json.getObjVal? key).bind decimal).mapError fun msg => s!"{key}: {msg}"

private def optionalDecimal (json : Json) (key : String) (fallback : Rat) : Except String Rat :=
  match json.getObjVal? key with
  | .ok val => (decimal val).mapError fun msg => s!"{key}: {msg}"
  | .error _ => .ok fallback

private def natField (json : Json) (key : String) : Except String Nat :=
  ((json.getObjVal? key).bind Json.getNat?).mapError fun msg => s!"{key}: {msg}"

private def readSampling (json : Json) (center : Parameter) : Except String Sampling := do
  let width ← natField json "width"
  let height ← natField json "height"
  let aa ← natField json "antialiasing"
  if hSize : 0 < width ∧ 0 < height ∧ 0 < aa then
    let ⟨hWidth, hHeight, hAA⟩ := hSize
    match json.getObjVal? "zoomFactor" with
    | .ok _ =>
      if ["minX", "maxX", "minY", "maxY"].any (fun key => (json.getObjVal? key).isOk) then
        throw "zoomFactor cannot be combined with explicit bounds"
      let zoom ← decimalField json "zoomFactor"
      if hZoom : 0 < zoom then
        return Sampling.ofZoom center zoom width height aa hZoom hWidth hHeight hAA
      else
        throw "zoomFactor must be positive"
    | .error _ =>
      let xMin ← decimalField json "minX"
      let xMax ← decimalField json "maxX"
      let yMin ← decimalField json "minY"
      let yMax ← decimalField json "maxY"
      if hBounds : xMin < xMax ∧ yMin < yMax then
        return ⟨xMin, xMax, yMin, yMax, width, height, aa,
          hBounds.1, hBounds.2, hWidth, hHeight, hAA⟩
      else
        throw "viewport spans must be positive"
  else
    throw "width, height, and antialiasing must be positive"

private def readRequest (json : Json) : Except String (Request × Nat) := do
  let center : Parameter := ⟨← decimalField json "centerX", ← decimalField json "centerY"⟩
  let sampling ← readSampling json center
  let last ← natField json "lastIndex"
  let frac ← optionalDecimal json "sampleFraction" (1 / 4)
  let radius ← optionalDecimal json "bailoutRadius" 16
  let backend ← match (← (← json.getObjVal? "backend").getStr?).toLower with
    | "st" => pure CPU.st
    | "mt3" => pure CPU.mt3
    | "mt9" => pure CPU.mt9
    | _ => throw "backend must be ST, MT3, or MT9"
  let maxBits ← match json.getObjVal? "maxBits" with
    | .ok val => val.getNat?
    | .error _ => pure defaultMaxBits
  unless 64 ≤ maxBits ∧ maxBits ≤ maxMPIRBits do
    throw s!"maxBits must be between 64 and {maxMPIRBits}"
  if h : 0 < frac ∧ frac ≤ 1 ∧ 0 < radius then
    return (⟨center, sampling, frac, h.1, backend, last, radius, h.2.2⟩, maxBits)
  else
    throw "bailoutRadius must be positive and sampleFraction must be in (0, 1]"

private def encoding (limbs : MPIR.Limbs) : Json :=
  Json.mkObj [("mantissa", toJson (toString limbs.mantissa)),
    ("binaryExponent", toJson (64 * (limbs.top - (limbs.used : Int))))]

private def failureName : Failure → String
  | .enclosureLimit => "enclosureLimit"
  | .precisionLimit => "precisionLimit"
  | .coordinateEncoding => "coordinateEncoding"

private def errorReport (failure : Failure) : Json :=
  Json.mkObj [("status", toJson (failureName failure))]

private def settingReport {r : Request} {factor : Rat} {maxBits : Nat} :
    Except Failure (Recommendation r factor maxBits) → Json
  | .error failure => errorReport failure
  | .ok result =>
    Json.mkObj [("status", toJson "certified"),
      ("requestedBits", toJson result.setting.requestedBits),
      ("effectiveBits", toJson result.setting.precision.bits),
      ("internalLimbs", toJson result.setting.precision.limbs),
      ("centerX", encoding result.x.limbs), ("centerY", encoding result.y.limbs)]

private def evaluate (json : Json) : Except String (Json × UInt32) := do
  let (request, maxBits) ← readRequest json
  match estimate request maxBits with
  | .error failure =>
    return (errorReport failure, 2)
  | .ok result =>
    let status := match result.combined with
      | .ok _ => 0
      | .error _ => 2
    return (Json.mkObj [
      ("lastIndex", toJson request.last),
      ("sampleTolerance", toJson (toString request.tolerance)),
      ("enclosureBits", toJson result.profile.enclosureBits),
      ("reference", settingReport result.reference),
      ("idealPerturbation", settingReport result.ideal),
      ("combined", settingReport result.combined)], status)

def runCalculator (args : List String) : IO UInt32 := do
  match args with
  | ["--help"] =>
    IO.println "Usage: lake exe orbit-precision input.json (use - to read JSON from stdin)"
    IO.println "Supply centerX and centerY with either zoomFactor or minX, maxX, minY, and maxY."
    return 0
  | [file] =>
    let source ← if file == "-" then (← IO.getStdin).readToEnd else IO.FS.readFile file
    match (Json.parse source).bind evaluate with
    | .ok (report, code) =>
      IO.println report.pretty
      return code
    | .error error =>
      IO.eprintln s!"Invalid input: {error}"
      return 1
  | _ =>
    IO.eprintln "Usage: lake exe orbit-precision input.json"
    return 1

end FractalSharkProofs.ReferenceOrbit.PrecisionEstimate

public def main (args : List String) : IO UInt32 :=
  FractalSharkProofs.ReferenceOrbit.PrecisionEstimate.runCalculator args
