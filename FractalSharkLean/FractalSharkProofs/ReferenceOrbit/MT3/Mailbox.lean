module

/-! Happens-before edges of an authoritative square worker's C++ mailbox.
Events model a first job and the next reuse of its memory. Atomic synchronization
edges apply to matching successful claims that read the publication in question.
-/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT3.Mailbox

inductive Event where
  | writeInput | publishInput | claimInput | readInput
  | writeResult | publishResult | claimResult | readResult
  | overwrite | publishNext | claimNext | readNext

inductive Sequenced : Event → Event → Prop where
  | inputPrepared : Sequenced .writeInput .publishInput
  | inputClaimed : Sequenced .claimInput .readInput
  | workerComputed : Sequenced .readInput .writeResult
  | resultPrepared : Sequenced .writeResult .publishResult
  | workerWaits : Sequenced .publishResult .claimNext
  | nextClaimed : Sequenced .claimNext .readNext
  | waitForResult : Sequenced .publishInput .claimResult
  | resultClaimed : Sequenced .claimResult .readResult
  | resultSaved : Sequenced .readResult .overwrite
  | nextPrepared : Sequenced .overwrite .publishNext

structure Orders where
  firstInputRelease : Bool
  resultRelease : Bool
  resultAcquire : Bool

def original : Orders := ⟨false, false, false⟩

def corrected : Orders := ⟨true, true, true⟩

/-- The worker's input claim is acquire in both protocols. The next input
publication is release in both, including the Y lookahead publication. -/
inductive HappensBefore (orders : Orders) : Event → Event → Prop where
  | sequenced {a b} : Sequenced a b → HappensBefore orders a b
  | first : orders.firstInputRelease = true → HappensBefore orders .publishInput .claimInput
  | result : orders.resultRelease = true → orders.resultAcquire = true →
      HappensBefore orders .publishResult .claimResult
  | next : HappensBefore orders .publishNext .claimNext
  | trans {a b c} : HappensBefore orders a b → HappensBefore orders b c → HappensBefore orders a c

private def earlyWorker : Event → Prop
  | .claimInput | .readInput | .writeResult | .publishResult => True
  | _ => False

private def worker : Event → Prop
  | .claimInput | .readInput | .writeResult | .publishResult | .claimNext | .readNext => True
  | _ => False

private theorem original_earlyWorker_predecessor {a b : Event}
    (h : HappensBefore original a b) : earlyWorker b → earlyWorker a := by
  induction h with
  | sequenced h => cases h <;> simp [earlyWorker]
  | first h => cases h
  | result h _ => cases h
  | next => simp [earlyWorker]
  | trans _ _ ih₁ ih₂ => exact fun hb => ih₁ (ih₂ hb)

private theorem original_worker_successor {a b : Event}
    (h : HappensBefore original a b) : worker a → worker b := by
  induction h with
  | sequenced h => cases h <;> simp [worker]
  | first h => cases h
  | result h _ => cases h
  | next => simp [worker]
  | trans _ _ ih₁ ih₂ => exact fun ha => ih₂ (ih₁ ha)

/-- The old graph leaves both conflicting access pairs unordered in either
direction, even with the release/acquire handoff for the next input. -/
theorem original_unordered :
    ¬ HappensBefore original .writeInput .readInput ∧
    ¬ HappensBefore original .readInput .writeInput ∧
    ¬ HappensBefore original .writeResult .readResult ∧
    ¬ HappensBefore original .readResult .writeResult :=
  ⟨fun h => original_earlyWorker_predecessor h trivial,
   fun h => original_worker_successor h trivial,
   fun h => original_worker_successor h trivial,
   fun h => original_earlyWorker_predecessor h trivial⟩

theorem corrected_payload_visibility :
    HappensBefore corrected .writeInput .readInput ∧
    HappensBefore corrected .writeResult .readResult :=
  ⟨HappensBefore.trans (.sequenced .inputPrepared)
      (.trans (.first rfl) (.sequenced .inputClaimed)),
   HappensBefore.trans (.sequenced .resultPrepared)
      (.trans (.result rfl rfl) (.sequenced .resultClaimed))⟩

theorem corrected_buffer_reuse :
    HappensBefore corrected .readInput .overwrite ∧
    HappensBefore corrected .readResult .readNext :=
  ⟨HappensBefore.trans (.sequenced .workerComputed)
      (.trans corrected_payload_visibility.2 (.sequenced .resultSaved)),
   HappensBefore.trans (.sequenced .resultSaved)
      (.trans (.sequenced .nextPrepared) (.trans .next (.sequenced .nextClaimed)))⟩

end FractalSharkProofs.ReferenceOrbit.MT3.Mailbox
