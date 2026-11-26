module

public import FractalSharkProofs.ReferenceOrbit.MPIRArithmetic

/-! Logical ownership of the two authoritative square jobs and their snapshots. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT3

structure Job (p : MPIR.Precision) where
  epoch : ℕ
  operand : Slot p

/-- The result belongs to the immutable snapshot handed to the worker. -/
structure Square {p : MPIR.Precision} (job : Job p) where
  value : Slot p
  squared : MPIR.Mul p job.operand.val job.operand.val value.val

inductive Worker (p : MPIR.Precision) where
  | idle
  | queued (job : Job p)
  | running (job : Job p) (stopPending : Bool)
  | ready (job : Job p) (result : Square job) (stopPending : Bool)
  | stopQueued
  | joined

/-- Each transition moves ownership of one job slot. The stop sentinel may be
posted while a claimed job runs. The worker then publishes its square before
exiting, and that last result can be discarded after the join. -/
inductive Handoff {p : MPIR.Precision} : Worker p → Worker p → Prop where
  | publish (j : Job p) : Handoff .idle (.queued j)
  | claim (j : Job p) : Handoff (.queued j) (.running j false)
  | finish {j : Job p} (r : Square j) (stop : Bool) :
      Handoff (.running j stop) (.ready j r stop)
  | consume {j : Job p} (r : Square j) : Handoff (.ready j r false) .idle
  | requestIdleStop : Handoff .idle .stopQueued
  | requestRunningStop (j : Job p) : Handoff (.running j false) (.running j true)
  | requestReadyStop {j : Job p} (r : Square j) :
      Handoff (.ready j r false) (.ready j r true)
  | joinIdle : Handoff .stopQueued .joined
  | joinReady {j : Job p} (r : Square j) : Handoff (.ready j r true) .joined

def Worker.active {p : MPIR.Precision} : Worker p → Option (Job p)
  | .queued j | .running j _ | .ready j _ _ => some j
  | _ => none

/-- Once claimed, a snapshot cannot be replaced by a different snapshot until
the slot is consumed. In particular finishing a square preserves its epoch. -/
theorem active_stable {p : MPIR.Precision} {a b : Worker p} {j k : Job p}
    (h : Handoff a b) (ha : a.active = some j) (hb : b.active = some k) :
    j = k := by
  cases h <;> simp_all [Worker.active]

theorem drain_running_lookahead {p : MPIR.Precision} {j : Job p} (r : Square j) :
    Handoff (.running j false) (.running j true) ∧
    Handoff (.running j true) (.ready j r true) ∧
    Handoff (.ready j r true) .joined :=
  ⟨Handoff.requestRunningStop j, Handoff.finish r true, Handoff.joinReady r⟩

/-- While In holds the lookahead job, stop publication waits for its claim. -/
theorem drain_queued_lookahead {p : MPIR.Precision} {j : Job p} (r : Square j) :
    Handoff (.queued j) (.running j false) ∧
    Handoff (.running j false) (.running j true) ∧
    Handoff (.running j true) (.ready j r true) ∧
    Handoff (.ready j r true) .joined :=
  ⟨Handoff.claim j, drain_running_lookahead r⟩

theorem joined_is_terminal {p : MPIR.Precision} {w : Worker p} :
    ¬ Handoff .joined w := by intro h; cases h

/-- X and Y have distinct mailboxes. A step on one leaves the other's state
available to the caller, irrespective of which completion is observed first. -/
inductive WorkerPairStep {p : MPIR.Precision} : Worker p × Worker p → Worker p × Worker p → Prop where
  | x {a a' : Worker p} (b : Worker p) : Handoff a a' → WorkerPairStep (a, b) (a', b)
  | y {b b' : Worker p} (a : Worker p) : Handoff b b' → WorkerPairStep (a, b) (a, b')

theorem either_completion_order {p : MPIR.Precision} {x y : Job p} (rx : Square x) (ry : Square y) :
    WorkerPairStep (.running x false, .running y false) (.ready x rx false, .running y false) ∧
    WorkerPairStep (.ready x rx false, .running y false) (.ready x rx false, .ready y ry false) ∧
    WorkerPairStep (.running x false, .running y false) (.running x false, .ready y ry false) ∧
    WorkerPairStep (.running x false, .ready y ry false) (.ready x rx false, .ready y ry false) :=
  ⟨.x _ (.finish rx false), .y _ (.finish ry false), .y _ (.finish ry false), .x _ (.finish rx false)⟩

def xJob {p : MPIR.Precision} (n : ℕ) (old : State p) : Job p := ⟨n, old.x⟩

def yJob {p : MPIR.Precision} (n : ℕ) (old : State p) : Job p := ⟨n, old.y⟩

/-- The caller can start this job as soon as the next imaginary value is
formed. The next real value is immaterial to this dispatch. -/
def yLookahead {p : MPIR.Precision} (n : ℕ) (nextY : Slot p) : Job p := ⟨n + 1, nextY⟩

theorem lookahead_matches_next {p : MPIR.Precision} (n : ℕ) (next : State p) :
    yLookahead n next.y = yJob (n + 1) next := rfl

/-- Save the current Y square before recycling its worker for lookahead. The
saved result remains indexed by the old epoch and old imaginary operand. -/
theorem consume_then_lookahead {p : MPIR.Precision} {n : ℕ} {old : State p}
    (saved : Square (yJob n old)) (nextY : Slot p) :
    Handoff (.ready (yJob n old) saved false) .idle ∧
    Handoff .idle (.queued (yLookahead n nextY)) :=
  ⟨Handoff.consume saved, Handoff.publish _⟩

end FractalSharkProofs.ReferenceOrbit.MT3
