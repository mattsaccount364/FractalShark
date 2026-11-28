module

public import FractalSharkProofs.ReferenceOrbit.MT9.Karatsuba

/-! One MT9 batch under release/acquire synchronization and atomic write/read coherence. -/

@[expose] public section

namespace FractalSharkProofs.ReferenceOrbit.MT9

inductive Job where
  | xHigh | xLow | xSum | yHigh | yLow | ySum | xyHigh | xyLow | xySum
  deriving DecidableEq

/-- The caller leads the cross product. The other eight jobs have persistent workers. -/
def Job.followers : Job → List Job
  | .xHigh => [.xLow, .xSum]
  | .yHigh => [.yLow, .ySum]
  | .xyHigh => [.xyLow, .xySum]
  | _ => []

inductive Event where
  | prepare (batch : ℕ)
  | dispatch (batch : ℕ) (job : Job)
  | acquireRun (batch : ℕ) (job : Job)
  | multiply (batch : ℕ) (job : Job)
  | wait (batch : ℕ) (lead follower : Job)
  | assemble (batch : ℕ) (lead : Job)
  | releaseIdle (batch : ℕ) (job : Job)
  | collect (batch : ℕ) (lead : Job)
  | store (batch : ℕ)
  | shutdown (batch : ℕ)

open Event

/-- Release/Acquire edges represent reads of the current batch's command or
completion. Sequenced-before edges also order aliasing stores and buffer reuse. -/
inductive HappensBefore : Event → Event → Prop where
  | prepare_dispatch {n : ℕ} {j : Job} : HappensBefore (.prepare n) (.dispatch n j)
  | followers_first {n : ℕ} {lead follower : Job} (h : follower ∈ lead.followers) :
      HappensBefore (.dispatch n follower) (.dispatch n lead)
  | run_sync {n : ℕ} {j : Job} : HappensBefore (.dispatch n j) (.acquireRun n j)
  | read_snapshot {n : ℕ} {j : Job} : HappensBefore (.acquireRun n j) (.multiply n j)
  | wait_after_start {n : ℕ} {lead follower : Job} (h : follower ∈ lead.followers) :
      HappensBefore (.acquireRun n lead) (.wait n lead follower)
  | follower_finishes {n : ℕ} {lead follower : Job} (h : follower ∈ lead.followers) :
      HappensBefore (.multiply n follower) (.releaseIdle n follower)
  | idle_sync {n : ℕ} {lead follower : Job} (h : follower ∈ lead.followers) :
      HappensBefore (.releaseIdle n follower) (.wait n lead follower)
  | wait_then_assemble {n : ℕ} {lead follower : Job} (h : follower ∈ lead.followers) :
      HappensBefore (.wait n lead follower) (.assemble n lead)
  | lead_product {n : ℕ} {lead : Job} (h : lead.followers ≠ []) :
      HappensBefore (.multiply n lead) (.assemble n lead)
  | assembled_idle {n : ℕ} {lead : Job} (h : lead.followers ≠ []) :
      HappensBefore (.assemble n lead) (.releaseIdle n lead)
  | caller_assembles {n : ℕ} : HappensBefore (.assemble n .xyHigh) (.collect n .xyHigh)
  | lead_idle_sync {n : ℕ} {lead : Job} (h : lead = .xHigh ∨ lead = .yHigh) :
      HappensBefore (.releaseIdle n lead) (.collect n lead)
  | all_before_store {n : ℕ} {lead : Job} (h : lead.followers ≠ []) :
      HappensBefore (.collect n lead) (.store n)
  | next_batch {n : ℕ} : HappensBefore (.store n) (.prepare (n + 1))
  | stop_after_batch {n : ℕ} : HappensBefore (.store n) (.shutdown n)
  | trans {a b c : Event} : HappensBefore a b → HappensBefore b c → HappensBefore a c

theorem snapshot_visible (batch : ℕ) (job : Job) :
    HappensBefore (.prepare batch) (.multiply batch job) :=
  .trans .prepare_dispatch (.trans .run_sync .read_snapshot)

theorem follower_run_before_wait {batch : ℕ} {lead follower : Job}
    (h : follower ∈ lead.followers) :
    HappensBefore (.dispatch batch follower) (.wait batch lead follower) :=
  .trans (.followers_first h) (.trans .run_sync (.wait_after_start h))

/-- A slot's modification order is initial Idle, Run 0, Idle 0, Run 1,
Idle 1, and so on. Write/read coherence supplies the lower inequality when
Run happens before the wait, and one outstanding batch supplies the upper. -/
theorem current_idle {batch completed : ℕ}
    (coherence : 2 * batch + 1 ≤ 2 * (completed + 1))
    (noFutureRun : completed ≤ batch) : completed = batch := by omega

theorem follower_visible {batch : ℕ} {lead follower : Job}
    (h : follower ∈ lead.followers) :
    HappensBefore (.multiply batch follower) (.assemble batch lead) :=
  .trans (.follower_finishes h) (.trans (.idle_sync h) (.wait_then_assemble h))

theorem assembly_before_store {batch : ℕ} {lead : Job} (h : lead.followers ≠ []) :
    HappensBefore (.assemble batch lead) (.store batch) := by
  apply HappensBefore.trans _ (HappensBefore.all_before_store h)
  cases lead <;> simp [Job.followers] at h
  · exact .trans (.assembled_idle (by simp [Job.followers])) (.lead_idle_sync (.inl rfl))
  · exact .trans (.assembled_idle (by simp [Job.followers])) (.lead_idle_sync (.inr rfl))
  · exact .caller_assembles

theorem follower_before_aliased_store {batch : ℕ} {lead follower : Job}
    (h : follower ∈ lead.followers) :
    HappensBefore (.multiply batch follower) (.store batch) :=
  .trans (follower_visible h) (assembly_before_store (List.ne_nil_of_mem h))

theorem workers_before_reuse {batch : ℕ} {lead follower : Job}
    (h : follower ∈ lead.followers) :
    HappensBefore (.multiply batch follower) (.prepare (batch + 1)) :=
  .trans (follower_before_aliased_store h) .next_batch

theorem workers_before_shutdown {batch : ℕ} {lead follower : Job}
    (h : follower ∈ lead.followers) :
    HappensBefore (.multiply batch follower) (.shutdown batch) :=
  .trans (follower_before_aliased_store h) .stop_after_batch

end FractalSharkProofs.ReferenceOrbit.MT9
