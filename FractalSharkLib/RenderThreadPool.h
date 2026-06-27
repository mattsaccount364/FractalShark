#pragma once

#include "GPU_Types.h"
#include "ItersMemoryContainer.h"
#include "OpenGLContext.h"
#include "PointZoomBBConverter.h"
#include "RenderAlgorithm.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_set>
#include <vector>

class Fractal;
class GPURenderer;

enum class RenderPresentationMode {
    Immediate,
    PacedAnimation,
};

// Snapshot of all render-relevant state, captured at enqueue time.
// Workers read exclusively from this, never from live Fractal members.
struct RenderWorkItem {
    // Monotonic sequence number assigned at enqueue time
    uint64_t SequenceNumber;

    // Location / zoom — always snapshotted from m_Ptz after command execution.
    PointZoomBBConverter Ptz{PointZoomBBConverter::TestMode::Enabled};

    // Algorithm
    RenderAlgorithm Algorithm;

    // Iteration config
    IterTypeEnum IterType;
    IterTypeFull NumIterations;
    uint32_t IterationPrecision;

    // Screen dimensions
    size_t ScrnWidth;
    size_t ScrnHeight;

    // GPU antialiasing (1=none, 2=4x, 3=9x, 4=16x)
    uint32_t GpuAntialiasing;

    // Changed flags
    bool ChangedWindow;
    bool ChangedScrn;
    bool ChangedIterations;

    // Back-pointer to Fractal for reference orbit access (not snapshotted).
    // Potential race deferred to future work.
    Fractal *FractalPtr;

    // Pre-render command: executed by the worker under m_CalcFractalMutex
    // before CalcFractal.  Use this to mutate Fractal state from the worker
    // thread instead of the UI thread, avoiding data races.
    // When set, the worker re-snapshots state after executing the command.
    std::function<void(Fractal &)> Command;

    // When true, the worker only executes the Command (under the lock)
    // and skips CalcFractal / GPU / frame production entirely.
    // Used for settings changes that take effect on the next render.
    bool MutationOnly = false;

    // When true (default), this render item can be superseded by a newer
    // enqueue — the older queued item is discarded.  Set to false for
    // pipelined renders (AutoZoomer) that must all execute.
    bool Supersedable = true;

    // Generation number stamped at enqueue time.  Workers skip items
    // whose generation is older than the latest enqueued supersedable item.
    uint64_t EnqueueGeneration = 0;

    // Paced animation frames retain normal rendering behavior but final
    // frames are presented at an adaptive cadence.  PresentationGroup separates
    // consecutive animations so each one receives its own pre-roll.
    RenderPresentationMode PresentationMode = RenderPresentationMode::Immediate;
    uint64_t PresentationGroup = 0;

    // Ordinary user renders clear a stale abort before starting.  Autozoom
    // clears once at startup, then preserves a live abort across its frames.
    bool ResetStopCalculatingBeforeRender = true;

    // Promise to signal job completion
    std::shared_ptr<std::promise<void>> CompletionPromise;
};

struct PresentedViewState {
    PointZoomBBConverter Ptz;
    IterTypeFull NumIterations;
};

// A rendered frame produced by a worker thread.
// Contains owned color data ready for GL texture upload.
struct RenderFrame {
    // Sequence number matching the originating RenderWorkItem
    uint64_t SequenceNumber;

    // Owned color buffer (RGBA16)
    std::unique_ptr<Color16[]> ColorData;

    // Dimensions of the color buffer
    size_t OutputWidth;
    size_t OutputHeight;

    // True if this is a progressive (partial) frame
    bool IsProgressive;

    // True if this is the final (complete) frame for this sequence
    bool IsFinal;

    // True when repainting is disabled and this frame should present the
    // legacy X placeholder instead of fractal pixels.
    bool IsRepaintPlaceholder;

    RenderPresentationMode PresentationMode;
    uint64_t PresentationGroup;
    std::optional<PresentedViewState> ViewState;

    // Actual allocation size of ColorData (may be larger than
    // OutputWidth*OutputHeight due to GPU block-rounding).
    size_t BufferPixelCount;
};

struct RenderFrameInfo {
    RenderPresentationMode PresentationMode;
    uint64_t PresentationGroup;
    bool IsProgressive;
    bool IsFinal;
};

// Manages checkout/return of GPURenderer instances from the Fractal's
// m_Renderers array. Workers acquire a renderer for the duration of a
// job and release it when done.
class RendererPool {
public:
    RendererPool();

    // Initialize with pointers to the Fractal's renderer array.
    // Called once during RenderThreadPool construction.
    void Initialize(GPURenderer *renderers, size_t count);

    // Acquire a free renderer. Blocks if all are in use.
    // Returns the RendererIndex of the acquired renderer.
    RendererIndex Acquire();

    // Release a previously acquired renderer back to the pool.
    void Release(RendererIndex idx);

private:
    std::mutex m_Mutex;
    std::condition_variable m_CV;
    std::vector<bool> m_Available;
    GPURenderer *m_Renderers;
    size_t m_Count;
};

// Handle returned by Enqueue. Callers can optionally Wait() for completion.
class RenderJobHandle {
public:
    RenderJobHandle() = default;
    explicit RenderJobHandle(std::shared_future<void> future);

    // Block until the job completes.
    void Wait();

    // Check if the job is done without blocking.
    bool IsReady() const;

private:
    std::shared_future<void> m_Future;
};

// Thread-safe queue of completed frames, supporting ordered retrieval.
class FrameCompletionQueue {
public:
    FrameCompletionQueue();

    // Push a completed frame (progressive or final).
    void Push(RenderFrame frame);

    // Try to pop the next frame in sequence order.
    // Returns true and fills 'frame' if the next expected frame is available.
    // expectedSeqNum: the sequence number the consumer expects next.
    // If progressive frames for expectedSeqNum are available, returns those first.
    // When a final frame for expectedSeqNum is returned, consumer should advance.
    bool TryPopNextInOrder(uint64_t expectedSeqNum, RenderFrame &frame);

    // Inspect the next frame without removing it.  Used by the presenter
    // to defer paced final frames while keeping progressive frames immediate.
    bool TryPeekNextInOrder(uint64_t expectedSeqNum, RenderFrameInfo &frameInfo);

    // Count consecutive completed paced-animation steps beginning at
    // expectedSeqNum.  Progressive frames do not affect the count.
    size_t CountConsecutivePacedFinalFrames(uint64_t expectedSeqNum, uint64_t presentationGroup);

    // Discard unpresented frames from an aborted paced animation and
    // replace each affected sequence with one tombstone.
    void CancelPacedAnimation(uint64_t presentationGroup);

    void PushTombstone(uint64_t sequenceNumber);

    // Wait until at least one frame for expectedSeqNum (or newer) is available, or shutdown.
    // Returns false on shutdown.
    bool WaitForFrame(uint64_t expectedSeqNum);

    // Wait until a frame for exactly seqNum is available, or shutdown.
    // Unlike WaitForFrame, does NOT wake for newer sequences.
    // Returns false on shutdown.
    bool WaitForFrameExact(uint64_t seqNum);

    // Wait for queue activity or a pacing deadline.  changeGeneration must
    // come from GetChangeGeneration() before checking presentation readiness.
    bool WaitForChangeUntil(uint64_t changeGeneration, std::chrono::steady_clock::time_point deadline);

    uint64_t GetChangeGeneration();

    // If the expected sequence is missing, advance expectedSeqNum to the
    // next available sequence in the queue.  Purges orphaned older frames.
    // Returns true if a sequence was found, false if queue is empty.
    bool SkipToNextAvailable(uint64_t &expectedSeqNum);

    // Signal shutdown to unblock any waiting consumers.
    void Shutdown();

    // Wake the consumer for an overlay-only repaint (no new frame).
    // Thread-safe: can be called from the UI thread.
    void NotifyOverlay();

    // Check and clear the overlay-dirty flag.  Called by the consumer
    // after waking to distinguish overlay wakeups from frame wakeups.
    bool ConsumeOverlayDirty();

private:
    void PushTombstoneLocked(uint64_t sequenceNumber);

    std::mutex m_Mutex;
    std::condition_variable m_CV;

    // Frames stored by sequence number. Each sequence can have
    // multiple progressive frames and one final frame.
    std::deque<RenderFrame> m_Frames;
    std::unordered_set<uint64_t> m_CancelledPresentationGroups;

    uint64_t m_ChangeGeneration = 0;
    bool m_ShutdownFlag;

    // Set by NotifyOverlay(), cleared by ConsumeOverlayDirty().
    // Protected by m_Mutex to share the same CV without missed wakeups.
    bool m_OverlayDirty = false;
};

// The main render thread pool. Owns 4 worker threads and a GL consumer thread.
// Replaces the old CalcFractal → DrawFractal → WaitForDrawFractal path.
class RenderThreadPool {
public:
    RenderThreadPool(Fractal *fractal, void *nativeWindow, bool hostOwnedGlPresentation = false);
    ~RenderThreadPool();

    // Enqueue a render job. Returns a handle for optional waiting.
    // Captures a snapshot of render state at call time.
    RenderJobHandle Enqueue(const RenderWorkItem &item);

    // Enqueue a command that mutates Fractal state, then renders.
    // The command lambda runs on a worker thread under m_CalcFractalMutex.
    // After the command, the worker snapshots state and renders.
    RenderJobHandle EnqueueCommand(
        std::function<void(Fractal &)> cmd,
        bool supersedable = true,
        RenderPresentationMode presentationMode = RenderPresentationMode::Immediate,
        uint64_t presentationGroup = 0,
        bool resetStopCalculatingBeforeRender = true);

    // Allocate an identity for a paced animation.  The presenter resets
    // pre-roll state whenever it sees a new group.
    uint64_t BeginPacedAnimation();

    // Abort queued, buffered, and in-flight work for one paced animation.
    void CancelPacedAnimation(uint64_t presentationGroup);

    std::optional<PresentedViewState> GetLastPresentedView() const;

    // Enqueue a mutation-only command: executes the lambda under the lock
    // but does NOT trigger CalcFractal or frame production.
    // Use for settings changes that take effect on the next render.
    RenderJobHandle EnqueueMutation(std::function<void(Fractal &)> cmd);

    // Shutdown the pool and join all threads.
    void Shutdown();

    // Wait for all queued and in-flight work items to complete.
    // Returns once no workers are processing and the queue is empty.
    void Drain();

    // Host-owned GL presentation mode (Linux GUI).  When the pool was
    // constructed with hostOwnedGlPresentation=true, no internal GL
    // consumer thread is started; the host's main loop is expected to
    // call TryPresentTick() each tick to upload the next ready frame and
    // present it on the GL context the host owns.  Returns true if a frame
    // was uploaded to GL (host should compose its overlay + swap),
    // false if no frame was ready.  No-op (returns false) when the
    // pool owns its own consumer thread.
    bool TryPresentTick(OpenGlContext &glContext);

    // Host-owned GL presentation mode only.  Returns the remaining wait
    // until a deferred paced frame becomes eligible for presentation.
    std::optional<std::chrono::milliseconds> GetTimeUntilNextPresentation() const;

    // Host-owned GL presentation mode only.  Re-uploads the most
    // recently rendered frame to GL.  Used for overlay-only repaints
    // when no new fractal frame is available (e.g. user is hovering
    // a popup with no fractal updates in flight).  Returns false if
    // no cached frame exists yet.
    bool RepresentLastFrame(OpenGlContext &glContext);

    // Update the drag-zoom selection rectangle overlay.
    // active=false hides it.  Thread-safe: called from the UI thread.
    void SetDragRect(bool active, int x0, int y0, int x1, int y1);

private:
    static constexpr size_t NumWorkers = NumRenderers;
    static constexpr size_t PacedAnimationPreRollFrames = 4;
    static constexpr auto PacedAnimationPreRollTimeout = std::chrono::milliseconds(500);
    static constexpr auto PacedAnimationFrameInterval = std::chrono::milliseconds(100);

    // Pre-seed enough buffers to cover in-flight frames across all workers
    // plus progressive frames queued for the GL consumer.
    static constexpr size_t PreSeedCount = 2 * NumRenderers;

    // Worker thread entry point
    void WorkerLoop(size_t workerIndex);

    // GL consumer thread entry point
    void GlConsumerLoop();

    // Dequeue the next work item. Blocks until available or shutdown.
    // Returns false on shutdown.
    bool DequeueWorkItem(RenderWorkItem &item);

    // Produce a RenderFrame from the current renderer state and push to completion queue.
    // Uses the worker's local ItersMemoryContainer, not the shared m_CurIters.
    // Returns true if a frame was pushed, false on early-out (dimension mismatch, error, etc.).
    bool ProduceFrame(const RenderWorkItem &item,
                      RendererIndex rendererIdx,
                      ItersMemoryContainer &workerIters,
                      bool isFinal);

    // Push a tombstone (empty IsFinal) frame so the GL consumer can advance
    // past the given sequence number.
    void PushTombstone(uint64_t sequenceNumber);

    // Push a final placeholder frame for a repaint-disabled render.
    void PushRepaintPlaceholder(const RenderWorkItem &item);

    enum class CalcRenderResult {
        Rendered,
        RepaintDisabled,
        Failed,
    };

    // Run CalcFractal under m_CalcFractalMutex with state save/restore.
    // If item.Command is set, executes it first, then re-snapshots state.
    // Swaps workerIters into m_CurIters, sets Ptz and dirty flags from the
    // work item, calls CalcFractal, then restores everything.  Exception-safe.
    // Returns RepaintDisabled when the render should present the legacy
    // placeholder without clearing dirty state.
    CalcRenderResult RunCalcFractal(Fractal *fractal,
                                    RenderWorkItem &item,
                                    RendererIndex rendererIdx,
                                    ItersMemoryContainer &workerIters);

    // Wait for GPU compute to finish, producing progressive frames periodically.
    void WaitForGpuAndProduceProgressiveFrames(Fractal *fractal,
                                               GPURenderer &renderer,
                                               const RenderWorkItem &item,
                                               RendererIndex rendererIdx,
                                               ItersMemoryContainer &workerIters,
                                               std::mutex &workerMutex,
                                               std::condition_variable &workerCV);

    // Upload a RenderFrame to the GL context as a textured quad.
    // If persistTexOut is non-null (hardware path), stores the new texture
    // ID there (caller owns lifetime) and does NOT delete it after drawing.
    // Deletes the old *persistTexOut first if non-zero.
    void RenderFrameToGL(OpenGlContext &glContext,
                         const RenderFrame &frame,
                         unsigned int *persistTexOut = nullptr);

    // Redraw the last persistent texture as a fullscreen quad (hardware path).
    void RedrawLastTexture(OpenGlContext &glContext, unsigned int texId, size_t width, size_t height);

    // Draw the drag-zoom selection rectangle overlay using GL_INVERT.
    // Reads drag rect state under m_DragRectMutex.  frameHeight is needed
    // to flip Y from screen coords (top=0) to GL coords (bottom=0).
    void DrawDragRectOverlay(size_t frameHeight);

    // Snapshot current Fractal state into a RenderWorkItem (everything except Ptz).
    RenderWorkItem SnapshotCurrentState() const;

    bool IsPresentationReady(uint64_t expectedSeqNum,
                             const RenderFrameInfo &frameInfo,
                             std::chrono::steady_clock::time_point now);
    void RecordPresentedFrame(const RenderFrame &frame, std::chrono::steady_clock::time_point now);
    static std::chrono::milliseconds GetPacedAnimationFrameInterval(size_t bufferedFrames);
    void ResetPacedAnimationState();
    bool IsPacedAnimationCancelled(const RenderWorkItem &item) const;

    Fractal *m_Fractal;
    void *m_NativeWindow;

    // Work queue
    std::mutex m_WorkQueueMutex;
    std::condition_variable m_WorkQueueCV;
    std::deque<RenderWorkItem> m_WorkQueue;
    std::atomic<uint64_t> m_NextSequenceNumber;

    // Renderer pool
    RendererPool m_RendererPool;

    // Serializes CalcFractal calls to avoid data races on shared Fractal state
    // (m_CurIters, changed flags, etc.). Multiple workers can compute concurrently
    // on different renderers, but CalcFractal itself modifies shared state.
    std::mutex m_CalcFractalMutex;

    // Frame completion queue
    FrameCompletionQueue m_FrameQueue;

    // Frame buffer pool: reuse Color16[] buffers to avoid per-frame alloc/free
    std::mutex m_BufferPoolMutex;
    std::vector<std::unique_ptr<Color16[]>> m_BufferPool;
    size_t m_BufferPoolPixelCount = 0;

    std::unique_ptr<Color16[]> AcquireFrameBuffer(size_t totalPixels);
    void ReleaseFrameBuffer(std::unique_ptr<Color16[]> buffer, size_t totalPixels);

    // Threads
    std::vector<std::thread> m_Workers;
    std::thread m_GlConsumerThread;

    // Shutdown flag
    std::atomic<bool> m_ShutdownFlag;

    // In-flight job tracking for Drain()
    std::atomic<size_t> m_InFlightCount{0};
    std::mutex m_DrainMutex;
    std::condition_variable m_DrainCV;

    // Generation counter for skipping stale in-flight renders.
    // Only incremented for supersedable items.  Non-atomic: single
    // writer (UI thread), multi-reader (workers) — safe on x64.
    std::atomic<uint64_t> m_EnqueueGeneration = 0;
    std::atomic<uint64_t> m_NextPresentationGroup = 0;

    mutable std::mutex m_CancelledPresentationGroupsMutex;
    std::unordered_set<uint64_t> m_CancelledPresentationGroups;

    // Host-owned GL presentation mode (Linux GUI).  When true the pool
    // does not start m_GlConsumerThread; the host's main loop drives
    // presentation by calling TryPresentTick / RepresentLastFrame.
    const bool m_HostOwnedGlPresentation = false;

    // Host-owned mode: per-host sequence-number tracker (counterpart
    // to GlConsumerLoop's nextExpectedSeqNum local).
    uint64_t m_HostExpectedSeqNum = 0;

    // Host-owned mode: most recently uploaded frame, retained for
    // overlay-only repaints via RepresentLastFrame.  ColorData is
    // released back to the pool only when superseded by a newer frame.
    RenderFrame m_HostLastFrame{};

    // Presenter-thread-only paced-animation state.  In Windows mode this
    // is owned by the GL consumer; in host-owned mode it is owned by the
    // Linux event-loop thread.
    uint64_t m_PacedAnimationGroup = 0;
    bool m_PacedAnimationStarted = false;
    std::optional<std::chrono::steady_clock::time_point> m_PacedAnimationPreRollStart;
    std::optional<std::chrono::steady_clock::time_point> m_NextPacedPresentation;
    std::mutex m_PresentationMutex;

    mutable std::mutex m_LastPresentedViewMutex;
    std::optional<PresentedViewState> m_LastPresentedView;

    // --- Drag-zoom selection rectangle overlay ---
    // Written by the UI thread via SetDragRect(), read by the GL consumer.
    std::mutex m_DragRectMutex;
    bool m_DragRectActive = false;
    int m_DragRectX0 = 0;
    int m_DragRectY0 = 0;
    int m_DragRectX1 = 0;
    int m_DragRectY1 = 0;
};
