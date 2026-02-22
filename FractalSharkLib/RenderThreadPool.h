#pragma once

#include "GPU_Types.h"
#include "ItersMemoryContainer.h"
#include "PointZoomBBConverter.h"
#include "RenderAlgorithm.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class Fractal;
class GPURenderer;

// Snapshot of all render-relevant state, captured at enqueue time.
// Workers read exclusively from this, never from live Fractal members.
struct RenderWorkItem {
    // Monotonic sequence number assigned at enqueue time
    uint64_t SequenceNumber;

    // Location / zoom
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

    // Promise to signal job completion
    std::shared_ptr<std::promise<void>> CompletionPromise;
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

    // Actual allocation size of ColorData (may be larger than
    // OutputWidth*OutputHeight due to GPU block-rounding).
    size_t BufferPixelCount;
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

    // Wait until at least one frame for expectedSeqNum (or newer) is available, or shutdown.
    // Returns false on shutdown.
    bool WaitForFrame(uint64_t expectedSeqNum);

    // Wait until a frame for exactly seqNum is available, or shutdown.
    // Unlike WaitForFrame, does NOT wake for newer sequences.
    // Returns false on shutdown.
    bool WaitForFrameExact(uint64_t seqNum);

    // If the expected sequence is missing, advance expectedSeqNum to the
    // next available sequence in the queue.  Purges orphaned older frames.
    // Returns true if a sequence was found, false if queue is empty.
    bool SkipToNextAvailable(uint64_t &expectedSeqNum);

    // Signal shutdown to unblock any waiting consumers.
    void Shutdown();

private:
    std::mutex m_Mutex;
    std::condition_variable m_CV;

    // Frames stored by sequence number. Each sequence can have
    // multiple progressive frames and one final frame.
    std::deque<RenderFrame> m_Frames;

    bool m_ShutdownFlag;
};

// The main render thread pool. Owns 4 worker threads and a GL consumer thread.
// Replaces the old CalcFractal → DrawFractal → WaitForDrawFractal path.
class RenderThreadPool {
public:
    RenderThreadPool(Fractal *fractal, HWND hWnd);
    ~RenderThreadPool();

    // Enqueue a render job. Returns a handle for optional waiting.
    // Captures a snapshot of render state at call time.
    RenderJobHandle Enqueue(const RenderWorkItem &item);

    // Convenience: snapshot current Fractal state and enqueue.
    RenderJobHandle EnqueueCurrentState();

    // Enqueue with explicit Ptz (coordinates come from caller, not m_Ptz).
    RenderJobHandle EnqueueCurrentState(const PointZoomBBConverter &ptz);

    // Shutdown the pool and join all threads.
    void Shutdown();

private:
    static constexpr size_t NumWorkers = NumRenderers;

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

    // Run CalcFractal under m_CalcFractalMutex with state save/restore.
    // Swaps workerIters into m_CurIters, sets Ptz and dirty flags from the
    // work item, calls CalcFractal, then restores everything.  Exception-safe.
    void RunCalcFractal(Fractal *fractal,
                        const RenderWorkItem &item,
                        RendererIndex rendererIdx,
                        ItersMemoryContainer &workerIters);

    // Wait for GPU compute to finish, producing progressive frames periodically.
    void WaitForGpuAndProduceProgressiveFrames(
        Fractal *fractal,
        GPURenderer &renderer,
        const RenderWorkItem &item,
        RendererIndex rendererIdx,
        ItersMemoryContainer &workerIters,
        std::mutex &workerMutex,
        std::condition_variable &workerCV);

    // Upload a RenderFrame to the GL context as a textured quad.
    void RenderFrameToGL(OpenGlContext &glContext, const RenderFrame &frame);

    // Snapshot current Fractal state into a RenderWorkItem (everything except Ptz).
    RenderWorkItem SnapshotCurrentState() const;

    Fractal *m_Fractal;
    HWND m_hWnd;

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
};
