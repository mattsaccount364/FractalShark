#include "stdafx.h"
#include "RenderThreadPool.h"
#include "Fractal.h"
#include "GPU_Render.h"

#include <algorithm>
#include <chrono>

// ============================================================================
// RendererPool
// ============================================================================

RendererPool::RendererPool()
    : m_Renderers(nullptr),
      m_Count(0) {
}

void RendererPool::Initialize(GPURenderer *renderers, size_t count) {
    m_Renderers = renderers;
    m_Count = count;
    m_Available.assign(count, true);
}

RendererIndex RendererPool::Acquire() {
    std::unique_lock lk(m_Mutex);
    m_CV.wait(lk, [this] {
        return std::any_of(m_Available.begin(), m_Available.end(),
                           [](bool v) { return v; });
    });

    for (size_t i = 0; i < m_Count; ++i) {
        if (m_Available[i]) {
            m_Available[i] = false;
            return static_cast<RendererIndex>(i);
        }
    }

    // Should never reach here
    return RendererIndex::Renderer0;
}

void RendererPool::Release(RendererIndex idx) {
    {
        std::lock_guard lk(m_Mutex);
        m_Available[static_cast<size_t>(idx)] = true;
    }
    m_CV.notify_one();
}

// ============================================================================
// RenderJobHandle
// ============================================================================

RenderJobHandle::RenderJobHandle(std::shared_future<void> future)
    : m_Future(std::move(future)) {
}

void RenderJobHandle::Wait() {
    if (m_Future.valid()) {
        m_Future.wait();
    }
}

bool RenderJobHandle::IsReady() const {
    if (!m_Future.valid()) {
        return true;
    }
    return m_Future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// ============================================================================
// FrameCompletionQueue
// ============================================================================

FrameCompletionQueue::FrameCompletionQueue()
    : m_ShutdownFlag(false) {
}

void FrameCompletionQueue::Push(RenderFrame frame) {
    {
        std::lock_guard lk(m_Mutex);
        m_Frames.push_back(std::move(frame));
    }
    m_CV.notify_one();
}

bool FrameCompletionQueue::TryPopNextInOrder(uint64_t expectedSeqNum, RenderFrame &frame) {
    std::lock_guard lk(m_Mutex);

    for (auto it = m_Frames.begin(); it != m_Frames.end(); ++it) {
        if (it->SequenceNumber == expectedSeqNum) {
            frame = std::move(*it);
            m_Frames.erase(it);
            return true;
        }
    }

    return false;
}

bool FrameCompletionQueue::WaitForFrame(uint64_t expectedSeqNum) {
    std::unique_lock lk(m_Mutex);
    // Wake on the exact sequence or any newer frame.  If a sequence is
    // permanently lost (e.g. worker crash), newer frames will still wake
    // us so the consumer can skip forward rather than blocking forever.
    m_CV.wait(lk, [this, expectedSeqNum] {
        if (m_ShutdownFlag) return true;
        return std::any_of(m_Frames.begin(), m_Frames.end(),
                           [expectedSeqNum](const RenderFrame &f) {
                               return f.SequenceNumber >= expectedSeqNum;
                           });
    });
    return !m_ShutdownFlag;
}

bool FrameCompletionQueue::WaitForFrameExact(uint64_t seqNum) {
    std::unique_lock lk(m_Mutex);
    m_CV.wait(lk, [this, seqNum] {
        if (m_ShutdownFlag) return true;
        return std::any_of(m_Frames.begin(), m_Frames.end(),
                           [seqNum](const RenderFrame &f) {
                               return f.SequenceNumber == seqNum;
                           });
    });
    return !m_ShutdownFlag;
}

bool FrameCompletionQueue::SkipToNextAvailable(uint64_t &expectedSeqNum) {
    std::lock_guard lk(m_Mutex);

    // Find the minimum sequence number >= expectedSeqNum
    uint64_t minSeq = UINT64_MAX;
    for (const auto &f : m_Frames) {
        if (f.SequenceNumber >= expectedSeqNum && f.SequenceNumber < minSeq) {
            minSeq = f.SequenceNumber;
        }
    }

    if (minSeq == UINT64_MAX) {
        return false; // No frames available
    }

    // Purge orphaned frames from lost sequences (seq < minSeq)
    std::erase_if(m_Frames, [minSeq](const RenderFrame &f) {
        return f.SequenceNumber < minSeq;
    });

    expectedSeqNum = minSeq;
    return true;
}

void FrameCompletionQueue::Shutdown() {
    {
        std::lock_guard lk(m_Mutex);
        m_ShutdownFlag = true;
    }
    m_CV.notify_all();
}

// ============================================================================
// RenderThreadPool
// ============================================================================

RenderThreadPool::RenderThreadPool(Fractal *fractal, HWND hWnd)
    : m_Fractal(fractal),
      m_hWnd(hWnd),
      m_NextSequenceNumber(0),
      m_ShutdownFlag(false) {

    m_RendererPool.Initialize(
        &fractal->GetRenderer(RendererIndex::Renderer0),
        NumRenderers);

    for (size_t i = 0; i < NumWorkers; ++i) {
        m_Workers.emplace_back(&RenderThreadPool::WorkerLoop, this, i);
    }

    m_GlConsumerThread = std::thread(&RenderThreadPool::GlConsumerLoop, this);
}

RenderThreadPool::~RenderThreadPool() {
    Shutdown();
}

std::unique_ptr<Color16[]> RenderThreadPool::AcquireFrameBuffer(size_t totalPixels) {
    std::lock_guard lk(m_BufferPoolMutex);
    if (!m_BufferPool.empty() && m_BufferPoolPixelCount == totalPixels) {
        auto buf = std::move(m_BufferPool.back());
        m_BufferPool.pop_back();
        return buf;
    }

    // Dimension changed or first call — pre-seed the pool so future
    // acquires hit the fast path even when the GL consumer hasn't
    // returned buffers yet (producer-consumer timing gap).
    if (m_BufferPoolPixelCount != totalPixels) {
        m_BufferPool.clear();
        m_BufferPoolPixelCount = totalPixels;
    }

    for (size_t i = 0; i < PreSeedCount; ++i) {
        m_BufferPool.emplace_back(new Color16[totalPixels]);
    }

    // Default-init (no zero-fill) — buffer is always fully overwritten
    // by ExtractItersAndColors or memcpy before use.
    return std::unique_ptr<Color16[]>(new Color16[totalPixels]);
}

void RenderThreadPool::ReleaseFrameBuffer(std::unique_ptr<Color16[]> buffer, size_t totalPixels) {
    std::lock_guard lk(m_BufferPoolMutex);
    if (m_BufferPoolPixelCount != totalPixels) {
        // Dimension changed — discard old buffers
        m_BufferPool.clear();
        m_BufferPoolPixelCount = totalPixels;
    }
    m_BufferPool.push_back(std::move(buffer));
}

RenderJobHandle RenderThreadPool::Enqueue(const RenderWorkItem &item) {
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future().share();

    RenderWorkItem workItem = item;
    workItem.CompletionPromise = std::move(promise);

    {
        std::lock_guard lk(m_WorkQueueMutex);
        workItem.SequenceNumber = m_NextSequenceNumber++;
        m_WorkQueue.push_back(std::move(workItem));
    }
    m_WorkQueueCV.notify_one();

    return RenderJobHandle(std::move(future));
}

RenderWorkItem RenderThreadPool::SnapshotCurrentState() const {
    RenderWorkItem item{};
    item.Ptz = m_Fractal->GetPtz();
    item.Algorithm = m_Fractal->GetRenderAlgorithm();
    item.IterType = m_Fractal->GetIterType();
    item.NumIterations = m_Fractal->GetNumIterationsRT();
    item.IterationPrecision = m_Fractal->GetIterationPrecision();
    item.ScrnWidth = m_Fractal->GetScrnWidth();
    item.ScrnHeight = m_Fractal->GetScrnHeight();
    item.GpuAntialiasing = m_Fractal->GetGpuAntialiasing();
    // Always mark dirty so CalcFractal actually computes.
    // If the snapshot captured clean flags (e.g. a prior enqueue already
    // snapshotted them), CalcFractal would skip computation and the worker
    // would produce frames from stale/garbage GPU memory.
    item.ChangedWindow = true;
    item.ChangedScrn = true;
    item.ChangedIterations = true;
    item.FractalPtr = m_Fractal;
    return item;
}

RenderJobHandle RenderThreadPool::EnqueueCurrentState() {
    return Enqueue(SnapshotCurrentState());
}

RenderJobHandle RenderThreadPool::EnqueueCurrentState(const PointZoomBBConverter &ptz) {
    auto item = SnapshotCurrentState();
    item.Ptz = ptz;
    return Enqueue(item);
}

void RenderThreadPool::PushTombstone(uint64_t sequenceNumber) {
    RenderFrame skipFrame{};
    skipFrame.SequenceNumber = sequenceNumber;
    skipFrame.IsFinal = true;
    m_FrameQueue.Push(std::move(skipFrame));
}

void RenderThreadPool::Shutdown() {
    if (m_ShutdownFlag.exchange(true)) {
        return; // Already shut down
    }

    // Wake up all workers
    m_WorkQueueCV.notify_all();

    // Shut down the frame queue to unblock GL consumer
    m_FrameQueue.Shutdown();

    for (auto &w : m_Workers) {
        if (w.joinable()) {
            w.join();
        }
    }

    if (m_GlConsumerThread.joinable()) {
        m_GlConsumerThread.join();
    }
}

bool RenderThreadPool::DequeueWorkItem(RenderWorkItem &item) {
    std::unique_lock lk(m_WorkQueueMutex);
    m_WorkQueueCV.wait(lk, [this] {
        return m_ShutdownFlag.load() || !m_WorkQueue.empty();
    });

    if (m_ShutdownFlag.load() && m_WorkQueue.empty()) {
        return false;
    }

    item = std::move(m_WorkQueue.front());
    m_WorkQueue.pop_front();
    return true;
}

void RenderThreadPool::RunCalcFractal(
    Fractal *fractal,
    const RenderWorkItem &item,
    RendererIndex rendererIdx,
    ItersMemoryContainer &workerIters) {

    // Hold the lock for the full CalcFractal call.  CalcFractalTypedIter
    // calls ChangedMakeClean() at the end, which clears the shared flags.
    // Without the lock covering CalcFractal, Worker A's clean can wipe
    // Worker B's flags between set and read, causing B to skip computation
    // and produce a final frame from stale GPU data.
    std::lock_guard lk(m_CalcFractalMutex);
    fractal->m_ChangedWindow = item.ChangedWindow;
    fractal->m_ChangedScrn = item.ChangedScrn;
    fractal->m_ChangedIterations = item.ChangedIterations;

    // Ptz and ItersMemory travel through CalcContext — no swap needed.
    CalcContext ctx{item.Ptz, workerIters};
    fractal->CalcFractal(rendererIdx, false, ctx);
}

void RenderThreadPool::WaitForGpuAndProduceProgressiveFrames(
    Fractal *fractal,
    GPURenderer &renderer,
    const RenderWorkItem &item,
    RendererIndex rendererIdx,
    ItersMemoryContainer &workerIters,
    std::mutex &workerMutex,
    std::condition_variable &workerCV) {

    if (fractal->m_BypassGpu) {
        return;
    }

    static constexpr auto ProgressiveDrawInterval = std::chrono::milliseconds(1000);

    renderer.ResetComputeDoneFlag();
    renderer.EnqueueComputeDoneCallback();

    for (;;) {
        std::unique_lock lk(workerMutex);
        auto computeDone = workerCV.wait_for(lk, ProgressiveDrawInterval, [&] {
            return renderer.IsComputeDone() || m_ShutdownFlag.load();
        });
        lk.unlock();

        if (m_ShutdownFlag.load()) {
            break;
        }

        if (computeDone) {
            uint32_t result = renderer.SyncComputeStream();
            if (result) {
                fractal->MessageBoxCudaError(result);
            }
            break;
        }

        ProduceFrame(item, rendererIdx, workerIters, false);
    }
}

void RenderThreadPool::RenderFrameToGL(OpenGlContext &glContext, const RenderFrame &frame) {
    glContext.glResetViewDim(frame.OutputWidth, frame.OutputHeight);

    GLuint texid;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA16,
                 (GLsizei)frame.OutputWidth,
                 (GLsizei)frame.OutputHeight,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_SHORT,
                 frame.ColorData.get());

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2i(0, (GLint)frame.OutputHeight);
    glTexCoord2i(0, 1);
    glVertex2i(0, 0);
    glTexCoord2i(1, 1);
    glVertex2i((GLint)frame.OutputWidth, 0);
    glTexCoord2i(1, 0);
    glVertex2i((GLint)frame.OutputWidth, (GLint)frame.OutputHeight);
    glEnd();
    glFlush();
    glDeleteTextures(1, &texid);
}

void RenderThreadPool::WorkerLoop(size_t workerIndex) {
    SetThreadDescription(GetCurrentThread(),
        std::format(L"RenderPool Worker {}", workerIndex).c_str());

    std::mutex workerMutex;
    std::condition_variable workerCV;

    while (!m_ShutdownFlag.load()) {
        RenderWorkItem item;
        if (!DequeueWorkItem(item)) {
            break;
        }

        RendererIndex rendererIdx = m_RendererPool.Acquire();
        Fractal *fractal = item.FractalPtr;
        auto &renderer = fractal->GetRenderer(rendererIdx);
        renderer.SetComputeDoneNotification(&workerMutex, &workerCV);
        ItersMemoryContainer workerIters = fractal->AcquireItersMemory();

        // Guarantee: every dequeued sequence gets at least one IsFinal frame.
        bool finalFramePushed = false;

        try {
            if (workerIters.m_OutputWidth != item.ScrnWidth ||
                workerIters.m_OutputHeight != item.ScrnHeight) {
                PushTombstone(item.SequenceNumber);
                finalFramePushed = true;
            } else {
                RunCalcFractal(fractal, item, rendererIdx, workerIters);

                WaitForGpuAndProduceProgressiveFrames(
                    fractal, renderer, item, rendererIdx,
                    workerIters, workerMutex, workerCV);

                bool pushed = !m_ShutdownFlag.load() &&
                              ProduceFrame(item, rendererIdx, workerIters, true);
                if (!pushed) {
                    PushTombstone(item.SequenceNumber);
                }
                finalFramePushed = true;
            }
        } catch (...) {
            // Exception during processing — fall through to cleanup.
        }

        if (!finalFramePushed) {
            PushTombstone(item.SequenceNumber);
        }

        // Return the container to the pool
        fractal->ReturnIterMemory(std::move(workerIters));

        // Signal completion to any waiters
        if (item.CompletionPromise) {
            try {
                item.CompletionPromise->set_value();
            } catch (...) {
                // Promise may already be fulfilled or broken
            }
        }

        // Release the renderer back to the pool
        m_RendererPool.Release(rendererIdx);
    }
}

void RenderThreadPool::GlConsumerLoop() {
    SetThreadDescription(GetCurrentThread(), L"RenderPool GL Consumer");

    // Create OpenGL context for this thread
    auto glContext = std::make_unique<OpenGlContext>(m_hWnd);
    if (!glContext->IsValid()) {
        return;
    }

    uint64_t nextExpectedSeqNum = 0;

    while (!m_ShutdownFlag.load()) {
        if (!m_FrameQueue.WaitForFrameExact(nextExpectedSeqNum)) {
            break; // Shutdown
        }

        RenderFrame frame;
        if (!m_FrameQueue.TryPopNextInOrder(nextExpectedSeqNum, frame)) {
            continue; // Spurious wakeup — wait again
        }

        // Process frames for this sequence number
        for (;;) {
            // Tombstone frame (skipped work item) — advance without rendering
            if (!frame.ColorData || frame.OutputWidth == 0 || frame.OutputHeight == 0) {
                if (frame.IsFinal) {
                    nextExpectedSeqNum++;
                    break;
                }
            } else {
                RenderFrameToGL(*glContext, frame);

                // Return buffer to pool for reuse (GL has copied the data)
                ReleaseFrameBuffer(
                    std::move(frame.ColorData),
                    frame.BufferPixelCount);

                if (frame.IsFinal) {
                    // Draw perturbation overlay on final frames.
                    m_Fractal->DrawAllPerturbationResults(true);
                    nextExpectedSeqNum++;
                    break;
                }
            }

            // Block-wait for next frame for this exact sequence.
            // Workers guarantee at least one IsFinal (or tombstone) per
            // sequence, so this never blocks indefinitely.
            if (!m_FrameQueue.WaitForFrameExact(nextExpectedSeqNum)) {
                break; // Shutdown
            }
            if (!m_FrameQueue.TryPopNextInOrder(nextExpectedSeqNum, frame)) {
                continue; // Spurious wakeup — wait again
            }
        }
    }
}

bool RenderThreadPool::ProduceFrame(
    const RenderWorkItem &item,
    RendererIndex rendererIdx,
    ItersMemoryContainer &workerIters,
    bool isFinal) {

    Fractal *fractal = item.FractalPtr;
    auto &renderer = fractal->GetRenderer(rendererIdx);

    // Extract colors from GPU to a local buffer via RenderCurrent
    ReductionResults gpuReductionResults;

    const size_t outputWidth = workerIters.m_OutputWidth;
    const size_t outputHeight = workerIters.m_OutputHeight;
    const size_t totalPixels = outputWidth * outputHeight;

    if (totalPixels == 0) {
        return false;
    }

    // Safety net: if the renderer's dimensions don't match the worker's
    // container (e.g., due to a data race on m_CurIters during resize),
    // skip the frame to prevent buffer overflow in ExtractItersAndColors.
    if (renderer.GetWidth() != workerIters.m_Width ||
        renderer.GetHeight() != workerIters.m_Height) {
        return false;
    }

    // Allocate for the block-rounded color count so ExtractItersAndColors
    // (which copies N_color_cu elements) never overflows the buffer.
    const size_t roundedColorTotal = workerIters.m_RoundedOutputColorTotal;

    // Acquire reusable color buffer for this frame
    auto colorData = AcquireFrameBuffer(roundedColorTotal);

    if (item.Algorithm.UseLocalColor) {
        // CPU coloring path: copy from worker's color memory
        if (workerIters.m_RoundedOutputColorMemory) {
            memcpy(colorData.get(),
                   workerIters.m_RoundedOutputColorMemory.get(),
                   totalPixels * sizeof(Color16));
        }
    } else {
        // GPU coloring path: call RenderCurrent to extract colors.
        // Progressive (non-final) frames use the display stream to avoid
        // blocking behind compute kernels on the compute stream.
        const bool progressive = !isFinal;
        uint32_t result = 0;
        if (item.IterType == IterTypeEnum::Bits32) {
            uint32_t *iter = isFinal ? workerIters.GetIters<uint32_t>() : nullptr;
            result = renderer.RenderCurrent<uint32_t>(
                static_cast<uint32_t>(item.NumIterations),
                iter,
                colorData.get(),
                &gpuReductionResults,
                progressive);
        } else {
            uint64_t *iter = isFinal ? workerIters.GetIters<uint64_t>() : nullptr;
            result = renderer.RenderCurrent<uint64_t>(
                static_cast<uint64_t>(item.NumIterations),
                iter,
                colorData.get(),
                &gpuReductionResults,
                progressive);
        }

        if (result) {
            fractal->MessageBoxCudaError(result);
            return false;
        }

        result = progressive
            ? renderer.SyncDisplayStream()
            : renderer.SyncComputeStream();
        if (result) {
            fractal->MessageBoxCudaError(result);
            return false;
        }
    }

    RenderFrame frame;
    frame.SequenceNumber = item.SequenceNumber;
    frame.ColorData = std::move(colorData);
    frame.OutputWidth = outputWidth;
    frame.OutputHeight = outputHeight;
    frame.IsProgressive = !isFinal;
    frame.IsFinal = isFinal;
    frame.BufferPixelCount = roundedColorTotal;

    m_FrameQueue.Push(std::move(frame));
    return true;
}
