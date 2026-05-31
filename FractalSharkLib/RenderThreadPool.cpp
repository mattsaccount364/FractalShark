#include "stdafx.h"
#include "Environment.h"

// clang-format off
#include "GlIncludes.h"
// clang-format on

#include "Exceptions.h"
#include "Fractal.h"
#include "FractalPalette.h"
#include "GPU_Render.h"
#include "OpenGLContext.h"
#include "RenderThreadPool.h"

#include <algorithm>
#include <chrono>

// Convert iteration counts to Color16 values for CPU render algorithms.
// Mirrors the logic from Fractal::DrawFractalThread but writes to the
// ItersMemoryContainer's m_RoundedOutputColorMemory buffer.
static void
ColorizeCpuIterations(ItersMemoryContainer &iters,
                      const FractalPalette &palette,
                      IterTypeFull numIterations,
                      IterTypeEnum iterType,
                      uint32_t gpuAntialiasing)
{

    if (!iters.m_RoundedOutputColorMemory) {
        return;
    }

    const IterTypeFull maxPossibleIters = (iterType == IterTypeEnum::Bits32)
                                              ? static_cast<IterTypeFull>(INT32_MAX - 1)
                                              : static_cast<IterTypeFull>(INT64_MAX - 1);
    const uint32_t palIters = palette.GetCurrentNumColors();
    const Color16 *pal = palette.GetCurrentPalInterleaved();
    const size_t totalAA = static_cast<size_t>(gpuAntialiasing) * gpuAntialiasing;
    const auto paletteType = palette.GetPaletteType();
    const auto auxDepth = palette.GetAuxDepth();
    const auto paletteRotation = palette.GetPaletteRotation();

    size_t basicFactor = 65536 / numIterations;
    if (basicFactor == 0) {
        basicFactor = 1;
    }

    const size_t outW = iters.m_OutputWidth;
    const size_t outH = iters.m_OutputHeight;

    auto GetBasicColor = [&](size_t numIters, size_t &acc_r, size_t &acc_g, size_t &acc_b) {
        auto shiftedIters = (numIters >> auxDepth);
        if (paletteType != FractalPaletteType::Basic) {
            auto palIndex = shiftedIters % palIters;
            acc_r += pal[palIndex].r;
            acc_g += pal[palIndex].g;
            acc_b += pal[palIndex].b;
        } else {
            acc_r += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
            acc_g += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
            acc_b += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
        }
    };

    auto colorize = [&](auto **ItersArray, auto NumIterations) {
        const auto maxIters = NumIterations;

        auto writePixel =
            [&](size_t output_x, size_t output_y, size_t acc_r, size_t acc_g, size_t acc_b) {
                size_t idx = output_y * outW + output_x;
                iters.m_RoundedOutputColorMemory[idx].r = static_cast<uint16_t>(acc_r);
                iters.m_RoundedOutputColorMemory[idx].g = static_cast<uint16_t>(acc_g);
                iters.m_RoundedOutputColorMemory[idx].b = static_cast<uint16_t>(acc_b);
                iters.m_RoundedOutputColorMemory[idx].a = 65535;
            };

        if (gpuAntialiasing == 1) {
            for (size_t output_y = 0; output_y < outH; output_y++) {
                for (size_t output_x = 0; output_x < outW; output_x++) {
                    size_t acc_r = 0, acc_g = 0, acc_b = 0;
                    size_t numIters = ItersArray[output_y][output_x];
                    if (numIters < maxIters) {
                        numIters += paletteRotation;
                        if (numIters >= maxPossibleIters) {
                            numIters = maxPossibleIters - 1;
                        }
                        GetBasicColor(numIters, acc_r, acc_g, acc_b);
                    }
                    writePixel(output_x, output_y, acc_r, acc_g, acc_b);
                }
            }
        } else {
            for (size_t output_y = 0; output_y < outH; output_y++) {
                for (size_t output_x = 0; output_x < outW; output_x++) {
                    size_t acc_r = 0, acc_g = 0, acc_b = 0;
                    for (size_t iy = output_y * gpuAntialiasing; iy < (output_y + 1) * gpuAntialiasing;
                         iy++) {
                        for (size_t ix = output_x * gpuAntialiasing;
                             ix < (output_x + 1) * gpuAntialiasing;
                             ix++) {
                            size_t numIters = ItersArray[iy][ix];
                            if (numIters < maxIters) {
                                numIters += paletteRotation;
                                if (numIters >= maxPossibleIters) {
                                    numIters = maxPossibleIters - 1;
                                }
                                GetBasicColor(numIters, acc_r, acc_g, acc_b);
                            }
                        }
                    }
                    acc_r /= totalAA;
                    acc_g /= totalAA;
                    acc_b /= totalAA;
                    writePixel(output_x, output_y, acc_r, acc_g, acc_b);
                }
            }
        }
    };

    if (iterType == IterTypeEnum::Bits32) {
        colorize(iters.GetItersArray<uint32_t>(), static_cast<uint32_t>(numIterations));
    } else {
        colorize(iters.GetItersArray<uint64_t>(), static_cast<uint64_t>(numIterations));
    }
}

// ============================================================================
// RendererPool
// ============================================================================

RendererPool::RendererPool() : m_Renderers(nullptr), m_Count(0) {}

void
RendererPool::Initialize(GPURenderer *renderers, size_t count)
{
    m_Renderers = renderers;
    m_Count = count;
    m_Available.assign(count, true);
}

RendererIndex
RendererPool::Acquire()
{
    std::unique_lock lk(m_Mutex);
    m_CV.wait(lk, [this] {
        return std::any_of(m_Available.begin(), m_Available.end(), [](bool v) { return v; });
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

void
RendererPool::Release(RendererIndex idx)
{
    {
        std::lock_guard lk(m_Mutex);
        m_Available[static_cast<size_t>(idx)] = true;
    }
    m_CV.notify_one();
}

// ============================================================================
// RenderJobHandle
// ============================================================================

RenderJobHandle::RenderJobHandle(std::shared_future<void> future) : m_Future(std::move(future)) {}

void
RenderJobHandle::Wait()
{
    if (m_Future.valid()) {
        m_Future.wait();
    }
}

bool
RenderJobHandle::IsReady() const
{
    if (!m_Future.valid()) {
        return true;
    }
    return m_Future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

// ============================================================================
// FrameCompletionQueue
// ============================================================================

FrameCompletionQueue::FrameCompletionQueue() : m_ShutdownFlag(false) {}

void
FrameCompletionQueue::Push(RenderFrame frame)
{
    {
        std::lock_guard lk(m_Mutex);
        m_Frames.push_back(std::move(frame));
    }
    m_CV.notify_one();
}

bool
FrameCompletionQueue::TryPopNextInOrder(uint64_t expectedSeqNum, RenderFrame &frame)
{
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

bool
FrameCompletionQueue::WaitForFrame(uint64_t expectedSeqNum)
{
    std::unique_lock lk(m_Mutex);
    // Wake on the exact sequence or any newer frame.  If a sequence is
    // permanently lost (e.g. worker crash), newer frames will still wake
    // us so the consumer can skip forward rather than blocking forever.
    m_CV.wait(lk, [this, expectedSeqNum] {
        if (m_ShutdownFlag)
            return true;
        return std::any_of(m_Frames.begin(), m_Frames.end(), [expectedSeqNum](const RenderFrame &f) {
            return f.SequenceNumber >= expectedSeqNum;
        });
    });
    return !m_ShutdownFlag;
}

bool
FrameCompletionQueue::WaitForFrameExact(uint64_t seqNum)
{
    std::unique_lock lk(m_Mutex);
    m_CV.wait(lk, [this, seqNum] {
        if (m_ShutdownFlag || m_OverlayDirty)
            return true;
        return std::any_of(m_Frames.begin(), m_Frames.end(), [seqNum](const RenderFrame &f) {
            return f.SequenceNumber == seqNum;
        });
    });
    return !m_ShutdownFlag;
}

bool
FrameCompletionQueue::SkipToNextAvailable(uint64_t &expectedSeqNum)
{
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
    std::erase_if(m_Frames, [minSeq](const RenderFrame &f) { return f.SequenceNumber < minSeq; });

    expectedSeqNum = minSeq;
    return true;
}

void
FrameCompletionQueue::Shutdown()
{
    {
        std::lock_guard lk(m_Mutex);
        m_ShutdownFlag = true;
    }
    m_CV.notify_all();
}

void
FrameCompletionQueue::NotifyOverlay()
{
    {
        std::lock_guard lk(m_Mutex);
        m_OverlayDirty = true;
    }
    m_CV.notify_all();
}

bool
FrameCompletionQueue::ConsumeOverlayDirty()
{
    std::lock_guard lk(m_Mutex);
    bool wasDirty = m_OverlayDirty;
    m_OverlayDirty = false;
    return wasDirty;
}

// ============================================================================
// RenderThreadPool
// ============================================================================

RenderThreadPool::RenderThreadPool(Fractal *fractal, void *nativeWindow, bool hostOwnedGlPresentation)
    : m_Fractal(fractal), m_NativeWindow(nativeWindow), m_NextSequenceNumber(0), m_ShutdownFlag(false),
      m_HostOwnedGlPresentation(hostOwnedGlPresentation)
{

    m_RendererPool.Initialize(&fractal->GetRenderer(RendererIndex::Renderer0), NumRenderers);

    for (size_t i = 0; i < NumWorkers; ++i) {
        m_Workers.emplace_back(&RenderThreadPool::WorkerLoop, this, i);
    }

    if (!m_HostOwnedGlPresentation) {
        m_GlConsumerThread = std::thread(&RenderThreadPool::GlConsumerLoop, this);
    }
}

RenderThreadPool::~RenderThreadPool() { Shutdown(); }

std::unique_ptr<Color16[]>
RenderThreadPool::AcquireFrameBuffer(size_t totalPixels)
{
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

void
RenderThreadPool::ReleaseFrameBuffer(std::unique_ptr<Color16[]> buffer, size_t totalPixels)
{
    std::lock_guard lk(m_BufferPoolMutex);
    if (m_BufferPoolPixelCount != totalPixels) {
        // Dimension changed — discard old buffers
        m_BufferPool.clear();
        m_BufferPoolPixelCount = totalPixels;
    }
    m_BufferPool.push_back(std::move(buffer));
}

RenderJobHandle
RenderThreadPool::Enqueue(const RenderWorkItem &item)
{
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future().share();

    RenderWorkItem workItem = item;
    workItem.CompletionPromise = std::move(promise);

    std::vector<uint64_t> tombstoneSeqs;

    {
        std::lock_guard lk(m_WorkQueueMutex);
        workItem.SequenceNumber = m_NextSequenceNumber++;

        // Supersede: when a new render arrives, discard earlier queued
        // renders that it replaces (e.g., rapid zoom — only last matters).
        // Keep mutations and non-supersedable items (AutoZoomer pipeline).
        if (!workItem.MutationOnly) {
            for (auto it = m_WorkQueue.begin(); it != m_WorkQueue.end();) {
                if (it->Supersedable && !it->MutationOnly) {
                    tombstoneSeqs.push_back(it->SequenceNumber);
                    if (it->CompletionPromise) {
                        it->CompletionPromise->set_value();
                    }
                    it = m_WorkQueue.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // Stamp generation for in-flight superseding.
        // Only supersedable items advance the counter.
        if (workItem.Supersedable) {
            workItem.EnqueueGeneration = ++m_EnqueueGeneration;
        }

        m_WorkQueue.push_back(std::move(workItem));
    }

    // Push tombstones for superseded sequences outside the work queue
    // lock to avoid nested locking with the FrameQueue mutex.
    for (auto seq : tombstoneSeqs) {
        PushTombstone(seq);
    }

    m_WorkQueueCV.notify_one();

    return RenderJobHandle(std::move(future));
}

RenderWorkItem
RenderThreadPool::SnapshotCurrentState() const
{
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

RenderJobHandle
RenderThreadPool::EnqueueCommand(std::function<void(Fractal &)> cmd, bool supersedable)
{
    RenderWorkItem item{};
    item.Command = std::move(cmd);
    item.Supersedable = supersedable;
    item.FractalPtr = m_Fractal;
    item.ChangedWindow = true;
    item.ChangedScrn = true;
    item.ChangedIterations = true;
    return Enqueue(item);
}

RenderJobHandle
RenderThreadPool::EnqueueMutation(std::function<void(Fractal &)> cmd)
{
    auto promise = std::make_shared<std::promise<void>>();
    auto future = promise->get_future().share();

    RenderWorkItem workItem{};
    workItem.Command = std::move(cmd);
    workItem.MutationOnly = true;
    workItem.FractalPtr = m_Fractal;
    workItem.CompletionPromise = std::move(promise);
    // No sequence number — mutations don't produce frames.
    // Assigning one would create a gap that hangs the GL consumer
    // on WaitForFrameExact for a frame that never arrives.

    {
        std::lock_guard lk(m_WorkQueueMutex);
        m_WorkQueue.push_back(std::move(workItem));
    }
    m_WorkQueueCV.notify_one();

    return RenderJobHandle(std::move(future));
}

void
RenderThreadPool::PushTombstone(uint64_t sequenceNumber)
{
    RenderFrame skipFrame{};
    skipFrame.SequenceNumber = sequenceNumber;
    skipFrame.IsFinal = true;
    m_FrameQueue.Push(std::move(skipFrame));
}

void
RenderThreadPool::Shutdown()
{
    if (m_ShutdownFlag.exchange(true)) {
        return; // Already shut down
    }

    // Discard all queued work — exit takes priority.
    {
        std::lock_guard lk(m_WorkQueueMutex);
        for (auto &item : m_WorkQueue) {
            if (item.CompletionPromise) {
                item.CompletionPromise->set_value();
            }
        }
        m_WorkQueue.clear();
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

void
RenderThreadPool::Drain()
{
    std::unique_lock lk(m_DrainMutex);
    m_DrainCV.wait(lk, [this] {
        std::lock_guard qlk(m_WorkQueueMutex);
        return m_WorkQueue.empty() && m_InFlightCount.load() == 0;
    });
}

void
RenderThreadPool::SetDragRect(bool active, int x0, int y0, int x1, int y1)
{
    {
        std::lock_guard lk(m_DragRectMutex);
        m_DragRectActive = active;
        m_DragRectX0 = x0;
        m_DragRectY0 = y0;
        m_DragRectX1 = x1;
        m_DragRectY1 = y1;
    }

    // Wake the GL consumer for an overlay-only repaint.
    m_FrameQueue.NotifyOverlay();
}

void
RenderThreadPool::DrawDragRectOverlay(size_t frameHeight)
{
    bool active;
    int x0, y0, x1, y1;
    {
        std::lock_guard lk(m_DragRectMutex);
        active = m_DragRectActive;
        x0 = m_DragRectX0;
        y0 = m_DragRectY0;
        x1 = m_DragRectX1;
        y1 = m_DragRectY1;
    }

    if (!active || frameHeight == 0)
        return;

    // Flip Y from screen coords (top=0) to GL coords (bottom=0).
    const int h = static_cast<int>(frameHeight);
    const int glY0 = h - y0;
    const int glY1 = h - y1;

    glEnable(GL_COLOR_LOGIC_OP);
    glLogicOp(GL_INVERT);
    glDisable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glVertex2i(x0, glY0);
    glVertex2i(x1, glY0);
    glVertex2i(x1, glY1);
    glVertex2i(x0, glY1);
    glEnd();

    glLineWidth(1.0f);
    glDisable(GL_COLOR_LOGIC_OP);
}

bool
RenderThreadPool::DequeueWorkItem(RenderWorkItem &item)
{
    std::unique_lock lk(m_WorkQueueMutex);
    m_WorkQueueCV.wait(lk, [this] { return m_ShutdownFlag.load() || !m_WorkQueue.empty(); });

    // Exit immediately on shutdown — don't process remaining items.
    if (m_ShutdownFlag.load()) {
        return false;
    }

    item = std::move(m_WorkQueue.front());
    m_WorkQueue.pop_front();
    // Increment in-flight count while queue lock is held to prevent
    // Drain() TOCTOU: without this, Drain() could see empty queue +
    // zero in-flight between pop and the old increment site.
    m_InFlightCount.fetch_add(1);
    return true;
}

bool
RenderThreadPool::RunCalcFractal(Fractal *fractal,
                                 RenderWorkItem &item,
                                 RendererIndex rendererIdx,
                                 ItersMemoryContainer &workerIters)
{

    // Hold the lock for the full CalcFractal call.  CalcFractalTypedIter
    // calls ChangedMakeClean() at the end, which clears the shared flags.
    // Without the lock covering CalcFractal, Worker A's clean can wipe
    // Worker B's flags between set and read, causing B to skip computation
    // and produce a final frame from stale GPU data.
    std::lock_guard lk(m_CalcFractalMutex);

    // Execute pre-render command if present.
    if (item.Command) {
        item.Command(*fractal);

        // Re-snapshot all state after the command has mutated Fractal.
        auto fresh = SnapshotCurrentState();
        item.Ptz = fresh.Ptz;
        item.Algorithm = fresh.Algorithm;
        item.IterType = fresh.IterType;
        item.NumIterations = fresh.NumIterations;
        item.IterationPrecision = fresh.IterationPrecision;
        item.ScrnWidth = fresh.ScrnWidth;
        item.ScrnHeight = fresh.ScrnHeight;
        item.GpuAntialiasing = fresh.GpuAntialiasing;
        item.ChangedWindow = fresh.ChangedWindow;
        item.ChangedScrn = fresh.ChangedScrn;
        item.ChangedIterations = fresh.ChangedIterations;

        // Always re-acquire ItersMemoryContainer after command execution.
        // The command may have changed dimensions, antialiasing, iter type,
        // or anything else that invalidates the pre-acquired container.
        fractal->ReturnIterMemory(std::move(workerIters));
        workerIters = fractal->AcquireItersMemory();
    }

    // Dimension/AA check (covers both command and non-command items).
    if (workerIters.m_OutputWidth != item.ScrnWidth || workerIters.m_OutputHeight != item.ScrnHeight ||
        workerIters.m_Antialiasing != item.GpuAntialiasing) {
        return false;
    }

    fractal->m_ChangedWindow = item.ChangedWindow;
    fractal->m_ChangedScrn = item.ChangedScrn;
    fractal->m_ChangedIterations = item.ChangedIterations;

    // Ptz and ItersMemory travel through CalcContext — no swap needed.
    CalcContext ctx{item.Ptz, workerIters};
    fractal->CalcFractal(rendererIdx, false, ctx);
    return true;
}

void
RenderThreadPool::WaitForGpuAndProduceProgressiveFrames(Fractal *fractal,
                                                        GPURenderer &renderer,
                                                        const RenderWorkItem &item,
                                                        RendererIndex rendererIdx,
                                                        ItersMemoryContainer &workerIters,
                                                        std::mutex &workerMutex,
                                                        std::condition_variable &workerCV)
{

    if (fractal->m_BypassGpu) {
        return;
    }

    static constexpr auto ProgressiveDrawInterval = std::chrono::milliseconds(1000);

    renderer.ResetComputeDoneFlag();
    renderer.EnqueueComputeDoneCallback();

    for (;;) {
        std::unique_lock lk(workerMutex);
        auto computeDone = workerCV.wait_for(lk, ProgressiveDrawInterval, [&] {
            return renderer.IsComputeDone() || m_ShutdownFlag.load() || fractal->GetStopCalculating();
        });
        lk.unlock();

        if (m_ShutdownFlag.load() || fractal->GetStopCalculating()) {
            renderer.SyncComputeStream();
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

static uint32_t
NextPOT(uint32_t v)
{
    if (v == 0)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

static long long
ElapsedMilliseconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                 start)
        .count();
}

void
RenderThreadPool::RenderFrameToGL(OpenGlContext &glContext,
                                  const RenderFrame &frame,
                                  GLuint *persistTexOut)
{
    glContext.glResetViewDim(frame.OutputWidth, frame.OutputHeight);

    if (glContext.IsSoftwareRenderer()) {
        // GL 1.1-safe path: tile the frame into chunks that fit within
        // GL_MAX_TEXTURE_SIZE (1024 on GDI Generic), RGBA8, power-of-two.
        const GLint maxTex = glContext.GetMaxTextureSize();
        if (maxTex <= 0) {
            GlLog("RenderFrameToGL: maxTex <= 0, skipping");
            return;
        }

        // One-shot diagnostic on first frame.
        static bool loggedOnce = false;
        if (!loggedOnce) {
            loggedOnce = true;
            char buf[256];
            snprintf(buf,
                     sizeof(buf),
                     "RenderFrameToGL: software path, frame=%zux%zu, maxTex=%d",
                     frame.OutputWidth,
                     frame.OutputHeight,
                     maxTex);
            GlLog(buf);
        }

        constexpr size_t SoftwareTileCap = 1024;
        const size_t frameW = frame.OutputWidth;
        const size_t frameH = frame.OutputHeight;
        const size_t tileW = std::min(static_cast<size_t>(maxTex), SoftwareTileCap);
        const size_t tileH = std::min(static_cast<size_t>(maxTex), SoftwareTileCap);
        const Color16 *src = frame.ColorData.get();

        {
            static bool loggedTileConfig = false;
            if (!loggedTileConfig) {
                loggedTileConfig = true;
                char buf[256];
                snprintf(buf,
                         sizeof(buf),
                         "RenderFrameToGL: software tiles=%zux%zu, stagingBytes=%zu",
                         tileW,
                         tileH,
                         tileW * tileH * 4);
                GlLog(buf);
            }
        }

        GLuint texid;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glColor4f(1.f, 1.f, 1.f, 1.f);

        // Reusable buffer for the largest capped software tile.
        std::vector<uint8_t> rgba8(tileW * tileH * 4, 0);

        for (size_t tileY = 0; tileY < frameH; tileY += tileH) {
            for (size_t tileX = 0; tileX < frameW; tileX += tileW) {
                // Actual pixel extent of this tile (may be smaller at edges).
                const size_t tw = std::min(tileW, frameW - tileX);
                const size_t th = std::min(tileH, frameH - tileY);

                // Pad to POT, capped at maxTex.
                const uint32_t potW =
                    std::min(NextPOT(static_cast<uint32_t>(tw)), static_cast<uint32_t>(maxTex));
                const uint32_t potH =
                    std::min(NextPOT(static_cast<uint32_t>(th)), static_cast<uint32_t>(maxTex));

                // Convert this tile's Color16 region to RGBA8.
                // Zero-fill the buffer first for the POT padding region.
                std::fill(
                    rgba8.begin(), rgba8.begin() + static_cast<size_t>(potW) * potH * 4, uint8_t{0});

                for (size_t y = 0; y < th; ++y) {
                    for (size_t x = 0; x < tw; ++x) {
                        const size_t srcIdx = (tileY + y) * frameW + (tileX + x);
                        const size_t dstIdx = (y * potW + x) * 4;
                        rgba8[dstIdx + 0] = static_cast<uint8_t>(src[srcIdx].r >> 8);
                        rgba8[dstIdx + 1] = static_cast<uint8_t>(src[srcIdx].g >> 8);
                        rgba8[dstIdx + 2] = static_cast<uint8_t>(src[srcIdx].b >> 8);
                        rgba8[dstIdx + 3] = static_cast<uint8_t>(src[srcIdx].a >> 8);
                    }
                }

                // One-shot GL error check on first tile of first frame.
                static bool checkedError = false;
                const bool diagnoseFirstTile = !checkedError;
                std::chrono::steady_clock::time_point uploadStart;
                if (diagnoseFirstTile) {
                    checkedError = true;
                    GlLog("RenderFrameToGL: first software tile glTexImage2D begin");
                    uploadStart = std::chrono::steady_clock::now();
                }

                glTexImage2D(GL_TEXTURE_2D,
                             0,
                             GL_RGBA8,
                             static_cast<GLsizei>(potW),
                             static_cast<GLsizei>(potH),
                             0,
                             GL_RGBA,
                             GL_UNSIGNED_BYTE,
                             rgba8.data());

                if (diagnoseFirstTile) {
                    const GLenum err = glGetError();
                    char buf[192];
                    snprintf(buf,
                             sizeof(buf),
                             "RenderFrameToGL: first software tile glTexImage2D end, "
                             "elapsedMs=%lld, error=0x%x, potW=%u, potH=%u",
                             ElapsedMilliseconds(uploadStart),
                             err,
                             potW,
                             potH);
                    GlLog(buf);
                }

                // Tex coords: only the content region of the POT texture.
                const GLfloat texMaxS = static_cast<GLfloat>(tw) / static_cast<GLfloat>(potW);
                const GLfloat texMaxT = static_cast<GLfloat>(th) / static_cast<GLfloat>(potH);

                // Screen-space quad for this tile. GL ortho has bottom-left
                // origin, so Y increases upward. The frame data is stored
                // with row 0 at the top (screen convention), but glResetViewDim
                // sets gluOrtho2D(0, W, 0, H) with Y=0 at bottom. The existing
                // quad mapping flips Y (texCoord 0,0 maps to vertex 0,H).
                // Maintain the same flip per tile.
                const GLint x0 = static_cast<GLint>(tileX);
                const GLint x1 = static_cast<GLint>(tileX + tw);
                const GLint y0 = static_cast<GLint>(frameH - tileY - th); // bottom
                const GLint y1 = static_cast<GLint>(frameH - tileY);      // top

                std::chrono::steady_clock::time_point drawStart;
                if (diagnoseFirstTile) {
                    GlLog("RenderFrameToGL: first software tile quad draw begin");
                    drawStart = std::chrono::steady_clock::now();
                }

                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f);
                glVertex2i(x0, y1);
                glTexCoord2f(0.0f, texMaxT);
                glVertex2i(x0, y0);
                glTexCoord2f(texMaxS, texMaxT);
                glVertex2i(x1, y0);
                glTexCoord2f(texMaxS, 0.0f);
                glVertex2i(x1, y1);
                glEnd();

                if (diagnoseFirstTile) {
                    char buf[128];
                    snprintf(buf,
                             sizeof(buf),
                             "RenderFrameToGL: first software tile quad draw end, elapsedMs=%lld",
                             ElapsedMilliseconds(drawStart));
                    GlLog(buf);
                }
            }
        }

        // Software path always deletes — persistent texture not supported
        // (tiled approach overwrites a single texture per tile).
        glDeleteTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    } else {
        // Hardware path: single RGBA16 texture with NPOT dimensions.
        // If persistTexOut is set, keep the texture alive for overlay repaints.
        if (persistTexOut && *persistTexOut != 0) {
            glDeleteTextures(1, persistTexOut);
            *persistTexOut = 0;
        }

        GLuint texid;
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        static bool diagnosedFirstHardwareFrame = false;
        const bool diagnoseHardwareFrame = !diagnosedFirstHardwareFrame;
        std::chrono::steady_clock::time_point uploadStart;
        if (diagnoseHardwareFrame) {
            diagnosedFirstHardwareFrame = true;
            GlLog("RenderFrameToGL: first hardware glTexImage2D begin");
            uploadStart = std::chrono::steady_clock::now();
        }

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA16,
                     (GLsizei)frame.OutputWidth,
                     (GLsizei)frame.OutputHeight,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_SHORT,
                     frame.ColorData.get());

        if (diagnoseHardwareFrame) {
            const GLenum err = glGetError();
            char buf[160];
            snprintf(buf,
                     sizeof(buf),
                     "RenderFrameToGL: first hardware glTexImage2D end, elapsedMs=%lld, error=0x%x",
                     ElapsedMilliseconds(uploadStart),
                     err);
            GlLog(buf);
        }

        glColor4f(1.f, 1.f, 1.f, 1.f);

        std::chrono::steady_clock::time_point drawStart;
        if (diagnoseHardwareFrame) {
            GlLog("RenderFrameToGL: first hardware quad draw begin");
            drawStart = std::chrono::steady_clock::now();
        }

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f);
        glVertex2i(0, (GLint)frame.OutputHeight);
        glTexCoord2f(0.0f, 1.0f);
        glVertex2i(0, 0);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2i((GLint)frame.OutputWidth, 0);
        glTexCoord2f(1.0f, 0.0f);
        glVertex2i((GLint)frame.OutputWidth, (GLint)frame.OutputHeight);
        glEnd();

        if (diagnoseHardwareFrame) {
            char buf[128];
            snprintf(buf,
                     sizeof(buf),
                     "RenderFrameToGL: first hardware quad draw end, elapsedMs=%lld",
                     ElapsedMilliseconds(drawStart));
            GlLog(buf);
        }

        if (persistTexOut) {
            // Keep texture alive for caller to reuse.
            *persistTexOut = texid;
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_TEXTURE_2D);
        } else {
            glDeleteTextures(1, &texid);
            glBindTexture(GL_TEXTURE_2D, 0);
            glDisable(GL_TEXTURE_2D);
        }
    }
}

void
RenderThreadPool::RedrawLastTexture(OpenGlContext &glContext, GLuint texId, size_t width, size_t height)
{
    glContext.glResetViewDim(width, height);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texId);
    glColor4f(1.f, 1.f, 1.f, 1.f);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2i(0, (GLint)height);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2i(0, 0);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2i((GLint)width, 0);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2i((GLint)width, (GLint)height);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}

void
RenderThreadPool::WorkerLoop(size_t workerIndex)
{
    Environment::SetCurrentThreadName(std::format(L"RenderPool Worker {}", workerIndex).c_str());

    std::mutex workerMutex;
    std::condition_variable workerCV;

    while (!m_ShutdownFlag.load()) {
        RenderWorkItem item;
        if (!DequeueWorkItem(item)) {
            break;
        }

        // Mutation-only fast path: execute command under lock, no render.
        if (item.MutationOnly) {
            // Scope guard: always decrement in-flight count on exit,
            // even if the command throws (prevents Drain() deadlock).
            struct DecrementGuard {
                RenderThreadPool &pool;
                ~DecrementGuard()
                {
                    pool.m_InFlightCount.fetch_sub(1);
                    pool.m_DrainCV.notify_all();
                }
            } guard{*this};

            {
                std::lock_guard lk(m_CalcFractalMutex);
                if (item.Command) {
                    item.Command(*item.FractalPtr);
                }
            }
            if (item.CompletionPromise) {
                item.CompletionPromise->set_value();
            }
            continue;
        }

        // Skip stale in-flight renders: if a newer supersedable item was
        // enqueued since this one, this render's output will never be seen.
        // Check before acquiring renderer/memory (expensive).
        if (item.Supersedable && item.EnqueueGeneration < m_EnqueueGeneration) {
            PushTombstone(item.SequenceNumber);
            if (item.CompletionPromise) {
                item.CompletionPromise->set_value();
            }
            m_InFlightCount.fetch_sub(1);
            m_DrainCV.notify_all();
            continue;
        }

        RendererIndex rendererIdx = m_RendererPool.Acquire();
        Fractal *fractal = item.FractalPtr;
        auto &renderer = fractal->GetRenderer(rendererIdx);
        renderer.SetComputeDoneNotification(&workerMutex, &workerCV);
        ItersMemoryContainer workerIters = fractal->AcquireItersMemory();

        // Guarantee: every dequeued sequence gets at least one IsFinal frame.
        bool finalFramePushed = false;

        try {
            fractal->m_BenchmarkData.m_Overall.StartTimer();

            if (!RunCalcFractal(fractal, item, rendererIdx, workerIters)) {
                // One-shot log for first tombstone from RunCalcFractal failure.
                static bool loggedCalcFail = false;
                if (!loggedCalcFail) {
                    loggedCalcFail = true;
                    char buf[256];
                    snprintf(buf,
                             sizeof(buf),
                             "WorkerLoop: RunCalcFractal returned false, seq=%llu, "
                             "itersWH=%zux%zu, scrnWH=%zux%zu, aa=%u",
                             item.SequenceNumber,
                             workerIters.m_OutputWidth,
                             workerIters.m_OutputHeight,
                             item.ScrnWidth,
                             item.ScrnHeight,
                             item.GpuAntialiasing);
                    GlLog(buf);
                }
                PushTombstone(item.SequenceNumber);
                finalFramePushed = true;
            } else {
                // For CPU algorithms, convert iteration counts to colors.
                if (item.Algorithm.UseLocalColor) {
                    ColorizeCpuIterations(workerIters,
                                          fractal->GetPalette(),
                                          item.NumIterations,
                                          item.IterType,
                                          item.GpuAntialiasing);
                }

                WaitForGpuAndProduceProgressiveFrames(
                    fractal, renderer, item, rendererIdx, workerIters, workerMutex, workerCV);

                fractal->m_BenchmarkData.m_PerPixel.StopTimer();

                bool pushed = !m_ShutdownFlag.load() && !fractal->GetStopCalculating() &&
                              ProduceFrame(item, rendererIdx, workerIters, true);
                if (pushed) {
                    static bool loggedSuccess = false;
                    if (!loggedSuccess) {
                        loggedSuccess = true;
                        char buf[128];
                        snprintf(buf,
                                 sizeof(buf),
                                 "WorkerLoop: first frame produced, seq=%llu, "
                                 "%zux%zu",
                                 item.SequenceNumber,
                                 workerIters.m_OutputWidth,
                                 workerIters.m_OutputHeight);
                        GlLog(buf);
                    }
                } else {
                    static bool loggedPushFail = false;
                    if (!loggedPushFail) {
                        loggedPushFail = true;
                        char buf[128];
                        snprintf(buf,
                                 sizeof(buf),
                                 "WorkerLoop: ProduceFrame returned false or skipped, "
                                 "seq=%llu",
                                 item.SequenceNumber);
                        GlLog(buf);
                    }
                    PushTombstone(item.SequenceNumber);
                }
                finalFramePushed = true;
            }

            fractal->m_BenchmarkData.m_Overall.StopTimer();
        } catch (const FractalSharkSeriousException &e) {
            auto msg = e.GetCallstack("Worker thread exception during CalcFractal");
            std::cerr << msg << std::endl;
            GlLog(msg.c_str());
            if (Environment::IsDebuggerAttached())
                Environment::DebugBreakpoint();
        } catch (const std::exception &e) {
            std::cerr << "Worker thread exception during CalcFractal: " << e.what() << std::endl;
            char buf[512];
            snprintf(buf, sizeof(buf), "WorkerLoop: exception: %s", e.what());
            GlLog(buf);
            if (Environment::IsDebuggerAttached())
                Environment::DebugBreakpoint();
        }

        if (!finalFramePushed) {
            PushTombstone(item.SequenceNumber);
        }

        // Swap the worker's iteration data into m_CurIters so that
        // SaveCurrentFractal, AutoZoom analysis, FindInterestingLocation,
        // etc. see the most recently completed render.  Only swap if
        // m_CurIters has matching dimensions — otherwise the old container
        // would be discarded by ReturnIterMemory, shrinking the pool.
        if (finalFramePushed) {
            std::lock_guard lk(m_CalcFractalMutex);
            if (fractal->m_CurIters.m_OutputWidth == workerIters.m_OutputWidth &&
                fractal->m_CurIters.m_OutputHeight == workerIters.m_OutputHeight &&
                fractal->m_CurIters.m_Antialiasing == workerIters.m_Antialiasing) {
                std::swap(fractal->m_CurIters, workerIters);
            }
        }

        // Return the container to the pool
        fractal->ReturnIterMemory(std::move(workerIters));

        // Signal completion to any waiters
        if (item.CompletionPromise) {
            item.CompletionPromise->set_value();
        }

        // Release the renderer back to the pool
        m_RendererPool.Release(rendererIdx);

        m_InFlightCount.fetch_sub(1);
        m_DrainCV.notify_all();
    }
}

void
RenderThreadPool::GlConsumerLoop()
{
    Environment::SetCurrentThreadName(L"RenderPool GL Consumer");

    // Create OpenGL context for this thread
    auto glContext = std::make_unique<OpenGlContext>(m_NativeWindow);
    if (!glContext->IsValid()) {
        GlLog("GlConsumerLoop: OpenGL context creation FAILED, no rendering will occur");
        return;
    }

    {
        char buf[256];
        snprintf(buf,
                 sizeof(buf),
                 "GlConsumerLoop: context valid, software=%d, maxTex=%d",
                 glContext->IsSoftwareRenderer() ? 1 : 0,
                 glContext->GetMaxTextureSize());
        GlLog(buf);
    }

    uint64_t nextExpectedSeqNum = 0;

    // Persistent GL texture for overlay-only repaints (hardware path).
    // Instead of caching the full CPU-side Color16 buffer, we keep the
    // last-uploaded texture alive and redraw from it.
    GLuint lastTex = 0;
    size_t lastFrameWidth = 0;
    size_t lastFrameHeight = 0;
    const bool isSoftwareRenderer = glContext->IsSoftwareRenderer();

    while (!m_ShutdownFlag.load()) {
        if (!m_FrameQueue.WaitForFrameExact(nextExpectedSeqNum)) {
            break; // Shutdown
        }

        // Check if we woke for an overlay-only repaint (no new frame).
        bool overlayDirty = m_FrameQueue.ConsumeOverlayDirty();

        RenderFrame frame;
        if (!m_FrameQueue.TryPopNextInOrder(nextExpectedSeqNum, frame)) {
            // No new frame — must be overlay-only wakeup.
            if (overlayDirty && lastTex != 0) {
                RedrawLastTexture(*glContext, lastTex, lastFrameWidth, lastFrameHeight);
                m_Fractal->DrawAllPerturbationResults(true);
                DrawDragRectOverlay(lastFrameHeight);
                glContext->SwapBuffers();
            }
            continue;
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
                // Upload frame to GL. On hardware path, persist the texture.
                GLuint *persistPtr = isSoftwareRenderer ? nullptr : &lastTex;
                RenderFrameToGL(*glContext, frame, persistPtr);

                lastFrameWidth = frame.OutputWidth;
                lastFrameHeight = frame.OutputHeight;

                if (frame.IsFinal) {
                    m_Fractal->DrawAllPerturbationResults(true);
                    DrawDragRectOverlay(lastFrameHeight);

                    // Release CPU buffer — GPU texture persists for overlay repaints.
                    ReleaseFrameBuffer(std::move(frame.ColorData), frame.BufferPixelCount);

                    glContext->SwapBuffers();
                    nextExpectedSeqNum++;
                    break;
                }

                // Progressive (non-final) frame: draw rect overlay, release buffer, swap.
                DrawDragRectOverlay(lastFrameHeight);
                ReleaseFrameBuffer(std::move(frame.ColorData), frame.BufferPixelCount);
                glContext->SwapBuffers();
            }

            // Block-wait for next frame for this exact sequence.
            if (!m_FrameQueue.WaitForFrameExact(nextExpectedSeqNum)) {
                break; // Shutdown
            }
            if (!m_FrameQueue.TryPopNextInOrder(nextExpectedSeqNum, frame)) {
                // Could be overlay-only wakeup while waiting for next progressive frame.
                m_FrameQueue.ConsumeOverlayDirty();
                continue;
            }
        }
    }

    // Clean up persistent texture on shutdown.
    if (lastTex != 0) {
        glDeleteTextures(1, &lastTex);
    }
}

bool
RenderThreadPool::TryPresentTick(OpenGlContext &glContext)
{
    if (!m_HostOwnedGlPresentation) {
        return false;
    }

    bool rendered = false;

    // Drain every frame currently ready for the host's expected
    // sequence; only the most recently uploaded frame remains visible
    // after the eventual SwapBuffers.  This mirrors the inner loop of
    // GlConsumerLoop except it is non-blocking and does not swap.
    for (;;) {
        RenderFrame frame;
        if (!m_FrameQueue.TryPopNextInOrder(m_HostExpectedSeqNum, frame)) {
            break;
        }

        const bool tombstone = !frame.ColorData || frame.OutputWidth == 0 || frame.OutputHeight == 0;

        if (!tombstone) {
            static bool diagnosedFirstHostFrame = false;
            const bool diagnoseHostFrame = !diagnosedFirstHostFrame;
            std::chrono::steady_clock::time_point presentationStart;
            if (diagnoseHostFrame) {
                diagnosedFirstHostFrame = true;
                GlLog("TryPresentTick: first host frame begin");
                presentationStart = std::chrono::steady_clock::now();
            }

            RenderFrameToGL(glContext, frame);

            // Release the previously cached frame's buffer back to the
            // pool now that a newer one supersedes it, then keep the
            // new frame for overlay-only repaints.
            if (m_HostLastFrame.ColorData) {
                ReleaseFrameBuffer(std::move(m_HostLastFrame.ColorData),
                                   m_HostLastFrame.BufferPixelCount);
            }
            m_HostLastFrame = std::move(frame);
            rendered = true;

            if (m_HostLastFrame.IsFinal) {
                static bool diagnosedFirstPerturbationOverlay = false;
                const bool diagnosePerturbationOverlay = !diagnosedFirstPerturbationOverlay;
                std::chrono::steady_clock::time_point perturbationStart;
                if (diagnosePerturbationOverlay) {
                    diagnosedFirstPerturbationOverlay = true;
                    GlLog("TryPresentTick: first perturbation overlay begin");
                    perturbationStart = std::chrono::steady_clock::now();
                }

                m_Fractal->DrawAllPerturbationResults(true);

                if (diagnosePerturbationOverlay) {
                    char buf[128];
                    snprintf(buf,
                             sizeof(buf),
                             "TryPresentTick: first perturbation overlay end, elapsedMs=%lld",
                             ElapsedMilliseconds(perturbationStart));
                    GlLog(buf);
                }

                ++m_HostExpectedSeqNum;
                if (diagnoseHostFrame) {
                    char buf[128];
                    snprintf(buf,
                             sizeof(buf),
                             "TryPresentTick: first host frame end, elapsedMs=%lld",
                             ElapsedMilliseconds(presentationStart));
                    GlLog(buf);
                }
                continue;
            }

            if (diagnoseHostFrame) {
                char buf[128];
                snprintf(buf,
                         sizeof(buf),
                         "TryPresentTick: first host frame end, elapsedMs=%lld",
                         ElapsedMilliseconds(presentationStart));
                GlLog(buf);
            }
        } else if (frame.IsFinal) {
            ++m_HostExpectedSeqNum;
        }
    }

    return rendered;
}

bool
RenderThreadPool::RepresentLastFrame(OpenGlContext &glContext)
{
    if (!m_HostOwnedGlPresentation || !m_HostLastFrame.ColorData) {
        return false;
    }
    RenderFrameToGL(glContext, m_HostLastFrame);
    if (m_HostLastFrame.IsFinal) {
        m_Fractal->DrawAllPerturbationResults(true);
    }
    return true;
}

bool
RenderThreadPool::ProduceFrame(const RenderWorkItem &item,
                               RendererIndex rendererIdx,
                               ItersMemoryContainer &workerIters,
                               bool isFinal)
{

    Fractal *fractal = item.FractalPtr;
    auto &renderer = fractal->GetRenderer(rendererIdx);

    // Extract colors from GPU to a local buffer via RenderCurrent
    ReductionResults gpuReductionResults;

    const size_t outputWidth = workerIters.m_OutputWidth;
    const size_t outputHeight = workerIters.m_OutputHeight;
    const size_t totalPixels = outputWidth * outputHeight;

    if (totalPixels == 0) {
        GlLog("ProduceFrame: totalPixels == 0");
        return false;
    }

    // Allocate for the block-rounded color count so ExtractItersAndColors
    // (which copies N_color_cu elements) never overflows the buffer.
    const size_t roundedColorTotal = workerIters.m_RoundedOutputColorTotal;

    // Acquire reusable color buffer for this frame
    auto colorData = AcquireFrameBuffer(roundedColorTotal);

    if (item.Algorithm.UseLocalColor) {
        // CPU coloring path: copy from worker's color memory.
        // No renderer dimension check needed — this path doesn't use the GPU renderer.
        if (workerIters.m_RoundedOutputColorMemory) {
            memcpy(colorData.get(),
                   workerIters.m_RoundedOutputColorMemory.get(),
                   totalPixels * sizeof(Color16));
        }
    } else {
        // GPU coloring path: call RenderCurrent to extract colors.
        // Safety net: if the renderer's dimensions don't match the worker's
        // container (e.g., due to a data race on m_CurIters during resize),
        // skip the frame to prevent buffer overflow in ExtractItersAndColors.
        if (renderer.GetWidth() != workerIters.m_Width || renderer.GetHeight() != workerIters.m_Height) {
            static bool loggedMismatch = false;
            if (!loggedMismatch) {
                loggedMismatch = true;
                char buf[256];
                snprintf(buf,
                         sizeof(buf),
                         "ProduceFrame: dimension mismatch, renderer=%ux%u, "
                         "iters=%zux%zu",
                         renderer.GetWidth(),
                         renderer.GetHeight(),
                         workerIters.m_Width,
                         workerIters.m_Height);
                GlLog(buf);
            }
            return false;
        }

        // Progressive (non-final) frames use the display stream to avoid
        // blocking behind compute kernels on the compute stream.
        const bool progressive = !isFinal;
        uint32_t result = 0;
        if (item.IterType == IterTypeEnum::Bits32) {
            uint32_t *iter = isFinal ? workerIters.GetIters<uint32_t>() : nullptr;
            result = renderer.RenderCurrent<uint32_t>(static_cast<uint32_t>(item.NumIterations),
                                                      iter,
                                                      colorData.get(),
                                                      &gpuReductionResults,
                                                      progressive);
        } else {
            uint64_t *iter = isFinal ? workerIters.GetIters<uint64_t>() : nullptr;
            result = renderer.RenderCurrent<uint64_t>(static_cast<uint64_t>(item.NumIterations),
                                                      iter,
                                                      colorData.get(),
                                                      &gpuReductionResults,
                                                      progressive);
        }

        if (result) {
            fractal->MessageBoxCudaError(result);
            return false;
        }

        result = progressive ? renderer.SyncDisplayStream() : renderer.SyncComputeStream();
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
