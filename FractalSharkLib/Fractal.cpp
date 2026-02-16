#include "stdafx.h"

#include "Fractal.h"
// #include "CBitmapWriter.h"

#include "BLAS.h"

#include <fstream>
#include <iostream>
#include <psapi.h>
#include <thread>

#include "AutoZoomer.h"
#include "ATInfo.h"
#include "FeatureFinder.h"
#include "FeatureSummary.h"
#include "HDRFloatComplex.h"
#include "LAInfoDeep.h"
#include "LAReference.h"

#include "BenchmarkData.h"
#include "CudaDblflt.h"
#include "Exceptions.h"
#include "FeatureFinder.h"
#include "FractalViewPresets.h"
#include "PerturbationResults.h"
#include "PrecisionCalculator.h"
#include "RecommendedSettings.h"

#include <chrono>

Fractal::Fractal(int width, int height, HWND hWnd, bool UseSensoCursor, uint64_t commitLimitInBytes)
    : m_RefOrbit{*this, commitLimitInBytes}, m_CommitLimitInBytes{commitLimitInBytes}
{
    Initialize(width, height, hWnd, UseSensoCursor);
}

Fractal::~Fractal() { Uninitialize(); }

void
Fractal::Initialize(int width, int height, HWND hWnd, bool UseSensoCursor)
{
    m_BypassGpu = true;
    auto SetupCuda = [&]() {
        auto res = GPURenderer::TestCudaIsWorking();

        if (!res) {
            std::cerr << "CUDA initialization failed.  GPU rendering will be disabled.\n";
            MessageBoxCudaError(res);
            m_BypassGpu = true;
            return;
        }

        m_BypassGpu = false;
    };

    auto setupThread = std::make_unique<std::thread>(SetupCuda);

    // Create the control-key-down/mouse-movement-monitoring thread.
    if (hWnd != nullptr) {
        m_AbortThreadQuitFlag = false;
        m_UseSensoCursor = UseSensoCursor;
        m_hWnd = hWnd;

        DWORD threadID;
        m_CheckForAbortThread =
            (HANDLE)CreateThread(nullptr, 0, CheckForAbortThread, this, 0, &threadID);
        if (m_CheckForAbortThread == nullptr) {
        }
    } else {
        m_AbortThreadQuitFlag = false;
        m_UseSensoCursor = false;
        m_hWnd = nullptr;

        m_CheckForAbortThread = nullptr;
    }

    m_AsyncRenderThreadCommand = { AsyncRenderThreadCommand::State::Idle, RendererIndex::Renderer0 };
    m_AsyncRenderThreadFinish = false;
    m_AsyncRenderThread = std::make_unique<std::thread>(DrawAsyncGpuFractalThreadStatic, this);

    srand((unsigned int)time(nullptr));

    // Initialize the palette
    m_Palette.InitializeAllPalettes();

    for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        m_DrawThreads.emplace_back(std::make_unique<DrawThreadSync>(i, nullptr, m_DrawThreadAtomics));
    }

    for (size_t i = 0; i < m_DrawThreads.size(); i++) {
        auto thread = std::make_unique<std::thread>(DrawFractalThread, i, this);
        m_DrawThreads[i]->m_Thread = std::move(thread);
    }

    // This one needs to be done before setting up the view.
    setupThread->join();

    InitialDefaultViewAndSettings(width, height);

    // Allocate the iterations array.
    InitializeMemory();
}

void
Fractal::InitializeMemory()
{
    // Wait until anyone using any of this memory is done.
    m_FractalSavesInProgress.clear();

    // Set up new memory.
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    m_ItersMemoryStorage.clear();
    for (size_t i = 0; i < 3; i++) {
        const size_t total_aa = GetGpuAntialiasing();
        ItersMemoryContainer container(GetIterType(), m_ScrnWidth, m_ScrnHeight, total_aa);
        m_ItersMemoryStorage.push_back(std::move(container));
    }

    m_CurIters = std::move(m_ItersMemoryStorage.back());
    m_ItersMemoryStorage.pop_back();

    m_DrawThreadAtomics.resize(m_ScrnHeight);
    m_DrawOutBytes = std::make_unique<GLushort[]>(m_ScrnWidth * m_ScrnHeight * 4); // RGBA
}

uint32_t
Fractal::InitializeGPUMemory(RendererIndex idx, bool expectedReuse)
{
    if (RequiresUseLocalColor()) {
        return 0;
    }

    if (m_BypassGpu) {
        return 1;
    }

    auto &renderer = GetRenderer(idx);
    uint32_t res;
    if (GetIterType() == IterTypeEnum::Bits32) {
        res = renderer.InitializeMemory<uint32_t>((uint32_t)m_CurIters.m_Width,
                                             (uint32_t)m_CurIters.m_Height,
                                             (uint32_t)m_CurIters.m_Antialiasing,
                                             m_Palette.GetCurrentPalR(),
                                             m_Palette.GetCurrentPalG(),
                                             m_Palette.GetCurrentPalB(),
                                             m_Palette.GetCurrentNumColors(),
                                             m_Palette.GetAuxDepth(),
                                             expectedReuse);
    } else {
        res = renderer.InitializeMemory<uint64_t>((uint32_t)m_CurIters.m_Width,
                                             (uint32_t)m_CurIters.m_Height,
                                             (uint32_t)m_CurIters.m_Antialiasing,
                                             m_Palette.GetCurrentPalR(),
                                             m_Palette.GetCurrentPalG(),
                                             m_Palette.GetCurrentPalB(),
                                             m_Palette.GetCurrentNumColors(),
                                             m_Palette.GetAuxDepth(),
                                             expectedReuse);
    }

    if (res) {
        m_BypassGpu = true;
    }

    return res;
}

void
Fractal::ReturnIterMemory(ItersMemoryContainer &&to_return)
{
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    m_ItersMemoryStorage.push_back(std::move(to_return));
}

void
Fractal::SetCurItersMemory()
{
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    if (!m_ItersMemoryStorage.empty()) {
        m_CurIters = std::move(m_ItersMemoryStorage.back());
        m_ItersMemoryStorage.pop_back();
        return;
    }

    for (;;) {
        lock.unlock();
        Sleep(100);
        lock.lock();
        if (!m_ItersMemoryStorage.empty()) {
            m_CurIters = std::move(m_ItersMemoryStorage.back());
            m_ItersMemoryStorage.pop_back();
            return;
        }
    }
}

void
Fractal::Uninitialize(void)
{

    {
        std::lock_guard lk(m_AsyncRenderThreadMutex);
        m_AsyncRenderThreadCommand.state = AsyncRenderThreadCommand::State::Finish;
        m_AsyncRenderThreadFinish = true;
    }
    m_AsyncRenderThreadCV.notify_one();

    if (m_AsyncRenderThread != nullptr) {
        m_AsyncRenderThread->join();
    }

    CleanupThreads(true);

    // Get rid of the abort thread, but only if we actually used it.
    if (m_CheckForAbortThread != nullptr) {
        m_AbortThreadQuitFlag = true;
        if (WaitForSingleObject(m_CheckForAbortThread, INFINITE) != WAIT_OBJECT_0) {
            std::wcerr << L"Error waiting for abort thread!" << std::endl;
        }
    }

    for (auto &thread : m_DrawThreads) {
        {
            std::lock_guard lk(thread->m_DrawThreadMutex);
            thread->m_DrawThreadReady = true;
            thread->m_TimeToExit = true;
        }
        thread->m_DrawThreadCV.notify_one();
    }

    for (auto &thread : m_DrawThreads) {
        thread->m_Thread->join();
    }
}

//////////////////////////////////////////////////////////////////////////////
// Resets the dimensions of the screen to width x height.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::ResetDimensions(size_t width, size_t height, uint32_t gpu_antialiasing)
{
    if (width == MAXSIZE_T) {
        width = m_ScrnWidth;
    }
    if (height == MAXSIZE_T) {
        height = m_ScrnHeight;
    }

    if (gpu_antialiasing == UINT32_MAX) {
        gpu_antialiasing = m_GpuAntialiasing;
    }

    if (gpu_antialiasing > 4) {
        std::wcerr << L"You're doing it wrong.  4x is max == 16 samples per pixel" << std::endl;
        gpu_antialiasing = 4;
    }

    if (m_ScrnWidth != width || m_ScrnHeight != height || m_GpuAntialiasing != gpu_antialiasing) {
        m_ScrnWidth = width;
        m_ScrnHeight = height;
        m_GpuAntialiasing = gpu_antialiasing;

        m_ChangedScrn = true;

        if (m_ScrnWidth != 0 && m_ScrnHeight != 0) {
            SquareCurrentView();
            InitializeMemory();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Recenters the view to the coordinates specified.  Note that the coordinates
// are "calculator coordinates," not screen coordinates.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::SetPrecision()
{
    auto precInBits = GetPrecision();
    m_Ptz.SetPrecision(precInBits);
}

void
Fractal::SetPrecision(uint64_t prec)
{
    m_Ptz.SetPrecision(prec);
}

// void Fractal::SetPrecision(PointZoomBBConverter &ptz, bool requiresReuse) {
//     auto precInBits = GetPrecision(ptz, requiresReuse);
//     ptz.SetPrecision(precInBits);
// }

uint64_t
Fractal::GetPrecision(void) const
{
    return Fractal::GetPrecision(m_Ptz, m_RefOrbit.RequiresReuse());
}

uint64_t
Fractal::GetPrecision(const PointZoomBBConverter &ptz, bool requiresReuse)
{

    return PrecisionCalculator::GetPrecision(ptz, requiresReuse);
}

bool
Fractal::RecenterViewCalc(const PointZoomBBConverter &ptz)
{
    SaveCurPos();

    m_Ptz = ptz;
    m_RefOrbit.ResetGuess();

    SetPrecision();
    SquareCurrentView();

    m_ChangedWindow = true;
    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Recenters the view to the the given screen coordinates.  Translates the
// screen coordinates to calculator coordinates, using whatever dimensions were
// specified when the Fractal object was constructed or whatever coordinates
// were given to the last call to the Reset function.
//////////////////////////////////////////////////////////////////////////////
bool
Fractal::RecenterViewScreen(RECT rect)
{
    SaveCurPos();

    HighPrecision newMinX = XFromScreenToCalc(HighPrecision{rect.left});
    HighPrecision newMaxX = XFromScreenToCalc(HighPrecision{rect.right});
    HighPrecision newMinY = YFromScreenToCalc(HighPrecision{rect.top});
    HighPrecision newMaxY = YFromScreenToCalc(HighPrecision{rect.bottom});

    if (newMaxX < newMinX) {
        HighPrecision temp = newMinX;
        newMinX = newMaxX;
        newMaxX = temp;
    }

    if (rect.right < rect.left) {
        auto tempInt = rect.left;
        rect.left = rect.right;
        rect.right = tempInt;
    }

    if (newMaxY < newMinY) {
        HighPrecision temp = newMinY;
        newMinY = newMaxY;
        newMaxY = temp;
    }

    if (rect.bottom < rect.top) {
        auto tempInt = rect.top;
        rect.top = rect.bottom;
        rect.bottom = tempInt;
    }

    if (newMinX == newMaxX) {
        return false;
    }

    if (newMinY == newMaxY) {
        return false;
    }

    // By default, don't guess.
    m_RefOrbit.ResetGuess();

    if (GetRenderAlgorithm().RequiresReferencePoints) {
        auto lambda = [&](auto **ItersArray) {
            double geometricMeanX = 0;
            double geometricMeanSum = 0;
            double geometricMeanY = 0;

            RECT antiRect = rect;
            antiRect.left *= GetGpuAntialiasing();
            antiRect.right *= GetGpuAntialiasing();
            antiRect.top *= GetGpuAntialiasing();
            antiRect.bottom *= GetGpuAntialiasing();

            double totaliters = 0;
            for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                for (auto x = antiRect.left; x < antiRect.right; x++) {
                    totaliters += ItersArray[y][x];
                }
            }

            double avgiters =
                totaliters / ((antiRect.bottom - antiRect.top) * (antiRect.right - antiRect.left));

            for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                for (auto x = antiRect.left; x < antiRect.right; x++) {
                    if (ItersArray[y][x] < avgiters) {
                        continue;
                    }

                    double sq = (double)(ItersArray[y][x]) * (ItersArray[y][x]);
                    geometricMeanSum += sq;
                    geometricMeanX += sq * x;
                    geometricMeanY += sq * y;
                }
            }

            if (geometricMeanSum != 0) {
                double meanX = geometricMeanX / geometricMeanSum / GetGpuAntialiasing();
                double meanY = geometricMeanY / geometricMeanSum / GetGpuAntialiasing();
                // m_PerturbationGuessCalcX = meanX * (double)m_ScrnWidth / (double)(rect.right -
                // rect.left); m_PerturbationGuessCalcY = meanY * (double)m_ScrnHeight /
                // (double)(rect.bottom - rect.top);

                auto tempMeanX = HighPrecision{meanX};
                auto tempMeanY = HighPrecision{meanY};

                tempMeanX = XFromScreenToCalc(tempMeanX);
                tempMeanY = YFromScreenToCalc(tempMeanY);

                m_RefOrbit.ResetGuess(tempMeanX, tempMeanY);

                // assert(!std::isnan(m_PerturbationGuessCalcX));
                // assert(!std::isnan(m_PerturbationGuessCalcY));

                // if (m_PerturbationGuessCalcX >= m_ScrnWidth || m_PerturbationGuessCalcX < 0 ||
                // std::isnan(m_PerturbationGuessCalcX)) {
                //     m_PerturbationGuessCalcX = 0;
                // }

                // if (m_PerturbationGuessCalcY >= m_ScrnHeight || m_PerturbationGuessCalcY < 0 ||
                // std::isnan(m_PerturbationGuessCalcY)) {
                //     m_PerturbationGuessCalcY = 0;
                // }
            } else {
                // Do nothing.  This case can occur if we e.g. change dimension, change antialiasing
                // etc.
            }
        };

        if (GetIterType() == IterTypeEnum::Bits32) {
            lambda(m_CurIters.GetItersArray<uint32_t>());
        } else {
            lambda(m_CurIters.GetItersArray<uint64_t>());
        }
    }

    // Set m_PerturbationGuessCalc<X|Y> = ... above.

    m_Ptz = PointZoomBBConverter{newMinX, newMinY, newMaxX, newMaxY};

    SetPrecision();
    SquareCurrentView();

    m_ChangedWindow = true;
    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Recenters the view to the given point specified by x, y, which are screen
// cordinates.  These screen coordinates are translated to calculator coordinates
// for you.
//////////////////////////////////////////////////////////////////////////////
bool
Fractal::CenterAtPoint(size_t x, size_t y)
{
    const HighPrecision newCenterX = XFromScreenToCalc((HighPrecision)x);
    const HighPrecision newCenterY = YFromScreenToCalc((HighPrecision)y);
    auto ptz = m_Ptz.Recentered(newCenterX, newCenterY);
    return RecenterViewCalc(ptz);
}

void
Fractal::ZoomAtCenter(double factor)
{
    auto ptz = m_Ptz.ZoomedAtCenter(factor);
    RecenterViewCalc(ptz);
}

// This one recenters and zooms in on the mouse cursor.
// This one is better with the keyboard.
void
Fractal::ZoomRecentered(size_t scrnX, size_t scrnY, double factor)
{
    const HighPrecision calcX = XFromScreenToCalc((HighPrecision)scrnX);
    const HighPrecision calcY = YFromScreenToCalc((HighPrecision)scrnY);
    auto ptz = m_Ptz.ZoomedRecentered(calcX, calcY, factor);
    RecenterViewCalc(ptz);
}

// The idea is to zoom in on the mouse cursor, without also recentering it.
// This one is better with the wheel.
void
Fractal::ZoomTowardPoint(size_t scrnX, size_t scrnY, double factor)
{
    const HighPrecision calcX = XFromScreenToCalc((HighPrecision)scrnX);
    const HighPrecision calcY = YFromScreenToCalc((HighPrecision)scrnY);
    auto ptz = m_Ptz.ZoomedTowardPoint(calcX, calcY, factor);
    RecenterViewCalc(ptz);
}

void
Fractal::InitialDefaultViewAndSettings(int width, int height)
{
    // Note!  This specific setting is overridden in MainWindow via a hardcoded
    // commandDispatcher.Dispatch(IDM_ALG_AUTO); call
    const bool success = SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO));
    if (!success) {
        std::cerr << "Error: could not set default render algorithm." << std::endl;
    }

    SetIterationPrecision(1);

    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighSTMed);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed1);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed4);
    m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::Auto);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::STPeriodicity);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
    // m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::GPU);
    m_RefOrbit.ResetGuess();

    DefaultCompressionErrorExp(CompressionError::Low);
    DefaultCompressionErrorExp(CompressionError::Intermediate);
    if (width != 0 && height != 0) {
        ResetDimensions(width, height, 1);
    } else {
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1);
    }

    // SetIterType(IterTypeEnum::Bits64);
    SetIterType(IterTypeEnum::Bits32);

    SetResultsAutosave(AddPointOptions::EnableWithoutSave);
    // SetResultsAutosave(AddPointOptions::DontSave);
    LoadPerturbationOrbits();
    View(0);
    // View(5);
    // View(11);
    // View(14);
    // View(27); // extremely hard

    m_Palette.SetDefaults();
    UsePaletteType(FractalPaletteType::Default);
    UsePalette(8);

    // Make sure the screen is completely redrawn the first time.
    ChangedMakeDirty();

    // Doesn't do anything with the palette.
}

template <Fractal::AutoZoomHeuristic h>
void
Fractal::AutoZoom()
{
    AutoZoomer(*this).Run<h>();
}

template void Fractal::AutoZoom<Fractal::AutoZoomHeuristic::Default>();
template void Fractal::AutoZoom<Fractal::AutoZoomHeuristic::Max>();
template void Fractal::AutoZoom<Fractal::AutoZoomHeuristic::Feature>();

//////////////////////////////////////////////////////////////////////////////
// Resets the fractal to the standard view.
// Make sure the view is square on all monitors at all weird aspect ratios.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::View(size_t view, bool includeMsgBox)
{
    auto preset = GetViewPreset(
        view,
        DefaultIterations,
        DefaultCompressionExp[static_cast<size_t>(CompressionError::Low)],
        DefaultCompressionExp[static_cast<size_t>(CompressionError::Intermediate)]);

    if (includeMsgBox && !preset.warningMessage.empty()) {
        ::MessageBox(
            nullptr,
            preset.warningMessage.c_str(),
            L"Warning",
            MB_OK | MB_APPLMODAL | MB_ICONWARNING);
    }

    ResetDimensions(MAXSIZE_T, MAXSIZE_T, preset.gpuAntialiasing);
    SetCompressionErrorExp(CompressionError::Low, preset.compressionErrorExpLow);
    SetCompressionErrorExp(CompressionError::Intermediate, preset.compressionErrorExpIntermediate);
    SetIterType(preset.iterType);
    SetNumIterations<IterTypeFull>(preset.numIterations);

    if (preset.setLADefaultsMaxPerf) {
        m_LAParameters.SetDefaults(LAParameters::LADefaults::MaxPerf);
    }

    PointZoomBBConverter convert{preset.minX, preset.minY, preset.maxX, preset.maxY};
    RecenterViewCalc(convert);
}

//////////////////////////////////////////////////////////////////////////////
// This function will "square" the current view, and it will take into account the
// aspect ratio of the monitor!  Wowee!  Thus, the view really is "square" even if
// your resolution is 1024x768 or some other "nonsquare" resolution.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::SquareCurrentView(void)
{
    m_Ptz.SquareAspectRatio(m_ScrnWidth, m_ScrnHeight);
    m_ChangedWindow = true;
    CleanupThreads(false);
}

// Used to gradually approach a given target.
// Used for creating sequences of still images.
// The images can then be made into a movie!
void
Fractal::ApproachTarget(void)
{
    HighPrecision targetIters{100000};
    int numFrames = 1000;

    HighPrecision deltaXMin;
    HighPrecision deltaYMin;
    HighPrecision deltaXMax;
    HighPrecision deltaYMax;

    SaveCurPos();

    auto MinX = GetMinX();
    auto MinY = GetMinY();
    auto MaxX = GetMaxX();
    auto MaxY = GetMaxY();

    IterTypeFull baseIters = GetNumIterations<IterTypeFull>();
    HighPrecision incIters = (HighPrecision)((HighPrecision)targetIters - (HighPrecision)baseIters) /
                             (HighPrecision)numFrames;
    for (int i = 0; i < numFrames; i++) {
        auto curMinX = m_Ptz.GetMinX();
        auto curMinY = m_Ptz.GetMinY();
        auto curMaxX = m_Ptz.GetMaxX();
        auto curMaxY = m_Ptz.GetMaxY();

        deltaXMin = (MinX - curMinX) / HighPrecision{75.0};
        deltaYMin = (MinY - curMinY) / HighPrecision{75.0};
        deltaXMax = (MaxX - curMaxX) / HighPrecision{75.0};
        deltaYMax = (MaxY - curMaxY) / HighPrecision{75.0};

        m_Ptz = PointZoomBBConverter{
            curMinX + deltaXMin, curMinY + deltaYMin, curMaxX + deltaXMax, curMaxY + deltaYMax};

        {
            HighPrecision result = ((HighPrecision)incIters * (HighPrecision)i);
            SetNumIterations<IterTypeFull>(baseIters + Convert<HighPrecision, unsigned long>(result));
        }

        m_ChangedWindow = true;
        m_ChangedIterations = true;

        wchar_t temp[256], temp2[256];
        wsprintf(temp, L"output%04d", i);
        wcscpy(temp2, temp);

        if (Utilities::FileExists(temp2) == false) { // Create a placeholder file
            FILE *file = _wfopen(temp2, L"w+");
            if (file == nullptr) // Fail silently.
            {
                break;
            }

            fwrite(temp2, sizeof(char), 256, file);
            fclose(file);

            // Render the fractal.
            // Draw the progress on the screen as necessary
            CalcFractal(true);

            // Stop _before_ saving the image, otherwise we will get a
            // corrupt render mixed in with good ones.
            // It's corrupt because it's incomplete!
            if (m_StopCalculating == true) {
                _wunlink(temp2); // Delete placeholder
                break;
            }

            // Save images as necessary.  This will usually be the case,
            // as simply watching it render one frame after another is not
            // particularly exciting.
            int ret = 0;
            ret = SaveCurrentFractal(temp2, false);
            if (ret == 0) {
                break;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Returns to the previous view.
//////////////////////////////////////////////////////////////////////////////
bool
Fractal::Back(void)
{
    if (!m_PrevPtz.empty()) {
        m_Ptz = m_PrevPtz.back();
        SetPrecision();
        m_RefOrbit.ResetGuess();

        m_PrevPtz.pop_back();

        m_ChangedWindow = true;
        return true;
    }

    return false;
}

// Used for screen saver program.
// It is supposed to find an interesting spot to zoom in on.
// It works rather well as is...
void
Fractal::FindInterestingLocation(RECT *rect)
{
    int x, y;
    int xMin = 0, yMin = 0, xMax = 0, yMax = 0;
    int xMinFinal = 0, yMinFinal = 0, xMaxFinal = 0, yMaxFinal = 0;

    HighPrecision ratio = (HighPrecision)m_ScrnWidth / (HighPrecision)m_ScrnHeight;
    int sizeY = rand() % (m_ScrnHeight / 3) + 1;

    HighPrecision sizeXHigh = ((HighPrecision)sizeY * ratio);
    int sizeX = Convert<HighPrecision, int>(sizeXHigh);

    uint64_t numberAtMax, numberAtMaxFinal = 0;
    HighPrecision percentAtMax, percentAtMaxFinal{10000000.0};
    HighPrecision desiredPercent{(rand() % 75) / 100.0 + .10};

    int i;
    for (i = 0; i < 1000; i++) {
        x = rand() % (m_ScrnWidth - sizeX * 2) + sizeX;
        y = rand() % (m_ScrnHeight - sizeY * 2) + sizeY;

        xMin = x - sizeX;
        yMin = y - sizeY;
        xMax = x + sizeX;
        yMax = y + sizeY;

        numberAtMax = 0;
        for (y = yMin; y <= yMax; y++) {
            for (x = xMin; x <= xMax; x++) {
                if (numberAtMax > 4200000000) {
                    break;
                }

                numberAtMax += m_CurIters.GetItersArrayValSlow(x, y);
            }
        }

        percentAtMax = (HighPrecision)numberAtMax / ((HighPrecision)sizeX * (HighPrecision)sizeY *
                                                     (HighPrecision)GetNumIterations<IterTypeFull>());

        if (abs(percentAtMax - desiredPercent) < abs(percentAtMaxFinal - desiredPercent)) {
            numberAtMaxFinal = numberAtMax;
            percentAtMaxFinal = percentAtMax;
            xMinFinal = xMin;
            yMinFinal = yMin;
            xMaxFinal = xMax;
            yMaxFinal = yMax;
        }

        if (m_StopCalculating == true) {
            break;
        }
    }

    rect->left = xMinFinal;
    rect->top = yMinFinal;
    rect->right = xMaxFinal;
    rect->bottom = yMaxFinal;
}

//////////////////////////////////////////////////////////////////////////////
// Call this before modifying the calculator window.
// This allows the user to use the "Back" function to return to
// the previous coordinates.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::SaveCurPos(void)
{
    if (!m_Ptz.Degenerate()) {
        m_PrevPtz.push_back(m_Ptz);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Functions for dealing with the number of iterations
///////////////////////////////////////////////////////////////////////////////
template <typename IterType>
void
Fractal::SetNumIterations(IterTypeFull num)
{
    if (num <= GetMaxIterations<IterType>()) {
        m_NumIterations = num;
    } else {
        m_NumIterations = GetMaxIterations<IterType>();
    }

    m_ChangedIterations = true;
}

template void Fractal::SetNumIterations<uint32_t>(IterTypeFull);
template void Fractal::SetNumIterations<uint64_t>(IterTypeFull);

template <typename IterType>
IterType
Fractal::GetNumIterations(void) const
{
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        if (m_NumIterations > GetMaxIterations<IterType>()) {
            std::wcerr << L"Iteration limit exceeded somehow." << std::endl;
            m_NumIterations = GetMaxIterations<IterType>();
            return GetMaxIterations<IterType>();
        }
    } else {
        static_assert(std::is_same<IterType, uint64_t>::value, "!");
    }
    return (IterType)m_NumIterations;
}

template uint32_t Fractal::GetNumIterations<uint32_t>(void) const;
template uint64_t Fractal::GetNumIterations<uint64_t>(void) const;

IterTypeFull
Fractal::GetMaxIterationsRT() const
{
    if (GetIterType() == IterTypeEnum::Bits32) {
        return GetMaxIterations<uint32_t>();
    } else {
        return GetMaxIterations<uint64_t>();
    }
}

void
Fractal::SetIterType(IterTypeEnum type)
{
    if (m_IterType != type) {
        m_IterType = type;
        InitializeMemory();
    }
}

HighPrecision
Fractal::GetZoomFactor() const
{
    return m_Ptz.GetZoomFactor();
}

void
Fractal::SetPerturbationAlg(RefOrbitCalc::PerturbationAlg alg)
{
    m_RefOrbit.SetPerturbationAlg(alg);
}

RefOrbitCalc::PerturbationAlg
Fractal::GetPerturbationAlg() const
{
    return m_RefOrbit.GetPerturbationAlg();
}

void
Fractal::ClearPerturbationResults(RefOrbitCalc::PerturbationResultType type)
{
    m_RefOrbit.ClearPerturbationResults(type);
}

void
Fractal::SavePerturbationOrbits()
{
    m_RefOrbit.SaveAllOrbits();
}

void
Fractal::LoadPerturbationOrbits()
{
    ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    m_RefOrbit.LoadAllOrbits();
}

IterTypeEnum
Fractal::GetIterType() const
{
    //
    // Returns the current iteration type
    //

    return m_IterType;
}

void
Fractal::ResetNumIterations(void)
{
    //
    // Resets the number of iterations to the default value.
    //

    m_NumIterations = DefaultIterations;
    m_ChangedIterations = true;
}

RenderAlgorithm
Fractal::GetRenderAlgorithm() const
{
    //
    // Returns the render algorithm to use
    //

    if (m_RenderAlgorithm.Algorithm == RenderAlgorithmEnum::AUTO) {
        HighPrecision zoomFactor = GetZoomFactor();
        if (zoomFactor < HighPrecision{1e4}) {
            // A bit borderline at 3840x1600 x16 AA
            return GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32);
        } else if (zoomFactor < HighPrecision{1e9}) {
            // Safe, maybe even full LA is a little better but barely
            return GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO);
        } else if (zoomFactor < HighPrecision{1e34}) {
            // This seems to work at x16 AA 3840x1600
            // This 1e34 is the same as at RefOrbitCalc::LoadOrbitConstInternal
            return GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu1x32PerturbedLAv2);
        } else {
            // Falls apart at high iteration counts with a lot of LA
            return GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2);
        }
    } else {
        // User-selected forced algorithm selection
        return m_RenderAlgorithm;
    }
}

bool
Fractal::SetRenderAlgorithm(RenderAlgorithm alg)
{
    bool ret = true;
    if (m_BypassGpu) {
        std::cerr << "Bypassing GPU in effect: CPU-only render algorithms enforced." << std::endl;
        if (alg.Gpu == RequiresGpu::Yes) {
            std::cerr << "Bypassing GPU: Forcing CPU64 render algorithm." << std::endl;
            alg = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);
            ret = false;
        }
    }

    m_RenderAlgorithm = alg;
    m_RefOrbit.ResetLastUsedOrbit();
    return ret;
}

const char *
Fractal::GetRenderAlgorithmName() const
{
    return GetRenderAlgorithmName(GetRenderAlgorithm());
}

const char *
Fractal::GetRenderAlgorithmName(RenderAlgorithm alg)
{
    //
    // Returns the name of the render algorithm
    //

    return alg.AlgorithmStr;
}

const HighPrecision &
Fractal::GetCompressionError(enum class CompressionError err) const
{
    return m_CompressionError[static_cast<size_t>(err)];
}

int32_t
Fractal::GetCompressionErrorExp(enum class CompressionError err) const
{
    return m_CompressionExp[static_cast<size_t>(err)];
}

void
Fractal::IncCompressionError(enum class CompressionError err, int32_t amount)
{
    size_t errIndex = static_cast<size_t>(err);
    if (m_CompressionExp[errIndex] >= 1000) {
        return;
    }

    m_CompressionExp[errIndex] += amount;
    m_CompressionError[errIndex] = HighPrecision{10.0}.power(m_CompressionExp[errIndex]);
    ChangedMakeDirty();
}

void
Fractal::DecCompressionError(enum class CompressionError err, int32_t amount)
{
    size_t errIndex = static_cast<size_t>(err);
    if (m_CompressionExp[errIndex] <= 1) {
        return;
    }

    m_CompressionExp[errIndex] -= amount;
    m_CompressionError[errIndex] = HighPrecision{10.0}.power(m_CompressionExp[errIndex]);
    ChangedMakeDirty();
}

void
Fractal::SetCompressionErrorExp(enum class CompressionError err, int32_t CompressionExp)
{
    size_t errIndex = static_cast<size_t>(err);

    if (CompressionExp >= 1000) {
        CompressionExp = 1000;
    }

    m_CompressionExp[errIndex] = CompressionExp;
    m_CompressionError[errIndex] = HighPrecision{10.0}.power(m_CompressionExp[errIndex]);
    ChangedMakeDirty();
}

void
Fractal::DefaultCompressionErrorExp(enum class CompressionError err)
{
    size_t errIndex = static_cast<size_t>(err);
    m_CompressionExp[errIndex] = DefaultCompressionExp[errIndex];
    m_CompressionError[errIndex] = HighPrecision{10.0}.power(m_CompressionExp[errIndex]);
    ChangedMakeDirty();
}

///////////////////////////////////////////////////////////////////////////////
// Functions for drawing the fractal
///////////////////////////////////////////////////////////////////////////////
bool
Fractal::RequiresUseLocalColor() const
{
    //
    // Returns true if the current render algorithm requires the use of
    // CPU-based color implementation
    //

    switch (GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::CpuHigh:
        case RenderAlgorithmEnum::Cpu64:
        case RenderAlgorithmEnum::CpuHDR32:
        case RenderAlgorithmEnum::CpuHDR64:

        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:

        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
            return true;
        default:
            return false;
    }
}

void
Fractal::CalcFractal(bool drawFractal)
{
    CalcFractal(RendererIndex::Renderer0, drawFractal);
}

void
Fractal::CalcFractal(RendererIndex idx, bool drawFractal)
{
    // if (m_glContextAsync->GetRepaint() == false)
    //{
    //     DrawFractal(RendererIndex::Renderer0);
    //     return;
    // }

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_Overall);

    // Bypass this function if the screen is too small.
    if (m_ScrnHeight == 0 || m_ScrnWidth == 0) {
        return;
    }

    if (GetIterType() == IterTypeEnum::Bits32) {
        CalcFractalTypedIter<uint32_t>(idx, drawFractal);
    } else {
        CalcFractalTypedIter<uint64_t>(idx, drawFractal);
    }
}

template <typename IterType>
void
Fractal::CalcFractalTypedIter(RendererIndex idx, bool drawFractal)
{
    // Test crash dump
    //{volatile int x = 5;
    // volatile int z = 0;
    // volatile int y = x / z;
    //}

    // Reset the flag should it be set.
    m_StopCalculating = false;

    // Do nothing if nothing has changed
    if (ChangedIsDirty() == false || (m_glContextAsync && m_glContextAsync->GetRepaint() == false)) {
        DrawFractal(idx);
        return;
    }

    // Draw the local fractal.
    // Note: This accounts for "Auto" being selected via the GetRenderAlgorithm call.
    switch (GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::CpuHigh:
            CalcCpuHDR<IterType, HighPrecision, double>();
            break;
        case RenderAlgorithmEnum::CpuHDR32:
            CalcCpuHDR<IterType, HDRFloat<float>, float>();
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
            CalcCpuPerturbationFractalBLA<IterType, HDRFloat<float>, float>();
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, float, PerturbExtras::Disable>();
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, float, PerturbExtras::SimpleCompression>();
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, double, PerturbExtras::Disable>();
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, double, PerturbExtras::Disable>();
            break;
        case RenderAlgorithmEnum::Cpu64:
            CalcCpuHDR<IterType, double, double>();
            break;
        case RenderAlgorithmEnum::CpuHDR64:
            CalcCpuHDR<IterType, HDRFloat<double>, double>();
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
            CalcCpuPerturbationFractalBLA<IterType, double, double>();
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:
            CalcCpuPerturbationFractalBLA<IterType, HDRFloat<double>, double>();
            break;
        case RenderAlgorithmEnum::Gpu1x64:
            CalcGpuFractal<IterType, double>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x64:
            CalcGpuFractal<IterType, MattDbldbl>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu4x64:
            CalcGpuFractal<IterType, MattQDbldbl>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32:
            CalcGpuFractal<IterType, float>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32:
            CalcGpuFractal<IterType, MattDblflt>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu4x32:
            CalcGpuFractal<IterType, MattQFltflt>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32:
            CalcGpuFractal<IterType, HDRFloat<double>>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedScaled:
            CalcGpuPerturbationFractalScaledBLA<IterType, double, double, float, float>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedScaled:
            CalcGpuPerturbationFractalScaledBLA<IterType, HDRFloat<float>, float, float, float>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, double, double>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedScaled:
            // TODO
            // CalcGpuPerturbationFractalBLA<IterType, double, double>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, HDRFloat<float>, float>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, HDRFloat<double>, double>(idx, drawFractal);
            break;

            // LAV2

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(idx, drawFractal);
            break;
        default:
            break;
    }

    // We are all updated now.
    ChangedMakeClean();
}

void
Fractal::UsePaletteType(FractalPaletteType type)
{
    m_Palette.UsePaletteType(type);
    for (size_t i = 0; i < NumRenderers; i++) {
        auto err = InitializeGPUMemory(static_cast<RendererIndex>(i), true);
        if (err) {
            MessageBoxCudaError(err);
            return;
        }
    }
}

FractalPaletteType
Fractal::GetPaletteType() const
{
    return m_Palette.GetPaletteType();
}

uint32_t
Fractal::GetPaletteDepthFromIndex(size_t index) const
{
    return m_Palette.GetPaletteDepthFromIndex(index);
}

void
Fractal::UsePalette(int depth)
{
    m_Palette.UsePalette(depth);
    for (size_t i = 0; i < NumRenderers; i++) {
        auto err = InitializeGPUMemory(static_cast<RendererIndex>(i), true);
        if (err) {
            MessageBoxCudaError(err);
            return;
        }
    }
}

void
Fractal::UseNextPaletteDepth()
{
    m_Palette.UseNextPaletteDepth();
    for (size_t i = 0; i < NumRenderers; i++) {
        auto err = InitializeGPUMemory(static_cast<RendererIndex>(i), true);
        if (err) {
            MessageBoxCudaError(err);
            return;
        }
    }
}

void
Fractal::SetPaletteAuxDepth(int32_t depth)
{
    m_Palette.SetPaletteAuxDepth(depth);
    for (size_t i = 0; i < NumRenderers; i++) {
        auto err = InitializeGPUMemory(static_cast<RendererIndex>(i), true);
        if (err) {
            MessageBoxCudaError(err);
            return;
        }
    }
}

void
Fractal::UseNextPaletteAuxDepth(int32_t inc)
{
    m_Palette.UseNextPaletteAuxDepth(inc);
    for (size_t i = 0; i < NumRenderers; i++) {
        auto err = InitializeGPUMemory(static_cast<RendererIndex>(i), true);
        if (err) {
            MessageBoxCudaError(err);
            return;
        }
    }
}

uint32_t
Fractal::GetPaletteDepth() const
{
    return m_Palette.GetPaletteDepth();
}

void
Fractal::ResetFractalPalette(void)
{
    m_Palette.ResetPaletteRotation();
}

void
Fractal::RotateFractalPalette(int delta)
{
    m_Palette.RotatePalette(delta, GetMaxIterationsRT());
}

void
Fractal::CreateNewFractalPalette(void)
{
    m_Palette.CreateNewRandomPalette();
}

FractalPalette &
Fractal::GetPalette()
{
    return m_Palette;
}

//////////////////////////////////////////////////////////////////////////////
// Redraws the fractal using OpenGL calls.
// Note that coordinates here are weird, so we have to make a few tweaks to
// get the image oriented right side up. In particular, the line which reads:
//       glVertex2i (px, m_ScrnHeight - py);
//////////////////////////////////////////////////////////////////////////////
void
Fractal::DrawFractal(RendererIndex idx)
{
    {
        std::unique_lock lk(m_AsyncRenderThreadMutex);

        m_AsyncRenderThreadCV.wait(
            lk, [&] { return m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::Idle; });

        m_AsyncRenderThreadCommand = { AsyncRenderThreadCommand::State::Start, idx };
    }

    m_AsyncRenderThreadCV.notify_one();

    if (m_BypassGpu == false) {
        uint32_t result = GetRenderer(idx).SyncComputeStream();
        if (result) {
            MessageBoxCudaError(result);
        }
    } else {
        std::cerr << "Bypassing GPU in effect: No GPU synchronization performed." << std::endl;
    }

    {
        std::unique_lock lk(m_AsyncRenderThreadMutex);

        m_AsyncRenderThreadCV.wait(
            lk, [&] { return m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::Idle; });

        m_AsyncRenderThreadCommand.state = AsyncRenderThreadCommand::State::SyncDone;
    }

    m_AsyncRenderThreadCV.notify_one();

    {
        std::unique_lock lk(m_AsyncRenderThreadMutex);

        m_AsyncRenderThreadCV.wait(lk, [&] { return m_AsyncRenderThreadFinish; });

        m_AsyncRenderThreadFinish = false;
    }
}

template <typename IterType>
void
Fractal::DrawGlFractal(RendererIndex idx, bool LocalColor, bool lastIter)
{
    ReductionResults gpuReductionResults;

    if (LocalColor) {
        for (auto &it : m_DrawThreadAtomics) {
            it.store(0);
        }

        for (auto &thread : m_DrawThreads) {
            {
                std::lock_guard lk(thread->m_DrawThreadMutex);
                thread->m_DrawThreadProcessed = false;
                thread->m_DrawThreadReady = true;
            }
            thread->m_DrawThreadCV.notify_one();
        }

        for (auto &thread : m_DrawThreads) {
            std::unique_lock lk(thread->m_DrawThreadMutex);
            thread->m_DrawThreadCV.wait(lk, [&] { return thread->m_DrawThreadProcessed; });
        }

        // In case we need this we have it.
        // m_CurIters.GetReductionResults(localReductionResults);
    } else {
        IterType *iter = nullptr;
        if (lastIter) {
            iter = m_CurIters.GetIters<IterType>();
        }

        auto &renderer = GetRenderer(idx);
        auto result = renderer.RenderCurrent<IterType>(GetNumIterations<IterType>(),
                                                  iter,
                                                  m_CurIters.m_RoundedOutputColorMemory.get(),
                                                  &gpuReductionResults);

        if (result) {
            MessageBoxCudaError(result);
            return;
        }

        result = renderer.SyncStream(true);
        if (result) {
            MessageBoxCudaError(result);
            return;
        }
    }

    GLuint texid;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap
    // levels of a texture. glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // Change m_DrawOutBytes size if GL_RGBA is changed
    if (LocalColor) {
        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA16,
                     (GLsizei)m_ScrnWidth,
                     (GLsizei)m_ScrnHeight,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_SHORT,
                     m_DrawOutBytes.get());
    } else {
        // glTexImage2D(
        //     GL_TEXTURE_2D, 0, GL_RGBA16,
        //     (GLsizei)m_CurIters.m_RoundedOutputColorWidth,
        //     (GLsizei)m_CurIters.m_RoundedOutputColorHeight,
        //     0, GL_RGBA, GL_UNSIGNED_SHORT, m_CurIters.m_RoundedOutputColorMemory.get());

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA16,
                     (GLsizei)m_CurIters.m_OutputWidth,
                     (GLsizei)m_CurIters.m_OutputHeight,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_SHORT,
                     m_CurIters.m_RoundedOutputColorMemory.get());
    }

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2i(0, (GLint)m_ScrnHeight);
    glTexCoord2i(0, 1);
    glVertex2i(0, 0);
    glTexCoord2i(1, 1);
    glVertex2i((GLint)m_ScrnWidth, 0);
    glTexCoord2i(1, 0);
    glVertex2i((GLint)m_ScrnWidth, (GLint)m_ScrnHeight);
    glEnd();
    glFlush();
    glDeleteTextures(1, &texid);

    DrawAllPerturbationResults(true);

    // TODO it'd be nice to do something like these but the message loop is now
    // on the other (main) thread.   Hmm  maybe a mistake after all to have this
    // separate thread
    //{
    //    MSG msg;
    //    PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
    //}

    //// Throw away any messages
    //{
    //    MSG msg;
    //    while (PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE)) {
    //        GetMessage(&msg, nullptr, 0, 0);
    //    }
    //}

    // while (GetMessage(&msg, nullptr, 0, 0) > 0)
    //{
    //     TranslateMessage(&msg);
    //     DispatchMessage(&msg);
    // }
}

void
Fractal::SetRepaint(bool repaint)
{
    m_glContextAsync->SetRepaint(repaint);
}

bool
Fractal::GetRepaint() const
{
    return m_glContextAsync->GetRepaint();
}

void
Fractal::ToggleRepainting()
{
    m_glContextAsync->ToggleRepaint();
}

static inline void
DrawFilledCircle2i(int cx, int cy, int radius, int segments = 24)
{
    if (segments < 8)
        segments = 8;

    glBegin(GL_TRIANGLE_FAN);
    glVertex2i(cx, cy); // center

    // simple fan; last point repeats first for closure
    for (int i = 0; i <= segments; ++i) {
        const double a = (2.0 * 3.14159265358979323846 * i) / (double)segments;
        const int x = cx + (int)std::lround(std::cos(a) * (double)radius);
        const int y = cy + (int)std::lround(std::sin(a) * (double)radius);
        glVertex2i(x, y);
    }
    glEnd();
}

void
Fractal::DrawAllPerturbationResults(bool LeaveScreen)
{
    if (!LeaveScreen) {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBegin(GL_POINTS);
    m_RefOrbit.DrawPerturbationResults();
    glEnd();

    if (!m_FeatureSummaries.empty()) {

        glEnable(GL_COLOR_LOGIC_OP);
        glLogicOp(GL_INVERT);

        bool showLines = (m_FeatureSummaries.size() == 1);
        if (showLines) {
            glLineWidth(2.0f);

            glBegin(GL_LINES);
            for (auto &fsPtr : m_FeatureSummaries) {
                if (!fsPtr)
                    continue;

                int x0, y0, x1, y1;
                fsPtr->EstablishScreenCoordinates(*this);
                fsPtr->GetScreenCoordinates(x0, y0, x1, y1);

                glVertex2i(x0, y0);
                glVertex2i(x1, y1);
            }
            glEnd();

            glLineWidth(1.0f); // restore default (optional)
        } else {
            // Filled inverted circles around the FOUND point (line end)
            // Tweak radius as desired (in pixels)
            const int radiusPx = 6;
            const int segments = 28;

            for (auto &fsPtr : m_FeatureSummaries) {
                if (!fsPtr)
                    continue;

                int x0, y0, x1, y1;
                fsPtr->EstablishScreenCoordinates(*this);
                fsPtr->GetScreenCoordinates(x0, y0, x1, y1);

                // Use end point as "found" center
                DrawFilledCircle2i(x1, y1, radiusPx, segments);
            }
        }

        glDisable(GL_COLOR_LOGIC_OP);
    }

    glFlush();
}


void
Fractal::DrawFractalThread(size_t index, Fractal *fractal)
{
    SetThreadDescription(GetCurrentThread(), L"Fractal Draw Thread");
    DrawThreadSync &sync = *fractal->m_DrawThreads[index].get();

    constexpr size_t BytesPerPixel = 4;
    IterTypeFull maxPossibleIters;

    for (;;) {
        // Wait until main() sends data
        std::unique_lock lk(sync.m_DrawThreadMutex);

        sync.m_DrawThreadCV.wait(lk, [&] { return sync.m_DrawThreadReady; });

        if (sync.m_TimeToExit) {
            break;
        }

        maxPossibleIters = fractal->GetMaxIterationsRT();

        sync.m_DrawThreadReady = false;

        auto lambda = [&](auto **ItersArray, auto NumIterations) {
            size_t acc_r, acc_g, acc_b;
            size_t outputIndex = 0;

            const size_t totalAA = fractal->GetGpuAntialiasing() * fractal->GetGpuAntialiasing();
            uint32_t palIters = fractal->m_Palette.GetCurrentNumColors();
            const uint16_t *palR = fractal->m_Palette.GetCurrentPalR();
            const uint16_t *palG = fractal->m_Palette.GetCurrentPalG();
            const uint16_t *palB = fractal->m_Palette.GetCurrentPalB();
            size_t basicFactor = 65536 / NumIterations;
            if (basicFactor == 0) {
                basicFactor = 1;
            }

            const auto GetBasicColor =
                [&](size_t numIters, size_t &acc_r, size_t &acc_g, size_t &acc_b) {
                    auto shiftedIters = (numIters >> fractal->m_Palette.GetAuxDepth());

                    if (fractal->m_Palette.GetPaletteType() != FractalPaletteType::Basic) {
                        auto palIndex = shiftedIters % palIters;
                        acc_r += palR[palIndex];
                        acc_g += palG[palIndex];
                        acc_b += palB[palIndex];
                    } else {
                        acc_r += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
                        acc_g += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
                        acc_b += (shiftedIters * basicFactor) & ((1llu << 16) - 1);
                    }
                };

            const size_t maxIters = NumIterations;
            for (size_t output_y = 0; output_y < fractal->m_ScrnHeight; output_y++) {
                if (sync.m_DrawThreadAtomics[output_y] != 0) {
                    continue;
                }

                uint64_t expected = 0;
                if (sync.m_DrawThreadAtomics[output_y].compare_exchange_strong(expected, 1llu) ==
                    false) {
                    continue;
                }

                outputIndex = output_y * fractal->m_ScrnWidth * BytesPerPixel;

                for (size_t output_x = 0; output_x < fractal->m_ScrnWidth; output_x++) {
                    if (fractal->GetGpuAntialiasing() == 1) {
                        acc_r = 0;
                        acc_g = 0;
                        acc_b = 0;

                        const size_t input_x = output_x;
                        const size_t input_y = output_y;
                        size_t numIters = ItersArray[input_y][input_x];

                        if (numIters < maxIters) {
                            numIters += fractal->m_Palette.GetPaletteRotation();
                            if (numIters >= maxPossibleIters) {
                                numIters = maxPossibleIters - 1;
                            }

                            GetBasicColor(numIters, acc_r, acc_g, acc_b);
                        }
                    } else {
                        acc_r = 0;
                        acc_g = 0;
                        acc_b = 0;

                        for (size_t input_x = output_x * fractal->GetGpuAntialiasing();
                             input_x < (output_x + 1) * fractal->GetGpuAntialiasing();
                             input_x++) {
                            for (size_t input_y = output_y * fractal->GetGpuAntialiasing();
                                 input_y < (output_y + 1) * fractal->GetGpuAntialiasing();
                                 input_y++) {

                                size_t numIters = ItersArray[input_y][input_x];
                                if (numIters < maxIters) {
                                    numIters += fractal->m_Palette.GetPaletteRotation();
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
                    }

                    fractal->m_DrawOutBytes[outputIndex] = (GLushort)acc_r;
                    fractal->m_DrawOutBytes[outputIndex + 1] = (GLushort)acc_g;
                    fractal->m_DrawOutBytes[outputIndex + 2] = (GLushort)acc_b;
                    fractal->m_DrawOutBytes[outputIndex + 3] = 255;
                    outputIndex += 4;
                }
            }
        };

        if (fractal->GetIterType() == IterTypeEnum::Bits32) {
            lambda(fractal->m_CurIters.GetItersArray<uint32_t>(), fractal->GetNumIterations<uint32_t>());
        } else {
            lambda(fractal->m_CurIters.GetItersArray<uint64_t>(), fractal->GetNumIterations<uint64_t>());
        }

        sync.m_DrawThreadProcessed = true;
        lk.unlock();
        sync.m_DrawThreadCV.notify_one();
    }
}

void
Fractal::DrawAsyncGpuFractalThread()
{
    m_glContextAsync = std::make_unique<OpenGlContext>(m_hWnd);
    if (!m_glContextAsync->IsValid()) {
        return;
    }

    auto lambda = [&]<typename IterType>(RendererIndex rendererIdx, bool lastIter) -> bool {
        m_glContextAsync->glResetViewDim(m_ScrnWidth, m_ScrnHeight);

        if (m_glContextAsync->GetRepaint() == false) {
            m_glContextAsync->DrawGlBox();
            return false;
        }

        const bool LocalColor = RequiresUseLocalColor();
        DrawGlFractal<IterType>(rendererIdx, LocalColor, lastIter);
        return false;
    };

    for (;;) {
        RendererIndex rendererIdx;
        {
            std::unique_lock lk(m_AsyncRenderThreadMutex);

            m_AsyncRenderThreadCV.wait(lk, [&] {
                return m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::Start ||
                       m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::Finish;
            });

            if (m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::Finish) {
                break;
            }

            rendererIdx = m_AsyncRenderThreadCommand.rendererIdx;
            m_AsyncRenderThreadCommand.state = AsyncRenderThreadCommand::State::Idle;
        }

        m_AsyncRenderThreadCV.notify_one();

        for (size_t i = 0;; i++) {
            // Setting the timeout lower works fine but puts more load on the GPU
            static constexpr auto set_time = std::chrono::milliseconds(1000);

            std::unique_lock lk(m_AsyncRenderThreadMutex);
            auto doneearly = m_AsyncRenderThreadCV.wait_for(lk, set_time, [&] {
                return m_AsyncRenderThreadCommand.state == AsyncRenderThreadCommand::State::SyncDone;
            });

            m_AsyncRenderThreadCommand.state = AsyncRenderThreadCommand::State::Idle;

            bool err;

            if (GetIterType() == IterTypeEnum::Bits32) {
                err = lambda.template operator()<uint32_t>(rendererIdx, doneearly);
            } else {
                err = lambda.template operator()<uint64_t>(rendererIdx, doneearly);
            }

            if (err) {
                std::wcerr << L"Unexpected error in DrawAsyncGpuFractalThread 1" << std::endl;
                break;
            }

            if (doneearly) {
                break;
            }
        }

        {
            std::unique_lock lk(m_AsyncRenderThreadMutex);
            m_AsyncRenderThreadFinish = true;
        }

        m_AsyncRenderThreadCV.notify_one();
    }

    m_glContextAsync = nullptr;
}

void
Fractal::DrawAsyncGpuFractalThreadStatic(Fractal *fractal)
{
    fractal->DrawAsyncGpuFractalThread();
}

void
Fractal::FillCoord(const HighPrecision &src, MattQFltflt &dest)
{
    // qflt
    dest.x = Convert<HighPrecision, float>(src);
    dest.y = Convert<HighPrecision, float>(src - HighPrecision{dest.x});
    dest.z = Convert<HighPrecision, float>(src - HighPrecision{dest.x} - HighPrecision{dest.y});
    dest.w = Convert<HighPrecision, float>(src - HighPrecision{dest.x} - HighPrecision{dest.y} -
                                           HighPrecision{dest.z});
}

void
Fractal::FillCoord(const HighPrecision &src, MattQDbldbl &dest)
{
    // qdbl
    dest.x = Convert<HighPrecision, double>(src);
    dest.y = Convert<HighPrecision, double>(src - HighPrecision{dest.x});
    dest.z = Convert<HighPrecision, double>(src - HighPrecision{dest.x} - HighPrecision{dest.y});
    dest.w = Convert<HighPrecision, double>(src - HighPrecision{dest.x} - HighPrecision{dest.y} -
                                            HighPrecision{dest.z});
}

void
Fractal::FillCoord(const HighPrecision &src, MattDbldbl &dest)
{
    // dbl
    dest.head = Convert<HighPrecision, double>(src);
    dest.tail = Convert<HighPrecision, double>(src - HighPrecision{dest.head});
}

void
Fractal::FillCoord(const HighPrecision &src, double &dest)
{
    // doubleOnly
    dest = Convert<HighPrecision, double>(src);
}

void
Fractal::FillCoord(const HighPrecision &src, HDRFloat<float> &dest)
{
    // hdrflt
    dest = (HDRFloat<float>)src;
    HdrReduce(dest);
}

void
Fractal::FillCoord(const HighPrecision &src, HDRFloat<double> &dest)
{
    //  hdrdbl
    dest = (HDRFloat<double>)src;
    HdrReduce(dest);
}

void
Fractal::FillCoord(const HighPrecision &src, MattDblflt &dest)
{
    // flt
    dest.head = Convert<HighPrecision, float>(src);
    dest.tail = Convert<HighPrecision, float>(src - HighPrecision{dest.head});
}

void
Fractal::FillCoord(const HighPrecision &src, float &dest)
{
    // floatOnly
    dest = Convert<HighPrecision, float>(src);
}

void
Fractal::FillCoord(const HighPrecision &src, CudaDblflt<MattDblflt> &dest)
{
    double destDbl = Convert<HighPrecision, double>(src);
    dest = CudaDblflt(destDbl);
}

void
Fractal::FillCoord(const HighPrecision &src, HDRFloat<CudaDblflt<MattDblflt>> &dest)
{
    HDRFloat<CudaDblflt<MattDblflt>> destDbl(src);
    dest = destDbl;
}

template <class T>
void
Fractal::FillGpuCoords(T &cx2, T &cy2, T &dx2, T &dy2)
{
    HighPrecision src_dy = m_Ptz.GetDeltaY(m_ScrnHeight, GetGpuAntialiasing());
    HighPrecision src_dx = m_Ptz.GetDeltaX(m_ScrnWidth, GetGpuAntialiasing());

    FillCoord(m_Ptz.GetMinX(), cx2);
    FillCoord(m_Ptz.GetMinY(), cy2);
    FillCoord(src_dx, dx2);
    FillCoord(src_dy, dy2);
}

void
Fractal::TryFindPeriodicPoint(size_t scrnX, size_t scrnY, FeatureFinderMode mode)     {
    if (GetIterType() == IterTypeEnum::Bits32) {
        TryFindPeriodicPointIterType<uint32_t>(scrnX, scrnY, mode);
    } else {
        TryFindPeriodicPointIterType<uint64_t>(scrnX, scrnY, mode);
    }
}

template <typename IterType>
void
Fractal::TryFindPeriodicPointIterType(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    // Note: This accounts for "Auto" being selected via the GetRenderAlgorithm call.
    switch (GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::CpuHigh:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint. RenderAlgorithmEnum::CpuHigh. ");
            break;

        case RenderAlgorithmEnum::CpuHDR32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAHDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::CpuHDR64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::CpuHDR64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Cpu64PerturbedBLAHDR>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x64:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu2x64. ");
            break;

        case RenderAlgorithmEnum::Gpu4x64:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu4x64. ");
            break;

        case RenderAlgorithmEnum::Gpu1x32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu2x32. ");
            break;

        case RenderAlgorithmEnum::Gpu4x32:
            throw FractalSharkSeriousException(
                "Unsupported Render Algorithm for TryFindPeriodicPoint: RenderAlgorithmEnum::Gpu4x32. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx32:
            TryFindPeriodicPointTemplate<IterType,
                                         RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32>,
                                         PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedScaled:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedScaled>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedScaled:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedScaled>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedScaled:
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedScaled. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedBLA:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedBLA>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

            // LAV2

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO. ");
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>,
            //    PerturbExtras::Disable>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
            //TryFindPeriodicPointTemplate<
            //    IterType,
            //    RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>,
            //    PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            throw FractalSharkSeriousException("Unsupported Render Algorithm for TryFindPeriodicPoint: "
                                               "RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO. ");
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            TryFindPeriodicPointTemplate<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(scrnX, scrnY, mode);
            break;

        default:
            std::cerr << "Current render algorithm does not support feature finding.\n";
            break;
    }
}

template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
void
Fractal::TryFindPeriodicPointTemplate(size_t scrnX, size_t scrnY, FeatureFinderMode mode)
{
    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_FeatureFinder);

    using T = RenderAlg::MainType;
    using SubType = RenderAlg::SubType;
    constexpr PerturbExtras PExtrasLocal = PerturbExtras::Disable;

    auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtrasLocal>>();

    m_FeatureSummaries.clear();

    const bool scan = mode == FeatureFinderMode::DirectScan || mode == FeatureFinderMode::PTScan ||
                      mode == FeatureFinderMode::LAScan;

    // Base mode
    FeatureFinderMode baseMode = mode;
    if (mode == FeatureFinderMode::DirectScan)
        baseMode = FeatureFinderMode::Direct;
    if (mode == FeatureFinderMode::PTScan)
        baseMode = FeatureFinderMode::PT;
    if (mode == FeatureFinderMode::LAScan)
        baseMode = FeatureFinderMode::LA;

    const T radiusY{T{GetMaxY() - GetMinY()} / T{2.0f}};
    HighPrecision radius{radiusY};
    radius /= HighPrecision{12};

    auto RunOne = [&](size_t px, size_t py) {
        const HighPrecision cx = XFromScreenToCalc(HighPrecision(px));
        const HighPrecision cy = YFromScreenToCalc(HighPrecision(py));

        auto fs = std::make_unique<FeatureSummary>(cx, cy, radius, baseMode);

        bool found = false;

        if (baseMode == FeatureFinderMode::Direct) {
            found = featureFinder->FindPeriodicPoint(GetNumIterations<IterType>(), *fs);
        } else if (baseMode == FeatureFinderMode::PT) {
            auto *results =
                m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                 T,
                                                                 SubType,
                                                                 PExtrasLocal,
                                                                 RefOrbitCalc::Extras::None>(m_Ptz);
            RuntimeDecompressor<IterType, T, PExtrasLocal> decompressor(*results);

            found = featureFinder->FindPeriodicPoint(
                GetNumIterations<IterType>(), *results, decompressor, *fs);
        } else if (baseMode == FeatureFinderMode::LA) {
            auto *results =
                m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                 T,
                                                                 SubType,
                                                                 PExtrasLocal,
                                                                 RefOrbitCalc::Extras::IncludeLAv2>(
                    m_Ptz);
            RuntimeDecompressor<IterType, T, PExtrasLocal> decompressor(*results);

            found = featureFinder->FindPeriodicPoint(
                GetNumIterations<IterType>(), *results, decompressor, *results->GetLaReference(), *fs);
        }

        if (found) {
            fs->SetNumIterationsAtFind(GetNumIterations<IterTypeFull>());
            m_FeatureSummaries.emplace_back(std::move(fs));
        }
    };

    if (!scan) {
        RunOne(scrnX, scrnY);
    } else {
        constexpr size_t NX = 12;
        constexpr size_t NY = 12;

        const size_t W = GetRenderWidth();
        const size_t H = GetRenderHeight();

        for (size_t gy = 0; gy < NY; ++gy) {
            const size_t y = (H * (2 * gy + 1)) / (2 * NY);
            for (size_t gx = 0; gx < NX; ++gx) {
                const size_t x = (W * (2 * gx + 1)) / (2 * NX);
                RunOne(x, y);
            }
        }
    }

    if (m_FeatureSummaries.empty()) {
        std::cout << "No periodic points found.\n";
    } else {
        std::cout << "Found " << m_FeatureSummaries.size() << " periodic points.\n";
    }
}

void
Fractal::ClearAllFoundFeatures()
{
    m_FeatureSummaries.clear();
    m_ChangedWindow = true;
}

bool
Fractal::ZoomToFoundFeature(FeatureSummary &feature, const HighPrecision *zoomFactor)
{
    // If we only have a candidate, refine now
    if (feature.HasCandidate()) {
        ScopedBenchmarkStopper stopper(m_BenchmarkData.m_FeatureFinderHP);

        using T = HDRFloat<double>;
        using SubType = double;
        using IterType = uint64_t;
        constexpr PerturbExtras PExtras = PerturbExtras::Disable;

        auto featureFinder = std::make_unique<FeatureFinder<IterType, T, PExtras>>();

        // Acquire PT data if possible (same logic as TryFindPeriodicPoint)
        auto extras = RefOrbitCalc::Extras::None;
        if (auto *cand = feature.GetCandidate()) {
            if (cand->modeFoundBy == FeatureFinderMode::LA)
                extras = RefOrbitCalc::Extras::IncludeLAv2;
        }

        if (!featureFinder->RefinePeriodicPoint_HighPrecision(feature)) {
            return false;
        }
    }

    const size_t featurePrec = feature.GetPrecision();
    if (featurePrec > GetPrecision()) {
        SetPrecision(featurePrec);
    }

    if (zoomFactor) {
        const HighPrecision ptX = feature.GetFoundX();
        const HighPrecision ptY = feature.GetFoundY();

        PointZoomBBConverter ptz(ptX, ptY, *zoomFactor);
        if (ptz.Degenerate())
            return false;

        return RecenterViewCalc(ptz);
    }

    return true;
}

FeatureSummary *
Fractal::ChooseClosestFeatureToMouse() const
{
    if (m_FeatureSummaries.empty())
        return nullptr;

    // Mouse in client pixels
    POINT pt{};
    if (!::GetCursorPos(&pt))
        return nullptr;

    HWND hwnd = m_hWnd;
    if (!hwnd)
        return nullptr;

    if (!::ScreenToClient(hwnd, &pt))
        return nullptr;

    // Clamp to render bounds
    const int w = (int)GetRenderWidth();
    const int h = (int)GetRenderHeight();
    if (w <= 0 || h <= 0)
        return nullptr;

    if (pt.x < 0)
        pt.x = 0;
    if (pt.y < 0)
        pt.y = 0;
    if (pt.x >= w)
        pt.x = w - 1;
    if (pt.y >= h)
        pt.y = h - 1;

    // Convert mouse -> calc coords (same space as found feature coords)
    const HighPrecision mx = XFromScreenToCalc(HighPrecision{(int64_t)pt.x});
    const HighPrecision my = YFromScreenToCalc(HighPrecision{(int64_t)pt.y});

    FeatureSummary *best = nullptr;
    HighPrecision bestDist2{};
    bool haveBest = false;

    for (auto &fsPtr : m_FeatureSummaries) {
        if (!fsPtr)
            continue;

        FeatureSummary &fs = *fsPtr;

        const HighPrecision dx = fs.GetFoundX() - mx;
        const HighPrecision dy = fs.GetFoundY() - my;
        const HighPrecision dist2 = dx * dx + dy * dy;

        if (!haveBest || dist2 < bestDist2) {
            best = &fs;
            bestDist2 = dist2;
            haveBest = true;
        }
    }

    return best;
}

bool
Fractal::ZoomToFoundFeature()
{
    FeatureSummary *best = ChooseClosestFeatureToMouse();
    if (!best) {
        std::cerr << "No feature found to zoom to.\n";
        return false;
    }

    const HighPrecision z = best->ComputeZoomFactor(m_Ptz);
    return ZoomToFoundFeature(*best, &z);
}

template <typename IterType, class T>
void
Fractal::CalcGpuFractal(RendererIndex idx, bool drawFractal)
{
    T cx2{}, cy2{}, dx2{}, dy2{};
    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    uint32_t err = InitializeGPUMemory(idx, true);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    auto &renderer = GetRenderer(idx);
    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    err = renderer.Render(
        GetRenderAlgorithm(), cx2, cy2, dx2, dy2, GetNumIterations<IterType>(), m_IterationPrecision);

    if (err) {
        MessageBoxCudaError(err);
    }

    if (drawFractal) {
        DrawFractal(idx);
    }
}

template <typename IterType>
void
Fractal::CalcCpuPerturbationFractal()
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     double,
                                                                     double,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>(m_Ptz);

    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    HighPrecision hiDx{(maxX - minX) / HighPrecision{m_ScrnWidth * GetGpuAntialiasing()}};
    HighPrecision hiDy{(maxY - minY) / HighPrecision{m_ScrnHeight * GetGpuAntialiasing()}};
    double dx = Convert<HighPrecision, double>(hiDx);
    double dy = Convert<HighPrecision, double>(hiDy);

    double centerX = (double)(results->GetHiX() - minX);
    double centerY = (double)(results->GetHiY() - maxY);

    static constexpr size_t num_threads = std::thread::hardware_concurrency();
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    //
    // From
    // https://fractalforums.org/fractal-mathematics-and-new-theories/28/another-solution-to-perturbation-glitches/4360/msg29835#msg29835
    //

    // complex Reference[]; // Reference orbit (MUST START WITH ZERO)
    // IterType MaxRefIteration; // The last valid iteration of the reference (the iteration just before
    // it escapes) complex dz = 0, dc; // Delta z and Delta c IterType Iteration = 0, RefIteration = 0;

    // while (Iteration < MaxIteration) {
    //     dz = 2 * dz * Reference[RefIteration] + dz * dz + dc;
    //     RefIteration++;

    //    complex z = Reference[RefIteration] + dz;
    //    if (| z | > BailoutRadius) break;
    //    if (| z | < | dz| || RefIteration == MaxRefIteration) {
    //        dz = z;
    //        RefIteration = 0;
    //    }
    //    Iteration++;
    //}

    auto one_thread = [&]() {
        SetThreadDescription(GetCurrentThread(), L"CalcCpuPerturbationFractal thread");
        auto compressionHelper{
            std::make_unique<RuntimeDecompressor<IterType, double, PerturbExtras::Disable>>(*results)};

        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                IterType iter = 0;
                IterType RefIteration = 0;
                double deltaReal = dx * x - centerX;
                double deltaImaginary = -dy * y - centerY;

                double DeltaSub0X = deltaReal;
                double DeltaSub0Y = deltaImaginary;
                double DeltaSubNX = 0;
                double DeltaSubNY = 0;

                while (iter < GetNumIterations<IterType>()) {
                    const double DeltaSubNXOrig = DeltaSubNX;
                    const double DeltaSubNYOrig = DeltaSubNY;

                    //
                    // wrn = (2 * Xr + wr * s) * wr - (2 * Xi + wi * s) * wi + ur;
                    // win = 2 * ((Xr + wr * s) * wi + Xi * wr) + ui;
                    //     = 2 * (Xr * wi + wr * s * wi + Xi * wr) + ui;
                    //     = 2 * Xr * wi + 2 * wr * s * wi + 2 * Xi * wr + ui;
                    //
                    // https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html#a2021-05-14_deep_zoom_theory_and_practice_rescaling
                    //
                    // DeltaSubN = 2 * DeltaSubN * results.complex[RefIteration] + DeltaSubN * DeltaSubN
                    // + DeltaSub0; S * w = 2 * S * w * results.complex[RefIteration] + S * w * S * w + S
                    // * d
                    //
                    // S * (DeltaSubNWX + DeltaSubNWY I) = 2 * S * (DeltaSubNWX + DeltaSubNWY I) *
                    // (results.x[RefIteration] + results.y[RefIteration] * I) +
                    //                                     S * S * (DeltaSubNWX + DeltaSubNWY I) *
                    //                                     (DeltaSubNWX + DeltaSubNWY I) + S * (dX + dY
                    //                                     I)
                    //
                    // (DeltaSubNWX + DeltaSubNWY I) = 2 * (DeltaSubNWX + DeltaSubNWY I) *
                    // (results.x[RefIteration] + results.y[RefIteration] * I) +
                    //                                 S * (DeltaSubNWX + DeltaSubNWY I) * (DeltaSubNWX +
                    //                                 DeltaSubNWY I) + (dX + dY I)
                    // DeltaSubNWX = 2 * (DeltaSubNWX * results.x[RefIteration] - DeltaSubNWY *
                    // results.y[RefIteration]) +
                    //               S * (DeltaSubNWX * DeltaSubNWX - DeltaSubNWY * DeltaSubNWY) +
                    //               dX
                    // DeltaSubNWX = 2 * DeltaSubNWX * results.x[RefIteration] - 2 * DeltaSubNWY *
                    // results.y[RefIteration] +
                    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
                    //               dX
                    //
                    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY *
                    // results.x[RefIteration] +
                    //               S * DeltaSubNWX * DeltaSubNWY + S * DeltaSubNWY * DeltaSubNWX +
                    //               dY
                    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY *
                    // results.x[RefIteration] +
                    //               2 * S * DeltaSubNWX * DeltaSubNWY +
                    //               dY

                    const auto tempZComplex = results->GetComplex(*compressionHelper, RefIteration);
                    DeltaSubNX = DeltaSubNXOrig * (tempZComplex.getRe() * 2 + DeltaSubNXOrig) -
                                 DeltaSubNYOrig * (tempZComplex.getIm() * 2 + DeltaSubNYOrig) +
                                 DeltaSub0X;
                    DeltaSubNY = DeltaSubNXOrig * (tempZComplex.getIm() * 2 + DeltaSubNYOrig) +
                                 DeltaSubNYOrig * (tempZComplex.getRe() * 2 + DeltaSubNXOrig) +
                                 DeltaSub0Y;

                    ++RefIteration;

                    const auto tempZComplex = results->GetComplex(*compressionHelper, RefIteration);
                    const double tempZX = tempZComplex.getRe() + DeltaSubNX;
                    const double tempZY = tempZComplex.getIm() + DeltaSubNY;
                    const double zn_size = tempZX * tempZX + tempZY * tempZY;
                    const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                    if (zn_size > 256) {
                        break;
                    }

                    if (zn_size < normDeltaSubN ||
                        RefIteration == (IterType)results->GetCountOrbitEntries() - 1) {
                        DeltaSubNX = tempZX;
                        DeltaSubNY = tempZY;
                        RefIteration = 0;
                    }

                    ++iter;
                }

                m_CurIters.SetItersArrayValSlow(x, y, iter);
            }
        }
    };

    for (size_t cur_thread = 0; cur_thread < num_threads; cur_thread++) {
        threads.push_back(std::make_unique<std::thread>(one_thread));
    }

    for (size_t cur_thread = 0; cur_thread < threads.size(); cur_thread++) {
        threads[cur_thread]->join();
    }

    DrawFractal(RendererIndex::Renderer0);
}

//////////////////////////////////////////////////////////////////////////////
// old comment
// This function recalculates the local component of the fractal.  That is,
// it draws only those parts of the fractal delegated to the client computer.
// It does not calculate the parts of the fractal delegated to the remote computer.
// If we are not using network rendering, this function calculates and draws the
// entire fractal.
//////////////////////////////////////////////////////////////////////////////
template <typename IterType, class T, class SubType>
void
Fractal::CalcCpuHDR()
{
    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    const T dx = T((maxX - minX) / HighPrecision{m_ScrnWidth * GetGpuAntialiasing()});
    const T dy = T((maxY - minY) / HighPrecision{m_ScrnHeight * GetGpuAntialiasing()});

    const size_t num_threads = std::thread::hardware_concurrency();
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    const T Four{4};
    const T Two{2};

    auto one_thread = [&]() {
        SetThreadDescription(GetCurrentThread(), L"CalcCpuHDR thread");

        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            T cx = T{minX};

            // This is kind of kludgy.  We cast the integer y to a float.
            T cy = T{T{maxY} - dy * T{static_cast<float>(y)}};
            T zx, zy;
            T zx2, zy2;
            T sum;
            unsigned int i;

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
                // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
                zx = cx;
                zy = cy;
                for (i = 0; i < GetNumIterations<IterType>(); i++) { // x^2+2*I*x*y-y^2
                    zx2 = zx * zx;
                    zy2 = zy * zy;
                    sum = zx2 + zy2;
                    HdrReduce(sum);
                    if (HdrCompareToBothPositiveReducedGT(sum, Four))
                        break;

                    zy = Two * zx * zy;
                    zx = zx2 - zy2;

                    zx += cx;
                    zy += cy;

                    HdrReduce(zx);
                    HdrReduce(zy);
                }
                cx += dx;
                // HdrReduce(cx);

                m_CurIters.SetItersArrayValSlow(x, y, i);
            }
        }
    };

    for (size_t cur_thread = 0; cur_thread < num_threads; cur_thread++) {
        threads.push_back(std::make_unique<std::thread>(one_thread));
    }

    for (size_t cur_thread = 0; cur_thread < threads.size(); cur_thread++) {
        threads[cur_thread]->join();
    }

    DrawFractal(RendererIndex::Renderer0);
}

template <typename IterType, class T, class SubType>
void
Fractal::CalcCpuPerturbationFractalBLA()
{
    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>(m_Ptz);

    BLAS<IterType, T> blas(*results);
    blas.Init((IterType)results->GetCountOrbitEntries(), results->GetMaxRadius());

    T dx = T((maxX - minX) / HighPrecision{m_ScrnWidth * GetGpuAntialiasing()});
    HdrReduce(dx);
    T dy = T((maxY - minY) / HighPrecision{m_ScrnHeight * GetGpuAntialiasing()});
    HdrReduce(dy);

    T centerX = (T)(results->GetHiX() - minX);
    HdrReduce(centerX);
    T centerY = (T)(results->GetHiY() - maxY);
    HdrReduce(centerY);

    const size_t num_threads = std::thread::hardware_concurrency();
    ;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    auto one_thread = [&]() {
        SetThreadDescription(GetCurrentThread(), L"CalcCpuPerturbationFractalBLA thread");
        // T dzdcX = T(1);
        // T dzdcY = T(0);
        // bool periodicity_should_break = false;

        auto compressionHelper{
            std::make_unique<RuntimeDecompressor<IterType, T, PerturbExtras::Disable>>(*results)};

        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                IterType iter = 0;
                IterType RefIteration = 0;
                T deltaReal = dx * (SubType)x;
                HdrReduce(deltaReal);
                deltaReal -= centerX;

                T deltaImaginary = -dy * (SubType)y;
                HdrReduce(deltaImaginary);
                deltaImaginary -= centerY;

                HdrReduce(deltaReal);
                HdrReduce(deltaImaginary);

                T DeltaSub0X = deltaReal;
                T DeltaSub0Y = deltaImaginary;
                T DeltaSubNX = T(0);
                T DeltaSubNY = T(0);
                T DeltaNormSquared = T(0);

                while (iter < GetNumIterations<IterType>()) {
                    BLA<T> *b = nullptr;
                    while ((b = blas.LookupBackwards(RefIteration, DeltaNormSquared)) != nullptr) {
                        int l = b->getL();

                        // TODO this first RefIteration + l check bugs me
                        if (RefIteration + l >= (IterType)results->GetCountOrbitEntries()) {
                            std::wcerr << L"Out of bounds! :(" << std::endl;
                            break;
                        }

                        if (iter + l >= GetNumIterations<IterType>()) {
                            break;
                        }

                        iter += l;

                        // double t1 = (double)DeltaSubNX;
                        // double t2 = (double)DeltaSubNY;
                        // b->getValue(t1, t2, (double)DeltaSub0X, (double)DeltaSub0Y);
                        // DeltaSubNX = t1;
                        // DeltaSubNY = t2;

                        b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                        RefIteration += l;

                        const auto tempZComplex = results->GetComplex(*compressionHelper, RefIteration);
                        auto tempZX = tempZComplex.getRe() + DeltaSubNX;
                        auto tempZY = tempZComplex.getIm() + DeltaSubNY;
                        auto normSquared = tempZX * tempZX + tempZY * tempZY;
                        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                        HdrReduce(normSquared);
                        HdrReduce(DeltaNormSquared);

                        if (HdrCompareToBothPositiveReducedGT(normSquared, T(256))) {
                            break;
                        }

                        if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                            RefIteration >= (IterType)results->GetCountOrbitEntries() - 1) {
                            DeltaSubNX = tempZX;
                            DeltaSubNY = tempZY;
                            DeltaNormSquared = normSquared;
                            RefIteration = 0;
                        }
                    }

                    if (iter >= GetNumIterations<IterType>()) {
                        break;
                    }

                    const T DeltaSubNXOrig = DeltaSubNX;
                    const T DeltaSubNYOrig = DeltaSubNY;

                    const auto tempZComplex = results->GetComplex(*compressionHelper, RefIteration);
                    T TermB1 = DeltaSubNXOrig * (tempZComplex.getRe() * 2 + DeltaSubNXOrig);
                    T TermB2 = DeltaSubNYOrig * (tempZComplex.getIm() * 2 + DeltaSubNYOrig);

                    DeltaSubNX = TermB1 - TermB2;
                    DeltaSubNX += DeltaSub0X;
                    HdrReduce(DeltaSubNX);

                    T Term3 = tempZComplex.getIm() * 2 + DeltaSubNYOrig;
                    T Term4 = tempZComplex.getRe() * 2 + DeltaSubNXOrig;
                    DeltaSubNY = DeltaSubNXOrig * Term3 + DeltaSubNYOrig * Term4;
                    DeltaSubNY += DeltaSub0Y;
                    HdrReduce(DeltaSubNY);

                    ++RefIteration;
                    if (RefIteration >= (IterType)results->GetCountOrbitEntries()) {
                        std::wcerr << L"Out of bounds 2! :(" << std::endl;
                        break;
                    }

                    if constexpr (true) {
                        // template<typename real>
                        // Temp<real> Prepare(const std::complex<real> &dz, const std::complex<real> &dc)
                        // const {
                        //     Temp<real> temp;
                        //     temp.newdz = dz * ((real(2.0) * std::complex<real>(Ref)) + dz);
                        //     vreal ChebyMagNewdz = ChebyshevNorm(temp.newdz);
                        //     temp.unusable = (ChebyMagNewdz >= real(LAThreshold));
                        //     return temp;
                        // }
                        // DzdzStep = 2.0 * (dz + Ref) * ZCoeff;
                        // DzdzStepXY = 2.0 * (dzX + dzY * i + RefX + RefY * i) * (ZCoeffX + ZCoeffY *
                        // i); DzdzStepXY = 2.0 * ((dzX + RefX) + (dzY + RefY) * i) * (ZCoeffX + ZCoeffY
                        // * i); DzdzStepXY = 2.0 * ((dzX + RefX) * ZCoeffX + (dzX + RefX) * ZCoeffY * i
                        // + (dzY + RefY) * i * ZCoeffX + (dzY + RefY) * i * ZCoeffY * i); DzdzStepX
                        // = 2.0 * ((dzX + RefX) * ZCoeffX - (dzY + RefY) * ZCoeffY); DzdzStepY = 2.0 *
                        // ((dzX + RefX) * ZCoeffY + (dzY + RefY) * ZCoeffX); DzdzStepX = 2.0 * (dzX +
                        // RefX) * ZCoeffX - 2.0 * (dzY + RefY) * ZCoeffY; DzdzStepY = 2.0 * (dzX + RefX)
                        // * ZCoeffY + 2.0 * (dzY + RefY) * ZCoeffX);

                        // dzdz = dzdz * DzdzStep;
                        // dzdzXY = (dzdzX + dzdzY * i) * (DzdzStepX + DzdzStepY * i);
                        // dzdzXY = dzdzX * DzdzStepX + dzdzX * DzdzStepY * i + dzdzY * i * DzdzStepX +
                        // dzdzY * i * DzdzStepY * i; dzdzX = dzdzX * DzdzStepX - dzdzY * DzdzStepY;
                        // dzdzY = dzdzX * DzdzStepY + dzdzY * DzdzStepX;
                        //
                        // dzdc = dzdc * DzdzStep + complex(CCoeff) * ScalingFactor;
                        // dzdcXY = (dzdcX + dzdcY * i) * (DzdzStepX + DzdzStepY * i) + (CCoeffX +
                        // CCoeffY * i) * (ScalingFactorX + ScalingFactorY * i); dzdcXY = (dzdcX *
                        // DzdzStepX + dzdcX * DzdzStepY * i + dzdcY * i * DzdzStepX + dzdcY * i *
                        // DzdzStepY * i) + (CCoeffX * ScalingFactorX + CCoeffX * ScalingFactorY * i +
                        // CCoeffY * i * ScalingFactorX + CCoeffY * i * ScalingFactorY * i); dzdcX =
                        // (dzdcX * DzdzStepX - dzdcY * DzdzStepY) + (CCoeffX * ScalingFactorX - CCoeffY
                        // * ScalingFactorY); dzdcY = (dzdcX * DzdzStepY + dzdcY * DzdzStepX) + (CCoeffX
                        // * ScalingFactorY + CCoeffY * ScalingFactorX);
                        /////////////////////

                        // HdrReduce(dzdcX);
                        // auto dzdcX1 = HdrAbs(dzdcX);

                        // HdrReduce(dzdcY);
                        // auto dzdcY1 = HdrAbs(dzdcY);

                        // HdrReduce(zxCopy);
                        // auto zxCopy1 = HdrAbs(zxCopy);

                        // HdrReduce(zyCopy);
                        // auto zyCopy1 = HdrAbs(zyCopy);

                        // T n2 = std::max(zxCopy1, zyCopy1);

                        // T r0 = std::max(dzdcX1, dzdcY1);
                        // T maxRadiusHdr{ 3840 };
                        // auto n3 = maxRadiusHdr * r0 * HighTwo;
                        // HdrReduce(n3);

                        // if (n2 < n3) {
                        //     periodicity_should_break = true;
                        // }
                        // else {
                        //     auto dzdcXOrig = dzdcX;
                        //     dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                        //     dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
                        // }
                    }

                    const auto tempZComplex2 = results->GetComplex(*compressionHelper, RefIteration);
                    T tempZX = tempZComplex2.getRe() + DeltaSubNX;
                    T tempZY = tempZComplex2.getIm() + DeltaSubNY;
                    T nT1 = tempZX * tempZX;
                    T nT2 = tempZY * tempZY;
                    T normSquared = nT1 + nT2;
                    HdrReduce(normSquared);

                    DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                    HdrReduce(DeltaNormSquared);

                    if (HdrCompareToBothPositiveReducedGT(normSquared, T(256))) {
                        break;
                    }

                    if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                        RefIteration >= (IterType)results->GetCountOrbitEntries() - 1) {
                        DeltaSubNX = tempZX;
                        DeltaSubNY = tempZY;
                        DeltaNormSquared = normSquared;
                        RefIteration = 0;
                    }

                    ++iter;
                }

                m_CurIters.SetItersArrayValSlow(x, y, iter);
            }
        }
    };

    for (size_t cur_thread = 0; cur_thread < num_threads; cur_thread++) {
        threads.push_back(std::make_unique<std::thread>(one_thread));
    }

    for (size_t cur_thread = 0; cur_thread < threads.size(); cur_thread++) {
        threads[cur_thread]->join();
    }

    DrawFractal(RendererIndex::Renderer0);
}

template <typename IterType, class SubType, PerturbExtras PExtras>
void
Fractal::CalcCpuPerturbationFractalLAV2()
{
    using T = HDRFloat<SubType>;
    using TComplex = HDRFloatComplex<SubType>;
    auto *results =
        m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                         T,
                                                         SubType,
                                                         PExtras,
                                                         RefOrbitCalc::Extras::IncludeLAv2>(m_Ptz);

    if (results->GetLaReference() == nullptr || results->GetOrbitData() == nullptr) {
        std::wcerr << L"Oops - a null pointer deref" << std::endl;
        return;
    }

    auto &LaReference = *results->GetLaReference();

    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    T dx = T((maxX - minX) / HighPrecision(m_ScrnWidth * GetGpuAntialiasing()));
    HdrReduce(dx);
    T dy = T((maxY - minY) / HighPrecision(m_ScrnHeight * GetGpuAntialiasing()));
    HdrReduce(dy);

    T centerX = (T)(results->GetHiX() - minX);
    HdrReduce(centerX);
    T centerY = (T)(results->GetHiY() - maxY);
    HdrReduce(centerY);

    const size_t num_threads = std::thread::hardware_concurrency();
    ;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    auto one_thread = [&]() {
        SetThreadDescription(GetCurrentThread(), L"CalcCpuPerturbationFractalLAV2 thread");

        auto compressionHelper{std::make_unique<RuntimeDecompressor<IterType, T, PExtras>>(*results)};

        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                IterType BLA2SkippedIterations;

                BLA2SkippedIterations = 0;

                TComplex DeltaSub0;
                TComplex DeltaSubN;

                T deltaReal = dx * (SubType)x;
                HdrReduce(deltaReal);
                deltaReal -= centerX;

                T deltaImaginary = -dy * (SubType)y;
                HdrReduce(deltaImaginary);
                deltaImaginary -= centerY;

                HdrReduce(deltaReal);
                HdrReduce(deltaImaginary);

                DeltaSub0 = {deltaReal, deltaImaginary};
                DeltaSubN = {0, 0};

                if (LaReference.IsValid() && LaReference.UseAT() &&
                    LaReference.GetAT().isValid(DeltaSub0)) {
                    ATResult<IterType, T, SubType> res;
                    LaReference.GetAT().PerformAT(GetNumIterations<IterType>(), DeltaSub0, res);
                    BLA2SkippedIterations = res.bla_iterations;
                    DeltaSubN = res.dz;
                }

                IterType iterations = 0;
                IterType RefIteration = 0;
                IterType MaxRefIteration = (IterType)results->GetCountOrbitEntries() - 1;

                iterations = BLA2SkippedIterations;

                TComplex complex0{deltaReal, deltaImaginary};

                if (iterations != 0 && RefIteration < MaxRefIteration) {
                    complex0 =
                        results->GetComplex<SubType>(*compressionHelper, RefIteration) + DeltaSubN;
                } else if (iterations != 0 && results->GetPeriodMaybeZero() != 0) {
                    RefIteration = RefIteration % results->GetPeriodMaybeZero();
                    complex0 =
                        results->GetComplex<SubType>(*compressionHelper, RefIteration) + DeltaSubN;
                }

                auto CurrentLAStage = LaReference.IsValid() ? LaReference.GetLAStageCount() : 0;

                while (CurrentLAStage > 0) {
                    CurrentLAStage--;

                    auto LAIndex = LaReference.getLAIndex(CurrentLAStage);

                    if (LaReference.isLAStageInvalid(LAIndex, DeltaSub0)) {
                        continue;
                    }

                    auto MacroItCount = LaReference.getMacroItCount(CurrentLAStage);
                    auto j = RefIteration;

                    while (iterations < GetNumIterations<IterType>()) {
                        auto las = LaReference.getLA(LAIndex,
                                                     DeltaSubN,
                                                     (IterType)j,
                                                     (IterType)iterations,
                                                     GetNumIterations<IterType>());

                        if (las.unusable) {
                            RefIteration = las.nextStageLAindex;
                            break;
                        }

                        iterations += las.step;
                        DeltaSubN = las.Evaluate(DeltaSub0);
                        complex0 = las.getZ(DeltaSubN);
                        j++;

                        auto lhs = complex0.chebychevNorm();
                        HdrReduce(lhs);
                        auto rhs = DeltaSubN.chebychevNorm();
                        HdrReduce(rhs);

                        if (HdrCompareToBothPositiveReducedLT(lhs, rhs) || j >= MacroItCount) {
                            DeltaSubN = complex0;
                            j = 0;
                        }
                    }

                    if (iterations >= GetNumIterations<IterType>()) {
                        break;
                    }
                }

                T normSquared{};

                if (iterations < GetNumIterations<IterType>()) {
                    normSquared = complex0.norm_squared();
                }

                for (; iterations < GetNumIterations<IterType>(); iterations++) {
                    auto curIter = results->GetComplex<SubType>(*compressionHelper, RefIteration);
                    curIter = curIter * T(2);
                    curIter = curIter + DeltaSubN;
                    DeltaSubN = DeltaSubN * curIter;
                    DeltaSubN = DeltaSubN + DeltaSub0;
                    HdrReduce(DeltaSubN);

                    RefIteration++;

                    complex0 =
                        results->GetComplex<SubType>(*compressionHelper, RefIteration) + DeltaSubN;
                    HdrReduce(complex0);

                    normSquared = complex0.norm_squared();
                    HdrReduce(normSquared);

                    auto DeltaNormSquared = DeltaSubN.norm_squared();
                    HdrReduce(DeltaNormSquared);

                    if (HdrCompareToBothPositiveReducedGT(normSquared, T(256))) {
                        break;
                    }

                    if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                        (RefIteration >= MaxRefIteration)) {
                        DeltaSubN = complex0;
                        RefIteration = 0;
                    }
                }

                m_CurIters.SetItersArrayValSlow(x, y, iterations);
            }
        }
    };

    for (size_t cur_thread = 0; cur_thread < num_threads; cur_thread++) {
        threads.push_back(std::make_unique<std::thread>(one_thread));
    }

    for (size_t cur_thread = 0; cur_thread < threads.size(); cur_thread++) {
        threads[cur_thread]->join();
    }

    DrawFractal(RendererIndex::Renderer0);
}

template <typename IterType, class T, class SubType>
void
Fractal::CalcGpuPerturbationFractalBLA(RendererIndex idx, bool drawFractal)
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>(m_Ptz);

    uint32_t err = InitializeGPUMemory(idx, true);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    auto &renderer = GetRenderer(idx);
    renderer.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->GetHiX() - m_Ptz.GetMinX();
    HighPrecision centerY = results->GetHiY() - m_Ptz.GetMaxY();

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    GPUPerturbResults<IterType, T, PerturbExtras::Disable> gpu_results{
        (IterType)results->GetCompressedOrUncompressedOrbitSize(),
        results->GetCountOrbitEntries(),
        results->GetOrbitXLow(),
        results->GetOrbitYLow(),
        results->GetOrbitData(),
        results->GetPeriodMaybeZero()};

    uint32_t result;

    BLAS<IterType, T> blas(*results);
    blas.Init(results->GetCountOrbitEntries(), results->GetMaxRadius());

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    result = renderer.RenderPerturbBLA<IterType, T>(GetRenderAlgorithm(),
                                               &gpu_results,
                                               &blas,
                                               cx2,
                                               cy2,
                                               dx2,
                                               dy2,
                                               centerX2,
                                               centerY2,
                                               GetNumIterations<IterType>(),
                                               m_IterationPrecision);

    if (drawFractal) {
        DrawFractal(idx);
    }

    if (result) {
        MessageBoxCudaError(result);
    }
}

template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
void
Fractal::CalcGpuPerturbationFractalLAv2(RendererIndex idx, bool drawFractal)
{

    using T = RenderAlg::MainType;
    using SubType = RenderAlg::SubType;
    constexpr LAv2Mode Mode = RenderAlg::LAv2;

    using ConditionalT = typename DoubleTo2x32Converter<T, SubType>::ConditionalT;
    using ConditionalSubType = typename DoubleTo2x32Converter<T, SubType>::ConditionalSubType;

    constexpr auto RefOrbitMode = (Mode == LAv2Mode::Full || Mode == LAv2Mode::LAO)
                                      ? RefOrbitCalc::Extras::IncludeLAv2
                                      : RefOrbitCalc::Extras::None;

    const PerturbationResults<IterType, T, PExtras> *results =
        m_RefOrbit.GetUsefulPerturbationResults<IterType,
                                                ConditionalT,
                                                ConditionalSubType,
                                                PExtras,
                                                RefOrbitMode,
                                                T>();

    // TODO pass perturb results via InitializeGPUMemory
    // Currently: keep this up here so it frees existing memory
    // before generating the new orbit.
    auto err = InitializeGPUMemory(idx, results != nullptr);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                               ConditionalT,
                                                               ConditionalSubType,
                                                               PExtras,
                                                               RefOrbitMode,
                                                               T>(m_Ptz);

    // Reference orbit is always required for LAv2
    // The LaReference is not required when running perturbation only
    if ((RefOrbitMode == RefOrbitCalc::Extras::IncludeLAv2 && results->GetLaReference() == nullptr) ||
        results->GetOrbitData() == nullptr) {
        std::wcerr << L"Oops - a null pointer deref" << std::endl;
        return;
    }

    GPUPerturbResults<IterType, T, PExtras> gpu_results{
        (IterType)results->GetCompressedOrUncompressedOrbitSize(),
        results->GetCountOrbitEntries(),
        results->GetOrbitXLow(),
        results->GetOrbitYLow(),
        results->GetOrbitData(),
        results->GetPeriodMaybeZero()};

    auto &renderer = GetRenderer(idx);
    err = renderer.InitializePerturb<IterType, T, SubType, PExtras, T>(
        results->GetGenerationNumber(), &gpu_results, 0, nullptr, results->GetLaReference());
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    renderer.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->GetHiX() - m_Ptz.GetMinX();
    HighPrecision centerY = results->GetHiY() - m_Ptz.GetMaxY();

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    auto result = renderer.RenderPerturbLAv2<IterType, T, SubType, Mode, PExtras>(
        GetRenderAlgorithm(), cx2, cy2, dx2, dy2, centerX2, centerY2, GetNumIterations<IterType>());

    if (result) {
        MessageBoxCudaError(result);
        return;
    }

    if (drawFractal) {
        DrawFractal(idx);
    }
}

template <typename IterType, class T, class SubType, class T2, class SubType2>
void
Fractal::CalcGpuPerturbationFractalScaledBLA(RendererIndex idx, bool drawFractal)
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Bad,
                                                                     RefOrbitCalc::Extras::None>(m_Ptz);
    auto *results2 =
        m_RefOrbit
            .CopyUsefulPerturbationResults<IterType, T, PerturbExtras::Bad, T2, PerturbExtras::Bad>(
                *results);

    uint32_t err = InitializeGPUMemory(idx, true);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    auto &renderer = GetRenderer(idx);
    renderer.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->GetHiX() - m_Ptz.GetMinX();
    HighPrecision centerY = results->GetHiY() - m_Ptz.GetMaxY();

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    GPUPerturbResults<IterType, T, PerturbExtras::Bad> gpu_results{
        (IterType)results->GetCountOrbitEntries(),
        results->GetCountOrbitEntries(),
        results->GetOrbitXLow(),
        results->GetOrbitYLow(),
        results->GetOrbitData(),
        results->GetPeriodMaybeZero()};

    GPUPerturbResults<IterType, T2, PerturbExtras::Bad> gpu_results2{
        (IterType)results2->GetCountOrbitEntries(),
        results2->GetCountOrbitEntries(),
        results2->GetOrbitXLow(),
        results2->GetOrbitYLow(),
        results2->GetOrbitData(),
        results2->GetPeriodMaybeZero()};

    if (gpu_results.GetCompressedSize() != gpu_results2.GetCompressedSize()) {
        std::wcerr << L"Mismatch on size" << std::endl;
        return;
    }

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    auto result = renderer.RenderPerturbBLAScaled<IterType, T>(GetRenderAlgorithm(),
                                                          &gpu_results,
                                                          &gpu_results2,
                                                          cx2,
                                                          cy2,
                                                          dx2,
                                                          dy2,
                                                          centerX2,
                                                          centerY2,
                                                          GetNumIterations<IterType>(),
                                                          m_IterationPrecision);

    if (drawFractal) {
        DrawFractal(idx);
    }

    if (result) {
        MessageBoxCudaError(result);
    }
}

void
Fractal::MessageBoxCudaError(uint32_t result)
{
    char error[256];
    sprintf(error,
            "Error from cuda: %u.  Message: \"%s\"\n",
            result,
            GPURenderer::ConvertErrorToString(result));
    std::cerr << error << std::endl;
}

int
Fractal::SaveCurrentFractal(std::wstring filename_base, bool copy_the_iters)
{
    return SaveFractalData<PngParallelSave::Type::PngImg>(filename_base, copy_the_iters);
}

template <PngParallelSave::Type Typ>
int
Fractal::SaveFractalData(std::wstring filename_base, bool copy_the_iters)
{
    auto lambda = [&]<typename T>(T &savesInProgress) {
        for (;;) {
            MEMORYSTATUSEX statex;
            statex.dwLength = sizeof(statex);
            GlobalMemoryStatusEx(&statex);

            if (savesInProgress.size() > std::thread::hardware_concurrency() ||
                (statex.dwMemoryLoad > 90 && !savesInProgress.empty())) {
                if (!CleanupThreads(false)) {
                    Sleep(100);
                }
            } else {
                auto newPtr =
                    std::make_unique<PngParallelSave>(Typ, filename_base, copy_the_iters, *this);
                savesInProgress.push_back(std::move(newPtr));
                savesInProgress.back()->StartThread();
                break;
            }
        }
    };

    lambda(m_FractalSavesInProgress);

    return 0;
}

bool
Fractal::CleanupThreads(bool all)
{
    bool ret = false;
    auto lambda = [&]<typename T>(T &savesInProgress) {
        bool continueCriteria = true;

        while (continueCriteria) {
            for (size_t i = 0; i < savesInProgress.size(); i++) {
                auto &it = savesInProgress[i];
                if (it->m_Destructable) {
                    savesInProgress.erase(savesInProgress.begin() + i);
                    ret = true;
                    break;
                }
            }

            if (all) {
                continueCriteria = !savesInProgress.empty();
            } else {
                break;
            }
        }
    };

    lambda(m_FractalSavesInProgress);
    return ret;
}

const BenchmarkDataCollection &
Fractal::GetBenchmark() const
{
    return m_BenchmarkData;
}

const HighPrecision &
Fractal::GetMinX(void) const
{
    return m_Ptz.GetMinX();
}

const HighPrecision &
Fractal::GetMaxX(void) const
{
    return m_Ptz.GetMaxX();
}

const HighPrecision &
Fractal::GetMinY(void) const
{
    return m_Ptz.GetMinY();
}

const HighPrecision &
Fractal::GetMaxY(void) const
{
    return m_Ptz.GetMaxY();
}

size_t
Fractal::GetRenderWidth(void) const
{
    return m_ScrnWidth;
}

size_t
Fractal::GetRenderHeight(void) const
{
    return m_ScrnHeight;
}

//////////////////////////////////////////////////////////////////////////////
// The resolution is 4096x4096.
// Re-renders the current fractal at very high resolution,
// and saves that to the the given file.
// Note that making the view "square" may screw with the aspect ratio
// in ways the person didn't anticipate.  For best results,
// make sure the window itself is square beforehand, then run the "square view"
// option.  This way, there is no distortion, since scaling from a smaller square
// view to a larger one doesn't screw anything up.
//////////////////////////////////////////////////////////////////////////////
int
Fractal::SaveHiResFractal(std::wstring filename)
{
    // CBitmapWriter bmpWriter;

    size_t OldScrnWidth = m_ScrnWidth;
    size_t OldScrnHeight = m_ScrnHeight;

    // Set us up with high res
    ResetDimensions(16384, 16384);
    SquareCurrentView();

    // Calculate the high res image.
    // Do it in memory! :)
    CalcFractal(true);

    // Save the bitmap.
    int ret = SaveCurrentFractal(filename, true);

    // Back to the previous res.
    ResetDimensions(OldScrnWidth, OldScrnHeight);

    return ret;
}

int
Fractal::SaveItersAsText(std::wstring filename_base)
{
    return SaveFractalData<PngParallelSave::Type::ItersText>(filename_base, true);
}

void
Fractal::SaveRefOrbit(CompressToDisk compression, std::wstring filename) const
{
    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_RefOrbitSave);
    m_RefOrbit.SaveOrbit(compression, filename);
}

void
Fractal::DiffRefOrbits(CompressToDisk compression,
                       std::wstring outFile,
                       std::wstring filename1,
                       std::wstring filename2) const
{

    m_RefOrbit.DiffOrbit(compression, outFile, filename1, filename2);
}

void
Fractal::LoadRefOrbit(RecommendedSettings *oldSettings,
                      CompressToDisk compression,
                      ImaginaSettings imaginaSettings,
                      std::wstring filename)
{

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_RefOrbitLoad);

    RecommendedSettings recommendedSettings{};

    // If the render algorithm is set to AUTO, then
    // use the saved settings.  The concept of "Auto"
    // seems to imply that we choose the best settings.
    if (m_RenderAlgorithm == RenderAlgorithmEnum::AUTO) {
        if (imaginaSettings != ImaginaSettings::UseSaved) {
            imaginaSettings = ImaginaSettings::UseSaved;
        }
    }

    // Store the old algorithm so we can restore it later.
    if (oldSettings) {
        // Store "RenderAlgorithm::AUTO" if applicable.
        *oldSettings = RecommendedSettings{GetPrecision(),
                                           m_Ptz.GetMinX(),
                                           m_Ptz.GetMinY(),
                                           m_Ptz.GetMaxX(),
                                           m_Ptz.GetMaxY(),
                                           m_RenderAlgorithm,
                                           GetNumIterations<IterTypeFull>()};

        oldSettings->OverrideIterType(GetIterType());
    }

    auto renderAlg = imaginaSettings == ImaginaSettings::UseSaved
                         ? GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::MAX)
                         : GetRenderAlgorithm();

    m_RefOrbit.LoadOrbit(imaginaSettings, compression, renderAlg, filename, &recommendedSettings);

    if (imaginaSettings == ImaginaSettings::UseSaved) {
        const bool success = SetRenderAlgorithm(recommendedSettings.GetRenderAlgorithm());
        if (!success) {
            std::wcerr << L"Warning: saved render algorithm is not supported on this system."
                       << std::endl;
        }

        SetIterType(recommendedSettings.GetIterType());
    }

    // This will force it to fit but it should
    SetNumIterations<IterTypeFull>(recommendedSettings.GetNumIterations());

    const auto &ptz = recommendedSettings.GetPointZoomBBConverter();
    m_Ptz = ptz;

    ChangedMakeDirty();
}

void
Fractal::SetResultsAutosave(AddPointOptions Enable)
{
    m_RefOrbit.SetOptions(Enable);
}

AddPointOptions
Fractal::GetResultsAutosave() const
{
    return m_RefOrbit.GetOptions();
}

// This function is used for benchmarking.
// Note that it does not return correct results
// if there are any pixels on the screen that took
// zero iterations to escape from.
uint64_t
Fractal::FindTotalItersUsed(void)
{
    uint64_t numIters = 0;
    int x, y;
    for (y = 0; y < m_ScrnHeight; y++) {
        for (x = 0; x < m_ScrnWidth; x++) {
            numIters += m_CurIters.GetItersArrayValSlow(x, y);
        }
    }

    return numIters;
}

//////////////////////////////////////////////////////////////////////////////
// Helper functions for converting from one coordinate system to another.
// "Calculator coordinates" are those one normally thinks of in math with
// the x-y axes.  "Screen coordinates" are those one thinks of with regards
// to most computer screens.
//////////////////////////////////////////////////////////////////////////////
template <bool IncludeGpuAntialiasing>
HighPrecision
Fractal::XFromScreenToCalc(HighPrecision x) const
{
    size_t aa = IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1;
    return m_Ptz.XFromScreenToCalc(std::move(x), m_ScrnWidth, aa);
}

template <bool IncludeGpuAntialiasing>
HighPrecision
Fractal::YFromScreenToCalc(HighPrecision y) const
{
    size_t aa = IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1;
    return m_Ptz.YFromScreenToCalc(std::move(y), m_ScrnHeight, aa);
}

template HighPrecision Fractal::XFromScreenToCalc<false>(HighPrecision x) const;
template HighPrecision Fractal::XFromScreenToCalc<true>(HighPrecision x) const;
template HighPrecision Fractal::YFromScreenToCalc<false>(HighPrecision y) const;
template HighPrecision Fractal::YFromScreenToCalc<true>(HighPrecision y) const;

HighPrecision
Fractal::XFromCalcToScreen(HighPrecision x) const
{
    return m_Ptz.XFromCalcToScreen(std::move(x), m_ScrnWidth);
}

HighPrecision
Fractal::YFromCalcToScreen(HighPrecision y) const
{
    return m_Ptz.YFromCalcToScreen(std::move(y), m_ScrnHeight);
}

void
Fractal::ForceRecalc()
{
    ChangedMakeDirty();
}

const LAParameters &
Fractal::GetLAParameters() const
{
    return m_LAParameters;
}

LAParameters &
Fractal::GetLAParameters()
{
    return m_LAParameters;
}

void
Fractal::GetRenderDetails(std::string &shortStr, std::string &longStr) const
{

    HighPrecision minX, minY;
    HighPrecision maxX, maxY;

    const auto prec = GetPrecision();

    minX = GetMinX();
    minY = GetMinY();
    maxX = GetMaxX();
    maxY = GetMaxY();

    std::stringstream ss;
    std::string s;

    const auto setupSS = [&](const HighPrecision &num) -> std::string {
        ss.str("");
        ss.clear();
        ss << std::setprecision(std::numeric_limits<HighPrecision>::max_digits10);
        ss << num;
        return ss.str();
    };

    s = setupSS(minX);
    const auto sminX = std::string(s.begin(), s.end());

    s = setupSS(minY);
    const auto sminY = std::string(s.begin(), s.end());

    s = setupSS(maxX);
    const auto smaxX = std::string(s.begin(), s.end());

    s = setupSS(maxY);
    const auto smaxY = std::string(s.begin(), s.end());

    const PointZoomBBConverter pz{minX, minY, maxX, maxY};
    s = setupSS(pz.GetPtX());
    const auto ptXStr = std::string(s.begin(), s.end());

    s = setupSS(pz.GetPtY());
    const auto ptYStr = std::string(s.begin(), s.end());

    auto reducedPrecZF = pz.GetZoomFactor();
    reducedPrecZF.precisionInBits(50);
    s = setupSS(reducedPrecZF);
    const auto zoomFactorStr = std::string(s.begin(), s.end());

    RefOrbitDetails details;
    GetSomeDetails(details);

    const auto ActualPeriodIfAny =
        (details.InternalPeriodMaybeZero > 0) ? (details.InternalPeriodMaybeZero - 1) : 0;

    const auto additionalDetailsStr =
        std::string("PerturbationAlg = ") + details.PerturbationAlg + "\r\n" +
        std::string("InternalPeriodIfAny = ") + std::to_string(details.InternalPeriodMaybeZero) +
        "\r\n" + std::string("ActualPeriodIfAny = ") + std::to_string(ActualPeriodIfAny) + "\r\n" +
        std::string("CompressedIters = ") + std::to_string(details.CompressedIters) + "\r\n" +
        std::string("UncompressedIters = ") + std::to_string(details.UncompressedIters) + "\r\n" +
        std::string("Compression ratio = ") +
        std::to_string((double)details.UncompressedIters / (double)details.CompressedIters) + "\r\n" +
        std::string("Compression error exp = ") + std::to_string(details.CompressionErrorExp) + "\r\n" +
        std::string("CompressedIntermediateIters = ") +
        std::to_string(details.CompressedIntermediateIters) + "\r\n" +
        std::string("Reuse compression error exp = ") +
        std::to_string(details.IntermediateCompressionErrorExp) + "\r\n" +
        std::string("Reuse compression ratio = ") +
        std::to_string((double)details.UncompressedIters / (double)details.CompressedIntermediateIters) +
        "\r\n" + std::string("DeltaIntermediatePrecision = ") +
        std::to_string(details.DeltaIntermediatePrecision) + "\r\n" +
        std::string("ExtraIntermediatePrecision = ") +
        std::to_string(details.ExtraIntermediatePrecision) + "\r\n" + std::string("ZoomFactor = ") +
        zoomFactorStr + "\r\n";

    const auto &laParameters = GetLAParameters();
    const auto threadingVal = laParameters.GetThreading();
    std::string threadingStr;
    if (threadingVal == LAParameters::LAThreadingAlgorithm::SingleThreaded) {
        threadingStr = "Single threaded";
    } else if (threadingVal == LAParameters::LAThreadingAlgorithm::MultiThreaded) {
        threadingStr = "Multi threaded";
    } else {
        threadingStr = "Unknown";
    }

    const auto laParametersStr =
        std::string("Detection method = ") + std::to_string(laParameters.GetDetectionMethod()) + "\r\n" +
        std::string("Threshold scale = ") + std::to_string(laParameters.GetLAThresholdScaleExp()) +
        "\r\n" + std::string("Threshold C scale = ") +
        std::to_string(laParameters.GetLAThresholdCScaleExp()) + "\r\n" +
        std::string("Stage 0 period detection threshold 2 = ") +
        std::to_string(laParameters.GetStage0PeriodDetectionThreshold2Exp()) + "\r\n" +
        std::string("Period detection threshold 2 = ") +
        std::to_string(laParameters.GetPeriodDetectionThreshold2Exp()) + "\r\n" +
        std::string("Stage 0 period detection threshold = ") +
        std::to_string(laParameters.GetStage0PeriodDetectionThresholdExp()) + "\r\n" +
        std::string("Period detection threshold = ") +
        std::to_string(laParameters.GetPeriodDetectionThresholdExp()) + "\r\n" +
        std::string("LA Threading: ") + threadingStr + "\r\n" + std::string("LA size: ") +
        std::to_string(details.LASize) + "\r\n";

    const auto benchmarkData =
        std::string("Overall (ms) = ") + std::to_string(GetBenchmark().m_Overall.GetDeltaInMs()) +
        "\r\n" + std::string("Per pixel (ms) = ") +
        std::to_string(GetBenchmark().m_PerPixel.GetDeltaInMs()) + "\r\n" +
        std::string("RefOrbit save (ms) = ") +
        std::to_string(GetBenchmark().m_RefOrbitSave.GetDeltaInMs()) + "\r\n" +
        std::string("RefOrbit load (ms) = ") +
        std::to_string(GetBenchmark().m_RefOrbitLoad.GetDeltaInMs()) + "\r\n" +
        std::string("RefOrbit (ms) = ") + std::to_string(details.OrbitMilliseconds) + "\r\n" +
        std::string("LA generation time (ms) = ") + std::to_string(details.LAMilliseconds) + "\r\n" +
        std::string("Feature Finder time low-prec (ms) = ") +
        std::to_string(GetBenchmark().m_FeatureFinder.GetDeltaInMs()) + "\r\n" +
        std::string("Feature Finder time high-prec (ms) = ") +
        std::to_string(GetBenchmark().m_FeatureFinderHP.GetDeltaInMs()) + "\r\n";

    shortStr = std::format("This text is copied to clipboard.  Using \"{}\"\r\n"
                           "Antialiasing: {}\r\n"
                           "Palette depth: {}\r\n"
                           "Coordinate precision = {};\r\n"
                           "\r\n"
                           "LA parameters:\r\n"
                           "{}\r\n"
                           "Benchmark data:\r\n"
                           "{}\r\n"
                           "\r\n"
                           "Additional details:\r\n"
                           "{}\r\n"
                           "SetNumIterations<IterTypeFull>({});\r\n",
                           GetRenderAlgorithmName(),
                           GetGpuAntialiasing(),
                           GetPaletteDepth(),
                           prec,
                           laParametersStr,
                           benchmarkData,
                           additionalDetailsStr,
                           GetNumIterations<IterTypeFull>());

    longStr = shortStr;

    // Put some extra information on the clipboard.
    auto tempStr = std::format("Bounding box:\r\n"
                               "Center X: \"{}\"\r\n"
                               "Center Y: \"{}\"\r\n"
                               "zoomFactor: \"{}\"\r\n"
                               "minX = HighPrecision{{ \"{}\" }};\r\n"
                               "minY = HighPrecision{{ \"{}\" }};\r\n"
                               "maxX = HighPrecision{{ \"{}\" }};\r\n"
                               "maxY = HighPrecision{{ \"{}\" }};\r\n",
                               ptXStr,
                               ptYStr,
                               zoomFactorStr,
                               sminX,
                               sminY,
                               smaxX,
                               smaxY);

    longStr += tempStr;
}

bool
Fractal::GpuBypassed() const
{
    return m_BypassGpu;
}

bool
Fractal::IsDownControl(void)
{
    return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000);
    // return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000) &&
    //     ((m_hWnd != nullptr && GetForegroundWindow() == m_hWnd) ||
    //     (m_hWnd == nullptr));
};

void
Fractal::CheckForAbort(void)
{
    POINT pt;
    GetCursorPos(&pt);
    int OrgX = pt.x;
    int OrgY = pt.y;

    for (;;) {
        if (m_AbortThreadQuitFlag == true) {
            break;
        }

        Sleep(250);

        if (IsDownControl()) {
            m_StopCalculating = true;
        }

        if (m_UseSensoCursor == true) {
            GetCursorPos(&pt);

            if (abs(pt.x - OrgX) >= 5 || abs(pt.y - OrgY) >= 5) {
                OrgX = pt.x; // Reset to current location.
                OrgY = pt.y;
                m_StopCalculating = true;
            }
        }
    }
}

unsigned long WINAPI
Fractal::CheckForAbortThread(void *fractal)
{
    ((Fractal *)fractal)->CheckForAbort();
    return 0;
}
