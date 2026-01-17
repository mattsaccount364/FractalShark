#include "stdafx.h"

#include "Fractal.h"
// #include "CBitmapWriter.h"

#include "BLAS.h"

#include <fstream>
#include <iostream>
#include <psapi.h>
#include <thread>

#include "ATInfo.h"
#include "HDRFloatComplex.h"
#include "LAInfoDeep.h"
#include "LAReference.h"

#include "BenchmarkData.h"
#include "CudaDblflt.h"

#include "PerturbationResults.h"
#include "PrecisionCalculator.h"
#include "RecommendedSettings.h"

#include <chrono>

void DefaultOutputMessage(const wchar_t *, ...);

Fractal::Fractal(int width, int height, HWND hWnd, bool UseSensoCursor, uint64_t commitLimitInBytes)
    : m_RefOrbit{*this, commitLimitInBytes}, m_CommitLimitInBytes{commitLimitInBytes}
{

    Initialize(width, height, hWnd, UseSensoCursor);
}

Fractal::~Fractal() { Uninitialize(); }

void
Fractal::Initialize(int width, int height, HWND hWnd, bool UseSensoCursor)
{
    SetupCuda();

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

    InitialDefaultViewAndSettings(width, height);

    m_AsyncRenderThreadState = AsyncRenderThreadState::Idle;
    m_AsyncRenderThreadFinish = false;
    m_AsyncRenderThread = std::make_unique<std::thread>(DrawAsyncGpuFractalThreadStatic, this);

    srand((unsigned int)time(nullptr));

    // Initialize the palette
    auto DefaultPaletteGen = [&](FractalPalette WhichPalette, size_t PaletteIndex, size_t Depth) {
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalR[WhichPalette][PaletteIndex].size();
    };

    auto PatrioticPaletteGen = [&](FractalPalette WhichPalette, size_t PaletteIndex, size_t Depth) {
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;

        // R=0xBB G=0x13 B=0x3E
        // R=0xB3 G=0x19 B=0x42
        // R=0xBF G=0x0A B=0x30
        const auto RR = (int)(((double)0xB3 / (double)0xFF) * max_val);
        const auto RG = (int)(((double)0x19 / (double)0xFF) * max_val);
        const auto RB = (int)(((double)0x42 / (double)0xFF) * max_val);

        // R=0x00 G=0x21 B=0x47
        // R=0x0A G=0x31 B=0x61
        // R=0x00 G=0x28 B=0x68
        const auto BR = (int)(((double)0x0A / (double)0xFF) * max_val);
        const auto BG = (int)(((double)0x31 / (double)0xFF) * max_val);
        const auto BB = (int)(((double)0x61 / (double)0xFF) * max_val);

        m_PalR[WhichPalette][PaletteIndex].push_back(static_cast<uint16_t>(max_val));
        m_PalG[WhichPalette][PaletteIndex].push_back(static_cast<uint16_t>(max_val));
        m_PalB[WhichPalette][PaletteIndex].push_back(static_cast<uint16_t>(max_val));

        PalTransition(WhichPalette, PaletteIndex, depth_total, RR, RG, RB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, BR, BG, BB);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalR[WhichPalette][PaletteIndex].size();
    };

    auto SummerPaletteGen = [&](FractalPalette WhichPalette, size_t PaletteIndex, size_t Depth) {
        int depth_total = (int)(1 << Depth);

        int max_val = 65535;

        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val / 2, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val / 2, max_val / 2, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val * 2 / 3, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t)m_PalR[WhichPalette][PaletteIndex].size();
    };

    for (size_t i = 0; i < FractalPalette::Num; i++) {
        m_PalIters[i].resize(NumBitDepths);
    }

    std::vector<std::unique_ptr<std::thread>> threads;

    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 0, 5));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 1, 6));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 2, 8));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 3, 12));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 4, 16));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, FractalPalette::Default, 5, 20));

    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 0, 5));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 1, 6));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 2, 8));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 3, 12));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 4, 16));
    threads.push_back(
        std::make_unique<std::thread>(PatrioticPaletteGen, FractalPalette::Patriotic, 5, 20));

    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 0, 5));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 1, 6));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 2, 8));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 3, 12));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 4, 16));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, FractalPalette::Summer, 5, 20));

    for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        m_DrawThreads.emplace_back(std::make_unique<DrawThreadSync>(i, nullptr, m_DrawThreadAtomics));
    }

    for (size_t i = 0; i < m_DrawThreads.size(); i++) {
        auto thread = std::make_unique<std::thread>(DrawFractalThread, i, this);
        m_DrawThreads[i]->m_Thread = std::move(thread);
    }

    // Allocate the iterations array.
    InitializeMemory();

    // Set up random palette.
    CreateNewFractalPalette();

    // Wait for all this shit to get done
    for (auto &it : threads) {
        it->join();
    }

    m_PaletteRotate = 0;
    m_PaletteDepthIndex = 2;
    m_PaletteAuxDepth = 0;
    UsePaletteType(FractalPalette::Default);
    UsePalette(8);

    // Make sure the screen is completely redrawn the first time.
    ChangedMakeDirty();
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
Fractal::InitializeGPUMemory(bool expectedReuse)
{
    if (RequiresUseLocalColor()) {
        return 0;
    }

    if (m_BypassGpu) {
        return 1;
    }

    uint32_t res;
    if (GetIterType() == IterTypeEnum::Bits32) {
        res = m_r.InitializeMemory<uint32_t>((uint32_t)m_CurIters.m_Width,
                                              (uint32_t)m_CurIters.m_Height,
                                              (uint32_t)m_CurIters.m_Antialiasing,
                                              m_PalR[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalG[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalB[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalIters[m_WhichPalette][m_PaletteDepthIndex],
                                              m_PaletteAuxDepth,
                                              expectedReuse);
    } else {
        res = m_r.InitializeMemory<uint64_t>((uint32_t)m_CurIters.m_Width,
                                              (uint32_t)m_CurIters.m_Height,
                                              (uint32_t)m_CurIters.m_Antialiasing,
                                              m_PalR[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalG[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalB[m_WhichPalette][m_PaletteDepthIndex].data(),
                                              m_PalIters[m_WhichPalette][m_PaletteDepthIndex],
                                              m_PaletteAuxDepth,
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
        m_AsyncRenderThreadState = AsyncRenderThreadState::Finish;
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
// Palette functions.  These functions make it easy to create a smoothly
// transitioning palette.
//////////////////////////////////////////////////////////////////////////////

// Given an empty array, a range of indexes to iterate over, and a start number
// and end number, this function will smoothly transition from val1 to val2
// over the indexes specified.
// length must be > 0
// total_length = number of elements in pal.
// e.g. unsigned char pal[256];
//   total_length == 256
void
Fractal::PalIncrease(std::vector<uint16_t> &pal, int length, int val1, int val2)
{
    double delta = (double)((double)(val2 - val1)) / length;
    for (int i = 0; i < length; i++) {
        double result = ((double)val1 + (double)delta * (i + 1));
        pal.push_back((unsigned short)result);
    }
}

// Transitions to the color specified.
// Allows for nice smooth palettes.
// length must be > 0
// Returns index immediately following the last index we filled here
// Returns -1 if we are at the end.
void
Fractal::PalTransition(size_t WhichPalette, size_t PaletteIndex, int length, int r, int g, int b)
{
    int curR, curB, curG;
    if (!m_PalR[WhichPalette][PaletteIndex].empty()) {
        curR = m_PalR[WhichPalette][PaletteIndex][m_PalR[WhichPalette][PaletteIndex].size() - 1];
        curG = m_PalG[WhichPalette][PaletteIndex][m_PalG[WhichPalette][PaletteIndex].size() - 1];
        curB = m_PalB[WhichPalette][PaletteIndex][m_PalB[WhichPalette][PaletteIndex].size() - 1];
    } else {
        curR = 0;
        curG = 0;
        curB = 0;
    }

    // This code will fill out the palettes to the very end.
    PalIncrease(m_PalR[WhichPalette][PaletteIndex], length, curR, r);
    PalIncrease(m_PalG[WhichPalette][PaletteIndex], length, curG, g);
    PalIncrease(m_PalB[WhichPalette][PaletteIndex], length, curB, b);
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
// screen coordinates to calculator coordinates for you, using whatever
// dimensions were specified when the Fractal object was constructed or whatever
// coordinates were given to the last call to the Reset function.
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
                // Do nothing.  This case can occur if we e.g. change dimension, change gpu antialiasing
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
    const HighPrecision width = m_Ptz.GetMaxX() - m_Ptz.GetMinX();
    const HighPrecision height = m_Ptz.GetMaxY() - m_Ptz.GetMinY();
    const auto two = HighPrecision{2};

    PointZoomBBConverter ptz{newCenterX - width / two,
                             newCenterY - height / two,
                             newCenterX + width / two,
                             newCenterY + height / two};

    return RecenterViewCalc(ptz);
}

void
Fractal::Zoom(double factor)
{
    const HighPrecision deltaX = (m_Ptz.GetMaxX() - m_Ptz.GetMinX()) * (HighPrecision)factor;
    const HighPrecision deltaY = (m_Ptz.GetMaxY() - m_Ptz.GetMinY()) * (HighPrecision)factor;

    PointZoomBBConverter ptz{m_Ptz.GetMinX() - deltaX,
                             m_Ptz.GetMinY() - deltaY,
                             m_Ptz.GetMaxX() + deltaX,
                             m_Ptz.GetMaxY() + deltaY};

    RecenterViewCalc(ptz);
}

// This one recenters and zooms in on the mouse cursor.
// This one is better with the keyboard.
void
Fractal::Zoom(size_t scrnX, size_t scrnY, double factor)
{
    CenterAtPoint(scrnX, scrnY);
    Zoom(factor);
}

// The idea is to zoom in on the mouse cursor, without also recentering it.
// This one is better with the wheel.
void
Fractal::Zoom2(size_t scrnX, size_t scrnY, double factor)
{
    const HighPrecision newCenterX = XFromScreenToCalc((HighPrecision)scrnX);
    const HighPrecision newCenterY = YFromScreenToCalc((HighPrecision)scrnY);

    const auto &minX = m_Ptz.GetMinX();
    const auto &minY = m_Ptz.GetMinY();
    const auto &maxX = m_Ptz.GetMaxX();
    const auto &maxY = m_Ptz.GetMaxY();

    const HighPrecision leftWeight = (newCenterX - minX) / (maxX - minX);
    const HighPrecision rightWeight = HighPrecision{1} - leftWeight;
    const HighPrecision topWeight = (newCenterY - minY) / (maxY - minY);
    const HighPrecision bottomWeight = HighPrecision{1} - topWeight;

    const HighPrecision minXFinal = minX - (maxX - minX) * leftWeight * (HighPrecision)factor;
    const HighPrecision minYFinal = minY - (maxY - minY) * topWeight * (HighPrecision)factor;
    const HighPrecision maxXFinal = maxX + (maxX - minX) * rightWeight * (HighPrecision)factor;
    const HighPrecision maxYFinal = maxY + (maxY - minY) * bottomWeight * (HighPrecision)factor;

    PointZoomBBConverter ptz{minXFinal, minYFinal, maxXFinal, maxYFinal};
    RecenterViewCalc(ptz);
}

void
Fractal::SetupCuda()
{
    auto res = GPURenderer::TestCudaIsWorking();

    if (!res) {
        std::cerr << "CUDA initialization failed.  GPU rendering will be disabled.\n";
        MessageBoxCudaError(res);
        m_BypassGpu = true;
        return;
    }

    m_BypassGpu = false;
}

void
Fractal::InitialDefaultViewAndSettings(int width, int height)
{
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2));
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2));
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2));
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu2x32PerturbedLAv2));
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO));
    // SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2));
    SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::AUTO));

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
    ChangedMakeDirty();

    // Doesn't do anything with the palette.
}

template <Fractal::AutoZoomHeuristic h>
void
Fractal::AutoZoom()
{
    const HighPrecision Two = HighPrecision{2};

    HighPrecision width = m_Ptz.GetMaxX() - m_Ptz.GetMinX();
    HighPrecision height = m_Ptz.GetMaxY() - m_Ptz.GetMinY();

    HighPrecision guessX;
    HighPrecision guessY;

    HighPrecision p10_9;

    size_t retries = 0;

    // static std::vector<size_t> top100;

    HighPrecision Divisor;
    if constexpr (h == AutoZoomHeuristic::Default) {
        Divisor = HighPrecision{3};
    }

    if constexpr (h == AutoZoomHeuristic::Max) {
        Divisor = HighPrecision{32};
    }

    for (;;) {
        // SaveCurPos();
        {
            MSG msg;
            PeekMessage(&msg, nullptr, 0, 0, PM_NOREMOVE);
        }

        double geometricMeanX = 0;
        double geometricMeanSum = 0;
        double geometricMeanY = 0;

        if (retries >= 0) {
            width = m_Ptz.GetMaxX() - m_Ptz.GetMinX();
            height = m_Ptz.GetMaxY() - m_Ptz.GetMinY();
            retries = 0;
        }

        size_t numAtMax = 0;
        size_t numAtLimit = 0;

        bool shouldBreak = false;

        auto lambda = [&](auto **ItersArray, auto NumIterations) {
            if constexpr (h == AutoZoomHeuristic::Default) {
                ULONG shiftWidth = (ULONG)m_ScrnWidth / 8;
                ULONG shiftHeight = (ULONG)m_ScrnHeight / 8;

                RECT antiRect;
                antiRect.left = shiftWidth;
                antiRect.right = (ULONG)m_ScrnWidth - shiftWidth;
                antiRect.top = shiftHeight;
                antiRect.bottom = (ULONG)m_ScrnHeight - shiftHeight;

                antiRect.left *= GetGpuAntialiasing();
                antiRect.right *= GetGpuAntialiasing();
                antiRect.top *= GetGpuAntialiasing();
                antiRect.bottom *= GetGpuAntialiasing();

                const auto antiRectWidthInt = antiRect.right - antiRect.left;
                const auto antiRectHeightInt = antiRect.bottom - antiRect.top;

                size_t maxiter = 0;
                double totaliters = 0;
                for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                    for (auto x = antiRect.left; x < antiRect.right; x++) {
                        auto curiter = ItersArray[y][x];
                        totaliters += curiter;

                        if (curiter > maxiter) {
                            maxiter = curiter;
                        }

                        // if (top100.empty() ||
                        //     (!top100.empty() && curiter > top100[0])) {
                        //     top100.push_back(curiter);
                        //     std::sort(top100.begin(), top100.end(), std::less<size_t>());

                        //    if (top100.size() > 100) {
                        //        top100.erase(top100.begin());
                        //    }
                        //}
                    }
                }

                double avgiters =
                    totaliters / ((antiRect.bottom - antiRect.top) * (antiRect.right - antiRect.left));

                // double midpointX = ((double) antiRect.left + antiRect.right) / 2.0;
                // double midpointY = ((double) antiRect.top + antiRect.bottom) / 2.0;
                double antirectWidth = antiRectWidthInt;
                double antirectHeight = antiRectHeightInt;

                double widthOver2 = antirectWidth / 2.0;
                double heightOver2 = antirectHeight / 2.0;
                double maxDistance = sqrt(widthOver2 * widthOver2 + heightOver2 * heightOver2);

                for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                    for (auto x = antiRect.left; x < antiRect.right; x++) {
                        auto curiter = ItersArray[y][x];

                        if (curiter == maxiter) {
                            numAtLimit++;
                        }

                        if (curiter < avgiters) {
                            continue;
                        }

                        // if (curiter > maxiter) {
                        //     maxiter = curiter;
                        //     geometricMeanX = x;
                        //     geometricMeanY = y;
                        //     geometricMeanSum = 1;
                        // }

                        // TODO this is all fucked up
                        double distanceX = fabs(widthOver2 - fabs(widthOver2 - fabs(x - antiRect.left)));
                        double distanceY =
                            fabs(heightOver2 - fabs(heightOver2 - fabs(y - antiRect.top)));
                        double normalizedIters = (double)curiter / (double)NumIterations;

                        if (curiter == maxiter) {
                            normalizedIters *= normalizedIters;
                        }
                        double normalizedDist =
                            (sqrt(distanceX * distanceX + distanceY * distanceY) / maxDistance);
                        double sq = normalizedIters * normalizedDist;
                        geometricMeanSum += sq;
                        geometricMeanX += sq * x;
                        geometricMeanY += sq * y;

                        if (curiter >= NumIterations) {
                            numAtMax++;
                        }
                    }
                }

                assert(geometricMeanSum != 0);

                if (geometricMeanSum != 0) {
                    double meanX = geometricMeanX / geometricMeanSum;
                    double meanY = geometricMeanY / geometricMeanSum;

                    guessX = XFromScreenToCalc<true>(HighPrecision{meanX});
                    guessY = YFromScreenToCalc<true>(HighPrecision{meanY});

                    // wchar_t temps[256];
                    // swprintf(temps, 256, L"Coords: %f %f", meanX, meanY);
                    //::MessageBox(nullptr, temps, L"", MB_OK | MB_APPLMODAL);
                } else {
                    shouldBreak = true;
                    return;
                }

                if (numAtLimit == antiRectWidthInt * antiRectHeightInt) {
                    std::wcerr << L"Flat screen! :(" << std::endl;
                    shouldBreak = true;
                    return;
                }
            }

            if constexpr (h == AutoZoomHeuristic::Max) {
                LONG targetX = -1;
                LONG targetY = -1;

                size_t maxiter = 0;
                for (auto y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
                    for (auto x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                        auto curiter = ItersArray[y][x];
                        if (curiter > maxiter) {
                            maxiter = curiter;
                        }
                    }
                }

                for (auto y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
                    for (auto x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                        auto curiter = ItersArray[y][x];
                        if (curiter == maxiter) {
                            numAtLimit++;
                            if (targetX == -1 && targetY == -1) {
                                targetX = x;
                                targetY = y;
                            }
                        }

                        if (curiter >= NumIterations) {
                            numAtMax++;
                        }
                    }
                }

                guessX = XFromScreenToCalc<true>(HighPrecision{targetX});
                guessY = YFromScreenToCalc<true>(HighPrecision{targetY});

                if (numAtLimit ==
                    m_ScrnWidth * m_ScrnHeight * GetGpuAntialiasing() * GetGpuAntialiasing()) {
                    std::wcerr << L"Flat screen! :(" << std::endl;
                    shouldBreak = true;
                    return;
                }
            }
        };

        if (GetIterType() == IterTypeEnum::Bits32) {
            lambda(m_CurIters.GetItersArray<uint32_t>(), GetNumIterations<uint32_t>());
        } else {
            lambda(m_CurIters.GetItersArray<uint64_t>(), GetNumIterations<uint64_t>());
        }

        if (shouldBreak) {
            break;
        }

        HighPrecision newMinX = guessX - width / Divisor;
        HighPrecision newMinY = guessY - height / Divisor;
        HighPrecision newMaxX = guessX + width / Divisor;
        HighPrecision newMaxY = guessY + height / Divisor;

        HighPrecision centerX = (m_Ptz.GetMinX() + m_Ptz.GetMaxX()) / Two;
        HighPrecision centerY = (m_Ptz.GetMinY() + m_Ptz.GetMaxY()) / Two;

        HighPrecision defaultMinX = centerX - width / Divisor;
        HighPrecision defaultMinY = centerY - height / Divisor;
        HighPrecision defaultMaxX = centerX + width / Divisor;
        HighPrecision defaultMaxY = centerY + height / Divisor;

        const HighPrecision defaultWeight{0};
        const HighPrecision newWeight{1};
        HighPrecision weightedNewMinX =
            (newWeight * newMinX + defaultWeight * defaultMinX) / (newWeight + defaultWeight);
        HighPrecision weightedNewMinY =
            (newWeight * newMinY + defaultWeight * defaultMinY) / (newWeight + defaultWeight);
        HighPrecision weightedNewMaxX =
            (newWeight * newMaxX + defaultWeight * defaultMaxX) / (newWeight + defaultWeight);
        HighPrecision weightedNewMaxY =
            (newWeight * newMaxY + defaultWeight * defaultMaxY) / (newWeight + defaultWeight);

        // HighPrecision weightedNewMinX = guessX - width / Divisor;
        // HighPrecision weightedNewMinY = guessY - height / Divisor;
        // HighPrecision weightedNewMaxX = guessX + width / Divisor;
        // HighPrecision weightedNewMaxY = guessY + height / Divisor;

        PointZoomBBConverter newPtz{weightedNewMinX, weightedNewMinY, weightedNewMaxX, weightedNewMaxY};
        RecenterViewCalc(newPtz);

        CalcFractal(false);
        // DrawFractal(false);

        if (numAtMax > 500) {
            break;
        }

        retries++;

        if (m_StopCalculating) {
            break;
        }
    }
}

template void Fractal::AutoZoom<Fractal::AutoZoomHeuristic::Default>();
template void Fractal::AutoZoom<Fractal::AutoZoomHeuristic::Max>();

//////////////////////////////////////////////////////////////////////////////
// Resets the fractal to the standard view.
// Make sure the view is square on all monitors at all weird aspect ratios.
//////////////////////////////////////////////////////////////////////////////
void
Fractal::View(size_t view, bool includeMsgBox)
{
    HighPrecision minX;
    HighPrecision minY;
    HighPrecision maxX;
    HighPrecision maxY;

    HighPrecision::defaultPrecisionInBits(MaxPrecisionLame);
    minX.precisionInBits(MaxPrecisionLame);
    minY.precisionInBits(MaxPrecisionLame);
    maxX.precisionInBits(MaxPrecisionLame);
    maxY.precisionInBits(MaxPrecisionLame);

    ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1);

    // Reset to default reference compression if applicable
    DefaultCompressionErrorExp(CompressionError::Low);
    DefaultCompressionErrorExp(CompressionError::Intermediate);

    switch (view) {
        case 1:
            // Limits of 4x64 GPU
            minX = HighPrecision{"-1."
                                 "7633991770667526958542201208184933948747647150755250706970853761736441"
                                 "56624573649873526729559691534754284706803085481158"};
            minY = HighPrecision{"0."
                                 "0428921126280651283647328562785831863573469575929186730211273062418894"
                                 "1270466703058975670804976478935827994844038526618063053858"};
            maxX = HighPrecision{"-1."
                                 "7633991770667526958542201208184933948747647150755250706970853558702728"
                                 "24868052108014289980411203646967653925705407102169"};
            maxY = HighPrecision{"0."
                                 "0428921126280651283647328562785831863573469575929186730211273146146333"
                                 "0922125985949917962768812338316745717506303752530265831841"};
            SetNumIterations<IterTypeFull>(196608);
            break;

        case 2: {
            // Limits of 4x32 GPU
            minX = HighPrecision{"-1."
                                 "7689694868673579727755649512754615510527514995099971856918819507862537"
                                 "43769635708375905775793656954725307354460920979983"};
            minY = HighPrecision{"0."
                                 "0569928069030467089311563689286064783317546364492265237591671271987259"
                                 "9382335388157040896288795946562522749757591414246314107544"};
            maxX = HighPrecision{"-1."
                                 "7689694868673579727755649509294879345534964945639413359110852926992503"
                                 "68065865432159590460057564657941788398574759610411"};
            maxY = HighPrecision{"0."
                                 "0569928069030467089311563690712797535512795230639169214127380470604178"
                                 "3710937876987367435542131127321801128526034375237132904264"};
            SetNumIterations<IterTypeFull>(196608);
            break;
        }

        case 3:
            // Limit of 1x32 + Perturbation with scaling
            minX = HighPrecision{"-1."
                                 "4465672699702273706229580697781780382944306168865611762380025631230375"
                                 "1202920456713778693247098684334495241572095045"};
            minY = HighPrecision{"7."
                                 "6416324526384045004431827961982015350830278953082652797942796664282935"
                                 "7717061175013838301474813332434725222956221212e-18"};
            maxX = HighPrecision{"-1."
                                 "4465672699702273706229580697781780382944260352995904063881267466752269"
                                 "7557115287788808403561611427018141845213679032"};
            maxY = HighPrecision{"7."
                                 "6416324526384045004431847051927726891871428188281863366518012036156697"
                                 "84193475322289855499574866715163283993737498e-18"};
            SetNumIterations<IterTypeFull>(196608);
            break;

        case 4:
            minX = HighPrecision{"-1."
                                 "4465672699702273706229580697781780382944276606223103446982143768032451"
                                 "5234695809677735314398112720680340773533658285"};
            minY = HighPrecision{"7."
                                 "6416324526384045004431831566549578255470090654562809940333742884829477"
                                 "9227472963485579985013605880788857454558128871e-18"};
            maxX = HighPrecision{"-1."
                                 "4465672699702273706229580697781780382944276606223103446982143768032451"
                                 "5234695809677735191376844462193327526231811052"};
            maxY = HighPrecision{"7."
                                 "6416324526384045004431831566549578255470090654562809940333742884829483"
                                 "0486334737855168838056042228491124080530744489e-18"};
            SetNumIterations<IterTypeFull>(196608);
            break;

        case 5:
            minX = HighPrecision{"-0."
                                 "5482057480704757084582125675467330293766992786373239327878603685103691"
                                 "07357663992406257053055723741951365216836802745"};
            minY = HighPrecision{"-0."
                                 "5775708389036038428051089822018505586755517301137385293646982654127795"
                                 "45002113555345006591372870167386914495276370477"};
            maxX = HighPrecision{"-0."
                                 "5482057480704757084582125675467330293766992706084409748610293006796228"
                                 "9200412659019319306589187062772276993544341295"};
            maxY = HighPrecision{"-0."
                                 "5775708389036038428051089822018505586755517268027721049520596403786942"
                                 "74662197291893029522164691495936927144187595881"};
            SetNumIterations<IterTypeFull>(4718592);
            // ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1); // TODO
            break;

        case 6:
            // Scale float with pixellation
            minX =
                HighPrecision{"-1."
                              "6225530545095544093937832714855193369815166490586925235310445917701797841"
                              "8891616690380136311469569647746535255597152879870544828084030266459696312"
                              "328585298881005139386870908363177752552421427177179281096147769415"};
            minY =
                HighPrecision{"0."
                              "0011175672388967686119452877936503680420978056943097961919136836510176758"
                              "4234238739006014642030867082584879980084600891029652194894033981012912620"
                              "372948556514051537500942007730195548392246463251930450398477496176544"};
            maxX =
                HighPrecision{"-1."
                              "6225530545095544093937832714855193369815166490586925235310445917701797841"
                              "8891616690380136311469569647746535255597152879870544828084030250153999905"
                              "750113975818926710341658168707379760602146485062960529816708172165"};
            maxY =
                HighPrecision{"0."
                              "0011175672388967686119452877936503680420978056943097961919136836510176758"
                              "4234238739006014642030867082584879980084600891029652194894033987737087528"
                              "857040088479840438460825120725713503099967399506797154756105787592431"};
            SetNumIterations<IterTypeFull>(4718592);
            break;

        case 7:
            // Scaled float limit with circle
            minX =
                HighPrecision{"-1."
                              "6225530545095544093937832714855193369815166490586925235310445917701797841"
                              "8891616690380136311469569647746535255597152879870544828084030252478540752"
                              "851056038295732180048485849836719480635256962570788141443758414653"};
            minY =
                HighPrecision{"0."
                              "0011175672388967686119452877936503680420978056943097961919136836510176758"
                              "4234238739006014642030867082584879980084600891029652194894033983298525557"
                              "03126748646225896578863028142955526188647404920935367487584932956791"};
            maxX =
                HighPrecision{"-1."
                              "6225530545095544093937832714855193369815166490586925235310445917701797841"
                              "8891616690380136311469569647746535255597152879870544828084030252478540752"
                              "851056038295729331767859389716189380600126795460282679236765495185"};
            maxY =
                HighPrecision{"0."
                              "0011175672388967686119452877936503680420978056943097961919136836510176758"
                              "4234238739006014642030867082584879980084600891029652194894033983298525557"
                              "031267486462261733835999658166408457695262521471675884626307237220407"};
            SetNumIterations<IterTypeFull>(4718592);
            break;

        case 8:
            // Full BLA test 10^500 or so.
            minX = HighPrecision{
                "-1."
                "622553054509554409393783271485519336981516649058692523531044591770179784188916166903801"
                "363114695696477465352555971528798705448280840302524785407528510560382957296336747734660"
                "781986878861832656627910563988831196767541174987070125253828371867979558889300125126916"
                "356006345148898733695397073332691494528591084305228375267624747632537984863862572085039"
                "742052399126752593493400012297784541851439677708432244697466044124608245754997623602165"
                "370794000587151474975819819456195944759807441120696911861556241411268002822443914381727"
                "486024297400684811891998011624082906915471367511682755431368949865497918093491140476583"
                "266495952404825038512305399236173787176914358157915436929123396631868909691124127724870"
                "282812097822811900868267134047713861692077304303606674327671011614600160831982637839953"
                "075022976105199342055519124699940622577737366867995911632623421203318214991899522128043"
                "854908268532674686185750586601514622901972501486131318643684877689470464141043269863071"
                "514177452698750631448571933928477764304097795881563844836072430658397548935468503784369"
                "712597732268753575785634704765454312860644160891741730383872538137423932379916978843068"
                "382051765967108389949845267672531392701348446374503446641582470472389970507273561113864"
                "394504324475127799616327175518893129735240410917146495207044134363795295340971836770720"
                "248397522954940368114066327257623890598806915959596944154000211971821372129091151250639"
                "111325886353330990473664640985669101380578852175948040908577542590682028736944864095586"
                "894535838946146934684858500305538133386107808126063840356032468846860679224355311791807"
                "304868263772346293246639323226366331923103758639636260155222956587652848435936698081529"
                "518277516241743555175696966399377638808834127105029423149039661060794001264217559981749"
                "222317343868825318580784448831838551186024233174975961446762084960937499999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999998"
                "109"};
            minY = HighPrecision{
                "0."
                "001117567238896768611945287793650368042097805694309796191913683651017675842342387390060"
                "146420308670825848799800846008910296521948940339832985255570312674864622611835522284075"
                "672543854847255264731946382328371214009526561663174941608909925216179620747558791394265"
                "628274748675295346034899998815173442512409345653708758722585674020064853646630373608679"
                "959493334284553551589569947877966948566416572492467339583663949970046379704495147267378"
                "153033462305115439398337489620790688829995544121329470083542063031324851802574649918811"
                "841750913168721314686845259594117073877271321100948129855768148854311161320084129267363"
                "499289251825454183589219881333850776847012695765759653605564436023656201715593927689666"
                "722447079663020705803627665637015104492497741636468267388256490548881346893345366125490"
                "399643435876691530713541432028774019720460412518615867427220487428798187975218145189083"
                "789569256006253505424279834364602251632081060904568420885095239536114820931152263030281"
                "510960264774165352065681307019706534613000980870274180073599675792637642781955545684799"
                "042499349664484696664255967704566757636662588724423743582737881875921718030051382037889"
                "290489109692568753441015738409173208113123967308658979176663175143694315257820174553147"
                "123170016957035575950933366295114766019884833988769556724019671487704066470206279346884"
                "135106807605702141085672617612292451502032657624487846200365061848714912762429355703562"
                "415874601262268865127750867393241343569164011564916899299372768445538725622751921653729"
                "817821602157920626985597292441933163224073178294594360672242702797827361918819138484230"
                "729076661426426985362989832789724044998791055083335658038803274851664414571288437875273"
                "224563220948706102106988695167966092545926424587115422685794647558401757470800335648264"
                "079478256438335825103058562986733937005823236177093349397182464599609374999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"
                "999368"};
            maxX = HighPrecision{
                "-1."
                "622553054509554409393783271485519336981516649058692523531044591770179784188916166903801"
                "363114695696477465352555971528798705448280840302524785407528510560382957296336747734660"
                "781986878861832656627910563988831196767541174987070125253828371867979558889300125126916"
                "356006345148898733695397073332691494528591084305228375267624747632537984863862572085039"
                "742052399126752593493400012297784541851439677708432244697466044124608245754997623602165"
                "370794000587151474975819819456195944759807441120696911861556241411268002822443914381727"
                "486024297400684811891998011624082906915471367511682755431368949865497918093491140476583"
                "266495952404825038512305399236173787176914358157915436929101074572363325485941890852869"
                "740547207893495401847216464403190377283252284936695221620192489736296880064364736168387"
                "364407018924498548865083468021039191959297956680950735192211117294472572549916766627933"
                "448887817452943083286885583312398485175176799261059924202852929698149189055624038345817"
                "137105972343305894016069756366146900994722671059851259685904384807877457739632787467911"
                "510131153946289523428325072359141842353594673786565545080689796074914723731611730652207"
                "617489386127306710995798281376160344598476271882907175206737026029853260611423835086663"
                "823662230295980596315444662332633999288401087473357654870330977577589362848522147173440"
                "735610957887415860064397118429150006470992796414518180674956981690673824954254846780389"
                "614421277378116832539432506584941678464221661845802155038330559171689808309409001223409"
                "532423237399439602230938059938450063207955010358991806721046406987145343805044073027784"
                "664829615863680850374937599099095789048554468885707214000693385718677220557313945289575"
                "462766123240889414425105608211440697730397516835728294646134282185031089311311699258534"
                "641301626048472044700465551168161448813975766825024038553237915039062499999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999999"
                "999999999999999999999999999999999999999999999999999999999999999999999999999999999999998"
                "107"};
            maxY = HighPrecision{
                "0."
                "001117567238896768611945287793650368042097805694309796191913683651017675842342387390060"
                "146420308670825848799800846008910296521948940339832985255570312674864622611835522284075"
                "672543854847255264731946382328371214009526561663174941608909925216179620747558791394265"
                "628274748675295346034899998815173442512409345653708758722585674020064853646630373608679"
                "959493334284553551589569947877966948566416572492467339583663949970046379704495147267378"
                "153033462305115439398337489620790688829995544121329470083542063031324851802574649918811"
                "841750913168721314686845259594117073877271321100948129855768148854311161320084129267363"
                "499289251825454183589219881333850776847012695765759653605573736881783528467753193053000"
                "281724117133569247062398777988899889662841499706014706016372541331507713879852825155309"
                "445733418035316861209556288978316282478143500096551357610725614057483872326044293314129"
                "792077777289475006632140252401733975684912603498348168568775217865832018883410276162470"
                "834740048255600659329223881004011060991907282879321090552836361563687680780220427483323"
                "293527090632178051813134981207196953681266541684913820792397357735300554966845235450747"
                "942390101292486119671868649365994478155987373346824092274515443661417944381090893731147"
                "361020889531680243992967746789389403706067885423681573530983486815289871675393650012417"
                "265434543050504019439701454624156569888621874101603997649966407799193057418611149232833"
                "039584855001941430933680923393544436450979507535811018411975678203452150800891864517137"
                "052035186135715348841397475928219859131636844030874374686820228572708751676865487969240"
                "162426098055037586559532217842753437863186592480806093936523929380404259520714584871920"
                "747692968032395327419735094412939817995275012199324226228671888756636304117844444282936"
                "821568138863483022553191437013266062994176763822906650602817535400390625000000000000000"
                "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                "000488"};
            SetNumIterations<IterTypeFull>(536870912);
            break;

        case 9:
            // Current debug spot
            minX = HighPrecision{"-0.81394199275609878875136988079089581768282857072187918528787"};
            minY = HighPrecision{"0.1940970983639155545870683099069152939779956506822050342848"};
            maxX = HighPrecision{"-0.81394199275609878875136264371649480889415711511526384374298"};
            maxY = HighPrecision{"0.19409709836391555458707132535458238097327542385162809326184"};
            SetNumIterations<IterTypeFull>(4718592);
            break;

        case 10:
            // Deep 64-bit BLA minibrot, expensive.  Definitely want BLA.
            // For Kalles Fraktaler etc:
            // -0.7483637942536300149503483214787034876591588411716933148186744630883569643446693595154771110299201854417716821473485
            // -0.0674478092770133511941572299505917091420441054948893985627670832452723589863935688773429127930946937306872903076587
            // 6.103515625e+105
            // 2147483646

            minX = HighPrecision{"-0."
                                 "7483637942536300149503483214787034876591588411716933148186744630883569"
                                 "643446693595154771110299201854417718816507212864046192442728464126"};
            minY = HighPrecision{"-0."
                                 "0674478092770133511941572299505917091420441054948893985627670832452723"
                                 "5898639356887734291279309469373068729038486837338182723971219233698"};
            maxX = HighPrecision{"-0."
                                 "7483637942536300149503483214787034876591588411716933148186744630883569"
                                 "643446693595154771110299201854417711330238558567565796397623653681"};
            maxY = HighPrecision{"-0."
                                 "0674478092770133511941572299505917091420441054948893985627670832452723"
                                 "5898639356887734291279309469373068697845700777769514407116615856846"};
            SetNumIterations<IterTypeFull>(2147483646);

            // Period detected:
            // SetNumIterations<IterTypeFull>(16125357);
            break;

        case 11:
            // Low iterations centered on minibrot ~10^750
            // For Kalles Fraktaler etc:
            // -1.76910833040747728089230624355777848356006391186234006507787883404189297159859080844234601990886820204375798539711744042815186207867460953613415842432757944592767829349335811233344529101414256593211862449597012320484442182435300186806636928772790198745922046142684209058217860629186666350353426119156588068800663472823306396040025965062872111452567447461778249159612483961826938045120616222919425540724435314735525341255820663672868054494834304368733487681716181154843412847462148699796875704432080579735378707932099259943340496912407404541830241707067347835140775121643369435009399783998167487300390631920225320820254728695870927954776295601127590160537830979541161007773653115528776772709046068463437303643702179432428030338435000806480
            // -0.00902068805707261760036093598494762011230558467412386688972791981712911000027307032573465274857726041164197728426795343558667918574806252145218561840172051237592827928284001732546143539590034733383167886737270960837393773203586137673223357330612418801177955434724672703229251690060990669896274784498860242840497701851849033318045752945025236339275315448898727818174162698044355889583608736485127229946909629832632750839622459160046934443901886226621505107842448159935017841941045354364042323342824193503392819832038805610324279071648709936971740195644028490370367453551798296170098754444868309749391071986751540687669916301587186977458689535397730634082592706549618026786866330689857703728968664890931318074163369981746242981795864371396
            // 4.0517578124997939E712
            // 2880000 iterations

            minX = HighPrecision{
                "-1."
                "769108330407477280892306243557778483560063911862340065077878834041892971598590808442346"
                "019908868202043757985397117440428151862078674609536134158424327579445927678293493358112"
                "333445291014142565932118624495970123204844421824353001868066369287727901987459220461426"
                "842090582178606291866663503534261191565880688006634728233063960400259650628721114525674"
                "474617782491596124839618269380451206162229194255407244353147355253412558206636728680544"
                "948343043687334876817161811548434128474621486997968757044320805797353787079320992599433"
                "404969124074045418302417070673478351407751216433694350093997839981674873003906319202253"
                "208202547286958709279547762956011275901605378309795411610077736531155287767727090460684"
                "634373036437021849939558081154827035966322541319864476469606578987384818141802086140871"
                "082590860960091777689102849781889131421151893374373923872281234460683461953146946783351"
                "680424907509438258067061062136495370430128426260576349247467954001002250166013425144708"
                "446174956227820466386363571779749724416127365"};
            minY = HighPrecision{
                "-0."
                "009020688057072617600360935984947620112305584674123866889727919817129110000273070325734"
                "652748577260411641977284267953435586679185748062521452185618401720512375928279282840017"
                "325461435395900347333831678867372709608373937732035861376732233573306124188011779554347"
                "246727032292516900609906698962747844988602428404977018518490333180457529450252363392753"
                "154488987278181741626980443558895836087364851272299469096298326327508396224591600469344"
                "439018862266215051078424481599350178419410453543640423233428241935033928198320388056103"
                "242790716487099369717401956440284903703674535517982961700987544448683097493910719867515"
                "406876699163015871869774586895353977306340825927065496180267868663306898577037289686648"
                "909313180741633720660506874259671345628579528566082199707266953582331381477204845958749"
                "313226525982433412485094171338066030864231223182859124950336577946409926601514095926041"
                "462597061333401178051740876104003994845997683393109587807745687517577651571044175605197"
                "417318412485990475703151931020039994351872796158"};
            maxX = HighPrecision{
                "-1."
                "769108330407477280892306243557778483560063911862340065077878834041892971598590808442346"
                "019908868202043757985397117440428151862078674609536134158424327579445927678293493358112"
                "333445291014142565932118624495970123204844421824353001868066369287727901987459220461426"
                "842090582178606291866663503534261191565880688006634728233063960400259650628721114525674"
                "474617782491596124839618269380451206162229194255407244353147355253412558206636728680544"
                "948343043687334876817161811548434128474621486997968757044320805797353787079320992599433"
                "404969124074045418302417070673478351407751216433694350093997839981674873003906319202253"
                "208202547286958709279547762956011275901605378309795411610077736531155287767727090460684"
                "634373036437021723023848332310708579010440266742180377555833452293488480928863794034113"
                "380717085247450477413409955521865484776069659968888982957089040699836537328435625787365"
                "406852114641414906701003244203291394842947052469974908497764264157006872226432364177441"
                "648413430211646947924195417953122886076275359"};
            maxY = HighPrecision{
                "-0."
                "009020688057072617600360935984947620112305584674123866889727919817129110000273070325734"
                "652748577260411641977284267953435586679185748062521452185618401720512375928279282840017"
                "325461435395900347333831678867372709608373937732035861376732233573306124188011779554347"
                "246727032292516900609906698962747844988602428404977018518490333180457529450252363392753"
                "154488987278181741626980443558895836087364851272299469096298326327508396224591600469344"
                "439018862266215051078424481599350178419410453543640423233428241935033928198320388056103"
                "242790716487099369717401956440284903703674535517982961700987544448683097493910719867515"
                "406876699163015871869774586895353977306340825927065496180267868663306898577037289686648"
                "909313180741633668322758782658686504512460185534571338350380449080021559310694091711156"
                "682758116900515226747059221239231515511117563446914128329486072019848707947252239523709"
                "242750148849525861066577473745400466613298543939063033542652770299644615337827641544194"
                "42516701718236383062903641742731769900290969182"};
            SetNumIterations<IterTypeFull>(2880000);
            break;

        case 12: {
            // If you want to see where precision runs out with the perterburation reuse.  This
            // coordinate is close to #14.
            PointZoomBBConverter convert{
                HighPrecision{
                    "-1."
                    "97308488401090040063694958492443779753220015673349789608662169368426273598640826637"
                    "76276894095720699803660849826377150236678266869162895747277682038789153500848403020"
                    "02338982933530918277669401478364960465526231129559636673835887666163842017990181946"
                    "96456029544018530169538910655163874159631160982932064231164379520817128850561282200"
                    "84977824592600941226781504831500617788951253600541052218205996303251019902118774731"
                    "94209720279420685096101082325563827641802006676447079563625439223552590683152787040"
                    "35870375461216535130276933432990851616241139414552036315452938086346174282061513753"
                    "11386416614654294781896847769011027399593827416596348585926638415387831668034450626"
                    "37231364917854421700620328817801375556117885055145015323164322089136214501067346357"
                    "39248722022468466491452278909525307347274576564635603129247264387268446197475571214"
                    "60568670138425980284333104053596071176664960072515836453762154139949467745612639240"
                    "80816769615132784776074810396671460033759548303142261919287488108511665292786520220"
                    "47218282832316623233367232804420628842999532153091182677889608570762793984250718984"
                    "68708755475674163910816740197285372853659099814362521184595483275079626485009480862"
                    "40094429194705896531858831493681063219085486032511247835905665703710790144750713217"
                    "46822627477961183585221011259411507847295013578693527476143912602324410577919951255"
                    "71371042201439402671820204346175701446696738836487696921542438015135186217375500175"
                    "95326878345321054448616765275696654584913054197929303287302209688152967055109135540"
                    "30500052757408196923315104000341749796173334047463765202724837923004153440717768387"
                    "40129369055199859559509283752264974608574663617596269811601088658759554717836023330"
                    "84546236272083452765962849673176348335319260478311505242710419866269714408080887949"
                    "26501680639131131259400767024040635636367169541854325951224019070548891135245989884"
                    "05542551394202806862082645514207243248096734228833972550421057513809216757121943858"
                    "41315384822688245503297879776122486500323991750543492700567918856974145567895431837"
                    "55949924343402419085698861928451597373738231915892215661750433528398250286798552040"
                    "27028554775402555195247442638315657622767106672548010847044590437898445050141498232"
                    "33249351337314624492588654437713950233472130513365820414244873505742304161069720038"
                    "02025815871230947070366477365979422635112157530625598997953597908769467862648814783"
                    "91631409936911370770328323786672662429231244709700210799990205546005576074315958539"
                    "51977112654613750097522433307144563062925986579737655884323956404772286898658859715"
                    "79792512212883746539208153550113539994002494214471966120413210248427452598833238026"
                    "56535674033629789084088345541956107675518697615475494177270806957776425591376648293"
                    "06417471773962999982075208404876602043224356596817214726337350254861268127961957197"
                    "39298182027936494343902768750477536159272267781743045693317634382292280924052540450"
                    "32621706688741083862962246706118727928460232044047528295996458418281083027118525433"
                    "53318205530402454626385152445570623550165301711064923021178620487891757425753574104"
                    "94603196665551068344163266082055772947321730032876022970207655957276117083610064542"
                    "13156981399643017240356626141748716877193844421313116826551006513038998269330698839"
                    "99424322353900165117112094644443317888402595617559312269340770771004041764746297797"
                    "06455895627856965249421782473143591037834255694170898813885818529903595016723524695"
                    "83419024732212843516278403332320787399066010024228114485929370778443567707315735642"
                    "29270919815113774360820308115332881959537780136730292724703068497525588844915776942"
                    "46597663568019018306664221689975744852080512223111253177116803914302702689490853082"
                    "28028609363418508677136682573466806390101093781871210170148452423627124083209403255"
                    "16875486109486586363012371949652013475793566890709250314883741685525318071046963252"
                    "92871579795209653940085962309478342764482576412510729339939577202798326663272861791"
                    "03785675334983739251791238544748561797064639811992461952881535413949545867116700507"
                    "61842400335354401893828989674510294415595258638669609101897789003104242436373996265"
                    "78616035717728086260868119156852151195485336067038311320769949938766043553784034991"
                    "43026063298436095903035099734771247711819317003806278255102250068293453071658755498"
                    "90893793424840039458995654461191198704102876688854929830894785754848462899988623774"
                    "75081159831982448666670961722199244591903714869792526857023216742974300339307010506"
                    "12121543992077476857820817154452697215023134671062081007089709186925057129010775047"
                    "25972367706334204531970855016084561469651568872545091789266025083241787316976851855"
                    "53844408249306132311368802731624425632525576623289420111811132346704804176520501013"
                    "25707008957863991756742092253824396923018735356096999733393307249519984295468686132"
                    "39315171294383552523411904962436933869637873479615701565328716956388331872251502617"
                    "90703365625122889067278553861747164255971540156295801363122446388931422347152810362"
                    "78417859278826579113502622837131830233065875699228555504935634775017160070677126557"
                    "11741546759888144175212491667741251186039790163679649894280795714850167219389617224"
                    "75181180670849356061383838659301994168398056123316663035782816581817929698703448274"
                    "01677138154890747354654527234450800757117612046941170041206370098590520258016438805"
                    "82734624150191767898237087878560322331490858562409750071724255124158063891071182194"
                    "41283342839681873448714606849121631399989489183539468492561918015603332064995098283"
                    "08878713332998322363074328871546467481830041012214398249395747431331501885417029901"
                    "99371361972544935452462339381634778862405067370498745003980699857810494678595885975"
                    "86857469601378089893613870602455525327609238589182284986721918903574874822102742652"
                    "48742954981989299289891457635665470665790060046923809716128463303325114387633116449"
                    "21405677658000481629592201773624381582766166623543178898657052997516811633246892132"
                    "12722200026906039173907271317980925159844222461467741354496053499023807467134749796"
                    "21550256594527573610402860693749337133458831050129073160922158048783746076551383646"
                    "79096934883117896837242496547774537431631545463764784174281796141673096329202817268"
                    "18012490235990486846554698476092664371027736311614324344702326706564632742301288369"
                    "24678088554760299336919422117647808313693753031530294806969355850778924305440175454"
                    "93203258594114614457282355532174372898911362715601365358713292298551513188561964831"
                    "91020252460145662944630584305881657040523143725116965869482362977529893481353911136"
                    "85366020493684260080871925719990080658959354229404922225019185016427250018365369186"
                    "99060251286808107731820441498428024634359389008014816695953457867167468392874667970"
                    "48570165600082422515209083381602092662223186033411184910951850255994889766216118813"
                    "62748794581880524725044959323427227683945289437133067825464024921188886617697054409"
                    "03214672005237253412367960690335872774369283538219595245731223967904408920966764553"
                    "58632246377995583266457117084999111511039407028115328000992506609966775189464675206"
                    "26987644977524189063143444575426735178482262547395052745836109904046956045308464759"
                    "120612876487999906884681472446490531921704919565674912015978542651e-01"},
                HighPrecision{
                    "1."
                    "10380129460262878671753405373211067547141793779098296086981118923118281163395933406"
                    "41577619320630631272359545249104602024464660368108622972257186686981113108472441866"
                    "35991363141917337570853402390895681866278184524313584991099338975501381491896291256"
                    "32174225603859894330856011786974257565889079312554554612541611036355421283696090526"
                    "45949788789782753943016708347835008968061806510471664104783265965008441112828566815"
                    "31040466490328311940277267787993163385904307364013945151036346313759637881311709653"
                    "57365677737296347068082915263755869837766823928630835304981671518066422455602862153"
                    "39977076499764785829538273147815417500738610285975563214323780827256126559177169128"
                    "84044991624970599713901556142817090179224089970659400299088416529327612873953041879"
                    "21453028224837043151419599901744693159292673387197049631858342623550634806182850594"
                    "45089066978996354473817691537479272678180963179437331577662923089833440065737104496"
                    "42073490102006233532563389079143770504533452824103080535247796577837149404624310098"
                    "67405126846735112633536340608844981131980057257402812030622871738390455461389757734"
                    "93464113071526061452896370126461759527654097782052908530271536005036257885481328814"
                    "85420008410551409504348363634906722755515540707886529775989765843429272940214664375"
                    "23774372326872317858043981227806904521309714726664782677388332883673576524183696579"
                    "21421509385464004483250668075409981057957028155380383774844014946124157709869335895"
                    "09145894335518966134826175660681916888190464321363376049127004242159680947991193224"
                    "77181654375682644246932531780723455845870281123253773614191679984071872224775351043"
                    "08636263925643937858005068044166766610159228671457426217966526451690376098935640862"
                    "68933966072975497716040181273569635992625543364783619033103022590329463531089770553"
                    "72569621991371867710533056631717075095984905239247213040127133412279025191329121386"
                    "56784187629279520283464682372091116074251283483215677118879253318704335016015638601"
                    "02618116998479234224893465141805358078510834542464730423067979312454459015044999990"
                    "69460845437241387664022478111911293130836721598077694690233955875465065094677876724"
                    "42265229039498439979201085794533651266184631069218294954302892235113890258245701532"
                    "68281020378788845303802201330521411867506088593941246434696262472963197646939633447"
                    "42733198385464659737498861168911541334986670881173939738174628182269367111271116485"
                    "16076590922700976744541375087032318027536398945874148381707159527704568353680858388"
                    "76566648315768857559503791954562318629453189013734252761778536338185918338155512025"
                    "09865488689756671993958726133575331917304729746521102205511973042989517498877058803"
                    "57469174641040381109130339080759503421828349453332549601783841275450597078683783143"
                    "80754958557853572150117293761953259810458660631556633967946669039063781732164974092"
                    "06910158476331518327808846705367878475237434817819633617913536554052770336997309699"
                    "00474065899818873788518055470405347184686467053608834170850165628554485694032539624"
                    "71796855264373339070508156216564803107685270217899788662994925437159825307003171287"
                    "66894016641663435892263502103599517011869869323463074607461164058940139177916288315"
                    "45553622598039668371221196573051562382869634534483507552145361562467748554076022224"
                    "64531908764408287213732150912964564893848409910096398924707504427595383209990144764"
                    "70690293996592964874696878801102388843903035664204282230339980623214384338872661875"
                    "16409781969401176250936898533615284296924590661852979836792229950450877310239293243"
                    "41984530671209223055558125528665152459671853303833155434939057661780796652710156436"
                    "03665523404126198286173911710498489947955894248023937372240043143162694209509685655"
                    "58670732270702832101704070575710647956999020673224209765787061270277055919813769714"
                    "15001169613113277140959558467703376627747028899541838453516683400146820025232862940"
                    "58410494676119401900602773613305618118156706572751351764125759949887229930877907648"
                    "24436411050333568729417091268489487239928791416409562990815995586612811144303083742"
                    "78080467630133731579384437690572023582056599515949939137356028031023240466854117528"
                    "47682458714817108641561393767758703562878524792639702373502310289333486593071487993"
                    "18322103425730473491502309993942222741961913311655476198442624558204326189065255491"
                    "71964858684904166149981512562702192062466216903482614293439877285139051498277691998"
                    "26839471318128861506907992102342439655728186190912856704021645199281536438470379813"
                    "80132541934184241820415132669218704569402639920739734046970463125908401501577960387"
                    "71787457907495935457236542929184335310858270705514530653108053052364743591681888875"
                    "15954831843731843606581892401513308351350852009833858160282617293335908550628700900"
                    "56433718577094029359344223255268019287066750013708586305552423906230682172136742570"
                    "84053613418011703912274484254427074866005885131206363053184583072217638857248281581"
                    "39322529151085157438010796913029857744759380904010557967626206817954073081603031431"
                    "40517216246924013539058244982513995657908481871389355139571180916313800267276871494"
                    "08514338190323977543129904892572606352707650372926356124791390673287757286614808099"
                    "52160390190278733730063117967914315918866396687898982093146660461318345025006854417"
                    "78933781200462436961733655747078895309963754838752239468209270026317583112802802239"
                    "91350794631032655210242205382339176258657891144866312781944472576496276425326256588"
                    "32596819899073531527415800759433845171172964483847168663094505760109662020933951948"
                    "82619269384816644573182809038618254600088333274182508740974285137888224275178520102"
                    "43336278167023404698335514300133668548390266534241311683638045023264863107420213756"
                    "87394883664903277338329930319091961191474618443905795312131530515414769184605937058"
                    "79450601392877424958052824329194856978479469281329764682033113052355629351290383907"
                    "43556862122445071551287499575504180656657017149005132232260588006199053754876930909"
                    "69210872696738534856789763852636456858781437820181410030777977810457894743403959120"
                    "91697012642458198870769412315746798645078827380398828763262497374496402919799955681"
                    "11846954232404479668025619495471153947199148769956079097986419471562617045231980783"
                    "93358214553427958065421392543067336523131113320277095960405927051546875592263849757"
                    "18745115744715689239586338455232337207917015300026972551829720471995871182230864222"
                    "24067626778849992340020226467303032935526410684855619651594541801280266893703312001"
                    "26016618405168844279342074156741313933142199313881360080713680316858942112232433669"
                    "95751282488626930943464429816450318167617476468739302390283560260867620522032412035"
                    "95172243922358422447571025451614415490353348979827919992459597505196154337803866958"
                    "90272178035398666009514883072512810156084762896945411668894917211769611117047507785"
                    "61110790760291691975406868896786614107277437831392672767421115138583628984345752439"
                    "92467430409205262070564363399295781124123425963939087687445870377411109637189958612"
                    "14450398185250200679605256634126597456745127797239460706353701273687357748723790008"
                    "03048871769715838343188429884448168047502526959137438527676581470700198935881581817"
                    "257151228400345298806207917216983888129443335080070098504609614366e+00"},
                HighPrecision{"5.10950250472583381635e+6710"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            SetNumIterations<IterTypeFull>(2147483646);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;
        }

        case 13:
            minX = HighPrecision{
                "-0."
                "198111600219673444807552285098794535437728241470875435752092821644604057966618589119204"
                "634147133554047306723399344586502123166791852244574645059479199518770447525743827045648"
                "248206215279344484040164933643205472293065196111715117157121769007179535831873152592592"
                "380535856614977769168004729410694921983490439479255134898961730705592943150343181762281"
                "983979426487228667570008297810243897996522125755423066463590274440694323707259382619156"
                "459966476609736571756072258537906749756185719175702305279572045264607014631739185152935"
                "018694484591850643445769994563598053850793835577765630048444683540083307364814184352968"
                "380120085005646941208192465553041685490219938103868444386742359911621621625526938685417"
                "352217301202267883975081432951152288915714403220710400358934691454323103471531355351812"
                "177770689580438051119804608582268412717189328419520989152213915137541303044546709335094"
                "594438221338371418722243617485380490801266669038347426812308930225779748804894215391155"
                "626562543109650255149100343371162294783221321454993343473818721954037869028880546064278"
                "383976310026873206872827456975335662171490851663080840802674577998999791968647124588462"
                "653688065305440561301461498488748184666221888029417841844644748340093107495901144135194"
                "685738334918629148761370090291216734598694339393515618419191541427112357798657939854556"
                "742081741022161596931437182380236850923273237276422892008348386951341746871678724814697"
                "230442153182201457599477934225402641281021437058244751099400311484622272533498686227725"
                "782443003131631757722024674204499453172453809886001670124646694979916009685246443382288"
                "078421788139328990006728259234615381831987626864548774698098027134345344450480587039233"
                "957893503239289621965395304652473004888909443566138618492472506696695591679777405466092"
                "726591063179078189308510237894395353459062165814678128194177531951349720847647292680583"
                "931075278991739211179593767620119405456704204102803865064779209634012165672865462269297"
                "651689859598283110370262158601908871169089798246556963197327330743451191117925581498690"
                "097810488357868812535335084998489691195880461864918562383917950923279661707998589911269"
                "729744936257292777150391663058737187998166601765841152667444301679476811490109341638052"
                "910110037649817205267383632433601194097526504746435748888288975852183039419281898366898"
                "490165764506973760837221951461565584264800995777346185313055332966732531505379701908272"
                "581814938194796875980222219743337989960393358475218540425073209034556528072572589035263"
                "322432282435115596154603100199044482094431370312604948184742466899244637110824058137373"
                "580216313930737328584827780247634365658524371546950571292163151489174876772226648341564"
                "785523410311136235436679838079073502825936660169781229830388229390991144050837751271456"
                "327398167954601850840646208635430656281144401565817560667525071549685048633371717638261"
                "052242817911789329270209898132125892929991182626482480160388973430921402251253189005436"
                "954717647464196445404184314692938118234840580159654070326499807553825365278744434383060"
                "284102201058599742922959152871543319920260722829205377298808887586272626564830371807858"
                "346382576921751517228602503871833181329198057110343523986979900258593705766938826095524"
                "236499443866516400824169897310065071847089667764414877902052856373092158442439421764377"
                "911609531206627182549323965342450823664931634593078269963041898851270454222873662324582"
                "588913280036584032306924717536189175782024760633109868078548563889192551984404668242363"
                "581278533423797517628585791320910995687897980718637273843382910223459688966244336284148"
                "781274622429171537617300868273140790221397769111228689080835391403927339854496284903663"
                "700367108115146500106508553441836924391713378950535155162859896327123516793807899055210"
                "285602295816023962157632709946385611103382630874103109885287567086480879683436401669629"
                "609021018647164777635677643894253874702751968583693606126555224224766114703716687664825"
                "951466422362025844599891668667880129606762024124270687871589323377116317387185331052126"
                "443341714503663602898772188795822318183912847582163880780852832245963966388010215643277"
                "479574574891744555789378479154266970433633006744368752420731200789512710969363916351566"
                "415177932208206101461618905926534348303152315841751282585409806732186505020763251404647"
                "462038002027398496887944634511637661472992356536286438022927560974160872894302055553525"
                "225505442612299706281501414733571541669152931089533436002281015166069851204283013003972"
                "37003185516353950264194714068535510113957115217817241050181960131736113134261363991"};
            minY = HighPrecision{
                "-1."
                "099513354265785711580863403134618967787009713630357293871554968929538942563258150313881"
                "224697763908772670990867742316118433518762927426387318843155318430511775352132934386612"
                "317977200122877100210736355033051770357999814059340906635731931139302629240805948070198"
                "390197048849989673582855385380283430190719760814989271358583343957716822777055572292844"
                "471227212959294924516709069109414872189800872726750746501940665304095713691670611851136"
                "894912873164636348093489376544023385965846691473088983927261190377467231573264102175454"
                "281815335420469527834834451138842916840082148809435565693669997592948229992384585056767"
                "442512294035976034796973628515135633793211316962417352267801343440813913057813228346606"
                "687207995053356114901582221712525892739105210526257661663294504578012022241942073453527"
                "319917129998014759645709586989947733935679557661787851001082799863017758817353885863783"
                "773145600765831031170287732436965954118090753687913510301876280057395083430796559988987"
                "879075064315676951665559748039076155397239101528372606746956374028775914420385853278239"
                "362932695894734534607586289867730058293718742273331688802857248602219814686939414193257"
                "644509326993869660910900461991385691212125161940707578099361558818630881213641990717782"
                "603616606922364210976164427024440851360044707564674755784128231997544714701991585061507"
                "256910828103255137657485601414505023962840547266577871594060608463979810193640852045766"
                "461574200840511858059560520467774833170916305407678181291567850685335123991827142535701"
                "882335426051842483328920920496146861047616132979844109842803008515472937044164806862416"
                "648701220287227494418733207871029478787328676609514606762997094260255687176213709415833"
                "288863651629314955743598384277765129333244374745424325559357955196807785224019903248737"
                "042789067455458487680815681129614565682660326099968166794905468667590339205381210825611"
                "618116038249461399715319913860356839382518677529211027588682956897111977598491918407735"
                "406370656690341907363147909585470286827619187538767438127753109444444151280017000080809"
                "458240702305594000301154045620425094853504394717956411684597185322407804582353643569846"
                "489125130426447126209614713223633493066999659205988957358410538409818656025005173984805"
                "407884578451107779775269172690071655129840811861725936474199989608497730789846568806332"
                "981762521438075054933080287039005346844253006400837904365867671968138441595483646401372"
                "372132865658456424501709348983510434823338001221059884507505745532078494277544334639995"
                "106961427935555424001182938417430836196309602822433534025971513716252823575192361680349"
                "215353368726387808593773647662232470573065643003225882223785820789409117600862036086755"
                "950283630091639112913449339323861684539232760592990453851238516482945369881722377701183"
                "522748850493415615158637204063870341108221946194800208723903266579109555217947933525029"
                "253934923483943351982096951985943080875556735676456543880396106482743455591549711377897"
                "707960544137721089309249539943235560899103619947751388806682815632679558389213345661459"
                "283056644254468737407612816338832047569948271589391544161150122701170547658678269044027"
                "759369280091028299995540647478727897042359287537848553957054962558564360345580169340756"
                "257161724603340177479659112912303962739246517343546674433318931238295749398732152506248"
                "003459252907751538758448456610068156518162096380530926057518593571577344907116636001281"
                "859580682339473242977284203256487612350299121494483100312719437020056069756329978649755"
                "801870392705691991553094158081736815524720905287351998855588373378728811683488254255410"
                "118399447224522580991579818654586271318642900527317990389055095859329563717487746794698"
                "222391672846172903695409422940277802620213802049100585954775071544153665929138941242311"
                "137638769187947473412931982559796979746286138011699115397450920728033795187324660769781"
                "890439167215104408511848724362620430054495768921452025437911700681948972507926018313928"
                "955501944633260019264071714546785536473858557681588193128419991770724668507792785023002"
                "944439274034171021453291770129960888217411938031104211710057129060666523029380411011684"
                "424134887989908624112272628965013632801164291138561093390693236971368751398401887782243"
                "869449010640403901875745772744519453440564732829746425928447202792811542584995187216941"
                "328231731659318395978815238843316499724087846837442422931062042417544975010728523403885"
                "812338231129753460256068635982749934119532125610835623298498841125496995090044721021405"
                "9234170765920468549231098085902093319040057298242654348820124671419516397663044813"};
            maxX = HighPrecision{
                "-0."
                "198111600219673444807552285098794535437728241470875435752092821644604057966618589119204"
                "634147133554047306723399344586502123166791852244574645059479199518770447525743827045648"
                "248206215279344484040164933643205472293065196111715117157121769007179535831873152592592"
                "380535856614977769168004729410694921983490439479255134898961730705592943150343181762281"
                "983979426487228667570008297810243897996522125755423066463590274440694323707259382619156"
                "459966476609736571756072258537906749756185719175702305279572045264607014631739185152935"
                "018694484591850643445769994563598053850793835577765630048444683540083307364814184352968"
                "380120085005646941208192465553041685490219938103868444386742359911621621625526938685417"
                "352217301202267883975081432951152288915714403220710400358934691454323103471531355351812"
                "177770689580438051119804608582268412717189328419520989152213915137541303044546709335094"
                "594438221338371418722243617485380490801266669038347426812308930225779748804894215391155"
                "626562543109650255149100343371162294783221321454993343473818721954037869028880546064278"
                "383976310026873206872827456975335662171490851663080840802674577998999791968647124588462"
                "653688065305440561301461498488748184666221888029417841844644748340093107495901144135194"
                "685738334918629148761370090291216734598694339393515618419191541427112357798657939854556"
                "742081741022161596931437182380236850923273237276422892008348386951341746871678724814697"
                "230442153182201457599477934225402641281021437058244751099400311484622272533498686227725"
                "782443003131631757722024674204499453172453809886001670124646694979916009685246443382288"
                "078421788139328990006728259234615381831987626864548774698098027134345344450480587039233"
                "957893503239289621965395304652473004888909443566138618492472506696695591679777405466092"
                "726591063179078189308510237894395353459062165814678128194177531951349720847647292680583"
                "931075278991739211179593767620119405456704204102803865064779209634012165672865462269297"
                "651689859598283110370262158601908871169089798246556963197327330743451191117925581498690"
                "097810488357868812535335084998489691195880461864918562383917950923279661707998589911269"
                "729744936257292777150391663058737187998166601765841152667444301679476811490109341638052"
                "910110037649817205267383632433601194097526504746435748888288975852183039419281898366898"
                "490165764506973760837221951461565584264800995777346185313055332966732531505379701908272"
                "581814938194796875980222219743337989960393358475218540425073209034556528072572589035263"
                "322432282435115596154603100199044482094431370312604948184742466899244637110824058137373"
                "580216313930737328584827780247634365658524371546950571292163151489174876772226648341564"
                "785523410311136235436679838079073502825936660169781229830388229390991144050837751271456"
                "327398167954601850840646208635430656281144401565817560667525071549685048633371717638261"
                "052242817911789329270209898132125892929991182626482480160388973430921402251253189005436"
                "954717647464196445404184314692938118234840580159654070326499807553825365278744434383060"
                "284102201058599742922959152871543319920260722829205377298808887586272626564830371807858"
                "346382576921751517228602503871833181329198057110343523986979900258593705766938826095524"
                "236499443866516400824169897310065071847089667764414877902052856373092158442439421764377"
                "911609531206627182549323965342450823664931634593078269963041898851270454222873662324582"
                "588913280036584032306924717536189175782024760633109868078548563889192551984404668242363"
                "581278533423797517628585791320910995687897980718637273843382910223459688966244336284148"
                "781274622429171537617300868273140790221397769111228689080835391403927339854496284903663"
                "700367108115146500106508553441836924391713378950535155162859896327123516793807899055210"
                "285602295816023962157632709946385611103382630874103109885287567086480879683436401669629"
                "609021018647164777635677643894253874702751968583693606126555224224766114703716687664825"
                "951466422362025844599891668667880129606762024124270687871589323377116317387185331052126"
                "443341714503663602898772188795822318183912847582163880780852832245963966388010215643277"
                "479574574891744555789378479154266970433633006744368752420731200789512710969363916351566"
                "415177932208206101461618905926534348303152315841751282585409806732186505020763251404647"
                "462038002027398496887944634511637661472992356536286438022927560974160872894302055553525"
                "225505442612299706281501414733571541669152931089533436002281015166069851204283013003972"
                "3700318551635395026419471406853551011395711521781724104902645634823828075018859827"};
            maxY = HighPrecision{
                "-1."
                "099513354265785711580863403134618967787009713630357293871554968929538942563258150313881"
                "224697763908772670990867742316118433518762927426387318843155318430511775352132934386612"
                "317977200122877100210736355033051770357999814059340906635731931139302629240805948070198"
                "390197048849989673582855385380283430190719760814989271358583343957716822777055572292844"
                "471227212959294924516709069109414872189800872726750746501940665304095713691670611851136"
                "894912873164636348093489376544023385965846691473088983927261190377467231573264102175454"
                "281815335420469527834834451138842916840082148809435565693669997592948229992384585056767"
                "442512294035976034796973628515135633793211316962417352267801343440813913057813228346606"
                "687207995053356114901582221712525892739105210526257661663294504578012022241942073453527"
                "319917129998014759645709586989947733935679557661787851001082799863017758817353885863783"
                "773145600765831031170287732436965954118090753687913510301876280057395083430796559988987"
                "879075064315676951665559748039076155397239101528372606746956374028775914420385853278239"
                "362932695894734534607586289867730058293718742273331688802857248602219814686939414193257"
                "644509326993869660910900461991385691212125161940707578099361558818630881213641990717782"
                "603616606922364210976164427024440851360044707564674755784128231997544714701991585061507"
                "256910828103255137657485601414505023962840547266577871594060608463979810193640852045766"
                "461574200840511858059560520467774833170916305407678181291567850685335123991827142535701"
                "882335426051842483328920920496146861047616132979844109842803008515472937044164806862416"
                "648701220287227494418733207871029478787328676609514606762997094260255687176213709415833"
                "288863651629314955743598384277765129333244374745424325559357955196807785224019903248737"
                "042789067455458487680815681129614565682660326099968166794905468667590339205381210825611"
                "618116038249461399715319913860356839382518677529211027588682956897111977598491918407735"
                "406370656690341907363147909585470286827619187538767438127753109444444151280017000080809"
                "458240702305594000301154045620425094853504394717956411684597185322407804582353643569846"
                "489125130426447126209614713223633493066999659205988957358410538409818656025005173984805"
                "407884578451107779775269172690071655129840811861725936474199989608497730789846568806332"
                "981762521438075054933080287039005346844253006400837904365867671968138441595483646401372"
                "372132865658456424501709348983510434823338001221059884507505745532078494277544334639995"
                "106961427935555424001182938417430836196309602822433534025971513716252823575192361680349"
                "215353368726387808593773647662232470573065643003225882223785820789409117600862036086755"
                "950283630091639112913449339323861684539232760592990453851238516482945369881722377701183"
                "522748850493415615158637204063870341108221946194800208723903266579109555217947933525029"
                "253934923483943351982096951985943080875556735676456543880396106482743455591549711377897"
                "707960544137721089309249539943235560899103619947751388806682815632679558389213345661459"
                "283056644254468737407612816338832047569948271589391544161150122701170547658678269044027"
                "759369280091028299995540647478727897042359287537848553957054962558564360345580169340756"
                "257161724603340177479659112912303962739246517343546674433318931238295749398732152506248"
                "003459252907751538758448456610068156518162096380530926057518593571577344907116636001281"
                "859580682339473242977284203256487612350299121494483100312719437020056069756329978649755"
                "801870392705691991553094158081736815524720905287351998855588373378728811683488254255410"
                "118399447224522580991579818654586271318642900527317990389055095859329563717487746794698"
                "222391672846172903695409422940277802620213802049100585954775071544153665929138941242311"
                "137638769187947473412931982559796979746286138011699115397450920728033795187324660769781"
                "890439167215104408511848724362620430054495768921452025437911700681948972507926018313928"
                "955501944633260019264071714546785536473858557681588193128419991770724668507792785023002"
                "944439274034171021453291770129960888217411938031104211710057129060666523029380411011684"
                "424134887989908624112272628965013632801164291138561093390693236971368751398401887782243"
                "869449010640403901875745772744519453440564732829746425928447202792811542584995187216941"
                "328231731659318395978815238843316499724087846837442422931062042417544975010728523403885"
                "812338231129753460256068635982749934119532125610835623298498841125496995090044721021405"
                "9234170765920468549231098085902093319040057298242654348771978680440440048326679578"};
            SetNumIterations<IterTypeFull>(113246208);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;

        case 14:
            minX = HighPrecision{
                "-0."
                "197308488401090040063694958492443779753220015673349789608662169368426273598640826637762"
                "768940957206998036608498263771502366782668691628957472776820387891535008484030200233898"
                "293353091827766940147836496046552623112955963667383588766616384201799018194696456029544"
                "018530169538910655163874159631160982932064231164379520817128850561282200849778245926009"
                "412267815048315006177889512536005410522182059963032510199021187747319420972027942068509"
                "610108232556382764180200667644707956362543922355259068315278704035870375461216535130276"
                "933432990851616241139414552036315452938086346174282061513753113864166146542947818968477"
                "690110273995938274165963485859266384153878316680344506263723136491785442170062032881780"
                "137555611788505514501532316432208913621450106734635739248722022468466491452278909525307"
                "347274576564635603129247264387268446197475571214605686701384259802843331040535960711766"
                "649600725158364537621541399494677456126392408081676961513278477607481039667146003375954"
                "830314226191928748810851166529278652022047218282832316623233367232804420628842999532153"
                "091182677889608570762793984250718984687087554756741639108167401972853728536590998143625"
                "211845954832750796264850094808624009442919470589653185883149368106321908548603251124783"
                "590566570371079014475071321746822627477961183585221011259411507847295013578693527476143"
                "912602324410577919951255713710422014394026718202043461757014466967388364876969215424380"
                "151351862173755001759532687834532105444861676527569665458491305419792930328730220968815"
                "296705510913554030500052757408196923315104000341749796173334047463765202724837923004153"
                "440717768387401293690551998595595092837522649746085746636175962698116010886587595547178"
                "360233308454623627208345276596284967317634833531926047831150524271041986626971440808088"
                "794926501680639131131259400767024040635636367169541854325951224019070548891135245989884"
                "055425513942028068620826455142072432480967342288339725504210575138092167571219438584131"
                "538482268824550329787977612248650032399175054349270056791885697414556789543183755949924"
                "343402419085698861928451597373738231915892215661750433528398250286798552040270285547754"
                "025551952474426383156576227671066725480108470445904378984450501414982323324935133731462"
                "449258865443771395023347213051336582041424487350574230416106972003802025815871230947070"
                "366477365979422635112157530625598997953597908769467862648814783916314099369113707703283"
                "237866726624292312447097002107999902055460055760743159585395197711265461375009752243330"
                "714456306292598657973765588432395640477228689865885971579792512212883746539208153550113"
                "539994002494214471966120413210248427452598833238026565356740336297890840883455419561076"
                "755186976154754941772708069577764255913766482930641747177396299998207520840487660204322"
                "435659681721472633735025486126812796195719739298182027936494343902768750477536159272267"
                "781743045693317634382292280924052540450326217066887410838629622467061187279284602320440"
                "475282959964584182810830271185254335331820553040245462638515244557062355016530171106492"
                "302117862048789175742575357410494603196665551068344163266082055772947321730032876022970"
                "207655957276117083610064542131569813996430172403566261417487168771938444213131168265510"
                "065130389982693306988399942432235390016511711209464444331788840259561755931226934077077"
                "100404176474629779706455895627856965249421782473143591037834255694170898813885818529903"
                "595016723524695834190247322128435162784033323207873990660100242281144859293707784435677"
                "073157356422927091981511377436082030811533288195953778013673029272470306849752558884491"
                "577694246597663568019018306664221689975744852080512223111253177116803914302702689490853"
                "082280286093634185086771366825734668063901010937818712101701484524236271240832094032551"
                "687548610948658636301237194965201347579356689070925031488374168552531807104696325292871"
                "579795209653940085962309478342764482576412510729339939577202798326663272861791037856753"
                "349837392517912385447485617970646398119924619528815354139495458671167005076184240033535"
                "440189382898967451029441559525863866960910189778900310424243637399626578616035717728086"
                "260868119156852151195485336067038311320769949938766043553784034991430260632984360959030"
                "350997347712477118193170038062782551022500682934530716587554989089379342484003945899565"
                "446119119870410287668885492983089478575484846289998862377475081159831982448666670961722"
                "199244591903714869792526857023216742974300339307010506121215439920774768578208171544526"
                "972150231346710620810070897091869250571290107750472597236770633420453197085501608456146"
                "965156887254509178926602508324178731697685185553844408249306132311368802731624425632525"
                "576623289420111811132346704804176520501013257070089578639917567420922538243969230187353"
                "560969997333933072495199842954686861323931517129438355252341190496243693386963787347961"
                "570156532871695638833187225150261790703365625122889067278553861747164255971540156295801"
                "363122446388931422347152810362784178592788265791135026228371318302330658756992285555049"
                "356347750171600706771265571174154675988814417521249166774125118603979016367964989428079"
                "571485016721938961722475181180670849356061383838659301994168398056123316663035782816581"
                "817929698703448274016771381548907473546545272344508007571176120469411700412063700985905"
                "202580164388058273462415019176789823708787856032233149085856240975007172425512415806389"
                "107118219441283342839681873448714606849121631399989489183539468492561918015603332064995"
                "098283088787133329983223630743288715464674818300410122143982493957474313315018854170299"
                "019937136197254493545246233938163477886240506737049874500398069985781049467859588597586"
                "857469601378089893613870602455525327609238589182284986721918903574874822102742652487429"
                "549819892992898914576356654706657900600469238097161284633033251143876331164492140567765"
                "800048162959220177362438158276616662354317889865705299751681163324689213212722200026906"
                "039173907271317980925159844222461467741354496053499023807467134749796215502565945275736"
                "104028606937493371334588310501290731609221580487837460765513836467909693488311789683724"
                "249654777453743163154546376478417428179614167309632920281726818012490235990486846554698"
                "476092664371027736311614324344702326706564632742301288369246780885547602993369194221176"
                "478083136937530315302948069693558507789243054401754549320325859411461445728235553217437"
                "289891136271560136535871329229855151318856196483191020252460145662944630584305881657040"
                "523143725116965869482362977529893481353911136853660204936842600808719257199900806589593"
                "542294049222250191850164272500183653691869906025128680810773182044149842802463435938900"
                "801481669595345786716746839287466797048570165600082422515209083381602092662224488599267"
                "251851208026255231278700567692114594621442548728955272296534325132682095766193071726709"
                "72296513149371536"};
            minY = HighPrecision{
                "1."
                "103801294602628786717534053732110675471417937790982960869811189231182811633959334064157"
                "761932063063127235954524910460202446466036810862297225718668698111310847244186635991363"
                "141917337570853402390895681866278184524313584991099338975501381491896291256321742256038"
                "598943308560117869742575658890793125545546125416110363554212836960905264594978878978275"
                "394301670834783500896806180651047166410478326596500844111282856681531040466490328311940"
                "277267787993163385904307364013945151036346313759637881311709653573656777372963470680829"
                "152637558698377668239286308353049816715180664224556028621533997707649976478582953827314"
                "781541750073861028597556321432378082725612655917716912884044991624970599713901556142817"
                "090179224089970659400299088416529327612873953041879214530282248370431514195999017446931"
                "592926733871970496318583426235506348061828505944508906697899635447381769153747927267818"
                "096317943733157766292308983344006573710449642073490102006233532563389079143770504533452"
                "824103080535247796577837149404624310098674051268467351126335363406088449811319800572574"
                "028120306228717383904554613897577349346411307152606145289637012646175952765409778205290"
                "853027153600503625788548132881485420008410551409504348363634906722755515540707886529775"
                "989765843429272940214664375237743723268723178580439812278069045213097147266647826773883"
                "328836735765241836965792142150938546400448325066807540998105795702815538038377484401494"
                "612415770986933589509145894335518966134826175660681916888190464321363376049127004242159"
                "680947991193224771816543756826442469325317807234558458702811232537736141916799840718722"
                "247753510430863626392564393785800506804416676661015922867145742621796652645169037609893"
                "564086268933966072975497716040181273569635992625543364783619033103022590329463531089770"
                "553725696219913718677105330566317170750959849052392472130401271334122790251913291213865"
                "678418762927952028346468237209111607425128348321567711887925331870433501601563860102618"
                "116998479234224893465141805358078510834542464730423067979312454459015044999990694608454"
                "372413876640224781119112931308367215980776946902339558754650650946778767244226522903949"
                "843997920108579453365126618463106921829495430289223511389025824570153268281020378788845"
                "303802201330521411867506088593941246434696262472963197646939633447427331983854646597374"
                "988611689115413349866708811739397381746281822693671112711164851607659092270097674454137"
                "508703231802753639894587414838170715952770456835368085838876566648315768857559503791954"
                "562318629453189013734252761778536338185918338155512025098654886897566719939587261335753"
                "319173047297465211022055119730429895174988770588035746917464104038110913033908075950342"
                "182834945333254960178384127545059707868378314380754958557853572150117293761953259810458"
                "660631556633967946669039063781732164974092069101584763315183278088467053678784752374348"
                "178196336179135365540527703369973096990047406589981887378851805547040534718468646705360"
                "883417085016562855448569403253962471796855264373339070508156216564803107685270217899788"
                "662994925437159825307003171287668940166416634358922635021035995170118698693234630746074"
                "611640589401391779162883154555362259803966837122119657305156238286963453448350755214536"
                "156246774855407602222464531908764408287213732150912964564893848409910096398924707504427"
                "595383209990144764706902939965929648746968788011023888439030356642042822303399806232143"
                "843388726618751640978196940117625093689853361528429692459066185297983679222995045087731"
                "023929324341984530671209223055558125528665152459671853303833155434939057661780796652710"
                "156436036655234041261982861739117104984899479558942480239373722400431431626942095096856"
                "555867073227070283210170407057571064795699902067322420976578706127027705591981376971415"
                "001169613113277140959558467703376627747028899541838453516683400146820025232862940584104"
                "946761194019006027736133056181181567065727513517641257599498872299308779076482443641105"
                "033356872941709126848948723992879141640956299081599558661281114430308374278080467630133"
                "731579384437690572023582056599515949939137356028031023240466854117528476824587148171086"
                "415613937677587035628785247926397023735023102893334865930714879931832210342573047349150"
                "230999394222274196191331165547619844262455820432618906525549171964858684904166149981512"
                "562702192062466216903482614293439877285139051498277691998268394713181288615069079921023"
                "424396557281861909128567040216451992815364384703798138013254193418424182041513266921870"
                "456940263992073973404697046312590840150157796038771787457907495935457236542929184335310"
                "858270705514530653108053052364743591681888875159548318437318436065818924015133083513508"
                "520098338581602826172933359085506287009005643371857709402935934422325526801928706675001"
                "370858630555242390623068217213674257084053613418011703912274484254427074866005885131206"
                "363053184583072217638857248281581393225291510851574380107969130298577447593809040105579"
                "676262068179540730816030314314051721624692401353905824498251399565790848187138935513957"
                "118091631380026727687149408514338190323977543129904892572606352707650372926356124791390"
                "673287757286614808099521603901902787337300631179679143159188663966878989820931466604613"
                "183450250068544177893378120046243696173365574707889530996375483875223946820927002631758"
                "311280280223991350794631032655210242205382339176258657891144866312781944472576496276425"
                "326256588325968198990735315274158007594338451711729644838471686630945057601096620209339"
                "519488261926938481664457318280903861825460008833327418250874097428513788822427517852010"
                "243336278167023404698335514300133668548390266534241311683638045023264863107420213756873"
                "948836649032773383299303190919611914746184439057953121315305154147691846059370587945060"
                "139287742495805282432919485697847946928132976468203311305235562935129038390743556862122"
                "445071551287499575504180656657017149005132232260588006199053754876930909692108726967385"
                "348567897638526364568587814378201814100307779778104578947434039591209169701264245819887"
                "076941231574679864507882738039882876326249737449640291979995568111846954232404479668025"
                "619495471153947199148769956079097986419471562617045231980783933582145534279580654213925"
                "430673365231311133202770959604059270515468755922638497571874511574471568923958633845523"
                "233720791701530002697255182972047199587118223086422224067626778849992340020226467303032"
                "935526410684855619651594541801280266893703312001260166184051688442793420741567413139331"
                "421993138813600807136803168589421122324336699575128248862693094346442981645031816761747"
                "646873930239028356026086762052203241203595172243922358422447571025451614415490353348979"
                "827919992459597505196154337803866958902721780353986660095148830725128101560847121904015"
                "178337333137469019839437159154264474823996673355314862895749128922220472265442653755787"
                "4607418286270335"};
            maxX = HighPrecision{
                "-0."
                "197308488401090040063694958492443779753220015673349789608662169368426273598640826637762"
                "768940957206998036608498263771502366782668691628957472776820387891535008484030200233898"
                "293353091827766940147836496046552623112955963667383588766616384201799018194696456029544"
                "018530169538910655163874159631160982932064231164379520817128850561282200849778245926009"
                "412267815048315006177889512536005410522182059963032510199021187747319420972027942068509"
                "610108232556382764180200667644707956362543922355259068315278704035870375461216535130276"
                "933432990851616241139414552036315452938086346174282061513753113864166146542947818968477"
                "690110273995938274165963485859266384153878316680344506263723136491785442170062032881780"
                "137555611788505514501532316432208913621450106734635739248722022468466491452278909525307"
                "347274576564635603129247264387268446197475571214605686701384259802843331040535960711766"
                "649600725158364537621541399494677456126392408081676961513278477607481039667146003375954"
                "830314226191928748810851166529278652022047218282832316623233367232804420628842999532153"
                "091182677889608570762793984250718984687087554756741639108167401972853728536590998143625"
                "211845954832750796264850094808624009442919470589653185883149368106321908548603251124783"
                "590566570371079014475071321746822627477961183585221011259411507847295013578693527476143"
                "912602324410577919951255713710422014394026718202043461757014466967388364876969215424380"
                "151351862173755001759532687834532105444861676527569665458491305419792930328730220968815"
                "296705510913554030500052757408196923315104000341749796173334047463765202724837923004153"
                "440717768387401293690551998595595092837522649746085746636175962698116010886587595547178"
                "360233308454623627208345276596284967317634833531926047831150524271041986626971440808088"
                "794926501680639131131259400767024040635636367169541854325951224019070548891135245989884"
                "055425513942028068620826455142072432480967342288339725504210575138092167571219438584131"
                "538482268824550329787977612248650032399175054349270056791885697414556789543183755949924"
                "343402419085698861928451597373738231915892215661750433528398250286798552040270285547754"
                "025551952474426383156576227671066725480108470445904378984450501414982323324935133731462"
                "449258865443771395023347213051336582041424487350574230416106972003802025815871230947070"
                "366477365979422635112157530625598997953597908769467862648814783916314099369113707703283"
                "237866726624292312447097002107999902055460055760743159585395197711265461375009752243330"
                "714456306292598657973765588432395640477228689865885971579792512212883746539208153550113"
                "539994002494214471966120413210248427452598833238026565356740336297890840883455419561076"
                "755186976154754941772708069577764255913766482930641747177396299998207520840487660204322"
                "435659681721472633735025486126812796195719739298182027936494343902768750477536159272267"
                "781743045693317634382292280924052540450326217066887410838629622467061187279284602320440"
                "475282959964584182810830271185254335331820553040245462638515244557062355016530171106492"
                "302117862048789175742575357410494603196665551068344163266082055772947321730032876022970"
                "207655957276117083610064542131569813996430172403566261417487168771938444213131168265510"
                "065130389982693306988399942432235390016511711209464444331788840259561755931226934077077"
                "100404176474629779706455895627856965249421782473143591037834255694170898813885818529903"
                "595016723524695834190247322128435162784033323207873990660100242281144859293707784435677"
                "073157356422927091981511377436082030811533288195953778013673029272470306849752558884491"
                "577694246597663568019018306664221689975744852080512223111253177116803914302702689490853"
                "082280286093634185086771366825734668063901010937818712101701484524236271240832094032551"
                "687548610948658636301237194965201347579356689070925031488374168552531807104696325292871"
                "579795209653940085962309478342764482576412510729339939577202798326663272861791037856753"
                "349837392517912385447485617970646398119924619528815354139495458671167005076184240033535"
                "440189382898967451029441559525863866960910189778900310424243637399626578616035717728086"
                "260868119156852151195485336067038311320769949938766043553784034991430260632984360959030"
                "350997347712477118193170038062782551022500682934530716587554989089379342484003945899565"
                "446119119870410287668885492983089478575484846289998862377475081159831982448666670961722"
                "199244591903714869792526857023216742974300339307010506121215439920774768578208171544526"
                "972150231346710620810070897091869250571290107750472597236770633420453197085501608456146"
                "965156887254509178926602508324178731697685185553844408249306132311368802731624425632525"
                "576623289420111811132346704804176520501013257070089578639917567420922538243969230187353"
                "560969997333933072495199842954686861323931517129438355252341190496243693386963787347961"
                "570156532871695638833187225150261790703365625122889067278553861747164255971540156295801"
                "363122446388931422347152810362784178592788265791135026228371318302330658756992285555049"
                "356347750171600706771265571174154675988814417521249166774125118603979016367964989428079"
                "571485016721938961722475181180670849356061383838659301994168398056123316663035782816581"
                "817929698703448274016771381548907473546545272344508007571176120469411700412063700985905"
                "202580164388058273462415019176789823708787856032233149085856240975007172425512415806389"
                "107118219441283342839681873448714606849121631399989489183539468492561918015603332064995"
                "098283088787133329983223630743288715464674818300410122143982493957474313315018854170299"
                "019937136197254493545246233938163477886240506737049874500398069985781049467859588597586"
                "857469601378089893613870602455525327609238589182284986721918903574874822102742652487429"
                "549819892992898914576356654706657900600469238097161284633033251143876331164492140567765"
                "800048162959220177362438158276616662354317889865705299751681163324689213212722200026906"
                "039173907271317980925159844222461467741354496053499023807467134749796215502565945275736"
                "104028606937493371334588310501290731609221580487837460765513836467909693488311789683724"
                "249654777453743163154546376478417428179614167309632920281726818012490235990486846554698"
                "476092664371027736311614324344702326706564632742301288369246780885547602993369194221176"
                "478083136937530315302948069693558507789243054401754549320325859411461445728235553217437"
                "289891136271560136535871329229855151318856196483191020252460145662944630584305881657040"
                "523143725116965869482362977529893481353911136853660204936842600808719257199900806589593"
                "542294049222250191850164272500183653691869906025128680810773182044149842802463435938900"
                "801481669595345786716746839287466797048570165600082422515209083381602092662222458583546"
                "027334317568999190369640995848143703959574459100524546445777938195204677065696030325787"
                "03859245649528511"};
            maxY = HighPrecision{
                "1."
                "103801294602628786717534053732110675471417937790982960869811189231182811633959334064157"
                "761932063063127235954524910460202446466036810862297225718668698111310847244186635991363"
                "141917337570853402390895681866278184524313584991099338975501381491896291256321742256038"
                "598943308560117869742575658890793125545546125416110363554212836960905264594978878978275"
                "394301670834783500896806180651047166410478326596500844111282856681531040466490328311940"
                "277267787993163385904307364013945151036346313759637881311709653573656777372963470680829"
                "152637558698377668239286308353049816715180664224556028621533997707649976478582953827314"
                "781541750073861028597556321432378082725612655917716912884044991624970599713901556142817"
                "090179224089970659400299088416529327612873953041879214530282248370431514195999017446931"
                "592926733871970496318583426235506348061828505944508906697899635447381769153747927267818"
                "096317943733157766292308983344006573710449642073490102006233532563389079143770504533452"
                "824103080535247796577837149404624310098674051268467351126335363406088449811319800572574"
                "028120306228717383904554613897577349346411307152606145289637012646175952765409778205290"
                "853027153600503625788548132881485420008410551409504348363634906722755515540707886529775"
                "989765843429272940214664375237743723268723178580439812278069045213097147266647826773883"
                "328836735765241836965792142150938546400448325066807540998105795702815538038377484401494"
                "612415770986933589509145894335518966134826175660681916888190464321363376049127004242159"
                "680947991193224771816543756826442469325317807234558458702811232537736141916799840718722"
                "247753510430863626392564393785800506804416676661015922867145742621796652645169037609893"
                "564086268933966072975497716040181273569635992625543364783619033103022590329463531089770"
                "553725696219913718677105330566317170750959849052392472130401271334122790251913291213865"
                "678418762927952028346468237209111607425128348321567711887925331870433501601563860102618"
                "116998479234224893465141805358078510834542464730423067979312454459015044999990694608454"
                "372413876640224781119112931308367215980776946902339558754650650946778767244226522903949"
                "843997920108579453365126618463106921829495430289223511389025824570153268281020378788845"
                "303802201330521411867506088593941246434696262472963197646939633447427331983854646597374"
                "988611689115413349866708811739397381746281822693671112711164851607659092270097674454137"
                "508703231802753639894587414838170715952770456835368085838876566648315768857559503791954"
                "562318629453189013734252761778536338185918338155512025098654886897566719939587261335753"
                "319173047297465211022055119730429895174988770588035746917464104038110913033908075950342"
                "182834945333254960178384127545059707868378314380754958557853572150117293761953259810458"
                "660631556633967946669039063781732164974092069101584763315183278088467053678784752374348"
                "178196336179135365540527703369973096990047406589981887378851805547040534718468646705360"
                "883417085016562855448569403253962471796855264373339070508156216564803107685270217899788"
                "662994925437159825307003171287668940166416634358922635021035995170118698693234630746074"
                "611640589401391779162883154555362259803966837122119657305156238286963453448350755214536"
                "156246774855407602222464531908764408287213732150912964564893848409910096398924707504427"
                "595383209990144764706902939965929648746968788011023888439030356642042822303399806232143"
                "843388726618751640978196940117625093689853361528429692459066185297983679222995045087731"
                "023929324341984530671209223055558125528665152459671853303833155434939057661780796652710"
                "156436036655234041261982861739117104984899479558942480239373722400431431626942095096856"
                "555867073227070283210170407057571064795699902067322420976578706127027705591981376971415"
                "001169613113277140959558467703376627747028899541838453516683400146820025232862940584104"
                "946761194019006027736133056181181567065727513517641257599498872299308779076482443641105"
                "033356872941709126848948723992879141640956299081599558661281114430308374278080467630133"
                "731579384437690572023582056599515949939137356028031023240466854117528476824587148171086"
                "415613937677587035628785247926397023735023102893334865930714879931832210342573047349150"
                "230999394222274196191331165547619844262455820432618906525549171964858684904166149981512"
                "562702192062466216903482614293439877285139051498277691998268394713181288615069079921023"
                "424396557281861909128567040216451992815364384703798138013254193418424182041513266921870"
                "456940263992073973404697046312590840150157796038771787457907495935457236542929184335310"
                "858270705514530653108053052364743591681888875159548318437318436065818924015133083513508"
                "520098338581602826172933359085506287009005643371857709402935934422325526801928706675001"
                "370858630555242390623068217213674257084053613418011703912274484254427074866005885131206"
                "363053184583072217638857248281581393225291510851574380107969130298577447593809040105579"
                "676262068179540730816030314314051721624692401353905824498251399565790848187138935513957"
                "118091631380026727687149408514338190323977543129904892572606352707650372926356124791390"
                "673287757286614808099521603901902787337300631179679143159188663966878989820931466604613"
                "183450250068544177893378120046243696173365574707889530996375483875223946820927002631758"
                "311280280223991350794631032655210242205382339176258657891144866312781944472576496276425"
                "326256588325968198990735315274158007594338451711729644838471686630945057601096620209339"
                "519488261926938481664457318280903861825460008833327418250874097428513788822427517852010"
                "243336278167023404698335514300133668548390266534241311683638045023264863107420213756873"
                "948836649032773383299303190919611914746184439057953121315305154147691846059370587945060"
                "139287742495805282432919485697847946928132976468203311305235562935129038390743556862122"
                "445071551287499575504180656657017149005132232260588006199053754876930909692108726967385"
                "348567897638526364568587814378201814100307779778104578947434039591209169701264245819887"
                "076941231574679864507882738039882876326249737449640291979995568111846954232404479668025"
                "619495471153947199148769956079097986419471562617045231980783933582145534279580654213925"
                "430673365231311133202770959604059270515468755922638497571874511574471568923958633845523"
                "233720791701530002697255182972047199587118223086422224067626778849992340020226467303032"
                "935526410684855619651594541801280266893703312001260166184051688442793420741567413139331"
                "421993138813600807136803168589421122324336699575128248862693094346442981645031816761747"
                "646873930239028356026086762052203241203595172243922358422447571025451614415490353348979"
                "827919992459597505196154337803866958902721780353986660095148830725128101560847967743899"
                "021886037494659036884878647422585679266441710700494332000230956812836063390649754339505"
                "2458971098763792"};
            SetNumIterations<IterTypeFull>(2147483646);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;

        case 15:
            minX =
                HighPrecision{"-1."
                              "2552386060808794544705762073214718782313298977374713834683672679219039239"
                              "1485262834542221765333926281210478722080772697161977970933710254724666689"
                              "5225513496666837065817527905487174554942327780991466572480483710152471757"
                              "1495723874748859332474368687432650230295564976108516327577711877380986"};
            minY =
                HighPrecision{"0."
                              "3821386782877920262924805588098523845507782625154048691830279750197815266"
                              "6818394826645189130797356608786903473371134867609900912124490114784833893"
                              "7814269399000905149775558707603837258390604413630021227035215627036116230"
                              "45547679106961127304013869141181720728437168142187095351869270963228793"};
            maxX =
                HighPrecision{"-1."
                              "2552386060808794544705762073214718782313298977374713834683672679219039239"
                              "1485262834542221765333926281210478722080772697161977970933710254724666689"
                              "5225513496666837065817527905487174554942327780991466572480483710152471757"
                              "1495723874748859332474368687432650230295523192890277752990779511763992"};
            maxY =
                HighPrecision{"0."
                              "3821386782877920262924805588098523845507782625154048691830279750197815266"
                              "6818394826645189130797356608786903473371134867609900912124490114784833893"
                              "7814269399000905149775558707603837258390604413630021227035215627036116230"
                              "45547679106961127304013869141181720728437342238929756079314822486632934"};
            SetNumIterations<IterTypeFull>(2147483646);
            break;

        case 16: {
            PointZoomBBConverter convert{
                HighPrecision{"-2."
                              "2817920769910571887912207570709061548015633738483510364900064462415074681"
                              "4756498835637162882690442602312850513201855450167543348272774154969477122"
                              "4847955228931798289163815438743631000392333635842525042153582202499655604"
                              "4494750661645865964591521051060641116244418227007881326557770689446951738"
                              "76856445256369203065186631701760525784081736244e-1"},
                HighPrecision{"1."
                              "1151567671155511043715094934051461395156851581586311660939652760704751491"
                              "6642413041968660280594843092549185744996529125136267112410593961023113773"
                              "2602258574689237558180672414305416362841285284232472179206525520835146101"
                              "8314092467920454966337318708594490924170564252580135863982080855021645197"
                              "51503113713282571658615520780020827777231498937"},
                HighPrecision{"1.4e301"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            SetNumIterations<IterTypeFull>(10100100);
            break;
        }

        case 17:
            minX = HighPrecision{"-0."
                                 "5398129258974168958622207952086918372374823102074872075260648021535803"
                                 "64920074745945503214288517608834423777345155755"};
            minY = HighPrecision{"0."
                                 "6611667021083996098993730317136609875652597471500246080021062112530517"
                                 "0240685980562572504049349703520845019259075699"};
            maxX = HighPrecision{"-0."
                                 "5398129258974168958622207952086918372374823102074872075260648021535803"
                                 "64920074745719107191125127561583883236084488469"};
            maxY = HighPrecision{"0."
                                 "6611667021083996098993730317136609875652597471500246080021062112530517"
                                 "02406859805720056716811576221562842084782701692"};
            SetNumIterations<IterTypeFull>(113246208);
            break;

        case 18: {
            // This one does not match #5 but is close - it'll differ by the screen aspect ratio.
            PointZoomBBConverter convert{
                HighPrecision{"-0."
                              "5482057480704757084582125675467330293766992746228824538244448345949959996"
                              "8089529129972505947379718"},
                HighPrecision{"-0."
                              "5775708389036038428051089822018505586755517284582553171583789528957369098"
                              "3215542361901805676878083"},
                HighPrecision{"4.98201309068883908096e+44"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            SetNumIterations<IterTypeFull>(4718592);
            break;
        }

        case 19: {
            minX = HighPrecision{"-0."
                                 "4806555079637494561219335091058799288119697276030121885990607686183285"
                                 "3774308262389069947355595792530735710085477971769115391712081609017535"
                                 "787686457114048952394910414276974524800303186844171743"};
            minY = HighPrecision{"0."
                                 "6374755901249708052094119195056120809799790056799661436909732420424829"
                                 "9914068987987862806005886435190126924708084397626352001791096751986215"
                                 "114695299109848932200215133076315301069004112043644706"};
            maxX = HighPrecision{"-0."
                                 "4806555079637494561219335091058799288119697276030121885990607686183285"
                                 "3774308262389069947355595792530735710085477971769115391712081609017535"
                                 "787686457114048948840611703973814164062372983621896847"};
            maxY = HighPrecision{"0."
                                 "6374755901249708052094119195056120809799790056799661436909732420424829"
                                 "9914068987987862806005886435190126924708084397626352001791096751986215"
                                 "114695299109848935754513843379475661806934315265919603"};
            SetNumIterations<IterTypeFull>(113246208);
            break;
        }

        case 20: {
            minX = HighPrecision{
                "-0.7086295598931993734622493611716274451479971071749675462585204452812848610912827"};
            minY = HighPrecision{
                "0.2588437605781938517662916915477487152993485711486090408513599516860639261853173"};
            maxX = HighPrecision{
                "-0.7086295598931993734622493611716274451479971071749675457247023721726823519349396"};
            maxY = HighPrecision{
                "0.2588437605781938517662916915477487152993485711486090413831737016165365135300597"};
            // ptX = HighPrecision{
            // "-0.7086295598931993734622493611716274451479971071749675459916114087269836065131112" };
            // ptY = HighPrecision{
            // "0.2588437605781938517662916915477487152993485711486090411172668266513002198576885" };
            // zoomFactor = HighPrecision{ "1.12397843052783475581e+55" };
            SetIterType(IterTypeEnum::Bits64);
            SetNumIterations<IterTypeFull>(50'000'000'000llu);
            break;
        }

        case 21:
            minX = HighPrecision{
                "-0."
                "165353054425834327068347550055335092120094241000026380415912481478846048996851264628574"
                "322423560266690587753860396550875628196105603940031447124507284796134645582380068094647"
                "607090297791540990497988541731977355180190520753027467465559816468674235485307967922195"
                "244020435598058452547945052640411625622380447543438758451505365411536195304664553213862"
                "187796767606743602535567713694658419041684193170231020290015597294881195401590590515687"
                "821394886994681228468823121619355762746143272331336588126133019869297364991848070238405"
                "297898977565831863782091635917688802769483971793868164049793382772144824691474021637422"
                "472581775598186763066262139585911209575570234948287138025858113770595417057765737841033"
                "586118537409729682731556704538168437889315511648671541179084394324381949630007598456990"
                "590122257489304269635986994729167818299362941847397307880932756823683547959153931416943"
                "468484997414042002226760352771502213545178733070190221064342199610334840881387030896218"
                "005684794531506673212756591521716283419037782654703941428719602480473210753190164633542"
                "506076510382975888406693578584596235047040945129391014749050860448846296782715940820928"
                "910430871080443380315527241808928932896928716039119443897790852490935543890817992936338"
                "937709268027467604182378253232485293312376943190272519179472927080530690648707898886007"
                "389680583190917306726096770998920761881206074115907690857056150891228681662452179492284"
                "790954328662430430656981970897085077471610297695950783746253592253527326193624697532694"
                "941942188724771208983480291886159387242725682612097829455543312890683809621756625033935"
                "103275511262132919872958003878843292448425265992348158407306443320832946249494297083297"
                "065386410865236181532967975846511540395885398967800451405872354471111541143118834281394"
                "086642513628585599191663767876630934261330738879007819927233496564762558362758418390567"
                "667209759340681400136029227832363343261284047156793237563769392991347582335842415147800"
                "214034819105679766895896190888487104322278865433401758695548688802612394860403944960107"
                "124678192573678989832899795415529229163840768056022104966650132921903672249536059325757"
                "089871496643859049887770030871044773077334163207114053337847994782596615178782492108495"
                "599767514996566744826625746330516783048178696834588961051566698640739429470125374731896"
                "593223653194204370639799254530974220119945225314385941207239585116703173521826441463626"
                "562708474709601500726612149467260167281041829123970618934980097324455704814315034834953"
                "682328758056771526998951795691697930367447004451518375515418169353884080315386805071944"
                "214945628326272206986611580426928212750995991822823860026677580567579124012249971331946"
                "408816729835427454567800436845776613039824177570729611229205776326703592784705416074092"
                "879853581588411484950856055747980311929635987723234642038360349774236752092955179056821"
                "468348024666783398270724421129003133127090314443705720069574303174193228724845820259116"
                "590995276466078757283233798452159659421707779991435140369151226234965251861878363669272"
                "417784157764671849926431585329404200513542914120936540490370523737679542261902716105031"
                "899047000574672763617580363870626629525533768506115991045406909783797753803746089087264"
                "977840263005711189494266590101966689231712756256470547792744777553927079757974120385717"
                "870790951384830911886978567420529971596544013707239416091341585384871990520957060109772"
                "736365748978193905106534801222821369395263033319870549174850155278336102979440391140479"
                "756393865532583065213267686335482313265711452613988394900061698931036844895917649122714"
                "468973535110221199701045387093364040088983148440901558565343569933842766230129144889523"
                "286868632078730811819589924157257618828836418991854941183745815617753130829885427605738"
                "332351815468800041614844411948734214373134474159935836830528838008902014084989040613766"
                "534015144169630407744947384870577540641433410805373824081271884349456445980467227805939"
                "858029823443636433375993565736010527634777140361646143747611462976125176422699484335744"
                "600731469227994871545083124569882828687719279622761878666910489636170143111644396194740"
                "107754249102526504602513511057314389106209059481710370014957772773493996517864927750259"
                "751417897780388304152123559071586995936393040390811267657212490363383794594817939326816"
                "915052108702617526830125590381626179273747346711161718632746010506896492152507657949850"
                "685205394267271069232829330314113981044847399353982571187682200220050685109212015810345"
                "847374980077166290560814788482099061174945177497777209829366064227056160852469309701939"
                "873197934997935146823283624577874026529594466861139705283234747077286219792074376315246"
                "333330706693906228230365874533010214032675713142686895233586242015050484327701262570458"
                "880321667614601233227675479227672224641408664879748669522025277685622575570050182195870"
                "035227678915279535025936583502036576519349295338276798472988538693559511220010233201916"
                "228704019126372083140407535079980191493057561086714211574430100432792620328609268319520"
                "304945854385497936141901443101199205052471093936689979860173883780344114252131947436942"
                "302349503752793374627732075522786101926735579515622542622317922002746879698386356592865"
                "907671648352406117662021351985545998783349330524914719609285041426256404473243191913367"
                "856856505245053658272307095369129220162266268420386638737793203594207202582540456171596"
                "060541143750558367075091060276815628477967236278834778223264263609185470196025869508929"
                "722398462967645617597794592799099406939069220012992699606111668791972386713160194753183"
                "837116190835484443893600248426067621563375877433923433053986834276941578172577169480739"
                "698040727927051070813667053486331251784829654417512411597664042111767367663367872406811"
                "253438297624215855991182989731954995212058872814504049337329693272918413538243524526839"
                "378807754264029690341466186361044512855032557883729700966098830281872407442880004870470"
                "584972051023774452512883045739712345718037055286717487106553056063485228487300596391087"
                "139454082167573918246552383569274381849009436037589909083139992944782141686680096524533"
                "568170424858925610407896477023227911013714730552946622590798621241352950011051227518196"
                "183154105030003364985768251976119377457107417619857186239273885250163431996516953720197"
                "171826621833921539290435578458700196733937764086363546782270852887856001397001975396643"
                "041276905170370449927708191342356000491510498182329289727839105416752909777142489501312"
                "569846474792156750775748670181730686556349529944551833099392024954663901742552812089011"
                "917592931200119202193015248773456711015371790869584374293209887842875062426035294797893"
                "890969787057912885144174509693572729486441063536053468860625486553264201741128373776071"
                "355340209336602300388938993187197178640398069203314482289353518901337019848286111326088"
                "235409923686674159952161562480844551090683390209994136598461029533764993305238873823339"
                "363942706577414784359492517592521760479266623019971610190663368517350825471750623101329"
                "733732676875134072821397170059942499383337984758966986663628920758061051763908186554368"
                "494548431347538961093279045258986114008848070935719505927929844387038392643409634869617"
                "879553144747707394902135413073731883743987911165789174348634634075053249592498046647045"
                "816047976293211941960381162024603750637280577686138131445369610346686061122266698851206"
                "484285402672323270264993345352262842622989814676592089051166636314896577991170492999858"
                "475643511699922806010673865196459988411983646144529395110965896266135582918090788910655"
                "653533977096537938917321840610415306482518110713209759360969812808423903607312327852649"
                "389369181278717908373707127250848414738760284626613909239977215884435918888044487983110"
                "245967872307235049034446988418503531952968982788520561075780654892652683629294901141771"
                "279510968052360539748738836432790792043526253652531307039188456936412054181860042128306"
                "868922924285956289336711532018612706267065365750882954648327718572091225202058150597266"
                "420194411112146793406903002760749685050404726389139753261108075164325117098447603641627"
                "172342335141252879668047403442496156686098362967003280803703751496079357587699689190716"
                "273868541142530992703910472555486651898428792369378140669670598661156878545238432732583"
                "504801709877711760652053644181148221932669718344567800083561816029230283328631350629687"
                "152285361481168846243317678250175658617315776244662741098180364247827206432903047652330"
                "073958450747982982320095249262502767794991533685844967431946590301552167994682578091262"
                "037108944986972736919718880408275845321966274622598243083637536104897294413921638954282"
                "376280884695105128267347686829582020636483391132026909471106841010974160244514587380606"
                "659287045028970009321283953825219937227139813381763670073075883942127389061316189749902"
                "383423167382658738580952213198804945769600776180384319100485879028253262657134710335293"
                "606535775389088853727315293100876079952109293344088103971951480494487754596407141467589"
                "307223384665151440511539065850505118855113045469618352702700421864081721157910734272327"
                "967917845850706068618905313969993629417189019440249915526790185446294203281664821715945"
                "912860879531821023789062130909466236932112134608209750365826535844639826854943861059761"
                "050817065936005572897716927656788711329860395413268756189922094161632961802975376296442"
                "212156031933282337702006954782696327044422873119138302780500207468796009428403959514219"
                "092778403693156424490285561767859067764070637929433238019135934620127159589148688663316"
                "333823324271609322641842758694003289203201044936650228178758315741220758011252572074991"
                "096068049106618012555191310446255165370526415173362113768333789599943675471326422117676"
                "334314266747968775024668083580670963192468694653336377644206322697597857619326333187182"
                "289423076171533936691489200188605165418650423116697995608160056812035697001549422601519"
                "842317117999137345435699155672279037430567745010659244889782939560113956308388433906461"
                "427370380377343080248029365213168450436915758404812661842554630512632919783992213543783"
                "257176855924460873683896564206216084026461806133687175646248722280968933802897270153863"
                "464293042013597529870325255088275718365644698293584089082828539443700361774585106480045"
                "756145535019626597646855793743408522811671335762791511468437595742168024409937201303441"
                "774585934220394538744700727446992991260442240520048636797756024810930146640135842185194"
                "897203606389769729463321849087906112691468658396157387297362275420411739066157736983641"
                "068114823469133313427620116880312881225892298889836488671172799822225946955020411149420"
                "933774586872339848592610488869714677843758358403321183181151962308158619307215281277740"
                "258543759337972503873005561599434391046500951009505597762560623708223965512068456513780"
                "654219899954375597512828103722798859948308156553994793288430241152986924485177472873015"
                "909873812123339226879255020117083749194681650812041777524412244774192327973441088010851"
                "283224914045003379940557946972586822089061969203048382548361630863652133124621331655531"
                "715660733353043243403778897569533996464116954872159905058467494495292935274845521175643"
                "055262623565410797581021224500703211015693978874893183400350124608416415017323117581688"
                "294856523114770863888377760625246237288617876540714450135755239993970314184532237097504"
                "605284789252685023259"};
            minY = HighPrecision{
                "1."
                "044277064897808805382972133514970033636669752029592433606621179011324460839154407760083"
                "401975587084514485665944078137412806407002865995435448320301379853171431018504164501791"
                "882937915959701460308680789833538940017615109204052114378534599805568913143433721403046"
                "439205869053847721440465614404636728874424936043699640930317900856017266059437572440389"
                "878746913170475829036230984669372185446989891582627851149298755344310847731020716343893"
                "905754382775262464527952786926687476749161966321307577475141911922933647690917552245194"
                "783623324663570183784753320876701599806414191859791194254383665847508813676303149564416"
                "002169543425982434473639175690502231884238779339636937927115677133856026720254304320912"
                "596326304195322539815227435108222145097458923796047746425910364932925829558006584166443"
                "664314352307264148382970891408988979062525968381056334335679045316392262774401410748913"
                "408401726603888508786068502561916923020865667192778753718590384998567343707540582590893"
                "641598755683111418781587221104872163730028193288726016114387032296975334783851084675237"
                "953433914071341923149575110764141419210252231173182740253477868436375249646101622314227"
                "416191146690240627169089555377962857953078414928044169108256927942349459910125948525888"
                "192514802979638130928884513689597676506227315633736859599410454347573595454383817394748"
                "935022334082255135551120400099019118998110307133409174242933088693341598449819928220259"
                "155678207221017769141187925530967420324226482642441837300165695640086994966587267298764"
                "816111987953532512141499935149352635859544158772359773227213299237214707647808099345149"
                "682795294672490521431109103068590372323430327921486018543311757823497567089023535287679"
                "052758409628828210249156117254088355377867901218420558414958281589870956330589688851321"
                "046690330125132544129530984221717837455562610519873328532560918889430601135107973345768"
                "817209608974207833250852667693795380646425391225392082513724949494476435790542977963418"
                "137359618212807646890057457386413399936496057488784167877323590824491613168801296594688"
                "801074154295032863107970487782416194310185519986529107358945248597178434959280827505375"
                "827413973770061227045403202358478813286345708064925273726005856759751937216053015686038"
                "333725737654461907455694752702861260355779262675620119674671289569048117869163731332885"
                "380961395242878038848030107587180419281007199403918358955680615012816392399655408092220"
                "719785110435589430058279851676911045619947317660628503348934794278085469249162338078185"
                "740166811898328201046771253939036701986468067018849544676093985646163517443669257597327"
                "605998680263234841056781142259599866627942176432187101723893470471368191462007769835291"
                "146141961003897316596588350625193110626780157374543624603924725978567445653023583594457"
                "342450439804247732783989513716615726986587278054179530389134873022538992769747001092644"
                "375570948891964111860493852920030762823906082513582995623847123977234957850465311060331"
                "249376671037656883228286707420864297276307091815601153092482618677009326779890727198773"
                "011360526132762184654019118927958254826608879455811016330056870395296558444575620915646"
                "310335239336746588690879426433929169514582809321864851565527270113459289837281507428943"
                "142483220577436139897580529033208015903282052194542725615926718405657412252008398951332"
                "832369317510331234855612895732416907835549029681481980174175748560255303898962522613183"
                "279466231226321065128314762519316857835691088178139182385060003331814183038200665820934"
                "733317514668065311540788543085235046904373432879721147296924262406434303230190469756858"
                "157502361592784352881670686400442237591092599939926289208865956714553475115072314960175"
                "023698570581415686650085571435036724201675256050465945743139385679245700279008060895893"
                "723193717497176961492846301344730092055262457187284623228467158575911349499113944752473"
                "670877111026290963422562736356746214395645393762593676501427537317746669560657944643073"
                "276253629018423022318290104936804325101867005532847827201774894991361980434936411653155"
                "762310125205831436785871715264909411705314195918518249868501298378047133779304309378677"
                "289045175036368183787608586139899142742148446798034554185499530469312263804426191814188"
                "792259918908005089337416158309492169729046978241108764800373199786835512775018459244859"
                "267531608074047213792485790714727394281686995009721892096313398856651364376835976485985"
                "577848331809780103472992243751739017814855135195390244578020544445155642189656411959818"
                "537113780293079843353672189952048766617739975099791375374874500190464500082083000996046"
                "562339788079629204480380125711374292052576290634217065477010746078997964370477540596950"
                "179494021104626979205963371667293248583566456157361250770591161294974738245422314899316"
                "567308929508634261348112806280532523476213178307252077997746083513294549227767091938067"
                "217624964628370906697291402801808940793284640660655062072285201193967242441893707135722"
                "290023979396671884835177432518035657429152687499442438672252127647437051589773194843058"
                "438225198896308860630489777828329667886104302882092289082288763901527187068327652071603"
                "624848895710581676120737165753641607798272808608702931524263579983568144332095173143917"
                "006791039477173888020650319663759808299494529128325593983395874886465461027967163402161"
                "841883781708743258815462635693304813049558585059423645696496789082810397724886054771581"
                "563831522836536737064904299544589884335904000319915121211509398654760775082923447627431"
                "810306326621272995546270200576623778725786936601643047562903876270563152346113203816665"
                "415647322772310805862830328907715627364519002365100137737699655671974565189869468321310"
                "071771026817931314136062271980786119418884667896744248988022776918612392621307244846715"
                "565141673730197159045126169762621237477118095627007495171039905668325659077157792159884"
                "943945222288639927266673422568383814713249358515769104841319551023369437288552466511955"
                "117808978763619586608932135694850483552263188003555050643236007295112136223460594779054"
                "726696786510462803932089824565856848396625577598749730172563451415247181522187861098101"
                "676294774399809634350068762970820625702542372795052750262991463099037894875056129499728"
                "445824201949824805059249789991826703880335086183597727635326230040287804809229904724327"
                "852548533423811116498779411377941487636025548910716603561208719468445816412340563590498"
                "400399244784036149284049457540915194474505097972683897906679365708314054347905676337878"
                "070783604662861248077878016527821790505946032823348841177178634386993182797943842539434"
                "419858352852503464295581430236588609770297176292756751968184573897538526914043047169814"
                "984561707551218821949790226202641140107370608506446237466149862964632615116143823940797"
                "596880439913349528709020788483323602475404095930729914237392360095230982170538229907966"
                "495086002892040173075678439373003701909762402992875028567037197838993284718386014290238"
                "070433830095663716147654734717059715386297217424718293321056467774762043970327984658152"
                "380500694244445759891469465190557700800470744175185680654189821923413620537583275222898"
                "761381663910864540456551646021987066654828544317400746914117387102118080351849066955339"
                "664552514032346001119032775166253582621422092613876844626462373903273430772755116511435"
                "087906670944894552654778284697192895688418054857236719421320513108694045127830937696976"
                "297674990198674156800094322748981626104687761720671181734337055644894970425380677001034"
                "000849743270912591141325563959537192085987620330490840476536083886276761925001294346732"
                "505633809631143917183216956647639752360631157837300580153010392919582696581794027662533"
                "624305037354263879163470222707636248134905305296685272910483908282847104831998228434858"
                "974130961151274297209438917064183987633962816357205519261740252040644213788393339141118"
                "739697979968256978909112973181868019620886889987187481890055895367983801459002799076818"
                "837986663883724217746910902316915065093257083145192704924689072159703769249070436309481"
                "736201756702232140567534778558365775966006770532140490701287831846815608863443209730182"
                "845519291896767911368267517992656584298826507477063195274462071851960638339558820957705"
                "945408982367241117986674718249918859668209281488699934059441044250362349944518950733087"
                "378952305246683734778048472358542456378916286920677595140976322183957754227863002796881"
                "445342473036060341816544972291059016840778902131080124362200441314757556055107299290272"
                "474386805268434702813296226582427924188685957363973334690057177075025448915737921847084"
                "496839222609113297210407902116666610614824213911106641486805467563380143891770790755077"
                "167646037291760287860918118093835326201781510890299315938053532911885053322064348985683"
                "872014986880624400851522484591793802210268438103911142640495772848835072529974685218226"
                "530680635000949565847378454967660431116817851497196098293776449567538417507298269860586"
                "641110286466501057094528795544149327856999756011735764809727666818188882692663129685627"
                "307308292841788932896782171967675480989562039413508190787222142371466915917300058033573"
                "956758035410403740286953590313023855419940156309291098762553907595055420194025188519773"
                "478532383738190125727961927627826957010912300607677153328682523185847768056223658161372"
                "063956594173958678984956434812265025341867438320127897206866145137431128120320675777927"
                "278079288185190540772516346474194464949754117537260776606321880409377893860252978938855"
                "936279098736016074980495680506751358874945024248382824839312022458903650881711401811099"
                "727132527170715917016452649648068281543613104412514657514133094728931930486732130330237"
                "860233300008851728771054892940018634866579694761191535837102038913521403159566337476438"
                "903506822942696860878465228375517633393711092355856879016328067674480021558426940911197"
                "537360973531609888843203147261886391526901762055829861631932676334918110816677328134287"
                "318403433644752025487960559636346260867461273710851076900517946468485272493974552249311"
                "774727074135667145199111533222466881966616063900028205814838832809205478383123779018284"
                "464095799291747600631432261662541444929709100872226373145461025029266814100794680558287"
                "960828016367377616377299551076861209811971060912998536363961949830635848960123950221836"
                "645040123367733155857607278961778722701329765831768695142739291420761507182337978946099"
                "417367599369925618514734841956072378859975096934798193328032286550363331350324986132500"
                "464738625624685084886366703070863472686243762955596814450727423669867329521462277177679"
                "629841212369453859103602491860237020095663604077905466151242813453152308103189869801194"
                "351814672108421447717521474366046024974865715446700166858200888354627705894167831647642"
                "637437058285732555360659777349669913067216996318888409694121138734328021188041286793705"
                "441757278118362122386622470123263158470474133857455191170203145959159030143474033525161"
                "733770964482552532659350912663742295237357567167312026515467171743013903994557270927628"
                "924485170298301840043308180786825680208009489568114015487209668749540967277041676669601"
                "080507867931068038810555954236536838616675351491571608872852352605168619256936034380349"
                "797163971307747341514911888068450063912908745787316316934334243862841506162896104851411"
                "629876199538145306775291656614453804323498925067782104388449505809724923401814110975161"
                "75795771302288600447"};
            maxX = HighPrecision{
                "-0."
                "165353054425834327068347550055335092120094241000026380415912481478846048996851264628574"
                "322423560266690587753860396550875628196105603940031447124507284796134645582380068094647"
                "607090297791540990497988541731977355180190520753027467465559816468674235485307967922195"
                "244020435598058452547945052640411625622380447543438758451505365411536195304664553213862"
                "187796767606743602535567713694658419041684193170231020290015597294881195401590590515687"
                "821394886994681228468823121619355762746143272331336588126133019869297364991848070238405"
                "297898977565831863782091635917688802769483971793868164049793382772144824691474021637422"
                "472581775598186763066262139585911209575570234948287138025858113770595417057765737841033"
                "586118537409729682731556704538168437889315511648671541179084394324381949630007598456990"
                "590122257489304269635986994729167818299362941847397307880932756823683547959153931416943"
                "468484997414042002226760352771502213545178733070190221064342199610334840881387030896218"
                "005684794531506673212756591521716283419037782654703941428719602480473210753190164633542"
                "506076510382975888406693578584596235047040945129391014749050860448846296782715940820928"
                "910430871080443380315527241808928932896928716039119443897790852490935543890817992936338"
                "937709268027467604182378253232485293312376943190272519179472927080530690648707898886007"
                "389680583190917306726096770998920761881206074115907690857056150891228681662452179492284"
                "790954328662430430656981970897085077471610297695950783746253592253527326193624697532694"
                "941942188724771208983480291886159387242725682612097829455543312890683809621756625033935"
                "103275511262132919872958003878843292448425265992348158407306443320832946249494297083297"
                "065386410865236181532967975846511540395885398967800451405872354471111541143118834281394"
                "086642513628585599191663767876630934261330738879007819927233496564762558362758418390567"
                "667209759340681400136029227832363343261284047156793237563769392991347582335842415147800"
                "214034819105679766895896190888487104322278865433401758695548688802612394860403944960107"
                "124678192573678989832899795415529229163840768056022104966650132921903672249536059325757"
                "089871496643859049887770030871044773077334163207114053337847994782596615178782492108495"
                "599767514996566744826625746330516783048178696834588961051566698640739429470125374731896"
                "593223653194204370639799254530974220119945225314385941207239585116703173521826441463626"
                "562708474709601500726612149467260167281041829123970618934980097324455704814315034834953"
                "682328758056771526998951795691697930367447004451518375515418169353884080315386805071944"
                "214945628326272206986611580426928212750995991822823860026677580567579124012249971331946"
                "408816729835427454567800436845776613039824177570729611229205776326703592784705416074092"
                "879853581588411484950856055747980311929635987723234642038360349774236752092955179056821"
                "468348024666783398270724421129003133127090314443705720069574303174193228724845820259116"
                "590995276466078757283233798452159659421707779991435140369151226234965251861878363669272"
                "417784157764671849926431585329404200513542914120936540490370523737679542261902716105031"
                "899047000574672763617580363870626629525533768506115991045406909783797753803746089087264"
                "977840263005711189494266590101966689231712756256470547792744777553927079757974120385717"
                "870790951384830911886978567420529971596544013707239416091341585384871990520957060109772"
                "736365748978193905106534801222821369395263033319870549174850155278336102979440391140479"
                "756393865532583065213267686335482313265711452613988394900061698931036844895917649122714"
                "468973535110221199701045387093364040088983148440901558565343569933842766230129144889523"
                "286868632078730811819589924157257618828836418991854941183745815617753130829885427605738"
                "332351815468800041614844411948734214373134474159935836830528838008902014084989040613766"
                "534015144169630407744947384870577540641433410805373824081271884349456445980467227805939"
                "858029823443636433375993565736010527634777140361646143747611462976125176422699484335744"
                "600731469227994871545083124569882828687719279622761878666910489636170143111644396194740"
                "107754249102526504602513511057314389106209059481710370014957772773493996517864927750259"
                "751417897780388304152123559071586995936393040390811267657212490363383794594817939326816"
                "915052108702617526830125590381626179273747346711161718632746010506896492152507657949850"
                "685205394267271069232829330314113981044847399353982571187682200220050685109212015810345"
                "847374980077166290560814788482099061174945177497777209829366064227056160852469309701939"
                "873197934997935146823283624577874026529594466861139705283234747077286219792074376315246"
                "333330706693906228230365874533010214032675713142686895233586242015050484327701262570458"
                "880321667614601233227675479227672224641408664879748669522025277685622575570050182195870"
                "035227678915279535025936583502036576519349295338276798472988538693559511220010233201916"
                "228704019126372083140407535079980191493057561086714211574430100432792620328609268319520"
                "304945854385497936141901443101199205052471093936689979860173883780344114252131947436942"
                "302349503752793374627732075522786101926735579515622542622317922002746879698386356592865"
                "907671648352406117662021351985545998783349330524914719609285041426256404473243191913367"
                "856856505245053658272307095369129220162266268420386638737793203594207202582540456171596"
                "060541143750558367075091060276815628477967236278834778223264263609185470196025869508929"
                "722398462967645617597794592799099406939069220012992699606111668791972386713160194753183"
                "837116190835484443893600248426067621563375877433923433053986834276941578172577169480739"
                "698040727927051070813667053486331251784829654417512411597664042111767367663367872406811"
                "253438297624215855991182989731954995212058872814504049337329693272918413538243524526839"
                "378807754264029690341466186361044512855032557883729700966098830281872407442880004870470"
                "584972051023774452512883045739712345718037055286717487106553056063485228487300596391087"
                "139454082167573918246552383569274381849009436037589909083139992944782141686680096524533"
                "568170424858925610407896477023227911013714730552946622590798621241352950011051227518196"
                "183154105030003364985768251976119377457107417619857186239273885250163431996516953720197"
                "171826621833921539290435578458700196733937764086363546782270852887856001397001975396643"
                "041276905170370449927708191342356000491510498182329289727839105416752909777142489501312"
                "569846474792156750775748670181730686556349529944551833099392024954663901742552812089011"
                "917592931200119202193015248773456711015371790869584374293209887842875062426035294797893"
                "890969787057912885144174509693572729486441063536053468860625486553264201741128373776071"
                "355340209336602300388938993187197178640398069203314482289353518901337019848286111326088"
                "235409923686674159952161562480844551090683390209994136598461029533764993305238873823339"
                "363942706577414784359492517592521760479266623019971610190663368517350825471750623101329"
                "733732676875134072821397170059942499383337984758966986663628920758061051763908186554368"
                "494548431347538961093279045258986114008848070935719505927929844387038392643409634869617"
                "879553144747707394902135413073731883743987911165789174348634634075053249592498046647045"
                "816047976293211941960381162024603750637280577686138131445369610346686061122266698851206"
                "484285402672323270264993345352262842622989814676592089051166636314896577991170492999858"
                "475643511699922806010673865196459988411983646144529395110965896266135582918090788910655"
                "653533977096537938917321840610415306482518110713209759360969812808423903607312327852649"
                "389369181278717908373707127250848414738760284626613909239977215884435918888044487983110"
                "245967872307235049034446988418503531952968982788520561075780654892652683629294901141771"
                "279510968052360539748738836432790792043526253652531307039188456936412054181860042128306"
                "868922924285956289336711532018612706267065365750882954648327718572091225202058150597266"
                "420194411112146793406903002760749685050404726389139753261108075164325117098447603641627"
                "172342335141252879668047403442496156686098362967003280803703751496079357587699689190716"
                "273868541142530992703910472555486651898428792369378140669670598661156878545238432732583"
                "504801709877711760652053644181148221932669718344567800083561816029230283328631350629687"
                "152285361481168846243317678250175658617315776244662741098180364247827206432903047652330"
                "073958450747982982320095249262502767794991533685844967431946590301552167994682578091262"
                "037108944986972736919718880408275845321966274622598243083637536104897294413921638954282"
                "376280884695105128267347686829582020636483391132026909471106841010974160244514587380606"
                "659287045028970009321283953825219937227139813381763670073075883942127389061316189749902"
                "383423167382658738580952213198804945769600776180384319100485879028253262657134710335293"
                "606535775389088853727315293100876079952109293344088103971951480494487754596407141467589"
                "307223384665151440511539065850505118855113045469618352702700421864081721157910734272327"
                "967917845850706068618905313969993629417189019440249915526790185446294203281664821715945"
                "912860879531821023789062130909466236932112134608209750365826535844639826854943861059761"
                "050817065936005572897716927656788711329860395413268756189922094161632961802975376296442"
                "212156031933282337702006954782696327044422873119138302780500207468796009428403959514219"
                "092778403693156424490285561767859067764070637929433238019135934620127159589148688663316"
                "333823324271609322641842758694003289203201044936650228178758315741220758011252572074991"
                "096068049106618012555191310446255165370526415173362113768333789599943675471326422117676"
                "334314266747968775024668083580670963192468694653336377644206322697597857619326333187182"
                "289423076171533936691489200188605165418650423116697995608160056812035697001549422601519"
                "842317117999137345435699155672279037430567745010659244889782939560113956308388433906461"
                "427370380377343080248029365213168450436915758404812661842554630512632919783992213543783"
                "257176855924460873683896564206216084026461806133687175646248722280968933802897270153863"
                "464293042013597529870325255088275718365644698293584089082828539443700361774585106480045"
                "756145535019626597646855793743408522811671335762791511468437595742168024409937201303441"
                "774585934220394538744700727446992991260442240520048636797756024810930146640135842185194"
                "897203606389769729463321849087906112691468658396157387297362275420411739066157736983641"
                "068114823469133313427620116880312881225892298889836488671172799822225946955020411149420"
                "933774586872339848592610488869714677843758358403321183181151962308158619307215281277740"
                "258543759337972503873005561599434391046500951009505597762560623708223965512068456513780"
                "654219899954375597512828103722798859948308156553994793288430241152986924485177472873015"
                "909873812123339226879255020117083749194681650812041777524412244774192327973441088010851"
                "283224914045003379940557946972586822089061969203048382548361630863652133124621331655531"
                "715660733353043243403778897569533996464116954872159905058467494495292935274845521175643"
                "055262623565410797581021224500703211015693978874893183400350124608416415017323117550185"
                "536204198832506366226190342179845404215472069530383611799921404450662049484630193282375"
                "132850487865342674796"};
            maxY = HighPrecision{
                "1."
                "044277064897808805382972133514970033636669752029592433606621179011324460839154407760083"
                "401975587084514485665944078137412806407002865995435448320301379853171431018504164501791"
                "882937915959701460308680789833538940017615109204052114378534599805568913143433721403046"
                "439205869053847721440465614404636728874424936043699640930317900856017266059437572440389"
                "878746913170475829036230984669372185446989891582627851149298755344310847731020716343893"
                "905754382775262464527952786926687476749161966321307577475141911922933647690917552245194"
                "783623324663570183784753320876701599806414191859791194254383665847508813676303149564416"
                "002169543425982434473639175690502231884238779339636937927115677133856026720254304320912"
                "596326304195322539815227435108222145097458923796047746425910364932925829558006584166443"
                "664314352307264148382970891408988979062525968381056334335679045316392262774401410748913"
                "408401726603888508786068502561916923020865667192778753718590384998567343707540582590893"
                "641598755683111418781587221104872163730028193288726016114387032296975334783851084675237"
                "953433914071341923149575110764141419210252231173182740253477868436375249646101622314227"
                "416191146690240627169089555377962857953078414928044169108256927942349459910125948525888"
                "192514802979638130928884513689597676506227315633736859599410454347573595454383817394748"
                "935022334082255135551120400099019118998110307133409174242933088693341598449819928220259"
                "155678207221017769141187925530967420324226482642441837300165695640086994966587267298764"
                "816111987953532512141499935149352635859544158772359773227213299237214707647808099345149"
                "682795294672490521431109103068590372323430327921486018543311757823497567089023535287679"
                "052758409628828210249156117254088355377867901218420558414958281589870956330589688851321"
                "046690330125132544129530984221717837455562610519873328532560918889430601135107973345768"
                "817209608974207833250852667693795380646425391225392082513724949494476435790542977963418"
                "137359618212807646890057457386413399936496057488784167877323590824491613168801296594688"
                "801074154295032863107970487782416194310185519986529107358945248597178434959280827505375"
                "827413973770061227045403202358478813286345708064925273726005856759751937216053015686038"
                "333725737654461907455694752702861260355779262675620119674671289569048117869163731332885"
                "380961395242878038848030107587180419281007199403918358955680615012816392399655408092220"
                "719785110435589430058279851676911045619947317660628503348934794278085469249162338078185"
                "740166811898328201046771253939036701986468067018849544676093985646163517443669257597327"
                "605998680263234841056781142259599866627942176432187101723893470471368191462007769835291"
                "146141961003897316596588350625193110626780157374543624603924725978567445653023583594457"
                "342450439804247732783989513716615726986587278054179530389134873022538992769747001092644"
                "375570948891964111860493852920030762823906082513582995623847123977234957850465311060331"
                "249376671037656883228286707420864297276307091815601153092482618677009326779890727198773"
                "011360526132762184654019118927958254826608879455811016330056870395296558444575620915646"
                "310335239336746588690879426433929169514582809321864851565527270113459289837281507428943"
                "142483220577436139897580529033208015903282052194542725615926718405657412252008398951332"
                "832369317510331234855612895732416907835549029681481980174175748560255303898962522613183"
                "279466231226321065128314762519316857835691088178139182385060003331814183038200665820934"
                "733317514668065311540788543085235046904373432879721147296924262406434303230190469756858"
                "157502361592784352881670686400442237591092599939926289208865956714553475115072314960175"
                "023698570581415686650085571435036724201675256050465945743139385679245700279008060895893"
                "723193717497176961492846301344730092055262457187284623228467158575911349499113944752473"
                "670877111026290963422562736356746214395645393762593676501427537317746669560657944643073"
                "276253629018423022318290104936804325101867005532847827201774894991361980434936411653155"
                "762310125205831436785871715264909411705314195918518249868501298378047133779304309378677"
                "289045175036368183787608586139899142742148446798034554185499530469312263804426191814188"
                "792259918908005089337416158309492169729046978241108764800373199786835512775018459244859"
                "267531608074047213792485790714727394281686995009721892096313398856651364376835976485985"
                "577848331809780103472992243751739017814855135195390244578020544445155642189656411959818"
                "537113780293079843353672189952048766617739975099791375374874500190464500082083000996046"
                "562339788079629204480380125711374292052576290634217065477010746078997964370477540596950"
                "179494021104626979205963371667293248583566456157361250770591161294974738245422314899316"
                "567308929508634261348112806280532523476213178307252077997746083513294549227767091938067"
                "217624964628370906697291402801808940793284640660655062072285201193967242441893707135722"
                "290023979396671884835177432518035657429152687499442438672252127647437051589773194843058"
                "438225198896308860630489777828329667886104302882092289082288763901527187068327652071603"
                "624848895710581676120737165753641607798272808608702931524263579983568144332095173143917"
                "006791039477173888020650319663759808299494529128325593983395874886465461027967163402161"
                "841883781708743258815462635693304813049558585059423645696496789082810397724886054771581"
                "563831522836536737064904299544589884335904000319915121211509398654760775082923447627431"
                "810306326621272995546270200576623778725786936601643047562903876270563152346113203816665"
                "415647322772310805862830328907715627364519002365100137737699655671974565189869468321310"
                "071771026817931314136062271980786119418884667896744248988022776918612392621307244846715"
                "565141673730197159045126169762621237477118095627007495171039905668325659077157792159884"
                "943945222288639927266673422568383814713249358515769104841319551023369437288552466511955"
                "117808978763619586608932135694850483552263188003555050643236007295112136223460594779054"
                "726696786510462803932089824565856848396625577598749730172563451415247181522187861098101"
                "676294774399809634350068762970820625702542372795052750262991463099037894875056129499728"
                "445824201949824805059249789991826703880335086183597727635326230040287804809229904724327"
                "852548533423811116498779411377941487636025548910716603561208719468445816412340563590498"
                "400399244784036149284049457540915194474505097972683897906679365708314054347905676337878"
                "070783604662861248077878016527821790505946032823348841177178634386993182797943842539434"
                "419858352852503464295581430236588609770297176292756751968184573897538526914043047169814"
                "984561707551218821949790226202641140107370608506446237466149862964632615116143823940797"
                "596880439913349528709020788483323602475404095930729914237392360095230982170538229907966"
                "495086002892040173075678439373003701909762402992875028567037197838993284718386014290238"
                "070433830095663716147654734717059715386297217424718293321056467774762043970327984658152"
                "380500694244445759891469465190557700800470744175185680654189821923413620537583275222898"
                "761381663910864540456551646021987066654828544317400746914117387102118080351849066955339"
                "664552514032346001119032775166253582621422092613876844626462373903273430772755116511435"
                "087906670944894552654778284697192895688418054857236719421320513108694045127830937696976"
                "297674990198674156800094322748981626104687761720671181734337055644894970425380677001034"
                "000849743270912591141325563959537192085987620330490840476536083886276761925001294346732"
                "505633809631143917183216956647639752360631157837300580153010392919582696581794027662533"
                "624305037354263879163470222707636248134905305296685272910483908282847104831998228434858"
                "974130961151274297209438917064183987633962816357205519261740252040644213788393339141118"
                "739697979968256978909112973181868019620886889987187481890055895367983801459002799076818"
                "837986663883724217746910902316915065093257083145192704924689072159703769249070436309481"
                "736201756702232140567534778558365775966006770532140490701287831846815608863443209730182"
                "845519291896767911368267517992656584298826507477063195274462071851960638339558820957705"
                "945408982367241117986674718249918859668209281488699934059441044250362349944518950733087"
                "378952305246683734778048472358542456378916286920677595140976322183957754227863002796881"
                "445342473036060341816544972291059016840778902131080124362200441314757556055107299290272"
                "474386805268434702813296226582427924188685957363973334690057177075025448915737921847084"
                "496839222609113297210407902116666610614824213911106641486805467563380143891770790755077"
                "167646037291760287860918118093835326201781510890299315938053532911885053322064348985683"
                "872014986880624400851522484591793802210268438103911142640495772848835072529974685218226"
                "530680635000949565847378454967660431116817851497196098293776449567538417507298269860586"
                "641110286466501057094528795544149327856999756011735764809727666818188882692663129685627"
                "307308292841788932896782171967675480989562039413508190787222142371466915917300058033573"
                "956758035410403740286953590313023855419940156309291098762553907595055420194025188519773"
                "478532383738190125727961927627826957010912300607677153328682523185847768056223658161372"
                "063956594173958678984956434812265025341867438320127897206866145137431128120320675777927"
                "278079288185190540772516346474194464949754117537260776606321880409377893860252978938855"
                "936279098736016074980495680506751358874945024248382824839312022458903650881711401811099"
                "727132527170715917016452649648068281543613104412514657514133094728931930486732130330237"
                "860233300008851728771054892940018634866579694761191535837102038913521403159566337476438"
                "903506822942696860878465228375517633393711092355856879016328067674480021558426940911197"
                "537360973531609888843203147261886391526901762055829861631932676334918110816677328134287"
                "318403433644752025487960559636346260867461273710851076900517946468485272493974552249311"
                "774727074135667145199111533222466881966616063900028205814838832809205478383123779018284"
                "464095799291747600631432261662541444929709100872226373145461025029266814100794680558287"
                "960828016367377616377299551076861209811971060912998536363961949830635848960123950221836"
                "645040123367733155857607278961778722701329765831768695142739291420761507182337978946099"
                "417367599369925618514734841956072378859975096934798193328032286550363331350324986132500"
                "464738625624685084886366703070863472686243762955596814450727423669867329521462277177679"
                "629841212369453859103602491860237020095663604077905466151242813453152308103189869801194"
                "351814672108421447717521474366046024974865715446700166858200888354627705894167831647642"
                "637437058285732555360659777349669913067216996318888409694121138734328021188041286793705"
                "441757278118362122386622470123263158470474133857455191170203145959159030143474033525161"
                "733770964482552532659350912663742295237357567167312026515467171743013903994557270927628"
                "924485170298301840043308180786825680208009489568114015487209668749540967277041676669601"
                "080507867931068038810555954236536838616675351491571608872852352605168619256936034380349"
                "797163971307747341514911888068450063912908745787316316934334243862841506162896104882759"
                "011173888435540435768368409617460551906764851402352987911282927860834380384330694648169"
                "753339317485999784"};
            SetNumIterations<IterTypeFull>(2147483646);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;

        case 22:
            minX = HighPrecision{"-0."
                                 "6907023001770585288601365254896697520219123564180210586974026317473887"
                                 "891717116357627686720779756408687304394424933003585901989015412142"};
            minY = HighPrecision{"0."
                                 "3270967712951810044828437245522451618089794418141623643693494431352454"
                                 "845268379424333502401886231660401278668233377275970460192695489989"};
            maxX = HighPrecision{"-0."
                                 "6907023001770585288601365254896697520219123564180210586974026317473887"
                                 "891717116357627686720779756408687304394424911840042901546067160599"};
            maxY = HighPrecision{"0."
                                 "3270967712951810044828437245522451618089794418141623643693494431352454"
                                 "845268379424333502401886231660401278668233395871670243248566020344"};
            SetNumIterations<IterTypeFull>(2147483646);
            ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
            break;

        case 23:
            minX = HighPrecision{"-0."
                                 "7485402233636429873317377178087497162167763288165931811664800714714691"
                                 "5304640714984571574330253031"};
            minY = HighPrecision{"0."
                                 "0646874933074049458482628420379299288470805576796381355016700052390982"
                                 "31744262964560868111971917423"};
            maxX = HighPrecision{"-0."
                                 "7485402233636429873317377178087497150392993040119672676435021612012040"
                                 "0755946688800146880673948663"};
            maxY = HighPrecision{"0."
                                 "0646874933074049458482628420379299300245575824842640490246479155093633"
                                 "77231203226405115048534961104"};
            SetIterType(IterTypeEnum::Bits64);
            SetNumIterations<IterTypeFull>(51'539'607'504llu);
            break;

        case 24:
            minX = HighPrecision{"-0."
                                 "3765623535167500135544978190514949279968288054594107843522817158282037"
                                 "4005540296873316709319956486"};
            minY = HighPrecision{"0."
                                 "6715894786762578641425805624574991939260604875743578679564780863333874"
                                 "0646688656232915470730827391"};
            maxX = HighPrecision{"-0."
                                 "3765623535167500135544978190514915320709811580213234195055793009997143"
                                 "1732574605202825177873455922"};
            maxY = HighPrecision{"0."
                                 "6715894786762578641425805624575025898519081350124452328031805011618768"
                                 "2919654347903407002177327955"};
            SetIterType(IterTypeEnum::Bits64);
            SetNumIterations<IterTypeFull>(51'539'607'504llu);
            break;

        case 25: {
            // from Claude Heiland-Allen
            // Low-period, but hard to render
            // 32-bit + perturbation only
            PointZoomBBConverter convert{HighPrecision{"3.56992006738525396399695724115347205e-01"},
                                         HighPrecision{"6.91411005282446050826514373514151521e-02"},
                                         HighPrecision{"1e19"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            SetNumIterations<IterTypeFull>(1'100'100'100);
            break;
        }

        case 26:
            minX = HighPrecision{
                "-0.1605261093438198889198301833197383152634624627924018099119331622744150582179648"};
            minY = HighPrecision{
                "-1.037616767084875731043652452075011878717204406363659116578730306577073971695717"};
            maxX = HighPrecision{
                "-0.1605261093438198838959220991033570359417441984488856377425735296717555605747732"};
            maxY = HighPrecision{
                "-1.037616767084875726019744367858630599395486142020142944409370673974414474052526"};
            SetNumIterations<IterTypeFull>(500'000'000);
            break;

        case 27:
            if (includeMsgBox) {
                ::MessageBox(
                    nullptr,
                    L"Warning: This is a very large image.  It will take a long time to render.",
                    L"Warning",
                    MB_OK | MB_APPLMODAL | MB_ICONWARNING);
            }

            // This text is copied to clipboard.Using "GpuHDRx2x32PerturbedRCLAv2"
            //     Antialiasing : 1
            //     Palette depth : 8
            //     Coordinate precision = 107;
            // Center X :
            // "-0.708629559893199373462249361171627445147997107174967545782137006452491235099644657748836185407480786537951038854249622"
            //     Center Y :
            //     "0.258843760578193851766291691547748715299348571148609040920235011011819852620117191013214718127019274971026034483874936"
            //     zoomFactor "7.90516141241318534606e+57"

            //    Additional details :
            // m_PeriodMaybeZero = 23387951809
            //    CompressedIters = 299845537
            //    UncompressedIters = 23387951809
            //    Compression ratio = 78.000000
            //    Compression error exp = 20

            //    Bounding box :
            // minX = HighPrecision{
            // "-0.708629559893199373462249361171627445147997107174967545782390005714069812517887859808545215572778069136090230432052003"
            // }; minY = HighPrecision{
            // "0.258843760578193851766291691547748715299348571148609040920129594652828778695849190155002622224812073888468037993123944"
            // }; maxX = HighPrecision{
            // "-0.708629559893199373462249361171627445147997107174967545781884007190912657681401455689127155242183503939811847276447242"
            // }; maxY = HighPrecision{
            // "0.258843760578193851766291691547748715299348571148609040920340427370810926544385191871426814029226476053584030974625928"
            // }; SetNumIterations<IterTypeFull>(50000000000000);

            //    Bounding box :
            minX = HighPrecision{"-0."
                                 "7086295598931993734622493611716274451479971071749675457822424228114823"
                                 "09023912658607048281309687987620509035345000615"};
            minY = HighPrecision{"0."
                                 "2588437605781938517662916915477487152993485711486090409201295946528287"
                                 "78695849190155002622224812073888468037993123944"};
            maxX = HighPrecision{"-0."
                                 "7086295598931993734622493611716274451479971071749675457820315900935001"
                                 "6117537665689062408950527358545539304236349863"};
            maxY = HighPrecision{"0."
                                 "2588437605781938517662916915477487152993485711486090409203404273708109"
                                 "26544385191871426814029226476053584030974625928"};
            m_LAParameters.SetDefaults(LAParameters::LADefaults::MaxPerf);
            SetCompressionErrorExp(Fractal::CompressionError::Low, 20);
            SetIterType(IterTypeEnum::Bits64);

            // 1 quadrillion works but 50 trillion is plenty.  1 trillion is visibly insufficient.
            // SetNumIterations<IterTypeFull>(1'000'000'000'000'000);
            SetNumIterations<IterTypeFull>(50'000'000'000'000);
            break;

        case 28: {
            // A different fast-rendering example of HDRx32 falling apart but HDRx64 working fine.
            minX = HighPrecision{"-5."
                                 "4820574807047570845821256754673302937669927499287325074403026784136146"
                                 "158613437596334009499566999e-01"};
            minY = HighPrecision{"-5."
                                 "7757083890360384280510898220185055867555172854045665290176278505246929"
                                 "025739738078012702647180541e-01"};
            maxX = HighPrecision{"-5."
                                 "4820574807047570845821256754673302937669927499285756222573292062768568"
                                 "705702095943792838057103099e-01"};
            maxY = HighPrecision{"-5."
                                 "7757083890360384280510898220185055867555172854045011601913889038010438"
                                 "420360012389453881212820582e-01"};
            SetNumIterations<IterTypeFull>(1'658'880'000);
            break;
        }
        case 29: {
            // A fast-rendering example of HDRx32 falling apart but HDRx64 working fine.
            minX = HighPrecision{
                "-1."
                "769108330407477280892306243557778483560063911862340065077878834041892971598590808442346"
                "019908868202043757985397117440428151862078674609536134158424327579445927678293493358112"
                "333445291014142565932118624495970123204844421824353001868066369287727901987459220461426"
                "842090582178606291866663503534261191565880688006634728233063960400259650628721114525674"
                "474617782491596124839618269380451206162229194255407244353147355253412558206636728680544"
                "948343043687334876817161811548434128474621486997968757044320805797353787079320992599433"
                "404969124074045418302417070673478351407751216433694350093997839981674873003906319202253"
                "208202547286958709279547762956011275901605378309795411610077736531155287767727090460684"
                "634373036437021799829783590529108042624544498697873839995455262813972110546"};
            minY = HighPrecision{
                "-0."
                "009020688057072617600360935984947620112305584674123866889727919817129110000273070325734"
                "652748577260411641977284267953435586679185748062521452185618401720512375928279282840017"
                "325461435395900347333831678867372709608373937732035861376732233573306124188011779554347"
                "246727032292516900609906698962747844988602428404977018518490333180457529450252363392753"
                "154488987278181741626980443558895836087364851272299469096298326327508396224591600469344"
                "439018862266215051078424481599350178419410453543640423233428241935033928198320388056103"
                "242790716487099369717401956440284903703674535517982961700987544448683097493910719867515"
                "406876699163015871869774586895353977306340825927065496180267868663306898577037289686648"
                "909313180741633692815372792496647017000301271474430650982039609621457257550363"};
            maxX = HighPrecision{
                "-1."
                "769108330407477280892306243557778483560063911862340065077878834041892971598590808442346"
                "019908868202043757985397117440428151862078674609536134158424327579445927678293493358112"
                "333445291014142565932118624495970123204844421824353001868066369287727901987459220461426"
                "842090582178606291866663503534261191565880688006634728233063960400259650628721114525674"
                "474617782491596124839618269380451206162229194255407244353147355253412558206636728680544"
                "948343043687334876817161811548434128474621486997968757044320805797353787079320992599433"
                "404969124074045418302417070673478351407751216433694350093997839981674873003906319202253"
                "208202547286958709279547762956011275901605378309795411610077736531155287767727090460684"
                "634373036437021799829477424384404967624358838889441067112448830585395145838"};
            maxY = HighPrecision{
                "-0."
                "009020688057072617600360935984947620112305584674123866889727919817129110000273070325734"
                "652748577260411641977284267953435586679185748062521452185618401720512375928279282840017"
                "325461435395900347333831678867372709608373937732035861376732233573306124188011779554347"
                "246727032292516900609906698962747844988602428404977018518490333180457529450252363392753"
                "154488987278181741626980443558895836087364851272299469096298326327508396224591600469344"
                "439018862266215051078424481599350178419410453543640423233428241935033928198320388056103"
                "242790716487099369717401956440284903703674535517982961700987544448683097493910719867515"
                "406876699163015871869774586895353977306340825927065496180267868663306898577037289686648"
                "909313180741633692815245223269687402416890579887583662280786929526216855587951"};
            SetNumIterations<IterTypeFull>(1'658'880'000);
            break;
        }

        case 30: {

#include "LargeCoords.h"

            mpf_t mpfX, mpfY;

            Hex64StringToMpf_Exact(strXHex, mpfX);
            Hex64StringToMpf_Exact(strYHex, mpfY);

            MpfNormalize(mpfX);
            MpfNormalize(mpfY);

            PointZoomBBConverter convert{
                HighPrecision{mpfX}, HighPrecision{mpfY}, HighPrecision{"1.36733731087e+114514"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            /// SetNumIterations<IterTypeFull>(700'000);
            SetNumIterations<IterTypeFull>(200'000'000);
            break;
        }

        case 0:
        default:
            // minX = HighPrecision{ "-2.5" };
            // minY = HighPrecision{ "-1.5" };
            // maxX = HighPrecision{ "1.5" };
            // maxY = HighPrecision{ "1.5" };

            PointZoomBBConverter convert{HighPrecision{"0"}, HighPrecision{"0"}, HighPrecision{"1"}};

            minX = convert.GetMinX();
            minY = convert.GetMinY();
            maxX = convert.GetMaxX();
            maxY = convert.GetMaxY();
            ResetNumIterations();
            break;
    }

    PointZoomBBConverter convert{minX, minY, maxX, maxY};
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
    auto minX = m_Ptz.GetMinX();
    auto minY = m_Ptz.GetMinY();
    auto maxX = m_Ptz.GetMaxX();
    auto maxY = m_Ptz.GetMaxY();

    HighPrecision ratio = (HighPrecision)m_ScrnWidth / (HighPrecision)m_ScrnHeight;
    HighPrecision mwidth = (maxX - minX) / ratio;
    HighPrecision height = maxY - minY;

    if (height > mwidth) {
        minX -= ratio * (height - mwidth) / HighPrecision{2.0};
        maxX += ratio * (height - mwidth) / HighPrecision{2.0};
        m_ChangedWindow = true;
    } else if (height < mwidth) {
        minY -= (mwidth - height) / HighPrecision{2.0};
        maxY += (mwidth - height) / HighPrecision{2.0};
        m_ChangedWindow = true;
    }

    m_Ptz = PointZoomBBConverter{minX, minY, maxX, maxY};

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
            CalcFractal(false);

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

void
Fractal::SetRenderAlgorithm(RenderAlgorithm alg)
{
    if (m_BypassGpu) {
        std::cerr << "Bypassing GPU: Ignoring request to set render algorithm."
                  << std::endl;
        alg = GetRenderAlgorithmTupleEntry(RenderAlgorithmEnum::Cpu64);
    }

    m_RenderAlgorithm = alg;
    m_RefOrbit.ResetLastUsedOrbit();
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
Fractal::CalcFractal(bool MemoryOnly)
{
    // if (m_glContext->GetRepaint() == false)
    //{
    //     DrawFractal(MemoryOnly);
    //     return;
    // }

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_Overall);

    // Bypass this function if the screen is too small.
    if (m_ScrnHeight == 0 || m_ScrnWidth == 0) {
        return;
    }

    if (GetIterType() == IterTypeEnum::Bits32) {
        CalcFractalTypedIter<uint32_t>(MemoryOnly);
    } else {
        CalcFractalTypedIter<uint64_t>(MemoryOnly);
    }
}

template <typename IterType>
void
Fractal::CalcFractalTypedIter(bool MemoryOnly)
{
    // Test crash dump
    //{volatile int x = 5;
    // volatile int z = 0;
    // volatile int y = x / z;
    //}

    // Reset the flag should it be set.
    m_StopCalculating = false;

    // Do nothing if nothing has changed
    if (ChangedIsDirty() == false || (m_glContext && m_glContext->GetRepaint() == false)) {
        DrawFractal(MemoryOnly);
        return;
    }

    // Clear the screen if we're drawing.
    // if (MemoryOnly == false)
    //{
    //    glClear(GL_COLOR_BUFFER_BIT);
    //}

    // Draw the local fractal.
    // Note: This accounts for "Auto" being selected via the GetRenderAlgorithm call.
    switch (GetRenderAlgorithm().Algorithm) {
        case RenderAlgorithmEnum::CpuHigh:
            CalcCpuHDR<IterType, HighPrecision, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::CpuHDR32:
            CalcCpuHDR<IterType, HDRFloat<float>, float>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedBLAHDR:
            CalcCpuPerturbationFractalBLA<IterType, HDRFloat<float>, float>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, float, PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu32PerturbedRCBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, float, PerturbExtras::SimpleCompression>(
                MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, double, PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedRCBLAV2HDR:
            CalcCpuPerturbationFractalLAV2<IterType, double, PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu64:
            CalcCpuHDR<IterType, double, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::CpuHDR64:
            CalcCpuHDR<IterType, HDRFloat<double>, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLA:
            CalcCpuPerturbationFractalBLA<IterType, double, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Cpu64PerturbedBLAHDR:
            CalcCpuPerturbationFractalBLA<IterType, HDRFloat<double>, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64:
            CalcGpuFractal<IterType, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x64:
            CalcGpuFractal<IterType, MattDbldbl>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu4x64:
            CalcGpuFractal<IterType, MattQDbldbl>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32:
            CalcGpuFractal<IterType, float>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32:
            CalcGpuFractal<IterType, MattDblflt>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu4x32:
            CalcGpuFractal<IterType, MattQFltflt>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32:
            CalcGpuFractal<IterType, HDRFloat<double>>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedScaled:
            CalcGpuPerturbationFractalScaledBLA<IterType, double, double, float, float>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedScaled:
            CalcGpuPerturbationFractalScaledBLA<IterType, HDRFloat<float>, float, float, float>(
                MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, double, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedScaled:
            // TODO
            // CalcGpuPerturbationFractalBLA<IterType, double, double>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, HDRFloat<float>, float>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedBLA:
            CalcGpuPerturbationFractalBLA<IterType, HDRFloat<double>, double>(MemoryOnly);
            break;

            // LAV2

        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;

        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;

        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::Gpu1x64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;

        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;

        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;

        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2PO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedLAv2LAO>,
                PerturbExtras::Disable>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2PO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        case RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO:
            CalcGpuPerturbationFractalLAv2<
                IterType,
                RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx64PerturbedRCLAv2LAO>,
                PerturbExtras::SimpleCompression>(MemoryOnly);
            break;
        default:
            break;
    }

    // We are all updated now.
    ChangedMakeClean();
}

void
Fractal::UsePaletteType(FractalPalette type)
{
    m_WhichPalette = type;
    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

uint32_t
Fractal::GetPaletteDepthFromIndex(size_t index) const
{
    switch (index) {
        case 0:
            return 5;
        case 1:
            return 6;
        case 2:
            return 8;
        case 3:
            return 12;
        case 4:
            return 16;
        case 5:
            return 20;
        default:
            return 8;
    }
}

void
Fractal::UsePalette(int depth)
{
    switch (depth) {
        case 5:
            m_PaletteDepthIndex = 0;
            break;
        case 6:
            m_PaletteDepthIndex = 1;
            break;
        case 8:
            m_PaletteDepthIndex = 2;
            break;
        case 12:
            m_PaletteDepthIndex = 3;
            break;
        case 16:
            m_PaletteDepthIndex = 4;
            break;
        case 20:
            m_PaletteDepthIndex = 5;
            break;
        default:
            m_PaletteDepthIndex = 0;
            break;
    }

    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

void
Fractal::UseNextPaletteDepth()
{
    m_PaletteDepthIndex = (m_PaletteDepthIndex + 1) % 6;
    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

void
Fractal::SetPaletteAuxDepth(int32_t depth)
{
    if (depth < 0 || depth > 16) {
        return;
    }

    m_PaletteAuxDepth = depth;
    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

void
Fractal::UseNextPaletteAuxDepth(int32_t inc)
{
    if (inc < -5 || inc > 5 || inc == 0) {
        return;
    }

    if (inc < 0) {
        if (m_PaletteAuxDepth == 0) {
            m_PaletteAuxDepth = 17 + inc;
        } else {
            m_PaletteAuxDepth += inc;
        }
    } else {
        if (m_PaletteAuxDepth >= 16) {
            m_PaletteAuxDepth = -1 + inc;
        } else {
            m_PaletteAuxDepth += inc;
        }
    }

    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

uint32_t
Fractal::GetPaletteDepth() const
{
    return GetPaletteDepthFromIndex(m_PaletteDepthIndex);
}

void
Fractal::ResetFractalPalette(void)
{
    m_PaletteRotate = 0;
}

void
Fractal::RotateFractalPalette(int delta)
{
    m_PaletteRotate += delta;
    if (m_PaletteRotate >= GetMaxIterationsRT()) {
        m_PaletteRotate = 0;
    }
}

void
Fractal::CreateNewFractalPalette(void)
{
    size_t rtime = __rdtsc();

    auto genNextColor = [](int m) -> int {
        const int max_val = 65535 / (m - 1);
        auto val = (rand() % m) * max_val;
        return val;
    };

    auto RandomPaletteGen = [&](size_t PaletteIndex, size_t Depth) {
        int depth_total = (int)(1 << Depth);

        srand((unsigned int)rtime);

        // Force a reallocation of the vectors to trigger re-initialization in the GPU
        std::vector<uint16_t>{}.swap(m_PalR[FractalPalette::Random][PaletteIndex]);
        std::vector<uint16_t>{}.swap(m_PalG[FractalPalette::Random][PaletteIndex]);
        std::vector<uint16_t>{}.swap(m_PalB[FractalPalette::Random][PaletteIndex]);

        const int m = 5;
        auto firstR = genNextColor(m);
        auto firstG = genNextColor(m);
        auto firstB = genNextColor(m);
        PalTransition(FractalPalette::Random, PaletteIndex, depth_total, firstR, firstG, firstB);
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random,
                      PaletteIndex,
                      depth_total,
                      genNextColor(m),
                      genNextColor(m),
                      genNextColor(m));
        PalTransition(FractalPalette::Random, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[FractalPalette::Random][PaletteIndex] =
            (uint32_t)m_PalR[FractalPalette::Random][PaletteIndex].size();
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 0, 5));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 1, 6));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 2, 8));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 3, 12));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 4, 16));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 5, 20));

    for (auto &it : threads) {
        it->join();
    }
}

//////////////////////////////////////////////////////////////////////////////
// Redraws the fractal using OpenGL calls.
// Note that coordinates here are weird, so we have to make a few tweaks to
// get the image oriented right side up. In particular, the line which reads:
//       glVertex2i (px, m_ScrnHeight - py);
//////////////////////////////////////////////////////////////////////////////
void
Fractal::DrawFractal(bool /*MemoryOnly*/)
{
    {
        std::unique_lock lk(m_AsyncRenderThreadMutex);

        m_AsyncRenderThreadCV.wait(
            lk, [&] { return m_AsyncRenderThreadState == AsyncRenderThreadState::Idle; });

        m_AsyncRenderThreadState = AsyncRenderThreadState::Start;
    }

    m_AsyncRenderThreadCV.notify_one();

    uint32_t result = m_r.SyncStream(false);
    if (result) {
        MessageBoxCudaError(result);
    }

    {
        std::unique_lock lk(m_AsyncRenderThreadMutex);

        m_AsyncRenderThreadCV.wait(
            lk, [&] { return m_AsyncRenderThreadState == AsyncRenderThreadState::Idle; });

        m_AsyncRenderThreadState = AsyncRenderThreadState::SyncDone;
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
Fractal::DrawGlFractal(bool LocalColor, bool lastIter)
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

        auto result = m_r.RenderCurrent<IterType>(GetNumIterations<IterType>(),
                                                  iter,
                                                  m_CurIters.m_RoundedOutputColorMemory.get(),
                                                  &gpuReductionResults);

        if (result) {
            MessageBoxCudaError(result);
            return;
        }

        result = m_r.SyncStream(true);
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
    m_glContext->SetRepaint(repaint);
}

bool
Fractal::GetRepaint() const
{
    return m_glContext->GetRepaint();
}

void
Fractal::ToggleRepainting()
{
    m_glContext->ToggleRepaint();
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
    glFlush();
}

void
Fractal::DrawFractalThread(size_t index, Fractal *fractal)
{
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
            uint32_t palIters =
                fractal->m_PalIters[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex];
            const uint16_t *palR =
                fractal->m_PalR[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            const uint16_t *palG =
                fractal->m_PalG[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            const uint16_t *palB =
                fractal->m_PalB[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            size_t basicFactor = 65536 / NumIterations;
            if (basicFactor == 0) {
                basicFactor = 1;
            }

            const auto GetBasicColor =
                [&](size_t numIters, size_t &acc_r, size_t &acc_g, size_t &acc_b) {
                    auto shiftedIters = (numIters >> fractal->m_PaletteAuxDepth);

                    if (fractal->m_WhichPalette != FractalPalette::Basic) {
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
                            numIters += fractal->m_PaletteRotate;
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
                                    numIters += fractal->m_PaletteRotate;
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
    m_glContext = std::make_unique<OpenGlContext>(m_hWnd);
    if (!m_glContext->IsValid()) {
        return;
    }

    auto lambda = [&]<typename IterType>(bool lastIter) -> bool {
        m_glContext->glResetViewDim(m_ScrnWidth, m_ScrnHeight);

        if (m_glContext->GetRepaint() == false) {
            m_glContext->DrawGlBox();
            return false;
        }

        const bool LocalColor = RequiresUseLocalColor();
        DrawGlFractal<IterType>(LocalColor, lastIter);
        return false;
    };

    for (;;) {
        {
            std::unique_lock lk(m_AsyncRenderThreadMutex);

            m_AsyncRenderThreadCV.wait(lk, [&] {
                return m_AsyncRenderThreadState == AsyncRenderThreadState::Start ||
                       m_AsyncRenderThreadState == AsyncRenderThreadState::Finish;
            });

            if (m_AsyncRenderThreadState == AsyncRenderThreadState::Finish) {
                break;
            }

            m_AsyncRenderThreadState = AsyncRenderThreadState::Idle;
        }

        m_AsyncRenderThreadCV.notify_one();

        for (size_t i = 0;; i++) {
            // Setting the timeout lower works fine but puts more load on the GPU
            static constexpr auto set_time = std::chrono::milliseconds(1000);

            std::unique_lock lk(m_AsyncRenderThreadMutex);
            auto doneearly = m_AsyncRenderThreadCV.wait_for(lk, set_time, [&] {
                return m_AsyncRenderThreadState == AsyncRenderThreadState::SyncDone;
            });

            m_AsyncRenderThreadState = AsyncRenderThreadState::Idle;

            bool err;

            if (GetIterType() == IterTypeEnum::Bits32) {
                err = lambda.template operator()<uint32_t>(doneearly);
            } else {
                err = lambda.template operator()<uint64_t>(doneearly);
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

    m_glContext = nullptr;
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
    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    HighPrecision src_dy = (maxY - minY) / (HighPrecision)(m_ScrnHeight * GetGpuAntialiasing());

    HighPrecision src_dx = (maxX - minX) / (HighPrecision)(m_ScrnWidth * GetGpuAntialiasing());

    FillCoord(minX, cx2);
    FillCoord(minY, cy2);
    FillCoord(src_dx, dx2);
    FillCoord(src_dy, dy2);
}

template <typename IterType, class T>
void
Fractal::CalcGpuFractal(bool MemoryOnly)
{
    T cx2{}, cy2{}, dx2{}, dy2{};
    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    err = m_r.Render(
        GetRenderAlgorithm(), cx2, cy2, dx2, dy2, GetNumIterations<IterType>(), m_IterationPrecision);

    if (err) {
        MessageBoxCudaError(err);
    }

    DrawFractal(MemoryOnly);
}

template <typename IterType>
void
Fractal::CalcCpuPerturbationFractal(bool MemoryOnly)
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     double,
                                                                     double,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>();

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

    DrawFractal(MemoryOnly);
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
Fractal::CalcCpuHDR(bool MemoryOnly)
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

    DrawFractal(MemoryOnly);
}

template <typename IterType, class T, class SubType>
void
Fractal::CalcCpuPerturbationFractalBLA(bool MemoryOnly)
{
    const auto &maxX = m_Ptz.GetMaxX();
    const auto &minX = m_Ptz.GetMinX();
    const auto &maxY = m_Ptz.GetMaxY();
    const auto &minY = m_Ptz.GetMinY();

    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>();

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

    DrawFractal(MemoryOnly);
}

template <typename IterType, class SubType, PerturbExtras PExtras>
void
Fractal::CalcCpuPerturbationFractalLAV2(bool MemoryOnly)
{
    using T = HDRFloat<SubType>;
    using TComplex = HDRFloatComplex<SubType>;
    auto *results =
        m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                         T,
                                                         SubType,
                                                         PExtras,
                                                         RefOrbitCalc::Extras::IncludeLAv2>();

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

    DrawFractal(MemoryOnly);
}

template <typename IterType, class T, class SubType>
void
Fractal::CalcGpuPerturbationFractalBLA(bool MemoryOnly)
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Disable,
                                                                     RefOrbitCalc::Extras::None>();

    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory<IterType>();

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
    result = m_r.RenderPerturbBLA<IterType, T>(GetRenderAlgorithm(),
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

    DrawFractal(MemoryOnly);

    if (result) {
        MessageBoxCudaError(result);
    }
}

template <typename IterType, typename RenderAlg, PerturbExtras PExtras>
void
Fractal::CalcGpuPerturbationFractalLAv2(bool MemoryOnly)
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
    auto err = InitializeGPUMemory(results != nullptr);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                               ConditionalT,
                                                               ConditionalSubType,
                                                               PExtras,
                                                               RefOrbitMode,
                                                               T>();

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

    err = m_r.InitializePerturb<IterType, T, SubType, PExtras, T>(
        results->GetGenerationNumber(), &gpu_results, 0, nullptr, results->GetLaReference());
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->GetHiX() - m_Ptz.GetMinX();
    HighPrecision centerY = results->GetHiY() - m_Ptz.GetMaxY();

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    ScopedBenchmarkStopper stopper(m_BenchmarkData.m_PerPixel);
    auto result = m_r.RenderPerturbLAv2<IterType, T, SubType, Mode, PExtras>(
        GetRenderAlgorithm(), cx2, cy2, dx2, dy2, centerX2, centerY2, GetNumIterations<IterType>());

    if (result) {
        MessageBoxCudaError(result);
        return;
    }

    DrawFractal(MemoryOnly);
}

template <typename IterType, class T, class SubType, class T2, class SubType2>
void
Fractal::CalcGpuPerturbationFractalScaledBLA(bool MemoryOnly)
{
    auto *results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<IterType,
                                                                     T,
                                                                     SubType,
                                                                     PerturbExtras::Bad,
                                                                     RefOrbitCalc::Extras::None>();
    auto *results2 =
        m_RefOrbit
            .CopyUsefulPerturbationResults<IterType, T, PerturbExtras::Bad, T2, PerturbExtras::Bad>(
                *results);

    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory<IterType>();

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
    auto result = m_r.RenderPerturbBLAScaled<IterType, T>(GetRenderAlgorithm(),
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

    DrawFractal(MemoryOnly);

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
        *oldSettings = RecommendedSettings{m_Ptz.GetMinX(),
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
        SetRenderAlgorithm(recommendedSettings.GetRenderAlgorithm());
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
    if (Enable == AddPointOptions::EnableWithSave) {
        m_RefOrbit.SetOptions(AddPointOptions::EnableWithSave);
    } else if (Enable == AddPointOptions::EnableWithoutSave) {
        m_RefOrbit.SetOptions(AddPointOptions::EnableWithoutSave);
    } else {
        m_RefOrbit.SetOptions(AddPointOptions::DontSave);
    }
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
Fractal::XFromScreenToCalc(HighPrecision x)
{
    HighPrecision aa(IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1);
    HighPrecision highHeight(m_ScrnHeight);
    HighPrecision highWidth(m_ScrnWidth);
    HighPrecision OriginX{highWidth * aa / (m_Ptz.GetMaxX() - m_Ptz.GetMinX()) * -m_Ptz.GetMinX()};
    return HighPrecision{(x - OriginX) * (m_Ptz.GetMaxX() - m_Ptz.GetMinX()) / (highWidth * aa)};
}

template <bool IncludeGpuAntialiasing>
HighPrecision
Fractal::YFromScreenToCalc(HighPrecision y)
{
    HighPrecision aa(IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1);
    HighPrecision highHeight(m_ScrnHeight);
    HighPrecision highWidth(m_ScrnWidth);
    HighPrecision OriginY =
        (HighPrecision)(highHeight * aa) / (m_Ptz.GetMaxY() - m_Ptz.GetMinY()) * m_Ptz.GetMaxY();
    return HighPrecision{-(y - OriginY) * (m_Ptz.GetMaxY() - m_Ptz.GetMinY()) / (highHeight * aa)};
}

HighPrecision
Fractal::XFromCalcToScreen(HighPrecision x) const
{
    HighPrecision highHeight(m_ScrnHeight);
    HighPrecision highWidth(m_ScrnWidth);
    return HighPrecision{(x - m_Ptz.GetMinX()) * (highWidth / (m_Ptz.GetMaxX() - m_Ptz.GetMinX()))};
}

HighPrecision
Fractal::YFromCalcToScreen(HighPrecision y) const
{
    HighPrecision highHeight(m_ScrnHeight);
    HighPrecision highWidth(m_ScrnWidth);
    return HighPrecision{highHeight -
                         (y - m_Ptz.GetMinY()) * highHeight / (m_Ptz.GetMaxY() - m_Ptz.GetMinY())};
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
        std::string("LA generation time (ms) = ") + std::to_string(details.LAMilliseconds) + "\r\n";

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
