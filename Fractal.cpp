#include "stdafx.h"
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <io.h>
#include <time.h>

#include <locale>
#include <codecvt>

#include <GL/gl.h>      /* OpenGL header file */
#include <GL/glu.h>     /* OpenGL utilities header file */

#include "Fractal.h"
#include "FractalNetwork.h"
#include "FractalSetupData.h"
//#include "CBitmapWriter.h"

#include "BLAS.h"

#include <thread>
#include <psapi.h>

// Match in render_gpu.cu
constexpr static auto NB_THREADS_W = 8; // W=16, H=8 previously seemed OK
constexpr static auto NB_THREADS_H = 8;

void DefaultOutputMessage(const wchar_t *, ...);

Fractal::ItersMemoryContainer::ItersMemoryContainer(size_t width, size_t height, size_t total_antialiasing)
    : m_ItersMemory(nullptr),
      m_ItersArray(nullptr),
      m_Width(),
      m_Height(),
      m_Total() {
    size_t antialias_width = width * total_antialiasing;
    size_t antialias_height = height * total_antialiasing;

    size_t w_block = antialias_width / NB_THREADS_W + (antialias_width % NB_THREADS_W != 0);
    size_t h_block = antialias_height / NB_THREADS_H + (antialias_height % NB_THREADS_H != 0);

    // This array must be identical in size to iter_matrix_cu in CUDA

    m_Width = w_block * NB_THREADS_W;
    m_Height = h_block * NB_THREADS_H;
    m_Total = m_Width * m_Height;
    m_ItersMemory = new uint32_t[m_Total];
    memset(m_ItersMemory, 0, m_Total * sizeof(uint32_t));

    m_ItersArray = new uint32_t*[m_Height];
    for (size_t i = 0; i < m_Height; i++) {
        m_ItersArray[i] = &m_ItersMemory[i * m_Width];
    }
};

Fractal::ItersMemoryContainer::ItersMemoryContainer(Fractal::ItersMemoryContainer&& other) noexcept {
    *this = std::move(other);
}

Fractal::ItersMemoryContainer &Fractal::ItersMemoryContainer::operator=(Fractal::ItersMemoryContainer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    m_ItersMemory = other.m_ItersMemory;
    other.m_ItersMemory = nullptr;

    m_ItersArray = other.m_ItersArray;
    other.m_ItersArray = nullptr;

    m_Width = other.m_Width;
    m_Height = other.m_Height;
    m_Total = other.m_Total;

    return *this;
}

Fractal::ItersMemoryContainer::~ItersMemoryContainer() {
    if (m_ItersMemory) {
        delete[] m_ItersMemory;
        m_ItersMemory = nullptr;
    }

    if (m_ItersArray) {
        delete[] m_ItersArray;
        m_ItersArray = nullptr;
    }
}

Fractal::Fractal(FractalSetupData* setupData,
    int width,
    int height,
    void(*pOutputMessage) (const wchar_t*, ...),
    HWND hWnd,
    bool UseSensoCursor) :
    m_CurIters(width, height, 1)
{
    setupData->CopyFromThisTo(&m_SetupData);
    Initialize(width, height, pOutputMessage, hWnd, UseSensoCursor);
}

Fractal::~Fractal()
{
    Uninitialize();
}

void Fractal::Initialize(int width,
    int height,
    void(*pOutputMessage) (const wchar_t*, ...),
    HWND hWnd,
    bool UseSensoCursor)
{ // Create the control-key-down/mouse-movement-monitoring thread.
    if (hWnd != NULL)
    {
        m_AbortThreadQuitFlag = false;
        m_UseSensoCursor = UseSensoCursor;
        m_hWnd = hWnd;

        DWORD threadID;
        m_CheckForAbortThread = (HANDLE)CreateThread(NULL, 0, CheckForAbortThread, this, 0, &threadID);
        if (m_CheckForAbortThread == NULL)
        {
        }
    }
    else
    {
        m_AbortThreadQuitFlag = false;
        m_UseSensoCursor = false;
        m_hWnd = NULL;

        m_CheckForAbortThread = NULL;
    }

    InitStatics();

    // Setup member variables with initial values:
    //SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLAHDR);
    //SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLA);
    SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedBLA);
    //SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedScaled);
    SetIterationPrecision(1);
    SetPerturbationAlg(PerturbationAlg::MTPeriodicity2);

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;

    ResetDimensions(width, height, 1);
    View(0);
    //View(5);

    m_ChangedWindow = true;
    m_ChangedScrn = true;
    m_ChangedIterations = true;

    srand((unsigned int) time(NULL));

    UsePaletteType(Palette::Default);

    // Initialize the palette
    auto DefaultPaletteGen = [&](Palette WhichPalette, size_t PaletteIndex, size_t Depth) {
        int depth_total = (int) (1 << Depth);

        int max_val = 65535;
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, 0);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, max_val, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, max_val, 0, max_val);
        PalTransition(WhichPalette, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[WhichPalette][PaletteIndex] = (uint32_t) m_PalR[WhichPalette][PaletteIndex].size();
    };

    auto PatrioticPaletteGen = [&](Palette WhichPalette, size_t PaletteIndex, size_t Depth) {
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
        
    auto SummerPaletteGen = [&](Palette WhichPalette, size_t PaletteIndex, size_t Depth) {
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

    for (size_t i = 0; i < Palette::Num; i++) {
        m_PalIters[i].resize(NumBitDepths);
    }

    std::vector<std::unique_ptr<std::thread>> threads;

    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 0, 6));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 1, 8));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 2, 12));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 3, 16));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 4, 20));

    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 0, 6));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 1, 8));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 2, 12));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 3, 16));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 4, 20));

    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 0, 6));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 1, 8));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 2, 12));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 3, 16));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 4, 20));

    for (size_t i = 0; i < std::thread::hardware_concurrency(); i++) {
        m_DrawThreads.emplace_back(
            std::make_unique<DrawThreadSync>(
                i,
                nullptr,
                m_DrawThreadAtomics
            )
        );
    }

    for (size_t i = 0; i < m_DrawThreads.size(); i++) {
        auto thread = std::make_unique<std::thread>(DrawFractalThread, i, this);
        m_DrawThreads[i]->m_Thread = std::move(thread);
    }

    UsePalette(8);
    m_PaletteRotate = 0;

    // Allocate the iterations array.
    int i;
    InitializeMemory();

    // Wait for all this shit to get done
    for (auto& it : threads) {
        it->join();
    }

    // Initialize the networking
    for (i = 0; i < MAXSERVERS; i++)
    {
        m_ClientMainNetwork[i] = NULL;
        m_ClientSubNetwork[i] = NULL;
    }

    m_ServerMainNetwork = NULL;
    m_ServerSubNetwork = NULL;

    // Allocate networking memory
    if (m_SetupData.m_BeNetworkClient == 'y' && m_SetupData.m_BeNetworkServer == 'y')
    {
        m_NetworkRender = '0';
    }
    else
    {
        if (m_SetupData.m_BeNetworkClient == 'y')
        {
            m_NetworkRender = 'a';

            for (i = 0; i < MAXSERVERS; i++)
            {
                if (m_SetupData.m_UseThisServer[i] == 'n')
                {
                    continue;
                }

                m_ClientMainNetwork[i] = new FractalNetwork();
                m_ClientSubNetwork[i] = new FractalNetwork();
            }
        }
        else if (m_SetupData.m_BeNetworkServer == 'y')
        {
            m_NetworkRender = 'S'; // 'S' is a placeholder until the server
            m_ServerMainNetwork = new FractalNetwork();
            m_ServerSubNetwork = new FractalNetwork();
        }
        else                     // gets its real ID from the client
        {
            m_NetworkRender = '0';
        }
    }

    // Create the workload for all the servers and clients.
    // This function delegates all work to the local machine if
    // we're not using networking
    if (pOutputMessage != NULL)
    {
        OutputMessage = pOutputMessage;
    }
    else
    {
        OutputMessage = DefaultOutputMessage;
    }

    NetworkCreateWorkload();
    if (m_NetworkRender == 'a')
    {
        ClientInitializeServers();
    }
    else if (m_NetworkRender == 'S')
    {
        DWORD threadID;
        m_ServerMainThread = (HANDLE)CreateThread(NULL, 0, ServerManageMainConnectionThread, this, 0, &threadID);
        m_ServerSubThread = (HANDLE)CreateThread(NULL, 0, ServerManageSubConnectionThread, this, 0, &threadID);
    }

    // Recreate the workload
    // This will be important if one of the servers doesn't respond.
    if (m_NetworkRender == '0')
    {
        NetworkCreateWorkload();
    }

    // Make sure the screen is completely redrawn the first time.
    ChangedMakeDirty();
}

void Fractal::InitializeMemory() {
    // Wait until anyone using any of this memory is done.
    m_FractalSavesInProgress.clear();

    // Set up new memory.
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    m_ItersMemoryStorage.clear();
    for (size_t i = 0; i < 3; i++) {
        const size_t total_aa = GetGpuAntialiasing();
        ItersMemoryContainer container(m_ScrnWidth,
                                       m_ScrnHeight,
                                       total_aa);
        m_ItersMemoryStorage.push_back(std::move(container));
    }

    m_CurIters = std::move(m_ItersMemoryStorage.back());
    m_ItersMemoryStorage.pop_back();

    m_DrawThreadAtomics.resize(m_ScrnHeight);
    m_DrawOutBytes = std::make_unique<GLushort[]>(m_ScrnWidth * m_ScrnHeight * 4); // RGBA
}

void Fractal::ReturnIterMemory(ItersMemoryContainer&& to_return) {
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    m_ItersMemoryStorage.push_back(std::move(to_return));
}

void Fractal::GetIterMemory() {
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

void Fractal::Uninitialize(void)
{ // Get rid of the abort thread, but only if we actually used it.
    if (m_CheckForAbortThread != NULL)
    {
        m_AbortThreadQuitFlag = true;
        if (WaitForSingleObject(m_CheckForAbortThread, INFINITE) != WAIT_OBJECT_0)
        {
            ::MessageBox(NULL, L"Error waiting for abort thread!", L"", MB_OK);
        }
    }

    for (auto& thread : m_DrawThreads) {
        {
            std::lock_guard lk(thread->m_DrawThreadMutex);
            thread->m_DrawThreadReady = true;
            thread->m_TimeToExit = true;
            std::cout << "main() signals data ready for processing\n";
        }
        thread->m_DrawThreadCV.notify_one();
    }

    for (auto& thread : m_DrawThreads) {
        thread->m_Thread->join();
    }

    // Disconnect from the remote server if necessary
    //if (m_NetworkRender == 'a')
    //{
    //    char data[512];
    //    strcpy(data, "done");

    //    for (int i = 0; i < MAXSERVERS; i++)
    //    {
    //        if (m_SetupData.m_UseThisServer[i] == 'n')
    //        {
    //            continue;
    //        }

    //        m_ClientMainNetwork[i]->SendData(data, 512);
    //        m_ClientMainNetwork[i]->ShutdownConnection();
    //        delete m_ClientMainNetwork[i];
    //        delete m_ClientSubNetwork[i];
    //    }
    //} // Shutdown the various server threads if necessary.
    //else if (m_NetworkRender >= 'b' || m_NetworkRender == 'S')
    //{
    //    FractalNetwork exitNetwork;

    //    char quittime[512];
    //    strcpy(quittime, "exit");

    //    // First, shutdown secondary thread.
    //    if (exitNetwork.CreateConnection(m_SetupData.m_LocalIP, PORTNUM) == true)
    //    {
    //        exitNetwork.SendData(quittime, 512);
    //        exitNetwork.ShutdownConnection();
    //    }
    //    else
    //    {
    //        ::MessageBox(NULL, L"Error connecting to server thread #1!", L"", MB_OK);
    //    }

    //    if (WaitForSingleObject(m_ServerSubThread, INFINITE) != WAIT_OBJECT_0)
    //    {
    //        ::MessageBox(NULL, L"Error waiting for server thread #1!", L"", MB_OK);
    //    }
    //    CloseHandle(m_ServerSubThread);

    //    // Then shutdown primary thread.
    //    if (exitNetwork.CreateConnection(m_SetupData.m_LocalIP, PERM_PORTNUM) == true)
    //    {
    //        exitNetwork.SendData(quittime, 512);
    //        exitNetwork.ShutdownConnection();
    //    }
    //    else
    //    {
    //        ::MessageBox(NULL, L"Error connecting to server thread #2!", L"", MB_OK);
    //    }

    //    if (WaitForSingleObject(m_ServerMainThread, INFINITE) != WAIT_OBJECT_0)
    //    {
    //        ::MessageBox(NULL, L"Error waiting for server thread #2!", L"", MB_OK);
    //    }
    //    CloseHandle(m_ServerMainThread);

    //    delete m_ServerMainNetwork;
    //    delete m_ServerSubNetwork;
    //}
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
void Fractal::PalIncrease(std::vector<uint16_t> &pal, int length, int val1, int val2)
{
    double delta = (double)((double)(val2 - val1)) / length;
    for (int i = 0; i < length; i++)
    {
        double result = ((double)val1 + (double)delta * (i + 1));
        pal.push_back((unsigned short)result);
    }
}

// Transitions to the color specified.
// Allows for nice smooth palettes.
// length must be > 0
// Returns index immediately following the last index we filled here
// Returns -1 if we are at the end.
void Fractal::PalTransition(size_t WhichPalette, size_t PaletteIndex, int length, int r, int g, int b)
{
    int curR, curB, curG;
    if (!m_PalR[WhichPalette][PaletteIndex].empty())
    {
        curR = m_PalR[WhichPalette][PaletteIndex][m_PalR[WhichPalette][PaletteIndex].size() - 1];
        curG = m_PalG[WhichPalette][PaletteIndex][m_PalG[WhichPalette][PaletteIndex].size() - 1];
        curB = m_PalB[WhichPalette][PaletteIndex][m_PalB[WhichPalette][PaletteIndex].size() - 1];
    }
    else
    {
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
void Fractal::ResetDimensions(size_t width,
                              size_t height,
                              uint32_t gpu_antialiasing)
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

    if (m_ScrnWidth != width ||
        m_ScrnHeight != height ||
        m_GpuAntialiasing != gpu_antialiasing)
    {
        m_ScrnWidth = width;
        m_ScrnHeight = height;
        m_GpuAntialiasing = gpu_antialiasing;

        m_ChangedScrn = true;

        SquareCurrentView();
        InitializeMemory();
    }
}

//////////////////////////////////////////////////////////////////////////////
// Recenters the view to the coordinates specified.  Note that the coordinates
// are "calculator coordinates," not screen coordinates.
//////////////////////////////////////////////////////////////////////////////
void Fractal::SetPrecision(
    size_t prec,
    HighPrecision& minX,
    HighPrecision& minY,
    HighPrecision& maxX,
    HighPrecision& maxY)
{
    if constexpr (DigitPrecision != 0) {
        return;
    }
    else {
#ifndef CONSTANT_PRECISION
        auto prec2 = static_cast<uint32_t>(prec);
        boost::multiprecision::mpf_float::default_precision(prec2);

        minX.precision(prec2);
        maxX.precision(prec2);
        minY.precision(prec2);
        maxY.precision(prec2);
#endif
    }
}

size_t Fractal::GetPrecision(
    const HighPrecision& minX,
    const HighPrecision& minY,
    const HighPrecision& maxX,
    const HighPrecision& maxY)
{
    if constexpr (DigitPrecision != 0) {
        return DigitPrecision;
    }
    else {
#ifndef CONSTANT_PRECISION
        auto deltaX = abs(maxX - minX);
        auto deltaY = abs(maxY - minY);

        int temp_expX;
        boost::multiprecision::frexp(deltaX, &temp_expX);
        int temp_expY;
        boost::multiprecision::frexp(deltaY, &temp_expY);

        double expX = temp_expX / log(10) * log(2);
        double expY = temp_expY / log(10) * log(2);
        size_t larger = (size_t)max(abs(expX), abs(expY));
        larger += 20;
        return larger;
#endif
    }
}

void Fractal::SetPrecision() {
    if constexpr (DigitPrecision != 0) {
        return;
    }
    else {
#ifndef CONSTANT_PRECISION
        uint32_t prec = static_cast<uint32_t>(GetPrecision());
        SetPrecision(prec, m_MinX, m_MinY, m_MaxX, m_MaxY);
#endif
    }
}

size_t Fractal::GetPrecision(void) const {
    if constexpr (DigitPrecision != 0) {
        return DigitPrecision;
    }
    else {
#ifndef CONSTANT_PRECISION
        return GetPrecision(m_MinX, m_MinY, m_MaxX, m_MaxY);
#endif
    }
}

bool Fractal::RecenterViewCalc(HighPrecision MinX, HighPrecision MinY, HighPrecision MaxX, HighPrecision MaxY)
{
    SaveCurPos();

    size_t prec = GetPrecision(MinX, MinY, MaxX, MaxY);
    SetPrecision(prec, MinX, MinY, MaxX, MaxY);

    m_MinX = MinX;
    m_MaxX = MaxX;
    m_MinY = MinY;
    m_MaxY = MaxY;

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;

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
bool Fractal::RecenterViewScreen(RECT rect)
{
    SaveCurPos();

    HighPrecision newMinX = XFromScreenToCalc(rect.left);
    HighPrecision newMaxX = XFromScreenToCalc(rect.right);
    HighPrecision newMinY = YFromScreenToCalc(rect.top);
    HighPrecision newMaxY = YFromScreenToCalc(rect.bottom);

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
    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;

    if (RequiresReferencePoints()) {
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
                totaliters += m_CurIters.m_ItersArray[y][x];
            }
        }

        double avgiters = totaliters / ((antiRect.bottom - antiRect.top) * (antiRect.right - antiRect.left));

        for (auto y = antiRect.top; y < antiRect.bottom; y++) {
            for (auto x = antiRect.left; x < antiRect.right; x++) {
                if (m_CurIters.m_ItersArray[y][x] < avgiters) {
                    continue;
                }

                double sq = (m_CurIters.m_ItersArray[y][x]) * 
                            (m_CurIters.m_ItersArray[y][x]);
                geometricMeanSum += sq;
                geometricMeanX += sq * x;
                geometricMeanY += sq * y;
            }
        }

        assert(geometricMeanSum != 0);

        if (geometricMeanSum != 0) {
            double meanX = geometricMeanX / geometricMeanSum / GetGpuAntialiasing();
            double meanY = geometricMeanY / geometricMeanSum / GetGpuAntialiasing();
            //m_PerturbationGuessCalcX = meanX * (double)m_ScrnWidth / (double)(rect.right - rect.left);
            //m_PerturbationGuessCalcY = meanY * (double)m_ScrnHeight / (double)(rect.bottom - rect.top);

            m_PerturbationGuessCalcX = meanX;
            m_PerturbationGuessCalcY = meanY;

            m_PerturbationGuessCalcX = XFromScreenToCalc(m_PerturbationGuessCalcX);
            m_PerturbationGuessCalcY = YFromScreenToCalc(m_PerturbationGuessCalcY);

            //assert(!std::isnan(m_PerturbationGuessCalcX));
            //assert(!std::isnan(m_PerturbationGuessCalcY));

            //if (m_PerturbationGuessCalcX >= m_ScrnWidth || m_PerturbationGuessCalcX < 0 || std::isnan(m_PerturbationGuessCalcX)) {
            //    m_PerturbationGuessCalcX = 0;
            //}

            //if (m_PerturbationGuessCalcY >= m_ScrnHeight || m_PerturbationGuessCalcY < 0 || std::isnan(m_PerturbationGuessCalcY)) {
            //    m_PerturbationGuessCalcY = 0;
            //}
        }
        else {
            // Do nothing.  This case can occur if we e.g. change dimension, change gpu antialiasing etc.
        }
    }

    // Set m_PerturbationGuessCalc<X|Y> = ... above.

    m_MinX = newMinX;
    m_MaxX = newMaxX;
    m_MinY = newMinY;
    m_MaxY = newMaxY;

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
bool Fractal::CenterAtPoint(size_t x, size_t y)
{
    HighPrecision newCenterX = XFromScreenToCalc((HighPrecision)x);
    HighPrecision newCenterY = YFromScreenToCalc((HighPrecision)y);
    HighPrecision width = m_MaxX - m_MinX;
    HighPrecision height = m_MaxY - m_MinY;

    return RecenterViewCalc(newCenterX - width / 2, newCenterY - height / 2,
        newCenterX + width / 2, newCenterY + height / 2);
}

void Fractal::Zoom(double factor) {
    HighPrecision deltaX = (m_MaxX - m_MinX) * (HighPrecision)factor;
    HighPrecision deltaY = (m_MaxY - m_MinY) * (HighPrecision)factor;
    RecenterViewCalc(m_MinX - deltaX, m_MinY - deltaY, m_MaxX + deltaX, m_MaxY + deltaY);
}

void Fractal::Zoom(size_t scrnX, size_t scrnY, double factor) {
    CenterAtPoint(scrnX, scrnY);
    Zoom(factor);
}

template<Fractal::AutoZoomHeuristic h>
void Fractal::AutoZoom() {
    const HighPrecision Two = 2;

    HighPrecision width = m_MaxX - m_MinX;
    HighPrecision height = m_MaxY - m_MinY;

    HighPrecision guessX;
    HighPrecision guessY;

    double newExp = 0;
    HighPrecision p10_9;
    if constexpr (DigitPrecision != 0) {
        newExp = -(double)DigitPrecision + 10;
        p10_9 = boost::multiprecision::pow(HighPrecision{ 10 }, newExp);
    }

    size_t retries = 0;

    //static std::vector<size_t> top100;

    HighPrecision Divisor;
    if constexpr (h == AutoZoomHeuristic::Default) {
        Divisor = 3;
    }

    if constexpr (h == AutoZoomHeuristic::Max) {
        Divisor = 32;
    }

    for (;;) {
        //SaveCurPos();

        double geometricMeanX = 0;
        double geometricMeanSum = 0;
        double geometricMeanY = 0;

        if (retries >= 0) {
            width = m_MaxX - m_MinX;
            height = m_MaxY - m_MinY;
            retries = 0;
        }

        if (DigitPrecision != 0 && width < p10_9 || height < p10_9) {
            break;
        }

        size_t numAtMax = 0;
        size_t numAtLimit = 0;

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
                    auto curiter = m_CurIters.m_ItersArray[y][x];
                    totaliters += curiter;

                    if (curiter > maxiter) {
                        maxiter = curiter;
                    }

                    //if (top100.empty() ||
                    //    (!top100.empty() && curiter > top100[0])) {
                    //    top100.push_back(curiter);
                    //    std::sort(top100.begin(), top100.end(), std::less<size_t>());

                    //    if (top100.size() > 100) {
                    //        top100.erase(top100.begin());
                    //    }
                    //}
                }
            }

            double avgiters = totaliters / ((antiRect.bottom - antiRect.top) * (antiRect.right - antiRect.left));

            //double midpointX = ((double) antiRect.left + antiRect.right) / 2.0;
            //double midpointY = ((double) antiRect.top + antiRect.bottom) / 2.0;
            double antirectWidth = antiRectWidthInt;
            double antirectHeight = antiRectHeightInt;

            double widthOver2 = antirectWidth / 2.0;
            double heightOver2 = antirectHeight / 2.0;
            double maxDistance = sqrt(widthOver2 * widthOver2 + heightOver2 * heightOver2);

            for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                for (auto x = antiRect.left; x < antiRect.right; x++) {
                    auto curiter = m_CurIters.m_ItersArray[y][x];

                    if (curiter == maxiter) {
                        numAtLimit++;
                    }

                    if (curiter < avgiters) {
                        continue;
                    }

                    //if (curiter > maxiter) {
                    //    maxiter = curiter;
                    //    geometricMeanX = x;
                    //    geometricMeanY = y;
                    //    geometricMeanSum = 1;
                    //}

                    // TODO this is all fucked up
                    double distanceX = fabs(widthOver2 - fabs(widthOver2 - fabs(x - antiRect.left)));
                    double distanceY = fabs(heightOver2 - fabs(heightOver2 - fabs(y - antiRect.top)));
                    double normalizedIters = (double)curiter / (double)GetNumIterations();

                    if (curiter == maxiter) {
                        normalizedIters *= normalizedIters;
                    }
                    double normalizedDist = (sqrt(distanceX * distanceX + distanceY * distanceY) / maxDistance);
                    double sq = normalizedIters * normalizedDist;
                    geometricMeanSum += sq;
                    geometricMeanX += sq * x;
                    geometricMeanY += sq * y;

                    if (curiter >= GetNumIterations()) {
                        numAtMax++;
                    }
                }
            }

            assert(geometricMeanSum != 0);

            if (geometricMeanSum != 0) {
                double meanX = geometricMeanX / geometricMeanSum;
                double meanY = geometricMeanY / geometricMeanSum;

                guessX = XFromScreenToCalc<true>(meanX);
                guessY = YFromScreenToCalc<true>(meanY);

                //wchar_t temps[256];
                //swprintf(temps, 256, L"Coords: %f %f", meanX, meanY);
                //::MessageBox(NULL, temps, L"", MB_OK);
            }
            else {
                break;
            }

            if (numAtLimit == antiRectWidthInt * antiRectHeightInt) {
                ::MessageBox(NULL, L"Flat screen! :(", L"", MB_OK);
                break;
            }
        }

        if constexpr (h == AutoZoomHeuristic::Max) {
            LONG targetX = -1;
            LONG targetY = -1;

            size_t maxiter = 0;
            for (auto y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
                for (auto x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                    auto curiter = m_CurIters.m_ItersArray[y][x];
                    if (curiter > maxiter) {
                        maxiter = curiter;
                    }
                }
            }

            for (auto y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
                for (auto x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++) {
                    auto curiter = m_CurIters.m_ItersArray[y][x];
                    if (curiter == maxiter) {
                        numAtLimit++;
                        if (targetX == -1 && targetY == -1) {
                            targetX = x;
                            targetY = y;
                        }
                    }

                    if (curiter >= GetNumIterations()) {
                        numAtMax++;
                    }
                }
            }

            guessX = XFromScreenToCalc<true>(targetX);
            guessY = YFromScreenToCalc<true>(targetY);

            if (numAtLimit == m_ScrnWidth * m_ScrnHeight * GetGpuAntialiasing() * GetGpuAntialiasing()) {
                ::MessageBox(NULL, L"Flat screen! :(", L"", MB_OK);
                break;
            }
        }

        HighPrecision newMinX = guessX - width / Divisor;
        HighPrecision newMinY = guessY - height / Divisor;
        HighPrecision newMaxX = guessX + width / Divisor;
        HighPrecision newMaxY = guessY + height / Divisor;

        HighPrecision centerX = (m_MinX + m_MaxX) / Two;
        HighPrecision centerY = (m_MinY + m_MaxY) / Two;

        HighPrecision defaultMinX = centerX - width / Divisor;
        HighPrecision defaultMinY = centerY - height / Divisor;
        HighPrecision defaultMaxX = centerX + width / Divisor;
        HighPrecision defaultMaxY = centerY + height / Divisor;

        const HighPrecision defaultWeight = 0;
        const HighPrecision newWeight = 1;
        HighPrecision weightedNewMinX = (newWeight * newMinX + defaultWeight * defaultMinX) / (newWeight + defaultWeight);
        HighPrecision weightedNewMinY = (newWeight * newMinY + defaultWeight * defaultMinY) / (newWeight + defaultWeight);
        HighPrecision weightedNewMaxX = (newWeight * newMaxX + defaultWeight * defaultMaxX) / (newWeight + defaultWeight);
        HighPrecision weightedNewMaxY = (newWeight * newMaxY + defaultWeight * defaultMaxY) / (newWeight + defaultWeight);

        //HighPrecision weightedNewMinX = guessX - width / Divisor;
        //HighPrecision weightedNewMinY = guessY - height / Divisor;
        //HighPrecision weightedNewMaxX = guessX + width / Divisor;
        //HighPrecision weightedNewMaxY = guessY + height / Divisor;

        RecenterViewCalc(
            weightedNewMinX,
            weightedNewMinY,
            weightedNewMaxX,
            weightedNewMaxY);

        CalcFractal(false);
        //DrawFractal(false);

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
void Fractal::View(size_t view)
{
    HighPrecision minX;
    HighPrecision minY;
    HighPrecision maxX;
    HighPrecision maxY;

    // Kludgy.  Resets at end of function.
    SetPrecision(50000, minX, minY, maxX, maxY);
    ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1);

    switch (view) {
    case 1:
        // Limits of 4x64 GPU
        minX = HighPrecision{ "-1.763399177066752695854220120818493394874764715075525070697085376173644156624573649873526729559691534754284706803085481158" };
        minY = HighPrecision{  "0.04289211262806512836473285627858318635734695759291867302112730624188941270466703058975670804976478935827994844038526618063053858" };
        maxX = HighPrecision{ "-1.763399177066752695854220120818493394874764715075525070697085355870272824868052108014289980411203646967653925705407102169" };
        maxY = HighPrecision{  "0.04289211262806512836473285627858318635734695759291867302112731461463330922125985949917962768812338316745717506303752530265831841" };
        SetNumIterations(196608);
        break;

    case 2:
        // Limits of 4x32 GPU
        minX = HighPrecision{ "-1.768969486867357972775564951275461551052751499509997185691881950786253743769635708375905775793656954725307354460920979983" };
        minY = HighPrecision{  "0.05699280690304670893115636892860647833175463644922652375916712719872599382335388157040896288795946562522749757591414246314107544" };
        maxX = HighPrecision{ "-1.768969486867357972775564950929487934553496494563941335911085292699250368065865432159590460057564657941788398574759610411" };
        maxY = HighPrecision{  "0.05699280690304670893115636907127975355127952306391692141273804706041783710937876987367435542131127321801128526034375237132904264" };
        SetNumIterations(196608);
        break;

    case 3:
        // Limit of 1x32 + Perturbation with no scaling
        minX = HighPrecision{ "-1.44656726997022737062295806977817803829443061688656117623800256312303751202920456713778693247098684334495241572095045" };
        minY = HighPrecision{  "7.64163245263840450044318279619820153508302789530826527979427966642829357717061175013838301474813332434725222956221212e-18" };
        maxX = HighPrecision{ "-1.44656726997022737062295806977817803829442603529959040638812674667522697557115287788808403561611427018141845213679032" };
        maxY = HighPrecision{  "7.641632452638404500443184705192772689187142818828186336651801203615669784193475322289855499574866715163283993737498e-18" };
        SetNumIterations(196608);
        break;

    case 4:
        minX = HighPrecision{ "-1.44656726997022737062295806977817803829442766062231034469821437680324515234695809677735314398112720680340773533658285" };
        minY = HighPrecision{  "7.64163245263840450044318315665495782554700906545628099403337428848294779227472963485579985013605880788857454558128871e-18" };
        maxX = HighPrecision{ "-1.44656726997022737062295806977817803829442766062231034469821437680324515234695809677735191376844462193327526231811052" };
        maxY = HighPrecision{  "7.64163245263840450044318315665495782554700906545628099403337428848294830486334737855168838056042228491124080530744489e-18" };
        SetNumIterations(196608);
        break;

    case 5:
        minX = HighPrecision{ "-0.548205748070475708458212567546733029376699278637323932787860368510369107357663992406257053055723741951365216836802745" };
        minY = HighPrecision{ "-0.577570838903603842805108982201850558675551730113738529364698265412779545002113555345006591372870167386914495276370477" };
        maxX = HighPrecision{ "-0.54820574807047570845821256754673302937669927060844097486102930067962289200412659019319306589187062772276993544341295" };
        maxY = HighPrecision{ "-0.577570838903603842805108982201850558675551726802772104952059640378694274662197291893029522164691495936927144187595881" };
        SetNumIterations(4718592);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 2);
        break;

    case 6:
        // Scale float with pixellation
        minX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030266459696312328585298881005139386870908363177752552421427177179281096147769415" };
        minY = HighPrecision{  "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033981012912620372948556514051537500942007730195548392246463251930450398477496176544" };
        maxX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030250153999905750113975818926710341658168707379760602146485062960529816708172165" };
        maxY = HighPrecision{  "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033987737087528857040088479840438460825120725713503099967399506797154756105787592431" };
        SetNumIterations(4718592);
        break;

    case 7:
        // Scaled float limit with circle
        minX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030252478540752851056038295732180048485849836719480635256962570788141443758414653" };
        minY = HighPrecision{  "0.0011175672388967686119452877936503680420978056943097961919136836510176758423423873900601464203086708258487998008460089102965219489403398329852555703126748646225896578863028142955526188647404920935367487584932956791" };
        maxX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030252478540752851056038295729331767859389716189380600126795460282679236765495185" };
        maxY = HighPrecision{  "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033983298525557031267486462261733835999658166408457695262521471675884626307237220407" };
        SetNumIterations(4718592);
        break;

    case 8:
        // Full BLA test 10^500 or so.
        minX = HighPrecision{ "-1.622553054509554409393783271485519336981516649058692523531044591770179784188916166903801363114695696477465352555971528798705448280840302524785407528510560382957296336747734660781986878861832656627910563988831196767541174987070125253828371867979558889300125126916356006345148898733695397073332691494528591084305228375267624747632537984863862572085039742052399126752593493400012297784541851439677708432244697466044124608245754997623602165370794000587151474975819819456195944759807441120696911861556241411268002822443914381727486024297400684811891998011624082906915471367511682755431368949865497918093491140476583266495952404825038512305399236173787176914358157915436929123396631868909691124127724870282812097822811900868267134047713861692077304303606674327671011614600160831982637839953075022976105199342055519124699940622577737366867995911632623421203318214991899522128043854908268532674686185750586601514622901972501486131318643684877689470464141043269863071514177452698750631448571933928477764304097795881563844836072430658397548935468503784369712597732268753575785634704765454312860644160891741730383872538137423932379916978843068382051765967108389949845267672531392701348446374503446641582470472389970507273561113864394504324475127799616327175518893129735240410917146495207044134363795295340971836770720248397522954940368114066327257623890598806915959596944154000211971821372129091151250639111325886353330990473664640985669101380578852175948040908577542590682028736944864095586894535838946146934684858500305538133386107808126063840356032468846860679224355311791807304868263772346293246639323226366331923103758639636260155222956587652848435936698081529518277516241743555175696966399377638808834127105029423149039661060794001264217559981749222317343868825318580784448831838551186024233174975961446762084960937499999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999998109" };
        minY = HighPrecision{ "0.001117567238896768611945287793650368042097805694309796191913683651017675842342387390060146420308670825848799800846008910296521948940339832985255570312674864622611835522284075672543854847255264731946382328371214009526561663174941608909925216179620747558791394265628274748675295346034899998815173442512409345653708758722585674020064853646630373608679959493334284553551589569947877966948566416572492467339583663949970046379704495147267378153033462305115439398337489620790688829995544121329470083542063031324851802574649918811841750913168721314686845259594117073877271321100948129855768148854311161320084129267363499289251825454183589219881333850776847012695765759653605564436023656201715593927689666722447079663020705803627665637015104492497741636468267388256490548881346893345366125490399643435876691530713541432028774019720460412518615867427220487428798187975218145189083789569256006253505424279834364602251632081060904568420885095239536114820931152263030281510960264774165352065681307019706534613000980870274180073599675792637642781955545684799042499349664484696664255967704566757636662588724423743582737881875921718030051382037889290489109692568753441015738409173208113123967308658979176663175143694315257820174553147123170016957035575950933366295114766019884833988769556724019671487704066470206279346884135106807605702141085672617612292451502032657624487846200365061848714912762429355703562415874601262268865127750867393241343569164011564916899299372768445538725622751921653729817821602157920626985597292441933163224073178294594360672242702797827361918819138484230729076661426426985362989832789724044998791055083335658038803274851664414571288437875273224563220948706102106988695167966092545926424587115422685794647558401757470800335648264079478256438335825103058562986733937005823236177093349397182464599609374999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999368" };
        maxX = HighPrecision{ "-1.622553054509554409393783271485519336981516649058692523531044591770179784188916166903801363114695696477465352555971528798705448280840302524785407528510560382957296336747734660781986878861832656627910563988831196767541174987070125253828371867979558889300125126916356006345148898733695397073332691494528591084305228375267624747632537984863862572085039742052399126752593493400012297784541851439677708432244697466044124608245754997623602165370794000587151474975819819456195944759807441120696911861556241411268002822443914381727486024297400684811891998011624082906915471367511682755431368949865497918093491140476583266495952404825038512305399236173787176914358157915436929101074572363325485941890852869740547207893495401847216464403190377283252284936695221620192489736296880064364736168387364407018924498548865083468021039191959297956680950735192211117294472572549916766627933448887817452943083286885583312398485175176799261059924202852929698149189055624038345817137105972343305894016069756366146900994722671059851259685904384807877457739632787467911510131153946289523428325072359141842353594673786565545080689796074914723731611730652207617489386127306710995798281376160344598476271882907175206737026029853260611423835086663823662230295980596315444662332633999288401087473357654870330977577589362848522147173440735610957887415860064397118429150006470992796414518180674956981690673824954254846780389614421277378116832539432506584941678464221661845802155038330559171689808309409001223409532423237399439602230938059938450063207955010358991806721046406987145343805044073027784664829615863680850374937599099095789048554468885707214000693385718677220557313945289575462766123240889414425105608211440697730397516835728294646134282185031089311311699258534641301626048472044700465551168161448813975766825024038553237915039062499999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999998107" };
        maxY = HighPrecision{ "0.001117567238896768611945287793650368042097805694309796191913683651017675842342387390060146420308670825848799800846008910296521948940339832985255570312674864622611835522284075672543854847255264731946382328371214009526561663174941608909925216179620747558791394265628274748675295346034899998815173442512409345653708758722585674020064853646630373608679959493334284553551589569947877966948566416572492467339583663949970046379704495147267378153033462305115439398337489620790688829995544121329470083542063031324851802574649918811841750913168721314686845259594117073877271321100948129855768148854311161320084129267363499289251825454183589219881333850776847012695765759653605573736881783528467753193053000281724117133569247062398777988899889662841499706014706016372541331507713879852825155309445733418035316861209556288978316282478143500096551357610725614057483872326044293314129792077777289475006632140252401733975684912603498348168568775217865832018883410276162470834740048255600659329223881004011060991907282879321090552836361563687680780220427483323293527090632178051813134981207196953681266541684913820792397357735300554966845235450747942390101292486119671868649365994478155987373346824092274515443661417944381090893731147361020889531680243992967746789389403706067885423681573530983486815289871675393650012417265434543050504019439701454624156569888621874101603997649966407799193057418611149232833039584855001941430933680923393544436450979507535811018411975678203452150800891864517137052035186135715348841397475928219859131636844030874374686820228572708751676865487969240162426098055037586559532217842753437863186592480806093936523929380404259520714584871920747692968032395327419735094412939817995275012199324226228671888756636304117844444282936821568138863483022553191437013266062994176763822906650602817535400390625000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000488" };
        SetNumIterations(536870912);
        break;

    case 9:
        // Current debug spot
        // Starts failing with scaled/BLA/32-bit but works with regular scaled 32-bit.
        minX = HighPrecision{ "-0.7483637942536300149503483214787034876591588411716933148186745513477244682570730516417622419808698292581482958014791704644565629000555509682153589709079382456689931580963129631758649894107825943703087439928414641651444362689519549254204846154926596121590912184282196146643043078678714060985887002716080586668490292983394730321504843621114015194797701155788204224604921" };
        minY = HighPrecision{ "-0.06744780927701335119415722995059170914204410549488939856276713700069965046725434093158528402364591969750087517693203035021583062328882637072820717521285184531690139123820284407899463906779357518574820657755838159003077331450979793198906608176322691865205598530852830385345767321526425688541101362130891363458886206558756783457806483014890069718507540521064981102965257" };
        maxX = HighPrecision{ "-0.74836379425363001495034832147870348765915884117169331481867433187401065255783016055089059668801973745851548723225680077302941610826313399595717357034234449928086303787522807162527281675222310279327102341439954670484722882754113271398864781791213972240089820570031805566451574870561115609430798868191313945870166698496668127704727678368440262539279975889176381385504076" };
        maxY = HighPrecision{ "-0.067447809277013351194157229950591709142044105494889398562767046493701193416045047963366441840970068619583043311554914622271582261091114020153740373090838894288008277652802673691394718968303669425227929361055051514588168357033981427169851116968741801469506295904472361894468267390743503211032923742242698263338165959443279104509270938940285203610910294845473266242724579" };
        SetNumIterations(4718592);
        break;

    case 10:
        // Deep 64-bit BLA minibrot, expensive.  Definitely want BLA.
        // For Kalles Fraktaler etc:
        // -0.7483637942536300149503483214787034876591588411716933148186744630883569643446693595154771110299201854417716821473485
        // -0.0674478092770133511941572299505917091420441054948893985627670832452723589863935688773429127930946937306872903076587
        // 6.103515625e+105
        // 2147483646

        minX = HighPrecision{ "-0.7483637942536300149503483214787034876591588411716933148186744630883569643446693595154771110299201854417718816507212864046192442728464126" };
        minY = HighPrecision{ "-0.06744780927701335119415722995059170914204410549488939856276708324527235898639356887734291279309469373068729038486837338182723971219233698" };
        maxX = HighPrecision{ "-0.7483637942536300149503483214787034876591588411716933148186744630883569643446693595154771110299201854417711330238558567565796397623653681" };
        maxY = HighPrecision{ "-0.06744780927701335119415722995059170914204410549488939856276708324527235898639356887734291279309469373068697845700777769514407116615856846" };
        SetNumIterations(2147483646);
        break;

    case 11:
        minX = HighPrecision{ "-0.9225837252220836442380682905062839979087904812701478975021202513277557678493257742585958941166199410512957938475766149929718155234080301303126877648228354119037152386584540549" };
        minY = HighPrecision{ "0.3101054229664775247118092309878450265997500181207805139942842842632061379116828179519893355354064444350287109494480557003773990753105163530757409662042930547521243602754174815" };
        maxX = HighPrecision{ "-0.9225837252220836442380682905062839979087904812701478975021202513277557678493257742585958941166199410512957938475766149929718155234080301303126877643767128233130766463441209464" };
        maxY = HighPrecision{ "0.3101054229664775247118092309878450265997500181207805139942842842632061379116828179519893355354064444350287109494480557003773990753105163530757409663901774666648904404063896101" };
        SetNumIterations(196608);
        break;

    case 12:
        // Low iterations centered on minibrot ~10^750
        // For Kalles Fraktaler etc:
        // -1.76910833040747728089230624355777848356006391186234006507787883404189297159859080844234601990886820204375798539711744042815186207867460953613415842432757944592767829349335811233344529101414256593211862449597012320484442182435300186806636928772790198745922046142684209058217860629186666350353426119156588068800663472823306396040025965062872111452567447461778249159612483961826938045120616222919425540724435314735525341255820663672868054494834304368733487681716181154843412847462148699796875704432080579735378707932099259943340496912407404541830241707067347835140775121643369435009399783998167487300390631920225320820254728695870927954776295601127590160537830979541161007773653115528776772709046068463437303643702179432428030338435000806480
        // -0.00902068805707261760036093598494762011230558467412386688972791981712911000027307032573465274857726041164197728426795343558667918574806252145218561840172051237592827928284001732546143539590034733383167886737270960837393773203586137673223357330612418801177955434724672703229251690060990669896274784498860242840497701851849033318045752945025236339275315448898727818174162698044355889583608736485127229946909629832632750839622459160046934443901886226621505107842448159935017841941045354364042323342824193503392819832038805610324279071648709936971740195644028490370367453551798296170098754444868309749391071986751540687669916301587186977458689535397730634082592706549618026786866330689857703728968664890931318074163369981746242981795864371396
        // 4.0517578124997939E712
        // 2880000 iterations

        minX = HighPrecision{ "-1.769108330407477280892306243557778483560063911862340065077878834041892971598590808442346019908868202043757985397117440428151862078674609536134158424327579445927678293493358112333445291014142565932118624495970123204844421824353001868066369287727901987459220461426842090582178606291866663503534261191565880688006634728233063960400259650628721114525674474617782491596124839618269380451206162229194255407244353147355253412558206636728680544948343043687334876817161811548434128474621486997968757044320805797353787079320992599433404969124074045418302417070673478351407751216433694350093997839981674873003906319202253208202547286958709279547762956011275901605378309795411610077736531155287767727090460684634373036437021849939558081154827035966322541319864476469606578987384818141802086140871082590860960091777689102849781889131421151893374373923872281234460683461953146946783351680424907509438258067061062136495370430128426260576349247467954001002250166013425144708446174956227820466386363571779749724416127365" };
        minY = HighPrecision{ "-0.009020688057072617600360935984947620112305584674123866889727919817129110000273070325734652748577260411641977284267953435586679185748062521452185618401720512375928279282840017325461435395900347333831678867372709608373937732035861376732233573306124188011779554347246727032292516900609906698962747844988602428404977018518490333180457529450252363392753154488987278181741626980443558895836087364851272299469096298326327508396224591600469344439018862266215051078424481599350178419410453543640423233428241935033928198320388056103242790716487099369717401956440284903703674535517982961700987544448683097493910719867515406876699163015871869774586895353977306340825927065496180267868663306898577037289686648909313180741633720660506874259671345628579528566082199707266953582331381477204845958749313226525982433412485094171338066030864231223182859124950336577946409926601514095926041462597061333401178051740876104003994845997683393109587807745687517577651571044175605197417318412485990475703151931020039994351872796158" };
        maxX = HighPrecision{ "-1.769108330407477280892306243557778483560063911862340065077878834041892971598590808442346019908868202043757985397117440428151862078674609536134158424327579445927678293493358112333445291014142565932118624495970123204844421824353001868066369287727901987459220461426842090582178606291866663503534261191565880688006634728233063960400259650628721114525674474617782491596124839618269380451206162229194255407244353147355253412558206636728680544948343043687334876817161811548434128474621486997968757044320805797353787079320992599433404969124074045418302417070673478351407751216433694350093997839981674873003906319202253208202547286958709279547762956011275901605378309795411610077736531155287767727090460684634373036437021723023848332310708579010440266742180377555833452293488480928863794034113380717085247450477413409955521865484776069659968888982957089040699836537328435625787365406852114641414906701003244203291394842947052469974908497764264157006872226432364177441648413430211646947924195417953122886076275359" };
        maxY = HighPrecision{ "-0.00902068805707261760036093598494762011230558467412386688972791981712911000027307032573465274857726041164197728426795343558667918574806252145218561840172051237592827928284001732546143539590034733383167886737270960837393773203586137673223357330612418801177955434724672703229251690060990669896274784498860242840497701851849033318045752945025236339275315448898727818174162698044355889583608736485127229946909629832632750839622459160046934443901886226621505107842448159935017841941045354364042323342824193503392819832038805610324279071648709936971740195644028490370367453551798296170098754444868309749391071986751540687669916301587186977458689535397730634082592706549618026786866330689857703728968664890931318074163366832275878265868650451246018553457133835038044908002155931069409171115668275811690051522674705922123923151551111756344691412832948607201984870794725223952370924275014884952586106657747374540046661329854393906303354265277029964461533782764154419442516701718236383062903641742731769900290969182" };
        SetNumIterations(2880000);
        break;

    case 13:
        minX = HighPrecision{ "-0.19811160021967344480755228509879453543772824147087543575209282164460405796661858911920463414713355404730672339934458650212316679185224457464505947919951877044752574382704564824820621527934448404016493364320547229306519611171511715712176900717953583187315259259238053585661497776916800472941069492198349043947925513489896173070559294315034318176228198397942648722866757000829781024389799652212575542306646359027444069432370725938261915645996647660973657175607225853790674975618571917570230527957204526460701463173918515293501869448459185064344576999456359805385079383557776563004844468354008330736481418435296838012008500564694120819246555304168549021993810386844438674235991162162162552693868541735221730120226788397508143295115228891571440322071040035893469145432310347153135535181217777068958043805111980460858226841271718932841952098915221391513754130304454670933509459443822133837141872224361748538049080126666903834742681230893022577974880489421539115562656254310965025514910034337116229478322132145499334347381872195403786902888054606427838397631002687320687282745697533566217149085166308084080267457799899979196864712458846265368806530544056130146149848874818466622188802941784184464474834009310749590114413519468573833491862914876137009029121673459869433939351561841919154142711235779865793985455674208174102216159693143718238023685092327323727642289200834838695134174687167872481469723044215318220145759947793422540264128102143705824475109940031148462227253349868622772578244300313163175772202467420449945317245380988600167012464669497991600968524644338228807842178813932899000672825923461538183198762686454877469809802713434534445048058703923395789350323928962196539530465247300488890944356613861849247250669669559167977740546609272659106317907818930851023789439535345906216581467812819417753195134972084764729268058393107527899173921117959376762011940545670420410280386506477920963401216567286546226929765168985959828311037026215860190887116908979824655696319732733074345119111792558149869009781048835786881253533508499848969119588046186491856238391795092327966170799858991126972974493625729277715039166305873718799816660176584115266744430167947681149010934163805291011003764981720526738363243360119409752650474643574888828897585218303941928189836689849016576450697376083722195146156558426480099577734618531305533296673253150537970190827258181493819479687598022221974333798996039335847521854042507320903455652807257258903526332243228243511559615460310019904448209443137031260494818474246689924463711082405813737358021631393073732858482778024763436565852437154695057129216315148917487677222664834156478552341031113623543667983807907350282593666016978122983038822939099114405083775127145632739816795460185084064620863543065628114440156581756066752507154968504863337171763826105224281791178932927020989813212589292999118262648248016038897343092140225125318900543695471764746419644540418431469293811823484058015965407032649980755382536527874443438306028410220105859974292295915287154331992026072282920537729880888758627262656483037180785834638257692175151722860250387183318132919805711034352398697990025859370576693882609552423649944386651640082416989731006507184708966776441487790205285637309215844243942176437791160953120662718254932396534245082366493163459307826996304189885127045422287366232458258891328003658403230692471753618917578202476063310986807854856388919255198440466824236358127853342379751762858579132091099568789798071863727384338291022345968896624433628414878127462242917153761730086827314079022139776911122868908083539140392733985449628490366370036710811514650010650855344183692439171337895053515516285989632712351679380789905521028560229581602396215763270994638561110338263087410310988528756708648087968343640166962960902101864716477763567764389425387470275196858369360612655522422476611470371668766482595146642236202584459989166866788012960676202412427068787158932337711631738718533105212644334171450366360289877218879582231818391284758216388078085283224596396638801021564327747957457489174455578937847915426697043363300674436875242073120078951271096936391635156641517793220820610146161890592653434830315231584175128258540980673218650502076325140464746203800202739849688794463451163766147299235653628643802292756097416087289430205555352522550544261229970628150141473357154166915293108953343600228101516606985120428301300397237003185516353950264194714068535510113957115217817241050181960131736113134261363991" };
        minY = HighPrecision{ "-1.0995133542657857115808634031346189677870097136303572938715549689295389425632581503138812246977639087726709908677423161184335187629274263873188431553184305117753521329343866123179772001228771002107363550330517703579998140593409066357319311393026292408059480701983901970488499896735828553853802834301907197608149892713585833439577168227770555722928444712272129592949245167090691094148721898008727267507465019406653040957136916706118511368949128731646363480934893765440233859658466914730889839272611903774672315732641021754542818153354204695278348344511388429168400821488094355656936699975929482299923845850567674425122940359760347969736285151356337932113169624173522678013434408139130578132283466066872079950533561149015822217125258927391052105262576616632945045780120222419420734535273199171299980147596457095869899477339356795576617878510010827998630177588173538858637837731456007658310311702877324369659541180907536879135103018762800573950834307965599889878790750643156769516655597480390761553972391015283726067469563740287759144203858532782393629326958947345346075862898677300582937187422733316888028572486022198146869394141932576445093269938696609109004619913856912121251619407075780993615588186308812136419907177826036166069223642109761644270244408513600447075646747557841282319975447147019915850615072569108281032551376574856014145050239628405472665778715940606084639798101936408520457664615742008405118580595605204677748331709163054076781812915678506853351239918271425357018823354260518424833289209204961468610476161329798441098428030085154729370441648068624166487012202872274944187332078710294787873286766095146067629970942602556871762137094158332888636516293149557435983842777651293332443747454243255593579551968077852240199032487370427890674554584876808156811296145656826603260999681667949054686675903392053812108256116181160382494613997153199138603568393825186775292110275886829568971119775984919184077354063706566903419073631479095854702868276191875387674381277531094444441512800170000808094582407023055940003011540456204250948535043947179564116845971853224078045823536435698464891251304264471262096147132236334930669996592059889573584105384098186560250051739848054078845784511077797752691726900716551298408118617259364741999896084977307898465688063329817625214380750549330802870390053468442530064008379043658676719681384415954836464013723721328656584564245017093489835104348233380012210598845075057455320784942775443346399951069614279355554240011829384174308361963096028224335340259715137162528235751923616803492153533687263878085937736476622324705730656430032258822237858207894091176008620360867559502836300916391129134493393238616845392327605929904538512385164829453698817223777011835227488504934156151586372040638703411082219461948002087239032665791095552179479335250292539349234839433519820969519859430808755567356764565438803961064827434555915497113778977079605441377210893092495399432355608991036199477513888066828156326795583892133456614592830566442544687374076128163388320475699482715893915441611501227011705476586782690440277593692800910282999955406474787278970423592875378485539570549625585643603455801693407562571617246033401774796591129123039627392465173435466744333189312382957493987321525062480034592529077515387584484566100681565181620963805309260575185935715773449071166360012818595806823394732429772842032564876123502991214944831003127194370200560697563299786497558018703927056919915530941580817368155247209052873519988555883733787288116834882542554101183994472245225809915798186545862713186429005273179903890550958593295637174877467946982223916728461729036954094229402778026202138020491005859547750715441536659291389412423111376387691879474734129319825597969797462861380116991153974509207280337951873246607697818904391672151044085118487243626204300544957689214520254379117006819489725079260183139289555019446332600192640717145467855364738585576815881931284199917707246685077927850230029444392740341710214532917701299608882174119380311042117100571290606665230293804110116844241348879899086241122726289650136328011642911385610933906932369713687513984018877822438694490106404039018757457727445194534405647328297464259284472027928115425849951872169413282317316593183959788152388433164997240878468374424229310620424175449750107285234038858123382311297534602560686359827499341195321256108356232984988411254969950900447210214059234170765920468549231098085902093319040057298242654348820124671419516397663044813" };
        maxX = HighPrecision{ "-0.1981116002196734448075522850987945354377282414708754357520928216446040579666185891192046341471335540473067233993445865021231667918522445746450594791995187704475257438270456482482062152793444840401649336432054722930651961117151171571217690071795358318731525925923805358566149777691680047294106949219834904394792551348989617307055929431503431817622819839794264872286675700082978102438979965221257554230664635902744406943237072593826191564599664766097365717560722585379067497561857191757023052795720452646070146317391851529350186944845918506434457699945635980538507938355777656300484446835400833073648141843529683801200850056469412081924655530416854902199381038684443867423599116216216255269386854173522173012022678839750814329511522889157144032207104003589346914543231034715313553518121777706895804380511198046085822684127171893284195209891522139151375413030445467093350945944382213383714187222436174853804908012666690383474268123089302257797488048942153911556265625431096502551491003433711622947832213214549933434738187219540378690288805460642783839763100268732068728274569753356621714908516630808408026745779989997919686471245884626536880653054405613014614984887481846662218880294178418446447483400931074959011441351946857383349186291487613700902912167345986943393935156184191915414271123577986579398545567420817410221615969314371823802368509232732372764228920083483869513417468716787248146972304421531822014575994779342254026412810214370582447510994003114846222725334986862277257824430031316317577220246742044994531724538098860016701246466949799160096852464433822880784217881393289900067282592346153818319876268645487746980980271343453444504805870392339578935032392896219653953046524730048889094435661386184924725066966955916797774054660927265910631790781893085102378943953534590621658146781281941775319513497208476472926805839310752789917392111795937676201194054567042041028038650647792096340121656728654622692976516898595982831103702621586019088711690897982465569631973273307434511911179255814986900978104883578688125353350849984896911958804618649185623839179509232796617079985899112697297449362572927771503916630587371879981666017658411526674443016794768114901093416380529101100376498172052673836324336011940975265047464357488882889758521830394192818983668984901657645069737608372219514615655842648009957773461853130553329667325315053797019082725818149381947968759802222197433379899603933584752185404250732090345565280725725890352633224322824351155961546031001990444820944313703126049481847424668992446371108240581373735802163139307373285848277802476343656585243715469505712921631514891748767722266483415647855234103111362354366798380790735028259366601697812298303882293909911440508377512714563273981679546018508406462086354306562811444015658175606675250715496850486333717176382610522428179117893292702098981321258929299911826264824801603889734309214022512531890054369547176474641964454041843146929381182348405801596540703264998075538253652787444343830602841022010585997429229591528715433199202607228292053772988088875862726265648303718078583463825769217515172286025038718331813291980571103435239869799002585937057669388260955242364994438665164008241698973100650718470896677644148779020528563730921584424394217643779116095312066271825493239653424508236649316345930782699630418988512704542228736623245825889132800365840323069247175361891757820247606331098680785485638891925519844046682423635812785334237975176285857913209109956878979807186372738433829102234596889662443362841487812746224291715376173008682731407902213977691112286890808353914039273398544962849036637003671081151465001065085534418369243917133789505351551628598963271235167938078990552102856022958160239621576327099463856111033826308741031098852875670864808796834364016696296090210186471647776356776438942538747027519685836936061265552242247661147037166876648259514664223620258445998916686678801296067620241242706878715893233771163173871853310521264433417145036636028987721887958223181839128475821638807808528322459639663880102156432774795745748917445557893784791542669704336330067443687524207312007895127109693639163515664151779322082061014616189059265343483031523158417512825854098067321865050207632514046474620380020273984968879446345116376614729923565362864380229275609741608728943020555535252255054426122997062815014147335715416691529310895334360022810151660698512042830130039723700318551635395026419471406853551011395711521781724104902645634823828075018859827" };
        maxY = HighPrecision{ "-1.0995133542657857115808634031346189677870097136303572938715549689295389425632581503138812246977639087726709908677423161184335187629274263873188431553184305117753521329343866123179772001228771002107363550330517703579998140593409066357319311393026292408059480701983901970488499896735828553853802834301907197608149892713585833439577168227770555722928444712272129592949245167090691094148721898008727267507465019406653040957136916706118511368949128731646363480934893765440233859658466914730889839272611903774672315732641021754542818153354204695278348344511388429168400821488094355656936699975929482299923845850567674425122940359760347969736285151356337932113169624173522678013434408139130578132283466066872079950533561149015822217125258927391052105262576616632945045780120222419420734535273199171299980147596457095869899477339356795576617878510010827998630177588173538858637837731456007658310311702877324369659541180907536879135103018762800573950834307965599889878790750643156769516655597480390761553972391015283726067469563740287759144203858532782393629326958947345346075862898677300582937187422733316888028572486022198146869394141932576445093269938696609109004619913856912121251619407075780993615588186308812136419907177826036166069223642109761644270244408513600447075646747557841282319975447147019915850615072569108281032551376574856014145050239628405472665778715940606084639798101936408520457664615742008405118580595605204677748331709163054076781812915678506853351239918271425357018823354260518424833289209204961468610476161329798441098428030085154729370441648068624166487012202872274944187332078710294787873286766095146067629970942602556871762137094158332888636516293149557435983842777651293332443747454243255593579551968077852240199032487370427890674554584876808156811296145656826603260999681667949054686675903392053812108256116181160382494613997153199138603568393825186775292110275886829568971119775984919184077354063706566903419073631479095854702868276191875387674381277531094444441512800170000808094582407023055940003011540456204250948535043947179564116845971853224078045823536435698464891251304264471262096147132236334930669996592059889573584105384098186560250051739848054078845784511077797752691726900716551298408118617259364741999896084977307898465688063329817625214380750549330802870390053468442530064008379043658676719681384415954836464013723721328656584564245017093489835104348233380012210598845075057455320784942775443346399951069614279355554240011829384174308361963096028224335340259715137162528235751923616803492153533687263878085937736476622324705730656430032258822237858207894091176008620360867559502836300916391129134493393238616845392327605929904538512385164829453698817223777011835227488504934156151586372040638703411082219461948002087239032665791095552179479335250292539349234839433519820969519859430808755567356764565438803961064827434555915497113778977079605441377210893092495399432355608991036199477513888066828156326795583892133456614592830566442544687374076128163388320475699482715893915441611501227011705476586782690440277593692800910282999955406474787278970423592875378485539570549625585643603455801693407562571617246033401774796591129123039627392465173435466744333189312382957493987321525062480034592529077515387584484566100681565181620963805309260575185935715773449071166360012818595806823394732429772842032564876123502991214944831003127194370200560697563299786497558018703927056919915530941580817368155247209052873519988555883733787288116834882542554101183994472245225809915798186545862713186429005273179903890550958593295637174877467946982223916728461729036954094229402778026202138020491005859547750715441536659291389412423111376387691879474734129319825597969797462861380116991153974509207280337951873246607697818904391672151044085118487243626204300544957689214520254379117006819489725079260183139289555019446332600192640717145467855364738585576815881931284199917707246685077927850230029444392740341710214532917701299608882174119380311042117100571290606665230293804110116844241348879899086241122726289650136328011642911385610933906932369713687513984018877822438694490106404039018757457727445194534405647328297464259284472027928115425849951872169413282317316593183959788152388433164997240878468374424229310620424175449750107285234038858123382311297534602560686359827499341195321256108356232984988411254969950900447210214059234170765920468549231098085902093319040057298242654348771978680440440048326679578" };
        SetNumIterations(113246208);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 16);
        break;

    case 0:
    default:
        minX = HighPrecision{ "-2.5" };
        minY = HighPrecision{ "-1.5" };
        maxX = HighPrecision{ "1.5" };
        maxY = HighPrecision{ "1.5" };
        ResetNumIterations();
        break;
    }

    RecenterViewCalc(minX, minY, maxX, maxY);
}

//////////////////////////////////////////////////////////////////////////////
// This function will "square" the current view, and it will take into account the
// aspect ratio of the monitor!  Wowee!  Thus, the view really is "square" even if
// your resolution is 1024x768 or some other "nonsquare" resolution.
//////////////////////////////////////////////////////////////////////////////
void Fractal::SquareCurrentView(void)
{
    HighPrecision ratio = (HighPrecision)m_ScrnWidth / (HighPrecision)m_ScrnHeight;
    HighPrecision mwidth = (m_MaxX - m_MinX) / ratio;
    HighPrecision height = m_MaxY - m_MinY;

    if (height > mwidth)
    {
        m_MinX -= ratio * (height - mwidth) / 2.0;
        m_MaxX += ratio * (height - mwidth) / 2.0;
    }
    else if (height < mwidth)
    {
        m_MinY -= (mwidth - height) / 2.0;
        m_MaxY += (mwidth - height) / 2.0;
    }

    m_ChangedWindow = true;

    CleanupThreads();
    OptimizeMemory();
}

// Used to gradually approach a given target.
// Used for creating sequences of still images.
// The images can then be made into a movie!
void Fractal::ApproachTarget(void)
{
    HighPrecision MinX = m_SetupData.m_L1MinX;
    HighPrecision MinY = m_SetupData.m_L1MinY;
    HighPrecision MaxX = m_SetupData.m_L1MaxX;
    HighPrecision MaxY = m_SetupData.m_L1MaxY;
    HighPrecision targetIters = m_SetupData.m_L1Iterations;
    int numFrames = m_SetupData.m_L1NumFrames;

    HighPrecision deltaXMin;
    HighPrecision deltaYMin;
    HighPrecision deltaXMax;
    HighPrecision deltaYMax;

    SaveCurPos();

    size_t baseIters = m_NumIterations;
    HighPrecision incIters = (HighPrecision)((HighPrecision)targetIters - (HighPrecision)baseIters) / (HighPrecision)numFrames;
    for (int i = 0; i < numFrames; i++)
    {
        deltaXMin = (MinX - m_MinX) / 75.0;
        deltaYMin = (MinY - m_MinY) / 75.0;
        deltaXMax = (MaxX - m_MaxX) / 75.0;
        deltaYMax = (MaxY - m_MaxY) / 75.0;

        m_MinX += deltaXMin;
        m_MinY += deltaYMin;
        m_MaxX += deltaXMax;
        m_MaxY += deltaYMax;

        {
            HighPrecision result = ((HighPrecision)incIters * (HighPrecision)i);
            m_NumIterations = baseIters + Convert<HighPrecision, unsigned long>(result);
        }

        m_ChangedWindow = true;
        m_ChangedIterations = true;

        wchar_t temp[256], temp2[256];
        wsprintf(temp, L"output%04d", i);
        wcscpy(temp2, m_SetupData.m_SaveDir);
        wcscpy(temp2, L"\\");
        wcscpy(temp2, temp);

        if (FileExists(temp2) == false)
        { // Create a placeholder file
            FILE *file = _wfopen(temp2, L"w+");
            if (file == NULL) // Fail silently.
            {
                break;
            }

            fwrite(temp2, sizeof(char), 256, file);
            fclose(file);

            // Render the fractal.
            // Draw the progress on the screen as necessary
            if (m_SetupData.m_AZDrawProgress == 'y')
            {
                CalcFractal(false);
            }
            else
            {
                CalcFractal(true);
            }

            // Stop _before_ saving the image, otherwise we will get a
            // corrupt render mixed in with good ones.
            // It's corrupt because it's incomplete!
            if (m_StopCalculating == true)
            {
                _wunlink(temp2); // Delete placeholder
                break;
            }

            // Save images as necessary.  This will usually be the case,
            // as simply watching it render one frame after another is not
            // particularly exciting.
            if (m_SetupData.m_AZSaveImages == 'y')
            {
                int ret = 0;
                ret = SaveCurrentFractal(temp2);
                if (ret == 0)
                {
                    break;
                }
            }
        }
    }
}

bool Fractal::FileExists(const wchar_t *filename)
{
    _wfinddata_t fileinfo;
    intptr_t handle = _wfindfirst(filename, &fileinfo);
    if (handle == -1)
    {
        return false;
    }

    _findclose(handle);

    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Returns to the previous view.
//////////////////////////////////////////////////////////////////////////////
bool Fractal::Back(void)
{
    if (!m_PrevMinX.empty())
    {
        m_MinX = m_PrevMinX.back();
        m_MinY = m_PrevMinY.back();
        m_MaxX = m_PrevMaxX.back();
        m_MaxY = m_PrevMaxY.back();

        SetPrecision();

        m_PerturbationGuessCalcX = 0;
        m_PerturbationGuessCalcY = 0;

        m_PrevMinX.pop_back();
        m_PrevMinY.pop_back();
        m_PrevMaxX.pop_back();
        m_PrevMaxY.pop_back();

        m_ChangedWindow = true;
        return true;
    }

    return false;
}

// Used for screen saver program.
// It is supposed to find an interesting spot to zoom in on.
// It works rather well as is...
void Fractal::FindInterestingLocation(RECT *rect)
{
    int x, y;
    int xMin = 0, yMin = 0, xMax = 0, yMax = 0;
    int xMinFinal = 0, yMinFinal = 0, xMaxFinal = 0, yMaxFinal = 0;

    HighPrecision ratio = (HighPrecision)m_ScrnWidth / (HighPrecision)m_ScrnHeight;
    int sizeY = rand() % (m_ScrnHeight / 3) + 1;

    HighPrecision sizeXHigh = ((HighPrecision)sizeY * ratio);
    int sizeX = Convert<HighPrecision, int>(sizeXHigh);

    unsigned int numberAtMax, numberAtMaxFinal = 0;
    HighPrecision percentAtMax, percentAtMaxFinal = 10000000.0;
    HighPrecision desiredPercent = (rand() % 75) / 100.0 + .10;

    int i;
    for (i = 0; i < 1000; i++)
    {
        x = rand() % (m_ScrnWidth - sizeX * 2) + sizeX;
        y = rand() % (m_ScrnHeight - sizeY * 2) + sizeY;

        xMin = x - sizeX;
        yMin = y - sizeY;
        xMax = x + sizeX;
        yMax = y + sizeY;

        numberAtMax = 0;
        for (y = yMin; y <= yMax; y++)
        {
            for (x = xMin; x <= xMax; x++)
            {
                if (numberAtMax > 4200000000)
                {
                    break;
                }

                numberAtMax += m_CurIters.m_ItersArray[y][x];
            }
        }

        percentAtMax = (HighPrecision)numberAtMax / ((HighPrecision)sizeX * (HighPrecision)sizeY * (HighPrecision)m_NumIterations);

        if (fabs(percentAtMax - desiredPercent) < fabs(percentAtMaxFinal - desiredPercent))
        {
            numberAtMaxFinal = numberAtMax;
            percentAtMaxFinal = percentAtMax;
            xMinFinal = xMin;
            yMinFinal = yMin;
            xMaxFinal = xMax;
            yMaxFinal = yMax;
        }

        if (m_StopCalculating == true)
        {
            break;
        }
    }

    rect->left = xMinFinal;
    rect->top = yMinFinal;
    rect->right = xMaxFinal;
    rect->bottom = yMaxFinal;
}

// Let's us know if we've zoomed in too far.
bool Fractal::IsValidLocation(void)
{
    return ((m_MaxX - m_MinX) / m_ScrnWidth > pow((HighPrecision)10, (HighPrecision)-16)) &&
        ((m_MaxY - m_MinY) / m_ScrnHeight > pow((HighPrecision)10, (HighPrecision)-16));
}

//////////////////////////////////////////////////////////////////////////////
// Call this before modifying the calculator window.
// This allows the user to use the "Back" function to return to
// the previous coordinates.
//////////////////////////////////////////////////////////////////////////////
void Fractal::SaveCurPos(void)
{
    m_PrevMinX.push_back(m_MinX);
    m_PrevMinY.push_back(m_MinY);
    m_PrevMaxX.push_back(m_MaxX);
    m_PrevMaxY.push_back(m_MaxY);
}

///////////////////////////////////////////////////////////////////////////////
// Functions for dealing with the number of iterations
///////////////////////////////////////////////////////////////////////////////
void Fractal::SetNumIterations(size_t num)
{
    if (num <= MAXITERS)
    {
        m_NumIterations = num;
    }
    else
    {
        m_NumIterations = MAXITERS;
    }

    m_ChangedIterations = true;
}

size_t Fractal::GetNumIterations(void) const
{
    return m_NumIterations;
}

void Fractal::ResetNumIterations(void)
{
    m_NumIterations = DefaultIterations;
    m_ChangedIterations = true;
}

///////////////////////////////////////////////////////////////////////////////
// Functions for drawing the fractal
///////////////////////////////////////////////////////////////////////////////
void Fractal::CalcFractal(bool MemoryOnly)
{
    SetCursor(LoadCursor(NULL, IDC_WAIT));

    // Reset the flag should it be set.
    m_StopCalculating = false;

    // Do nothing if nothing has changed
    if (ChangedIsDirty() == false)
    {
        DrawFractal(MemoryOnly);
        return;
    }

    // Clear the screen if we're drawing.
    if (MemoryOnly == false)
    {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // Create a network connection as necessary.
    if (m_NetworkRender == 'a')
    {
        ClientCreateSubConnections();
    }

    // Draw the local fractal.
    switch(GetRenderAlgorithm()) {
    case RenderAlgorithm::CpuHigh:
        CalcCpuHDR<HighPrecision, double>(MemoryOnly);
        break;
    case RenderAlgorithm::CpuHDR32:
        CalcCpuHDR<HDRFloat<float>, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu32PerturbedBLAHDR:
        CalcCpuPerturbationFractalBLA<HDRFloat<float>, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64:
        CalcCpuHDR<double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::CpuHDR64:
        CalcCpuHDR<HDRFloat<double>, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLA:
        CalcCpuPerturbationFractalBLA<double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
        CalcCpuPerturbationFractalBLA<HDRFloat<double>, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64:
    case RenderAlgorithm::Gpu2x64:
    case RenderAlgorithm::Gpu4x64:
    case RenderAlgorithm::Gpu1x32:
    case RenderAlgorithm::Gpu2x32:
    case RenderAlgorithm::Gpu4x32:
        CalcGpuFractal(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32Perturbed:
        CalcGpuPerturbationFractalBLA<float, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
        CalcGpuPerturbationFractalScaledBLA<double, double, float, float>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
        CalcGpuPerturbationFractalScaledBLA<HDRFloat<float>, float, float, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64Perturbed:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
        CalcGpuPerturbationFractalBLA<double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x32Perturbed:
        //CalcGpuPerturbationFractalBLA<dblflt, dblflt>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        //CalcGpuPerturbationFractalBLA<double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
        CalcGpuPerturbationFractalBLA<HDRFloat<float>, float>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
        CalcGpuPerturbationFractalBLA<HDRFloat<double>, double>(MemoryOnly);
        break;
    default:
        break;
    }

    // Read in data from the network and draw it.
    if (m_NetworkRender == 'a')
    {
        CalcNetworkFractal(MemoryOnly);
    }

    // Shutdown the network connection
    if (m_NetworkRender == 'a')
    {
        ClientShutdownSubConnections();
    }

    // We are all updated now.
    ChangedMakeClean();

    // Reset the cursor
    SetCursor(LoadCursor(NULL, IDC_ARROW));
}

void Fractal::UsePaletteType(Palette type)
{
    m_WhichPalette = type;
}

int Fractal::GetPaletteDepthFromIndex(size_t index)
{
    switch (index) {
    case 0: return 6;
    case 1: return 8;
    case 2: return 12;
    case 3: return 16;
    case 4: return 20;
    default: return 6;
    }
}

void Fractal::UsePalette(int depth)
{
    switch (depth) {
    case 6:
        m_PaletteDepthIndex = 0;
        break;
    case 8:
        m_PaletteDepthIndex = 1;
        break;
    case 12:
        m_PaletteDepthIndex = 2;
        break;
    case 16:
        m_PaletteDepthIndex = 3;
        break;
    case 20:
        m_PaletteDepthIndex = 4;
        break;
    default:
        m_PaletteDepthIndex = 0;
        break;
    }
}

void Fractal::ResetFractalPalette(void)
{
    m_PaletteRotate = 0;
}

void Fractal::RotateFractalPalette(int delta)
{
    m_PaletteRotate += delta;
    if (m_PaletteRotate >= MAXITERS)
    {
        m_PaletteRotate = 0;
    }
}

void Fractal::CreateNewFractalPalette(void)
{
    //int index, i;

    //srand((unsigned int)time(NULL));
    //index = 0;
    //index = PalTransition(index, 256, MAXITERS, (rand() % 2) * 255, (rand() % 2) * 255, (rand() % 2) * 255);
    //if (index != -1)
    //{
    //    for (i = 0; i < 1000; i++)
    //    {
    //        index = PalTransition(index, 256, MAXITERS, (rand() % 4) * 85, (rand() % 4) * 85, (rand() % 4) * 85);
    //        if (index == -1)
    //        {
    //            break;
    //        }
    //    }
    //}

    //DrawFractal(false);
}

template<class T>
void Fractal::DrawPerturbationResults(bool LeaveScreen) {
    if (!LeaveScreen) {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBegin(GL_POINTS);

    auto& results = GetPerturbationResults<T>();
    for (size_t i = 0; i < results.size(); i++)
    {
        if (IsPerturbationResultUsefulHere<T>(i)) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            GLint scrnX = Convert<HighPrecision, GLint>(XFromCalcToScreen(results[i].hiX));
            GLint scrnY = static_cast<GLint>(m_ScrnHeight) - Convert<HighPrecision, GLint>(YFromCalcToScreen(results[i].hiY));

            // Coordinates are weird in OGL mode.
            glVertex2i(scrnX, scrnY);
        }
    }

    glEnd();
    glFlush();
}

void Fractal::ClearPerturbationResults() {
    m_PerturbationResultsDouble.clear();
    m_PerturbationResultsFloat.clear();
    m_PerturbationResultsHDRDouble.clear();
    m_PerturbationResultsHDRFloat.clear();

    m_PerturbationGuessCalcX = 0;
    m_PerturbationGuessCalcY = 0;
}

//////////////////////////////////////////////////////////////////////////////
// Redraws the fractal using OpenGL calls.
// Note that coordinates here are weird, so we have to make a few tweaks to
// get the image oriented right side up. In particular, the line which reads:
//       glVertex2i (px, m_ScrnHeight - py);
//////////////////////////////////////////////////////////////////////////////
void Fractal::DrawFractal(bool MemoryOnly)
{

    if (MemoryOnly) {
        return;
    }

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

    for (auto& thread : m_DrawThreads) {
        std::unique_lock lk(thread->m_DrawThreadMutex);
        thread->m_DrawThreadCV.wait(lk, [&] {
            return thread->m_DrawThreadProcessed;
            }
        );
    }

    GLuint texid;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap levels of a texture.
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // Change m_DrawOutBytes size if GL_RGBA is changed
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA16, (GLsizei)m_ScrnWidth, (GLsizei)m_ScrnHeight, 0,
        GL_RGBA, GL_UNSIGNED_SHORT, m_DrawOutBytes.get());

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, (GLint)m_ScrnHeight);
    glTexCoord2i(0, 1); glVertex2i(0, 0);
    glTexCoord2i(1, 1); glVertex2i((GLint)m_ScrnWidth, 0);
    glTexCoord2i(1, 0); glVertex2i((GLint)m_ScrnWidth, (GLint)m_ScrnHeight);
    glEnd();
    glFlush();

    glDeleteTextures(1, &texid);

    DrawPerturbationResults<double>(true);
    DrawPerturbationResults<float>(true);
    DrawPerturbationResults<HDRFloat<double>>(true);
    DrawPerturbationResults<HDRFloat<float>>(true);

    //while (GetMessage(&msg, NULL, 0, 0) > 0)
    //{
    //    TranslateMessage(&msg);
    //    DispatchMessage(&msg);
    //}
}

void Fractal::DrawFractalThread(size_t index, Fractal* fractal) {
    DrawThreadSync &sync = *fractal->m_DrawThreads[index].get();

    constexpr size_t BytesPerPixel = 4;

    for (;;) {
        // Wait until main() sends data
        std::unique_lock lk(sync.m_DrawThreadMutex);
        
        sync.m_DrawThreadCV.wait(lk, [&] {
            return sync.m_DrawThreadReady;
            }
        );

        if (sync.m_TimeToExit) {
            break;
        }

        sync.m_DrawThreadReady = false;

        size_t acc_r, acc_g, acc_b;
        size_t outputIndex = 0;

        size_t totalAA = fractal->GetGpuAntialiasing() * fractal->GetGpuAntialiasing();
        size_t palIters = fractal->m_PalIters[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex];
        const uint16_t* palR = fractal->m_PalR[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
        const uint16_t* palG = fractal->m_PalG[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
        const uint16_t* palB = fractal->m_PalB[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
        size_t basicFactor = 65536 / fractal->GetNumIterations();
        if (basicFactor == 0) {
            basicFactor = 1;
        }

        auto GetBasicColor = [&](
            size_t numIters,
            size_t& acc_r,
            size_t& acc_g,
            size_t& acc_b) {

            if (fractal->m_WhichPalette != Palette::Basic) {
                auto palIndex = numIters % palIters;
                acc_r += palR[palIndex];
                acc_g += palG[palIndex];
                acc_b += palB[palIndex];
            }
            else {
                acc_r += (numIters * basicFactor) & ((1llu << 16) - 1);
                acc_g += (numIters * basicFactor) & ((1llu << 16) - 1);
                acc_b += (numIters * basicFactor) & ((1llu << 16) - 1);
            }
        };

        for (size_t output_y = 0; output_y < fractal->m_ScrnHeight; output_y++) {
            if (sync.m_DrawThreadAtomics[output_y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (sync.m_DrawThreadAtomics[output_y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            outputIndex = output_y * fractal->m_ScrnWidth * BytesPerPixel;

            for (size_t output_x = 0;
                output_x < fractal->m_ScrnWidth;
                output_x++)
            {
                if (fractal->GetGpuAntialiasing() == 1) {
                    acc_r = 0;
                    acc_g = 0;
                    acc_b = 0;

                    const size_t input_x = output_x;
                    const size_t input_y = output_y;
                    size_t numIters = fractal->m_CurIters.m_ItersArray[input_y][input_x];

                    if (numIters < fractal->GetNumIterations())
                    {
                        numIters += fractal->m_PaletteRotate;
                        if (numIters >= MAXITERS) {
                            numIters = MAXITERS - 1;
                        }

                        GetBasicColor(numIters, acc_r, acc_g, acc_b);
                    }
                }
                else {
                    acc_r = 0;
                    acc_g = 0;
                    acc_b = 0;

                    for (size_t input_x = output_x * fractal->GetGpuAntialiasing();
                        input_x < (output_x + 1) * fractal->GetGpuAntialiasing();
                        input_x++) {
                        for (size_t input_y = output_y * fractal->GetGpuAntialiasing();
                            input_y < (output_y + 1) * fractal->GetGpuAntialiasing();
                            input_y++) {

                            size_t numIters = fractal->m_CurIters.m_ItersArray[input_y][input_x];
                            if (numIters < fractal->GetNumIterations())
                            {
                                numIters += fractal->m_PaletteRotate;
                                if (numIters >= MAXITERS) {
                                    numIters = MAXITERS - 1;
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

        sync.m_DrawThreadProcessed = true;
        lk.unlock();
        sync.m_DrawThreadCV.notify_one();
    }
}

void Fractal::FillCoord(HighPrecision& src, MattCoords& dest) {
    dest.qflt.v1 = Convert<HighPrecision, float>(src);
    dest.qflt.v2 = Convert<HighPrecision, float>(src - HighPrecision{ dest.qflt.v1 });
    dest.qflt.v3 = Convert<HighPrecision, float>(src - HighPrecision{ dest.qflt.v1 } - HighPrecision{ dest.qflt.v2 });
    dest.qflt.v4 = Convert<HighPrecision, float>(src - HighPrecision{ dest.qflt.v1 } - HighPrecision{ dest.qflt.v2 } - HighPrecision{ dest.qflt.v3 });

    dest.qdbl.v1 = Convert<HighPrecision, double>(src);
    dest.qdbl.v2 = Convert<HighPrecision, double>(src - HighPrecision{ dest.qdbl.v1 });
    dest.qdbl.v3 = Convert<HighPrecision, double>(src - HighPrecision{ dest.qdbl.v1 } - HighPrecision{ dest.qdbl.v2 });
    dest.qdbl.v4 = Convert<HighPrecision, double>(src - HighPrecision{ dest.qdbl.v1 } - HighPrecision{ dest.qdbl.v2 } - HighPrecision{ dest.qdbl.v3 });

    dest.dbl.head = Convert<HighPrecision, double>(src);
    dest.dbl.tail = Convert<HighPrecision, double>(src - HighPrecision{ dest.dbl.head });

    dest.doubleOnly = Convert<HighPrecision, double>(src);

    dest.hdrflt = (HDRFloat<float>)src;

    dest.hdrdbl = (HDRFloat<double>)src;

    dest.flt.y = Convert<HighPrecision, float>(src);
    dest.flt.x = Convert<HighPrecision, float>(src - HighPrecision{ dest.flt.y });

    dest.floatOnly = Convert<HighPrecision, float>(src);
}

void Fractal::FillGpuCoords(MattCoords &cx2, MattCoords &cy2, MattCoords &dx2, MattCoords &dy2) {
    HighPrecision src_dy =
        (m_MaxY - m_MinY) /
        (HighPrecision)(m_ScrnHeight * GetGpuAntialiasing());

    HighPrecision src_dx =
        (m_MaxX - m_MinX) /
        (HighPrecision)(m_ScrnWidth * GetGpuAntialiasing());

    FillCoord(m_MinX, cx2);
    FillCoord(m_MinY, cy2);
    FillCoord(src_dx, dx2);
    FillCoord(src_dy, dy2);
}

void Fractal::CalcGpuFractal(bool MemoryOnly)
{
    OutputMessage(L"\r\n");

    MattCoords cx2{}, cy2{}, dx2{}, dy2{};

    FillGpuCoords(cx2, cy2, dx2, dy2);

    uint32_t err =
        m_r.InitializeMemory(m_CurIters.m_Width,
                             m_CurIters.m_Height);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    err = m_r.Render(m_RenderAlgorithm,
                     (uint32_t*)m_CurIters.m_ItersMemory,
                     cx2,
                     cy2,
                     dx2,
                     dy2,
                     (uint32_t)m_NumIterations,
                     m_IterationPrecision);
    if (err) {
        MessageBoxCudaError(err);
    }

    DrawFractal(MemoryOnly);
}

//////////////////////////////////////////////////////////////////////////////
// Draw the network part of the fractal.  This function is very similar
// to the previous one, except it reads data in from the network instead
// of calculating it locally.  The network can be much faster.
//////////////////////////////////////////////////////////////////////////////
void Fractal::CalcNetworkFractal(bool MemoryOnly)
{
    size_t nx, ny;
    uint32_t numItersL;
    WORD numItersS;
    uint32_t numIters = 0;

    int i;

    for (i = 0; i < MAXSERVERS; i++)
    {
        if (m_SetupData.m_UseThisServer[i] == 'n')
        {
            continue;
        }

        m_ClientSubNetwork[i]->BufferedReadEmpty();

        for (ny = 0; ny < m_ScrnHeight; ny++)
        {
            if (m_ProcessPixelRow[ny] != i + 'b')
            {
                continue;
            }

            for (nx = 0; nx < m_ScrnWidth; nx++)
            {
                if (m_NumIterations >= 65536)
                {
                    m_ClientSubNetwork[i]->BufferedReadLong(&numItersL);
                    numIters = numItersL;
                }
                else
                {
                    m_ClientSubNetwork[i]->BufferedReadShort(&numItersS);
                    numIters = numItersS;
                }

                m_CurIters.m_ItersArray[ny][nx] = numIters;
            }

            if (MemoryOnly == false)
            {
                assert(false);
                // TODO broken
                //DrawFractalLine(ny);
            }
        }
    }
}

void Fractal::CalcCpuPerturbationFractal(bool MemoryOnly) {
    auto* results = GetUsefulPerturbationResults<double, double>();

    double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / m_ScrnWidth);
    double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / m_ScrnHeight);

    double centerX = (double)(results->hiX - m_MinX);
    double centerY = (double)(results->hiY - m_MaxY);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight);
    threads.reserve(num_threads);

    //
    // From https://fractalforums.org/fractal-mathematics-and-new-theories/28/another-solution-to-perturbation-glitches/4360/msg29835#msg29835
    //

    //complex Reference[]; // Reference orbit (MUST START WITH ZERO)
    //int MaxRefIteration; // The last valid iteration of the reference (the iteration just before it escapes)
    //complex dz = 0, dc; // Delta z and Delta c
    //int Iteration = 0, RefIteration = 0;

    //while (Iteration < MaxIteration) {
    //    dz = 2 * dz * Reference[RefIteration] + dz * dz + dc;
    //    RefIteration++;

    //    complex z = Reference[RefIteration] + dz;
    //    if (| z | > BailoutRadius) break;
    //    if (| z | < | dz| || RefIteration == MaxRefIteration) {
    //        dz = z;
    //        RefIteration = 0;
    //    }
    //    Iteration++;
    //}

    auto one_thread = [&]() {
        for (size_t y = 0; y < m_ScrnHeight; y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth; x++)
            {
                size_t iter = 0;
                size_t RefIteration = 0;
                double deltaReal = dx * x - centerX;
                double deltaImaginary = -dy * y - centerY;

                double DeltaSub0X = deltaReal;
                double DeltaSub0Y = deltaImaginary;
                double DeltaSubNX = 0;
                double DeltaSubNY = 0;

                while (iter < m_NumIterations) {
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
                    // DeltaSubN = 2 * DeltaSubN * results.complex[RefIteration] + DeltaSubN * DeltaSubN + DeltaSub0;
                    // S * w = 2 * S * w * results.complex[RefIteration] + S * w * S * w + S * d
                    // 
                    // S * (DeltaSubNWX + DeltaSubNWY I) = 2 * S * (DeltaSubNWX + DeltaSubNWY I) * (results.x[RefIteration] + results.y[RefIteration] * I) +
                    //                                     S * S * (DeltaSubNWX + DeltaSubNWY I) * (DeltaSubNWX + DeltaSubNWY I) +
                    //                                     S * (dX + dY I)
                    // 
                    // (DeltaSubNWX + DeltaSubNWY I) = 2 * (DeltaSubNWX + DeltaSubNWY I) * (results.x[RefIteration] + results.y[RefIteration] * I) +
                    //                                 S * (DeltaSubNWX + DeltaSubNWY I) * (DeltaSubNWX + DeltaSubNWY I) +
                    //                                 (dX + dY I)
                    // DeltaSubNWX = 2 * (DeltaSubNWX * results.x[RefIteration] - DeltaSubNWY * results.y[RefIteration]) +
                    //               S * (DeltaSubNWX * DeltaSubNWX - DeltaSubNWY * DeltaSubNWY) +
                    //               dX
                    // DeltaSubNWX = 2 * DeltaSubNWX * results.x[RefIteration] - 2 * DeltaSubNWY * results.y[RefIteration] +
                    //               S * DeltaSubNWX * DeltaSubNWX - S * DeltaSubNWY * DeltaSubNWY +
                    //               dX
                    //
                    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY * results.x[RefIteration] +
                    //               S * DeltaSubNWX * DeltaSubNWY + S * DeltaSubNWY * DeltaSubNWX +
                    //               dY
                    // DeltaSubNWY = 2 * DeltaSubNWX * results.y[RefIteration] + 2 * DeltaSubNWY * results.x[RefIteration] +
                    //               2 * S * DeltaSubNWX * DeltaSubNWY +
                    //               dY

                    DeltaSubNX = DeltaSubNXOrig * (results->x[RefIteration] * 2 + DeltaSubNXOrig) -
                        DeltaSubNYOrig * (results->y[RefIteration] * 2 + DeltaSubNYOrig) +
                        DeltaSub0X;
                    DeltaSubNY = DeltaSubNXOrig * (results->y[RefIteration] * 2 + DeltaSubNYOrig) +
                        DeltaSubNYOrig * (results->x[RefIteration] * 2 + DeltaSubNXOrig) +
                        DeltaSub0Y;

                    ++RefIteration;

                    const double tempZX = results->x[RefIteration] + DeltaSubNX;
                    const double tempZY = results->y[RefIteration] + DeltaSubNY;
                    const double zn_size = tempZX * tempZX + tempZY * tempZY;
                    const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                    if (zn_size > 256) {
                        break;
                    }

                    if (zn_size < normDeltaSubN ||
                        RefIteration == results->x.size() - 1) {
                        DeltaSubNX = tempZX;
                        DeltaSubNY = tempZY;
                        RefIteration = 0;
                    }

                    ++iter;
                }

                m_CurIters.m_ItersArray[y][x] = (uint32_t)iter;
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
template<class T, class SubType>
void Fractal::CalcCpuHDR(bool MemoryOnly) {
    const T dx = T((m_MaxX - m_MinX) / m_ScrnWidth);
    const T dy = T((m_MaxY - m_MinY) / m_ScrnHeight);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight);
    threads.reserve(num_threads);

    const T Four{ 4 };
    const T Two{ 2 };

    auto one_thread = [&]() {
        for (size_t y = 0; y < m_ScrnHeight; y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            T cx = T{ m_MinX };
            T cy = T{ m_MaxY - dy * (SubType)y };
            T zx, zy;
            T zx2, zy2;
            T sum;
            unsigned int i;

            for (size_t x = 0; x < m_ScrnWidth; x++)
            {
                // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
                // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
                zx = cx;
                zy = cy;
                for (i = 0; i < m_NumIterations; i++)
                { // x^2+2*I*x*y-y^2
                    zx2 = zx * zx;
                    zy2 = zy * zy;
                    sum = zx2 + zy2;
                    HdrReduce(sum);
                    if (sum > Four) break;

                    zy = Two * zx * zy;
                    zx = zx2 - zy2;

                    zx += cx;
                    zy += cy;

                    HdrReduce(zx);
                    HdrReduce(zy);
                }
                cx += dx;
                //HdrReduce(cx);

                m_CurIters.m_ItersArray[y][x] = (uint32_t)i;
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

template<class T, class SubType>
void Fractal::CalcCpuPerturbationFractalBLA(bool MemoryOnly) {
    auto* results = GetUsefulPerturbationResults<T, SubType>();

    BLAS<T> blas(*results);
    blas.Init(results->x.size(), T{ results->maxRadius });

    T dx = T((m_MaxX - m_MinX) / m_ScrnWidth);
    T dy = T((m_MaxY - m_MinY) / m_ScrnHeight);

    T centerX = (T)(results->hiX - m_MinX);
    HdrReduce(centerX);
    T centerY = (T)(results->hiY - m_MaxY);
    HdrReduce(centerY);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight);
    threads.reserve(num_threads);

    auto one_thread = [&]() {
        //T dzdcX = T(1.0);
        //T dzdcY = T(0.0);
        //bool periodicity_should_break = false;

        for (size_t y = 0; y < m_ScrnHeight; y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth; x++)
            {
                size_t iter = 0;
                size_t RefIteration = 0;
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

                while (iter < m_NumIterations) {
                    BLA<T> *b = nullptr;
                    while ((b = blas.LookupBackwards(RefIteration, DeltaNormSquared)) != nullptr) {
                        int l = b->getL();

                        // TODO this first RefIteration + l check bugs me
                        if (RefIteration + l >= results->x.size()) {
                            //::MessageBox(NULL, L"Out of bounds! :(", L"", MB_OK);
                            break;
                        }

                        if (iter + l >= m_NumIterations) {
                            break;
                        }
                        
                        iter += l;

                        //double t1 = (double)DeltaSubNX;
                        //double t2 = (double)DeltaSubNY;
                        //b->getValue(t1, t2, (double)DeltaSub0X, (double)DeltaSub0Y);
                        //DeltaSubNX = t1;
                        //DeltaSubNY = t2;

                        b->getValue(DeltaSubNX, DeltaSubNY, DeltaSub0X, DeltaSub0Y);

                        RefIteration += l;

                        T tempZX = results->x[RefIteration] + DeltaSubNX;
                        T tempZY = results->y[RefIteration] + DeltaSubNY;
                        T normSquared = tempZX * tempZX + tempZY * tempZY;
                        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                        HdrReduce(normSquared);
                        HdrReduce(DeltaNormSquared);

                        if (normSquared > T(256)) {
                            break;
                        }

                        if (normSquared < DeltaNormSquared ||
                            RefIteration >= results->x.size() - 1) {
                            DeltaSubNX = tempZX;
                            DeltaSubNY = tempZY;
                            DeltaNormSquared = normSquared;
                            RefIteration = 0;
                        }
                    }

                    if (iter >= m_NumIterations) {
                        break;
                    }

                    const T DeltaSubNXOrig = DeltaSubNX;
                    const T DeltaSubNYOrig = DeltaSubNY;

                    T TermB1 = DeltaSubNXOrig * (results->x[RefIteration] * 2 + DeltaSubNXOrig);
                    T TermB2 = DeltaSubNYOrig * (results->y[RefIteration] * 2 + DeltaSubNYOrig);

                    DeltaSubNX = TermB1 - TermB2;
                    DeltaSubNX += DeltaSub0X;
                    HdrReduce(DeltaSubNX);

                    T Term3 = results->y[RefIteration] * 2 + DeltaSubNYOrig;
                    T Term4 = results->x[RefIteration] * 2 + DeltaSubNXOrig;
                    DeltaSubNY = DeltaSubNXOrig * Term3 + DeltaSubNYOrig * Term4;
                    DeltaSubNY += DeltaSub0Y;
                    HdrReduce(DeltaSubNY);

                    ++RefIteration;
                    if (RefIteration >= results->x.size()) {
                        ::MessageBox(NULL, L"Out of bounds 2! :(", L"", MB_OK);
                        break;
                    }

                    if constexpr (true) {
                        //template<typename real>
                        //Temp<real> Prepare(const std::complex<real> &dz, const std::complex<real> &dc) const {
                        //    Temp<real> temp;
                        //    temp.newdz = dz * ((real(2.0) * std::complex<real>(Ref)) + dz);
                        //    vreal ChebyMagNewdz = ChebyshevNorm(temp.newdz);
                        //    temp.unusable = (ChebyMagNewdz >= real(LAThreshold));
                        //    return temp;
                        //}
                        //DzdzStep = 2.0 * (dz + Ref) * ZCoeff;
                        //DzdzStepXY = 2.0 * (dzX + dzY * i + RefX + RefY * i) * (ZCoeffX + ZCoeffY * i);
                        //DzdzStepXY = 2.0 * ((dzX + RefX) + (dzY + RefY) * i) * (ZCoeffX + ZCoeffY * i);
                        //DzdzStepXY = 2.0 * ((dzX + RefX) * ZCoeffX + (dzX + RefX) * ZCoeffY * i + (dzY + RefY) * i * ZCoeffX + (dzY + RefY) * i * ZCoeffY * i);
                        //DzdzStepX = 2.0 * ((dzX + RefX) * ZCoeffX - (dzY + RefY) * ZCoeffY);
                        //DzdzStepY = 2.0 * ((dzX + RefX) * ZCoeffY + (dzY + RefY) * ZCoeffX);
                        //DzdzStepX = 2.0 * (dzX + RefX) * ZCoeffX - 2.0 * (dzY + RefY) * ZCoeffY;
                        //DzdzStepY = 2.0 * (dzX + RefX) * ZCoeffY + 2.0 * (dzY + RefY) * ZCoeffX);

                        //dzdz = dzdz * DzdzStep;
                        //dzdzXY = (dzdzX + dzdzY * i) * (DzdzStepX + DzdzStepY * i);
                        //dzdzXY = dzdzX * DzdzStepX + dzdzX * DzdzStepY * i + dzdzY * i * DzdzStepX + dzdzY * i * DzdzStepY * i;
                        //dzdzX = dzdzX * DzdzStepX - dzdzY * DzdzStepY;
                        //dzdzY = dzdzX * DzdzStepY + dzdzY * DzdzStepX;
                        // 
                        //dzdc = dzdc * DzdzStep + complex(CCoeff) * ScalingFactor;
                        //dzdcXY = (dzdcX + dzdcY * i) * (DzdzStepX + DzdzStepY * i) + (CCoeffX + CCoeffY * i) * (ScalingFactorX + ScalingFactorY * i);
                        //dzdcXY = (dzdcX * DzdzStepX + dzdcX * DzdzStepY * i + dzdcY * i * DzdzStepX + dzdcY * i * DzdzStepY * i) + (CCoeffX * ScalingFactorX + CCoeffX * ScalingFactorY * i + CCoeffY * i * ScalingFactorX + CCoeffY * i * ScalingFactorY * i);
                        //dzdcX = (dzdcX * DzdzStepX - dzdcY * DzdzStepY) + (CCoeffX * ScalingFactorX - CCoeffY * ScalingFactorY);
                        //dzdcY = (dzdcX * DzdzStepY + dzdcY * DzdzStepX) + (CCoeffX * ScalingFactorY + CCoeffY * ScalingFactorX);
                        /////////////////////

                        //HdrReduce(dzdcX);
                        //auto dzdcX1 = HdrAbs(dzdcX);

                        //HdrReduce(dzdcY);
                        //auto dzdcY1 = HdrAbs(dzdcY);

                        //HdrReduce(zxCopy);
                        //auto zxCopy1 = HdrAbs(zxCopy);

                        //HdrReduce(zyCopy);
                        //auto zyCopy1 = HdrAbs(zyCopy);

                        //T n2 = max(zxCopy1, zyCopy1);

                        //T r0 = max(dzdcX1, dzdcY1);
                        //T maxRadiusHdr{ 3840 };
                        //auto n3 = maxRadiusHdr * r0 * HighTwo;
                        //HdrReduce(n3);

                        //if (n3 > n2) {
                        //    periodicity_should_break = true;
                        //}
                        //else {
                        //    auto dzdcXOrig = dzdcX;
                        //    dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                        //    dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
                        //}
                    }

                    T tempZX = results->x[RefIteration] + DeltaSubNX;
                    T tempZY = results->y[RefIteration] + DeltaSubNY;
                    T nT1 = tempZX * tempZX;
                    T nT2 = tempZY * tempZY;
                    T normSquared = nT1 + nT2;
                    HdrReduce(normSquared);

                    DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                    HdrReduce(DeltaNormSquared);

                    if (normSquared > T(256)) {
                        break;
                    }

                    if (normSquared < DeltaNormSquared ||
                        RefIteration >= results->x.size() - 1) {
                        DeltaSubNX = tempZX;
                        DeltaSubNY = tempZY;
                        DeltaNormSquared = normSquared;
                        RefIteration = 0;
                    }

                    ++iter;
                }

                m_CurIters.m_ItersArray[y][x] = (uint32_t)iter;
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

bool Fractal::IsThisPerturbationArrayUsed(void* check) const {
    switch (GetRenderAlgorithm()) {
        case RenderAlgorithm::CpuHDR32:
        case RenderAlgorithm::CpuHigh:
        case RenderAlgorithm::Cpu64:
        case RenderAlgorithm::CpuHDR64:
        case RenderAlgorithm::Gpu1x64:
        case RenderAlgorithm::Gpu2x64:
        case RenderAlgorithm::Gpu4x64:
        case RenderAlgorithm::Gpu1x32:
        case RenderAlgorithm::Gpu2x32:
        case RenderAlgorithm::Gpu4x32:
            return false;
        case RenderAlgorithm::Cpu32PerturbedBLAHDR:
        case RenderAlgorithm::GpuHDRx32PerturbedBLA:
        case RenderAlgorithm::GpuHDRx32PerturbedScaled:
            return check == &m_PerturbationResultsHDRFloat;
        case RenderAlgorithm::Cpu64PerturbedBLAHDR:
        case RenderAlgorithm::GpuHDRx64PerturbedBLA:
            return check == &m_PerturbationResultsHDRDouble;
        case RenderAlgorithm::Gpu1x32Perturbed:
            return check == &m_PerturbationResultsFloat;
        case RenderAlgorithm::Cpu64PerturbedBLA:
        case RenderAlgorithm::Gpu1x32PerturbedScaled:
        case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
        case RenderAlgorithm::Gpu1x64Perturbed:
        case RenderAlgorithm::Gpu1x64PerturbedBLA:
            return check == &m_PerturbationResultsDouble;
        case RenderAlgorithm::Gpu2x32Perturbed:
            // TODO
            //CalcGpuPerturbationFractalBLA<dblflt, dblflt>(MemoryOnly);
            assert(false);
            return false;
        case RenderAlgorithm::Gpu2x32PerturbedScaled:
            // TODO
            //CalcGpuPerturbationFractalBLA<double, double>(MemoryOnly);
            assert(false);
            return false;
        default:
            return false;
    }
}

void Fractal::OptimizeMemory() {
    PROCESS_MEMORY_COUNTERS_EX checkHappy;
    const size_t OMGAlotOfMemory = 128llu * 1024llu * 1024llu * 1024llu;
    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));

    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsDouble)) {
            m_PerturbationResultsDouble.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsFloat)) {
            m_PerturbationResultsFloat.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsHDRDouble)) {
            m_PerturbationResultsHDRDouble.clear();
        }

        if (!IsThisPerturbationArrayUsed(&m_PerturbationResultsHDRFloat)) {
            m_PerturbationResultsHDRFloat.clear();
        }
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ClearPerturbationResults();
    }

    GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&checkHappy, sizeof(checkHappy));
    if (checkHappy.PagefileUsage > OMGAlotOfMemory) {
        ::MessageBox(NULL, L"Watch the memory use... this is just a warning", L"", MB_OK);
        assert(false);
    }
}

template<class T>
std::vector<PerturbationResults<T>>& Fractal::GetPerturbationResults() {
    if constexpr (std::is_same<T, double>::value) {
        return m_PerturbationResultsDouble;
    }
    else if constexpr (std::is_same<T, float>::value) {
        return m_PerturbationResultsFloat;
    } 
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        return m_PerturbationResultsHDRDouble;
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        return m_PerturbationResultsHDRFloat;
    }
}

template<class T, class SubType, bool BenchmarkMode>
void Fractal::AddPerturbationReferencePoint() {
    if (m_PerturbationGuessCalcX == 0 && m_PerturbationGuessCalcY == 0) {
        m_PerturbationGuessCalcX = (m_MaxX + m_MinX) / 2;
        m_PerturbationGuessCalcY = (m_MaxY + m_MinY) / 2;
    }

    if (m_PerturbationAlg == PerturbationAlg::ST) {
        AddPerturbationReferencePointST<T, SubType, false, BenchmarkMode>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
    }
    else if (m_PerturbationAlg == PerturbationAlg::MT) {
        AddPerturbationReferencePointMT2<T, SubType, false, BenchmarkMode>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
    }
    else if (m_PerturbationAlg == PerturbationAlg::STPeriodicity) {
        AddPerturbationReferencePointST<T, SubType, true, BenchmarkMode>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
    }
    else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity2) {
        AddPerturbationReferencePointMT2<T, SubType, true, BenchmarkMode>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
    }
    else if (m_PerturbationAlg == PerturbationAlg::MTPeriodicity5) {
        AddPerturbationReferencePointMT5<T, SubType, true, BenchmarkMode>(m_PerturbationGuessCalcX, m_PerturbationGuessCalcY);
    }
}

template<class T, class SubType, bool Periodicity, bool BenchmarkMode>
void Fractal::AddPerturbationReferencePointST(HighPrecision initX, HighPrecision initY) {
    auto &PerturbationResults = GetPerturbationResults<T>();
    PerturbationResults.push_back({});
    auto *results = &PerturbationResults[PerturbationResults.size() - 1];

    HighPrecision radiusX = fabs(m_MaxX - initX) + fabs(m_MinX - initX);
    HighPrecision radiusY = fabs(m_MaxY - initY) + fabs(m_MinY - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_NumIterations + 1; // +1 for push_back(0) below

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    unsigned int i;

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_NumIterations);
    results->y.reserve(m_NumIterations);
    results->bad.reserve(m_NumIterations);

    results->x.push_back(T(0));
    results->y.push_back(T(0));
    // Note: results->bad is not here.  See end of this function.

    //SubType glitch;
    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

    SubType glitch = (SubType)0.0000001;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    T zxCopy;
    T zyCopy;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    zx = cx;
    zy = cy;
    for (i = 0; i < m_NumIterations; i++)
    {
        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };
        }

        zx2 = zx * 2;
        zy2 = zy * 2;

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        T sq_x = double_zx * double_zx;
        T sq_y = double_zy * double_zy;
        T norm = (sq_x + sq_y) * glitch;

        bool underflow = (HdrAbs((T)zx) <= small_float ||
                          HdrAbs((T)zy) <= small_float ||
                          norm <= small_float);
        results->bad.push_back(underflow);

        if constexpr (Periodicity) {
            // x^2+2*I*x*y-y^2
            //dzdc = 2.0 * z * dzdc + real(1.0);
            //dzdc = 2.0 * (zx + zy * i) * (dzdcX + dzdcY * i) + HighPrecision(1.0);
            //dzdc = 2.0 * (zx * dzdcX + zx * dzdcY * i + zy * i * dzdcX + zy * i * dzdcY * i) + HighPrecision(1.0);
            //dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX + 2.0 * zy * i * dzdcY * i + HighPrecision(1.0);
            //dzdc = 2.0 * zx * dzdcX + 2.0 * zx * dzdcY * i + 2.0 * zy * i * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0);
            //
            // dzdcX = 2.0 * zx * dzdcX - 2.0 * zy * dzdcY + HighPrecision(1.0)
            // dzdcY = 2.0 * zx * dzdcY + 2.0 * zy * dzdcX

            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n3 > n2) {
                if constexpr (!BenchmarkMode) {
                    break;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        zx = zx * zx - zy * zy + cx;
        zy = zx2 * zy + cy;

        T tempZX = double_zx + (T)cx;
        T tempZY = double_zy + (T)cy;
        T zn_size = tempZX * tempZX + tempZY * tempZY;
        if (zn_size > TwoFiftySix) {
            break;
        }
    }

    results->bad.push_back(false);
    assert(results->bad.size() == results->x.size());
}

template<class Type>
struct ThreadPtrs {
    std::atomic<Type*> In;
    std::atomic<Type*> Out;
};

#define ENABLE_PREFETCH(ARG0, ARG1) _mm_prefetch(ARG0, ARG1)
//#define ENABLE_PREFETCH(ARG0, ARG1)

#define CheckStartCriteria \
    _mm_pause(); \
    if (expected == nullptr || \
        ThreadMemory->In.compare_exchange_weak( \
            expected, \
            nullptr, \
            std::memory_order_relaxed) == false) { \
        continue; \
    } \
 \
    if (expected == (void*)0x1) { \
        break; \
    } \
    ENABLE_PREFETCH((const char*)expected, _MM_HINT_T0); \
    ok = expected; \

#define CheckFinishCriteria \
    expected = nullptr; \
    for (;;) { \
        _mm_pause(); \
        bool result = ThreadMemory->Out.compare_exchange_weak( \
            expected, \
            ok, \
            std::memory_order_relaxed); \
        if (result) { \
            break; \
        } \
    } \

static inline void prefetch_range(void* addr, std::size_t len) {
    constexpr uintptr_t prefetch_stride = 64;
    void* vp = addr;
    void* end = reinterpret_cast<void*>(
        reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(len));
    while (vp < end) {
        ENABLE_PREFETCH((const char*)vp, _MM_HINT_T0);
        vp = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(vp) + static_cast<uintptr_t>(prefetch_stride));
    }
}

static inline void PrefetchHighPrec(HighPrecision& target) {
    ENABLE_PREFETCH((const char*)&target.backend().data(), _MM_HINT_T0);
    size_t lastindex = abs(target.backend().data()->_mp_size);
    size_t size_elt = sizeof(mp_limb_t);
    size_t total = size_elt * lastindex;
    prefetch_range(target.backend().data()->_mp_d, total);
}

template<class T, class SubType, bool Periodicity, bool BenchmarkMode>
void Fractal::AddPerturbationReferencePointMT2(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResults = GetPerturbationResults<T>();
    PerturbationResults.push_back({});
    auto* results = &PerturbationResults[PerturbationResults.size() - 1];

    HighPrecision radiusX = fabs(m_MaxX - initX) + fabs(m_MinX - initX);
    HighPrecision radiusY = fabs(m_MaxY - initY) + fabs(m_MinY - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    HighPrecision zx, zy;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_NumIterations + 1; // +1 for push_back(0) below

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_NumIterations);
    results->y.reserve(m_NumIterations);
    results->bad.reserve(m_NumIterations);
    
    results->x.push_back(T(0));
    results->y.push_back(T(0));
    // Note: results->bad is not here.  See end of this function.

    //SubType glitch;
    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        HighPrecision zx;
        HighPrecision zx_sq;
    };

    struct ThreadZyData {
        HighPrecision zy;
        HighPrecision zy_sq;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto ThreadSqZx = [](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zx);

            ok->zx_sq = ok->zx * ok->zx;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto ThreadSqZy = [](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zy);

            ok->zy_sq = ok->zy * ok->zy;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};

    //threadZxdata->zx_sq = &zx_sq;

    //threadZydata->zy_sq = &zy_sq;

    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;

    bool done1 = false;
    bool done2 = false;

    HighPrecision zy_sq_orig;

    zx = cx;
    zy = cy;

    T zxCopy;
    T zyCopy;
    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    bool zyStarted = false;

    for (size_t i = 0; i < m_NumIterations; i++)
    {
        // Start Zx squaring thread
        threadZxdata->zx = zx;

        if (!zyStarted) {
            threadZydata->zy = zy;
        }

        ThreadZxMemory->In.store(
            threadZxdata,
            std::memory_order_release);

        if (!zyStarted) {
            // Start Zy squaring thread
            ThreadZyMemory->In.store(
                threadZydata,
                std::memory_order_relaxed);

            zyStarted = true;
        }

        T double_zx;
        T double_zy;

        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };

            double_zx = zxCopy;
            double_zy = zyCopy;
        }
        else {
            double_zx = (T)zx;
            double_zy = (T)zy;
        }
        
        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        T norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
        //HdrReduce(norm);
        bool underflow = (HdrAbs(double_zx) <= small_float ||
            HdrAbs(double_zy) <= small_float ||
            norm <= small_float);
        results->bad.push_back(underflow);

        // Note: not T.
        const SubType tempZX = (SubType)double_zx + (SubType)cx;
        const SubType tempZY = (SubType)double_zy + (SubType)cy;
        const SubType zn_size = tempZX * tempZX + tempZY * tempZY;

        if constexpr (Periodicity) {
            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);

            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);

            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n3 > n2) {
                if constexpr (!BenchmarkMode) {
                    periodicity_should_break = true;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        zy = zx * 2 * zy + cy;

        done1 = false;
        done2 = false;
        bool quitting = false;

        for (;;) {
            expectedZy = threadZydata;

            _mm_pause();
            if (!done2 &&
                ThreadZyMemory->Out.compare_exchange_weak(expectedZy,
                    nullptr,
                    std::memory_order_release)) {
                done2 = true;

                PrefetchHighPrec(threadZydata->zy_sq);

                if constexpr (Periodicity) {
                    if (periodicity_should_break) {
                        results->m_Periodic = true;
                        quitting = true;
                    }
                }

                if (zn_size > 256) {
                    quitting = true;
                }

                if (!quitting) {
                    zy_sq_orig = threadZydata->zy_sq;

                    // Restart right away!
                    threadZydata->zy = zy;

                    ThreadZyMemory->In.store(
                        threadZydata,
                        std::memory_order_release);
                }
            }

            expectedZx = threadZxdata;

            _mm_pause();
            if (!done1 &&
                ThreadZxMemory->Out.compare_exchange_weak(expectedZx,
                    nullptr,
                    std::memory_order_release)) {
                done1 = true;

                PrefetchHighPrec(threadZxdata->zx_sq);
            }

            if (done1 && done2) {
                break;
            }
        }

        zx = threadZxdata->zx_sq - zy_sq_orig + cx;

        if (!quitting) {
            continue;
        }

        break;
    }

    results->bad.push_back(false);
    assert(results->bad.size() == results->x.size());

    expectedZx = nullptr;
    ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);

    expectedZy = nullptr;
    ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);

    tZx->join();
    tZy->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);
}

template<class T, class SubType, bool Periodicity, bool BenchmarkMode>
void Fractal::AddPerturbationReferencePointMT5(HighPrecision initX, HighPrecision initY) {
    auto& PerturbationResults = GetPerturbationResults<T>();
    PerturbationResults.push_back({});
    auto* results = &PerturbationResults[PerturbationResults.size() - 1];

    HighPrecision radiusX = fabs(m_MaxX - initX) + fabs(m_MinX - initX);
    HighPrecision radiusY = fabs(m_MaxY - initY) + fabs(m_MinY - initY);
    HighPrecision cx = initX;
    HighPrecision cy = initY;

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    HighPrecision zx_sq, zy_sq;

    T dzdcX = T(1.0);
    T dzdcY = T(0.0);

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = T((radiusX > radiusY) ? radiusX : radiusY);
    results->MaxIterations = m_NumIterations + 1; // +1 for push_back(0) below

    //volatile double tempX = Convert<HighPrecision, double>(initX);
    //volatile double tempY = Convert<HighPrecision, double>(initY);
    //volatile double intX = Convert<HighPrecision, double>(XFromCalcToScreen(initX));
    //volatile double intY = Convert<HighPrecision, double>(YFromCalcToScreen(initY));

    const T small_float = T((SubType)1.1754944e-38);

    results->x.reserve(m_NumIterations);
    results->y.reserve(m_NumIterations);
    results->bad.reserve(m_NumIterations);

    results->x.push_back(T(0));
    results->y.push_back(T(0));
    // Note: results->bad is not here.  See end of this function.

    //if constexpr (std::is_same<SubType, double>::value) {
    //    glitch = (SubType)0.0000001;
    //}
    //else if constexpr (std::is_same<SubType, float>::value) {
    //    glitch = (SubType)0.0001;
    //}
    //else {
    //    ::MessageBox(NULL, L"Confused", L"", MB_OK);
    //    assert(false);
    //    return;
    //}

    SubType glitch = (SubType)0.0000001;

    struct ThreadZxData {
        HighPrecision zx;
        HighPrecision* zx_sq;
    };

    struct ThreadZyData {
        HighPrecision zy;
        HighPrecision* zy_sq;
    };

    struct Thread1Data {
        HighPrecision* zx_sq;
        HighPrecision* zy_sq;
        HighPrecision* zx2;
        HighPrecision* zy2;
        HighPrecision* zx;
        HighPrecision zy;
        HighPrecision* cx;
    };

    struct Thread2Data {
        HighPrecision zx2;
        HighPrecision* zy2;
        HighPrecision zy;
        HighPrecision* cy;
    };

    auto* ThreadZxMemory = (ThreadPtrs<ThreadZxData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZxData>), 64);
    memset(ThreadZxMemory, 0, sizeof(*ThreadZxMemory));

    auto* ThreadZyMemory = (ThreadPtrs<ThreadZyData> *)
        _aligned_malloc(sizeof(ThreadPtrs<ThreadZyData>), 64);
    memset(ThreadZyMemory, 0, sizeof(*ThreadZyMemory));

    auto* Thread1Memory = (ThreadPtrs<Thread1Data> *)
        _aligned_malloc(sizeof(ThreadPtrs<Thread1Data>), 64);
    memset(Thread1Memory, 0, sizeof(*Thread1Memory));

    auto* Thread2Memory = (ThreadPtrs<Thread2Data>*)
        _aligned_malloc(sizeof(ThreadPtrs<Thread2Data>), 64);
    memset(Thread2Memory, 0, sizeof(*Thread2Memory));

    auto ThreadSqZx = [](ThreadPtrs<ThreadZxData>* ThreadMemory) {
        for (;;) {
            ThreadZxData* expected = ThreadMemory->In.load();
            ThreadZxData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zx);

            *ok->zx_sq = ok->zx * ok->zx;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto ThreadSqZy = [](ThreadPtrs<ThreadZyData>* ThreadMemory) {
        for (;;) {
            ThreadZyData* expected = ThreadMemory->In.load();
            ThreadZyData* ok = nullptr;

            CheckStartCriteria;
            PrefetchHighPrec(ok->zy);

            *ok->zy_sq = ok->zy * ok->zy;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto Thread1 = [](ThreadPtrs<Thread1Data>* ThreadMemory,
                      ThreadZxData *threadZxdata,
                      ThreadZyData* threadZydata,
                      ThreadPtrs<ThreadZxData>* ThreadZxMemory,
                      ThreadPtrs<ThreadZyData>* ThreadZyMemory) {
        HighPrecision temp3;

        ThreadZxData* expectedZx = nullptr;
        ThreadZyData* expectedZy = nullptr;

        for (;;) {
            Thread1Data* expected = ThreadMemory->In.load();
            Thread1Data* ok = nullptr;

            CheckStartCriteria;

            PrefetchHighPrec(*ok->cx);
            PrefetchHighPrec(*ok->zx);

            // Wait for squaring
            bool zxOk = false;
            bool zyOk = false;
            for (;;) {
                expectedZx = threadZxdata;

                if (!zxOk && ThreadZxMemory->Out.compare_exchange_weak(expectedZx, nullptr, std::memory_order_relaxed)) {
                    zxOk = true;

                    std::atomic_thread_fence(std::memory_order_release);
                    PrefetchHighPrec(*ok->zx_sq);
                }

                expectedZy = threadZydata;

                if (!zyOk && ThreadZyMemory->Out.compare_exchange_weak(expectedZy, nullptr, std::memory_order_relaxed)) {
                    zyOk = true;
                    std::atomic_thread_fence(std::memory_order_release);
                    PrefetchHighPrec(*ok->zy_sq);
                }

                if (zxOk && zyOk) {
                    break;
                }
            }

            temp3 = *ok->zx_sq - *ok->zy_sq;
            *ok->zx = temp3 + *ok->cx;
            *ok->zx2 = *ok->zx * 2;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto Thread2 = [](ThreadPtrs<Thread2Data>* ThreadMemory) {
        HighPrecision temp1;
        for (;;) {
            Thread2Data* expected = ThreadMemory->In.load();
            Thread2Data* ok = nullptr;

            CheckStartCriteria;

            // _mm_prefetch((const char*)ok->zx_sq, _MM_HINT_T0);
            // _mm_prefetch((const char*)&ok->zx_sq->backend().data(), _MM_HINT_T0);
            PrefetchHighPrec(*ok->cy);
            PrefetchHighPrec(ok->zx2);
            PrefetchHighPrec(ok->zy);

            temp1 = ok->zx2 * ok->zy;
            ok->zy = temp1 + *ok->cy;
            *ok->zy2 = ok->zy * 2;

            // Give result back.
            CheckFinishCriteria;
        }
    };

    auto* threadZxdata = (ThreadZxData*)_aligned_malloc(sizeof(ThreadZxData), 64);
    auto* threadZydata = (ThreadZyData*)_aligned_malloc(sizeof(ThreadZyData), 64);
    auto* thread1data = (Thread1Data*)_aligned_malloc(sizeof(Thread1Data), 64);
    auto* thread2data = (Thread2Data*)_aligned_malloc(sizeof(Thread2Data), 64);

    new (threadZxdata) (ThreadZxData){};
    new (threadZydata) (ThreadZyData){};
    new (thread1data) (Thread1Data){};
    new (thread2data) (Thread2Data){};

    threadZxdata->zx_sq = &zx_sq;

    threadZydata->zy_sq = &zy_sq;

    thread1data->zx2 = &zx2;
    thread1data->zy2 = &zy2;
    thread1data->zx_sq = &zx_sq;
    thread1data->zy_sq = &zy_sq;
    thread1data->zx = &zx;
    thread1data->cx = &cx;

    thread2data->zy2 = &zy2;
    thread2data->cy = &cy;

    // Five threads + use rest for HDRFloat
    std::unique_ptr<std::thread> tZx(new std::thread(ThreadSqZx, ThreadZxMemory));
    std::unique_ptr<std::thread> tZy(new std::thread(ThreadSqZy, ThreadZyMemory));
    std::unique_ptr<std::thread> t1(new std::thread(Thread1, Thread1Memory, threadZxdata, threadZydata, ThreadZxMemory, ThreadZyMemory));
    std::unique_ptr<std::thread> t2(new std::thread(Thread2, Thread2Memory));

    SetThreadAffinityMask(GetCurrentThread(), 0x1 << 3);
    SetThreadAffinityMask(tZx->native_handle(), 0x1 << 5);
    SetThreadAffinityMask(tZy->native_handle(), 0x1 << 7);
    SetThreadAffinityMask(t1->native_handle(), 0x1 << 9);
    SetThreadAffinityMask(t2->native_handle(), 0x1 << 11);

    ThreadZxData* expectedZx = nullptr;
    ThreadZyData* expectedZy = nullptr;
    Thread1Data* expected1 = nullptr;
    Thread2Data* expected2 = nullptr;

    bool done1 = false;
    bool done2 = false;

    zx = cx;
    zy = cy;
    zx2 = zx * 2;
    zy2 = zy * 2;

    thread2data->zy = zy;

    T zxCopy;
    T zyCopy;
    bool periodicity_should_break = false;

    static const T HighOne = T{ 1.0 };
    static const T HighTwo = T{ 2.0 };
    static const T TwoFiftySix = T(256);

    for (size_t i = 0; i < m_NumIterations; i++)
    {
        if constexpr (Periodicity) {
            zxCopy = T{ zx };
            zyCopy = T{ zy };
        }

        // Start Thread 2: zy = 2 * zx * zy + cy;
        thread2data->zx2 = zx2;

        Thread2Memory->In.store(
            thread2data,
            std::memory_order_release);

        // Start Zx squaring thread
        threadZxdata->zx = zx;

        ThreadZxMemory->In.store(
            threadZxdata,
            std::memory_order_release);

        // Start Zy squaring thread
        threadZydata->zy = zy;

        ThreadZyMemory->In.store(
            threadZydata,
            std::memory_order_release);

        T double_zx = (T)zx;
        T double_zy = (T)zy;

        // Start Thread 1: zx = zx * zx - zy * zy + cx;
        thread1data->zy = zy;

        Thread1Memory->In.store(
            thread1data,
            std::memory_order_release);

        results->x.push_back(double_zx);
        results->y.push_back(double_zy);

        T norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
        bool underflow = (HdrAbs(double_zx) <= small_float ||
            HdrAbs(double_zy) <= small_float ||
            norm <= small_float);
        results->bad.push_back(underflow);

        // Note: not T.
        const SubType tempZX = (SubType)double_zx + (SubType)cx;
        const SubType tempZY = (SubType)double_zy + (SubType)cy;
        const SubType zn_size = tempZX * tempZX + tempZY * tempZY;

        if constexpr (Periodicity) {
            //function p  = findPeriodM3(c0,dx,dy,n,doCont,mpow)
            //% in ball centered on c0 find period (up to n) of nucleus
            //% use 1nd order Taylor ball
            //% M-power mpow set
            //% doCont = 0 normally
            //
            //r0 = min(abs(dx),abs(dy));
            //z = c0*0;
            //r = (r0);
            //p = [];
            //maxR = 1e5;
            //az = abs(z);
            //
            //for k=1:n
            //    r = (az+r).^mpow - az.^mpow + r0;
            //    z = z.^mpow + c0;
            //    az = abs(z);
            //    if(r>az)
            //        p = [p k];
            //        fprintf('findPeriodBallM3: N-period found: %d\n',k);
            //        if(~doCont)
            //            break;
            //        end
            //    end
            //    if(az>maxR | r>maxR)
            //        fprintf('Ball: escaping\n',k);
            //        break;
            //    end
            //end

            HdrReduce(dzdcX);
            auto dzdcX1 = HdrAbs(dzdcX);

            HdrReduce(dzdcY);
            auto dzdcY1 = HdrAbs(dzdcY);
            
            HdrReduce(zxCopy);
            auto zxCopy1 = HdrAbs(zxCopy);
            
            HdrReduce(zyCopy);
            auto zyCopy1 = HdrAbs(zyCopy);

            T n2 = max(zxCopy1, zyCopy1);

            T r0 = max(dzdcX1, dzdcY1);
            T maxRadiusHdr{ results->maxRadius };
            auto n3 = maxRadiusHdr * r0 * HighTwo;
            HdrReduce(n3);

            if (n3 > n2) {
                if constexpr (!BenchmarkMode) {
                    periodicity_should_break = true;
                }
            }
            else {
                auto dzdcXOrig = dzdcX;
                dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
            }
        }

        done1 = false;
        done2 = false;

        for (;;) {
            expected1 = thread1data;

            if (!done1 &&
                Thread1Memory->Out.compare_exchange_weak(expected1,
                                                         nullptr,
                                                         std::memory_order_release)) {
                done1 = true;
            }

            expected2 = thread2data;

            if (!done2 &&
                Thread2Memory->Out.compare_exchange_weak(expected2,
                                                         nullptr,
                                                         std::memory_order_release)) {
                done2 = true;
            }

            if (done1 && done2) {
                break;
            }
        }

        zy = thread2data->zy;

        if (zn_size > 256) {
            break;
        }

        if constexpr (Periodicity) {
            if (periodicity_should_break) {
                results->m_Periodic = true;
                break;
            }
        }
    }

    results->bad.push_back(false);
    assert(results->bad.size() == results->x.size());

    expectedZx = nullptr;
    ThreadZxMemory->In.compare_exchange_strong(expectedZx, (ThreadZxData*)0x1, std::memory_order_release);

    expectedZy = nullptr;
    ThreadZyMemory->In.compare_exchange_strong(expectedZy, (ThreadZyData*)0x1, std::memory_order_release);

    expected1 = nullptr;
    Thread1Memory->In.compare_exchange_strong(expected1, (Thread1Data*)0x1, std::memory_order_release);

    expected2 = nullptr;
    Thread2Memory->In.compare_exchange_strong(expected2, (Thread2Data*)0x1, std::memory_order_release);

    tZx->join();
    tZy->join();
    t1->join();
    t2->join();

    _aligned_free(ThreadZxMemory);
    _aligned_free(ThreadZyMemory);
    _aligned_free(Thread1Memory);
    _aligned_free(Thread2Memory);

    threadZxdata->~ThreadZxData();
    threadZydata->~ThreadZyData();
    thread1data->~Thread1Data();
    thread2data->~Thread2Data();

    _aligned_free(threadZxdata);
    _aligned_free(threadZydata);
    _aligned_free(thread1data);
    _aligned_free(thread2data);
}

bool Fractal::RequiresReferencePoints() const {
    switch (GetRenderAlgorithm()) {
        case RenderAlgorithm::Cpu64PerturbedBLA:
        case RenderAlgorithm::Cpu64PerturbedBLAHDR:
        case RenderAlgorithm::Gpu1x32Perturbed:
        case RenderAlgorithm::Gpu1x32PerturbedScaled:
        case RenderAlgorithm::GpuHDRx32PerturbedScaled:
        case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
        case RenderAlgorithm::Gpu1x64Perturbed:
        case RenderAlgorithm::Gpu1x64PerturbedBLA:
        case RenderAlgorithm::Gpu2x32Perturbed:
        case RenderAlgorithm::Gpu2x32PerturbedScaled:
        case RenderAlgorithm::GpuHDRx32PerturbedBLA:
        case RenderAlgorithm::GpuHDRx64PerturbedBLA:
            return true;
    }

    return false;
}

template<class T>
bool Fractal::IsPerturbationResultUsefulHere(size_t i) const {
    if constexpr (std::is_same<T, double>::value) {
        return
            m_PerturbationResultsDouble[i].hiX >= m_MinX &&
            m_PerturbationResultsDouble[i].hiX <= m_MaxX &&
            m_PerturbationResultsDouble[i].hiY >= m_MinY &&
            m_PerturbationResultsDouble[i].hiY <= m_MaxY &&
            (m_PerturbationResultsDouble[i].MaxIterations > m_PerturbationResultsDouble[i].x.size() ||
                m_PerturbationResultsDouble[i].MaxIterations >= m_NumIterations);
    } else if constexpr (std::is_same<T, float>::value) {
        return
            m_PerturbationResultsFloat[i].hiX >= m_MinX &&
            m_PerturbationResultsFloat[i].hiX <= m_MaxX &&
            m_PerturbationResultsFloat[i].hiY >= m_MinY &&
            m_PerturbationResultsFloat[i].hiY <= m_MaxY &&
            (m_PerturbationResultsFloat[i].MaxIterations > m_PerturbationResultsFloat[i].x.size() ||
                m_PerturbationResultsFloat[i].MaxIterations >= m_NumIterations);
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        return
            m_PerturbationResultsHDRDouble[i].hiX >= m_MinX &&
            m_PerturbationResultsHDRDouble[i].hiX <= m_MaxX &&
            m_PerturbationResultsHDRDouble[i].hiY >= m_MinY &&
            m_PerturbationResultsHDRDouble[i].hiY <= m_MaxY &&
            (m_PerturbationResultsHDRDouble[i].MaxIterations > m_PerturbationResultsHDRDouble[i].x.size() ||
                m_PerturbationResultsHDRDouble[i].MaxIterations >= m_NumIterations);
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        return
            m_PerturbationResultsHDRFloat[i].hiX >= m_MinX &&
            m_PerturbationResultsHDRFloat[i].hiX <= m_MaxX &&
            m_PerturbationResultsHDRFloat[i].hiY >= m_MinY &&
            m_PerturbationResultsHDRFloat[i].hiY <= m_MaxY &&
            (m_PerturbationResultsHDRFloat[i].MaxIterations > m_PerturbationResultsHDRFloat[i].x.size() ||
                m_PerturbationResultsHDRFloat[i].MaxIterations >= m_NumIterations);
    }
}

template<class T, class SubType>
PerturbationResults<T>* Fractal::GetUsefulPerturbationResults() {
    std::vector<PerturbationResults<T>*> useful_results;
    std::vector<PerturbationResults<T>>* cur_array = nullptr;

    if constexpr (std::is_same<T, double>::value) {
        cur_array = &m_PerturbationResultsDouble;
    }
    else if constexpr (std::is_same<T, float>::value) {
        cur_array = &m_PerturbationResultsFloat;
    }
    else if constexpr (std::is_same<T, HDRFloat<double>>::value) {
        cur_array = &m_PerturbationResultsHDRDouble;
    }
    else if constexpr (std::is_same<T, HDRFloat<float>>::value) {
        cur_array = &m_PerturbationResultsHDRFloat;
    }

    if (!cur_array->empty()) {
        if (cur_array->size() > 64) {
            cur_array->erase(cur_array->begin());
        }

        for (size_t i = 0; i < cur_array->size(); i++) {
            if (IsPerturbationResultUsefulHere<T>(i)) {
                useful_results.push_back(&(*cur_array)[i]);
            }
        }
    }

    PerturbationResults<T>* results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }
    else {
        AddPerturbationReferencePoint<T, SubType>();

        results = &(*cur_array)[cur_array->size() - 1];
    }

    return results;
}

template<class SrcT, class DestT>
PerturbationResults<DestT> *Fractal::CopyUsefulPerturbationResults(
    PerturbationResults<SrcT> &src_array)
{
    if constexpr (std::is_same<SrcT, double>::value) {
        m_PerturbationResultsFloat.push_back({});
        auto* dest = &m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1];
        dest->Copy(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, float>::value) {
        return nullptr;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<double>>::value) {
        m_PerturbationResultsHDRFloat.push_back({});
        auto* dest = &m_PerturbationResultsHDRFloat[m_PerturbationResultsHDRFloat.size() - 1];
        dest->Copy(src_array);
        return dest;
    }
    else if constexpr (std::is_same<SrcT, HDRFloat<float>>::value) {
        m_PerturbationResultsFloat.push_back({});
        auto* dest = &m_PerturbationResultsFloat[m_PerturbationResultsFloat.size() - 1];
        dest->Copy(src_array);
        return dest;
    }
    else {
        return nullptr;
    }
}

template<class T, class SubType>
void Fractal::CalcGpuPerturbationFractalBLA(bool MemoryOnly) {
    auto* results = GetUsefulPerturbationResults<T, SubType>();

    BLAS<T> blas(*results);
    blas.Init(results->x.size(), T(results->maxRadius));

    uint32_t err =
        m_r.InitializeMemory(m_CurIters.m_Width,
                             m_CurIters.m_Height);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory();

    MattCoords cx2{}, cy2{}, dx2{}, dy2{};
    MattCoords centerX2{}, centerY2{};

    FillGpuCoords(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    MattPerturbResults<T> gpu_results{
        results->x.size(),
        results->x.data(),
        results->y.data(),
        results->bad.data() };

    // all sizes should be the same anyway just pick one
    gpu_results.size = results->x.size();

    auto result = m_r.RenderPerturbBLA(m_RenderAlgorithm,
        (uint32_t*)m_CurIters.m_ItersMemory,
        &gpu_results,
        &blas,
        cx2,
        cy2,
        dx2,
        dy2,
        centerX2,
        centerY2,
        (uint32_t)m_NumIterations,
        m_IterationPrecision);

    DrawFractal(MemoryOnly);

    if (result) {
        MessageBoxCudaError(result);
    }
}

template<class T, class SubType, class T2, class SubType2>
void Fractal::CalcGpuPerturbationFractalScaledBLA(bool MemoryOnly) {
    auto* results = GetUsefulPerturbationResults<T, SubType>();
    auto* results2 = CopyUsefulPerturbationResults<T, T2>(*results);

    BLAS<T> blas(*results);
    blas.Init(results->x.size(), T(results->maxRadius));

    uint32_t err =
        m_r.InitializeMemory(m_CurIters.m_Width,
                             m_CurIters.m_Height);
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory();

    MattCoords cx2{}, cy2{}, dx2{}, dy2{};
    MattCoords centerX2{}, centerY2{};

    FillGpuCoords(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    MattPerturbResults<T> gpu_results{
        results->x.size(),
        results->x.data(),
        results->y.data(),
        results->bad.data() };

    // all sizes should be the same anyway just pick one
    gpu_results.size = results->x.size();

    MattPerturbResults<T2> gpu_results2{
        results2->x.size(),
        results2->x.data(),
        results2->y.data(),
        results2->bad.data() };

    gpu_results2.size = results2->x.size();

    if (gpu_results.size != gpu_results2.size) {
        ::MessageBox(NULL, L"Mismatch on size", L"", MB_OK);
        return;
    }

    auto result = m_r.RenderPerturbBLA<T>(m_RenderAlgorithm,
        (uint32_t*)m_CurIters.m_ItersMemory,
        &gpu_results,
        &gpu_results2,
        &blas,
        cx2,
        cy2,
        dx2,
        dy2,
        centerX2,
        centerY2,
        (uint32_t)m_NumIterations,
        m_IterationPrecision);

    DrawFractal(MemoryOnly);

    if (result) {
        MessageBoxCudaError(result);
    }
}

void Fractal::MessageBoxCudaError(uint32_t result) {
    wchar_t error[256];
    wsprintf(error, L"Error from cuda: %u\n", result);
    ::MessageBox(NULL, error, L"", MB_OK);
}

bool Fractal::CalcPixelRow_Exp(unsigned int *rowBuffer, size_t row)
{
    double dy = (Convert<HighPrecision, double>(m_MaxY) - Convert<HighPrecision, double>(m_MinY)) / ((double)m_ScrnHeight);
    double dx = (Convert<HighPrecision, double>(m_MaxX) - Convert<HighPrecision, double>(m_MinX)) / ((double)m_ScrnWidth);

    double cx = Convert<HighPrecision, double>(m_MinX);
    double cy = Convert<HighPrecision, double>(m_MaxY) - dy * ((double)row);
    double zx, zy;
    double term1, term2, term3, term4, term5;
    unsigned int i;

    // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
    // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
    for (size_t x = 0; x < m_ScrnWidth; x++)
    { // Calc Pixel
        zx = cx;
        zy = cy;
        for (i = 0; i < m_NumIterations; i++)
        { // x^4+4*I*x^3*y-6*x^2*y^2-4*I*x*y^3+y^4
            term1 = zx * zx * zx * zx;
            term2 = 4 * zx * zx * zx * zy;
            term3 = -6.0 * zx * zx * zy * zy;
            term4 = -4.0 * zx * zy * zy * zy;
            term5 = zy * zy * zy * zy;
            if (term1 - term3 + term5 > 10) break;
            zy = term2 + term4;
            zx = term1 + term3 + term5;
            zx += cx;
            zy += cy;
        }
        cx += dx;
        if (i == m_NumIterations)
        {
            *rowBuffer = 0;
        }
        else
        {
            *rowBuffer = i;
        }

        rowBuffer++;
    }

    return true;
}


void Fractal::CalcPixelRow_Multi(unsigned int* rowBuffer, size_t row)
{
    HighPrecision dy = (m_MaxY - m_MinY) / ((HighPrecision)m_ScrnHeight);
    HighPrecision dx = (m_MaxX - m_MinX) / ((HighPrecision)m_ScrnWidth);

    HighPrecision cx = m_MinX;
    HighPrecision cy = m_MaxY - dy * ((HighPrecision)row);
    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    unsigned int i;

    // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
    // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
    for (int x = 0; x < m_ScrnWidth; x++)
    { // Calc Pixel
        zx = cx;
        zy = cy;
        for (i = 0; i < m_NumIterations; i++)
        { // x^2+2*I*x*y-y^2
            zx2 = zx * zx;
            zy2 = zy * zy;
            if ((zx2 + zy2) > 4) break;
            zy = 2 * zx * zy;
            zx = zx2 - zy2;
            zx += cx;
            zy += cy;
        }
        cx += dx;
        *rowBuffer = 0;
        *rowBuffer = i;

        rowBuffer++;
    }
}

bool Fractal::CalcPixelRow_C(unsigned int* rowBuffer, size_t row)
{
    double dy = (Convert<HighPrecision, double>(m_MaxY) - Convert<HighPrecision, double>(m_MinY)) / ((double)m_ScrnHeight);
    double dx = (Convert<HighPrecision, double>(m_MaxX) - Convert<HighPrecision, double>(m_MinX)) / ((double)m_ScrnWidth);

    double cx = Convert<HighPrecision, double>(m_MinX);
    double cy = Convert<HighPrecision, double>(m_MaxY) - dy * ((double)row);
    double zx, zy;
    double zx2, zy2;
    unsigned int i;

    // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
    // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
    for (size_t x = 0; x < m_ScrnWidth; x++)
    { // Calc Pixel
        zx = cx;
        zy = cy;
        for (i = 0; i < m_NumIterations; i++)
        { // x^2+2*I*x*y-y^2
            zx2 = zx * zx;
            zy2 = zy * zy;
            if ((zx2 + zy2) > 4) break;
            zy = 2 * zx * zy;
            zx = zx2 - zy2;
            zx += cx;
            zy += cy;
        }
        cx += dx;
        *rowBuffer = i;
        rowBuffer++;
    }

    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Find the maximum number of iterations used
// in the fractal.
//////////////////////////////////////////////////////////////////////////////
size_t Fractal::FindMaxItersUsed(void) const
{
    size_t prevMaxIters = 0;
    int x, y;

    for (y = 0; y < m_ScrnHeight; y++)
    {
        for (x = 0; x < m_ScrnWidth; x++)
        {
            if (m_CurIters.m_ItersArray[y][x] > prevMaxIters)
            {
                prevMaxIters = m_CurIters.m_ItersArray[y][x];
            }
        }
    }

    return prevMaxIters;
}

//////////////////////////////////////////////////////////////////////////////
// Saves the current fractal as a bitmap to the given file.
// If halfImage is true, a bitmap with half the dimensions of the current
// fractal is saved instead.  Thus, 1024x768 is resized to 512x384.
//////////////////////////////////////////////////////////////////////////////

Fractal::CurrentFractalSave::CurrentFractalSave(
    std::wstring filename_base,
    Fractal& fractal)
    : m_FilenameBase(filename_base),
    m_Fractal(fractal),
    m_ScrnWidth(fractal.m_ScrnWidth),
    m_ScrnHeight(fractal.m_ScrnHeight),
    m_GpuAntialiasing(fractal.m_GpuAntialiasing),
    m_NumIterations(fractal.m_NumIterations),
    m_PaletteRotate(fractal.m_PaletteRotate),
    m_PaletteDepthIndex(fractal.m_PaletteDepthIndex),
    m_WhichPalette(fractal.m_WhichPalette),
    m_CurIters(std::move(fractal.m_CurIters)) {

    fractal.GetIterMemory();

    for (size_t i = 0; i < Fractal::Palette::Num; i++) {
        m_PalR[i] = fractal.m_PalR[i];
        m_PalG[i] = fractal.m_PalG[i];
        m_PalB[i] = fractal.m_PalB[i];

        m_PalIters[i] = fractal.m_PalIters[i];
    }

    m_Thread = nullptr;
    m_Destructable = false;
}

Fractal::CurrentFractalSave::~CurrentFractalSave() {
    if (m_Thread) {
        m_Thread->join();
    }
}

void Fractal::CurrentFractalSave::StartThread() {
    assert(m_Thread == nullptr);
    m_Thread = std::unique_ptr<std::thread>(new std::thread(&Fractal::CurrentFractalSave::Run, this));
}

void Fractal::CurrentFractalSave::Run() {
    //CBitmapWriter bmpWriter;
    int ret;
    //unsigned char *data = new unsigned char[m_ScrnWidth * m_ScrnHeight * 3];
    double acc_r, acc_b, acc_g;
    size_t input_x, input_y;
    size_t output_x, output_y;
    size_t numIters;

    std::wstring filename_png;

    if (m_FilenameBase != L"") {
        wchar_t temp[512];
        wsprintf(temp, L"%s", m_FilenameBase.c_str());
        filename_png = std::wstring(temp) + std::wstring(L".png");
        if (Fractal::FileExists(filename_png.c_str())) {
            ::MessageBox(NULL, L"Not saving, file exists", L"", MB_OK);
            return;
        }
    }
    else {
        size_t i = 0;
        do {
            wchar_t temp[512];
            wsprintf(temp, L"output%05d", i);
            filename_png = std::wstring(temp) + std::wstring(L".png");
            i++;
        } while (Fractal::FileExists(filename_png.c_str()));
    }

    WPngImage image((int)m_ScrnWidth, (int)m_ScrnHeight, WPngImage::Pixel16(0, 0, 0));

    for (output_y = 0; output_y < m_ScrnHeight; output_y++)
    {
        for (output_x = 0; output_x < m_ScrnWidth; output_x++)
        {
            acc_r = 0;
            acc_g = 0;
            acc_b = 0;

            for (input_x = output_x * m_GpuAntialiasing;
                input_x < (output_x + 1) * m_GpuAntialiasing;
                input_x++) {
                for (input_y = output_y * m_GpuAntialiasing;
                    input_y < (output_y + 1) * m_GpuAntialiasing;
                    input_y++) {

                    numIters = m_CurIters.m_ItersArray[input_y][input_x];
                    if (numIters < m_NumIterations)
                    {
                        numIters += m_PaletteRotate;
                        if (numIters >= MAXITERS) {
                            numIters = MAXITERS - 1;
                        }

                        auto palIndex = numIters % m_PalIters[m_WhichPalette][m_PaletteDepthIndex];

                        acc_r += m_PalR[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                        acc_g += m_PalG[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                        acc_b += m_PalB[m_WhichPalette][m_PaletteDepthIndex][palIndex];
                    }
                }
            }

            acc_r /= m_GpuAntialiasing * m_GpuAntialiasing;
            acc_g /= m_GpuAntialiasing * m_GpuAntialiasing;
            acc_b /= m_GpuAntialiasing * m_GpuAntialiasing;

            //if (index > MAXITERS) {
            //    index = MAXITERS - 1;
            //}

            //data[i] = (unsigned char)acc_r;
            //i++;
            //data[i] = (unsigned char)acc_g;
            //i++;
            //data[i] = (unsigned char)acc_b;
            //i++;

            image.set((int)output_x,
                (int)output_y,
                WPngImage::Pixel16((uint16_t)acc_r, (uint16_t)acc_g, (uint16_t)acc_b));
        }
    }

    m_Fractal.ReturnIterMemory(std::move(m_CurIters));

    //setup converter deprecated
    //using convert_type = std::codecvt_utf8<wchar_t>;
    //std::wstring_convert<convert_type, wchar_t> converter;
    ////use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    //const std::string filename_png_c = converter.to_bytes(filename_png);

    std::string filename_png_c;
    std::transform(filename_png.begin(), filename_png.end(), std::back_inserter(filename_png_c), [](wchar_t c) {
        return (char)c;
    });

    ret = image.saveImage(filename_png_c, WPngImage::PngFileFormat::kPngFileFormat_RGBA16);

    m_Destructable = true;
    return;
}

int Fractal::SaveCurrentFractal(const std::wstring filename_base)
{
    for (;;) {
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        GlobalMemoryStatusEx(&statex);

        if (m_FractalSavesInProgress.size() > 32 || statex.dwMemoryLoad > 90) {
            CleanupThreads();
            Sleep(100);
        }
        else {
            auto newPtr = std::make_unique<CurrentFractalSave>(filename_base, *this);
            m_FractalSavesInProgress.push_back(std::move(newPtr));
            m_FractalSavesInProgress.back()->StartThread();
            break;
        }
    }
    return 0;
}

void Fractal::CleanupThreads() {
    for (size_t i = 0; i < m_FractalSavesInProgress.size(); i++) {
        auto& it = m_FractalSavesInProgress[i];
        if (it->m_Destructable) {
            m_FractalSavesInProgress.erase(m_FractalSavesInProgress.begin() + i);
            break;
        }
    }
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
int Fractal::SaveHiResFractal(const std::wstring filename)
{
    //CBitmapWriter bmpWriter;

    size_t OldScrnWidth = m_ScrnWidth;
    size_t OldScrnHeight = m_ScrnHeight;

    // Set us up with high res
    ResetDimensions(16384, 16384);
    SquareCurrentView();

    // Calculate the high res image.
    // Do it in memory! :)
    CalcFractal(true);

    // Save the bitmap.
    int ret = SaveCurrentFractal(filename);

    // Back to the previous res.
    ResetDimensions(OldScrnWidth, OldScrnHeight);

    return ret;
}

HighPrecision Fractal::Benchmark(size_t numIters, size_t& milliseconds)
{
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (RequiresReferencePoints()) {
        AddPerturbationReferencePoint<double, double>();
    }

    if (!bm.StartTimer()) {
        return {};
    }

    CalcFractal(true);
    auto result = bm.StopTimer(milliseconds);
    bm.BenchmarkFinish();
    return result;
}

template<class T, class SubType>
HighPrecision Fractal::BenchmarkReferencePoint(size_t numIters, size_t& milliseconds) {
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (!bm.StartTimer()) {
        return {};
    }

    AddPerturbationReferencePoint<T, SubType, true>();

    auto result = bm.StopTimerNoIters<T>(milliseconds);
    bm.BenchmarkFinish();
    return result;
}

template HighPrecision Fractal::BenchmarkReferencePoint<float, float>(size_t numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<double, double>(size_t numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<HDRFloat<double>, double>(size_t numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<HDRFloat<float>, float>(size_t numIters, size_t& milliseconds);

HighPrecision Fractal::BenchmarkThis(size_t& milliseconds) {
    BenchmarkData bm(*this);

    if (!bm.StartTimer()) {
        return {};
    }

    ChangedMakeDirty();
    CalcFractal(true);

    return bm.StopTimer(milliseconds);
}

Fractal::BenchmarkData::BenchmarkData(Fractal& fractal) :
    fractal(fractal) {}

void Fractal::BenchmarkData::BenchmarkSetup(size_t numIters) {
    prevScrnWidth = fractal.m_ScrnWidth;
    prevScrnHeight = fractal.m_ScrnHeight;

    fractal.ClearPerturbationResults();
    fractal.SetNumIterations(numIters);
    fractal.ResetDimensions(500, 500, 1);
    //fractal.SetIterationPrecision(1);
    //fractal.RecenterViewCalc(-.1, -.1, .1, .1);
    fractal.RecenterViewCalc(-1.5, -.75, 1, .75);
}

bool Fractal::BenchmarkData::StartTimer() {
    startTime.QuadPart = 0;
    endTime.QuadPart = 1;
    if (QueryPerformanceFrequency(&freq) == 0) {
        return false;
    }

    QueryPerformanceCounter(&startTime);
    return true;
}

HighPrecision Fractal::BenchmarkData::StopTimer(size_t& milliseconds) {
    QueryPerformanceCounter(&endTime);

    uint64_t freq64 = freq.QuadPart;
    uint64_t startTime64 = startTime.QuadPart;
    uint64_t endTime64 = endTime.QuadPart;
    uint64_t totalIters = fractal.FindTotalItersUsed();

    uint64_t deltaTime = endTime64 - startTime64;
    HighPrecision timeTaken = (HighPrecision)((HighPrecision)deltaTime / (HighPrecision)freq64);
    milliseconds = (size_t)(timeTaken * HighPrecision{ 1000 }).convert_to<size_t>();
    return (HighPrecision)(totalIters / timeTaken) / 1000000.0;
}

template<class T>
HighPrecision Fractal::BenchmarkData::StopTimerNoIters(size_t &milliseconds) {
    QueryPerformanceCounter(&endTime);

    uint64_t freq64 = freq.QuadPart;
    uint64_t startTime64 = startTime.QuadPart;
    uint64_t endTime64 = endTime.QuadPart;
    uint64_t totalIters = fractal.GetNumIterations();

    uint64_t deltaTime = endTime64 - startTime64;
    HighPrecision timeTaken = (HighPrecision)((HighPrecision)deltaTime / (HighPrecision)freq64);
    milliseconds = (size_t)(timeTaken * HighPrecision{ 1000 }).convert_to<size_t>();
    return (HighPrecision)(totalIters / timeTaken) / 1000000.0;
}

void Fractal::BenchmarkData::BenchmarkFinish() {

    fractal.ResetDimensions(prevScrnWidth, prevScrnHeight);
    fractal.View(0);
    fractal.ChangedMakeDirty();
}

// This function is used for benchmarking.
// Note that it does not return correct results
// if there are any pixels on the screen that took
// zero iterations to escape from.
__int64 Fractal::FindTotalItersUsed(void)
{
    __int64 numIters = 0;
    int x, y;
    for (y = 0; y < m_ScrnHeight; y++)
    {
        for (x = 0; x < m_ScrnWidth; x++)
        {
            numIters += (__int64)m_CurIters.m_ItersArray[y][x];
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
template<bool IncludeGpuAntialiasing>
HighPrecision Fractal::XFromScreenToCalc(HighPrecision x)
{
    size_t aa = (IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1);
    HighPrecision OriginX = (HighPrecision)(m_ScrnWidth * aa) / (m_MaxX - m_MinX) * -m_MinX;
    return (x - OriginX) * (m_MaxX - m_MinX) / (m_ScrnWidth * aa);
}

template<bool IncludeGpuAntialiasing>
HighPrecision Fractal::YFromScreenToCalc(HighPrecision y)
{
    size_t aa = (IncludeGpuAntialiasing ? GetGpuAntialiasing() : 1);
    HighPrecision OriginY = (HighPrecision)(m_ScrnHeight * aa) / (m_MaxY - m_MinY) * m_MaxY;
    return -(y - OriginY) * (m_MaxY - m_MinY) / (m_ScrnHeight * aa);
}

HighPrecision Fractal::XFromCalcToScreen(HighPrecision x)
{
    return (x - m_MinX) * ((HighPrecision)m_ScrnWidth / (m_MaxX - m_MinX));
}

HighPrecision Fractal::YFromCalcToScreen(HighPrecision y)
{
    return (HighPrecision)m_ScrnHeight - (y - m_MinY) * ((HighPrecision)m_ScrnHeight / (m_MaxY - m_MinY));
}

void Fractal::ClientInitializeServers(void)
{ // Send the array to the servers
    int i, j;
    for (i = 0; i < MAXSERVERS; i++)
    {
        if (m_SetupData.m_UseThisServer[i] == 'n')
        {
            continue;
        }

        if (m_ClientMainNetwork[i]->CreateConnection(m_SetupData.m_ServerIPs[i], PERM_PORTNUM) == true)
        { // Send preliminary handshake string
            char temp[512];
            strcpy(temp, "Initialize 1.2 step 1");
            m_ClientMainNetwork[i]->SendData(temp, 512);

            // Read reply.
            char ack[512];
            m_ClientMainNetwork[i]->ReadData(ack, 512);

            // Check if reply is correct.
            if (strcmp(ack, "Fractal Server 1.2") == 0)
            {
                strcpy(temp, "Initialize 1.2 step 2");
                m_ClientMainNetwork[i]->SendData(temp, 512);

                // Tell the server who is processing what
                for (j = 0; j < BrokenMaxFractalSize; j++)
                {
                    m_ClientMainNetwork[i]->BufferedSendByte(m_ProcessPixelRow[j]);
                }
                m_ClientMainNetwork[i]->BufferedSendFlush();

                // Tell the server which parts of the fractal it is supposed to process
                sprintf(temp, "%c", 'b' + i);
                m_ClientMainNetwork[i]->SendData(temp, 512);
            }
            else // Reply was unknown
            {
                m_NetworkRender = '0';
                m_ClientMainNetwork[i]->ShutdownConnection();
            }
        }
        else // The connection failed.
        {
            m_NetworkRender = '0';
            m_ClientMainNetwork[i]->ShutdownConnection();
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
// Fills the ProcessPixel array with a workload to send to all the connected
// computers.  '0' = local (no network), 'a' = client, 'b' = server1, c = server2
// and so forth.  If we are acting as a server ourselves, we assign 'S' as a
// placeholder until we download the actual workload from the client.
//////////////////////////////////////////////////////////////////////////////
void Fractal::NetworkCreateWorkload(void)
{
    if (m_NetworkRender == '0') // Do everything locally
    {
        int i;
        for (i = 0; i < BrokenMaxFractalSize; i++)
        {
            m_ProcessPixelRow[i] = '0';
        }
    }
    else if (m_NetworkRender == 'a')
    {
        int i;

        // Make it so the server does some of the work.
        int total = m_SetupData.m_WorkClient;
        for (i = 0; i < MAXSERVERS; i++)
        {
            if (m_SetupData.m_UseThisServer[i] == 'n')
            {
                continue;
            }

            total += m_SetupData.m_WorkServers[i];
        }

        for (i = 0; i < BrokenMaxFractalSize; i++) // Default to having the client do everything
        {
            m_ProcessPixelRow[i] = 'a';
        }

        // Make sure the numbers jibe
        if (total == 100)
        {
            for (i = 0; i < MAXSERVERS; i++)
            {
                if (m_SetupData.m_UseThisServer[i] == 'n')
                {
                    continue;
                }

                double counter;
                for (counter = 0; counter < BrokenMaxFractalSize; counter += 100.0 / m_SetupData.m_WorkServers[i])
                { // If this row is already assigned, move on to the next one.
                    while (m_ProcessPixelRow[(int)counter] != 'a')
                    {
                        counter++;
                    }

                    // Don't overwrite other memory!
                    if (counter >= BrokenMaxFractalSize)
                    {
                        break;
                    }

                    // Assign this row to this server.
                    m_ProcessPixelRow[(int)counter] = (char)(i + 'b');
                }
            }
        }

        //    char temp[101];
        //    for (i = 0; i < 100; i++)
        //    { temp[i] = m_ProcessPixelRow[i]; }
        //    temp[100] = 0;
        //    ::MessageBox (NULL, temp, "", MB_OK);
    }
    else if (m_NetworkRender == 'S')
    {
        int i;
        for (i = 0; i < BrokenMaxFractalSize; i++)      // Placeholder until the server
        {
            m_ProcessPixelRow[i] = 'S';
        } // gets its actual workload from the client
    }
}

//////////////////////////////////////////////////////////////////////////////
// Client specific functions.
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Connects to the specified fractal server, and sends it the data
// necessary for it to begin calculating.
//////////////////////////////////////////////////////////////////////////////
void Fractal::ClientCreateSubConnections(void)
{
    int i;
    for (i = 0; i < MAXSERVERS; i++)
    {
        if (m_SetupData.m_UseThisServer[i] == 'n')
        {
            continue;
        }

        if (m_ClientSubNetwork[i]->CreateConnection(m_SetupData.m_ServerIPs[i], PORTNUM) == false)
        {
            ::MessageBox(NULL, L"Failure establishing connection.", L"", MB_OK);
            m_NetworkRender = '0'; // Fall back to local only
            m_ClientSubNetwork[i]->ShutdownConnection();
            return;
        }

        // Send the remote server a chunk of data describing
        // the work that needs to be done.
        char data[512];
        // TODO: support high precision
        //sprintf(data, "%c %d %d %d %.30lf %.30lf %.30lf %.30lf",
        //    m_RenderAlgorithm, m_NumIterations, m_ScrnWidth, m_ScrnHeight,
        //    m_MinX, m_MaxX, m_MinY, m_MaxY);

        if (m_ClientSubNetwork[i]->SendData(data, 512) == -1)
        {
            ::MessageBox(NULL, L"Failure sending initial data.", L"", MB_OK);
            m_NetworkRender = '0'; // Fall back to local only
            m_ClientSubNetwork[i]->ShutdownConnection();
            return;
        }
    }

    return;
}

void Fractal::ClientShutdownSubConnections(void)
{
    int i;
    for (i = 0; i < MAXSERVERS; i++)
    {
        if (m_SetupData.m_UseThisServer[i] == 'n')
        {
            continue;
        }

        m_ClientSubNetwork[i]->ShutdownConnection();
    }
}

void Fractal::ClientSendQuitMessages(void)
{
    char temp[512];
    strcpy(temp, "Stop Calculating");

    int i;
    for (i = 0; i < MAXSERVERS; i++)
    {
        if (m_SetupData.m_UseThisServer[i] == 'n')
        {
            continue;
        }

        m_ClientMainNetwork[i]->SendData(temp, 512);
    }
}

//////////////////////////////////////////////////////////////////////////////
// Server specific functions.
//////////////////////////////////////////////////////////////////////////////
bool Fractal::ServerRespondToInitialization(void)
{
    char ack[512];
    strcpy(ack, "Fractal Server 1.2");

    int ret = m_ServerMainNetwork->SendData(ack, 512);
    if (ret == -1)
    {
        OutputMessage(L"Failure sending acknowledgement %d", ret);
        return false;
    }

    OutputMessage(L"Acknowledgement sent.  Awaiting initialization data...\r\n");

    return true;
}

// Read the array specifying which computers process what pixels.
void Fractal::ServerReadProcessPixelRow(void)
{
    int i;

    m_ServerMainNetwork->BufferedReadEmpty();

    for (i = 0; i < BrokenMaxFractalSize; i++)
    {
        m_ServerMainNetwork->BufferedReadByte(&m_ProcessPixelRow[i]);
    }

    OutputMessage(L"Process Pixel row: %c%c%c%c\r\n", m_ProcessPixelRow[0], m_ProcessPixelRow[1], m_ProcessPixelRow[2], m_ProcessPixelRow[3]);
}

// Read the ID of this server, assigned by the client.
void Fractal::ServerReadID(void)
{
    char temp[512];
    m_ServerMainNetwork->ReadData(temp, 512);
    m_NetworkRender = temp[0];

    OutputMessage(L"Network Render: %c\r\n", m_NetworkRender);
}

// Main server loop
void Fractal::ServerManageMainConnection(void)
{
    if (m_ServerMainNetwork->SetUpListener(m_SetupData.m_LocalIP, PERM_PORTNUM) == false)
    {
        OutputMessage(L"MainConnectionThread - SetUpListener failed on address %s:%d\r\n", m_SetupData.m_LocalIP, PERM_PORTNUM);
        return;
    }

    int ret;
    sockaddr_in sinRemote;
    for (;;)
    {
        OutputMessage(L"Awaiting primary connection...\r\n");

        if (m_ServerMainNetwork->AcceptConnection(sinRemote) == false)
        {
            OutputMessage(L"Error waiting for new connection!\r\n");
            break;
        }

        OutputMessage(L"Accepted!\r\n");

        do
        {
            ret = ServerManageMainState();
        } while (ret == 1);

        m_ServerMainNetwork->ShutdownConnection();

        if (ret == -1)
        {
            break;
        }
    }
}

// -1 = time to exit this program
// 0 = that client is done, we can wait for a new client now
// 1 = that client is not done, so we will keep running and await
//     more work to do.
int Fractal::ServerManageMainState(void)
{
    char buffer[512];
    m_ServerMainNetwork->ReadData(buffer, 512);

    OutputMessage(L"ManageMainState - Data received:\r\n");
    OutputMessage(L"%s\r\n", buffer);

    if (strcmp(buffer, "exit") == 0)
    {
        return -1;
    }      // Time to exit this program
    else if (strcmp(buffer, "done") == 0)
    {
        return 0;
    }       // The client is done, we can wait for a new client
    else if (strcmp(buffer, "Initialize 1.2 step 1") == 0)
    {
        bool ret = ServerRespondToInitialization();
        if (ret == false)
        {
            return 0;
        }
        return 1;
    }
    else if (strcmp(buffer, "Initialize 1.2 step 2") == 0)
    {
        ServerReadProcessPixelRow();
        ServerReadID();
        return 1;
    }
    else if (strcmp(buffer, "Stop Calculating") == 0)
    {
        m_StopCalculating = true;
        return 1;
    }
    else
    {
        OutputMessage(L"Unexpected data received on main connection!");
        return -1;
    }
}

void Fractal::ServerManageSubConnection(void)
{
    if (m_ServerSubNetwork->SetUpListener(m_SetupData.m_LocalIP, PORTNUM) == false)
    {
        OutputMessage(L"SubConnectionThread - SetUpListener failed on address %s:%d\r\n", m_SetupData.m_LocalIP, PORTNUM);
        return;
    }

    // Wait for connections forever, until the client tells the server to shutdown
    sockaddr_in sinRemote;
    for (;;)
    {
        OutputMessage(L"Awaiting secondary connection...\r\n");

        if (m_ServerSubNetwork->AcceptConnection(sinRemote) == false)
        {
            OutputMessage(L"Error waiting for new connection!\r\n");
            break;
        }

        OutputMessage(L"Accepted!\r\n");
        bool ret = ServerBeginCalc();
        m_ServerSubNetwork->ShutdownConnection();

        if (ret == false)
        {
            break;
        }
    }
}

bool Fractal::ServerBeginCalc(void)
{
    OutputMessage(L"Awaiting instructions...\r\n");
    char buffer[512];
    m_ServerSubNetwork->ReadData(buffer, 512);

    // TODO:
    RenderAlgorithm RenderAlgorithm = RenderAlgorithm::Cpu64;
    int ScreenWidth = 0, ScreenHeight = 0;
    uint32_t NumIters = 0;
    HighPrecision MinX = 0, MaxX = 0, MinY = 0, MaxY = 0;

    OutputMessage(L"Data received:\r\n");
    OutputMessage(L"%s\r\n", buffer);

    // Secondary connection should quit.
    if (strcmp(buffer, "exit") == 0)
    {
        return false;
    }

    // Anything else must be data for setting up a calculation.
    // TODO support high precision
    //sscanf(buffer, "%c %d %d %d %lf %lf %lf %lf",
    //    &RenderAlgorithm, &NumIters, &ScreenWidth, &ScreenHeight,
    //    &MinX, &MaxX, &MinY, &MaxY);

    OutputMessage(L"Received instructions.\r\n");
    OutputMessage(L"Interpretation:\r\n");
    OutputMessage(L"Render Mode:  %d\r\n", RenderAlgorithm);
    OutputMessage(L"NumIters:     %d\r\n", NumIters);
    OutputMessage(L"ScreenWidth:  %d\r\n", ScreenWidth);
    OutputMessage(L"ScreenHeight: %d\r\n", ScreenHeight);
    OutputMessage(L"MinX:         %.15f\r\n", Convert<HighPrecision, double>(MinX));
    OutputMessage(L"MaxX:         %.15f\r\n", Convert<HighPrecision, double>(MaxX));
    OutputMessage(L"MinY:         %.15f\r\n", Convert<HighPrecision, double>(MinY));
    OutputMessage(L"MaxY:         %.15f\r\n", Convert<HighPrecision, double>(MaxY));

    ResetDimensions(ScreenWidth, ScreenHeight);
    RecenterViewCalc(MinX, MinY, MaxX, MaxY);
    SetNumIterations(NumIters);
    SetRenderAlgorithm(RenderAlgorithm);
    CalcFractal(true);

    int x, y;
    for (y = 0; y < ScreenHeight; y++)
    {
        if (m_ProcessPixelRow[y] != m_NetworkRender)
        {
            continue;
        }

        for (x = 0; x < ScreenWidth; x++)
        {
            if (NumIters >= 65536)
            {
                m_ServerSubNetwork->BufferedSendLong(m_CurIters.m_ItersArray[y][x]);
            }
            else
            {
                m_ServerSubNetwork->BufferedSendShort((unsigned short)m_CurIters.m_ItersArray[y][x]);
            }
        }
    }

    m_ServerSubNetwork->BufferedSendFlush();

    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Server threads
//////////////////////////////////////////////////////////////////////////////
DWORD WINAPI Fractal::ServerManageMainConnectionThread(void *fractal)
{
    ((Fractal *)fractal)->ServerManageMainConnection();
    return 0;
}

DWORD WINAPI Fractal::ServerManageSubConnectionThread(void *fractal)
{ // Lower our priority a bit--this isn't mission critical :)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    ((Fractal *)fractal)->ServerManageSubConnection();
    return 0;
}

void DefaultOutputMessage(const wchar_t *, ...)
{
}

bool Fractal::IsDownControl(void)
{
    return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000);
    //return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000) &&
    //    ((m_hWnd != NULL && GetForegroundWindow() == m_hWnd) ||
    //    (m_hWnd == NULL));
};

void Fractal::CheckForAbort(void)
{
    POINT pt;
    GetCursorPos(&pt);
    int OrgX = pt.x;
    int OrgY = pt.y;

    for (;;)
    {
        if (m_AbortThreadQuitFlag == true)
        {
            break;
        }

        Sleep(250);

        if (IsDownControl())
        {
            m_StopCalculating = true;
        }

        if (m_UseSensoCursor == true)
        {
            GetCursorPos(&pt);

            if (abs(pt.x - OrgX) >= 5 ||
                abs(pt.y - OrgY) >= 5)
            {
                OrgX = pt.x; // Reset to current location.
                OrgY = pt.y;
                m_StopCalculating = true;
            }
        }
    }
}

unsigned long WINAPI Fractal::CheckForAbortThread(void *fractal)
{
    ((Fractal *)fractal)->CheckForAbort();
    return 0;
}