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
#include <iostream>
#include <fstream>

#include "HDRFloatComplex.h"
#include "ATInfo.h"
#include "LAInfoDeep.h"
#include "LAReference.h"

#include "CudaDblflt.h"

void DefaultOutputMessage(const wchar_t *, ...);

Fractal::ItersMemoryContainer::ItersMemoryContainer(
    IterTypeEnum type,
    size_t width,
    size_t height,
    size_t total_antialiasing)
    : m_IterType(type),
      m_ItersMemory32(nullptr),
      m_ItersArray32(nullptr),
      m_ItersMemory64(nullptr),
      m_ItersArray64(nullptr),
      m_Width(),
      m_Height(),
      m_Total(),
      m_OutputWidth(),
      m_OutputHeight(),
      m_OutputTotal(),
      m_RoundedWidth(),
      m_RoundedHeight(),
      m_RoundedTotal(),
      m_RoundedOutputColorWidth(),
      m_RoundedOutputColorHeight(),
      m_RoundedOutputColorTotal(),
      m_Antialiasing(total_antialiasing) {

    // This array must be identical in size to OutputIterMatrix in CUDA
    m_Width = width * total_antialiasing;
    m_Height = height * total_antialiasing;
    m_Total = m_Width * m_Height;

    m_OutputWidth = width;
    m_OutputHeight = height;
    m_OutputTotal = m_OutputWidth * m_OutputHeight;

    const size_t w_block = m_Width / GPURenderer::NB_THREADS_W +
        (m_Width % GPURenderer::NB_THREADS_W != 0);
    const size_t h_block = m_Height / GPURenderer::NB_THREADS_H +
        (m_Height % GPURenderer::NB_THREADS_H != 0);

    m_RoundedWidth = w_block * GPURenderer::NB_THREADS_W;
    m_RoundedHeight = h_block * GPURenderer::NB_THREADS_H;
    m_RoundedTotal = m_RoundedWidth * m_RoundedHeight;

    if (m_IterType == IterTypeEnum::Bits32) {
        m_ItersMemory32 = std::make_unique<uint32_t[]>(m_RoundedTotal);
        m_ItersArray32 = new uint32_t * [m_RoundedHeight];
        for (size_t i = 0; i < m_RoundedHeight; i++) {
            m_ItersArray32[i] = &m_ItersMemory32[i * m_RoundedWidth];
        }
    }
    else {
        m_ItersMemory64 = std::make_unique<uint64_t[]>(m_RoundedTotal);
        m_ItersArray64 = new uint64_t * [m_RoundedHeight];
        for (size_t i = 0; i < m_RoundedHeight; i++) {
            m_ItersArray64[i] = &m_ItersMemory64[i * m_RoundedWidth];
        }
    }

    const size_t w_color_block = m_OutputWidth / GPURenderer::NB_THREADS_W_AA +
        (m_OutputWidth % GPURenderer::NB_THREADS_W_AA != 0);
    const size_t h_color_block = m_OutputHeight / GPURenderer::NB_THREADS_H_AA +
        (m_OutputHeight % GPURenderer::NB_THREADS_H_AA != 0);
    m_RoundedOutputColorWidth = w_color_block * GPURenderer::NB_THREADS_W_AA;
    m_RoundedOutputColorHeight = h_color_block * GPURenderer::NB_THREADS_H_AA;
    m_RoundedOutputColorTotal = m_RoundedOutputColorWidth * m_RoundedOutputColorHeight;
    m_RoundedOutputColorMemory = std::make_unique<Color16[]>(m_RoundedOutputColorTotal);
};

Fractal::ItersMemoryContainer::ItersMemoryContainer(Fractal::ItersMemoryContainer&& other) noexcept {
    *this = std::move(other);
}

Fractal::ItersMemoryContainer &Fractal::ItersMemoryContainer::operator=(Fractal::ItersMemoryContainer&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    m_IterType = other.m_IterType;

    m_ItersMemory32 = std::move(other.m_ItersMemory32);
    m_ItersMemory64 = std::move(other.m_ItersMemory64);

    m_ItersArray32 = other.m_ItersArray32;
    other.m_ItersArray32 = nullptr;

    m_ItersArray64 = other.m_ItersArray64;
    other.m_ItersArray64 = nullptr;

    m_Width = other.m_Width;
    m_Height = other.m_Height;
    m_Total = other.m_Total;
    
    m_OutputWidth = other.m_OutputWidth;
    m_OutputHeight = other.m_OutputHeight;
    m_OutputTotal = other.m_OutputTotal;

    m_RoundedWidth = other.m_RoundedWidth;
    m_RoundedHeight = other.m_RoundedHeight;
    m_RoundedTotal = other.m_RoundedTotal;

    m_RoundedOutputColorWidth = other.m_RoundedOutputColorWidth;
    m_RoundedOutputColorHeight = other.m_RoundedOutputColorHeight;
    m_RoundedOutputColorTotal = other.m_RoundedOutputColorTotal;
    m_RoundedOutputColorMemory = std::move(other.m_RoundedOutputColorMemory);

    m_Antialiasing = other.m_Antialiasing;

    return *this;
}

Fractal::ItersMemoryContainer::~ItersMemoryContainer() {
    m_ItersMemory32 = nullptr;
    m_ItersMemory64 = nullptr;
    m_RoundedOutputColorMemory = nullptr;

    if (m_ItersArray32) {
        delete[] m_ItersArray32;
        m_ItersArray32 = nullptr;
    }

    if (m_ItersArray64) {
        delete[] m_ItersArray64;
        m_ItersArray64 = nullptr;
    }
}

Fractal::Fractal(FractalSetupData* setupData,
    int width,
    int height,
    void(*pOutputMessage) (const wchar_t*, ...),
    HWND hWnd,
    bool UseSensoCursor) :
    m_CurIters(IterTypeEnum::Bits32, width, height, 1),
    m_RefOrbit(*this)
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
    //SetRenderAlgorithm(RenderAlgorithm::Cpu32PerturbedBLAV2HDR);
    //SetRenderAlgorithm(RenderAlgorithm::GpuHDRx32PerturbedLAv2);
    //SetRenderAlgorithm(RenderAlgorithm::GpuHDRx2x32PerturbedLAv2);
    SetRenderAlgorithm(RenderAlgorithm::AUTO);

    SetIterationPrecision(1);
    //m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed);
    m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
    //m_RefOrbit.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MT);
    m_RefOrbit.ResetGuess();

    ResetDimensions(width, height, 2);
    SetIterType(IterTypeEnum::Bits32);

    View(0);
    //View(5);
    //View(15);

    m_ChangedWindow = true;
    m_ChangedScrn = true;
    m_ChangedIterations = true;

    srand((unsigned int) time(NULL));

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

    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 0, 5));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 1, 6));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 2, 8));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 3, 12));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 4, 16));
    threads.push_back(std::make_unique<std::thread>(DefaultPaletteGen, Palette::Default, 5, 20));

    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 0, 5));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 1, 6));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 2, 8));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 3, 12));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 4, 16));
    threads.push_back(std::make_unique<std::thread>(PatrioticPaletteGen, Palette::Patriotic, 5, 20));

    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 0, 5));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 1, 6));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 2, 8));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 3, 12));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 4, 16));
    threads.push_back(std::make_unique<std::thread>(SummerPaletteGen, Palette::Summer, 5, 20));

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

    // Allocate the iterations array.
    int i;
    InitializeMemory();

    // Set up random palette.
    CreateNewFractalPalette();

    // Wait for all this shit to get done
    for (auto& it : threads) {
        it->join();
    }

    m_PaletteRotate = 0;
    m_PaletteDepthIndex = 2;
    m_PaletteAuxDepth = 0;
    UsePaletteType(Palette::Default);
    UsePalette(8);

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
        ItersMemoryContainer container(GetIterType(), m_ScrnWidth, m_ScrnHeight, total_aa);
        m_ItersMemoryStorage.push_back(std::move(container));
    }

    m_CurIters = std::move(m_ItersMemoryStorage.back());
    m_ItersMemoryStorage.pop_back();

    m_DrawThreadAtomics.resize(m_ScrnHeight);
    m_DrawOutBytes = std::make_unique<GLushort[]>(m_ScrnWidth * m_ScrnHeight * 4); // RGBA
}

uint32_t Fractal::InitializeGPUMemory() {
    if (RequiresUseLocalColor()) {
        return 0;
    }

    if (GetIterType() == IterTypeEnum::Bits32) {
        return m_r.InitializeMemory<uint32_t>(
            (uint32_t)m_CurIters.m_Width,
            (uint32_t)m_CurIters.m_Height,
            (uint32_t)m_CurIters.m_Antialiasing,
            m_PalR[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalG[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalB[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalIters[m_WhichPalette][m_PaletteDepthIndex],
            m_PaletteAuxDepth);
    }
    else {
        return m_r.InitializeMemory<uint64_t>(
            (uint32_t)m_CurIters.m_Width,
            (uint32_t)m_CurIters.m_Height,
            (uint32_t)m_CurIters.m_Antialiasing,
            m_PalR[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalG[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalB[m_WhichPalette][m_PaletteDepthIndex].data(),
            m_PalIters[m_WhichPalette][m_PaletteDepthIndex],
            m_PaletteAuxDepth);
    }
}

void Fractal::ReturnIterMemory(ItersMemoryContainer&& to_return) {
    std::unique_lock<std::mutex> lock(m_ItersMemoryStorageLock);
    m_ItersMemoryStorage.push_back(std::move(to_return));
}

void Fractal::SetCurItersMemory() {
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
{
    CleanupThreads(true);

    // Get rid of the abort thread, but only if we actually used it.
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

    if (gpu_antialiasing > 4) {
        ::MessageBox(NULL, L"You're doing it wrong.  4x is max == 16 samples per pixel", L"", MB_OK);
        gpu_antialiasing = 4;
    }

    if (m_ScrnWidth != width ||
        m_ScrnHeight != height ||
        m_GpuAntialiasing != gpu_antialiasing)
    {
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
        HighPrecision::default_precision(prec2);
        //HighPrecision::default_variable_precision_options(
        //    boost::multiprecision::variable_precision_options::preserve_target_precision);

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
    const HighPrecision& maxY,
    bool RequiresReuse)
{
    if constexpr (DigitPrecision != 0) {
        return DigitPrecision;
    }
    else {
#ifndef CONSTANT_PRECISION
        static_assert(DigitPrecision == 0, "!");
        auto deltaX = abs(maxX - minX);
        auto deltaY = abs(maxY - minY);

        int temp_expX;
        boost::multiprecision::frexp(deltaX, &temp_expX);
        int temp_expY;
        boost::multiprecision::frexp(deltaY, &temp_expY);

        double expX = temp_expX / log(10) * log(2);
        double expY = temp_expY / log(10) * log(2);
        size_t larger = (size_t)std::max(abs(expX), abs(expY));

        if (RequiresReuse) {
            larger += AuthoritativeReuseExtraPrecision;
        }
        else {
            larger += AuthoritativeMinExtraPrecision;
        }
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
        return GetPrecision(m_MinX, m_MinY, m_MaxX, m_MaxY, m_RefOrbit.RequiresReuse());
#endif
    }
}

bool Fractal::RecenterViewCalc(HighPrecision MinX, HighPrecision MinY, HighPrecision MaxX, HighPrecision MaxY)
{
    SaveCurPos();

    size_t prec = GetPrecision(MinX, MinY, MaxX, MaxY, m_RefOrbit.RequiresReuse());
    SetPrecision(prec, MinX, MinY, MaxX, MaxY);

    m_MinX = MinX;
    m_MaxX = MaxX;
    m_MinY = MinY;
    m_MaxY = MaxY;

    m_RefOrbit.ResetGuess();

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
    m_RefOrbit.ResetGuess();

    if (m_RefOrbit.RequiresReferencePoints()) {
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

            double avgiters = totaliters / ((antiRect.bottom - antiRect.top) * (antiRect.right - antiRect.left));

            for (auto y = antiRect.top; y < antiRect.bottom; y++) {
                for (auto x = antiRect.left; x < antiRect.right; x++) {
                    if (ItersArray[y][x] < avgiters) {
                        continue;
                    }

                    double sq = (double)
                        (ItersArray[y][x]) *
                        (ItersArray[y][x]);
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

                HighPrecision tempMeanX = meanX;
                HighPrecision tempMeanY = meanY;

                tempMeanX = XFromScreenToCalc(tempMeanX);
                tempMeanY = YFromScreenToCalc(tempMeanY);

                m_RefOrbit.ResetGuess(tempMeanX, tempMeanY);

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
        };

        if (GetIterType() == IterTypeEnum::Bits32) {
            lambda(m_CurIters.GetItersArray<uint32_t>());
        }
        else {
            lambda(m_CurIters.GetItersArray<uint64_t>());
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
        {
            MSG msg;
            PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);
        }

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

        bool shouldBreak = false;

        auto lambda = [&](auto** ItersArray, auto NumIterations) {
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
                        auto curiter = ItersArray[y][x];

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
                        double normalizedIters = (double)curiter / (double)NumIterations;

                        if (curiter == maxiter) {
                            normalizedIters *= normalizedIters;
                        }
                        double normalizedDist = (sqrt(distanceX * distanceX + distanceY * distanceY) / maxDistance);
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

                    guessX = XFromScreenToCalc<true>(meanX);
                    guessY = YFromScreenToCalc<true>(meanY);

                    //wchar_t temps[256];
                    //swprintf(temps, 256, L"Coords: %f %f", meanX, meanY);
                    //::MessageBox(NULL, temps, L"", MB_OK);
                }
                else {
                    shouldBreak = true;
                    return;
                }

                if (numAtLimit == antiRectWidthInt * antiRectHeightInt) {
                    ::MessageBox(NULL, L"Flat screen! :(", L"", MB_OK);
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

                guessX = XFromScreenToCalc<true>(targetX);
                guessY = YFromScreenToCalc<true>(targetY);

                if (numAtLimit == m_ScrnWidth * m_ScrnHeight * GetGpuAntialiasing() * GetGpuAntialiasing()) {
                    ::MessageBox(NULL, L"Flat screen! :(", L"", MB_OK);
                    shouldBreak = true;
                    return;
                }
            }
        };

        if (GetIterType() == IterTypeEnum::Bits32) {
            lambda(m_CurIters.GetItersArray<uint32_t>(), GetNumIterations<uint32_t>());
        }
        else {
            lambda(m_CurIters.GetItersArray<uint64_t>(), GetNumIterations<uint64_t>());
        }

        if (shouldBreak) {
            break;
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

Fractal::PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision ptX,
    HighPrecision ptY,
    HighPrecision zoomFactor)
    : ptX(ptX),
      ptY(ptY),
      zoomFactor(zoomFactor) {
    minX = ptX - (HighPrecision{ factor } / zoomFactor);
    // minX + (HighPrecision{ factor } / zoomFactor) = ptX;
    // (HighPrecision{ factor } / zoomFactor) = ptX - minX;
    // HighPrecision{ factor } = (ptX - minX) * zoomFactor;
    // HighPrecision{ factor } / (ptX - minX) = zoomFactor;

    minY = ptY - (HighPrecision{ factor } / zoomFactor);

    maxX = ptX + (HighPrecision{ factor } / zoomFactor);
    // maxX - ptX = (HighPrecision{ factor } / zoomFactor);
    // zoomFactor * (maxX - ptX) = (HighPrecision{ factor });
    // zoomFactor = (HighPrecision{ factor }) / (maxX - ptX);

    maxY = ptY + (HighPrecision{ factor } / zoomFactor);
}

Fractal::PointZoomBBConverter::PointZoomBBConverter(
    HighPrecision minX,
    HighPrecision minY,
    HighPrecision maxX,
    HighPrecision maxY) :
    minX(minX),
    minY(minY),
    maxX(maxX),
    maxY(maxY) {
    ptX = (minX + maxX) / HighPrecision(2);
    ptY = (minY + maxY) / HighPrecision(2);

    auto zf1 = HighPrecision{ factor } / (ptX - minX);
    auto zf2 = HighPrecision{ factor } / (ptY - minY);
    auto zf3 = HighPrecision{ factor } / (maxX - ptX);
    auto zf4 = HighPrecision{ factor } / (maxY - ptY);
    zoomFactor = std::min(std::min(zf1, zf2), std::min(zf3, zf4));
}

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
        minY = HighPrecision{ "0.04289211262806512836473285627858318635734695759291867302112730624188941270466703058975670804976478935827994844038526618063053858" };
        maxX = HighPrecision{ "-1.763399177066752695854220120818493394874764715075525070697085355870272824868052108014289980411203646967653925705407102169" };
        maxY = HighPrecision{ "0.04289211262806512836473285627858318635734695759291867302112731461463330922125985949917962768812338316745717506303752530265831841" };
        SetNumIterations<IterTypeFull>(196608);
        break;

    case 2:
    {
        // Limits of 4x32 GPU
        minX = HighPrecision{ "-1.768969486867357972775564951275461551052751499509997185691881950786253743769635708375905775793656954725307354460920979983" };
        minY = HighPrecision{ "0.05699280690304670893115636892860647833175463644922652375916712719872599382335388157040896288795946562522749757591414246314107544" };
        maxX = HighPrecision{ "-1.768969486867357972775564950929487934553496494563941335911085292699250368065865432159590460057564657941788398574759610411" };
        maxY = HighPrecision{ "0.05699280690304670893115636907127975355127952306391692141273804706041783710937876987367435542131127321801128526034375237132904264" };
        SetNumIterations<IterTypeFull>(196608);
        break;
    }

    case 3:
        // Limit of 1x32 + Perturbation with scaling
        minX = HighPrecision{ "-1.44656726997022737062295806977817803829443061688656117623800256312303751202920456713778693247098684334495241572095045" };
        minY = HighPrecision{ "7.64163245263840450044318279619820153508302789530826527979427966642829357717061175013838301474813332434725222956221212e-18" };
        maxX = HighPrecision{ "-1.44656726997022737062295806977817803829442603529959040638812674667522697557115287788808403561611427018141845213679032" };
        maxY = HighPrecision{ "7.641632452638404500443184705192772689187142818828186336651801203615669784193475322289855499574866715163283993737498e-18" };
        SetNumIterations<IterTypeFull>(196608);
        break;

    case 4:
        minX = HighPrecision{ "-1.44656726997022737062295806977817803829442766062231034469821437680324515234695809677735314398112720680340773533658285" };
        minY = HighPrecision{ "7.64163245263840450044318315665495782554700906545628099403337428848294779227472963485579985013605880788857454558128871e-18" };
        maxX = HighPrecision{ "-1.44656726997022737062295806977817803829442766062231034469821437680324515234695809677735191376844462193327526231811052" };
        maxY = HighPrecision{ "7.64163245263840450044318315665495782554700906545628099403337428848294830486334737855168838056042228491124080530744489e-18" };
        SetNumIterations<IterTypeFull>(196608);
        break;

    case 5:
        minX = HighPrecision{ "-0.548205748070475708458212567546733029376699278637323932787860368510369107357663992406257053055723741951365216836802745" };
        minY = HighPrecision{ "-0.577570838903603842805108982201850558675551730113738529364698265412779545002113555345006591372870167386914495276370477" };
        maxX = HighPrecision{ "-0.54820574807047570845821256754673302937669927060844097486102930067962289200412659019319306589187062772276993544341295" };
        maxY = HighPrecision{ "-0.577570838903603842805108982201850558675551726802772104952059640378694274662197291893029522164691495936927144187595881" };
        SetNumIterations<IterTypeFull>(4718592);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 1); // TODO
        break;

    case 6:
        // Scale float with pixellation
        minX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030266459696312328585298881005139386870908363177752552421427177179281096147769415" };
        minY = HighPrecision{ "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033981012912620372948556514051537500942007730195548392246463251930450398477496176544" };
        maxX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030250153999905750113975818926710341658168707379760602146485062960529816708172165" };
        maxY = HighPrecision{ "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033987737087528857040088479840438460825120725713503099967399506797154756105787592431" };
        SetNumIterations<IterTypeFull>(4718592);
        break;

    case 7:
        // Scaled float limit with circle
        minX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030252478540752851056038295732180048485849836719480635256962570788141443758414653" };
        minY = HighPrecision{ "0.0011175672388967686119452877936503680420978056943097961919136836510176758423423873900601464203086708258487998008460089102965219489403398329852555703126748646225896578863028142955526188647404920935367487584932956791" };
        maxX = HighPrecision{ "-1.62255305450955440939378327148551933698151664905869252353104459177017978418891616690380136311469569647746535255597152879870544828084030252478540752851056038295729331767859389716189380600126795460282679236765495185" };
        maxY = HighPrecision{ "0.00111756723889676861194528779365036804209780569430979619191368365101767584234238739006014642030867082584879980084600891029652194894033983298525557031267486462261733835999658166408457695262521471675884626307237220407" };
        SetNumIterations<IterTypeFull>(4718592);
        break;

    case 8:
        // Full BLA test 10^500 or so.
        minX = HighPrecision{ "-1.622553054509554409393783271485519336981516649058692523531044591770179784188916166903801363114695696477465352555971528798705448280840302524785407528510560382957296336747734660781986878861832656627910563988831196767541174987070125253828371867979558889300125126916356006345148898733695397073332691494528591084305228375267624747632537984863862572085039742052399126752593493400012297784541851439677708432244697466044124608245754997623602165370794000587151474975819819456195944759807441120696911861556241411268002822443914381727486024297400684811891998011624082906915471367511682755431368949865497918093491140476583266495952404825038512305399236173787176914358157915436929123396631868909691124127724870282812097822811900868267134047713861692077304303606674327671011614600160831982637839953075022976105199342055519124699940622577737366867995911632623421203318214991899522128043854908268532674686185750586601514622901972501486131318643684877689470464141043269863071514177452698750631448571933928477764304097795881563844836072430658397548935468503784369712597732268753575785634704765454312860644160891741730383872538137423932379916978843068382051765967108389949845267672531392701348446374503446641582470472389970507273561113864394504324475127799616327175518893129735240410917146495207044134363795295340971836770720248397522954940368114066327257623890598806915959596944154000211971821372129091151250639111325886353330990473664640985669101380578852175948040908577542590682028736944864095586894535838946146934684858500305538133386107808126063840356032468846860679224355311791807304868263772346293246639323226366331923103758639636260155222956587652848435936698081529518277516241743555175696966399377638808834127105029423149039661060794001264217559981749222317343868825318580784448831838551186024233174975961446762084960937499999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999998109" };
        minY = HighPrecision{ "0.001117567238896768611945287793650368042097805694309796191913683651017675842342387390060146420308670825848799800846008910296521948940339832985255570312674864622611835522284075672543854847255264731946382328371214009526561663174941608909925216179620747558791394265628274748675295346034899998815173442512409345653708758722585674020064853646630373608679959493334284553551589569947877966948566416572492467339583663949970046379704495147267378153033462305115439398337489620790688829995544121329470083542063031324851802574649918811841750913168721314686845259594117073877271321100948129855768148854311161320084129267363499289251825454183589219881333850776847012695765759653605564436023656201715593927689666722447079663020705803627665637015104492497741636468267388256490548881346893345366125490399643435876691530713541432028774019720460412518615867427220487428798187975218145189083789569256006253505424279834364602251632081060904568420885095239536114820931152263030281510960264774165352065681307019706534613000980870274180073599675792637642781955545684799042499349664484696664255967704566757636662588724423743582737881875921718030051382037889290489109692568753441015738409173208113123967308658979176663175143694315257820174553147123170016957035575950933366295114766019884833988769556724019671487704066470206279346884135106807605702141085672617612292451502032657624487846200365061848714912762429355703562415874601262268865127750867393241343569164011564916899299372768445538725622751921653729817821602157920626985597292441933163224073178294594360672242702797827361918819138484230729076661426426985362989832789724044998791055083335658038803274851664414571288437875273224563220948706102106988695167966092545926424587115422685794647558401757470800335648264079478256438335825103058562986733937005823236177093349397182464599609374999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999368" };
        maxX = HighPrecision{ "-1.622553054509554409393783271485519336981516649058692523531044591770179784188916166903801363114695696477465352555971528798705448280840302524785407528510560382957296336747734660781986878861832656627910563988831196767541174987070125253828371867979558889300125126916356006345148898733695397073332691494528591084305228375267624747632537984863862572085039742052399126752593493400012297784541851439677708432244697466044124608245754997623602165370794000587151474975819819456195944759807441120696911861556241411268002822443914381727486024297400684811891998011624082906915471367511682755431368949865497918093491140476583266495952404825038512305399236173787176914358157915436929101074572363325485941890852869740547207893495401847216464403190377283252284936695221620192489736296880064364736168387364407018924498548865083468021039191959297956680950735192211117294472572549916766627933448887817452943083286885583312398485175176799261059924202852929698149189055624038345817137105972343305894016069756366146900994722671059851259685904384807877457739632787467911510131153946289523428325072359141842353594673786565545080689796074914723731611730652207617489386127306710995798281376160344598476271882907175206737026029853260611423835086663823662230295980596315444662332633999288401087473357654870330977577589362848522147173440735610957887415860064397118429150006470992796414518180674956981690673824954254846780389614421277378116832539432506584941678464221661845802155038330559171689808309409001223409532423237399439602230938059938450063207955010358991806721046406987145343805044073027784664829615863680850374937599099095789048554468885707214000693385718677220557313945289575462766123240889414425105608211440697730397516835728294646134282185031089311311699258534641301626048472044700465551168161448813975766825024038553237915039062499999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999998107" };
        maxY = HighPrecision{ "0.001117567238896768611945287793650368042097805694309796191913683651017675842342387390060146420308670825848799800846008910296521948940339832985255570312674864622611835522284075672543854847255264731946382328371214009526561663174941608909925216179620747558791394265628274748675295346034899998815173442512409345653708758722585674020064853646630373608679959493334284553551589569947877966948566416572492467339583663949970046379704495147267378153033462305115439398337489620790688829995544121329470083542063031324851802574649918811841750913168721314686845259594117073877271321100948129855768148854311161320084129267363499289251825454183589219881333850776847012695765759653605573736881783528467753193053000281724117133569247062398777988899889662841499706014706016372541331507713879852825155309445733418035316861209556288978316282478143500096551357610725614057483872326044293314129792077777289475006632140252401733975684912603498348168568775217865832018883410276162470834740048255600659329223881004011060991907282879321090552836361563687680780220427483323293527090632178051813134981207196953681266541684913820792397357735300554966845235450747942390101292486119671868649365994478155987373346824092274515443661417944381090893731147361020889531680243992967746789389403706067885423681573530983486815289871675393650012417265434543050504019439701454624156569888621874101603997649966407799193057418611149232833039584855001941430933680923393544436450979507535811018411975678203452150800891864517137052035186135715348841397475928219859131636844030874374686820228572708751676865487969240162426098055037586559532217842753437863186592480806093936523929380404259520714584871920747692968032395327419735094412939817995275012199324226228671888756636304117844444282936821568138863483022553191437013266062994176763822906650602817535400390625000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000488" };
        SetNumIterations<IterTypeFull>(536870912);
        break;

    case 9:
        // Current debug spot
        minX = HighPrecision{ "-0.81394199275609878875136988079089581768282857072187918528787" };
        minY = HighPrecision{ "0.1940970983639155545870683099069152939779956506822050342848" };
        maxX = HighPrecision{ "-0.81394199275609878875136264371649480889415711511526384374298" };
        maxY = HighPrecision{ "0.19409709836391555458707132535458238097327542385162809326184" };
        SetNumIterations<IterTypeFull>(4718592);
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
        SetNumIterations<IterTypeFull>(2147483646);
        break;

    case 11:
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
        SetNumIterations<IterTypeFull>(2880000);
        break;

    case 12:
        // If you want to see where precision runs out with the perterburation reuse.  This coordinate is close to #12.
        minX = HighPrecision{ "-1.76910833040747728089230624355777848356006391186234006507787883404189297159859080844234601990886820204375798539711744042815186207867460953613415842432757944592767829349335811233344529101414256593211862449597012320484442182435300186806636928772790198745922046142684209058217860629186666350353426119156588068800663472823306396040025965062872111452567447461778249159612483961826938045120616222919425540724435314735525341255820663672868054494834304368733487681716181154843412847462148699796875704432080579735378707932099259943340496912407404541830241707067347835140775121643369435009399783998167487300390631920225320820254728695870927954776295601127590160537830979541161007773653115528776772709046068463437303643702175866642817304546107357112279873422484157718757129196810795693518958255039624778938475617445111016013086373475393233874716585338450230947310601250058690103023099689884201242694561516539745287838587918221627869932196806413565377232147111814588288577692048770788367460786433140889655843806647494371013905365699785200397560100006174937533306996141097840422070704064095237194112968121825183138665438571716381211796515694088396844959354095816866457114012153080019487232640050790708341014511315467013780272449347246902044715468312677230258459389504777848301760439914501243265396675318470277209199681587731097409818146696069676026130248510784235126335481494016193477694875359431995784189336594107507951846006501510431760177564290044552302075876143634173549988486753099154361843704143883372113456385" };
        minY = HighPrecision{ "-0.009020688057072617600360935984947620112305584674123866889727919817129110000273070325734652748577260411641977284267953435586679185748062521452185618401720512375928279282840017325461435395900347333831678867372709608373937732035861376732233573306124188011779554347246727032292516900609906698962747844988602428404977018518490333180457529450252363392753154488987278181741626980443558895836087364851272299469096298326327508396224591600469344439018862266215051078424481599350178419410453543640423233428241935033928198320388056103242790716487099369717401956440284903703674535517982961700987544448683097493910719867515406876699163015871869774586895353977306340825927065496180267868663306898577037289686648909313180741633711594021238074083115011086084628994833124149301314187302815457604493822108774400721225992087235996576176785127709984995821832988395033815431121124998792735163242913946558946611429605326561855235032093444769463577270874309902856442991193336799063416924854140617882170403247064602364981494053319833003188019116845480582222146591517526508505952626386869858474988126349184687934707366040142267796734715322832680253126560501329537526948788885996605418667272386217032588591270129447669824878906593971427474744417524322105140328661650742118353945952181192001604129127134047924585037918409069156107155889600191116937956191818015662980620885082359270174251013804407449814771577348678491812396912541905464641582286685375259221243722824553293480523618266671834657116667527540483323781858022176882851892592" };
        maxX = HighPrecision{ "-1.769108330407477280892306243557778483560063911862340065077878834041892971598590808442346019908868202043757985397117440428151862078674609536134158424327579445927678293493358112333445291014142565932118624495970123204844421824353001868066369287727901987459220461426842090582178606291866663503534261191565880688006634728233063960400259650628721114525674474617782491596124839618269380451206162229194255407244353147355253412558206636728680544948343043687334876817161811548434128474621486997968757044320805797353787079320992599433404969124074045418302417070673478351407751216433694350093997839981674873003906319202253208202547286958709279547762956011275901605378309795411610077736531155287767727090460684634373036437021758666428173045461073571122798734224841577187571291968107956935189582550332713134767221606194196966909920597154394601298418939500178310837089551519160325239057070335415353536614608854514772124036118106343484358740826468466191170631617074356641470804366320767734027197600492508276959709045822217267444775906367676211597027697730374929265711213570006888083159881614118166294004101997940358650624155974698074597135162493239227908862635120101181090672161735534444832481866257541957313604804765868699997708623273968618775578446964784635786300536926262073677381598605535198075397975409884727132919471518256860297054049156169291470558697171898240446929162870950544501171570301739748623953818158151161089954910461946243649370432772020344606460814389902136131967374425389185723446125203589050303538177" };
        maxY = HighPrecision{ "-0.009020688057072617600360935984947620112305584674123866889727919817129110000273070325734652748577260411641977284267953435586679185748062521452185618401720512375928279282840017325461435395900347333831678867372709608373937732035861376732233573306124188011779554347246727032292516900609906698962747844988602428404977018518490333180457529450252363392753154488987278181741626980443558895836087364851272299469096298326327508396224591600469344439018862266215051078424481599350178419410453543640423233428241935033928198320388056103242790716487099369717401956440284903703674535517982961700987544448683097493910719867515406876699163015871869774586895353977306340825927065496180267868663306898577037289686648909313180741633711594021238074083115011086084628994833124149301314187302815457604493822082301627963919921980188832734117144461235927725510618869926701050424262382737719488840773512518784408973510309125444874255964978497771821668461876114293439072218675091281807178360617915555529082793314189344198844402114683815213905623020938900424483270620111507313773141025982071467830338531834932334965716024243695294450805655962108713240628873982687645555738023024545155223684024358821182653368665810645129599750411260039008718131834815821408650230260822547619065046734882688109844628902351866850181380512234883756491595739994310366468198773264903666837375911772313096663562651632994834907611872106924651004252002990272786097767889536177779052406169314062287523045682250172098024827873526558027077566760003564036591195503" };
        SetNumIterations<IterTypeFull>(2880000);
        break;

    case 13:
        minX = HighPrecision{ "-0.19811160021967344480755228509879453543772824147087543575209282164460405796661858911920463414713355404730672339934458650212316679185224457464505947919951877044752574382704564824820621527934448404016493364320547229306519611171511715712176900717953583187315259259238053585661497776916800472941069492198349043947925513489896173070559294315034318176228198397942648722866757000829781024389799652212575542306646359027444069432370725938261915645996647660973657175607225853790674975618571917570230527957204526460701463173918515293501869448459185064344576999456359805385079383557776563004844468354008330736481418435296838012008500564694120819246555304168549021993810386844438674235991162162162552693868541735221730120226788397508143295115228891571440322071040035893469145432310347153135535181217777068958043805111980460858226841271718932841952098915221391513754130304454670933509459443822133837141872224361748538049080126666903834742681230893022577974880489421539115562656254310965025514910034337116229478322132145499334347381872195403786902888054606427838397631002687320687282745697533566217149085166308084080267457799899979196864712458846265368806530544056130146149848874818466622188802941784184464474834009310749590114413519468573833491862914876137009029121673459869433939351561841919154142711235779865793985455674208174102216159693143718238023685092327323727642289200834838695134174687167872481469723044215318220145759947793422540264128102143705824475109940031148462227253349868622772578244300313163175772202467420449945317245380988600167012464669497991600968524644338228807842178813932899000672825923461538183198762686454877469809802713434534445048058703923395789350323928962196539530465247300488890944356613861849247250669669559167977740546609272659106317907818930851023789439535345906216581467812819417753195134972084764729268058393107527899173921117959376762011940545670420410280386506477920963401216567286546226929765168985959828311037026215860190887116908979824655696319732733074345119111792558149869009781048835786881253533508499848969119588046186491856238391795092327966170799858991126972974493625729277715039166305873718799816660176584115266744430167947681149010934163805291011003764981720526738363243360119409752650474643574888828897585218303941928189836689849016576450697376083722195146156558426480099577734618531305533296673253150537970190827258181493819479687598022221974333798996039335847521854042507320903455652807257258903526332243228243511559615460310019904448209443137031260494818474246689924463711082405813737358021631393073732858482778024763436565852437154695057129216315148917487677222664834156478552341031113623543667983807907350282593666016978122983038822939099114405083775127145632739816795460185084064620863543065628114440156581756066752507154968504863337171763826105224281791178932927020989813212589292999118262648248016038897343092140225125318900543695471764746419644540418431469293811823484058015965407032649980755382536527874443438306028410220105859974292295915287154331992026072282920537729880888758627262656483037180785834638257692175151722860250387183318132919805711034352398697990025859370576693882609552423649944386651640082416989731006507184708966776441487790205285637309215844243942176437791160953120662718254932396534245082366493163459307826996304189885127045422287366232458258891328003658403230692471753618917578202476063310986807854856388919255198440466824236358127853342379751762858579132091099568789798071863727384338291022345968896624433628414878127462242917153761730086827314079022139776911122868908083539140392733985449628490366370036710811514650010650855344183692439171337895053515516285989632712351679380789905521028560229581602396215763270994638561110338263087410310988528756708648087968343640166962960902101864716477763567764389425387470275196858369360612655522422476611470371668766482595146642236202584459989166866788012960676202412427068787158932337711631738718533105212644334171450366360289877218879582231818391284758216388078085283224596396638801021564327747957457489174455578937847915426697043363300674436875242073120078951271096936391635156641517793220820610146161890592653434830315231584175128258540980673218650502076325140464746203800202739849688794463451163766147299235653628643802292756097416087289430205555352522550544261229970628150141473357154166915293108953343600228101516606985120428301300397237003185516353950264194714068535510113957115217817241050181960131736113134261363991" };
        minY = HighPrecision{ "-1.0995133542657857115808634031346189677870097136303572938715549689295389425632581503138812246977639087726709908677423161184335187629274263873188431553184305117753521329343866123179772001228771002107363550330517703579998140593409066357319311393026292408059480701983901970488499896735828553853802834301907197608149892713585833439577168227770555722928444712272129592949245167090691094148721898008727267507465019406653040957136916706118511368949128731646363480934893765440233859658466914730889839272611903774672315732641021754542818153354204695278348344511388429168400821488094355656936699975929482299923845850567674425122940359760347969736285151356337932113169624173522678013434408139130578132283466066872079950533561149015822217125258927391052105262576616632945045780120222419420734535273199171299980147596457095869899477339356795576617878510010827998630177588173538858637837731456007658310311702877324369659541180907536879135103018762800573950834307965599889878790750643156769516655597480390761553972391015283726067469563740287759144203858532782393629326958947345346075862898677300582937187422733316888028572486022198146869394141932576445093269938696609109004619913856912121251619407075780993615588186308812136419907177826036166069223642109761644270244408513600447075646747557841282319975447147019915850615072569108281032551376574856014145050239628405472665778715940606084639798101936408520457664615742008405118580595605204677748331709163054076781812915678506853351239918271425357018823354260518424833289209204961468610476161329798441098428030085154729370441648068624166487012202872274944187332078710294787873286766095146067629970942602556871762137094158332888636516293149557435983842777651293332443747454243255593579551968077852240199032487370427890674554584876808156811296145656826603260999681667949054686675903392053812108256116181160382494613997153199138603568393825186775292110275886829568971119775984919184077354063706566903419073631479095854702868276191875387674381277531094444441512800170000808094582407023055940003011540456204250948535043947179564116845971853224078045823536435698464891251304264471262096147132236334930669996592059889573584105384098186560250051739848054078845784511077797752691726900716551298408118617259364741999896084977307898465688063329817625214380750549330802870390053468442530064008379043658676719681384415954836464013723721328656584564245017093489835104348233380012210598845075057455320784942775443346399951069614279355554240011829384174308361963096028224335340259715137162528235751923616803492153533687263878085937736476622324705730656430032258822237858207894091176008620360867559502836300916391129134493393238616845392327605929904538512385164829453698817223777011835227488504934156151586372040638703411082219461948002087239032665791095552179479335250292539349234839433519820969519859430808755567356764565438803961064827434555915497113778977079605441377210893092495399432355608991036199477513888066828156326795583892133456614592830566442544687374076128163388320475699482715893915441611501227011705476586782690440277593692800910282999955406474787278970423592875378485539570549625585643603455801693407562571617246033401774796591129123039627392465173435466744333189312382957493987321525062480034592529077515387584484566100681565181620963805309260575185935715773449071166360012818595806823394732429772842032564876123502991214944831003127194370200560697563299786497558018703927056919915530941580817368155247209052873519988555883733787288116834882542554101183994472245225809915798186545862713186429005273179903890550958593295637174877467946982223916728461729036954094229402778026202138020491005859547750715441536659291389412423111376387691879474734129319825597969797462861380116991153974509207280337951873246607697818904391672151044085118487243626204300544957689214520254379117006819489725079260183139289555019446332600192640717145467855364738585576815881931284199917707246685077927850230029444392740341710214532917701299608882174119380311042117100571290606665230293804110116844241348879899086241122726289650136328011642911385610933906932369713687513984018877822438694490106404039018757457727445194534405647328297464259284472027928115425849951872169413282317316593183959788152388433164997240878468374424229310620424175449750107285234038858123382311297534602560686359827499341195321256108356232984988411254969950900447210214059234170765920468549231098085902093319040057298242654348820124671419516397663044813" };
        maxX = HighPrecision{ "-0.1981116002196734448075522850987945354377282414708754357520928216446040579666185891192046341471335540473067233993445865021231667918522445746450594791995187704475257438270456482482062152793444840401649336432054722930651961117151171571217690071795358318731525925923805358566149777691680047294106949219834904394792551348989617307055929431503431817622819839794264872286675700082978102438979965221257554230664635902744406943237072593826191564599664766097365717560722585379067497561857191757023052795720452646070146317391851529350186944845918506434457699945635980538507938355777656300484446835400833073648141843529683801200850056469412081924655530416854902199381038684443867423599116216216255269386854173522173012022678839750814329511522889157144032207104003589346914543231034715313553518121777706895804380511198046085822684127171893284195209891522139151375413030445467093350945944382213383714187222436174853804908012666690383474268123089302257797488048942153911556265625431096502551491003433711622947832213214549933434738187219540378690288805460642783839763100268732068728274569753356621714908516630808408026745779989997919686471245884626536880653054405613014614984887481846662218880294178418446447483400931074959011441351946857383349186291487613700902912167345986943393935156184191915414271123577986579398545567420817410221615969314371823802368509232732372764228920083483869513417468716787248146972304421531822014575994779342254026412810214370582447510994003114846222725334986862277257824430031316317577220246742044994531724538098860016701246466949799160096852464433822880784217881393289900067282592346153818319876268645487746980980271343453444504805870392339578935032392896219653953046524730048889094435661386184924725066966955916797774054660927265910631790781893085102378943953534590621658146781281941775319513497208476472926805839310752789917392111795937676201194054567042041028038650647792096340121656728654622692976516898595982831103702621586019088711690897982465569631973273307434511911179255814986900978104883578688125353350849984896911958804618649185623839179509232796617079985899112697297449362572927771503916630587371879981666017658411526674443016794768114901093416380529101100376498172052673836324336011940975265047464357488882889758521830394192818983668984901657645069737608372219514615655842648009957773461853130553329667325315053797019082725818149381947968759802222197433379899603933584752185404250732090345565280725725890352633224322824351155961546031001990444820944313703126049481847424668992446371108240581373735802163139307373285848277802476343656585243715469505712921631514891748767722266483415647855234103111362354366798380790735028259366601697812298303882293909911440508377512714563273981679546018508406462086354306562811444015658175606675250715496850486333717176382610522428179117893292702098981321258929299911826264824801603889734309214022512531890054369547176474641964454041843146929381182348405801596540703264998075538253652787444343830602841022010585997429229591528715433199202607228292053772988088875862726265648303718078583463825769217515172286025038718331813291980571103435239869799002585937057669388260955242364994438665164008241698973100650718470896677644148779020528563730921584424394217643779116095312066271825493239653424508236649316345930782699630418988512704542228736623245825889132800365840323069247175361891757820247606331098680785485638891925519844046682423635812785334237975176285857913209109956878979807186372738433829102234596889662443362841487812746224291715376173008682731407902213977691112286890808353914039273398544962849036637003671081151465001065085534418369243917133789505351551628598963271235167938078990552102856022958160239621576327099463856111033826308741031098852875670864808796834364016696296090210186471647776356776438942538747027519685836936061265552242247661147037166876648259514664223620258445998916686678801296067620241242706878715893233771163173871853310521264433417145036636028987721887958223181839128475821638807808528322459639663880102156432774795745748917445557893784791542669704336330067443687524207312007895127109693639163515664151779322082061014616189059265343483031523158417512825854098067321865050207632514046474620380020273984968879446345116376614729923565362864380229275609741608728943020555535252255054426122997062815014147335715416691529310895334360022810151660698512042830130039723700318551635395026419471406853551011395711521781724104902645634823828075018859827" };
        maxY = HighPrecision{ "-1.0995133542657857115808634031346189677870097136303572938715549689295389425632581503138812246977639087726709908677423161184335187629274263873188431553184305117753521329343866123179772001228771002107363550330517703579998140593409066357319311393026292408059480701983901970488499896735828553853802834301907197608149892713585833439577168227770555722928444712272129592949245167090691094148721898008727267507465019406653040957136916706118511368949128731646363480934893765440233859658466914730889839272611903774672315732641021754542818153354204695278348344511388429168400821488094355656936699975929482299923845850567674425122940359760347969736285151356337932113169624173522678013434408139130578132283466066872079950533561149015822217125258927391052105262576616632945045780120222419420734535273199171299980147596457095869899477339356795576617878510010827998630177588173538858637837731456007658310311702877324369659541180907536879135103018762800573950834307965599889878790750643156769516655597480390761553972391015283726067469563740287759144203858532782393629326958947345346075862898677300582937187422733316888028572486022198146869394141932576445093269938696609109004619913856912121251619407075780993615588186308812136419907177826036166069223642109761644270244408513600447075646747557841282319975447147019915850615072569108281032551376574856014145050239628405472665778715940606084639798101936408520457664615742008405118580595605204677748331709163054076781812915678506853351239918271425357018823354260518424833289209204961468610476161329798441098428030085154729370441648068624166487012202872274944187332078710294787873286766095146067629970942602556871762137094158332888636516293149557435983842777651293332443747454243255593579551968077852240199032487370427890674554584876808156811296145656826603260999681667949054686675903392053812108256116181160382494613997153199138603568393825186775292110275886829568971119775984919184077354063706566903419073631479095854702868276191875387674381277531094444441512800170000808094582407023055940003011540456204250948535043947179564116845971853224078045823536435698464891251304264471262096147132236334930669996592059889573584105384098186560250051739848054078845784511077797752691726900716551298408118617259364741999896084977307898465688063329817625214380750549330802870390053468442530064008379043658676719681384415954836464013723721328656584564245017093489835104348233380012210598845075057455320784942775443346399951069614279355554240011829384174308361963096028224335340259715137162528235751923616803492153533687263878085937736476622324705730656430032258822237858207894091176008620360867559502836300916391129134493393238616845392327605929904538512385164829453698817223777011835227488504934156151586372040638703411082219461948002087239032665791095552179479335250292539349234839433519820969519859430808755567356764565438803961064827434555915497113778977079605441377210893092495399432355608991036199477513888066828156326795583892133456614592830566442544687374076128163388320475699482715893915441611501227011705476586782690440277593692800910282999955406474787278970423592875378485539570549625585643603455801693407562571617246033401774796591129123039627392465173435466744333189312382957493987321525062480034592529077515387584484566100681565181620963805309260575185935715773449071166360012818595806823394732429772842032564876123502991214944831003127194370200560697563299786497558018703927056919915530941580817368155247209052873519988555883733787288116834882542554101183994472245225809915798186545862713186429005273179903890550958593295637174877467946982223916728461729036954094229402778026202138020491005859547750715441536659291389412423111376387691879474734129319825597969797462861380116991153974509207280337951873246607697818904391672151044085118487243626204300544957689214520254379117006819489725079260183139289555019446332600192640717145467855364738585576815881931284199917707246685077927850230029444392740341710214532917701299608882174119380311042117100571290606665230293804110116844241348879899086241122726289650136328011642911385610933906932369713687513984018877822438694490106404039018757457727445194534405647328297464259284472027928115425849951872169413282317316593183959788152388433164997240878468374424229310620424175449750107285234038858123382311297534602560686359827499341195321256108356232984988411254969950900447210214059234170765920468549231098085902093319040057298242654348771978680440440048326679578" };
        SetNumIterations<IterTypeFull>(113246208);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
        break;

    case 14:
        minX = HighPrecision{ "-0.19730848840109004006369495849244377975322001567334978960866216936842627359864082663776276894095720699803660849826377150236678266869162895747277682038789153500848403020023389829335309182776694014783649604655262311295596366738358876661638420179901819469645602954401853016953891065516387415963116098293206423116437952081712885056128220084977824592600941226781504831500617788951253600541052218205996303251019902118774731942097202794206850961010823255638276418020066764470795636254392235525906831527870403587037546121653513027693343299085161624113941455203631545293808634617428206151375311386416614654294781896847769011027399593827416596348585926638415387831668034450626372313649178544217006203288178013755561178850551450153231643220891362145010673463573924872202246846649145227890952530734727457656463560312924726438726844619747557121460568670138425980284333104053596071176664960072515836453762154139949467745612639240808167696151327847760748103966714600337595483031422619192874881085116652927865202204721828283231662323336723280442062884299953215309118267788960857076279398425071898468708755475674163910816740197285372853659099814362521184595483275079626485009480862400944291947058965318588314936810632190854860325112478359056657037107901447507132174682262747796118358522101125941150784729501357869352747614391260232441057791995125571371042201439402671820204346175701446696738836487696921542438015135186217375500175953268783453210544486167652756966545849130541979293032873022096881529670551091355403050005275740819692331510400034174979617333404746376520272483792300415344071776838740129369055199859559509283752264974608574663617596269811601088658759554717836023330845462362720834527659628496731763483353192604783115052427104198662697144080808879492650168063913113125940076702404063563636716954185432595122401907054889113524598988405542551394202806862082645514207243248096734228833972550421057513809216757121943858413153848226882455032978797761224865003239917505434927005679188569741455678954318375594992434340241908569886192845159737373823191589221566175043352839825028679855204027028554775402555195247442638315657622767106672548010847044590437898445050141498232332493513373146244925886544377139502334721305133658204142448735057423041610697200380202581587123094707036647736597942263511215753062559899795359790876946786264881478391631409936911370770328323786672662429231244709700210799990205546005576074315958539519771126546137500975224333071445630629259865797376558843239564047722868986588597157979251221288374653920815355011353999400249421447196612041321024842745259883323802656535674033629789084088345541956107675518697615475494177270806957776425591376648293064174717739629999820752084048766020432243565968172147263373502548612681279619571973929818202793649434390276875047753615927226778174304569331763438229228092405254045032621706688741083862962246706118727928460232044047528295996458418281083027118525433533182055304024546263851524455706235501653017110649230211786204878917574257535741049460319666555106834416326608205577294732173003287602297020765595727611708361006454213156981399643017240356626141748716877193844421313116826551006513038998269330698839994243223539001651171120946444433178884025956175593122693407707710040417647462977970645589562785696524942178247314359103783425569417089881388581852990359501672352469583419024732212843516278403332320787399066010024228114485929370778443567707315735642292709198151137743608203081153328819595377801367302927247030684975255888449157769424659766356801901830666422168997574485208051222311125317711680391430270268949085308228028609363418508677136682573466806390101093781871210170148452423627124083209403255168754861094865863630123719496520134757935668907092503148837416855253180710469632529287157979520965394008596230947834276448257641251072933993957720279832666327286179103785675334983739251791238544748561797064639811992461952881535413949545867116700507618424003353544018938289896745102944155952586386696091018977890031042424363739962657861603571772808626086811915685215119548533606703831132076994993876604355378403499143026063298436095903035099734771247711819317003806278255102250068293453071658755498908937934248400394589956544611911987041028766888549298308947857548484628999886237747508115983198244866667096172219924459190371486979252685702321674297430033930701050612121543992077476857820817154452697215023134671062081007089709186925057129010775047259723677063342045319708550160845614696515688725450917892660250832417873169768518555384440824930613231136880273162442563252557662328942011181113234670480417652050101325707008957863991756742092253824396923018735356096999733393307249519984295468686132393151712943835525234119049624369338696378734796157015653287169563883318722515026179070336562512288906727855386174716425597154015629580136312244638893142234715281036278417859278826579113502622837131830233065875699228555504935634775017160070677126557117415467598881441752124916677412511860397901636796498942807957148501672193896172247518118067084935606138383865930199416839805612331666303578281658181792969870344827401677138154890747354654527234450800757117612046941170041206370098590520258016438805827346241501917678982370878785603223314908585624097500717242551241580638910711821944128334283968187344871460684912163139998948918353946849256191801560333206499509828308878713332998322363074328871546467481830041012214398249395747431331501885417029901993713619725449354524623393816347788624050673704987450039806998578104946785958859758685746960137808989361387060245552532760923858918228498672191890357487482210274265248742954981989299289891457635665470665790060046923809716128463303325114387633116449214056776580004816295922017736243815827661666235431788986570529975168116332468921321272220002690603917390727131798092515984422246146774135449605349902380746713474979621550256594527573610402860693749337133458831050129073160922158048783746076551383646790969348831178968372424965477745374316315454637647841742817961416730963292028172681801249023599048684655469847609266437102773631161432434470232670656463274230128836924678088554760299336919422117647808313693753031530294806969355850778924305440175454932032585941146144572823555321743728989113627156013653587132922985515131885619648319102025246014566294463058430588165704052314372511696586948236297752989348135391113685366020493684260080871925719990080658959354229404922225019185016427250018365369186990602512868081077318204414984280246343593890080148166959534578671674683928746679704857016560008242251520908338160209266222448859926725185120802625523127870056769211459462144254872895527229653432513268209576619307172670972296513149371536" };
        minY = HighPrecision{ "1.1038012946026287867175340537321106754714179377909829608698111892311828116339593340641577619320630631272359545249104602024464660368108622972257186686981113108472441866359913631419173375708534023908956818662781845243135849910993389755013814918962912563217422560385989433085601178697425756588907931255455461254161103635542128369609052645949788789782753943016708347835008968061806510471664104783265965008441112828566815310404664903283119402772677879931633859043073640139451510363463137596378813117096535736567773729634706808291526375586983776682392863083530498167151806642245560286215339977076499764785829538273147815417500738610285975563214323780827256126559177169128840449916249705997139015561428170901792240899706594002990884165293276128739530418792145302822483704315141959990174469315929267338719704963185834262355063480618285059445089066978996354473817691537479272678180963179437331577662923089833440065737104496420734901020062335325633890791437705045334528241030805352477965778371494046243100986740512684673511263353634060884498113198005725740281203062287173839045546138975773493464113071526061452896370126461759527654097782052908530271536005036257885481328814854200084105514095043483636349067227555155407078865297759897658434292729402146643752377437232687231785804398122780690452130971472666478267738833288367357652418369657921421509385464004483250668075409981057957028155380383774844014946124157709869335895091458943355189661348261756606819168881904643213633760491270042421596809479911932247718165437568264424693253178072345584587028112325377361419167998407187222477535104308636263925643937858005068044166766610159228671457426217966526451690376098935640862689339660729754977160401812735696359926255433647836190331030225903294635310897705537256962199137186771053305663171707509598490523924721304012713341227902519132912138656784187629279520283464682372091116074251283483215677118879253318704335016015638601026181169984792342248934651418053580785108345424647304230679793124544590150449999906946084543724138766402247811191129313083672159807769469023395587546506509467787672442265229039498439979201085794533651266184631069218294954302892235113890258245701532682810203787888453038022013305214118675060885939412464346962624729631976469396334474273319838546465973749886116891154133498667088117393973817462818226936711127111648516076590922700976744541375087032318027536398945874148381707159527704568353680858388765666483157688575595037919545623186294531890137342527617785363381859183381555120250986548868975667199395872613357533191730472974652110220551197304298951749887705880357469174641040381109130339080759503421828349453332549601783841275450597078683783143807549585578535721501172937619532598104586606315566339679466690390637817321649740920691015847633151832780884670536787847523743481781963361791353655405277033699730969900474065899818873788518055470405347184686467053608834170850165628554485694032539624717968552643733390705081562165648031076852702178997886629949254371598253070031712876689401664166343589226350210359951701186986932346307460746116405894013917791628831545553622598039668371221196573051562382869634534483507552145361562467748554076022224645319087644082872137321509129645648938484099100963989247075044275953832099901447647069029399659296487469687880110238884390303566420428223033998062321438433887266187516409781969401176250936898533615284296924590661852979836792229950450877310239293243419845306712092230555581255286651524596718533038331554349390576617807966527101564360366552340412619828617391171049848994795589424802393737224004314316269420950968565558670732270702832101704070575710647956999020673224209765787061270277055919813769714150011696131132771409595584677033766277470288995418384535166834001468200252328629405841049467611940190060277361330561811815670657275135176412575994988722993087790764824436411050333568729417091268489487239928791416409562990815995586612811144303083742780804676301337315793844376905720235820565995159499391373560280310232404668541175284768245871481710864156139376775870356287852479263970237350231028933348659307148799318322103425730473491502309993942222741961913311655476198442624558204326189065255491719648586849041661499815125627021920624662169034826142934398772851390514982776919982683947131812886150690799210234243965572818619091285670402164519928153643847037981380132541934184241820415132669218704569402639920739734046970463125908401501577960387717874579074959354572365429291843353108582707055145306531080530523647435916818888751595483184373184360658189240151330835135085200983385816028261729333590855062870090056433718577094029359344223255268019287066750013708586305552423906230682172136742570840536134180117039122744842544270748660058851312063630531845830722176388572482815813932252915108515743801079691302985774475938090401055796762620681795407308160303143140517216246924013539058244982513995657908481871389355139571180916313800267276871494085143381903239775431299048925726063527076503729263561247913906732877572866148080995216039019027873373006311796791431591886639668789898209314666046131834502500685441778933781200462436961733655747078895309963754838752239468209270026317583112802802239913507946310326552102422053823391762586578911448663127819444725764962764253262565883259681989907353152741580075943384517117296448384716866309450576010966202093395194882619269384816644573182809038618254600088333274182508740974285137888224275178520102433362781670234046983355143001336685483902665342413116836380450232648631074202137568739488366490327733832993031909196119147461844390579531213153051541476918460593705879450601392877424958052824329194856978479469281329764682033113052355629351290383907435568621224450715512874995755041806566570171490051322322605880061990537548769309096921087269673853485678976385263645685878143782018141003077797781045789474340395912091697012642458198870769412315746798645078827380398828763262497374496402919799955681118469542324044796680256194954711539471991487699560790979864194715626170452319807839335821455342795806542139254306733652313111332027709596040592705154687559226384975718745115744715689239586338455232337207917015300026972551829720471995871182230864222240676267788499923400202264673030329355264106848556196515945418012802668937033120012601661840516884427934207415674131393314219931388136008071368031685894211223243366995751282488626930943464429816450318167617476468739302390283560260867620522032412035951722439223584224475710254516144154903533489798279199924595975051961543378038669589027217803539866600951488307251281015608471219040151783373331374690198394371591542644748239966733553148628957491289222204722654426537557874607418286270335" };
        maxX = HighPrecision{ "-0.19730848840109004006369495849244377975322001567334978960866216936842627359864082663776276894095720699803660849826377150236678266869162895747277682038789153500848403020023389829335309182776694014783649604655262311295596366738358876661638420179901819469645602954401853016953891065516387415963116098293206423116437952081712885056128220084977824592600941226781504831500617788951253600541052218205996303251019902118774731942097202794206850961010823255638276418020066764470795636254392235525906831527870403587037546121653513027693343299085161624113941455203631545293808634617428206151375311386416614654294781896847769011027399593827416596348585926638415387831668034450626372313649178544217006203288178013755561178850551450153231643220891362145010673463573924872202246846649145227890952530734727457656463560312924726438726844619747557121460568670138425980284333104053596071176664960072515836453762154139949467745612639240808167696151327847760748103966714600337595483031422619192874881085116652927865202204721828283231662323336723280442062884299953215309118267788960857076279398425071898468708755475674163910816740197285372853659099814362521184595483275079626485009480862400944291947058965318588314936810632190854860325112478359056657037107901447507132174682262747796118358522101125941150784729501357869352747614391260232441057791995125571371042201439402671820204346175701446696738836487696921542438015135186217375500175953268783453210544486167652756966545849130541979293032873022096881529670551091355403050005275740819692331510400034174979617333404746376520272483792300415344071776838740129369055199859559509283752264974608574663617596269811601088658759554717836023330845462362720834527659628496731763483353192604783115052427104198662697144080808879492650168063913113125940076702404063563636716954185432595122401907054889113524598988405542551394202806862082645514207243248096734228833972550421057513809216757121943858413153848226882455032978797761224865003239917505434927005679188569741455678954318375594992434340241908569886192845159737373823191589221566175043352839825028679855204027028554775402555195247442638315657622767106672548010847044590437898445050141498232332493513373146244925886544377139502334721305133658204142448735057423041610697200380202581587123094707036647736597942263511215753062559899795359790876946786264881478391631409936911370770328323786672662429231244709700210799990205546005576074315958539519771126546137500975224333071445630629259865797376558843239564047722868986588597157979251221288374653920815355011353999400249421447196612041321024842745259883323802656535674033629789084088345541956107675518697615475494177270806957776425591376648293064174717739629999820752084048766020432243565968172147263373502548612681279619571973929818202793649434390276875047753615927226778174304569331763438229228092405254045032621706688741083862962246706118727928460232044047528295996458418281083027118525433533182055304024546263851524455706235501653017110649230211786204878917574257535741049460319666555106834416326608205577294732173003287602297020765595727611708361006454213156981399643017240356626141748716877193844421313116826551006513038998269330698839994243223539001651171120946444433178884025956175593122693407707710040417647462977970645589562785696524942178247314359103783425569417089881388581852990359501672352469583419024732212843516278403332320787399066010024228114485929370778443567707315735642292709198151137743608203081153328819595377801367302927247030684975255888449157769424659766356801901830666422168997574485208051222311125317711680391430270268949085308228028609363418508677136682573466806390101093781871210170148452423627124083209403255168754861094865863630123719496520134757935668907092503148837416855253180710469632529287157979520965394008596230947834276448257641251072933993957720279832666327286179103785675334983739251791238544748561797064639811992461952881535413949545867116700507618424003353544018938289896745102944155952586386696091018977890031042424363739962657861603571772808626086811915685215119548533606703831132076994993876604355378403499143026063298436095903035099734771247711819317003806278255102250068293453071658755498908937934248400394589956544611911987041028766888549298308947857548484628999886237747508115983198244866667096172219924459190371486979252685702321674297430033930701050612121543992077476857820817154452697215023134671062081007089709186925057129010775047259723677063342045319708550160845614696515688725450917892660250832417873169768518555384440824930613231136880273162442563252557662328942011181113234670480417652050101325707008957863991756742092253824396923018735356096999733393307249519984295468686132393151712943835525234119049624369338696378734796157015653287169563883318722515026179070336562512288906727855386174716425597154015629580136312244638893142234715281036278417859278826579113502622837131830233065875699228555504935634775017160070677126557117415467598881441752124916677412511860397901636796498942807957148501672193896172247518118067084935606138383865930199416839805612331666303578281658181792969870344827401677138154890747354654527234450800757117612046941170041206370098590520258016438805827346241501917678982370878785603223314908585624097500717242551241580638910711821944128334283968187344871460684912163139998948918353946849256191801560333206499509828308878713332998322363074328871546467481830041012214398249395747431331501885417029901993713619725449354524623393816347788624050673704987450039806998578104946785958859758685746960137808989361387060245552532760923858918228498672191890357487482210274265248742954981989299289891457635665470665790060046923809716128463303325114387633116449214056776580004816295922017736243815827661666235431788986570529975168116332468921321272220002690603917390727131798092515984422246146774135449605349902380746713474979621550256594527573610402860693749337133458831050129073160922158048783746076551383646790969348831178968372424965477745374316315454637647841742817961416730963292028172681801249023599048684655469847609266437102773631161432434470232670656463274230128836924678088554760299336919422117647808313693753031530294806969355850778924305440175454932032585941146144572823555321743728989113627156013653587132922985515131885619648319102025246014566294463058430588165704052314372511696586948236297752989348135391113685366020493684260080871925719990080658959354229404922225019185016427250018365369186990602512868081077318204414984280246343593890080148166959534578671674683928746679704857016560008242251520908338160209266222245858354602733431756899919036964099584814370395957445910052454644577793819520467706569603032578703859245649528511" };
        maxY = HighPrecision{ "1.1038012946026287867175340537321106754714179377909829608698111892311828116339593340641577619320630631272359545249104602024464660368108622972257186686981113108472441866359913631419173375708534023908956818662781845243135849910993389755013814918962912563217422560385989433085601178697425756588907931255455461254161103635542128369609052645949788789782753943016708347835008968061806510471664104783265965008441112828566815310404664903283119402772677879931633859043073640139451510363463137596378813117096535736567773729634706808291526375586983776682392863083530498167151806642245560286215339977076499764785829538273147815417500738610285975563214323780827256126559177169128840449916249705997139015561428170901792240899706594002990884165293276128739530418792145302822483704315141959990174469315929267338719704963185834262355063480618285059445089066978996354473817691537479272678180963179437331577662923089833440065737104496420734901020062335325633890791437705045334528241030805352477965778371494046243100986740512684673511263353634060884498113198005725740281203062287173839045546138975773493464113071526061452896370126461759527654097782052908530271536005036257885481328814854200084105514095043483636349067227555155407078865297759897658434292729402146643752377437232687231785804398122780690452130971472666478267738833288367357652418369657921421509385464004483250668075409981057957028155380383774844014946124157709869335895091458943355189661348261756606819168881904643213633760491270042421596809479911932247718165437568264424693253178072345584587028112325377361419167998407187222477535104308636263925643937858005068044166766610159228671457426217966526451690376098935640862689339660729754977160401812735696359926255433647836190331030225903294635310897705537256962199137186771053305663171707509598490523924721304012713341227902519132912138656784187629279520283464682372091116074251283483215677118879253318704335016015638601026181169984792342248934651418053580785108345424647304230679793124544590150449999906946084543724138766402247811191129313083672159807769469023395587546506509467787672442265229039498439979201085794533651266184631069218294954302892235113890258245701532682810203787888453038022013305214118675060885939412464346962624729631976469396334474273319838546465973749886116891154133498667088117393973817462818226936711127111648516076590922700976744541375087032318027536398945874148381707159527704568353680858388765666483157688575595037919545623186294531890137342527617785363381859183381555120250986548868975667199395872613357533191730472974652110220551197304298951749887705880357469174641040381109130339080759503421828349453332549601783841275450597078683783143807549585578535721501172937619532598104586606315566339679466690390637817321649740920691015847633151832780884670536787847523743481781963361791353655405277033699730969900474065899818873788518055470405347184686467053608834170850165628554485694032539624717968552643733390705081562165648031076852702178997886629949254371598253070031712876689401664166343589226350210359951701186986932346307460746116405894013917791628831545553622598039668371221196573051562382869634534483507552145361562467748554076022224645319087644082872137321509129645648938484099100963989247075044275953832099901447647069029399659296487469687880110238884390303566420428223033998062321438433887266187516409781969401176250936898533615284296924590661852979836792229950450877310239293243419845306712092230555581255286651524596718533038331554349390576617807966527101564360366552340412619828617391171049848994795589424802393737224004314316269420950968565558670732270702832101704070575710647956999020673224209765787061270277055919813769714150011696131132771409595584677033766277470288995418384535166834001468200252328629405841049467611940190060277361330561811815670657275135176412575994988722993087790764824436411050333568729417091268489487239928791416409562990815995586612811144303083742780804676301337315793844376905720235820565995159499391373560280310232404668541175284768245871481710864156139376775870356287852479263970237350231028933348659307148799318322103425730473491502309993942222741961913311655476198442624558204326189065255491719648586849041661499815125627021920624662169034826142934398772851390514982776919982683947131812886150690799210234243965572818619091285670402164519928153643847037981380132541934184241820415132669218704569402639920739734046970463125908401501577960387717874579074959354572365429291843353108582707055145306531080530523647435916818888751595483184373184360658189240151330835135085200983385816028261729333590855062870090056433718577094029359344223255268019287066750013708586305552423906230682172136742570840536134180117039122744842544270748660058851312063630531845830722176388572482815813932252915108515743801079691302985774475938090401055796762620681795407308160303143140517216246924013539058244982513995657908481871389355139571180916313800267276871494085143381903239775431299048925726063527076503729263561247913906732877572866148080995216039019027873373006311796791431591886639668789898209314666046131834502500685441778933781200462436961733655747078895309963754838752239468209270026317583112802802239913507946310326552102422053823391762586578911448663127819444725764962764253262565883259681989907353152741580075943384517117296448384716866309450576010966202093395194882619269384816644573182809038618254600088333274182508740974285137888224275178520102433362781670234046983355143001336685483902665342413116836380450232648631074202137568739488366490327733832993031909196119147461844390579531213153051541476918460593705879450601392877424958052824329194856978479469281329764682033113052355629351290383907435568621224450715512874995755041806566570171490051322322605880061990537548769309096921087269673853485678976385263645685878143782018141003077797781045789474340395912091697012642458198870769412315746798645078827380398828763262497374496402919799955681118469542324044796680256194954711539471991487699560790979864194715626170452319807839335821455342795806542139254306733652313111332027709596040592705154687559226384975718745115744715689239586338455232337207917015300026972551829720471995871182230864222240676267788499923400202264673030329355264106848556196515945418012802668937033120012601661840516884427934207415674131393314219931388136008071368031685894211223243366995751282488626930943464429816450318167617476468739302390283560260867620522032412035951722439223584224475710254516144154903533489798279199924595975051961543378038669589027217803539866600951488307251281015608479677438990218860374946590368848786474225856792664417107004943320002309568128360633906497543395052458971098763792" };
        SetNumIterations<IterTypeFull>(2147483646);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
        break;

    case 15:
        minX = HighPrecision{ "-1.2552386060808794544705762073214718782313298977374713834683672679219039239148526283454222176533392628121047872208077269716197797093371025472466668952255134966668370658175279054871745549423277809914665724804837101524717571495723874748859332474368687432650230295564976108516327577711877380986" };
        minY = HighPrecision{ "0.38213867828779202629248055880985238455077826251540486918302797501978152666818394826645189130797356608786903473371134867609900912124490114784833893781426939900090514977555870760383725839060441363002122703521562703611623045547679106961127304013869141181720728437168142187095351869270963228793" };
        maxX = HighPrecision{ "-1.2552386060808794544705762073214718782313298977374713834683672679219039239148526283454222176533392628121047872208077269716197797093371025472466668952255134966668370658175279054871745549423277809914665724804837101524717571495723874748859332474368687432650230295523192890277752990779511763992" };
        maxY = HighPrecision{ "0.38213867828779202629248055880985238455077826251540486918302797501978152666818394826645189130797356608786903473371134867609900912124490114784833893781426939900090514977555870760383725839060441363002122703521562703611623045547679106961127304013869141181720728437342238929756079314822486632934" };
        SetNumIterations<IterTypeFull>(2147483646);
        break;

    case 16:
    {
        PointZoomBBConverter convert{
            HighPrecision{ "-2.281792076991057188791220757070906154801563373848351036490006446241507468147564988356371628826904426023128505132018554501675433482727741549694771224847955228931798289163815438743631000392333635842525042153582202499655604449475066164586596459152105106064111624441822700788132655777068944695173876856445256369203065186631701760525784081736244e-1" },
            HighPrecision{ "1.115156767115551104371509493405146139515685158158631166093965276070475149166424130419686602805948430925491857449965291251362671124105939610231137732602258574689237558180672414305416362841285284232472179206525520835146101831409246792045496633731870859449092417056425258013586398208085502164519751503113713282571658615520780020827777231498937" },
            HighPrecision{ "1.4e301" }
        };

        minX = convert.minX;
        minY = convert.minY;
        maxX = convert.maxX;
        maxY = convert.maxY;
        SetNumIterations<IterTypeFull>(10100100);
        break;
    }

    case 17:
        minX = HighPrecision{ "-0.539812925897416895862220795208691837237482310207487207526064802153580364920074745945503214288517608834423777345155755" };
        minY = HighPrecision{ "0.66116670210839960989937303171366098756525974715002460800210621125305170240685980562572504049349703520845019259075699" };
        maxX = HighPrecision{ "-0.539812925897416895862220795208691837237482310207487207526064802153580364920074745719107191125127561583883236084488469" };
        maxY = HighPrecision{ "0.661166702108399609899373031713660987565259747150024608002106211253051702406859805720056716811576221562842084782701692" };
        SetNumIterations<IterTypeFull>(113246208);
        break;

    case 18:
    {
        // This one does not match #5 but is close - it'll differ by the screen aspect ratio.
        PointZoomBBConverter convert{
            HighPrecision{ "-0.54820574807047570845821256754673302937669927462288245382444483459499599968089529129972505947379718" },
            HighPrecision{ "-0.57757083890360384280510898220185055867555172845825531715837895289573690983215542361901805676878083" },
            HighPrecision{ "4.98201309068883908096e+44" }
        };

        minX = convert.minX;
        minY = convert.minY;
        maxX = convert.maxX;
        maxY = convert.maxY;
        SetNumIterations<IterTypeFull>(4718592);
        break;
    }

    case 19:
    {
        minX = HighPrecision{ "-0.48065550796374945612193350910587992881196972760301218859906076861832853774308262389069947355595792530735710085477971769115391712081609017535787686457114048952394910414276974524800303186844171743" };
        minY = HighPrecision{ "0.63747559012497080520941191950561208097997900567996614369097324204248299914068987987862806005886435190126924708084397626352001791096751986215114695299109848932200215133076315301069004112043644706" };
        maxX = HighPrecision{ "-0.48065550796374945612193350910587992881196972760301218859906076861832853774308262389069947355595792530735710085477971769115391712081609017535787686457114048948840611703973814164062372983621896847" };
        maxY = HighPrecision{ "0.63747559012497080520941191950561208097997900567996614369097324204248299914068987987862806005886435190126924708084397626352001791096751986215114695299109848935754513843379475661806934315265919603" };
        SetNumIterations<IterTypeFull>(113246208);
        break;
    }

    case 20:
    {
        minX = HighPrecision{ "-0.7086295598931993734622493611716274451479971071749675462585204452812848610912827" };
        minY = HighPrecision{ "0.2588437605781938517662916915477487152993485711486090408513599516860639261853173" };
        maxX = HighPrecision{ "-0.7086295598931993734622493611716274451479971071749675457247023721726823519349396" };
        maxY = HighPrecision{ "0.2588437605781938517662916915477487152993485711486090413831737016165365135300597" };
        //ptX = HighPrecision{ "-0.7086295598931993734622493611716274451479971071749675459916114087269836065131112" };
        //ptY = HighPrecision{ "0.2588437605781938517662916915477487152993485711486090411172668266513002198576885" };
        //zoomFactor = HighPrecision{ "1.12397843052783475581e+55" };
        SetIterType(IterTypeEnum::Bits64);
        SetNumIterations<IterTypeFull>(50'000'000'000llu);
        break;
    }

    case 21:
        minX = HighPrecision{ "-0.165353054425834327068347550055335092120094241000026380415912481478846048996851264628574322423560266690587753860396550875628196105603940031447124507284796134645582380068094647607090297791540990497988541731977355180190520753027467465559816468674235485307967922195244020435598058452547945052640411625622380447543438758451505365411536195304664553213862187796767606743602535567713694658419041684193170231020290015597294881195401590590515687821394886994681228468823121619355762746143272331336588126133019869297364991848070238405297898977565831863782091635917688802769483971793868164049793382772144824691474021637422472581775598186763066262139585911209575570234948287138025858113770595417057765737841033586118537409729682731556704538168437889315511648671541179084394324381949630007598456990590122257489304269635986994729167818299362941847397307880932756823683547959153931416943468484997414042002226760352771502213545178733070190221064342199610334840881387030896218005684794531506673212756591521716283419037782654703941428719602480473210753190164633542506076510382975888406693578584596235047040945129391014749050860448846296782715940820928910430871080443380315527241808928932896928716039119443897790852490935543890817992936338937709268027467604182378253232485293312376943190272519179472927080530690648707898886007389680583190917306726096770998920761881206074115907690857056150891228681662452179492284790954328662430430656981970897085077471610297695950783746253592253527326193624697532694941942188724771208983480291886159387242725682612097829455543312890683809621756625033935103275511262132919872958003878843292448425265992348158407306443320832946249494297083297065386410865236181532967975846511540395885398967800451405872354471111541143118834281394086642513628585599191663767876630934261330738879007819927233496564762558362758418390567667209759340681400136029227832363343261284047156793237563769392991347582335842415147800214034819105679766895896190888487104322278865433401758695548688802612394860403944960107124678192573678989832899795415529229163840768056022104966650132921903672249536059325757089871496643859049887770030871044773077334163207114053337847994782596615178782492108495599767514996566744826625746330516783048178696834588961051566698640739429470125374731896593223653194204370639799254530974220119945225314385941207239585116703173521826441463626562708474709601500726612149467260167281041829123970618934980097324455704814315034834953682328758056771526998951795691697930367447004451518375515418169353884080315386805071944214945628326272206986611580426928212750995991822823860026677580567579124012249971331946408816729835427454567800436845776613039824177570729611229205776326703592784705416074092879853581588411484950856055747980311929635987723234642038360349774236752092955179056821468348024666783398270724421129003133127090314443705720069574303174193228724845820259116590995276466078757283233798452159659421707779991435140369151226234965251861878363669272417784157764671849926431585329404200513542914120936540490370523737679542261902716105031899047000574672763617580363870626629525533768506115991045406909783797753803746089087264977840263005711189494266590101966689231712756256470547792744777553927079757974120385717870790951384830911886978567420529971596544013707239416091341585384871990520957060109772736365748978193905106534801222821369395263033319870549174850155278336102979440391140479756393865532583065213267686335482313265711452613988394900061698931036844895917649122714468973535110221199701045387093364040088983148440901558565343569933842766230129144889523286868632078730811819589924157257618828836418991854941183745815617753130829885427605738332351815468800041614844411948734214373134474159935836830528838008902014084989040613766534015144169630407744947384870577540641433410805373824081271884349456445980467227805939858029823443636433375993565736010527634777140361646143747611462976125176422699484335744600731469227994871545083124569882828687719279622761878666910489636170143111644396194740107754249102526504602513511057314389106209059481710370014957772773493996517864927750259751417897780388304152123559071586995936393040390811267657212490363383794594817939326816915052108702617526830125590381626179273747346711161718632746010506896492152507657949850685205394267271069232829330314113981044847399353982571187682200220050685109212015810345847374980077166290560814788482099061174945177497777209829366064227056160852469309701939873197934997935146823283624577874026529594466861139705283234747077286219792074376315246333330706693906228230365874533010214032675713142686895233586242015050484327701262570458880321667614601233227675479227672224641408664879748669522025277685622575570050182195870035227678915279535025936583502036576519349295338276798472988538693559511220010233201916228704019126372083140407535079980191493057561086714211574430100432792620328609268319520304945854385497936141901443101199205052471093936689979860173883780344114252131947436942302349503752793374627732075522786101926735579515622542622317922002746879698386356592865907671648352406117662021351985545998783349330524914719609285041426256404473243191913367856856505245053658272307095369129220162266268420386638737793203594207202582540456171596060541143750558367075091060276815628477967236278834778223264263609185470196025869508929722398462967645617597794592799099406939069220012992699606111668791972386713160194753183837116190835484443893600248426067621563375877433923433053986834276941578172577169480739698040727927051070813667053486331251784829654417512411597664042111767367663367872406811253438297624215855991182989731954995212058872814504049337329693272918413538243524526839378807754264029690341466186361044512855032557883729700966098830281872407442880004870470584972051023774452512883045739712345718037055286717487106553056063485228487300596391087139454082167573918246552383569274381849009436037589909083139992944782141686680096524533568170424858925610407896477023227911013714730552946622590798621241352950011051227518196183154105030003364985768251976119377457107417619857186239273885250163431996516953720197171826621833921539290435578458700196733937764086363546782270852887856001397001975396643041276905170370449927708191342356000491510498182329289727839105416752909777142489501312569846474792156750775748670181730686556349529944551833099392024954663901742552812089011917592931200119202193015248773456711015371790869584374293209887842875062426035294797893890969787057912885144174509693572729486441063536053468860625486553264201741128373776071355340209336602300388938993187197178640398069203314482289353518901337019848286111326088235409923686674159952161562480844551090683390209994136598461029533764993305238873823339363942706577414784359492517592521760479266623019971610190663368517350825471750623101329733732676875134072821397170059942499383337984758966986663628920758061051763908186554368494548431347538961093279045258986114008848070935719505927929844387038392643409634869617879553144747707394902135413073731883743987911165789174348634634075053249592498046647045816047976293211941960381162024603750637280577686138131445369610346686061122266698851206484285402672323270264993345352262842622989814676592089051166636314896577991170492999858475643511699922806010673865196459988411983646144529395110965896266135582918090788910655653533977096537938917321840610415306482518110713209759360969812808423903607312327852649389369181278717908373707127250848414738760284626613909239977215884435918888044487983110245967872307235049034446988418503531952968982788520561075780654892652683629294901141771279510968052360539748738836432790792043526253652531307039188456936412054181860042128306868922924285956289336711532018612706267065365750882954648327718572091225202058150597266420194411112146793406903002760749685050404726389139753261108075164325117098447603641627172342335141252879668047403442496156686098362967003280803703751496079357587699689190716273868541142530992703910472555486651898428792369378140669670598661156878545238432732583504801709877711760652053644181148221932669718344567800083561816029230283328631350629687152285361481168846243317678250175658617315776244662741098180364247827206432903047652330073958450747982982320095249262502767794991533685844967431946590301552167994682578091262037108944986972736919718880408275845321966274622598243083637536104897294413921638954282376280884695105128267347686829582020636483391132026909471106841010974160244514587380606659287045028970009321283953825219937227139813381763670073075883942127389061316189749902383423167382658738580952213198804945769600776180384319100485879028253262657134710335293606535775389088853727315293100876079952109293344088103971951480494487754596407141467589307223384665151440511539065850505118855113045469618352702700421864081721157910734272327967917845850706068618905313969993629417189019440249915526790185446294203281664821715945912860879531821023789062130909466236932112134608209750365826535844639826854943861059761050817065936005572897716927656788711329860395413268756189922094161632961802975376296442212156031933282337702006954782696327044422873119138302780500207468796009428403959514219092778403693156424490285561767859067764070637929433238019135934620127159589148688663316333823324271609322641842758694003289203201044936650228178758315741220758011252572074991096068049106618012555191310446255165370526415173362113768333789599943675471326422117676334314266747968775024668083580670963192468694653336377644206322697597857619326333187182289423076171533936691489200188605165418650423116697995608160056812035697001549422601519842317117999137345435699155672279037430567745010659244889782939560113956308388433906461427370380377343080248029365213168450436915758404812661842554630512632919783992213543783257176855924460873683896564206216084026461806133687175646248722280968933802897270153863464293042013597529870325255088275718365644698293584089082828539443700361774585106480045756145535019626597646855793743408522811671335762791511468437595742168024409937201303441774585934220394538744700727446992991260442240520048636797756024810930146640135842185194897203606389769729463321849087906112691468658396157387297362275420411739066157736983641068114823469133313427620116880312881225892298889836488671172799822225946955020411149420933774586872339848592610488869714677843758358403321183181151962308158619307215281277740258543759337972503873005561599434391046500951009505597762560623708223965512068456513780654219899954375597512828103722798859948308156553994793288430241152986924485177472873015909873812123339226879255020117083749194681650812041777524412244774192327973441088010851283224914045003379940557946972586822089061969203048382548361630863652133124621331655531715660733353043243403778897569533996464116954872159905058467494495292935274845521175643055262623565410797581021224500703211015693978874893183400350124608416415017323117581688294856523114770863888377760625246237288617876540714450135755239993970314184532237097504605284789252685023259" };
        minY = HighPrecision{ "1.04427706489780880538297213351497003363666975202959243360662117901132446083915440776008340197558708451448566594407813741280640700286599543544832030137985317143101850416450179188293791595970146030868078983353894001761510920405211437853459980556891314343372140304643920586905384772144046561440463672887442493604369964093031790085601726605943757244038987874691317047582903623098466937218544698989158262785114929875534431084773102071634389390575438277526246452795278692668747674916196632130757747514191192293364769091755224519478362332466357018378475332087670159980641419185979119425438366584750881367630314956441600216954342598243447363917569050223188423877933963693792711567713385602672025430432091259632630419532253981522743510822214509745892379604774642591036493292582955800658416644366431435230726414838297089140898897906252596838105633433567904531639226277440141074891340840172660388850878606850256191692302086566719277875371859038499856734370754058259089364159875568311141878158722110487216373002819328872601611438703229697533478385108467523795343391407134192314957511076414141921025223117318274025347786843637524964610162231422741619114669024062716908955537796285795307841492804416910825692794234945991012594852588819251480297963813092888451368959767650622731563373685959941045434757359545438381739474893502233408225513555112040009901911899811030713340917424293308869334159844981992822025915567820722101776914118792553096742032422648264244183730016569564008699496658726729876481611198795353251214149993514935263585954415877235977322721329923721470764780809934514968279529467249052143110910306859037232343032792148601854331175782349756708902353528767905275840962882821024915611725408835537786790121842055841495828158987095633058968885132104669033012513254412953098422171783745556261051987332853256091888943060113510797334576881720960897420783325085266769379538064642539122539208251372494949447643579054297796341813735961821280764689005745738641339993649605748878416787732359082449161316880129659468880107415429503286310797048778241619431018551998652910735894524859717843495928082750537582741397377006122704540320235847881328634570806492527372600585675975193721605301568603833372573765446190745569475270286126035577926267562011967467128956904811786916373133288538096139524287803884803010758718041928100719940391835895568061501281639239965540809222071978511043558943005827985167691104561994731766062850334893479427808546924916233807818574016681189832820104677125393903670198646806701884954467609398564616351744366925759732760599868026323484105678114225959986662794217643218710172389347047136819146200776983529114614196100389731659658835062519311062678015737454362460392472597856744565302358359445734245043980424773278398951371661572698658727805417953038913487302253899276974700109264437557094889196411186049385292003076282390608251358299562384712397723495785046531106033124937667103765688322828670742086429727630709181560115309248261867700932677989072719877301136052613276218465401911892795825482660887945581101633005687039529655844457562091564631033523933674658869087942643392916951458280932186485156552727011345928983728150742894314248322057743613989758052903320801590328205219454272561592671840565741225200839895133283236931751033123485561289573241690783554902968148198017417574856025530389896252261318327946623122632106512831476251931685783569108817813918238506000333181418303820066582093473331751466806531154078854308523504690437343287972114729692426240643430323019046975685815750236159278435288167068640044223759109259993992628920886595671455347511507231496017502369857058141568665008557143503672420167525605046594574313938567924570027900806089589372319371749717696149284630134473009205526245718728462322846715857591134949911394475247367087711102629096342256273635674621439564539376259367650142753731774666956065794464307327625362901842302231829010493680432510186700553284782720177489499136198043493641165315576231012520583143678587171526490941170531419591851824986850129837804713377930430937867728904517503636818378760858613989914274214844679803455418549953046931226380442619181418879225991890800508933741615830949216972904697824110876480037319978683551277501845924485926753160807404721379248579071472739428168699500972189209631339885665136437683597648598557784833180978010347299224375173901781485513519539024457802054444515564218965641195981853711378029307984335367218995204876661773997509979137537487450019046450008208300099604656233978807962920448038012571137429205257629063421706547701074607899796437047754059695017949402110462697920596337166729324858356645615736125077059116129497473824542231489931656730892950863426134811280628053252347621317830725207799774608351329454922776709193806721762496462837090669729140280180894079328464066065506207228520119396724244189370713572229002397939667188483517743251803565742915268749944243867225212764743705158977319484305843822519889630886063048977782832966788610430288209228908228876390152718706832765207160362484889571058167612073716575364160779827280860870293152426357998356814433209517314391700679103947717388802065031966375980829949452912832559398339587488646546102796716340216184188378170874325881546263569330481304955858505942364569649678908281039772488605477158156383152283653673706490429954458988433590400031991512121150939865476077508292344762743181030632662127299554627020057662377872578693660164304756290387627056315234611320381666541564732277231080586283032890771562736451900236510013773769965567197456518986946832131007177102681793131413606227198078611941888466789674424898802277691861239262130724484671556514167373019715904512616976262123747711809562700749517103990566832565907715779215988494394522228863992726667342256838381471324935851576910484131955102336943728855246651195511780897876361958660893213569485048355226318800355505064323600729511213622346059477905472669678651046280393208982456585684839662557759874973017256345141524718152218786109810167629477439980963435006876297082062570254237279505275026299146309903789487505612949972844582420194982480505924978999182670388033508618359772763532623004028780480922990472432785254853342381111649877941137794148763602554891071660356120871946844581641234056359049840039924478403614928404945754091519447450509797268389790667936570831405434790567633787807078360466286124807787801652782179050594603282334884117717863438699318279794384253943441985835285250346429558143023658860977029717629275675196818457389753852691404304716981498456170755121882194979022620264114010737060850644623746614986296463261511614382394079759688043991334952870902078848332360247540409593072991423739236009523098217053822990796649508600289204017307567843937300370190976240299287502856703719783899328471838601429023807043383009566371614765473471705971538629721742471829332105646777476204397032798465815238050069424444575989146946519055770080047074417518568065418982192341362053758327522289876138166391086454045655164602198706665482854431740074691411738710211808035184906695533966455251403234600111903277516625358262142209261387684462646237390327343077275511651143508790667094489455265477828469719289568841805485723671942132051310869404512783093769697629767499019867415680009432274898162610468776172067118173433705564489497042538067700103400084974327091259114132556395953719208598762033049084047653608388627676192500129434673250563380963114391718321695664763975236063115783730058015301039291958269658179402766253362430503735426387916347022270763624813490530529668527291048390828284710483199822843485897413096115127429720943891706418398763396281635720551926174025204064421378839333914111873969797996825697890911297318186801962088688998718748189005589536798380145900279907681883798666388372421774691090231691506509325708314519270492468907215970376924907043630948173620175670223214056753477855836577596600677053214049070128783184681560886344320973018284551929189676791136826751799265658429882650747706319527446207185196063833955882095770594540898236724111798667471824991885966820928148869993405944104425036234994451895073308737895230524668373477804847235854245637891628692067759514097632218395775422786300279688144534247303606034181654497229105901684077890213108012436220044131475755605510729929027247438680526843470281329622658242792418868595736397333469005717707502544891573792184708449683922260911329721040790211666661061482421391110664148680546756338014389177079075507716764603729176028786091811809383532620178151089029931593805353291188505332206434898568387201498688062440085152248459179380221026843810391114264049577284883507252997468521822653068063500094956584737845496766043111681785149719609829377644956753841750729826986058664111028646650105709452879554414932785699975601173576480972766681818888269266312968562730730829284178893289678217196767548098956203941350819078722214237146691591730005803357395675803541040374028695359031302385541994015630929109876255390759505542019402518851977347853238373819012572796192762782695701091230060767715332868252318584776805622365816137206395659417395867898495643481226502534186743832012789720686614513743112812032067577792727807928818519054077251634647419446494975411753726077660632188040937789386025297893885593627909873601607498049568050675135887494502424838282483931202245890365088171140181109972713252717071591701645264964806828154361310441251465751413309472893193048673213033023786023330000885172877105489294001863486657969476119153583710203891352140315956633747643890350682294269686087846522837551763339371109235585687901632806767448002155842694091119753736097353160988884320314726188639152690176205582986163193267633491811081667732813428731840343364475202548796055963634626086746127371085107690051794646848527249397455224931177472707413566714519911153322246688196661606390002820581483883280920547838312377901828446409579929174760063143226166254144492970910087222637314546102502926681410079468055828796082801636737761637729955107686120981197106091299853636396194983063584896012395022183664504012336773315585760727896177872270132976583176869514273929142076150718233797894609941736759936992561851473484195607237885997509693479819332803228655036333135032498613250046473862562468508488636670307086347268624376295559681445072742366986732952146227717767962984121236945385910360249186023702009566360407790546615124281345315230810318986980119435181467210842144771752147436604602497486571544670016685820088835462770589416783164764263743705828573255536065977734966991306721699631888840969412113873432802118804128679370544175727811836212238662247012326315847047413385745519117020314595915903014347403352516173377096448255253265935091266374229523735756716731202651546717174301390399455727092762892448517029830184004330818078682568020800948956811401548720966874954096727704167666960108050786793106803881055595423653683861667535149157160887285235260516861925693603438034979716397130774734151491188806845006391290874578731631693433424386284150616289610485141162987619953814530677529165661445380432349892506778210438844950580972492340181411097516175795771302288600447" };
        maxX = HighPrecision{ "-0.165353054425834327068347550055335092120094241000026380415912481478846048996851264628574322423560266690587753860396550875628196105603940031447124507284796134645582380068094647607090297791540990497988541731977355180190520753027467465559816468674235485307967922195244020435598058452547945052640411625622380447543438758451505365411536195304664553213862187796767606743602535567713694658419041684193170231020290015597294881195401590590515687821394886994681228468823121619355762746143272331336588126133019869297364991848070238405297898977565831863782091635917688802769483971793868164049793382772144824691474021637422472581775598186763066262139585911209575570234948287138025858113770595417057765737841033586118537409729682731556704538168437889315511648671541179084394324381949630007598456990590122257489304269635986994729167818299362941847397307880932756823683547959153931416943468484997414042002226760352771502213545178733070190221064342199610334840881387030896218005684794531506673212756591521716283419037782654703941428719602480473210753190164633542506076510382975888406693578584596235047040945129391014749050860448846296782715940820928910430871080443380315527241808928932896928716039119443897790852490935543890817992936338937709268027467604182378253232485293312376943190272519179472927080530690648707898886007389680583190917306726096770998920761881206074115907690857056150891228681662452179492284790954328662430430656981970897085077471610297695950783746253592253527326193624697532694941942188724771208983480291886159387242725682612097829455543312890683809621756625033935103275511262132919872958003878843292448425265992348158407306443320832946249494297083297065386410865236181532967975846511540395885398967800451405872354471111541143118834281394086642513628585599191663767876630934261330738879007819927233496564762558362758418390567667209759340681400136029227832363343261284047156793237563769392991347582335842415147800214034819105679766895896190888487104322278865433401758695548688802612394860403944960107124678192573678989832899795415529229163840768056022104966650132921903672249536059325757089871496643859049887770030871044773077334163207114053337847994782596615178782492108495599767514996566744826625746330516783048178696834588961051566698640739429470125374731896593223653194204370639799254530974220119945225314385941207239585116703173521826441463626562708474709601500726612149467260167281041829123970618934980097324455704814315034834953682328758056771526998951795691697930367447004451518375515418169353884080315386805071944214945628326272206986611580426928212750995991822823860026677580567579124012249971331946408816729835427454567800436845776613039824177570729611229205776326703592784705416074092879853581588411484950856055747980311929635987723234642038360349774236752092955179056821468348024666783398270724421129003133127090314443705720069574303174193228724845820259116590995276466078757283233798452159659421707779991435140369151226234965251861878363669272417784157764671849926431585329404200513542914120936540490370523737679542261902716105031899047000574672763617580363870626629525533768506115991045406909783797753803746089087264977840263005711189494266590101966689231712756256470547792744777553927079757974120385717870790951384830911886978567420529971596544013707239416091341585384871990520957060109772736365748978193905106534801222821369395263033319870549174850155278336102979440391140479756393865532583065213267686335482313265711452613988394900061698931036844895917649122714468973535110221199701045387093364040088983148440901558565343569933842766230129144889523286868632078730811819589924157257618828836418991854941183745815617753130829885427605738332351815468800041614844411948734214373134474159935836830528838008902014084989040613766534015144169630407744947384870577540641433410805373824081271884349456445980467227805939858029823443636433375993565736010527634777140361646143747611462976125176422699484335744600731469227994871545083124569882828687719279622761878666910489636170143111644396194740107754249102526504602513511057314389106209059481710370014957772773493996517864927750259751417897780388304152123559071586995936393040390811267657212490363383794594817939326816915052108702617526830125590381626179273747346711161718632746010506896492152507657949850685205394267271069232829330314113981044847399353982571187682200220050685109212015810345847374980077166290560814788482099061174945177497777209829366064227056160852469309701939873197934997935146823283624577874026529594466861139705283234747077286219792074376315246333330706693906228230365874533010214032675713142686895233586242015050484327701262570458880321667614601233227675479227672224641408664879748669522025277685622575570050182195870035227678915279535025936583502036576519349295338276798472988538693559511220010233201916228704019126372083140407535079980191493057561086714211574430100432792620328609268319520304945854385497936141901443101199205052471093936689979860173883780344114252131947436942302349503752793374627732075522786101926735579515622542622317922002746879698386356592865907671648352406117662021351985545998783349330524914719609285041426256404473243191913367856856505245053658272307095369129220162266268420386638737793203594207202582540456171596060541143750558367075091060276815628477967236278834778223264263609185470196025869508929722398462967645617597794592799099406939069220012992699606111668791972386713160194753183837116190835484443893600248426067621563375877433923433053986834276941578172577169480739698040727927051070813667053486331251784829654417512411597664042111767367663367872406811253438297624215855991182989731954995212058872814504049337329693272918413538243524526839378807754264029690341466186361044512855032557883729700966098830281872407442880004870470584972051023774452512883045739712345718037055286717487106553056063485228487300596391087139454082167573918246552383569274381849009436037589909083139992944782141686680096524533568170424858925610407896477023227911013714730552946622590798621241352950011051227518196183154105030003364985768251976119377457107417619857186239273885250163431996516953720197171826621833921539290435578458700196733937764086363546782270852887856001397001975396643041276905170370449927708191342356000491510498182329289727839105416752909777142489501312569846474792156750775748670181730686556349529944551833099392024954663901742552812089011917592931200119202193015248773456711015371790869584374293209887842875062426035294797893890969787057912885144174509693572729486441063536053468860625486553264201741128373776071355340209336602300388938993187197178640398069203314482289353518901337019848286111326088235409923686674159952161562480844551090683390209994136598461029533764993305238873823339363942706577414784359492517592521760479266623019971610190663368517350825471750623101329733732676875134072821397170059942499383337984758966986663628920758061051763908186554368494548431347538961093279045258986114008848070935719505927929844387038392643409634869617879553144747707394902135413073731883743987911165789174348634634075053249592498046647045816047976293211941960381162024603750637280577686138131445369610346686061122266698851206484285402672323270264993345352262842622989814676592089051166636314896577991170492999858475643511699922806010673865196459988411983646144529395110965896266135582918090788910655653533977096537938917321840610415306482518110713209759360969812808423903607312327852649389369181278717908373707127250848414738760284626613909239977215884435918888044487983110245967872307235049034446988418503531952968982788520561075780654892652683629294901141771279510968052360539748738836432790792043526253652531307039188456936412054181860042128306868922924285956289336711532018612706267065365750882954648327718572091225202058150597266420194411112146793406903002760749685050404726389139753261108075164325117098447603641627172342335141252879668047403442496156686098362967003280803703751496079357587699689190716273868541142530992703910472555486651898428792369378140669670598661156878545238432732583504801709877711760652053644181148221932669718344567800083561816029230283328631350629687152285361481168846243317678250175658617315776244662741098180364247827206432903047652330073958450747982982320095249262502767794991533685844967431946590301552167994682578091262037108944986972736919718880408275845321966274622598243083637536104897294413921638954282376280884695105128267347686829582020636483391132026909471106841010974160244514587380606659287045028970009321283953825219937227139813381763670073075883942127389061316189749902383423167382658738580952213198804945769600776180384319100485879028253262657134710335293606535775389088853727315293100876079952109293344088103971951480494487754596407141467589307223384665151440511539065850505118855113045469618352702700421864081721157910734272327967917845850706068618905313969993629417189019440249915526790185446294203281664821715945912860879531821023789062130909466236932112134608209750365826535844639826854943861059761050817065936005572897716927656788711329860395413268756189922094161632961802975376296442212156031933282337702006954782696327044422873119138302780500207468796009428403959514219092778403693156424490285561767859067764070637929433238019135934620127159589148688663316333823324271609322641842758694003289203201044936650228178758315741220758011252572074991096068049106618012555191310446255165370526415173362113768333789599943675471326422117676334314266747968775024668083580670963192468694653336377644206322697597857619326333187182289423076171533936691489200188605165418650423116697995608160056812035697001549422601519842317117999137345435699155672279037430567745010659244889782939560113956308388433906461427370380377343080248029365213168450436915758404812661842554630512632919783992213543783257176855924460873683896564206216084026461806133687175646248722280968933802897270153863464293042013597529870325255088275718365644698293584089082828539443700361774585106480045756145535019626597646855793743408522811671335762791511468437595742168024409937201303441774585934220394538744700727446992991260442240520048636797756024810930146640135842185194897203606389769729463321849087906112691468658396157387297362275420411739066157736983641068114823469133313427620116880312881225892298889836488671172799822225946955020411149420933774586872339848592610488869714677843758358403321183181151962308158619307215281277740258543759337972503873005561599434391046500951009505597762560623708223965512068456513780654219899954375597512828103722798859948308156553994793288430241152986924485177472873015909873812123339226879255020117083749194681650812041777524412244774192327973441088010851283224914045003379940557946972586822089061969203048382548361630863652133124621331655531715660733353043243403778897569533996464116954872159905058467494495292935274845521175643055262623565410797581021224500703211015693978874893183400350124608416415017323117550185536204198832506366226190342179845404215472069530383611799921404450662049484630193282375132850487865342674796" };
        maxY = HighPrecision{ "1.044277064897808805382972133514970033636669752029592433606621179011324460839154407760083401975587084514485665944078137412806407002865995435448320301379853171431018504164501791882937915959701460308680789833538940017615109204052114378534599805568913143433721403046439205869053847721440465614404636728874424936043699640930317900856017266059437572440389878746913170475829036230984669372185446989891582627851149298755344310847731020716343893905754382775262464527952786926687476749161966321307577475141911922933647690917552245194783623324663570183784753320876701599806414191859791194254383665847508813676303149564416002169543425982434473639175690502231884238779339636937927115677133856026720254304320912596326304195322539815227435108222145097458923796047746425910364932925829558006584166443664314352307264148382970891408988979062525968381056334335679045316392262774401410748913408401726603888508786068502561916923020865667192778753718590384998567343707540582590893641598755683111418781587221104872163730028193288726016114387032296975334783851084675237953433914071341923149575110764141419210252231173182740253477868436375249646101622314227416191146690240627169089555377962857953078414928044169108256927942349459910125948525888192514802979638130928884513689597676506227315633736859599410454347573595454383817394748935022334082255135551120400099019118998110307133409174242933088693341598449819928220259155678207221017769141187925530967420324226482642441837300165695640086994966587267298764816111987953532512141499935149352635859544158772359773227213299237214707647808099345149682795294672490521431109103068590372323430327921486018543311757823497567089023535287679052758409628828210249156117254088355377867901218420558414958281589870956330589688851321046690330125132544129530984221717837455562610519873328532560918889430601135107973345768817209608974207833250852667693795380646425391225392082513724949494476435790542977963418137359618212807646890057457386413399936496057488784167877323590824491613168801296594688801074154295032863107970487782416194310185519986529107358945248597178434959280827505375827413973770061227045403202358478813286345708064925273726005856759751937216053015686038333725737654461907455694752702861260355779262675620119674671289569048117869163731332885380961395242878038848030107587180419281007199403918358955680615012816392399655408092220719785110435589430058279851676911045619947317660628503348934794278085469249162338078185740166811898328201046771253939036701986468067018849544676093985646163517443669257597327605998680263234841056781142259599866627942176432187101723893470471368191462007769835291146141961003897316596588350625193110626780157374543624603924725978567445653023583594457342450439804247732783989513716615726986587278054179530389134873022538992769747001092644375570948891964111860493852920030762823906082513582995623847123977234957850465311060331249376671037656883228286707420864297276307091815601153092482618677009326779890727198773011360526132762184654019118927958254826608879455811016330056870395296558444575620915646310335239336746588690879426433929169514582809321864851565527270113459289837281507428943142483220577436139897580529033208015903282052194542725615926718405657412252008398951332832369317510331234855612895732416907835549029681481980174175748560255303898962522613183279466231226321065128314762519316857835691088178139182385060003331814183038200665820934733317514668065311540788543085235046904373432879721147296924262406434303230190469756858157502361592784352881670686400442237591092599939926289208865956714553475115072314960175023698570581415686650085571435036724201675256050465945743139385679245700279008060895893723193717497176961492846301344730092055262457187284623228467158575911349499113944752473670877111026290963422562736356746214395645393762593676501427537317746669560657944643073276253629018423022318290104936804325101867005532847827201774894991361980434936411653155762310125205831436785871715264909411705314195918518249868501298378047133779304309378677289045175036368183787608586139899142742148446798034554185499530469312263804426191814188792259918908005089337416158309492169729046978241108764800373199786835512775018459244859267531608074047213792485790714727394281686995009721892096313398856651364376835976485985577848331809780103472992243751739017814855135195390244578020544445155642189656411959818537113780293079843353672189952048766617739975099791375374874500190464500082083000996046562339788079629204480380125711374292052576290634217065477010746078997964370477540596950179494021104626979205963371667293248583566456157361250770591161294974738245422314899316567308929508634261348112806280532523476213178307252077997746083513294549227767091938067217624964628370906697291402801808940793284640660655062072285201193967242441893707135722290023979396671884835177432518035657429152687499442438672252127647437051589773194843058438225198896308860630489777828329667886104302882092289082288763901527187068327652071603624848895710581676120737165753641607798272808608702931524263579983568144332095173143917006791039477173888020650319663759808299494529128325593983395874886465461027967163402161841883781708743258815462635693304813049558585059423645696496789082810397724886054771581563831522836536737064904299544589884335904000319915121211509398654760775082923447627431810306326621272995546270200576623778725786936601643047562903876270563152346113203816665415647322772310805862830328907715627364519002365100137737699655671974565189869468321310071771026817931314136062271980786119418884667896744248988022776918612392621307244846715565141673730197159045126169762621237477118095627007495171039905668325659077157792159884943945222288639927266673422568383814713249358515769104841319551023369437288552466511955117808978763619586608932135694850483552263188003555050643236007295112136223460594779054726696786510462803932089824565856848396625577598749730172563451415247181522187861098101676294774399809634350068762970820625702542372795052750262991463099037894875056129499728445824201949824805059249789991826703880335086183597727635326230040287804809229904724327852548533423811116498779411377941487636025548910716603561208719468445816412340563590498400399244784036149284049457540915194474505097972683897906679365708314054347905676337878070783604662861248077878016527821790505946032823348841177178634386993182797943842539434419858352852503464295581430236588609770297176292756751968184573897538526914043047169814984561707551218821949790226202641140107370608506446237466149862964632615116143823940797596880439913349528709020788483323602475404095930729914237392360095230982170538229907966495086002892040173075678439373003701909762402992875028567037197838993284718386014290238070433830095663716147654734717059715386297217424718293321056467774762043970327984658152380500694244445759891469465190557700800470744175185680654189821923413620537583275222898761381663910864540456551646021987066654828544317400746914117387102118080351849066955339664552514032346001119032775166253582621422092613876844626462373903273430772755116511435087906670944894552654778284697192895688418054857236719421320513108694045127830937696976297674990198674156800094322748981626104687761720671181734337055644894970425380677001034000849743270912591141325563959537192085987620330490840476536083886276761925001294346732505633809631143917183216956647639752360631157837300580153010392919582696581794027662533624305037354263879163470222707636248134905305296685272910483908282847104831998228434858974130961151274297209438917064183987633962816357205519261740252040644213788393339141118739697979968256978909112973181868019620886889987187481890055895367983801459002799076818837986663883724217746910902316915065093257083145192704924689072159703769249070436309481736201756702232140567534778558365775966006770532140490701287831846815608863443209730182845519291896767911368267517992656584298826507477063195274462071851960638339558820957705945408982367241117986674718249918859668209281488699934059441044250362349944518950733087378952305246683734778048472358542456378916286920677595140976322183957754227863002796881445342473036060341816544972291059016840778902131080124362200441314757556055107299290272474386805268434702813296226582427924188685957363973334690057177075025448915737921847084496839222609113297210407902116666610614824213911106641486805467563380143891770790755077167646037291760287860918118093835326201781510890299315938053532911885053322064348985683872014986880624400851522484591793802210268438103911142640495772848835072529974685218226530680635000949565847378454967660431116817851497196098293776449567538417507298269860586641110286466501057094528795544149327856999756011735764809727666818188882692663129685627307308292841788932896782171967675480989562039413508190787222142371466915917300058033573956758035410403740286953590313023855419940156309291098762553907595055420194025188519773478532383738190125727961927627826957010912300607677153328682523185847768056223658161372063956594173958678984956434812265025341867438320127897206866145137431128120320675777927278079288185190540772516346474194464949754117537260776606321880409377893860252978938855936279098736016074980495680506751358874945024248382824839312022458903650881711401811099727132527170715917016452649648068281543613104412514657514133094728931930486732130330237860233300008851728771054892940018634866579694761191535837102038913521403159566337476438903506822942696860878465228375517633393711092355856879016328067674480021558426940911197537360973531609888843203147261886391526901762055829861631932676334918110816677328134287318403433644752025487960559636346260867461273710851076900517946468485272493974552249311774727074135667145199111533222466881966616063900028205814838832809205478383123779018284464095799291747600631432261662541444929709100872226373145461025029266814100794680558287960828016367377616377299551076861209811971060912998536363961949830635848960123950221836645040123367733155857607278961778722701329765831768695142739291420761507182337978946099417367599369925618514734841956072378859975096934798193328032286550363331350324986132500464738625624685084886366703070863472686243762955596814450727423669867329521462277177679629841212369453859103602491860237020095663604077905466151242813453152308103189869801194351814672108421447717521474366046024974865715446700166858200888354627705894167831647642637437058285732555360659777349669913067216996318888409694121138734328021188041286793705441757278118362122386622470123263158470474133857455191170203145959159030143474033525161733770964482552532659350912663742295237357567167312026515467171743013903994557270927628924485170298301840043308180786825680208009489568114015487209668749540967277041676669601080507867931068038810555954236536838616675351491571608872852352605168619256936034380349797163971307747341514911888068450063912908745787316316934334243862841506162896104882759011173888435540435768368409617460551906764851402352987911282927860834380384330694648169753339317485999784" };
        SetNumIterations<IterTypeFull>(2147483646);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
        break;

    case 22:
        minX = HighPrecision{ "-0.6907023001770585288601365254896697520219123564180210586974026317473887891717116357627686720779756408687304394424933003585901989015412142" };
        minY = HighPrecision{ "0.3270967712951810044828437245522451618089794418141623643693494431352454845268379424333502401886231660401278668233377275970460192695489989" };
        maxX = HighPrecision{ "-0.6907023001770585288601365254896697520219123564180210586974026317473887891717116357627686720779756408687304394424911840042901546067160599" };
        maxY = HighPrecision{ "0.3270967712951810044828437245522451618089794418141623643693494431352454845268379424333502401886231660401278668233395871670243248566020344" };
        SetNumIterations<IterTypeFull>(2147483646);
        ResetDimensions(MAXSIZE_T, MAXSIZE_T, 4);
        break;

    case 23:
        minX = HighPrecision{ "-0.74854022336364298733173771780874971621677632881659318116648007147146915304640714984571574330253031" };
        minY = HighPrecision{ "0.064687493307404945848262842037929928847080557679638135501670005239098231744262964560868111971917423" };
        maxX = HighPrecision{ "-0.74854022336364298733173771780874971503929930401196726764350216120120400755946688800146880673948663" };
        maxY = HighPrecision{ "0.064687493307404945848262842037929930024557582484264049024647915509363377231203226405115048534961104" };
        SetIterType(IterTypeEnum::Bits64);
        SetNumIterations<IterTypeFull>(51'539'607'504llu);
        break;

    case 24:
        minX = HighPrecision{ "-0.37656235351675001355449781905149492799682880545941078435228171582820374005540296873316709319956486" };
        minY = HighPrecision{ "0.67158947867625786414258056245749919392606048757435786795647808633338740646688656232915470730827391" };
        maxX = HighPrecision{ "-0.37656235351675001355449781905149153207098115802132341950557930099971431732574605202825177873455922" };
        maxY = HighPrecision{ "0.67158947867625786414258056245750258985190813501244523280318050116187682919654347903407002177327955" };
        SetIterType(IterTypeEnum::Bits64);
        SetNumIterations<IterTypeFull>(51'539'607'504llu);
        break;

    case 25:
    {
        // from Claude Heiland-Allen
        // Low-period, but hard to render
        // 32-bit + perturbation only
        PointZoomBBConverter convert{
            HighPrecision{ "3.56992006738525396399695724115347205e-01" },
            HighPrecision{ "6.91411005282446050826514373514151521e-02" },
            HighPrecision{ "1e19" }
        };

        minX = convert.minX;
        minY = convert.minY;
        maxX = convert.maxX;
        maxY = convert.maxY;
        SetNumIterations<IterTypeFull>(1'100'100'100);
        break;
    }

    case 26:
        minX = HighPrecision{ "-0.1605261093438198889198301833197383152634624627924018099119331622744150582179648" };
        minY = HighPrecision{ "-1.037616767084875731043652452075011878717204406363659116578730306577073971695717" };
        maxX = HighPrecision{ "-0.1605261093438198838959220991033570359417441984488856377425735296717555605747732" };
        maxY = HighPrecision{ "-1.037616767084875726019744367858630599395486142020142944409370673974414474052526" };
        SetNumIterations<IterTypeFull>(500'000'000);
        break;

    case 27:
    case 28:
    case 29:

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

    CleanupThreads(false);
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

    IterTypeFull baseIters = GetNumIterations<IterTypeFull>();
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
            SetNumIterations<IterTypeFull>(baseIters + Convert<HighPrecision, unsigned long>(result));
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

        m_RefOrbit.ResetGuess();

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

    uint64_t numberAtMax, numberAtMaxFinal = 0;
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

                numberAtMax += m_CurIters.GetItersArrayValSlow(x, y);
            }
        }

        percentAtMax = (HighPrecision)numberAtMax / ((HighPrecision)sizeX * (HighPrecision)sizeY * (HighPrecision)GetNumIterations<IterTypeFull>());

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
    if (m_MinX != m_MaxX ||
        m_MinY != m_MaxY) {
        m_PrevMinX.push_back(m_MinX);
        m_PrevMinY.push_back(m_MinY);
        m_PrevMaxX.push_back(m_MaxX);
        m_PrevMaxY.push_back(m_MaxY);
    }
}

///////////////////////////////////////////////////////////////////////////////
// Functions for dealing with the number of iterations
///////////////////////////////////////////////////////////////////////////////
template<typename IterType>
void Fractal::SetNumIterations(IterTypeFull num)
{
    if (num <= GetMaxIterations<IterType>())
    {
        m_NumIterations = num;
    }
    else
    {
        m_NumIterations = GetMaxIterations<IterType>();
    }

    m_ChangedIterations = true;
}

template void Fractal::SetNumIterations<uint32_t>(IterTypeFull);
template void Fractal::SetNumIterations<uint64_t>(IterTypeFull);

template<typename IterType>
IterType Fractal::GetNumIterations(void) const
{
    if constexpr (std::is_same<IterType, uint32_t>::value) {
        if (m_NumIterations > GetMaxIterations<IterType>()) {
            ::MessageBox(NULL, L"Iteration limit exceeded somehow.", L"", MB_OK);
            m_NumIterations = GetMaxIterations<IterType>();
            return GetMaxIterations<IterType>();
        }
    }
    else {
        static_assert(std::is_same<IterType, uint64_t>::value, "!");
    }
    return (IterType)m_NumIterations;
}

template uint32_t Fractal::GetNumIterations<uint32_t>(void) const;
template uint64_t Fractal::GetNumIterations<uint64_t>(void) const;


template<typename IterType>
constexpr IterType Fractal::GetMaxIterations(void) const {
    return ((sizeof(IterType) == 4) ? (INT32_MAX - 1) : (INT64_MAX - 1));
}

IterTypeFull Fractal::GetMaxIterationsRT() const {
    if (GetIterType() == IterTypeEnum::Bits32) {
        return GetMaxIterations<uint32_t>();
    }
    else {
        return GetMaxIterations<uint64_t>();
    }
}

void Fractal::SetIterType(IterTypeEnum type) {
    if (m_IterType != type) {
        m_IterType = type;
        m_RefOrbit.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        InitializeMemory();
    }
}

void Fractal::SavePerturbationOrbit() {
    m_RefOrbit.SaveAllOrbits();
}

void Fractal::LoadPerturbationOrbit() {
    m_RefOrbit.LoadAllOrbits();
}

Fractal::IterTypeEnum Fractal::GetIterType() const {
    //
    // Returns the current iteration type
    //

    return m_IterType;
}

void Fractal::ResetNumIterations(void)
{
    //
    // Resets the number of iterations to the default value.
    //

    m_NumIterations = DefaultIterations;
    m_ChangedIterations = true;
}

RenderAlgorithm Fractal::GetRenderAlgorithm() const {
    //
    // Returns the render algorithm to use
    //

    if (m_RenderAlgorithm == RenderAlgorithm::AUTO) {
        PointZoomBBConverter converter{ m_MinX, m_MinY, m_MaxX, m_MaxY };
        if (converter.zoomFactor < HighPrecision{ 1e4 }) {
            // A bit borderline at 3840x1600 x16 AA
            return RenderAlgorithm::Gpu1x32;
        }
        else if (converter.zoomFactor < HighPrecision{ 1e9 }) {
            // Safe, maybe even full LA is a little better but barely
            return RenderAlgorithm::Gpu1x32PerturbedLAv2PO;
        }
        else if (converter.zoomFactor < HighPrecision{ 1e34 }) {
            // This seems to work at x16 AA 3840x1600
            return RenderAlgorithm::Gpu1x32PerturbedLAv2;
        }
        else {
            // Falls apart at high iteration counts with a lot of LA
            return RenderAlgorithm::GpuHDRx32PerturbedLAv2;
        }
    }
    else {
        // User-selected forced algorithm selection
        return m_RenderAlgorithm;
    }
}

std::string Fractal::GetRenderAlgorithmName() const {
    //
    // Returns the name of the render algorithm in use
    //

    return RenderAlgorithmStr[static_cast<size_t>(GetRenderAlgorithm())];
}

///////////////////////////////////////////////////////////////////////////////
// Functions for drawing the fractal
///////////////////////////////////////////////////////////////////////////////
bool Fractal::RequiresUseLocalColor() const {
    //
    // Returns true if the current render algorithm requires the use of
    // CPU-based color implementation
    //

    switch (GetRenderAlgorithm()) {
    case RenderAlgorithm::CpuHigh:
    case RenderAlgorithm::CpuHDR32:
    case RenderAlgorithm::CpuHDR64:
    case RenderAlgorithm::Cpu64:
    case RenderAlgorithm::Cpu64PerturbedBLA:
    case RenderAlgorithm::Cpu32PerturbedBLAHDR:
    case RenderAlgorithm::Cpu32PerturbedBLAV2HDR:
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
    case RenderAlgorithm::Cpu64PerturbedBLAV2HDR:
        return true;
    default:
        return false;
    }
}

void Fractal::CalcFractal(bool MemoryOnly)
{
    if (GetIterType() == IterTypeEnum::Bits32) {
        CalcFractalTypedIter<uint32_t>(MemoryOnly);
    }
    else {
        CalcFractalTypedIter<uint64_t>(MemoryOnly);
    }
}

template<typename IterType>
void Fractal::CalcFractalTypedIter(bool MemoryOnly) {
    SetCursor(LoadCursor(NULL, IDC_WAIT));

    // Test crash dump
    //{volatile int x = 5;
    //volatile int z = 0;
    //volatile int y = x / z;
    //}

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
    // Note: This accounts for "Auto" being selected via the GetRenderAlgorithm call.
    switch (GetRenderAlgorithm()) {
    case RenderAlgorithm::CpuHigh:
        CalcCpuHDR<IterType, HighPrecision, double>(MemoryOnly);
        break;
    case RenderAlgorithm::CpuHDR32:
        CalcCpuHDR<IterType, HDRFloat<float>, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu32PerturbedBLAHDR:
        CalcCpuPerturbationFractalBLA<IterType, HDRFloat<float>, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu32PerturbedBLAV2HDR:
        CalcCpuPerturbationFractalLAV2<IterType, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLAV2HDR:
        CalcCpuPerturbationFractalLAV2<IterType, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64:
        CalcCpuHDR<IterType, double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::CpuHDR64:
        CalcCpuHDR<IterType, HDRFloat<double>, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLA:
        CalcCpuPerturbationFractalBLA<IterType, double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLAHDR:
        CalcCpuPerturbationFractalBLA<IterType, HDRFloat<double>, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64:
        CalcGpuFractal<IterType, double>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x64:
        CalcGpuFractal<IterType, MattDbldbl>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu4x64:
        CalcGpuFractal<IterType, MattQDbldbl>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32:
        CalcGpuFractal<IterType, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x32:
        CalcGpuFractal<IterType, MattDblflt>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu4x32:
        CalcGpuFractal<IterType, MattQFltflt>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32:
        CalcGpuFractal<IterType, HDRFloat<double>>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32Perturbed:
    case RenderAlgorithm::Gpu1x32PerturbedPeriodic:
        CalcGpuPerturbationFractalBLA<IterType, float, float, false>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32Perturbed:
        //case RenderAlgorithm::GpuHDRx32PerturbedPeriodic:
        CalcGpuPerturbationFractalBLA<IterType, HDRFloat<float>, float, false>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x32PerturbedScaledBLA:
        CalcGpuPerturbationFractalScaledBLA<IterType, double, double, float, float>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedScaled:
        CalcGpuPerturbationFractalScaledBLA<IterType, HDRFloat<float>, float, float, float>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64Perturbed:
        CalcGpuPerturbationFractalBLA<IterType, double, double, false>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
        CalcGpuPerturbationFractalBLA<IterType, double, double, true>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x32Perturbed:
        // TODO
        //CalcGpuPerturbationFractalBLA<IterType, dblflt, dblflt>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        // TODO
        //CalcGpuPerturbationFractalBLA<IterType, double, double>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedBLA:
        CalcGpuPerturbationFractalBLA<IterType, HDRFloat<float>, float, true>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx64PerturbedBLA:
        CalcGpuPerturbationFractalBLA<IterType, HDRFloat<double>, double, true>(MemoryOnly);
        break;


    case RenderAlgorithm::Gpu1x32PerturbedLAv2:
        CalcGpuPerturbationFractalLAv2<IterType, float, float, LAv2Mode::Full>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedLAv2PO:
        CalcGpuPerturbationFractalLAv2<IterType, float, float, LAv2Mode::PO>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedLAv2LAO:
        CalcGpuPerturbationFractalLAv2<IterType, float, float, LAv2Mode::LAO>(MemoryOnly);
        break;

    case RenderAlgorithm::Gpu1x64PerturbedLAv2:
        CalcGpuPerturbationFractalLAv2<IterType, double, double, LAv2Mode::Full>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64PerturbedLAv2PO:
        CalcGpuPerturbationFractalLAv2<IterType, double, double, LAv2Mode::PO>(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64PerturbedLAv2LAO:
        CalcGpuPerturbationFractalLAv2<IterType, double, double, LAv2Mode::LAO>(MemoryOnly);
        break;

    case RenderAlgorithm::GpuHDRx32PerturbedLAv2:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<float>, float, LAv2Mode::Full>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2PO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<float>, float, LAv2Mode::PO>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<float>, float, LAv2Mode::LAO>(MemoryOnly);
        break;

    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::Full>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::PO>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>, LAv2Mode::LAO>(MemoryOnly);
        break;

    case RenderAlgorithm::GpuHDRx64PerturbedLAv2:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<double>, double, LAv2Mode::Full>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2PO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<double>, double, LAv2Mode::PO>(MemoryOnly);
        break;
    case RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO:
        CalcGpuPerturbationFractalLAv2<IterType, HDRFloat<double>, double, LAv2Mode::LAO>(MemoryOnly);
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
    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

int Fractal::GetPaletteDepthFromIndex(size_t index) const
{
    switch (index) {
    case 0: return 5;
    case 1: return 6;
    case 2: return 8;
    case 3: return 12;
    case 4: return 16;
    case 5: return 20;
    default: return 8;
    }
}

void Fractal::UsePalette(int depth)
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

void Fractal::UseNextPaletteDepth() {
    m_PaletteDepthIndex = (m_PaletteDepthIndex + 1) % 6;
    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

void Fractal::UseNextPaletteAuxDepth(int32_t inc) {
    if (inc < -5 || inc > 5 || inc == 0) {
        return;
    }

    if (inc < 0) {
        if (m_PaletteAuxDepth == 0) {
            m_PaletteAuxDepth = 17 + inc;
        }
        else {
            m_PaletteAuxDepth += inc;
        }
    }
    else {
        if (m_PaletteAuxDepth >= 16) {
            m_PaletteAuxDepth = inc;
        }
        else {
            m_PaletteAuxDepth += inc;
        }
    }

    auto err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }
}

int Fractal::GetPaletteDepth() const {
    return GetPaletteDepthFromIndex(m_PaletteDepthIndex);
}

void Fractal::ResetFractalPalette(void)
{
    m_PaletteRotate = 0;
}

void Fractal::RotateFractalPalette(int delta)
{
    m_PaletteRotate += delta;
    if (m_PaletteRotate >= GetMaxIterationsRT())
    {
        m_PaletteRotate = 0;
    }
}

void Fractal::CreateNewFractalPalette(void)
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
        std::vector<uint16_t>{}.swap(m_PalR[Palette::Random][PaletteIndex]);
        std::vector<uint16_t>{}.swap(m_PalG[Palette::Random][PaletteIndex]);
        std::vector<uint16_t>{}.swap(m_PalB[Palette::Random][PaletteIndex]);

        const int m = 5;
        auto firstR = genNextColor(m);
        auto firstG = genNextColor(m);
        auto firstB = genNextColor(m);
        PalTransition(Palette::Random, PaletteIndex, depth_total, firstR, firstG, firstB);
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, genNextColor(m), genNextColor(m), genNextColor(m));
        PalTransition(Palette::Random, PaletteIndex, depth_total, 0, 0, 0);

        m_PalIters[Palette::Random][PaletteIndex] = (uint32_t)m_PalR[Palette::Random][PaletteIndex].size();
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 0, 5));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 1, 6));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 2, 8));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 3, 12));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 4, 16));
    threads.push_back(std::make_unique<std::thread>(RandomPaletteGen, 5, 20));

    for (auto& it : threads) {
        it->join();
    }
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

    const bool LocalColor = RequiresUseLocalColor();

    if (LocalColor) {
        for (auto& it : m_DrawThreadAtomics) {
            it.store(0);
        }

        for (auto& thread : m_DrawThreads) {
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
    }
    else {
        uint32_t result;
        if (GetIterType() == IterTypeEnum::Bits32) {
            result = m_r.OnlyAA(
                m_CurIters.m_RoundedOutputColorMemory.get(),
                GetNumIterations<uint32_t>());
        }
        else {
            result = m_r.OnlyAA(
                m_CurIters.m_RoundedOutputColorMemory.get(),
                GetNumIterations<uint64_t>());
        }

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
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);  //Always set the base and max mipmap levels of a texture.
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

    // Change m_DrawOutBytes size if GL_RGBA is changed
    if (LocalColor) {
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA16, (GLsizei)m_ScrnWidth, (GLsizei)m_ScrnHeight, 0,
            GL_RGBA, GL_UNSIGNED_SHORT, m_DrawOutBytes.get());
    }
    else {
        //glTexImage2D(
        //    GL_TEXTURE_2D, 0, GL_RGBA16,
        //    (GLsizei)m_CurIters.m_RoundedOutputColorWidth,
        //    (GLsizei)m_CurIters.m_RoundedOutputColorHeight,
        //    0, GL_RGBA, GL_UNSIGNED_SHORT, m_CurIters.m_RoundedOutputColorMemory.get());

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA16,
            (GLsizei)m_CurIters.m_OutputWidth,
            (GLsizei)m_CurIters.m_OutputHeight,
            0, GL_RGBA, GL_UNSIGNED_SHORT, m_CurIters.m_RoundedOutputColorMemory.get());
    }

    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, (GLint)m_ScrnHeight);
    glTexCoord2i(0, 1); glVertex2i(0, 0);
    glTexCoord2i(1, 1); glVertex2i((GLint)m_ScrnWidth, 0);
    glTexCoord2i(1, 0); glVertex2i((GLint)m_ScrnWidth, (GLint)m_ScrnHeight);
    glEnd();
    glFlush();

    glDeleteTextures(1, &texid);

    DrawAllPerturbationResults();

    //while (GetMessage(&msg, NULL, 0, 0) > 0)
    //{
    //    TranslateMessage(&msg);
    //    DispatchMessage(&msg);
    //}
}

void Fractal::DrawAllPerturbationResults() {
    DrawPerturbationResults<uint32_t, double, CalcBad::Disable>(true);
    DrawPerturbationResults<uint32_t, float, CalcBad::Disable>(true);
    DrawPerturbationResults<uint32_t, HDRFloat<double>, CalcBad::Disable>(true);
    DrawPerturbationResults<uint32_t, HDRFloat<float>, CalcBad::Disable>(true);

    DrawPerturbationResults<uint32_t, double, CalcBad::Enable>(true);
    DrawPerturbationResults<uint32_t, float, CalcBad::Enable>(true);
    DrawPerturbationResults<uint32_t, HDRFloat<double>, CalcBad::Enable>(true);
    DrawPerturbationResults<uint32_t, HDRFloat<float>, CalcBad::Enable>(true);

    DrawPerturbationResults<uint64_t, double, CalcBad::Disable>(true);
    DrawPerturbationResults<uint64_t, float, CalcBad::Disable>(true);
    DrawPerturbationResults<uint64_t, HDRFloat<double>, CalcBad::Disable>(true);
    DrawPerturbationResults<uint64_t, HDRFloat<float>, CalcBad::Disable>(true);

    DrawPerturbationResults<uint64_t, double, CalcBad::Enable>(true);
    DrawPerturbationResults<uint64_t, float, CalcBad::Enable>(true);
    DrawPerturbationResults<uint64_t, HDRFloat<double>, CalcBad::Enable>(true);
    DrawPerturbationResults<uint64_t, HDRFloat<float>, CalcBad::Enable>(true);

}

template<typename IterType, class T, CalcBad Bad>
void Fractal::DrawPerturbationResults(bool LeaveScreen) {
    if (!LeaveScreen) {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBegin(GL_POINTS);

    // TODO can we just integrate all this with DrawFractal

    auto& results = m_RefOrbit.GetPerturbationResults<IterType, T, Bad>();
    for (size_t i = 0; i < results.size(); i++)
    {
        if (m_RefOrbit.IsPerturbationResultUsefulHere<IterType, T, false, Bad>(i)) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            GLint scrnX = Convert<HighPrecision, GLint>(XFromCalcToScreen(results[i]->hiX));
            GLint scrnY = static_cast<GLint>(m_ScrnHeight) - Convert<HighPrecision, GLint>(YFromCalcToScreen(results[i]->hiY));

            // Coordinates are weird in OGL mode.
            glVertex2i(scrnX, scrnY);
        }
    }

    glEnd();
    glFlush();
}

void Fractal::DrawFractalThread(size_t index, Fractal* fractal) {
    DrawThreadSync &sync = *fractal->m_DrawThreads[index].get();

    constexpr size_t BytesPerPixel = 4;
    IterTypeFull maxPossibleIters;

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

        maxPossibleIters = fractal->GetMaxIterationsRT();

        sync.m_DrawThreadReady = false;

        auto lambda = [&](auto** ItersArray, auto NumIterations) {
            size_t acc_r, acc_g, acc_b;
            size_t outputIndex = 0;

            const size_t totalAA = fractal->GetGpuAntialiasing() * fractal->GetGpuAntialiasing();
            uint32_t palIters = fractal->m_PalIters[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex];
            const uint16_t* palR = fractal->m_PalR[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            const uint16_t* palG = fractal->m_PalG[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            const uint16_t* palB = fractal->m_PalB[fractal->m_WhichPalette][fractal->m_PaletteDepthIndex].data();
            size_t basicFactor = 65536 / NumIterations;
            if (basicFactor == 0) {
                basicFactor = 1;
            }

            const auto GetBasicColor = [&](
                size_t numIters,
                size_t& acc_r,
                size_t& acc_g,
                size_t& acc_b) {

                if (fractal->m_WhichPalette != Palette::Basic) {
                    auto palIndex = (numIters >> fractal->m_PaletteAuxDepth) % palIters;
                    acc_r += palR[palIndex];
                    acc_g += palG[palIndex];
                    acc_b += palB[palIndex];
                }
                else {
                    acc_r += ((numIters >> fractal->m_PaletteAuxDepth) * basicFactor) & ((1llu << 16) - 1);
                    acc_g += ((numIters >> fractal->m_PaletteAuxDepth) * basicFactor) & ((1llu << 16) - 1);
                    acc_b += ((numIters >> fractal->m_PaletteAuxDepth) * basicFactor) & ((1llu << 16) - 1);
                }
            };

            const size_t maxIters = NumIterations;
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
                        size_t numIters = ItersArray[input_y][input_x];

                        if (numIters < maxIters)
                        {
                            numIters += fractal->m_PaletteRotate;
                            if (numIters >= maxPossibleIters) {
                                numIters = maxPossibleIters - 1;
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

                                size_t numIters = ItersArray[input_y][input_x];
                                if (numIters < maxIters)
                                {
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
        }
        else {
            lambda(fractal->m_CurIters.GetItersArray<uint64_t>(), fractal->GetNumIterations<uint64_t>());
        }

        sync.m_DrawThreadProcessed = true;
        lk.unlock();
        sync.m_DrawThreadCV.notify_one();
    }
}

void Fractal::FillCoord(HighPrecision& src, MattQFltflt& dest) {
    // qflt
    dest.x = Convert<HighPrecision, float>(src);
    dest.y = Convert<HighPrecision, float>(src - HighPrecision{ dest.x });
    dest.z = Convert<HighPrecision, float>(src - HighPrecision{ dest.x } - HighPrecision{ dest.y });
    dest.w = Convert<HighPrecision, float>(src - HighPrecision{ dest.x } - HighPrecision{ dest.y } - HighPrecision{ dest.z });
}

void Fractal::FillCoord(HighPrecision& src, MattQDbldbl& dest) {
    // qdbl
    dest.x = Convert<HighPrecision, double>(src);
    dest.y = Convert<HighPrecision, double>(src - HighPrecision{ dest.x });
    dest.z = Convert<HighPrecision, double>(src - HighPrecision{ dest.x } - HighPrecision{ dest.y });
    dest.w = Convert<HighPrecision, double>(src - HighPrecision{ dest.x } - HighPrecision{ dest.y } - HighPrecision{ dest.z });
}

void Fractal::FillCoord(HighPrecision& src, MattDbldbl& dest) {
    // dbl
    dest.head = Convert<HighPrecision, double>(src);
    dest.tail = Convert<HighPrecision, double>(src - HighPrecision{ dest.head });
}

void Fractal::FillCoord(HighPrecision& src, double& dest) {
    // doubleOnly
    dest = Convert<HighPrecision, double>(src);
}

void Fractal::FillCoord(HighPrecision& src, HDRFloat<float>& dest) {
    // hdrflt
    dest = (HDRFloat<float>)src;
    HdrReduce(dest);
}

void Fractal::FillCoord(HighPrecision& src, HDRFloat<double>& dest) {
    //  hdrdbl
    dest = (HDRFloat<double>)src;
    HdrReduce(dest);
}

void Fractal::FillCoord(HighPrecision& src, MattDblflt& dest) {
    // flt
    dest.y = Convert<HighPrecision, float>(src);
    dest.x = Convert<HighPrecision, float>(src - HighPrecision{ dest.y });
}

void Fractal::FillCoord(HighPrecision& src, float& dest) {
    // floatOnly
    dest = Convert<HighPrecision, float>(src);
}

void Fractal::FillCoord(HighPrecision& src, CudaDblflt<MattDblflt>& dest) {
    double destDbl = Convert<HighPrecision, double>(src);
    dest = CudaDblflt(destDbl);
}

void Fractal::FillCoord(HighPrecision& src, HDRFloat<CudaDblflt<MattDblflt>>& dest) {
    HDRFloat<CudaDblflt<MattDblflt>> destDbl(src);
    dest = destDbl;
}


template<class T>
void Fractal::FillGpuCoords(T &cx2, T &cy2, T &dx2, T &dy2) {
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

template<typename IterType, class T>
void Fractal::CalcGpuFractal(bool MemoryOnly)
{
    OutputMessage(L"\r\n");

    T cx2{}, cy2{}, dx2{}, dy2{};
    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    err = m_r.Render(GetRenderAlgorithm(),
                     m_CurIters.GetIters<IterType>(),
                     m_CurIters.m_RoundedOutputColorMemory.get(),
                     cx2,
                     cy2,
                     dx2,
                     dy2,
                     GetNumIterations<IterType>(),
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
void Fractal::CalcNetworkFractal(bool /*MemoryOnly*/)
{
    // TODO broken
    //size_t nx, ny;
    //IterType numItersL;
    //WORD numItersS;
    //IterType numIters = 0;

    //int i;

    //for (i = 0; i < MAXSERVERS; i++)
    //{
    //    if (m_SetupData.m_UseThisServer[i] == 'n')
    //    {
    //        continue;
    //    }

    //    m_ClientSubNetwork[i]->BufferedReadEmpty();

    //    for (ny = 0; ny < m_ScrnHeight; ny++)
    //    {
    //        if (m_ProcessPixelRow[ny] != i + 'b')
    //        {
    //            continue;
    //        }

    //        for (nx = 0; nx < m_ScrnWidth; nx++)
    //        {
    //            if (m_NumIterations >= 65536)
    //            {
    //                m_ClientSubNetwork[i]->BufferedReadLong(&numItersL);
    //                numIters = numItersL;
    //            }
    //            else
    //            {
    //                m_ClientSubNetwork[i]->BufferedReadShort(&numItersS);
    //                numIters = numItersS;
    //            }

    //            m_CurIters.m_ItersArray[ny][nx] = numIters;
    //        }

    //        if (MemoryOnly == false)
    //        {
    //            assert(false);
    //            // TODO broken
    //            //DrawFractalLine(ny);
    //        }
    //    }
    //}
}

template<typename IterType>
void Fractal::CalcCpuPerturbationFractal(bool MemoryOnly) {
    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        double,
        double,
        CalcBad::Disable,
        RefOrbitCalc::Extras::None>();

    double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / (m_ScrnWidth * GetGpuAntialiasing()));
    double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / (m_ScrnHeight * GetGpuAntialiasing()));

    double centerX = (double)(results->hiX - m_MinX);
    double centerY = (double)(results->hiY - m_MaxY);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    //
    // From https://fractalforums.org/fractal-mathematics-and-new-theories/28/another-solution-to-perturbation-glitches/4360/msg29835#msg29835
    //

    //complex Reference[]; // Reference orbit (MUST START WITH ZERO)
    //IterType MaxRefIteration; // The last valid iteration of the reference (the iteration just before it escapes)
    //complex dz = 0, dc; // Delta z and Delta c
    //IterType Iteration = 0, RefIteration = 0;

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
        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++)
            {
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

                    DeltaSubNX = DeltaSubNXOrig * (results->orb[RefIteration].x * 2 + DeltaSubNXOrig) -
                        DeltaSubNYOrig * (results->orb[RefIteration].y * 2 + DeltaSubNYOrig) +
                        DeltaSub0X;
                    DeltaSubNY = DeltaSubNXOrig * (results->orb[RefIteration].y * 2 + DeltaSubNYOrig) +
                        DeltaSubNYOrig * (results->orb[RefIteration].x * 2 + DeltaSubNXOrig) +
                        DeltaSub0Y;

                    ++RefIteration;

                    const double tempZX = results->orb[RefIteration].x + DeltaSubNX;
                    const double tempZY = results->orb[RefIteration].y + DeltaSubNY;
                    const double zn_size = tempZX * tempZX + tempZY * tempZY;
                    const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                    if (zn_size > 256) {
                        break;
                    }

                    if (zn_size < normDeltaSubN ||
                        RefIteration == (IterType)results->orb.size() - 1) {
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
template<typename IterType, class T, class SubType>
void Fractal::CalcCpuHDR(bool MemoryOnly) {
    const T dx = T((m_MaxX - m_MinX) / (m_ScrnWidth * GetGpuAntialiasing()));
    const T dy = T((m_MaxY - m_MinY) / (m_ScrnHeight * GetGpuAntialiasing()));

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    const T Four{ 4 };
    const T Two{ 2 };

    auto one_thread = [&]() {
        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
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

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++)
            {
                // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
                // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
                zx = cx;
                zy = cy;
                for (i = 0; i < GetNumIterations<IterType>(); i++)
                { // x^2+2*I*x*y-y^2
                    zx2 = zx * zx;
                    zy2 = zy * zy;
                    sum = zx2 + zy2;
                    HdrReduce(sum);
                    if (HdrCompareToBothPositiveReducedGT(sum, Four)) break;

                    zy = Two * zx * zy;
                    zx = zx2 - zy2;

                    zx += cx;
                    zy += cy;

                    HdrReduce(zx);
                    HdrReduce(zy);
                }
                cx += dx;
                //HdrReduce(cx);

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

template<typename IterType, class T, class SubType>
void Fractal::CalcCpuPerturbationFractalBLA(bool MemoryOnly) {
    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        T,
        SubType,
        CalcBad::Disable,
        RefOrbitCalc::Extras::None>();

    BLAS<IterType, T> blas(*results);
    blas.Init((IterType)results->orb.size(), results->maxRadius);

    T dx = T((m_MaxX - m_MinX) / (m_ScrnWidth * GetGpuAntialiasing()));
    T dy = T((m_MaxY - m_MinY) / (m_ScrnHeight * GetGpuAntialiasing()));

    T centerX = (T)(results->hiX - m_MinX);
    HdrReduce(centerX);
    T centerY = (T)(results->hiY - m_MaxY);
    HdrReduce(centerY);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    auto one_thread = [&]() {
        //T dzdcX = T(1);
        //T dzdcY = T(0);
        //bool periodicity_should_break = false;

        for (size_t y = 0; y < m_ScrnHeight * GetGpuAntialiasing(); y++) {
            if (atomics[y] != 0) {
                continue;
            }

            uint64_t expected = 0;
            if (atomics[y].compare_exchange_strong(expected, 1llu) == false) {
                continue;
            }

            for (size_t x = 0; x < m_ScrnWidth * GetGpuAntialiasing(); x++)
            {
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
                        if (RefIteration + l >= (IterType)results->orb.size()) {
                            //::MessageBox(NULL, L"Out of bounds! :(", L"", MB_OK);
                            break;
                        }

                        if (iter + l >= GetNumIterations<IterType>()) {
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

                        auto tempZX = results->orb[RefIteration].x + DeltaSubNX;
                        auto tempZY = results->orb[RefIteration].y + DeltaSubNY;
                        auto normSquared = tempZX * tempZX + tempZY * tempZY;
                        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;
                        HdrReduce(normSquared);
                        HdrReduce(DeltaNormSquared);

                        if (HdrCompareToBothPositiveReducedGT(normSquared, T(256))) {
                            break;
                        }

                        if (HdrCompareToBothPositiveReducedLT(normSquared, DeltaNormSquared) ||
                            RefIteration >= (IterType)results->orb.size() - 1) {
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

                    T TermB1 = DeltaSubNXOrig * (results->orb[RefIteration].x * 2 + DeltaSubNXOrig);
                    T TermB2 = DeltaSubNYOrig * (results->orb[RefIteration].y * 2 + DeltaSubNYOrig);

                    DeltaSubNX = TermB1 - TermB2;
                    DeltaSubNX += DeltaSub0X;
                    HdrReduce(DeltaSubNX);

                    T Term3 = results->orb[RefIteration].y * 2 + DeltaSubNYOrig;
                    T Term4 = results->orb[RefIteration].x * 2 + DeltaSubNXOrig;
                    DeltaSubNY = DeltaSubNXOrig * Term3 + DeltaSubNYOrig * Term4;
                    DeltaSubNY += DeltaSub0Y;
                    HdrReduce(DeltaSubNY);

                    ++RefIteration;
                    if (RefIteration >= (IterType)results->orb.size()) {
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

                        //T n2 = std::max(zxCopy1, zyCopy1);

                        //T r0 = std::max(dzdcX1, dzdcY1);
                        //T maxRadiusHdr{ 3840 };
                        //auto n3 = maxRadiusHdr * r0 * HighTwo;
                        //HdrReduce(n3);

                        //if (n2 < n3) {
                        //    periodicity_should_break = true;
                        //}
                        //else {
                        //    auto dzdcXOrig = dzdcX;
                        //    dzdcX = HighTwo * (zxCopy * dzdcX - zyCopy * dzdcY) + HighOne;
                        //    dzdcY = HighTwo * (zxCopy * dzdcY + zyCopy * dzdcXOrig);
                        //}
                    }

                    T tempZX = results->orb[RefIteration].x + DeltaSubNX;
                    T tempZY = results->orb[RefIteration].y + DeltaSubNY;
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
                        RefIteration >= (IterType)results->orb.size() - 1) {
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

template<typename IterType, class SubType>
void Fractal::CalcCpuPerturbationFractalLAV2(bool MemoryOnly) {
    using T = HDRFloat<SubType>;
    using TComplex = HDRFloatComplex<SubType>;
    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        T,
        SubType,
        CalcBad::Disable,
        RefOrbitCalc::Extras::IncludeLAv2>();

    if (results->GetLaReference() == nullptr) {
        ::MessageBox(NULL, L"Oops - a null pointer deref", L"", MB_OK);
        return;
    }

    auto &LaReference = *results->GetLaReference();

    T dx = T((m_MaxX - m_MinX) / (m_ScrnWidth * GetGpuAntialiasing()));
    HdrReduce(dx);
    T dy = T((m_MaxY - m_MinY) / (m_ScrnHeight * GetGpuAntialiasing()));
    HdrReduce(dy);

    T centerX = (T)(results->hiX - m_MinX);
    HdrReduce(centerX);
    T centerY = (T)(results->hiY - m_MaxY);
    HdrReduce(centerY);

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight * GetGpuAntialiasing());
    threads.reserve(num_threads);

    auto one_thread = [&]() {
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

                DeltaSub0 = { deltaReal, deltaImaginary };
                DeltaSubN = { 0, 0 };

                if (LaReference.isValid && LaReference.UseAT && LaReference.AT.isValid(DeltaSub0)) {
                    ATResult<IterType, T, SubType> res;
                    LaReference.AT.PerformAT(GetNumIterations<IterType>(), DeltaSub0, res);
                    BLA2SkippedIterations = res.bla_iterations;
                    DeltaSubN = res.dz;
                }

                IterType iterations = 0;
                IterType RefIteration = 0;
                IterType MaxRefIteration = (IterType)results->orb.size() - 1;

                iterations = BLA2SkippedIterations;

                TComplex complex0{ deltaReal, deltaImaginary };

                if (iterations != 0 && RefIteration < MaxRefIteration) {
                    complex0 = results->GetComplex<SubType>(RefIteration) + DeltaSubN;
                } else if (iterations != 0 && results->PeriodMaybeZero != 0) {
                    RefIteration = RefIteration % results->PeriodMaybeZero;
                    complex0 = results->GetComplex<SubType>(RefIteration) + DeltaSubN;
                }

                auto CurrentLAStage = LaReference.isValid ? LaReference.LAStageCount : 0;

                while (CurrentLAStage > 0) {
                    CurrentLAStage--;

                    auto LAIndex = LaReference.getLAIndex(CurrentLAStage);

                    if (LaReference.isLAStageInvalid(LAIndex, DeltaSub0)) {
                        continue;
                    }

                    auto MacroItCount = LaReference.getMacroItCount(CurrentLAStage);
                    auto j = RefIteration;

                    while (iterations < GetNumIterations<IterType>()) {
                        auto las = LaReference.getLA(
                            LAIndex,
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
                    auto curIter = results->GetComplex<SubType>(RefIteration);
                    curIter = curIter * T(2);
                    curIter = curIter + DeltaSubN;
                    DeltaSubN = DeltaSubN * curIter;
                    DeltaSubN = DeltaSubN + DeltaSub0;
                    HdrReduce(DeltaSubN);

                    RefIteration++;

                    complex0 = results->GetComplex<SubType>(RefIteration) + DeltaSubN;
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

template<typename IterType, class T, class SubType, bool BLA>
void Fractal::CalcGpuPerturbationFractalBLA(bool MemoryOnly) {
    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        T,
        SubType,
        CalcBad::Disable,
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

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    MattPerturbResults<IterType, T> gpu_results{
        (IterType)results->orb.size(),
        results->orb.data(),
        results->PeriodMaybeZero };

    uint32_t result;
    if constexpr (BLA) {
        BLAS<IterType, T> blas(*results);
        blas.Init(results->orb.size(), results->maxRadius);

        result = m_r.RenderPerturbBLA<IterType, T>(GetRenderAlgorithm(),
            m_CurIters.GetIters<IterType>(),
            m_CurIters.m_RoundedOutputColorMemory.get(),
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
    }
    else {
        result = m_r.RenderPerturb(GetRenderAlgorithm(),
            m_CurIters.GetIters<IterType>(),
            m_CurIters.m_RoundedOutputColorMemory.get(),
            &gpu_results,
            cx2,
            cy2,
            dx2,
            dy2,
            centerX2,
            centerY2,
            GetNumIterations<IterType>(),
            m_IterationPrecision);
    }

    DrawFractal(MemoryOnly);

    if (result) {
        MessageBoxCudaError(result);
    }
}

template<typename IterType, class T, class SubType, LAv2Mode Mode>
void Fractal::CalcGpuPerturbationFractalLAv2(bool MemoryOnly) {
    using ConditionalT = typename DoubleTo2x32Converter<T, SubType>::ConditionalT;
    using ConditionalSubType = typename DoubleTo2x32Converter<T, SubType>::ConditionalSubType;

    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        ConditionalT,
        ConditionalSubType,
        CalcBad::Disable,
        RefOrbitCalc::Extras::IncludeLAv2,
        T>();

    // TODO pass perturb results via InitializeGPUMemory
    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    MattPerturbResults<IterType, T> gpu_results{
        (IterType)results->orb.size(),
        results->orb.data(),
        results->PeriodMaybeZero };

    if (results->GetLaReference() == nullptr) {
        ::MessageBox(NULL, L"Oops - a null pointer deref", L"", MB_OK);
        return;
    }

    m_r.InitializePerturb<IterType, T, SubType, CalcBad::Disable, T>(
        results->GetGenerationNumber(),
        &gpu_results,
        0,
        nullptr,
        results->GetLaReference(),
        results->GetLaGenerationNumber());
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    auto result = m_r.RenderPerturbLAv2<IterType, T, SubType, Mode>(GetRenderAlgorithm(),
        m_CurIters.GetIters<IterType>(),
        m_CurIters.m_RoundedOutputColorMemory.get(),
        cx2,
        cy2,
        dx2,
        dy2,
        centerX2,
        centerY2,
        GetNumIterations<IterType>());

    DrawFractal(MemoryOnly);

    if (result) {
        MessageBoxCudaError(result);
    }
}

template<typename IterType, class T, class SubType, class T2, class SubType2>
void Fractal::CalcGpuPerturbationFractalScaledBLA(bool MemoryOnly) {
    auto* results = m_RefOrbit.GetAndCreateUsefulPerturbationResults<
        IterType,
        T,
        SubType,
        CalcBad::Enable,
        RefOrbitCalc::Extras::None>();
    auto* results2 = m_RefOrbit.CopyUsefulPerturbationResults<
        IterType,
        T,
        CalcBad::Enable,
        T2,
        CalcBad::Enable>(*results);

    BLAS<IterType, T, CalcBad::Enable> blas(*results);
    blas.Init(results->orb.size(), results->maxRadius);

    uint32_t err = InitializeGPUMemory();
    if (err) {
        MessageBoxCudaError(err);
        return;
    }

    m_r.ClearMemory<IterType>();

    T cx2{}, cy2{}, dx2{}, dy2{};
    T centerX2{}, centerY2{};

    FillGpuCoords<T>(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    MattPerturbResults<IterType, T, CalcBad::Enable> gpu_results{
        (IterType)results->orb.size(),
        results->orb.data(),
        results->PeriodMaybeZero };

    MattPerturbResults<IterType, T2, CalcBad::Enable> gpu_results2{
        (IterType)results2->orb.size(),
        results2->orb.data(),
        results2->PeriodMaybeZero };

    if (gpu_results.size != gpu_results2.size) {
        ::MessageBox(NULL, L"Mismatch on size", L"", MB_OK);
        return;
    }

    auto result = m_r.RenderPerturbBLAScaled<IterType, T>(GetRenderAlgorithm(),
        m_CurIters.GetIters<IterType>(),
        m_CurIters.m_RoundedOutputColorMemory.get(),
        &gpu_results,
        &gpu_results2,
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

void Fractal::MessageBoxCudaError(uint32_t result) {
    char error[256];
    sprintf(error, "Error from cuda: %u.  Message: \"%s\"\n", result, GPURenderer::ConvertErrorToString(result));
    ::MessageBoxA(NULL, error, "", MB_OK);
}

//////////////////////////////////////////////////////////////////////////////
// Saves the current fractal as a bitmap to the given file.
// If halfImage is true, a bitmap with half the dimensions of the current
// fractal is saved instead.  Thus, 1024x768 is resized to 512x384.
//////////////////////////////////////////////////////////////////////////////

Fractal::CurrentFractalSave::CurrentFractalSave(
    enum Type typ,
    std::wstring filename_base,
    Fractal& fractal)
    : m_Type(typ),
    m_FilenameBase(filename_base),
    m_Fractal(fractal),
    m_ScrnWidth(fractal.m_ScrnWidth),
    m_ScrnHeight(fractal.m_ScrnHeight),
    m_GpuAntialiasing(fractal.m_GpuAntialiasing),
    m_NumIterations(fractal.m_NumIterations),
    m_PaletteRotate(fractal.m_PaletteRotate),
    m_PaletteDepthIndex(fractal.m_PaletteDepthIndex),
    m_PaletteAuxDepth(fractal.m_PaletteAuxDepth),
    m_WhichPalette(fractal.m_WhichPalette),
    m_CurIters(std::move(fractal.m_CurIters)) {

    //
    // TODO Note we pass off ownership of m_CurIters.
    // Implication is that if you save multiple copies of the same bit map, it's not
    // going to work sensibly.  This is a bug.
    //

    fractal.SetCurItersMemory();

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
    int ret;
    std::wstring final_filename;

    std::wstring ext;
    if (m_Type == Type::PngImg) {
        ext = L".png";
    }
    else {
        ext = L".txt";
    }

    if (m_FilenameBase != L"") {
        wchar_t temp[512];
        wsprintf(temp, L"%s", m_FilenameBase.c_str());
        final_filename = std::wstring(temp) + ext;
        if (Fractal::FileExists(final_filename.c_str())) {
            ::MessageBox(NULL, L"Not saving, file exists", L"", MB_OK);
            return;
        }
    }
    else {
        size_t i = 0;
        do {
            wchar_t temp[512];
            wsprintf(temp, L"output%05d", i);
            final_filename = std::wstring(temp) + ext;
            i++;
        } while (Fractal::FileExists(final_filename.c_str()));
    }

    // TODO racy bug, changing iteration type while save in progress.
    IterTypeFull maxPossibleIters = m_Fractal.GetMaxIterationsRT();

    //setup converter deprecated
    //using convert_type = std::codecvt_utf8<wchar_t>;
    //std::wstring_convert<convert_type, wchar_t> converter;
    ////use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    //const std::string filename_c = converter.to_bytes(final_filename);

    std::string filename_c;
    std::transform(final_filename.begin(), final_filename.end(), std::back_inserter(filename_c), [](wchar_t c) {
        return (char)c;
        });

    if (m_Type == Type::PngImg) {
        double acc_r, acc_b, acc_g;
        size_t input_x, input_y;
        size_t output_x, output_y;
        size_t numIters;

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

                        numIters = m_CurIters.GetItersArrayValSlow(input_x, input_y);
                        if (numIters < m_NumIterations)
                        {
                            numIters += m_PaletteRotate;
                            if (numIters >= maxPossibleIters) {
                                numIters = maxPossibleIters - 1;
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

                //if (index > GetMaxIterations<IterType>()) {
                //    index = GetMaxIterations<IterType>() - 1;
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

        ret = image.saveImage(filename_c, WPngImage::PngFileFormat::kPngFileFormat_RGBA16);
    }
    else {
        constexpr size_t buf_size = 128;
        char one_val[buf_size];
        std::string out_str;

        for (uint32_t output_y = 0; output_y < m_ScrnHeight * m_GpuAntialiasing; output_y++) {
            for (uint32_t output_x = 0; output_x < m_ScrnWidth * m_GpuAntialiasing; output_x++) {
                IterTypeFull numiters = m_CurIters.GetItersArrayValSlow(output_x, output_y);
                memset(one_val, ' ', sizeof(one_val));

                //static_assert(sizeof(IterType) == 8, "!");
                //char(*__kaboom1)[sizeof(IterType)] = 1;
                sprintf(one_val, "(%u,%u):%llu ", output_x, output_y, (IterTypeFull)numiters);

                // Wow what a kludge
                //size_t orig_len = strlen(one_val);
                //one_val[orig_len] = ' ';
                //one_val[orig_len + 1] = 0;

                out_str += one_val;
            }

            out_str += "\n";
        }

        std::ofstream out(filename_c);
        out << out_str;
        out.close();

        m_Fractal.ReturnIterMemory(std::move(m_CurIters));
    }

    m_Destructable = true;
    return;
}

int Fractal::SaveCurrentFractal(const std::wstring filename_base) {
    return SaveFractalData<CurrentFractalSave::Type::PngImg>(filename_base);
}

template<Fractal::CurrentFractalSave::Type Typ>
int Fractal::SaveFractalData(const std::wstring filename_base)
{
    auto lambda = [&]<typename T>(T &savesInProgress) {
        for (;;) {
            MEMORYSTATUSEX statex;
            statex.dwLength = sizeof(statex);
            GlobalMemoryStatusEx(&statex);

            if (savesInProgress.size() > 32 || statex.dwMemoryLoad > 90) {
                CleanupThreads(false);
                Sleep(100);
            }
            else {
                auto newPtr = std::make_unique<CurrentFractalSave>(Typ, filename_base, *this);
                savesInProgress.push_back(std::move(newPtr));
                savesInProgress.back()->StartThread();
                break;
            }
        }
    };

    lambda(m_FractalSavesInProgress);

    return 0;
}

void Fractal::CleanupThreads(bool all) {
    auto lambda = [&]<typename T>(T & savesInProgress) {
        bool continueCriteria = true;

        while (continueCriteria) {
            for (size_t i = 0; i < savesInProgress.size(); i++) {
                auto& it = savesInProgress[i];
                if (it->m_Destructable) {
                    savesInProgress.erase(savesInProgress.begin() + i);
                    break;
                }
            }

            if (all) {
                continueCriteria = !savesInProgress.empty();
            }
            else {
                break;
            }
        }
    };

    lambda(m_FractalSavesInProgress);
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

int Fractal::SaveItersAsText(const std::wstring filename_base) {
    return SaveFractalData<CurrentFractalSave::Type::ItersText>(filename_base);
}

HighPrecision Fractal::Benchmark(IterTypeFull numIters, size_t& milliseconds)
{
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (m_RefOrbit.RequiresReferencePoints()) {
        m_RefOrbit.AddPerturbationReferencePoint<uint32_t, double, double, RefOrbitCalc::BenchmarkMode::Disable>();
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
HighPrecision Fractal::BenchmarkReferencePoint(IterTypeFull numIters, size_t& milliseconds) {
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (!bm.StartTimer()) {
        return {};
    }

    m_RefOrbit.AddPerturbationReferencePoint<uint32_t, T, SubType, RefOrbitCalc::BenchmarkMode::Enable>();

    auto result = bm.StopTimerNoIters<T>(milliseconds);
    bm.BenchmarkFinish();
    return result;
}

template HighPrecision Fractal::BenchmarkReferencePoint<float, float>(IterTypeFull numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<double, double>(IterTypeFull numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<HDRFloat<double>, double>(IterTypeFull numIters, size_t& milliseconds);
template HighPrecision Fractal::BenchmarkReferencePoint<HDRFloat<float>, float>(IterTypeFull numIters, size_t& milliseconds);

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

void Fractal::BenchmarkData::BenchmarkSetup(IterTypeFull numIters) {
    prevScrnWidth = fractal.m_ScrnWidth;
    prevScrnHeight = fractal.m_ScrnHeight;

    fractal.m_RefOrbit.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    fractal.SetNumIterations<IterTypeFull>(numIters);
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
    uint64_t totalIters = fractal.GetNumIterations<IterTypeFull>();

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
uint64_t Fractal::FindTotalItersUsed(void)
{
    uint64_t numIters = 0;
    int x, y;
    for (y = 0; y < m_ScrnHeight; y++)
    {
        for (x = 0; x < m_ScrnWidth; x++)
        {
            numIters += m_CurIters.GetItersArrayValSlow(x,y);
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
    //OutputMessage(L"Awaiting instructions...\r\n");
    //char buffer[512];
    //m_ServerSubNetwork->ReadData(buffer, 512);

    //// TODO:
    //RenderAlgorithm RenderAlgorithm = RenderAlgorithm::Cpu64;
    //int ScreenWidth = 0, ScreenHeight = 0;
    //IterType NumIters = 0;
    //HighPrecision MinX = 0, MaxX = 0, MinY = 0, MaxY = 0;

    //OutputMessage(L"Data received:\r\n");
    //OutputMessage(L"%s\r\n", buffer);

    //// Secondary connection should quit.
    //if (strcmp(buffer, "exit") == 0)
    //{
    //    return false;
    //}

    //// Anything else must be data for setting up a calculation.
    //// TODO support high precision
    ////sscanf(buffer, "%c %d %d %d %lf %lf %lf %lf",
    ////    &RenderAlgorithm, &NumIters, &ScreenWidth, &ScreenHeight,
    ////    &MinX, &MaxX, &MinY, &MaxY);

    //OutputMessage(L"Received instructions.\r\n");
    //OutputMessage(L"Interpretation:\r\n");
    //OutputMessage(L"Render Mode:  %d\r\n", RenderAlgorithm);
    //OutputMessage(L"NumIters:     %d\r\n", NumIters);
    //OutputMessage(L"ScreenWidth:  %d\r\n", ScreenWidth);
    //OutputMessage(L"ScreenHeight: %d\r\n", ScreenHeight);
    //OutputMessage(L"MinX:         %.15f\r\n", Convert<HighPrecision, double>(MinX));
    //OutputMessage(L"MaxX:         %.15f\r\n", Convert<HighPrecision, double>(MaxX));
    //OutputMessage(L"MinY:         %.15f\r\n", Convert<HighPrecision, double>(MinY));
    //OutputMessage(L"MaxY:         %.15f\r\n", Convert<HighPrecision, double>(MaxY));

    //ResetDimensions(ScreenWidth, ScreenHeight);
    //RecenterViewCalc(MinX, MinY, MaxX, MaxY);
    //SetNumIterations(NumIters);
    //SetRenderAlgorithm(RenderAlgorithm);
    //CalcFractal(true);

    //int x, y;
    //for (y = 0; y < ScreenHeight; y++)
    //{
    //    if (m_ProcessPixelRow[y] != m_NetworkRender)
    //    {
    //        continue;
    //    }

    //    for (x = 0; x < ScreenWidth; x++)
    //    {
    //        if (NumIters >= 65536)
    //        {
    //            m_ServerSubNetwork->BufferedSendLong(m_CurIters.m_ItersArray[y][x]);
    //        }
    //        else
    //        {
    //            m_ServerSubNetwork->BufferedSendShort((unsigned short)m_CurIters.m_ItersArray[y][x]);
    //        }
    //    }
    //}

    //m_ServerSubNetwork->BufferedSendFlush();

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