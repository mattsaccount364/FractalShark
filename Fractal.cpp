// Search for TODO
// Make the screen saver render high-res versions of screens that have
//      been saved to a queue.  High res images made at idle time
// Make this code run in a separate thread so it doesn't interfere with the windows
//      message pump.  Make it use callback functions so whoever is using this code
//      can see the progress, be notified when it is done, whatever.

#include "stdafx.h"
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <io.h>
#include <time.h>

#include <deque>
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

void DefaultOutputMessage(const wchar_t *, ...);

Fractal::ItersMemoryContainer::ItersMemoryContainer(size_t width, size_t height, size_t total_antialiasing)
    : m_ItersMemory(nullptr),
      m_ItersArray(nullptr),
      m_Width((width + 32) * total_antialiasing),
      m_Height((height + 32) * total_antialiasing),
      m_Total(m_Width * m_Height) {
    m_ItersMemory = new uint32_t[m_Total];
    memset(m_ItersMemory, 0, m_Total * sizeof(uint32_t));

    m_ItersArray = new uint32_t*[m_Height];
    for (size_t i = 0; i < m_Height; i++) {
        m_ItersArray[i] = &m_ItersMemory[i * m_Width];
    }
};

Fractal::ItersMemoryContainer::ItersMemoryContainer(Fractal::ItersMemoryContainer&& other) {
    *this = std::move(other);
}

Fractal::ItersMemoryContainer &Fractal::ItersMemoryContainer::operator=(Fractal::ItersMemoryContainer&& other) {
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

    // Setup member variables with initial values:
    //SetRenderAlgorithm(RenderAlgorithm::Gpu1x32PerturbedScaled);
    SetRenderAlgorithm(RenderAlgorithm::Cpu64PerturbedBLA);
    SetIterationPrecision(1);

    ResetDimensions(width, height, 1, 1);
    View(8);

    m_ChangedWindow = true;
    m_ChangedScrn = true;
    m_ChangedIterations = true;

    srand((unsigned int) time(NULL));

    // Initialize the palette
    auto PaletteGen = [&](size_t PaletteIndex, size_t Depth) {
        m_PalR[PaletteIndex].resize(MAXITERS);
        m_PalG[PaletteIndex].resize(MAXITERS);
        m_PalB[PaletteIndex].resize(MAXITERS);
        int depth_total = (int) (1 << Depth);

        int index = 0;
        PalTransition(PaletteIndex, 0, MAXITERS, 0, 0, 0);
        for (;;)
        {
            int max_val = 65535;
            index = PalTransition(PaletteIndex, index, depth_total, max_val, 0, 0);       if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, max_val, max_val, 0); if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, 0, max_val, 0);       if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, 0, max_val, max_val); if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, 0, 0, max_val);       if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, max_val, 0, max_val); if (index == -1) break;
            index = PalTransition(PaletteIndex, index, depth_total, 0, 0, 0);             if (index == -1) break;
        }
    };

    std::vector<std::unique_ptr<std::thread>> threads;
    std::unique_ptr<std::thread> t1(new std::thread(PaletteGen, 0, 8));
    std::unique_ptr<std::thread> t2(new std::thread(PaletteGen, 1, 12));
    std::unique_ptr<std::thread> t3(new std::thread(PaletteGen, 2, 16));

    m_PaletteDepth = 8;
    m_PaletteDepthIndex = 0;
    m_PaletteRotate = 0;

    // Allocate the iterations array.
    int i;
    InitializeMemory();

    // Wait for all this shit to get done
    t1->join();
    t2->join();
    t3->join();

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
        const size_t total_aa = GetGpuAntialiasing() * GetIterationAntialiasing();
        ItersMemoryContainer container(m_ScrnWidth,
                                       m_ScrnHeight,
                                       total_aa);
        m_ItersMemoryStorage.push_back(std::move(container));
    }

    m_CurIters = std::move(m_ItersMemoryStorage.back());
    m_ItersMemoryStorage.pop_back();
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
bool Fractal::PalIncrease(std::vector<uint16_t> &pal, int i1, int length, int val1, int val2)
{
    bool ret = true;
    if (pal.size() > INT32_MAX) {
        return false;
    }

    if (i1 + length >= pal.size())
    {
        length = (int)pal.size() - i1;
        ret = false;
    }

    double delta = (double)((double)(val2 - val1)) / length;
    for (int i = i1; i < i1 + length; i++)
    {
        double result = ((double)val1 + (double)delta * (i - i1 + 1));
        pal[i] = (unsigned short)result;
    }

    return ret;
}

// Transitions to the color specified.
// Allows for nice smooth palettes.
// length must be > 0
// Returns index immediately following the last index we filled here
// Returns -1 if we are at the end.
int Fractal::PalTransition(size_t PaletteIndex, int i1, int length, int r, int g, int b)
{
    int curR, curB, curG;
    if (i1 > 0)
    {
        curR = m_PalR[PaletteIndex][i1 - 1];
        curG = m_PalG[PaletteIndex][i1 - 1];
        curB = m_PalB[PaletteIndex][i1 - 1];
    }
    else
    {
        curR = 0;
        curG = 0;
        curB = 0;
    }

    // This code will fill out the palettes to the very end.
    bool shouldcontinue;
    shouldcontinue = PalIncrease(m_PalR[PaletteIndex], i1, length, curR, r);
    shouldcontinue &= PalIncrease(m_PalG[PaletteIndex], i1, length, curG, g);
    shouldcontinue &= PalIncrease(m_PalB[PaletteIndex], i1, length, curB, b);

    if (shouldcontinue)
    {
        return i1 + length;
    }

    return -1;
}

//////////////////////////////////////////////////////////////////////////////
// Resets the dimensions of the screen to width x height.
//////////////////////////////////////////////////////////////////////////////
void Fractal::ResetDimensions(size_t width,
                              size_t height,
                              uint32_t iteration_antialiasing,
                              uint32_t gpu_antialiasing)
{
    if (width == MAXSIZE_T) {
        width = m_ScrnWidth;
    }
    if (height == MAXSIZE_T) {
        height = m_ScrnHeight;
    }

    if (iteration_antialiasing == UINT32_MAX) {
        iteration_antialiasing = m_IterationAntialiasing;
    }
    
    if (gpu_antialiasing == UINT32_MAX) {
        gpu_antialiasing = m_GpuAntialiasing;
    }

    if (m_ScrnWidth != width ||
        m_ScrnHeight != height ||
        m_IterationAntialiasing != iteration_antialiasing ||
        m_GpuAntialiasing != gpu_antialiasing)
    {
        m_ScrnWidth = width;
        m_ScrnHeight = height;
        m_IterationAntialiasing = iteration_antialiasing;
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
bool Fractal::RecenterViewCalc(HighPrecision MinX, HighPrecision MinY, HighPrecision MaxX, HighPrecision MaxY)
{
    SaveCurPos();

    m_MinX = MinX;
    m_MaxX = MaxX;
    m_MinY = MinY;
    m_MaxY = MaxY;

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

    if (newMaxX < newMinX)
    {
        HighPrecision temp = newMinX;
        newMinX = newMaxX;
        newMaxX = temp;
    }

    if (newMaxY < newMinY)
    {
        HighPrecision temp = newMinY;
        newMinY = newMaxY;
        newMaxY = temp;
    }

    if (newMinX == newMaxX)
    {
        return false;
    }
    if (newMinY == newMaxY)
    {
        return false;
    }

    m_MinX = newMinX;
    m_MaxX = newMaxX;
    m_MinY = newMinY;
    m_MaxY = newMaxY;

    SquareCurrentView();

    m_ChangedWindow = true;
    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Recenters the view to the given point specified by x, y, which are screen
// cordinates.  These screen coordinates are translated to calculator coordinates
// for you.
//////////////////////////////////////////////////////////////////////////////
bool Fractal::CenterAtPoint(int x, int y)
{
    HighPrecision newCenterX = XFromScreenToCalc((HighPrecision)x);
    HighPrecision newCenterY = YFromScreenToCalc((HighPrecision)y);
    HighPrecision width = m_MaxX - m_MinX;
    HighPrecision height = m_MaxY - m_MinY;

    return RecenterViewCalc(newCenterX - width / 2, newCenterY - height / 2,
        newCenterX + width / 2, newCenterY + height / 2);
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

    m_PerturbationResults.clear();

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
        // Full BLA test
        minX = HighPrecision{ "-0.73198732629843320440204976927241262302865557668926323067907189191303572618456166820968874072402159997625420009666704345781677506621292458045712758673688395452668227635979898014441200747973251590103395489752085660025815039468080358849516199571780945485167631234003282215249532325026337114967584315527302929014070769647388129102508004668427421878414139639523504366866712" };
        minY = HighPrecision{ "0.168551219955066602450857684291218156300083497911317973583050597001478465218981183137439786526519040565783979013189488828294624083190238716864177970162720141151059941811388858157678439839291010444609585826087853669052343805389112843515477164832278357299322499467357831668268389147768565099109917523901197980150988627409178625493832131786965502660398647751643363880613" };
        maxX = HighPrecision{ "-0.73198711278411562860976923611757670326501681760613040979799153622365168352615277812643778690141397882000970796454699989818273281342296259355135081024529675472346701207658925643942943829071362022273606437127895303527746921408670822016734143274746985669135866602338077444815138791879406189673678065195261513943708472873576703784395775345318271550455800744078719690529401" };
        maxY = HighPrecision{ "0.16855130800451760694861870163628690390460609216071062689497565448933568323026627631972480735787769231227305079485074498667466459247298274714116671429828548167851387556091815660055162944189796322695068863491583698598561526811049739729599498355981609295200333367768758481392123970943429283930409545024424389959120250602521105099244814567974510474736220689025616952385081" };
        SetNumIterations(4718592);
        break;

    case 9:
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

size_t Fractal::GetNumIterations(void)
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
void Fractal::CalcDiskFractal(wchar_t * /*filename*/)
{
    // TODO busted, no support for antialiasing etc.
    return;

    //CBitmapWriter writer;
    //writer.beginWrite(filename, m_ScrnWidth, m_ScrnHeight);
    //uint32_t *data = new uint32_t[m_ScrnWidth + (4 - m_ScrnWidth % 4)];
    //int maxLines = 1000000 / (m_ScrnWidth * 3);
    //unsigned char *bitmapData = new unsigned char[maxLines * m_ScrnWidth * 3];

    //int numLinesInBuff = 0, i = 0;
    //for (int py = m_ScrnHeight - 1; py >= 0; py--)
    //{
    //    if (m_RenderAlgorithm == 'h')
    //    {
    //        CalcPixelRow_Multi((unsigned int *)data, py);
    //    }
    //    else if (m_RenderAlgorithm == 'l')
    //    {
    //        CalcPixelRow_C((unsigned int*)data, py);
    //    }

    //    int x;
    //    for (x = 0; x < m_ScrnWidth; x++)
    //    {
    //        bitmapData[i] = m_PalB[(data[x] == m_NumIterations) ? 0 : data[x]];
    //        i++;
    //        bitmapData[i] = m_PalG[(data[x] == m_NumIterations) ? 0 : data[x]];
    //        i++;
    //        bitmapData[i] = m_PalR[(data[x] == m_NumIterations) ? 0 : data[x]];
    //        i++;
    //    }

    //    numLinesInBuff++;
    //    if (numLinesInBuff == maxLines)
    //    {
    //        writer.writeLines(bitmapData, m_ScrnWidth, numLinesInBuff);
    //        i = 0;
    //        numLinesInBuff = 0;
    //    }

    //    if (py % 10 == 0)
    //    {
    //        OutputMessage(L"%.2f", (double)((double)m_ScrnHeight - (double)py) / (double)m_ScrnHeight * 100.0f);
    //    }
    //}

    //writer.writeLines(bitmapData, m_ScrnWidth, numLinesInBuff);
    //writer.endWrite();

    //delete[] data;
    //delete[] bitmapData;
}

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
    case RenderAlgorithm::Cpu64:
        CalcCpuFractal(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLA:
        CalcCpuPerturbationFractalBLA(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64:
    case RenderAlgorithm::Gpu2x64:
    case RenderAlgorithm::Gpu4x64:
    case RenderAlgorithm::Gpu1x32:
    case RenderAlgorithm::Gpu2x32:
    case RenderAlgorithm::Gpu4x32:
        CalcGpuFractal(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedBLA:
    case RenderAlgorithm::Gpu1x32PerturbedScaled:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
    case RenderAlgorithm::Gpu2x32PerturbedBLA:
    case RenderAlgorithm::Gpu2x32PerturbedScaled:
        CalcGpuPerturbationFractalBLA(MemoryOnly);
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

void Fractal::UsePalette(int depth)
{
    switch (depth) {
    case 8:
        m_PaletteDepthIndex = 0;
        m_PaletteDepth = depth;
        break;
    case 12:
        m_PaletteDepthIndex = 1;
        m_PaletteDepth = depth;
        break;
    case 16:
        m_PaletteDepthIndex = 2;
        m_PaletteDepth = depth;
        break;
    default:
        m_PaletteDepthIndex = 0;
        m_PaletteDepth = 8;
        break;
    }

    DrawFractal(false);
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

void Fractal::DrawPerturbationResults(bool LeaveScreen) {
    if (!LeaveScreen) {
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBegin(GL_POINTS);

    for (size_t i = 0; i < m_PerturbationResults.size(); i++)
    {
        if (IsPerturbationResultUsefulHere(i)) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            // Coordinates are weird in OGL mode.
            glVertex2i((GLint)m_PerturbationResults[i].scrnX, (GLint)m_PerturbationResults[i].scrnY);
        }
    }

    glEnd();
    glFlush();
}

void Fractal::ClearPerturbationResults() {
    m_PerturbationResults.clear();
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

    size_t py;

    //glClear(GL_COLOR_BUFFER_BIT);

    for (py = 0; py < m_ScrnHeight; py++) {
        DrawFractalLine(py);
    }

    DrawPerturbationResults(true);
}

void Fractal::DrawFractalLine(size_t output_y)
{
    uint32_t numIters;
    size_t input_x = 0;
    size_t input_y;
    size_t output_x;

    double acc_r, acc_g, acc_b;

    glBegin(GL_POINTS);

    for (output_x = 0;
         output_x < m_ScrnWidth;
         output_x++)
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

                    acc_r += m_PalR[m_PaletteDepthIndex][numIters] / 65536.0f;
                    acc_g += m_PalG[m_PaletteDepthIndex][numIters] / 65536.0f;
                    acc_b += m_PalB[m_PaletteDepthIndex][numIters] / 65536.0f;
                }
            }
        }

        acc_r /= m_GpuAntialiasing * m_GpuAntialiasing;
        acc_g /= m_GpuAntialiasing * m_GpuAntialiasing;
        acc_b /= m_GpuAntialiasing * m_GpuAntialiasing;

        glColor3f((GLfloat)acc_r, (GLfloat)acc_g, (GLfloat)acc_b);

        // Coordinates are weird in OGL mode.
        glVertex2i((GLint) output_x, (GLint) (m_ScrnHeight - output_y));
    }

    glEnd();
    glFlush();
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

    dest.flt.y = Convert<HighPrecision, float>(src);
    dest.flt.x = Convert<HighPrecision, float>(src - HighPrecision{ dest.flt.y });

    dest.floatOnly = Convert<HighPrecision, float>(src);
}

void Fractal::FillGpuCoords(MattCoords &cx2, MattCoords &cy2, MattCoords &dx2, MattCoords &dy2) {
    HighPrecision src_dy =
        (m_MaxY - m_MinY) /
        (HighPrecision)(m_ScrnHeight * m_IterationAntialiasing * m_GpuAntialiasing);

    HighPrecision src_dx =
        (m_MaxX - m_MinX) /
        (HighPrecision)(m_ScrnWidth * m_IterationAntialiasing * m_GpuAntialiasing);

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
        m_r.InitializeMemory(m_ScrnWidth * m_GpuAntialiasing,
                             m_ScrnHeight * m_GpuAntialiasing,
                             m_IterationAntialiasing,
                             m_CurIters.m_Width);
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
// This function recalculates the local component of the fractal.  That is,
// it draws only those parts of the fractal delegated to the client computer.
// It does not calculate the parts of the fractal delegated to the remote computer.
// If we are not using network rendering, this function calculates and draws the
// entire fractal.
//////////////////////////////////////////////////////////////////////////////
void Fractal::CalcCpuFractal(bool MemoryOnly)
{
    size_t py;
    //uint32_t prevMaxIters = FindMaxItersUsed ();

    int displayProgress = 0;

    OutputMessage(L"\r\n");

    std::vector<std::unique_ptr<std::thread>> threads;
    std::vector<size_t> threads_py;
    for (int i = 0; i < 10; i++)
    {
        py = i;
        while (py < m_ScrnHeight)
        { // Calculate only those pixels assigned to us
            //if (m_ProcessPixelRow[py] == m_NetworkRender)
            {
                if (m_RenderAlgorithm == RenderAlgorithm::CpuHigh)
                {
                    std::unique_ptr<std::thread> t1(new std::thread(&Fractal::CalcPixelRow_Multi, this, (unsigned int *)m_CurIters.m_ItersArray[py], py));
                    threads.push_back(std::move(t1));
                    threads_py.push_back(py);
                }
                else if (m_RenderAlgorithm == RenderAlgorithm::Cpu64)
                {
                    std::unique_ptr<std::thread> t1(new std::thread(&Fractal::CalcPixelRow_C, this, (unsigned int*)m_CurIters.m_ItersArray[py], py));
                    threads.push_back(std::move(t1));
                    threads_py.push_back(py);
                }

                // Display progress if we're doing this locally or as a client.
                //if (m_NetworkRender == 'a' || m_NetworkRender == '0')
                //{ if (MemoryOnly == false)
                //  { DrawFractalLine (py); }
                //}
            }
            py += 10;

            // todo check this status indicator
            //if (m_NetworkRender >= 'b')
            //{
            //    displayProgress++;
            //    if (displayProgress == 10)
            //    {
            //        OutputMessage(L"%03.0f\b\b\b", (double)i * 10.0 + (double)py / m_ScrnHeight * 10.0);
            //        displayProgress = 0;
            //    }
            //}

            //if (m_StopCalculating == true)
            //{
            //    if (m_NetworkRender == 'a')
            //    {
            //        ClientSendQuitMessages();
            //    }
            //    break;
            //}

            if (threads.size() > 128) {
                threads[0].get()->join();
                threads.erase(threads.begin());

                //DrawFractalLine(threads_py[0]);
                threads_py.erase(threads_py.begin());
            }
        }
    }

    for (size_t i = 0; i < threads.size(); i++) {
        threads[i].get()->join();
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
                DrawFractalLine(ny);
            }
        }
    }
}

void Fractal::CalcCpuPerturbationFractal(bool MemoryOnly) {
    PerturbationResults* results = GetUsefulPerturbationResults();

    BLAS blas(*results);
    blas.init(results->x.size(), Convert<HighPrecision, double>(results->maxRadius));

    double initX = (double)m_ScrnWidth / 2;
    double initY = (double)m_ScrnHeight / 2;

    double centerX, centerY;

    double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / m_ScrnWidth);
    double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / m_ScrnHeight);

    {
        HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
        HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

        HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)initX);
        HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)initY);

        centerX = (double)(cx - m_MinX);
        centerY = (double)(cy - m_MaxY);
    }

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

                    DeltaSubNX = DeltaSubNXOrig * (results->x2[RefIteration] + DeltaSubNXOrig) -
                        DeltaSubNYOrig * (results->y2[RefIteration] + DeltaSubNYOrig) +
                        DeltaSub0X;
                    DeltaSubNY = DeltaSubNXOrig * (results->y2[RefIteration] + DeltaSubNYOrig) +
                        DeltaSubNYOrig * (results->x2[RefIteration] + DeltaSubNXOrig) +
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

void Fractal::CalcCpuPerturbationFractalBLA(bool MemoryOnly) {
    PerturbationResults* results = GetUsefulPerturbationResults();

    BLAS blas(*results);
    blas.init(results->x.size(), Convert<HighPrecision, double>(results->maxRadius));

    double initX = (double)m_ScrnWidth / 2;
    double initY = (double)m_ScrnHeight / 2;

    double centerX, centerY;

    double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / m_ScrnWidth);
    double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / m_ScrnHeight);

    {
        HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
        HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

        HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)initX);
        HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)initY);

        centerX = (double)(cx - m_MinX);
        centerY = (double)(cy - m_MaxY);
    }

    static constexpr size_t num_threads = 32;
    std::deque<std::atomic_uint64_t> atomics;
    std::vector<std::unique_ptr<std::thread>> threads;
    atomics.resize(m_ScrnHeight);
    threads.reserve(num_threads);

    auto one_thread = [&]() {
        for (size_t y = 0; y < m_ScrnHeight; y++) {
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
                double DeltaNormSquared = 0;
                std::complex<double> DeltaSub0(DeltaSub0X, DeltaSub0Y);
                std::complex<double> DeltaSubN{};

                while (iter < m_NumIterations) {
                    BLA *b = nullptr;
                    while ((b = blas.lookupBackwards(RefIteration, DeltaNormSquared)) != nullptr) {
                        int l = b->getL();

                        if (iter + l >= m_NumIterations) {
                            break;
                        }

                        iter += l;

                        DeltaSubN = { DeltaSubNX, DeltaSubNY };

                        DeltaSubN = b->getValue(DeltaSubN, DeltaSub0);
                        // DeltaNormSquared = std::norm(DeltaSubN);

                        DeltaSubNX = DeltaSubN.real();
                        DeltaSubNY = DeltaSubN.imag();

                        RefIteration += l;

                        const double tempZX = results->x[RefIteration] + DeltaSubNX;
                        const double tempZY = results->y[RefIteration] + DeltaSubNY;
                        const double normSquared = tempZX * tempZX + tempZY * tempZY;
                        DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                        if (normSquared > 256) {
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

                    const double DeltaSubNXOrig = DeltaSubNX;
                    const double DeltaSubNYOrig = DeltaSubNY;

                    DeltaSubNX = DeltaSubNXOrig * (results->x2[RefIteration] + DeltaSubNXOrig) -
                        DeltaSubNYOrig * (results->y2[RefIteration] + DeltaSubNYOrig) +
                        DeltaSub0X;
                    DeltaSubNY = DeltaSubNXOrig * (results->y2[RefIteration] + DeltaSubNYOrig) +
                        DeltaSubNYOrig * (results->x2[RefIteration] + DeltaSubNXOrig) +
                        DeltaSub0Y;

                    ++RefIteration;

                    const double tempZX = results->x[RefIteration] + DeltaSubNX;
                    const double tempZY = results->y[RefIteration] + DeltaSubNY;
                    const double normSquared = tempZX * tempZX + tempZY * tempZY;
                    DeltaNormSquared = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                    if (normSquared > 256) {
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

void Fractal::AddPerturbationReferencePoint() {
    m_PerturbationResults.push_back({});
    PerturbationResults *results = &m_PerturbationResults[m_PerturbationResults.size() - 1];

    double initX = (double)m_ScrnWidth / 2;
    double initY = (double)m_ScrnHeight / 2;

    Point pt = { initX, initY };

    HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
    HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

    HighPrecision radiusX = dxHigh * ((HighPrecision)pt.x);
    HighPrecision radiusY = dyHigh * ((HighPrecision)pt.y);
    HighPrecision cx = m_MinX + radiusX;
    HighPrecision cy = m_MaxY - radiusY;

    results->hiX = cx;
    results->hiY = cy;
    results->radiusX = radiusX;
    results->radiusY = radiusY;
    results->maxRadius = (radiusX > radiusY) ? radiusX : radiusY;
    results->scrnX = (size_t)pt.x;
    results->scrnY = (size_t)pt.y;
    results->MaxIterations = m_NumIterations + 1; // +1 for push_back(0) below

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    unsigned int i;

    const double small_float = 1.1754944e-38;

    results->x.reserve(m_NumIterations);
    results->x2.reserve(m_NumIterations);
    results->y.reserve(m_NumIterations);
    results->y2.reserve(m_NumIterations);
    results->bad.reserve(m_NumIterations);
    results->bad_counts.reserve(m_NumIterations);

    results->x.push_back(0);
    results->x2.push_back(0);
    results->y.push_back(0);
    results->y2.push_back(0);
    results->bad_counts.push_back(0);
    bool cur_bad_count_underflow = false;
    // Note: results->bad is not here.  See end of this function.

    //double glitch = std::exp(std::log(0.0001));
    double glitch = 0.0001;

    zx = cx;
    zy = cy;
    for (i = 0; i < m_NumIterations; i++)
    {
        zx2 = zx * 2;
        zy2 = zy * 2;

        double double_zx = (double)zx;
        double double_zx2 = (double)zx2;
        double double_zy = (double)zy;
        double double_zy2 = (double)zy2;

        results->x.push_back(double_zx);
        results->x2.push_back(double_zx2);
        results->y.push_back(double_zy);
        results->y2.push_back(double_zy2);

        double norm = (double_zx * double_zx + double_zy * double_zy) * glitch;
        bool underflow = (fabs(zx) <= small_float ||
                          fabs(zy) <= small_float ||
                          norm <= small_float);
        results->bad.push_back(underflow);
        if (cur_bad_count_underflow != underflow) {
            results->bad_counts.push_back(1);
            cur_bad_count_underflow = !cur_bad_count_underflow;
        }
        else {
            results->bad_counts.back()++;
        }

        // x^2+2*I*x*y-y^2
        zx = zx * zx - zy * zy + cx;
        zy = zx2 * zy + cy;

        const double tempZX = double_zx + (double)cx;
        const double tempZY = double_zy + (double)cy;
        const double zn_size = tempZX * tempZX + tempZY * tempZY;
        if (zn_size > 256) {
            break;
        }
    }

    results->bad.push_back(false);
    assert(results->bad.size() == results->x.size());
}

bool Fractal::RequiresReferencePoints() const {
    switch (GetRenderAlgorithm()) {
        case RenderAlgorithm::Cpu64PerturbedBLA:
        case RenderAlgorithm::Gpu1x32PerturbedBLA:
        case RenderAlgorithm::Gpu1x32PerturbedScaled:
        case RenderAlgorithm::Gpu1x64PerturbedBLA:
        case RenderAlgorithm::Gpu2x32PerturbedBLA:
        case RenderAlgorithm::Gpu2x32PerturbedScaled:
            return true;
    }

    return false;
}

bool Fractal::IsPerturbationResultUsefulHere(size_t i) const {
    return m_PerturbationResults[i].hiX >= m_MinX &&
           m_PerturbationResults[i].hiX <= m_MaxX &&
           m_PerturbationResults[i].hiY >= m_MinY &&
           m_PerturbationResults[i].hiY <= m_MaxY &&
           (m_PerturbationResults[i].MaxIterations > m_PerturbationResults[i].x.size() ||
            m_PerturbationResults[i].MaxIterations >= m_NumIterations);
}

PerturbationResults* Fractal::GetUsefulPerturbationResults() {
    std::vector<PerturbationResults*> useful_results;

    if (!m_PerturbationResults.empty()) {
        if (m_PerturbationResults.size() > 64) {
            m_PerturbationResults.erase(m_PerturbationResults.begin());
        }

        for (size_t i = 0; i < m_PerturbationResults.size(); i++) {
            if (IsPerturbationResultUsefulHere(i)) {
                useful_results.push_back(&m_PerturbationResults[i]);

                m_PerturbationResults[i].scrnX = (size_t)((m_PerturbationResults[i].hiX - m_MinX) / (m_MaxX - m_MinX) * HighPrecision { m_ScrnWidth });
                m_PerturbationResults[i].scrnY = (size_t)((m_PerturbationResults[i].hiY - m_MinY) / (m_MaxY - m_MinY) * HighPrecision { m_ScrnHeight });
            }
            else {
                m_PerturbationResults[i].scrnX = MAXSIZE_T;
                m_PerturbationResults[i].scrnY = MAXSIZE_T;
            }
        }
    }

    PerturbationResults* results = nullptr;

    if (!useful_results.empty()) {
        results = useful_results[useful_results.size() - 1];
    }
    else {
        AddPerturbationReferencePoint();

        results = &m_PerturbationResults[m_PerturbationResults.size() - 1];
    }

    return results;
}

void Fractal::CalcGpuPerturbationFractalBLA(bool MemoryOnly) {
    PerturbationResults* results = GetUsefulPerturbationResults();

    BLAS blas(*results);
    blas.init(results->x.size(), Convert<HighPrecision, double>(results->maxRadius));

    uint32_t err =
        m_r.InitializeMemory(m_ScrnWidth * m_GpuAntialiasing,
                             m_ScrnHeight * m_GpuAntialiasing,
                             m_IterationAntialiasing,
                             m_CurIters.m_Width);
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

    MattPerturbResults gpu_results{
        results->x.size(),
        results->bad_counts.size(),
        results->x.data(),
        results->x2.data(),
        results->y.data(),
        results->y2.data(),
        results->bad.data(),
        results->bad_counts.data() };

    // all sizes should be the same anyway just pick one
    gpu_results.size = results->x.size();

    auto result = m_r.RenderPerturbBLA(m_RenderAlgorithm,
        (uint32_t*)m_CurIters.m_ItersMemory,
        &gpu_results,
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
uint32_t Fractal::FindMaxItersUsed(void)
{
    uint32_t prevMaxIters = 0;
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
    m_PaletteDepth(fractal.m_PaletteDepth),
    m_PaletteDepthIndex(fractal.m_PaletteDepthIndex),
    m_PalR(fractal.m_PalR),
    m_PalG(fractal.m_PalG),
    m_PalB(fractal.m_PalB),
    m_CurIters(std::move(fractal.m_CurIters)) {

    fractal.GetIterMemory();

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
                            numIters -= MAXITERS;
                        }

                        acc_r += m_PalR[m_PaletteDepthIndex][numIters];
                        acc_g += m_PalG[m_PaletteDepthIndex][numIters];
                        acc_b += m_PalB[m_PaletteDepthIndex][numIters];
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

    //setup converter
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;
    //use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    const std::string filename_png_c = converter.to_bytes(filename_png);

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

HighPrecision Fractal::Benchmark(size_t numIters)
{
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (RequiresReferencePoints()) {
        AddPerturbationReferencePoint();
    }

    if (!bm.StartTimer()) {
        return {};
    }

    CalcFractal(true);
    auto result = bm.StopTimer();
    bm.BenchmarkFinish();
    return result;
}

HighPrecision Fractal::BenchmarkReferencePoint(size_t numIters) {
    BenchmarkData bm(*this);
    bm.BenchmarkSetup(numIters);

    if (!bm.StartTimer()) {
        return {};
    }

    AddPerturbationReferencePoint();

    auto result = bm.StopTimer();
    bm.BenchmarkFinish();
    return result;
}

HighPrecision Fractal::BenchmarkThis() {
    BenchmarkData bm(*this);

    if (!bm.StartTimer()) {
        return {};
    }

    ChangedMakeDirty();
    CalcFractal(true);

    return bm.StopTimer();
}

Fractal::BenchmarkData::BenchmarkData(Fractal& fractal) :
    fractal(fractal) {}

void Fractal::BenchmarkData::BenchmarkSetup(size_t numIters) {
    prevScrnWidth = fractal.m_ScrnWidth;
    prevScrnHeight = fractal.m_ScrnHeight;

    fractal.ClearPerturbationResults();
    fractal.SetNumIterations(numIters);
    fractal.ResetDimensions(500, 500, 1, 1);
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

HighPrecision Fractal::BenchmarkData::StopTimer() {
    QueryPerformanceCounter(&endTime);

    __int64 freq64 = freq.QuadPart;
    __int64 startTime64 = startTime.QuadPart;
    __int64 endTime64 = endTime.QuadPart;
    __int64 totalIters = fractal.FindTotalItersUsed();

    __int64 deltaTime = endTime64 - startTime64;
    HighPrecision timeTaken = (HighPrecision)((HighPrecision)deltaTime / (HighPrecision)freq64);
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
HighPrecision Fractal::XFromScreenToCalc(HighPrecision x)
{
    HighPrecision OriginX = (HighPrecision)m_ScrnWidth / (m_MaxX - m_MinX) * -m_MinX;
    return (x - OriginX) * (m_MaxX - m_MinX) / m_ScrnWidth;
}

HighPrecision Fractal::YFromScreenToCalc(HighPrecision y)
{
    HighPrecision OriginY = (HighPrecision)m_ScrnHeight / (m_MaxY - m_MinY) * m_MaxY;
    return -(y - OriginY) * (m_MaxY - m_MinY) / (double)m_ScrnHeight;
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
    return ((GetAsyncKeyState(VK_CONTROL) & 0x8000) == 0x8000) &&
        ((m_hWnd != NULL && GetForegroundWindow() == m_hWnd) ||
        (m_hWnd == NULL));
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