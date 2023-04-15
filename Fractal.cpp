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

#include <locale>
#include <codecvt>

#include <GL/gl.h>      /* OpenGL header file */
#include <GL/glu.h>     /* OpenGL utilities header file */

#include "Fractal.h"
#include "FractalNetwork.h"
#include "FractalSetupData.h"
//#include "CBitmapWriter.h"

#include <thread>

void DefaultOutputMessage(const wchar_t *, ...);



Fractal::Fractal(FractalSetupData *setupData,
    int width,
    int height,
    void(*pOutputMessage) (const wchar_t *, ...),
    HWND hWnd,
    bool UseSensoCursor)
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

    // Reset our location in the Back array.
    m_CurPos = 0;

    // Setup member variables with initial values:
    Reset(width, height);
    StandardView();
    SetRenderAlgorithm(RenderAlgorithm::Gpu2x32);
    SetIterationAntialiasing(1);
    SetGpuAntialiasing(1);
    SetIterationPrecision(1);

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
    m_ItersMemory = new uint32_t[MaxFractalSize * MaxFractalSize];
    memset(m_ItersMemory, 0, MaxFractalSize * MaxFractalSize * sizeof(uint32_t));
    for (i = 0; i < MaxFractalSize; i++)
    {
        m_ItersArray[i] = &m_ItersMemory[i * MaxFractalSize];
    }

    m_RatioMemory = nullptr; // Lazy init later

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

    // Deallocate the big arrays.
    int i;
    for (i = 0; i < MaxFractalSize; i++)
    {
        m_ItersArray[i] = nullptr;
        m_RatioArray[i] = nullptr;
    }

    if (m_RatioArray) {
        delete[] m_RatioMemory;
        m_RatioMemory = nullptr;
    }

    delete[] m_ItersMemory;
    m_ItersMemory = nullptr;
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
void Fractal::Reset(size_t width, size_t height)
{
    if (m_ScrnWidth != width ||
        m_ScrnHeight != height)
    {
        m_ScrnWidth = width;
        m_ScrnHeight = height;
        m_ChangedScrn = true;

        SquareCurrentView();
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
void Fractal::StandardView(void)
{
    // Limits of 4x64 GPU
    //HighPrecision minX = HighPrecision{ "-1.763399177066752695854220120818493394874764715075525070697085376173644156624573649873526729559691534754284706803085481158" };
    //HighPrecision minY = HighPrecision{ "0.04289211262806512836473285627858318635734695759291867302112730624188941270466703058975670804976478935827994844038526618063053858" };
    //HighPrecision maxX = HighPrecision{ "-1.763399177066752695854220120818493394874764715075525070697085355870272824868052108014289980411203646967653925705407102169" };
    //HighPrecision maxY = HighPrecision{ "0.04289211262806512836473285627858318635734695759291867302112731461463330922125985949917962768812338316745717506303752530265831841" };
    //RecenterViewCalc(minX, minY, maxX, maxY);

    //// Limits of 4x32 GPU
    //HighPrecision minX = HighPrecision{ "-1.768969486867357972775564951275461551052751499509997185691881950786253743769635708375905775793656954725307354460920979983" };
    //HighPrecision minY = HighPrecision{ "0.05699280690304670893115636892860647833175463644922652375916712719872599382335388157040896288795946562522749757591414246314107544" };
    //HighPrecision maxX = HighPrecision{ "-1.768969486867357972775564950929487934553496494563941335911085292699250368065865432159590460057564657941788398574759610411" };
    //HighPrecision maxY = HighPrecision{ "0.05699280690304670893115636907127975355127952306391692141273804706041783710937876987367435542131127321801128526034375237132904264" };
    //RecenterViewCalc(minX, minY, maxX, maxY);

    //HighPrecision minX = HighPrecision{"-0.58495503626130003601029848213118066005878420809359614438890887303577302491190715010346117018574877216397030492770124781195671802"};
    //HighPrecision minY = HighPrecision{"0.65539077238267043815632402306524090571012503143059005580600414572477379600345575745505251230357715199720080866919744892255565016"};
    //HighPrecision maxX = HighPrecision{"-0.58495503626130003601029848186508480376748852380830163262290451155972369401636122663125472706742227530136237906642559212881715395"};
    //HighPrecision maxY = HighPrecision{"0.65539077238267043815632402317497403427516092222190248300856418146463464284705444362984488601344316801714300684703725236796367606" };
    //RecenterViewCalc(minX, minY, maxX, maxY);

    RecenterViewCalc(-2.5, -1.5, 1.5, 1.5);
    ResetNumIterations();
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
    if (m_CurPos > 0)
    {
        m_CurPos--;
        m_MinX = m_PrevMinX[m_CurPos];
        m_MinY = m_PrevMinY[m_CurPos];
        m_MaxX = m_PrevMaxX[m_CurPos];
        m_MaxY = m_PrevMaxY[m_CurPos];

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

                numberAtMax += m_ItersArray[y][x];
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
    m_PrevMinX[m_CurPos] = m_MinX;
    m_PrevMinY[m_CurPos] = m_MinY;
    m_PrevMaxX[m_CurPos] = m_MaxX;
    m_PrevMaxY[m_CurPos] = m_MaxY;
    m_CurPos++;
    if (m_CurPos >= 256) // Don't allow the "back" option to work
    {
        m_CurPos = 255;
    } // after so many levels.
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
    case RenderAlgorithm::Cpu64PerturbedGlitchy:
        CalcCpuPerturbationFractalGlitchy(MemoryOnly);
        break;
    case RenderAlgorithm::Cpu64PerturbedBLA:
        CalcCpuPerturbationFractalBLA(MemoryOnly);
        break;
    case RenderAlgorithm::Blend:
        FillRatioArrayIfNeeded();
        // fall through
    case RenderAlgorithm::Gpu1x64:
    case RenderAlgorithm::Gpu2x64:
    case RenderAlgorithm::Gpu4x64:
    case RenderAlgorithm::Gpu1x32:
    case RenderAlgorithm::Gpu2x32:
    case RenderAlgorithm::Gpu4x32:
        CalcGpuFractal(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x64PerturbedGlitchy:
    case RenderAlgorithm::Gpu2x32PerturbedGlitchy:
        CalcGpuPerturbationFractalGlitchy(MemoryOnly);
        break;
    case RenderAlgorithm::Gpu1x32PerturbedBLA:
    case RenderAlgorithm::Gpu1x64PerturbedBLA:
    case RenderAlgorithm::Gpu2x32PerturbedBLA:
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
        if (m_PerturbationResults[i].scrnX != MAXSIZE_T &&
            m_PerturbationResults[i].scrnY != MAXSIZE_T &&
            m_PerturbationResults[i].MaxIterations >= m_PerturbationResults[i].x.size()) {
            glColor3f((GLfloat)255, (GLfloat)255, (GLfloat)255);

            // Coordinates are weird in OGL mode.
            glVertex2i((GLint)m_PerturbationResults[i].scrnX, (GLint)m_PerturbationResults[i].scrnY);
        }
    }

    glEnd();
    glFlush();
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

    //if (m_SetupData.m_AltDraw == 'y') {
        // DrawFractal2();
    //}
    //else {
        size_t py;

        //glClear(GL_COLOR_BUFFER_BIT);

        for (py = 0; py < m_ScrnHeight; py++) {
            DrawFractalLine(py);
        }
    //}

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

                numIters = m_ItersArray[input_y][input_x];
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

void Fractal::DrawFractal2(void)
{
    //glClear(GL_COLOR_BUFFER_BIT);
    //glRasterPos2i(0, 0);

    //unsigned short *data = new unsigned short[m_ScrnWidth * m_ScrnHeight * 3];

    //size_t x, y, i = 0;
    //for (y = m_ScrnHeight - 1; y >= 0; y--)
    //{
    //    for (x = 0; x < m_ScrnWidth; x++)
    //    {
    //        data[i] = m_PalR[(m_ItersArray[y][x] == m_NumIterations) ? 0 : ((m_ItersArray[y][x] + m_PaletteRotate) % MAXITERS)];
    //        i++;
    //        data[i] = m_PalG[(m_ItersArray[y][x] == m_NumIterations) ? 0 : ((m_ItersArray[y][x] + m_PaletteRotate) % MAXITERS)];
    //        i++;
    //        data[i] = m_PalB[(m_ItersArray[y][x] == m_NumIterations) ? 0 : ((m_ItersArray[y][x] + m_PaletteRotate) % MAXITERS)];
    //        i++;
    //    }
    //}

    //glDrawPixels((GLint)m_ScrnWidth, (GLint)m_ScrnHeight, GL_RGB, GL_UNSIGNED_SHORT, data);

    //glFlush();
    //delete[] data;
}

void Fractal::FillRatioMemory(double Ratio, double &FinalRatio) {
    size_t count_d = 1, count_D = 1;

    for (size_t y = 0; y < MaxFractalSize; y++)
    {
        for (size_t x = 0; x < MaxFractalSize; x++)
        {
            FinalRatio = (double)count_D / (double)count_d;
            if (FinalRatio < Ratio) {
                m_RatioArray[y][x] = 'D'; // 'D'
                count_D++;
            }
            else {
                m_RatioArray[y][x] = 'f'; //'f'
                count_d++;
            }
        }
    }
}

void Fractal::FillRatioArrayIfNeeded() {
    if (m_RatioMemory == nullptr) {
        m_RatioMemory = new uint8_t[MaxFractalSize * MaxFractalSize];

        int i;
        memset(m_RatioMemory, 0, MaxFractalSize * MaxFractalSize * sizeof(uint8_t));
        for (i = 0; i < MaxFractalSize; i++)
        {
            m_RatioArray[i] = &m_RatioMemory[i * MaxFractalSize];
        }

        SetRenderAlgorithm(RenderAlgorithm::Blend);

        double FinalRatio1, FinalRatio2, FinalRatio3;

        //const size_t numIters = 5000000;
        const size_t numIters = 20000;

        FillRatioMemory(-1.0, FinalRatio1); // Force all 'f'
        m_r.SetRatioMemory(m_RatioMemory, MaxFractalSize);
        HighPrecision Result_1_64 = Benchmark(numIters);

        FillRatioMemory(DBL_MAX, FinalRatio2); // Force all 'D'
        m_r.SetRatioMemory(m_RatioMemory, MaxFractalSize);
        HighPrecision Result_2_32 = Benchmark(numIters);

        Result_2_32 *= 3;

        double Ratio = Convert<HighPrecision, double>(Result_2_32) / Convert<HighPrecision, double>(Result_1_64);

        //constexpr static auto NB_THREADS_W = 16; // Match in render_gpu.cu
        //constexpr static auto NB_THREADS_H = 8;
        //unsigned int w_block = MaxFractalSize / NB_THREADS_W;
        //unsigned int h_block = MaxFractalSize / NB_THREADS_H;

        FillRatioMemory(Ratio, FinalRatio3);
        m_r.SetRatioMemory(m_RatioMemory, MaxFractalSize);

        HighPrecision Result_combo;
        Result_combo = Benchmark(5000);
        Result_combo = Benchmark(numIters);
        
        std::stringstream ss;
        ss << std::string("Your computer calculated ");
        ss << Ratio
            << ", result_1_64: " << Convert<HighPrecision, double>(Result_1_64)
            << ", result_2_32: " << Convert<HighPrecision, double>(Result_2_32)
            << ", result_combo: " << Convert<HighPrecision, double>(Result_combo)
            << ", final ratio: " << FinalRatio1 << ", " << FinalRatio2 << ", " << FinalRatio3;
        std::string s = ss.str();
        const std::wstring ws(s.begin(), s.end());
        MessageBox(NULL, ws.c_str(), L"", MB_OK);

    }
}

void Fractal::FillCoordArray(const double *src, size_t size, MattCoordsArray &dest) {
    // note incomplete
    for (size_t i = 0; i < size; i++) {
        dest.doubleOnly[i] = src[i];
        dest.floatOnly[i] = (float)src[i];
        dest.flt[i].head = (float)src[i];
        dest.flt[i].tail = (float)(src[i] - (double) ((float)dest.flt[i].head));
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

    dest.flt.head = Convert<HighPrecision, float>(src);
    dest.flt.tail = Convert<HighPrecision, float>(src - HighPrecision{ dest.flt.head });

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

    m_r.InitializeMemory(m_ScrnWidth * m_GpuAntialiasing,
                         m_ScrnHeight * m_GpuAntialiasing,
                         m_IterationAntialiasing,
                         MaxFractalSize);

    m_r.Render(m_RenderAlgorithm,
               (uint32_t*)m_ItersMemory,
               cx2,
               cy2,
               dx2,
               dy2,
               (uint32_t)m_NumIterations,
               m_IterationPrecision);

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
            if (m_ProcessPixelRow[py] == m_NetworkRender)
            {
                if (m_RenderAlgorithm == RenderAlgorithm::CpuHigh)
                {
                    //CalcPixelRow_Multi ((unsigned int *) m_ItersArray[py], py);
                    std::unique_ptr<std::thread> t1(new std::thread(&Fractal::CalcPixelRow_Multi, this, (unsigned int *)m_ItersArray[py], py));
                    threads.push_back(std::move(t1));
                    threads_py.push_back(py);
                }
                else if (m_RenderAlgorithm == RenderAlgorithm::Cpu64)
                {
                    //CalcPixelRow_Multi ((unsigned int *) m_ItersArray[py], py);
                    std::unique_ptr<std::thread> t1(new std::thread(&Fractal::CalcPixelRow_C, this, (unsigned int*)m_ItersArray[py], py));
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
            if (m_NetworkRender >= 'b')
            {
                displayProgress++;
                if (displayProgress == 10)
                {
                    OutputMessage(L"%03.0f\b\b\b", (double)i * 10.0 + (double)py / m_ScrnHeight * 10.0);
                    displayProgress = 0;
                }
            }

            if (m_StopCalculating == true)
            {
                if (m_NetworkRender == 'a')
                {
                    ClientSendQuitMessages();
                }
                break;
            }

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

                m_ItersArray[ny][nx] = numIters;
            }

            if (MemoryOnly == false)
            {
                DrawFractalLine(ny);
            }
        }
    }
}

void Fractal::CalcCpuPerturbationFractalGlitchy(bool MemoryOnly) {
    PerturbationResults results;
    std::vector<Point> glitched;

    double initX = (double)(rand() % m_ScrnWidth);
    double initY = (double)(rand() % m_ScrnHeight);
    //double initX = m_ScrnWidth / 2;
    //double initY = m_ScrnHeight / 2;
    glitched.push_back({ initX, initY, 0 });

    for (size_t row = 0; row < m_ScrnHeight; row++) {
        memset(m_ItersArray[row], 0, MaxFractalSize * sizeof(uint32_t));
    }

    for (size_t count = 0; count < 5; count++) {
        double centerX, centerY;

        int randIndex = rand() % glitched.size();
        auto pt = glitched[randIndex];

        results.clear();

        double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / m_ScrnWidth);
        double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / m_ScrnHeight);

        {
            HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
            HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

            HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)pt.x);
            HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)pt.y);

            HighPrecision zx, zy;
            HighPrecision zx2, zy2;
            unsigned int i;

            centerX = (double)(cx - m_MinX);
            centerY = (double)(cy - m_MaxY);

            zx = cx;
            zy = cy;
            for (i = 0; i < m_NumIterations; i++)
            {
                // x^2+2*I*x*y-y^2
                zx2 = zx * 2;
                zy2 = zy * 2;

                results.x.push_back((double)zx);
                results.x2.push_back((double)zx2);
                results.y.push_back((double)zy);
                results.y2.push_back((double)zy2);
                //results.complex2.push_back({ (double)zx2, (double)zy2 });

                const double glitchTolerancy = 0.0000001;
                double tolerancyReal = (double)zx * glitchTolerancy;
                double tolerancyImag = (double)zy * glitchTolerancy;
                std::complex<double> tolerancy(tolerancyReal, tolerancyImag);
                results.tolerancy.push_back(std::norm(tolerancy));

                zx = zx * zx - zy * zy + cx;
                zy = zx2 * zy + cy;
            }
        }

        size_t iter;
        const size_t max_iter = results.x.size() - 1;

        double zn_size;

        // (zx + zy)^2 = zx^2 + 2*zx*zy + zy^2
        // (zx + zy)^3 = zx^3 + 3*zx^2*zy + 3*zx*zy^2 + zy
        for (size_t y = 0; y < m_ScrnHeight; y++) {
            for (size_t x = 0; x < m_ScrnWidth; x++)
            {
                if (m_ItersArray[y][x] != 0) {
                    continue;
                }

                iter = 0;

                //double deltaReal = dx * x - centerX;
                //double deltaImaginary = -dy * y - centerY;

                //std::complex<double> DeltaSub0(deltaReal, deltaImaginary);
                //std::complex<double> DeltaSubN;

                //DeltaSubN = DeltaSub0;

                //bool glitched = false;

                //do
                //{
                //    DeltaSubN *= results.complex2[iter] + DeltaSubN;
                //    DeltaSubN += DeltaSub0;

                //    ++iter;
                //    zn_size = std::norm(results.complex2[iter] * 0.5 + DeltaSubN);

                //    if (glitched == false &&
                //        zn_size < results.tolerancy[iter]) {
                //        glitched = true;
                //        break;
                //    }
                //} while (zn_size < 256 && iter < max_iter);

                double deltaReal = dx * x - centerX;
                double deltaImaginary = -dy * y - centerY;

                double DeltaSub0X = deltaReal;
                double DeltaSub0Y = deltaImaginary;
                double DeltaSubNX, DeltaSubNY;

                DeltaSubNX = DeltaSub0X;
                DeltaSubNY = DeltaSub0Y;

                bool bglitched = false;

                do {
                    const double DeltaSubNXOrig = DeltaSubNX;
                    const double DeltaSubNYOrig = DeltaSubNY;

                    DeltaSubNX = DeltaSubNXOrig * (results.x2[iter] + DeltaSubNXOrig) -
                                 DeltaSubNYOrig * (results.y2[iter] + DeltaSubNYOrig);
                    DeltaSubNX += DeltaSub0X;

                    DeltaSubNY = DeltaSubNXOrig * (results.y2[iter] + DeltaSubNYOrig) +
                                 DeltaSubNYOrig * (results.x2[iter] + DeltaSubNXOrig);
                    DeltaSubNY += DeltaSub0Y;

                    ++iter;

                    const double tempX = results.x[iter] + DeltaSubNX;
                    const double tempY = results.y[iter] + DeltaSubNY;
                    zn_size = tempX * tempX + tempY * tempY;

                    if (bglitched == false &&
                        zn_size < results.tolerancy[iter]) {
                        bglitched = true;
                        break;
                    }
                } while (zn_size < 256 && iter < max_iter);

                if (bglitched == false) {
                    m_ItersArray[y][x] = (uint32_t)iter;
                }
                else {
                    m_ItersArray[y][x] = 0;
                    glitched.push_back({ (double)x,(double)y,iter });
                }
            }
        }

        if (glitched.size() < 100) {
            break;
        }
    }

    DrawFractal(MemoryOnly);
}

void Fractal::CalcCpuPerturbationFractalBLA(bool MemoryOnly) {
    PerturbationResults results;

    double initX = (double)m_ScrnWidth / 2;
    double initY = (double)m_ScrnHeight / 2;

    for (size_t row = 0; row < m_ScrnHeight; row++) {
        memset(m_ItersArray[row], 0, MaxFractalSize * sizeof(uint32_t));
    }

    double centerX, centerY;

    results.clear();

    double dx = Convert<HighPrecision, double>((m_MaxX - m_MinX) / m_ScrnWidth);
    double dy = Convert<HighPrecision, double>((m_MaxY - m_MinY) / m_ScrnHeight);

    {
        HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
        HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

        HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)initX);
        HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)initY);

        HighPrecision zx, zy;
        HighPrecision zx2, zy2;
        unsigned int i;

        centerX = (double)(cx - m_MinX);
        centerY = (double)(cy - m_MaxY);

        results.x.push_back(0);
        results.x2.push_back(0);
        results.y.push_back(0);
        results.y2.push_back(0);
        results.complex.push_back({ 0,0 });
        results.complex2.push_back({ 0,0 });
        results.tolerancy.push_back(0);

        zx = cx;
        zy = cy;
        for (i = 0; i < m_NumIterations; i++)
        {
            // x^2+2*I*x*y-y^2
            zx2 = zx * 2;
            zy2 = zy * 2;

            results.x.push_back((double)zx);
            results.x2.push_back((double)zx2);
            results.y.push_back((double)zy);
            results.y2.push_back((double)zy2);
            results.complex.push_back({ (double)zx, (double)zy });
            results.complex2.push_back({ (double)zx2, (double)zy2 });

            const double glitchTolerancy = 0.0000001;
            double tolerancyReal = (double)zx * glitchTolerancy;
            double tolerancyImag = (double)zy * glitchTolerancy;
            std::complex<double> tolerancy(tolerancyReal, tolerancyImag);
            results.tolerancy.push_back(std::norm(tolerancy));

            zx = zx * zx - zy * zy + cx;
            zy = zx2 * zy + cy;
        }
    }


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

    for (size_t y = 0; y < m_ScrnHeight; y++) {
        for (size_t x = 0; x < m_ScrnWidth; x++)
        {
            if (m_ItersArray[y][x] != 0) {
                continue;
            }

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
                // https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html#a2021-05-14_deep_zoom_theory_and_practice_rescaling
                // 
                // DeltaSubN = 2 * DeltaSubN * results.complex[RefIteration] + DeltaSubN * DeltaSubN + DeltaSub0;
                // S * w = 2 * S * w * results.complex[RefIteration] + S * w * S * w + S * c
                // 
                // S * (DeltaSubNWX + DeltaSubNWY I) = 2 * S * (DeltaSubNWX + DeltaSubNWY I) * (results.x[RefIteration] + results.y[RefIteration] * I) +
                //                                     S * S * (DeltaSubNWX + DeltaSubNWY I) * (DeltaSubNWX + DeltaSubNWY I) +
                //                                     S * d
                // 
                // (DeltaSubNWX + DeltaSubNWY I) = 2 * (DeltaSubNWX + DeltaSubNWY I) * (results.x[RefIteration] + results.y[RefIteration] * I) +
                //                                 S * (DeltaSubNWX + DeltaSubNWY I) * (DeltaSubNWX + DeltaSubNWY I) +
                //                                 d
                // 

                DeltaSubNX = DeltaSubNXOrig * (results.x2[RefIteration] + DeltaSubNXOrig) -
                             DeltaSubNYOrig * (results.y2[RefIteration] + DeltaSubNYOrig) +
                             DeltaSub0X;
                DeltaSubNY = DeltaSubNXOrig * (results.y2[RefIteration] + DeltaSubNYOrig) +
                             DeltaSubNYOrig * (results.x2[RefIteration] + DeltaSubNXOrig) +
                             DeltaSub0Y;

                ++RefIteration;

                const double tempZX = results.x[RefIteration] + DeltaSubNX;
                const double tempZY = results.y[RefIteration] + DeltaSubNY;
                const double zn_size = tempZX * tempZX + tempZY * tempZY;
                const double normDeltaSubN = DeltaSubNX * DeltaSubNX + DeltaSubNY * DeltaSubNY;

                if (zn_size > 256) {
                    break;
                }

                if (zn_size < normDeltaSubN ||
                    RefIteration == results.x.size() - 1) {
                    DeltaSubNX = tempZX;
                    DeltaSubNY = tempZY;
                    RefIteration = 0;
                }

                ++iter;
            }

            m_ItersArray[y][x] = (uint32_t)iter;
        }
    }

    DrawFractal(MemoryOnly);
}

void Fractal::CalcGpuPerturbationFractalGlitchy(bool MemoryOnly) {
    //m_PerturbationResults.clear(); // TODO

    std::vector<Point> glitched;

    if (!m_PerturbationResults.empty()) {
        for (size_t i = m_PerturbationResults.size() - 1; i >= 0; i--) {
            if (m_PerturbationResults[i].hiX >= m_MinX && m_PerturbationResults[i].hiX <= m_MaxX &&
                m_PerturbationResults[i].hiY >= m_MinY && m_PerturbationResults[i].hiY <= m_MaxY) {
                continue;
            }

            m_PerturbationResults.erase(m_PerturbationResults.begin() + i);
        }
    }

    for (size_t row = 0; row < m_ScrnHeight; row++) {
        memset(m_ItersArray[row], 0, MaxFractalSize * sizeof(uint32_t));
    }

    MattCoords cx2{}, cy2{}, dx2{}, dy2{};
    MattCoords centerX2{}, centerY2{};

    FillGpuCoords(cx2, cy2, dx2, dy2);

    m_r.InitializeMemory(m_ScrnWidth * m_GpuAntialiasing,
        m_ScrnHeight * m_GpuAntialiasing,
        m_IterationAntialiasing,
        MaxFractalSize);

    m_r.ClearMemory();

    if (glitched.empty()) {
        double initX = (double)m_ScrnWidth / 2;
        double initY = (double)m_ScrnHeight / 2;
        glitched.push_back({ initX, initY, 0 });
    }

    if (!m_PerturbationResults.empty()) {
        size_t max_iters = 0;
        size_t max_index = 0;

        for (int count = 0; count < m_PerturbationResults.size(); count++) {
            PerturbationResults& results = m_PerturbationResults[count];
            if (results.x.size() > max_iters) {
                max_iters = results.x.size();
                max_index = count;
            }
        }

        PerturbationResults& results = m_PerturbationResults[max_index];
        glitched.clear();

        HighPrecision centerX = results.hiX - m_MinX;
        HighPrecision centerY = results.hiY - m_MaxY;

        FillCoord(centerX, centerX2);
        FillCoord(centerY, centerY2);

        MattPerturbResults gpu_results{ results.x.size() };
        FillCoordArray(results.x.data(), results.x.size(), gpu_results.x);
        FillCoordArray(results.x2.data(), results.x2.size(), gpu_results.x2);
        FillCoordArray(results.y.data(), results.y.size(), gpu_results.y);
        FillCoordArray(results.y2.data(), results.y2.size(), gpu_results.y2);
        FillCoordArray(results.tolerancy.data(), results.tolerancy.size(), gpu_results.tolerancy);

        // all sizes should be the same anyway just pick one
        gpu_results.size = results.tolerancy.size();

        m_r.RenderPerturbGlitchy (m_RenderAlgorithm,
            (uint32_t*)m_ItersMemory,
            &gpu_results,
            cx2,
            cy2,
            dx2,
            dy2,
            centerX2,
            centerY2,
            (uint32_t)m_NumIterations,
            m_IterationPrecision);

        for (size_t y = 0; y < m_ScrnHeight; y++) {
            for (size_t x = 0; x < m_ScrnWidth; x++)
            {
                if (m_ItersArray[y][x] == 0) {
                    glitched.push_back({ (double)x,(double)y, 0 });
                }
            }
        }
    }

    if (!glitched.empty()) {

        double totalX = 0;
        double totalY = 0;
        bool alt = false;

        for (size_t count = 0; count < 10; count++) {
            Point pt;

            if (alt == false && (totalX != 0 || totalY != 0)) {
                pt = Point{ totalX, totalY };
                alt = true;
            }
            else if (!glitched.empty()) {
                int randIndex = rand() % glitched.size();
                pt = glitched[randIndex];
                alt = false;
            }
            else
            {
                break;
            }

            m_PerturbationResults.push_back({});
            PerturbationResults& results = m_PerturbationResults[count];

            HighPrecision dxHigh = (m_MaxX - m_MinX) / m_ScrnWidth;
            HighPrecision dyHigh = (m_MaxY - m_MinY) / m_ScrnHeight;

            HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)pt.x);
            HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)pt.y);

            results.hiX = cx;
            results.hiY = cy;
            results.scrnX = (size_t)pt.x;
            results.scrnY = (size_t)pt.y;
            results.MaxIterations = m_NumIterations;

            HighPrecision zx, zy;
            HighPrecision zx2, zy2;
            unsigned int i;

            HighPrecision centerX = cx - m_MinX;
            HighPrecision centerY = cy - m_MaxY;

            FillCoord(centerX, centerX2);
            FillCoord(centerY, centerY2);

            zx = cx;
            zy = cy;
            for (i = 0; i < m_NumIterations; i++)
            {
                // x^2+2*I*x*y-y^2
                zx2 = zx * 2;
                zy2 = zy * 2;

                results.x.push_back((double)zx);
                results.x2.push_back((double)zx2);
                results.y.push_back((double)zy);
                results.y2.push_back((double)zy2);

                const double glitchTolerancy = 0.0000001;
                double tolerancyReal = (double)zx * glitchTolerancy;
                double tolerancyImag = (double)zy * glitchTolerancy;
                std::complex<double> tolerancy(tolerancyReal, tolerancyImag);
                results.tolerancy.push_back(std::norm(tolerancy));

                zx = zx * zx - zy * zy + cx;
                zy = zx2 * zy + cy;
            }


            glitched.clear();

            MattPerturbResults gpu_results{ results.x.size() };
            FillCoordArray(results.x.data(), results.x.size(), gpu_results.x);
            FillCoordArray(results.x2.data(), results.x2.size(), gpu_results.x2);
            FillCoordArray(results.y.data(), results.y.size(), gpu_results.y);
            FillCoordArray(results.y2.data(), results.y2.size(), gpu_results.y2);
            FillCoordArray(results.tolerancy.data(), results.tolerancy.size(), gpu_results.tolerancy);

            // all sizes should be the same anyway just pick one
            gpu_results.size = results.tolerancy.size();

            m_r.RenderPerturbGlitchy (m_RenderAlgorithm,
                (uint32_t*)m_ItersMemory,
                &gpu_results,
                cx2,
                cy2,
                dx2,
                dy2,
                centerX2,
                centerY2,
                (uint32_t)m_NumIterations,
                m_IterationPrecision);

            totalX = 0;
            totalY = 0;

            for (size_t y = 0; y < m_ScrnHeight; y++) {
                for (size_t x = 0; x < m_ScrnWidth; x++)
                {
                    if (m_ItersArray[y][x] == 0) {
                        glitched.push_back({ (double)x,(double)y, 0 });
                        totalX += x;
                        totalY += y;
                    }
                }
            }

            totalX /= (double)glitched.size();
            totalY /= (double)glitched.size();

            if (m_ItersArray[(size_t)totalY][(size_t)totalX] != 0)
            {
                totalX = 0;
                totalY = 0;
            }

            if (glitched.size() < 100) {
                break;
            }
        }
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

    HighPrecision cx = m_MinX + dxHigh * ((HighPrecision)pt.x);
    HighPrecision cy = m_MaxY - dyHigh * ((HighPrecision)pt.y);

    results->hiX = cx;
    results->hiY = cy;
    results->scrnX = (size_t)pt.x;
    results->scrnY = (size_t)pt.y;
    results->MaxIterations = m_NumIterations + 1; // +1 for push_back(0) below

    HighPrecision zx, zy;
    HighPrecision zx2, zy2;
    unsigned int i;

    results->x.reserve(1000000);
    results->x2.reserve(1000000);
    results->y.reserve(1000000);
    results->y2.reserve(1000000);

    results->x.push_back(0);
    results->x2.push_back(0);
    results->y.push_back(0);
    results->y2.push_back(0);

    zx = cx;
    zy = cy;
    for (i = 0; i < m_NumIterations; i++)
    {
        zx2 = zx * 2;
        zy2 = zy * 2;

        results->x.push_back((double)zx);
        results->x2.push_back((double)zx2);
        results->y.push_back((double)zy);
        results->y2.push_back((double)zy2);

        // x^2+2*I*x*y-y^2
        zx = zx * zx - zy * zy + cx;
        zy = zx2 * zy + cy;

        const double tempZX = (double)zx + (double)cx;
        const double tempZY = (double)zy + (double)cy;
        const double zn_size = tempZX * tempZX + tempZY * tempZY;
        if (zn_size > 256) {
            break;
        }
    }
}

void Fractal::CalcGpuPerturbationFractalBLA(bool MemoryOnly) {
    std::vector<PerturbationResults*> useful_results;

    if (!m_PerturbationResults.empty()) {
        for (size_t i = 0; i < m_PerturbationResults.size(); i++) {
            if (m_PerturbationResults[i].hiX >= m_MinX && m_PerturbationResults[i].hiX <= m_MaxX &&
                m_PerturbationResults[i].hiY >= m_MinY && m_PerturbationResults[i].hiY <= m_MaxY &&
                m_PerturbationResults[i].MaxIterations >= m_PerturbationResults[i].x.size()) {
                useful_results.push_back(&m_PerturbationResults[i]);

                m_PerturbationResults[i].scrnX = (size_t) ((m_PerturbationResults[i].hiX - m_MinX) / (m_MaxX - m_MinX) * HighPrecision{ m_ScrnWidth });
                m_PerturbationResults[i].scrnY = (size_t) ((m_PerturbationResults[i].hiY - m_MinY) / (m_MaxY - m_MinY) * HighPrecision { m_ScrnHeight });
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

    m_r.InitializeMemory(m_ScrnWidth * m_GpuAntialiasing,
        m_ScrnHeight * m_GpuAntialiasing,
        m_IterationAntialiasing,
        MaxFractalSize);

    m_r.ClearMemory();

    MattCoords cx2{}, cy2{}, dx2{}, dy2{};
    MattCoords centerX2{}, centerY2{};

    FillGpuCoords(cx2, cy2, dx2, dy2);

    HighPrecision centerX = results->hiX - m_MinX;
    HighPrecision centerY = results->hiY - m_MaxY;

    FillCoord(centerX, centerX2);
    FillCoord(centerY, centerY2);

    MattPerturbResults gpu_results{ results->x.size() };
    FillCoordArray(results->x.data(), results->x.size(), gpu_results.x);
    FillCoordArray(results->x2.data(), results->x2.size(), gpu_results.x2);
    FillCoordArray(results->y.data(), results->y.size(), gpu_results.y);
    FillCoordArray(results->y2.data(), results->y2.size(), gpu_results.y2);

    // all sizes should be the same anyway just pick one
    gpu_results.size = results->x.size();

    m_r.RenderPerturbBLA(m_RenderAlgorithm,
        (uint32_t*)m_ItersMemory,
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
            if (m_ItersArray[y][x] > prevMaxIters)
            {
                prevMaxIters = m_ItersArray[y][x];
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
int Fractal::SaveCurrentFractal(const std::wstring filename_base)
{
    //CBitmapWriter bmpWriter;
    int ret;
    //unsigned char *data = new unsigned char[m_ScrnWidth * m_ScrnHeight * 3];
    double acc_r, acc_b, acc_g;
    size_t input_x, input_y;
    size_t output_x, output_y;
    size_t i;
    size_t numIters;

    std::wstring filename_bmp = filename_base + std::wstring(L".bmp");
    std::wstring filename_png;

    i = 0;
    do {
        wchar_t temp[512];
        wsprintf(temp, L"%s%05d", filename_base.c_str(), i);
        filename_png = std::wstring(temp) + std::wstring(L".png");
        i++;
    } while (FileExists(filename_png.c_str()));

    i = 0;

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

                    numIters = m_ItersArray[input_y][input_x];
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

    //ret = bmpWriter.Write(filename_bmp.c_str(), data, (int)m_ScrnWidth, (int)m_ScrnHeight);

    //// bmp -> png
    //HANDLE bmpFile = CreateFile(filename_bmp.c_str(), 0, 0, NULL, OPEN_EXISTING, 0, NULL);
    //DWORD upperFilesize, lowerFilesize;
    //lowerFilesize = GetFileSize(bmpFile, &upperFilesize);
    //CloseHandle(bmpFile);

    //if (lowerFilesize != INVALID_FILE_SIZE && lowerFilesize < 1000000000 && upperFilesize == 0)
    //{
    //    CxImage image;
    //    image.Load(filename_bmp.c_str(), CXIMAGE_FORMAT_BMP);
    //    if (image.IsValid())
    //    {
    //        if (image.Save(filename_png.c_str(), CXIMAGE_FORMAT_PNG) == true)
    //        {
    //            _wunlink(filename_bmp.c_str());
    //        }
    //    }
    //}

    //const std::string filename_png_c(filename_png.begin(), filename_png.end());

    //setup converter
    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;
    //use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    const std::string filename_png_c = converter.to_bytes(filename_png);

    ret = image.saveImage(filename_png_c, WPngImage::PngFileFormat::kPngFileFormat_RGBA16);

    //delete[] data;

    return ret;
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
    Reset(16384, 16384);
    SquareCurrentView();

    if (m_ScrnWidth * m_GpuAntialiasing * m_IterationAntialiasing >= MaxFractalSize ||
        m_ScrnHeight * m_GpuAntialiasing * m_IterationAntialiasing >= MaxFractalSize) {
        ::MessageBox(NULL, L"Increase MaxFractalSize and recompile", L"", MB_OK);
        return 0;
    }

    // Calculate the high res image.
    // Do it in memory! :)
    CalcFractal(true);

    // Save the bitmap.
    int ret = SaveCurrentFractal(filename);

    // Back to the previous res.
    Reset(OldScrnWidth, OldScrnHeight);

    return ret;
}

HighPrecision Fractal::Benchmark(size_t numIters)
{
    BenchmarkData bm(*this);
    if (!bm.BenchmarkSetup(numIters)) {
        return {};
    }

    CalcFractal(true);
    return bm.BenchmarkFinish();
}

HighPrecision Fractal::BenchmarkReferencePoint(size_t numIters) {
    BenchmarkData bm(*this);
    if (!bm.BenchmarkSetup(numIters)) {
        return {};
    }

    AddPerturbationReferencePoint();
    return bm.BenchmarkFinish();

}

Fractal::BenchmarkData::BenchmarkData(Fractal& fractal) :
    fractal(fractal) {}

bool Fractal::BenchmarkData::BenchmarkSetup(size_t numIters) {
    prevScrnWidth = fractal.m_ScrnWidth;
    prevScrnHeight = fractal.m_ScrnHeight;

    fractal.Reset(500, 500);
    fractal.SetNumIterations(numIters);
    fractal.SetIterationAntialiasing(1);
    fractal.SetGpuAntialiasing(1);
    //fractal.SetIterationPrecision(1);
    //fractal.RecenterViewCalc(-.1, -.1, .1, .1);
    fractal.RecenterViewCalc(-1.5, -.75, 1, .75);

    startTime.QuadPart = 0;
    endTime.QuadPart = 1;
    if (QueryPerformanceFrequency(&freq) == 0) {
        return false;
    }

    QueryPerformanceCounter(&startTime);
    return true;
}

HighPrecision Fractal::BenchmarkData::BenchmarkFinish() {
    QueryPerformanceCounter(&endTime);

    __int64 freq64 = freq.QuadPart;
    __int64 startTime64 = startTime.QuadPart;
    __int64 endTime64 = endTime.QuadPart;
    __int64 totalIters = fractal.FindTotalItersUsed();

    __int64 deltaTime = endTime64 - startTime64;
    HighPrecision timeTaken = (HighPrecision)((HighPrecision)deltaTime / (HighPrecision)freq64);

    fractal.Reset(prevScrnWidth, prevScrnHeight);
    fractal.StandardView();

    fractal.ChangedMakeDirty();

    return (HighPrecision)(totalIters / timeTaken) / 1000000.0;
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
            //if (m_ItersArray[y][x] == 0) // note possible bug
            //{
            //    numIters += (__int64)m_NumIterations;
            //}
            //else
            //{
            //    numIters += (__int64)m_ItersArray[y][x];
            //}

            numIters += (__int64)m_ItersArray[y][x];
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
                for (j = 0; j < MaxFractalSize; j++)
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
        for (i = 0; i < MaxFractalSize; i++)
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

        for (i = 0; i < MaxFractalSize; i++) // Default to having the client do everything
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
                for (counter = 0; counter < MaxFractalSize; counter += 100.0 / m_SetupData.m_WorkServers[i])
                { // If this row is already assigned, move on to the next one.
                    while (m_ProcessPixelRow[(int)counter] != 'a')
                    {
                        counter++;
                    }

                    // Don't overwrite other memory!
                    if (counter >= MaxFractalSize)
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
        for (i = 0; i < MaxFractalSize; i++)      // Placeholder until the server
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

    for (i = 0; i < MaxFractalSize; i++)
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

    Reset(ScreenWidth, ScreenHeight);
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
                m_ServerSubNetwork->BufferedSendLong(m_ItersArray[y][x]);
            }
            else
            {
                m_ServerSubNetwork->BufferedSendShort((unsigned short)m_ItersArray[y][x]);
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