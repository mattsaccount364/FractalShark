#ifndef FRACTAL_H
#define FRACTAL_H

//
// Comments for Matthew:
// - Re-use information from points in set between frames so that the all-black
//   pixels don't suck up so much time (skip chunks on the interior).
// - Multisample antialiasing, where we look for the "edges" and recalculating
//   those only?
// - Add a FP32 GPU for quick rendering at start up.
// - Put all this in GIT
// - Can we do a e.g. 256 bit float on GPU e.g. using fixed point integers etc
//   like fractint
// - Can we calculate several frames in parallel on GPU
// - Profile 128-bit implementation on GPU and optimize?
// - Add text box in UI to import a set of coordinates so we can navigate
//   somewhere saved.
//

#include "FractalSetupData.h"
#include "FractalNetwork.h"
#include "render_gpu.h"
#include "PerturbationResults.h"

#include "..\WPngImage\WPngImage.hh"
#include "HighPrecision.h"

//const int MAXITERS = 256 * 32; // 256 * 256 * 256 * 32
const int MAXITERS = 256 * 256 * 256 * 32;

class Fractal
{
public:
    Fractal(FractalSetupData *setupData,
        int width,
        int height,
        void(*pOutputMessage) (const wchar_t *, ...),
        HWND hWnd,
        bool UseSensoCursor);
    ~Fractal();

private:
    void Initialize(int width,
        int height,
        void(*OutputMessage) (const wchar_t *, ...),
        HWND hWnd,
        bool UseSensoCursor);
    void Uninitialize(void);
    bool PalIncrease(std::vector<uint16_t>& pal, int i1, int length, int val1, int val2);
    int PalTransition(size_t paletteIndex, int i1, int length, int r, int g, int b);

public:
    static unsigned long WINAPI CheckForAbortThread(void *fractal);

private:
    bool IsDownControl(void);
    void CheckForAbort(void);

public: // Changing the view
    void ResetDimensions(size_t width = MAXSIZE_T,
                         size_t height = MAXSIZE_T,
                         uint32_t iteration_antialiasing = UINT32_MAX,
                         uint32_t gpu_antialiasing = UINT32_MAX);
    bool RecenterViewCalc(HighPrecision MinX, HighPrecision MinY, HighPrecision MaxX, HighPrecision MaxY);
    bool RecenterViewScreen(RECT rect);
    bool CenterAtPoint(int x, int y);
    void View(size_t i);
    void SquareCurrentView(void);
    void ApproachTarget(void);

private:
    static bool FileExists(const wchar_t *filename);  // Used only by ApproachTarget

public:
    bool Back(void);

public: // For screen saver specifically
    void FindInterestingLocation(RECT *rect);
    bool IsValidLocation(void);

private: // For saving the current location
    void SaveCurPos(void);

public: // Iterations
    void SetNumIterations(size_t num);
    size_t GetNumIterations(void);
    void ResetNumIterations(void);

public:
    inline RenderAlgorithm GetRenderAlgorithm(void) const { return m_RenderAlgorithm; }
    inline void SetRenderAlgorithm(RenderAlgorithm alg) { m_RenderAlgorithm = alg; }

    inline uint32_t GetIterationAntialiasing(void) const { return m_IterationAntialiasing; }
    inline uint32_t GetGpuAntialiasing(void) const { return m_GpuAntialiasing; }
    inline uint32_t GetIterationPrecision(void) const { return m_IterationPrecision; }
    inline void SetIterationPrecision(uint32_t iteration_precision) { m_IterationPrecision = iteration_precision;  }

private: // Keeps track of what has changed and what hasn't since the last draw
    inline void ChangedMakeClean(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = false; }
    inline void ChangedMakeDirty(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = true; }
    inline bool ChangedIsDirty(void) const { return (m_ChangedWindow || m_ChangedScrn || m_ChangedIterations); }
    inline bool ChangedItersOnly(void) const { return (m_ChangedIterations && !(m_ChangedScrn || m_ChangedWindow)); }

public: // Drawing functions
    void CalcDiskFractal(wchar_t *filename);
    void CalcFractal(bool MemoryOnly);
    void DrawFractal(bool MemoryOnly);

    void UsePalette(int depth);
    void ResetFractalPalette(void);
    void RotateFractalPalette(int delta);
    void CreateNewFractalPalette(void);

    void DrawPerturbationResults(bool MemoryOnly);
    void ClearPerturbationResults();

private:
    void DrawRotatedFractal(void);
    void DrawFractalLine(size_t row);

    void CalcCpuFractal(bool MemoryOnly);

    struct Point {
        double x, y;
        size_t iteration;
    };

    std::vector<PerturbationResults> m_PerturbationResults;
    void AddPerturbationReferencePoint();
    bool RequiresReferencePoints() const;
    PerturbationResults* GetUsefulPerturbationResults();

    static void FillCoord(HighPrecision& src, MattCoords& dest);
    void FillGpuCoords(MattCoords& cx2, MattCoords& cy2, MattCoords& dx2, MattCoords& dy2);
    void CalcGpuFractal(bool MemoryOnly);
    void CalcNetworkFractal(bool MemoryOnly);
    void CalcCpuPerturbationFractal(bool MemoryOnly);
    void CalcCpuPerturbationFractalBLA(bool MemoryOnly);
    void CalcGpuPerturbationFractalBLA(bool MemoryOnly);

    void CalcPixelRow_Multi(unsigned int *rowBuffer, size_t row); // Multiprecision
    bool CalcPixelRow_Exp(unsigned int *rowBuffer, size_t row); // Experimental
    bool CalcPixelRow_C(unsigned int *rowBuffer, size_t row);

    bool IsPerturbationResultUsefulHere(size_t i) const;

private:
    uint32_t FindMaxItersUsed(void);

public: // Saving images of the fractal
    int SaveCurrentFractal(const std::wstring filename_base);
    int SaveHiResFractal(const std::wstring filename_base);
    void CleanupThreads();

public: // Benchmarking
    HighPrecision Benchmark(size_t numIters);
    HighPrecision BenchmarkReferencePoint(size_t numIters);
    HighPrecision BenchmarkThis();

private:
    struct BenchmarkData {
        BenchmarkData(Fractal& fractal);
        Fractal& fractal;

        LARGE_INTEGER freq;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

        size_t prevScrnWidth;
            size_t prevScrnHeight;

        void BenchmarkSetup(size_t numIters);
        bool StartTimer();
        HighPrecision StopTimer();
        void BenchmarkFinish();
    };

    __int64 FindTotalItersUsed(void);

private: // Unit conversion helpers
    HighPrecision XFromScreenToCalc(HighPrecision x);
    HighPrecision YFromScreenToCalc(HighPrecision y);
    HighPrecision XFromCalcToScreen(HighPrecision x);
    HighPrecision YFromCalcToScreen(HighPrecision y);

public: // Used for retrieving our current location
    inline HighPrecision GetMinX(void) { return m_MinX; }
    inline HighPrecision GetMaxX(void) { return m_MaxX; }
    inline HighPrecision GetMinY(void) { return m_MinY; }
    inline HighPrecision GetMaxY(void) { return m_MaxY; }
    inline size_t GetRenderWidth(void) { return m_ScrnWidth; }
    inline size_t GetRenderHeight(void) { return m_ScrnHeight; }

    // Networking functions.
private:
    void NetworkCreateWorkload(void);
    void ClientInitializeServers(void);
    void ClientCreateSubConnections(void);
    void ClientShutdownSubConnections(void);
    void ClientSendQuitMessages(void);

    bool ServerRespondToInitialization(void);
    void ServerReadProcessPixelRow(void);
    void ServerReadID(void);
    void ServerManageMainConnection(void);
    int ServerManageMainState(void);
    void ServerManageSubConnection(void);
    bool ServerBeginCalc(void);

public:
    static DWORD WINAPI ServerManageMainConnectionThread(void *);
    static DWORD WINAPI ServerManageSubConnectionThread(void *);

    // Member Variables
private:
    // Holds some customizations the user can make.  Saves/Loads from disk
    FractalSetupData m_SetupData;

    // Defaults
    static constexpr size_t DefaultIterations = 256 * 32;
    //static constexpr size_t DefaultIterations = 256;

    // Handle to the thread which checks to see if we should quit or not
    HANDLE m_CheckForAbortThread;
    volatile bool m_AbortThreadQuitFlag;
    bool m_UseSensoCursor;
    HWND m_hWnd;
    volatile bool m_StopCalculating;

    // Holds all previous positions within the fractal.
    // Allows us to go "back."
    std::vector<HighPrecision> m_PrevMinX;
    std::vector<HighPrecision> m_PrevMinY;
    std::vector<HighPrecision> m_PrevMaxX;
    std::vector<HighPrecision> m_PrevMaxY;

    // Describes the exact spot within the fractal
    // we are currently looking at.
    HighPrecision m_MinX, m_MaxX;
    HighPrecision m_MinY, m_MaxY;
    bool m_ChangedWindow;

    // Holds the dimensions of the window onto which we are drawing.
    size_t m_ScrnWidth, m_ScrnHeight;
    bool m_ChangedScrn;

    // The maximum number of iterations to go through on a pixel
    // in an effort to determine whether the point is within the set
    // or not.
    size_t m_NumIterations;
    bool m_ChangedIterations;

    // Antialiasing;
    // 1 = none
    // 2 = 2x2 grid average
    // 3 = 3x3 grid average
    // 4 = 4x4 grid average

    // m_IterationAntialiasing = average iteration count
    // m_GpuAntialiasing = average final colors picked
    uint32_t m_IterationAntialiasing;
    uint32_t m_GpuAntialiasing;

    // Iteration precision
    // 1 = precise
    // 2 = unroll inner mandel loop, so iterations are multiples of 2.
    // 3 = unroll more, multiples of 3.
    // higher = faster but less precise iteration count
    uint32_t m_IterationPrecision;

    // Render algorithm
    RenderAlgorithm m_RenderAlgorithm;

    // The palette!
    std::vector<uint16_t> m_PalR[3], m_PalG[3], m_PalB[3];
    int m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepth; // 8, 12, 16
    int m_PaletteDepthIndex; // 0, 1, 2
    static constexpr int NumPalettes = 3;

    // Holds the number of iterations it took to decide if
    // we were in or not in the fractal.  Has a number
    // for every point on the screen.
    struct ItersMemoryContainer {
        ItersMemoryContainer(size_t width, size_t height, size_t total_antialiasing);
        ItersMemoryContainer(ItersMemoryContainer&&);
        ItersMemoryContainer& operator=(ItersMemoryContainer&&);
        ~ItersMemoryContainer();

        ItersMemoryContainer(ItersMemoryContainer&) = delete;
        ItersMemoryContainer& operator=(const ItersMemoryContainer&) = delete;

        uint32_t* m_ItersMemory;
        uint32_t** m_ItersArray;

        // These are a bit bigger than m_ScrnWidth / m_ScrnHeight!
        size_t m_Width;
        size_t m_Height;
        size_t m_Total;
    };

    void InitializeMemory();
    void GetIterMemory();
    void ReturnIterMemory(ItersMemoryContainer&& to_return);

    ItersMemoryContainer m_CurIters;
    std::mutex m_ItersMemoryStorageLock;
    std::vector<struct ItersMemoryContainer> m_ItersMemoryStorage;

    static constexpr size_t BrokenMaxFractalSize = 1; // TODO
    BYTE m_ProcessPixelRow[BrokenMaxFractalSize];

    // Network member variables
    FractalNetwork *m_ClientMainNetwork[MAXSERVERS], *m_ClientSubNetwork[MAXSERVERS];
    FractalNetwork *m_ServerMainNetwork, *m_ServerSubNetwork;
    char m_NetworkRender;
    HANDLE m_ServerMainThread;
    HANDLE m_ServerSubThread;
    void(*OutputMessage) (const wchar_t *, ...);

    // GPU rendering
    GPURenderer m_r;
    void MessageBoxCudaError(uint32_t err);

    // 
    struct CurrentFractalSave {
        CurrentFractalSave(std::wstring filename_base, Fractal& fractal);
        ~CurrentFractalSave();
        void Run();
        void StartThread();

        CurrentFractalSave(CurrentFractalSave&&) = default;

        Fractal &m_Fractal;
        size_t m_ScrnWidth;
        size_t m_ScrnHeight;
        uint32_t m_GpuAntialiasing;
        size_t m_NumIterations;
        int m_PaletteRotate; // Used to shift the palette
        int m_PaletteDepth; // 8, 12, 16
        int m_PaletteDepthIndex; // 0, 1, 2
        std::vector<uint16_t> *m_PalR, *m_PalG, *m_PalB;
        ItersMemoryContainer m_CurIters;
        std::wstring m_FilenameBase;
        std::unique_ptr<std::thread> m_Thread;
        bool m_Destructable;
    };

    std::vector<std::unique_ptr<CurrentFractalSave>> m_FractalSavesInProgress;
};

#endif