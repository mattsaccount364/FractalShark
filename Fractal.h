#ifndef FRACTAL_H
#define FRACTAL_H

//
// Comments for Matthew:
// - Re-use information from points in set between frames so that the all-black
//   pixels don't suck up so much time (skip chunks on the interior).
// - Multisample antialiasing, where we look for the "edges" and recalculating
//   those only?
// - Add text box in UI to import a set of coordinates so we can navigate
//   somewhere saved.
//
// Search for TODO
// Make the screen saver render high-res versions of screens that have
//      been saved to a queue.  High res images made at idle time
// Make this code run in a separate thread so it doesn't interfere with the windows
//      message pump.  Make it use callback functions so whoever is using this code
//      can see the progress, be notified when it is done, whatever.


#include "FractalSetupData.h"
#include "FractalNetwork.h"
#include "GPU_Render.h"
#include "PerturbationResults.h"

#include "..\WPngImage\WPngImage.hh"
#include "HighPrecision.h"
#include "HDRFloat.h"

#include "RefOrbitCalc.h"

#include <deque>

template<typename IterType, class SubType>
class LAReference;

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
    void PalIncrease(std::vector<uint16_t>& pal, int length, int val1, int val2);
    void PalTransition(size_t WhichPalette, size_t paletteIndex, int length, int r, int g, int b);

public:
    static unsigned long WINAPI CheckForAbortThread(void *fractal);

    static size_t GetPrecision(
        const HighPrecision& minX,
        const HighPrecision& minY,
        const HighPrecision& maxX,
        const HighPrecision& maxY,
        bool RequiresReuse);
    size_t GetPrecision(void) const;
    static void SetPrecision(
        size_t prec,
        HighPrecision& minX,
        HighPrecision& minY,
        HighPrecision& maxX,
        HighPrecision& maxY);
    void SetPrecision();

private:
    bool IsDownControl(void);
    void CheckForAbort(void);

public: // Changing the view
    void ResetDimensions(size_t width = MAXSIZE_T,
                         size_t height = MAXSIZE_T,
                         uint32_t gpu_antialiasing = UINT32_MAX);
    bool RecenterViewCalc(HighPrecision MinX, HighPrecision MinY, HighPrecision MaxX, HighPrecision MaxY);
    bool RecenterViewScreen(RECT rect);
    bool CenterAtPoint(size_t x, size_t y);
    void Zoom(double factor);
    void Zoom(size_t scrnX, size_t scrnY, double factor);

    enum class AutoZoomHeuristic {
        Default,
        Max
    };

    template<AutoZoomHeuristic h>
    void AutoZoom();

    struct PointZoomBBConverter {
        PointZoomBBConverter(
            HighPrecision ptX,
            HighPrecision ptY,
            HighPrecision zoomFactor);

        PointZoomBBConverter(
            HighPrecision minX,
            HighPrecision minY,
            HighPrecision maxX,
            HighPrecision maxY);

        HighPrecision ptX, ptY;
        HighPrecision zoomFactor;

        HighPrecision minX, minY;
        HighPrecision maxX, maxY;
    };

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
    template<typename IterType>
    void SetNumIterations(IterTypeFull num);

    template<typename IterType>
    IterType GetNumIterations(void) const;

    template<typename IterType>
    constexpr IterType GetMaxIterations() const;

    enum class IterTypeEnum {
        Bits32,
        Bits64
    };

    IterTypeFull GetMaxIterationsRT() const;

    void SetIterType(IterTypeEnum type);
    IterTypeEnum GetIterType() const;

    void ResetNumIterations(void);

public:
    inline RenderAlgorithm GetRenderAlgorithm(void) const { return m_RenderAlgorithm; }
    inline void SetRenderAlgorithm(RenderAlgorithm alg) { m_RenderAlgorithm = alg; }

    inline uint32_t GetGpuAntialiasing(void) const { return m_GpuAntialiasing; }
    inline uint32_t GetIterationPrecision(void) const { return m_IterationPrecision; }
    inline void SetIterationPrecision(uint32_t iteration_precision) { m_IterationPrecision = iteration_precision;  }

    void SetPerturbationAlg(RefOrbitCalc::PerturbationAlg alg) { m_RefOrbit.SetPerturbationAlg(alg); }
    void ClearPerturbationResults(RefOrbitCalc::PerturbationResultType type) { m_RefOrbit.ClearPerturbationResults(type); }

private: // Keeps track of what has changed and what hasn't since the last draw
    inline void ChangedMakeClean(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = false; }
    inline void ChangedMakeDirty(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = true; }
    inline bool ChangedIsDirty(void) const { return (m_ChangedWindow || m_ChangedScrn || m_ChangedIterations); }
    inline bool ChangedItersOnly(void) const { return (m_ChangedIterations && !(m_ChangedScrn || m_ChangedWindow)); }

public: // Drawing functions
    bool RequiresUseLocalColor() const;
    void CalcFractal(bool MemoryOnly);
    void DrawFractal(bool MemoryOnly);

    // The palette!
    enum Palette : size_t {
        Basic = 0,
        Default,
        Patriotic,
        Summer,
        Num
    };

    int GetPaletteDepthFromIndex(size_t index) const;
    int GetPaletteDepth() const; 
    void UsePalette(int depth);
    void UseNextPaletteDepth();
    void UsePaletteType(Palette type);
    void ResetFractalPalette(void);
    void RotateFractalPalette(int delta);
    void CreateNewFractalPalette(void);

    template<typename IterType, class T, CalcBad Bad>
    void DrawPerturbationResults(bool MemoryOnly);
    void DrawAllPerturbationResults();

private:
    template<typename IterType>
    void CalcFractalTypedIter(bool MemoryOnly);

    static void DrawFractalThread(size_t index, Fractal* fractal);

    RefOrbitCalc m_RefOrbit;
    std::unique_ptr<uint16_t[]> m_DrawOutBytes;
    std::deque<std::atomic_uint64_t> m_DrawThreadAtomics;
    struct DrawThreadSync {
        //DrawThreadSync& operator=(const DrawThreadSync&) = delete;
        //DrawThreadSync(const DrawThreadSync&) = delete;
        DrawThreadSync(
            size_t index,
            std::unique_ptr<std::thread> thread,
            std::deque<std::atomic_uint64_t>& draw_thread_atomics
        ) :
            m_Index(index),
            m_Thread(std::move(thread)),
            m_DrawThreadAtomics(draw_thread_atomics),
            m_DrawThreadReady{},
            m_DrawThreadProcessed{},
            m_TimeToExit{}
        {
        }

        size_t m_Index;
        std::mutex m_DrawThreadMutex;
        std::condition_variable m_DrawThreadCV;
        std::unique_ptr<std::thread> m_Thread;
        std::deque<std::atomic_uint64_t> &m_DrawThreadAtomics;
        bool m_DrawThreadReady;
        bool m_DrawThreadProcessed;
        bool m_TimeToExit;
    };

    std::vector<std::unique_ptr<DrawThreadSync>> m_DrawThreads;

    void FillCoord(HighPrecision& src, MattQFltflt& dest);
    void FillCoord(HighPrecision& src, MattQDbldbl& dest);
    void FillCoord(HighPrecision& src, MattDbldbl& dest);
    void FillCoord(HighPrecision& src, double& dest);
    void FillCoord(HighPrecision& src, HDRFloat<float>& dest);
    void FillCoord(HighPrecision& src, HDRFloat<double>& dest);
    void FillCoord(HighPrecision& src, MattDblflt& dest);
    void FillCoord(HighPrecision& src, float& dest);

    template<class T>
    void FillGpuCoords(T& cx2, T& cy2, T& dx2, T& dy2);

    template<typename IterType, class T>
    void CalcGpuFractal(bool MemoryOnly);

    void CalcNetworkFractal(bool MemoryOnly);

    template<typename IterType>
    void CalcCpuPerturbationFractal(bool MemoryOnly);

    template<typename IterType, class T, class SubType>
    void CalcCpuHDR(bool MemoryOnly);

    template<typename IterType, class T, class SubType>
    void CalcCpuPerturbationFractalBLA(bool MemoryOnly);

    //HDRFloatComplex<float>* initializeFromBLA2(
    //    LAReference& laReference,
    //    HDRFloatComplex<float> dpixel,
    //    IterType& BLA2SkippedIterations,
    //    IterType& BLA2SkippedSteps);

    template<typename IterType, class SubType>
    void CalcCpuPerturbationFractalLAV2(bool MemoryOnly);

    template<typename IterType, class T, class SubType, bool BLA>
    void CalcGpuPerturbationFractalBLA(bool MemoryOnly);

    template<typename IterType, class T, class SubType, LAv2Mode Mode>
    void CalcGpuPerturbationFractalLAv2(bool MemoryOnly);

    template<typename IterType, class T, class SubType, class T2, class SubType2>
    void CalcGpuPerturbationFractalScaledBLA(bool MemoryOnly);

public: // Saving images of the fractal
    int SaveCurrentFractal(const std::wstring filename_base);
    int SaveHiResFractal(const std::wstring filename_base);
    int SaveItersAsText(const std::wstring filename_base);
    void CleanupThreads(bool all);

private:
    // Holds the number of iterations it took to decide if
    // we were in or not in the fractal.  Has a number
    // for every point on the screen.
    struct ItersMemoryContainer {
        ItersMemoryContainer(
            IterTypeEnum type,
            size_t width,
            size_t height,
            size_t total_antialiasing);
        ItersMemoryContainer(ItersMemoryContainer&&) noexcept;
        ItersMemoryContainer& operator=(ItersMemoryContainer&&) noexcept;
        ~ItersMemoryContainer();

        ItersMemoryContainer(ItersMemoryContainer&) = delete;
        ItersMemoryContainer& operator=(const ItersMemoryContainer&) = delete;

        template<typename IterType>
        IterType* GetIters() {
            if constexpr (std::is_same<IterType, uint32_t>::value) {
                return m_ItersMemory32.get();
            }
            else {
                return m_ItersMemory64.get();
            }
        }

        template<typename IterType>
        IterType** GetItersArray() {
            if constexpr (sizeof(IterType) == sizeof(uint32_t)) {
                return m_ItersArray32;
            }
            else {
                return m_ItersArray64;
            }
        }

        IterTypeFull GetItersArrayValSlow(size_t x, size_t y) {
            if (m_IterType == IterTypeEnum::Bits32) {
                return m_ItersArray32[y][x];
            }
            else {
                return m_ItersArray64[y][x];
            }
        }

        void SetItersArrayValSlow(size_t x, size_t y, uint64_t val) {
            if (m_IterType == IterTypeEnum::Bits32) {
                m_ItersArray32[y][x] = (uint32_t)val;
            }
            else {
                m_ItersArray64[y][x] = val;
            }
        }

        // These include antialiasing, so 4x antialiasing implies each is ~2x screen dimension
        size_t m_Width;
        size_t m_Height;
        size_t m_Total;

        // These are the originally-input desired dimensions
        size_t m_OutputWidth;
        size_t m_OutputHeight;
        size_t m_OutputTotal;

        // These are a bit bigger than m_ScrnWidth / m_ScrnHeight, and increased
        // to account for AA.
        size_t m_RoundedWidth;
        size_t m_RoundedHeight;
        size_t m_RoundedTotal;

        // Also a bit bigger, but much closer to actual screen size.  These sizes
        // are independent of antialiasing.
        size_t m_RoundedOutputColorWidth;
        size_t m_RoundedOutputColorHeight;
        size_t m_RoundedOutputColorTotal;
        std::unique_ptr<Color16[]> m_RoundedOutputColorMemory;

        // Antialiasing for reference: 1 (1x), 2 (4x), 3 (9x), 4 (16x)
        size_t m_Antialiasing;

    private:
        IterTypeEnum m_IterType;

        std::unique_ptr<uint32_t[]> m_ItersMemory32;
        uint32_t** m_ItersArray32;

        std::unique_ptr<uint64_t[]> m_ItersMemory64;
        uint64_t** m_ItersArray64;

    };

    struct CurrentFractalSave {
        enum class Type {
            ItersText,
            PngImg
        };

        CurrentFractalSave(enum Type typ, std::wstring filename_base, Fractal& fractal);
        ~CurrentFractalSave();
        void Run();
        void StartThread();

        CurrentFractalSave(CurrentFractalSave&&) = default;

        Type m_Type;
        Fractal& m_Fractal;
        size_t m_ScrnWidth;
        size_t m_ScrnHeight;
        uint32_t m_GpuAntialiasing;
        IterTypeFull m_NumIterations;
        IterTypeFull m_PaletteRotate; // Used to shift the palette
        int m_PaletteDepthIndex; // 0, 1, 2
        std::vector<uint16_t>* m_PalR[Fractal::Palette::Num], * m_PalG[Fractal::Palette::Num], * m_PalB[Fractal::Palette::Num];
        Fractal::Palette m_WhichPalette;
        std::vector<uint32_t> m_PalIters[Fractal::Palette::Num];
        ItersMemoryContainer m_CurIters;
        std::wstring m_FilenameBase;
        std::unique_ptr<std::thread> m_Thread;
        bool m_Destructable;
    };

    std::vector<std::unique_ptr<CurrentFractalSave>> m_FractalSavesInProgress;

    template<CurrentFractalSave::Type Typ>
    int SaveFractalData(const std::wstring filename_base);

public: // Benchmarking
 
    HighPrecision Benchmark(IterTypeFull numIters, size_t &millseconds);

    template<class T, class SubType>
    HighPrecision BenchmarkReferencePoint(IterTypeFull numIters, size_t& millseconds);
    HighPrecision BenchmarkThis(size_t& millseconds);

private:
    struct BenchmarkData {
        BenchmarkData(Fractal& fractal);
        Fractal& fractal;

        LARGE_INTEGER freq;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

        size_t prevScrnWidth;
            size_t prevScrnHeight;

        void BenchmarkSetup(IterTypeFull numIters);
        bool StartTimer();
        HighPrecision StopTimer(size_t &milliseconds);

        template<class T>
        HighPrecision StopTimerNoIters(size_t &milliseconds);
        void BenchmarkFinish();
    };

    uint64_t FindTotalItersUsed(void);

private: // Unit conversion helpers
    template<bool IncludeGpuAntialiasing = false>
    HighPrecision XFromScreenToCalc(HighPrecision x);

    template<bool IncludeGpuAntialiasing = false>
    HighPrecision YFromScreenToCalc(HighPrecision y);

    HighPrecision XFromCalcToScreen(HighPrecision x);
    HighPrecision YFromCalcToScreen(HighPrecision y);

public: // Used for retrieving our current location
    inline const HighPrecision  &GetMinX(void) const { return m_MinX; }
    inline const HighPrecision  &GetMaxX(void) const { return m_MaxX; }
    inline const HighPrecision  &GetMinY(void) const { return m_MinY; }
    inline const HighPrecision  &GetMaxY(void) const { return m_MaxY; }
    inline size_t GetRenderWidth(void) const { return m_ScrnWidth; }
    inline size_t GetRenderHeight(void) const { return m_ScrnHeight; }

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
    static constexpr IterTypeFull DefaultIterations = 256 * 32;
    //static constexpr IterType DefaultIterations = 256;

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
    IterTypeFull m_NumIterations;
    bool m_ChangedIterations;

    // Antialiasing;
    // 1 = none
    // 2 = 2x2 grid average
    // 3 = 3x3 grid average
    // 4 = 4x4 grid average

    // m_GpuAntialiasing = average final colors picked
    uint32_t m_GpuAntialiasing;

    // Iteration precision
    // 1 = precise
    // 2 = unroll inner mandel loop, so iterations are multiples of 2.
    // 3 = unroll more, multiples of 3.
    // higher = faster but less precise iteration count
    uint32_t m_IterationPrecision;

    // Render algorithm
    RenderAlgorithm m_RenderAlgorithm;

    static constexpr const size_t NumBitDepths = 6;

    std::vector<uint16_t> m_PalR[Palette::Num][NumBitDepths];
    std::vector<uint16_t> m_PalG[Palette::Num][NumBitDepths];
    std::vector<uint16_t> m_PalB[Palette::Num][NumBitDepths];
    std::vector<uint32_t> m_PalIters[Palette::Num];
    enum Palette m_WhichPalette;

    IterTypeFull m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepthIndex; // 0, 1, 2
    static constexpr int NumPalettes = 3;

    uint32_t InitializeGPUMemory();
    void InitializeMemory();
    void GetIterMemory();

    void ReturnIterMemory(ItersMemoryContainer&& to_return);

    IterTypeEnum m_IterType;

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
};

#endif