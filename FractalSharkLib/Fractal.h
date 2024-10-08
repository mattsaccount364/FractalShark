#pragma once

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


#include "GPU_Render.h"

#include "WPngImage\WPngImage.hh"
#include "ItersMemoryContainer.h"
#include "HighPrecision.h"
#include "HDRFloat.h"
#include "CudaDblflt.h"
#include "OpenGLContext.h"
#include "RefOrbitCalc.h"

#include "PointZoomBBConverter.h"
#include "DrawThreadSync.h"

#include "PngParallelSave.h"
#include "Utilities.h"
#include "BenchmarkDataCollection.h"

#include "LAParameters.h"

#include "RefOrbitDetails.h"

#include <string>
#include <deque>

template<typename IterType, class T, class SubType, PerturbExtras PExtras>
class LAReference;

class Fractal {
public:
    // TODO get rid of this junk:
    friend class PngParallelSave;
    friend class BenchmarkData;

    Fractal(
        int width,
        int height,
        HWND hWnd,
        bool UseSensoCursor,
        uint64_t commitLimitInBytes);
    ~Fractal();

    void InitialDefaultViewAndSettings(int width = 0, int height = 0);

    static unsigned long WINAPI CheckForAbortThread(void *fractal);

    // Kludgy.  Resets at end of function.
    // Roughly 50000 digits of precision (50000 * 3.321)
    //SetPrecision(166050, minX, minY, maxX, maxY);
    static constexpr size_t MaxPrecisionLame = 400000;

    uint64_t GetPrecision(void) const;
    static uint64_t GetPrecision(const PointZoomBBConverter &ptz, bool requiresReuse);
    void SetPrecision();
    void SetPrecision(uint64_t prec);

    void ResetDimensions(size_t width = MAXSIZE_T,
        size_t height = MAXSIZE_T,
        uint32_t gpu_antialiasing = UINT32_MAX);
    bool RecenterViewCalc(const PointZoomBBConverter &ptz);
    bool RecenterViewScreen(RECT rect);
    bool CenterAtPoint(size_t x, size_t y);
    void Zoom(double factor);
    void Zoom(size_t scrnX, size_t scrnY, double factor);
    void Zoom2(size_t scrnX, size_t scrnY, double factor);

    void TestBasic();

    enum class AutoZoomHeuristic {
        Default,
        Max
    };

    template<AutoZoomHeuristic h>
    void AutoZoom();

    void View(size_t i, bool includeMsgBox = true);
    void SquareCurrentView(void);
    void ApproachTarget(void);
    bool Back(void);

    void FindInterestingLocation(RECT *rect);

    template<typename IterType>
    void SetNumIterations(IterTypeFull num);

    template<typename IterType>
    IterType GetNumIterations(void) const;

    template<typename IterType>
    static constexpr IterType GetMaxIterations(void) {
        return ((sizeof(IterType) == 4) ? (INT32_MAX - 1) : (INT64_MAX - 1));
    }

    IterTypeFull GetMaxIterationsRT() const;

    void SetIterType(IterTypeEnum type);
    IterTypeEnum GetIterType() const;
    void ResetNumIterations(void);

    RenderAlgorithm GetRenderAlgorithm(void) const;
    inline void SetRenderAlgorithm(RenderAlgorithm alg);
    const char *GetRenderAlgorithmName() const;
    static const char *GetRenderAlgorithmName(RenderAlgorithm alg);

    enum class CompressionError : size_t {
        Low,
        Intermediate,
        Num
    };

    static constexpr int32_t DefaultCompressionExp[] = {
        20,
        450 // Consider AuthoritativeMinExtraPrecisionInBits.  TODO: 450 is a guess.
    };

    const HighPrecision &GetCompressionError(enum class CompressionError) const;
    int32_t GetCompressionErrorExp(enum class CompressionError) const;
    void IncCompressionError(enum class CompressionError, int32_t amount);
    void DecCompressionError(enum class CompressionError, int32_t amount);
    void SetCompressionErrorExp(enum class CompressionError, int32_t CompressionExp);
    void DefaultCompressionErrorExp(enum class CompressionError);

    inline uint32_t GetGpuAntialiasing(void) const { return m_GpuAntialiasing; }
    inline uint32_t GetIterationPrecision(void) const { return m_IterationPrecision; }
    inline void SetIterationPrecision(uint32_t iteration_precision) { m_IterationPrecision = iteration_precision; }

    HighPrecision GetZoomFactor() const;
    void SetPerturbationAlg(RefOrbitCalc::PerturbationAlg alg);
    void ClearPerturbationResults(RefOrbitCalc::PerturbationResultType type);
    void SavePerturbationOrbits();
    void LoadPerturbationOrbits();

    // Drawing functions
    bool RequiresUseLocalColor() const;
    void CalcFractal(bool MemoryOnly);
    void DrawFractal(bool MemoryOnly);

    template<typename IterType>
    void DrawGlFractal(bool LocalColor, bool LastIter);

    void SetRepaint(bool repaint);
    bool GetRepaint() const;
    void ToggleRepainting();

    // Palette functions
    uint32_t GetPaletteDepthFromIndex(size_t index) const;
    uint32_t GetPaletteDepth() const;
    void UsePalette(int depth);
    void UseNextPaletteDepth();
    void SetPaletteAuxDepth(int32_t aux_depth);
    void UseNextPaletteAuxDepth(int32_t inc);
    void UsePaletteType(FractalPalette type);
    void ResetFractalPalette(void);
    void RotateFractalPalette(int delta);
    void CreateNewFractalPalette(void);

    void DrawAllPerturbationResults(bool LeaveScreen);

    // Saving images of the fractal
    int SaveCurrentFractal(std::wstring filename_base, bool copy_the_iters);
    int SaveHiResFractal(std::wstring filename_base);
    int SaveItersAsText(std::wstring filename_base);
    void SaveRefOrbit(CompressToDisk compression, std::wstring filename) const;
    void DiffRefOrbits(
        CompressToDisk compression,
        std::wstring outFile,
        std::wstring filename1,
        std::wstring filename2) const;

    void LoadRefOrbit(
        RecommendedSettings *oldSettings,
        CompressToDisk compression,
        ImaginaSettings imaginaSettings,
        std::wstring filename);

    bool CleanupThreads(bool all);

    // Benchmark results
    const BenchmarkDataCollection &GetBenchmark() const;

    // Used for retrieving our current location
    const HighPrecision &GetMinX(void) const;
    const HighPrecision &GetMaxX(void) const;
    const HighPrecision &GetMinY(void) const;
    const HighPrecision &GetMaxY(void) const;
    size_t GetRenderWidth(void) const;
    size_t GetRenderHeight(void) const;

    void GetSomeDetails(RefOrbitDetails &details) const {
        m_RefOrbit.GetSomeDetails(details);
    }

    void SetResultsAutosave(AddPointOptions Enable);

    // Unit conversion helpers
    template<bool IncludeGpuAntialiasing = false>
    HighPrecision XFromScreenToCalc(HighPrecision x);

    template<bool IncludeGpuAntialiasing = false>
    HighPrecision YFromScreenToCalc(HighPrecision y);

    HighPrecision XFromCalcToScreen(HighPrecision x) const;
    HighPrecision YFromCalcToScreen(HighPrecision y) const;

    static DWORD WINAPI ServerManageMainConnectionThread(void *);
    static DWORD WINAPI ServerManageSubConnectionThread(void *);

    void ForceRecalc();

    const LAParameters &GetLAParameters() const;
    LAParameters &GetLAParameters();

    void GetRenderDetails(
        std::string &shortStr,
        std::string &longStr) const;

private:
    void Initialize(int width,
        int height,
        HWND hWnd,
        bool UseSensoCursor);
    void Uninitialize(void);
    void PalIncrease(std::vector<uint16_t> &pal, int length, int val1, int val2);
    void PalTransition(size_t WhichPalette, size_t paletteIndex, int length, int r, int g, int b);
    bool IsDownControl(void);
    void CheckForAbort(void);

    void SaveCurPos(void);

    // Keeps track of what has changed and what hasn't since the last draw
    inline void ChangedMakeClean(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = false; }
    inline void ChangedMakeDirty(void) { m_ChangedWindow = m_ChangedScrn = m_ChangedIterations = true; }
    inline bool ChangedIsDirty(void) const { return (m_ChangedWindow || m_ChangedScrn || m_ChangedIterations); }
    inline bool ChangedItersOnly(void) const { return (m_ChangedIterations && !(m_ChangedScrn || m_ChangedWindow)); }

    template<typename IterType>
    void CalcFractalTypedIter(bool MemoryOnly);

    static void DrawFractalThread(size_t index, Fractal *fractal);

    void FillCoord(const HighPrecision &src, MattQFltflt &dest);
    void FillCoord(const HighPrecision &src, MattQDbldbl &dest);
    void FillCoord(const HighPrecision &src, MattDbldbl &dest);
    void FillCoord(const HighPrecision &src, double &dest);
    void FillCoord(const HighPrecision &src, HDRFloat<float> &dest);
    void FillCoord(const HighPrecision &src, HDRFloat<double> &dest);
    void FillCoord(const HighPrecision &src, MattDblflt &dest);
    void FillCoord(const HighPrecision &src, float &dest);
    void FillCoord(const HighPrecision &src, CudaDblflt<MattDblflt> &dest);
    void FillCoord(const HighPrecision &src, HDRFloat<CudaDblflt<MattDblflt>> &dest);

    template<class T>
    void FillGpuCoords(T &cx2, T &cy2, T &dx2, T &dy2);

    template<typename IterType>
    void CalcAutoFractal();

    template<typename IterType, class T>
    void CalcGpuFractal(bool MemoryOnly);

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

    template<typename IterType, class SubType, PerturbExtras PExtras>
    void CalcCpuPerturbationFractalLAV2(bool MemoryOnly);

    template<typename IterType, class T, class SubType>
    void CalcGpuPerturbationFractalBLA(bool MemoryOnly);

    template<typename IterType, typename RenderAlg, PerturbExtras PExtras>
    void CalcGpuPerturbationFractalLAv2(bool MemoryOnly);

    template<typename IterType, class T, class SubType, class T2, class SubType2>
    void CalcGpuPerturbationFractalScaledBLA(bool MemoryOnly);

    template<PngParallelSave::Type Typ>
    int SaveFractalData(const std::wstring filename_base, bool copy_the_iters);

    uint64_t FindTotalItersUsed(void);

    // Member Variables
    RefOrbitCalc m_RefOrbit;

    std::unique_ptr<uint16_t[]> m_DrawOutBytes;
    std::deque<std::atomic_uint64_t> m_DrawThreadAtomics;
    std::vector<std::unique_ptr<DrawThreadSync>> m_DrawThreads;

    std::vector<std::unique_ptr<PngParallelSave>> m_FractalSavesInProgress;

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
    std::vector<PointZoomBBConverter> m_PrevPtz;

    // Describes the exact spot within the fractal
    // we are currently looking at.
    PointZoomBBConverter m_Ptz;
    bool m_ChangedWindow;

    // Holds the dimensions of the window onto which we are drawing.
    size_t m_ScrnWidth, m_ScrnHeight;
    bool m_ChangedScrn;

    // The maximum number of iterations to go through on a pixel
    // in an effort to determine whether the point is within the set
    // or not.
    mutable IterTypeFull m_NumIterations;
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

    std::vector<uint16_t> m_PalR[FractalPalette::Num][NumBitDepths];
    std::vector<uint16_t> m_PalG[FractalPalette::Num][NumBitDepths];
    std::vector<uint16_t> m_PalB[FractalPalette::Num][NumBitDepths];
    std::vector<uint32_t> m_PalIters[FractalPalette::Num];
    FractalPalette m_WhichPalette;

    IterTypeFull m_PaletteRotate; // Used to shift the palette
    int m_PaletteDepthIndex; // 0, 1, 2
    int m_PaletteAuxDepth; // 0..16
    static constexpr int NumPalettes = 3;

    uint32_t InitializeGPUMemory(bool expectedReuse = true);

    void InitializeMemory();
    void SetCurItersMemory();

    void ReturnIterMemory(ItersMemoryContainer &&to_return);

    IterTypeEnum m_IterType;

    ItersMemoryContainer m_CurIters;
    std::mutex m_ItersMemoryStorageLock;
    std::vector<ItersMemoryContainer> m_ItersMemoryStorage;

    static constexpr size_t BrokenMaxFractalSize = 1; // TODO
    BYTE m_ProcessPixelRow[BrokenMaxFractalSize];

    // Reference compression
    int32_t m_CompressionExp[static_cast<size_t>(CompressionError::Num)];
    HighPrecision m_CompressionError[static_cast<size_t>(CompressionError::Num)];

    // GPU rendering
    GPURenderer m_r;
    std::unique_ptr<OpenGlContext> m_glContext;

    std::mutex m_AsyncRenderThreadMutex;
    std::condition_variable m_AsyncRenderThreadCV;

    enum class AsyncRenderThreadState {
        Idle,
        Start,
        SyncDone,
        Finish
    };
    AsyncRenderThreadState m_AsyncRenderThreadState;
    bool m_AsyncRenderThreadFinish;

    // Benchmarking
    mutable BenchmarkDataCollection m_BenchmarkData;

    std::unique_ptr<std::thread> m_AsyncRenderThread;
    std::atomic<uint32_t> m_AsyncGpuRenderIsAtomic;
    void DrawAsyncGpuFractalThread();
    static void DrawAsyncGpuFractalThreadStatic(Fractal *fractal);
    void MessageBoxCudaError(uint32_t err);

    LAParameters m_LAParameters;

    const uint64_t m_CommitLimitInBytes;
};
