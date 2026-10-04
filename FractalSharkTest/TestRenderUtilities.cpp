#include "Exceptions.h"
#include "FloatComplex.h"
#include "Fractal.h"
#include "HDRFloatComplex.h"
#include "RenderTestSupport.h"
#include "RenderThreadPool.h"
#include "TestFramework.h"
#include "Vectors.h"
#include <sstream>

TEST(RenderRegression_StringConversion)
{
    double double1 = 5.5555;
    double double2 = 6.6666;
    float float1 = 15.5555f;
    float float2 = 16.6666f;
    float float3 = 17.7777f;
    float float4 = 18.8888f;
    int32_t testExp = 5;
    int32_t testHdrExp = 1234;

    HDRFloat<float> f1(testHdrExp, float1);
    HDRFloat<double> d1(testHdrExp, double1);
    HDRFloat<CudaDblflt<MattDblflt>> m1(testExp, CudaDblflt<MattDblflt>{float1, float2});
    HDRFloat<CudaDblflt<dblflt>> c1(testExp, CudaDblflt<MattDblflt>{float3, float4});

    HDRFloatComplex<float> f2(float1, float2, testHdrExp);
    HDRFloatComplex<double> d2(double1, double2, testHdrExp);
    HDRFloatComplex<CudaDblflt<MattDblflt>> m2(
        CudaDblflt<MattDblflt>(float1, float2), CudaDblflt<MattDblflt>(float3, float4), testHdrExp);
    HDRFloatComplex<CudaDblflt<dblflt>> c2(
        CudaDblflt<dblflt>(float1, float2), CudaDblflt<dblflt>(float3, float4), testHdrExp);

    CudaDblflt<MattDblflt> cudaDblflt1(float1, float2);
    CudaDblflt<dblflt> cudaDblflt2(float1, float2);

    FloatComplex<float> floatComplex1(float1, float2);
    FloatComplex<double> floatComplex2(double1, double2);
    FloatComplex<CudaDblflt<MattDblflt>> floatComplex3(CudaDblflt<MattDblflt>(float1, float2),
                                                       CudaDblflt<MattDblflt>(float3, float4));
    FloatComplex<CudaDblflt<dblflt>> floatComplex4(CudaDblflt<dblflt>(float1, float2),
                                                   CudaDblflt<dblflt>(float3, float4));

    std::stringstream is;
    auto toString = [&]<bool IntOut>(auto &hdr) {
        auto ret = std::string("Descriptor: ");
        ret += HdrToString<IntOut>(hdr);
        // append newline to ret
        ret += std::string("\n");

        // append string to stringstream
        is << ret;
        return ret;
    };

    std::string allStr;
    allStr += toString.template operator()<false>(double1);
    allStr += toString.template operator()<false>(double2);
    allStr += toString.template operator()<false>(float1);
    allStr += toString.template operator()<false>(float2);
    allStr += toString.template operator()<false>(float3);
    allStr += toString.template operator()<false>(float4);

    allStr += toString.template operator()<false>(f1);
    allStr += toString.template operator()<false>(d1);
    allStr += toString.template operator()<false>(m1);
    allStr += toString.template operator()<false>(c1);

    allStr += toString.template operator()<false>(f2);
    allStr += toString.template operator()<false>(d2);
    allStr += toString.template operator()<false>(m2);
    allStr += toString.template operator()<false>(c2);

    allStr += toString.template operator()<false>(cudaDblflt1);
    allStr += toString.template operator()<false>(cudaDblflt2);

    allStr += toString.template operator()<false>(floatComplex1);
    allStr += toString.template operator()<false>(floatComplex2);
    allStr += toString.template operator()<false>(floatComplex3);
    allStr += toString.template operator()<false>(floatComplex4);

    allStr += toString.template operator()<true>(double1);
    allStr += toString.template operator()<true>(double2);
    allStr += toString.template operator()<true>(float1);
    allStr += toString.template operator()<true>(float2);
    allStr += toString.template operator()<true>(float3);
    allStr += toString.template operator()<true>(float4);

    allStr += toString.template operator()<true>(f1);
    allStr += toString.template operator()<true>(d1);
    allStr += toString.template operator()<true>(m1);
    allStr += toString.template operator()<true>(c1);
    allStr += toString.template operator()<true>(f2);
    allStr += toString.template operator()<true>(d2);
    allStr += toString.template operator()<true>(m2);
    allStr += toString.template operator()<true>(c2);

    allStr += toString.template operator()<true>(cudaDblflt1);
    allStr += toString.template operator()<true>(cudaDblflt2);

    allStr += toString.template operator()<true>(floatComplex1);
    allStr += toString.template operator()<true>(floatComplex2);
    allStr += toString.template operator()<true>(floatComplex3);
    allStr += toString.template operator()<true>(floatComplex4);

    // convert stringstream to istream
    std::istringstream iss(is.str());

    // read from istream
    double readBackDouble1 = 0;
    double readBackDouble2 = 0;
    float readBackFloat1 = 0;
    float readBackFloat2 = 0;
    float readBackFloat3 = 0;
    float readBackFloat4 = 0;

    HDRFloat<float> readBackF1;
    HDRFloat<double> readBackD1;
    HDRFloat<CudaDblflt<MattDblflt>> readBackM1;
    HDRFloat<CudaDblflt<dblflt>> readBackC1;

    HDRFloatComplex<float> readBackF2;
    HDRFloatComplex<double> readBackD2;
    HDRFloatComplex<CudaDblflt<MattDblflt>> readBackM2;
    HDRFloatComplex<CudaDblflt<dblflt>> readBackC2;

    CudaDblflt<MattDblflt> readBackCudaDblflt1;
    CudaDblflt<dblflt> readBackCudaDblflt2;

    FloatComplex<float> readBackFloatComplex1;
    FloatComplex<double> readBackFloatComplex2;
    FloatComplex<CudaDblflt<MattDblflt>> readBackFloatComplex3;
    FloatComplex<CudaDblflt<dblflt>> readBackFloatComplex4;

    HdrFromIfStream<false, double, double>(readBackDouble1, iss);
    HdrFromIfStream<false, double, double>(readBackDouble2, iss);
    HdrFromIfStream<false, float, float>(readBackFloat1, iss);
    HdrFromIfStream<false, float, float>(readBackFloat2, iss);
    HdrFromIfStream<false, float, float>(readBackFloat3, iss);
    HdrFromIfStream<false, float, float>(readBackFloat4, iss);

    HdrFromIfStream<false, HDRFloat<float>, float>(readBackF1, iss);
    HdrFromIfStream<false, HDRFloat<double>, double>(readBackD1, iss);
    HdrFromIfStream<false, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM1, iss);
    HdrFromIfStream<false, HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC1, iss);

    HdrFromIfStream<false, HDRFloatComplex<float>, float>(readBackF2, iss);
    HdrFromIfStream<false, HDRFloatComplex<double>, double>(readBackD2, iss);
    HdrFromIfStream<false, HDRFloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM2,
                                                                                            iss);
    HdrFromIfStream<false, HDRFloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC2, iss);

    HdrFromIfStream<false, CudaDblflt<MattDblflt>, MattDblflt>(readBackCudaDblflt1, iss);
    HdrFromIfStream<false, CudaDblflt<dblflt>, dblflt>(readBackCudaDblflt2, iss);

    HdrFromIfStream<false, FloatComplex<float>, float>(readBackFloatComplex1, iss);
    HdrFromIfStream<false, FloatComplex<double>, double>(readBackFloatComplex2, iss);
    HdrFromIfStream<false, FloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(
        readBackFloatComplex3, iss);
    HdrFromIfStream<false, FloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackFloatComplex4,
                                                                                 iss);

    auto checker = [&](auto &a, auto &b) {
        if (a != b) {
            throw FractalSharkSeriousException("String conversion failed!");
        }
    };

    checker(double1, readBackDouble1);
    checker(double2, readBackDouble2);
    checker(float1, readBackFloat1);
    checker(float2, readBackFloat2);
    checker(float3, readBackFloat3);
    checker(float4, readBackFloat4);

    checker(f1, readBackF1);
    checker(d1, readBackD1);
    checker(m1, readBackM1);
    checker(c1, readBackC1);

    checker(f2, readBackF2);
    checker(d2, readBackD2);
    checker(m2, readBackM2);
    checker(c2, readBackC2);

    checker(cudaDblflt1, readBackCudaDblflt1);
    checker(cudaDblflt2, readBackCudaDblflt2);

    checker(floatComplex1, readBackFloatComplex1);
    checker(floatComplex2, readBackFloatComplex2);
    checker(floatComplex3, readBackFloatComplex3);
    checker(floatComplex4, readBackFloatComplex4);

    HdrFromIfStream<true, double, double>(readBackDouble1, iss);
    HdrFromIfStream<true, double, double>(readBackDouble2, iss);
    HdrFromIfStream<true, float, float>(readBackFloat1, iss);
    HdrFromIfStream<true, float, float>(readBackFloat2, iss);
    HdrFromIfStream<true, float, float>(readBackFloat3, iss);
    HdrFromIfStream<true, float, float>(readBackFloat4, iss);

    HdrFromIfStream<true, HDRFloat<float>, float>(readBackF1, iss);
    HdrFromIfStream<true, HDRFloat<double>, double>(readBackD1, iss);
    HdrFromIfStream<true, HDRFloat<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM1, iss);
    HdrFromIfStream<true, HDRFloat<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC1, iss);

    HdrFromIfStream<true, HDRFloatComplex<float>, float>(readBackF2, iss);
    HdrFromIfStream<true, HDRFloatComplex<double>, double>(readBackD2, iss);
    HdrFromIfStream<true, HDRFloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM2,
                                                                                           iss);
    HdrFromIfStream<true, HDRFloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC2, iss);

    HdrFromIfStream<true, CudaDblflt<MattDblflt>, MattDblflt>(readBackCudaDblflt1, iss);
    HdrFromIfStream<true, CudaDblflt<dblflt>, dblflt>(readBackCudaDblflt2, iss);

    HdrFromIfStream<true, FloatComplex<float>, float>(readBackFloatComplex1, iss);
    HdrFromIfStream<true, FloatComplex<double>, double>(readBackFloatComplex2, iss);
    HdrFromIfStream<true, FloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(
        readBackFloatComplex3, iss);
    HdrFromIfStream<true, FloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackFloatComplex4,
                                                                                iss);

    checker(double1, readBackDouble1);
    checker(double2, readBackDouble2);
    checker(float1, readBackFloat1);
    checker(float2, readBackFloat2);
    checker(float3, readBackFloat3);
    checker(float4, readBackFloat4);

    checker(f1, readBackF1);
    checker(d1, readBackD1);
    checker(m1, readBackM1);
    checker(c1, readBackC1);

    checker(f2, readBackF2);
    checker(d2, readBackD2);
    checker(m2, readBackM2);
    checker(c2, readBackC2);

    checker(cudaDblflt1, readBackCudaDblflt1);
    checker(cudaDblflt2, readBackCudaDblflt2);

    checker(floatComplex1, readBackFloatComplex1);
    checker(floatComplex2, readBackFloatComplex2);
    checker(floatComplex3, readBackFloatComplex3);
    checker(floatComplex4, readBackFloatComplex4);
}

namespace {
void
CheckVector(uint64_t count)
{
    auto verifyContents = [](const GrowableVector<uint64_t> &testVector, uint64_t manyElts) {
        for (uint64_t i = 0; i < manyElts; i++) {
            if (testVector[i] != i) {
                throw FractalSharkSeriousException("GrowableVector contents incorrect!");
            }
        }
    };

    constexpr size_t maxInstantiations = 3;
    for (size_t numInstantiations = 0; numInstantiations < maxInstantiations; numInstantiations++) {
        // Defaults to AddPointOptions::DontSave
        GrowableVector<uint64_t> testVector;

        const uint64_t manyElts = count;
        for (uint64_t i = 0; i < manyElts; i++) {
            testVector.PushBack(static_cast<uint64_t>(i));
        }

        // Try moving the vector
        GrowableVector<uint64_t> testVector2(std::move(testVector));

        // Verify contents
        verifyContents(testVector2, manyElts);

        // Try the move assignment operator
        GrowableVector<uint64_t> testVector3;
        testVector3 = std::move(testVector2);

        // Verify contents
        verifyContents(testVector3, manyElts);

        // Verify filename
        std::wstring filename = testVector3.GetFilename();
        if (filename != L"") {
            throw FractalSharkSeriousException("GrowableVector filename incorrect!");
        }

        // Verify GetAddPointOptions
        AddPointOptions options = testVector3.GetAddPointOptions();
        if (options != AddPointOptions::DontSave) {
            throw FractalSharkSeriousException("GrowableVector AddPointOptions incorrect!");
        }

        // Trim
        testVector3.Trim();
        verifyContents(testVector3, manyElts);

        // ValidFile
        if (testVector3.ValidFile() != true) {
            throw FractalSharkSeriousException("GrowableVector ValidFile failed!");
        }
    }
}

void
CheckResize(bool useGpu)
{
    const std::string caseId = useGpu ? "RenderGolden_ResizeGpu" : "RenderGolden_ResizeCpu";
    RenderTests::ScopedDirectory directory{caseId};
    Fractal fractal{
        256, 256, nullptr, false, UINT64_MAX, true, useGpu ? GpuMode::Auto : GpuMode::Disabled};
    fractal.GetRenderPool()->Drain();
    fractal.SetResultsAutosave(AddPointOptions::DontSave);
    ASSERT_TRUE(fractal.SetRenderAlgorithm(GetRenderAlgorithmTupleEntry(
        useGpu ? RenderAlgorithmEnum::Gpu1x32 : RenderAlgorithmEnum::Cpu64)));
    ASSERT_FALSE(useGpu && fractal.GpuBypassed());
    size_t frame = 0;
    auto capture = [&] {
        fractal.ForceRecalc();
        fractal.CalcFractal(true);
        RenderTests::SaveAndCheck(fractal, caseId + "_Frame" + std::to_string(frame++));
    };
    // Pick a sane starting size that should always be valid
    constexpr int baseW = 1280;
    constexpr int baseH = 800;

    const auto initWidth = fractal.GetRenderWidth();
    const auto initHeight = fractal.GetRenderHeight();

    fractal.ResetDimensions(baseW, baseH);

    // Simulate a smooth drag: small deltas, many calls
    int w = baseW;
    int h = baseH;

    // Phase 1: grow slowly
    for (int i = 0; i < 120; ++i) {
        w += (i & 1) ? 1 : 2;
        h += (i & 1) ? 2 : 1;
        fractal.ResetDimensions(w, h);
        capture();
    }

    // Phase 2: jitter + minor reversals (real mouse behavior)
    for (int i = 0; i < 200; ++i) {
        const int dx = (i % 3) - 1; // -1, 0, +1
        const int dy = ((i + 1) % 3) - 1;

        w += dx;
        h += dy;

        // Avoid pathological zero / negative sizes
        if (w < 64)
            w = 64;
        if (h < 64)
            h = 64;

        fractal.ResetDimensions(w, h);
        capture();
    }

    // Phase 3: shrink back down
    for (int i = 0; i < 100; ++i) {
        w -= 2;
        h -= 1;

        if (w < 256)
            w = 256;
        if (h < 256)
            h = 256;

        fractal.ResetDimensions(w, h);
        capture();
    }

    // Phase 4: oscillate around a fixed size
    for (int i = 0; i < 100; ++i) {
        const int wobble = (i & 1) ? 3 : -3;
        fractal.ResetDimensions(w + wobble, h - wobble);
        capture();
    }

    // The final settled dimensions catch cases where an earlier resize wins.
    fractal.ResetDimensions(initWidth, initHeight);
    capture();
    ASSERT_EQ(frame, size_t{521});
}

const bool registrations = [] {
    TestFramework::RegisterCase("RenderGolden_ResizeCpu", [] { CheckResize(false); }, false, "", true);
    TestFramework::RegisterCase("RenderGolden_ResizeGpu", [] { CheckResize(true); }, true, "", true);
    TestFramework::RegisterCase(
        "RenderRegression_VectorBillion",
        [] { CheckVector(1024ull * 1024 * 1024); },
        false,
        "INCOMPLETE: billion-element stress case remains disabled",
        false);
    return true;
}();
} // namespace

TEST(RenderRegression_VectorMoveAndTrim) { CheckVector(1024 * 1024); }
