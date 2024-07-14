#include "stdafx.h"
#include "FractalTest.h"

#include "Exceptions.h"
#include "Fractal.h"
#include "HDRFloat.h"

#include <shellapi.h>

FractalTest::FractalTest(Fractal &fractal) : m_Fractal(fractal) {
}

void FractalTest::TestPreReq(const wchar_t *dirName) {
    DWORD ftyp = GetFileAttributesW(dirName);
    if (ftyp != INVALID_FILE_ATTRIBUTES && (ftyp & FILE_ATTRIBUTE_DIRECTORY)) {
        SHFILEOPSTRUCT fileOp = { 0 };
        fileOp.wFunc = FO_DELETE;
        fileOp.pFrom = dirName;
        fileOp.fFlags = FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT;
        auto shRet = SHFileOperation(&fileOp);
        if (shRet != 0) {
            ::MessageBox(nullptr, L"Error deleting directory!", L"", MB_OK | MB_APPLMODAL);
            return;
        }
    }

    // Create the directory
    auto ret = CreateDirectory(dirName, nullptr);
    if (ret == 0 && GetLastError() != ERROR_ALREADY_EXISTS) {
        ::MessageBox(nullptr, L"Error creating directory!", L"", MB_OK | MB_APPLMODAL);
        return;
    }
}

void FractalTest::BasicTestInternal(
    const wchar_t *dirName,
    size_t &testIndex,
    IterTypeEnum iterType) {

    constexpr bool IncludeSlow = false;

    // First, iterate over all the supported RenderAlgorithm entries and render the default view:
    // Skip AUTO plus all LAO-only algorithms.  They produce a black screen for the default view.
    for (size_t i = 0; i < (size_t)RenderAlgorithm::AUTO; i++) {
        auto curAlg = static_cast<RenderAlgorithm>(i);
        if (curAlg != RenderAlgorithm::Gpu1x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu2x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu1x64PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO &&

            curAlg != RenderAlgorithm::Gpu1x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu2x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO &&
            curAlg != RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO) {
            BasicOneTest(0, testIndex, dirName, L"View0", curAlg, iterType);
            ++testIndex;
        }
    }

    // Now, iterate over all the RenderAlgorithm entries that should work with View #5.
    RenderAlgorithm View5Algs[] = {
        RenderAlgorithm::Gpu1x32PerturbedScaled,
        RenderAlgorithm::Gpu2x32PerturbedScaled,
        RenderAlgorithm::GpuHDRx32PerturbedScaled,

        RenderAlgorithm::Gpu1x64PerturbedBLA,
        RenderAlgorithm::GpuHDRx32PerturbedBLA,
        RenderAlgorithm::GpuHDRx64PerturbedBLA,

        RenderAlgorithm::Gpu1x64PerturbedLAv2,
        RenderAlgorithm::Gpu1x64PerturbedLAv2PO,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO,

        RenderAlgorithm::Gpu1x64PerturbedLAv2LAO,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
    };

    for (auto curAlg : View5Algs) {
        BasicOneTest(5, testIndex, dirName, L"View5", curAlg, iterType);
        ++testIndex;
    }

    if constexpr (IncludeSlow) {
        // This one is quite slow.  Be advised.
        RenderAlgorithm View10Algs[] = {
            RenderAlgorithm::GpuHDRx32PerturbedLAv2,
            RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
            RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
            RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
            RenderAlgorithm::GpuHDRx64PerturbedLAv2,
            RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,

            RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
            RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
            RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
            RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
        };

        for (auto curAlg : View10Algs) {
            BasicOneTest(10, testIndex, dirName, L"View10", curAlg, iterType);
            ++testIndex;
        }
    }

    // Finally, iterate over all the RenderAlgorithm entries that should work with View #11.
    RenderAlgorithm View11Algs[] = {
        RenderAlgorithm::GpuHDRx32PerturbedScaled,

        RenderAlgorithm::GpuHDRx32PerturbedBLA,
        RenderAlgorithm::GpuHDRx64PerturbedBLA,

        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2PO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2PO,

        RenderAlgorithm::GpuHDRx32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2LAO,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2LAO,
    };

    for (auto curAlg : View11Algs) {
        BasicOneTest(11, testIndex, dirName, L"View11", curAlg, iterType);
        ++testIndex;
    }
}

std::string FractalTest::GenFilename(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm algToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    std::string baseName) {

    auto iterTypeStr =
        IterTypeEnum::Bits32 == iterType ? "32" : "64";

    auto compErrStr = compressionError >= 0 ? std::string(" - CompErr#") + std::to_string(compressionError) : "";

    auto nameWithIntPrefix =
        std::to_string(testIndex) +
        " - View#" + std::to_string(viewIndex) +
        " - Alg#" + std::to_string(static_cast<size_t>(algToTest)) +
        compErrStr +
        " - Bits# " + iterTypeStr +
        " - " + baseName;
    return nameWithIntPrefix;
}

std::wstring FractalTest::GenFilenameW(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm algToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    const wchar_t *testPrefix,
    const wchar_t *dirName,
    std::string baseName) {

    auto nameWithIntPrefix = GenFilename(
        testIndex,
        viewIndex,
        algToTest,
        compressionError,
        iterType,
        baseName);

    std::wstring filenameW;
    std::transform(
        nameWithIntPrefix.begin(),
        nameWithIntPrefix.end(),
        std::back_inserter(filenameW), [](char c) {
            return (wchar_t)c;
        });

    filenameW = std::wstring(dirName) + L"\\" + std::wstring(testPrefix) + L" - " + filenameW;

    return filenameW;
}

void FractalTest::BasicOneTest(
    size_t viewIndex,
    size_t testIndex,
    const wchar_t *dirName,
    const wchar_t *testPrefix,
    RenderAlgorithm algToTest,
    IterTypeEnum iterType) {

    m_Fractal.SetRenderAlgorithm(algToTest);
    m_Fractal.View(viewIndex);
    m_Fractal.ForceRecalc();

    auto name = m_Fractal.GetRenderAlgorithmName();
    m_Fractal.CalcFractal(false);

    const auto filenameW = GenFilenameW(
        testIndex,
        viewIndex,
        algToTest,
        -1,
        iterType,
        testPrefix,
        dirName,
        name);

    m_Fractal.SaveCurrentFractal(filenameW, false);
    //SaveItersAsText(filenameW);
}

void FractalTest::TestBasic() {
    const wchar_t *dirName = L"TestBasic";
    TestPreReq(dirName);

    m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

    size_t testIndex = 0;

    auto lambda = [&]() {
        IterTypeEnum curIterType = IterTypeEnum::Bits32;
        m_Fractal.SetIterType(curIterType);
        BasicTestInternal(dirName, testIndex, curIterType);

        curIterType = IterTypeEnum::Bits64;
        m_Fractal.SetIterType(curIterType);
        BasicTestInternal(dirName, testIndex, curIterType);
        };

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);
    lambda();

    m_Fractal.SetResultsAutosave(AddPointOptions::EnableWithoutSave);
    lambda();

    m_Fractal.SavePerturbationOrbits();

    m_Fractal.InitialDefaultViewAndSettings();
    m_Fractal.CalcFractal(false);
}

void FractalTest::ReferenceSaveLoad (
    Fractal &fractal,
    const wchar_t *dirName,
    size_t viewIndex,
    size_t testIndex,
    IterTypeEnum iterType,
    RenderAlgorithm algToTest,
    int32_t compressionError) {

    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

    fractal.SetIterType(iterType);
    fractal.SetRenderAlgorithm(algToTest);
    fractal.View(viewIndex);
    fractal.ForceRecalc();
    fractal.CalcFractal(false);

    const wchar_t *testPrefix = L"RefOrbitTest";
    const auto algStr = fractal.GetRenderAlgorithmName(algToTest);

    const auto genLocalFilename = [&](const std::wstring extraPrefix) {
        std::wstring fullPrefix = testPrefix + std::wstring(L" - ") + extraPrefix;
        return GenFilenameW(
            testIndex,
            viewIndex,
            algToTest,
            compressionError,
            iterType,
            fullPrefix.c_str(),
            dirName,
            algStr);
        };

    const auto expectedIterations = fractal.GetNumIterations<IterTypeFull>();

    if (compressionError >= 0) {
        fractal.SetCompressionErrorExp(Fractal::CompressionError::Low, compressionError);
    }

    // Only save view 5 since that's small enough to be reasonable.
    // Running this test on all views would take a long time and
    // consume substantial disk space.
    if (viewIndex == 5) {
        const auto disableFilename = genLocalFilename(L"Disable") + L".txt";
        fractal.SaveRefOrbit(CompressToDisk::Disable, disableFilename);

        const auto simpleFilename = genLocalFilename(L"Simple") + L".txt";
        fractal.SaveRefOrbit(CompressToDisk::SimpleCompression, simpleFilename);
    }

    const auto maxFilename = genLocalFilename(L"Max") + L".txt";
    fractal.SaveRefOrbit(CompressToDisk::MaxCompression, maxFilename);

    const auto maxImaginaFilename = genLocalFilename(L"MaxImagina") + L".im";
    fractal.SaveRefOrbit(CompressToDisk::MaxCompressionImagina, maxImaginaFilename);

    const auto originalImageFilename = genLocalFilename(L"Original");
    fractal.SaveCurrentFractal(originalImageFilename, false);

    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    fractal.View(0);
    fractal.ForceRecalc();
    fractal.CalcFractal(false);

    // Force 64-bit on load
    fractal.SetIterType(IterTypeEnum::Bits64);

    fractal.LoadRefOrbit(CompressToDisk::MaxCompressionImagina, maxImaginaFilename);
    if (fractal.GetNumIterations<IterTypeFull>() != expectedIterations) {
        throw FractalSharkSeriousException("LoadRefOrbit failed to set the correct number of iterations!");
    }

    fractal.CalcFractal(false);

    const auto decompressedResultFilename = genLocalFilename(L"Decompressed");
    fractal.SaveCurrentFractal(decompressedResultFilename, false);
}

void FractalTest::TestReferenceSave() {

    const wchar_t *dirName = L"TestReferenceSave";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);

    size_t testIndex = 0;

    m_Fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    int32_t compressionError = m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low);

    auto loopAll = [&](size_t view, std::vector<RenderAlgorithm> algsToTest) {
        for (auto curAlg : algsToTest) {
            ReferenceSaveLoad(
                m_Fractal,
                dirName,
                view,
                testIndex,
                IterTypeEnum::Bits64,
                curAlg,
                compressionError);

            testIndex++;
        }
    };

    size_t viewToTest;

    const auto View5and10Algs = {
        RenderAlgorithm::Gpu1x64PerturbedLAv2,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
    };

    viewToTest = 5;
    loopAll(viewToTest, View5and10Algs);

    viewToTest = 10;
    loopAll(viewToTest, View5and10Algs);

    const auto View13and14Algs = {
        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
    };

    viewToTest = 13;
    loopAll(viewToTest, View13and14Algs);

    viewToTest = 14;
    loopAll(viewToTest, View13and14Algs);
}

void FractalTest::TestVariedCompression() {
    const wchar_t *dirName = L"TestVariedCompression";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);

    size_t testIndex = 0;

    m_Fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    int32_t compressionError = m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low);

    auto loopAll = [&](size_t view, std::vector<RenderAlgorithm> algsToTest) {
        for (auto curAlg : algsToTest) {
            for (int32_t compressionErrorExp = 1; compressionErrorExp <= compressionError; compressionErrorExp++) {
                ReferenceSaveLoad(
                    m_Fractal,
                    dirName,
                    view,
                    testIndex,
                    IterTypeEnum::Bits64,
                    curAlg,
                    compressionErrorExp);

                testIndex++;
            }
        }
        };

    size_t viewToTest;

    const auto View5Algs = {
        RenderAlgorithm::Gpu1x64PerturbedLAv2,
        RenderAlgorithm::Gpu1x64PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedLAv2,
        RenderAlgorithm::GpuHDRx32PerturbedRCLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedLAv2,
        //RenderAlgorithm::GpuHDRx2x32PerturbedRCLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedLAv2,
        RenderAlgorithm::GpuHDRx64PerturbedRCLAv2,
    };

    viewToTest = 5;
    loopAll(viewToTest, View5Algs);
}

void FractalTest::TestStringConversion() {
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
    HDRFloatComplex<CudaDblflt<MattDblflt>> m2(CudaDblflt<MattDblflt>(float1, float2), CudaDblflt<MattDblflt>(float3, float4), testHdrExp);
    HDRFloatComplex<CudaDblflt<dblflt>> c2(CudaDblflt<dblflt>(float1, float2), CudaDblflt<dblflt>(float3, float4), testHdrExp);

    CudaDblflt<MattDblflt> cudaDblflt1(float1, float2);
    CudaDblflt<dblflt> cudaDblflt2(float1, float2);

    FloatComplex<float> floatComplex1(float1, float2);
    FloatComplex<double> floatComplex2(double1, double2);
    FloatComplex<CudaDblflt<MattDblflt>> floatComplex3(CudaDblflt<MattDblflt>(float1, float2), CudaDblflt<MattDblflt>(float3, float4));
    FloatComplex<CudaDblflt<dblflt>> floatComplex4(CudaDblflt<dblflt>(float1, float2), CudaDblflt<dblflt>(float3, float4));

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

    // Concatenate all the strings together to prevent the compiler from optimizing them away.
    //::MessageBoxA(nullptr, allStr.c_str(), "String conversion test", MB_OK | MB_APPLMODAL);

    {
        // Copy to clipboard
        if (!OpenClipboard(nullptr)) {
            throw FractalSharkSeriousException("OpenClipboard failed!");
        }

        if (!EmptyClipboard()) {
            throw FractalSharkSeriousException("EmptyClipboard failed!");
        }

        HGLOBAL hg = GlobalAlloc(GMEM_MOVEABLE, allStr.size() + 1);
        if (!hg) {
            CloseClipboard();
            throw FractalSharkSeriousException("GlobalAlloc failed!");
        }

        memcpy(GlobalLock(hg), allStr.c_str(), allStr.size() + 1);
        GlobalUnlock(hg);
        SetClipboardData(CF_TEXT, hg);
        CloseClipboard();
    }

    // convert stringstream to istream
    std::istringstream iss(is.str());

    // read from istream
    double readBackDouble1 = 5.5555;
    double readBackDouble2 = 6.6666;
    float readBackFloat1 = 15.5555f;
    float readBackFloat2 = 16.6666f;
    float readBackFloat3 = 17.7777f;
    float readBackFloat4 = 18.8888f;

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
    HdrFromIfStream<false, HDRFloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM2, iss);
    HdrFromIfStream<false, HDRFloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC2, iss);

    HdrFromIfStream<false, CudaDblflt<MattDblflt>, MattDblflt>(readBackCudaDblflt1, iss);
    HdrFromIfStream<false, CudaDblflt<dblflt>, dblflt>(readBackCudaDblflt2, iss);

    HdrFromIfStream<false, FloatComplex<float>, float>(readBackFloatComplex1, iss);
    HdrFromIfStream<false, FloatComplex<double>, double>(readBackFloatComplex2, iss);
    HdrFromIfStream<false, FloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackFloatComplex3, iss);
    HdrFromIfStream<false, FloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackFloatComplex4, iss);

    auto checker = [&](auto &a, auto &b) {
        if (a != b) {
            throw FractalSharkSeriousException("String conversion failed!");
        }
    };

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
    HdrFromIfStream<true, HDRFloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackM2, iss);
    HdrFromIfStream<true, HDRFloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackC2, iss);

    HdrFromIfStream<true, CudaDblflt<MattDblflt>, MattDblflt>(readBackCudaDblflt1, iss);
    HdrFromIfStream<true, CudaDblflt<dblflt>, dblflt>(readBackCudaDblflt2, iss);

    HdrFromIfStream<true, FloatComplex<float>, float>(readBackFloatComplex1, iss);
    HdrFromIfStream<true, FloatComplex<double>, double>(readBackFloatComplex2, iss);
    HdrFromIfStream<true, FloatComplex<CudaDblflt<MattDblflt>>, CudaDblflt<MattDblflt>>(readBackFloatComplex3, iss);
    HdrFromIfStream<true, FloatComplex<CudaDblflt<dblflt>>, CudaDblflt<dblflt>>(readBackFloatComplex4, iss);

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

void FractalTest::Benchmark(RefOrbitCalc::PerturbationResultType type) {
    static constexpr size_t NumIterations = 5;

    if (m_Fractal.GetRepaint() == false) {
        m_Fractal.ToggleRepainting();
    }

    std::vector<size_t> overallTimes;
    std::vector<size_t> perPixelTimes;
    std::vector<size_t> refOrbitTimes;
    std::vector<size_t> LAGenerationTimes;

    for (size_t i = 0; i < NumIterations; i++) {
        m_Fractal.ClearPerturbationResults(type);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        RefOrbitDetails details;
        m_Fractal.GetSomeDetails(details);

        overallTimes.push_back(m_Fractal.GetBenchmark().m_Overall.GetDeltaInMs());
        perPixelTimes.push_back(m_Fractal.GetBenchmark().m_PerPixel.GetDeltaInMs());
        refOrbitTimes.push_back(details.OrbitMilliseconds);
        LAGenerationTimes.push_back(details.LAMilliseconds);
    }

    // Write benchmarkData to the file BenchmarkResults.txt.  Truncate
    // the file if it already exists.
    std::ofstream file("BenchmarkResults.txt", std::ios::binary | std::ios::trunc);

    auto printVectorWithDescription = [&](const std::string &description, const std::vector<size_t> &vec) {
        file << description << "\r\n";
        for (size_t i = 0; i < NumIterations; i++) {
            file << vec[i] << "\r\n";
        }
        file << "\r\n";
        };

    printVectorWithDescription("Overall times (ms)", overallTimes);
    printVectorWithDescription("Per pixel times (ms)", perPixelTimes);
    printVectorWithDescription("RefOrbit times (ms)", refOrbitTimes);
    printVectorWithDescription("LA generation times (ms)", LAGenerationTimes);

    file.close();
}