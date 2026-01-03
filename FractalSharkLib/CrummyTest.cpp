#include "stdafx.h"
#include "CrummyTest.h"

#include "Exceptions.h"
#include "Fractal.h"
#include "HDRFloat.h"

#include "..\FractalShark\resource.h"

#include <shellapi.h>

CrummyTest::CrummyTest(Fractal &fractal) : m_Fractal(fractal) {
}

void CrummyTest::TestAll() {
    TestBasic();
    TestReferenceSave();
    TestVariedCompression();
    TestImaginaLoad();
    TestStringConversion();
    TestPerturbedPerturb();
    TestGrowableVector();
}

void CrummyTest::TestPreReq(const wchar_t *dirName) {
    DWORD ftyp = GetFileAttributesW(dirName);
    if (ftyp != INVALID_FILE_ATTRIBUTES && (ftyp & FILE_ATTRIBUTE_DIRECTORY)) {
        // Delete the directory recursively
        // Double-null terminated string
        std::vector<wchar_t> dirNameDoubleNull{};
        dirNameDoubleNull.resize(std::wcslen(dirName) + 2, L'\0');
        std::copy(dirName, dirName + std::wcslen(dirName) + 1, dirNameDoubleNull.data());

        SHFILEOPSTRUCT fileOp{};
        fileOp.wFunc = FO_DELETE;
        fileOp.pFrom = dirNameDoubleNull.data();
        fileOp.fFlags = FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT | FOF_ALLOWUNDO;
        auto shRet = SHFileOperation(&fileOp);
        if (shRet != 0) {
            auto wstrMsg = std::wstring(L"Error deleting directory! SHFileOperation returned: ") + std::to_wstring(shRet);
            std::wcerr << wstrMsg.c_str() << std::endl;
            return;
        }
    }

    // Create the directory
    auto ret = CreateDirectory(dirName, nullptr);
    if (ret == 0 && GetLastError() != ERROR_ALREADY_EXISTS) {
        auto lastError = GetLastError();
        std::wstring msg = L"Error creating directory! Last error: " + std::to_wstring(lastError);
        std::wcerr << msg.c_str() << std::endl;
        return;
    }
}

void CrummyTest::BasicTestInternal(
    const wchar_t *dirName,
    size_t &testIndex,
    IterTypeEnum iterType) {

    constexpr bool IncludeSlow = false;

    // First, iterate over all the supported RenderAlgorithm entries and render the default view:
    // Skip AUTO plus all LAO-only algorithms.  They produce a black screen for the default view.
    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        BasicOneTest<TestTypeEnum::View0>(alg, testIndex, dirName, L"Default", iterType);
        });

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        BasicOneTest<TestTypeEnum::View5>(alg, testIndex, dirName, L"View5", iterType);
        });

    if constexpr (IncludeSlow) {
        IterateRenderAlgs([&](auto i) {
            constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
            BasicOneTest<TestTypeEnum::View10>(alg, testIndex, dirName, L"View10", iterType);
            });
    }

    // Finally, iterate over all the RenderAlgorithm entries that should work with View #11.
    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        BasicOneTest<TestTypeEnum::View11>(alg, testIndex, dirName, L"View11", iterType);
        });
}

std::string CrummyTest::GenFilename(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm origAlgToTest,
    RenderAlgorithm convertAlgToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    std::string baseName) {

    auto iterTypeStr =
        IterTypeEnum::Bits32 == iterType ? "32" : "64";

    std::string algStr;
    if (origAlgToTest == convertAlgToTest) {
        algStr =
            std::string(" - Alg#") +
            origAlgToTest.AlgorithmStr;
    } else {
        algStr =
            std::string(" - Alg#") +
            origAlgToTest.AlgorithmStr +
            std::string(" to ") +
            convertAlgToTest.AlgorithmStr;
    }

    auto compErrStr = compressionError >= 0 ? std::string(" - CompErr#") + std::to_string(compressionError) : "";

    auto nameWithIntPrefix =
        std::to_string(testIndex) +
        " - View#" + std::to_string(viewIndex) +
        algStr +
        compErrStr +
        " - Bits# " + iterTypeStr +
        " - " + baseName;
    return nameWithIntPrefix;

}

std::string CrummyTest::GenFilename(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm algToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    std::string baseName) {

    return GenFilename(
        testIndex,
        viewIndex,
        algToTest,
        algToTest,
        compressionError,
        iterType,
        baseName);
}

std::wstring CrummyTest::GenFilenameW(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm origAlgToTest,
    RenderAlgorithm convertAlgToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    const wchar_t *testPrefix,
    const wchar_t *dirName,
    std::string baseName) {

    auto nameWithIntPrefix = GenFilename(
        testIndex,
        viewIndex,
        origAlgToTest,
        convertAlgToTest,
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

std::wstring CrummyTest::GenFilenameW(
    size_t testIndex,
    size_t viewIndex,
    RenderAlgorithm algToTest,
    int32_t compressionError,
    IterTypeEnum iterType,
    const wchar_t *testPrefix,
    const wchar_t *dirName,
    std::string baseName) {

    return GenFilenameW(
        testIndex,
        viewIndex,
        algToTest,
        algToTest,
        compressionError,
        iterType,
        testPrefix,
        dirName,
        baseName);
}

template<TestTypeEnum testEnumIndex>
void CrummyTest::BasicOneTest(
    auto algToTest,
    size_t &testIndex,
    const wchar_t *dirName,
    const wchar_t *testPrefix,
    IterTypeEnum iterType) {

    if constexpr (algToTest.TestInclude.Lookup(testEnumIndex) != TestViewEnum::Disabled) {
        const auto viewIndex = static_cast<size_t>(algToTest.TestInclude.Lookup(testEnumIndex));

        m_Fractal.SetRenderAlgorithm(algToTest);
        m_Fractal.View(viewIndex);
        m_Fractal.ForceRecalc();

        auto name = m_Fractal.GetRenderAlgorithmName();
        m_Fractal.CalcFractal(false);

        const auto filenameW = GenFilenameW(
            testIndex,
            viewIndex,
            GetRenderAlgorithmTupleEntry(algToTest.Algorithm),
            -1,
            iterType,
            testPrefix,
            dirName,
            name);

        m_Fractal.SaveCurrentFractal(filenameW, false);
        //SaveItersAsText(filenameW);

        ++testIndex;
    }
}

void CrummyTest::TestBasic() {
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

    ++testIndex;
}

template<typename OrigAlgToTest, typename ConvertAlgToTest>
void CrummyTest::ReferenceSaveLoad(
    Fractal &fractal,
    const wchar_t *dirName,
    size_t viewIndex,
    size_t testIndex,
    IterTypeEnum iterType,
    ImaginaSettings imaginaSettings,
    OrigAlgToTest origAlgToTest,
    ConvertAlgToTest convertAlgToTest,
    int32_t compressionError) {

    fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
    fractal.SetIterType(iterType);
    fractal.SetRenderAlgorithm(origAlgToTest);
    fractal.View(viewIndex);
    fractal.ForceRecalc();
    fractal.CalcFractal(false);

    const wchar_t *testPrefix = L"RefOrbitTest";
    const auto algStr = fractal.GetRenderAlgorithmName(origAlgToTest);

    const auto genLocalFilename = [&](const std::wstring extraPrefix) {
        std::wstring fullPrefix = testPrefix + std::wstring(L" - ") + extraPrefix;
        return GenFilenameW(
            testIndex,
            viewIndex,
            origAlgToTest,
            convertAlgToTest,
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

        // Only run this for non-2x32 algorithms
        // TODO introduce better way to sort out different RenderAlgorithm properties
        try {
            const auto simpleFilename = genLocalFilename(L"Simple") + L".txt";
            fractal.SaveRefOrbit(CompressToDisk::SimpleCompression, simpleFilename);
        }
        catch (FractalSharkSeriousException &e) {
            if constexpr (origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2 ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2PO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedLAv2LAO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2 ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2PO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2LAO ||

                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2 ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2PO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedLAv2LAO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2 ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2PO ||
                origAlgToTest.Algorithm == RenderAlgorithmEnum::Gpu2x32PerturbedRCLAv2LAO) {

                // This is expected to fail for 2x32 algorithms
            } else {

                // Scaled algs also throw but we're not testing that currently
                throw e;
            }
        }
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

    if (origAlgToTest == convertAlgToTest) {
        fractal.SetRenderAlgorithm(RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>{}});
    }

    fractal.LoadRefOrbit(
        nullptr,
        CompressToDisk::MaxCompressionImagina,
        imaginaSettings,
        maxImaginaFilename);

    if (fractal.GetNumIterations<IterTypeFull>() != expectedIterations) {
        throw FractalSharkSeriousException("LoadRefOrbit failed to set the correct number of iterations!");
    }

    fractal.CalcFractal(false);

    const auto decompressedResultFilename = genLocalFilename(L"Decompressed");
    fractal.SaveCurrentFractal(decompressedResultFilename, false);
}

void CrummyTest::TestReferenceSave() {

    const wchar_t *dirName = L"TestReferenceSave";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);

    size_t testIndex = 0;

    m_Fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    int32_t compressionError = m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low);

    auto loopAll = [&]<TestTypeEnum viewEnum>(auto algToTest) {
        const auto view = algToTest.TestInclude.Lookup(viewEnum);
        if constexpr (view != TestViewEnum::Disabled) {
            const auto viewIndex = static_cast<size_t>(view);

            const auto testSettings = { ImaginaSettings::ConvertToCurrent, ImaginaSettings::UseSaved };
            for (auto curSettings : testSettings) {
                ReferenceSaveLoad(
                    m_Fractal,
                    dirName,
                    viewIndex,
                    testIndex,
                    IterTypeEnum::Bits64,
                    curSettings,
                    algToTest,
                    algToTest,
                    compressionError);

                testIndex++;
            }
        }
    };

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave0>(alg);
        });

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave5>(alg);
        });

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave10>(alg);
        });

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave13>(alg);
        });

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave14>(alg);
        });
}

void CrummyTest::TestVariedCompression() {
    const wchar_t *dirName = L"TestVariedCompression";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);

    size_t testIndex = 0;

    m_Fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    int32_t compressionError = m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low);

    auto loopAll = [&]<TestTypeEnum viewEnum>(
        auto algToTest) {

        constexpr auto view = algToTest.TestInclude.Lookup(viewEnum);
        if constexpr (view != TestViewEnum::Disabled) {
            const auto viewIndex = static_cast<size_t>(view);

            for (int32_t compressionErrorExp = 1; compressionErrorExp <= compressionError; compressionErrorExp++) {
                const auto origAlg = algToTest;
                const auto convertToAlg = algToTest;
                ReferenceSaveLoad(
                    m_Fractal,
                    dirName,
                    viewIndex,
                    testIndex,
                    IterTypeEnum::Bits64,
                    ImaginaSettings::ConvertToCurrent,
                    origAlg,
                    convertToAlg,
                    compressionErrorExp);

                testIndex++;
            }
        }
    };

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll.operator()<TestTypeEnum::ReferenceSave5 >(alg);
    });
}

void CrummyTest::TestImaginaLoad() {
    // First, write out all the Imagina resources to files

    struct Pair {
        DWORD rc;
        const wchar_t *name;
        int view;
        const wchar_t *origPngName;
        const wchar_t *imaginaPngName;
    };

    const wchar_t *dirName = L"TestImaginaLoad";
    TestPreReq(dirName);

    std::vector<Pair> pairs;
    pairs.push_back({ IDR_IMAGINA_VIEW0, L"View0.im", 0, L"View0 Orig.png", L"View0 Imagina.png" });
    pairs.push_back({ IDR_IMAGINA_VIEW5, L"View5.im", 5, L"View5 Orig.png", L"View5 Imagina.png" });
    pairs.push_back({ IDR_IMAGINA_VIEW14, L"View14.im", 14, L"View14 Orig.png", L"View14 Imagina.png" });
    pairs.push_back({ IDR_IMAGINA_VIEW15, L"View15.im", 15, L"View15 Orig.png", L"View15 Imagina.png" });
    pairs.push_back({ IDR_IMAGINA_VIEW19, L"View19.im", 19, L"View19 Orig.png", L"View19 Imagina.png" });
    pairs.push_back({ IDR_IMAGINA_VIEWEASY1, L"ViewEasy1.im", -1, L"ViewEasy1 Orig.png", L"ViewEasy1 Imagina.png" });

   auto processOnePair = [this, dirName](const auto &pair) {
        auto hInst = GetModuleHandle(nullptr);

        auto throwErr = [](const std::string &msg) {
            auto lastError = GetLastError();
            std::string msgFull = msg + " Last error: " + std::to_string(lastError);
            throw FractalSharkSeriousException(msgFull);
            };

        // Load the Imagina resource
        auto hRes = FindResource(hInst, MAKEINTRESOURCE(pair.rc), L"SHARK_DATA");
        if (hRes == nullptr) {
            throwErr("FindResource failed!");
        }

        auto hGlobal = LoadResource(hInst, hRes);
        if (hGlobal == nullptr) {
            throwErr("LoadResource failed!");
        }

        auto pRes = LockResource(hGlobal);
        if (pRes == nullptr) {
            throwErr("LockResource failed!");
        }

        auto size = SizeofResource(hInst, hRes);
        // Write the resource to a file
        auto filename = pair.name;
        auto hFile = CreateFile(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            throwErr("CreateFile failed!");
        }

        DWORD bytesWritten;
        auto ret = WriteFile(hFile, pRes, size, &bytesWritten, nullptr);
        if (ret == 0) {
            CloseHandle(hFile);
            throw FractalSharkSeriousException("WriteFile failed!");
        }

        CloseHandle(hFile);

        // Unlock
        UnlockResource(hGlobal);

        m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
        m_Fractal.SetIterType(IterTypeEnum::Bits64);
        m_Fractal.SetRenderAlgorithm(RenderAlgorithm{RenderAlgorithmCompileTime<RenderAlgorithmEnum::AUTO>{}});
        m_Fractal.View(pair.view);
        m_Fractal.ForceRecalc();
        m_Fractal.CalcFractal(false);

        const auto outOrigFilename = dirName + std::wstring(L"\\") + pair.origPngName;
        m_Fractal.SaveCurrentFractal(outOrigFilename, false);

        // Load the Imagina file
        m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);

        ImaginaSettings imaginaSettings = ImaginaSettings::UseSaved;
        auto filenameW = std::wstring(pair.name);
        m_Fractal.LoadRefOrbit(nullptr, CompressToDisk::MaxCompressionImagina, imaginaSettings, filenameW);
        m_Fractal.CalcFractal(false);

        const auto outImaginaFilename = dirName + std::wstring(L"\\") + pair.imaginaPngName;
        m_Fractal.SaveCurrentFractal(outImaginaFilename, false);

        // Delete the Imagina file
        auto ret2 = DeleteFile(filename);
        if (ret2 == 0) {
            throwErr("DeleteFile failed!");
        }
    };

    for (const auto &pair : pairs) {
        processOnePair(pair);
    }
}

void CrummyTest::TestStringConversion() {
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
    auto toString = [&]<bool IntOut>(auto & hdr) {
        auto ret = std::string("Descriptor: ");
        ret += HdrToString<IntOut>(hdr);
        // append newline to ret
        ret += std::string("\n");

        // append string to stringstream
        is << ret;
        return ret;
    };

    std::string allStr;
    allStr += toString.template operator() < false > (double1);
    allStr += toString.template operator() < false > (double2);
    allStr += toString.template operator() < false > (float1);
    allStr += toString.template operator() < false > (float2);
    allStr += toString.template operator() < false > (float3);
    allStr += toString.template operator() < false > (float4);

    allStr += toString.template operator() < false > (f1);
    allStr += toString.template operator() < false > (d1);
    allStr += toString.template operator() < false > (m1);
    allStr += toString.template operator() < false > (c1);

    allStr += toString.template operator() < false > (f2);
    allStr += toString.template operator() < false > (d2);
    allStr += toString.template operator() < false > (m2);
    allStr += toString.template operator() < false > (c2);

    allStr += toString.template operator() < false > (cudaDblflt1);
    allStr += toString.template operator() < false > (cudaDblflt2);

    allStr += toString.template operator() < false > (floatComplex1);
    allStr += toString.template operator() < false > (floatComplex2);
    allStr += toString.template operator() < false > (floatComplex3);
    allStr += toString.template operator() < false > (floatComplex4);

    allStr += toString.template operator() < true > (double1);
    allStr += toString.template operator() < true > (double2);
    allStr += toString.template operator() < true > (float1);
    allStr += toString.template operator() < true > (float2);
    allStr += toString.template operator() < true > (float3);
    allStr += toString.template operator() < true > (float4);

    allStr += toString.template operator() < true > (f1);
    allStr += toString.template operator() < true > (d1);
    allStr += toString.template operator() < true > (m1);
    allStr += toString.template operator() < true > (c1);
    allStr += toString.template operator() < true > (f2);
    allStr += toString.template operator() < true > (d2);
    allStr += toString.template operator() < true > (m2);
    allStr += toString.template operator() < true > (c2);

    allStr += toString.template operator() < true > (cudaDblflt1);
    allStr += toString.template operator() < true > (cudaDblflt2);

    allStr += toString.template operator() < true > (floatComplex1);
    allStr += toString.template operator() < true > (floatComplex2);
    allStr += toString.template operator() < true > (floatComplex3);
    allStr += toString.template operator() < true > (floatComplex4);

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

        auto *result = GlobalLock(hg);
        if (result != nullptr) {
            memcpy(result, allStr.c_str(), allStr.size() + 1);
        }

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

void CrummyTest::TestPerturbedPerturb() {
    const wchar_t *dirName = L"TestPerturbedPerturb";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::DontSave);

    size_t testIndex = 0;

    m_Fractal.DefaultCompressionErrorExp(Fractal::CompressionError::Low);
    int32_t compressionError = m_Fractal.GetCompressionErrorExp(Fractal::CompressionError::Low);

    auto loopAll = [&](auto algToTest) {
        if constexpr (algToTest.TestInclude.Lookup(TestTypeEnum::PerturbedPerturb12) != TestViewEnum::Disabled) {
            const auto viewIndex = 14;
            const auto perturbedViewIndex = 12;
            const auto iterType = IterTypeEnum::Bits32;
            const wchar_t *testPrefix = L"PerturbedPerturb";
            const auto algStr = std::string(algToTest.AlgorithmStr);
            RenderAlgorithm algToTestRT{ GetRenderAlgorithmTupleEntry(algToTest.Algorithm) };

            const auto genLocalFilename = [&](const std::wstring extraPrefix) {
                std::wstring fullPrefix = testPrefix + std::wstring(L" - ") + extraPrefix;
                return GenFilenameW(
                    testIndex,
                    viewIndex,
                    algToTestRT,
                    algToTestRT,
                    compressionError,
                    iterType,
                    fullPrefix.c_str(),
                    dirName,
                    algStr);
                };

            m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
            m_Fractal.SetIterType(iterType);
            m_Fractal.SetRenderAlgorithm(algToTestRT);
            m_Fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3PerturbMTHighMTMed3);
            m_Fractal.View(viewIndex);
            m_Fractal.ForceRecalc();
            m_Fractal.CalcFractal(false);
            m_Fractal.SaveCurrentFractal(genLocalFilename(L"Original"), false);

            m_Fractal.View(perturbedViewIndex);
            m_Fractal.ForceRecalc();
            m_Fractal.CalcFractal(false);

            m_Fractal.SaveCurrentFractal(genLocalFilename(L"Perturbed"), false);
        }
    };

    IterateRenderAlgs([&](auto i) {
        constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
        loopAll(alg);
        });
}

void CrummyTest::TestGrowableVector() {
    auto VerifyContents = [](const GrowableVector<uint64_t> &testVector, uint64_t manyElts) {
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

        constexpr uint64_t ManyElts = 1024 * 1024 * 1024;
        for (uint64_t i = 0; i < ManyElts; i++) {
            testVector.PushBack(static_cast<uint64_t>(i));
        }

        // Try moving the vector
        GrowableVector<uint64_t> testVector2(std::move(testVector));

        // Verify contents
        VerifyContents(testVector2, ManyElts);

        // Try the move assignment operator
        GrowableVector<uint64_t> testVector3;
        testVector3 = std::move(testVector2);

        // Verify contents
        VerifyContents(testVector3, ManyElts);

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
        VerifyContents(testVector3, ManyElts);

        // ValidFile
        if (testVector3.ValidFile() != true) {
            throw FractalSharkSeriousException("GrowableVector ValidFile failed!");
        }
    }
}

void CrummyTest::TestReallyHardView27() {
    const wchar_t *dirName = L"ReallyHardView27";
    TestPreReq(dirName);

    m_Fractal.SetResultsAutosave(AddPointOptions::EnableWithoutSave);

    size_t testIndex = 0;

    static constexpr auto StartError = 18;
    static constexpr auto EndError = 22;

    static constexpr auto TestView = TestTypeEnum::View27;
    //static constexpr auto TestView = TestTypeEnum::View5;

    auto loopAll = [&]<TestTypeEnum viewEnum>(
        auto algToTest) {

        constexpr auto view = algToTest.TestInclude.Lookup(viewEnum);
        if constexpr (view != TestViewEnum::Disabled) {
            const auto viewIndex = static_cast<size_t>(view);

            for (int32_t compressionErrorExp = StartError;
                compressionErrorExp <= EndError;
                compressionErrorExp++) {

                m_Fractal.ClearPerturbationResults(RefOrbitCalc::PerturbationResultType::All);
                const auto iterType = IterTypeEnum::Bits64;
                m_Fractal.SetIterType(iterType);
                m_Fractal.SetRenderAlgorithm(algToTest);
                m_Fractal.SetPerturbationAlg(RefOrbitCalc::PerturbationAlg::MTPeriodicity3);
                m_Fractal.View(viewIndex, false);
                m_Fractal.SetCompressionErrorExp(Fractal::CompressionError::Low, compressionErrorExp);
                m_Fractal.GetLAParameters().SetDefaults(LAParameters::LADefaults::MaxPerf);
                m_Fractal.ForceRecalc();
                m_Fractal.CalcFractal(false);

                auto name = m_Fractal.GetRenderAlgorithmName();
                const auto testPrefix = L"LAMaxPerf";

                const auto filenameW = GenFilenameW(
                    testIndex,
                    viewIndex,
                    GetRenderAlgorithmTupleEntry(algToTest.Algorithm),
                    compressionErrorExp,
                    iterType,
                    testPrefix,
                    dirName,
                    name);

                std::string renderDetailsShort, renderDetailsLong;
                m_Fractal.GetRenderDetails(renderDetailsShort, renderDetailsLong);
                m_Fractal.SaveCurrentFractal(filenameW, false);

                // Write renderDetailsShort to a separate text file
                const auto textTestPrefix = L"RenderDetailsShort";
                const auto textFilenameW = GenFilenameW(
                    testIndex,
                    viewIndex,
                    GetRenderAlgorithmTupleEntry(algToTest.Algorithm),
                    compressionErrorExp,
                    iterType,
                    textTestPrefix,
                    dirName,
                    name) + L".txt";

                std::ofstream file(textFilenameW, std::ios::binary | std::ios::trunc);
                file << renderDetailsShort;
                file.close();

                const auto maxImaginaFilename =
                    GenFilenameW(
                        testIndex,
                        viewIndex,
                        GetRenderAlgorithmTupleEntry(algToTest.Algorithm),
                        compressionErrorExp,
                        iterType,
                        L"MaxImagina",
                        dirName,
                        name) + L".im";
                m_Fractal.SaveRefOrbit(CompressToDisk::MaxCompressionImagina, maxImaginaFilename);

                testIndex++;
            }
        }
    };

    //IterateRenderAlgs([&](auto i) {
    //    constexpr auto alg = std::get<i>(RenderAlgorithmsTuple);
    //    loopAll.operator() < TestTypeEnum::View5 > (alg);
    //    });

    loopAll.operator() < TestView > (
        RenderAlgorithmCompileTime<RenderAlgorithmEnum::GpuHDRx2x32PerturbedRCLAv2>{});
}

void CrummyTest::Benchmark(RefOrbitCalc::PerturbationResultType type) {
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
