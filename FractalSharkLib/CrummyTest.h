#pragma once

#include <stdint.h>
#include "GPU_Types.h"
#include "RefOrbitCalc.h"

class Fractal;
enum class IterTypeEnum;

class CrummyTest {
public:
    CrummyTest(Fractal &fractal);

    void TestAll();

    // TestAll includes these:
    void TestBasic();
    void TestReferenceSave();
    void TestVariedCompression();
    void TestImaginaLoad();
    void TestStringConversion();
    void TestPerturbedPerturb();
    void TestGrowableVector();
    void TestWindowResize();

    // TestAll does not include:
    void TestReallyHardView27();

    void Benchmark(RefOrbitCalc::PerturbationResultType type);

private:
    void TestPreReq(const wchar_t *dirName);

    void BasicTestInternal(
        const wchar_t *dirName,
        size_t &testIndex,
        IterTypeEnum iterType);

    template<TestTypeEnum viewIndex>
    void BasicOneTest(
        auto algToTest,
        size_t &testIndex,
        const wchar_t *dirName,
        const wchar_t *testPrefix,
        IterTypeEnum iterType);

    static std::string GenFilename(
        size_t testIndex,
        size_t viewIndex,
        RenderAlgorithm origAlgToTest,
        RenderAlgorithm convertAlgToTest,
        int32_t compressionError,
        IterTypeEnum iterType,
        std::string baseName);

    static std::string GenFilename(
        size_t testIndex,
        size_t viewIndex,
        RenderAlgorithm algToTest,
        int32_t compressionError,
        IterTypeEnum iterType,
        std::string baseName);

    static std::wstring GenFilenameW(
        size_t testIndex,
        size_t viewIndex,
        RenderAlgorithm algToTest,
        int32_t compressionError,
        IterTypeEnum iterType,
        const wchar_t *testPrefix,
        const wchar_t *dirName,
        std::string baseName);

    static std::wstring GenFilenameW(
        size_t testIndex,
        size_t viewIndex,
        RenderAlgorithm origAlgToTest,
        RenderAlgorithm convertAlgToTest,
        int32_t compressionError,
        IterTypeEnum iterType,
        const wchar_t *testPrefix,
        const wchar_t *dirName,
        std::string baseName);

    template<typename OrigAlgToTest, typename ConvertAlgToTest>
    void ReferenceSaveLoad(
        Fractal &fractal,
        const wchar_t *dirName,
        size_t viewIndex,
        size_t testIndex,
        IterTypeEnum iterType,
        ImaginaSettings imaginaSettings,
        OrigAlgToTest origAlgToTest,
        ConvertAlgToTest convertAlgToTest,
        int32_t compressionError);

    Fractal &m_Fractal;
};
