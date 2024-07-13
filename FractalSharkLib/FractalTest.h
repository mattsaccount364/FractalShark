#pragma once

#include <stdint.h>
#include "GPU_Types.h"
#include "RefOrbitCalc.h"

class Fractal;
enum class IterTypeEnum;

class FractalTest {
public:
    FractalTest(Fractal &fractal);

    void TestBasic();
    void TestReferenceSave();
    void TestVariedCompression();

    void Benchmark(RefOrbitCalc::PerturbationResultType type);

private:
    void TestPreReq(const wchar_t *dirName);

    void BasicTestInternal(
        const wchar_t *dirName,
        size_t &testIndex,
        IterTypeEnum iterType);

    void BasicOneTest(
        size_t viewIndex,
        size_t testIndex,
        const wchar_t *dirName,
        const wchar_t *testPrefix,
        RenderAlgorithm algToTest,
        IterTypeEnum iterType);

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

    void ReferenceSaveLoad(
        Fractal &fractal,
        const wchar_t *dirName,
        size_t viewIndex,
        size_t testIndex,
        IterTypeEnum iterType,
        RenderAlgorithm algToTest,
        int32_t compressionError);

    Fractal &m_Fractal;
};
