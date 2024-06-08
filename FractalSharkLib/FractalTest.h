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
        RenderAlgorithm algToTest,
        IterTypeEnum iterType,
        std::string baseName);

    static std::wstring GenFilenameW(
        size_t testIndex,
        RenderAlgorithm algToTest,
        IterTypeEnum iterType,
        const wchar_t *testPrefix,
        const wchar_t *dirName,
        std::string baseName);

    Fractal &m_Fractal;
};
