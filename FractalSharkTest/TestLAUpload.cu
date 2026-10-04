#include "stdafx.h"

#include <cuda_runtime.h>

#include "GPU_LAReference.h"
#include "LAReference.h"
#include "TestFramework.h"

namespace {

struct UploadResult {
    double Value;
    double Z;
    uint64_t Step;
    uint64_t Descent;
    bool BelowThreshold;
    bool AtThreshold;
    bool BudgetRejected;
};

class TestStream {
public:
    cudaStream_t m_Stream{};
    TestStream() { ASSERT_EQ(cudaStreamCreate(&m_Stream), cudaSuccess); }
    ~TestStream() { cudaStreamDestroy(m_Stream); }
};

template <class T> class TestDeviceBuffer {
public:
    T *m_Data{};
    TestDeviceBuffer() { ASSERT_EQ(cudaMalloc(&m_Data, sizeof(T)), cudaSuccess); }
    ~TestDeviceBuffer() { cudaFree(m_Data); }
};

template <typename IterType, class Float, class SubType>
__global__ void
InspectUploaded(const GPU_LAReference<IterType, Float, SubType> *reference,
                IterType index,
                UploadResult *output)
{
    const auto step = reference->getLA(0, {Float{0.125}, Float{0}}, index, 0, 100);
    const auto value = step.Evaluate({Float{0.0625}, Float{0}});
    if constexpr (GPU_LAInfoDeep<IterType, Float, SubType>::IsHDR) {
        output->Value = value.getRe().toDouble();
        output->Z = step.getZ(value).getRe().toDouble();
    } else {
        output->Value = static_cast<double>(value.getRe());
        output->Z = static_cast<double>(step.getZ(value).getRe());
    }
    output->Step = step.step;
    output->Descent = step.nextStageLAindex;
    output->BelowThreshold = reference->IsLAStageInvalid(0, {Float{0.03125}, Float{0}});
    output->AtThreshold = reference->IsLAStageInvalid(0, {Float{0.0625}, Float{0}});
    output->BudgetRejected = reference->getLA(0, {}, index, 0, 1).unusable;
}

template <typename IterType, class SourceFloat, class SourceSubType, class Float, class SubType>
void
CheckUpload(size_t rows, FractalShark::LA::StorageLocation location)
{
    LAReference<IterType, SourceFloat, SourceSubType, PerturbExtras::Disable> source{
        AddPointOptions::DontSave, L"", L""};
    source.GetLAs().MutableResize(rows + 1);
    source.GetLAStages().MutableResize(1);
    source.GetLAStages()[0] =
        LAStageInfo<IterType, SourceFloat>{0, static_cast<IterType>(rows), SourceFloat{0.0625}};
    for (size_t i = 0; i <= rows; ++i) {
        auto &row = source.GetLAs()[i];
        row.Ref = {SourceFloat{i == rows ? 7.0f : 1.0f}, SourceFloat{0}};
        row.ZCoeff = {SourceFloat{2}, SourceFloat{0}};
        row.CCoeff = {SourceFloat{3}, SourceFloat{0}};
        row.LAThreshold = SourceFloat{100};
        row.LAi = LAInfoI<IterType>{2, 17};
    }
    TestStream stream;
    GPU_LAReference<IterType, Float, SubType> reference{source, stream.m_Stream, location};
    ASSERT_EQ(reference.CheckValid(), static_cast<uint32_t>(cudaSuccess));
    TestDeviceBuffer<GPU_LAReference<IterType, Float, SubType>> descriptor;
    TestDeviceBuffer<UploadResult> output;
    ASSERT_EQ(
        cudaMemcpyAsync(
            descriptor.m_Data, &reference, sizeof(reference), cudaMemcpyHostToDevice, stream.m_Stream),
        cudaSuccess);
    InspectUploaded<<<1, 1, 0, stream.m_Stream>>>(
        descriptor.m_Data, static_cast<IterType>(rows - 1), output.m_Data);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    UploadResult result{};
    ASSERT_EQ(
        cudaMemcpyAsync(&result, output.m_Data, sizeof(result), cudaMemcpyDeviceToHost, stream.m_Stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream.m_Stream), cudaSuccess);
    ASSERT_NEAR(result.Value, 0.71875, 0.0);
    ASSERT_NEAR(result.Z, 7.71875, 0.0);
    ASSERT_EQ(result.Step, uint64_t{2});
    ASSERT_EQ(result.Descent, uint64_t{17});
    ASSERT_FALSE(result.BelowThreshold);
    ASSERT_TRUE(result.AtThreshold);
    ASSERT_TRUE(result.BudgetRejected);
}

} // namespace

TEST(CudaLAUpload_MatchingLayouts)
{
    CheckUpload<uint32_t, HDRFloat<float>, float, HDRFloat<float>, float>(
        3, FractalShark::LA::StorageLocation::DevicePreferred);
    CheckUpload<uint64_t, HDRFloat<double>, double, HDRFloat<double>, double>(
        3, FractalShark::LA::StorageLocation::DevicePreferred);
    CheckUpload<uint32_t, float, float, float, float>(
        3, FractalShark::LA::StorageLocation::DevicePreferred);
}

TEST(CudaLAUpload_ConvertedChunks)
{
    CheckUpload<uint32_t, HDRFloat<double>, double, HDRFloat<float>, float>(
        170000, FractalShark::LA::StorageLocation::DevicePreferred);
}

TEST(CudaLAUpload_HostFallback)
{
    CheckUpload<uint32_t, HDRFloat<float>, float, HDRFloat<float>, float>(
        3, FractalShark::LA::StorageLocation::Host);
    CheckUpload<uint32_t, HDRFloat<double>, double, HDRFloat<float>, float>(
        3, FractalShark::LA::StorageLocation::Host);
}
