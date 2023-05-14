#pragma once

#include "BLA.h"
#include <vector>

template<class T>
class BLAS;

template<class T>
class GPUBLAS {
public:
    using GPUBLA_TYPE = BLA<double>;

    GPUBLAS(const std::vector<std::vector<GPUBLA_TYPE>>& B,
        int32_t LM2,
        size_t FirstLevel);
    ~GPUBLAS();

    GPUBLAS(const GPUBLAS& other);

    uint32_t CheckValid() const;

    GPUBLAS(GPUBLAS&& other) = delete;
    GPUBLAS& operator=(const GPUBLAS& other) = delete;
    GPUBLAS& operator=(GPUBLAS&& other) = delete;

    __host__ __device__ GPUBLA_TYPE* LookupBackwards(size_t m, T z2);

protected:
    size_t* m_ElementsPerLevel;

    size_t m_NumLevels;
    GPUBLA_TYPE** m_B;

    static constexpr size_t BLA_BITS = 23;
    static constexpr size_t BLA_STARTING_LEVEL = 2;

    int32_t m_LM2;//Level -1 is not attainable due to Zero R
    size_t m_FirstLevel;

    cudaError_t m_Err;

    bool m_Owned;
};

class ScaledGPUBLAS : public GPUBLAS<double> {
public:
    using GPUBLA_TYPE = ScaledBLA<float>;

    ScaledGPUBLAS(const std::vector<std::vector<GPUBLAS<double>::GPUBLA_TYPE>>& B,
        int32_t LM2,
        size_t FirstLevel);
    ~ScaledGPUBLAS();

    ScaledGPUBLAS(const ScaledGPUBLAS& other);

    ScaledGPUBLAS(ScaledGPUBLAS&& other) = delete;
    ScaledGPUBLAS& operator=(const ScaledGPUBLAS& other) = delete;
    ScaledGPUBLAS& operator=(ScaledGPUBLAS&& other) = delete;

    __host__ __device__ GPUBLA_TYPE* LookupBackwards(size_t m, float z2, float scale);

private:
    GPUBLA_TYPE** m_ScaledB;
};
