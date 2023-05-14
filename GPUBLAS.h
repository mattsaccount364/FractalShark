#pragma once

#include "BLA.h"
#include <vector>

class BLAS;

class GPUBLAS {
public:
    GPUBLAS(const std::vector<std::vector<BLA>>& B,
        int32_t LM2,
        size_t FirstLevel);
    ~GPUBLAS();

    GPUBLAS(const GPUBLAS& other);

    uint32_t CheckValid() const;

    GPUBLAS(GPUBLAS&& other) = delete;
    GPUBLAS& operator=(const GPUBLAS& other) = delete;
    GPUBLAS& operator=(GPUBLAS&& other) = delete;

    __host__ __device__ BLA* LookupBackwards(size_t m, double z2);

private:
    size_t* m_ElementsPerLevel;

    size_t m_NumLevels;
    BLA** m_B;

    static constexpr size_t BLA_BITS = 23;
    static constexpr size_t BLA_STARTING_LEVEL = 2;

    int32_t m_LM2;//Level -1 is not attainable due to Zero R
    size_t m_FirstLevel;

    cudaError_t m_Err;

    bool m_Owned;
};
