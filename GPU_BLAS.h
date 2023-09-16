#pragma once

#include "BLA.h"
#include <vector>

template<class T>
class BLAS;

template<class T, class GPUBLA_TYPE, int32_t LM2>
class GPU_BLAS {
public:
    GPU_BLAS(const std::vector<std::vector<GPUBLA_TYPE>>& B);
    ~GPU_BLAS();

    GPU_BLAS(const GPU_BLAS& other);

    uint32_t CheckValid() const;

    GPU_BLAS(GPU_BLAS&& other) = delete;
    GPU_BLAS& operator=(const GPU_BLAS& other) = delete;
    GPU_BLAS& operator=(GPU_BLAS&& other) = delete;

#ifdef __CUDA_ARCH__
    CUDA_CRAP const GPUBLA_TYPE* LookupBackwards(
        const GPUBLA_TYPE* __restrict__ *altB,
        /*T* curBR2,*/
        //const GPUBLA_TYPE * __restrict__ nullBla,
        size_t m,
        T z2) const;
#endif

    static constexpr size_t m_NumLevels = LM2 + 2;

    CUDA_CRAP GPUBLA_TYPE** GetB() const {
        return m_B;
    }

protected:
    GPUBLA_TYPE* m_BMem;
    GPUBLA_TYPE** m_B;

    static constexpr size_t BLA_STARTING_LEVEL = 3;
    static constexpr size_t m_FirstLevel = BLA_STARTING_LEVEL - 1;

    cudaError_t m_Err;

    const bool m_Owned;
};
